//! Strict conversation statement verifier with inline full GKR verifier execution.
//!
//! This is the recursive target for real LLM generations: the Cairo execution
//! proven by `cairo-prove --recursive-160` verifies one full GKR proof per
//! generated token, binds it to the conversation step, then emits the canonical
//! 19-felt conversation/action statement.

use core::poseidon::poseidon_hash_span;
use elo_cairo_verifier::aggregated_binding::{
    deserialize_binding_proof_packed, verify_aggregated_binding,
};
use elo_cairo_verifier::channel::{
    channel_default, channel_draw_qm31s, channel_mix_felt, channel_mix_secure_field,
    channel_mix_u64,
};
use elo_cairo_verifier::field::{
    QM31, evaluate_mle_from_io_span_2d, log2_ceil, next_power_of_two, qm31_eq,
};
use elo_cairo_verifier::model_verifier::{WeightClaimData, verify_gkr_model_with_trace_dp};
use elo_cairo_verifier::types::GKRClaim;

const DOMAIN_SEQ: felt252 = 0x53455131; // "SEQ1"
const DOMAIN_STEP: felt252 = 0x53544550; // "STEP"
const DOMAIN_ACTION: felt252 = 0x4143544E; // "ACTN"
const DOMAIN_CONVERSATION: felt252 = 0x434F4E56; // "CONV"
const DOMAIN_BATCH: felt252 = 0x43424154; // "CBAT"
const DOMAIN_KV_ROOT: felt252 = 0x4B565254; // "KVRT"
const DOMAIN_WEIGHT_GROUPS: felt252 = 0x57475254; // "WGRT"
const DOMAIN_ML_RECEIPT: felt252 = 0x4D4C5243; // "MLRC"
const STATEMENT_VERSION: felt252 = 1;
const WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK: u32 = 4;
const GROUPED_BINDING_MARKER: felt252 = 0x47525044; // "GRPD"

#[derive(Copy, Drop, Serde)]
struct Conversation {
    conversation_index: felt252,
    conversation_id_hash: felt252,
    prompt_commitment: felt252,
    transcript_commitment: felt252,
    action_root: felt252,
    initial_kv_commitment: felt252,
    final_kv_commitment: felt252,
    n_turns: felt252,
    n_prefill_tokens: felt252,
    n_generated_tokens: felt252,
    first_step_index: felt252,
    n_steps: felt252,
}

#[derive(Copy, Drop, Serde)]
struct Step {
    global_step_index: felt252,
    conversation_index: felt252,
    turn_index: felt252,
    token_index: felt252,
    generated_token_id: felt252,
    io_commitment: felt252,
    sampling_commitment: felt252,
    prev_kv_commitment: felt252,
    kv_commitment: felt252,
    ml_receipt_hash: felt252,
}

#[derive(Copy, Drop, Serde)]
struct Action {
    conversation_index: felt252,
    turn_index: felt252,
    action_index: felt252,
    action_kind_hash: felt252,
    tool_name_hash: felt252,
    input_commitment: felt252,
    output_commitment: felt252,
    policy_commitment: felt252,
}

#[derive(Drop, Serde)]
struct GkrStepProof {
    raw_io_data: Array<felt252>,
    circuit_depth: u32,
    num_layers: u32,
    matmul_dims: Array<u32>,
    dequantize_bits: Array<u64>,
    proof_data: Array<felt252>,
    weight_commitments: Array<felt252>,
    weight_binding_mode: u32,
    weight_binding_data: Array<felt252>,
    packed: bool,
    double_packed: bool,
    has_kv_cache: bool,
}

#[derive(Drop, Serde)]
struct ConversationGkrWitness {
    model_id: felt252,
    verifier_program_hash: felt252,
    circuit_hash: felt252,
    weight_super_root: felt252,
    policy_commitment: felt252,
    tokenizer_config_hash: felt252,
    hades_commitment: felt252,
    security_bits: felt252,
    conversations: Array<Conversation>,
    steps: Array<Step>,
    actions: Array<Action>,
    gkr_proofs: Array<GkrStepProof>,
}

#[derive(Copy, Drop)]
struct GkrVerificationOutput {
    model_id: felt252,
    io_commitment: felt252,
    weight_commitment: felt252,
    num_layers: u32,
    num_matmuls: u32,
    verified: bool,
}

#[executable]
fn main(witness: ConversationGkrWitness) -> Array<felt252> {
    let ConversationGkrWitness {
        model_id,
        verifier_program_hash,
        circuit_hash,
        weight_super_root,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        security_bits,
        conversations,
        steps,
        actions,
        mut gkr_proofs,
    } = witness;

    let security_bits_u32: u32 = security_bits.try_into().unwrap();
    assert!(model_id != 0, "model_id cannot be zero");
    assert!(circuit_hash != 0, "circuit_hash cannot be zero");
    assert!(weight_super_root != 0, "weight root cannot be zero");
    assert!(policy_commitment != 0, "policy cannot be zero");
    assert!(security_bits_u32 >= 160, "security below 160");
    assert!(conversations.len() != 0, "no conversations");
    assert!(gkr_proofs.len() == steps.len(), "GKR proof count mismatch");

    let mut conversation_root_acc = sequence_init(DOMAIN_CONVERSATION, conversations.len().into());
    let mut initial_kv_root_acc = sequence_init(DOMAIN_KV_ROOT, conversations.len().into());
    let mut final_kv_root_acc = sequence_init(DOMAIN_KV_ROOT, conversations.len().into());
    let mut total_prefill_tokens: felt252 = 0;
    let mut total_generated_tokens: felt252 = 0;

    let mut conversation_index: u32 = 0;
    loop {
        if conversation_index >= conversations.len() {
            break;
        }
        let conversation = *conversations.at(conversation_index);
        assert!(
            conversation.conversation_index == conversation_index.into(),
            "conversation index mismatch",
        );
        conversation_root_acc =
            sequence_push(
                DOMAIN_CONVERSATION,
                conversation_root_acc,
                conversation_index.into(),
                commit_conversation(conversation),
            );
        initial_kv_root_acc =
            sequence_push(
                DOMAIN_KV_ROOT,
                initial_kv_root_acc,
                conversation_index.into(),
                conversation.initial_kv_commitment,
            );
        final_kv_root_acc =
            sequence_push(
                DOMAIN_KV_ROOT,
                final_kv_root_acc,
                conversation_index.into(),
                conversation.final_kv_commitment,
            );
        total_prefill_tokens += conversation.n_prefill_tokens;
        total_generated_tokens += conversation.n_generated_tokens;
        conversation_index += 1;
    }

    let mut generation_root_acc = sequence_init(DOMAIN_STEP, steps.len().into());
    let mut step_index: u32 = 0;
    loop {
        if step_index >= steps.len() {
            break;
        }
        let step = *steps.at(step_index);
        assert!(step.global_step_index == step_index.into(), "step index mismatch");

        let gkr_proof = gkr_proofs.pop_front().unwrap();
        let gkr_output = verify_gkr_step(
            model_id, circuit_hash, weight_super_root, policy_commitment, step, gkr_proof,
        );
        validate_gkr_output(model_id, weight_super_root, step, @gkr_output);

        generation_root_acc =
            sequence_push(DOMAIN_STEP, generation_root_acc, step_index.into(), commit_step(step));
        step_index += 1;
    }
    assert!(gkr_proofs.len() == 0, "unused GKR proofs");

    let mut action_root_acc = sequence_init(DOMAIN_ACTION, actions.len().into());
    let mut action_index: u32 = 0;
    loop {
        if action_index >= actions.len() {
            break;
        }
        let action = *actions.at(action_index);
        action_root_acc =
            sequence_push(
                DOMAIN_ACTION, action_root_acc, action_index.into(), commit_action(action),
            );
        action_index += 1;
    }

    validate_statement_consistency(conversations.span(), steps.span(), actions.span());

    array![
        DOMAIN_BATCH, STATEMENT_VERSION, model_id, verifier_program_hash, circuit_hash,
        weight_super_root, policy_commitment, tokenizer_config_hash, hades_commitment,
        conversation_root_acc, generation_root_acc, action_root_acc, initial_kv_root_acc,
        final_kv_root_acc, conversations.len().into(), steps.len().into(), total_prefill_tokens,
        total_generated_tokens, security_bits,
    ]
}

fn verify_gkr_step(
    model_id: felt252,
    circuit_hash: felt252,
    weight_super_root: felt252,
    policy_commitment: felt252,
    step: Step,
    proof: GkrStepProof,
) -> GkrVerificationOutput {
    assert!(
        proof.weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
        "only aggregated oracle weight binding supported",
    );
    assert!(proof.raw_io_data.len() >= 6, "raw IO too short");

    let raw_io = proof.raw_io_data.span();
    let io_commitment = poseidon_hash_span(raw_io);
    assert!(io_commitment == step.io_commitment, "raw IO commitment mismatch");

    let in_rows: u32 = (*raw_io.at(0)).try_into().unwrap();
    let in_cols: u32 = (*raw_io.at(1)).try_into().unwrap();
    let in_len: u32 = (*raw_io.at(2)).try_into().unwrap();
    assert!(in_len == in_rows * in_cols, "input length mismatch");
    let out_header = 3 + in_len;
    assert!(out_header + 2 < raw_io.len(), "output header underrun");
    let out_rows: u32 = (*raw_io.at(out_header)).try_into().unwrap();
    let out_cols: u32 = (*raw_io.at(out_header + 1)).try_into().unwrap();
    let out_len: u32 = (*raw_io.at(out_header + 2)).try_into().unwrap();
    assert!(out_len == out_rows * out_cols, "output length mismatch");
    assert!(proof.raw_io_data.len() == out_header + 3 + out_len, "raw IO trailing mismatch");

    let mut ch = channel_default();
    if proof.has_kv_cache {
        assert!(step.kv_commitment != 0, "missing KV commitment");
        channel_mix_felt(ref ch, step.kv_commitment);
        channel_mix_felt(ref ch, step.prev_kv_commitment);
    }
    channel_mix_u64(ref ch, proof.circuit_depth.into());
    channel_mix_u64(ref ch, in_rows.into());
    channel_mix_u64(ref ch, in_cols.into());
    channel_mix_felt(ref ch, policy_commitment);

    let padded_out_rows = next_power_of_two(out_rows);
    let padded_out_cols = next_power_of_two(out_cols);
    let out_n_vars = log2_ceil(padded_out_rows) + log2_ceil(padded_out_cols);
    let r_out = channel_draw_qm31s(ref ch, out_n_vars);
    let out_data_off = out_header + 3;
    let output_value = evaluate_mle_from_io_span_2d(
        raw_io, out_data_off, out_rows, out_cols, padded_out_rows, padded_out_cols, r_out.span(),
    );
    channel_mix_secure_field(ref ch, output_value);
    let initial_claim = GKRClaim { point: r_out, value: output_value };

    let (final_claim, weight_claims, layer_tags, deferred_weight_commitments) =
        verify_gkr_model_with_trace_dp(
        proof.proof_data.span(),
        proof.num_layers,
        proof.matmul_dims.span(),
        proof.dequantize_bits.span(),
        initial_claim,
        ref ch,
        proof.packed,
        proof.double_packed,
    );

    let observed_circuit_hash = compute_circuit_hash(proof.circuit_depth, layer_tags.span());
    assert!(observed_circuit_hash == circuit_hash, "circuit hash mismatch");

    let mut all_weight_commitments: Array<felt252> = array![];
    let mut wc_i: u32 = 0;
    loop {
        if wc_i >= proof.weight_commitments.len() {
            break;
        }
        all_weight_commitments.append(*proof.weight_commitments.at(wc_i));
        wc_i += 1;
    }
    let deferred_span = deferred_weight_commitments.span();
    let mut dw_i: u32 = 0;
    loop {
        if dw_i >= deferred_span.len() {
            break;
        }
        all_weight_commitments.append(*deferred_span.at(dw_i));
        dw_i += 1;
    }

    let weight_claims_span = weight_claims.span();
    verify_weight_binding(
        weight_super_root,
        weight_claims_span,
        all_weight_commitments.span(),
        ref ch,
        proof.weight_binding_data.span(),
    );

    let padded_in_rows = next_power_of_two(in_rows);
    let padded_in_cols = next_power_of_two(in_cols);
    let expected_in_vars = log2_ceil(padded_in_rows) + log2_ceil(padded_in_cols);
    let final_point_span = final_claim.point.span();
    assert!(final_point_span.len() == expected_in_vars, "input claim point size mismatch");
    let input_value = evaluate_mle_from_io_span_2d(
        raw_io, 3, in_rows, in_cols, padded_in_rows, padded_in_cols, final_point_span,
    );
    assert!(qm31_eq(input_value, final_claim.value), "input MLE mismatch");

    GkrVerificationOutput {
        model_id,
        io_commitment,
        weight_commitment: weight_super_root,
        num_layers: proof.num_layers,
        num_matmuls: weight_claims_span.len(),
        verified: true,
    }
}

fn verify_weight_binding(
    weight_super_root: felt252,
    weight_claims: Span<WeightClaimData>,
    weight_commitments: Span<felt252>,
    ref ch: elo_cairo_verifier::channel::PoseidonChannel,
    binding_data: Span<felt252>,
) {
    assert!(binding_data.len() != 0, "missing weight binding data");
    let mut binding_span = binding_data;

    if *binding_span.at(0) == GROUPED_BINDING_MARKER {
        binding_span.pop_front().unwrap();
        let n_groups: u32 = Serde::<u32>::deserialize(ref binding_span).unwrap();
        assert!(n_groups != 0, "empty weight binding groups");

        let mut group_root_acc = sequence_init(DOMAIN_WEIGHT_GROUPS, n_groups.into());
        let mut claim_offset: u32 = 0;
        let mut group_idx: u32 = 0;
        loop {
            if group_idx >= n_groups {
                break;
            }

            let binding_proof = deserialize_binding_proof_packed(ref binding_span);
            let group_n_claims = binding_proof.config.n_claims;
            assert!(group_n_claims != 0, "empty weight binding group");
            assert!(claim_offset + group_n_claims <= weight_claims.len(), "group claims overrun");
            assert!(
                claim_offset + group_n_claims <= weight_commitments.len(),
                "group commitments overrun",
            );

            let group_claims = copy_weight_claims(weight_claims, claim_offset, group_n_claims);
            let group_commitments = copy_felts(weight_commitments, claim_offset, group_n_claims);
            let binding_ok = verify_aggregated_binding(
                @binding_proof, group_claims.span(), group_commitments.span(), ref ch,
            );
            assert!(binding_ok, "grouped weight binding failed");
            group_root_acc =
                sequence_push(
                    DOMAIN_WEIGHT_GROUPS,
                    group_root_acc,
                    group_idx.into(),
                    binding_proof.super_root,
                );

            claim_offset += group_n_claims;
            group_idx += 1;
        }

        assert!(binding_span.len() == 0, "weight binding trailing data");
        assert!(claim_offset == weight_claims.len(), "unbound weight claims");
        assert!(claim_offset == weight_commitments.len(), "unbound weight commitments");
        assert!(group_root_acc == weight_super_root, "grouped weight root mismatch");
    } else {
        let binding_proof = deserialize_binding_proof_packed(ref binding_span);
        assert!(binding_span.len() == 0, "weight binding trailing data");
        assert!(binding_proof.super_root == weight_super_root, "weight super-root mismatch");
        let binding_ok = verify_aggregated_binding(
            @binding_proof, weight_claims, weight_commitments, ref ch,
        );
        assert!(binding_ok, "aggregated weight binding failed");
    }
}

fn copy_weight_claims(
    claims: Span<WeightClaimData>, offset: u32, len: u32,
) -> Array<WeightClaimData> {
    let mut out: Array<WeightClaimData> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= len {
            break;
        }
        let claim = claims.at(offset + i);
        out
            .append(
                WeightClaimData {
                    eval_point: clone_qm31_array(claim.eval_point),
                    expected_value: *claim.expected_value,
                },
            );
        i += 1;
    }
    out
}

fn clone_qm31_array(values: @Array<QM31>) -> Array<QM31> {
    let mut out: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= values.len() {
            break;
        }
        out.append(*values.at(i));
        i += 1;
    }
    out
}

fn copy_felts(values: Span<felt252>, offset: u32, len: u32) -> Array<felt252> {
    let mut out: Array<felt252> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= len {
            break;
        }
        out.append(*values.at(offset + i));
        i += 1;
    }
    out
}

fn compute_circuit_hash(circuit_depth: u32, layer_tags: Span<u32>) -> felt252 {
    let mut tags_hash: felt252 = 0;
    let mut i: u32 = 0;
    loop {
        if i >= layer_tags.len() {
            break;
        }
        tags_hash = poseidon_hash_span(array![tags_hash, (*layer_tags.at(i)).into()].span());
        i += 1;
    }
    poseidon_hash_span(array![circuit_depth.into(), tags_hash].span())
}

fn validate_gkr_output(
    model_id: felt252, weight_super_root: felt252, step: Step, output: @GkrVerificationOutput,
) {
    assert!(*output.verified, "GKR proof did not verify");
    assert!(*output.model_id == model_id, "GKR model mismatch");
    assert!(*output.io_commitment == step.io_commitment, "GKR IO mismatch");
    assert!(*output.weight_commitment == weight_super_root, "GKR weight mismatch");
    let receipt_hash = ml_receipt_hash(output);
    assert!(step.ml_receipt_hash == receipt_hash, "GKR receipt mismatch");
}

fn ml_receipt_hash(output: @GkrVerificationOutput) -> felt252 {
    let verified_felt = if *output.verified {
        1
    } else {
        0
    };
    poseidon_hash_span(
        array![
            DOMAIN_ML_RECEIPT, *output.model_id, *output.io_commitment, *output.weight_commitment,
            (*output.num_layers).into(), (*output.num_matmuls).into(), verified_felt,
        ]
            .span(),
    )
}

fn sequence_init(domain: felt252, len: felt252) -> felt252 {
    poseidon_hash_span(array![DOMAIN_SEQ, domain, len].span())
}

fn sequence_push(domain: felt252, prev: felt252, index: felt252, commitment: felt252) -> felt252 {
    poseidon_hash_span(array![DOMAIN_SEQ, domain, prev, index, commitment].span())
}

fn commit_step(step: Step) -> felt252 {
    poseidon_hash_span(
        array![
            DOMAIN_STEP, step.global_step_index, step.conversation_index, step.turn_index,
            step.token_index, step.generated_token_id, step.io_commitment, step.sampling_commitment,
            step.prev_kv_commitment, step.kv_commitment, step.ml_receipt_hash,
        ]
            .span(),
    )
}

fn commit_action(action: Action) -> felt252 {
    poseidon_hash_span(
        array![
            DOMAIN_ACTION, action.conversation_index, action.turn_index, action.action_index,
            action.action_kind_hash, action.tool_name_hash, action.input_commitment,
            action.output_commitment, action.policy_commitment,
        ]
            .span(),
    )
}

fn commit_conversation(conversation: Conversation) -> felt252 {
    poseidon_hash_span(
        array![
            DOMAIN_CONVERSATION, conversation.conversation_index, conversation.conversation_id_hash,
            conversation.prompt_commitment, conversation.transcript_commitment,
            conversation.action_root, conversation.initial_kv_commitment,
            conversation.final_kv_commitment, conversation.n_turns, conversation.n_prefill_tokens,
            conversation.n_generated_tokens, conversation.first_step_index, conversation.n_steps,
        ]
            .span(),
    )
}

fn validate_statement_consistency(
    conversations: Span<Conversation>, steps: Span<Step>, actions: Span<Action>,
) {
    let mut expected_step_index: felt252 = 0;
    let mut total_generated: felt252 = 0;
    let mut conv_idx: u32 = 0;
    loop {
        if conv_idx >= conversations.len() {
            break;
        }
        let conversation = *conversations.at(conv_idx);
        assert!(conversation.conversation_index == conv_idx.into(), "conversation order");
        assert!(conversation.first_step_index == expected_step_index, "first step mismatch");

        let mut prev_kv = conversation.initial_kv_commitment;
        let n_steps: u32 = conversation.n_steps.try_into().unwrap();
        let mut local_step: u32 = 0;
        loop {
            if local_step >= n_steps {
                break;
            }
            let global_idx_felt = conversation.first_step_index + local_step.into();
            let global_idx: u32 = global_idx_felt.try_into().unwrap();
            assert!(global_idx < steps.len(), "step out of range");
            let step = *steps.at(global_idx);
            assert!(step.global_step_index == global_idx_felt, "global step mismatch");
            assert!(
                step.conversation_index == conversation.conversation_index,
                "step conversation mismatch",
            );
            assert!(step.prev_kv_commitment == prev_kv, "KV chain break");
            prev_kv = step.kv_commitment;
            local_step += 1;
        }
        assert!(prev_kv == conversation.final_kv_commitment, "final KV mismatch");
        assert!(conversation.n_steps == conversation.n_generated_tokens, "step/token mismatch");
        assert!(
            conversation
                .action_root == conversation_action_root(conversation.conversation_index, actions),
            "conversation action root mismatch",
        );

        expected_step_index += conversation.n_steps;
        total_generated += conversation.n_generated_tokens;
        conv_idx += 1;
    }
    assert!(expected_step_index == steps.len().into(), "unused steps");
    assert!(total_generated == steps.len().into(), "generated count mismatch");

    let mut action_idx: u32 = 0;
    loop {
        if action_idx >= actions.len() {
            break;
        }
        let action = *actions.at(action_idx);
        let conv_i: u32 = action.conversation_index.try_into().unwrap();
        assert!(conv_i < conversations.len(), "action conversation out of range");
        let conversation = *conversations.at(conv_i);
        let action_turn: u32 = action.turn_index.try_into().unwrap();
        let n_turns: u32 = conversation.n_turns.try_into().unwrap();
        assert!(action_turn < n_turns, "action turn out of range");
        assert!(action.policy_commitment != 0, "action policy missing");
        action_idx += 1;
    };
}

fn conversation_action_root(conversation_index: felt252, actions: Span<Action>) -> felt252 {
    let mut count: u32 = 0;
    let mut i: u32 = 0;
    loop {
        if i >= actions.len() {
            break;
        }
        let action = *actions.at(i);
        if action.conversation_index == conversation_index {
            count += 1;
        }
        i += 1;
    }

    let mut acc = sequence_init(DOMAIN_ACTION, count.into());
    let mut local_index: u32 = 0;
    let mut j: u32 = 0;
    loop {
        if j >= actions.len() {
            break;
        }
        let action = *actions.at(j);
        if action.conversation_index == conversation_index {
            assert!(action.action_index == local_index.into(), "action index mismatch");
            acc = sequence_push(DOMAIN_ACTION, acc, local_index.into(), commit_action(action));
            local_index += 1;
        }
        j += 1;
    }
    acc
}
