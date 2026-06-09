//! Strict conversation statement verifier with inline ML verifier execution.
//!
//! This is the production STARK-in-STARK target: the Cairo execution proven by
//! `cairo-prove --recursive-160` verifies every supplied `MLProof`, checks each
//! verifier output against its generation row, then emits the canonical 19-felt
//! conversation statement.

use core::poseidon::poseidon_hash_span;
use obelysk_ml_air::claim::{MLProofV2, MLVerificationOutput};
use obelysk_ml_air::verify_ml_v2;

const DOMAIN_SEQ: felt252 = 0x53455131; // "SEQ1"
const DOMAIN_STEP: felt252 = 0x53544550; // "STEP"
const DOMAIN_ACTION: felt252 = 0x4143544E; // "ACTN"
const DOMAIN_CONVERSATION: felt252 = 0x434F4E56; // "CONV"
const DOMAIN_BATCH: felt252 = 0x43424154; // "CBAT"
const DOMAIN_KV_ROOT: felt252 = 0x4B565254; // "KVRT"
const DOMAIN_ML_RECEIPT: felt252 = 0x4D4C5243; // "MLRC"
const STATEMENT_VERSION: felt252 = 1;

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
struct ConversationMlWitness {
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
    ml_proofs: Array<MLProofV2>,
}

#[executable]
fn main(witness: ConversationMlWitness) -> Array<felt252> {
    let ConversationMlWitness {
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
        mut ml_proofs,
    } = witness;

    let security_bits_u32: u32 = security_bits.try_into().unwrap();
    assert!(model_id != 0, "model_id cannot be zero");
    assert!(circuit_hash != 0, "circuit_hash cannot be zero");
    assert!(weight_super_root != 0, "weight root cannot be zero");
    assert!(policy_commitment != 0, "policy cannot be zero");
    assert!(security_bits_u32 >= 160, "security below 160");
    assert!(conversations.len() != 0, "no conversations");
    assert!(ml_proofs.len() == steps.len(), "ML proof count mismatch");

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
        assert!(conversation.conversation_index == conversation_index.into(), "conversation index mismatch");
        conversation_root_acc = sequence_push(
            DOMAIN_CONVERSATION,
            conversation_root_acc,
            conversation_index.into(),
            commit_conversation(conversation),
        );
        initial_kv_root_acc = sequence_push(
            DOMAIN_KV_ROOT,
            initial_kv_root_acc,
            conversation_index.into(),
            conversation.initial_kv_commitment,
        );
        final_kv_root_acc = sequence_push(
            DOMAIN_KV_ROOT,
            final_kv_root_acc,
            conversation_index.into(),
            conversation.final_kv_commitment,
        );
        total_prefill_tokens += conversation.n_prefill_tokens;
        total_generated_tokens += conversation.n_generated_tokens;
        conversation_index += 1;
    };

    let mut generation_root_acc = sequence_init(DOMAIN_STEP, steps.len().into());
    let mut step_index: u32 = 0;
    loop {
        if step_index >= steps.len() {
            break;
        }
        let step = *steps.at(step_index);
        assert!(step.global_step_index == step_index.into(), "step index mismatch");

        let ml_proof = ml_proofs.pop_front().unwrap();
        let ml_output = verify_ml_v2(ml_proof);
        validate_ml_output(model_id, weight_super_root, step, @ml_output);

        generation_root_acc = sequence_push(
            DOMAIN_STEP,
            generation_root_acc,
            step_index.into(),
            commit_step(step),
        );
        step_index += 1;
    };
    assert!(ml_proofs.len() == 0, "unused ML proofs");

    let mut action_root_acc = sequence_init(DOMAIN_ACTION, actions.len().into());
    let mut action_index: u32 = 0;
    loop {
        if action_index >= actions.len() {
            break;
        }
        let action = *actions.at(action_index);
        action_root_acc = sequence_push(
            DOMAIN_ACTION,
            action_root_acc,
            action_index.into(),
            commit_action(action),
        );
        action_index += 1;
    };

    validate_statement_consistency(conversations.span(), steps.span(), actions.span());

    array![
        DOMAIN_BATCH,
        STATEMENT_VERSION,
        model_id,
        verifier_program_hash,
        circuit_hash,
        weight_super_root,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        conversation_root_acc,
        generation_root_acc,
        action_root_acc,
        initial_kv_root_acc,
        final_kv_root_acc,
        conversations.len().into(),
        steps.len().into(),
        total_prefill_tokens,
        total_generated_tokens,
        security_bits,
    ]
}

fn validate_ml_output(
    model_id: felt252, weight_super_root: felt252, step: Step, output: @MLVerificationOutput,
) {
    assert!(*output.verified, "ML proof did not verify");
    assert!(*output.model_id == model_id, "ML model mismatch");
    assert!(*output.io_commitment == step.io_commitment, "ML IO mismatch");
    assert!(*output.weight_commitment == weight_super_root, "ML weight mismatch");
    let receipt_hash = ml_receipt_hash(output);
    assert!(step.ml_receipt_hash == receipt_hash, "ML receipt mismatch");
}

fn ml_receipt_hash(output: @MLVerificationOutput) -> felt252 {
    let verified_felt = if *output.verified {
        1
    } else {
        0
    };
    poseidon_hash_span(
        array![
            DOMAIN_ML_RECEIPT,
            *output.model_id,
            *output.io_commitment,
            *output.weight_commitment,
            (*output.num_layers).into(),
            (*output.num_matmuls).into(),
            verified_felt,
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
            DOMAIN_STEP,
            step.global_step_index,
            step.conversation_index,
            step.turn_index,
            step.token_index,
            step.generated_token_id,
            step.io_commitment,
            step.sampling_commitment,
            step.prev_kv_commitment,
            step.kv_commitment,
            step.ml_receipt_hash,
        ]
            .span(),
    )
}

fn commit_action(action: Action) -> felt252 {
    poseidon_hash_span(
        array![
            DOMAIN_ACTION,
            action.conversation_index,
            action.turn_index,
            action.action_index,
            action.action_kind_hash,
            action.tool_name_hash,
            action.input_commitment,
            action.output_commitment,
            action.policy_commitment,
        ]
            .span(),
    )
}

fn commit_conversation(conversation: Conversation) -> felt252 {
    poseidon_hash_span(
        array![
            DOMAIN_CONVERSATION,
            conversation.conversation_index,
            conversation.conversation_id_hash,
            conversation.prompt_commitment,
            conversation.transcript_commitment,
            conversation.action_root,
            conversation.initial_kv_commitment,
            conversation.final_kv_commitment,
            conversation.n_turns,
            conversation.n_prefill_tokens,
            conversation.n_generated_tokens,
            conversation.first_step_index,
            conversation.n_steps,
        ]
            .span(),
    )
}

fn validate_statement_consistency(
    conversations: Span<Conversation>, steps: Span<Step>, actions: Span<Action>,
) {
    let mut i: u32 = 0;
    let mut covered_steps: u32 = 0;
    loop {
        if i >= conversations.len() {
            break;
        }
        let conversation = *conversations.at(i);
        let first_step_index: u32 = conversation.first_step_index.try_into().unwrap();
        let n_steps: u32 = conversation.n_steps.try_into().unwrap();
        assert!(conversation.conversation_index == i.into(), "conversation index mismatch");
        assert!(first_step_index == covered_steps, "conversation step range gap");
        assert!(conversation.n_generated_tokens == conversation.n_steps, "generated token count mismatch");

        let mut expected_prev = conversation.initial_kv_commitment;
        let mut local_offset: u32 = 0;
        loop {
            if local_offset >= n_steps {
                break;
            }
            let step = *steps.at(first_step_index + local_offset);
            assert!(step.conversation_index == conversation.conversation_index, "step conversation mismatch");
            assert!(step.prev_kv_commitment == expected_prev, "KV continuity mismatch");
            expected_prev = step.kv_commitment;
            local_offset += 1;
        };
        assert!(expected_prev == conversation.final_kv_commitment, "final KV mismatch");
        assert!(
            conversation.action_root == conversation_action_root(conversation.conversation_index, actions),
            "conversation action root mismatch",
        );

        covered_steps += n_steps;
        i += 1;
    };
    assert!(covered_steps == steps.len(), "uncovered generation steps");
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
    };

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
    };
    acc
}
