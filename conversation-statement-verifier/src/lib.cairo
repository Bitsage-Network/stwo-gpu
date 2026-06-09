//! Conversation statement verifier executable.
//!
//! This is the Cairo program that should be proven by `cairo-prove --recursive-160`
//! for STARK-in-STARK conversation/action verification. It consumes a flat
//! statement witness, checks KV continuity and ordered roots, and returns the
//! 19 canonical statement felts. With `poseidon_outputs_packing`, the on-chain
//! STWO verifier sees `output_hash = poseidon_hash_span(statement_felts)`.

use core::poseidon::poseidon_hash_span;

const DOMAIN_SEQ: felt252 = 0x53455131; // "SEQ1"
const DOMAIN_STEP: felt252 = 0x53544550; // "STEP"
const DOMAIN_ACTION: felt252 = 0x4143544E; // "ACTN"
const DOMAIN_CONVERSATION: felt252 = 0x434F4E56; // "CONV"
const DOMAIN_BATCH: felt252 = 0x43424154; // "CBAT"
const DOMAIN_KV_ROOT: felt252 = 0x4B565254; // "KVRT"
const STATEMENT_VERSION: felt252 = 1;
const PRODUCTION_SECURITY_BITS: felt252 = 160;

#[derive(Copy, Drop)]
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

#[derive(Copy, Drop)]
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
    recursive_proof_hash: felt252,
}

#[derive(Copy, Drop)]
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

#[executable]
fn main(input: Array<felt252>) -> Array<felt252> {
    let mut span = input.span();

    let model_id = next(ref span);
    let verifier_program_hash = next(ref span);
    let circuit_hash = next(ref span);
    let weight_super_root = next(ref span);
    let policy_commitment = next(ref span);
    let tokenizer_config_hash = next(ref span);
    let hades_commitment = next(ref span);
    let security_bits = next(ref span);
    let n_conversations = next(ref span);
    let n_steps = next(ref span);
    let n_actions = next(ref span);
    let security_bits_u32: u32 = security_bits.try_into().unwrap();
    let n_conversations_u32: u32 = n_conversations.try_into().unwrap();
    let n_steps_u32: u32 = n_steps.try_into().unwrap();
    let n_actions_u32: u32 = n_actions.try_into().unwrap();

    assert!(model_id != 0, "model_id cannot be zero");
    assert!(circuit_hash != 0, "circuit_hash cannot be zero");
    assert!(weight_super_root != 0, "weight root cannot be zero");
    assert!(policy_commitment != 0, "policy cannot be zero");
    assert!(security_bits_u32 >= 160, "security below 160");
    assert!(n_conversations != 0, "no conversations");

    let mut conversation_root_acc = sequence_init(DOMAIN_CONVERSATION, n_conversations);
    let mut initial_kv_root_acc = sequence_init(DOMAIN_KV_ROOT, n_conversations);
    let mut final_kv_root_acc = sequence_init(DOMAIN_KV_ROOT, n_conversations);
    let mut total_prefill_tokens: felt252 = 0;
    let mut total_generated_tokens: felt252 = 0;

    let mut conversations: Array<Conversation> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= n_conversations_u32 {
            break;
        }
        let conversation = read_conversation(ref span);
        assert!(conversation.conversation_index == i.into(), "conversation index mismatch");
        let commitment = commit_conversation(conversation);
        conversation_root_acc = sequence_push(
            DOMAIN_CONVERSATION, conversation_root_acc, i.into(), commitment,
        );
        initial_kv_root_acc = sequence_push(
            DOMAIN_KV_ROOT, initial_kv_root_acc, i.into(), conversation.initial_kv_commitment,
        );
        final_kv_root_acc = sequence_push(
            DOMAIN_KV_ROOT, final_kv_root_acc, i.into(), conversation.final_kv_commitment,
        );
        total_prefill_tokens += conversation.n_prefill_tokens;
        total_generated_tokens += conversation.n_generated_tokens;
        conversations.append(conversation);
        i += 1;
    };

    let conversation_root = conversation_root_acc;
    let initial_kv_root = initial_kv_root_acc;
    let final_kv_root = final_kv_root_acc;

    let mut generation_root_acc = sequence_init(DOMAIN_STEP, n_steps);
    let mut steps: Array<Step> = array![];
    let mut step_index: u32 = 0;
    loop {
        if step_index >= n_steps_u32 {
            break;
        }
        let step = read_step(ref span);
        assert!(step.global_step_index == step_index.into(), "step index mismatch");
        generation_root_acc = sequence_push(
            DOMAIN_STEP, generation_root_acc, step_index.into(), commit_step(step),
        );
        steps.append(step);
        step_index += 1;
    };
    let generation_root = generation_root_acc;

    let mut action_root_acc = sequence_init(DOMAIN_ACTION, n_actions);
    let mut actions: Array<Action> = array![];
    let mut action_index: u32 = 0;
    loop {
        if action_index >= n_actions_u32 {
            break;
        }
        let action = read_action(ref span);
        action_root_acc = sequence_push(
            DOMAIN_ACTION, action_root_acc, action_index.into(), commit_action(action),
        );
        actions.append(action);
        action_index += 1;
    };
    let action_root = action_root_acc;

    assert!(span.is_empty(), "trailing input");
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
        conversation_root,
        generation_root,
        action_root,
        initial_kv_root,
        final_kv_root,
        n_conversations,
        n_steps,
        total_prefill_tokens,
        total_generated_tokens,
        security_bits,
    ]
}

fn next(ref span: Span<felt252>) -> felt252 {
    *span.pop_front().unwrap()
}

fn read_conversation(ref span: Span<felt252>) -> Conversation {
    Conversation {
        conversation_index: next(ref span),
        conversation_id_hash: next(ref span),
        prompt_commitment: next(ref span),
        transcript_commitment: next(ref span),
        action_root: next(ref span),
        initial_kv_commitment: next(ref span),
        final_kv_commitment: next(ref span),
        n_turns: next(ref span),
        n_prefill_tokens: next(ref span),
        n_generated_tokens: next(ref span),
        first_step_index: next(ref span),
        n_steps: next(ref span),
    }
}

fn read_step(ref span: Span<felt252>) -> Step {
    Step {
        global_step_index: next(ref span),
        conversation_index: next(ref span),
        turn_index: next(ref span),
        token_index: next(ref span),
        generated_token_id: next(ref span),
        io_commitment: next(ref span),
        sampling_commitment: next(ref span),
        prev_kv_commitment: next(ref span),
        kv_commitment: next(ref span),
        recursive_proof_hash: next(ref span),
    }
}

fn read_action(ref span: Span<felt252>) -> Action {
    Action {
        conversation_index: next(ref span),
        turn_index: next(ref span),
        action_index: next(ref span),
        action_kind_hash: next(ref span),
        tool_name_hash: next(ref span),
        input_commitment: next(ref span),
        output_commitment: next(ref span),
        policy_commitment: next(ref span),
    }
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
            step.recursive_proof_hash,
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
    let n_conversations: u32 = conversations.len().try_into().unwrap();
    let mut covered_steps: u32 = 0;
    loop {
        if i >= n_conversations {
            break;
        }
        let conversation = *conversations.at(i);
        assert!(conversation.conversation_index == i.into(), "conversation index mismatch");

        let first_step_index: u32 = conversation.first_step_index.try_into().unwrap();
        let n_steps: u32 = conversation.n_steps.try_into().unwrap();
        assert!(first_step_index == covered_steps, "conversation step range gap");
        assert!(
            conversation.n_generated_tokens == conversation.n_steps,
            "generated token count mismatch",
        );

        let mut expected_prev = conversation.initial_kv_commitment;
        let mut local_offset: u32 = 0;
        loop {
            if local_offset >= n_steps {
                break;
            }
            let step = *steps.at(first_step_index + local_offset);
            assert!(
                step.conversation_index == conversation.conversation_index,
                "step conversation mismatch",
            );
            assert!(step.prev_kv_commitment == expected_prev, "KV continuity mismatch");
            expected_prev = step.kv_commitment;
            local_offset += 1;
        };
        assert!(expected_prev == conversation.final_kv_commitment, "final KV mismatch");

        let expected_action_root = conversation_action_root(conversation.conversation_index, actions);
        assert!(conversation.action_root == expected_action_root, "conversation action root mismatch");

        covered_steps += n_steps;
        i += 1;
    };
    let total_steps: u32 = steps.len().try_into().unwrap();
    assert!(covered_steps == total_steps, "uncovered generation steps");
}

fn conversation_action_root(conversation_index: felt252, actions: Span<Action>) -> felt252 {
    let n_actions: u32 = actions.len().try_into().unwrap();
    let mut count: u32 = 0;
    let mut i: u32 = 0;
    loop {
        if i >= n_actions {
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
        if j >= n_actions {
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
