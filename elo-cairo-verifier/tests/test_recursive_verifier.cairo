use elo_cairo_verifier::recursive_verifier::{
    IRecursiveVerifierDispatcher, IRecursiveVerifierDispatcherTrait,
};
use snforge_std::{ContractClassTrait, DeclareResultTrait, declare, start_cheat_caller_address};
use starknet::ContractAddress;

const OWNER_ADDR: felt252 = 0x1234;
const ATTACKER_ADDR: felt252 = 0xBAD;
const MODEL_ID: felt252 = 0xABC;
const CIRCUIT_HASH: felt252 = 0x123456;
const WEIGHT_ROOT: felt252 = 0x789ABC;
const IO_COMMITMENT: felt252 = 0xDEF123;
const POLICY_COMMITMENT: felt252 = 0x0370c9;
const N_LAYERS: u32 = 30;
const TRACE_LOG_SIZE: u32 = 14;
const N_MATMULS: u32 = 192;
const HIDDEN_SIZE: u32 = 5120;
const NUM_TRANSFORMER_BLOCKS: u32 = 48;
const EXPECTED_N_POSEIDON_PERMS: u32 = 145;
const LEVEL1_PROOF_HASH: felt252 = 0xCAFE;

fn deploy_verifier() -> IRecursiveVerifierDispatcher {
    let contract = declare("RecursiveVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into()]).unwrap();
    IRecursiveVerifierDispatcher { contract_address: address }
}

fn as_owner(verifier: @IRecursiveVerifierDispatcher) {
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*verifier.contract_address, owner);
}

/// Build a fake proof header (30 felts) with specified circuit_hash and weight_root.
/// Values are placed in the low limb (ch3/wr3) for simplicity.
/// This proof will fail at STARK verification but is sufficient to test
/// pre-STARK asserts (circuit hash, weight binding, io commitment).
fn build_fake_proof(
    circuit_hash: felt252, io_commit: felt252, weight_root: felt252,
) -> Array<felt252> {
    let mut data: Array<felt252> = array![];
    // circuit_hash: QM31 as 4 M31 limbs [ch0, ch1, ch2, ch3]
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(circuit_hash);
    // io_commitment: QM31 as 4 M31 limbs
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(io_commit);
    // weight_super_root: QM31 as 4 M31 limbs
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(weight_root);
    // n_layers, n_poseidon_perms
    data.append(N_LAYERS.into());
    data.append(EXPECTED_N_POSEIDON_PERMS.into());
    // seed_digest: QM31 as 4 M31 limbs
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    // hades_commitment, full io_commitment, pass1_final_digest
    data.append(0);
    data.append(io_commit);
    data.append(0x9999);
    // final_digest, log_size, chain/arithmetic/sumcheck/draw row counts
    data.append(0x1234);
    data.append(TRACE_LOG_SIZE.into());
    data.append(EXPECTED_N_POSEIDON_PERMS.into());
    data.append(0);
    data.append(0);
    data.append(0);
    // logup_claimed_sum
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    // prev_kv_cache_commitment, kv_cache_commitment, conversation_statement_hash
    data.append(0);
    data.append(0);
    data.append(0);
    data
}

// ═══════════════════════════════════════════════════════════════
// GROUP A: Registration CRUD
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_register_recursive_model() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert!(info.circuit_hash == CIRCUIT_HASH, "circuit_hash mismatch");
    assert!(info.weight_super_root == WEIGHT_ROOT, "weight_root mismatch");

    let count = verifier.get_recursive_verification_count(MODEL_ID);
    assert!(count == 0, "count should start at 0");
}

#[test]
#[should_panic(expected: 'Only owner can register')]
fn test_register_recursive_model_non_owner_rejected() {
    let verifier = deploy_verifier();
    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(verifier.contract_address, attacker);

    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
}

#[test]
fn test_query_unregistered_model_returns_zero() {
    let verifier = deploy_verifier();
    let info = verifier.get_recursive_model_info(0xDEAD);
    assert!(info.circuit_hash == 0, "unregistered model should have circuit_hash=0");
}

#[test]
fn test_is_recursive_proof_verified_default_false() {
    let verifier = deploy_verifier();
    let verified = verifier.is_recursive_proof_verified(0x999);
    assert!(!verified, "should default to false");
}

#[test]
fn test_verification_count_default_zero() {
    let verifier = deploy_verifier();
    let count = verifier.get_recursive_verification_count(0xDEAD);
    assert!(count == 0, "unregistered model count should be 0");
}

// ═══════════════════════════════════════════════════════════════
// GROUP A2: Registration Edge Cases
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_register_multiple_models() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            0x1,
            0xAA,
            0xBB,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
    verifier
        .register_model_recursive(
            0x2,
            0xCC,
            0xDD,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let info1 = verifier.get_recursive_model_info(0x1);
    let info2 = verifier.get_recursive_model_info(0x2);

    assert!(info1.circuit_hash == 0xAA, "model 1 circuit_hash wrong");
    assert!(info2.circuit_hash == 0xCC, "model 2 circuit_hash wrong");
    assert!(info1.weight_super_root == 0xBB, "model 1 weight_root wrong");
    assert!(info2.weight_super_root == 0xDD, "model 2 weight_root wrong");
}

#[test]
#[should_panic(expected: 'Model already registered')]
fn test_re_register_model_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            MODEL_ID,
            0x111,
            0x222,
            0,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
    verifier
        .register_model_recursive(
            MODEL_ID,
            0x333,
            0x444,
            0,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
}

#[test]
fn test_register_model_owner_stored_correctly() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let info = verifier.get_recursive_model_info(MODEL_ID);
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    assert!(info.owner == owner, "owner should be the registrar");
}

#[test]
fn test_register_with_zero_circuit_hash() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Zero circuit_hash is allowed at registration time
    // but verify_recursive will reject it with "Model not registered"
    // because the check is `model.circuit_hash != 0`
    verifier
        .register_model_recursive(
            MODEL_ID,
            0,
            WEIGHT_ROOT,
            0,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert!(info.circuit_hash == 0, "zero circuit_hash should be stored");
}

// ═══════════════════════════════════════════════════════════════
// GROUP B: Pre-STARK Adversarial Rejection Tests
// ═══════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected: 'Model not registered')]
fn test_verify_unregistered_model_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Submit proof for model that was never registered
    let fake_proof = build_fake_proof(CIRCUIT_HASH, IO_COMMITMENT, WEIGHT_ROOT);
    verifier
        .verify_recursive(
            0xDEAD,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            fake_proof,
        );
}

#[test]
#[should_panic(expected: "Proof too short")]
fn test_verify_proof_too_short() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    // Submit proof with only 10 felts (need >= 34)
    let short_proof: Array<felt252> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    verifier
        .verify_recursive(
            MODEL_ID,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            short_proof,
        );
}

#[test]
#[should_panic(expected: 'Circuit hash mismatch')]
fn test_verify_circuit_hash_mismatch() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register model with CIRCUIT_HASH
    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    // Build proof with DIFFERENT circuit_hash (0xBADBAD instead of 0x123456)
    let tampered_proof = build_fake_proof(0xBADBAD, IO_COMMITMENT, WEIGHT_ROOT);
    verifier
        .verify_recursive(
            MODEL_ID,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            tampered_proof,
        );
}

#[test]
#[should_panic(expected: 'Weight binding mismatch')]
fn test_verify_weight_binding_mismatch() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register model with WEIGHT_ROOT
    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    // Build proof with correct circuit_hash but WRONG weight_root
    let tampered_proof = build_fake_proof(CIRCUIT_HASH, IO_COMMITMENT, 0xBADBAD);
    verifier
        .verify_recursive(
            MODEL_ID,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            tampered_proof,
        );
}

// NOTE: io_commitment is NOT directly comparable between the parameter (Poseidon hash)
// and the proof header (QM31 limbs). The STARK proof binds IO through Fiat-Shamir.
// A separate io_commitment mismatch test would require matching encodings.

#[test]
#[should_panic(expected: "Proof too short")]
fn test_verify_empty_proof_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);
    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let empty_proof: Array<felt252> = array![];
    verifier
        .verify_recursive(
            MODEL_ID,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            empty_proof,
        );
}

#[test]
#[should_panic(expected: "Proof too short")]
fn test_verify_proof_boundary_33_felts_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);
    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    // Exactly 33 felts — one short of the 34 minimum
    let mut short: Array<felt252> = array![];
    let mut i: u32 = 0;
    while i < 33 {
        short.append(0);
        i += 1;
    }
    verifier
        .verify_recursive(
            MODEL_ID,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            short,
        );
}

#[test]
#[should_panic(expected: 'Model not registered')]
fn test_verify_model_with_zero_circuit_hash_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register model with circuit_hash=0, then try to verify
    // The verify_recursive check is `model.circuit_hash != 0`
    // so this should be rejected even though the model was "registered"
    verifier
        .register_model_recursive(
            MODEL_ID,
            0,
            WEIGHT_ROOT,
            0,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let fake_proof = build_fake_proof(0, IO_COMMITMENT, WEIGHT_ROOT);
    verifier
        .verify_recursive(
            MODEL_ID,
            IO_COMMITMENT,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            fake_proof,
        );
}

#[test]
#[should_panic(expected: 'Circuit hash mismatch')]
fn test_verify_swapped_models_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register two models with different circuit hashes
    verifier
        .register_model_recursive(
            0x1,
            0xAAA,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
    verifier
        .register_model_recursive(
            0x2,
            0xBBB,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    // Try to verify model 0x1 with model 0x2's circuit hash
    let wrong_proof = build_fake_proof(0xBBB, IO_COMMITMENT, WEIGHT_ROOT);
    verifier
        .verify_recursive(
            0x1,
            IO_COMMITMENT,
            0xAAA,
            WEIGHT_ROOT,
            N_LAYERS,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            POLICY_COMMITMENT,
            TRACE_LOG_SIZE,
            wrong_proof,
        );
}

// ═══════════════════════════════════════════════════════════════
// GROUP B2: Policy Commitment Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_register_model_with_policy() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            POLICY_COMMITMENT,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert!(info.policy_commitment == POLICY_COMMITMENT, "policy_commitment mismatch");

    let policy = verifier.get_model_policy(MODEL_ID);
    assert!(policy == POLICY_COMMITMENT, "get_model_policy should return policy");
}

#[test]
fn test_register_model_without_policy() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Zero policy = any policy accepted (backward compatible)
    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            0,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );

    let policy = verifier.get_model_policy(MODEL_ID);
    assert!(policy == 0, "zero policy should be stored");
}

#[test]
fn test_unregistered_model_policy_is_zero() {
    let verifier = deploy_verifier();
    let policy = verifier.get_model_policy(0xDEAD);
    assert!(policy == 0, "unregistered model policy should be 0");
}

#[test]
#[should_panic(expected: 'Model already registered')]
fn test_re_register_model_policy_update_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            0x111,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
    verifier
        .register_model_recursive(
            MODEL_ID,
            CIRCUIT_HASH,
            WEIGHT_ROOT,
            0x222,
            N_MATMULS,
            HIDDEN_SIZE,
            NUM_TRANSFORMER_BLOCKS,
            EXPECTED_N_POSEIDON_PERMS,
            LEVEL1_PROOF_HASH,
        );
}

// ═══════════════════════════════════════════════════════════════
// GROUP C: Real STARK Proof Tests
// Uses a Rust-generated tiny 1-layer proof with Hades AIR + LogUp + 160-bit PCS.
// ═══════════════════════════════════════════════════════════════

use super::test_recursive_data;

fn register_tiny_recursive_model(verifier: @IRecursiveVerifierDispatcher) {
    verifier
        .register_model_recursive(
            test_recursive_data::tiny_model_id(),
            test_recursive_data::tiny_circuit_hash(),
            test_recursive_data::tiny_weight_root(),
            test_recursive_data::tiny_policy_commitment(),
            test_recursive_data::tiny_n_matmuls(),
            test_recursive_data::tiny_hidden_size(),
            test_recursive_data::tiny_num_transformer_blocks(),
            test_recursive_data::tiny_expected_n_poseidon_perms(),
            test_recursive_data::tiny_level1_proof_hash(),
        );
}

#[test]
#[ignore] // 56,545-felt production fixture overflows Sierra when embedded in snforge source
fn test_verify_recursive_proof_valid() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    register_tiny_recursive_model(@verifier);

    let result = verifier
        .verify_recursive_with_statement(
            test_recursive_data::tiny_model_id(),
            test_recursive_data::tiny_io_commitment(),
            test_recursive_data::tiny_circuit_hash(),
            test_recursive_data::tiny_weight_root(),
            test_recursive_data::tiny_n_layers(),
            test_recursive_data::tiny_n_matmuls(),
            test_recursive_data::tiny_hidden_size(),
            test_recursive_data::tiny_num_transformer_blocks(),
            test_recursive_data::tiny_policy_commitment(),
            test_recursive_data::tiny_trace_log_size(),
            test_recursive_data::tiny_statement_hash(),
            test_recursive_data::tiny_calldata(),
        );
    assert!(result, "proof should be valid");

    let count = verifier.get_recursive_verification_count(test_recursive_data::tiny_model_id());
    assert!(count == 1, "count should be 1 after verification");
}

#[test]
#[ignore] // 56,545-felt production fixture overflows Sierra when embedded in snforge source
#[should_panic(expected: 'Already verified')]
fn test_verify_proof_replay_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    register_tiny_recursive_model(@verifier);

    verifier
        .verify_recursive_with_statement(
            test_recursive_data::tiny_model_id(),
            test_recursive_data::tiny_io_commitment(),
            test_recursive_data::tiny_circuit_hash(),
            test_recursive_data::tiny_weight_root(),
            test_recursive_data::tiny_n_layers(),
            test_recursive_data::tiny_n_matmuls(),
            test_recursive_data::tiny_hidden_size(),
            test_recursive_data::tiny_num_transformer_blocks(),
            test_recursive_data::tiny_policy_commitment(),
            test_recursive_data::tiny_trace_log_size(),
            test_recursive_data::tiny_statement_hash(),
            test_recursive_data::tiny_calldata(),
        );

    // Second submission → should panic
    verifier
        .verify_recursive_with_statement(
            test_recursive_data::tiny_model_id(),
            test_recursive_data::tiny_io_commitment(),
            test_recursive_data::tiny_circuit_hash(),
            test_recursive_data::tiny_weight_root(),
            test_recursive_data::tiny_n_layers(),
            test_recursive_data::tiny_n_matmuls(),
            test_recursive_data::tiny_hidden_size(),
            test_recursive_data::tiny_num_transformer_blocks(),
            test_recursive_data::tiny_policy_commitment(),
            test_recursive_data::tiny_trace_log_size(),
            test_recursive_data::tiny_statement_hash(),
            test_recursive_data::tiny_calldata(),
        );
}

#[test]
#[ignore] // 56,545-felt production fixture overflows Sierra when embedded in snforge source
#[should_panic] // STARK verification rejects tampered proof
fn test_verify_bit_flip_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    register_tiny_recursive_model(@verifier);

    let real = test_recursive_data::tiny_calldata();
    let mut tampered: Array<felt252> = array![];
    let real_span = real.span();
    let mut i: u32 = 0;
    loop {
        if i >= real_span.len() {
            break;
        }
        if i == 20 {
            tampered.append(*real_span.at(i) + 0xDEAD);
        } else {
            tampered.append(*real_span.at(i));
        }
        i += 1;
    }

    verifier
        .verify_recursive_with_statement(
            test_recursive_data::tiny_model_id(),
            test_recursive_data::tiny_io_commitment(),
            test_recursive_data::tiny_circuit_hash(),
            test_recursive_data::tiny_weight_root(),
            test_recursive_data::tiny_n_layers(),
            test_recursive_data::tiny_n_matmuls(),
            test_recursive_data::tiny_hidden_size(),
            test_recursive_data::tiny_num_transformer_blocks(),
            test_recursive_data::tiny_policy_commitment(),
            test_recursive_data::tiny_trace_log_size(),
            test_recursive_data::tiny_statement_hash(),
            tampered,
        );
}

// ═══════════════════════════════════════════════════════════════
// GROUP D: Upgrade Timelock Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_propose_upgrade() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    let new_class: starknet::ClassHash = 0xABCDEF.try_into().unwrap();
    verifier.propose_upgrade(new_class);

    let (pending, _ts) = verifier.get_pending_upgrade();
    assert!(pending == new_class, "pending should match proposed class");
}

#[test]
#[should_panic(expected: "Only owner")]
fn test_propose_upgrade_non_owner_rejected() {
    let verifier = deploy_verifier();
    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(verifier.contract_address, attacker);

    let new_class: starknet::ClassHash = 0xABCDEF.try_into().unwrap();
    verifier.propose_upgrade(new_class);
}

#[test]
#[should_panic(expected: "Class hash cannot be zero")]
fn test_propose_upgrade_zero_class_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    let zero_class: starknet::ClassHash = 0.try_into().unwrap();
    verifier.propose_upgrade(zero_class);
}

#[test]
#[should_panic(expected: "Upgrade already pending")]
fn test_propose_upgrade_double_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    let class1: starknet::ClassHash = 0xAAA.try_into().unwrap();
    let class2: starknet::ClassHash = 0xBBB.try_into().unwrap();
    verifier.propose_upgrade(class1);
    verifier.propose_upgrade(class2); // should fail
}

#[test]
fn test_cancel_upgrade() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    let new_class: starknet::ClassHash = 0xABCDEF.try_into().unwrap();
    verifier.propose_upgrade(new_class);
    verifier.cancel_upgrade();

    let (pending, _ts) = verifier.get_pending_upgrade();
    let pending_felt: felt252 = pending.into();
    assert!(pending_felt == 0, "pending should be cleared after cancel");
}

#[test]
#[should_panic(expected: "No upgrade pending")]
fn test_cancel_upgrade_when_none_pending() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.cancel_upgrade(); // nothing to cancel
}

#[test]
#[should_panic(expected: "No upgrade pending")]
fn test_execute_upgrade_when_none_pending() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.execute_upgrade(); // nothing to execute
}

#[test]
fn test_get_pending_upgrade_default_zero() {
    let verifier = deploy_verifier();
    let (pending, ts) = verifier.get_pending_upgrade();
    let pending_felt: felt252 = pending.into();
    assert!(pending_felt == 0, "default pending should be zero");
    assert!(ts == 0, "default timestamp should be zero");
}
