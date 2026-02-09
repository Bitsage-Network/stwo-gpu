use snforge_std::{declare, ContractClassTrait, DeclareResultTrait};
use starknet::ContractAddress;

use stwo_ml_verifier::contract::{IStweMlVerifierDispatcher, IStweMlVerifierDispatcherTrait};
use stwo_ml_verifier::sumcheck::{
    MatMulSumcheckProof, MleOpeningProof,
    qm31_zero,
};

fn deploy_contract() -> IStweMlVerifierDispatcher {
    let contract = declare("StweMlVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = 0x1_felt252.try_into().unwrap();
    let mut calldata = array![];
    calldata.append(owner.into());
    let (contract_address, _) = contract.deploy(@calldata).unwrap();
    IStweMlVerifierDispatcher { contract_address }
}

// ============================================================================
// register_model
// ============================================================================

#[test]
fn test_register_model() {
    let dispatcher = deploy_contract();
    let model_id: felt252 = 0x42;
    let commitment: felt252 = 0xABCD;
    dispatcher.register_model(model_id, commitment);

    let stored = dispatcher.get_model_commitment(model_id);
    assert!(stored == commitment, "commitment should match");
}

#[test]
#[should_panic(expected: "Commitment cannot be zero")]
fn test_register_model_zero_commitment_fails() {
    let dispatcher = deploy_contract();
    dispatcher.register_model(0x42, 0);
}

#[test]
#[should_panic(expected: "Model already registered")]
fn test_register_model_duplicate_fails() {
    let dispatcher = deploy_contract();
    dispatcher.register_model(0x42, 0xABCD);
    dispatcher.register_model(0x42, 0xDEF0); // duplicate
}

// ============================================================================
// get_model_commitment / get_verification_count / is_verified
// ============================================================================

#[test]
fn test_get_unregistered_model() {
    let dispatcher = deploy_contract();
    let stored = dispatcher.get_model_commitment(0x999);
    assert!(stored == 0, "unregistered model should return 0");
}

#[test]
fn test_verification_count_starts_zero() {
    let dispatcher = deploy_contract();
    let count = dispatcher.get_verification_count(0x999);
    assert!(count == 0, "unregistered model should have 0 verifications");
}

#[test]
fn test_is_verified_default_false() {
    let dispatcher = deploy_contract();
    let verified = dispatcher.is_verified(0xDEAD);
    assert!(!verified, "random hash should not be verified");
}

// ============================================================================
// verify_matmul — error paths
// ============================================================================

#[test]
#[should_panic(expected: "Model not registered")]
fn test_verify_matmul_unregistered_model() {
    let dispatcher = deploy_contract();
    // Create a minimal proof (will fail at "Model not registered" before proof validation)
    let proof = MatMulSumcheckProof {
        m: 2, k: 2, n: 2, num_rounds: 1,
        claimed_sum: qm31_zero(),
        round_polys: array![],
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0,
        b_commitment: 0,
        a_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
        b_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
    };
    dispatcher.verify_matmul(0x42, proof);
}

// ============================================================================
// verify_ml_inference — edge cases
// ============================================================================

#[test]
fn test_verify_ml_inference_empty_proofs() {
    let dispatcher = deploy_contract();
    dispatcher.register_model(0x42, 0xABCD);

    // No matmul proofs, no layers — should succeed (empty inference)
    let result = dispatcher.verify_ml_inference(
        0x42,
        0xABCD,     // matching model commitment
        0x1234,     // io_commitment
        array![],   // no matmul proofs
        array![],   // no layer headers
        0x0,        // no tee
    );
    assert!(result, "empty inference should verify");
}

#[test]
fn test_verify_ml_inference_commitment_mismatch() {
    let dispatcher = deploy_contract();
    dispatcher.register_model(0x42, 0xABCD);

    // model_commitment doesn't match registered value
    let result = dispatcher.verify_ml_inference(
        0x42,
        0x9999,     // wrong commitment
        0x1234,
        array![],
        array![],
        0x0,
    );
    assert!(!result, "mismatched commitment should fail");
}
