use elo_cairo_verifier::statement_fact_registry::{
    IStatementFactRegistryDispatcher, IStatementFactRegistryDispatcherTrait,
};
use elo_cairo_verifier::statement_verification_session::{
    IStatementVerificationSessionDispatcher, IStatementVerificationSessionDispatcherTrait,
};
use snforge_std::{
    ContractClassTrait, DeclareResultTrait, declare, start_cheat_caller_address,
    stop_cheat_caller_address,
};
use starknet::ContractAddress;

const OWNER_ADDR: felt252 = 0x1234;
const SUBMITTER_ADDR: felt252 = 0x2345;
const STAGE0_VERIFIER_ADDR: felt252 = 0x3456;
const STAGE1_VERIFIER_ADDR: felt252 = 0x4567;
const ATTACKER_ADDR: felt252 = 0xBAD;
const STATEMENT_HASH: felt252 = 0xC0FFEE01;
const PROOF_HASH: felt252 = 0xC0FFEE02;
const PROGRAM_HASH: felt252 = 0xC0FFEE03;
const MODEL_ID: felt252 = 0xC0FFEE04;
const SECURITY_BITS: u32 = 160;
const STAGE0_AND_STAGE1_MASK: u32 = 3;

fn deploy_registry() -> IStatementFactRegistryDispatcher {
    let contract = declare("StatementFactRegistryContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let zero: ContractAddress = 0_felt252.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into(), zero.into()]).unwrap();
    IStatementFactRegistryDispatcher { contract_address: address }
}

fn deploy_session(
    registry: @IStatementFactRegistryDispatcher,
) -> IStatementVerificationSessionDispatcher {
    let contract = declare("StatementVerificationSessionContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract
        .deploy(@array![owner.into(), (*registry.contract_address).into()])
        .unwrap();
    IStatementVerificationSessionDispatcher { contract_address: address }
}

fn deploy_session_with_registry_verifier() -> (
    IStatementFactRegistryDispatcher, IStatementVerificationSessionDispatcher,
) {
    let registry = deploy_registry();
    let session = deploy_session(@registry);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(registry.contract_address, owner);
    registry.set_verifier(session.contract_address);
    stop_cheat_caller_address(registry.contract_address);

    (registry, session)
}

fn setup() -> (IStatementFactRegistryDispatcher, IStatementVerificationSessionDispatcher) {
    let (registry, session) = deploy_session_with_registry_verifier();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    let stage1: ContractAddress = STAGE1_VERIFIER_ADDR.try_into().unwrap();
    session.set_stage_verifier(0, stage0);
    session.set_stage_verifier(1, stage1);

    (registry, session)
}

fn open_statement_session(session: @IStatementVerificationSessionDispatcher) -> u64 {
    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*session.contract_address, submitter);
    session
        .open_session(
            STATEMENT_HASH,
            PROOF_HASH,
            PROGRAM_HASH,
            MODEL_ID,
            SECURITY_BITS,
            STAGE0_AND_STAGE1_MASK,
        )
}

#[test]
fn test_statement_session_finalizes_and_records_registry_fact() {
    let (registry, session) = setup();
    let session_id = open_statement_session(@session);

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);

    let stage1: ContractAddress = STAGE1_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage1);
    session.attest_stage(session_id, 1, 0xBBB);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let composition_proof_hash = session.get_composition_proof_hash(session_id);
    session.finalize_session(session_id);

    assert!(registry.is_statement_verified(STATEMENT_HASH), "statement should be recorded");
    assert!(
        registry.get_statement_proof_hash(STATEMENT_HASH) == composition_proof_hash,
        "composition proof hash should match",
    );
    assert!(
        registry.get_statement_proof_hash(STATEMENT_HASH) != PROOF_HASH,
        "caller proof hash must not be copied directly",
    );
    let info = session.get_session(session_id);
    assert!(info.completed_stage_mask == STAGE0_AND_STAGE1_MASK, "mask should be complete");
    assert!(info.finalized, "session should be finalized");
}

#[test]
#[should_panic(expected: "required stages missing")]
fn test_statement_session_rejects_finalize_before_required_stages() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);

    session.finalize_session(session_id);
}

#[test]
#[should_panic(expected: "Session: stage verifier only")]
fn test_statement_session_rejects_unauthorized_stage_attestation() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, attacker);
    session.attest_stage(session_id, 0, 0xAAA);
}

#[test]
#[should_panic(expected: "stage not required")]
fn test_statement_session_rejects_unrequested_stage_attestation() {
    let (registry, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    let stage2: ContractAddress = 0x5678.try_into().unwrap();
    session.set_stage_verifier(2, stage2);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_id = session
        .open_session(0xDAD01, 0xDAD02, 0xDAD03, 0xDAD04, SECURITY_BITS, STAGE0_AND_STAGE1_MASK);

    start_cheat_caller_address(session.contract_address, stage2);
    session.attest_stage(session_id, 2, 0xCCC);

    assert!(!registry.is_statement_verified(STATEMENT_HASH), "statement should not record");
}

#[test]
#[should_panic(expected: "stage already attested")]
fn test_statement_session_rejects_duplicate_stage_attestation() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);
    session.attest_stage(session_id, 0, 0xAAB);
}

#[test]
#[should_panic(expected: "session finalized")]
fn test_statement_session_rejects_duplicate_finalize() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);

    let stage1: ContractAddress = STAGE1_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage1);
    session.attest_stage(session_id, 1, 0xBBB);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    session.finalize_session(session_id);
    session.finalize_session(session_id);
}

#[test]
#[should_panic(expected: "required verifier missing")]
fn test_statement_session_rejects_open_with_missing_required_verifier() {
    let (_, session) = deploy_session_with_registry_verifier();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    session.set_stage_verifier(0, stage0);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    session
        .open_session(
            STATEMENT_HASH,
            PROOF_HASH,
            PROGRAM_HASH,
            MODEL_ID,
            SECURITY_BITS,
            STAGE0_AND_STAGE1_MASK,
        );
}

#[test]
fn test_statement_session_binds_stage_fact_to_session_metadata() {
    let (_, session) = setup();

    let session_a = open_statement_session(@session);
    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_b = session
        .open_session(
            0xFACE01, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS, STAGE0_AND_STAGE1_MASK,
        );

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_a, 0, 0xAAA);
    session.attest_stage(session_b, 0, 0xAAA);

    assert!(
        session.get_stage_result_hash(session_a, 0) == session.get_stage_result_hash(session_b, 0),
        "raw stage result should match",
    );
    assert!(
        session.get_stage_fact_hash(session_a, 0) != session.get_stage_fact_hash(session_b, 0),
        "bound stage facts should differ across statements",
    );
}

#[test]
fn test_statement_session_composition_hash_changes_with_stage_result() {
    let (_, session) = setup();

    let session_a = open_statement_session(@session);
    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_b = session
        .open_session(
            STATEMENT_HASH,
            PROOF_HASH,
            PROGRAM_HASH,
            MODEL_ID,
            SECURITY_BITS,
            STAGE0_AND_STAGE1_MASK,
        );

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_a, 0, 0xAAA);
    session.attest_stage(session_b, 0, 0xAAC);

    let stage1: ContractAddress = STAGE1_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage1);
    session.attest_stage(session_a, 1, 0xBBB);
    session.attest_stage(session_b, 1, 0xBBB);

    assert!(
        session
            .get_composition_proof_hash(session_a) != session
            .get_composition_proof_hash(session_b),
        "composition hash should bind stage result",
    );
}

#[test]
fn test_statement_session_composition_uses_attesting_verifier_after_rotation() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);

    let stage1: ContractAddress = STAGE1_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage1);
    session.attest_stage(session_id, 1, 0xBBB);

    let before_rotation = session.get_composition_proof_hash(session_id);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    let rotated_stage0: ContractAddress = 0xABCD.try_into().unwrap();
    session.set_stage_verifier(0, rotated_stage0);

    assert!(session.get_stage_verifier(0) == rotated_stage0, "configured verifier should rotate");
    assert!(
        session.get_stage_attestor(session_id, 0) == stage0, "session attestor should be stable",
    );
    assert!(
        session.get_composition_proof_hash(session_id) == before_rotation,
        "composition hash should use attesting verifier",
    );
}

#[test]
fn test_statement_session_uses_open_time_stage_verifier_after_rotation() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    let rotated_stage0: ContractAddress = 0xABCD.try_into().unwrap();
    session.set_stage_verifier(0, rotated_stage0);

    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    assert!(session.get_stage_verifier(0) == rotated_stage0, "live verifier should rotate");
    assert!(
        session.get_session_stage_verifier(session_id, 0) == stage0,
        "session verifier should stay snapshotted",
    );

    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);

    assert!(session.get_stage_attestor(session_id, 0) == stage0, "old verifier should attest");
}

#[test]
#[should_panic(expected: "Session: stage verifier only")]
fn test_statement_session_rejects_rotated_verifier_for_existing_session() {
    let (_, session) = setup();
    let session_id = open_statement_session(@session);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    let rotated_stage0: ContractAddress = 0xABCD.try_into().unwrap();
    session.set_stage_verifier(0, rotated_stage0);

    start_cheat_caller_address(session.contract_address, rotated_stage0);
    session.attest_stage(session_id, 0, 0xAAA);
}

#[test]
#[should_panic(expected: "Session: owner only")]
fn test_statement_session_rejects_non_owner_stage_freeze() {
    let (_, session) = setup();

    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, attacker);
    session.freeze_stage_verifier(0);
}

#[test]
#[should_panic(expected: "stage verifier frozen")]
fn test_statement_session_rejects_stage_update_after_freeze() {
    let (_, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    session.freeze_stage_verifier(0);
    assert!(session.is_stage_verifier_frozen(0), "stage verifier should be frozen");
    let rotated_stage0: ContractAddress = 0xABCD.try_into().unwrap();
    session.set_stage_verifier(0, rotated_stage0);
}

#[test]
fn test_statement_session_can_open_and_attest_with_frozen_stage_verifier() {
    let (_, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    session.freeze_stage_verifier(0);
    session.freeze_stage_verifier(1);

    let session_id = open_statement_session(@session);
    let stage0: ContractAddress = STAGE0_VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, stage0);
    session.attest_stage(session_id, 0, 0xAAA);

    assert!(session.get_stage_attestor(session_id, 0) == stage0, "frozen stage should attest");
}

#[test]
#[should_panic(expected: "required policy stages missing")]
fn test_statement_session_rejects_stage_mask_below_model_policy() {
    let (_, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_policy(PROGRAM_HASH, MODEL_ID, STAGE0_AND_STAGE1_MASK);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    session.open_session(STATEMENT_HASH, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS, 1);
}

#[test]
fn test_statement_session_accepts_stage_mask_covering_model_policy() {
    let (_, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_policy(PROGRAM_HASH, MODEL_ID, 1);

    let session_id = open_statement_session(@session);
    assert!(
        session
            .get_session_stage_verifier(session_id, 0) == STAGE0_VERIFIER_ADDR
            .try_into()
            .unwrap(),
        "stage 0 should snapshot",
    );
    assert!(
        session
            .get_session_stage_verifier(session_id, 1) == STAGE1_VERIFIER_ADDR
            .try_into()
            .unwrap(),
        "stage 1 should snapshot",
    );
}

#[test]
#[should_panic(expected: "Session: owner only")]
fn test_statement_session_rejects_non_owner_stage_policy_update() {
    let (_, session) = setup();

    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, attacker);
    session.set_stage_policy(PROGRAM_HASH, MODEL_ID, STAGE0_AND_STAGE1_MASK);
}

#[test]
#[should_panic(expected: "stage policy frozen")]
fn test_statement_session_rejects_stage_policy_update_after_freeze() {
    let (_, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_policy(PROGRAM_HASH, MODEL_ID, STAGE0_AND_STAGE1_MASK);
    session.freeze_stage_policy(PROGRAM_HASH, MODEL_ID);
    assert!(session.is_stage_policy_frozen(PROGRAM_HASH, MODEL_ID), "policy should be frozen");
    session.set_stage_policy(PROGRAM_HASH, MODEL_ID, 1);
}

#[test]
#[should_panic(expected: "stage policy missing")]
fn test_statement_session_rejects_freezing_missing_stage_policy() {
    let (_, session) = setup();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, owner);
    session.freeze_stage_policy(PROGRAM_HASH, MODEL_ID);
}
