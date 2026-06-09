use elo_cairo_verifier::mock_recursive_statement_fact_source::{
    IMockRecursiveStatementFactSourceDispatcher, IMockRecursiveStatementFactSourceDispatcherTrait,
};
use elo_cairo_verifier::recursive_statement_stage_verifier::{
    IRecursiveStatementStageVerifierDispatcher, IRecursiveStatementStageVerifierDispatcherTrait,
};
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
const STATEMENT_HASH: felt252 = 0xF00D01;
const SESSION_PROOF_HASH: felt252 = 0xF00D02;
const PROGRAM_HASH: felt252 = 0xF00D03;
const MODEL_ID: felt252 = 0xF00D04;
const RECURSIVE_PROOF_HASH: felt252 = 0xF00D05;
const SECURITY_BITS: u32 = 160;
const RECURSIVE_STAGE_INDEX: u32 = 1;
const RECURSIVE_STAGE_MASK: u32 = 2;

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

fn deploy_stage_verifier(
    fact_source: @IMockRecursiveStatementFactSourceDispatcher,
) -> IRecursiveStatementStageVerifierDispatcher {
    let contract = declare("RecursiveStatementStageVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract
        .deploy(@array![owner.into(), (*fact_source.contract_address).into()])
        .unwrap();
    IRecursiveStatementStageVerifierDispatcher { contract_address: address }
}

fn deploy_fact_source() -> IMockRecursiveStatementFactSourceDispatcher {
    let contract = declare("MockRecursiveStatementFactSourceContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![]).unwrap();
    IMockRecursiveStatementFactSourceDispatcher { contract_address: address }
}

fn setup() -> (
    IStatementFactRegistryDispatcher,
    IStatementVerificationSessionDispatcher,
    IRecursiveStatementStageVerifierDispatcher,
    IMockRecursiveStatementFactSourceDispatcher,
) {
    let registry = deploy_registry();
    let session = deploy_session(@registry);
    let fact_source = deploy_fact_source();
    let stage = deploy_stage_verifier(@fact_source);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(registry.contract_address, owner);
    registry.set_verifier(session.contract_address);
    stop_cheat_caller_address(registry.contract_address);

    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_verifier(RECURSIVE_STAGE_INDEX, stage.contract_address);
    stop_cheat_caller_address(session.contract_address);

    (registry, session, stage, fact_source)
}

fn open_recursive_session(session: @IStatementVerificationSessionDispatcher) -> u64 {
    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*session.contract_address, submitter);
    let session_id = session
        .open_session(
            STATEMENT_HASH,
            SESSION_PROOF_HASH,
            PROGRAM_HASH,
            MODEL_ID,
            SECURITY_BITS,
            RECURSIVE_STAGE_MASK,
        );
    stop_cheat_caller_address(*session.contract_address);
    session_id
}

#[test]
fn test_recursive_statement_stage_verifier_attests_existing_recursive_fact() {
    let (registry, session, stage, fact_source) = setup();
    let session_id = open_recursive_session(@session);
    fact_source.set_statement_fact(STATEMENT_HASH, RECURSIVE_PROOF_HASH, true);

    let returned_hash = stage
        .verify_and_attest(
            session.contract_address, session_id, RECURSIVE_STAGE_INDEX, RECURSIVE_PROOF_HASH,
        );
    assert!(returned_hash != RECURSIVE_PROOF_HASH, "stage result should bind source and statement");
    assert!(
        session.get_stage_result_hash(session_id, RECURSIVE_STAGE_INDEX) == returned_hash,
        "stage result should match returned hash",
    );

    session.finalize_session(session_id);
    assert!(registry.is_statement_verified(STATEMENT_HASH), "statement should finalize");
}

#[test]
#[should_panic(expected: "recursive statement fact missing")]
fn test_recursive_statement_stage_verifier_rejects_missing_recursive_fact() {
    let (_, session, stage, _) = setup();
    let session_id = open_recursive_session(@session);

    stage.verify_and_attest(session.contract_address, session_id, RECURSIVE_STAGE_INDEX, 0);
}

#[test]
#[should_panic(expected: "recursive proof hash mismatch")]
fn test_recursive_statement_stage_verifier_rejects_mismatched_recursive_proof_hash() {
    let (_, session, stage, fact_source) = setup();
    let session_id = open_recursive_session(@session);
    fact_source.set_statement_fact(STATEMENT_HASH, RECURSIVE_PROOF_HASH, true);

    stage.verify_and_attest(session.contract_address, session_id, RECURSIVE_STAGE_INDEX, 0xBADF00D);
}

#[test]
#[should_panic(expected: "Stage: owner only")]
fn test_recursive_statement_stage_verifier_rejects_non_owner_fact_source_update() {
    let (_, _, stage, fact_source) = setup();
    let attacker: ContractAddress = 0xBAD.try_into().unwrap();
    start_cheat_caller_address(stage.contract_address, attacker);
    stage.set_fact_source(fact_source.contract_address);
}

#[test]
#[should_panic(expected: "fact source frozen")]
fn test_recursive_statement_stage_verifier_rejects_update_after_freeze() {
    let (_, _, stage, fact_source) = setup();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(stage.contract_address, owner);
    stage.freeze_fact_source();
    assert!(stage.is_fact_source_frozen(), "fact source should be frozen");
    stage.set_fact_source(fact_source.contract_address);
}

#[test]
fn test_recursive_statement_stage_result_changes_with_fact_source() {
    let (_, session_a, stage, fact_source_a) = setup();
    let fact_source_b = deploy_fact_source();

    let session_a_id = open_recursive_session(@session_a);
    fact_source_a.set_statement_fact(STATEMENT_HASH, RECURSIVE_PROOF_HASH, true);
    fact_source_b.set_statement_fact(STATEMENT_HASH, RECURSIVE_PROOF_HASH, true);

    let result_a = stage
        .verify_and_attest(
            session_a.contract_address, session_a_id, RECURSIVE_STAGE_INDEX, RECURSIVE_PROOF_HASH,
        );

    let (registry_b, session_b) = deploy_session_with_stage(@stage);
    let session_b_id = open_recursive_session(@session_b);
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(stage.contract_address, owner);
    stage.set_fact_source(fact_source_b.contract_address);

    let result_b = stage
        .verify_and_attest(
            session_b.contract_address, session_b_id, RECURSIVE_STAGE_INDEX, RECURSIVE_PROOF_HASH,
        );

    assert!(result_a != result_b, "stage result should bind fact source");
    assert!(!registry_b.is_statement_verified(STATEMENT_HASH), "session should not auto-finalize");
}

fn deploy_session_with_stage(
    stage: @IRecursiveStatementStageVerifierDispatcher,
) -> (IStatementFactRegistryDispatcher, IStatementVerificationSessionDispatcher) {
    let registry = deploy_registry();
    let session = deploy_session(@registry);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(registry.contract_address, owner);
    registry.set_verifier(session.contract_address);
    stop_cheat_caller_address(registry.contract_address);

    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_verifier(RECURSIVE_STAGE_INDEX, *stage.contract_address);
    stop_cheat_caller_address(session.contract_address);

    (registry, session)
}
