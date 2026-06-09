use core::poseidon::poseidon_hash_span;
use elo_cairo_verifier::conversation_statement_stage_verifier::{
    IConversationStatementStageVerifierDispatcher,
    IConversationStatementStageVerifierDispatcherTrait,
};
use elo_cairo_verifier::hades_logup_stage_verifier::{
    IHadesLogupStageVerifierDispatcher, IHadesLogupStageVerifierDispatcherTrait,
};
use elo_cairo_verifier::mock_hades_logup_fact_source::{
    IMockHadesLogupFactSourceDispatcher, IMockHadesLogupFactSourceDispatcherTrait,
};
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
const DOMAIN_BATCH: felt252 = 0x43424154;
const MODEL_ID: felt252 = 0xC0FFEE04;
const PROGRAM_HASH: felt252 = 0xC0FFEE03;
const SESSION_PROOF_HASH: felt252 = 0xC0FFEE02;
const RECURSIVE_PROOF_HASH: felt252 = 0xC0FFEE05;
const HADES_PROOF_HASH: felt252 = 0xC0FFEE06;
const HADES_COMMITMENT: felt252 = 0xC0FFEE07;
const SECURITY_BITS: u32 = 160;
const STATEMENT_STAGE_INDEX: u32 = 0;
const RECURSIVE_STAGE_INDEX: u32 = 1;
const HADES_STAGE_INDEX: u32 = 2;
const REQUIRED_STAGE_MASK: u32 = 7;

fn statement_felts() -> Array<felt252> {
    array![
        DOMAIN_BATCH, 1, MODEL_ID, PROGRAM_HASH, 0xC100, 0xC101, 0xC102, 0xC103, 0xC104, 0xC105,
        0xC106, 0xC107, 0xC108, 0xC109, 2, 8, 0xC10A, 0xC10B, SECURITY_BITS.into(),
    ]
}

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

fn deploy_statement_stage() -> IConversationStatementStageVerifierDispatcher {
    let contract = declare("ConversationStatementStageVerifierContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![]).unwrap();
    IConversationStatementStageVerifierDispatcher { contract_address: address }
}

fn deploy_recursive_stage(
    source: @IMockRecursiveStatementFactSourceDispatcher,
) -> IRecursiveStatementStageVerifierDispatcher {
    let contract = declare("RecursiveStatementStageVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract
        .deploy(@array![owner.into(), (*source.contract_address).into()])
        .unwrap();
    IRecursiveStatementStageVerifierDispatcher { contract_address: address }
}

fn deploy_hades_stage(
    source: @IMockHadesLogupFactSourceDispatcher,
) -> IHadesLogupStageVerifierDispatcher {
    let contract = declare("HadesLogupStageVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract
        .deploy(@array![owner.into(), (*source.contract_address).into()])
        .unwrap();
    IHadesLogupStageVerifierDispatcher { contract_address: address }
}

fn deploy_recursive_source() -> IMockRecursiveStatementFactSourceDispatcher {
    let contract = declare("MockRecursiveStatementFactSourceContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![]).unwrap();
    IMockRecursiveStatementFactSourceDispatcher { contract_address: address }
}

fn deploy_hades_source() -> IMockHadesLogupFactSourceDispatcher {
    let contract = declare("MockHadesLogupFactSourceContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![]).unwrap();
    IMockHadesLogupFactSourceDispatcher { contract_address: address }
}

#[test]
fn test_three_stage_statement_pipeline_finalizes_only_after_all_facts() {
    let registry = deploy_registry();
    let session = deploy_session(@registry);
    let statement_stage = deploy_statement_stage();
    let recursive_source = deploy_recursive_source();
    let hades_source = deploy_hades_source();
    let recursive_stage = deploy_recursive_stage(@recursive_source);
    let hades_stage = deploy_hades_stage(@hades_source);

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(registry.contract_address, owner);
    registry.set_verifier(session.contract_address);
    registry.freeze_verifier();
    stop_cheat_caller_address(registry.contract_address);

    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_verifier(STATEMENT_STAGE_INDEX, statement_stage.contract_address);
    session.set_stage_verifier(RECURSIVE_STAGE_INDEX, recursive_stage.contract_address);
    session.set_stage_verifier(HADES_STAGE_INDEX, hades_stage.contract_address);
    session.set_stage_policy(PROGRAM_HASH, MODEL_ID, REQUIRED_STAGE_MASK);
    session.freeze_stage_verifier(STATEMENT_STAGE_INDEX);
    session.freeze_stage_verifier(RECURSIVE_STAGE_INDEX);
    session.freeze_stage_verifier(HADES_STAGE_INDEX);
    session.freeze_stage_policy(PROGRAM_HASH, MODEL_ID);
    stop_cheat_caller_address(session.contract_address);

    start_cheat_caller_address(recursive_stage.contract_address, owner);
    recursive_stage.freeze_fact_source();
    stop_cheat_caller_address(recursive_stage.contract_address);

    start_cheat_caller_address(hades_stage.contract_address, owner);
    hades_stage.freeze_fact_source();
    stop_cheat_caller_address(hades_stage.contract_address);

    let felts = statement_felts();
    let statement_hash = poseidon_hash_span(felts.span());
    recursive_source.set_statement_fact(statement_hash, RECURSIVE_PROOF_HASH, true);
    hades_source.set_hades_logup_fact(statement_hash, HADES_PROOF_HASH, HADES_COMMITMENT, true);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_id = session
        .open_session(
            statement_hash,
            SESSION_PROOF_HASH,
            PROGRAM_HASH,
            MODEL_ID,
            SECURITY_BITS,
            REQUIRED_STAGE_MASK,
        );
    stop_cheat_caller_address(session.contract_address);

    statement_stage
        .verify_and_attest(session.contract_address, session_id, STATEMENT_STAGE_INDEX, felts);
    recursive_stage
        .verify_and_attest(
            session.contract_address, session_id, RECURSIVE_STAGE_INDEX, RECURSIVE_PROOF_HASH,
        );
    hades_stage
        .verify_and_attest(
            session.contract_address,
            session_id,
            HADES_STAGE_INDEX,
            HADES_PROOF_HASH,
            HADES_COMMITMENT,
        );

    let composition_hash = session.get_composition_proof_hash(session_id);
    session.finalize_session(session_id);

    assert!(registry.is_statement_verified(statement_hash), "statement should finalize");
    assert!(
        registry.get_statement_proof_hash(statement_hash) == composition_hash,
        "registry should store composed proof hash",
    );
}
