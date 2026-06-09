use core::poseidon::poseidon_hash_span;
use elo_cairo_verifier::conversation_statement_stage_verifier::{
    IConversationStatementStageVerifierDispatcher,
    IConversationStatementStageVerifierDispatcherTrait,
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
const PROOF_HASH: felt252 = 0xC0FFEE02;
const SECURITY_BITS: u32 = 160;
const STATEMENT_STAGE_INDEX: u32 = 0;
const STATEMENT_STAGE_MASK: u32 = 1;

fn statement_felts(
    model_id: felt252, program_hash: felt252, security_bits: felt252,
) -> Array<felt252> {
    array![
        DOMAIN_BATCH, 1, model_id, program_hash, 0xC100, 0xC101, 0xC102, 0xC103, 0xC104, 0xC105,
        0xC106, 0xC107, 0xC108, 0xC109, 2, 8, 0xC10A, 0xC10B, security_bits,
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

fn deploy_stage_verifier() -> IConversationStatementStageVerifierDispatcher {
    let contract = declare("ConversationStatementStageVerifierContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![]).unwrap();
    IConversationStatementStageVerifierDispatcher { contract_address: address }
}

fn setup() -> (
    IStatementFactRegistryDispatcher,
    IStatementVerificationSessionDispatcher,
    IConversationStatementStageVerifierDispatcher,
) {
    let registry = deploy_registry();
    let session = deploy_session(@registry);
    let stage = deploy_stage_verifier();

    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(registry.contract_address, owner);
    registry.set_verifier(session.contract_address);
    stop_cheat_caller_address(registry.contract_address);

    start_cheat_caller_address(session.contract_address, owner);
    session.set_stage_verifier(STATEMENT_STAGE_INDEX, stage.contract_address);

    (registry, session, stage)
}

#[test]
fn test_conversation_statement_stage_verifier_attests_and_finalizes() {
    let (registry, session, stage) = setup();
    let felts = statement_felts(MODEL_ID, PROGRAM_HASH, SECURITY_BITS.into());
    let statement_hash = poseidon_hash_span(felts.span());

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_id = session
        .open_session(
            statement_hash, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS, STATEMENT_STAGE_MASK,
        );
    stop_cheat_caller_address(session.contract_address);

    let returned_hash = stage
        .verify_and_attest(session.contract_address, session_id, STATEMENT_STAGE_INDEX, felts);
    assert!(returned_hash == statement_hash, "stage should return statement hash");
    assert!(
        session.get_stage_result_hash(session_id, STATEMENT_STAGE_INDEX) == statement_hash,
        "stage result should be statement hash",
    );

    session.finalize_session(session_id);
    assert!(registry.is_statement_verified(statement_hash), "statement should finalize");
}

#[test]
#[should_panic(expected: "statement program mismatch")]
fn test_conversation_statement_stage_verifier_rejects_relabelled_program() {
    let (_, session, stage) = setup();
    let original = statement_felts(MODEL_ID, PROGRAM_HASH, SECURITY_BITS.into());
    let statement_hash = poseidon_hash_span(original.span());
    let relabelled = statement_felts(MODEL_ID, 0xBADF00D, SECURITY_BITS.into());

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_id = session
        .open_session(
            statement_hash, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS, STATEMENT_STAGE_MASK,
        );
    stop_cheat_caller_address(session.contract_address);

    stage
        .verify_and_attest(session.contract_address, session_id, STATEMENT_STAGE_INDEX, relabelled);
}

#[test]
#[should_panic(expected: "statement security mismatch")]
fn test_conversation_statement_stage_verifier_rejects_relabelled_security() {
    let (_, session, stage) = setup();
    let original = statement_felts(MODEL_ID, PROGRAM_HASH, SECURITY_BITS.into());
    let statement_hash = poseidon_hash_span(original.span());
    let relabelled = statement_felts(MODEL_ID, PROGRAM_HASH, 192);

    let submitter: ContractAddress = SUBMITTER_ADDR.try_into().unwrap();
    start_cheat_caller_address(session.contract_address, submitter);
    let session_id = session
        .open_session(
            statement_hash, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS, STATEMENT_STAGE_MASK,
        );
    stop_cheat_caller_address(session.contract_address);

    stage
        .verify_and_attest(session.contract_address, session_id, STATEMENT_STAGE_INDEX, relabelled);
}

#[test]
#[should_panic(expected: "statement conversations missing")]
fn test_conversation_statement_stage_verifier_rejects_empty_conversation_count() {
    let (_, _, stage) = setup();
    let felts = array![
        DOMAIN_BATCH, 1, MODEL_ID, PROGRAM_HASH, 0xC100, 0xC101, 0xC102, 0xC103, 0xC104, 0xC105,
        0xC106, 0xC107, 0xC108, 0xC109, 0, 8, 0xC10A, 0xC10B, SECURITY_BITS.into(),
    ];

    stage.compute_statement_hash(felts);
}
