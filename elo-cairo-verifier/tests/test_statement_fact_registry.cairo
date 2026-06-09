use elo_cairo_verifier::statement_fact_registry::{
    IStatementFactRegistryDispatcher, IStatementFactRegistryDispatcherTrait,
};
use snforge_std::{ContractClassTrait, DeclareResultTrait, declare, start_cheat_caller_address};
use starknet::ContractAddress;

const OWNER_ADDR: felt252 = 0x1234;
const VERIFIER_ADDR: felt252 = 0x5678;
const ATTACKER_ADDR: felt252 = 0xBAD;
const STATEMENT_HASH: felt252 = 0xCAFE01;
const PROOF_HASH: felt252 = 0xCAFE02;
const PROGRAM_HASH: felt252 = 0xCAFE03;
const MODEL_ID: felt252 = 0xCAFE04;
const SECURITY_BITS: u32 = 160;

fn deploy_registry() -> IStatementFactRegistryDispatcher {
    let contract = declare("StatementFactRegistryContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let verifier: ContractAddress = VERIFIER_ADDR.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into(), verifier.into()]).unwrap();
    IStatementFactRegistryDispatcher { contract_address: address }
}

fn as_verifier(registry: @IStatementFactRegistryDispatcher) {
    let verifier: ContractAddress = VERIFIER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*registry.contract_address, verifier);
}

fn as_owner(registry: @IStatementFactRegistryDispatcher) {
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*registry.contract_address, owner);
}

fn as_attacker(registry: @IStatementFactRegistryDispatcher) {
    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*registry.contract_address, attacker);
}

#[test]
fn test_statement_fact_registry_records_verified_statement() {
    let registry = deploy_registry();
    as_verifier(@registry);

    registry
        .record_statement_fact(STATEMENT_HASH, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS);

    assert!(registry.is_statement_verified(STATEMENT_HASH), "statement should be verified");
    assert!(
        registry.get_statement_proof_hash(STATEMENT_HASH) == PROOF_HASH, "proof hash should match",
    );
    assert!(
        registry.get_statement_program_hash(STATEMENT_HASH) == PROGRAM_HASH,
        "program hash should match",
    );
    assert!(registry.get_statement_model_id(STATEMENT_HASH) == MODEL_ID, "model should match");
    assert!(
        registry.get_statement_security_bits(STATEMENT_HASH) == SECURITY_BITS,
        "security bits should match",
    );
}

#[test]
#[should_panic(expected: "Registry: verifier only")]
fn test_statement_fact_registry_rejects_non_verifier() {
    let registry = deploy_registry();
    as_attacker(@registry);

    registry
        .record_statement_fact(STATEMENT_HASH, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS);
}

#[test]
#[should_panic(expected: "statement already recorded")]
fn test_statement_fact_registry_rejects_duplicate_statement() {
    let registry = deploy_registry();
    as_verifier(@registry);

    registry
        .record_statement_fact(STATEMENT_HASH, PROOF_HASH, PROGRAM_HASH, MODEL_ID, SECURITY_BITS);
    registry.record_statement_fact(STATEMENT_HASH, 0xBEEF02, PROGRAM_HASH, MODEL_ID, SECURITY_BITS);
}

#[test]
#[should_panic(expected: "security below 160")]
fn test_statement_fact_registry_rejects_low_security_fact() {
    let registry = deploy_registry();
    as_verifier(@registry);

    registry.record_statement_fact(STATEMENT_HASH, PROOF_HASH, PROGRAM_HASH, MODEL_ID, 100);
}

#[test]
#[should_panic(expected: "Registry: owner only")]
fn test_statement_fact_registry_rejects_non_owner_freeze() {
    let registry = deploy_registry();
    as_attacker(@registry);

    registry.freeze_verifier();
}

#[test]
#[should_panic(expected: "verifier frozen")]
fn test_statement_fact_registry_rejects_verifier_update_after_freeze() {
    let registry = deploy_registry();
    as_owner(@registry);

    registry.freeze_verifier();
    assert!(registry.is_verifier_frozen(), "verifier should be frozen");
    let next_verifier: ContractAddress = 0x9999.try_into().unwrap();
    registry.set_verifier(next_verifier);
}
