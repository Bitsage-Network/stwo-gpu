use starknet::ContractAddress;

#[starknet::interface]
pub trait IRecursiveStatementFactSource<TContractState> {
    fn is_recursive_statement_verified(
        self: @TContractState, conversation_statement_hash: felt252,
    ) -> bool;
    fn get_recursive_statement_proof_hash(
        self: @TContractState, conversation_statement_hash: felt252,
    ) -> felt252;
}

#[starknet::interface]
pub trait IRecursiveStatementStageVerifier<TContractState> {
    fn set_fact_source(ref self: TContractState, recursive_fact_source: ContractAddress);
    fn freeze_fact_source(ref self: TContractState);
    fn get_fact_source(self: @TContractState) -> ContractAddress;
    fn is_fact_source_frozen(self: @TContractState) -> bool;
    fn verify_and_attest(
        ref self: TContractState,
        session: ContractAddress,
        session_id: u64,
        stage_index: u32,
        expected_recursive_proof_hash: felt252,
    ) -> felt252;
}

#[starknet::contract]
pub mod RecursiveStatementStageVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::storage::{StoragePointerReadAccess, StoragePointerWriteAccess};
    use starknet::{ContractAddress, get_caller_address};
    use crate::statement_verification_session::{
        IStatementVerificationSessionDispatcher, IStatementVerificationSessionDispatcherTrait,
    };
    use super::{
        IRecursiveStatementFactSourceDispatcher, IRecursiveStatementFactSourceDispatcherTrait,
        IRecursiveStatementStageVerifier,
    };

    const DOMAIN_RECURSIVE_STATEMENT_FACT: felt252 =
        0x5245435552534956455f46414354; // "RECURSIVE_FACT"

    #[storage]
    struct Storage {
        owner: ContractAddress,
        recursive_fact_source: ContractAddress,
        fact_source_frozen: bool,
    }

    #[constructor]
    fn constructor(
        ref self: ContractState, owner: ContractAddress, recursive_fact_source: ContractAddress,
    ) {
        let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
        assert!(owner != zero_addr, "owner cannot be zero");
        assert!(recursive_fact_source != zero_addr, "fact source cannot be zero");
        self.owner.write(owner);
        self.recursive_fact_source.write(recursive_fact_source);
        self.fact_source_frozen.write(false);
    }

    #[abi(embed_v0)]
    impl RecursiveStatementStageVerifierImpl of IRecursiveStatementStageVerifier<ContractState> {
        fn set_fact_source(ref self: ContractState, recursive_fact_source: ContractAddress) {
            assert!(get_caller_address() == self.owner.read(), "Stage: owner only");
            assert!(!self.fact_source_frozen.read(), "fact source frozen");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(recursive_fact_source != zero_addr, "fact source cannot be zero");
            self.recursive_fact_source.write(recursive_fact_source);
        }

        fn freeze_fact_source(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Stage: owner only");
            self.fact_source_frozen.write(true);
        }

        fn get_fact_source(self: @ContractState) -> ContractAddress {
            self.recursive_fact_source.read()
        }

        fn is_fact_source_frozen(self: @ContractState) -> bool {
            self.fact_source_frozen.read()
        }

        fn verify_and_attest(
            ref self: ContractState,
            session: ContractAddress,
            session_id: u64,
            stage_index: u32,
            expected_recursive_proof_hash: felt252,
        ) -> felt252 {
            let session_dispatcher = IStatementVerificationSessionDispatcher {
                contract_address: session,
            };
            let info = session_dispatcher.get_session(session_id);
            assert!(info.statement_hash != 0, "statement hash missing");

            let fact_source_address = self.recursive_fact_source.read();
            let fact_source = IRecursiveStatementFactSourceDispatcher {
                contract_address: fact_source_address,
            };
            assert!(
                fact_source.is_recursive_statement_verified(info.statement_hash),
                "recursive statement fact missing",
            );
            let recursive_proof_hash = fact_source
                .get_recursive_statement_proof_hash(info.statement_hash);
            assert!(recursive_proof_hash != 0, "recursive proof hash missing");
            if expected_recursive_proof_hash != 0 {
                assert!(
                    recursive_proof_hash == expected_recursive_proof_hash,
                    "recursive proof hash mismatch",
                );
            }

            let result_hash = poseidon_hash_span(
                array![
                    DOMAIN_RECURSIVE_STATEMENT_FACT, info.statement_hash,
                    fact_source_address.into(), recursive_proof_hash,
                ]
                    .span(),
            );
            session_dispatcher.attest_stage(session_id, stage_index, result_hash);
            result_hash
        }
    }
}
