use starknet::ContractAddress;

#[starknet::interface]
pub trait IHadesLogupFactSource<TContractState> {
    fn is_hades_logup_verified(self: @TContractState, statement_hash: felt252) -> bool;
    fn get_hades_logup_proof_hash(self: @TContractState, statement_hash: felt252) -> felt252;
    fn get_hades_commitment(self: @TContractState, statement_hash: felt252) -> felt252;
}

#[starknet::interface]
pub trait IHadesLogupStageVerifier<TContractState> {
    fn set_fact_source(ref self: TContractState, hades_fact_source: ContractAddress);
    fn freeze_fact_source(ref self: TContractState);
    fn get_fact_source(self: @TContractState) -> ContractAddress;
    fn is_fact_source_frozen(self: @TContractState) -> bool;
    fn verify_and_attest(
        ref self: TContractState,
        session: ContractAddress,
        session_id: u64,
        stage_index: u32,
        expected_hades_proof_hash: felt252,
        expected_hades_commitment: felt252,
    ) -> felt252;
}

#[starknet::contract]
pub mod HadesLogupStageVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::storage::{StoragePointerReadAccess, StoragePointerWriteAccess};
    use starknet::{ContractAddress, get_caller_address};
    use crate::statement_verification_session::{
        IStatementVerificationSessionDispatcher, IStatementVerificationSessionDispatcherTrait,
    };
    use super::{
        IHadesLogupFactSourceDispatcher, IHadesLogupFactSourceDispatcherTrait,
        IHadesLogupStageVerifier,
    };

    const DOMAIN_HADES_LOGUP_FACT: felt252 = 0x48414445535f4c4f4750; // "HADES_LOGUP"

    #[storage]
    struct Storage {
        owner: ContractAddress,
        hades_fact_source: ContractAddress,
        fact_source_frozen: bool,
    }

    #[constructor]
    fn constructor(
        ref self: ContractState, owner: ContractAddress, hades_fact_source: ContractAddress,
    ) {
        let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
        assert!(owner != zero_addr, "owner cannot be zero");
        assert!(hades_fact_source != zero_addr, "fact source cannot be zero");
        self.owner.write(owner);
        self.hades_fact_source.write(hades_fact_source);
        self.fact_source_frozen.write(false);
    }

    #[abi(embed_v0)]
    impl HadesLogupStageVerifierImpl of IHadesLogupStageVerifier<ContractState> {
        fn set_fact_source(ref self: ContractState, hades_fact_source: ContractAddress) {
            assert!(get_caller_address() == self.owner.read(), "Stage: owner only");
            assert!(!self.fact_source_frozen.read(), "fact source frozen");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(hades_fact_source != zero_addr, "fact source cannot be zero");
            self.hades_fact_source.write(hades_fact_source);
        }

        fn freeze_fact_source(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Stage: owner only");
            self.fact_source_frozen.write(true);
        }

        fn get_fact_source(self: @ContractState) -> ContractAddress {
            self.hades_fact_source.read()
        }

        fn is_fact_source_frozen(self: @ContractState) -> bool {
            self.fact_source_frozen.read()
        }

        fn verify_and_attest(
            ref self: ContractState,
            session: ContractAddress,
            session_id: u64,
            stage_index: u32,
            expected_hades_proof_hash: felt252,
            expected_hades_commitment: felt252,
        ) -> felt252 {
            let session_dispatcher = IStatementVerificationSessionDispatcher {
                contract_address: session,
            };
            let info = session_dispatcher.get_session(session_id);
            assert!(info.statement_hash != 0, "statement hash missing");

            let fact_source_address = self.hades_fact_source.read();
            let fact_source = IHadesLogupFactSourceDispatcher {
                contract_address: fact_source_address,
            };
            assert!(
                fact_source.is_hades_logup_verified(info.statement_hash),
                "hades logup fact missing",
            );
            let hades_proof_hash = fact_source.get_hades_logup_proof_hash(info.statement_hash);
            let hades_commitment = fact_source.get_hades_commitment(info.statement_hash);
            assert!(hades_proof_hash != 0, "hades proof hash missing");
            assert!(hades_commitment != 0, "hades commitment missing");
            if expected_hades_proof_hash != 0 {
                assert!(hades_proof_hash == expected_hades_proof_hash, "hades proof hash mismatch");
            }
            if expected_hades_commitment != 0 {
                assert!(hades_commitment == expected_hades_commitment, "hades commitment mismatch");
            }

            let result_hash = poseidon_hash_span(
                array![
                    DOMAIN_HADES_LOGUP_FACT, info.statement_hash, fact_source_address.into(),
                    hades_proof_hash, hades_commitment,
                ]
                    .span(),
            );
            session_dispatcher.attest_stage(session_id, stage_index, result_hash);
            result_hash
        }
    }
}
