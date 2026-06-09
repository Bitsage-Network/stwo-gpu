#[starknet::interface]
pub trait IMockHadesLogupFactSource<TContractState> {
    fn set_hades_logup_fact(
        ref self: TContractState,
        statement_hash: felt252,
        proof_hash: felt252,
        hades_commitment: felt252,
        verified: bool,
    );
    fn is_hades_logup_verified(self: @TContractState, statement_hash: felt252) -> bool;
    fn get_hades_logup_proof_hash(self: @TContractState, statement_hash: felt252) -> felt252;
    fn get_hades_commitment(self: @TContractState, statement_hash: felt252) -> felt252;
}

#[starknet::contract]
pub mod MockHadesLogupFactSourceContract {
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess};
    use super::IMockHadesLogupFactSource;

    #[storage]
    struct Storage {
        verified: Map<felt252, bool>,
        proof_hash: Map<felt252, felt252>,
        hades_commitment: Map<felt252, felt252>,
    }

    #[abi(embed_v0)]
    impl MockHadesLogupFactSourceImpl of IMockHadesLogupFactSource<ContractState> {
        fn set_hades_logup_fact(
            ref self: ContractState,
            statement_hash: felt252,
            proof_hash: felt252,
            hades_commitment: felt252,
            verified: bool,
        ) {
            self.verified.write(statement_hash, verified);
            self.proof_hash.write(statement_hash, proof_hash);
            self.hades_commitment.write(statement_hash, hades_commitment);
        }

        fn is_hades_logup_verified(self: @ContractState, statement_hash: felt252) -> bool {
            self.verified.read(statement_hash)
        }

        fn get_hades_logup_proof_hash(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.proof_hash.read(statement_hash)
        }

        fn get_hades_commitment(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.hades_commitment.read(statement_hash)
        }
    }
}
