#[starknet::interface]
pub trait IMockRecursiveStatementFactSource<TContractState> {
    fn set_statement_fact(
        ref self: TContractState, statement_hash: felt252, proof_hash: felt252, verified: bool,
    );
    fn is_recursive_statement_verified(
        self: @TContractState, conversation_statement_hash: felt252,
    ) -> bool;
    fn get_recursive_statement_proof_hash(
        self: @TContractState, conversation_statement_hash: felt252,
    ) -> felt252;
}

#[starknet::contract]
pub mod MockRecursiveStatementFactSourceContract {
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess};
    use super::IMockRecursiveStatementFactSource;

    #[storage]
    struct Storage {
        verified: Map<felt252, bool>,
        proof_hash: Map<felt252, felt252>,
    }

    #[abi(embed_v0)]
    impl MockRecursiveStatementFactSourceImpl of IMockRecursiveStatementFactSource<ContractState> {
        fn set_statement_fact(
            ref self: ContractState, statement_hash: felt252, proof_hash: felt252, verified: bool,
        ) {
            self.verified.write(statement_hash, verified);
            self.proof_hash.write(statement_hash, proof_hash);
        }

        fn is_recursive_statement_verified(
            self: @ContractState, conversation_statement_hash: felt252,
        ) -> bool {
            self.verified.read(conversation_statement_hash)
        }

        fn get_recursive_statement_proof_hash(
            self: @ContractState, conversation_statement_hash: felt252,
        ) -> felt252 {
            self.proof_hash.read(conversation_statement_hash)
        }
    }
}
