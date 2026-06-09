use starknet::ContractAddress;

#[starknet::interface]
pub trait IMockStatementVerifier<TContractState> {
    fn set_statement_verified(
        ref self: TContractState, statement_hash: felt252, proof_hash: felt252, verified: bool,
    );
    fn is_statement_verified(self: @TContractState, statement_hash: felt252) -> bool;
    fn get_statement_proof_hash(self: @TContractState, statement_hash: felt252) -> felt252;
}

#[starknet::contract]
pub mod MockStatementVerifierContract {
    use starknet::get_caller_address;
    use starknet::storage::{
        Map, StoragePathEntry, StoragePointerReadAccess, StoragePointerWriteAccess,
    };
    use super::{ContractAddress, IMockStatementVerifier};

    #[storage]
    struct Storage {
        owner: ContractAddress,
        verified: Map<felt252, bool>,
        proof_hash: Map<felt252, felt252>,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    #[abi(embed_v0)]
    impl MockStatementVerifierImpl of IMockStatementVerifier<ContractState> {
        fn set_statement_verified(
            ref self: ContractState, statement_hash: felt252, proof_hash: felt252, verified: bool,
        ) {
            assert!(get_caller_address() == self.owner.read(), "Mock: owner only");
            self.verified.entry(statement_hash).write(verified);
            self.proof_hash.entry(statement_hash).write(proof_hash);
        }

        fn is_statement_verified(self: @ContractState, statement_hash: felt252) -> bool {
            self.verified.entry(statement_hash).read()
        }

        fn get_statement_proof_hash(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.proof_hash.entry(statement_hash).read()
        }
    }
}
