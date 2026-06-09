// ContractRegistry: on-chain attestation of contract metadata.
//
// Stores (contract_address → is_verified, has_source) attestations.
// The firewall reads from this during resolve_action_with_proof to
// cross-check target_flags features from the proven classifier IO.
//
// Design: owner-gated writes, anyone can read. The owner is typically
// a multisig or governance contract that aggregates attestation data
// from block explorers (Starkscan, Voyager) or auditors.

use starknet::ContractAddress;

#[starknet::interface]
pub trait IContractRegistry<TContractState> {
    /// Attest contract metadata (owner only).
    fn attest(
        ref self: TContractState, contract_address: felt252, is_verified: bool, has_source: bool,
    );

    /// Batch attest multiple contracts (owner only).
    fn attest_batch(
        ref self: TContractState,
        addresses: Array<felt252>,
        verified: Array<bool>,
        sources: Array<bool>,
    );

    /// Query whether a contract is verified.
    fn is_verified(self: @TContractState, contract_address: felt252) -> bool;

    /// Query whether a contract has published source.
    fn has_source(self: @TContractState, contract_address: felt252) -> bool;

    /// Query whether any attestation exists for this contract.
    fn is_attested(self: @TContractState, contract_address: felt252) -> bool;

    /// Get the registry owner.
    fn get_owner(self: @TContractState) -> ContractAddress;
}

#[starknet::contract]
pub mod ContractRegistry {
    use starknet::storage::{
        Map, StoragePathEntry, StoragePointerReadAccess, StoragePointerWriteAccess,
    };
    use starknet::{ContractAddress, get_caller_address};

    #[storage]
    struct Storage {
        owner: ContractAddress,
        contract_verified: Map<felt252, bool>,
        contract_has_source: Map<felt252, bool>,
        contract_attested: Map<felt252, bool>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ContractAttested: ContractAttested,
    }

    #[derive(Drop, starknet::Event)]
    struct ContractAttested {
        #[key]
        contract_address: felt252,
        is_verified: bool,
        has_source: bool,
        attester: ContractAddress,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    #[abi(embed_v0)]
    impl ContractRegistryImpl of super::IContractRegistry<ContractState> {
        fn attest(
            ref self: ContractState, contract_address: felt252, is_verified: bool, has_source: bool,
        ) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(contract_address != 0, "ADDRESS_CANNOT_BE_ZERO");
            self.contract_verified.entry(contract_address).write(is_verified);
            self.contract_has_source.entry(contract_address).write(has_source);
            self.contract_attested.entry(contract_address).write(true);
            self
                .emit(
                    ContractAttested {
                        contract_address, is_verified, has_source, attester: get_caller_address(),
                    },
                );
        }

        fn attest_batch(
            ref self: ContractState,
            addresses: Array<felt252>,
            verified: Array<bool>,
            sources: Array<bool>,
        ) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(addresses.len() == verified.len(), "LENGTH_MISMATCH");
            assert!(addresses.len() == sources.len(), "LENGTH_MISMATCH");

            let mut i: u32 = 0;
            loop {
                if i >= addresses.len() {
                    break;
                }
                let addr = *addresses.at(i);
                let v = *verified.at(i);
                let s = *sources.at(i);
                self.contract_verified.entry(addr).write(v);
                self.contract_has_source.entry(addr).write(s);
                self.contract_attested.entry(addr).write(true);
                self
                    .emit(
                        ContractAttested {
                            contract_address: addr,
                            is_verified: v,
                            has_source: s,
                            attester: get_caller_address(),
                        },
                    );
                i += 1;
            };
        }

        fn is_verified(self: @ContractState, contract_address: felt252) -> bool {
            self.contract_verified.entry(contract_address).read()
        }

        fn has_source(self: @ContractState, contract_address: felt252) -> bool {
            self.contract_has_source.entry(contract_address).read()
        }

        fn is_attested(self: @ContractState, contract_address: felt252) -> bool {
            self.contract_attested.entry(contract_address).read()
        }

        fn get_owner(self: @ContractState) -> ContractAddress {
            self.owner.read()
        }
    }
}
