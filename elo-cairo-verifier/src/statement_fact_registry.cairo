use starknet::ContractAddress;

#[starknet::interface]
pub trait IStatementFactRegistry<TContractState> {
    fn set_verifier(ref self: TContractState, verifier: ContractAddress);
    fn freeze_verifier(ref self: TContractState);
    fn record_statement_fact(
        ref self: TContractState,
        statement_hash: felt252,
        proof_hash: felt252,
        program_hash: felt252,
        model_id: felt252,
        security_bits: u32,
    );
    fn is_statement_verified(self: @TContractState, statement_hash: felt252) -> bool;
    fn get_statement_proof_hash(self: @TContractState, statement_hash: felt252) -> felt252;
    fn get_statement_program_hash(self: @TContractState, statement_hash: felt252) -> felt252;
    fn get_statement_model_id(self: @TContractState, statement_hash: felt252) -> felt252;
    fn get_statement_security_bits(self: @TContractState, statement_hash: felt252) -> u32;
    fn get_verifier(self: @TContractState) -> ContractAddress;
    fn is_verifier_frozen(self: @TContractState) -> bool;
}

#[starknet::contract]
pub mod StatementFactRegistryContract {
    use starknet::storage::{
        Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess,
        StoragePointerWriteAccess,
    };
    use starknet::{ContractAddress, get_block_timestamp, get_caller_address};
    use super::IStatementFactRegistry;

    const MIN_SECURITY_BITS: u32 = 160;

    #[storage]
    struct Storage {
        owner: ContractAddress,
        verifier: ContractAddress,
        verifier_frozen: bool,
        verified: Map<felt252, bool>,
        proof_hash: Map<felt252, felt252>,
        program_hash: Map<felt252, felt252>,
        model_id: Map<felt252, felt252>,
        security_bits: Map<felt252, u32>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        VerifierUpdated: VerifierUpdated,
        VerifierFrozen: VerifierFrozen,
        StatementFactRecorded: StatementFactRecorded,
    }

    #[derive(Drop, starknet::Event)]
    struct VerifierUpdated {
        verifier: ContractAddress,
        updated_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct VerifierFrozen {
        verifier: ContractAddress,
        frozen_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StatementFactRecorded {
        #[key]
        statement_hash: felt252,
        proof_hash: felt252,
        program_hash: felt252,
        model_id: felt252,
        security_bits: u32,
        recorded_at: u64,
        verifier: ContractAddress,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress, verifier: ContractAddress) {
        let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
        assert!(owner != zero_addr, "owner cannot be zero");
        self.owner.write(owner);
        self.verifier.write(verifier);
        self.verifier_frozen.write(false);
    }

    #[abi(embed_v0)]
    impl StatementFactRegistryImpl of IStatementFactRegistry<ContractState> {
        fn set_verifier(ref self: ContractState, verifier: ContractAddress) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "Registry: owner only");
            assert!(!self.verifier_frozen.read(), "verifier frozen");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(verifier != zero_addr, "verifier cannot be zero");
            self.verifier.write(verifier);
            self.emit(VerifierUpdated { verifier, updated_by: caller });
        }

        fn freeze_verifier(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "Registry: owner only");
            let verifier = self.verifier.read();
            let verifier_felt: felt252 = verifier.into();
            assert!(verifier_felt != 0, "verifier cannot be zero");
            self.verifier_frozen.write(true);
            self.emit(VerifierFrozen { verifier, frozen_by: caller });
        }

        fn record_statement_fact(
            ref self: ContractState,
            statement_hash: felt252,
            proof_hash: felt252,
            program_hash: felt252,
            model_id: felt252,
            security_bits: u32,
        ) {
            let caller = get_caller_address();
            assert!(caller == self.verifier.read(), "Registry: verifier only");
            assert!(statement_hash != 0, "statement hash missing");
            assert!(proof_hash != 0, "proof hash missing");
            assert!(program_hash != 0, "program hash missing");
            assert!(model_id != 0, "model id missing");
            assert!(security_bits >= MIN_SECURITY_BITS, "security below 160");
            assert!(!self.verified.read(statement_hash), "statement already recorded");

            self.verified.write(statement_hash, true);
            self.proof_hash.write(statement_hash, proof_hash);
            self.program_hash.write(statement_hash, program_hash);
            self.model_id.write(statement_hash, model_id);
            self.security_bits.write(statement_hash, security_bits);

            self
                .emit(
                    StatementFactRecorded {
                        statement_hash,
                        proof_hash,
                        program_hash,
                        model_id,
                        security_bits,
                        recorded_at: get_block_timestamp(),
                        verifier: caller,
                    },
                );
        }

        fn is_statement_verified(self: @ContractState, statement_hash: felt252) -> bool {
            self.verified.read(statement_hash)
        }

        fn get_statement_proof_hash(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.proof_hash.read(statement_hash)
        }

        fn get_statement_program_hash(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.program_hash.read(statement_hash)
        }

        fn get_statement_model_id(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.model_id.read(statement_hash)
        }

        fn get_statement_security_bits(self: @ContractState, statement_hash: felt252) -> u32 {
            self.security_bits.read(statement_hash)
        }

        fn get_verifier(self: @ContractState) -> ContractAddress {
            self.verifier.read()
        }

        fn is_verifier_frozen(self: @ContractState) -> bool {
            self.verifier_frozen.read()
        }
    }
}
