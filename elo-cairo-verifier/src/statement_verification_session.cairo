use starknet::ContractAddress;

#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct StatementSessionInfo {
    pub submitter: ContractAddress,
    pub statement_hash: felt252,
    /// Caller-provided statement proof/result hash. This is an input to the
    /// final composition hash, not the hash recorded in the registry.
    pub proof_hash: felt252,
    pub program_hash: felt252,
    pub model_id: felt252,
    pub security_bits: u32,
    pub required_stage_mask: u32,
    pub completed_stage_mask: u32,
    pub finalized: bool,
}

#[starknet::interface]
pub trait IStatementVerificationSession<TContractState> {
    fn set_stage_policy(
        ref self: TContractState,
        program_hash: felt252,
        model_id: felt252,
        required_stage_mask: u32,
    );
    fn freeze_stage_policy(ref self: TContractState, program_hash: felt252, model_id: felt252);
    fn set_stage_verifier(ref self: TContractState, stage_index: u32, verifier: ContractAddress);
    fn freeze_stage_verifier(ref self: TContractState, stage_index: u32);
    fn open_session(
        ref self: TContractState,
        statement_hash: felt252,
        proof_hash: felt252,
        program_hash: felt252,
        model_id: felt252,
        security_bits: u32,
        required_stage_mask: u32,
    ) -> u64;
    fn attest_stage(
        ref self: TContractState, session_id: u64, stage_index: u32, fact_hash: felt252,
    );
    fn finalize_session(ref self: TContractState, session_id: u64);
    fn get_session(self: @TContractState, session_id: u64) -> StatementSessionInfo;
    fn get_composition_proof_hash(self: @TContractState, session_id: u64) -> felt252;
    fn get_stage_policy(self: @TContractState, program_hash: felt252, model_id: felt252) -> u32;
    fn is_stage_policy_frozen(
        self: @TContractState, program_hash: felt252, model_id: felt252,
    ) -> bool;
    fn get_stage_verifier(self: @TContractState, stage_index: u32) -> ContractAddress;
    fn is_stage_verifier_frozen(self: @TContractState, stage_index: u32) -> bool;
    fn get_session_stage_verifier(
        self: @TContractState, session_id: u64, stage_index: u32,
    ) -> ContractAddress;
    fn get_stage_attestor(
        self: @TContractState, session_id: u64, stage_index: u32,
    ) -> ContractAddress;
    fn get_stage_result_hash(self: @TContractState, session_id: u64, stage_index: u32) -> felt252;
    fn get_stage_fact_hash(self: @TContractState, session_id: u64, stage_index: u32) -> felt252;
}

#[starknet::contract]
pub mod StatementVerificationSessionContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::storage::{
        Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess,
        StoragePointerWriteAccess,
    };
    use starknet::{ContractAddress, get_block_timestamp, get_caller_address};
    use crate::statement_fact_registry::{
        IStatementFactRegistryDispatcher, IStatementFactRegistryDispatcherTrait,
    };
    use super::{IStatementVerificationSession, StatementSessionInfo};

    const MIN_SECURITY_BITS: u32 = 160;
    const MAX_STAGE_INDEX: u32 = 7;
    const MAX_STAGE_MASK: u32 = 255;
    const DOMAIN_STAGE_FACT: felt252 = 0x53544147455f46414354; // "STAGE_FACT"
    const DOMAIN_COMPOSITION_FACT: felt252 = 0x434f4d504f534954494f4e; // "COMPOSITION"

    #[storage]
    struct Storage {
        owner: ContractAddress,
        registry: ContractAddress,
        next_session_id: u64,
        stage_policy: Map<(felt252, felt252), u32>,
        stage_policy_frozen: Map<(felt252, felt252), bool>,
        stage_verifier: Map<u32, ContractAddress>,
        stage_verifier_frozen: Map<u32, bool>,
        submitter: Map<u64, ContractAddress>,
        statement_hash: Map<u64, felt252>,
        proof_hash: Map<u64, felt252>,
        program_hash: Map<u64, felt252>,
        model_id: Map<u64, felt252>,
        security_bits: Map<u64, u32>,
        required_stage_mask: Map<u64, u32>,
        completed_stage_mask: Map<u64, u32>,
        finalized: Map<u64, bool>,
        session_stage_verifier: Map<(u64, u32), ContractAddress>,
        stage_attestor: Map<(u64, u32), ContractAddress>,
        stage_result_hash: Map<(u64, u32), felt252>,
        stage_fact_hash: Map<(u64, u32), felt252>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        StageVerifierSet: StageVerifierSet,
        StageVerifierFrozen: StageVerifierFrozen,
        StagePolicySet: StagePolicySet,
        StagePolicyFrozen: StagePolicyFrozen,
        StatementSessionOpened: StatementSessionOpened,
        StatementStageAttested: StatementStageAttested,
        StatementSessionFinalized: StatementSessionFinalized,
    }

    #[derive(Drop, starknet::Event)]
    struct StageVerifierSet {
        #[key]
        stage_index: u32,
        verifier: ContractAddress,
        updated_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StageVerifierFrozen {
        #[key]
        stage_index: u32,
        verifier: ContractAddress,
        frozen_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StagePolicySet {
        #[key]
        program_hash: felt252,
        #[key]
        model_id: felt252,
        required_stage_mask: u32,
        updated_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StagePolicyFrozen {
        #[key]
        program_hash: felt252,
        #[key]
        model_id: felt252,
        required_stage_mask: u32,
        frozen_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StatementSessionOpened {
        #[key]
        session_id: u64,
        #[key]
        statement_hash: felt252,
        proof_hash: felt252,
        program_hash: felt252,
        model_id: felt252,
        security_bits: u32,
        required_stage_mask: u32,
        submitter: ContractAddress,
        opened_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct StatementStageAttested {
        #[key]
        session_id: u64,
        #[key]
        stage_index: u32,
        stage_bit: u32,
        completed_stage_mask: u32,
        fact_hash: felt252,
        result_hash: felt252,
        verifier: ContractAddress,
        attested_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct StatementSessionFinalized {
        #[key]
        session_id: u64,
        #[key]
        statement_hash: felt252,
        proof_hash: felt252,
        program_hash: felt252,
        model_id: felt252,
        security_bits: u32,
        finalized_by: ContractAddress,
        finalized_at: u64,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress, registry: ContractAddress) {
        let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
        assert!(owner != zero_addr, "owner cannot be zero");
        assert!(registry != zero_addr, "registry cannot be zero");
        self.owner.write(owner);
        self.registry.write(registry);
        self.next_session_id.write(1);
    }

    #[abi(embed_v0)]
    impl StatementVerificationSessionImpl of IStatementVerificationSession<ContractState> {
        fn set_stage_policy(
            ref self: ContractState,
            program_hash: felt252,
            model_id: felt252,
            required_stage_mask: u32,
        ) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "Session: owner only");
            assert!(program_hash != 0, "program hash missing");
            assert!(model_id != 0, "model id missing");
            assert!(required_stage_mask != 0, "stage mask missing");
            assert!(required_stage_mask <= MAX_STAGE_MASK, "stage mask too high");
            assert!(
                !self.stage_policy_frozen.read((program_hash, model_id)), "stage policy frozen",
            );

            self.stage_policy.write((program_hash, model_id), required_stage_mask);
            self
                .emit(
                    StagePolicySet {
                        program_hash, model_id, required_stage_mask, updated_by: caller,
                    },
                );
        }

        fn freeze_stage_policy(ref self: ContractState, program_hash: felt252, model_id: felt252) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "Session: owner only");
            assert!(program_hash != 0, "program hash missing");
            assert!(model_id != 0, "model id missing");
            let required_stage_mask = self.stage_policy.read((program_hash, model_id));
            assert!(required_stage_mask != 0, "stage policy missing");
            self.stage_policy_frozen.write((program_hash, model_id), true);
            self
                .emit(
                    StagePolicyFrozen {
                        program_hash, model_id, required_stage_mask, frozen_by: caller,
                    },
                );
        }

        fn set_stage_verifier(
            ref self: ContractState, stage_index: u32, verifier: ContractAddress,
        ) {
            assert!(get_caller_address() == self.owner.read(), "Session: owner only");
            assert!(stage_index <= MAX_STAGE_INDEX, "stage index too high");
            assert!(!self.stage_verifier_frozen.read(stage_index), "stage verifier frozen");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(verifier != zero_addr, "verifier cannot be zero");

            self.stage_verifier.write(stage_index, verifier);
            self.emit(StageVerifierSet { stage_index, verifier, updated_by: get_caller_address() });
        }

        fn freeze_stage_verifier(ref self: ContractState, stage_index: u32) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "Session: owner only");
            assert!(stage_index <= MAX_STAGE_INDEX, "stage index too high");
            let verifier = self.stage_verifier.read(stage_index);
            let verifier_felt: felt252 = verifier.into();
            assert!(verifier_felt != 0, "verifier cannot be zero");
            self.stage_verifier_frozen.write(stage_index, true);
            self.emit(StageVerifierFrozen { stage_index, verifier, frozen_by: caller });
        }

        fn open_session(
            ref self: ContractState,
            statement_hash: felt252,
            proof_hash: felt252,
            program_hash: felt252,
            model_id: felt252,
            security_bits: u32,
            required_stage_mask: u32,
        ) -> u64 {
            assert!(statement_hash != 0, "statement hash missing");
            assert!(proof_hash != 0, "proof hash missing");
            assert!(program_hash != 0, "program hash missing");
            assert!(model_id != 0, "model id missing");
            assert!(security_bits >= MIN_SECURITY_BITS, "security below 160");
            assert!(required_stage_mask != 0, "stage mask missing");
            assert!(required_stage_mask <= MAX_STAGE_MASK, "stage mask too high");
            self._assert_stage_policy(program_hash, model_id, required_stage_mask);

            let session_id = self.next_session_id.read();
            self.next_session_id.write(session_id + 1);
            self._snapshot_required_stage_verifiers(session_id, required_stage_mask);

            let submitter = get_caller_address();
            self.submitter.write(session_id, submitter);
            self.statement_hash.write(session_id, statement_hash);
            self.proof_hash.write(session_id, proof_hash);
            self.program_hash.write(session_id, program_hash);
            self.model_id.write(session_id, model_id);
            self.security_bits.write(session_id, security_bits);
            self.required_stage_mask.write(session_id, required_stage_mask);
            self.completed_stage_mask.write(session_id, 0);
            self.finalized.write(session_id, false);

            self
                .emit(
                    StatementSessionOpened {
                        session_id,
                        statement_hash,
                        proof_hash,
                        program_hash,
                        model_id,
                        security_bits,
                        required_stage_mask,
                        submitter,
                        opened_at: get_block_timestamp(),
                    },
                );
            session_id
        }

        fn attest_stage(
            ref self: ContractState, session_id: u64, stage_index: u32, fact_hash: felt252,
        ) {
            let submitter_felt: felt252 = self.submitter.read(session_id).into();
            assert!(submitter_felt != 0, "session missing");
            assert!(!self.finalized.read(session_id), "session finalized");
            assert!(stage_index <= MAX_STAGE_INDEX, "stage index too high");
            assert!(fact_hash != 0, "fact hash missing");

            let stage_bit = stage_bit_for(stage_index);
            let required = self.required_stage_mask.read(session_id);
            assert!(has_stage(required, stage_bit), "stage not required");

            let verifier = self.session_stage_verifier.read((session_id, stage_index));
            let verifier_felt: felt252 = verifier.into();
            assert!(verifier_felt != 0, "session stage verifier missing");
            assert!(get_caller_address() == verifier, "Session: stage verifier only");

            let completed = self.completed_stage_mask.read(session_id);
            assert!(!has_stage(completed, stage_bit), "stage already attested");
            let new_completed = completed + stage_bit;
            let stage_fact_hash = bound_stage_fact_hash(
                session_id,
                self.statement_hash.read(session_id),
                self.proof_hash.read(session_id),
                self.program_hash.read(session_id),
                self.model_id.read(session_id),
                self.security_bits.read(session_id),
                self.required_stage_mask.read(session_id),
                stage_index,
                verifier,
                fact_hash,
            );

            self.stage_attestor.write((session_id, stage_index), verifier);
            self.stage_result_hash.write((session_id, stage_index), fact_hash);
            self.stage_fact_hash.write((session_id, stage_index), stage_fact_hash);
            self.completed_stage_mask.write(session_id, new_completed);

            self
                .emit(
                    StatementStageAttested {
                        session_id,
                        stage_index,
                        stage_bit,
                        completed_stage_mask: new_completed,
                        fact_hash: stage_fact_hash,
                        result_hash: fact_hash,
                        verifier,
                        attested_at: get_block_timestamp(),
                    },
                );
        }

        fn finalize_session(ref self: ContractState, session_id: u64) {
            let submitter_felt: felt252 = self.submitter.read(session_id).into();
            assert!(submitter_felt != 0, "session missing");
            assert!(!self.finalized.read(session_id), "session finalized");
            let required = self.required_stage_mask.read(session_id);
            let completed = self.completed_stage_mask.read(session_id);
            assert!(mask_satisfied(required, completed), "required stages missing");

            self.finalized.write(session_id, true);

            let statement_hash = self.statement_hash.read(session_id);
            let proof_hash = self._compute_composition_proof_hash(session_id);
            let program_hash = self.program_hash.read(session_id);
            let model_id = self.model_id.read(session_id);
            let security_bits = self.security_bits.read(session_id);
            let registry = IStatementFactRegistryDispatcher {
                contract_address: self.registry.read(),
            };
            registry
                .record_statement_fact(
                    statement_hash, proof_hash, program_hash, model_id, security_bits,
                );

            self
                .emit(
                    StatementSessionFinalized {
                        session_id,
                        statement_hash,
                        proof_hash,
                        program_hash,
                        model_id,
                        security_bits,
                        finalized_by: get_caller_address(),
                        finalized_at: get_block_timestamp(),
                    },
                );
        }

        fn get_session(self: @ContractState, session_id: u64) -> StatementSessionInfo {
            StatementSessionInfo {
                submitter: self.submitter.read(session_id),
                statement_hash: self.statement_hash.read(session_id),
                proof_hash: self.proof_hash.read(session_id),
                program_hash: self.program_hash.read(session_id),
                model_id: self.model_id.read(session_id),
                security_bits: self.security_bits.read(session_id),
                required_stage_mask: self.required_stage_mask.read(session_id),
                completed_stage_mask: self.completed_stage_mask.read(session_id),
                finalized: self.finalized.read(session_id),
            }
        }

        fn get_composition_proof_hash(self: @ContractState, session_id: u64) -> felt252 {
            self._compute_composition_proof_hash(session_id)
        }

        fn get_stage_policy(self: @ContractState, program_hash: felt252, model_id: felt252) -> u32 {
            self.stage_policy.read((program_hash, model_id))
        }

        fn is_stage_policy_frozen(
            self: @ContractState, program_hash: felt252, model_id: felt252,
        ) -> bool {
            self.stage_policy_frozen.read((program_hash, model_id))
        }

        fn get_stage_verifier(self: @ContractState, stage_index: u32) -> ContractAddress {
            self.stage_verifier.read(stage_index)
        }

        fn is_stage_verifier_frozen(self: @ContractState, stage_index: u32) -> bool {
            self.stage_verifier_frozen.read(stage_index)
        }

        fn get_session_stage_verifier(
            self: @ContractState, session_id: u64, stage_index: u32,
        ) -> ContractAddress {
            self.session_stage_verifier.read((session_id, stage_index))
        }

        fn get_stage_attestor(
            self: @ContractState, session_id: u64, stage_index: u32,
        ) -> ContractAddress {
            self.stage_attestor.read((session_id, stage_index))
        }

        fn get_stage_result_hash(
            self: @ContractState, session_id: u64, stage_index: u32,
        ) -> felt252 {
            self.stage_result_hash.read((session_id, stage_index))
        }

        fn get_stage_fact_hash(self: @ContractState, session_id: u64, stage_index: u32) -> felt252 {
            self.stage_fact_hash.read((session_id, stage_index))
        }
    }

    #[generate_trait]
    impl InternalImpl of InternalTrait {
        fn _assert_stage_policy(
            self: @ContractState,
            program_hash: felt252,
            model_id: felt252,
            requested_stage_mask: u32,
        ) {
            let policy_mask = self.stage_policy.read((program_hash, model_id));
            if policy_mask != 0 {
                assert!(
                    mask_satisfied(policy_mask, requested_stage_mask),
                    "required policy stages missing",
                );
            }
        }

        fn _snapshot_required_stage_verifiers(
            ref self: ContractState, session_id: u64, required_stage_mask: u32,
        ) {
            let mut stage_index: u32 = 0;
            loop {
                if stage_index > MAX_STAGE_INDEX {
                    break;
                }
                let bit = stage_bit_for(stage_index);
                if has_stage(required_stage_mask, bit) {
                    let verifier = self.stage_verifier.read(stage_index);
                    let verifier_felt: felt252 = verifier.into();
                    assert!(verifier_felt != 0, "required verifier missing");
                    self.session_stage_verifier.write((session_id, stage_index), verifier);
                }
                stage_index += 1;
            };
        }

        fn _compute_composition_proof_hash(self: @ContractState, session_id: u64) -> felt252 {
            let submitter_felt: felt252 = self.submitter.read(session_id).into();
            assert!(submitter_felt != 0, "session missing");

            let mut hash_input = array![
                DOMAIN_COMPOSITION_FACT, session_id.into(), self.statement_hash.read(session_id),
                self.proof_hash.read(session_id), self.program_hash.read(session_id),
                self.model_id.read(session_id), self.security_bits.read(session_id).into(),
                self.required_stage_mask.read(session_id).into(),
                self.completed_stage_mask.read(session_id).into(),
            ];
            let required = self.required_stage_mask.read(session_id);
            let mut stage_index: u32 = 0;
            loop {
                if stage_index > MAX_STAGE_INDEX {
                    break;
                }
                let bit = stage_bit_for(stage_index);
                if has_stage(required, bit) {
                    let fact_hash = self.stage_fact_hash.read((session_id, stage_index));
                    let attestor = self.stage_attestor.read((session_id, stage_index));
                    let attestor_felt: felt252 = attestor.into();
                    assert!(fact_hash != 0, "stage fact missing");
                    assert!(attestor_felt != 0, "stage attestor missing");
                    hash_input.append(stage_index.into());
                    hash_input.append(attestor.into());
                    hash_input.append(fact_hash);
                }
                stage_index += 1;
            }
            poseidon_hash_span(hash_input.span())
        }
    }

    fn bound_stage_fact_hash(
        session_id: u64,
        statement_hash: felt252,
        proof_hash: felt252,
        program_hash: felt252,
        model_id: felt252,
        security_bits: u32,
        required_stage_mask: u32,
        stage_index: u32,
        verifier: ContractAddress,
        result_hash: felt252,
    ) -> felt252 {
        poseidon_hash_span(
            array![
                DOMAIN_STAGE_FACT, session_id.into(), statement_hash, proof_hash, program_hash,
                model_id, security_bits.into(), required_stage_mask.into(), stage_index.into(),
                verifier.into(), result_hash,
            ]
                .span(),
        )
    }

    fn stage_bit_for(stage_index: u32) -> u32 {
        if stage_index == 0 {
            1
        } else if stage_index == 1 {
            2
        } else if stage_index == 2 {
            4
        } else if stage_index == 3 {
            8
        } else if stage_index == 4 {
            16
        } else if stage_index == 5 {
            32
        } else if stage_index == 6 {
            64
        } else {
            128
        }
    }

    fn has_stage(mask: u32, stage_bit: u32) -> bool {
        let q = mask / stage_bit;
        q - ((q / 2) * 2) == 1
    }

    fn mask_satisfied(required: u32, completed: u32) -> bool {
        let mut stage_index: u32 = 0;
        loop {
            if stage_index > MAX_STAGE_INDEX {
                break;
            }
            let bit = stage_bit_for(stage_index);
            if has_stage(required, bit) {
                if !has_stage(completed, bit) {
                    return false;
                }
            }
            stage_index += 1;
        }
        true
    }
}
