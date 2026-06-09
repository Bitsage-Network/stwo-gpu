// ═══════════════════════════════════════════════════════════════════════════
// General-Purpose STWO On-Chain Verifier
// ═══════════════════════════════════════════════════════════════════════════
//
// Verifies ANY Cairo program's STWO STARK proof on-chain.
// First general-purpose STWO STARK verifier on any blockchain.
//
// Two verification modes:
//   1. Single-TX: For proofs ≤ 5000 felts (small programs or with M31 packing)
//   2. Streaming: For proofs > 5000 felts (split across multiple TXs)
//
// Security: 160 bits minimum (pow + log_blowup × n_queries ≥ 160)
// No trusted setup. Pure algebraic verification.

use stwo_cairo_air::{CairoProof, VerificationOutput, get_verification_output, verify_cairo};

/// Registered program info.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct ProgramInfo {
    pub program_hash: felt252,
    pub min_security_bits: u32,
    pub owner: starknet::ContractAddress,
}

#[starknet::interface]
pub trait IGeneralStwoVerifier<TContractState> {
    /// Register a program for on-chain verification.
    fn register_program(ref self: TContractState, program_hash: felt252, min_security_bits: u32);

    /// Single-TX verification (for proofs ≤ 5000 felts).
    fn verify_stwo(ref self: TContractState, proof: CairoProof) -> VerificationOutput;

    /// Statement-bound verification for ML conversation/action proofs.
    ///
    /// The Cairo program being proven must emit the canonical 19 statement felts.
    /// With `general_stwo_poseidon`, stwo_cairo_air exposes this as
    /// `VerificationOutput.output_hash = poseidon_hash_span(statement_felts)`.
    fn verify_conversation_stwo(
        ref self: TContractState,
        proof: CairoProof,
        expected_program_hash: felt252,
        expected_output_hash: felt252,
        statement_hash: felt252,
        model_id: felt252,
    ) -> VerificationOutput;

    /// Strict statement-bound verification. Production callers should prefer
    /// this entrypoint because it recomputes the canonical statement hash from
    /// the 19 emitted statement felts and rejects caller-side relabeling of
    /// model/program/security metadata.
    fn verify_conversation_stwo_with_statement(
        ref self: TContractState,
        proof: CairoProof,
        expected_program_hash: felt252,
        statement_hash: felt252,
        model_id: felt252,
        statement_felts: Array<felt252>,
    ) -> VerificationOutput;

    /// Streaming: open a verification session.
    fn stream_open(ref self: TContractState, expected_total_felts: u32) -> u64;

    /// Streaming: upload a chunk of proof data.
    fn stream_chunk(
        ref self: TContractState, session_id: u64, chunk_idx: u32, chunk: Array<felt252>,
    );

    /// Streaming: finalize — reassemble proof, verify, record on-chain.
    fn stream_verify(ref self: TContractState, session_id: u64) -> VerificationOutput;

    fn is_verified(self: @TContractState, proof_hash: felt252) -> bool;
    fn is_statement_verified(self: @TContractState, statement_hash: felt252) -> bool;
    fn get_statement_proof_hash(self: @TContractState, statement_hash: felt252) -> felt252;
    fn get_verification_count(self: @TContractState, program_hash: felt252) -> u64;

    /// Propose a contract class upgrade (owner only, subject to timelock).
    fn propose_upgrade(ref self: TContractState, new_class_hash: starknet::ClassHash);
    /// Execute a proposed upgrade after the timelock has elapsed.
    fn execute_upgrade(ref self: TContractState);
    /// Cancel a pending upgrade.
    fn cancel_upgrade(ref self: TContractState);
    /// Get the pending upgrade info.
    fn get_pending_upgrade(self: @TContractState) -> (starknet::ClassHash, u64);
}

#[starknet::contract]
mod GeneralStwoVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::storage::{
        Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess,
        StoragePointerWriteAccess,
    };
    use starknet::{ContractAddress, get_block_timestamp, get_caller_address};
    use stwo_verifier_core::pcs::PcsConfigTrait;
    use super::{
        CairoProof, IGeneralStwoVerifier, ProgramInfo, VerificationOutput, get_verification_output,
        verify_cairo,
    };

    const MIN_SECURITY_BITS: u32 = 160;
    const MAX_CHUNK_SIZE: u32 = 4900; // Leave room for TX overhead
    const DOMAIN_BATCH: felt252 = 0x43424154; // "CBAT"
    const CONVERSATION_STATEMENT_VERSION: felt252 = 1;
    const CONVERSATION_STATEMENT_FELTS: u32 = 19;

    #[storage]
    struct Storage {
        owner: ContractAddress,
        // Program registry
        programs: Map<felt252, ProgramInfo>,
        // Verification results
        verified_proofs: Map<felt252, bool>,
        verified_statements: Map<felt252, bool>,
        statement_proof_hash: Map<felt252, felt252>,
        verification_count: Map<felt252, u64>,
        // Last verification per program
        last_proof_hash: Map<felt252, felt252>,
        last_verified_at: Map<felt252, u64>,
        // Streaming state
        next_session_id: u64,
        session_owner: Map<u64, ContractAddress>,
        session_total_felts: Map<u64, u32>,
        session_received_felts: Map<u64, u32>,
        session_chunks_received: Map<u64, u32>,
        session_sealed: Map<u64, bool>,
        // Streaming data: (session_id, flat_index) → felt252
        session_data: Map<(u64, u32), felt252>,
        // Running hash of all chunks for integrity
        session_data_hash: Map<u64, felt252>,
        // Upgradability (timelock)
        pending_upgrade: starknet::ClassHash,
        upgrade_proposed_at: u64,
    }

    // 5 minutes for dev/Sepolia. Bump to 86400 (24h) before mainnet.
    const UPGRADE_DELAY: u64 = 300;

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ProgramRegistered: ProgramRegistered,
        StwoProofVerified: StwoProofVerified,
        ConversationStatementVerified: ConversationStatementVerified,
        StreamOpened: StreamOpened,
        StreamChunkReceived: StreamChunkReceived,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
    }

    #[derive(Drop, starknet::Event)]
    struct ProgramRegistered {
        #[key]
        program_hash: felt252,
        min_security_bits: u32,
        registered_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StwoProofVerified {
        #[key]
        program_hash: felt252,
        proof_hash: felt252,
        verification_count: u64,
        verified_at: u64,
        submitter: ContractAddress,
        mode: felt252 // 'single' or 'stream'
    }

    #[derive(Drop, starknet::Event)]
    struct ConversationStatementVerified {
        #[key]
        statement_hash: felt252,
        #[key]
        model_id: felt252,
        program_hash: felt252,
        output_hash: felt252,
        proof_hash: felt252,
        verified_at: u64,
        submitter: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct UpgradeProposed {
        new_class_hash: starknet::ClassHash,
        proposed_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct UpgradeExecuted {
        new_class_hash: starknet::ClassHash,
        executed_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct StreamOpened {
        #[key]
        session_id: u64,
        expected_total_felts: u32,
        owner: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StreamChunkReceived {
        #[key]
        session_id: u64,
        chunk_idx: u32,
        chunk_size: u32,
        total_received: u32,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
        self.next_session_id.write(1);
    }

    #[abi(embed_v0)]
    impl GeneralStwoVerifierImpl of IGeneralStwoVerifier<ContractState> {
        fn register_program(
            ref self: ContractState, program_hash: felt252, min_security_bits: u32,
        ) {
            assert!(min_security_bits >= MIN_SECURITY_BITS, "min_security_bits must be >= 160");
            assert!(program_hash != 0, "program_hash cannot be zero");

            let caller = get_caller_address();
            self
                .programs
                .write(
                    program_hash, ProgramInfo { program_hash, min_security_bits, owner: caller },
                );
            self.emit(ProgramRegistered { program_hash, min_security_bits, registered_by: caller });
        }

        // ─── Single-TX verification
        // ───────────────────────────────────────
        fn verify_stwo(ref self: ContractState, proof: CairoProof) -> VerificationOutput {
            let output = get_verification_output(proof: @proof);

            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Security {} < 160", security);

            // FULL CRYPTOGRAPHIC STARK VERIFICATION
            verify_cairo(proof);

            let proof_hash = poseidon_hash_span(
                array![output.program_hash, output.output_hash].span(),
            );
            assert!(!self.verified_proofs.read(proof_hash), "Already verified");

            self._record_verification(output.program_hash, proof_hash, 'single');
            output
        }

        // ─── Statement-bound ML conversation/action verification ─────────
        fn verify_conversation_stwo(
            ref self: ContractState,
            proof: CairoProof,
            expected_program_hash: felt252,
            expected_output_hash: felt252,
            statement_hash: felt252,
            model_id: felt252,
        ) -> VerificationOutput {
            assert!(expected_program_hash != 0, "program_hash cannot be zero");
            assert!(expected_output_hash != 0, "output_hash cannot be zero");
            assert!(statement_hash != 0, "statement_hash cannot be zero");
            assert!(model_id != 0, "model_id cannot be zero");
            assert!(expected_output_hash == statement_hash, "statement/output hash mismatch");

            let registered = self.programs.read(expected_program_hash);
            assert!(registered.program_hash == expected_program_hash, "program not registered");

            let output = get_verification_output(proof: @proof);
            assert!(output.program_hash == expected_program_hash, "program_hash mismatch");
            assert!(output.output_hash == expected_output_hash, "output_hash mismatch");

            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Security {} < 160", security);
            assert!(security >= registered.min_security_bits, "Security below program minimum");

            // FULL CRYPTOGRAPHIC STARK VERIFICATION of the Cairo verifier run.
            verify_cairo(proof);

            let proof_hash = poseidon_hash_span(
                array![output.program_hash, output.output_hash, statement_hash, model_id].span(),
            );
            assert!(!self.verified_proofs.read(proof_hash), "Already verified");
            assert!(!self.verified_statements.read(statement_hash), "Statement already verified");

            self._record_verification(output.program_hash, proof_hash, 'conversation');
            self
                ._record_statement_verification(
                    statement_hash, model_id, output.program_hash, output.output_hash, proof_hash,
                );
            output
        }

        fn verify_conversation_stwo_with_statement(
            ref self: ContractState,
            proof: CairoProof,
            expected_program_hash: felt252,
            statement_hash: felt252,
            model_id: felt252,
            statement_felts: Array<felt252>,
        ) -> VerificationOutput {
            assert!(expected_program_hash != 0, "program_hash cannot be zero");
            assert!(statement_hash != 0, "statement_hash cannot be zero");
            assert!(model_id != 0, "model_id cannot be zero");
            assert_conversation_statement(
                statement_felts.span(), expected_program_hash, statement_hash, model_id,
            );

            let registered = self.programs.read(expected_program_hash);
            assert!(registered.program_hash == expected_program_hash, "program not registered");

            let output = get_verification_output(proof: @proof);
            assert!(output.program_hash == expected_program_hash, "program_hash mismatch");
            assert!(output.output_hash == statement_hash, "output_hash mismatch");

            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Security {} < 160", security);
            assert!(security >= registered.min_security_bits, "Security below program minimum");

            // FULL CRYPTOGRAPHIC STARK VERIFICATION of the Cairo verifier run.
            verify_cairo(proof);

            let proof_hash = poseidon_hash_span(
                array![output.program_hash, output.output_hash, statement_hash, model_id].span(),
            );
            assert!(!self.verified_proofs.read(proof_hash), "Already verified");
            assert!(!self.verified_statements.read(statement_hash), "Statement already verified");

            self._record_verification(output.program_hash, proof_hash, 'conversation');
            self
                ._record_statement_verification(
                    statement_hash, model_id, output.program_hash, output.output_hash, proof_hash,
                );
            output
        }

        // ─── Streaming: open session
        // ──────────────────────────────────────
        fn stream_open(ref self: ContractState, expected_total_felts: u32) -> u64 {
            let session_id = self.next_session_id.read();
            self.next_session_id.write(session_id + 1);

            let caller = get_caller_address();
            self.session_owner.write(session_id, caller);
            self.session_total_felts.write(session_id, expected_total_felts);
            self.session_received_felts.write(session_id, 0);
            self.session_chunks_received.write(session_id, 0);
            self.session_sealed.write(session_id, false);
            self.session_data_hash.write(session_id, 0);

            self.emit(StreamOpened { session_id, expected_total_felts, owner: caller });
            session_id
        }

        // ─── Streaming: upload chunk
        // ──────────────────────────────────────
        fn stream_chunk(
            ref self: ContractState, session_id: u64, chunk_idx: u32, chunk: Array<felt252>,
        ) {
            // Validate session
            let caller = get_caller_address();
            assert!(self.session_owner.read(session_id) == caller, "Not session owner");
            assert!(!self.session_sealed.read(session_id), "Session sealed");

            let expected_chunk_idx = self.session_chunks_received.read(session_id);
            assert!(chunk_idx == expected_chunk_idx, "Chunks must be sequential");

            let chunk_len: u32 = chunk.len().try_into().unwrap();
            assert!(chunk_len <= MAX_CHUNK_SIZE, "Chunk too large");

            // Store chunk data at flat offset
            let offset = self.session_received_felts.read(session_id);
            let mut i: u32 = 0;
            let chunk_span = chunk.span();
            loop {
                if i >= chunk_len {
                    break;
                }
                self.session_data.write((session_id, offset + i), *chunk_span.at(i.into()));
                i += 1;
            }

            // Update running hash for integrity
            let prev_hash = self.session_data_hash.read(session_id);
            let mut hash_input = array![prev_hash];
            for felt in chunk_span {
                hash_input.append(*felt);
            }
            self.session_data_hash.write(session_id, poseidon_hash_span(hash_input.span()));

            // Update counters
            let new_received = offset + chunk_len;
            self.session_received_felts.write(session_id, new_received);
            self.session_chunks_received.write(session_id, expected_chunk_idx + 1);

            // Auto-seal when all felts received
            let total = self.session_total_felts.read(session_id);
            if new_received >= total {
                self.session_sealed.write(session_id, true);
            }

            self
                .emit(
                    StreamChunkReceived {
                        session_id, chunk_idx, chunk_size: chunk_len, total_received: new_received,
                    },
                );
        }

        // ─── Streaming: finalize and verify
        // ───────────────────────────────
        fn stream_verify(ref self: ContractState, session_id: u64) -> VerificationOutput {
            // Validate session is sealed
            let caller = get_caller_address();
            assert!(self.session_owner.read(session_id) == caller, "Not session owner");
            assert!(self.session_sealed.read(session_id), "Session not sealed");

            // Reassemble proof from storage
            let total_felts = self.session_total_felts.read(session_id);
            let mut proof_data: Array<felt252> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= total_felts {
                    break;
                }
                proof_data.append(self.session_data.read((session_id, i)));
                i += 1;
            }

            // Deserialize CairoProof from reassembled data
            let mut proof_span = proof_data.span();
            let proof: CairoProof = Serde::deserialize(ref proof_span).expect('PROOF_DESER');

            // Extract output BEFORE verification
            let output = get_verification_output(proof: @proof);

            // Enforce security
            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Security {} < 160", security);

            // FULL CRYPTOGRAPHIC STARK VERIFICATION
            verify_cairo(proof);

            // Record on-chain
            let proof_hash = poseidon_hash_span(
                array![output.program_hash, output.output_hash].span(),
            );
            assert!(!self.verified_proofs.read(proof_hash), "Already verified");

            self._record_verification(output.program_hash, proof_hash, 'stream');
            output
        }

        fn is_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.verified_proofs.read(proof_hash)
        }

        fn is_statement_verified(self: @ContractState, statement_hash: felt252) -> bool {
            self.verified_statements.read(statement_hash)
        }

        fn get_statement_proof_hash(self: @ContractState, statement_hash: felt252) -> felt252 {
            self.statement_proof_hash.read(statement_hash)
        }

        fn get_verification_count(self: @ContractState, program_hash: felt252) -> u64 {
            self.verification_count.read(program_hash)
        }

        // ─── Upgradability (timelocked)
        // ───────────────────────────────────
        fn propose_upgrade(ref self: ContractState, new_class_hash: starknet::ClassHash) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            assert!(new_class_hash.into() != 0_felt252, "Zero class hash");

            let now = get_block_timestamp();
            self.pending_upgrade.write(new_class_hash);
            self.upgrade_proposed_at.write(now);
            self.emit(UpgradeProposed { new_class_hash, proposed_at: now });
        }

        fn execute_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            let new_class_hash = self.pending_upgrade.read();
            assert!(new_class_hash.into() != 0_felt252, "No upgrade pending");

            let proposed_at = self.upgrade_proposed_at.read();
            let now = get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "Upgrade delay not elapsed");

            // Clear pending state
            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeExecuted { new_class_hash, executed_at: now });

            // Replace contract class (takes effect immediately)
            starknet::syscalls::replace_class_syscall(new_class_hash).unwrap();
        }

        fn cancel_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);
        }

        fn get_pending_upgrade(self: @ContractState) -> (starknet::ClassHash, u64) {
            (self.pending_upgrade.read(), self.upgrade_proposed_at.read())
        }
    }

    #[generate_trait]
    impl InternalImpl of InternalTrait {
        fn _record_verification(
            ref self: ContractState, program_hash: felt252, proof_hash: felt252, mode: felt252,
        ) {
            self.verified_proofs.write(proof_hash, true);
            let count = self.verification_count.read(program_hash);
            self.verification_count.write(program_hash, count + 1);
            let block_ts = get_block_timestamp();
            self.last_proof_hash.write(program_hash, proof_hash);
            self.last_verified_at.write(program_hash, block_ts);

            self
                .emit(
                    StwoProofVerified {
                        program_hash,
                        proof_hash,
                        verification_count: count + 1,
                        verified_at: block_ts,
                        submitter: get_caller_address(),
                        mode,
                    },
                );
        }

        fn _record_statement_verification(
            ref self: ContractState,
            statement_hash: felt252,
            model_id: felt252,
            program_hash: felt252,
            output_hash: felt252,
            proof_hash: felt252,
        ) {
            self.verified_statements.write(statement_hash, true);
            self.statement_proof_hash.write(statement_hash, proof_hash);

            self
                .emit(
                    ConversationStatementVerified {
                        statement_hash,
                        model_id,
                        program_hash,
                        output_hash,
                        proof_hash,
                        verified_at: get_block_timestamp(),
                        submitter: get_caller_address(),
                    },
                );
        }
    }

    fn assert_conversation_statement(
        statement_felts: Span<felt252>,
        expected_program_hash: felt252,
        statement_hash: felt252,
        model_id: felt252,
    ) {
        assert!(statement_felts.len() == CONVERSATION_STATEMENT_FELTS, "statement felts len");
        assert!(*statement_felts.at(0) == DOMAIN_BATCH, "statement domain mismatch");
        assert!(
            *statement_felts.at(1) == CONVERSATION_STATEMENT_VERSION, "statement version mismatch",
        );
        assert!(*statement_felts.at(2) == model_id, "statement model mismatch");
        assert!(*statement_felts.at(3) == expected_program_hash, "statement program mismatch");
        assert!(*statement_felts.at(4) != 0, "statement circuit missing");
        assert!(*statement_felts.at(5) != 0, "statement weight missing");
        assert!(*statement_felts.at(6) != 0, "statement policy missing");
        assert!(*statement_felts.at(14) != 0, "statement conversations missing");
        assert!(*statement_felts.at(15) != 0, "statement steps missing");

        let statement_security: u32 = (*statement_felts.at(18)).try_into().unwrap();
        assert!(statement_security >= MIN_SECURITY_BITS, "statement security below 160");

        let computed_hash = poseidon_hash_span(statement_felts);
        assert!(computed_hash == statement_hash, "statement hash mismatch");
    }
}
