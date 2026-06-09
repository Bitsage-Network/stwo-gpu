// Recursive STARK Verifier for ObelyZK
//
// Verifies a recursive STARK proof that attests "the GKR verifier accepted."
// The proof is a standard STWO STARK — no GKR-specific logic on-chain.
//
// This replaces the 18-TX streaming GKR verification with a single TX.
//
// Public inputs (committed inside the STARK):
//   - circuit_hash: Poseidon hash of the model's circuit descriptor
//   - io_commitment: Poseidon hash of the packed inference IO
//   - weight_super_root: Poseidon Merkle root of all weight matrices
//   - conversation_statement_hash: Poseidon hash of the full
//     conversation/action statement (0 for legacy single-inference proofs)
//
// On-chain, we check these against the registered model and record verification.

// Recursive STARK verifier types

/// Public inputs for recursive verification.
#[derive(Drop, Copy, Serde)]
pub struct RecursivePublicInputs {
    /// Poseidon hash of the LayeredCircuit descriptor.
    pub circuit_hash: felt252,
    /// Poseidon hash of the packed IO felts.
    pub io_commitment: felt252,
    /// Poseidon Merkle root of all weight matrices.
    pub weight_super_root: felt252,
    /// Batch-level conversation/action statement hash.
    pub conversation_statement_hash: felt252,
}

/// Registered model info for recursive verification.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct RecursiveModelInfo {
    /// Expected circuit hash (from registration).
    pub circuit_hash: felt252,
    /// Expected weight super root (from registration).
    pub weight_super_root: felt252,
    /// Expected policy commitment (Poseidon hash of PolicyConfig). 0 = any policy.
    pub policy_commitment: felt252,
    /// Model architecture metadata (set at registration, validated at verification).
    pub n_matmuls: u32,
    pub hidden_size: u32,
    pub num_transformer_blocks: u32,
    /// Expected number of Poseidon permutations in the verifier trace.
    /// SECURITY: Prevents trace miniaturization attack. Without this, an attacker
    /// could submit a 2-row chain that satisfies all AIR constraints without
    /// running the GKR verifier. Set at registration from a reference proof.
    pub expected_n_poseidon_perms: u32,
    /// keccak256 of off-chain cairo-prove Hades Level 1 proof; 0 = not yet attached.
    /// Phase A is auditable hash only — off-chain auditors fetch the Level 1
    /// proof and verify it matches. No on-chain enforcement at verify time.
    pub level1_proof_hash: felt252,
    /// Owner who registered the model.
    pub owner: starknet::ContractAddress,
}

/// Minimal interface implemented by the statement-bound STWO verifier.
///
/// The production implementation is `GeneralStwoVerifierContract` built with
/// `general_stwo_poseidon`. Tests use a mock with the same view surface.
#[starknet::interface]
pub trait IStatementFactVerifier<TContractState> {
    fn is_statement_verified(self: @TContractState, statement_hash: felt252) -> bool;
    fn get_statement_proof_hash(self: @TContractState, statement_hash: felt252) -> felt252;
}

/// Per-session state for streaming/decoding workloads (KV-cache continuity).
///
/// Each `verify_decode_step` call advances `step_count` by 1 and rolls
/// `last_kv_commitment` forward via the proof body's KV-cache fields.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct DecodeSession {
    /// Model bound to this session (from start_decode_session).
    pub model_id: felt252,
    /// Block timestamp of session start.
    pub started_at: u64,
    /// Number of accepted steps so far. Next step MUST equal this value.
    pub step_count: u32,
    /// Rolling KV-cache commitment. 0 at start; updated to proof[32] each step.
    pub last_kv_commitment: felt252,
    /// True after finalize_decode_session — no more steps accepted.
    pub finalized: bool,
    /// Address that opened the session (sole finalizer).
    pub initiator: starknet::ContractAddress,
}

#[starknet::interface]
pub trait IRecursiveVerifier<TContractState> {
    /// Register a model for recursive verification.
    ///
    /// The circuit_hash and weight_super_root are committed at registration time.
    /// Subsequent verify calls check the proof's public inputs match these values.
    fn register_model_recursive(
        ref self: TContractState,
        model_id: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        policy_commitment: felt252,
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        expected_n_poseidon_perms: u32,
        level1_proof_hash: felt252,
    );

    /// Verify a recursive STARK proof for a registered model.
    ///
    /// Single-TX on-chain verification of a full ML inference proof.
    /// The STARK proof attests that the GKR verifier accepted the original
    /// proof covering all matmul, attention, norm, and activation layers.
    ///
    /// All parameters are visible in block explorers for full transparency.
    fn verify_recursive(
        ref self: TContractState,
        /// Unique model identifier (Poseidon hash of weight commitments).
        model_id: felt252,
        /// Poseidon hash of packed inference IO (input tokens + output logits).
        io_commitment: felt252,
        /// Model architecture fingerprint (Poseidon hash of circuit descriptor).
        circuit_hash: felt252,
        /// Poseidon Merkle root binding all weight matrices.
        weight_super_root: felt252,
        /// Number of GKR layers proven (e.g., 337 for 48-layer transformer).
        n_layers: u32,
        /// Number of matmul reductions in the GKR proof (e.g., 192 for Qwen2.5-14B).
        n_matmuls: u32,
        /// Model hidden dimension (e.g., 5120 for 14B params).
        hidden_size: u32,
        /// Number of transformer blocks (e.g., 48 for Qwen2.5-14B).
        num_transformer_blocks: u32,
        /// Proving policy commitment (Poseidon hash of PolicyConfig).
        policy_commitment: felt252,
        /// STARK execution trace log₂ size (e.g., 15 = 32768 rows).
        trace_log_size: u32,
        /// The recursive STARK proof body (FRI + Merkle decommitments).
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Production statement-bound single-pass verification.
    ///
    /// Rejects unless the proof body carries exactly
    /// `expected_conversation_statement_hash` in header slot [33]. Use this
    /// entrypoint for conversation/action proofs; `verify_recursive` remains
    /// available only for legacy zero-statement proofs.
    fn verify_recursive_with_statement(
        ref self: TContractState,
        model_id: felt252,
        io_commitment: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        n_layers: u32,
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        policy_commitment: felt252,
        trace_log_size: u32,
        expected_conversation_statement_hash: felt252,
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Fully composed statement-bound single-pass verification.
    ///
    /// Requires an already verified STWO/Cairo statement proof for the same
    /// `expected_conversation_statement_hash` before accepting the recursive
    /// STARK proof. This is the production entrypoint for STARK-in-STARK
    /// composition until both verifiers are folded into one contract.
    fn verify_recursive_with_statement_fact(
        ref self: TContractState,
        model_id: felt252,
        io_commitment: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        n_layers: u32,
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        policy_commitment: felt252,
        trace_log_size: u32,
        expected_conversation_statement_hash: felt252,
        statement_verifier: starknet::ContractAddress,
        expected_statement_proof_hash: felt252,
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Check if a recursive proof has been verified.
    fn is_recursive_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;

    /// Get the number of recursive verifications for a model.
    fn get_recursive_verification_count(self: @TContractState, model_id: felt252) -> u64;

    /// Get registered model info.
    fn get_recursive_model_info(self: @TContractState, model_id: felt252) -> RecursiveModelInfo;

    /// Get the registered policy commitment for a model. Returns 0 if no policy bound.
    fn get_model_policy(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the keccak256 of the off-chain Level 1 Hades proof attached at
    /// registration. Returns 0 if not yet attached. Phase A: auditable only.
    fn get_level1_proof_hash(self: @TContractState, model_id: felt252) -> felt252;

    /// Get full details of the last verification for a model.
    /// Returns (io_commitment, proof_hash, timestamp, proof_felts, n_layers, trace_log_size,
    /// verification_count).
    fn get_last_verification(
        self: @TContractState, model_id: felt252,
    ) -> (felt252, felt252, u64, u32, u32, u32, u64);

    /// Get the conversation/action statement hash bound to the last verification.
    /// Returns 0 for legacy single-inference proofs.
    fn get_last_conversation_statement_hash(self: @TContractState, model_id: felt252) -> felt252;

    /// Check whether a nonzero conversation/action statement hash was accepted
    /// by the recursive verifier.
    fn is_recursive_statement_verified(
        self: @TContractState, conversation_statement_hash: felt252,
    ) -> bool;

    /// Return the recursive proof hash that accepted a statement hash, or 0.
    fn get_recursive_statement_proof_hash(
        self: @TContractState, conversation_statement_hash: felt252,
    ) -> felt252;

    /// Open a new streaming decode session for a registered model.
    ///
    /// Each session tracks a rolling KV-cache commitment. Steps must arrive
    /// in order (no gaps, no reorder). Returns the new session_id.
    ///
    /// `initial_kv_commitment`: the anchor state for the session. Pass `0` for
    /// decode-from-fresh sessions, or the post-prefill KV-cache commitment when
    /// continuing an off-chain prefill. The first decode step's
    /// `prev_kv_cache_commitment` MUST equal this value.
    fn start_decode_session(
        ref self: TContractState, model_id: felt252, initial_kv_commitment: felt252,
    ) -> u64;

    /// Verify the next step of an open decode session.
    ///
    /// Performs full STARK verification (identical to `verify_recursive`) plus
    /// continuity checks: the proof body's `prev_kv_cache_commitment` (proof[31])
    /// MUST equal `session.last_kv_commitment`, and on success the session's
    /// rolling commitment is rolled forward to the body's `kv_cache_commitment`
    /// (proof[32]). `expected_step_idx` MUST equal `session.step_count`.
    fn verify_decode_step(
        ref self: TContractState,
        session_id: u64,
        expected_step_idx: u32,
        model_id: felt252,
        io_commitment: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        n_layers: u32,
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        policy_commitment: felt252,
        trace_log_size: u32,
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Production statement-bound decode-step verification.
    ///
    /// Same as `verify_decode_step`, but rejects unless header slot [33]
    /// equals `expected_conversation_statement_hash`.
    fn verify_decode_step_with_statement(
        ref self: TContractState,
        session_id: u64,
        expected_step_idx: u32,
        model_id: felt252,
        io_commitment: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        n_layers: u32,
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        policy_commitment: felt252,
        trace_log_size: u32,
        expected_conversation_statement_hash: felt252,
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Fully composed statement-bound decode-step verification.
    ///
    /// Requires the external statement verifier to have accepted the exact
    /// statement hash before the recursive decode step can advance the session.
    fn verify_decode_step_with_statement_fact(
        ref self: TContractState,
        session_id: u64,
        expected_step_idx: u32,
        model_id: felt252,
        io_commitment: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        n_layers: u32,
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        policy_commitment: felt252,
        trace_log_size: u32,
        expected_conversation_statement_hash: felt252,
        statement_verifier: starknet::ContractAddress,
        expected_statement_proof_hash: felt252,
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Finalize a decode session. Only the initiator may finalize. Requires
    /// at least one accepted step. Returns the final rolling KV commitment.
    fn finalize_decode_session(ref self: TContractState, session_id: u64) -> felt252;

    /// Read a decode session's full state.
    fn get_decode_session(self: @TContractState, session_id: u64) -> DecodeSession;

    /// Propose a contract class upgrade (owner only, subject to timelock).
    fn propose_upgrade(ref self: TContractState, new_class_hash: starknet::ClassHash);

    /// Execute a proposed upgrade after the timelock has elapsed.
    fn execute_upgrade(ref self: TContractState);

    /// Cancel a pending upgrade.
    fn cancel_upgrade(ref self: TContractState);

    /// Get the pending upgrade class hash and proposal timestamp.
    fn get_pending_upgrade(self: @TContractState) -> (starknet::ClassHash, u64);
}

#[starknet::contract]
pub mod RecursiveVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::storage::{
        Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess,
        StoragePointerWriteAccess,
    };
    use starknet::{ContractAddress, get_block_timestamp, get_caller_address};
    use stwo_constraint_framework::{CommonLookupElements, LookupElementsTrait};
    use stwo_verifier_core::channel::ChannelTrait;
    use stwo_verifier_core::circle::ChannelGetRandomCirclePointImpl;
    use stwo_verifier_core::fields::m31::{M31, m31};
    use stwo_verifier_core::fields::qm31::{QM31, QM31Trait, QM31Zero};
    use stwo_verifier_core::pcs::PcsConfigTrait;
    use stwo_verifier_core::pcs::verifier::CommitmentSchemeVerifierImpl;
    use crate::recursive_air::{LIMBS_PER_FELT, RecursiveAir};
    use super::{
        DecodeSession, IStatementFactVerifierDispatcher, IStatementFactVerifierDispatcherTrait,
        RecursiveModelInfo,
    };

    #[storage]
    struct Storage {
        /// Contract owner.
        owner: ContractAddress,
        /// Registered models for recursive verification.
        /// model_id → RecursiveModelInfo
        recursive_models: Map<felt252, RecursiveModelInfo>,
        /// Verified recursive proof hashes.
        /// proof_hash → verified (true/false)
        recursive_verified: Map<felt252, bool>,
        /// Verified nonzero conversation/action statement hashes.
        /// conversation_statement_hash → verified (true/false)
        recursive_statement_verified: Map<felt252, bool>,
        /// conversation_statement_hash → recursive proof hash.
        recursive_statement_proof_hash: Map<felt252, felt252>,
        /// conversation_statement_hash → STWO/Cairo statement proof hash.
        recursive_statement_stwo_proof_hash: Map<felt252, felt252>,
        /// Verification count per model.
        recursive_count: Map<felt252, u64>,
        /// Last proof details per model (queryable on-chain).
        /// model_id → (io_commitment, proof_hash, timestamp, proof_felts)
        last_io: Map<felt252, felt252>,
        last_proof_hash: Map<felt252, felt252>,
        last_verified_at: Map<felt252, u64>,
        last_proof_felts: Map<felt252, u32>,
        last_n_layers: Map<felt252, u32>,
        last_trace_log_size: Map<felt252, u32>,
        last_conversation_statement_hash: Map<felt252, felt252>,
        /// Pending upgrade class hash (0 = no pending upgrade).
        pending_upgrade: starknet::ClassHash,
        /// Timestamp when upgrade was proposed.
        upgrade_proposed_at: u64,
        /// Streaming decode sessions: session_id → DecodeSession.
        decode_sessions: Map<u64, DecodeSession>,
        /// Monotonically increasing session id counter.
        next_session_id: u64,
    }

    /// Minimum delay (seconds) between propose_upgrade and execute_upgrade.
    // Development: 5 minutes. Set to 86400 (24h) before public mainnet launch.
    const UPGRADE_DELAY: u64 = 300;

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        RecursiveModelRegistered: RecursiveModelRegistered,
        RecursiveProofVerified: RecursiveProofVerified,
        RecursiveStatementComposed: RecursiveStatementComposed,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
        UpgradeCancelled: UpgradeCancelled,
        DecodeSessionStarted: DecodeSessionStarted,
        DecodeStepVerified: DecodeStepVerified,
        DecodeSessionFinalized: DecodeSessionFinalized,
    }

    #[derive(Drop, starknet::Event)]
    pub struct RecursiveModelRegistered {
        #[key]
        pub model_id: felt252,
        pub circuit_hash: felt252,
        pub weight_super_root: felt252,
        pub policy_commitment: felt252,
        /// keccak256 of the off-chain Hades Level 1 proof (0 = not attached).
        pub level1_proof_hash: felt252,
        pub owner: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct RecursiveProofVerified {
        #[key]
        pub model_id: felt252,
        #[key]
        pub proof_hash: felt252,
        /// Poseidon hash of packed inference IO (input + output).
        pub io_commitment: felt252,
        /// Poseidon hash of the model's circuit descriptor (architecture fingerprint).
        pub circuit_hash: felt252,
        /// Poseidon Merkle root binding all weight matrices.
        pub weight_super_root: felt252,
        /// Poseidon hash of the policy config used during proving.
        pub policy_commitment: felt252,
        /// Poseidon hash of the full conversation/action statement.
        /// 0 for legacy single-inference proofs.
        pub conversation_statement_hash: felt252,
        /// Number of transformer layers in the model (e.g., 48 for Qwen2.5-14B).
        pub n_layers: u32,
        /// STARK trace log_size (log₂ of execution trace rows).
        pub trace_log_size: u32,
        /// Number of calldata felts in the STARK proof.
        pub proof_felts: u32,
        /// Verification sequence number for this model.
        pub verification_count: u64,
        /// Block timestamp of verification.
        pub verified_at: u64,
        /// Submitter address.
        pub submitter: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct RecursiveStatementComposed {
        #[key]
        pub conversation_statement_hash: felt252,
        #[key]
        pub recursive_proof_hash: felt252,
        pub statement_verifier: ContractAddress,
        pub statement_proof_hash: felt252,
        pub model_id: felt252,
        pub submitter: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeProposed {
        pub new_class_hash: starknet::ClassHash,
        pub proposed_at: u64,
        pub proposer: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeExecuted {
        pub new_class_hash: starknet::ClassHash,
        pub executed_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeCancelled {
        pub cancelled_class_hash: starknet::ClassHash,
        pub cancelled_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct DecodeSessionStarted {
        #[key]
        pub session_id: u64,
        #[key]
        pub model_id: felt252,
        pub initiator: ContractAddress,
        pub started_at: u64,
        pub initial_kv_commitment: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct DecodeStepVerified {
        #[key]
        pub session_id: u64,
        pub step_idx: u32,
        /// KV-cache commitment expected on entry (matches session.last_kv_commitment).
        pub prev_kv: felt252,
        /// KV-cache commitment after this step (rolled into session).
        pub new_kv: felt252,
        /// Poseidon hash of the full conversation/action statement.
        /// 0 for legacy single-inference proofs.
        pub conversation_statement_hash: felt252,
        /// Poseidon dedup hash of the step proof (over model_id, io_commitment).
        pub proof_hash: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct DecodeSessionFinalized {
        #[key]
        pub session_id: u64,
        pub n_steps: u32,
        pub final_kv: felt252,
        pub finalized_at: u64,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    #[abi(embed_v0)]
    impl RecursiveVerifierImpl of super::IRecursiveVerifier<ContractState> {
        fn register_model_recursive(
            ref self: ContractState,
            model_id: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            policy_commitment: felt252,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            expected_n_poseidon_perms: u32,
            level1_proof_hash: felt252,
        ) {
            // Only owner can register models
            let caller = get_caller_address();
            assert(caller == self.owner.read(), 'Only owner can register');

            // Prevent accidental overwrite of existing registrations
            let existing = self.recursive_models.read(model_id);
            assert(existing.circuit_hash == 0, 'Model already registered');

            // SECURITY: n_poseidon_perms must be > 0 (prevents miniaturization attack)
            assert(expected_n_poseidon_perms > 0, 'n_poseidon_perms must be > 0');

            let info = RecursiveModelInfo {
                circuit_hash,
                weight_super_root,
                policy_commitment,
                n_matmuls,
                hidden_size,
                num_transformer_blocks,
                expected_n_poseidon_perms,
                level1_proof_hash,
                owner: caller,
            };
            self.recursive_models.write(model_id, info);

            self
                .emit(
                    RecursiveModelRegistered {
                        model_id,
                        circuit_hash,
                        weight_super_root,
                        policy_commitment,
                        level1_proof_hash,
                        owner: caller,
                    },
                );
        }

        fn verify_recursive(
            ref self: ContractState,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            // Single-pass wrapper: runs the shared inner verifier, then enforces
            // the single-pass invariant (no KV-cache continuity), and finally
            // performs dedup write + verification-count update + event emit.
            let stark_proof_data_len: u32 = stark_proof_data.len();
            let (proof_hash, prev_kv, _new_kv, conversation_statement_hash) = self
                ._verify_recursive_inner(
                    model_id,
                    io_commitment,
                    circuit_hash,
                    weight_super_root,
                    n_layers,
                    n_matmuls,
                    hidden_size,
                    num_transformer_blocks,
                    policy_commitment,
                    trace_log_size,
                    0,
                    stark_proof_data,
                );

            // Single-pass mode: prover MUST set prev_kv_cache_commitment = 0.
            // Streaming continuity (nonzero prev) is only valid through
            // verify_decode_step.
            assert(prev_kv == 0, 'Single-pass requires prev_kv=0');
            self._record_statement_binding(conversation_statement_hash, proof_hash);

            // Record verification + rich on-chain state.
            self.recursive_verified.write(proof_hash, true);
            let count = self.recursive_count.read(model_id);
            self.recursive_count.write(model_id, count + 1);
            let block_ts = get_block_timestamp();
            self.last_io.write(model_id, io_commitment);
            self.last_proof_hash.write(model_id, proof_hash);
            self.last_verified_at.write(model_id, block_ts);
            self.last_proof_felts.write(model_id, stark_proof_data_len);
            self.last_n_layers.write(model_id, n_layers);
            self.last_trace_log_size.write(model_id, trace_log_size);
            self.last_conversation_statement_hash.write(model_id, conversation_statement_hash);

            // Emit rich verification event — full provenance in one TX.
            self
                .emit(
                    RecursiveProofVerified {
                        model_id,
                        proof_hash,
                        io_commitment,
                        circuit_hash,
                        weight_super_root,
                        policy_commitment,
                        conversation_statement_hash,
                        n_layers,
                        trace_log_size,
                        proof_felts: stark_proof_data_len,
                        verification_count: count + 1,
                        verified_at: block_ts,
                        submitter: get_caller_address(),
                    },
                );

            true
        }

        fn verify_recursive_with_statement(
            ref self: ContractState,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            expected_conversation_statement_hash: felt252,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            assert(expected_conversation_statement_hash != 0, 'statement hash required');
            let stark_proof_data_len: u32 = stark_proof_data.len();
            let (proof_hash, prev_kv, _new_kv, conversation_statement_hash) = self
                ._verify_recursive_inner(
                    model_id,
                    io_commitment,
                    circuit_hash,
                    weight_super_root,
                    n_layers,
                    n_matmuls,
                    hidden_size,
                    num_transformer_blocks,
                    policy_commitment,
                    trace_log_size,
                    expected_conversation_statement_hash,
                    stark_proof_data,
                );

            assert(prev_kv == 0, 'Single-pass requires prev_kv=0');
            self._record_statement_binding(conversation_statement_hash, proof_hash);

            self.recursive_verified.write(proof_hash, true);
            let count = self.recursive_count.read(model_id);
            self.recursive_count.write(model_id, count + 1);
            let block_ts = get_block_timestamp();
            self.last_io.write(model_id, io_commitment);
            self.last_proof_hash.write(model_id, proof_hash);
            self.last_verified_at.write(model_id, block_ts);
            self.last_proof_felts.write(model_id, stark_proof_data_len);
            self.last_n_layers.write(model_id, n_layers);
            self.last_trace_log_size.write(model_id, trace_log_size);
            self.last_conversation_statement_hash.write(model_id, conversation_statement_hash);

            self
                .emit(
                    RecursiveProofVerified {
                        model_id,
                        proof_hash,
                        io_commitment,
                        circuit_hash,
                        weight_super_root,
                        policy_commitment,
                        conversation_statement_hash,
                        n_layers,
                        trace_log_size,
                        proof_felts: stark_proof_data_len,
                        verification_count: count + 1,
                        verified_at: block_ts,
                        submitter: get_caller_address(),
                    },
                );

            true
        }

        fn verify_recursive_with_statement_fact(
            ref self: ContractState,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            expected_conversation_statement_hash: felt252,
            statement_verifier: ContractAddress,
            expected_statement_proof_hash: felt252,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            assert(expected_conversation_statement_hash != 0, 'statement hash required');
            let statement_proof_hash = self
                ._assert_statement_fact(
                    expected_conversation_statement_hash,
                    statement_verifier,
                    expected_statement_proof_hash,
                );

            let stark_proof_data_len: u32 = stark_proof_data.len();
            let (proof_hash, prev_kv, _new_kv, conversation_statement_hash) = self
                ._verify_recursive_inner(
                    model_id,
                    io_commitment,
                    circuit_hash,
                    weight_super_root,
                    n_layers,
                    n_matmuls,
                    hidden_size,
                    num_transformer_blocks,
                    policy_commitment,
                    trace_log_size,
                    expected_conversation_statement_hash,
                    stark_proof_data,
                );

            assert(prev_kv == 0, 'Single-pass requires prev_kv=0');
            self._record_statement_binding(conversation_statement_hash, proof_hash);
            self
                ._record_statement_composition(
                    model_id,
                    conversation_statement_hash,
                    proof_hash,
                    statement_verifier,
                    statement_proof_hash,
                );

            self.recursive_verified.write(proof_hash, true);
            let count = self.recursive_count.read(model_id);
            self.recursive_count.write(model_id, count + 1);
            let block_ts = get_block_timestamp();
            self.last_io.write(model_id, io_commitment);
            self.last_proof_hash.write(model_id, proof_hash);
            self.last_verified_at.write(model_id, block_ts);
            self.last_proof_felts.write(model_id, stark_proof_data_len);
            self.last_n_layers.write(model_id, n_layers);
            self.last_trace_log_size.write(model_id, trace_log_size);
            self.last_conversation_statement_hash.write(model_id, conversation_statement_hash);

            self
                .emit(
                    RecursiveProofVerified {
                        model_id,
                        proof_hash,
                        io_commitment,
                        circuit_hash,
                        weight_super_root,
                        policy_commitment,
                        conversation_statement_hash,
                        n_layers,
                        trace_log_size,
                        proof_felts: stark_proof_data_len,
                        verification_count: count + 1,
                        verified_at: block_ts,
                        submitter: get_caller_address(),
                    },
                );

            true
        }

        fn start_decode_session(
            ref self: ContractState, model_id: felt252, initial_kv_commitment: felt252,
        ) -> u64 {
            // Model must be registered.
            let model = self.recursive_models.read(model_id);
            assert(model.circuit_hash != 0, 'Model not registered');

            // Allocate next session id (monotonic, never reused).
            let session_id = self.next_session_id.read() + 1;
            self.next_session_id.write(session_id);

            let caller = get_caller_address();
            let now = get_block_timestamp();
            let session = DecodeSession {
                model_id,
                started_at: now,
                step_count: 0,
                last_kv_commitment: initial_kv_commitment,
                finalized: false,
                initiator: caller,
            };
            self.decode_sessions.write(session_id, session);

            self
                .emit(
                    DecodeSessionStarted {
                        session_id,
                        model_id,
                        initiator: caller,
                        started_at: now,
                        initial_kv_commitment,
                    },
                );

            session_id
        }

        fn verify_decode_step(
            ref self: ContractState,
            session_id: u64,
            expected_step_idx: u32,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            // Load session and enforce continuity preconditions BEFORE the
            // expensive STARK verification, so cheap rejections fail fast.
            let mut session = self.decode_sessions.read(session_id);
            assert(session.model_id != 0, 'Session not found');
            assert(!session.finalized, 'Session finalized');
            assert(session.model_id == model_id, 'Session model_id mismatch');
            assert(expected_step_idx == session.step_count, 'Step idx mismatch');

            // Run shared inner verifier — performs full STARK verification and
            // returns the proof's KV-cache commitments (proof[31], proof[32]).
            let (proof_hash, prev_kv, new_kv, conversation_statement_hash) = self
                ._verify_recursive_inner(
                    model_id,
                    io_commitment,
                    circuit_hash,
                    weight_super_root,
                    n_layers,
                    n_matmuls,
                    hidden_size,
                    num_transformer_blocks,
                    policy_commitment,
                    trace_log_size,
                    0,
                    stark_proof_data,
                );

            // Continuity check: the body's prev_kv MUST match the rolling
            // commitment of the session. This binds each step to its predecessor.
            assert(prev_kv == session.last_kv_commitment, 'KV cache continuity broken');
            self._record_statement_binding(conversation_statement_hash, proof_hash);
            self.recursive_verified.write(proof_hash, true);

            // Roll the session forward and persist.
            session.last_kv_commitment = new_kv;
            session.step_count = session.step_count + 1;
            self.decode_sessions.write(session_id, session);

            self
                .emit(
                    DecodeStepVerified {
                        session_id,
                        step_idx: expected_step_idx,
                        prev_kv,
                        new_kv,
                        conversation_statement_hash,
                        proof_hash,
                    },
                );

            true
        }

        fn verify_decode_step_with_statement(
            ref self: ContractState,
            session_id: u64,
            expected_step_idx: u32,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            expected_conversation_statement_hash: felt252,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            assert(expected_conversation_statement_hash != 0, 'statement hash required');
            let mut session = self.decode_sessions.read(session_id);
            assert(session.model_id != 0, 'Session not found');
            assert(!session.finalized, 'Session finalized');
            assert(session.model_id == model_id, 'Session model_id mismatch');
            assert(expected_step_idx == session.step_count, 'Step idx mismatch');

            let (proof_hash, prev_kv, new_kv, conversation_statement_hash) = self
                ._verify_recursive_inner(
                    model_id,
                    io_commitment,
                    circuit_hash,
                    weight_super_root,
                    n_layers,
                    n_matmuls,
                    hidden_size,
                    num_transformer_blocks,
                    policy_commitment,
                    trace_log_size,
                    expected_conversation_statement_hash,
                    stark_proof_data,
                );

            assert(prev_kv == session.last_kv_commitment, 'KV cache continuity broken');
            self._record_statement_binding(conversation_statement_hash, proof_hash);
            self.recursive_verified.write(proof_hash, true);

            session.last_kv_commitment = new_kv;
            session.step_count = session.step_count + 1;
            self.decode_sessions.write(session_id, session);

            self
                .emit(
                    DecodeStepVerified {
                        session_id,
                        step_idx: expected_step_idx,
                        prev_kv,
                        new_kv,
                        conversation_statement_hash,
                        proof_hash,
                    },
                );

            true
        }

        fn verify_decode_step_with_statement_fact(
            ref self: ContractState,
            session_id: u64,
            expected_step_idx: u32,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            expected_conversation_statement_hash: felt252,
            statement_verifier: ContractAddress,
            expected_statement_proof_hash: felt252,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            assert(expected_conversation_statement_hash != 0, 'statement hash required');
            let statement_proof_hash = self
                ._assert_statement_fact(
                    expected_conversation_statement_hash,
                    statement_verifier,
                    expected_statement_proof_hash,
                );

            let mut session = self.decode_sessions.read(session_id);
            assert(session.model_id != 0, 'Session not found');
            assert(!session.finalized, 'Session finalized');
            assert(session.model_id == model_id, 'Session model_id mismatch');
            assert(expected_step_idx == session.step_count, 'Step idx mismatch');

            let (proof_hash, prev_kv, new_kv, conversation_statement_hash) = self
                ._verify_recursive_inner(
                    model_id,
                    io_commitment,
                    circuit_hash,
                    weight_super_root,
                    n_layers,
                    n_matmuls,
                    hidden_size,
                    num_transformer_blocks,
                    policy_commitment,
                    trace_log_size,
                    expected_conversation_statement_hash,
                    stark_proof_data,
                );

            assert(prev_kv == session.last_kv_commitment, 'KV cache continuity broken');
            self._record_statement_binding(conversation_statement_hash, proof_hash);
            self
                ._record_statement_composition(
                    model_id,
                    conversation_statement_hash,
                    proof_hash,
                    statement_verifier,
                    statement_proof_hash,
                );
            self.recursive_verified.write(proof_hash, true);

            session.last_kv_commitment = new_kv;
            session.step_count = session.step_count + 1;
            self.decode_sessions.write(session_id, session);

            self
                .emit(
                    DecodeStepVerified {
                        session_id,
                        step_idx: expected_step_idx,
                        prev_kv,
                        new_kv,
                        conversation_statement_hash,
                        proof_hash,
                    },
                );

            true
        }

        fn finalize_decode_session(ref self: ContractState, session_id: u64) -> felt252 {
            let mut session = self.decode_sessions.read(session_id);
            assert(session.model_id != 0, 'Session not found');
            assert(!session.finalized, 'Session finalized');
            assert(session.step_count > 0, 'Session has no steps');
            assert(get_caller_address() == session.initiator, 'Only initiator finalizes');

            session.finalized = true;
            self.decode_sessions.write(session_id, session);

            let now = get_block_timestamp();
            self
                .emit(
                    DecodeSessionFinalized {
                        session_id,
                        n_steps: session.step_count,
                        final_kv: session.last_kv_commitment,
                        finalized_at: now,
                    },
                );

            session.last_kv_commitment
        }

        fn get_decode_session(self: @ContractState, session_id: u64) -> DecodeSession {
            self.decode_sessions.read(session_id)
        }

        fn is_recursive_proof_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.recursive_verified.read(proof_hash)
        }

        fn is_recursive_statement_verified(
            self: @ContractState, conversation_statement_hash: felt252,
        ) -> bool {
            self.recursive_statement_verified.read(conversation_statement_hash)
        }

        fn get_recursive_statement_proof_hash(
            self: @ContractState, conversation_statement_hash: felt252,
        ) -> felt252 {
            self.recursive_statement_proof_hash.read(conversation_statement_hash)
        }

        fn get_recursive_verification_count(self: @ContractState, model_id: felt252) -> u64 {
            self.recursive_count.read(model_id)
        }

        fn get_recursive_model_info(self: @ContractState, model_id: felt252) -> RecursiveModelInfo {
            self.recursive_models.read(model_id)
        }

        fn get_model_policy(self: @ContractState, model_id: felt252) -> felt252 {
            self.recursive_models.read(model_id).policy_commitment
        }

        fn get_level1_proof_hash(self: @ContractState, model_id: felt252) -> felt252 {
            self.recursive_models.read(model_id).level1_proof_hash
        }

        fn get_last_verification(
            self: @ContractState, model_id: felt252,
        ) -> (felt252, felt252, u64, u32, u32, u32, u64) {
            (
                self.last_io.read(model_id),
                self.last_proof_hash.read(model_id),
                self.last_verified_at.read(model_id),
                self.last_proof_felts.read(model_id),
                self.last_n_layers.read(model_id),
                self.last_trace_log_size.read(model_id),
                self.recursive_count.read(model_id),
            )
        }

        fn get_last_conversation_statement_hash(
            self: @ContractState, model_id: felt252,
        ) -> felt252 {
            self.last_conversation_statement_hash.read(model_id)
        }

        fn propose_upgrade(ref self: ContractState, new_class_hash: starknet::ClassHash) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            assert!(new_class_hash.into() != 0_felt252, "Class hash cannot be zero");

            let existing: felt252 = self.pending_upgrade.read().into();
            assert!(existing == 0, "Upgrade already pending, cancel first");

            let now = starknet::get_block_timestamp();
            self.pending_upgrade.write(new_class_hash);
            self.upgrade_proposed_at.write(now);

            self
                .emit(
                    UpgradeProposed {
                        new_class_hash, proposed_at: now, proposer: get_caller_address(),
                    },
                );
        }

        fn execute_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            let new_class_hash = self.pending_upgrade.read();
            assert!(new_class_hash.into() != 0_felt252, "No upgrade pending");

            let proposed_at = self.upgrade_proposed_at.read();
            let now = starknet::get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "Upgrade delay not elapsed");

            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeExecuted { new_class_hash, executed_at: now });

            starknet::syscalls::replace_class_syscall(new_class_hash).unwrap();
        }

        fn cancel_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            let pending: starknet::ClassHash = self.pending_upgrade.read();
            assert!(pending.into() != 0_felt252, "No upgrade pending");

            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self
                .emit(
                    UpgradeCancelled {
                        cancelled_class_hash: pending, cancelled_by: get_caller_address(),
                    },
                );
        }

        fn get_pending_upgrade(self: @ContractState) -> (starknet::ClassHash, u64) {
            (self.pending_upgrade.read(), self.upgrade_proposed_at.read())
        }
    }

    /// Private helpers — not part of the public ABI.
    #[generate_trait]
    impl RecursiveVerifierInternalImpl of RecursiveVerifierInternal {
        /// Shared inner verifier used by both `verify_recursive` (single-pass)
        /// and `verify_decode_step` (streaming).
        ///
        /// Performs ALL the existing verify_recursive logic (header parse,
        /// metadata cross-checks, channel binding, OODS/Merkle/FRI/PoW), but
        /// does NOT perform the dedup write, verification-count update,
        /// last_* writes, or event emit — the public callers do that.
        ///
        /// Returns `(proof_hash, prev_kv_cache_commitment, kv_cache_commitment,
        /// conversation_statement_hash)`. `proof_hash` is the Poseidon dedup
        /// hash; the KV commitments come from proof body slots [31] and [32],
        /// and the statement hash comes from slot [33].
        fn _verify_recursive_inner(
            ref self: ContractState,
            model_id: felt252,
            io_commitment: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            expected_conversation_statement_hash: felt252,
            stark_proof_data: Array<felt252>,
        ) -> (felt252, felt252, felt252, felt252) {
            // 1. Look up registered model
            let model = self.recursive_models.read(model_id);
            assert(model.circuit_hash != 0, 'Model not registered');

            // Verify caller-supplied metadata matches registration
            assert(circuit_hash == model.circuit_hash, 'Circuit hash mismatch (param)');
            assert(weight_super_root == model.weight_super_root, 'Weight root mismatch (param)');

            // 4. Parse header and deserialize STARK proof.
            //
            // Calldata layout (from Rust serialize_recursive_proof_calldata):
            //   [0..4)   circuit_hash: QM31 (4 felts)
            //   [4..8)   io_commitment: QM31 (4 felts)
            //   [8..12)  weight_super_root: QM31 (4 felts)
            //   [12]     n_layers: u32
            //   [13]     n_poseidon_perms: u32
            //   [14..18) seed_digest: QM31 (4 felts, channel seeding checkpoint)
            //   [18]     hades_commitment: felt252 (Level 1 Hades recursive proof binding)
            //   [19]     io_commitment_felt252: felt252 (full 252-bit hash)
            //   [20]     pass1_final_digest: felt252 (Pass 1 GKR verification digest)
            //   [21]     final_digest: felt252 (Pass 2 chain AIR boundary)
            //   [22]     log_size: u32
            //   [23]     n_real_rows: u32 (active chain rows for accumulator)
            //   [24]     n_arithmetic_rows: u32 (active primitive arithmetic rows)
            //   [25]     n_sumcheck_rows: u32 (active recorded sumcheck rows)
            //   [26]     n_draw_rows: u32 (active channel draw rows)
            //   [27..31) logup_claimed_sum: QM31 (required for interaction AIR)
            //   [31]     prev_kv_cache_commitment: felt252 (0 = single-pass)
            //   [32]     kv_cache_commitment: felt252 (0 = single-pass)
            //   [33]     conversation_statement_hash: felt252 (0 = legacy)
            //   [34..)   CommitmentSchemeProof (Serde-compatible)

            let mut proof_span = stark_proof_data.span();
            assert!(proof_span.len() >= 34, "Proof too short");

            // Parse circuit_hash: QM31 (4 M31 limbs)
            let ch0: felt252 = *proof_span.pop_front().unwrap();
            let ch1: felt252 = *proof_span.pop_front().unwrap();
            let ch2: felt252 = *proof_span.pop_front().unwrap();
            let ch3: felt252 = *proof_span.pop_front().unwrap();

            // Parse io_commitment: QM31 (4 M31 limbs)
            let io0: felt252 = *proof_span.pop_front().unwrap();
            let io1: felt252 = *proof_span.pop_front().unwrap();
            let io2: felt252 = *proof_span.pop_front().unwrap();
            let io3: felt252 = *proof_span.pop_front().unwrap();

            // Parse weight_super_root: QM31 (4 M31 limbs)
            let wr0: felt252 = *proof_span.pop_front().unwrap();
            let wr1: felt252 = *proof_span.pop_front().unwrap();
            let wr2: felt252 = *proof_span.pop_front().unwrap();
            let wr3: felt252 = *proof_span.pop_front().unwrap();

            // Parse n_layers from proof body
            let proof_n_layers: felt252 = *proof_span.pop_front().unwrap();

            // Parse n_poseidon_perms from proof body
            let proof_n_poseidon_perms: u32 = (*proof_span.pop_front().unwrap())
                .try_into()
                .unwrap();

            // Parse seed_digest: QM31 (4 M31 limbs) — channel seeding checkpoint
            let sd0: felt252 = *proof_span.pop_front().unwrap();
            let sd1: felt252 = *proof_span.pop_front().unwrap();
            let sd2: felt252 = *proof_span.pop_front().unwrap();
            let sd3: felt252 = *proof_span.pop_front().unwrap();

            // Level 1 Hades recursive proof commitment (two-level recursion)
            let hades_commitment: felt252 = *proof_span.pop_front().unwrap();

            // Full felt252 IO commitment (preserves all 252 bits)
            let proof_io_commitment_felt252: felt252 = *proof_span.pop_front().unwrap();

            // Pass 1 (full GKR verification) final digest — channel-bound.
            // This prevents a malicious prover from skipping Pass 1 (the full
            // GKR verification) and fabricating a partial witness from Pass 2 only.
            let pass1_final_digest: felt252 = *proof_span.pop_front().unwrap();

            let final_digest: felt252 = *proof_span.pop_front().unwrap();
            let proof_log_size: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();
            let proof_n_real_rows: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();
            let proof_n_arithmetic_rows: u32 = (*proof_span.pop_front().unwrap())
                .try_into()
                .unwrap();
            let proof_n_sumcheck_rows: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();
            let proof_n_draw_rows: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();

            // Recursive LogUp claimed sum: public input to the interaction AIR.
            // It is consumed by LogUp composition constraints once the recursive
            // interaction AIR is ported. Until then, chain-only proofs must
            // carry zero and interaction-tree proofs fail closed below.
            let logup_sum0: felt252 = *proof_span.pop_front().unwrap();
            let logup_sum1: felt252 = *proof_span.pop_front().unwrap();
            let logup_sum2: felt252 = *proof_span.pop_front().unwrap();
            let logup_sum3: felt252 = *proof_span.pop_front().unwrap();
            let logup_claimed_sum = QM31Trait::from_fixed_array(
                [
                    felt252_to_m31(logup_sum0), felt252_to_m31(logup_sum1),
                    felt252_to_m31(logup_sum2), felt252_to_m31(logup_sum3),
                ],
            );

            // Streaming KV-cache commitments (NEW in v4).
            // Single-pass callers MUST emit ZERO for both. Streaming callers
            // emit the previous step's KV root in [31] and the new one in [32].
            // Both are channel-bound below; no extra cross-checks needed here.
            let prev_kv_cache_commitment: felt252 = *proof_span.pop_front().unwrap();
            let kv_cache_commitment: felt252 = *proof_span.pop_front().unwrap();

            // Full conversation/action statement hash (NEW in v5).
            // This answers "what exact transcript, generated tokens, actions,
            // KV transitions, model, policy, and security level did the proof
            // attest?" Production conversation proofs MUST set this to
            // ConversationBatchStatement::statement_hash().
            let conversation_statement_hash: felt252 = *proof_span.pop_front().unwrap();
            if expected_conversation_statement_hash != 0 {
                assert(conversation_statement_hash != 0, 'statement hash missing');
                assert(
                    conversation_statement_hash == expected_conversation_statement_hash,
                    'statement hash mismatch',
                );
            }

            // 3. Compute proof hash for dedup.
            // Hash covers the explicit statement hash in addition to model and
            // IO, so the same IO cannot be relabeled across different
            // conversation/action statements. It intentionally excludes the
            // submitter address; commit-reveal is still needed to prevent
            // submission-attribution frontrunning.
            let mut hash_input = array![model_id, io_commitment, conversation_statement_hash];
            let proof_hash = poseidon_hash_span(hash_input.span());

            // Check not already verified before doing expensive STARK work.
            assert(!self.recursive_verified.read(proof_hash), 'Already verified');

            // Verify proof binds to the registered model's circuit and weights.
            // Pack 4 M31 limbs into felt252: a * 2^93 + b * 2^62 + c * 2^31 + d
            let circuit_hash_packed = ch0 * 0x80000000 * 0x80000000 * 0x80000000
                + ch1 * 0x80000000 * 0x80000000
                + ch2 * 0x80000000
                + ch3;
            let weight_root_packed = wr0 * 0x80000000 * 0x80000000 * 0x80000000
                + wr1 * 0x80000000 * 0x80000000
                + wr2 * 0x80000000
                + wr3;
            assert(circuit_hash_packed == model.circuit_hash, 'Circuit hash mismatch');
            assert(weight_root_packed == model.weight_super_root, 'Weight binding mismatch');

            // Cross-check ALL caller parameters against the proof body.
            // This prevents the relabeling attack: same proof body, different
            // caller metadata. Every value emitted in the event MUST match
            // what's cryptographically bound inside the proof.

            // io_commitment: full 252-bit felt252 from proof body, compare with caller.
            // The QM31 io_commitment only carries 124 bits (lossy conversion via
            // felt_to_securefield). The felt252 field preserves the full hash.
            assert(io_commitment == proof_io_commitment_felt252, 'io_commitment mismatch (proof)');

            // n_layers: compare caller param with proof body
            let proof_n_layers_u32: u32 = proof_n_layers.try_into().unwrap();
            assert(n_layers == proof_n_layers_u32, 'n_layers mismatch (param/proof)');

            // policy_commitment: check against registered model
            assert(
                policy_commitment == model.policy_commitment || model.policy_commitment == 0,
                'Policy mismatch',
            );

            // trace_log_size: must match proof body
            assert(trace_log_size == proof_log_size, 'trace_log_size mismatch');

            // Model architecture metadata: must match registration.
            // These are fixed per model — set once at register_model_recursive,
            // validated here so the event cannot contain false architecture claims.
            assert(n_matmuls == model.n_matmuls, 'n_matmuls mismatch');
            assert(hidden_size == model.hidden_size, 'hidden_size mismatch');
            assert(
                num_transformer_blocks == model.num_transformer_blocks,
                'num_transformer_blocks mismatch',
            );

            // SECURITY: n_poseidon_perms from proof body must match registration.
            // This prevents the trace miniaturization attack: without this check,
            // an attacker could submit a proof with n_poseidon_perms=2 (trivially
            // small chain of 2 Hades permutations) that satisfies all chain AIR
            // constraints without ever running the GKR verifier.
            assert(
                proof_n_poseidon_perms == model.expected_n_poseidon_perms,
                'n_poseidon_perms mismatch',
            );
            assert(proof_n_real_rows != 0, 'n_real_rows missing');
            let max_rows = pow2(proof_log_size);
            let proof_n_real_rows_u256: u256 = proof_n_real_rows.into();
            assert(proof_n_real_rows_u256 <= max_rows, 'n_real_rows too large');
            let proof_n_arithmetic_rows_u256: u256 = proof_n_arithmetic_rows.into();
            let proof_n_sumcheck_rows_u256: u256 = proof_n_sumcheck_rows.into();
            let proof_n_draw_rows_u256: u256 = proof_n_draw_rows.into();
            assert(proof_n_arithmetic_rows_u256 <= max_rows, 'arith rows too large');
            assert(proof_n_sumcheck_rows_u256 <= max_rows, 'sumcheck rows too large');
            assert(proof_n_draw_rows_u256 <= max_rows, 'draw rows too large');

            // Build RecursiveAir from public inputs.
            // Initial digest is always zero (fresh Poseidon channel).
            // Final digest limbs: decompose the felt252 into 9 M31 limbs (28 bits each).
            let mut initial_limbs: Array<QM31> = array![];
            let mut final_limbs: Array<QM31> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= LIMBS_PER_FELT {
                    break;
                }
                initial_limbs.append(QM31Zero::zero());
                // Each limb is 28 bits of the felt252, from LSB
                // For the boundary constraint check, we extract M31 limbs
                // via bit shifting. The STARK's boundary constraints enforce
                // that the trace's digest_after on the last row matches these values.
                let limb_val = felt252_extract_limb(final_digest, i);
                final_limbs.append(m31_to_qm31(limb_val));
                i += 1;
            }

            // 5. Deserialize + verify
            let csp: stwo_verifier_core::pcs::verifier::CommitmentSchemeProof = Serde::deserialize(
                ref proof_span,
            )
                .expect('CSP_DESER');

            let pcs_config = csp.config;
            let log_blowup = pcs_config.fri_config.log_blowup_factor;

            // Enforce minimum proof security level.
            assert(pcs_config.pow_bits >= 20, 'pow_bits too low');
            assert(log_blowup >= 5, 'log_blowup too low');
            assert(pcs_config.fri_config.n_queries >= 28, 'n_queries too low');
            let commitments_span = csp.commitments;
            let n_commitments = commitments_span.len();

            // Detect Hades mode from commitment count:
            //   3 trees = chain-only (preprocessed, trace, composition)
            //   4 trees = Hades+LogUp (preprocessed, trace+hades, interaction, composition)
            let hades_enabled = n_commitments >= 4;
            let arithmetic_enabled = proof_n_arithmetic_rows != 0;
            let sumcheck_enabled = proof_n_sumcheck_rows != 0;
            let draw_enabled = proof_n_draw_rows != 0;
            let logup_claimed_sum_nonzero = logup_sum0 != 0
                || logup_sum1 != 0
                || logup_sum2 != 0
                || logup_sum3 != 0;

            if !hades_enabled {
                assert(!logup_claimed_sum_nonzero, 'unexpected logup sum');
            }

            let preprocessed_commitment: stwo_verifier_core::Hash = *commitments_span.at(0);
            let trace_commitment: stwo_verifier_core::Hash = *commitments_span.at(1);
            // Composition is the LAST tree (index 2 for chain-only, index 3 for Hades)
            let composition_commitment: stwo_verifier_core::Hash = *commitments_span
                .at(n_commitments - 1);

            // Preprocessed sizes: always 3 columns (is_first, is_last, is_chain)
            let n_preprocess: u32 = 3;
            let mut preprocessed_sizes: Array<u32> = array![];
            i = 0;
            loop {
                if i >= n_preprocess {
                    break;
                }
                preprocessed_sizes.append(proof_log_size);
                i += 1;
            }

            // Trace sizes follow the Rust tree-1 layout:
            // chain, optional Hades, optional arithmetic, sumcheck, and draw.
            let mut n_trace: u32 = 59;
            if hades_enabled {
                n_trace += 1281;
            }
            if arithmetic_enabled {
                n_trace += 30;
            }
            if sumcheck_enabled {
                n_trace += 54;
            }
            if draw_enabled {
                n_trace += 137;
            }
            let mut trace_sizes: Array<u32> = array![];
            i = 0;
            loop {
                if i >= n_trace {
                    break;
                }
                trace_sizes.append(proof_log_size);
                i += 1;
            }

            let mut channel = Default::default();
            pcs_config.mix_into(ref channel);

            // ── Bind public inputs to Fiat-Shamir channel ────────────
            // Reconstruct QM31 values from the M31 limbs parsed from the
            // proof header, then mix into the channel in the same order
            // as the Rust prover: [circuit_hash, io_commitment,
            // weight_super_root] via mix_felts, then n_layers via mix_u64.
            //
            // This makes the STARK proof cryptographically bound to these
            // values.  Submitting different metadata causes channel
            // divergence → FRI verification failure.
            let _z = m31(0);
            let circuit_hash_qm31 = QM31Trait::from_fixed_array(
                [
                    felt252_to_m31(ch0), felt252_to_m31(ch1), felt252_to_m31(ch2),
                    felt252_to_m31(ch3),
                ],
            );
            let io_commitment_qm31 = QM31Trait::from_fixed_array(
                [
                    felt252_to_m31(io0), felt252_to_m31(io1), felt252_to_m31(io2),
                    felt252_to_m31(io3),
                ],
            );
            let weight_root_qm31 = QM31Trait::from_fixed_array(
                [
                    felt252_to_m31(wr0), felt252_to_m31(wr1), felt252_to_m31(wr2),
                    felt252_to_m31(wr3),
                ],
            );
            channel
                .mix_felts(array![circuit_hash_qm31, io_commitment_qm31, weight_root_qm31].span());
            let proof_n_layers_u64: u64 = proof_n_layers.try_into().unwrap();
            channel.mix_u64(proof_n_layers_u64);
            // SECURITY: n_poseidon_perms bound to channel — prevents miniaturization
            let proof_n_poseidon_perms_u64: u64 = proof_n_poseidon_perms.into();
            channel.mix_u64(proof_n_poseidon_perms_u64);
            // SECURITY: bind active row counts for every recursive AIR component.
            channel.mix_u64(proof_n_real_rows.into());
            channel.mix_u64(proof_n_arithmetic_rows.into());
            channel.mix_u64(proof_n_sumcheck_rows.into());
            channel.mix_u64(proof_n_draw_rows.into());

            // SECURITY: seed_digest checkpoint — binds chain to model dimensions
            let seed_digest_qm31 = QM31Trait::from_fixed_array(
                [
                    felt252_to_m31(sd0), felt252_to_m31(sd1), felt252_to_m31(sd2),
                    felt252_to_m31(sd3),
                ],
            );
            channel.mix_felts(array![seed_digest_qm31].span());

            // SECURITY: Bind Level 1 Hades recursive proof commitment.
            // This cryptographically ties the chain STARK to the set of verified
            // Hades permutations. An attacker cannot substitute different permutations
            // without changing the commitment, which invalidates the STARK proof.
            let hc_u256: u256 = hades_commitment.into();
            channel
                .mix_u64(
                    (hc_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((hc_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256)
                        & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((hc_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel.mix_u64((hc_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            // SECURITY: Bind streaming KV-cache commitments (v4).
            // prev_kv FIRST, then new_kv — order MUST match the Rust prover at
            // recursive/prover.rs (mixed immediately after hades_commitment,
            // before io_commitment_felt252). For single-pass both are zero,
            // so the eight mix_u64 calls inject 64 bytes of zeros — still
            // channel-affecting and identical on prover & verifier sides.
            let prev_kv_u256: u256 = prev_kv_cache_commitment.into();
            channel
                .mix_u64(
                    (prev_kv_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((prev_kv_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256)
                        & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((prev_kv_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel.mix_u64((prev_kv_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            let new_kv_u256: u256 = kv_cache_commitment.into();
            channel
                .mix_u64(
                    (new_kv_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((new_kv_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256)
                        & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((new_kv_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel.mix_u64((new_kv_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            // SECURITY: Bind conversation/action statement hash (v5).
            // Order MUST match Rust: prev_kv, new_kv, conversation_statement_hash,
            // then full felt252 io_commitment and pass1_final_digest.
            let stmt_u256: u256 = conversation_statement_hash.into();
            channel
                .mix_u64(
                    (stmt_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((stmt_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256)
                        & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((stmt_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel.mix_u64((stmt_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            // Bind the full felt252 io_commitment into the channel.
            // This ensures the proof body's io_commitment_felt252 field
            // cannot be tampered without invalidating the STARK.
            // Split into 4 × u64 to match the Rust prover's 4 × mix_u64 calls.
            let io_u256: u256 = proof_io_commitment_felt252.into();
            channel
                .mix_u64(
                    (io_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((io_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256)
                        & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((io_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel.mix_u64((io_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            // Bind the Pass 1 digest into Fiat-Shamir. This binds the proof to
            // the digest declared in the proof body. It is not, by itself, a
            // proof that full GKR verification executed correctly; that requires
            // complete verifier AIR or Cairo STARK-in-STARK recursion.
            // Split into 4 × u64 to match the Rust prover's 4 × mix_u64 calls.
            let p1_u256: u256 = pass1_final_digest.into();
            channel
                .mix_u64(
                    (p1_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256
                        / 0x10000000000000000_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((p1_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256)
                        & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel
                .mix_u64(
                    ((p1_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256)
                        .try_into()
                        .unwrap(),
                );
            channel.mix_u64((p1_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            let mut commitment_scheme =
                stwo_verifier_core::pcs::verifier::CommitmentSchemeVerifierImpl::new();
            commitment_scheme
                .commit(
                    preprocessed_commitment, preprocessed_sizes.span(), ref channel, log_blowup,
                );
            commitment_scheme.commit(trace_commitment, trace_sizes.span(), ref channel, log_blowup);

            // If Hades+LogUp enabled, draw relation challenges and commit the
            // interaction tree in the same order as the Rust recursive verifier:
            // HadesPerm, optional DrawFelt, optional SumcheckChallenge.
            let dummy_lookup_elements: CommonLookupElements = LookupElementsTrait::from_z_alpha(
                QM31Zero::zero(), QM31Zero::zero(),
            );
            let mut hades_lookup_elements = dummy_lookup_elements.clone();
            let mut draw_felt_lookup_elements = dummy_lookup_elements.clone();
            let mut challenge_lookup_elements = dummy_lookup_elements.clone();
            if hades_enabled {
                hades_lookup_elements = LookupElementsTrait::draw(ref channel);
                if draw_enabled {
                    draw_felt_lookup_elements = LookupElementsTrait::draw(ref channel);
                }
                if sumcheck_enabled && draw_enabled {
                    challenge_lookup_elements = LookupElementsTrait::draw(ref channel);
                }

                // Commit interaction tree (tree index 2)
                let interaction_commitment: stwo_verifier_core::Hash = *commitments_span.at(2);
                let mut interaction_sizes: Array<u32> = array![];
                i = 0;
                // One LogUp relation entry is represented by 4 trace columns
                // (QM31 partial-eval cumulative sums). Rust emits entries in:
                //   Hades provider, optional sumcheck challenge consumer,
                //   optional draw raw-felt consumer, optional draw challenge
                //   provider, optional chain draw raw-felt provider, chain Hades consumer.
                let mut interaction_col_count: u32 = 8; // Hades provider + chain consumer.
                if draw_enabled {
                    interaction_col_count += 8; // draw raw-felt consumer + chain provider.
                }
                if sumcheck_enabled && draw_enabled {
                    interaction_col_count += 8; // challenge consumer + provider.
                }
                loop {
                    if i >= interaction_col_count {
                        break;
                    }
                    interaction_sizes.append(proof_log_size);
                    i += 1;
                }
                commitment_scheme
                    .commit(
                        interaction_commitment, interaction_sizes.span(), ref channel, log_blowup,
                    );
            }

            let air = RecursiveAir {
                log_n_rows: proof_log_size,
                n_real_rows: proof_n_real_rows,
                n_draw_rows: proof_n_draw_rows,
                n_arithmetic_rows: proof_n_arithmetic_rows,
                n_sumcheck_rows: proof_n_sumcheck_rows,
                initial_digest_limbs: initial_limbs,
                final_digest_limbs: final_limbs,
                hades_enabled,
                arithmetic_enabled,
                sumcheck_enabled,
                draw_enabled,
                logup_claimed_sum,
                hades_lookup_elements,
                draw_felt_lookup_elements,
                challenge_lookup_elements,
            };

            // 7. FULL cryptographic STARK verification — ALL checks:
            // - OODS: AIR constraint evaluation matches composition polynomial
            // - Merkle: decommitment paths verify tree commitments
            // - FRI: proximity proof verifies polynomial low-degree
            // - PoW: proof of work prevents grinding
            let stark_proof = stwo_verifier_core::verifier::StarkProof {
                commitment_scheme_proof: csp,
            };
            // v1.2.2 signature: verify(proof, air, composition_log_degree_bound,
            //   composition_commitment, commitment_scheme, ref channel, min_security_bits)
            let composition_log_degree_bound = proof_log_size + 1;

            stwo_verifier_core::verifier::verify(
                stark_proof,
                air,
                composition_log_degree_bound,
                composition_commitment,
                commitment_scheme,
                ref channel,
                0,
            );

            (proof_hash, prev_kv_cache_commitment, kv_cache_commitment, conversation_statement_hash)
        }

        fn _record_statement_binding(
            ref self: ContractState, conversation_statement_hash: felt252, proof_hash: felt252,
        ) {
            if conversation_statement_hash != 0 {
                assert(
                    !self.recursive_statement_verified.read(conversation_statement_hash),
                    'Statement already verified',
                );
                self.recursive_statement_verified.write(conversation_statement_hash, true);
                self.recursive_statement_proof_hash.write(conversation_statement_hash, proof_hash);
            }
        }

        fn _assert_statement_fact(
            self: @ContractState,
            conversation_statement_hash: felt252,
            statement_verifier_address: ContractAddress,
            expected_statement_proof_hash: felt252,
        ) -> felt252 {
            assert(conversation_statement_hash != 0, 'statement hash required');
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert(statement_verifier_address != zero_addr, 'statement verifier required');

            let statement_verifier = IStatementFactVerifierDispatcher {
                contract_address: statement_verifier_address,
            };
            assert(
                statement_verifier.is_statement_verified(conversation_statement_hash),
                'statement fact missing',
            );
            let statement_proof_hash = statement_verifier
                .get_statement_proof_hash(conversation_statement_hash);
            if expected_statement_proof_hash != 0 {
                assert(
                    statement_proof_hash == expected_statement_proof_hash,
                    'statement proof mismatch',
                );
            }
            statement_proof_hash
        }

        fn _record_statement_composition(
            ref self: ContractState,
            model_id: felt252,
            conversation_statement_hash: felt252,
            recursive_proof_hash: felt252,
            statement_verifier: ContractAddress,
            statement_proof_hash: felt252,
        ) {
            self
                .recursive_statement_stwo_proof_hash
                .write(conversation_statement_hash, statement_proof_hash);
            self
                .emit(
                    RecursiveStatementComposed {
                        conversation_statement_hash,
                        recursive_proof_hash,
                        statement_verifier,
                        statement_proof_hash,
                        model_id,
                        submitter: get_caller_address(),
                    },
                );
        }
    }

    /// Convert a felt252 that holds an M31 value (0..2^31-1) to M31.
    /// Used to reconstruct QM31 from proof header limbs.
    fn felt252_to_m31(value: felt252) -> M31 {
        let v_u32: u32 = value.try_into().unwrap();
        m31(v_u32)
    }

    /// Extract the i-th 28-bit M31 limb from a felt252.
    fn felt252_extract_limb(value: felt252, limb_idx: u32) -> M31 {
        let v: u256 = value.into();
        let shift = limb_idx * 28;
        let mask: u256 = 0xFFFFFFF;
        let limb: u256 = (v / pow2(shift)) & mask;
        let limb_u32: u32 = limb.try_into().unwrap();
        stwo_verifier_core::fields::m31::m31(limb_u32 % 0x7FFFFFFF)
    }

    /// Convert an M31 to QM31 by embedding in the real component.
    fn m31_to_qm31(v: M31) -> QM31 {
        let z = stwo_verifier_core::fields::m31::m31(0);
        // QM31 = ((v, 0), (0, 0)) — v in real part, rest zero
        let arr: [M31; 4] = [v, z, z, z];
        stwo_verifier_core::fields::qm31::QM31Trait::from_fixed_array(arr)
    }

    fn pow2(n: u32) -> u256 {
        let mut r: u256 = 1;
        let mut i: u32 = 0;
        loop {
            if i >= n {
                break;
            }
            r = r * 2;
            i += 1;
        }
        r
    }
}
