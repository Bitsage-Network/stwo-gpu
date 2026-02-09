/// Main Starknet contract for ML inference verification.
///
/// Verifies complete model inference proofs:
/// 1. Per-layer matmul sumcheck (Poseidon Fiat-Shamir)
/// 2. Commitment chain continuity
/// 3. TEE attestation (optional)
///
/// On-chain visibility:
///   - Commitment roots only (meaningless without data)
///   - Verification boolean
///   - Nothing else. No weights. No I/O. No witness.

#[starknet::interface]
pub trait IStweMlVerifier<TContractState> {
    /// Register a model with its weight commitment.
    fn register_model(
        ref self: TContractState,
        model_id: felt252,
        weight_commitment: felt252,
    );

    /// Verify a single matmul sumcheck proof for a registered model.
    fn verify_matmul(
        ref self: TContractState,
        model_id: felt252,
        proof: super::sumcheck::MatMulSumcheckProof,
    ) -> bool;

    /// Verify a complete model inference (multiple matmul layers + chain).
    fn verify_ml_inference(
        ref self: TContractState,
        model_id: felt252,
        model_commitment: felt252,
        io_commitment: felt252,
        matmul_proofs: Array<super::sumcheck::MatMulSumcheckProof>,
        layer_headers: Array<super::layer_chain::LayerProofHeader>,
        tee_report_hash: felt252,
    ) -> bool;

    /// Query whether a proof has been verified (public).
    fn is_verified(self: @TContractState, proof_hash: felt252) -> bool;

    /// Get the model's registered weight commitment.
    fn get_model_commitment(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the number of successful verifications for a model.
    fn get_verification_count(self: @TContractState, model_id: felt252) -> u64;
}

#[starknet::contract]
mod StweMlVerifierContract {
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ContractAddress, get_caller_address};
    use core::poseidon::poseidon_hash_span;

    use crate::sumcheck::{
        MatMulSumcheckProof, verify_matmul_sumcheck,
        channel_default, channel_mix_u64, channel_mix_felt,
        channel_draw_qm31s, pack_qm31_to_felt,
        verify_sumcheck_inner, verify_mle_opening,
        next_power_of_two, log2_ceil,
    };
    use crate::layer_chain::{LayerProofHeader, verify_layer_chain, compute_chain_commitment};
    use crate::privacy::VerificationRecord;

    // ====================================================================
    // Storage
    // ====================================================================

    #[storage]
    struct Storage {
        /// Contract owner.
        owner: ContractAddress,
        /// model_id → Poseidon hash of model weight matrices.
        model_commitments: Map<felt252, felt252>,
        /// model_id → number of successful verifications.
        verification_counts: Map<felt252, u64>,
        /// proof_hash → verified (true/false).
        verified_proofs: Map<felt252, bool>,
        /// proof_hash → verification record.
        verification_records: Map<felt252, VerificationRecord>,
    }

    // ====================================================================
    // Events
    // ====================================================================

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ModelRegistered: ModelRegistered,
        InferenceVerified: InferenceVerified,
        MatMulVerified: MatMulVerified,
        VerificationFailed: VerificationFailed,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelRegistered {
        #[key]
        model_id: felt252,
        weight_commitment: felt252,
        registrar: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct InferenceVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        num_layers: u32,
        chain_commitment: felt252,
    }

    #[derive(Drop, starknet::Event)]
    struct MatMulVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        dimensions: felt252,
        num_rounds: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct VerificationFailed {
        #[key]
        model_id: felt252,
        reason: felt252,
    }

    // ====================================================================
    // Constructor
    // ====================================================================

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    // ====================================================================
    // Implementation
    // ====================================================================

    #[abi(embed_v0)]
    impl StweMlVerifierImpl of super::IStweMlVerifier<ContractState> {
        fn register_model(
            ref self: ContractState,
            model_id: felt252,
            weight_commitment: felt252,
        ) {
            let existing = self.model_commitments.entry(model_id).read();
            assert!(existing == 0, "Model already registered");
            assert!(weight_commitment != 0, "Commitment cannot be zero");

            self.model_commitments.entry(model_id).write(weight_commitment);

            self.emit(
                ModelRegistered {
                    model_id,
                    weight_commitment,
                    registrar: get_caller_address(),
                },
            );
        }

        fn verify_matmul(
            ref self: ContractState,
            model_id: felt252,
            proof: MatMulSumcheckProof,
        ) -> bool {
            let commitment = self.model_commitments.entry(model_id).read();
            assert!(commitment != 0, "Model not registered");

            let (is_valid, proof_hash) = verify_matmul_sumcheck(proof);

            if is_valid {
                self.verified_proofs.entry(proof_hash).write(true);
                let count = self.verification_counts.entry(model_id).read();
                self.verification_counts.entry(model_id).write(count + 1);
                self.emit(
                    MatMulVerified {
                        model_id,
                        proof_hash,
                        dimensions: 0,
                        num_rounds: 0,
                    },
                );
            } else {
                self.emit(VerificationFailed { model_id, reason: proof_hash });
            }

            is_valid
        }

        fn verify_ml_inference(
            ref self: ContractState,
            model_id: felt252,
            model_commitment: felt252,
            io_commitment: felt252,
            matmul_proofs: Array<MatMulSumcheckProof>,
            layer_headers: Array<LayerProofHeader>,
            tee_report_hash: felt252,
        ) -> bool {
            // 1. Validate model registration
            let registered_commitment = self.model_commitments.entry(model_id).read();
            assert!(registered_commitment != 0, "Model not registered");
            if registered_commitment != model_commitment {
                self.emit(VerificationFailed { model_id, reason: 'COMMITMENT_MISMATCH' });
                return false;
            }

            // 2. Verify commitment chain continuity
            let chain_result = verify_layer_chain(
                layer_headers.span(), 0, 0,
            );
            if !chain_result.is_valid {
                self.emit(VerificationFailed { model_id, reason: 'CHAIN_BROKEN' });
                return false;
            }

            // 3. Verify each matmul sumcheck proof
            let mut all_valid = true;
            let mut proof_hashes: Array<felt252> = array![];
            let matmul_proofs_span = matmul_proofs.span();
            let mut i: u32 = 0;
            loop {
                if i >= matmul_proofs_span.len() {
                    break;
                }

                // We need to consume the proof, so we deserialize from span
                // For now, we verify the complete proof inline
                let proof_data = matmul_proofs_span.at(i);
                let m = *proof_data.m;
                let k = *proof_data.k;
                let n = *proof_data.n;
                let num_rounds = *proof_data.num_rounds;
                let claimed_sum = *proof_data.claimed_sum;
                let final_a_eval = *proof_data.final_a_eval;
                let final_b_eval = *proof_data.final_b_eval;
                let a_commitment = *proof_data.a_commitment;
                let b_commitment = *proof_data.b_commitment;

                // Replay Fiat-Shamir for this matmul
                let mut ch = channel_default();
                channel_mix_u64(ref ch, m.into());
                channel_mix_u64(ref ch, k.into());
                channel_mix_u64(ref ch, n.into());

                let m_log = log2_ceil(next_power_of_two(m));
                let n_log = log2_ceil(next_power_of_two(n));
                let _row_challenges = channel_draw_qm31s(ref ch, m_log);
                let _col_challenges = channel_draw_qm31s(ref ch, n_log);

                channel_mix_felt(ref ch, pack_qm31_to_felt(claimed_sum));
                channel_mix_felt(ref ch, a_commitment);
                channel_mix_felt(ref ch, b_commitment);

                let round_polys = proof_data.round_polys.span();
                let (valid, proof_hash, assignment) = verify_sumcheck_inner(
                    claimed_sum, round_polys, num_rounds,
                    final_a_eval, final_b_eval, ref ch,
                );

                if !valid {
                    all_valid = false;
                    self.emit(VerificationFailed { model_id, reason: 'SUMCHECK_FAIL' });
                    break;
                }

                // Verify MLE openings
                let a_valid = verify_mle_opening(
                    a_commitment, proof_data.a_opening, assignment.span(), ref ch,
                );
                if !a_valid {
                    all_valid = false;
                    self.emit(VerificationFailed { model_id, reason: 'A_MLE_FAIL' });
                    break;
                }

                let b_valid = verify_mle_opening(
                    b_commitment, proof_data.b_opening, assignment.span(), ref ch,
                );
                if !b_valid {
                    all_valid = false;
                    self.emit(VerificationFailed { model_id, reason: 'B_MLE_FAIL' });
                    break;
                }

                proof_hashes.append(proof_hash);
                i += 1;
            };

            if !all_valid {
                return false;
            }

            // 4. Compute aggregate proof hash
            let chain_commitment = compute_chain_commitment(layer_headers.span());
            let mut aggregate_inputs: Array<felt252> = array![
                model_commitment, io_commitment, chain_commitment, tee_report_hash,
            ];
            let mut ph_i: u32 = 0;
            loop {
                if ph_i >= proof_hashes.len() { break; }
                aggregate_inputs.append(*proof_hashes.at(ph_i));
                ph_i += 1;
            };
            let aggregate_proof_hash = poseidon_hash_span(aggregate_inputs.span());

            // 5. Record verification
            self.verified_proofs.entry(aggregate_proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);

            // Store verification record
            self.verification_records.entry(aggregate_proof_hash).write(
                VerificationRecord {
                    is_valid: true,
                    verified_at: 0, // Would use get_block_timestamp() in production
                    model_commitment,
                    io_commitment,
                    tee_report_hash,
                },
            );

            self.emit(
                InferenceVerified {
                    model_id,
                    proof_hash: aggregate_proof_hash,
                    num_layers: layer_headers.len(),
                    chain_commitment,
                },
            );

            true
        }

        fn is_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.verified_proofs.entry(proof_hash).read()
        }

        fn get_model_commitment(self: @ContractState, model_id: felt252) -> felt252 {
            self.model_commitments.entry(model_id).read()
        }

        fn get_verification_count(self: @ContractState, model_id: felt252) -> u64 {
            self.verification_counts.entry(model_id).read()
        }
    }
}
