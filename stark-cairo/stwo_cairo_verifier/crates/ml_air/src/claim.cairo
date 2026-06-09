/// Claims and proof structures for ML inference verification.
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use stwo_verifier_core::fields::qm31::{QM31, QM31Serde};
use stwo_verifier_core::pcs::PcsConfig;
use stwo_verifier_core::verifier::StarkProof;
use super::components::activation::{
    ActivationClaim, ActivationInteractionClaim, N_ACTIVATION_PREPROCESSED_COLUMNS,
    N_ACTIVATION_TRACE_COLUMNS,
};

/// The complete ML inference proof.
#[derive(Drop, Serde)]
pub struct MLProof {
    /// Model and inference metadata.
    pub claim: MLClaim,
    /// Per-matmul sumcheck proofs (Poseidon Fiat-Shamir).
    pub matmul_proofs: Array<MatMulSumcheckProofOnChain>,
    /// Optional channel salt for rerandomization.
    pub channel_salt: Option<u64>,
    /// Optional activation STARK proof data.
    pub activation_stark_proof: Option<ActivationStarkProof>,
}

/// Versioned full-model proof layout.
///
/// This mirrors the wider Rust recursive serializer. `verify_ml_v2()` verifies
/// every currently supported section and rejects unsupported non-empty sections
/// instead of silently accepting a relabelled "full model" proof.
#[derive(Drop, Serde)]
pub struct MLProofV2 {
    pub claim: MLClaim,
    pub matmul_proofs: Array<MatMulSumcheckProofOnChain>,
    pub batched_matmul_proofs: Array<BatchedMatMulProofOnChain>,
    pub channel_salt: Option<u64>,
    pub unified_stark_proof: Option<UnifiedStarkProofV2>,
    pub add_claims: Array<ElementwiseClaim>,
    pub mul_claims: Array<ElementwiseClaim>,
    pub layernorm_claims: Array<LayerNormClaim>,
    pub embedding_claims: Array<ElementwiseClaim>,
    pub attention_proofs: Array<AttentionProofOnChain>,
    pub tee_attestation_hash: Option<felt252>,
}

/// STARK proof for activation LogUp verification.
///
/// Contains all data needed to run the generic STARK verifier on
/// activation constraint quotients. When present, `verify_ml()` will
/// verify both the matmul sumchecks and the activation STARK.
#[derive(Drop, Serde)]
pub struct ActivationStarkProof {
    /// Per-layer activation claims.
    pub activation_claims: Array<ActivationClaim>,
    /// Per-layer activation interaction claims.
    pub activation_interaction_claims: Array<ActivationInteractionClaim>,
    /// Global interaction claim (LogUp sum).
    pub interaction_claim: MLInteractionClaim,
    /// PCS configuration used during proving.
    pub pcs_config: PcsConfig,
    /// Proof-of-work nonce for interaction phase.
    pub interaction_pow: u64,
    /// The STARK proof itself (commitments, FRI, decommitments).
    pub stark_proof: StarkProof,
}

/// Wider unified STARK proof layout used by the Rust full-model serializer.
///
/// The current Cairo verifier can trustlessly verify the activation subset.
/// Other component arrays must be empty until their AIR evaluators are wired
/// into the Cairo verifier.
#[derive(Drop, Serde)]
pub struct UnifiedStarkProofV2 {
    pub activation_claims: Array<ActivationClaim>,
    pub activation_interaction_claims: Array<ActivationInteractionClaim>,
    pub add_claims: Array<ElementwiseComponentClaim>,
    pub mul_claims: Array<ElementwiseComponentClaim>,
    pub layernorm_claims: Array<ElementwiseComponentClaim>,
    pub layernorm_interaction_claims: Array<ActivationInteractionClaim>,
    pub embedding_claims: Array<ElementwiseComponentClaim>,
    pub interaction_claim: MLInteractionClaim,
    pub pcs_config: PcsConfig,
    pub interaction_pow: u64,
    pub stark_proof: StarkProof,
}

/// Claim about the ML inference computation.
#[derive(Drop, Serde, Copy)]
pub struct MLClaim {
    /// Unique model identifier (Poseidon hash of architecture + weights).
    pub model_id: felt252,
    /// Number of layers in the model.
    pub num_layers: u32,
    /// Activation function type: 0=ReLU, 1=GELU, 2=Sigmoid.
    pub activation_type: u8,
    /// Poseidon commitment over (input || output) tensors.
    pub io_commitment: felt252,
    /// Poseidon Merkle root of all weight matrices.
    pub weight_commitment: felt252,
}

#[generate_trait]
pub impl MLClaimMixImpl of MLClaimMixTrait {
    fn mix_into(self: @MLClaim, ref channel: Channel) {
        channel.mix_u64((*self.num_layers).into());
        channel.mix_u64((*self.activation_type).into());
    }
}

/// ML interaction claim (logup sums for activation lookups).
#[derive(Drop, Serde, Copy)]
pub struct MLInteractionClaim {
    /// Claimed LogUp sum for all activation layers combined.
    pub activation_claimed_sum: QM31,
}

#[generate_trait]
pub impl MLInteractionClaimMixImpl of MLInteractionClaimMixTrait {
    fn mix_into(self: @MLInteractionClaim, ref channel: Channel) {
        channel.mix_felts([*self.activation_claimed_sum].span());
    }
}

/// Compute column log_sizes per commitment tree for the activation STARK.
///
/// Returns `[preprocessed_log_sizes, trace_log_sizes, interaction_trace_log_sizes]`
/// needed by `CommitmentSchemeVerifier::commit()`.
#[generate_trait]
pub impl MLClaimLogSizesImpl of MLClaimLogSizesTrait {
    fn log_sizes(activation_claims: Span<ActivationClaim>) -> Array<Array<u32>> {
        let mut preprocessed_sizes: Array<u32> = array![];
        let mut trace_sizes: Array<u32> = array![];
        let mut interaction_sizes: Array<u32> = array![];

        let mut i: u32 = 0;
        while i < activation_claims.len() {
            let log_size = *activation_claims.at(i).log_size;
            // Preprocessed: N_ACTIVATION_PREPROCESSED_COLUMNS columns (table_input, table_output)
            let mut p: u32 = 0;
            while p < N_ACTIVATION_PREPROCESSED_COLUMNS {
                preprocessed_sizes.append(log_size);
                p += 1;
            }
            // Trace: N_ACTIVATION_TRACE_COLUMNS columns (input, output, multiplicity)
            let mut t: u32 = 0;
            while t < N_ACTIVATION_TRACE_COLUMNS {
                trace_sizes.append(log_size);
                t += 1;
            }
            // Interaction: 4 LogUp cumulative sum columns (QM31 partial evals)
            interaction_sizes.append(log_size);
            interaction_sizes.append(log_size);
            interaction_sizes.append(log_size);
            interaction_sizes.append(log_size);
            i += 1;
        }

        array![preprocessed_sizes, trace_sizes, interaction_sizes]
    }
}

/// A single round polynomial from the sumcheck protocol.
///
/// Represents p(x) = c0 + c1*x + c2*x^2 where:
///   p(0) + p(1) = current_sum (verified by verifier)
///   p(r) = next_sum (r drawn from channel)
#[derive(Drop, Serde, Copy)]
pub struct RoundPoly {
    pub c0: QM31,
    pub c1: QM31,
    pub c2: QM31,
}

/// On-chain matmul sumcheck proof (Poseidon Fiat-Shamir).
#[derive(Drop, Serde)]
pub struct MatMulSumcheckProofOnChain {
    /// Matrix dimensions.
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Number of sumcheck rounds (log2(k_padded)).
    pub num_rounds: u32,
    /// Claimed sum: MLE_C(r_i, r_j).
    pub claimed_sum: QM31,
    /// Round polynomials: one per sumcheck round.
    pub round_polys: Array<RoundPoly>,
    /// Final evaluation of MLE_A at the sumcheck point.
    pub final_a_eval: QM31,
    /// Final evaluation of MLE_B at the sumcheck point.
    pub final_b_eval: QM31,
    /// Poseidon commitment to MLE_A (restricted to r_i).
    pub a_commitment: felt252,
    /// Poseidon commitment to MLE_B (restricted to r_j).
    pub b_commitment: felt252,
}

/// One entry inside a batched matmul sumcheck.
#[derive(Drop, Serde)]
pub struct BatchedMatMulEntryOnChain {
    pub node_id: u32,
    pub m: u32,
    pub n: u32,
    pub claimed_sum: QM31,
    pub final_a_eval: QM31,
    pub final_b_eval: QM31,
    pub a_commitment: felt252,
    pub b_commitment: felt252,
}

/// Combined sumcheck for many matmuls with the same `k` dimension.
#[derive(Drop, Serde)]
pub struct BatchedMatMulProofOnChain {
    pub k: u32,
    pub num_rounds: u32,
    pub lambda: QM31,
    pub combined_claimed_sum: QM31,
    pub round_polys: Array<RoundPoly>,
    pub entries: Array<BatchedMatMulEntryOnChain>,
}

/// Lightweight graph metadata for add/mul/embedding components.
#[derive(Drop, Serde, Copy)]
pub struct ElementwiseClaim {
    pub layer_index: u32,
    pub trace_rows: u32,
}

/// Lightweight graph metadata for layernorm components.
#[derive(Drop, Serde, Copy)]
pub struct LayerNormClaim {
    pub layer_index: u32,
    pub trace_rows: u32,
    pub claimed_sum: QM31,
}

/// Component claim inside the unified STARK proof.
#[derive(Drop, Serde, Copy)]
pub struct ElementwiseComponentClaim {
    pub layer_index: u32,
    pub log_size: u32,
}

/// Attention proof layout emitted by the Rust recursive serializer.
#[derive(Drop, Serde)]
pub struct AttentionProofOnChain {
    pub num_heads: u32,
    pub q_proof: MatMulSumcheckProofOnChain,
    pub k_proof: MatMulSumcheckProofOnChain,
    pub v_proof: MatMulSumcheckProofOnChain,
    pub score_proofs: Array<MatMulSumcheckProofOnChain>,
    pub attn_v_proofs: Array<MatMulSumcheckProofOnChain>,
    pub output_proof: MatMulSumcheckProofOnChain,
}

/// Output of ML verification (returned by the executable).
#[derive(Drop, Serde)]
pub struct MLVerificationOutput {
    pub model_id: felt252,
    pub io_commitment: felt252,
    pub weight_commitment: felt252,
    pub num_layers: u32,
    pub num_matmuls: u32,
    pub verified: bool,
}
