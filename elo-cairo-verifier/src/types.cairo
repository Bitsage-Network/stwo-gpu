// Proof types for on-chain sumcheck verification.
//
// All types derive Serde for calldata deserialization and match
// the serialization layout from stwo-ml's cairo_serde.rs.

use crate::field::QM31;

/// A single round polynomial: p(x) = c0 + c1·x + c2·x².
/// Coefficients in monomial basis, matching STWO's UnivariatePoly<SecureField>.
#[derive(Drop, Copy, Serde)]
pub struct RoundPoly {
    pub c0: QM31,
    pub c1: QM31,
    pub c2: QM31,
}

/// Data for a single query at a single folding round of the MLE opening protocol.
#[derive(Drop, Serde)]
pub struct MleQueryRoundData {
    /// Value at the lo half (L_i[idx]).
    pub left_value: QM31,
    /// Value at the hi half (L_i[mid + idx]).
    pub right_value: QM31,
    /// Merkle path siblings for the left value (bottom-up).
    pub left_siblings: Array<felt252>,
    /// Merkle path siblings for the right value (bottom-up).
    pub right_siblings: Array<felt252>,
}

/// Complete data for a single query across all folding rounds.
#[derive(Drop, Serde)]
pub struct MleQueryProof {
    /// Initial query index in layer 0.
    pub initial_pair_index: u32,
    /// Authentication data at each folding round.
    pub rounds: Array<MleQueryRoundData>,
}

/// Opening proof for MLE(point) = claimed_eval using multilinear folding.
#[derive(Drop, Serde)]
pub struct MleOpeningProof {
    /// Merkle roots of intermediate folded layers (R_1, ..., R_{n-1}).
    pub intermediate_roots: Array<felt252>,
    /// Spot-check query proofs.
    pub queries: Array<MleQueryProof>,
    /// The final value after all folds.
    pub final_value: QM31,
}

/// Complete sumcheck proof with MLE commitment openings for on-chain verification.
///
/// Field order matches cairo_serde.rs serialize_matmul_sumcheck_proof():
/// m, k, n, num_rounds, claimed_sum, round_polys, final_a_eval, final_b_eval,
/// a_commitment, b_commitment, a_opening, b_opening
#[derive(Drop, Serde)]
pub struct MatMulSumcheckProof {
    /// Matrix dimensions: A is m×k, B is k×n, C is m×n.
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Number of sumcheck rounds (= ceil_log2(k)).
    pub num_rounds: u32,
    /// The claimed value: MLE_C evaluated at the random point.
    pub claimed_sum: QM31,
    /// One degree-2 polynomial per sumcheck round.
    pub round_polys: Array<RoundPoly>,
    /// MLE_A evaluated at (row_challenges, assignment).
    pub final_a_eval: QM31,
    /// MLE_B evaluated at (assignment, col_challenges).
    pub final_b_eval: QM31,
    /// Poseidon Merkle root of matrix A entries.
    pub a_commitment: felt252,
    /// Poseidon Merkle root of matrix B entries.
    pub b_commitment: felt252,
    /// MLE opening proof verifying final_a_eval against a_commitment.
    pub a_opening: MleOpeningProof,
    /// MLE opening proof verifying final_b_eval against b_commitment.
    pub b_opening: MleOpeningProof,
}

// ============================================================================
// Batched Matmul Sumcheck Types
// ============================================================================

/// Per-matmul entry within a batched proof.
///
/// Field order matches cairo_serde.rs serialize_batched_matmul_for_recursive()
/// per-entry layout: node_id, m, n, claimed_sum, final_a_eval, final_b_eval,
/// a_commitment, b_commitment.
#[derive(Drop, Copy, Serde)]
pub struct BatchedMatMulEntry {
    /// Graph node identifier for this matmul.
    pub node_id: u32,
    /// Row dimension of matrix A.
    pub m: u32,
    /// Column dimension of matrix B.
    pub n: u32,
    /// MLE_C evaluated at (r_i, r_j) for this matmul.
    pub claimed_sum: QM31,
    /// MLE_A evaluated at (row_challenges, assignment).
    pub final_a_eval: QM31,
    /// MLE_B evaluated at (assignment, col_challenges).
    pub final_b_eval: QM31,
    /// Poseidon Merkle root of restricted MLE_A.
    pub a_commitment: felt252,
    /// Poseidon Merkle root of restricted MLE_B.
    pub b_commitment: felt252,
}

/// Batched matmul sumcheck proof — multiple matmuls combined with lambda weighting.
///
/// Instead of N individual sumcheck proofs, a batch combines them:
///   h(x) = Σ λ^i · f_a_i(x) · f_b_i(x)
/// One set of shared round polynomials + per-matmul final evaluations.
///
/// Field order matches cairo_serde.rs serialize_batched_matmul_for_recursive():
/// k, num_rounds, lambda, combined_claimed_sum, round_polys[], entries[].
#[derive(Drop, Serde)]
pub struct BatchedMatMulProof {
    /// Padded k dimension (shared by all entries in this batch).
    pub k: u32,
    /// Number of sumcheck rounds (= log2(k)).
    pub num_rounds: u32,
    /// Lambda batching weight drawn from Fiat-Shamir.
    pub lambda: QM31,
    /// Combined claimed sum: Σ λ^i · claimed_sum_i.
    pub combined_claimed_sum: QM31,
    /// Shared round polynomials (one degree-2 polynomial per round).
    pub round_polys: Array<RoundPoly>,
    /// Per-matmul entries with individual evaluations and commitments.
    pub entries: Array<BatchedMatMulEntry>,
}

/// Verification result emitted as event data.
#[derive(Drop, Copy, Serde)]
pub struct VerificationResult {
    pub verified: bool,
    pub proof_hash: felt252,
    pub num_rounds: u32,
}
