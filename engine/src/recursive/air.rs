//! AIR circuit for the recursive STARK — felt252 Hades chain.
//!
//! # Architecture
//!
//! The GKR verifier's Fiat-Shamir transcript is a chain of Hades permutations
//! over felt252 (Starknet's Poseidon). Each mix/draw operation is one Hades call
//! on a 3-element state: `[digest, value, capacity]`.
//!
//! The chain AIR constrains this transcript by:
//! 1. Storing each Hades input/output as M31 limbs (9 limbs per felt252)
//! 2. Constraining the chain: output digest of row i == input digest of row i+1
//! 3. Constraining boundaries: first row starts from zero digest, last row
//!    produces the expected final digest
//!
//! Production proofs additionally enable the merged Hades AIR and shared LogUp
//! relation. In that mode the same STARK constrains the Hades S-box, MDS,
//! round transitions, and repacked permutation input/output digests, then
//! LogUp binds every chain row to a verified Hades row. Chain-only mode remains
//! as a legacy diagnostic path and must not be used for production claims.
//!
//! # Trace Layout
//!
//! Each row = one Hades permutation call. Columns per row:
//!
//! | Columns | Count | Description |
//! |---------|-------|-------------|
//! | digest_before | 9 | felt252 → 9 M31 limbs |
//! | digest_after  | 9 | felt252 → 9 M31 limbs |
//! | shifted_next  | 9 | next row's digest_before |
//! | addition_dig  | 9 | intermediate addition |
//! | carry_pos     | 8 | positive carry indicator (boolean) |
//! | carry_neg     | 8 | negative carry / borrow indicator (boolean) |
//! | k, is_active, acc, acc_next | 4 | selectors |
//! | **Total**     | **56** | |
//!
//! Signed carry at limb j = carry_pos[j] - carry_neg[j] ∈ {-1, 0, 1}.
//! Mutually exclusive (pos*neg = 0) — both AIR-enforced via degree-2 constraints.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::BaseField as M31;
use stwo_constraint_framework::{
    preprocessed_columns::PreProcessedColumnId, EvalAtRow, FrameworkComponent, FrameworkEval,
    Relation, RelationEntry,
};

use super::hades_air::{cube_252_constraint, mds_constraint, stark_prime_9bit_limbs, LIMBS_28};

// ── LogUp relation: binds chain AIR ↔ Hades AIR ─────────────────────
// Key: (digest_before[9], digest_after[9]) = 18 M31 columns (28-bit limbs).
// This binds the chain's digest transitions to verified Hades permutations.
// Chain AIR contributes +1 multiplicity (consumer) for each active row.
// Hades AIR contributes -1 multiplicity (provider) for each permutation's
// last round, using the first-round digest and last-round MDS output digest.
//
// Digest-only binding is sufficient because the Poseidon digest uniquely
// identifies the channel state — if the Hades permutation is correct for
// a given (input_state[0], input_state[1], input_state[2]), the output
// digest (output_state[0]) is deterministic.
stwo_constraint_framework::relation!(HadesPermRelation, 18);
// Key: raw felt252 draw output as 9 28-bit limbs plus draw ordinal. Chain draw rows provide
// the Hades output felt and draw rows consume it before unpacking to QM31.
stwo_constraint_framework::relation!(DrawFeltRelation, 10);
// Key: drawn QM31 challenge limbs plus draw ordinal. This binds sumcheck rows
// to the exact recorded channel draw, not just to the challenge-value multiset.
stwo_constraint_framework::relation!(SumcheckChallengeRelation, 5);

/// Number of M31 limbs to represent one felt252.
/// 9 * 31 = 279 bits ≥ 252 bits.
pub const LIMBS_PER_FELT: usize = 9;

/// Columns for one Hades state element (3 felt252 = 3 * 9 = 27 limbs).
pub const COLS_PER_STATE: usize = 3 * LIMBS_PER_FELT; // 27

/// Columns for the shifted next-row input digest (for chain constraints).
pub const COLS_SHIFTED_DIGEST: usize = LIMBS_PER_FELT; // 9

/// Columns for one digest: 9 M31 limbs.
pub const COLS_PER_DIGEST: usize = LIMBS_PER_FELT; // 9

/// Columns for the full Hades state that are NOT the digest (value + capacity = 2 × 9 = 18).
pub const COLS_EXTRA_STATE: usize = 2 * LIMBS_PER_FELT; // 18

/// Total columns per row:
///   [0..9)    digest_before[9]
///   [9..18)   digest_after[9]
///   [18..27)  shifted_next_before[9]
///   [27..36)  addition_digest[9]
///   [36..44)  addition_carry_pos[8]
///   [44..52)  addition_carry_neg[8]
///   [52]      addition_k
///   [53]      is_active
///   [54]      active_count
///   [55]      active_count_next
///   [56]      is_draw
///   [57]      draw_count
///   [58]      shifted_next_draw_count
///
/// Each row = one Hades permutation (not one ChannelOp). Multi-Hades ChannelOps
/// (mix_poly_coeffs: 2 Hades calls) produce 2 rows. The addition_digest column
/// holds the intermediate value added to element 0 between consecutive perms
/// within the same ChannelOp.
///
/// Chain constraint: is_chain × (digest_after + addition_digest - shifted_next_before) = 0
///
/// SECURITY: Execution-trace selectors with UNCONDITIONAL amortized accumulator
/// constraint block the all-zeros-selector attack.
pub const CHAIN_COL_DIGEST_BEFORE: usize = 0;
pub const CHAIN_COL_DIGEST_AFTER: usize = CHAIN_COL_DIGEST_BEFORE + COLS_PER_DIGEST;
pub const CHAIN_COL_SHIFTED_NEXT_BEFORE: usize = CHAIN_COL_DIGEST_AFTER + COLS_PER_DIGEST;
pub const CHAIN_COL_ADDITION_DIGEST: usize = CHAIN_COL_SHIFTED_NEXT_BEFORE + COLS_PER_DIGEST;
pub const CHAIN_COL_CARRY_POS: usize = CHAIN_COL_ADDITION_DIGEST + COLS_PER_DIGEST;
pub const CHAIN_COL_CARRY_NEG: usize = CHAIN_COL_CARRY_POS + 8;
pub const CHAIN_COL_ADDITION_K: usize = CHAIN_COL_CARRY_NEG + 8;
pub const CHAIN_COL_IS_ACTIVE: usize = CHAIN_COL_ADDITION_K + 1;
pub const CHAIN_COL_ACTIVE_COUNT: usize = CHAIN_COL_IS_ACTIVE + 1;
pub const CHAIN_COL_ACTIVE_COUNT_NEXT: usize = CHAIN_COL_ACTIVE_COUNT + 1;
pub const CHAIN_COL_IS_DRAW: usize = CHAIN_COL_ACTIVE_COUNT_NEXT + 1;
pub const CHAIN_COL_DRAW_COUNT: usize = CHAIN_COL_IS_DRAW + 1;
pub const CHAIN_COL_SHIFTED_NEXT_DRAW_COUNT: usize = CHAIN_COL_DRAW_COUNT + 1;
pub const COLS_PER_ROW: usize = CHAIN_COL_SHIFTED_NEXT_DRAW_COUNT + 1;
// Total: 9 + 9 + 9 + 9 + 8 + 8 + 1 + 6 = 59.
// Effective signed carry at limb j: addition_carry_pos[j] - addition_carry_neg[j].
// Constraint enforces pos*neg = 0 so they're mutually exclusive
// → effective carry ∈ {-1, 0, 1}.

/// Columns for primitive verifier arithmetic rows recorded by the instrumented
/// GKR verifier. These rows constrain local QM31/equality facts inside the same
/// recursive STARK as the transcript/Hades rows.
///
/// Layout:
///   [0..4)    a limbs
///   [4..8)    b limbs
///   [8..12)   result limbs
///   [12..16)  mul_a limbs (zero unless is_mul)
///   [16..20)  mul_b limbs (zero unless is_mul)
///   [20..24)  mul_result limbs (zero unless is_mul)
///   [24]      is_add
///   [25]      is_mul
///   [26]      is_eq
///   [27]      is_active
///   [28]      active_count
///   [29]      active_count_next
pub const ARITH_VALUE_LIMBS: usize = 4;
pub const ARITH_COL_A: usize = 0;
pub const ARITH_COL_B: usize = ARITH_COL_A + ARITH_VALUE_LIMBS;
pub const ARITH_COL_RESULT: usize = ARITH_COL_B + ARITH_VALUE_LIMBS;
pub const ARITH_COL_MUL_A: usize = ARITH_COL_RESULT + ARITH_VALUE_LIMBS;
pub const ARITH_COL_MUL_B: usize = ARITH_COL_MUL_A + ARITH_VALUE_LIMBS;
pub const ARITH_COL_MUL_RESULT: usize = ARITH_COL_MUL_B + ARITH_VALUE_LIMBS;
pub const ARITH_COL_IS_ADD: usize = ARITH_COL_MUL_RESULT + ARITH_VALUE_LIMBS;
pub const ARITH_COL_IS_MUL: usize = ARITH_COL_IS_ADD + 1;
pub const ARITH_COL_IS_EQ: usize = ARITH_COL_IS_MUL + 1;
pub const ARITH_COL_IS_ACTIVE: usize = ARITH_COL_IS_EQ + 1;
pub const ARITH_COL_ACTIVE_COUNT: usize = ARITH_COL_IS_ACTIVE + 1;
pub const ARITH_COL_ACTIVE_COUNT_NEXT: usize = ARITH_COL_ACTIVE_COUNT + 1;
pub const ARITH_COLS_PER_ROW: usize = ARITH_COL_ACTIVE_COUNT_NEXT + 1;

/// Columns for locally checking recorded sumcheck verifier rounds.
///
/// Layout stores QM31 values as 4 M31 limbs:
///   c0, c1, c2, c3, claim, challenge, next_claim,
///   r2, r3, c1*r, c2*r2, c3*r3,
///   is_deg2, is_deg3, is_active, active_count, active_count_next.
///
/// Degree-2 rows set c3=0 and c3*r3=0. Padding rows are all-zero, so the
/// arithmetic constraints hold without selector-gated high-degree terms.
pub const SUMCHECK_VALUE_LIMBS: usize = 4;
pub const SUMCHECK_COL_C0: usize = 0;
pub const SUMCHECK_COL_C1: usize = SUMCHECK_COL_C0 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_C2: usize = SUMCHECK_COL_C1 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_C3: usize = SUMCHECK_COL_C2 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_CLAIM: usize = SUMCHECK_COL_C3 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_CHALLENGE: usize = SUMCHECK_COL_CLAIM + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_CHALLENGE_DRAW_INDEX: usize = SUMCHECK_COL_CHALLENGE + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_NEXT_CLAIM: usize = SUMCHECK_COL_CHALLENGE_DRAW_INDEX + 1;
pub const SUMCHECK_COL_R2: usize = SUMCHECK_COL_NEXT_CLAIM + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_R3: usize = SUMCHECK_COL_R2 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_C1R: usize = SUMCHECK_COL_R3 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_C2R2: usize = SUMCHECK_COL_C1R + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_C3R3: usize = SUMCHECK_COL_C2R2 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_IS_DEG2: usize = SUMCHECK_COL_C3R3 + SUMCHECK_VALUE_LIMBS;
pub const SUMCHECK_COL_IS_DEG3: usize = SUMCHECK_COL_IS_DEG2 + 1;
pub const SUMCHECK_COL_IS_ACTIVE: usize = SUMCHECK_COL_IS_DEG3 + 1;
pub const SUMCHECK_COL_ACTIVE_COUNT: usize = SUMCHECK_COL_IS_ACTIVE + 1;
pub const SUMCHECK_COL_ACTIVE_COUNT_NEXT: usize = SUMCHECK_COL_ACTIVE_COUNT + 1;
pub const SUMCHECK_COLS_PER_ROW: usize = SUMCHECK_COL_ACTIVE_COUNT_NEXT + 1;

/// Columns for recorded channel draws consumed by sumcheck verifier rounds.
pub const DRAW_VALUE_LIMBS: usize = 4;
pub const DRAW_COL_VALUE: usize = 0;
pub const DRAW_COL_DRAW_INDEX: usize = DRAW_COL_VALUE + DRAW_VALUE_LIMBS;
pub const DRAW_COL_RAW_FELT: usize = DRAW_COL_DRAW_INDEX + 1;
pub const DRAW_RAW_FELT_LIMBS: usize = LIMBS_PER_FELT;
pub const DRAW_COL_SPLIT_LO: usize = DRAW_COL_RAW_FELT + DRAW_RAW_FELT_LIMBS;
pub const DRAW_SPLIT_COUNT: usize = 4;
pub const DRAW_COL_SPLIT_HI: usize = DRAW_COL_SPLIT_LO + DRAW_SPLIT_COUNT;
pub const DRAW_COL_SPLIT_BITS: usize = DRAW_COL_SPLIT_HI + DRAW_SPLIT_COUNT;
pub const DRAW_SPLIT_BITS_TOTAL: usize = 112;
pub const DRAW_COL_IS_ACTIVE: usize = DRAW_COL_SPLIT_BITS + DRAW_SPLIT_BITS_TOTAL;
pub const DRAW_COL_ACTIVE_COUNT: usize = DRAW_COL_IS_ACTIVE + 1;
pub const DRAW_COL_ACTIVE_COUNT_NEXT: usize = DRAW_COL_ACTIVE_COUNT + 1;
pub const DRAW_COLS_PER_ROW: usize = DRAW_COL_ACTIVE_COUNT_NEXT + 1;

// ═══════════════════════════════════════════════════════════════════════
// Felt252 ↔ M31 limb decomposition
// ═══════════════════════════════════════════════════════════════════════

/// Stark prime P decomposed into 9 × 28-bit limbs (LSB first).
/// P = 2^251 + 17 * 2^192 + 1
pub const P_LIMBS_28: [u32; 9] = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728];

/// Compute carry-chain witnesses for modular limb addition:
///   a[j] + b[j] + (pos[j-1] - neg[j-1]) = result[j] + k*P[j] + (pos[j] - neg[j])*2^28
///
/// Returns (carry_pos[8], carry_neg[8], k) where:
/// - carry_pos[j] ∈ {0,1}: positive carry indicator for limb j
/// - carry_neg[j] ∈ {0,1}: negative carry (borrow) indicator for limb j
/// - signed carry at position j = pos[j] - neg[j] ∈ {-1, 0, 1}
/// - pos[j] * neg[j] = 0 enforced by the AIR (mutual exclusion)
/// - k ∈ {0,1}: modular reduction quotient
///
/// `a_limbs` = digest_after, `b_limbs` = addition_digest, `result_limbs` = shifted_next_before.
///
/// Negative carries (borrows) are needed when P's high-bit limbs (P_LIMBS_28[6..9]
/// non-zero due to 2^251 + 17*2^192 + 1) cause the running integer sum's limb to
/// fall below the result's limb at some position. This happens for Llama-class
/// decode-mode channel ops.
pub fn compute_addition_carry_chain(
    a_limbs: &[M31; LIMBS_PER_FELT],
    b_limbs: &[M31; LIMBS_PER_FELT],
    result_limbs: &[M31; LIMBS_PER_FELT],
) -> ([M31; 8], [M31; 8], M31) {
    for k in 0..=1u32 {
        let mut carries_signed = [0i64; 8];
        let mut carry: i64 = 0;
        let mut valid = true;

        for j in 0..LIMBS_PER_FELT {
            let lhs = a_limbs[j].0 as i64 + b_limbs[j].0 as i64 + carry;
            let rhs_base = result_limbs[j].0 as i64 + (k as i64) * (P_LIMBS_28[j] as i64);
            let diff = lhs - rhs_base;
            // diff = carry_out * 2^28; must be exact multiple
            if diff.rem_euclid(1i64 << 28) != 0 {
                valid = false;
                break;
            }
            carry = diff / (1i64 << 28);
            if j < 8 {
                if !(-1..=1).contains(&carry) {
                    valid = false;
                    break;
                }
                carries_signed[j] = carry;
            }
        }
        if valid && carry == 0 {
            // Split signed carries into pos/neg indicator columns:
            //   c = +1 → pos=1, neg=0
            //   c =  0 → pos=0, neg=0
            //   c = -1 → pos=0, neg=1
            let pos = std::array::from_fn(|j| {
                M31::from_u32_unchecked(if carries_signed[j] == 1 { 1 } else { 0 })
            });
            let neg = std::array::from_fn(|j| {
                M31::from_u32_unchecked(if carries_signed[j] == -1 { 1 } else { 0 })
            });
            return (pos, neg, M31::from_u32_unchecked(k));
        }
    }
    // Fallback: should never happen if a + b ≡ result (mod P).
    eprintln!(
        "[carry-chain FALLBACK] a_limbs={:?} b_limbs={:?} result_limbs={:?}",
        a_limbs.map(|m| m.0),
        b_limbs.map(|m| m.0),
        result_limbs.map(|m| m.0),
    );
    (
        [M31::from_u32_unchecked(0); 8],
        [M31::from_u32_unchecked(0); 8],
        M31::from_u32_unchecked(0),
    )
}

/// Decompose a felt252 into 9 M31 limbs (LSB first).
///
/// Each limb holds 28 bits (not 31) to leave room for carry propagation
/// in future constraint additions. 9 * 28 = 252 bits = exact fit.
pub fn felt252_to_limbs(felt: &FieldElement) -> [M31; LIMBS_PER_FELT] {
    let bytes = felt.to_bytes_be();
    let mut limbs = [M31::from_u32_unchecked(0); LIMBS_PER_FELT];

    // Convert 32 bytes (256 bits) to 9 limbs of 28 bits each.
    // Total capacity: 9 * 28 = 252 bits (perfect for felt252).
    // We use 28-bit limbs for future carry-chain constraints.
    let mut bits_remaining = 252u32;
    let mut byte_idx = 31usize; // start from LSB
    let mut bit_offset = 0u32;

    for limb in limbs.iter_mut() {
        let limb_bits = 28u32.min(bits_remaining);
        let mut value = 0u32;

        for b in 0..limb_bits {
            let global_bit = bit_offset + b;
            let bi = (global_bit / 8) as usize;
            let bpos = (global_bit % 8) as u32;
            if bi <= 31 {
                let byte_val = bytes[31 - bi];
                if (byte_val >> bpos) & 1 == 1 {
                    value |= 1u32 << b;
                }
            }
        }

        *limb = M31::from_u32_unchecked(value);
        bit_offset += limb_bits;
        bits_remaining = bits_remaining.saturating_sub(limb_bits);
    }

    limbs
}

/// Reconstruct a felt252 from 9 28-bit M31 limbs (LSB first).
/// Inverse of `felt252_to_limbs` for diagnostic use.
pub fn limbs_to_felt252(limbs: &[M31; LIMBS_PER_FELT]) -> FieldElement {
    let mut result = FieldElement::ZERO;
    let mut shift = FieldElement::ONE;
    let two_pow_28 = FieldElement::from(1u64 << 28);
    for limb in limbs {
        result = result + FieldElement::from(limb.0 as u64) * shift;
        shift = shift * two_pow_28;
    }
    result
}

/// Decompose a 3-element Hades state into 27 M31 limbs.
pub fn hades_state_to_limbs(state: &[FieldElement; 3]) -> [M31; COLS_PER_STATE] {
    let mut limbs = [M31::from_u32_unchecked(0); COLS_PER_STATE];
    for (i, felt) in state.iter().enumerate() {
        let felt_limbs = felt252_to_limbs(felt);
        limbs[i * LIMBS_PER_FELT..(i + 1) * LIMBS_PER_FELT].copy_from_slice(&felt_limbs);
    }
    limbs
}

// ═══════════════════════════════════════════════════════════════════════
// FrameworkEval: Hades Chain AIR
// ═══════════════════════════════════════════════════════════════════════

/// AIR evaluator for the recursive STARK's Hades chain.
///
/// Each row constrains one Hades permutation from the verifier's transcript.
/// Chain constraints link consecutive rows via the digest limbs.
#[derive(Debug, Clone)]
pub struct RecursiveVerifierEval {
    /// log2 of the number of trace rows.
    pub log_n_rows: u32,

    /// Number of real (non-padding) rows in the trace.
    /// SECURITY: This determines where is_first, is_last, is_chain are placed.
    /// Both prover and verifier compute the preprocessed columns from this value.
    /// Without this, a malicious prover could place is_last at row 1 (miniaturized
    /// chain) and the verifier couldn't detect it.
    pub n_real_rows: u32,

    /// Initial digest (usually zero, decomposed into limbs).
    pub initial_digest_limbs: [M31; LIMBS_PER_FELT],

    /// Expected final digest after all verifier operations (decomposed into limbs).
    pub final_digest_limbs: [M31; LIMBS_PER_FELT],

    /// LogUp lookup elements for the Hades permutation relation.
    /// When `None`, LogUp is disabled (backward-compatible mode).
    pub hades_lookup: Option<HadesPermRelation>,

    /// LogUp lookup elements binding draw trace raw felts to Hades draw outputs.
    pub draw_felt_lookup: Option<DrawFeltRelation>,

    /// LogUp lookup elements binding sumcheck challenges to channel draws.
    pub challenge_lookup: Option<SumcheckChallengeRelation>,

    /// When true, the Hades AIR columns (1281 columns) follow the chain
    /// columns in the committed trace and their constraints are evaluated
    /// inline. This merges chain + Hades into a single component.
    pub hades_enabled: bool,

    /// When true, primitive verifier arithmetic columns follow the chain and
    /// optional Hades columns in the committed trace.
    pub arithmetic_enabled: bool,

    /// Number of real primitive arithmetic rows.
    pub n_arithmetic_rows: u32,

    /// When true, recorded sumcheck verifier round columns follow the chain,
    /// optional Hades columns, and optional primitive arithmetic columns.
    pub sumcheck_enabled: bool,

    /// Number of real recorded sumcheck rows.
    pub n_sumcheck_rows: u32,

    /// When true, recorded channel draw columns follow the sumcheck columns.
    pub draw_enabled: bool,

    /// Number of real recorded channel draw rows.
    pub n_draw_rows: u32,
}

impl FrameworkEval for RecursiveVerifierEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        if self.hades_enabled {
            // The merged Hades constraints are quadratic at most. The bound
            // must stay tight; over-declaring it changes STWO's quotient
            // composition path and can fail the proof sanity check.
            self.log_n_rows + 1
        } else if self.arithmetic_enabled || self.sumcheck_enabled || self.draw_enabled {
            self.log_n_rows + 1
        } else {
            // Preserve the existing chain-only proof profile.
            self.log_n_rows + 1
        }
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // ── Preprocessed selectors ─────────────────────────────────────
        // is_first IS used for the initial boundary (row 0). It represents
        // the first circle domain point and is deterministic. The amortized
        // accumulator (C3) prevents the all-zeros attack even if is_first
        // is tampered, because the correction term is unconditionally non-zero.
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_first".into(),
        });
        let _is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_last".into(),
        });
        let _is_chain = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_chain".into(),
        });

        // ── Read execution trace columns (chain layout with pos/neg carries) ─────
        // digest_before[9]
        let digest_before: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());

        // digest_after[9]
        let digest_after: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());

        // shifted_next_before[9]: digest_before of the NEXT row
        let shifted_next_before: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // addition_digest[9]: intermediate value added to digest between consecutive
        // Hades permutations within the same ChannelOp (e.g., felt2 in mix_poly_coeffs).
        // Zero for single-perm ops.
        let addition_digest: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // Carry chain for modular limb addition:
        //   digest_after[j] + addition[j] + signed_carry[j-1]
        //     = result[j] + k*P[j] + signed_carry[j]*2^28
        // where signed_carry[j] = addition_carry_pos[j] - addition_carry_neg[j],
        // result[j] = shifted_next_before[j], k ∈ {0,1}, pos/neg ∈ {0,1} mutually exclusive
        // (so signed_carry ∈ {-1, 0, 1}).
        // Negative carries (borrows) are needed when P's high-bit limbs cause the
        // limb-level integer sum to fall below the result's limb at some position
        // (e.g., decode-mode channel ops on Llama-class models).
        let addition_carry_pos: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
        let addition_carry_neg: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
        let addition_k = eval.next_trace_mask();

        // ── Execution-trace selectors ────────────────────────────────
        let is_active = eval.next_trace_mask();
        let active_count = eval.next_trace_mask();
        let active_count_next = eval.next_trace_mask();
        let is_draw = eval.next_trace_mask();
        let draw_count = eval.next_trace_mask();
        let shifted_next_draw_count = eval.next_trace_mask();

        // ══════════════════════════════════════════════════════════════
        // UNCONDITIONAL CONSTRAINTS (no selector gating)
        // ══════════════════════════════════════════════════════════════

        // C1: is_active is boolean [degree 2, unconditional]
        eval.add_constraint(is_active.clone() * (E::F::from(M31::from(1u32)) - is_active.clone()));
        eval.add_constraint(is_draw.clone() * (E::F::from(M31::from(1u32)) - is_draw.clone()));
        eval.add_constraint(is_draw.clone() * (E::F::from(M31::from(1u32)) - is_active.clone()));

        // C2: amortized accumulator [degree 1, unconditional]
        // SECURITY: CRITICAL — prevents all-zeros-selector attack.
        // For an all-zeros trace: 0 - 0 - 0 + correction = correction ≠ 0.
        let n = 1u32 << self.log_n_rows;
        let n_inv = M31::from(n).inverse();
        let correction = E::F::from(M31::from(self.n_real_rows)) * E::F::from(n_inv);
        eval.add_constraint(
            active_count_next.clone() - active_count.clone() - is_active.clone() + correction,
        );

        // ══════════════════════════════════════════════════════════════
        // PREPROCESSED-GATED CONSTRAINTS (existing, proven working)
        // These use preprocessed is_first/is_last/is_chain selectors.
        // Combined with C2 (accumulator), these provide full security:
        // - C2 forces exactly n_real_rows active rows (prevents miniaturization)
        // - is_first/is_last/is_chain enforce chain integrity
        // ══════════════════════════════════════════════════════════════

        // C3: Initial boundary — row 0's digest_before = initial [degree 2]
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_first.clone()
                    * (digest_before[j].clone() - E::F::from(self.initial_digest_limbs[j])),
            );
        }
        eval.add_constraint(is_first.clone() * draw_count.clone());

        // C4: Final boundary — last active row's digest_after = final [degree 2]
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                _is_last.clone()
                    * (digest_after[j].clone() - E::F::from(self.final_digest_limbs[j])),
            );
        }
        eval.add_constraint(
            _is_last.clone()
                * (draw_count.clone() + is_draw.clone() - E::F::from(M31::from(self.n_draw_rows))),
        );

        // C5: Chain — carry-chain modular addition [degree 2]
        // digest_after[j] + addition[j] + carry[j-1] - result[j] - k*P[j] - carry[j]*2^28 = 0
        // where result[j] = shifted_next_before[j]
        //
        // Stark prime P in 28-bit limbs (LSB first):
        // P = 2^251 + 17*2^192 + 1
        // P_limbs = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728]
        {
            let p_limbs_28: [u32; 9] = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728];
            let two_pow_28 = E::F::from(M31::from(1u32 << 28));

            // k must be boolean
            eval.add_constraint(
                _is_chain.clone()
                    * addition_k.clone()
                    * (addition_k.clone() - E::F::from(M31::from(1u32))),
            );
            // pos/neg carries must each be boolean (∈ {0, 1})
            // and mutually exclusive (pos*neg = 0). Net signed carry = pos - neg ∈ {-1, 0, 1}.
            // All three constraints stay degree 2.
            let one = E::F::from(M31::from(1u32));
            for j in 0..8 {
                eval.add_constraint(
                    _is_chain.clone()
                        * addition_carry_pos[j].clone()
                        * (addition_carry_pos[j].clone() - one.clone()),
                );
                eval.add_constraint(
                    _is_chain.clone()
                        * addition_carry_neg[j].clone()
                        * (addition_carry_neg[j].clone() - one.clone()),
                );
                eval.add_constraint(
                    _is_chain.clone()
                        * addition_carry_pos[j].clone()
                        * addition_carry_neg[j].clone(),
                );
            }

            // Per-limb carry-chain constraint
            for j in 0..LIMBS_PER_FELT {
                let carry_in = if j == 0 {
                    E::F::from(M31::from(0u32))
                } else {
                    addition_carry_pos[j - 1].clone() - addition_carry_neg[j - 1].clone()
                };
                let carry_out_term = if j < 8 {
                    (addition_carry_pos[j].clone() - addition_carry_neg[j].clone())
                        * two_pow_28.clone()
                } else {
                    // Last limb: carry out must be 0 (no overflow past 252 bits)
                    E::F::from(M31::from(0u32))
                };
                let p_j = E::F::from(M31::from(p_limbs_28[j]));

                // da[j] + add[j] + carry_in - snb[j] - k*P[j] - carry_out*2^28 = 0
                eval.add_constraint(
                    _is_chain.clone()
                        * (digest_after[j].clone() + addition_digest[j].clone() + carry_in
                            - shifted_next_before[j].clone()
                            - addition_k.clone() * p_j
                            - carry_out_term),
                );
            }
        }

        eval.add_constraint(
            _is_chain.clone()
                * (shifted_next_draw_count.clone() - draw_count.clone() - is_draw.clone()),
        );

        // ══════════════════════════════════════════════════════════════
        // HADES AIR (merged inline when enabled)
        // ══════════════════════════════════════════════════════════════
        //
        // When hades_enabled is true, we read the 1281 Hades trace columns
        // that follow the chain columns in the committed trace and
        // evaluate the core Hades constraints (boolean selectors, S-box/cube,
        // post-sbox interpolation, MDS, round transition).
        //
        // The Hades trace has its OWN is_real selector (independent of the
        // chain's is_active) because the two have different numbers of real
        // rows: chain has ~13 rows, Hades has ~1183 (13 perms * 91 rounds).

        if self.hades_enabled {
            let h_zero_f = || E::F::from(M31::from(0u32));

            // ── Hades column reads (1281 columns) ────────────────────
            // state_before: 3 x 28 limbs
            let mut h_state_before: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_state_before[elem][j] = eval.next_trace_mask();
                }
            }

            // sbox_input: 3 x 28 limbs
            let mut h_sbox_input: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_sbox_input[elem][j] = eval.next_trace_mask();
                }
            }

            // cube_result: 3 x 28 limbs
            let mut h_cube_result: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_cube_result[elem][j] = eval.next_trace_mask();
                }
            }

            // cube_sq: 3 x 28 limbs (x^2 intermediate)
            let mut h_cube_sq: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_cube_sq[elem][j] = eval.next_trace_mask();
                }
            }

            // Multiplication witness: 54 carries + 28 k_limbs per mul (6 total)
            let mut h_mul_carries: [[[E::F; 54]; 2]; 3] = std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()))
            });
            let mut h_mul_k_limbs: [[[[E::F; LIMBS_28]; 1]; 2]; 3] = std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()))
                })
            });
            for elem in 0..3 {
                for mul_idx in 0..2 {
                    for j in 0..54 {
                        h_mul_carries[elem][mul_idx][j] = eval.next_trace_mask();
                    }
                    for j in 0..LIMBS_28 {
                        h_mul_k_limbs[elem][mul_idx][0][j] = eval.next_trace_mask();
                    }
                }
            }

            // post_sbox: 3 x 28 limbs (MDS input)
            let mut h_post_sbox: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_post_sbox[elem][j] = eval.next_trace_mask();
                }
            }

            // mds_result: 3 x 28 limbs
            let mut h_mds_result: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_mds_result[elem][j] = eval.next_trace_mask();
                }
            }

            // MDS carries (29) + k (1) per element
            let mut h_mds_carries: [[E::F; 29]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            let mut h_mds_k: [E::F; 3] = std::array::from_fn(|_| h_zero_f());
            for elem in 0..3 {
                for j in 0..29 {
                    h_mds_carries[elem][j] = eval.next_trace_mask();
                }
                h_mds_k[elem] = eval.next_trace_mask();
            }

            // shifted_next_state: state_before of the NEXT row (round transition)
            let mut h_shifted_next_state: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_shifted_next_state[elem][j] = eval.next_trace_mask();
                }
            }

            // Selectors
            let h_is_full_round = eval.next_trace_mask();
            let h_is_real = eval.next_trace_mask();
            let h_is_chain_round = eval.next_trace_mask();

            // Boundary selectors for Hades blocks.
            let h_is_first_round = eval.next_trace_mask();
            let h_is_last_round = eval.next_trace_mask();

            // Repack columns used by LogUp provider.
            let h_input_digest_9bit: [E::F; LIMBS_28] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let h_shifted_next_input_digest_9bit: [E::F; LIMBS_28] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let h_input_digest_28bit: [E::F; 9] = std::array::from_fn(|_| eval.next_trace_mask());
            let h_output_digest_28bit: [E::F; 9] = std::array::from_fn(|_| eval.next_trace_mask());
            let h_split_lo_in: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
            let h_split_hi_in: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
            let h_split_lo_out: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
            let h_split_hi_out: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());

            // ── Hades constraints ────────────────────────────────────

            let h_p_limbs = stark_prime_9bit_limbs();
            let h_one = E::F::from(M31::from(1u32));
            #[cfg(test)]
            let hades_debug_level = std::env::var("OBELYZK_HADES_DEBUG_LEVEL")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(u32::MAX);
            #[cfg(not(test))]
            let hades_debug_level = u32::MAX;

            // Boolean selector constraints
            if hades_debug_level >= 1 {
                eval.add_constraint(h_is_real.clone() * (h_is_real.clone() - h_one.clone()));
                eval.add_constraint(
                    h_is_full_round.clone() * (h_is_full_round.clone() - h_one.clone()),
                );
            }

            // S-box constraints: cube_result = sbox_input^3
            // No selector gating — padding rows are all-zero so constraints
            // hold trivially (0^3 = 0).
            let h_rc_ref: Option<&super::hades_air::RangeCheck20> = None;
            if hades_debug_level >= 2 {
                for elem in 0..3 {
                    cube_252_constraint::<E>(
                        &h_sbox_input[elem],
                        &h_cube_sq[elem],
                        &h_cube_result[elem],
                        &h_mul_k_limbs[elem][0][0],
                        &h_mul_carries[elem][0],
                        &h_mul_k_limbs[elem][1][0],
                        &h_mul_carries[elem][1],
                        &mut eval,
                        &h_p_limbs,
                        &h_one,
                        h_rc_ref,
                    );
                }
            }

            // Post-sbox linking:
            // Element 2: always cubed -> post_sbox[2] = cube_result[2]
            if hades_debug_level >= 3 {
                for j in 0..LIMBS_28 {
                    eval.add_constraint(h_post_sbox[2][j].clone() - h_cube_result[2][j].clone());
                }
                // Elements 0,1: interpolate between cube and passthrough
                // post_sbox[e] = is_full_round * cube_result[e] + (1 - is_full_round) * sbox_input[e]
                for elem in 0..2 {
                    for j in 0..LIMBS_28 {
                        let expected = h_is_full_round.clone() * h_cube_result[elem][j].clone()
                            + (h_one.clone() - h_is_full_round.clone())
                                * h_sbox_input[elem][j].clone();
                        eval.add_constraint(h_post_sbox[elem][j].clone() - expected);
                    }
                }
            }

            // MDS constraint
            if hades_debug_level >= 4 {
                mds_constraint::<E>(
                    &h_post_sbox[0],
                    &h_post_sbox[1],
                    &h_post_sbox[2],
                    &h_mds_result[0],
                    &h_mds_result[1],
                    &h_mds_result[2],
                    &h_mds_carries[0],
                    &h_mds_carries[1],
                    &h_mds_carries[2],
                    &h_mds_k[0],
                    &h_mds_k[1],
                    &h_mds_k[2],
                    &mut eval,
                    &h_p_limbs,
                    &h_is_real,
                    h_rc_ref,
                );
            }

            // Round transition: mds_result[row] == state_before[row+1]
            // Active on is_chain_round (all real rows except last in block).
            if hades_debug_level >= 5 {
                for elem in 0..3 {
                    for j in 0..LIMBS_28 {
                        eval.add_constraint(
                            h_is_chain_round.clone()
                                * (h_mds_result[elem][j].clone()
                                    - h_shifted_next_state[elem][j].clone()),
                        );
                    }
                }
                for j in 0..LIMBS_28 {
                    eval.add_constraint(
                        h_is_first_round.clone()
                            * (h_input_digest_9bit[j].clone() - h_state_before[0][j].clone()),
                    );
                    eval.add_constraint(
                        h_is_chain_round.clone()
                            * (h_input_digest_9bit[j].clone()
                                - h_shifted_next_input_digest_9bit[j].clone()),
                    );
                }
            }

            // Boundary selector constraints. These selectors gate the Hades
            // provider rows in the shared chain↔Hades LogUp relation.
            if hades_debug_level >= 6 {
                eval.add_constraint(
                    h_is_first_round.clone() * (h_is_first_round.clone() - h_one.clone()),
                );
                eval.add_constraint(
                    h_is_last_round.clone() * (h_is_last_round.clone() - h_one.clone()),
                );
            }

            // Repack verification for provider keys. On each last-round row,
            // prove that the provider key really is the 28-bit digest repack of
            // the verified Hades input/output.
            if hades_debug_level >= 6 {
                let split_limbs: [usize; 8] = [3, 6, 9, 12, 15, 18, 21, 24];
                let lo_bits: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

                for s in 0..8 {
                    let two_pow_lo = E::F::from(M31::from(1u32 << lo_bits[s]));
                    eval.add_constraint(
                        h_is_last_round.clone()
                            * (h_input_digest_9bit[split_limbs[s]].clone()
                                - h_split_lo_in[s].clone()
                                - h_split_hi_in[s].clone() * two_pow_lo.clone()),
                    );
                    eval.add_constraint(
                        h_is_last_round.clone()
                            * (h_mds_result[0][split_limbs[s]].clone()
                                - h_split_lo_out[s].clone()
                                - h_split_hi_out[s].clone() * two_pow_lo),
                    );
                }

                let expected_in_b0 = h_input_digest_9bit[0].clone()
                    + h_input_digest_9bit[1].clone() * E::F::from(M31::from(1u32 << 9))
                    + h_input_digest_9bit[2].clone() * E::F::from(M31::from(1u32 << 18))
                    + h_split_lo_in[0].clone() * E::F::from(M31::from(1u32 << 27));
                eval.add_constraint(
                    h_is_last_round.clone() * (h_input_digest_28bit[0].clone() - expected_in_b0),
                );

                let expected_out_b0 = h_mds_result[0][0].clone()
                    + h_mds_result[0][1].clone() * E::F::from(M31::from(1u32 << 9))
                    + h_mds_result[0][2].clone() * E::F::from(M31::from(1u32 << 18))
                    + h_split_lo_out[0].clone() * E::F::from(M31::from(1u32 << 27));
                eval.add_constraint(
                    h_is_last_round.clone() * (h_output_digest_28bit[0].clone() - expected_out_b0),
                );

                for k in 1..8usize {
                    let hi_bits_prev = 9 - lo_bits[k - 1];
                    let j_base = split_limbs[k - 1] + 1;
                    let shift_lo = 28 - lo_bits[k];
                    let expected_in = h_split_hi_in[k - 1].clone()
                        + h_input_digest_9bit[j_base].clone()
                            * E::F::from(M31::from(1u32 << hi_bits_prev))
                        + h_input_digest_9bit[j_base + 1].clone()
                            * E::F::from(M31::from(1u32 << (hi_bits_prev + 9)))
                        + h_split_lo_in[k].clone() * E::F::from(M31::from(1u32 << shift_lo));
                    eval.add_constraint(
                        h_is_last_round.clone() * (h_input_digest_28bit[k].clone() - expected_in),
                    );

                    let expected_out = h_split_hi_out[k - 1].clone()
                        + h_mds_result[0][j_base].clone()
                            * E::F::from(M31::from(1u32 << hi_bits_prev))
                        + h_mds_result[0][j_base + 1].clone()
                            * E::F::from(M31::from(1u32 << (hi_bits_prev + 9)))
                        + h_split_lo_out[k].clone() * E::F::from(M31::from(1u32 << shift_lo));
                    eval.add_constraint(
                        h_is_last_round.clone() * (h_output_digest_28bit[k].clone() - expected_out),
                    );
                }

                let expected_in_b8 = h_split_hi_in[7].clone()
                    + h_input_digest_9bit[25].clone() * E::F::from(M31::from(1u32 << 1))
                    + h_input_digest_9bit[26].clone() * E::F::from(M31::from(1u32 << 10))
                    + h_input_digest_9bit[27].clone() * E::F::from(M31::from(1u32 << 19));
                eval.add_constraint(
                    h_is_last_round.clone() * (h_input_digest_28bit[8].clone() - expected_in_b8),
                );

                let expected_out_b8 = h_split_hi_out[7].clone()
                    + h_mds_result[0][25].clone() * E::F::from(M31::from(1u32 << 1))
                    + h_mds_result[0][26].clone() * E::F::from(M31::from(1u32 << 10))
                    + h_mds_result[0][27].clone() * E::F::from(M31::from(1u32 << 19));
                eval.add_constraint(
                    h_is_last_round.clone() * (h_output_digest_28bit[8].clone() - expected_out_b8),
                );
            }

            // Hades provider side of the shared LogUp relation. Each verified
            // Hades block contributes -1 for its (input_digest, output_digest)
            // key; chain rows contribute +1 below.
            if hades_debug_level >= 7 {
                if let Some(ref hades_rel) = self.hades_lookup {
                    let mut key_values: Vec<E::F> = Vec::with_capacity(18);
                    for v in &h_input_digest_28bit {
                        key_values.push(v.clone());
                    }
                    for v in &h_output_digest_28bit {
                        key_values.push(v.clone());
                    }

                    let neg_one = E::F::from(M31::from(0u32)) - E::F::from(M31::from(1u32));
                    eval.add_to_relation(RelationEntry::new(
                        hades_rel,
                        E::EF::from(h_is_last_round.clone() * neg_one),
                        &key_values,
                    ));
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // PRIMITIVE VERIFIER ARITHMETIC AIR
        // ══════════════════════════════════════════════════════════════
        //
        // These rows constrain the local field facts recorded during verifier
        // replay: QM31 addition, QM31 multiplication, and equality assertions.
        // This is intentionally a primitive row checker. It does not yet bind
        // each row to a full verifier-step state machine, so higher-level
        // metadata must not call this "full verifier execution AIR" yet.

        if self.arithmetic_enabled {
            let arith_a: [E::F; ARITH_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let arith_b: [E::F; ARITH_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let arith_result: [E::F; ARITH_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let arith_mul_a: [E::F; ARITH_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let arith_mul_b: [E::F; ARITH_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let arith_mul_result: [E::F; ARITH_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let arith_is_add = eval.next_trace_mask();
            let arith_is_mul = eval.next_trace_mask();
            let arith_is_eq = eval.next_trace_mask();
            let arith_is_active = eval.next_trace_mask();
            let arith_active_count = eval.next_trace_mask();
            let arith_active_count_next = eval.next_trace_mask();

            let one = E::F::from(M31::from(1u32));
            let two = E::F::from(M31::from(2u32));

            for selector in [
                arith_is_add.clone(),
                arith_is_mul.clone(),
                arith_is_eq.clone(),
                arith_is_active.clone(),
            ] {
                eval.add_constraint(selector.clone() * (one.clone() - selector));
            }

            eval.add_constraint(
                arith_is_add.clone() + arith_is_mul.clone() + arith_is_eq.clone()
                    - arith_is_active.clone(),
            );

            let n = 1u32 << self.log_n_rows;
            let n_inv = M31::from(n).inverse();
            let correction = E::F::from(M31::from(self.n_arithmetic_rows)) * E::F::from(n_inv);
            eval.add_constraint(
                arith_active_count_next.clone()
                    - arith_active_count.clone()
                    - arith_is_active.clone()
                    + correction,
            );

            for j in 0..ARITH_VALUE_LIMBS {
                eval.add_constraint(
                    arith_is_eq.clone() * (arith_a[j].clone() - arith_b[j].clone()),
                );
                eval.add_constraint(
                    arith_is_add.clone()
                        * (arith_a[j].clone() + arith_b[j].clone() - arith_result[j].clone()),
                );
                eval.add_constraint(
                    arith_mul_a[j].clone() - arith_is_mul.clone() * arith_a[j].clone(),
                );
                eval.add_constraint(
                    arith_mul_b[j].clone() - arith_is_mul.clone() * arith_b[j].clone(),
                );
                eval.add_constraint(
                    arith_mul_result[j].clone() - arith_is_mul.clone() * arith_result[j].clone(),
                );
            }

            // QM31 = (a0 + a1*i) + (a2 + a3*i)u, i^2 = -1, u^2 = 2+i.
            // (x0 + x1*u)(y0 + y1*u) =
            //   (x0*y0 + (2+i)*x1*y1) + (x0*y1 + x1*y0)u.
            let cmul = |x_re: E::F, x_im: E::F, y_re: E::F, y_im: E::F| -> (E::F, E::F) {
                (
                    x_re.clone() * y_re.clone() - x_im.clone() * y_im.clone(),
                    x_re * y_im + x_im * y_re,
                )
            };

            let (x0y0_re, x0y0_im) = cmul(
                arith_mul_a[0].clone(),
                arith_mul_a[1].clone(),
                arith_mul_b[0].clone(),
                arith_mul_b[1].clone(),
            );
            let (x1y1_re, x1y1_im) = cmul(
                arith_mul_a[2].clone(),
                arith_mul_a[3].clone(),
                arith_mul_b[2].clone(),
                arith_mul_b[3].clone(),
            );
            let (x0y1_re, x0y1_im) = cmul(
                arith_mul_a[0].clone(),
                arith_mul_a[1].clone(),
                arith_mul_b[2].clone(),
                arith_mul_b[3].clone(),
            );
            let (x1y0_re, x1y0_im) = cmul(
                arith_mul_a[2].clone(),
                arith_mul_a[3].clone(),
                arith_mul_b[0].clone(),
                arith_mul_b[1].clone(),
            );

            let expected_mul = [
                x0y0_re + two.clone() * x1y1_re.clone() - x1y1_im.clone(),
                x0y0_im + x1y1_re + two.clone() * x1y1_im,
                x0y1_re + x1y0_re,
                x0y1_im + x1y0_im,
            ];
            for j in 0..ARITH_VALUE_LIMBS {
                eval.add_constraint(arith_mul_result[j].clone() - expected_mul[j].clone());
            }
        }

        // ══════════════════════════════════════════════════════════════
        // RECORDED SUMCHECK ROUND AIR
        // ══════════════════════════════════════════════════════════════
        //
        // Locally constrains each recorded verifier round:
        //   p(0) + p(1) = claim
        //   p(challenge) = next_claim
        // for degree-2 and degree-3 round polynomials. This still does not
        // bind rounds into a complete verifier control-flow state machine.

        if self.sumcheck_enabled {
            let sc_c0: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_c1: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_c2: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_c3: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_claim: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_challenge: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_challenge_draw_index = eval.next_trace_mask();
            let sc_next_claim: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_r2: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_r3: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_c1r: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_c2r2: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_c3r3: [E::F; SUMCHECK_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let sc_is_deg2 = eval.next_trace_mask();
            let sc_is_deg3 = eval.next_trace_mask();
            let sc_is_active = eval.next_trace_mask();
            let sc_active_count = eval.next_trace_mask();
            let sc_active_count_next = eval.next_trace_mask();

            let one = E::F::from(M31::from(1u32));
            let two = E::F::from(M31::from(2u32));

            for selector in [sc_is_deg2.clone(), sc_is_deg3.clone(), sc_is_active.clone()] {
                eval.add_constraint(selector.clone() * (one.clone() - selector));
            }
            eval.add_constraint(sc_is_deg2.clone() + sc_is_deg3.clone() - sc_is_active.clone());

            let n = 1u32 << self.log_n_rows;
            let n_inv = M31::from(n).inverse();
            let correction = E::F::from(M31::from(self.n_sumcheck_rows)) * E::F::from(n_inv);
            eval.add_constraint(
                sc_active_count_next.clone() - sc_active_count.clone() - sc_is_active.clone()
                    + correction,
            );

            for j in 0..SUMCHECK_VALUE_LIMBS {
                // Degree-2 rows must have no cubic term.
                eval.add_constraint(sc_is_deg2.clone() * sc_c3[j].clone());

                // p(0)+p(1) = c0 + (c0+c1+c2+c3) = claim.
                eval.add_constraint(
                    two.clone() * sc_c0[j].clone()
                        + sc_c1[j].clone()
                        + sc_c2[j].clone()
                        + sc_c3[j].clone()
                        - sc_claim[j].clone(),
                );
            }

            let cmul = |x_re: E::F, x_im: E::F, y_re: E::F, y_im: E::F| -> (E::F, E::F) {
                (
                    x_re.clone() * y_re.clone() - x_im.clone() * y_im.clone(),
                    x_re * y_im + x_im * y_re,
                )
            };
            let qm31_mul_expected = |a: &[E::F; 4], b: &[E::F; 4]| -> [E::F; 4] {
                let (x0y0_re, x0y0_im) =
                    cmul(a[0].clone(), a[1].clone(), b[0].clone(), b[1].clone());
                let (x1y1_re, x1y1_im) =
                    cmul(a[2].clone(), a[3].clone(), b[2].clone(), b[3].clone());
                let (x0y1_re, x0y1_im) =
                    cmul(a[0].clone(), a[1].clone(), b[2].clone(), b[3].clone());
                let (x1y0_re, x1y0_im) =
                    cmul(a[2].clone(), a[3].clone(), b[0].clone(), b[1].clone());
                [
                    x0y0_re + two.clone() * x1y1_re.clone() - x1y1_im.clone(),
                    x0y0_im + x1y1_re + two.clone() * x1y1_im,
                    x0y1_re + x1y0_re,
                    x0y1_im + x1y0_im,
                ]
            };

            let expected_r2 = qm31_mul_expected(&sc_challenge, &sc_challenge);
            let expected_r3 = qm31_mul_expected(&sc_r2, &sc_challenge);
            let expected_c1r = qm31_mul_expected(&sc_c1, &sc_challenge);
            let expected_c2r2 = qm31_mul_expected(&sc_c2, &sc_r2);
            let expected_c3r3 = qm31_mul_expected(&sc_c3, &sc_r3);

            for j in 0..SUMCHECK_VALUE_LIMBS {
                eval.add_constraint(sc_r2[j].clone() - expected_r2[j].clone());
                eval.add_constraint(sc_r3[j].clone() - expected_r3[j].clone());
                eval.add_constraint(sc_c1r[j].clone() - expected_c1r[j].clone());
                eval.add_constraint(sc_c2r2[j].clone() - expected_c2r2[j].clone());
                eval.add_constraint(sc_c3r3[j].clone() - expected_c3r3[j].clone());
                eval.add_constraint(
                    sc_c0[j].clone() + sc_c1r[j].clone() + sc_c2r2[j].clone() + sc_c3r3[j].clone()
                        - sc_next_claim[j].clone(),
                );
            }

            if let Some(ref challenge_rel) = self.challenge_lookup {
                let mut key_values: Vec<E::F> = sc_challenge.iter().cloned().collect();
                key_values.push(sc_challenge_draw_index.clone());
                eval.add_to_relation(RelationEntry::new(
                    challenge_rel,
                    E::EF::from(sc_is_active.clone()),
                    &key_values,
                ));
            }
        }

        // ══════════════════════════════════════════════════════════════
        // RECORDED CHANNEL DRAW AIR
        // ══════════════════════════════════════════════════════════════
        //
        // Locally constrains the draw-row selector/count and provides the
        // drawn QM31 values to the sumcheck challenge LogUp relation. This
        // proves every constrained sumcheck challenge is present in the
        // recorded draw multiset.

        if self.draw_enabled {
            let draw_value: [E::F; DRAW_VALUE_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let draw_index = eval.next_trace_mask();
            let draw_raw_felt: [E::F; DRAW_RAW_FELT_LIMBS] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let draw_split_lo: [E::F; DRAW_SPLIT_COUNT] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let draw_split_hi: [E::F; DRAW_SPLIT_COUNT] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let draw_split_bits: [E::F; DRAW_SPLIT_BITS_TOTAL] =
                std::array::from_fn(|_| eval.next_trace_mask());
            let draw_is_active = eval.next_trace_mask();
            let draw_active_count = eval.next_trace_mask();
            let draw_active_count_next = eval.next_trace_mask();

            let one = E::F::from(M31::from(1u32));
            eval.add_constraint(draw_is_active.clone() * (one.clone() - draw_is_active.clone()));

            let n = 1u32 << self.log_n_rows;
            let n_inv = M31::from(n).inverse();
            let correction = E::F::from(M31::from(self.n_draw_rows)) * E::F::from(n_inv);
            eval.add_constraint(
                draw_active_count_next.clone() - draw_active_count.clone() - draw_is_active.clone()
                    + correction,
            );

            for bit in &draw_split_bits {
                eval.add_constraint(bit.clone() * (one.clone() - bit.clone()));
            }

            let split_lo_bits = [3u32, 6, 9, 12];
            let mut bit_offset = 0usize;
            for split_idx in 0..DRAW_SPLIT_COUNT {
                let lo_bits = split_lo_bits[split_idx] as usize;
                let hi_bits = 28usize - lo_bits;
                let mut lo_acc = E::F::from(M31::from(0u32));
                for bit in 0..lo_bits {
                    lo_acc = lo_acc
                        + draw_split_bits[bit_offset + bit].clone()
                            * E::F::from(M31::from(1u32 << bit));
                }
                bit_offset += lo_bits;

                let mut hi_acc = E::F::from(M31::from(0u32));
                for bit in 0..hi_bits {
                    hi_acc = hi_acc
                        + draw_split_bits[bit_offset + bit].clone()
                            * E::F::from(M31::from(1u32 << bit));
                }
                bit_offset += hi_bits;

                let two_pow_lo = E::F::from(M31::from(1u32 << split_lo_bits[split_idx]));
                eval.add_constraint(draw_split_lo[split_idx].clone() - lo_acc);
                eval.add_constraint(draw_split_hi[split_idx].clone() - hi_acc);
                eval.add_constraint(
                    draw_raw_felt[split_idx + 1].clone()
                        - draw_split_lo[split_idx].clone()
                        - draw_split_hi[split_idx].clone() * two_pow_lo,
                );
            }

            // draw_qm31 extracts four 31-bit chunks from the raw felt, LSB first.
            // The chunks cross 28-bit felt limbs at offsets 31, 62, 93, and 124.
            let expected_draw = [
                draw_raw_felt[0].clone()
                    + draw_split_lo[0].clone() * E::F::from(M31::from(1u32 << 28)),
                draw_split_hi[0].clone()
                    + draw_split_lo[1].clone() * E::F::from(M31::from(1u32 << 25)),
                draw_split_hi[1].clone()
                    + draw_split_lo[2].clone() * E::F::from(M31::from(1u32 << 22)),
                draw_split_hi[2].clone()
                    + draw_split_lo[3].clone() * E::F::from(M31::from(1u32 << 19)),
            ];
            for j in 0..DRAW_VALUE_LIMBS {
                eval.add_constraint(draw_value[j].clone() - expected_draw[j].clone());
            }

            if let Some(ref draw_felt_rel) = self.draw_felt_lookup {
                let mut key_values: Vec<E::F> = draw_raw_felt.iter().cloned().collect();
                key_values.push(draw_index.clone());
                eval.add_to_relation(RelationEntry::new(
                    draw_felt_rel,
                    E::EF::from(draw_is_active.clone()),
                    &key_values,
                ));
            }

            if let Some(ref challenge_rel) = self.challenge_lookup {
                let mut key_values: Vec<E::F> = draw_value.iter().cloned().collect();
                key_values.push(draw_index.clone());
                let neg_one = E::F::from(M31::from(0u32)) - E::F::from(M31::from(1u32));
                eval.add_to_relation(RelationEntry::new(
                    challenge_rel,
                    E::EF::from(draw_is_active.clone() * neg_one),
                    &key_values,
                ));
            }
        }

        if let Some(ref draw_felt_rel) = self.draw_felt_lookup {
            let mut key_values: Vec<E::F> = digest_after.iter().cloned().collect();
            key_values.push(draw_count.clone());
            let neg_one = E::F::from(M31::from(0u32)) - E::F::from(M31::from(1u32));
            eval.add_to_relation(RelationEntry::new(
                draw_felt_rel,
                E::EF::from(is_draw.clone() * neg_one),
                &key_values,
            ));
        }

        // ── LogUp: Hades permutation binding ─────────────────────────
        // Key: (digest_before[9], digest_after[9]) = 18 M31 elements.
        // Only active rows contribute (+1 multiplicity).
        // Padding rows contribute 0 multiplicity (is_active = 0).
        if let Some(ref hades_rel) = self.hades_lookup {
            let mut key_values: Vec<E::F> = Vec::with_capacity(18);
            for v in &digest_before {
                key_values.push(v.clone());
            }
            for v in &digest_after {
                key_values.push(v.clone());
            }

            eval.add_to_relation(RelationEntry::new(
                hades_rel,
                E::EF::from(is_active.clone()),
                &key_values,
            ));
        }

        if self.hades_lookup.is_some()
            || self.draw_felt_lookup.is_some()
            || self.challenge_lookup.is_some()
        {
            eval.finalize_logup();
        }

        eval
    }
}

/// The recursive verifier STARK component.
pub type RecursiveVerifierComponent = FrameworkComponent<RecursiveVerifierEval>;

// ═══════════════════════════════════════════════════════════════════════
// Trace building — populates real Hades states from the witness
// ═══════════════════════════════════════════════════════════════════════

use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};

/// A single chain trace row — one Hades permutation.
struct ChainRow {
    full_input: [FieldElement; 3],
    full_output: [FieldElement; 3],
    is_draw: bool,
    /// Value added to output[0] before the NEXT perm's input[0].
    /// Zero for most rows. Non-zero for intermediate perms within
    /// multi-Hades ChannelOps (e.g., felt2 in mix_poly_coeffs).
    addition_digest: FieldElement,
}

/// Build the execution trace from real Hades permutation states.
///
/// Each row stores the actual felt252 Hades input/output from the GKR
/// verifier's Fiat-Shamir transcript, decomposed into M31 limbs.
///
/// Layout:
///   [CHAIN_COL_DIGEST_BEFORE..CHAIN_COL_DIGEST_AFTER) digest_before
///   [CHAIN_COL_DIGEST_AFTER..CHAIN_COL_SHIFTED_NEXT_BEFORE) digest_after
///   [CHAIN_COL_SHIFTED_NEXT_BEFORE..CHAIN_COL_ADDITION_DIGEST) shifted_next_before
///   [CHAIN_COL_ADDITION_DIGEST..CHAIN_COL_CARRY_POS) addition_digest
///   [CHAIN_COL_CARRY_POS..CHAIN_COL_CARRY_NEG) addition_carry_pos[8]
///   [CHAIN_COL_CARRY_NEG..CHAIN_COL_ADDITION_K) addition_carry_neg[8]
///   [CHAIN_COL_ADDITION_K] addition_k
///   [CHAIN_COL_IS_ACTIVE] is_active
///   [CHAIN_COL_ACTIVE_COUNT] active_count
///   [CHAIN_COL_ACTIVE_COUNT_NEXT] active_count_next
///   [CHAIN_COL_IS_DRAW] is_draw
///   [CHAIN_COL_DRAW_COUNT] draw_count
///   [CHAIN_COL_SHIFTED_NEXT_DRAW_COUNT] shifted_next_draw_count
pub fn build_recursive_trace(witness: &super::types::GkrVerifierWitness) -> RecursiveTraceData {
    use super::types::WitnessOp;

    // Build one row per HadesPerm — 1:1 correspondence with Hades AIR for LogUp.
    // Multi-Hades ChannelOps (mix_poly_coeffs: 2 calls) produce 2 rows.
    // The addition_digest column stores the value added to output[0] before
    // the next perm's input[0]. The carry-chain constraint handles modular
    // limb addition overflow.
    let mut rows: Vec<ChainRow> = Vec::new();

    let hades_ops: Vec<([FieldElement; 3], [FieldElement; 3], bool)> = witness
        .ops
        .iter()
        .enumerate()
        .filter_map(|(idx, op)| {
            if let WitnessOp::HadesPerm { input, output } = op {
                let is_draw = matches!(
                    (witness.ops.get(idx + 1), witness.ops.get(idx + 2)),
                    (
                        Some(WitnessOp::ChannelOp { .. }),
                        Some(WitnessOp::ChannelDraw { .. })
                    )
                );
                Some((*input, *output, is_draw))
            } else {
                None
            }
        })
        .collect();

    for i in 0..hades_ops.len() {
        let (input, output, is_draw) = hades_ops[i];
        let addition = if i + 1 < hades_ops.len() {
            hades_ops[i + 1].0[0] - output[0]
        } else {
            FieldElement::ZERO
        };
        rows.push(ChainRow {
            full_input: input,
            full_output: output,
            is_draw,
            addition_digest: addition,
        });
    }

    let n_ops = rows.len();
    let n_real_rows = n_ops;
    let n_for_sizing = witness.n_poseidon_perms.max(n_ops);

    // SECURITY: Ensure at least one padding row (n_padded > n_real_rows).
    // This guarantees is_active transitions from 1→0 within the trace,
    // so the final boundary constraint fires correctly without relying
    // on preprocessed selectors.
    let log_size = if n_for_sizing <= 1 {
        2 // minimum log_size=2 → 4 rows (even for 1-2 real rows)
    } else {
        // +1 ensures n_padded > n_for_sizing (at least one padding row)
        ((n_for_sizing + 1) as u32)
            .next_power_of_two()
            .ilog2()
            .max(2)
    };
    let n_padded_rows = 1usize << log_size;
    assert!(
        n_padded_rows > n_real_rows,
        "n_padded ({n_padded_rows}) must be > n_real ({n_real_rows}) for boundary constraints"
    );

    // Build trace columns (pos/neg carry layout)
    let mut execution_trace: Vec<Vec<M31>> = Vec::with_capacity(COLS_PER_ROW);
    for _ in 0..COLS_PER_ROW {
        execution_trace.push(vec![M31::from_u32_unchecked(0); n_padded_rows]);
    }

    let col_digest_before = CHAIN_COL_DIGEST_BEFORE;
    let col_digest_after = CHAIN_COL_DIGEST_AFTER;
    let col_shifted = CHAIN_COL_SHIFTED_NEXT_BEFORE;
    let col_addition = CHAIN_COL_ADDITION_DIGEST;
    let col_carry_pos = CHAIN_COL_CARRY_POS;
    let col_carry_neg = CHAIN_COL_CARRY_NEG;
    let col_k = CHAIN_COL_ADDITION_K;
    let col_is_active = CHAIN_COL_IS_ACTIVE;
    let col_active_count = CHAIN_COL_ACTIVE_COUNT;
    let col_active_count_next = CHAIN_COL_ACTIVE_COUNT_NEXT;
    let col_is_draw = CHAIN_COL_IS_DRAW;
    let col_draw_count = CHAIN_COL_DRAW_COUNT;
    let col_shifted_next_draw_count = CHAIN_COL_SHIFTED_NEXT_DRAW_COUNT;

    // Populate trace
    for row_idx in 0..n_padded_rows {
        if row_idx < n_ops {
            let r = &rows[row_idx];
            // digest_before: only input state[0] (9 limbs)
            let input_digest_limbs = felt252_to_limbs(&r.full_input[0]);
            for j in 0..LIMBS_PER_FELT {
                execution_trace[col_digest_before + j][row_idx] = input_digest_limbs[j];
            }
            // digest_after: only output state[0] (9 limbs)
            let output_digest_limbs = felt252_to_limbs(&r.full_output[0]);
            for j in 0..LIMBS_PER_FELT {
                execution_trace[col_digest_after + j][row_idx] = output_digest_limbs[j];
            }
            // addition_digest: 9 limbs
            let add_limbs = felt252_to_limbs(&r.addition_digest);
            for j in 0..LIMBS_PER_FELT {
                execution_trace[col_addition + j][row_idx] = add_limbs[j];
            }
            if r.is_draw {
                execution_trace[col_is_draw][row_idx] = M31::from(1);
            }
        }
        // Padding rows remain zero (initialized above)
    }

    // Second pass: shifted_next_before[row_i] = digest_before[row_{(i+1) mod N}]
    // Circle domain wrap-around: last row shifts to first row's digest.
    for row_idx in 0..n_padded_rows {
        let next_idx = (row_idx + 1) % n_padded_rows;
        for j in 0..LIMBS_PER_FELT {
            execution_trace[col_shifted + j][row_idx] =
                execution_trace[col_digest_before + j][next_idx];
        }
    }

    // Third pass: compute carry-chain witnesses for modular addition
    {
        for row_idx in 0..n_real_rows.saturating_sub(1) {
            let da_limbs: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| execution_trace[col_digest_after + j][row_idx]);
            let add_limbs: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| execution_trace[col_addition + j][row_idx]);
            let next_before_limbs: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| execution_trace[col_digest_before + j][row_idx + 1]);
            let (carry_pos, carry_neg, k) =
                compute_addition_carry_chain(&da_limbs, &add_limbs, &next_before_limbs);
            for j in 0..8 {
                execution_trace[col_carry_pos + j][row_idx] = carry_pos[j];
                execution_trace[col_carry_neg + j][row_idx] = carry_neg[j];
            }
            execution_trace[col_k][row_idx] = k;
        }
    }

    // ── Execution-trace selectors (columns 45..47) ──────────────────
    // is_active: 1 for real rows, 0 for padding
    for i in 0..n_real_rows.min(n_padded_rows) {
        execution_trace[col_is_active][i] = M31::from_u32_unchecked(1);
    }

    // active_count: amortized accumulator
    // active_count[i+1] = active_count[i] + is_active[i] - n_real_rows * N_inv
    let n_m31 = M31::from(n_padded_rows as u32);
    let n_inv = n_m31.inverse();
    let correction = M31::from(n_real_rows as u32) * n_inv;

    execution_trace[col_active_count][0] = M31::from_u32_unchecked(0);
    for i in 0..n_padded_rows - 1 {
        let is_act = execution_trace[col_is_active][i];
        execution_trace[col_active_count][i + 1] =
            execution_trace[col_active_count][i] + is_act - correction;
    }

    // active_count_next[i] = active_count[(i+1) mod N]
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        execution_trace[col_active_count_next][i] = execution_trace[col_active_count][next];
    }

    let mut draw_count = M31::from_u32_unchecked(0);
    for i in 0..n_real_rows.min(n_padded_rows) {
        execution_trace[col_draw_count][i] = draw_count;
        if execution_trace[col_is_draw][i] == M31::from_u32_unchecked(1) {
            draw_count += M31::from_u32_unchecked(1);
        }
    }
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        execution_trace[col_shifted_next_draw_count][i] = execution_trace[col_draw_count][next];
    }

    // Preprocessed columns (kept for Tree 0 structural compatibility)
    let mut is_first = vec![M31::from_u32_unchecked(0); n_padded_rows];
    let mut is_last = vec![M31::from_u32_unchecked(0); n_padded_rows];
    let mut is_chain = vec![M31::from_u32_unchecked(0); n_padded_rows];

    is_first[0] = M31::from_u32_unchecked(1);
    if n_real_rows > 0 && n_real_rows <= n_padded_rows {
        is_last[n_real_rows - 1] = M31::from_u32_unchecked(1);
    }
    for i in 0..n_real_rows.saturating_sub(1).min(n_padded_rows) {
        is_chain[i] = M31::from_u32_unchecked(1);
    }

    RecursiveTraceData {
        execution_trace,
        preprocessed_is_first: is_first,
        preprocessed_is_last: is_last,
        preprocessed_is_chain: is_chain,
        log_size,
        n_real_rows,
        n_channel_ops: n_ops,
    }
}

/// Build primitive verifier arithmetic rows from recorded QM31/equality ops.
///
/// This constrains local arithmetic facts, but it is not a complete verifier
/// state-machine AIR. Semantic linking to sumcheck/verifier control flow is a
/// separate hardening step.
pub fn build_arithmetic_trace(witness: &super::types::GkrVerifierWitness) -> ArithmeticTraceData {
    use super::types::WitnessOp;

    #[derive(Clone, Copy)]
    enum ArithKind {
        Add,
        Mul,
        Eq,
    }

    let mut rows: Vec<(
        [M31; ARITH_VALUE_LIMBS],
        [M31; ARITH_VALUE_LIMBS],
        [M31; ARITH_VALUE_LIMBS],
        ArithKind,
    )> = Vec::new();

    for op in &witness.ops {
        match op {
            WitnessOp::QM31Add { a, b, result } => {
                rows.push((
                    a.to_m31_array(),
                    b.to_m31_array(),
                    result.to_m31_array(),
                    ArithKind::Add,
                ));
            }
            WitnessOp::QM31Mul { a, b, result } => {
                rows.push((
                    a.to_m31_array(),
                    b.to_m31_array(),
                    result.to_m31_array(),
                    ArithKind::Mul,
                ));
            }
            WitnessOp::EqualityCheck { lhs, rhs } => {
                rows.push((
                    lhs.to_m31_array(),
                    rhs.to_m31_array(),
                    [M31::from_u32_unchecked(0); ARITH_VALUE_LIMBS],
                    ArithKind::Eq,
                ));
            }
            _ => {}
        }
    }

    let n_real_rows = rows.len();
    let log_size = if n_real_rows <= 1 {
        2
    } else {
        ((n_real_rows + 1) as u32)
            .next_power_of_two()
            .ilog2()
            .max(2)
    };
    let n_padded_rows = 1usize << log_size;

    let mut trace: Vec<Vec<M31>> = (0..ARITH_COLS_PER_ROW)
        .map(|_| vec![M31::from_u32_unchecked(0); n_padded_rows])
        .collect();

    for (row_idx, (a, b, result, kind)) in rows.iter().enumerate() {
        for j in 0..ARITH_VALUE_LIMBS {
            trace[ARITH_COL_A + j][row_idx] = a[j];
            trace[ARITH_COL_B + j][row_idx] = b[j];
            trace[ARITH_COL_RESULT + j][row_idx] = result[j];
        }
        match kind {
            ArithKind::Add => trace[ARITH_COL_IS_ADD][row_idx] = M31::from_u32_unchecked(1),
            ArithKind::Mul => {
                trace[ARITH_COL_IS_MUL][row_idx] = M31::from_u32_unchecked(1);
                for j in 0..ARITH_VALUE_LIMBS {
                    trace[ARITH_COL_MUL_A + j][row_idx] = a[j];
                    trace[ARITH_COL_MUL_B + j][row_idx] = b[j];
                    trace[ARITH_COL_MUL_RESULT + j][row_idx] = result[j];
                }
            }
            ArithKind::Eq => trace[ARITH_COL_IS_EQ][row_idx] = M31::from_u32_unchecked(1),
        }
        trace[ARITH_COL_IS_ACTIVE][row_idx] = M31::from_u32_unchecked(1);
    }

    recompute_arithmetic_accumulator(&mut trace, n_real_rows, log_size);

    ArithmeticTraceData {
        trace,
        log_size,
        n_real_rows,
    }
}

/// Resize arithmetic columns and recompute the amortized active-row accumulator
/// for the committed domain size.
pub fn pad_arithmetic_trace_to_log_size(trace_data: &mut ArithmeticTraceData, log_size: u32) {
    if log_size <= trace_data.log_size {
        return;
    }
    let n_padded_rows = 1usize << log_size;
    for col in trace_data.trace.iter_mut() {
        col.resize(n_padded_rows, M31::from_u32_unchecked(0));
    }
    recompute_arithmetic_accumulator(&mut trace_data.trace, trace_data.n_real_rows, log_size);
    trace_data.log_size = log_size;
}

fn recompute_arithmetic_accumulator(trace: &mut [Vec<M31>], n_real_rows: usize, log_size: u32) {
    let n_padded_rows = 1usize << log_size;
    let n_m31 = M31::from(n_padded_rows as u32);
    let n_inv = n_m31.inverse();
    let correction = M31::from(n_real_rows as u32) * n_inv;

    trace[ARITH_COL_ACTIVE_COUNT][0] = M31::from_u32_unchecked(0);
    for i in 0..n_padded_rows - 1 {
        let is_act = trace[ARITH_COL_IS_ACTIVE][i];
        trace[ARITH_COL_ACTIVE_COUNT][i + 1] =
            trace[ARITH_COL_ACTIVE_COUNT][i] + is_act - correction;
    }
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        trace[ARITH_COL_ACTIVE_COUNT_NEXT][i] = trace[ARITH_COL_ACTIVE_COUNT][next];
    }
}

/// Build locally constrained rows for recorded sumcheck verifier rounds.
pub fn build_sumcheck_trace(witness: &super::types::GkrVerifierWitness) -> SumcheckTraceData {
    use super::types::WitnessOp;
    use stwo::core::fields::qm31::SecureField;

    #[derive(Clone, Copy)]
    enum SumcheckKind {
        Deg2,
        Deg3,
    }

    #[derive(Clone, Copy)]
    struct SumcheckRow {
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
        c3: SecureField,
        claim: SecureField,
        challenge: SecureField,
        challenge_draw_index: M31,
        next_claim: SecureField,
        kind: SumcheckKind,
    }

    let mut rows = Vec::new();
    let mut recorded_draws: Vec<([M31; SUMCHECK_VALUE_LIMBS], bool)> = Vec::new();
    for op in &witness.ops {
        match op {
            WitnessOp::ChannelDraw { result } => {
                recorded_draws.push((result.to_m31_array(), false));
            }
            WitnessOp::SumcheckRoundDeg2 {
                round_poly,
                claim,
                challenge,
                next_claim,
            } => {
                let challenge_limbs = challenge.to_m31_array();
                let draw_index = recorded_draws
                    .iter_mut()
                    .enumerate()
                    .find_map(|(idx, (draw, used))| {
                        if !*used && *draw == challenge_limbs {
                            *used = true;
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                rows.push(SumcheckRow {
                    c0: round_poly.c0,
                    c1: round_poly.c1,
                    c2: round_poly.c2,
                    c3: SecureField::default(),
                    claim: *claim,
                    challenge: *challenge,
                    challenge_draw_index: M31::from(draw_index as u32),
                    next_claim: *next_claim,
                    kind: SumcheckKind::Deg2,
                });
            }
            WitnessOp::SumcheckRoundDeg3 {
                round_poly,
                claim,
                challenge,
                next_claim,
            } => {
                let challenge_limbs = challenge.to_m31_array();
                let draw_index = recorded_draws
                    .iter_mut()
                    .enumerate()
                    .find_map(|(idx, (draw, used))| {
                        if !*used && *draw == challenge_limbs {
                            *used = true;
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                rows.push(SumcheckRow {
                    c0: round_poly.c0,
                    c1: round_poly.c1,
                    c2: round_poly.c2,
                    c3: round_poly.c3,
                    claim: *claim,
                    challenge: *challenge,
                    challenge_draw_index: M31::from(draw_index as u32),
                    next_claim: *next_claim,
                    kind: SumcheckKind::Deg3,
                });
            }
            _ => {}
        }
    }

    let n_real_rows = rows.len();
    let log_size = if n_real_rows <= 1 {
        2
    } else {
        ((n_real_rows + 1) as u32)
            .next_power_of_two()
            .ilog2()
            .max(2)
    };
    let n_padded_rows = 1usize << log_size;
    let mut trace: Vec<Vec<M31>> = (0..SUMCHECK_COLS_PER_ROW)
        .map(|_| vec![M31::from_u32_unchecked(0); n_padded_rows])
        .collect();

    let write_qm31 = |trace: &mut [Vec<M31>], col: usize, row: usize, value: SecureField| {
        let limbs = value.to_m31_array();
        for j in 0..SUMCHECK_VALUE_LIMBS {
            trace[col + j][row] = limbs[j];
        }
    };

    for (row_idx, row) in rows.iter().enumerate() {
        let r2 = row.challenge * row.challenge;
        let r3 = r2 * row.challenge;
        let c1r = row.c1 * row.challenge;
        let c2r2 = row.c2 * r2;
        let c3r3 = row.c3 * r3;

        write_qm31(&mut trace, SUMCHECK_COL_C0, row_idx, row.c0);
        write_qm31(&mut trace, SUMCHECK_COL_C1, row_idx, row.c1);
        write_qm31(&mut trace, SUMCHECK_COL_C2, row_idx, row.c2);
        write_qm31(&mut trace, SUMCHECK_COL_C3, row_idx, row.c3);
        write_qm31(&mut trace, SUMCHECK_COL_CLAIM, row_idx, row.claim);
        write_qm31(&mut trace, SUMCHECK_COL_CHALLENGE, row_idx, row.challenge);
        trace[SUMCHECK_COL_CHALLENGE_DRAW_INDEX][row_idx] = row.challenge_draw_index;
        write_qm31(&mut trace, SUMCHECK_COL_NEXT_CLAIM, row_idx, row.next_claim);
        write_qm31(&mut trace, SUMCHECK_COL_R2, row_idx, r2);
        write_qm31(&mut trace, SUMCHECK_COL_R3, row_idx, r3);
        write_qm31(&mut trace, SUMCHECK_COL_C1R, row_idx, c1r);
        write_qm31(&mut trace, SUMCHECK_COL_C2R2, row_idx, c2r2);
        write_qm31(&mut trace, SUMCHECK_COL_C3R3, row_idx, c3r3);

        match row.kind {
            SumcheckKind::Deg2 => trace[SUMCHECK_COL_IS_DEG2][row_idx] = M31::from(1),
            SumcheckKind::Deg3 => trace[SUMCHECK_COL_IS_DEG3][row_idx] = M31::from(1),
        }
        trace[SUMCHECK_COL_IS_ACTIVE][row_idx] = M31::from(1);
    }

    recompute_sumcheck_accumulator(&mut trace, n_real_rows, log_size);

    SumcheckTraceData {
        trace,
        log_size,
        n_real_rows,
    }
}

/// Resize sumcheck columns and recompute the active-row accumulator for the
/// committed domain size.
pub fn pad_sumcheck_trace_to_log_size(trace_data: &mut SumcheckTraceData, log_size: u32) {
    if log_size <= trace_data.log_size {
        return;
    }
    let n_padded_rows = 1usize << log_size;
    for col in trace_data.trace.iter_mut() {
        col.resize(n_padded_rows, M31::from_u32_unchecked(0));
    }
    recompute_sumcheck_accumulator(&mut trace_data.trace, trace_data.n_real_rows, log_size);
    trace_data.log_size = log_size;
}

fn recompute_sumcheck_accumulator(trace: &mut [Vec<M31>], n_real_rows: usize, log_size: u32) {
    let n_padded_rows = 1usize << log_size;
    let n_m31 = M31::from(n_padded_rows as u32);
    let n_inv = n_m31.inverse();
    let correction = M31::from(n_real_rows as u32) * n_inv;

    trace[SUMCHECK_COL_ACTIVE_COUNT][0] = M31::from_u32_unchecked(0);
    for i in 0..n_padded_rows - 1 {
        let is_act = trace[SUMCHECK_COL_IS_ACTIVE][i];
        trace[SUMCHECK_COL_ACTIVE_COUNT][i + 1] =
            trace[SUMCHECK_COL_ACTIVE_COUNT][i] + is_act - correction;
    }
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        trace[SUMCHECK_COL_ACTIVE_COUNT_NEXT][i] = trace[SUMCHECK_COL_ACTIVE_COUNT][next];
    }
}

/// Build rows for QM31 values drawn from the Fiat-Shamir channel.
pub fn build_draw_trace(witness: &super::types::GkrVerifierWitness) -> DrawTraceData {
    use super::types::WitnessOp;

    #[derive(Clone, Copy)]
    struct DrawRow {
        value: [M31; DRAW_VALUE_LIMBS],
        draw_index: M31,
        raw_felt: [M31; DRAW_RAW_FELT_LIMBS],
    }

    let mut rows = Vec::new();
    let mut draw_index = 0usize;
    for (idx, op) in witness.ops.iter().enumerate() {
        if let WitnessOp::ChannelDraw { result } = op {
            let raw_output = witness.ops[..idx].iter().rev().find_map(|prev| {
                if let WitnessOp::HadesPerm { output, .. } = prev {
                    Some(output[0])
                } else {
                    None
                }
            });
            let raw_felt = raw_output
                .map(|felt| felt252_to_limbs(&felt))
                .unwrap_or([M31::from_u32_unchecked(0); DRAW_RAW_FELT_LIMBS]);
            rows.push(DrawRow {
                value: result.to_m31_array(),
                draw_index: M31::from(draw_index as u32),
                raw_felt,
            });
            draw_index += 1;
        }
    }

    let n_real_rows = rows.len();
    let log_size = if n_real_rows <= 1 {
        2
    } else {
        ((n_real_rows + 1) as u32)
            .next_power_of_two()
            .ilog2()
            .max(2)
    };
    let n_padded_rows = 1usize << log_size;
    let mut trace: Vec<Vec<M31>> = (0..DRAW_COLS_PER_ROW)
        .map(|_| vec![M31::from_u32_unchecked(0); n_padded_rows])
        .collect();

    for (row_idx, row) in rows.iter().enumerate() {
        for j in 0..DRAW_VALUE_LIMBS {
            trace[DRAW_COL_VALUE + j][row_idx] = row.value[j];
        }
        trace[DRAW_COL_DRAW_INDEX][row_idx] = row.draw_index;
        for j in 0..DRAW_RAW_FELT_LIMBS {
            trace[DRAW_COL_RAW_FELT + j][row_idx] = row.raw_felt[j];
        }

        let split_lo_bits = [3u32, 6, 9, 12];
        let mut bit_offset = 0usize;
        for split_idx in 0..DRAW_SPLIT_COUNT {
            let raw = row.raw_felt[split_idx + 1].0;
            let lo_bits = split_lo_bits[split_idx];
            let lo_mask = (1u32 << lo_bits) - 1;
            let lo = raw & lo_mask;
            let hi = raw >> lo_bits;
            trace[DRAW_COL_SPLIT_LO + split_idx][row_idx] = M31::from(lo);
            trace[DRAW_COL_SPLIT_HI + split_idx][row_idx] = M31::from(hi);

            for bit in 0..lo_bits {
                trace[DRAW_COL_SPLIT_BITS + bit_offset][row_idx] = M31::from((lo >> bit) & 1);
                bit_offset += 1;
            }
            for bit in 0..(28 - lo_bits) {
                trace[DRAW_COL_SPLIT_BITS + bit_offset][row_idx] = M31::from((hi >> bit) & 1);
                bit_offset += 1;
            }
        }

        trace[DRAW_COL_IS_ACTIVE][row_idx] = M31::from(1);
    }

    recompute_draw_accumulator(&mut trace, n_real_rows, log_size);

    DrawTraceData {
        trace,
        log_size,
        n_real_rows,
    }
}

/// Resize draw columns and recompute the active-row accumulator for the
/// committed domain size.
pub fn pad_draw_trace_to_log_size(trace_data: &mut DrawTraceData, log_size: u32) {
    if log_size <= trace_data.log_size {
        return;
    }
    let n_padded_rows = 1usize << log_size;
    for col in trace_data.trace.iter_mut() {
        col.resize(n_padded_rows, M31::from_u32_unchecked(0));
    }
    recompute_draw_accumulator(&mut trace_data.trace, trace_data.n_real_rows, log_size);
    trace_data.log_size = log_size;
}

fn recompute_draw_accumulator(trace: &mut [Vec<M31>], n_real_rows: usize, log_size: u32) {
    let n_padded_rows = 1usize << log_size;
    let n_m31 = M31::from(n_padded_rows as u32);
    let n_inv = n_m31.inverse();
    let correction = M31::from(n_real_rows as u32) * n_inv;

    trace[DRAW_COL_ACTIVE_COUNT][0] = M31::from_u32_unchecked(0);
    for i in 0..n_padded_rows - 1 {
        let is_act = trace[DRAW_COL_IS_ACTIVE][i];
        trace[DRAW_COL_ACTIVE_COUNT][i + 1] = trace[DRAW_COL_ACTIVE_COUNT][i] + is_act - correction;
    }
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        trace[DRAW_COL_ACTIVE_COUNT_NEXT][i] = trace[DRAW_COL_ACTIVE_COUNT][next];
    }
}

/// Container for the recursive STARK trace data.
pub struct RecursiveTraceData {
    /// Execution trace columns (COLS_PER_ROW columns x 2^log_size rows).
    pub execution_trace: Vec<Vec<M31>>,

    /// Preprocessed column: 1 on row 0.
    pub preprocessed_is_first: Vec<M31>,

    /// Preprocessed column: 1 on the last real row.
    pub preprocessed_is_last: Vec<M31>,

    /// Preprocessed column: 1 on all real rows except the last.
    pub preprocessed_is_chain: Vec<M31>,

    /// log2 of padded trace height.
    pub log_size: u32,

    /// Number of real (non-padding) rows.
    pub n_real_rows: usize,

    /// Number of channel operations (ChannelOp entries) in the trace.
    pub n_channel_ops: usize,
}

/// Container for primitive verifier arithmetic trace data.
pub struct ArithmeticTraceData {
    /// Execution trace columns (ARITH_COLS_PER_ROW columns x 2^log_size rows).
    pub trace: Vec<Vec<M31>>,

    /// log2 of padded trace height.
    pub log_size: u32,

    /// Number of real (non-padding) arithmetic rows.
    pub n_real_rows: usize,
}

/// Container for locally constrained sumcheck verifier rows.
pub struct SumcheckTraceData {
    /// Execution trace columns (SUMCHECK_COLS_PER_ROW columns x 2^log_size rows).
    pub trace: Vec<Vec<M31>>,

    /// log2 of padded trace height.
    pub log_size: u32,

    /// Number of real (non-padding) sumcheck rows.
    pub n_real_rows: usize,
}

/// Container for recorded channel draw trace data.
pub struct DrawTraceData {
    /// Execution trace columns (DRAW_COLS_PER_ROW columns x 2^log_size rows).
    pub trace: Vec<Vec<M31>>,

    /// log2 of padded trace height.
    pub log_size: u32,

    /// Number of real (non-padding) draw rows.
    pub n_real_rows: usize,
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::types::{GkrVerifierWitness, WitnessOp};
    use stwo::core::fields::qm31::SecureField;

    #[test]
    fn test_cols_per_row() {
        assert_eq!(LIMBS_PER_FELT, 9);
        assert_eq!(COLS_PER_DIGEST, 9);
        assert_eq!(COLS_PER_STATE, 27);
        // 9 + 9 + 9 + 9 (addition) + 8 (carry_pos) + 8 (carry_neg) + 1 (k) + 6 selectors/counts = 59
        assert_eq!(COLS_PER_ROW, 59);
        assert_eq!(ARITH_COLS_PER_ROW, 30);
        assert_eq!(SUMCHECK_COLS_PER_ROW, 54);
        assert_eq!(DRAW_COLS_PER_ROW, 137);
    }

    #[test]
    fn test_build_arithmetic_trace_records_add_mul_and_equality_rows() {
        use stwo::core::fields::qm31::SecureField;

        let a =
            SecureField::from_m31_array([M31::from(1), M31::from(2), M31::from(3), M31::from(4)]);
        let b =
            SecureField::from_m31_array([M31::from(5), M31::from(6), M31::from(7), M31::from(8)]);
        let sum = a + b;
        let product = a * b;

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::QM31Add { a, b, result: sum },
                WitnessOp::QM31Mul {
                    a,
                    b,
                    result: product,
                },
                WitnessOp::EqualityCheck {
                    lhs: product,
                    rhs: product,
                },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: SecureField::default(),
                io_commitment: SecureField::default(),
                hades_commitment: FieldElement::ZERO,
                weight_super_root: SecureField::default(),
                n_layers: 1,
                n_poseidon_perms: 0,
                seed_digest: SecureField::default(),
                kv_cache_commitment: FieldElement::ZERO,
                prev_kv_cache_commitment: FieldElement::ZERO,
                conversation_statement_hash: FieldElement::ZERO,
            },
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 2,
            final_digest: FieldElement::ZERO,
            n_equality_checks: 1,
        };

        let trace = build_arithmetic_trace(&witness);
        assert_eq!(trace.n_real_rows, 3);
        assert_eq!(trace.trace.len(), ARITH_COLS_PER_ROW);
        assert_eq!(trace.trace[ARITH_COL_IS_ADD][0], M31::from(1));
        assert_eq!(trace.trace[ARITH_COL_IS_MUL][1], M31::from(1));
        assert_eq!(trace.trace[ARITH_COL_IS_EQ][2], M31::from(1));
        assert_eq!(trace.trace[ARITH_COL_IS_ACTIVE][0], M31::from(1));
        assert_eq!(trace.trace[ARITH_COL_IS_ACTIVE][3], M31::from(0));
        assert_eq!(trace.trace[ARITH_COL_A][0], a.to_m31_array()[0]);
        assert_eq!(
            trace.trace[ARITH_COL_RESULT + 2][1],
            product.to_m31_array()[2]
        );
        assert_eq!(trace.trace[ARITH_COL_MUL_A][0], M31::from(0));
        assert_eq!(trace.trace[ARITH_COL_MUL_A][1], a.to_m31_array()[0]);
        assert_eq!(
            trace.trace[ARITH_COL_MUL_RESULT + 3][1],
            product.to_m31_array()[3]
        );

        let aa = a.to_m31_array();
        let bb = b.to_m31_array();
        let cmul = |x_re: M31, x_im: M31, y_re: M31, y_im: M31| -> (M31, M31) {
            (x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re)
        };
        let (x0y0_re, x0y0_im) = cmul(aa[0], aa[1], bb[0], bb[1]);
        let (x1y1_re, x1y1_im) = cmul(aa[2], aa[3], bb[2], bb[3]);
        let (x0y1_re, x0y1_im) = cmul(aa[0], aa[1], bb[2], bb[3]);
        let (x1y0_re, x1y0_im) = cmul(aa[2], aa[3], bb[0], bb[1]);
        let expected = [
            x0y0_re + M31::from(2) * x1y1_re - x1y1_im,
            x0y0_im + x1y1_re + M31::from(2) * x1y1_im,
            x0y1_re + x1y0_re,
            x0y1_im + x1y0_im,
        ];
        assert_eq!(expected, product.to_m31_array());
    }

    #[test]
    fn test_build_sumcheck_trace_records_deg2_and_deg3_rows() {
        use crate::components::matmul::RoundPoly;
        use crate::gkr::types::RoundPolyDeg3;
        use stwo::core::fields::qm31::SecureField;

        let r =
            SecureField::from_m31_array([M31::from(2), M31::from(0), M31::from(0), M31::from(0)]);
        let deg2 = RoundPoly {
            c0: SecureField::from_m31_array([
                M31::from(1),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c1: SecureField::from_m31_array([
                M31::from(3),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c2: SecureField::from_m31_array([
                M31::from(5),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
        };
        let deg2_claim = deg2.c0 + (deg2.c0 + deg2.c1 + deg2.c2);
        let deg2_next = deg2.c0 + deg2.c1 * r + deg2.c2 * r * r;

        let deg3 = RoundPolyDeg3 {
            c0: SecureField::from_m31_array([
                M31::from(2),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c1: SecureField::from_m31_array([
                M31::from(4),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c2: SecureField::from_m31_array([
                M31::from(6),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c3: SecureField::from_m31_array([
                M31::from(8),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
        };
        let deg3_claim = deg3.c0 + (deg3.c0 + deg3.c1 + deg3.c2 + deg3.c3);
        let deg3_next = deg3.eval(r);

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::SumcheckRoundDeg2 {
                    round_poly: deg2,
                    claim: deg2_claim,
                    challenge: r,
                    next_claim: deg2_next,
                },
                WitnessOp::SumcheckRoundDeg3 {
                    round_poly: deg3,
                    claim: deg3_claim,
                    challenge: r,
                    next_claim: deg3_next,
                },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: SecureField::default(),
                io_commitment: SecureField::default(),
                hades_commitment: FieldElement::ZERO,
                weight_super_root: SecureField::default(),
                n_layers: 1,
                n_poseidon_perms: 0,
                seed_digest: SecureField::default(),
                kv_cache_commitment: FieldElement::ZERO,
                prev_kv_cache_commitment: FieldElement::ZERO,
                conversation_statement_hash: FieldElement::ZERO,
            },
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 2,
            n_qm31_ops: 0,
            final_digest: FieldElement::ZERO,
            n_equality_checks: 0,
        };

        let trace = build_sumcheck_trace(&witness);
        assert_eq!(trace.n_real_rows, 2);
        assert_eq!(trace.trace.len(), SUMCHECK_COLS_PER_ROW);
        assert_eq!(trace.trace[SUMCHECK_COL_IS_DEG2][0], M31::from(1));
        assert_eq!(trace.trace[SUMCHECK_COL_IS_DEG3][1], M31::from(1));
        assert_eq!(trace.trace[SUMCHECK_COL_C3][0], M31::from(0));
        assert_eq!(
            trace.trace[SUMCHECK_COL_NEXT_CLAIM][0],
            deg2_next.to_m31_array()[0]
        );
        assert_eq!(
            trace.trace[SUMCHECK_COL_C3R3][1],
            (deg3.c3 * r * r * r).to_m31_array()[0]
        );
    }

    fn assert_sumcheck_trace_locally_valid(trace: &SumcheckTraceData, expected_rows: usize) {
        assert_eq!(trace.n_real_rows, expected_rows);
        for row in 0..trace.trace[0].len() {
            let is_deg2 = trace.trace[SUMCHECK_COL_IS_DEG2][row];
            let is_deg3 = trace.trace[SUMCHECK_COL_IS_DEG3][row];
            let is_active = trace.trace[SUMCHECK_COL_IS_ACTIVE][row];
            assert!(
                (is_deg2 == M31::from(0) || is_deg2 == M31::from(1))
                    && (is_deg3 == M31::from(0) || is_deg3 == M31::from(1))
                    && (is_active == M31::from(0) || is_active == M31::from(1)),
                "sumcheck selectors must be boolean at row {row}"
            );
            assert_eq!(
                is_deg2 + is_deg3,
                is_active,
                "sumcheck row kind must match active flag at row {row}"
            );

            let read_qm31 = |base: usize| -> SecureField {
                SecureField::from_m31_array([
                    trace.trace[base][row],
                    trace.trace[base + 1][row],
                    trace.trace[base + 2][row],
                    trace.trace[base + 3][row],
                ])
            };
            let c0 = read_qm31(SUMCHECK_COL_C0);
            let c1 = read_qm31(SUMCHECK_COL_C1);
            let c2 = read_qm31(SUMCHECK_COL_C2);
            let c3 = read_qm31(SUMCHECK_COL_C3);
            let claim = read_qm31(SUMCHECK_COL_CLAIM);
            let challenge = read_qm31(SUMCHECK_COL_CHALLENGE);
            let next_claim = read_qm31(SUMCHECK_COL_NEXT_CLAIM);
            let r2 = read_qm31(SUMCHECK_COL_R2);
            let r3 = read_qm31(SUMCHECK_COL_R3);
            let c1r = read_qm31(SUMCHECK_COL_C1R);
            let c2r2 = read_qm31(SUMCHECK_COL_C2R2);
            let c3r3 = read_qm31(SUMCHECK_COL_C3R3);

            if is_deg2 == M31::from(1) {
                assert_eq!(
                    c3,
                    SecureField::default(),
                    "degree-2 sumcheck row must not contain a cubic term at row {row}"
                );
            }
            assert_eq!(
                c0 + c0 + c1 + c2 + c3,
                claim,
                "sumcheck p(0)+p(1)=claim failed at row {row}"
            );
            assert_eq!(
                challenge * challenge,
                r2,
                "sumcheck r^2 witness failed at row {row}"
            );
            assert_eq!(
                r2 * challenge,
                r3,
                "sumcheck r^3 witness failed at row {row}"
            );
            assert_eq!(
                c1 * challenge,
                c1r,
                "sumcheck c1*r witness failed at row {row}"
            );
            assert_eq!(c2 * r2, c2r2, "sumcheck c2*r^2 witness failed at row {row}");
            assert_eq!(c3 * r3, c3r3, "sumcheck c3*r^3 witness failed at row {row}");
            assert_eq!(
                c0 + c1r + c2r2 + c3r3,
                next_claim,
                "sumcheck p(r)=next_claim failed at row {row}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "sumcheck p(r)=next_claim failed")]
    fn test_tampered_sumcheck_next_claim_fails_local_constraints() {
        use crate::components::matmul::RoundPoly;
        use stwo::core::fields::qm31::SecureField;

        let r =
            SecureField::from_m31_array([M31::from(2), M31::from(0), M31::from(0), M31::from(0)]);
        let round_poly = RoundPoly {
            c0: SecureField::from_m31_array([
                M31::from(1),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c1: SecureField::from_m31_array([
                M31::from(3),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
            c2: SecureField::from_m31_array([
                M31::from(5),
                M31::from(0),
                M31::from(0),
                M31::from(0),
            ]),
        };
        let claim = round_poly.c0 + (round_poly.c0 + round_poly.c1 + round_poly.c2);
        let next_claim = round_poly.c0 + round_poly.c1 * r + round_poly.c2 * r * r;
        let witness = GkrVerifierWitness {
            ops: vec![WitnessOp::SumcheckRoundDeg2 {
                round_poly,
                claim,
                challenge: r,
                next_claim,
            }],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: SecureField::default(),
                io_commitment: SecureField::default(),
                hades_commitment: FieldElement::ZERO,
                weight_super_root: SecureField::default(),
                n_layers: 1,
                n_poseidon_perms: 0,
                seed_digest: SecureField::default(),
                kv_cache_commitment: FieldElement::ZERO,
                prev_kv_cache_commitment: FieldElement::ZERO,
                conversation_statement_hash: FieldElement::ZERO,
            },
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 1,
            n_qm31_ops: 0,
            final_digest: FieldElement::ZERO,
            n_equality_checks: 0,
        };

        let mut trace = build_sumcheck_trace(&witness);
        assert_sumcheck_trace_locally_valid(&trace, 1);
        trace.trace[SUMCHECK_COL_NEXT_CLAIM][0] += M31::from(1);

        assert_sumcheck_trace_locally_valid(&trace, 1);
    }

    #[test]
    fn test_draw_trace_binds_raw_hades_output_and_unpacking() {
        let mut channel = crate::recursive::witness::InstrumentedChannel::new();
        channel.mix_felt(FieldElement::from(42u64));
        let draw0 = channel.draw_qm31();
        let draw1 = channel.draw_qm31();
        assert_ne!(draw0, draw1, "test requires distinct draw rows");

        let final_digest = channel.inner().digest();
        let witness = GkrVerifierWitness {
            ops: channel.into_ops(),
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: SecureField::default(),
                io_commitment: SecureField::default(),
                hades_commitment: FieldElement::ZERO,
                weight_super_root: SecureField::default(),
                n_layers: 1,
                n_poseidon_perms: 3,
                seed_digest: SecureField::default(),
                kv_cache_commitment: FieldElement::ZERO,
                prev_kv_cache_commitment: FieldElement::ZERO,
                conversation_statement_hash: FieldElement::ZERO,
            },
            n_poseidon_perms: 3,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest,
            n_equality_checks: 0,
        };

        let chain = build_recursive_trace(&witness);
        let draw = build_draw_trace(&witness);
        assert_eq!(draw.n_real_rows, 2);
        let last_chain_row = chain.n_real_rows - 1;
        assert_eq!(
            chain.execution_trace[CHAIN_COL_DRAW_COUNT][last_chain_row]
                + chain.execution_trace[CHAIN_COL_IS_DRAW][last_chain_row],
            M31::from(draw.n_real_rows as u32),
            "final chain draw count must account for a transcript ending in a draw"
        );

        let chain_draw_keys: Vec<([M31; LIMBS_PER_FELT], M31)> = (0..chain.n_real_rows)
            .filter(|&row| chain.execution_trace[CHAIN_COL_IS_DRAW][row] == M31::from(1))
            .map(|row| {
                (
                    std::array::from_fn(|j| chain.execution_trace[CHAIN_COL_DIGEST_AFTER + j][row]),
                    chain.execution_trace[CHAIN_COL_DRAW_COUNT][row],
                )
            })
            .collect();
        assert_eq!(chain_draw_keys.len(), 2);

        for row in 0..draw.n_real_rows {
            let raw: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| draw.trace[DRAW_COL_RAW_FELT + j][row]);
            let draw_index = draw.trace[DRAW_COL_DRAW_INDEX][row];
            assert_eq!(
                (raw, draw_index),
                chain_draw_keys[row],
                "draw raw felt must be the matching Hades output at row {row}"
            );

            let lo0 = raw[1].0 & 0b111;
            let hi0 = raw[1].0 >> 3;
            let lo1 = raw[2].0 & 0b11_1111;
            let hi1 = raw[2].0 >> 6;
            let lo2 = raw[3].0 & 0b1_1111_1111;
            let hi2 = raw[3].0 >> 9;
            let lo3 = raw[4].0 & 0b1111_1111_1111;
            let expected = [
                M31::from(raw[0].0 + (lo0 << 28)),
                M31::from(hi0 + (lo1 << 25)),
                M31::from(hi1 + (lo2 << 22)),
                M31::from(hi2 + (lo3 << 19)),
            ];
            for (j, expected_limb) in expected.iter().enumerate() {
                assert_eq!(
                    draw.trace[DRAW_COL_VALUE + j][row],
                    *expected_limb,
                    "draw QM31 limb {j} must unpack from raw Hades felt at row {row}"
                );
            }
        }

        let mut swapped = draw.trace.clone();
        for j in 0..DRAW_RAW_FELT_LIMBS {
            swapped[DRAW_COL_RAW_FELT + j].swap(0, 1);
        }
        let swapped_raw0: [M31; LIMBS_PER_FELT] =
            std::array::from_fn(|j| swapped[DRAW_COL_RAW_FELT + j][0]);
        let swapped_lo0 = swapped_raw0[1].0 & 0b111;
        let swapped_expected0 = M31::from(swapped_raw0[0].0 + (swapped_lo0 << 28));
        assert_ne!(
            swapped[DRAW_COL_VALUE][0], swapped_expected0,
            "swapping Hades draw raw felts independently of draw values must break unpacking"
        );
    }

    #[test]
    fn test_felt252_to_limbs_zero() {
        let limbs = felt252_to_limbs(&FieldElement::ZERO);
        for l in &limbs {
            assert_eq!(*l, M31::from_u32_unchecked(0));
        }
    }

    #[test]
    fn test_felt252_to_limbs_one() {
        let limbs = felt252_to_limbs(&FieldElement::ONE);
        assert_eq!(limbs[0], M31::from_u32_unchecked(1));
        for l in &limbs[1..] {
            assert_eq!(*l, M31::from_u32_unchecked(0));
        }
    }

    #[test]
    fn test_felt252_to_limbs_small() {
        let felt = FieldElement::from(0xDEADBEEFu64);
        let limbs = felt252_to_limbs(&felt);
        // First limb holds lowest 28 bits of 0xDEADBEEF = 0xEADBEEF
        assert_eq!(limbs[0], M31::from_u32_unchecked(0xEADBEEF));
        // Second limb holds next 4 bits = 0xD
        assert_eq!(limbs[1], M31::from_u32_unchecked(0xD));
    }

    #[test]
    fn test_hades_logup_provider_keys_match_chain_keys_and_detect_cross_swap() {
        let input0 = [
            FieldElement::ZERO,
            FieldElement::from(11u64),
            FieldElement::from(2u64),
        ];
        let mut output0 = input0;
        crate::crypto::hades::hades_permutation(&mut output0);

        let input1 = [
            output0[0],
            FieldElement::from(22u64),
            FieldElement::from(2u64),
        ];
        let mut output1 = input1;
        crate::crypto::hades::hades_permutation(&mut output1);

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::HadesPerm {
                    input: input0,
                    output: output0,
                },
                WitnessOp::HadesPerm {
                    input: input1,
                    output: output1,
                },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                n_poseidon_perms: 2,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                conversation_statement_hash: starknet_ff::FieldElement::ZERO,
            },
            n_poseidon_perms: 2,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: output1[0],
            n_equality_checks: 0,
        };

        let chain = build_recursive_trace(&witness);
        let hades =
            crate::recursive::hades_air::build_hades_trace(&[(input0, output0), (input1, output1)]);

        let chain_key = |row: usize| -> Vec<M31> {
            (0..LIMBS_PER_FELT)
                .map(|j| chain.execution_trace[j][row])
                .chain(
                    (0..LIMBS_PER_FELT)
                        .map(|j| chain.execution_trace[CHAIN_COL_DIGEST_AFTER + j][row]),
                )
                .collect()
        };
        let provider_key = |perm_idx: usize| -> Vec<M31> {
            let row = perm_idx * crate::recursive::hades_air::N_ROUNDS
                + crate::recursive::hades_air::N_ROUNDS
                - 1;
            (0..9)
                .map(|j| {
                    hades.trace[crate::recursive::hades_air::HADES_INPUT_DIGEST_28BIT_COL + j][row]
                })
                .chain((0..9).map(|j| {
                    hades.trace[crate::recursive::hades_air::HADES_OUTPUT_DIGEST_28BIT_COL + j][row]
                }))
                .collect()
        };

        let chain_keys = vec![chain_key(0), chain_key(1)];
        let provider_keys = vec![provider_key(0), provider_key(1)];
        assert_eq!(provider_keys, chain_keys);

        let mut cross_swapped = provider_keys[0][0..9].to_vec();
        cross_swapped.extend_from_slice(&provider_keys[1][9..18]);
        assert!(
            !chain_keys.contains(&cross_swapped),
            "independently swapping Hades input/output rows must change the LogUp multiset"
        );
    }

    #[test]
    fn test_hades_state_to_limbs_roundtrip() {
        let state = [
            FieldElement::from(42u64),
            FieldElement::from(100u64),
            FieldElement::TWO,
        ];
        let limbs = hades_state_to_limbs(&state);
        assert_eq!(limbs.len(), COLS_PER_STATE);

        // First felt252 (42) should have limbs[0] = 42
        assert_eq!(limbs[0], M31::from_u32_unchecked(42));
        // Second felt252 (100) starts at offset 9
        assert_eq!(limbs[LIMBS_PER_FELT], M31::from_u32_unchecked(100));
        // Third felt252 (2) starts at offset 18
        assert_eq!(limbs[2 * LIMBS_PER_FELT], M31::from_u32_unchecked(2));
    }

    #[test]
    #[ignore = "stale: passes WitnessOp::ChannelOp directly to build_recursive_trace; current API requires HadesPerm (G7)"]
    fn test_trace_with_channel_op() {
        // Build a witness with ChannelOp and verify trace population.
        use crate::recursive::types::WitnessOp;

        let digest_before = FieldElement::ZERO;
        let mut state = [digest_before, FieldElement::from(42u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut state);
        let digest_after = state[0];

        let witness = GkrVerifierWitness {
            ops: vec![WitnessOp::ChannelOp {
                digest_before,
                digest_after,
            }],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                n_poseidon_perms: 1,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                conversation_statement_hash: starknet_ff::FieldElement::ZERO,
            },
            n_poseidon_perms: 1,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: digest_after,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);

        assert_eq!(trace.log_size, 1);
        assert_eq!(trace.n_real_rows, 1);
        assert_eq!(trace.n_channel_ops, 1);
        assert_eq!(trace.execution_trace.len(), COLS_PER_ROW);

        // Verify digest_before limbs (first 9 columns)
        let before_limbs = felt252_to_limbs(&digest_before);
        for j in 0..LIMBS_PER_FELT {
            assert_eq!(trace.execution_trace[j][0], before_limbs[j]);
        }

        // Verify digest_after limbs (columns 27-35 in expanded layout)
        let after_limbs = felt252_to_limbs(&digest_after);
        for j in 0..LIMBS_PER_FELT {
            assert_eq!(trace.execution_trace[COLS_PER_STATE + j][0], after_limbs[j]);
        }

        assert_eq!(trace.preprocessed_is_first[0], M31::from_u32_unchecked(1));
        assert_eq!(trace.preprocessed_is_last[0], M31::from_u32_unchecked(1));
    }

    #[test]
    #[ignore = "stale: passes WitnessOp::ChannelOp directly to build_recursive_trace; current API requires HadesPerm (G7)"]
    fn test_trace_chain_correctness() {
        // Two channel ops: verify digest_after[0] == digest_before[1].
        use crate::recursive::types::WitnessOp;

        let d0 = FieldElement::ZERO;
        let mut s1 = [d0, FieldElement::from(42u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut s1);
        let d1 = s1[0];

        let mut s2 = [d1, FieldElement::from(100u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut s2);
        let d2 = s2[0];

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::ChannelOp {
                    digest_before: d0,
                    digest_after: d1,
                },
                WitnessOp::ChannelOp {
                    digest_before: d1,
                    digest_after: d2,
                },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                n_poseidon_perms: 2,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                conversation_statement_hash: starknet_ff::FieldElement::ZERO,
            },
            n_poseidon_perms: 2,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: d2,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);

        assert_eq!(trace.n_channel_ops, 2);

        // Verify chain: digest_after[row0] == digest_before[row1]
        // digest_after starts at column COLS_PER_STATE (27) in expanded layout
        for j in 0..LIMBS_PER_FELT {
            let after_row0 = trace.execution_trace[COLS_PER_STATE + j][0];
            let before_row1 = trace.execution_trace[j][1];
            assert_eq!(
                after_row0, before_row1,
                "chain broken at limb {j}: digest_after[0] != digest_before[1]"
            );
        }
    }

    #[test]
    #[ignore = "stale: indexes columns at 2*COLS_PER_STATE assuming 89-col layout; current slim layout is 48 cols (G7)"]
    fn test_accumulator_constraint_satisfaction() {
        // Verify the amortized accumulator constraint holds on each row.
        use crate::recursive::types::WitnessOp;
        let d0 = FieldElement::ZERO;
        let mut s1 = [d0, FieldElement::from(42u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut s1);
        let d1 = s1[0];
        let mut s2 = [d1, FieldElement::from(100u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut s2);
        let d2 = s2[0];

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::ChannelOp {
                    digest_before: d0,
                    digest_after: d1,
                },
                WitnessOp::ChannelOp {
                    digest_before: d1,
                    digest_after: d2,
                },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                n_poseidon_perms: 2,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                conversation_statement_hash: starknet_ff::FieldElement::ZERO,
            },
            n_poseidon_perms: 2,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: d2,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);
        let n = 1usize << trace.log_size;
        let n_real = trace.n_real_rows;
        let col_is_active = 2 * COLS_PER_STATE + LIMBS_PER_FELT + 1;
        let col_active_count = col_is_active + 3;
        let col_active_count_next = col_is_active + 4;

        // Compute expected correction
        let n_m31 = M31::from(n as u32);
        let n_inv = n_m31.inverse();
        let correction = M31::from(n_real as u32) * n_inv;

        // Check C3: active_count_next - active_count - is_active + correction = 0
        for i in 0..n {
            let ac = trace.execution_trace[col_active_count][i];
            let ac_next = trace.execution_trace[col_active_count_next][i];
            let is_act = trace.execution_trace[col_is_active][i];
            let residual = ac_next - ac - is_act + correction;
            assert_eq!(
                residual,
                M31::from_u32_unchecked(0),
                "C3 accumulator constraint fails at row {i}: ac={ac:?}, ac_next={ac_next:?}, is_active={is_act:?}, correction={correction:?}"
            );
        }
        eprintln!("[test] accumulator constraint satisfied on all {} rows", n);

        // Check C1: is_active boolean
        for i in 0..n {
            let a = trace.execution_trace[col_is_active][i];
            let residual = a * (M31::from(1u32) - a);
            assert_eq!(
                residual,
                M31::from_u32_unchecked(0),
                "C1 boolean fails at row {i}"
            );
        }

        // Check C4: initial boundary (row where is_active=1 and is_active_prev=0)
        let col_is_active_prev = col_is_active + 2;
        for i in 0..n {
            let a = trace.execution_trace[col_is_active][i];
            let a_prev = trace.execution_trace[col_is_active_prev][i];
            if a == M31::from(1u32) && a_prev == M31::from_u32_unchecked(0) {
                // Initial boundary should fire here
                for j in 0..LIMBS_PER_FELT {
                    let db = trace.execution_trace[j][i];
                    assert_eq!(
                        db,
                        M31::from_u32_unchecked(0),
                        "C4 initial boundary fails at row {i} limb {j}"
                    );
                }
                eprintln!("[test] initial boundary fires at row {i}");
            }
        }

        // Check C5: final boundary (row where is_active=1 and is_active_next=0)
        let col_is_active_next = col_is_active + 1;
        for i in 0..n {
            let a = trace.execution_trace[col_is_active][i];
            let a_next = trace.execution_trace[col_is_active_next][i];
            if a == M31::from(1u32) && a_next == M31::from_u32_unchecked(0) {
                let final_limbs = felt252_to_limbs(&d2);
                for j in 0..LIMBS_PER_FELT {
                    let da = trace.execution_trace[COLS_PER_STATE + j][i];
                    assert_eq!(
                        da, final_limbs[j],
                        "C5 final boundary fails at row {i} limb {j}"
                    );
                }
                eprintln!("[test] final boundary fires at row {i}");
            }
        }
    }

    #[test]
    fn test_eval_properties() {
        let eval = RecursiveVerifierEval {
            log_n_rows: 14,
            n_real_rows: 100, // test value
            initial_digest_limbs: [M31::from_u32_unchecked(0); LIMBS_PER_FELT],
            final_digest_limbs: [M31::from_u32_unchecked(42); LIMBS_PER_FELT],
            hades_lookup: None,
            draw_felt_lookup: None,
            challenge_lookup: None,
            hades_enabled: false,
            arithmetic_enabled: false,
            n_arithmetic_rows: 0,
            sumcheck_enabled: false,
            n_sumcheck_rows: 0,
            draw_enabled: false,
            n_draw_rows: 0,
        };
        assert_eq!(eval.log_size(), 14);
        assert_eq!(eval.max_constraint_log_degree_bound(), 15); // +1 for degree-2 constraints
    }
}
