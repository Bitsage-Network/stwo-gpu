use stwo_constraint_framework::{CommonLookupElements, LookupElementsTrait};
use stwo_verifier_core::circle::CirclePoint;
use stwo_verifier_core::fields::Invertible;
use stwo_verifier_core::fields::m31::{M31, m31};
/// Cairo AIR implementation for the Recursive STARK verifier.
///
/// Verifies that a chain of Poseidon channel operations was executed correctly.
///
/// Trace layout (59 columns per row, signed carry via pos/neg split):
///   [0..9)    digest_before
///   [9..18)   digest_after
///   [18..27)  shifted_next_before
///   [27..36)  addition_digest
///   [36..44)  addition_carry_pos[8]   (positive carry indicator ∈ {0,1})
///   [44..52)  addition_carry_neg[8]   (negative carry / borrow indicator ∈ {0,1})
///   [52]      addition_k
///   [53]      is_active
///   [54]      active_count
///   [55]      active_count_next
///   [56]      is_draw
///   [57]      draw_count
///   [58]      shifted_next_draw_count
///
/// Plus 3 preprocessed selector columns (is_first, is_last, is_chain).
///
/// Signed carry at limb j = pos[j] - neg[j] ∈ {-1, 0, 1} via pos*neg=0 enforcement.
/// Negative carries (borrows) are required for Llama-class decode-mode channel ops
/// because P's high-bit limbs (P_LIMBS_28[6]=2^24, [7]=1, [8]=2^27) cause
/// limb-level integer sums to fall below the result limb at some positions.
///
/// Constraints (59 total, all degree ≤ 2):
///   C1:  is_active boolean                     [1, unconditional]
///   C1b: is_draw boolean and active implication [2, unconditional]
///   C2:  amortized accumulator                 [1, unconditional — BLOCKS all-zeros]
///   C3:  initial boundary (is_first × 9 limbs) [9]
///   C3b: initial draw_count boundary            [1]
///   C4:  final boundary (is_last × 9 limbs)    [9]
///   C4b: final draw_count boundary              [1]
///   C5k: k boolean                             [1]
///   C5pos: pos[j] boolean                      [8]
///   C5neg: neg[j] boolean                      [8]
///   C5excl: pos[j] * neg[j] = 0                [8]
///   C5:  carry-chain modular addition (9 limbs) [9]
///   C6:  shifted draw-count continuity          [1]

use stwo_verifier_core::fields::qm31::{QM31, QM31One, QM31Trait, QM31Zero};
use stwo_verifier_core::poly::circle::CanonicCosetImpl;
use stwo_verifier_core::verifier::Air;
use stwo_verifier_core::{ColumnSpan, TreeSpan};

/// Number of M31 limbs per felt252 (9 × 28 = 252 bits).
pub const LIMBS_PER_FELT: u32 = 9;

/// Total chain trace columns.
const TRACE_COLS: u32 = 59;

/// Total Hades trace columns appended after the chain trace in Hades mode.
const HADES_TRACE_COLS: u32 = 1281;

/// Primitive verifier component trace widths, matching libs/engine Rust AIR.
const ARITH_COLS: u32 = 30;
const SUMCHECK_COLS: u32 = 54;
const DRAW_COLS: u32 = 137;

/// Relative Hades columns used by the shared HadesPerm LogUp provider.
const HADES_IS_LAST_ROUND_COL: u32 = 1174;
const HADES_INPUT_DIGEST_28BIT_COL: u32 = 1231;
const HADES_OUTPUT_DIGEST_28BIT_COL: u32 = 1240;

/// Columns per full Hades state (3 felt252 × 9 limbs).
const COLS_PER_STATE: u32 = 27;

/// Number of preprocessed selector columns: is_first, is_last, is_chain.
const PREPROCESS_COLS: u32 = 3;

/// Total constraints: 59.
const N_CONSTRAINTS: u32 = 59;

/// Stark prime P in 28-bit limbs (LSB first).
/// P = 2^251 + 17*2^192 + 1
fn stark_prime_28bit_limbs() -> Array<u32> {
    array![1, 0, 0, 0, 0, 0, 16777216, 1, 134217728]
}

/// The recursive verifier AIR.
#[derive(Drop)]
pub struct RecursiveAir {
    /// log2(number of trace rows).
    pub log_n_rows: u32,
    /// Number of real (active) rows in the trace.
    pub n_real_rows: u32,
    /// Number of real channel draw rows in the trace.
    pub n_draw_rows: u32,
    /// Number of real primitive arithmetic rows in the trace.
    pub n_arithmetic_rows: u32,
    /// Number of real sumcheck rows in the trace.
    pub n_sumcheck_rows: u32,
    /// Initial digest decomposed into 9 M31 limbs (usually all zero).
    pub initial_digest_limbs: Array<QM31>,
    /// Expected final digest decomposed into 9 M31 limbs.
    pub final_digest_limbs: Array<QM31>,
    /// When true, evaluates 1281 Hades columns after the 59 chain columns.
    pub hades_enabled: bool,
    /// When true, evaluates primitive arithmetic columns.
    pub arithmetic_enabled: bool,
    /// When true, evaluates recorded sumcheck round columns.
    pub sumcheck_enabled: bool,
    /// When true, evaluates channel draw columns.
    pub draw_enabled: bool,
    /// Claimed sum for the recursive LogUp interaction trace.
    pub logup_claimed_sum: QM31,
    /// HadesPerm relation lookup elements.
    pub hades_lookup_elements: CommonLookupElements,
    /// DrawFelt relation lookup elements.
    pub draw_felt_lookup_elements: CommonLookupElements,
    /// SumcheckChallenge relation lookup elements.
    pub challenge_lookup_elements: CommonLookupElements,
}

impl RecursiveAirImpl of Air<RecursiveAir> {
    // composition_log_degree_bound removed in v1.2.2 — passed to verify() directly.

    fn eval_composition_polynomial_at_point(
        self: @RecursiveAir,
        point: CirclePoint<QM31>,
        mask_values: TreeSpan<ColumnSpan<Span<QM31>>>,
        random_coeff: QM31,
    ) -> QM31 {
        // Chain-only: 3 trees [preprocessed, trace, composition]
        // Hades mode: 4 trees [preprocessed, trace+hades, interaction, composition]
        let _n_trees = mask_values.len();
        let preprocessed_vals: ColumnSpan<Span<QM31>> = *mask_values[0];
        let trace_vals: ColumnSpan<Span<QM31>> = *mask_values[1];

        // Preprocessed selectors
        let is_first = extract_single_val(preprocessed_vals, 0);
        let is_last = extract_single_val(preprocessed_vals, 1);
        let is_chain = extract_single_val(preprocessed_vals, 2);

        // Trace columns
        let one: QM31 = QM31One::one();
        let zero: QM31 = QM31Zero::zero();

        // Column extraction helpers
        // 59-column layout (signed carry via pos/neg split):
        // digest_before [0..9)
        // digest_after [9..18)
        // shifted_next_before [18..27)
        // addition_digest [27..36)
        // addition_carry_pos [36..44)
        // addition_carry_neg [44..52)
        // addition_k [52]
        // is_active [53]
        // active_count [54]
        // active_count_next [55]
        // is_draw [56]
        // draw_count [57]
        // shifted_next_draw_count [58]

        let is_active = extract_single_val(trace_vals, 53);
        let active_count = extract_single_val(trace_vals, 54);
        let active_count_next = extract_single_val(trace_vals, 55);
        let is_draw = extract_single_val(trace_vals, 56);
        let draw_count = extract_single_val(trace_vals, 57);
        let shifted_next_draw_count = extract_single_val(trace_vals, 58);
        let addition_k = extract_single_val(trace_vals, 52);

        // NOTE: Do NOT divide by vanishing polynomial here.
        // verify() multiplies the result by denominator_inv (vanishing^{-1}) externally.
        // Dividing here would double-apply the inverse.

        let mut quotients: Array<QM31> = array![];

        // ═══════════════════════════════════════════════════════════
        // C1: is_active boolean [unconditional]
        // ═══════════════════════════════════════════════════════════
        quotients.append(is_active * (one - is_active));
        quotients.append(is_draw * (one - is_draw));
        quotients.append(is_draw * (one - is_active));

        // ═══════════════════════════════════════════════════════════
        // C2: amortized accumulator [unconditional — BLOCKS all-zeros]
        // active_count_next - active_count - is_active + n_real * N_inv = 0
        // ═══════════════════════════════════════════════════════════
        // Compute N = 2^log_n_rows and N_inv in the M31 field
        let n_val: u64 = pow2(*self.log_n_rows);
        let n_m31: M31 = m31(n_val.try_into().unwrap());
        let n_inv_m31: M31 = n_m31.inverse();
        let correction_m31: M31 = m31(*self.n_real_rows) * n_inv_m31;
        let correction: QM31 = m31_to_qm31(correction_m31);
        quotients.append((active_count_next - active_count - is_active + correction));

        // ═══════════════════════════════════════════════════════════
        // C3: Initial boundary — is_first × (digest_before - initial) [9]
        // ═══════════════════════════════════════════════════════════
        let mut j: u32 = 0;
        loop {
            if j >= LIMBS_PER_FELT {
                break;
            }
            let db = extract_single_val(trace_vals, j);
            let init = *self.initial_digest_limbs.at(j);
            quotients.append(is_first * (db - init));
            j += 1;
        }
        quotients.append(is_first * draw_count);

        // ═══════════════════════════════════════════════════════════
        // C4: Final boundary — is_last × (digest_after - final) [9]
        // ═══════════════════════════════════════════════════════════
        j = 0;
        loop {
            if j >= LIMBS_PER_FELT {
                break;
            }
            let da = extract_single_val(trace_vals, 9 + j); // digest_after
            let fin = *self.final_digest_limbs.at(j);
            quotients.append(is_last * (da - fin));
            j += 1;
        }
        quotients.append(is_last * (draw_count + is_draw - m31_to_qm31(m31(*self.n_draw_rows))));

        // ═══════════════════════════════════════════════════════════
        // C5k: k boolean — is_chain × k × (k - 1)
        // ═══════════════════════════════════════════════════════════
        quotients.append(is_chain * addition_k * (addition_k - one));

        // ═══════════════════════════════════════════════════════════
        // C5pos: pos[j] boolean — is_chain × pos[j] × (pos[j] - 1) [8]
        // C5neg: neg[j] boolean — is_chain × neg[j] × (neg[j] - 1) [8]
        // C5excl: pos[j] * neg[j] = 0 — is_chain × pos[j] × neg[j]   [8]
        // Net signed carry at limb j = pos[j] - neg[j] ∈ {-1, 0, 1}.
        // ═══════════════════════════════════════════════════════════
        j = 0;
        loop {
            if j >= 8 {
                break;
            }
            let pos_j = extract_single_val(trace_vals, 36 + j); // addition_carry_pos
            let neg_j = extract_single_val(trace_vals, 44 + j); // addition_carry_neg
            quotients.append(is_chain * pos_j * (pos_j - one));
            quotients.append(is_chain * neg_j * (neg_j - one));
            quotients.append(is_chain * pos_j * neg_j);
            j += 1;
        }

        // ═══════════════════════════════════════════════════════════
        // C5: Carry-chain modular addition [9 limbs]
        // da[j] + add[j] + (pos[j-1] - neg[j-1])
        //   - snb[j] - k*P[j] - (pos[j] - neg[j])*2^28 = 0
        // ═══════════════════════════════════════════════════════════
        let p_limbs = stark_prime_28bit_limbs();
        let two_pow_28: QM31 = m31_to_qm31(m31(268435456)); // 2^28

        j = 0;
        loop {
            if j >= LIMBS_PER_FELT {
                break;
            }
            let da = extract_single_val(trace_vals, 9 + j); // digest_after
            let add = extract_single_val(trace_vals, 27 + j); // addition_digest
            let snb = extract_single_val(trace_vals, 18 + j); // shifted_next_before
            let p_j: QM31 = m31_to_qm31(m31(*p_limbs.at(j)));

            let carry_in: QM31 = if j == 0 {
                zero
            } else {
                let pos_prev = extract_single_val(trace_vals, 36 + j - 1);
                let neg_prev = extract_single_val(trace_vals, 44 + j - 1);
                pos_prev - neg_prev
            };

            let carry_out_term: QM31 = if j < 8 {
                let pos_j = extract_single_val(trace_vals, 36 + j);
                let neg_j = extract_single_val(trace_vals, 44 + j);
                (pos_j - neg_j) * two_pow_28
            } else {
                zero
            };

            quotients
                .append(is_chain * (da + add + carry_in - snb - addition_k * p_j - carry_out_term));
            j += 1;
        }
        quotients.append(is_chain * (shifted_next_draw_count - draw_count - is_draw));

        // If Hades enabled, append Hades constraint quotients to the same array.
        // Hades columns start at offset 59 (after chain columns).
        let mut component_offset = TRACE_COLS;
        let hades_offset = component_offset;
        if *self.hades_enabled {
            let hades_quotients = crate::recursive_hades_air::evaluate_hades_constraints_array(
                trace_vals, TRACE_COLS,
            );
            let mut hi: u32 = 0;
            loop {
                if hi >= hades_quotients.len() {
                    break;
                }
                quotients.append(*hades_quotients.at(hi));
                hi += 1;
            }
            component_offset += HADES_TRACE_COLS;
        }

        // Primitive verifier arithmetic rows. These locally constrain recorded
        // QM31 addition/multiplication/equality facts; they do not by themselves
        // prove full verifier control flow.
        let mut arithmetic_offset: u32 = 0;
        if *self.arithmetic_enabled {
            arithmetic_offset = component_offset;
            let arith_a: u32 = component_offset;
            let arith_b: u32 = component_offset + 4;
            let arith_result: u32 = component_offset + 8;
            let arith_mul_a: u32 = component_offset + 12;
            let arith_mul_b: u32 = component_offset + 16;
            let arith_mul_result: u32 = component_offset + 20;
            let arith_is_add = extract_single_val(trace_vals, component_offset + 24);
            let arith_is_mul = extract_single_val(trace_vals, component_offset + 25);
            let arith_is_eq = extract_single_val(trace_vals, component_offset + 26);
            let arith_is_active = extract_single_val(trace_vals, component_offset + 27);
            let arith_active_count = extract_single_val(trace_vals, component_offset + 28);
            let arith_active_count_next = extract_single_val(trace_vals, component_offset + 29);

            quotients.append(arith_is_add * (one - arith_is_add));
            quotients.append(arith_is_mul * (one - arith_is_mul));
            quotients.append(arith_is_eq * (one - arith_is_eq));
            quotients.append(arith_is_active * (one - arith_is_active));
            quotients.append(arith_is_add + arith_is_mul + arith_is_eq - arith_is_active);

            let correction_m31: M31 = m31(*self.n_arithmetic_rows) * n_inv_m31;
            let correction: QM31 = m31_to_qm31(correction_m31);
            quotients
                .append(
                    arith_active_count_next - arith_active_count - arith_is_active + correction,
                );

            j = 0;
            loop {
                if j >= 4 {
                    break;
                }
                let a_j = extract_single_val(trace_vals, arith_a + j);
                let b_j = extract_single_val(trace_vals, arith_b + j);
                let result_j = extract_single_val(trace_vals, arith_result + j);
                let mul_a_j = extract_single_val(trace_vals, arith_mul_a + j);
                let mul_b_j = extract_single_val(trace_vals, arith_mul_b + j);
                let mul_result_j = extract_single_val(trace_vals, arith_mul_result + j);
                quotients.append(arith_is_eq * (a_j - b_j));
                quotients.append(arith_is_add * (a_j + b_j - result_j));
                quotients.append(mul_a_j - arith_is_mul * a_j);
                quotients.append(mul_b_j - arith_is_mul * b_j);
                quotients.append(mul_result_j - arith_is_mul * result_j);
                j += 1;
            }

            let (expected0, expected1, expected2, expected3) = qm31_mul_at_offsets(
                trace_vals, arith_mul_a, arith_mul_b,
            );
            quotients.append(extract_single_val(trace_vals, arith_mul_result) - expected0);
            quotients.append(extract_single_val(trace_vals, arith_mul_result + 1) - expected1);
            quotients.append(extract_single_val(trace_vals, arith_mul_result + 2) - expected2);
            quotients.append(extract_single_val(trace_vals, arith_mul_result + 3) - expected3);

            component_offset += ARITH_COLS;
        }

        // Recorded sumcheck round rows. This constrains p(0)+p(1)=claim and
        // p(challenge)=next_claim for each row, without claiming full verifier
        // control-flow execution.
        let mut sumcheck_offset: u32 = 0;
        if *self.sumcheck_enabled {
            sumcheck_offset = component_offset;
            let sc_c0: u32 = component_offset;
            let sc_c1: u32 = component_offset + 4;
            let sc_c2: u32 = component_offset + 8;
            let sc_c3: u32 = component_offset + 12;
            let sc_claim: u32 = component_offset + 16;
            let sc_challenge: u32 = component_offset + 20;
            let _sc_challenge_draw_index = extract_single_val(trace_vals, component_offset + 24);
            let sc_next_claim: u32 = component_offset + 25;
            let sc_r2: u32 = component_offset + 29;
            let sc_r3: u32 = component_offset + 33;
            let sc_c1r: u32 = component_offset + 37;
            let sc_c2r2: u32 = component_offset + 41;
            let sc_c3r3: u32 = component_offset + 45;
            let sc_is_deg2 = extract_single_val(trace_vals, component_offset + 49);
            let sc_is_deg3 = extract_single_val(trace_vals, component_offset + 50);
            let sc_is_active = extract_single_val(trace_vals, component_offset + 51);
            let sc_active_count = extract_single_val(trace_vals, component_offset + 52);
            let sc_active_count_next = extract_single_val(trace_vals, component_offset + 53);
            let two = m31_to_qm31(m31(2));

            quotients.append(sc_is_deg2 * (one - sc_is_deg2));
            quotients.append(sc_is_deg3 * (one - sc_is_deg3));
            quotients.append(sc_is_active * (one - sc_is_active));
            quotients.append(sc_is_deg2 + sc_is_deg3 - sc_is_active);

            let correction_m31: M31 = m31(*self.n_sumcheck_rows) * n_inv_m31;
            let correction: QM31 = m31_to_qm31(correction_m31);
            quotients.append(sc_active_count_next - sc_active_count - sc_is_active + correction);

            j = 0;
            loop {
                if j >= 4 {
                    break;
                }
                let c0_j = extract_single_val(trace_vals, sc_c0 + j);
                let c1_j = extract_single_val(trace_vals, sc_c1 + j);
                let c2_j = extract_single_val(trace_vals, sc_c2 + j);
                let c3_j = extract_single_val(trace_vals, sc_c3 + j);
                let claim_j = extract_single_val(trace_vals, sc_claim + j);
                quotients.append(sc_is_deg2 * c3_j);
                quotients.append(two * c0_j + c1_j + c2_j + c3_j - claim_j);
                j += 1;
            }

            append_qm31_mul_quotients(ref quotients, trace_vals, sc_challenge, sc_challenge, sc_r2);
            append_qm31_mul_quotients(ref quotients, trace_vals, sc_r2, sc_challenge, sc_r3);
            append_qm31_mul_quotients(ref quotients, trace_vals, sc_c1, sc_challenge, sc_c1r);
            append_qm31_mul_quotients(ref quotients, trace_vals, sc_c2, sc_r2, sc_c2r2);
            append_qm31_mul_quotients(ref quotients, trace_vals, sc_c3, sc_r3, sc_c3r3);

            j = 0;
            loop {
                if j >= 4 {
                    break;
                }
                quotients
                    .append(
                        extract_single_val(trace_vals, sc_c0 + j)
                            + extract_single_val(trace_vals, sc_c1r + j)
                            + extract_single_val(trace_vals, sc_c2r2 + j)
                            + extract_single_val(trace_vals, sc_c3r3 + j)
                            - extract_single_val(trace_vals, sc_next_claim + j),
                    );
                j += 1;
            }

            component_offset += SUMCHECK_COLS;
        }

        // Recorded channel draw rows. This constrains raw felt decomposition
        // into the QM31 challenge value. The LogUp consumer/provider relation is
        // deliberately still gated in recursive_verifier.cairo.
        let mut draw_offset: u32 = 0;
        if *self.draw_enabled {
            draw_offset = component_offset;
            let draw_value: u32 = component_offset;
            let _draw_index = extract_single_val(trace_vals, component_offset + 4);
            let draw_raw_felt: u32 = component_offset + 5;
            let draw_split_lo: u32 = component_offset + 14;
            let draw_split_hi: u32 = component_offset + 18;
            let draw_split_bits: u32 = component_offset + 22;
            let draw_is_active = extract_single_val(trace_vals, component_offset + 134);
            let draw_active_count = extract_single_val(trace_vals, component_offset + 135);
            let draw_active_count_next = extract_single_val(trace_vals, component_offset + 136);

            quotients.append(draw_is_active * (one - draw_is_active));
            let correction_m31: M31 = m31(*self.n_draw_rows) * n_inv_m31;
            let correction: QM31 = m31_to_qm31(correction_m31);
            quotients
                .append(draw_active_count_next - draw_active_count - draw_is_active + correction);

            j = 0;
            loop {
                if j >= 112 {
                    break;
                }
                let bit = extract_single_val(trace_vals, draw_split_bits + j);
                quotients.append(bit * (one - bit));
                j += 1;
            }

            let split_lo_bits = array![3, 6, 9, 12];
            let mut split_idx: u32 = 0;
            let mut bit_offset: u32 = 0;
            loop {
                if split_idx >= 4 {
                    break;
                }
                let lo_bits = *split_lo_bits.at(split_idx);
                let hi_bits = 28 - lo_bits;
                let mut lo_acc: QM31 = zero;
                let mut bit: u32 = 0;
                loop {
                    if bit >= lo_bits {
                        break;
                    }
                    lo_acc += extract_single_val(trace_vals, draw_split_bits + bit_offset + bit)
                        * qm31_pow2(bit);
                    bit += 1;
                }
                bit_offset += lo_bits;

                let mut hi_acc: QM31 = zero;
                bit = 0;
                loop {
                    if bit >= hi_bits {
                        break;
                    }
                    hi_acc += extract_single_val(trace_vals, draw_split_bits + bit_offset + bit)
                        * qm31_pow2(bit);
                    bit += 1;
                }
                bit_offset += hi_bits;

                quotients
                    .append(extract_single_val(trace_vals, draw_split_lo + split_idx) - lo_acc);
                quotients
                    .append(extract_single_val(trace_vals, draw_split_hi + split_idx) - hi_acc);
                quotients
                    .append(
                        extract_single_val(trace_vals, draw_raw_felt + split_idx + 1)
                            - extract_single_val(trace_vals, draw_split_lo + split_idx)
                            - extract_single_val(trace_vals, draw_split_hi + split_idx)
                                * qm31_pow2(lo_bits),
                    );
                split_idx += 1;
            }

            let expected0 = extract_single_val(trace_vals, draw_raw_felt)
                + extract_single_val(trace_vals, draw_split_lo) * qm31_pow2(28);
            let expected1 = extract_single_val(trace_vals, draw_split_hi)
                + extract_single_val(trace_vals, draw_split_lo + 1) * qm31_pow2(25);
            let expected2 = extract_single_val(trace_vals, draw_split_hi + 1)
                + extract_single_val(trace_vals, draw_split_lo + 2) * qm31_pow2(22);
            let expected3 = extract_single_val(trace_vals, draw_split_hi + 2)
                + extract_single_val(trace_vals, draw_split_lo + 3) * qm31_pow2(19);
            quotients.append(extract_single_val(trace_vals, draw_value) - expected0);
            quotients.append(extract_single_val(trace_vals, draw_value + 1) - expected1);
            quotients.append(extract_single_val(trace_vals, draw_value + 2) - expected2);
            quotients.append(extract_single_val(trace_vals, draw_value + 3) - expected3);
        }

        // Recursive LogUp interaction constraints. This mirrors Rust
        // FrameworkEval::finalize_logup() with unbatched relation entries:
        // each relation entry gets one QM31 cumulative-sum interaction column.
        if *self.hades_enabled {
            let interaction_vals: ColumnSpan<Span<QM31>> = *mask_values[2];
            let mut interaction_col: u32 = 0;
            let mut prev_col_cumsum: QM31 = zero;

            // 1. Hades provider: -1 on Hades last-round rows.
            let mut hades_key: Array<QM31> = array![];
            j = 0;
            loop {
                if j >= LIMBS_PER_FELT {
                    break;
                }
                hades_key
                    .append(
                        extract_single_val(
                            trace_vals, hades_offset + HADES_INPUT_DIGEST_28BIT_COL + j,
                        ),
                    );
                j += 1;
            }
            j = 0;
            loop {
                if j >= LIMBS_PER_FELT {
                    break;
                }
                hades_key
                    .append(
                        extract_single_val(
                            trace_vals, hades_offset + HADES_OUTPUT_DIGEST_28BIT_COL + j,
                        ),
                    );
                j += 1;
            }
            append_logup_intermediate_constraint(
                ref quotients,
                interaction_vals,
                ref interaction_col,
                ref prev_col_cumsum,
                self.hades_lookup_elements.combine_qm31(hades_key.span()),
                zero - extract_single_val(trace_vals, hades_offset + HADES_IS_LAST_ROUND_COL),
            );

            if *self.sumcheck_enabled && *self.draw_enabled {
                // 2. Sumcheck rows consume recorded QM31 draw challenges (+1).
                let sc_challenge: u32 = sumcheck_offset + 20;
                let mut challenge_key: Array<QM31> = array![];
                j = 0;
                loop {
                    if j >= 4 {
                        break;
                    }
                    challenge_key.append(extract_single_val(trace_vals, sc_challenge + j));
                    j += 1;
                }
                challenge_key.append(extract_single_val(trace_vals, sumcheck_offset + 24));
                append_logup_intermediate_constraint(
                    ref quotients,
                    interaction_vals,
                    ref interaction_col,
                    ref prev_col_cumsum,
                    self.challenge_lookup_elements.combine_qm31(challenge_key.span()),
                    extract_single_val(trace_vals, sumcheck_offset + 51),
                );
            }

            if *self.draw_enabled {
                // 3. Draw rows consume raw Hades output felts (+1).
                let mut draw_raw_key: Array<QM31> = array![];
                j = 0;
                loop {
                    if j >= LIMBS_PER_FELT {
                        break;
                    }
                    draw_raw_key.append(extract_single_val(trace_vals, draw_offset + 5 + j));
                    j += 1;
                }
                draw_raw_key.append(extract_single_val(trace_vals, draw_offset + 4));
                append_logup_intermediate_constraint(
                    ref quotients,
                    interaction_vals,
                    ref interaction_col,
                    ref prev_col_cumsum,
                    self.draw_felt_lookup_elements.combine_qm31(draw_raw_key.span()),
                    extract_single_val(trace_vals, draw_offset + 134),
                );
            }

            if *self.sumcheck_enabled && *self.draw_enabled {
                // 4. Draw rows provide unpacked QM31 challenges (-1).
                let mut draw_challenge_key: Array<QM31> = array![];
                j = 0;
                loop {
                    if j >= 4 {
                        break;
                    }
                    draw_challenge_key.append(extract_single_val(trace_vals, draw_offset + j));
                    j += 1;
                }
                draw_challenge_key.append(extract_single_val(trace_vals, draw_offset + 4));
                append_logup_intermediate_constraint(
                    ref quotients,
                    interaction_vals,
                    ref interaction_col,
                    ref prev_col_cumsum,
                    self.challenge_lookup_elements.combine_qm31(draw_challenge_key.span()),
                    zero - extract_single_val(trace_vals, draw_offset + 134),
                );
            }

            if *self.draw_enabled {
                // 5. Chain draw rows provide raw Hades output felts (-1).
                let mut chain_draw_key: Array<QM31> = array![];
                j = 0;
                loop {
                    if j >= LIMBS_PER_FELT {
                        break;
                    }
                    chain_draw_key.append(extract_single_val(trace_vals, 9 + j));
                    j += 1;
                }
                chain_draw_key.append(draw_count);
                append_logup_intermediate_constraint(
                    ref quotients,
                    interaction_vals,
                    ref interaction_col,
                    ref prev_col_cumsum,
                    self.draw_felt_lookup_elements.combine_qm31(chain_draw_key.span()),
                    zero - is_draw,
                );
            }

            // Final relation entry: chain rows consume HadesPerm keys (+1).
            // The final cumulative column is shifted by claimed_sum / N and
            // uses the previous-row mask, exactly like Rust finalize_logup().
            let mut chain_hades_key: Array<QM31> = array![];
            j = 0;
            loop {
                if j >= LIMBS_PER_FELT {
                    break;
                }
                chain_hades_key.append(extract_single_val(trace_vals, j));
                j += 1;
            }
            j = 0;
            loop {
                if j >= LIMBS_PER_FELT {
                    break;
                }
                chain_hades_key.append(extract_single_val(trace_vals, 9 + j));
                j += 1;
            }
            append_logup_final_constraint(
                ref quotients,
                interaction_vals,
                interaction_col,
                prev_col_cumsum,
                self.hades_lookup_elements.combine_qm31(chain_hades_key.span()),
                is_active,
                *self.logup_claimed_sum * m31_to_qm31(n_inv_m31),
            );
        }

        // Accumulate ALL quotients (chain + Hades) using Horner's method
        let n_quotients = quotients.len();
        let mut acc: QM31 = QM31Zero::zero();
        let mut idx: u32 = 0;
        loop {
            if idx >= n_quotients {
                break;
            }
            acc = acc * random_coeff + *quotients.at(idx);
            idx += 1;
        }

        acc
    }
}

/// Convert M31 to QM31 (embed in the base component).
fn m31_to_qm31(v: M31) -> QM31 {
    QM31Trait::from_fixed_array([v, m31(0), m31(0), m31(0)])
}

fn qm31_pow2(n: u32) -> QM31 {
    m31_to_qm31(m31(pow2(n).try_into().unwrap()))
}

fn cmul(x_re: QM31, x_im: QM31, y_re: QM31, y_im: QM31) -> (QM31, QM31) {
    (x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re)
}

fn qm31_mul_at_offsets(
    trace_vals: ColumnSpan<Span<QM31>>, a_offset: u32, b_offset: u32,
) -> (QM31, QM31, QM31, QM31) {
    let two = m31_to_qm31(m31(2));
    let (x0y0_re, x0y0_im) = cmul(
        extract_single_val(trace_vals, a_offset),
        extract_single_val(trace_vals, a_offset + 1),
        extract_single_val(trace_vals, b_offset),
        extract_single_val(trace_vals, b_offset + 1),
    );
    let (x1y1_re, x1y1_im) = cmul(
        extract_single_val(trace_vals, a_offset + 2),
        extract_single_val(trace_vals, a_offset + 3),
        extract_single_val(trace_vals, b_offset + 2),
        extract_single_val(trace_vals, b_offset + 3),
    );
    let (x0y1_re, x0y1_im) = cmul(
        extract_single_val(trace_vals, a_offset),
        extract_single_val(trace_vals, a_offset + 1),
        extract_single_val(trace_vals, b_offset + 2),
        extract_single_val(trace_vals, b_offset + 3),
    );
    let (x1y0_re, x1y0_im) = cmul(
        extract_single_val(trace_vals, a_offset + 2),
        extract_single_val(trace_vals, a_offset + 3),
        extract_single_val(trace_vals, b_offset),
        extract_single_val(trace_vals, b_offset + 1),
    );
    (
        x0y0_re + two * x1y1_re - x1y1_im,
        x0y0_im + x1y1_re + two * x1y1_im,
        x0y1_re + x1y0_re,
        x0y1_im + x1y0_im,
    )
}

fn append_qm31_mul_quotients(
    ref quotients: Array<QM31>,
    trace_vals: ColumnSpan<Span<QM31>>,
    a_offset: u32,
    b_offset: u32,
    result_offset: u32,
) {
    let (expected0, expected1, expected2, expected3) = qm31_mul_at_offsets(
        trace_vals, a_offset, b_offset,
    );
    quotients.append(extract_single_val(trace_vals, result_offset) - expected0);
    quotients.append(extract_single_val(trace_vals, result_offset + 1) - expected1);
    quotients.append(extract_single_val(trace_vals, result_offset + 2) - expected2);
    quotients.append(extract_single_val(trace_vals, result_offset + 3) - expected3);
}

fn interaction_qm31_current(interaction_vals: ColumnSpan<Span<QM31>>, col_offset: u32) -> QM31 {
    let c0 = *interaction_vals.at(col_offset);
    let c1 = *interaction_vals.at(col_offset + 1);
    let c2 = *interaction_vals.at(col_offset + 2);
    let c3 = *interaction_vals.at(col_offset + 3);
    let i0 = c0.len() - 1;
    let i1 = c1.len() - 1;
    let i2 = c2.len() - 1;
    let i3 = c3.len() - 1;
    QM31Trait::from_partial_evals([*c0.at(i0), *c1.at(i1), *c2.at(i2), *c3.at(i3)])
}

fn interaction_qm31_prev_row(interaction_vals: ColumnSpan<Span<QM31>>, col_offset: u32) -> QM31 {
    let c0 = *interaction_vals.at(col_offset);
    let c1 = *interaction_vals.at(col_offset + 1);
    let c2 = *interaction_vals.at(col_offset + 2);
    let c3 = *interaction_vals.at(col_offset + 3);
    assert!(c0.len() == 2, "logup prev missing");
    assert!(c1.len() == 2, "logup prev missing");
    assert!(c2.len() == 2, "logup prev missing");
    assert!(c3.len() == 2, "logup prev missing");
    QM31Trait::from_partial_evals([*c0.at(0), *c1.at(0), *c2.at(0), *c3.at(0)])
}

fn append_logup_intermediate_constraint(
    ref quotients: Array<QM31>,
    interaction_vals: ColumnSpan<Span<QM31>>,
    ref interaction_col: u32,
    ref prev_col_cumsum: QM31,
    denominator: QM31,
    numerator: QM31,
) {
    let cur_cumsum = interaction_qm31_current(interaction_vals, interaction_col);
    quotients.append((cur_cumsum - prev_col_cumsum) * denominator - numerator);
    prev_col_cumsum = cur_cumsum;
    interaction_col += 4;
}

fn append_logup_final_constraint(
    ref quotients: Array<QM31>,
    interaction_vals: ColumnSpan<Span<QM31>>,
    interaction_col: u32,
    prev_col_cumsum: QM31,
    denominator: QM31,
    numerator: QM31,
    claimed_sum_shift: QM31,
) {
    let cur_cumsum = interaction_qm31_current(interaction_vals, interaction_col);
    let prev_row_cumsum = interaction_qm31_prev_row(interaction_vals, interaction_col);
    quotients
        .append(
            (cur_cumsum - prev_row_cumsum - prev_col_cumsum + claimed_sum_shift) * denominator
                - numerator,
        );
}

/// Extract a single QM31 value from the j-th column of a tree's mask values.
pub fn extract_single_val(tree_vals: ColumnSpan<Span<QM31>>, col_idx: u32) -> QM31 {
    let col = *tree_vals.at(col_idx);
    *col.at(0)
}

/// Compute 2^n for small n.
fn pow2(n: u32) -> u64 {
    let mut result: u64 = 1;
    let mut i: u32 = 0;
    loop {
        if i >= n {
            break;
        }
        result = result * 2;
        i += 1;
    }
    result
}
