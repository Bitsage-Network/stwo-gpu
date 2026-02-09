/// Sumcheck verifier for ML matmul proofs.
///
/// Adapted from BitSage-Cairo-Smart-Contracts/src/obelysk/sumcheck_verifier.cairo
/// for use with the stwo-ml-verifier pipeline.
///
/// Verifies on-chain that C = A × B via the sumcheck protocol over
/// multilinear extensions with Poseidon252 Fiat-Shamir channel.
use core::poseidon::{poseidon_hash_span, hades_permutation};

// ============================================================================
// M31 Field Arithmetic (p = 2^31 - 1)
// ============================================================================

const M31_P: u64 = 0x7FFFFFFF;
const M31_SHIFT: felt252 = 0x80000000;

pub fn m31_add(a: u64, b: u64) -> u64 {
    let sum = a + b;
    if sum >= M31_P { sum - M31_P } else { sum }
}

pub fn m31_sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { M31_P - (b - a) }
}

pub fn m31_mul(a: u64, b: u64) -> u64 {
    (a * b) % M31_P
}

pub fn m31_reduce(val: u64) -> u64 {
    val % M31_P
}

// ============================================================================
// CM31 = M31[i] / (i² + 1)
// ============================================================================

#[derive(Drop, Copy, Serde)]
pub struct CM31 {
    pub a: u64,
    pub b: u64,
}

pub fn cm31_add(x: CM31, y: CM31) -> CM31 {
    CM31 { a: m31_add(x.a, y.a), b: m31_add(x.b, y.b) }
}

pub fn cm31_sub(x: CM31, y: CM31) -> CM31 {
    CM31 { a: m31_sub(x.a, y.a), b: m31_sub(x.b, y.b) }
}

pub fn cm31_mul(x: CM31, y: CM31) -> CM31 {
    let ac = m31_mul(x.a, y.a);
    let bd = m31_mul(x.b, y.b);
    let ad = m31_mul(x.a, y.b);
    let bc = m31_mul(x.b, y.a);
    CM31 { a: m31_sub(ac, bd), b: m31_add(ad, bc) }
}

pub fn cm31_eq(x: CM31, y: CM31) -> bool {
    x.a == y.a && x.b == y.b
}

// ============================================================================
// QM31 = CM31[j] / (j² - (2 + i))
// ============================================================================

#[derive(Drop, Copy, Serde)]
pub struct QM31 {
    pub a: CM31,
    pub b: CM31,
}

pub fn qm31_zero() -> QM31 {
    QM31 { a: CM31 { a: 0, b: 0 }, b: CM31 { a: 0, b: 0 } }
}

pub fn qm31_add(x: QM31, y: QM31) -> QM31 {
    QM31 { a: cm31_add(x.a, y.a), b: cm31_add(x.b, y.b) }
}

pub fn qm31_mul(x: QM31, y: QM31) -> QM31 {
    let ac = cm31_mul(x.a, y.a);
    let bd = cm31_mul(x.b, y.b);
    let bd_times_irred = CM31 {
        a: m31_sub(m31_add(bd.a, bd.a), bd.b),
        b: m31_add(bd.a, m31_add(bd.b, bd.b)),
    };
    let real = cm31_add(ac, bd_times_irred);
    let apb = cm31_add(x.a, x.b);
    let cpd = cm31_add(y.a, y.b);
    let apb_cpd = cm31_mul(apb, cpd);
    let j_part = cm31_sub(cm31_sub(apb_cpd, ac), bd);
    QM31 { a: real, b: j_part }
}

pub fn qm31_eq(x: QM31, y: QM31) -> bool {
    cm31_eq(x.a, y.a) && cm31_eq(x.b, y.b)
}

pub fn qm31_sub(x: QM31, y: QM31) -> QM31 {
    QM31 { a: cm31_sub(x.a, y.a), b: cm31_sub(x.b, y.b) }
}

// ============================================================================
// Polynomial Evaluation
// ============================================================================

pub fn poly_eval_degree2(c0: QM31, c1: QM31, c2: QM31, x: QM31) -> QM31 {
    let inner = qm31_add(c1, qm31_mul(x, c2));
    qm31_add(c0, qm31_mul(x, inner))
}

// ============================================================================
// Poseidon252 Channel (matches STWO's Poseidon252Channel)
// ============================================================================

#[derive(Drop, Copy)]
pub struct PoseidonChannel {
    pub digest: felt252,
    pub n_draws: u32,
}

pub fn channel_default() -> PoseidonChannel {
    PoseidonChannel { digest: 0, n_draws: 0 }
}

pub fn channel_mix_u64(ref ch: PoseidonChannel, value: u64) {
    let (s0, _, _) = hades_permutation(ch.digest, value.into(), 2);
    ch.digest = s0;
    ch.n_draws = 0;
}

pub fn channel_mix_felt(ref ch: PoseidonChannel, value: felt252) {
    let (s0, _, _) = hades_permutation(ch.digest, value, 2);
    ch.digest = s0;
    ch.n_draws = 0;
}

fn channel_draw_felt252(ref ch: PoseidonChannel) -> felt252 {
    let (s0, _, _) = hades_permutation(ch.digest, ch.n_draws.into(), 3);
    ch.n_draws += 1;
    s0
}

fn felt252_to_m31_array_8(
    value: felt252,
) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    let shift: u256 = 0x80000000;
    let mut cur: u256 = value.into();
    let r0: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r1: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r2: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r3: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r4: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r5: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r6: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r7: u64 = (cur % shift).try_into().unwrap();
    (
        m31_reduce(r0), m31_reduce(r1), m31_reduce(r2), m31_reduce(r3),
        m31_reduce(r4), m31_reduce(r5), m31_reduce(r6), m31_reduce(r7),
    )
}

pub fn channel_draw_qm31(ref ch: PoseidonChannel) -> QM31 {
    let felt = channel_draw_felt252(ref ch);
    let (m0, m1, m2, m3, _, _, _, _) = felt252_to_m31_array_8(felt);
    QM31 { a: CM31 { a: m0, b: m1 }, b: CM31 { a: m2, b: m3 } }
}

pub fn channel_draw_qm31s(ref ch: PoseidonChannel, count: u32) -> Array<QM31> {
    let mut result: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= count { break; }
        result.append(channel_draw_qm31(ref ch));
        i += 1;
    };
    result
}

fn pack_qm31_into_felt(mut cur: felt252, v: QM31) -> felt252 {
    cur = cur * M31_SHIFT + v.a.a.into();
    cur = cur * M31_SHIFT + v.a.b.into();
    cur = cur * M31_SHIFT + v.b.a.into();
    cur = cur * M31_SHIFT + v.b.b.into();
    cur
}

pub fn channel_mix_poly_coeffs(ref ch: PoseidonChannel, c0: QM31, c1: QM31, c2: QM31) {
    let mut packed1: felt252 = 1;
    packed1 = pack_qm31_into_felt(packed1, c0);
    packed1 = pack_qm31_into_felt(packed1, c1);
    let mut packed2: felt252 = 1;
    packed2 = pack_qm31_into_felt(packed2, c2);
    ch.digest = poseidon_hash_span(array![ch.digest, packed1, packed2].span());
    ch.n_draws = 0;
}

pub fn pack_qm31_to_felt(v: QM31) -> felt252 {
    let shift: felt252 = 0x80000000;
    let mut result: felt252 = 1;
    result = result * shift + v.a.a.into();
    result = result * shift + v.a.b.into();
    result = result * shift + v.b.a.into();
    result = result * shift + v.b.b.into();
    result
}

// ============================================================================
// Proof Structures
// ============================================================================

#[derive(Drop, Copy, Serde)]
pub struct RoundPoly {
    pub c0: QM31,
    pub c1: QM31,
    pub c2: QM31,
}

#[derive(Drop, Serde)]
pub struct MleQueryRoundData {
    pub left_value: QM31,
    pub right_value: QM31,
    pub left_siblings: Array<felt252>,
    pub right_siblings: Array<felt252>,
}

#[derive(Drop, Serde)]
pub struct MleQueryProof {
    pub initial_pair_index: u32,
    pub rounds: Array<MleQueryRoundData>,
}

#[derive(Drop, Serde)]
pub struct MleOpeningProof {
    pub intermediate_roots: Array<felt252>,
    pub queries: Array<MleQueryProof>,
    pub final_value: QM31,
}

#[derive(Drop, Serde)]
pub struct MatMulSumcheckProof {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub num_rounds: u32,
    pub claimed_sum: QM31,
    pub round_polys: Array<RoundPoly>,
    pub final_a_eval: QM31,
    pub final_b_eval: QM31,
    pub a_commitment: felt252,
    pub b_commitment: felt252,
    pub a_opening: MleOpeningProof,
    pub b_opening: MleOpeningProof,
}

// ============================================================================
// Helpers
// ============================================================================

pub fn next_power_of_two(n: u32) -> u32 {
    if n == 0 { return 1; }
    let mut v = n - 1;
    v = v | (v / 2);
    v = v | (v / 4);
    v = v | (v / 16);
    v = v | (v / 256);
    v = v | (v / 65536);
    v + 1
}

pub fn log2_ceil(n: u32) -> u32 {
    assert!(n > 0, "log2(0) undefined");
    let mut result: u32 = 0;
    let mut val = n - 1;
    loop {
        if val == 0 { break; }
        val = val / 2;
        result += 1;
    };
    result
}

pub fn pow2(n: u32) -> u32 {
    let mut result: u32 = 1;
    let mut i: u32 = 0;
    loop {
        if i >= n { break; }
        result = result * 2;
        i += 1;
    };
    result
}

// ============================================================================
// MLE Opening Verification
// ============================================================================

const MLE_NUM_QUERIES: u32 = 14;

fn verify_merkle_path(
    leaf_hash: felt252, index: u32, siblings: Span<felt252>, root: felt252,
) -> bool {
    let mut current = leaf_hash;
    let mut idx = index;
    let mut i: u32 = 0;
    loop {
        if i >= siblings.len() { break; }
        let sibling = *siblings.at(i);
        if idx % 2 == 0 {
            let (s0, _, _) = hades_permutation(current, sibling, 2);
            current = s0;
        } else {
            let (s0, _, _) = hades_permutation(sibling, current, 2);
            current = s0;
        }
        idx = idx / 2;
        i += 1;
    };
    current == root
}

fn channel_draw_query_indices(
    ref ch: PoseidonChannel, half_n: u32, n_queries: u32,
) -> Array<u32> {
    let mut indices: Array<u32> = array![];
    let half_n_u64: u64 = half_n.into();
    let mut i: u32 = 0;
    loop {
        if i >= n_queries { break; }
        let felt = channel_draw_felt252(ref ch);
        let hash_u256: u256 = felt.into();
        let val_u64: u64 = (hash_u256 % 0x10000000000000000).try_into().unwrap();
        let index: u32 = (val_u64 % half_n_u64).try_into().unwrap();
        indices.append(index);
        i += 1;
    };
    indices
}

fn next_query_pair_index(current_idx: u32, layer_mid: u32) -> u32 {
    let next_half = layer_mid / 2;
    if next_half == 0 { 0 } else { current_idx % next_half }
}

pub fn verify_mle_opening(
    commitment_root: felt252,
    proof: @MleOpeningProof,
    challenges: Span<QM31>,
    ref ch: PoseidonChannel,
) -> bool {
    let n_rounds: u32 = challenges.len();

    channel_mix_felt(ref ch, commitment_root);
    let intermediate_roots_span = proof.intermediate_roots.span();
    let mut ir_i: u32 = 0;
    loop {
        if ir_i >= intermediate_roots_span.len() { break; }
        channel_mix_felt(ref ch, *intermediate_roots_span.at(ir_i));
        ir_i += 1;
    };

    let layer_roots_len: u32 = 1 + intermediate_roots_span.len();

    if n_rounds == 0 {
        return proof.queries.len() == 0;
    }

    let half_n: u32 = pow2(n_rounds - 1);
    let n_queries: u32 = if MLE_NUM_QUERIES < half_n { MLE_NUM_QUERIES } else { half_n };
    let query_indices = channel_draw_query_indices(ref ch, half_n, n_queries);

    let queries_span = proof.queries.span();
    if queries_span.len() != n_queries { return false; }

    let mut q_idx: u32 = 0;
    loop {
        if q_idx >= n_queries { break; }

        let query = queries_span.at(q_idx);
        let rounds_span = query.rounds.span();
        if rounds_span.len() != n_rounds { return false; }
        if *query.initial_pair_index != *query_indices.at(q_idx) { return false; }

        let mut current_idx: u32 = *query.initial_pair_index;
        let mut layer_size: u32 = pow2(n_rounds);

        let mut round: u32 = 0;
        loop {
            if round >= n_rounds { break; }

            let rd = rounds_span.at(round);
            let left_value: QM31 = *rd.left_value;
            let right_value: QM31 = *rd.right_value;
            let left_siblings = rd.left_siblings.span();
            let right_siblings = rd.right_siblings.span();

            let mid: u32 = layer_size / 2;

            if round < layer_roots_len {
                let layer_root = if round == 0 {
                    commitment_root
                } else {
                    *intermediate_roots_span.at(round - 1)
                };

                let left_leaf = pack_qm31_to_felt(left_value);
                let right_leaf = pack_qm31_to_felt(right_value);

                if !verify_merkle_path(left_leaf, current_idx, left_siblings, layer_root) {
                    return false;
                }
                if !verify_merkle_path(right_leaf, mid + current_idx, right_siblings, layer_root) {
                    return false;
                }
            }

            let challenge: QM31 = *challenges.at(round);
            let diff = qm31_sub(right_value, left_value);
            let fold_val = qm31_add(left_value, qm31_mul(challenge, diff));

            if round == n_rounds - 1 {
                if !qm31_eq(fold_val, *proof.final_value) { return false; }
            }

            current_idx = next_query_pair_index(current_idx, mid);
            layer_size = mid;
            round += 1;
        };

        q_idx += 1;
    };

    true
}

// ============================================================================
// Core Sumcheck Verification
// ============================================================================

/// Verify sumcheck rounds + final product check.
/// Returns (is_valid, proof_hash, assignment).
pub fn verify_sumcheck_inner(
    claimed_sum: QM31,
    round_polys: Span<RoundPoly>,
    num_rounds: u32,
    final_a_eval: QM31,
    final_b_eval: QM31,
    ref ch: PoseidonChannel,
) -> (bool, felt252, Array<QM31>) {
    let mut expected_sum = claimed_sum;
    let initial_digest = ch.digest;
    let mut assignment: Array<QM31> = array![];

    let mut round: u32 = 0;
    loop {
        if round >= num_rounds { break; }
        let poly = *round_polys.at(round);
        let eval_at_0 = poly.c0;
        let eval_at_1 = qm31_add(qm31_add(poly.c0, poly.c1), poly.c2);
        let round_sum = qm31_add(eval_at_0, eval_at_1);

        if !qm31_eq(round_sum, expected_sum) {
            let proof_hash = poseidon_hash_span(
                array![initial_digest, round.into(), 'FAIL'].span(),
            );
            return (false, proof_hash, array![]);
        }

        channel_mix_poly_coeffs(ref ch, poly.c0, poly.c1, poly.c2);
        let challenge = channel_draw_qm31(ref ch);
        assignment.append(challenge);
        expected_sum = poly_eval_degree2(poly.c0, poly.c1, poly.c2, challenge);
        round += 1;
    };

    let product = qm31_mul(final_a_eval, final_b_eval);
    if !qm31_eq(expected_sum, product) {
        let proof_hash = poseidon_hash_span(
            array![initial_digest, num_rounds.into(), 'FINAL_FAIL'].span(),
        );
        return (false, proof_hash, array![]);
    }

    let proof_hash = poseidon_hash_span(
        array![
            initial_digest,
            num_rounds.into(),
            claimed_sum.a.a.into(),
            claimed_sum.a.b.into(),
            claimed_sum.b.a.into(),
            claimed_sum.b.b.into(),
            final_a_eval.a.a.into(),
            final_a_eval.a.b.into(),
            final_a_eval.b.a.into(),
            final_a_eval.b.b.into(),
            final_b_eval.a.a.into(),
            final_b_eval.a.b.into(),
            final_b_eval.b.a.into(),
            final_b_eval.b.b.into(),
        ].span(),
    );

    (true, proof_hash, assignment)
}

/// Full matmul sumcheck verification (transcript replay + MLE opening).
pub fn verify_matmul_sumcheck(
    proof: MatMulSumcheckProof,
) -> (bool, felt252) {
    let MatMulSumcheckProof {
        m, k, n, num_rounds, claimed_sum, round_polys,
        final_a_eval, final_b_eval,
        a_commitment, b_commitment, a_opening, b_opening,
    } = proof;

    // Validate structure
    assert!(num_rounds > 0, "Proof must have at least one round");
    assert!(round_polys.len() == num_rounds, "Round count mismatch");

    let k_pow2 = next_power_of_two(k);
    let expected_rounds = log2_ceil(k_pow2);
    assert!(num_rounds == expected_rounds, "Wrong number of rounds");

    // Replay Fiat-Shamir transcript
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

    let (is_valid, proof_hash, assignment) = verify_sumcheck_inner(
        claimed_sum, round_polys.span(), num_rounds,
        final_a_eval, final_b_eval, ref ch,
    );

    if !is_valid { return (false, proof_hash); }

    if !verify_mle_opening(a_commitment, @a_opening, assignment.span(), ref ch) {
        return (false, 'A_MLE_FAIL');
    }

    if !verify_mle_opening(b_commitment, @b_opening, assignment.span(), ref ch) {
        return (false, 'B_MLE_FAIL');
    }

    (true, proof_hash)
}
