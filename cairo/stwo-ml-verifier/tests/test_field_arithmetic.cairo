use stwo_ml_verifier::sumcheck::{
    m31_add, m31_sub, m31_mul, m31_reduce,
    CM31, cm31_add, cm31_sub, cm31_mul, cm31_eq,
    QM31, qm31_zero, qm31_add, qm31_sub, qm31_mul, qm31_eq,
    poly_eval_degree2,
    channel_default, channel_mix_u64, channel_mix_felt,
    channel_draw_qm31, channel_draw_qm31s,
    pack_qm31_to_felt,
    next_power_of_two, log2_ceil, pow2,
};

const P: u64 = 0x7FFFFFFF; // 2^31 - 1

// ============================================================================
// M31 Arithmetic
// ============================================================================

#[test]
fn test_m31_add_basic() {
    assert!(m31_add(3, 5) == 8, "3+5 should be 8");
}

#[test]
fn test_m31_add_wraps() {
    // (P-1) + 2 = P+1, P+1 >= P so result = P+1-P = 1
    let result = m31_add(P - 1, 2);
    assert!(result == 1, "should wrap around P");
}

#[test]
fn test_m31_add_at_boundary() {
    // (P-1) + 1 = P, P >= P so result = P-P = 0
    let result = m31_add(P - 1, 1);
    assert!(result == 0, "P-1 + 1 should wrap to 0");
}

#[test]
fn test_m31_add_zero() {
    assert!(m31_add(42, 0) == 42, "x+0 should be x");
    assert!(m31_add(0, 42) == 42, "0+x should be x");
}

#[test]
fn test_m31_sub_basic() {
    assert!(m31_sub(10, 3) == 7, "10-3 should be 7");
}

#[test]
fn test_m31_sub_underflow() {
    // 3 - 5 mod P = P - 2
    let result = m31_sub(3, 5);
    assert!(result == P - 2, "3-5 should wrap to P-2");
}

#[test]
fn test_m31_sub_equal() {
    assert!(m31_sub(100, 100) == 0, "x-x should be 0");
}

#[test]
fn test_m31_mul_basic() {
    assert!(m31_mul(3, 7) == 21, "3*7 should be 21");
}

#[test]
fn test_m31_mul_zero() {
    assert!(m31_mul(999, 0) == 0, "x*0 should be 0");
    assert!(m31_mul(0, 999) == 0, "0*x should be 0");
}

#[test]
fn test_m31_mul_one() {
    assert!(m31_mul(42, 1) == 42, "x*1 should be x");
    assert!(m31_mul(1, 42) == 42, "1*x should be x");
}

#[test]
fn test_m31_mul_large() {
    // Verify modular reduction: (P-1) * 2 = 2P - 2 mod P = P - 2
    let result = m31_mul(P - 1, 2);
    assert!(result == P - 2, "(P-1)*2 should be P-2");
}

#[test]
fn test_m31_reduce_no_op() {
    assert!(m31_reduce(42) == 42, "42 < P should stay 42");
    assert!(m31_reduce(0) == 0, "0 should stay 0");
}

#[test]
fn test_m31_reduce_wraps() {
    assert!(m31_reduce(P) == 0, "P should reduce to 0");
    assert!(m31_reduce(P + 1) == 1, "P+1 should reduce to 1");
    assert!(m31_reduce(P * 2) == 0, "2P should reduce to 0");
}

// ============================================================================
// CM31 Extension Field
// ============================================================================

#[test]
fn test_cm31_add() {
    let x = CM31 { a: 3, b: 5 };
    let y = CM31 { a: 7, b: 11 };
    let result = cm31_add(x, y);
    assert!(result.a == 10 && result.b == 16, "cm31 add component-wise");
}

#[test]
fn test_cm31_sub() {
    let x = CM31 { a: 10, b: 20 };
    let y = CM31 { a: 3, b: 5 };
    let result = cm31_sub(x, y);
    assert!(result.a == 7 && result.b == 15, "cm31 sub component-wise");
}

#[test]
fn test_cm31_mul() {
    // (3+5i) * (7+11i) = (3*7 - 5*11) + (3*11 + 5*7)i = (21-55) + (33+35)i
    // Real: 21 - 55 = -34 mod P = P - 34
    // Imag: 33 + 35 = 68
    let x = CM31 { a: 3, b: 5 };
    let y = CM31 { a: 7, b: 11 };
    let result = cm31_mul(x, y);
    assert!(result.a == P - 34, "cm31 mul real part");
    assert!(result.b == 68, "cm31 mul imag part");
}

#[test]
fn test_cm31_mul_by_zero() {
    let x = CM31 { a: 42, b: 99 };
    let zero = CM31 { a: 0, b: 0 };
    let result = cm31_mul(x, zero);
    assert!(result.a == 0 && result.b == 0, "cm31 * 0 should be 0");
}

#[test]
fn test_cm31_eq_true() {
    let x = CM31 { a: 42, b: 99 };
    let y = CM31 { a: 42, b: 99 };
    assert!(cm31_eq(x, y), "identical CM31 should be equal");
}

#[test]
fn test_cm31_eq_false() {
    let x = CM31 { a: 42, b: 99 };
    let y = CM31 { a: 42, b: 100 };
    assert!(!cm31_eq(x, y), "different CM31 should not be equal");
}

// ============================================================================
// QM31 Extension Field
// ============================================================================

#[test]
fn test_qm31_add_identity() {
    let x = QM31 { a: CM31 { a: 10, b: 20 }, b: CM31 { a: 30, b: 40 } };
    let zero = qm31_zero();
    let result = qm31_add(x, zero);
    assert!(qm31_eq(result, x), "x + 0 should equal x");
}

#[test]
fn test_qm31_add_commutative() {
    let a = QM31 { a: CM31 { a: 3, b: 5 }, b: CM31 { a: 7, b: 11 } };
    let b = QM31 { a: CM31 { a: 13, b: 17 }, b: CM31 { a: 19, b: 23 } };
    let ab = qm31_add(a, b);
    let ba = qm31_add(b, a);
    assert!(qm31_eq(ab, ba), "addition should be commutative");
}

#[test]
fn test_qm31_sub_inverse() {
    let x = QM31 { a: CM31 { a: 100, b: 200 }, b: CM31 { a: 300, b: 400 } };
    let result = qm31_sub(x, x);
    let zero = qm31_zero();
    assert!(qm31_eq(result, zero), "x - x should be zero");
}

#[test]
fn test_qm31_mul_identity() {
    // QM31 identity = (1+0i, 0+0j)
    let one = QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let x = QM31 { a: CM31 { a: 42, b: 99 }, b: CM31 { a: 7, b: 13 } };
    let result = qm31_mul(x, one);
    assert!(qm31_eq(result, x), "x * 1 should equal x");
}

#[test]
fn test_qm31_mul_zero() {
    let x = QM31 { a: CM31 { a: 42, b: 99 }, b: CM31 { a: 7, b: 13 } };
    let zero = qm31_zero();
    let result = qm31_mul(x, zero);
    assert!(qm31_eq(result, zero), "x * 0 should be zero");
}

#[test]
fn test_qm31_eq_true() {
    let x = QM31 { a: CM31 { a: 42, b: 99 }, b: CM31 { a: 7, b: 13 } };
    let y = QM31 { a: CM31 { a: 42, b: 99 }, b: CM31 { a: 7, b: 13 } };
    assert!(qm31_eq(x, y), "identical QM31 should be equal");
}

#[test]
fn test_qm31_eq_false() {
    let x = QM31 { a: CM31 { a: 42, b: 99 }, b: CM31 { a: 7, b: 13 } };
    let y = QM31 { a: CM31 { a: 42, b: 99 }, b: CM31 { a: 7, b: 14 } };
    assert!(!qm31_eq(x, y), "different QM31 should not be equal");
}

// ============================================================================
// Polynomial Evaluation
// ============================================================================

#[test]
fn test_poly_eval_degree2_constant() {
    // p(x) = 5 + 0*x + 0*x^2 = 5 for all x
    let c0 = QM31 { a: CM31 { a: 5, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let zero = qm31_zero();
    let x = QM31 { a: CM31 { a: 42, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let result = poly_eval_degree2(c0, zero, zero, x);
    assert!(qm31_eq(result, c0), "constant poly should eval to c0");
}

#[test]
fn test_poly_eval_degree2_linear() {
    // p(x) = 0 + 1*x + 0*x^2 = x
    let zero = qm31_zero();
    let one = QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let x = QM31 { a: CM31 { a: 7, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let result = poly_eval_degree2(zero, one, zero, x);
    assert!(qm31_eq(result, x), "identity poly should eval to x");
}

// ============================================================================
// Poseidon Channel
// ============================================================================

#[test]
fn test_channel_default() {
    let ch = channel_default();
    assert!(ch.digest == 0, "fresh channel should have digest 0");
    assert!(ch.n_draws == 0, "fresh channel should have 0 draws");
}

#[test]
fn test_channel_mix_changes_digest() {
    let mut ch = channel_default();
    let before = ch.digest;
    channel_mix_u64(ref ch, 42);
    assert!(ch.digest != before, "mixing should change digest");
    assert!(ch.n_draws == 0, "mixing resets draw count");
}

#[test]
fn test_channel_mix_felt_changes_digest() {
    let mut ch = channel_default();
    channel_mix_felt(ref ch, 0x1234);
    let d1 = ch.digest;
    channel_mix_felt(ref ch, 0x5678);
    assert!(ch.digest != d1, "different felt should change digest");
}

#[test]
fn test_channel_draw_qm31_valid_m31() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 1);
    let q = channel_draw_qm31(ref ch);
    // All components must be < P (valid M31)
    assert!(q.a.a < P, "a.a must be < P");
    assert!(q.a.b < P, "a.b must be < P");
    assert!(q.b.a < P, "b.a must be < P");
    assert!(q.b.b < P, "b.b must be < P");
}

#[test]
fn test_channel_draw_deterministic() {
    // Same seed → same draw
    let mut ch1 = channel_default();
    channel_mix_u64(ref ch1, 42);
    let q1 = channel_draw_qm31(ref ch1);

    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 42);
    let q2 = channel_draw_qm31(ref ch2);

    assert!(qm31_eq(q1, q2), "same seed should produce same draw");
}

#[test]
fn test_channel_draw_different_seeds() {
    let mut ch1 = channel_default();
    channel_mix_u64(ref ch1, 42);
    let q1 = channel_draw_qm31(ref ch1);

    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 43);
    let q2 = channel_draw_qm31(ref ch2);

    assert!(!qm31_eq(q1, q2), "different seeds should produce different draws");
}

#[test]
fn test_channel_draw_qm31s_count() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 1);
    let draws = channel_draw_qm31s(ref ch, 5);
    assert!(draws.len() == 5, "should draw exactly 5 QM31s");
}

#[test]
fn test_channel_draw_qm31s_empty() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 1);
    let draws = channel_draw_qm31s(ref ch, 0);
    assert!(draws.len() == 0, "drawing 0 should return empty");
}

#[test]
fn test_pack_qm31_to_felt_nonzero() {
    let q = QM31 { a: CM31 { a: 1, b: 2 }, b: CM31 { a: 3, b: 4 } };
    let packed = pack_qm31_to_felt(q);
    assert!(packed != 0, "packed non-zero QM31 should be nonzero");
}

#[test]
fn test_pack_qm31_to_felt_different_inputs() {
    let q1 = QM31 { a: CM31 { a: 1, b: 2 }, b: CM31 { a: 3, b: 4 } };
    let q2 = QM31 { a: CM31 { a: 5, b: 6 }, b: CM31 { a: 7, b: 8 } };
    let p1 = pack_qm31_to_felt(q1);
    let p2 = pack_qm31_to_felt(q2);
    assert!(p1 != p2, "different QM31 should pack differently");
}

// ============================================================================
// Utility Helpers
// ============================================================================

#[test]
fn test_next_power_of_two() {
    assert!(next_power_of_two(0) == 1, "0 -> 1");
    assert!(next_power_of_two(1) == 1, "1 -> 1");
    assert!(next_power_of_two(2) == 2, "2 -> 2");
    assert!(next_power_of_two(3) == 4, "3 -> 4");
    assert!(next_power_of_two(4) == 4, "4 -> 4");
    assert!(next_power_of_two(5) == 8, "5 -> 8");
    assert!(next_power_of_two(7) == 8, "7 -> 8");
    assert!(next_power_of_two(8) == 8, "8 -> 8");
    assert!(next_power_of_two(9) == 16, "9 -> 16");
}

#[test]
fn test_log2_ceil() {
    assert!(log2_ceil(1) == 0, "log2(1) = 0");
    assert!(log2_ceil(2) == 1, "log2(2) = 1");
    assert!(log2_ceil(3) == 2, "log2(3) = 2");
    assert!(log2_ceil(4) == 2, "log2(4) = 2");
    assert!(log2_ceil(5) == 3, "log2(5) = 3");
    assert!(log2_ceil(8) == 3, "log2(8) = 3");
    assert!(log2_ceil(16) == 4, "log2(16) = 4");
}

#[test]
fn test_pow2() {
    assert!(pow2(0) == 1, "2^0 = 1");
    assert!(pow2(1) == 2, "2^1 = 2");
    assert!(pow2(2) == 4, "2^2 = 4");
    assert!(pow2(3) == 8, "2^3 = 8");
    assert!(pow2(10) == 1024, "2^10 = 1024");
}
