use stwo_ml_verifier::sumcheck::{
    CM31, QM31, RoundPoly, MatMulSumcheckProof, MleOpeningProof,
    qm31_zero,
    channel_default, channel_mix_u64, channel_draw_qm31,
    verify_sumcheck_inner, verify_matmul_sumcheck,
};

// ============================================================================
// verify_sumcheck_inner
// ============================================================================

#[test]
fn test_verify_sumcheck_inner_invalid_sum() {
    // Round poly where c0 + (c0 + c1 + c2) != claimed_sum
    // c0 = (1,0,0,0), c1 = (1,0,0,0), c2 = (0,0,0,0)
    // eval_at_0 = c0 = 1, eval_at_1 = c0+c1+c2 = 2, round_sum = 3
    // But claimed_sum = (999,0,0,0) → mismatch
    let claimed_sum = QM31 { a: CM31 { a: 999, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let round_polys = array![
        RoundPoly {
            c0: QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } },
            c1: QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } },
            c2: qm31_zero(),
        },
    ];
    let final_a = QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let final_b = QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let mut ch = channel_default();

    let (is_valid, _proof_hash, _assignment) = verify_sumcheck_inner(
        claimed_sum, round_polys.span(), 1, final_a, final_b, ref ch,
    );
    assert!(!is_valid, "mismatched sum should fail");
}

#[test]
fn test_verify_sumcheck_inner_zero_rounds() {
    // 0 rounds: final product check only
    // claimed_sum should equal final_a * final_b
    let one = QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } };
    let round_polys: Array<RoundPoly> = array![];
    let mut ch = channel_default();

    let (is_valid, _hash, assignment) = verify_sumcheck_inner(
        one,              // claimed_sum = 1
        round_polys.span(),
        0,                // num_rounds
        one,              // final_a = 1
        one,              // final_b = 1, product = 1 = claimed_sum
        ref ch,
    );
    assert!(is_valid, "0 rounds with matching product should pass");
    assert!(assignment.len() == 0, "0 rounds should have empty assignment");
}

// ============================================================================
// verify_matmul_sumcheck — structural validation
// ============================================================================

#[test]
#[should_panic(expected: "Proof must have at least one round")]
fn test_verify_matmul_sumcheck_zero_rounds() {
    let proof = MatMulSumcheckProof {
        m: 2, k: 2, n: 2,
        num_rounds: 0, // invalid
        claimed_sum: qm31_zero(),
        round_polys: array![],
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0,
        b_commitment: 0,
        a_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
        b_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
    };
    verify_matmul_sumcheck(proof);
}

#[test]
#[should_panic(expected: "Round count mismatch")]
fn test_verify_matmul_sumcheck_mismatched_polys() {
    let proof = MatMulSumcheckProof {
        m: 2, k: 2, n: 2,
        num_rounds: 1, // expects 1 round poly
        claimed_sum: qm31_zero(),
        round_polys: array![], // but 0 provided
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0,
        b_commitment: 0,
        a_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
        b_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
    };
    verify_matmul_sumcheck(proof);
}

#[test]
#[should_panic(expected: "Wrong number of rounds")]
fn test_verify_matmul_sumcheck_wrong_rounds_for_dimensions() {
    // k=4, so expected rounds = log2(4) = 2, but num_rounds=1
    let proof = MatMulSumcheckProof {
        m: 2, k: 4, n: 2,
        num_rounds: 1,
        claimed_sum: qm31_zero(),
        round_polys: array![
            RoundPoly { c0: qm31_zero(), c1: qm31_zero(), c2: qm31_zero() },
        ],
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0,
        b_commitment: 0,
        a_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
        b_opening: MleOpeningProof {
            intermediate_roots: array![],
            queries: array![],
            final_value: qm31_zero(),
        },
    };
    verify_matmul_sumcheck(proof);
}

// ============================================================================
// Cross-language vector: channel_draw_qm31 after mix(42)
// ============================================================================

#[test]
fn test_cross_language_channel_draw_qm31() {
    // From Rust: draw_qm31 after mix(42): (693979178, 1709616127, 206777170, 2102168542)
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 42);
    let q = channel_draw_qm31(ref ch);

    assert!(q.a.a == 693979178, "a.a mismatch with Rust");
    assert!(q.a.b == 1709616127, "a.b mismatch with Rust");
    assert!(q.b.a == 206777170, "b.a mismatch with Rust");
    assert!(q.b.b == 2102168542, "b.b mismatch with Rust");
}

#[test]
fn test_cross_language_channel_mix_dimensions() {
    // Verify channel state after mixing m=2, k=2, n=2
    // Same sequence used by verify_matmul_sumcheck
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 2); // m
    let d1 = ch.digest;
    channel_mix_u64(ref ch, 2); // k
    let d2 = ch.digest;
    channel_mix_u64(ref ch, 2); // n
    let d3 = ch.digest;

    // Digests should be different after each mix
    assert!(d1 != 0, "digest after mix(m) should be non-zero");
    assert!(d1 != d2, "digest should change after mix(k)");
    assert!(d2 != d3, "digest should change after mix(n)");

    // Draw row challenge (1 QM31 since log2(2)=1)
    let row_ch = channel_draw_qm31(ref ch);
    // Values should be valid M31
    assert!(row_ch.a.a < 0x7FFFFFFF, "row challenge a.a should be valid M31");
}
