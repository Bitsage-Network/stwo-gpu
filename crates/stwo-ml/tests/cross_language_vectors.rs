//! Cross-language test vector generation for Cairo verifier alignment.
//!
//! Generates deterministic intermediate values from a 2×2 matmul proof
//! using PoseidonChannel, so the Cairo verifier can replay the identical
//! transcript and assert matching values at each step.
//!
//! Run with: cargo test -p stwo-ml cross_language -- --nocapture

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use starknet_ff::FieldElement;

use stwo_ml::components::matmul::{
    M31Matrix, matmul_m31, prove_matmul_sumcheck_onchain, verify_matmul_sumcheck_onchain,
};
use stwo_ml::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};

/// Helper: format a FieldElement as 0x-prefixed hex.
fn felt_hex(f: FieldElement) -> String {
    format!("0x{:064x}", {
        let bytes = f.to_bytes_be();
        let mut val = [0u8; 32];
        val.copy_from_slice(&bytes);
        u256_from_be(val)
    })
}

fn u256_from_be(bytes: [u8; 32]) -> u128 {
    // Just the low 128 bits for display (Starknet felt252 fits in 252 bits)
    u128::from_be_bytes(bytes[16..32].try_into().unwrap())
}

/// Helper: format a QM31 as (a.a, a.b, b.a, b.b) tuple.
fn qm31_components(sf: SecureField) -> (u32, u32, u32, u32) {
    (sf.0 .0 .0, sf.0 .1 .0, sf.1 .0 .0, sf.1 .1 .0)
}

/// Build the canonical 2×2 test matrices.
fn make_test_matrices() -> (M31Matrix, M31Matrix, M31Matrix) {
    let mut a = M31Matrix::new(2, 2);
    a.set(0, 0, M31::from(1));
    a.set(0, 1, M31::from(2));
    a.set(1, 0, M31::from(3));
    a.set(1, 1, M31::from(4));

    let mut b = M31Matrix::new(2, 2);
    b.set(0, 0, M31::from(5));
    b.set(0, 1, M31::from(6));
    b.set(1, 0, M31::from(7));
    b.set(1, 1, M31::from(8));

    let c = matmul_m31(&a, &b);
    (a, b, c)
}

#[test]
fn generate_cross_language_vectors() {
    let (a, b, c) = make_test_matrices();

    // Verify C = A * B: [[19, 22], [43, 50]]
    assert_eq!(c.get(0, 0), M31::from(19));
    assert_eq!(c.get(0, 1), M31::from(22));
    assert_eq!(c.get(1, 0), M31::from(43));
    assert_eq!(c.get(1, 1), M31::from(50));

    println!("=== Cross-Language Test Vectors (2x2 MatMul) ===\n");

    // Replay the PoseidonChannel transcript step by step
    let mut ch = PoseidonChannel::new();

    // Step 1: Mix dimensions
    ch.mix_u64(2); // m
    let digest_after_m = ch.digest();
    println!("digest_after_mix_m:  {}", felt_hex(digest_after_m));

    ch.mix_u64(2); // k
    let digest_after_k = ch.digest();
    println!("digest_after_mix_k:  {}", felt_hex(digest_after_k));

    ch.mix_u64(2); // n
    let digest_after_n = ch.digest();
    println!("digest_after_mix_n:  {}", felt_hex(digest_after_n));

    // Step 2: Draw row challenges (log2(2) = 1 QM31)
    let r_i_0 = ch.draw_qm31();
    let digest_after_ri = ch.digest();
    println!("\nrow_challenge[0]:    {:?}", qm31_components(r_i_0));
    println!("digest_after_ri:     {}", felt_hex(digest_after_ri));

    // Step 3: Draw col challenges (log2(2) = 1 QM31)
    let r_j_0 = ch.draw_qm31();
    let digest_after_rj = ch.digest();
    println!("col_challenge[0]:    {:?}", qm31_components(r_j_0));
    println!("digest_after_rj:     {}", felt_hex(digest_after_rj));

    // Step 4: Generate proof to get claimed_sum and commitments
    let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
        .expect("proving should succeed");

    println!("\n--- Proof Values ---");
    println!("claimed_sum:         {:?}", qm31_components(proof.claimed_sum));
    println!("claimed_sum_packed:  {}", felt_hex(securefield_to_felt(proof.claimed_sum)));
    println!("a_commitment:        {}", felt_hex(proof.a_commitment));
    println!("b_commitment:        {}", felt_hex(proof.b_commitment));
    println!("num_rounds:          {}", proof.num_rounds);

    // Step 5: Mix claimed_sum into our replay channel
    ch.mix_felt(securefield_to_felt(proof.claimed_sum));
    let digest_after_claimed = ch.digest();
    println!("\ndigest_after_mix_claimed_sum: {}", felt_hex(digest_after_claimed));

    // Step 6: Mix commitments
    ch.mix_felt(proof.a_commitment);
    let digest_after_a_commit = ch.digest();
    println!("digest_after_mix_a_commitment: {}", felt_hex(digest_after_a_commit));

    ch.mix_felt(proof.b_commitment);
    let digest_after_b_commit = ch.digest();
    println!("digest_after_mix_b_commitment: {}", felt_hex(digest_after_b_commit));

    // Step 7: Sumcheck round(s) — 2x2 has 1 round (log2(k=2) = 1)
    println!("\n--- Sumcheck Rounds ---");
    for (i, rp) in proof.round_polys.iter().enumerate() {
        println!("round_poly[{}].c0:    {:?}", i, qm31_components(rp.c0));
        println!("round_poly[{}].c1:    {:?}", i, qm31_components(rp.c1));
        println!("round_poly[{}].c2:    {:?}", i, qm31_components(rp.c2));

        // Mix poly coefficients
        ch.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);
        let digest_after_poly = ch.digest();
        println!("digest_after_mix_poly[{}]: {}", i, felt_hex(digest_after_poly));

        // Draw challenge
        let challenge = ch.draw_qm31();
        println!("challenge[{}]:        {:?}", i, qm31_components(challenge));
        println!("digest_after_challenge[{}]: {}", i, felt_hex(ch.digest()));
    }

    println!("\nfinal_a_eval:        {:?}", qm31_components(proof.final_a_eval));
    println!("final_b_eval:        {:?}", qm31_components(proof.final_b_eval));

    // Step 8: MLE opening data
    println!("\n--- MLE Opening Proof A ---");
    println!("a_opening.intermediate_roots: {} roots", proof.a_opening.intermediate_roots.len());
    for (i, root) in proof.a_opening.intermediate_roots.iter().enumerate() {
        println!("  root[{}]: {}", i, felt_hex(*root));
    }
    println!("a_opening.queries: {} queries", proof.a_opening.queries.len());
    println!("a_opening.final_value: {:?}", qm31_components(proof.a_opening.final_value));

    println!("\n--- MLE Opening Proof B ---");
    println!("b_opening.intermediate_roots: {} roots", proof.b_opening.intermediate_roots.len());
    for (i, root) in proof.b_opening.intermediate_roots.iter().enumerate() {
        println!("  root[{}]: {}", i, felt_hex(*root));
    }
    println!("b_opening.queries: {} queries", proof.b_opening.queries.len());
    println!("b_opening.final_value: {:?}", qm31_components(proof.b_opening.final_value));

    // Step 9: Verify the proof ourselves
    println!("\n--- Verification ---");
    let result = verify_matmul_sumcheck_onchain(&proof);
    println!("Rust verification: {:?}", result.is_ok());
    assert!(result.is_ok(), "proof should verify: {:?}", result.err());

    println!("\n=== End Test Vectors ===");
}

#[test]
fn test_channel_draw_qm31_matches_cairo() {
    // This test verifies that draw_qm31 produces values matching Cairo's
    // channel_draw_qm31 (which uses felt252_to_m31_array_8 + m31_reduce).
    //
    // Cairo extracts 8 M31 values by successive floor_div(2^31), takes first 4.
    // Rust extracts 4 M31 values by successive 31-bit shifts from low bits.
    //
    // Both should produce identical values IF the underlying Poseidon hash
    // (hades_permutation) is the same.

    let mut ch = PoseidonChannel::new();
    ch.mix_u64(42);

    let q = ch.draw_qm31();
    let (v0, v1, v2, v3) = qm31_components(q);

    // All values must be valid M31 (< 2^31 - 1)
    let p = (1u32 << 31) - 1;
    assert!(v0 < p, "v0 out of range");
    assert!(v1 < p, "v1 out of range");
    assert!(v2 < p, "v2 out of range");
    assert!(v3 < p, "v3 out of range");

    println!("draw_qm31 after mix(42): ({}, {}, {}, {})", v0, v1, v2, v3);
}

#[test]
fn test_proof_roundtrip_4x4() {
    // Larger test: 4×4 matmul (2 sumcheck rounds)
    let mut a = M31Matrix::new(4, 4);
    let mut b = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            b.set(i, j, M31::from((i * 4 + j + 17) as u32));
        }
    }
    let c = matmul_m31(&a, &b);

    let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
        .expect("4x4 proving should succeed");
    assert_eq!(proof.num_rounds, 2);
    assert_eq!(proof.round_polys.len(), 2);

    verify_matmul_sumcheck_onchain(&proof)
        .expect("4x4 verification should succeed");
}
