// Core Sumcheck Verification
//
// Verifies a sumcheck proof by replaying the Fiat-Shamir transcript.
//
// Matches STWO's partially_verify() + final evaluation check:
//   For each round:
//     1. Check: p_i(0) + p_i(1) = expected_sum
//     2. channel.mix_felts(round_poly coefficients)
//     3. challenge = channel.draw_secure_felt()
//     4. expected_sum ← p_i(challenge)
//   Final: expected_sum = final_a_eval × final_b_eval

use core::poseidon::poseidon_hash_span;
use crate::field::{
    QM31, qm31_add, qm31_mul, qm31_eq, qm31_zero, qm31_one, poly_eval_degree2,
    pack_qm31_to_felt,
};
use crate::channel::{
    PoseidonChannel, channel_default, channel_mix_u64, channel_mix_felt,
    channel_mix_poly_coeffs, channel_draw_qm31,
};
use crate::types::{RoundPoly, BatchedMatMulProof, BatchedMatMulEntry};

/// Verify sumcheck rounds and return (is_valid, proof_hash, assignment).
///
/// The channel state must match the prover's state at sumcheck entry
/// (after mixing dimensions, drawing row/col challenges, mixing claimed_sum
/// and commitments).
///
/// Returns:
/// - `is_valid`: true if all round checks and final check pass
/// - `proof_hash`: Poseidon commitment to the proof transcript
/// - `assignment`: the challenges drawn at each round (needed for MLE opening)
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
        if round >= num_rounds {
            break;
        }

        let poly = *round_polys.at(round);

        // p_i(0) = c0, p_i(1) = c0 + c1 + c2
        let eval_at_0 = poly.c0;
        let eval_at_1 = qm31_add(qm31_add(poly.c0, poly.c1), poly.c2);
        let round_sum = qm31_add(eval_at_0, eval_at_1);

        if !qm31_eq(round_sum, expected_sum) {
            let proof_hash = poseidon_hash_span(
                array![initial_digest, round.into(), 'ROUND_FAIL'].span(),
            );
            return (false, proof_hash, array![]);
        }

        // Mix round polynomial into channel
        channel_mix_poly_coeffs(ref ch, poly.c0, poly.c1, poly.c2);

        // Draw random challenge
        let challenge = channel_draw_qm31(ref ch);
        assignment.append(challenge);

        // Update expected sum: expected_sum ← p_i(challenge)
        expected_sum = poly_eval_degree2(poly.c0, poly.c1, poly.c2, challenge);

        round += 1;
    };

    // Final check: expected_sum = f_A(assignment) × f_B(assignment)
    let product = qm31_mul(final_a_eval, final_b_eval);

    if !qm31_eq(expected_sum, product) {
        let proof_hash = poseidon_hash_span(
            array![initial_digest, num_rounds.into(), 'FINAL_FAIL'].span(),
        );
        return (false, proof_hash, array![]);
    }

    // Compute proof hash for on-chain recording
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
        ]
            .span(),
    );

    (true, proof_hash, assignment)
}

/// Verify a single sumcheck round check: p(0) + p(1) == expected_sum.
/// Exposed for unit testing.
pub fn check_round_sum(poly: RoundPoly, expected_sum: QM31) -> bool {
    let eval_at_0 = poly.c0;
    let eval_at_1 = qm31_add(qm31_add(poly.c0, poly.c1), poly.c2);
    let round_sum = qm31_add(eval_at_0, eval_at_1);
    qm31_eq(round_sum, expected_sum)
}

// ============================================================================
// Batched Sumcheck Verification
// ============================================================================

/// Verify a batched matmul sumcheck proof with full Fiat-Shamir transcript replay.
///
/// Replays the exact prover transcript (gpu_sumcheck.rs prove_matmul_batch_onchain_gpu):
///   1. channel.mix_u64(num_entries), channel.mix_u64(k)
///   2. For each entry: mix_felt(claimed_sum), mix_felt(a_commit), mix_felt(b_commit)
///   3. lambda = channel.draw_qm31()
///   4. Validate lambda matches proof, combined_claimed_sum = Σ λ^i · claimed_sum_i
///   5. Per round: check p(0)+p(1)=current_sum, mix poly, draw challenge, eval p(challenge)
///   6. Final: current_sum = Σ λ^i · final_a_eval_i · final_b_eval_i
///
/// Returns (is_valid, proof_hash).
pub fn verify_batched_sumcheck(
    proof: @BatchedMatMulProof,
) -> (bool, felt252) {
    let k = *proof.k;
    let num_rounds = *proof.num_rounds;
    let entries = proof.entries;
    let round_polys = proof.round_polys;
    let num_entries: u32 = entries.len();

    // Basic validation
    if num_entries == 0 {
        return (false, 'EMPTY_BATCH');
    }
    if round_polys.len() != num_rounds {
        return (false, 'ROUND_COUNT');
    }
    if num_rounds == 0 {
        return (false, 'ZERO_ROUNDS');
    }

    // ---- Step 1-3: Replay Fiat-Shamir to derive lambda ----
    let mut ch = channel_default();

    // Mix batch metadata (matches prover: mix_u64(num_entries), mix_u64(k))
    channel_mix_u64(ref ch, num_entries.into());
    channel_mix_u64(ref ch, k.into());

    // Mix per-entry commitments and claimed sums
    let mut i: u32 = 0;
    loop {
        if i >= num_entries {
            break;
        }
        let entry: @BatchedMatMulEntry = entries.at(i);
        channel_mix_felt(ref ch, pack_qm31_to_felt(*entry.claimed_sum));
        channel_mix_felt(ref ch, *entry.a_commitment);
        channel_mix_felt(ref ch, *entry.b_commitment);
        i += 1;
    };

    // Draw lambda
    let lambda = channel_draw_qm31(ref ch);

    // Verify lambda matches the one stored in proof
    if !qm31_eq(lambda, *proof.lambda) {
        return (false, 'LAMBDA_MISMATCH');
    }

    // ---- Step 4: Validate combined_claimed_sum = Σ λ^i · claimed_sum_i ----
    let mut expected_combined = qm31_zero();
    let mut lambda_pow = qm31_one();
    i = 0;
    loop {
        if i >= num_entries {
            break;
        }
        let entry: @BatchedMatMulEntry = entries.at(i);
        expected_combined = qm31_add(expected_combined, qm31_mul(lambda_pow, *entry.claimed_sum));
        lambda_pow = qm31_mul(lambda_pow, lambda);
        i += 1;
    };

    if !qm31_eq(expected_combined, *proof.combined_claimed_sum) {
        return (false, 'COMBINED_SUM');
    }

    // ---- Step 5: Verify sumcheck rounds with Fiat-Shamir ----
    let mut current_sum = *proof.combined_claimed_sum;
    let mut round: u32 = 0;
    loop {
        if round >= num_rounds {
            break;
        }

        let poly: RoundPoly = *round_polys.at(round);

        // Check: p(0) + p(1) = current_sum
        let eval_at_0 = poly.c0;
        let eval_at_1 = qm31_add(qm31_add(poly.c0, poly.c1), poly.c2);
        let round_sum = qm31_add(eval_at_0, eval_at_1);

        if !qm31_eq(round_sum, current_sum) {
            return (false, 'ROUND_FAIL');
        }

        // Mix round polynomial into channel (matches prover)
        channel_mix_poly_coeffs(ref ch, poly.c0, poly.c1, poly.c2);

        // Draw challenge from Fiat-Shamir (matches prover)
        let challenge = channel_draw_qm31(ref ch);

        // Update expected sum: current_sum = p(challenge)
        current_sum = poly_eval_degree2(poly.c0, poly.c1, poly.c2, challenge);

        round += 1;
    };

    // ---- Step 6: Final check — current_sum = Σ λ^i · a_eval_i · b_eval_i ----
    let mut expected_final = qm31_zero();
    lambda_pow = qm31_one();
    i = 0;
    loop {
        if i >= num_entries {
            break;
        }
        let entry: @BatchedMatMulEntry = entries.at(i);
        let ab = qm31_mul(*entry.final_a_eval, *entry.final_b_eval);
        expected_final = qm31_add(expected_final, qm31_mul(lambda_pow, ab));
        lambda_pow = qm31_mul(lambda_pow, lambda);
        i += 1;
    };

    if !qm31_eq(current_sum, expected_final) {
        return (false, 'FINAL_EVAL');
    }

    // Compute proof hash for on-chain recording
    let proof_hash = poseidon_hash_span(
        array![
            ch.digest,
            num_rounds.into(),
            num_entries.into(),
            (*proof.combined_claimed_sum).a.a.into(),
            (*proof.combined_claimed_sum).a.b.into(),
            (*proof.combined_claimed_sum).b.a.into(),
            (*proof.combined_claimed_sum).b.b.into(),
        ]
            .span(),
    );

    (true, proof_hash)
}
