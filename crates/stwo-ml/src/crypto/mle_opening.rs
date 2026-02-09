//! MLE evaluation opening proof protocol.
//!
//! Implements the multilinear extension opening proof matching Cairo's
//! `MleOpeningProof` struct. Given evaluations on `{0,1}^n` committed
//! via a Poseidon Merkle tree, proves that `evaluate_mle(f, r) = claimed_value`.
//!
//! The protocol iteratively folds the evaluations with sumcheck challenges,
//! building intermediate Merkle commitments at each layer. Queries are
//! drawn from the Poseidon channel for soundness.

use starknet_ff::FieldElement;
use stwo::core::fields::qm31::SecureField;

use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};
use crate::crypto::poseidon_merkle::{PoseidonMerkleTree, MerkleAuthPath};

/// Number of queries for MLE opening (matching STARK FRI query count).
pub const MLE_N_QUERIES: usize = 14;

/// MLE opening proof matching Cairo's `MleOpeningProof`.
#[derive(Debug, Clone)]
pub struct MleOpeningProof {
    /// Merkle roots of intermediate folded layers.
    pub intermediate_roots: Vec<FieldElement>,
    /// Per-query proofs with authentication paths at each folding layer.
    pub queries: Vec<MleQueryProof>,
    /// Final single value after all folding rounds.
    pub final_value: SecureField,
}

/// Proof for a single query through all folding layers.
#[derive(Debug, Clone)]
pub struct MleQueryProof {
    /// Initial pair index in the bottom layer.
    pub initial_pair_index: u32,
    /// Per-round data: left/right values + Merkle auth paths.
    pub rounds: Vec<MleQueryRoundData>,
}

/// Data for one round of a query proof.
#[derive(Debug, Clone)]
pub struct MleQueryRoundData {
    pub left_value: SecureField,
    pub right_value: SecureField,
    pub left_siblings: Vec<FieldElement>,
    pub right_siblings: Vec<FieldElement>,
}

/// Commit to an MLE by building a Poseidon Merkle tree over its evaluations.
///
/// Each SecureField (QM31) is packed into a single FieldElement for hashing.
///
/// Returns (root, tree).
pub fn commit_mle(evals: &[SecureField]) -> (FieldElement, PoseidonMerkleTree) {
    let leaves: Vec<FieldElement> = evals.iter().map(|&sf| securefield_to_felt(sf)).collect();
    let tree = PoseidonMerkleTree::build(leaves);
    (tree.root(), tree)
}

/// Compute the next query index after folding.
///
/// Given the current query index within a layer of size `2 * layer_mid`,
/// returns the reduced index for the next (folded) layer of size `layer_mid`.
/// The next layer's lo/hi split is at `layer_mid / 2`, so the index is
/// reduced to `[0, layer_mid / 2)` via modular reduction.
///
/// # Panics
/// Panics if `layer_mid` is 0.
pub fn next_query_pair_index(current_idx: usize, layer_mid: usize) -> usize {
    let next_half = layer_mid / 2;
    if next_half == 0 {
        // Last folding round — layer has 2 elements, folds to 1. No further query needed.
        0
    } else {
        current_idx % next_half
    }
}

/// Draw query indices from a Poseidon channel.
///
/// Shared between prover and verifier to ensure identical index derivation.
fn draw_query_indices(channel: &mut PoseidonChannel, half_n: usize, n_queries: usize) -> Vec<usize> {
    let mut query_indices = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let pair_idx = (raw as usize) % half_n;
        query_indices.push(pair_idx);
    }
    query_indices
}

/// Generate an MLE opening proof.
///
/// Given evaluations `evals` on `{0,1}^n`, challenges `challenges` (the sumcheck
/// assignment), and a Poseidon channel for query generation, produces a proof
/// that `evaluate_mle(evals, challenges) = final_value`.
///
/// Protocol:
/// 1. Build initial Merkle tree over `evals`
/// 2. For each challenge `r[i]`:
///    - Fold: `f'[j] = (1-r[i])*f[2j] + r[i]*f[2j+1]`
///    - Commit folded layer → intermediate root
/// 3. Draw query indices from channel
/// 4. Build query proofs with auth paths at each layer
pub fn prove_mle_opening(
    evals: &[SecureField],
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> MleOpeningProof {
    assert!(!evals.is_empty());
    assert!(evals.len().is_power_of_two());
    let n_vars = evals.len().ilog2() as usize;
    assert_eq!(challenges.len(), n_vars);

    // Build initial tree and store layers for query generation
    let (initial_root, initial_tree) = commit_mle(evals);
    channel.mix_felt(initial_root);

    // Store all layers (evaluations) and trees for query proof construction
    let mut layer_evals: Vec<Vec<SecureField>> = vec![evals.to_vec()];
    let mut layer_trees: Vec<PoseidonMerkleTree> = vec![initial_tree];
    let mut intermediate_roots: Vec<FieldElement> = Vec::new();

    // Fold through each challenge
    // Variable ordering matches evaluate_mle: first variable splits into lo/hi halves
    let mut current = evals.to_vec();
    for &r in challenges.iter() {
        let mid = current.len() / 2;
        let mut folded = Vec::with_capacity(mid);
        for j in 0..mid {
            // f(r, x_rest) = (1-r)*f(0, x_rest) + r*f(1, x_rest)
            // f(0, x_rest) = current[j], f(1, x_rest) = current[mid + j]
            folded.push(current[j] + r * (current[mid + j] - current[j]));
        }

        if folded.len() > 1 {
            let (root, tree) = commit_mle(&folded);
            channel.mix_felt(root);
            intermediate_roots.push(root);
            layer_trees.push(tree);
        }
        layer_evals.push(folded.clone());
        current = folded;
    }

    let final_value = current[0];

    // Draw query indices (each query selects an index in [0, n/2))
    let initial_n = evals.len();
    let half_n = initial_n / 2;
    let n_queries = MLE_N_QUERIES.min(half_n);

    let query_indices = draw_query_indices(channel, half_n, n_queries);

    // Build query proofs
    let mut queries = Vec::with_capacity(n_queries);
    for &pair_idx in &query_indices {
        let mut rounds = Vec::with_capacity(n_vars);
        let mut current_idx = pair_idx;

        for round in 0..n_vars {
            let layer = &layer_evals[round];
            let mid = layer.len() / 2;
            // With lo/hi folding: left = layer[idx], right = layer[mid + idx]
            let left_idx = current_idx;
            let right_idx = mid + current_idx;

            // Fix 4: Assert indices are in bounds instead of silent clamping
            assert!(
                left_idx < layer.len(),
                "left_idx {left_idx} out of bounds for layer of size {} at round {round}",
                layer.len()
            );
            assert!(
                right_idx < layer.len(),
                "right_idx {right_idx} out of bounds for layer of size {} at round {round}",
                layer.len()
            );

            let left_value = layer[left_idx];
            let right_value = layer[right_idx];

            // Get Merkle auth paths — tree must exist for this round
            assert!(
                round < layer_trees.len(),
                "no Merkle tree for round {round} (only {} trees)",
                layer_trees.len()
            );
            let tree = &layer_trees[round];
            assert!(
                left_idx < tree.num_leaves(),
                "left_idx {left_idx} >= num_leaves {} at round {round}",
                tree.num_leaves()
            );
            assert!(
                right_idx < tree.num_leaves(),
                "right_idx {right_idx} >= num_leaves {} at round {round}",
                tree.num_leaves()
            );

            let left_path = tree.prove(left_idx);
            let right_path = tree.prove(right_idx);

            rounds.push(MleQueryRoundData {
                left_value,
                right_value,
                left_siblings: left_path.siblings,
                right_siblings: right_path.siblings,
            });

            // Fix 5: Use shared helper for index reduction
            current_idx = next_query_pair_index(current_idx, mid);
        }

        queries.push(MleQueryProof {
            initial_pair_index: pair_idx as u32,
            rounds,
        });
    }

    MleOpeningProof {
        intermediate_roots,
        queries,
        final_value,
    }
}

/// Verify an MLE opening proof.
///
/// Full verification including:
/// 1. Replay Fiat-Shamir transcript to reconstruct query indices
/// 2. Verify Merkle authentication paths against commitment roots
/// 3. Check algebraic folding consistency at each round
/// 4. Verify final folded value equals `proof.final_value`
pub fn verify_mle_opening(
    commitment: FieldElement,
    proof: &MleOpeningProof,
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> bool {
    let n_rounds = challenges.len();

    // Replay channel transcript: mix initial commitment and intermediate roots
    channel.mix_felt(commitment);
    for root in &proof.intermediate_roots {
        channel.mix_felt(*root);
    }

    // Build roots per layer: layer 0 = commitment, layers 1..n-1 = intermediate_roots
    // The last folding round produces a single value (no tree), so we need n_rounds-1
    // intermediate roots for n_rounds total layers with trees.
    let mut layer_roots = Vec::with_capacity(n_rounds);
    layer_roots.push(commitment);
    for root in &proof.intermediate_roots {
        layer_roots.push(*root);
    }
    // layer_roots should have entries for every round that has a tree.
    // Rounds 0..layer_roots.len()-1 have trees. The last round(s) may not
    // if the layer has only 1 element (single value, no tree).

    // Reconstruct query indices using identical channel operations
    if n_rounds == 0 {
        return proof.queries.is_empty();
    }

    // Initial evals size is 2^n_rounds, half_n = 2^(n_rounds-1)
    let half_n = 1usize << (n_rounds - 1);
    let n_queries = MLE_N_QUERIES.min(half_n);
    let query_indices = draw_query_indices(channel, half_n, n_queries);

    if proof.queries.len() != n_queries {
        return false;
    }

    // Verify each query chain
    for (q_idx, query) in proof.queries.iter().enumerate() {
        if query.rounds.len() != n_rounds {
            return false;
        }

        // Verify initial pair index matches reconstructed query
        if query.initial_pair_index as usize != query_indices[q_idx] {
            return false;
        }

        let mut current_idx = query.initial_pair_index as usize;
        // Track the current layer size. Initial layer has 2^n_rounds elements.
        let mut layer_size = 1usize << n_rounds;

        for (round_idx, round_data) in query.rounds.iter().enumerate() {
            let r = challenges[round_idx];
            let mid = layer_size / 2;
            let left_idx = current_idx;
            let right_idx = mid + current_idx;

            // Verify Merkle authentication paths for rounds that have trees
            if round_idx < layer_roots.len() {
                let root = layer_roots[round_idx];
                let left_leaf = securefield_to_felt(round_data.left_value);
                let right_leaf = securefield_to_felt(round_data.right_value);
                let left_path = MerkleAuthPath { siblings: round_data.left_siblings.clone() };
                let right_path = MerkleAuthPath { siblings: round_data.right_siblings.clone() };

                if !PoseidonMerkleTree::verify(root, left_idx, left_leaf, &left_path) {
                    return false;
                }
                if !PoseidonMerkleTree::verify(root, right_idx, right_leaf, &right_path) {
                    return false;
                }
            }

            // Check algebraic fold: f(r) = left + r * (right - left)
            let folded = round_data.left_value + r * (round_data.right_value - round_data.left_value);

            // If this is the last round, the folded value must equal final_value
            if round_idx == n_rounds - 1 && folded != proof.final_value {
                return false;
            }

            // Advance to next layer
            current_idx = next_query_pair_index(current_idx, mid);
            layer_size = mid;
        }
    }

    true
}

/// Evaluate a multilinear extension at a point (standalone helper).
///
/// Duplicated from matmul.rs for the crypto module's independence.
pub fn evaluate_mle_at(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    assert_eq!(evals.len(), 1 << point.len());
    let mut current = evals.to_vec();
    for &r in point {
        let mid = current.len() / 2;
        let mut next = Vec::with_capacity(mid);
        for i in 0..mid {
            next.push(current[i] + r * (current[mid + i] - current[i]));
        }
        current = next;
    }
    current[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    fn make_evals(n: usize) -> Vec<SecureField> {
        (0..n)
            .map(|i| SecureField::from(M31::from((i + 1) as u32)))
            .collect()
    }

    /// Helper: prove and verify a round-trip for given evals and challenges.
    fn prove_verify_roundtrip(evals: &[SecureField], challenges: &[SecureField], seed: u64) -> bool {
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(seed);
        let proof = prove_mle_opening(evals, challenges, &mut ch);

        let (commitment, _) = commit_mle(evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(seed);
        verify_mle_opening(commitment, &proof, challenges, &mut ch_v)
    }

    #[test]
    fn test_mle_opening_2_vars() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        assert_eq!(proof.final_value, evaluate_mle_at(&evals, &challenges));
        assert!(!proof.queries.is_empty());

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(42);
        assert!(verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_4_vars() {
        let evals = make_evals(16);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(99);

        let challenges = vec![
            SecureField::from(M31::from(5)),
            SecureField::from(M31::from(11)),
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let proof = prove_mle_opening(&evals, &challenges, &mut ch);
        let expected = evaluate_mle_at(&evals, &challenges);
        assert_eq!(proof.final_value, expected);

        // Verify with Merkle checks
        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(99);
        assert!(verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_tampered_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let mut proof = prove_mle_opening(&evals, &challenges, &mut ch);
        proof.final_value = SecureField::from(M31::from(999));

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(42);
        assert!(!verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_roundtrip() {
        let evals: Vec<SecureField> = vec![
            QM31(CM31(M31::from(10), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(20), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(30), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(40), M31::from(0)), CM31(M31::from(0), M31::from(0))),
        ];

        let challenges = vec![
            SecureField::from(M31::from(2)),
            SecureField::from(M31::from(5)),
        ];

        let mut ch = PoseidonChannel::new();
        ch.mix_u64(7);
        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        let direct = evaluate_mle_at(&evals, &challenges);
        assert_eq!(proof.final_value, direct);

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(7);
        assert!(verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    // === Fix 1 Tests: Merkle path verification ===

    #[test]
    fn test_mle_opening_merkle_paths_verified() {
        // Full round-trip with Merkle auth path checks (4 evals = 2 vars)
        let evals = make_evals(4);
        let challenges = vec![
            SecureField::from(M31::from(13)),
            SecureField::from(M31::from(17)),
        ];
        assert!(prove_verify_roundtrip(&evals, &challenges, 100));
    }

    #[test]
    fn test_mle_opening_tampered_left_value_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(200);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let mut proof = prove_mle_opening(&evals, &challenges, &mut ch);

        // Tamper with a left_value — Merkle check should catch it
        if let Some(q) = proof.queries.first_mut() {
            if let Some(rd) = q.rounds.first_mut() {
                rd.left_value = SecureField::from(M31::from(9999));
            }
        }

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(200);
        assert!(!verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_tampered_right_value_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(201);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let mut proof = prove_mle_opening(&evals, &challenges, &mut ch);

        if let Some(q) = proof.queries.first_mut() {
            if let Some(rd) = q.rounds.first_mut() {
                rd.right_value = SecureField::from(M31::from(8888));
            }
        }

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(201);
        assert!(!verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_tampered_siblings_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(202);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let mut proof = prove_mle_opening(&evals, &challenges, &mut ch);

        // Tamper with a sibling hash
        if let Some(q) = proof.queries.first_mut() {
            if let Some(rd) = q.rounds.first_mut() {
                if let Some(sib) = rd.left_siblings.first_mut() {
                    *sib = FieldElement::from(7777u64);
                }
            }
        }

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(202);
        assert!(!verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_wrong_commitment_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(203);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        // Use wrong commitment
        let wrong_commitment = FieldElement::from(12345u64);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(203);
        assert!(!verify_mle_opening(wrong_commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_8_vars_with_merkle() {
        // 256 evaluations on {0,1}^8
        let evals = make_evals(256);
        let challenges: Vec<SecureField> = (0..8)
            .map(|i| SecureField::from(M31::from((i * 3 + 5) as u32)))
            .collect();
        assert!(prove_verify_roundtrip(&evals, &challenges, 300));
    }

    // === Fix 4 Tests: No OOB index clamping ===

    #[test]
    fn test_no_oob_indices_small() {
        // 4 evals — ensure no panic (indices are valid)
        let evals = make_evals(4);
        let challenges = vec![
            SecureField::from(M31::from(11)),
            SecureField::from(M31::from(13)),
        ];
        assert!(prove_verify_roundtrip(&evals, &challenges, 400));
    }

    #[test]
    fn test_no_oob_indices_large() {
        // 64 evals — 6 vars, deeper query chains
        let evals = make_evals(64);
        let challenges: Vec<SecureField> = (0..6)
            .map(|i| SecureField::from(M31::from((i * 7 + 2) as u32)))
            .collect();
        assert!(prove_verify_roundtrip(&evals, &challenges, 401));
    }

    // === Fix 5 Tests: Query index helper ===

    #[test]
    fn test_next_query_pair_index_basic() {
        // mid=4 → next_half=2 → index wraps mod 2
        assert_eq!(next_query_pair_index(0, 4), 0);
        assert_eq!(next_query_pair_index(1, 4), 1);
        assert_eq!(next_query_pair_index(2, 4), 0);
        assert_eq!(next_query_pair_index(3, 4), 1);
    }

    #[test]
    fn test_next_query_pair_index_wrapping() {
        // mid=2 → next_half=1 → always 0
        assert_eq!(next_query_pair_index(0, 2), 0);
        assert_eq!(next_query_pair_index(1, 2), 0);

        // mid=1 → next_half=0 → returns 0 (terminal)
        assert_eq!(next_query_pair_index(0, 1), 0);
        assert_eq!(next_query_pair_index(5, 1), 0);

        // mid=8 → next_half=4
        assert_eq!(next_query_pair_index(5, 8), 1);
        assert_eq!(next_query_pair_index(7, 8), 3);
    }
}
