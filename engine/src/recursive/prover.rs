//! Recursive STARK prover — proves "I verified the GKR proof and it passed."
//!
//! This module wires together the witness generator and AIR circuit to produce
//! a recursive STARK proof using STWO's standard `prove()` function.
//!
//! # Pipeline
//!
//! ```text
//! GKRProof + Circuit + Output + Weights
//!     → generate_witness()     → GkrVerifierWitness
//!     → build_recursive_trace()→ RecursiveTraceData
//!     → commit traces          → CommitmentSchemeProver
//!     → stwo::prove()          → StarkProof
//!     → RecursiveProof
//! ```

/// Debug logging for recursive prover — only prints in debug builds.
macro_rules! recursive_log {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        eprintln!($($arg)*);
    };
}

use num_traits::Zero;
use starknet_ff::FieldElement;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::{SecureField, QM31};
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::poseidon252_merkle::{
    Poseidon252MerkleChannel, Poseidon252MerkleHasher,
};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use crate::backend::convert_evaluations;

use crate::compiler::graph::GraphWeights;
use crate::components::matmul::M31Matrix;
use crate::gkr::circuit::LayeredCircuit;
use crate::gkr::types::GKRProof;

use super::air::{
    build_arithmetic_trace, build_draw_trace, build_recursive_trace, build_sumcheck_trace,
    pad_arithmetic_trace_to_log_size, pad_draw_trace_to_log_size, pad_sumcheck_trace_to_log_size,
    RecursiveVerifierEval,
};
use super::types::{RecursiveProof, RecursiveProofMetadata, RecursivePublicInputs};
use super::witness::generate_witness_with_policy;

/// Error type for recursive proving.
#[derive(Debug)]
pub enum RecursiveError {
    /// The GKR proof failed verification (Pass 1).
    GkrVerificationFailed(String),
    /// Trace building failed.
    TraceBuildFailed(String),
    /// STWO proving failed.
    ProvingFailed(String),
}

impl std::fmt::Display for RecursiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecursiveError::GkrVerificationFailed(e) => write!(f, "GKR verification failed: {e}"),
            RecursiveError::TraceBuildFailed(e) => write!(f, "trace build failed: {e}"),
            RecursiveError::ProvingFailed(e) => write!(f, "recursive proving failed: {e}"),
        }
    }
}

impl std::error::Error for RecursiveError {}

/// Produce a recursive STARK proof for a GKR proof.
///
/// This is the main entry point for recursive composition. It:
/// 1. Generates the verifier witness (validates the GKR proof via production verifier)
/// 2. Builds the execution trace from the witness
/// 3. Commits the trace using STWO's commitment scheme
/// 4. Calls `stwo::prove()` to produce the recursive STARK proof
///
/// # Arguments
///
/// * `circuit` - The model's layered circuit
/// * `gkr_proof` - The GKR proof to verify recursively
/// * `output` - The model's output matrix
/// * `weights` - Model weights (needed for aggregated binding verification)
/// * `weight_super_root` - Poseidon root of all weight commitments
/// * `io_commitment` - Poseidon hash of packed IO
/// * `gkr_prove_time_secs` - Time taken to produce the GKR proof (for metadata)
///
/// # Returns
///
/// A `RecursiveProof` containing the STARK proof + public inputs.
/// On-chain, only this proof is submitted (not the original GKR proof).
pub fn prove_recursive(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    weight_super_root: QM31,
    io_commitment: QM31,
    gkr_prove_time_secs: f64,
) -> Result<RecursiveProof, RecursiveError> {
    // Legacy entrypoint: reconstructs felt252 from QM31 (lossy — 124 bits + sentinel).
    // Production callers should use prove_recursive_with_policy_and_io_felt()
    // and pass the original full Poseidon felt252 commitment.
    prove_recursive_with_policy(
        circuit,
        gkr_proof,
        output,
        weights,
        weight_super_root,
        io_commitment,
        gkr_prove_time_secs,
        None,
    )
}

/// Generate a recursive STARK proof with explicit policy binding.
pub fn prove_recursive_with_policy(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    weight_super_root: QM31,
    io_commitment: QM31,
    gkr_prove_time_secs: f64,
    policy: Option<&crate::policy::PolicyConfig>,
) -> Result<RecursiveProof, RecursiveError> {
    prove_recursive_with_policy_and_io_felt(
        circuit,
        gkr_proof,
        output,
        weights,
        weight_super_root,
        io_commitment,
        None,
        gkr_prove_time_secs,
        policy,
    )
}

/// Generate a recursive STARK proof with explicit policy and full felt252 IO binding.
pub fn prove_recursive_with_policy_and_io_felt(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    weight_super_root: QM31,
    io_commitment: QM31,
    io_commitment_felt252: Option<FieldElement>,
    gkr_prove_time_secs: f64,
    policy: Option<&crate::policy::PolicyConfig>,
) -> Result<RecursiveProof, RecursiveError> {
    prove_recursive_with_policy_io_and_statement(
        circuit,
        gkr_proof,
        output,
        weights,
        weight_super_root,
        io_commitment,
        io_commitment_felt252,
        None,
        gkr_prove_time_secs,
        policy,
    )
}

/// Generate a recursive STARK proof with explicit policy, full felt252 IO, and
/// optional conversation/action statement binding.
#[allow(clippy::too_many_arguments)]
pub fn prove_recursive_with_policy_io_and_statement(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    weight_super_root: QM31,
    io_commitment: QM31,
    io_commitment_felt252: Option<FieldElement>,
    conversation_statement_hash: Option<FieldElement>,
    gkr_prove_time_secs: f64,
    policy: Option<&crate::policy::PolicyConfig>,
) -> Result<RecursiveProof, RecursiveError> {
    let t_start = std::time::Instant::now();

    // ── Step 1: Generate witness ─────────────────────────────────────
    recursive_log!("  [Recursive] Step 1/5: Generating verifier witness...");
    let mut witness = generate_witness_with_policy(
        circuit,
        gkr_proof,
        output,
        Some(weights),
        weight_super_root,
        io_commitment,
        policy,
    )
    .map_err(|e| RecursiveError::GkrVerificationFailed(format!("{e:?}")))?;

    witness.public_inputs.conversation_statement_hash =
        conversation_statement_hash.unwrap_or(FieldElement::ZERO);

    recursive_log!(
        "  [Recursive] Witness: {} poseidon perms, {} sumcheck rounds, {} qm31 ops",
        witness.n_poseidon_perms,
        witness.n_sumcheck_rounds,
        witness.n_qm31_ops,
    );
    let unconstrained_verifier_checks =
        witness.n_sumcheck_rounds + witness.n_qm31_ops + witness.n_equality_checks;
    recursive_log!(
        "  [Recursive] Coverage: Hades transcript, primitive arithmetic, sumcheck rounds, channel draws, and LogUp bindings active; full verifier step-machine AIR is not yet implemented",
    );

    // ── Step 1b: Verify all Hades permutations offline ──────────────
    // This ensures every (input, output) pair in the witness is a valid
    // Hades permutation, providing soundness at the prover level even
    // before the Hades AIR is fully integrated into the multi-component STARK.
    let n_hades_verified = verify_hades_perms_offline(&witness).map_err(|e| {
        RecursiveError::GkrVerificationFailed(format!("Hades permutation check failed: {e}"))
    })?;
    recursive_log!(
        "  [Recursive] Verified {} Hades permutations offline",
        n_hades_verified
    );

    // Production recursive proofs must include the inline Hades AIR. The only
    // non-test escape hatch is the explicitly named legacy chain-only mode,
    // which should not be used for externally reviewed proof claims.
    #[cfg(not(test))]
    let hades_enabled = {
        let legacy_chain_only = std::env::var("OBELYZK_RECURSIVE_LEGACY_CHAIN_ONLY")
            .map(|v| v == "1")
            .unwrap_or(false);
        if legacy_chain_only {
            false
        } else {
            std::env::var("OBELYZK_HADES_AIR")
                .map(|v| v != "0")
                .unwrap_or(true)
        }
    };
    #[cfg(test)]
    let hades_enabled = std::env::var("OBELYZK_HADES_AIR")
        .map(|v| v == "1")
        .unwrap_or(false); // OFF by default in tests

    #[cfg(not(test))]
    if !hades_enabled {
        let legacy_chain_only = std::env::var("OBELYZK_RECURSIVE_LEGACY_CHAIN_ONLY")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !legacy_chain_only {
            return Err(RecursiveError::ProvingFailed(
                "production recursive proofs require Hades AIR; set OBELYZK_RECURSIVE_LEGACY_CHAIN_ONLY=1 only for legacy diagnostics".to_string(),
            ));
        }
    }

    // ── Step 2: Build traces ─────────────────────────────────────────
    recursive_log!("  [Recursive] Step 2/5: Building chain execution trace...");
    let mut trace_data = build_recursive_trace(&witness);
    let mut arithmetic_trace = build_arithmetic_trace(&witness);
    let arithmetic_enabled = arithmetic_trace.n_real_rows > 0;
    let mut sumcheck_trace = build_sumcheck_trace(&witness);
    let sumcheck_enabled = sumcheck_trace.n_real_rows > 0;
    let mut draw_trace = build_draw_trace(&witness);
    let draw_enabled = draw_trace.n_real_rows > 0;

    // Build Hades verification trace from HadesPerm witness ops
    let hades_perms = extract_hades_perms(&witness);
    let hades_trace = if hades_enabled {
        recursive_log!("  [Recursive] Step 2b/5: Building Hades verification trace...");
        super::hades_air::build_hades_trace(&hades_perms)
    } else {
        // Placeholder empty trace when Hades AIR is disabled
        super::hades_air::HadesTraceData {
            trace: Vec::new(),
            log_size: 1,
            n_real_rows: 0,
            n_perms: 0,
        }
    };

    recursive_log!(
        "  [Recursive] Trace: {} rows (log_size={}), {} cols/row, {} real rows",
        1u32 << trace_data.log_size,
        trace_data.log_size,
        super::air::COLS_PER_ROW,
        trace_data.n_real_rows,
    );
    if arithmetic_enabled {
        recursive_log!(
            "  [Recursive] Arithmetic trace: {} rows (log_size={}), {} cols/row, {} real rows",
            1u32 << arithmetic_trace.log_size,
            arithmetic_trace.log_size,
            super::air::ARITH_COLS_PER_ROW,
            arithmetic_trace.n_real_rows,
        );
    }
    if sumcheck_enabled {
        recursive_log!(
            "  [Recursive] Sumcheck trace: {} rows (log_size={}), {} cols/row, {} real rows",
            1u32 << sumcheck_trace.log_size,
            sumcheck_trace.log_size,
            super::air::SUMCHECK_COLS_PER_ROW,
            sumcheck_trace.n_real_rows,
        );
    }
    if draw_enabled {
        recursive_log!(
            "  [Recursive] Channel draw trace: {} rows (log_size={}), {} cols/row, {} real rows",
            1u32 << draw_trace.log_size,
            draw_trace.log_size,
            super::air::DRAW_COLS_PER_ROW,
            draw_trace.n_real_rows,
        );
    }

    // ── Step 3: Commit traces ────────────────────────────────────────
    recursive_log!("  [Recursive] Step 3/5: Committing traces...");
    // Security-hardened PCS config for recursive proofs.
    //
    // PcsConfig::default() gives only 13 bits of security
    // (pow_bits=10, log_blowup=1, n_queries=3).
    //
    // Production config: 160-bit target, matching `cairo-prove --recursive-160`.
    //   log_blowup=5  (32x blowup)
    //   n_queries=28
    //   pow_bits=20
    //
    // Test-only override via thread-local guard `RecursiveTestModeGuard::enter()`.
    // Production builds never enter the test path (the entire `#[cfg(test)]`
    // arm compiles out). The thread-local replaces a process-global env var
    // (`OBELYZK_RECURSIVE_SECURITY`) that previously raced across parallel
    // test threads.
    let config = {
        #[cfg(test)]
        let test_mode = super::recursive_test_mode_active();
        #[cfg(not(test))]
        let test_mode = false;
        if test_mode {
            #[cfg(test)]
            {
                PcsConfig::default()
            } // 13 bits — unit tests only
            #[cfg(not(test))]
            {
                unreachable!()
            }
        } else {
            PcsConfig {
                pow_bits: 20,
                // 160-bit target: pow(20) + blowup(5)*queries(28) = 20+140 = 160.
                // log_last_layer_degree_bound=0 required by Cairo FRI verifier.
                fri_config: stwo::core::fri::FriConfig::new(0, 5, 28, 1),
                lifting_log_size: None,
            }
        }
    };
    let chain_log_size = trace_data.log_size;
    let hades_log_size = hades_trace.log_size;
    let arithmetic_log_size = arithmetic_trace.log_size;
    let sumcheck_log_size = sumcheck_trace.log_size;
    let draw_log_size = draw_trace.log_size;
    // Twiddles computed after unified_log_size is known (below).

    // Use Poseidon252MerkleChannel so the STARK is verifiable by stwo-cairo-verifier
    // (Cairo's native Poseidon). This eliminates the need to constrain felt252 Hades
    // in the M31 AIR — the STARK proof itself uses Poseidon252 for Fiat-Shamir and
    // Merkle commitments, matching what the Cairo verifier expects.
    let channel = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();
    // Mix PcsConfig into channel BEFORE any tree commits.
    // MUST use config.mix_into() to match Cairo verifier's PcsConfig::mix_into
    // which packs fields into 2 QM31 values via mix_felts:
    //   QM31(pow_bits, log_blowup, n_queries, log_last_layer)
    //   QM31(fold_step, lifting_log_size.unwrap_or(0), 0, 0)
    recursive_log!(
        "  [Recursive] Channel after default: {:?}",
        channel.digest()
    );
    config.mix_into(channel);
    recursive_log!(
        "  [Recursive] Channel after PcsConfig: {:?}",
        channel.digest()
    );

    // ── Bind public inputs to Fiat-Shamir channel ────────────────────
    // By mixing circuit_hash, io_commitment, weight_super_root, and
    // n_layers into the channel BEFORE any tree commits, the STARK proof
    // becomes cryptographically bound to these values.  A verifier that
    // supplies different metadata will initialize a different channel
    // state, causing the FRI verification to fail.
    //
    // Order: [circuit_hash, io_commitment, weight_super_root] via
    // mix_felts (chunks-of-2 QM31 packing), then n_layers via mix_u64.
    // The Cairo verifier MUST replicate this exact sequence.
    channel.mix_felts(&[
        witness.public_inputs.circuit_hash,
        witness.public_inputs.io_commitment,
        witness.public_inputs.weight_super_root,
    ]);
    channel.mix_u64(witness.public_inputs.n_layers as u64);
    // SECURITY: n_poseidon_perms prevents trace miniaturization attack.
    channel.mix_u64(witness.public_inputs.n_poseidon_perms as u64);
    // SECURITY: bind the active row counts for every recursive AIR component.
    // These values are needed by the verifier to reconstruct accumulator
    // corrections and component enablement; mixing them prevents calldata
    // relabeling from relying only on a later AIR-shape mismatch.
    channel.mix_u64(trace_data.n_real_rows as u64);
    channel.mix_u64(arithmetic_trace.n_real_rows as u64);
    channel.mix_u64(sumcheck_trace.n_real_rows as u64);
    channel.mix_u64(draw_trace.n_real_rows as u64);
    // SECURITY: seed_digest checkpoint — binds chain content to model dimensions.
    channel.mix_felts(&[witness.public_inputs.seed_digest]);
    // SECURITY: hades_commitment — binds to Level 1 Hades recursive proof.
    // This is the Poseidon hash of all verified (input, output) Hades pairs.
    // Two-level recursion: the chain STARK transitively attests Hades correctness.
    {
        let bytes = witness.public_inputs.hades_commitment.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }

    // SECURITY: KV-cache continuity binding. Multi-token autoregressive sessions
    // require step N's input cache == step N-1's output cache. Mixing both
    // commitments here makes any tampering with the chain (reorder, skip a
    // step, swap caches) cause Fiat-Shamir divergence and FRI rejection.
    // Both are FieldElement::ZERO on prefill / single-pass / non-decode proofs.
    for fe in [
        witness.public_inputs.prev_kv_cache_commitment,
        witness.public_inputs.kv_cache_commitment,
        witness.public_inputs.conversation_statement_hash,
    ] {
        let bytes = fe.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }

    let io_felt252 = io_commitment_felt252.unwrap_or_else(|| {
        crate::crypto::poseidon_channel::securefield_to_felt(witness.public_inputs.io_commitment)
    });

    // Bind the felt252 io_commitment into the channel.
    // IMPORTANT: This MUST use the SAME value that ends up in the proof body
    // (io_commitment_felt252). The Cairo verifier reads this from the proof
    // and mixes it into its channel — both must match.
    // The default is the legacy lossy QM31→felt252 conversion. Production
    // callers pass the original Poseidon felt252 commitment explicitly.
    {
        let bytes = io_felt252.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }
    // Bind the Pass 1 digest into the Fiat-Shamir channel.
    // This binds the proof to the digest produced by the Rust verifier witness.
    // It is not, by itself, a proof that Pass 1 executed correctly; full trustless
    // verifier execution requires either complete verifier AIR or Cairo STARK-in-STARK.
    {
        let bytes = witness.final_digest.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }
    recursive_log!(
        "  [Recursive] Channel after public inputs: {:?}",
        channel.digest()
    );

    // When Hades AIR is enabled, use the SAME domain for both components
    // to avoid STWO's SIMD mixed-size column evaluation issues.
    // Chain columns are padded to unified_log_size (zeros beyond n_real_rows).
    // DEBUG: force unified to hades size but DO NOT pad chain — this tests
    // whether the padding recomputation is the issue.
    let unified_log_size = if hades_enabled {
        chain_log_size
            .max(hades_log_size)
            .max(if arithmetic_enabled {
                arithmetic_log_size
            } else {
                0
            })
            .max(if sumcheck_enabled {
                sumcheck_log_size
            } else {
                0
            })
            .max(if draw_enabled { draw_log_size } else { 0 })
    } else {
        // DEBUG: test chain at forced larger size
        let forced = std::env::var("OBELYZK_FORCE_CHAIN_LOG")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(chain_log_size);
        chain_log_size
            .max(forced)
            .max(if arithmetic_enabled {
                arithmetic_log_size
            } else {
                0
            })
            .max(if sumcheck_enabled {
                sumcheck_log_size
            } else {
                0
            })
            .max(if draw_enabled { draw_log_size } else { 0 })
    };
    eprintln!(
        "  [SIZES] chain_log={}, hades_log={}, arithmetic_log={}, sumcheck_log={}, draw_log={}, unified={}",
        chain_log_size, hades_log_size, arithmetic_log_size, sumcheck_log_size, draw_log_size, unified_log_size
    );
    // Must match RecursiveVerifierEval::max_constraint_log_degree_bound().
    let max_degree_bound = unified_log_size + 1;
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();

    let chain_domain = CanonicCoset::new(unified_log_size).circle_domain();
    let hades_domain = CanonicCoset::new(unified_log_size).circle_domain();
    let arithmetic_domain = CanonicCoset::new(unified_log_size).circle_domain();
    let sumcheck_domain = CanonicCoset::new(unified_log_size).circle_domain();
    let draw_domain = CanonicCoset::new(unified_log_size).circle_domain();

    // Tree 0: Preprocessed columns (is_first, is_last, is_chain)
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        // Pad preprocessed columns to unified_log_size if needed
        let pad_to = 1 << unified_log_size;
        let mut is_first = trace_data.preprocessed_is_first.clone();
        let mut is_last = trace_data.preprocessed_is_last.clone();
        let mut is_chain = trace_data.preprocessed_is_chain.clone();
        is_first.resize(pad_to, M31::from_u32_unchecked(0));
        is_last.resize(pad_to, M31::from_u32_unchecked(0));
        is_chain.resize(pad_to, M31::from_u32_unchecked(0));
        let is_first_col = simd_column_from_vec(&is_first);
        let is_last_col = simd_column_from_vec(&is_last);
        let is_chain_col = simd_column_from_vec(&is_chain);
        let simd_evals = vec![
            CircleEvaluation::new(chain_domain, is_first_col),
            CircleEvaluation::new(chain_domain, is_last_col),
            CircleEvaluation::new(chain_domain, is_chain_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
            simd_evals,
        ));
        tree_builder.commit(channel);
        recursive_log!(
            "  [Recursive] Channel after preprocessed commit: {:?}",
            channel.digest()
        );
    }

    // Tree 1: All execution traces (chain + Hades in same tree, mixed sizes)
    // STWO's tree builder supports mixed-size columns within one commit.
    {
        let mut tree_builder = commitment_scheme.tree_builder();

        // Recompute accumulator for unified_log_size if chain was built at a smaller size.
        // The amortized accumulator correction depends on N = 2^log_size. If the chain
        // trace was built at chain_log_size but we're committing at unified_log_size
        // (which may be larger due to Hades AIR), the accumulator must be recomputed.
        if unified_log_size > chain_log_size {
            let unified_n = 1usize << unified_log_size;
            let n_real = trace_data.n_real_rows;
            let col_is_active = super::air::CHAIN_COL_IS_ACTIVE;
            let col_ac = super::air::CHAIN_COL_ACTIVE_COUNT;
            let col_ac_next = super::air::CHAIN_COL_ACTIVE_COUNT_NEXT;
            let col_shifted = super::air::CHAIN_COL_SHIFTED_NEXT_BEFORE;
            let col_digest_before = super::air::CHAIN_COL_DIGEST_BEFORE;
            let col_digest_after = super::air::CHAIN_COL_DIGEST_AFTER;
            let col_addition = super::air::CHAIN_COL_ADDITION_DIGEST;
            let col_carry_pos = super::air::CHAIN_COL_CARRY_POS;
            let col_carry_neg = super::air::CHAIN_COL_CARRY_NEG;
            let col_k = super::air::CHAIN_COL_ADDITION_K;
            let col_draw_count = super::air::CHAIN_COL_DRAW_COUNT;
            let col_shifted_next_draw_count = super::air::CHAIN_COL_SHIFTED_NEXT_DRAW_COUNT;

            // Pad execution columns to unified size first
            for col in trace_data.execution_trace.iter_mut() {
                col.resize(unified_n, M31::from_u32_unchecked(0));
            }

            // Recompute shifted_next_before with wrap-around for unified domain
            for i in 0..unified_n {
                let next = (i + 1) % unified_n;
                for j in 0..super::air::LIMBS_PER_FELT {
                    trace_data.execution_trace[col_shifted + j][i] =
                        trace_data.execution_trace[col_digest_before + j][next];
                }
            }

            // Recompute accumulator with unified N
            let n_m31 = M31::from(unified_n as u32);
            let n_inv = n_m31.inverse();
            let correction = M31::from(n_real as u32) * n_inv;

            trace_data.execution_trace[col_ac][0] = M31::from_u32_unchecked(0);
            for i in 0..unified_n - 1 {
                let is_act = trace_data.execution_trace[col_is_active][i];
                trace_data.execution_trace[col_ac][i + 1] =
                    trace_data.execution_trace[col_ac][i] + is_act - correction;
            }
            for i in 0..unified_n {
                let next = (i + 1) % unified_n;
                trace_data.execution_trace[col_ac_next][i] =
                    trace_data.execution_trace[col_ac][next];
            }

            let mut draw_count = M31::from_u32_unchecked(0);
            for i in 0..n_real.min(unified_n) {
                trace_data.execution_trace[col_draw_count][i] = draw_count;
                if trace_data.execution_trace[super::air::CHAIN_COL_IS_DRAW][i]
                    == M31::from_u32_unchecked(1)
                {
                    draw_count += M31::from_u32_unchecked(1);
                }
            }
            for i in 0..unified_n {
                let next = (i + 1) % unified_n;
                trace_data.execution_trace[col_shifted_next_draw_count][i] =
                    trace_data.execution_trace[col_draw_count][next];
            }

            // Recompute carry chain for unified domain (only chain rows need carries)
            for row_idx in 0..n_real.saturating_sub(1) {
                let da_limbs: [M31; super::air::LIMBS_PER_FELT] = std::array::from_fn(|j| {
                    trace_data.execution_trace[col_digest_after + j][row_idx]
                });
                let add_limbs: [M31; super::air::LIMBS_PER_FELT] =
                    std::array::from_fn(|j| trace_data.execution_trace[col_addition + j][row_idx]);
                let next_before_limbs: [M31; super::air::LIMBS_PER_FELT] =
                    std::array::from_fn(|j| {
                        trace_data.execution_trace[col_digest_before + j][row_idx + 1]
                    });
                let (carry_pos, carry_neg, k) = super::air::compute_addition_carry_chain(
                    &da_limbs,
                    &add_limbs,
                    &next_before_limbs,
                );
                for j in 0..8 {
                    trace_data.execution_trace[col_carry_pos + j][row_idx] = carry_pos[j];
                    trace_data.execution_trace[col_carry_neg + j][row_idx] = carry_neg[j];
                }
                trace_data.execution_trace[col_k][row_idx] = k;
            }

            recursive_log!(
                "  [Recursive] Recomputed selectors/accumulator for unified_log_size={} (was chain_log_size={})",
                unified_log_size, chain_log_size
            );
        }
        if arithmetic_enabled {
            pad_arithmetic_trace_to_log_size(&mut arithmetic_trace, unified_log_size);
        }
        if sumcheck_enabled {
            pad_sumcheck_trace_to_log_size(&mut sumcheck_trace, unified_log_size);
        }
        if draw_enabled {
            pad_draw_trace_to_log_size(&mut draw_trace, unified_log_size);
        }

        // Chain columns, padded to unified_log_size.
        let chain_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = trace_data
            .execution_trace
            .iter()
            .map(|col| {
                let padded = if col.len() < (1 << unified_log_size) {
                    let mut p = col.clone();
                    p.resize(1 << unified_log_size, M31::from_u32_unchecked(0));
                    p
                } else {
                    col.clone()
                };
                let simd_col = simd_column_from_vec(&padded);
                CircleEvaluation::new(chain_domain, simd_col)
            })
            .collect();
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
            chain_evals,
        ));

        // Hades columns in same tree as chain
        if hades_enabled {
            let hades_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = hades_trace
                .trace
                .iter()
                .map(|col| {
                    let simd_col = simd_column_from_vec(col);
                    CircleEvaluation::new(hades_domain, simd_col)
                })
                .collect();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
                hades_evals,
            ));
        }

        if arithmetic_enabled {
            let arithmetic_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = arithmetic_trace
                .trace
                .iter()
                .map(|col| {
                    let simd_col = simd_column_from_vec(col);
                    CircleEvaluation::new(arithmetic_domain, simd_col)
                })
                .collect();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
                arithmetic_evals,
            ));
        }

        if sumcheck_enabled {
            let sumcheck_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = sumcheck_trace
                .trace
                .iter()
                .map(|col| {
                    let simd_col = simd_column_from_vec(col);
                    CircleEvaluation::new(sumcheck_domain, simd_col)
                })
                .collect();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
                sumcheck_evals,
            ));
        }

        if draw_enabled {
            let draw_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = draw_trace
                .trace
                .iter()
                .map(|col| {
                    let simd_col = simd_column_from_vec(col);
                    CircleEvaluation::new(draw_domain, simd_col)
                })
                .collect();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
                draw_evals,
            ));
        }

        tree_builder.commit(channel);
        recursive_log!(
            "  [Recursive] Channel after trace commit: {:?}",
            channel.digest()
        );
    }

    // ── Step 3c: LogUp interaction trace ─────────────────────────────
    // Production path: always bind chain rows to Hades provider rows via LogUp.
    // Tests keep it opt-in because the full Hades AIR is intentionally heavy.
    // Legacy chain-only diagnostics must opt out explicitly through
    // OBELYZK_RECURSIVE_LEGACY_CHAIN_ONLY=1.
    #[cfg(test)]
    let logup_requested = std::env::var("OBELYZK_LOGUP")
        .map(|v| v == "1")
        .unwrap_or(false);
    #[cfg(not(test))]
    let logup_requested = if std::env::var("OBELYZK_RECURSIVE_LEGACY_CHAIN_ONLY")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        false
    } else {
        std::env::var("OBELYZK_LOGUP")
            .map(|v| v != "0")
            .unwrap_or(true)
    };
    if logup_requested && !hades_enabled {
        return Err(RecursiveError::ProvingFailed(
            "Hades LogUp requires OBELYZK_HADES_AIR=1".to_string(),
        ));
    }
    #[cfg(not(test))]
    if hades_enabled && !logup_requested {
        return Err(RecursiveError::ProvingFailed(
            "production recursive proofs require Hades LogUp; set OBELYZK_RECURSIVE_LEGACY_CHAIN_ONLY=1 only for legacy diagnostics".to_string(),
        ));
    }
    let logup_enabled = logup_requested && hades_enabled;

    let (logup_relation, draw_felt_relation, challenge_relation, chain_claimed_sum) =
        if logup_enabled {
            use num_traits::One;
            use stwo::prover::backend::simd::m31::N_LANES;
            use stwo::prover::backend::simd::qm31::PackedSecureField;
            use stwo_constraint_framework::{LogupTraceGenerator, Relation};

            // Draw shared LogUp random elements from channel
            let relation = super::air::HadesPermRelation::draw(channel);
            let draw_felt_relation = if draw_enabled {
                Some(super::air::DrawFeltRelation::draw(channel))
            } else {
                None
            };
            let challenge_relation = if sumcheck_enabled && draw_enabled {
                Some(super::air::SumcheckChallengeRelation::draw(channel))
            } else {
                None
            };
            recursive_log!("  [Recursive] LogUp relation drawn from channel");

            // Generate Hades provider interaction trace first, matching
            // RecursiveVerifierEval's add_to_relation order.
            let mut logup_gen = LogupTraceGenerator::new(unified_log_size);
            {
                let mut col = logup_gen.new_col();
                let n_total = 1usize << unified_log_size;
                let n_vec_rows = n_total / N_LANES;
                for vec_row in 0..n_vec_rows {
                    let mut nums = [SecureField::zero(); N_LANES];
                    let mut denoms = [SecureField::one(); N_LANES];
                    for lane in 0..N_LANES {
                        let row = vec_row * N_LANES + lane;
                        if row < hades_trace.trace[0].len()
                            && hades_trace.trace[super::hades_air::HADES_IS_LAST_ROUND_COL][row]
                                == M31::from_u32_unchecked(1)
                        {
                            nums[lane] = SecureField::zero() - SecureField::one();
                            let key_vals: Vec<M31> = (0..9)
                                .map(|j| {
                                    hades_trace.trace
                                        [super::hades_air::HADES_INPUT_DIGEST_28BIT_COL + j][row]
                                })
                                .chain((0..9).map(|j| {
                                    hades_trace.trace
                                        [super::hades_air::HADES_OUTPUT_DIGEST_28BIT_COL + j][row]
                                }))
                                .collect();
                            denoms[lane] = relation.combine(&key_vals);
                        }
                    }
                    col.write_frac(
                        vec_row,
                        PackedSecureField::from_array(nums),
                        PackedSecureField::from_array(denoms),
                    );
                }
                col.finalize_col();
            }

            if let Some(ref draw_relation) = challenge_relation {
                // Sumcheck rows consume the challenge multiset (+1). This order
                // matches RecursiveVerifierEval::evaluate().
                {
                    let mut col = logup_gen.new_col();
                    let n_total = 1usize << unified_log_size;
                    let n_vec_rows = n_total / N_LANES;
                    for vec_row in 0..n_vec_rows {
                        let mut nums = [SecureField::zero(); N_LANES];
                        let mut denoms = [SecureField::one(); N_LANES];
                        for lane in 0..N_LANES {
                            let row = vec_row * N_LANES + lane;
                            if row < sumcheck_trace.n_real_rows {
                                nums[lane] = SecureField::one();
                                let mut key_vals: Vec<M31> = (0..super::air::SUMCHECK_VALUE_LIMBS)
                                    .map(|j| {
                                        sumcheck_trace.trace[super::air::SUMCHECK_COL_CHALLENGE + j]
                                            [row]
                                    })
                                    .collect();
                                key_vals.push(
                                    sumcheck_trace.trace
                                        [super::air::SUMCHECK_COL_CHALLENGE_DRAW_INDEX][row],
                                );
                                denoms[lane] = draw_relation.combine(&key_vals);
                            }
                        }
                        col.write_frac(
                            vec_row,
                            PackedSecureField::from_array(nums),
                            PackedSecureField::from_array(denoms),
                        );
                    }
                    col.finalize_col();
                }
            }

            if let Some(ref draw_felt_relation) = draw_felt_relation {
                // Draw rows consume the raw felt emitted by the matching Hades draw row (+1).
                {
                    let mut col = logup_gen.new_col();
                    let n_total = 1usize << unified_log_size;
                    let n_vec_rows = n_total / N_LANES;
                    for vec_row in 0..n_vec_rows {
                        let mut nums = [SecureField::zero(); N_LANES];
                        let mut denoms = [SecureField::one(); N_LANES];
                        for lane in 0..N_LANES {
                            let row = vec_row * N_LANES + lane;
                            if row < draw_trace.n_real_rows {
                                nums[lane] = SecureField::one();
                                let mut key_vals: Vec<M31> = (0..super::air::DRAW_RAW_FELT_LIMBS)
                                    .map(|j| {
                                        draw_trace.trace[super::air::DRAW_COL_RAW_FELT + j][row]
                                    })
                                    .collect();
                                key_vals
                                    .push(draw_trace.trace[super::air::DRAW_COL_DRAW_INDEX][row]);
                                denoms[lane] = draw_felt_relation.combine(&key_vals);
                            }
                        }
                        col.write_frac(
                            vec_row,
                            PackedSecureField::from_array(nums),
                            PackedSecureField::from_array(denoms),
                        );
                    }
                    col.finalize_col();
                }
            }

            if let Some(ref draw_relation) = challenge_relation {
                // Recorded channel draws provide the challenge multiset (-1).
                {
                    let mut col = logup_gen.new_col();
                    let n_total = 1usize << unified_log_size;
                    let n_vec_rows = n_total / N_LANES;
                    for vec_row in 0..n_vec_rows {
                        let mut nums = [SecureField::zero(); N_LANES];
                        let mut denoms = [SecureField::one(); N_LANES];
                        for lane in 0..N_LANES {
                            let row = vec_row * N_LANES + lane;
                            if row < draw_trace.n_real_rows {
                                nums[lane] = SecureField::zero() - SecureField::one();
                                let mut key_vals: Vec<M31> = (0..super::air::DRAW_VALUE_LIMBS)
                                    .map(|j| draw_trace.trace[super::air::DRAW_COL_VALUE + j][row])
                                    .collect();
                                key_vals
                                    .push(draw_trace.trace[super::air::DRAW_COL_DRAW_INDEX][row]);
                                denoms[lane] = draw_relation.combine(&key_vals);
                            }
                        }
                        col.write_frac(
                            vec_row,
                            PackedSecureField::from_array(nums),
                            PackedSecureField::from_array(denoms),
                        );
                    }
                    col.finalize_col();
                }
            }

            if let Some(ref draw_felt_relation) = draw_felt_relation {
                // Chain draw Hades rows provide the raw output felt consumed by draw rows (-1).
                {
                    let mut col = logup_gen.new_col();
                    let n_total = 1usize << unified_log_size;
                    let n_vec_rows = n_total / N_LANES;
                    for vec_row in 0..n_vec_rows {
                        let mut nums = [SecureField::zero(); N_LANES];
                        let mut denoms = [SecureField::one(); N_LANES];
                        for lane in 0..N_LANES {
                            let row = vec_row * N_LANES + lane;
                            if row < trace_data.n_real_rows
                                && trace_data.execution_trace[super::air::CHAIN_COL_IS_DRAW][row]
                                    == M31::from_u32_unchecked(1)
                            {
                                nums[lane] = SecureField::zero() - SecureField::one();
                                let mut key_vals: Vec<M31> = (0..super::air::LIMBS_PER_FELT)
                                    .map(|j| {
                                        trace_data.execution_trace
                                            [super::air::CHAIN_COL_DIGEST_AFTER + j][row]
                                    })
                                    .collect();
                                key_vals.push(
                                    trace_data.execution_trace[super::air::CHAIN_COL_DRAW_COUNT]
                                        [row],
                                );
                                denoms[lane] = draw_felt_relation.combine(&key_vals);
                            }
                        }
                        col.write_frac(
                            vec_row,
                            PackedSecureField::from_array(nums),
                            PackedSecureField::from_array(denoms),
                        );
                    }
                    col.finalize_col();
                }
            }

            // Generate chain consumer interaction trace (+1 per active HadesPerm row).
            {
                let mut col = logup_gen.new_col();
                let n_total = 1usize << unified_log_size;
                let n_vec_rows = n_total / N_LANES;
                for vec_row in 0..n_vec_rows {
                    let mut nums = [SecureField::zero(); N_LANES];
                    let mut denoms = [SecureField::one(); N_LANES];
                    for lane in 0..N_LANES {
                        let row = vec_row * N_LANES + lane;
                        if row < trace_data.n_real_rows {
                            nums[lane] = SecureField::one();
                            let key_vals: Vec<M31> = (0..super::air::LIMBS_PER_FELT)
                                .map(|j| trace_data.execution_trace[j][row])
                                .chain((0..super::air::LIMBS_PER_FELT).map(|j| {
                                    trace_data.execution_trace
                                        [super::air::CHAIN_COL_DIGEST_AFTER + j][row]
                                }))
                                .collect();
                            denoms[lane] = relation.combine(&key_vals);
                        }
                    }
                    col.write_frac(
                        vec_row,
                        PackedSecureField::from_array(nums),
                        PackedSecureField::from_array(denoms),
                    );
                }
                col.finalize_col();
            }
            let (interaction_trace, claimed_sum) = logup_gen.finalize_last();

            // Commit interaction trace as Tree 2
            {
                let mut tree_builder = commitment_scheme.tree_builder();
                tree_builder.extend_evals(interaction_trace);
                tree_builder.commit(channel);
            }
            recursive_log!(
                "  [Recursive] LogUp interaction committed (claimed_sum={:?})",
                claimed_sum
            );

            (
                Some(relation),
                draw_felt_relation,
                challenge_relation,
                claimed_sum,
            )
        } else {
            (None, None, None, SecureField::zero())
        };

    recursive_log!(
        "  [Recursive] Channel before prove(): {:?}",
        channel.digest()
    );

    // ── Step 4: Prove ────────────────────────────────────────────────
    recursive_log!("  [Recursive] Step 4/5: Proving (STARK)...");

    // Compute initial/final digest limbs.
    // Initial = zero (fresh channel).
    // Final = digest_after of the last recorded channel operation. Draws run a
    // Hades permutation but do not update the channel digest, so using the last
    // raw Hades output would be wrong whenever the transcript ends with a draw.
    let zero_limbs = super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);

    let last_channel_digest = witness.ops.iter().rev().find_map(|op| {
        if let super::types::WitnessOp::ChannelOp { digest_after, .. } = op {
            Some(*digest_after)
        } else {
            None
        }
    });

    let final_digest_felt = last_channel_digest.unwrap_or(starknet_ff::FieldElement::ZERO);

    if final_digest_felt != witness.final_digest {
        let recorded_channel_ops = witness
            .ops
            .iter()
            .filter(|op| matches!(op, super::types::WitnessOp::ChannelOp { .. }))
            .count();
        recursive_log!(
            "  [Recursive] NOTE: Pass 2 final digest differs from Pass 1 \
             ({} recorded ops vs {} total Poseidon calls). \
             Chain covers the instrumented subset.",
            recorded_channel_ops,
            witness.n_poseidon_perms,
        );

        #[cfg(not(test))]
        {
            let allow_partial = std::env::var("OBELYZK_RECURSIVE_ALLOW_PARTIAL_INSTRUMENTATION")
                .map(|v| v == "1")
                .unwrap_or(false);
            if !allow_partial {
                return Err(RecursiveError::ProvingFailed(format!(
                    "recursive witness is partial: instrumented final digest {final_digest_felt:?} \
                     does not match production verifier final digest {:?} \
                     ({} recorded channel ops vs {} production Poseidon calls). \
                     Full STARK-in-STARK production proving requires instrumenting the complete \
                     verifier path; set OBELYZK_RECURSIVE_ALLOW_PARTIAL_INSTRUMENTATION=1 only \
                     for legacy diagnostics.",
                    witness.final_digest, recorded_channel_ops, witness.n_poseidon_perms
                )));
            }
        }
    }

    let final_limbs = super::air::felt252_to_limbs(&final_digest_felt);

    // Print limbs for Cairo comparison
    recursive_log!(
        "  [Recursive] Initial limbs: {:?}",
        zero_limbs.iter().map(|l| l.0).collect::<Vec<_>>()
    );
    recursive_log!(
        "  [Recursive] Final limbs: {:?}",
        final_limbs.iter().map(|l| l.0).collect::<Vec<_>>()
    );

    recursive_log!(
        "  [Recursive] Final digest: {:?} (production: {:?}, match: {})",
        final_digest_felt,
        witness.final_digest,
        final_digest_felt == witness.final_digest,
    );

    // Create both AIR components with shared allocator
    let mut allocator = TraceLocationAllocator::default();

    // Component 1: Chain AIR (digest chain + boundary constraints + LogUp consumer)
    let chain_eval = RecursiveVerifierEval {
        log_n_rows: unified_log_size,
        n_real_rows: trace_data.n_real_rows as u32,
        initial_digest_limbs: zero_limbs,
        final_digest_limbs: final_limbs,
        hades_lookup: logup_relation.clone(),
        draw_felt_lookup: draw_felt_relation.clone(),
        challenge_lookup: challenge_relation.clone(),
        hades_enabled, // true = chain + Hades, false = chain-only
        arithmetic_enabled,
        n_arithmetic_rows: arithmetic_trace.n_real_rows as u32,
        sumcheck_enabled,
        n_sumcheck_rows: sumcheck_trace.n_real_rows as u32,
        draw_enabled,
        n_draw_rows: draw_trace.n_real_rows as u32,
    };
    let chain_component = FrameworkComponent::new(&mut allocator, chain_eval, chain_claimed_sum);

    use stwo::core::air::Component;
    let bounds = Component::trace_log_degree_bounds(&chain_component);
    eprintln!(
        "  [Recursive] Chain component: {} constraints, {} trees, bounds: {:?}",
        chain_component.n_constraints(),
        bounds.len(),
        bounds.iter().map(|t| t.len()).collect::<Vec<_>>(),
    );

    // Merged Hades AIR is evaluated inside the chain component when enabled.
    // Its provider rows are already included in the shared LogUp interaction
    // trace, so the committed trace proves both the Hades computation and the
    // chain↔Hades multiset equality in one active proof.
    let diag_enabled = std::env::var("OBELYZK_RECURSIVE_DIAG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Diagnostic: check Hades trace basic validity
    if hades_enabled && diag_enabled {
        let n_hades_padded = 1usize << unified_log_size;
        let n_hades_real = hades_trace.n_real_rows;
        // Find is_real column by scanning for the boolean pattern
        let mut is_real_col = 0;
        for c in 0..hades_trace.trace.len() {
            let mut is_bool = true;
            for r in 0..n_hades_real.min(10) {
                let v = hades_trace.trace[c][r].0;
                if v != 0 && v != 1 {
                    is_bool = false;
                    break;
                }
            }
            // Check: all real rows are 1, all padding rows are 0
            if is_bool {
                let all_real_one = (0..n_hades_real.min(5)).all(|r| hades_trace.trace[c][r].0 == 1);
                let all_pad_zero = if n_hades_padded > n_hades_real {
                    (n_hades_real..n_hades_padded.min(n_hades_real + 5))
                        .all(|r| hades_trace.trace[c][r].0 == 0)
                } else {
                    true
                };
                if all_real_one && all_pad_zero {
                    is_real_col = c;
                    break;
                }
            }
        }
        eprintln!("  [HADES DIAG] Found is_real at column {}", is_real_col);
        let mut bad_rows = 0;
        for row in 0..n_hades_padded.min(hades_trace.trace[0].len()) {
            let is_real = hades_trace.trace[is_real_col][row].0;
            let expected = if row < n_hades_real { 1 } else { 0 };
            if is_real != expected {
                if bad_rows < 3 {
                    eprintln!(
                        "  [HADES DIAG] is_real mismatch at row {}: got {} expected {}",
                        row, is_real, expected
                    );
                }
                bad_rows += 1;
            }
        }
        if bad_rows > 0 {
            eprintln!(
                "  [HADES DIAG] {} is_real mismatches out of {} rows",
                bad_rows, n_hades_padded
            );
        } else {
            eprintln!(
                "  [HADES DIAG] is_real check OK ({} real, {} padded)",
                n_hades_real, n_hades_padded
            );
        }
        eprintln!(
            "  [HADES DIAG] trace cols: {}, expected: {}",
            hades_trace.trace.len(),
            super::hades_air::N_HADES_TRACE_COLUMNS
        );
        eprintln!(
            "  [HADES DIAG] hades_log_size: {}, unified: {}, rows: {}/{}",
            hades_log_size, unified_log_size, n_hades_real, n_hades_padded
        );

        // Row-by-row constraint check
        let (fails, first_fail) = super::hades_air::check_hades_constraints_rowwise(
            &hades_trace.trace,
            n_hades_real,
            n_hades_padded,
        );
        if fails > 0 {
            eprintln!("  [HADES DIAG] CONSTRAINT FAILURES: {} total", fails);
            eprintln!("  [HADES DIAG] First: {}", first_fail);
        } else {
            eprintln!(
                "  [HADES DIAG] All row-by-row constraints PASS ({} rows checked)",
                n_hades_padded
            );
        }

        // Check cube constraint on row 0: cube_result[2] = sbox_input[2]³
        // sbox_input starts at col 84, element 2 at col 84+56..84+84
        // cube_result starts at col 168, element 2 at col 168+56..168+84
        if n_hades_real > 0 {
            let row = 0;
            let sbox_in_2: Vec<u32> = (0..28)
                .map(|j| hades_trace.trace[84 + 56 + j][row].0)
                .collect();
            let cube_out_2: Vec<u32> = (0..28)
                .map(|j| hades_trace.trace[168 + 56 + j][row].0)
                .collect();

            // Reconstruct felt252 from 9-bit limbs
            let sbox_felt = super::hades_air::limbs_9bit_to_felt252(
                &sbox_in_2
                    .iter()
                    .map(|v| M31::from_u32_unchecked(*v))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            );
            let cube_felt = super::hades_air::limbs_9bit_to_felt252(
                &cube_out_2
                    .iter()
                    .map(|v| M31::from_u32_unchecked(*v))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            );
            let expected_cube = sbox_felt * sbox_felt * sbox_felt;
            eprintln!(
                "  [HADES DIAG] Row 0 cube check: sbox_in[2]³ == cube_out[2]? {}",
                expected_cube == cube_felt
            );
            if expected_cube != cube_felt {
                eprintln!("    sbox_in[2]  = {:?}", sbox_felt);
                eprintln!("    cube_out[2] = {:?}", cube_felt);
                eprintln!("    expected    = {:?}", expected_cube);
            }
        }
    }

    // Optional diagnostic: verify chain constraints before proving.
    if diag_enabled {
        let n_real = trace_data.n_real_rows;
        let n_padded = 1usize << unified_log_size;
        let col_digest_after = super::air::CHAIN_COL_DIGEST_AFTER;
        let col_shifted = super::air::CHAIN_COL_SHIFTED_NEXT_BEFORE;
        let col_addition = super::air::CHAIN_COL_ADDITION_DIGEST;
        let col_carry_pos = super::air::CHAIN_COL_CARRY_POS;
        let col_carry_neg = super::air::CHAIN_COL_CARRY_NEG;
        let col_k_idx = super::air::CHAIN_COL_ADDITION_K;
        let col_is_active_offset = super::air::CHAIN_COL_IS_ACTIVE;
        let col_ac = super::air::CHAIN_COL_ACTIVE_COUNT;
        let col_ac_next = super::air::CHAIN_COL_ACTIVE_COUNT_NEXT;

        let mut chain_failures = 0usize;
        let mut failed_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
        // Signed carry at limb j = pos[j] - neg[j] ∈ {-1, 0, 1}.
        let signed_carry = |i: usize, j: usize| -> i64 {
            let pos = trace_data.execution_trace[col_carry_pos + j][i].0 as i64;
            let neg = trace_data.execution_trace[col_carry_neg + j][i].0 as i64;
            pos - neg
        };
        for i in 0..n_real.saturating_sub(1) {
            for j in 0..super::air::LIMBS_PER_FELT {
                let da = trace_data.execution_trace[col_digest_after + j][i].0 as i64;
                let add = trace_data.execution_trace[col_addition + j][i].0 as i64;
                let carry_in = if j == 0 { 0i64 } else { signed_carry(i, j - 1) };
                let snb = trace_data.execution_trace[col_shifted + j][i].0 as i64;
                let k = trace_data.execution_trace[col_k_idx][i].0 as i64;
                let carry_out = if j < 8 { signed_carry(i, j) } else { 0 };
                let p_j = super::air::P_LIMBS_28[j] as i64;
                let residual = da + add + carry_in - snb - k * p_j - carry_out * (1i64 << 28);
                if residual != 0 {
                    if chain_failures < 5 {
                        eprintln!("[chain-check] FAIL row {i} limb {j}: residual={residual} (da={da} add={add} cin={carry_in} snb={snb} k={k} cout={carry_out})");
                    }
                    chain_failures += 1;
                    failed_rows.insert(i);
                }
            }
        }
        // Reconstruct felts at each failing row to compare felt-level vs limb-level.
        for &i in failed_rows.iter().take(3) {
            let da_limbs: [M31; super::air::LIMBS_PER_FELT] =
                std::array::from_fn(|j| trace_data.execution_trace[col_digest_after + j][i]);
            let add_limbs: [M31; super::air::LIMBS_PER_FELT] =
                std::array::from_fn(|j| trace_data.execution_trace[col_addition + j][i]);
            let snb_limbs: [M31; super::air::LIMBS_PER_FELT] =
                std::array::from_fn(|j| trace_data.execution_trace[col_shifted + j][i]);
            let da_felt = super::air::limbs_to_felt252(&da_limbs);
            let add_felt = super::air::limbs_to_felt252(&add_limbs);
            let snb_felt = super::air::limbs_to_felt252(&snb_limbs);
            let sum = da_felt + add_felt;
            let diff = sum - snb_felt;
            eprintln!(
                "[chain-check FELT] row {i}: da={:#066x} add={:#066x} snb={:#066x} sum-snb={:#066x}",
                da_felt, add_felt, snb_felt, diff,
            );
        }
        if chain_failures > 0 {
            eprintln!(
                "[chain-check] {} total constraint failures across {} chain rows",
                chain_failures,
                n_real - 1
            );
        } else {
            eprintln!(
                "[chain-check] PASSED: all {} chain rows OK (n_padded={}, n_real={})",
                n_real - 1,
                n_padded,
                n_real
            );
        }

        // Check initial boundary: row 0's digest_before should be zero
        for j in 0..super::air::LIMBS_PER_FELT {
            let v = trace_data.execution_trace[j][0].0;
            if v != 0 {
                eprintln!("[boundary-check] INITIAL FAIL: row 0 limb {j} = {v} (expected 0)");
            }
        }

        // Check final boundary: row n_real-1's digest_after should match final_digest
        if n_real > 0 {
            let final_limbs_ref = super::air::felt252_to_limbs(&final_digest_felt);
            for j in 0..super::air::LIMBS_PER_FELT {
                let da = trace_data.execution_trace[col_digest_after + j][n_real - 1];
                if da != final_limbs_ref[j] {
                    eprintln!(
                        "[boundary-check] FINAL FAIL: row {} limb {j}: got {:?}, expected {:?}",
                        n_real - 1,
                        da,
                        final_limbs_ref[j]
                    );
                }
            }
        }

        // Check accumulator: verify correction term
        let n_m31 = M31::from(n_padded as u32);
        let n_inv_m31 = n_m31.inverse();
        let correction_m31 = M31::from(n_real as u32) * n_inv_m31;
        let mut accum_failures = 0;
        for i in 0..n_padded {
            let ac = trace_data.execution_trace[col_ac][i];
            let ac_next = trace_data.execution_trace[col_ac_next][i];
            let is_act = trace_data.execution_trace[col_is_active_offset][i];
            let residual = ac_next - ac - is_act + correction_m31;
            if residual != M31::from_u32_unchecked(0) {
                if accum_failures < 3 {
                    eprintln!("[accum-check] FAIL row {i}: ac={:?} ac_next={:?} is_act={:?} correction={:?} residual={:?}", ac, ac_next, is_act, correction_m31, residual);
                }
                accum_failures += 1;
            }
        }
        eprintln!(
            "[accum-check] {} failures out of {} rows",
            accum_failures, n_padded
        );
    }

    // Unified single-component proving: chain + Hades in one evaluate().
    recursive_log!(
        "  [Recursive] Proving unified component (hades_enabled={})",
        hades_enabled
    );
    let stark_proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
        &[&chain_component],
        channel,
        commitment_scheme,
    )
    .map_err(|e| RecursiveError::ProvingFailed(format!("{e:?}")))?;

    let recursive_prove_time = t_start.elapsed().as_secs_f64();
    recursive_log!(
        "  [Recursive] Done in {:.2}s (chain: {}x{}, hades: {}x{}, arithmetic: {}x{}, sumcheck: {}x{}, draws: {}x{}, proof size: {} bytes)",
        recursive_prove_time,
        1u32 << chain_log_size,
        super::air::COLS_PER_ROW,
        1u32 << hades_log_size,
        super::hades_air::N_HADES_TRACE_COLUMNS,
        if arithmetic_enabled {
            1u32 << arithmetic_log_size
        } else {
            0
        },
        if arithmetic_enabled {
            super::air::ARITH_COLS_PER_ROW
        } else {
            0
        },
        if sumcheck_enabled {
            1u32 << sumcheck_log_size
        } else {
            0
        },
        if sumcheck_enabled {
            super::air::SUMCHECK_COLS_PER_ROW
        } else {
            0
        },
        if draw_enabled {
            1u32 << draw_log_size
        } else {
            0
        },
        if draw_enabled {
            super::air::DRAW_COLS_PER_ROW
        } else {
            0
        },
        estimate_proof_size(&stark_proof),
    );

    Ok(RecursiveProof {
        stark_proof: stark_proof,
        public_inputs: witness.public_inputs,
        io_commitment_felt252: io_felt252,
        pass1_final_digest: witness.final_digest,
        final_digest: final_digest_felt,
        logup_claimed_sum: chain_claimed_sum,
        n_real_rows: trace_data.n_real_rows as u32,
        n_arithmetic_rows: arithmetic_trace.n_real_rows as u32,
        n_sumcheck_rows: sumcheck_trace.n_real_rows as u32,
        n_draw_rows: draw_trace.n_real_rows as u32,
        log_size: unified_log_size,
        hades_pairs: hades_perms.clone(),
        // This field is set after Level 1 proof generation. For now, compute the
        // commitment deterministically from the pairs (matching Cairo program output).
        // The chain STARK binds to this value via Fiat-Shamir channel.
        metadata: RecursiveProofMetadata {
            recursive_prove_time_secs: recursive_prove_time,
            gkr_prove_time_secs,
            n_poseidon_perms: witness.n_poseidon_perms,
            n_sumcheck_rounds: witness.n_sumcheck_rounds,
            n_qm31_ops: witness.n_qm31_ops,
            n_equality_checks: witness.n_equality_checks,
            n_primitive_arithmetic_rows: arithmetic_trace.n_real_rows,
            n_sumcheck_rows_air_constrained: sumcheck_trace.n_real_rows,
            n_channel_draw_rows_air_constrained: draw_trace.n_real_rows,
            transcript_air_constrained: hades_enabled && logup_enabled,
            primitive_arithmetic_air_constrained: arithmetic_enabled,
            primitive_sumcheck_air_constrained: sumcheck_enabled,
            sumcheck_challenge_draw_logup_bound: challenge_relation.is_some(),
            draw_felt_unpack_air_constrained: draw_enabled,
            draw_felt_hades_logup_bound: draw_felt_relation.is_some(),
            verifier_arithmetic_air_constrained: false,
            unconstrained_verifier_checks,
            trace_log_size: unified_log_size,
            n_trace_columns: super::air::COLS_PER_ROW
                + if hades_enabled {
                    super::hades_air::N_HADES_TRACE_COLUMNS
                } else {
                    0
                }
                + if arithmetic_enabled {
                    super::air::ARITH_COLS_PER_ROW
                } else {
                    0
                }
                + if sumcheck_enabled {
                    super::air::SUMCHECK_COLS_PER_ROW
                } else {
                    0
                }
                + if draw_enabled {
                    super::air::DRAW_COLS_PER_ROW
                } else {
                    0
                },
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Hardened Recursive Proving (with Hades AIR)
// ═══════════════════════════════════════════════════════════════════════

/// Extract (input, output) pairs for every HadesPerm in the witness.
///
/// These pairs are used to build the Hades verification trace that
/// constrains the actual Hades permutation computation.
pub fn extract_hades_perms(
    witness: &super::types::GkrVerifierWitness,
) -> Vec<(
    [starknet_ff::FieldElement; 3],
    [starknet_ff::FieldElement; 3],
)> {
    witness
        .ops
        .iter()
        .filter_map(|op| {
            if let super::types::WitnessOp::HadesPerm { input, output } = op {
                Some((*input, *output))
            } else {
                None
            }
        })
        .collect()
}

/// Compute the Hades commitment matching the Cairo verifier program's output.
/// The Cairo program chains: commitment = Hades(commitment, actual_out0, 2)[0]
/// for each pair, starting from commitment = 0.
pub fn compute_hades_commitment(
    pairs: &[(
        [starknet_ff::FieldElement; 3],
        [starknet_ff::FieldElement; 3],
    )],
) -> starknet_ff::FieldElement {
    let mut commitment = starknet_ff::FieldElement::ZERO;
    for (_input, output) in pairs {
        let mut state = [commitment, output[0], starknet_ff::FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut state);
        commitment = state[0];
    }
    commitment
}

/// Export Hades permutation pairs as Cairo arguments JSON.
///
/// Format: `["n_pairs", "in0", "in1", "in2", "out0", "out1", "out2", ...]`
/// Each felt252 is hex-encoded. This file is fed to cairo-prove --arguments-file.
pub fn export_hades_pairs_cairo_args(
    pairs: &[(
        [starknet_ff::FieldElement; 3],
        [starknet_ff::FieldElement; 3],
    )],
) -> String {
    let mut args: Vec<String> = Vec::with_capacity(1 + pairs.len() * 6);
    args.push(format!("\"{}\"", pairs.len()));
    for (input, output) in pairs {
        for v in input.iter().chain(output.iter()) {
            args.push(format!("\"{:#066x}\"", v));
        }
    }
    format!("[{}]", args.join(", "))
}

/// Verify all HadesPerm operations in a witness via step-by-step execution.
///
/// This is a standalone soundness check that can be run without generating
/// a STARK proof. It verifies that every (input, output) pair in the witness
/// corresponds to a correct Hades permutation.
///
/// Returns Ok(n_verified) on success, or Err with the first mismatch.
pub fn verify_hades_perms_offline(
    witness: &super::types::GkrVerifierWitness,
) -> Result<usize, RecursiveError> {
    let perms = extract_hades_perms(witness);
    for (i, (input, expected_output)) in perms.iter().enumerate() {
        let mut actual = *input;
        starknet_crypto::poseidon_permute_comp(&mut actual);
        if actual != *expected_output {
            return Err(RecursiveError::ProvingFailed(format!(
                "HadesPerm #{} mismatch: input={:?}, expected={:?}, got={:?}",
                i, input, expected_output, actual,
            )));
        }
    }
    Ok(perms.len())
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Convert a Vec<M31> to a SIMD column for STWO.
fn simd_column_from_vec(data: &[M31]) -> Col<SimdBackend, M31> {
    let mut col = Col::<SimdBackend, M31>::zeros(data.len());
    for (i, &val) in data.iter().enumerate() {
        col.set(i, val);
    }
    col
}

/// Rough estimate of serialized proof size.
fn estimate_proof_size(_proof: &StarkProof<Poseidon252MerkleHasher>) -> usize {
    4096 // placeholder
}

/// Serialize a STARK proof to bytes (placeholder — binary format in Phase 2D).
fn serialize_stark_proof(_proof: &StarkProof<Poseidon252MerkleHasher>) -> Vec<u8> {
    Vec::new()
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use stwo::core::fields::cm31::CM31;

    #[test]
    fn test_prove_recursive_1layer() {
        let _g = super::super::RecursiveTestModeGuard::enter();
        // End-to-end: prove a 1-layer MatMul GKR → recursive STARK.
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

        let result = prove_recursive(
            &circuit,
            gkr,
            &proof.execution.output,
            &weights,
            zero,
            zero,
            0.0,
        );

        let recursive_proof = result.expect("recursive proving should succeed");
        assert!(recursive_proof.metadata.n_poseidon_perms > 0);
        assert!(recursive_proof.metadata.recursive_prove_time_secs > 0.0);
        assert!(
            recursive_proof.metadata.transcript_air_constrained,
            "Hades+LogUp recursive test should constrain the transcript"
        );
        assert!(
            recursive_proof
                .metadata
                .primitive_arithmetic_air_constrained,
            "recursive test should locally constrain primitive verifier arithmetic rows"
        );
        assert!(
            recursive_proof.metadata.primitive_sumcheck_air_constrained,
            "recursive test should locally constrain recorded sumcheck rows"
        );
        assert!(
            recursive_proof.metadata.sumcheck_challenge_draw_logup_bound,
            "recursive test should bind sumcheck challenges to recorded channel draws via LogUp"
        );
        assert!(
            recursive_proof.metadata.n_channel_draw_rows_air_constrained
                >= recursive_proof.metadata.n_sumcheck_rows_air_constrained,
            "draw trace should cover the sumcheck challenge rows"
        );
        assert!(
            !recursive_proof.metadata.verifier_arithmetic_air_constrained,
            "custom recursive AIR still does not link arithmetic/sumcheck rows into a full verifier state machine"
        );
        assert!(
            recursive_proof.metadata.unconstrained_verifier_checks > 0,
            "coverage metadata must expose recorded verifier checks not consumed by AIR"
        );
        recursive_log!(
            "Recursive proof: {:.3}s, {} poseidon perms, log_size={}",
            recursive_proof.metadata.recursive_prove_time_secs,
            recursive_proof.metadata.n_poseidon_perms,
            recursive_proof.metadata.trace_log_size,
        );
    }

    #[test]
    fn test_prove_recursive_binds_full_io_felt252() {
        let _g = super::super::RecursiveTestModeGuard::enter();
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );
        let full_io = FieldElement::from(0x123456789abcdefu64);
        let statement_hash = FieldElement::from(0xfeed_cafe_u64);

        let recursive_proof = prove_recursive_with_policy_io_and_statement(
            &circuit,
            gkr,
            &proof.execution.output,
            &weights,
            zero,
            zero,
            Some(full_io),
            Some(statement_hash),
            0.0,
            None,
        )
        .expect("recursive proving should succeed");

        assert_eq!(recursive_proof.io_commitment_felt252, full_io);
        assert_eq!(
            recursive_proof.public_inputs.conversation_statement_hash,
            statement_hash
        );
        crate::recursive::verify_recursive_with_io_felt(
            &recursive_proof.stark_proof,
            &recursive_proof.public_inputs,
            recursive_proof.io_commitment_felt252,
            recursive_proof.pass1_final_digest,
            recursive_proof.n_real_rows,
            recursive_proof.n_arithmetic_rows,
            recursive_proof.n_sumcheck_rows,
            recursive_proof.n_draw_rows,
            recursive_proof.log_size,
            recursive_proof.final_digest,
            recursive_proof.logup_claimed_sum,
        )
        .expect("full felt252 IO-bound proof should verify");

        let mut tampered_inputs = recursive_proof.public_inputs.clone();
        tampered_inputs.conversation_statement_hash += FieldElement::ONE;
        let err = crate::recursive::verify_recursive_with_io_felt(
            &recursive_proof.stark_proof,
            &tampered_inputs,
            recursive_proof.io_commitment_felt252,
            recursive_proof.pass1_final_digest,
            recursive_proof.n_real_rows,
            recursive_proof.n_arithmetic_rows,
            recursive_proof.n_sumcheck_rows,
            recursive_proof.n_draw_rows,
            recursive_proof.log_size,
            recursive_proof.final_digest,
            recursive_proof.logup_claimed_sum,
        );
        assert!(
            err.is_err(),
            "relabelled conversation statement hash must be rejected"
        );
    }

    #[cfg(feature = "cli")]
    fn pack_qm31_header_limbs(limbs: &[FieldElement]) -> FieldElement {
        let shift31 = FieldElement::from(1u64 << 31);
        let mut result = limbs[0];
        result = result * shift31 + limbs[1];
        result = result * shift31 + limbs[2];
        result = result * shift31 + limbs[3];
        result
    }

    #[cfg(feature = "cli")]
    fn write_tiny_recursive_cairo_fixture(
        path: &str,
        recursive_proof: &RecursiveProof,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::fmt::Write as _;

        let calldata = crate::cairo_serde::serialize_recursive_proof_calldata(recursive_proof);
        let circuit_hash = pack_qm31_header_limbs(&calldata[0..4]);
        let weight_root = pack_qm31_header_limbs(&calldata[8..12]);
        let summary = crate::cairo_serde::recursive_proof_calldata_summary(recursive_proof);

        let mut out = String::new();
        writeln!(
            out,
            "// Auto-generated by `cargo test --features cli --lib recursive::prover::tests::dump_tiny_recursive_160_cairo_fixture -- --ignored --nocapture`."
        )?;
        writeln!(
            out,
            "// Tiny 1-layer recursive proof with Hades AIR + LogUp + 160-bit PCS."
        )?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_model_id() -> felt252 {{")?;
        writeln!(out, "    0x54494e595f524543555253495645")?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_circuit_hash() -> felt252 {{")?;
        writeln!(out, "    0x{:x}", circuit_hash)?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_weight_root() -> felt252 {{")?;
        writeln!(out, "    0x{:x}", weight_root)?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_io_commitment() -> felt252 {{")?;
        writeln!(out, "    0x{:x}", recursive_proof.io_commitment_felt252)?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_policy_commitment() -> felt252 {{")?;
        writeln!(out, "    0")?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_level1_proof_hash() -> felt252 {{")?;
        writeln!(out, "    0")?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_statement_hash() -> felt252 {{")?;
        writeln!(
            out,
            "    0x{:x}",
            recursive_proof.public_inputs.conversation_statement_hash
        )?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_n_layers() -> u32 {{")?;
        writeln!(out, "    {}", recursive_proof.public_inputs.n_layers)?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_n_matmuls() -> u32 {{")?;
        writeln!(out, "    1")?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_hidden_size() -> u32 {{")?;
        writeln!(out, "    4")?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_num_transformer_blocks() -> u32 {{")?;
        writeln!(out, "    1")?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_expected_n_poseidon_perms() -> u32 {{")?;
        writeln!(
            out,
            "    {}",
            recursive_proof.public_inputs.n_poseidon_perms
        )?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_trace_log_size() -> u32 {{")?;
        writeln!(out, "    {}", recursive_proof.log_size)?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_total_felts() -> u32 {{")?;
        writeln!(out, "    {}", summary.total_felts)?;
        writeln!(out, "}}")?;
        writeln!(out)?;
        writeln!(out, "pub fn tiny_calldata() -> Array<felt252> {{")?;
        writeln!(out, "    array![")?;
        for chunk in calldata.chunks(6) {
            write!(out, "        ")?;
            for (idx, felt) in chunk.iter().enumerate() {
                if idx > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "0x{:x}", felt)?;
            }
            writeln!(out, ",")?;
        }
        writeln!(out, "    ]")?;
        writeln!(out, "}}")?;

        std::fs::write(path, out)?;
        Ok(())
    }

    #[test]
    #[ignore = "writes a production 160-bit Cairo verifier fixture; run manually when refreshing test_recursive_data.cairo"]
    #[cfg(feature = "cli")]
    fn dump_tiny_recursive_160_cairo_fixture() {
        // Intentionally no RecursiveTestModeGuard here. This fixture must pass
        // the on-chain verifier's 160-bit security checks.
        std::env::set_var("OBELYZK_HADES_AIR", "1");
        std::env::set_var("OBELYZK_LOGUP", "1");

        let out = std::env::var("OBELYZK_FIXTURE_OUT")
            .expect("set OBELYZK_FIXTURE_OUT=/path/to/test_recursive_data.cairo");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let full_io = FieldElement::from(0x123456789abcdefu64);
        let weight_root = FieldElement::from(0x789abcu64);
        let statement_hash = FieldElement::from(0xfeed_cafe_u64);
        let recursive_proof = prove_recursive_with_policy_io_and_statement(
            &circuit,
            gkr,
            &proof.execution.output,
            &weights,
            crate::crypto::poseidon_channel::felt_to_securefield(weight_root),
            crate::crypto::poseidon_channel::felt_to_securefield(full_io),
            Some(full_io),
            Some(statement_hash),
            0.0,
            None,
        )
        .expect("recursive proving should succeed");

        crate::recursive::verify_recursive_with_io_felt(
            &recursive_proof.stark_proof,
            &recursive_proof.public_inputs,
            recursive_proof.io_commitment_felt252,
            recursive_proof.pass1_final_digest,
            recursive_proof.n_real_rows,
            recursive_proof.n_arithmetic_rows,
            recursive_proof.n_sumcheck_rows,
            recursive_proof.n_draw_rows,
            recursive_proof.log_size,
            recursive_proof.final_digest,
            recursive_proof.logup_claimed_sum,
        )
        .expect("fresh production recursive proof should verify in Rust");

        let written_path = match write_tiny_recursive_cairo_fixture(&out, &recursive_proof) {
            Ok(()) => out.clone(),
            Err(err) => {
                let fallback = "/private/tmp/test_recursive_data.cairo";
                eprintln!("could not write {out}: {err}; writing {fallback} instead");
                write_tiny_recursive_cairo_fixture(fallback, &recursive_proof)
                    .expect("write fallback Cairo fixture");
                fallback.to_string()
            }
        };
        eprintln!(
            "wrote {} felts to {} (log_size={}, n_poseidon_perms={})",
            crate::cairo_serde::serialize_recursive_proof_calldata(&recursive_proof).len(),
            written_path,
            recursive_proof.log_size,
            recursive_proof.public_inputs.n_poseidon_perms
        );
    }
}
