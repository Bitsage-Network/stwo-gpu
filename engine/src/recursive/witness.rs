//! GKR verifier witness generator for recursive STARK composition.
//!
//! This module re-executes the GKR verifier with an instrumented channel that
//! records transcript permutations and selected QM31 verifier checks. The
//! recorded operations become the execution trace for the recursive STARK.
//!
//! # Design Principle
//!
//! The witness generator executes the **exact same verifier code path** as the
//! production verifier (`verify_gkr_inner`) by running it over
//! `InstrumentedChannel`. Pass 1 still runs the concrete `PoseidonChannel`
//! verifier first as an independent preflight and to capture the production
//! digest/count.
//!
//! The intended end-state is:
//!
//! 1. Wrapping `PoseidonChannel` in `InstrumentedChannel` that records ops
//! 2. Using `InstrumentedChannel` as a drop-in replacement during verification
//! 3. Running differential tests to confirm both paths produce identical transcripts
//!
//! # Generic Verifier
//!
//! The verifier is generic over the channel:
//! ```ignore
//! fn verify_gkr_generic<C: VerifierChannel>(channel: &mut C, ...) -> Result<...>
//! ```
//! Both `PoseidonChannel` and `InstrumentedChannel` implement `VerifierChannel`.
//! This guarantees transcript consistency by construction. Arithmetic hooks are
//! recorded for the core MatMul/Add/Mul checks today; broader AIR constraints
//! are still required before calling the custom recursive AIR a complete
//! STARK-in-STARK verifier.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::crypto::poseidon_channel::{
    felt_to_securefield, pack_m31s, securefield_to_felt, PoseidonChannel, VerifierChannel,
};
use crate::gkr::types::SecureField;

use super::types::{GkrVerifierWitness, RecursivePublicInputs, WitnessOp};

// =========================================================================
// InstrumentedChannel — records every Poseidon call
// =========================================================================

/// A Fiat-Shamir channel that wraps `PoseidonChannel` and records every
/// operation for use as a STARK witness.
///
/// Every `mix_*` and `draw_*` call delegates to the inner channel AND
/// appends a `WitnessOp` to the ops log. The ops log becomes the
/// execution trace for the recursive STARK.
#[derive(Debug, Clone)]
pub struct InstrumentedChannel {
    /// The real channel — produces identical output to production.
    inner: PoseidonChannel,

    /// Recorded operations (in execution order).
    ops: Vec<WitnessOp>,

    /// Counters for trace sizing.
    n_poseidon_perms: usize,
    n_sumcheck_rounds: usize,
    n_qm31_ops: usize,
    n_equality_checks: usize,
}

impl InstrumentedChannel {
    /// Create a new instrumented channel wrapping a fresh PoseidonChannel.
    pub fn new() -> Self {
        Self {
            inner: PoseidonChannel::new(),
            ops: Vec::with_capacity(32_000),
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            n_equality_checks: 0,
        }
    }

    /// Create from an existing channel state (for mid-stream instrumentation).
    pub fn from_channel(channel: PoseidonChannel) -> Self {
        Self {
            inner: channel,
            ops: Vec::with_capacity(32_000),
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            n_equality_checks: 0,
        }
    }

    /// Get the accumulated operations log.
    pub fn ops(&self) -> &[WitnessOp] {
        &self.ops
    }

    /// Consume the channel and return the operations log.
    pub fn into_ops(self) -> Vec<WitnessOp> {
        self.ops
    }

    /// Get a reference to the inner (production) channel.
    pub fn inner(&self) -> &PoseidonChannel {
        &self.inner
    }

    /// Get a mutable reference to the inner channel.
    pub fn inner_mut(&mut self) -> &mut PoseidonChannel {
        &mut self.inner
    }

    /// Get counters for trace sizing.
    pub fn counters(&self) -> (usize, usize, usize, usize) {
        (
            self.n_poseidon_perms,
            self.n_sumcheck_rounds,
            self.n_qm31_ops,
            self.n_equality_checks,
        )
    }

    // ── Channel operations (delegate to inner + record Hades states) ──

    /// Mix a u64 value into the channel. Records the Hades permutation.
    pub fn mix_u64(&mut self, value: u64) {
        self.mix_felt(FieldElement::from(value));
    }

    /// Mix a felt252 value into the channel. Records Hades + ChannelOp.
    pub fn mix_felt(&mut self, value: FieldElement) {
        let digest_before = self.inner.digest();

        // Record the Hades call
        let input = [digest_before, value, FieldElement::TWO];
        let mut output = input;
        crate::crypto::hades::hades_permutation(&mut output);
        self.ops.push(WitnessOp::HadesPerm { input, output });
        self.n_poseidon_perms += 1;

        // Record the channel operation (digest chain unit)
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after: output[0],
        });

        // Advance inner channel
        self.inner.mix_felt(value);
        debug_assert_eq!(self.inner.digest(), output[0]);
    }

    /// Mix a SecureField (QM31) into the channel.
    pub fn mix_securefield(&mut self, value: SecureField) {
        let felt = securefield_to_felt(value);
        self.mix_felt(felt);
    }

    /// Draw a QM31 from the channel. Records Hades + ChannelOp.
    pub fn draw_qm31(&mut self) -> SecureField {
        let digest_before = self.inner.digest();
        let n_draws = self.inner.n_draws();

        let result = self.inner.draw_qm31();
        let digest_after = self.inner.digest();

        // Record the Hades call
        let input = [
            digest_before,
            FieldElement::from(n_draws as u64),
            FieldElement::THREE,
        ];
        let mut output = input;
        crate::crypto::hades::hades_permutation(&mut output);
        self.ops.push(WitnessOp::HadesPerm { input, output });
        self.n_poseidon_perms += 1;

        // Note: draw doesn't change the digest (only increments n_draws),
        // but the channel state changes. We still record the ChannelOp.
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after,
        });
        self.ops.push(WitnessOp::ChannelDraw { result });
        result
    }

    /// Draw multiple QM31s.
    pub fn draw_qm31s(&mut self, count: usize) -> Vec<SecureField> {
        (0..count).map(|_| self.draw_qm31()).collect()
    }

    /// Mix degree-2 polynomial coefficients (3 QM31s).
    ///
    /// Replicates the exact Hades calls from `poseidon_hash_many([digest, felt1, felt2])`.
    ///
    /// From starknet-crypto source:
    /// ```text
    /// state = [0, 0, 0]
    /// state[0] += digest; state[1] += felt1; hades(&state)   // absorb pair
    /// state[0] += felt2; state[1] += 1;      hades(&state)   // absorb remainder + padding
    /// return state[0]
    /// ```
    /// Total: 2 Hades calls.
    pub fn mix_poly_coeffs(&mut self, c0: SecureField, c1: SecureField, c2: SecureField) {
        use crate::crypto::poseidon_channel::pack_m31s;
        let m31s: Vec<M31> = vec![
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1, c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1, c2.0 .0,
            c2.0 .1, c2.1 .0, c2.1 .1,
        ];
        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);
        let digest_before = self.inner.digest();

        // Replicate poseidon_hash_many([digest, felt1, felt2]) exactly:
        // Call 1: absorb pair [digest, felt1]
        let mut state = [FieldElement::ZERO; 3];
        state[0] += digest_before;
        state[1] += felt1;
        let input1 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm {
            input: input1,
            output: state,
        });
        self.n_poseidon_perms += 1;

        // Call 2: absorb remainder [felt2] + padding
        state[0] += felt2;
        state[1] += FieldElement::ONE; // padding at state[remainder.len()] = state[1]
        let input2 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm {
            input: input2,
            output: state,
        });
        self.n_poseidon_perms += 1;

        // Record channel operation (atomic digest transition)
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after: state[0],
        });

        // Advance inner channel and verify
        self.inner.mix_poly_coeffs(c0, c1, c2);
        debug_assert_eq!(
            self.inner.digest(),
            state[0],
            "mix_poly_coeffs decomposition mismatch"
        );
    }

    /// Mix a variable-length array of QM31 values into the channel.
    ///
    /// This records the exact `poseidon_hash_many([digest, packed_chunks...])`
    /// permutation sequence used by `PoseidonChannel::mix_felts`.
    pub fn mix_felts(&mut self, felts: &[SecureField]) {
        if felts.is_empty() {
            return;
        }

        let mut hash_inputs = vec![self.inner.digest()];
        let mut i = 0;
        while i < felts.len() {
            let remaining = felts.len() - i;
            if remaining >= 2 {
                let m31s: Vec<M31> = vec![
                    felts[i].0 .0,
                    felts[i].0 .1,
                    felts[i].1 .0,
                    felts[i].1 .1,
                    felts[i + 1].0 .0,
                    felts[i + 1].0 .1,
                    felts[i + 1].1 .0,
                    felts[i + 1].1 .1,
                ];
                hash_inputs.push(pack_m31s(&m31s));
                i += 2;
            } else {
                let m31s: Vec<M31> =
                    vec![felts[i].0 .0, felts[i].0 .1, felts[i].1 .0, felts[i].1 .1];
                hash_inputs.push(pack_m31s(&m31s));
                i += 1;
            }
        }

        let digest_before = self.inner.digest();
        let digest_after = self.record_poseidon_hash_many(&hash_inputs);
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after,
        });

        self.inner.mix_felts(felts);
        debug_assert_eq!(
            self.inner.digest(),
            digest_after,
            "mix_felts decomposition mismatch"
        );
    }

    fn record_poseidon_hash_many(&mut self, inputs: &[FieldElement]) -> FieldElement {
        let mut state = [FieldElement::ZERO, FieldElement::ZERO, FieldElement::ZERO];
        let mut chunks = inputs.chunks_exact(2);

        for chunk in chunks.by_ref() {
            state[0] += chunk[0];
            state[1] += chunk[1];
            let input = state;
            crate::crypto::hades::hades_permutation(&mut state);
            self.ops.push(WitnessOp::HadesPerm {
                input,
                output: state,
            });
            self.n_poseidon_perms += 1;
        }

        let remainder = chunks.remainder();
        if remainder.len() == 1 {
            state[0] += remainder[0];
        }
        state[remainder.len()] += FieldElement::ONE;
        let input = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm {
            input,
            output: state,
        });
        self.n_poseidon_perms += 1;

        state[0]
    }

    /// Mix degree-3 polynomial coefficients (4 QM31s).
    ///
    /// Same sponge construction for `poseidon_hash_many([digest, felt1, felt2])`.
    /// 16 M31s → 2 felt252s → 2 Hades calls.
    pub fn mix_poly_coeffs_deg3(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
        c3: SecureField,
    ) {
        use crate::crypto::poseidon_channel::pack_m31s;
        let m31s: Vec<M31> = vec![
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1, c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1, c2.0 .0,
            c2.0 .1, c2.1 .0, c2.1 .1, c3.0 .0, c3.0 .1, c3.1 .0, c3.1 .1,
        ];
        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);
        let digest_before = self.inner.digest();

        // poseidon_hash_many([digest, felt1, felt2]): 2 Hades calls
        let mut state = [FieldElement::ZERO; 3];
        state[0] += digest_before;
        state[1] += felt1;
        let input1 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm {
            input: input1,
            output: state,
        });
        self.n_poseidon_perms += 1;

        state[0] += felt2;
        state[1] += FieldElement::ONE;
        let input2 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm {
            input: input2,
            output: state,
        });
        self.n_poseidon_perms += 1;

        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after: state[0],
        });

        self.inner.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        debug_assert_eq!(
            self.inner.digest(),
            state[0],
            "mix_poly_coeffs_deg3 decomposition mismatch"
        );
    }

    /// Draw a raw felt252.
    pub fn draw_felt252(&mut self) -> FieldElement {
        let digest = self.inner.digest();
        let n_draws = self.inner.n_draws();
        let input = [
            digest,
            FieldElement::from(n_draws as u64),
            FieldElement::THREE,
        ];
        let mut output = input;
        crate::crypto::hades::hades_permutation(&mut output);
        self.ops.push(WitnessOp::HadesPerm { input, output });
        self.n_poseidon_perms += 1;

        self.inner.draw_felt252()
    }

    // ── Arithmetic recording (called by the verifier replay) ─────────

    /// Record a QM31 multiplication that the verifier computed.
    pub fn record_mul(&mut self, a: SecureField, b: SecureField, result: SecureField) {
        self.ops.push(WitnessOp::QM31Mul { a, b, result });
        self.n_qm31_ops += 1;
    }

    /// Record a QM31 addition that the verifier computed.
    pub fn record_add(&mut self, a: SecureField, b: SecureField, result: SecureField) {
        self.ops.push(WitnessOp::QM31Add { a, b, result });
        self.n_qm31_ops += 1;
    }

    /// Record an equality check that the verifier asserted.
    pub fn record_equality_check(&mut self, lhs: SecureField, rhs: SecureField) {
        self.ops.push(WitnessOp::EqualityCheck { lhs, rhs });
        self.n_equality_checks += 1;
    }

    /// Record a sumcheck round (degree-2).
    pub fn record_sumcheck_round_deg2(
        &mut self,
        round_poly: crate::components::matmul::RoundPoly,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    ) {
        self.ops.push(WitnessOp::SumcheckRoundDeg2 {
            round_poly,
            claim,
            challenge,
            next_claim,
        });
        self.n_sumcheck_rounds += 1;
    }

    /// Record a sumcheck round (degree-3).
    pub fn record_sumcheck_round_deg3(
        &mut self,
        round_poly: crate::gkr::types::RoundPolyDeg3,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    ) {
        self.ops.push(WitnessOp::SumcheckRoundDeg3 {
            round_poly,
            claim,
            challenge,
            next_claim,
        });
        self.n_sumcheck_rounds += 1;
    }
}

impl Default for InstrumentedChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifierChannel for InstrumentedChannel {
    fn mix_u64(&mut self, value: u64) {
        InstrumentedChannel::mix_u64(self, value);
    }

    fn mix_felt(&mut self, value: FieldElement) {
        InstrumentedChannel::mix_felt(self, value);
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        InstrumentedChannel::mix_felts(self, felts);
    }

    fn mix_poly_coeffs(&mut self, c0: SecureField, c1: SecureField, c2: SecureField) {
        InstrumentedChannel::mix_poly_coeffs(self, c0, c1, c2);
    }

    fn mix_poly_coeffs_deg3(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
        c3: SecureField,
    ) {
        InstrumentedChannel::mix_poly_coeffs_deg3(self, c0, c1, c2, c3);
    }

    fn draw_felt252(&mut self) -> FieldElement {
        InstrumentedChannel::draw_felt252(self)
    }

    fn draw_qm31(&mut self) -> SecureField {
        InstrumentedChannel::draw_qm31(self)
    }

    fn draw_qm31s(&mut self, count: usize) -> Vec<SecureField> {
        InstrumentedChannel::draw_qm31s(self, count)
    }

    fn mix_securefield(&mut self, value: SecureField) {
        InstrumentedChannel::mix_securefield(self, value);
    }

    fn record_qm31_mul(&mut self, a: SecureField, b: SecureField, result: SecureField) {
        InstrumentedChannel::record_mul(self, a, b, result);
    }

    fn record_qm31_add(&mut self, a: SecureField, b: SecureField, result: SecureField) {
        InstrumentedChannel::record_add(self, a, b, result);
    }

    fn record_equality_check(&mut self, lhs: SecureField, rhs: SecureField) {
        InstrumentedChannel::record_equality_check(self, lhs, rhs);
    }

    fn record_sumcheck_round_deg2(
        &mut self,
        round_poly: crate::components::matmul::RoundPoly,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    ) {
        InstrumentedChannel::record_sumcheck_round_deg2(
            self, round_poly, claim, challenge, next_claim,
        );
    }

    fn record_sumcheck_round_deg3(
        &mut self,
        round_poly: crate::gkr::types::RoundPolyDeg3,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    ) {
        InstrumentedChannel::record_sumcheck_round_deg3(
            self, round_poly, claim, challenge, next_claim,
        );
    }

    fn digest(&self) -> FieldElement {
        self.inner.digest()
    }

    fn n_draws(&self) -> u32 {
        self.inner.n_draws()
    }

    fn hash_count(&self) -> u64 {
        self.inner.hash_count()
    }
}

// =========================================================================
// Witness generation
// =========================================================================

/// Generate the recursive STARK witness by replaying the GKR verifier.
///
/// This uses a two-pass approach:
///
/// **Pass 1 (production verification)**: Runs the real `verify_gkr` with a
/// `PoseidonChannel` to confirm the proof is valid and measure the exact
/// number of Poseidon calls (via `hash_count()`).
///
/// **Pass 2 (instrumented verification)**: Runs the same verifier over an
/// `InstrumentedChannel`, recording every transcript permutation plus the core
/// MatMul/Add/Mul arithmetic hooks that the recursive AIR currently consumes.
///
/// Matching the Pass 1 digest proves transcript coverage. It is still not a
/// complete custom-AIR verifier for every non-core arithmetic relation until
/// those verifier hooks/constraints are added.
pub fn generate_witness(
    circuit: &crate::gkr::circuit::LayeredCircuit,
    proof: &crate::gkr::types::GKRProof,
    output: &crate::components::matmul::M31Matrix,
    weights: Option<&crate::compiler::graph::GraphWeights>,
    weight_super_root: QM31,
    io_commitment: QM31,
) -> Result<GkrVerifierWitness, crate::gkr::types::GKRError> {
    generate_witness_with_policy(
        circuit,
        proof,
        output,
        weights,
        weight_super_root,
        io_commitment,
        None,
    )
}

/// Generate witness with explicit policy binding.
///
/// The policy must match the one used during proving — otherwise the
/// Fiat-Shamir channel diverges and verification fails in Pass 1.
pub fn generate_witness_with_policy(
    circuit: &crate::gkr::circuit::LayeredCircuit,
    proof: &crate::gkr::types::GKRProof,
    output: &crate::components::matmul::M31Matrix,
    weights: Option<&crate::compiler::graph::GraphWeights>,
    weight_super_root: QM31,
    io_commitment: QM31,
    policy: Option<&crate::policy::PolicyConfig>,
) -> Result<GkrVerifierWitness, crate::gkr::types::GKRError> {
    // ── Pass 1: production verification ──────────────────────────────
    // Run the real verifier to (a) confirm validity, (b) measure hash_count,
    // and (c) capture the final channel digest.
    //
    // The channel must match the prover's GKR channel state exactly.
    // The pure GKR path (prove_model_pure_gkr_inner) seeds the GKR channel
    // ONLY with optional KV-cache commitment — no io_commitment or policy
    // outer seeding (that's only in prove_model_aggregated_onchain_gkr_auto).
    let mut prod_channel = crate::crypto::poseidon_channel::PoseidonChannel::new();

    // Decode-step continuity: when the proof carries KV commitments (set by
    // `prove_model_pure_gkr_decode_step_incremental`), the decode prover
    // seeds the GKR channel with them in this order before any layer prove:
    //   gkr_channel.mix_felt(new_kv_commitment);    // = kv_cache_commitment
    //   gkr_channel.mix_felt(prev_kv_commitment);   // = prev_kv_cache_commitment
    // The witness's verifier replay MUST mirror this exactly or the
    // Fiat-Shamir challenges from round 0 onward diverge — which manifested
    // as the `rmsnorm RMS² round 1: p(0)+p(1) != sum` failure for decode
    // proofs. (See aggregation.rs:6099-6100 for the prover side.)
    if let Some(kv) = proof.kv_cache_commitment {
        prod_channel.mix_felt(kv);
        if let Some(prev_kv) = proof.prev_kv_cache_commitment {
            prod_channel.mix_felt(prev_kv);
        }
    }

    let _claim = if let Some(p) = policy {
        crate::gkr::verifier::verify_gkr_with_policy(
            circuit,
            proof,
            output,
            weights,
            &mut prod_channel,
            p,
        )?
    } else if let Some(w) = weights {
        crate::gkr::verifier::verify_gkr_with_weights(circuit, proof, output, w, &mut prod_channel)?
    } else {
        crate::gkr::verifier::verify_gkr(circuit, proof, output, &mut prod_channel)?
    };
    let total_poseidon_calls = prod_channel.hash_count() as usize;
    let final_digest = prod_channel.digest();

    // ── Pass 2: instrumented verifier replay ─────────────────────────
    // Run the same verifier code path over InstrumentedChannel so every
    // transcript operation is captured for the recursive trace.
    let mut channel = InstrumentedChannel::new();
    let d = circuit.layers.len();

    if let Some(kv) = proof.kv_cache_commitment {
        channel.mix_felt(kv);
        if let Some(prev_kv) = proof.prev_kv_cache_commitment {
            channel.mix_felt(prev_kv);
        }
    }

    // SECURITY: Capture the seed digest — deterministic given the model.
    // This becomes a public input and AIR checkpoint constraint.
    let seed_digest = {
        let mut seed_channel = PoseidonChannel::new();
        if let Some(kv) = proof.kv_cache_commitment {
            seed_channel.mix_felt(kv);
            if let Some(prev_kv) = proof.prev_kv_cache_commitment {
                seed_channel.mix_felt(prev_kv);
            }
        }
        seed_channel.mix_u64(d as u64);
        seed_channel.mix_u64(circuit.input_shape.0 as u64);
        seed_channel.mix_u64(circuit.input_shape.1 as u64);
        felt_to_securefield(seed_channel.digest())
    };

    crate::gkr::verifier::verify_gkr_on_channel(
        circuit,
        proof,
        output,
        weights,
        &mut channel,
        policy,
    )?;

    let circuit_hash = compute_circuit_hash(circuit);
    let (_instrumented_poseidon, n_sumcheck_rounds, n_qm31_ops, n_equality_checks) =
        channel.counters();

    // Compute hades_commitment from all HadesPerm pairs in the ops.
    // This matches the Cairo Hades verifier program's output.
    let hades_commitment = {
        let pairs: Vec<_> = channel
            .ops()
            .iter()
            .filter_map(|op| {
                if let WitnessOp::HadesPerm { input, output } = op {
                    Some((*input, *output))
                } else {
                    None
                }
            })
            .collect();
        super::prover::compute_hades_commitment(&pairs)
    };

    // Pull KV-cache commitments from the GKR proof (set by decode-step prover).
    // FieldElement::ZERO on prefill / single-pass / non-decode proofs.
    let kv_cache_commitment = proof
        .kv_cache_commitment
        .unwrap_or(starknet_ff::FieldElement::ZERO);
    let prev_kv_cache_commitment = proof
        .prev_kv_cache_commitment
        .unwrap_or(starknet_ff::FieldElement::ZERO);

    let witness = GkrVerifierWitness {
        ops: channel.into_ops(),
        public_inputs: RecursivePublicInputs {
            circuit_hash,
            io_commitment,
            weight_super_root,
            n_layers: d as u32,
            n_poseidon_perms: total_poseidon_calls as u32,
            seed_digest,
            hades_commitment,
            kv_cache_commitment,
            prev_kv_cache_commitment,
            conversation_statement_hash: starknet_ff::FieldElement::ZERO,
        },
        // Use the production verifier's total count (covers ALL layer types)
        n_poseidon_perms: total_poseidon_calls,
        n_sumcheck_rounds,
        n_qm31_ops,
        final_digest,
        n_equality_checks,
    };

    Ok(witness)
}

/// Compute a deterministic hash of the circuit structure.
///
/// This binds the recursive proof to a specific model architecture.
/// The hash covers layer types and shapes but NOT weight values
/// (those are bound via weight_super_root).
pub fn compute_circuit_hash(circuit: &crate::gkr::circuit::LayeredCircuit) -> QM31 {
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::gkr::circuit::LayerType;

    let mut hasher = PoseidonChannel::new();

    // Mix circuit dimensions
    hasher.mix_u64(circuit.layers.len() as u64);
    hasher.mix_u64(circuit.input_shape.0 as u64);
    hasher.mix_u64(circuit.input_shape.1 as u64);

    // Mix each layer's type and shape
    for layer in &circuit.layers {
        let type_tag: u64 = match &layer.layer_type {
            LayerType::Input => 0,
            LayerType::Identity => 1,
            LayerType::MatMul { .. } => 2,
            LayerType::Add { .. } => 3,
            LayerType::Mul { .. } => 4,
            LayerType::Activation { .. } => 5,
            LayerType::LayerNorm { .. } => 6,
            LayerType::RMSNorm { .. } => 7,
            LayerType::Embedding { .. } => 8,
            LayerType::RoPE { .. } => 9,
            _ => 99,
        };
        hasher.mix_u64(type_tag);
        hasher.mix_u64(layer.output_shape.0 as u64);
        hasher.mix_u64(layer.output_shape.1 as u64);
    }

    // Extract hash as QM31
    hasher.draw_qm31()
}
