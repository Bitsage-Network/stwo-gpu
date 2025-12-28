//! GPU Component Prover implementation.
//!
//! This module provides the `ComponentProver<GpuBackend>` implementation for
//! `FrameworkComponent`, enabling GPU-accelerated constraint evaluation.
//!
//! # Architecture
//!
//! The GPU prover follows the same flow as SIMD, but with GPU acceleration:
//!
//! 1. Extend trace polynomials to evaluation domain (GPU FFT)
//! 2. Compute denominator inverses
//! 3. Evaluate constraints row-by-row (GPU parallel or batched)
//! 4. Accumulate results with random coefficient weighting
//!
//! # Backend Design
//!
//! GpuBackend shares the same column types as SimdBackend, allowing us to
//! use SIMD-style vectorized constraint evaluation. The GPU acceleration
//! comes from:
//!
//! - GPU-accelerated FFT for polynomial extension
//! - GPU-accelerated column operations
//! - Future: Direct GPU kernel constraint evaluation

use std::borrow::Cow;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::air::Component;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo::prover::backend::gpu::conversion::secure_col_coords_mut_to_simd;
use stwo::prover::backend::gpu::GpuBackend;
use stwo::prover::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{ColumnAccumulator, ComponentProver, DomainEvaluationAccumulator, Trace};
use tracing::{span, Level};

use super::GpuDomainEvaluator;
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

/// Chunk size for parallel constraint evaluation.
/// Larger chunks reduce parallel overhead but may hurt load balancing.
const CHUNK_SIZE: usize = 1;

impl<E: FrameworkEval + Sync> ComponentProver<GpuBackend> for FrameworkComponent<E> {
    /// Evaluate constraint quotients on the evaluation domain using GPU acceleration.
    ///
    /// This method:
    /// 1. Extends trace polynomials to the evaluation domain
    /// 2. Computes denominator inverses for the vanishing polynomial
    /// 3. Evaluates all constraints at each domain point
    /// 4. Accumulates results weighted by powers of a random coefficient
    ///
    /// # Performance
    ///
    /// GPU acceleration comes from:
    /// - GPU-accelerated FFT in polynomial extension
    /// - Parallel constraint evaluation using SIMD-style vectorization
    /// - Future: Direct GPU kernel for constraint evaluation
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, GpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<GpuBackend>,
    ) {
        // Early return if no constraints
        if self.n_constraints() == 0 {
            return;
        }

        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.eval.log_size());

        // Build component polynomial references
        let mut component_polys = trace.polys.sub_tree(&self.trace_locations);
        component_polys[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
            .collect();

        // Check if we need to extend polynomials to the evaluation domain
        let need_to_extend = component_polys
            .iter()
            .flatten()
            .any(|c| c.evals.domain.log_size() != eval_domain.log_size());

        // Extend trace using GPU-accelerated FFT
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "GPU Constraint Extension").entered();
            // Use GPU backend for twiddle precomputation and evaluation
            let twiddles = GpuBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles)))
        } else {
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
        };

        // Compute denominator inverses
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        // Get accumulator column
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "GPU Constraint point-wise eval",
            class = "GpuConstraintEval"
        )
        .entered();

        // Use vectorized SIMD-style evaluation.
        // This works for GpuBackend because it uses the same column types as SimdBackend.
        self.evaluate_constraints_vectorized(&trace, &denom_inv, &mut accum, trace_domain, eval_domain);
    }
}

impl<E: FrameworkEval + Sync> FrameworkComponent<E> {
    /// Vectorized constraint evaluation using SIMD-style processing.
    ///
    /// This method works for GpuBackend because it shares column types with SimdBackend.
    /// The conversion is done via `secure_col_coords_mut_to_simd` which uses transmute
    /// since both backends have identical memory layouts.
    fn evaluate_constraints_vectorized(
        &self,
        trace: &TreeVec<Vec<Cow<'_, CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>>>>,
        denom_inv: &[BaseField],
        accum: &mut ColumnAccumulator<'_, GpuBackend>,
        trace_domain: CanonicCoset,
        eval_domain: stwo::core::poly::circle::CircleDomain,
    ) {
        let _span = span!(Level::INFO, "GPU Vectorized Constraint Eval").entered();

        // Convert GpuBackend column to SimdBackend column reference.
        // This is safe because both backends use identical column types.
        let simd_col = secure_col_coords_mut_to_simd(accum.col);

        // Transform to VeryPacked representation for vectorized SIMD processing
        let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(simd_col) };

        let range = 0..(1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS));

        #[cfg(not(feature = "parallel"))]
        let iter = range.step_by(CHUNK_SIZE).zip(col.chunks_mut(CHUNK_SIZE));

        #[cfg(feature = "parallel")]
        let iter = range
            .into_par_iter()
            .step_by(CHUNK_SIZE)
            .zip(col.chunks_mut(CHUNK_SIZE));

        let self_eval = &self.eval;
        let self_claimed_sum = self.claimed_sum;

        iter.for_each(|(chunk_idx, mut chunk)| {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

            for idx_in_chunk in 0..CHUNK_SIZE {
                let vec_row = chunk_idx * CHUNK_SIZE + idx_in_chunk;

                // Evaluate constraints using GPU domain evaluator
                let eval = GpuDomainEvaluator::new(
                    &trace_cols,
                    vec_row,
                    &accum.random_coeff_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self_eval.log_size(),
                    self_claimed_sum,
                );
                let row_res = self_eval.evaluate(eval).row_res;

                // Apply denominator inverse and accumulate
                unsafe {
                    let denom = VeryPackedBaseField::broadcast(
                        denom_inv[vec_row
                            >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                    );
                    chunk.set_packed(
                        idx_in_chunk,
                        chunk.packed_at(idx_in_chunk) + row_res * denom,
                    )
                }
            }
        });

        tracing::debug!(
            "GPU vectorized constraint evaluation completed for {} rows",
            1 << eval_domain.log_size()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_size() {
        assert_eq!(CHUNK_SIZE, 1);
    }
}
