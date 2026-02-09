//! GPU-accelerated quotient operations.
//!
//! This module implements [`QuotientOps`] for [`GpuBackend`].
//!
//! # Algorithm
//!
//! The new quotient pipeline is split into two phases:
//! 1. **accumulate_numerators** - Accumulates FRI numerators per (log_size, sample_point).
//! 2. **compute_quotients_and_combine** - Computes denominators, multiplies by accumulated
//!    numerators, and sums across sample points.
//!
//! # GPU Strategy
//!
//! Both phases delegate to `SimdBackend` via zero-cost transmute conversions.
//! When CUDA kernels are available (future work), the delegation can be replaced
//! with native GPU kernels for large domains.

use crate::core::fields::m31::BaseField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::pcs::quotient_ops::{AccumulatedNumerators, QuotientOps};
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;

use super::conversion::{circle_eval_ref_to_simd, secure_eval_to_gpu};
use super::GpuBackend;

impl QuotientOps for GpuBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        // Convert GpuBackend column references to SimdBackend references (zero-copy transmute).
        let simd_columns: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            columns.iter().map(|c| circle_eval_ref_to_simd(*c)).collect();

        // Delegate to SimdBackend, collecting into a SimdBackend-typed vec.
        let mut simd_acc: Vec<AccumulatedNumerators<SimdBackend>> = Vec::new();
        SimdBackend::accumulate_numerators(&simd_columns, sample_batches, &mut simd_acc);

        // Convert AccumulatedNumerators<SimdBackend> -> AccumulatedNumerators<GpuBackend>.
        // Both backends share identical column types, so transmute is safe.
        for acc in simd_acc {
            accumulated_numerators_vec.push(unsafe { std::mem::transmute(acc) });
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // Convert AccumulatedNumerators<GpuBackend> -> AccumulatedNumerators<SimdBackend>.
        let simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = accs
            .into_iter()
            .map(|acc| unsafe { std::mem::transmute(acc) })
            .collect();

        let result = SimdBackend::compute_quotients_and_combine(simd_accs);

        // Convert SecureEvaluation<SimdBackend> -> SecureEvaluation<GpuBackend>.
        secure_eval_to_gpu(result)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpu_backend_quotient_ops_compiles() {
        // Compile-time check: GpuBackend implements QuotientOps.
        fn _assert_impl<T: super::QuotientOps>() {}
        _assert_impl::<super::GpuBackend>();
    }
}
