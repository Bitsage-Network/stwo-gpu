//! GPU-accelerated quotient operations.
//!
//! This module implements [`QuotientOps`] for [`GpuBackend`].
//!
//! Quotient accumulation is moderately compute-intensive but has complex
//! memory access patterns that don't map well to GPU. We delegate to SIMD
//! for now and can optimize later if profiling shows benefit.

use crate::core::poly::circle::CircleDomain;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::prover::pcs::quotient_ops::QuotientOps;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::core::fields::m31::BaseField;

use super::GpuBackend;

impl QuotientOps for GpuBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // Quotient accumulation has irregular memory access patterns
        // GPU doesn't help much here, delegate to SIMD
        
        // Convert column references
        let simd_columns: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = 
            columns.iter().map(|c| unsafe {
                std::mem::transmute::<
                    &&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
                    &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>
                >(c)
            }).collect();
        
        let result = SimdBackend::accumulate_quotients(
            domain,
            &simd_columns,
            random_coeff,
            sample_batches,
            log_blowup_factor,
        );
        
        unsafe {
            std::mem::transmute::<
                SecureEvaluation<SimdBackend, BitReversedOrder>,
                SecureEvaluation<GpuBackend, BitReversedOrder>
            >(result)
        }
    }
}

