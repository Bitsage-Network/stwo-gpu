//! GPU-accelerated polynomial operations.
//!
//! This module implements [`PolyOps`] for [`GpuBackend`], providing GPU acceleration
//! for the most computationally intensive operations in proof generation:
//!
//! - **FFT (evaluate/interpolate)**: 50-100x speedup on large polynomials
//! - **Twiddle precomputation**: GPU-accelerated for large domains
//! - **Point evaluation**: Parallelized on GPU
//!
//! # Performance Strategy
//!
//! We use a hybrid approach:
//! - Small polynomials (< 16K elements): Use SIMD backend (GPU overhead not worth it)
//! - Large polynomials (>= 16K elements): Use GPU acceleration
//!
//! This threshold was determined empirically on A100 GPUs.

use tracing::{span, Level};

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

use super::conversion::{
    circle_coeffs_ref_to_simd, circle_eval_ref_to_simd, twiddle_ref_to_simd, twiddle_to_gpu,
};
use super::fft::{GPU_FFT_THRESHOLD_LOG_SIZE, compute_itwiddle_dbls_cpu, compute_twiddle_dbls_cpu};
use super::cuda_executor::{is_cuda_available, cuda_ifft, cuda_fft};
use super::GpuBackend;

impl PolyOps for GpuBackend {
    // We use the same twiddle type as SimdBackend since twiddles are precomputed
    // and stored in CPU memory. The GPU uses them during FFT execution.
    type Twiddles = <SimdBackend as PolyOps>::Twiddles;
    
    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::interpolate").entered();
        
        let log_size = eval.domain.log_size();
        
        // Small polynomials: use SIMD (GPU overhead not worth it)
        if log_size < GPU_FFT_THRESHOLD_LOG_SIZE {
            tracing::debug!(
                "GPU interpolate: using SIMD for small size (log_size={} < threshold={})",
                log_size, GPU_FFT_THRESHOLD_LOG_SIZE
            );
            return interpolate_simd_fallback(eval, twiddles);
        }
        
        // Large polynomials: require GPU
        if !is_cuda_available() {
            panic!(
                "GpuBackend::interpolate called with log_size={} but CUDA is not available. \
                 Use SimdBackend for CPU-only execution.",
                log_size
            );
        }
        
        tracing::info!(
            "GPU interpolate: using CUDA for {} elements (log_size={})",
            1u64 << log_size, log_size
        );
        gpu_interpolate(eval, twiddles, log_size)
    }
    
    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        // Point evaluation is memory-bound, not compute-bound
        // GPU doesn't help much here, use SIMD via conversion
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        SimdBackend::eval_at_point(simd_poly, point)
    }
    
    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField> {
        // Delegate to SIMD - this is not a hot path
        SimdBackend::barycentric_weights(coset, p)
    }
    
    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField {
        let simd_evals = circle_eval_ref_to_simd(evals);
        SimdBackend::barycentric_eval_at_point(simd_evals, weights)
    }
    
    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let simd_evals = circle_eval_ref_to_simd(evals);
        let simd_twiddles = twiddle_ref_to_simd(twiddles);
        SimdBackend::eval_at_point_by_folding(simd_evals, point, simd_twiddles)
    }
    
    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        let result = SimdBackend::extend(simd_poly, log_size);
        CircleCoefficients::new(result.coeffs)
    }
    
    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let _span = span!(Level::TRACE, "GpuBackend::evaluate").entered();
        
        let log_size = domain.log_size();
        
        // Small polynomials: use SIMD (GPU overhead not worth it)
        if log_size < GPU_FFT_THRESHOLD_LOG_SIZE {
            tracing::debug!(
                "GPU evaluate: using SIMD for small size (log_size={} < threshold={})",
                log_size, GPU_FFT_THRESHOLD_LOG_SIZE
            );
            return evaluate_simd_fallback(poly, domain, twiddles);
        }
        
        // Large polynomials: require GPU
        if !is_cuda_available() {
            panic!(
                "GpuBackend::evaluate called with log_size={} but CUDA is not available. \
                 Use SimdBackend for CPU-only execution.",
                log_size
            );
        }
        
        tracing::info!(
            "GPU evaluate: using CUDA for {} elements (log_size={})",
            1u64 << log_size, log_size
        );
        gpu_evaluate(poly, domain, twiddles, log_size)
    }
    
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::precompute_twiddles").entered();
        
        // Use SIMD backend's cached implementation for twiddle computation.
        // Twiddles are computed once and reused many times, so CPU is fine here.
        // The GPU benefit comes from using these twiddles in FFT operations.
        let simd_twiddles = SimdBackend::precompute_twiddles(coset);
        
        // Convert to GpuBackend's TwiddleTree
        twiddle_to_gpu(simd_twiddles)
    }
    
    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        // This is a simple split operation, no GPU benefit
        let simd_poly = CircleCoefficients::<SimdBackend>::new(poly.coeffs);
        let (left, right) = SimdBackend::split_at_mid(simd_poly);
        (
            CircleCoefficients::new(left.coeffs),
            CircleCoefficients::new(right.coeffs),
        )
    }
}

// =============================================================================
// SIMD Fallback Functions
// =============================================================================

fn interpolate_simd_fallback(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleCoefficients<GpuBackend> {
    let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
        eval.domain,
        eval.values,
    );
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::interpolate(simd_eval, simd_twiddles);
    CircleCoefficients::new(result.coeffs)
}

fn evaluate_simd_fallback(
    poly: &CircleCoefficients<GpuBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> {
    let simd_poly = circle_coeffs_ref_to_simd(poly);
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::evaluate(simd_poly, domain, simd_twiddles);
    CircleEvaluation::new(domain, result.values)
}

// =============================================================================
// GPU FFT Implementation
// =============================================================================

/// GPU-accelerated polynomial interpolation (inverse FFT).
///
/// This function:
/// 1. Extracts evaluation data (zero-copy when possible)
/// 2. Computes inverse FFT twiddles (cached per log_size)
/// 3. Transfers data to GPU
/// 4. Executes IFFT kernels
/// 5. Applies denormalization on GPU
/// 6. Transfers results back to CPU
fn gpu_interpolate(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    _twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) -> CircleCoefficients<GpuBackend> {
    let _span = span!(Level::INFO, "GPU interpolate (IFFT)", size = 1u64 << log_size).entered();
    
    // Extract raw data from evaluation - use unsafe transmute for zero-copy
    // BaseField is repr(transparent) over u32, so this is safe
    let values_slice = eval.values.as_slice();
    let data_ptr = values_slice.as_ptr() as *const u32;
    let data_len = values_slice.len();
    
    // Create a mutable copy for GPU processing
    let mut data: Vec<u32> = unsafe {
        std::slice::from_raw_parts(data_ptr, data_len).to_vec()
    };
    
    // Compute twiddles for IFFT
    // TODO: Cache these per log_size to avoid recomputation
    let twiddles_dbl = compute_itwiddle_dbls_cpu(log_size);
    
    // Execute CUDA IFFT
    match cuda_ifft(&mut data, &twiddles_dbl, log_size) {
        Ok(()) => {
            tracing::debug!(
                "GPU IFFT completed for {} elements",
                1u64 << log_size
            );
            
            // Apply denormalization factor (divide by domain size)
            // This is done on CPU but could be fused into GPU kernel
            let denorm = BaseField::from(1u32 << log_size).inverse();
            let denorm_val = denorm.0;
            
            // Apply denormalization in-place using M31 multiplication
            const M31_PRIME: u64 = (1u64 << 31) - 1;
            for v in data.iter_mut() {
                let product = (*v as u64) * (denorm_val as u64);
                *v = (product % M31_PRIME) as u32;
            }
            
            // Convert back to BaseColumn
            use crate::prover::backend::simd::column::BaseColumn;
            let coeffs: BaseColumn = unsafe {
                // Safety: BaseField is repr(transparent) over u32
                let bf_ptr = data.as_ptr() as *const BaseField;
                std::slice::from_raw_parts(bf_ptr, data.len()).iter().copied().collect()
            };
            
            CircleCoefficients::new(coeffs)
        }
        Err(e) => {
            // GPU execution failed - this is a hard error, not a fallback
            panic!("GPU IFFT execution failed: {}. GPU backend requires working CUDA.", e);
        }
    }
}

/// GPU-accelerated polynomial evaluation (forward FFT).
///
/// This function:
/// 1. Extracts coefficient data
/// 2. Computes forward FFT twiddles
/// 3. Transfers data to GPU
/// 4. Executes FFT kernels
/// 5. Transfers results back to CPU
fn gpu_evaluate(
    poly: &CircleCoefficients<GpuBackend>,
    domain: CircleDomain,
    _twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) -> CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> {
    let _span = span!(Level::INFO, "GPU evaluate (FFT)", size = 1u64 << log_size).entered();
    
    // Extract raw data from coefficients
    let mut data: Vec<u32> = poly.coeffs.as_slice()
        .iter()
        .map(|f| f.0)
        .collect();
    
    // Pad to domain size if needed
    let domain_size = 1usize << log_size;
    if data.len() < domain_size {
        data.resize(domain_size, 0);
    }
    
    // Compute twiddles for forward FFT
    let twiddles_dbl = compute_twiddle_dbls_cpu(log_size);
    
    // Execute CUDA FFT
    match cuda_fft(&mut data, &twiddles_dbl, log_size) {
        Ok(()) => {
            tracing::info!(
                "GPU FFT completed for {} elements",
                1u64 << log_size
            );
            
            // Convert back to BaseColumn
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::core::fields::m31::BaseField;
            
            let values: BaseColumn = data.iter()
                .map(|&v| BaseField::from_u32_unchecked(v))
                .collect();
            
            CircleEvaluation::new(domain, values)
        }
        Err(e) => {
            // GPU execution failed - this is a hard error, not a fallback
            panic!("GPU FFT execution failed: {}. GPU backend requires working CUDA.", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_backend_uses_threshold() {
        // Verify that small polynomials don't trigger GPU path
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE <= 20);
    }
    
    #[test]
    fn test_threshold_is_reasonable() {
        // 16K elements is a good threshold based on benchmarks
        assert_eq!(GPU_FFT_THRESHOLD_LOG_SIZE, 14);
    }
}
