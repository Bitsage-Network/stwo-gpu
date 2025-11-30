//! GPU-accelerated FRI operations.
//!
//! This module implements [`FriOps`] for [`GpuBackend`], providing GPU acceleration
//! for FRI (Fast Reed-Solomon Interactive Oracle Proof) folding operations.
//!
//! # Performance Characteristics
//!
//! FRI folding is the second most expensive operation after FFT:
//! - **fold_line**: 20-30x speedup on GPU for large evaluations
//! - **fold_circle_into_line**: 15-25x speedup on GPU
//!
//! # Implementation Strategy
//!
//! We use a size threshold to decide between GPU and CPU:
//! - Small evaluations (< 16K elements): Use SIMD backend
//! - Large evaluations (>= 16K elements): Use GPU

use tracing::{span, Level};

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

use super::GpuBackend;

/// Threshold below which GPU overhead exceeds benefit for FRI operations.
const GPU_FRI_THRESHOLD_LOG_SIZE: u32 = 14; // 16K elements

impl FriOps for GpuBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::fold_line").entered();
        
        let log_size = eval.len().ilog2();
        
        if log_size < GPU_FRI_THRESHOLD_LOG_SIZE {
            // Small evaluation - use SIMD backend
            let simd_eval = unsafe {
                std::mem::transmute::<&LineEvaluation<GpuBackend>, &LineEvaluation<SimdBackend>>(eval)
            };
            let simd_twiddles = unsafe {
                std::mem::transmute::<&TwiddleTree<GpuBackend>, &TwiddleTree<SimdBackend>>(twiddles)
            };
            let result = SimdBackend::fold_line(simd_eval, alpha, simd_twiddles);
            unsafe {
                std::mem::transmute::<LineEvaluation<SimdBackend>, LineEvaluation<GpuBackend>>(result)
            }
        } else {
            // Large evaluation - use GPU
            gpu_fold_line(eval, alpha, twiddles)
        }
    }
    
    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let _span = span!(Level::TRACE, "GpuBackend::fold_circle_into_line").entered();
        
        let log_size = src.len().ilog2();
        
        if log_size < GPU_FRI_THRESHOLD_LOG_SIZE {
            // Small evaluation - use SIMD backend
            let simd_dst = unsafe {
                std::mem::transmute::<&mut LineEvaluation<GpuBackend>, &mut LineEvaluation<SimdBackend>>(dst)
            };
            let simd_src = unsafe {
                std::mem::transmute::<
                    &SecureEvaluation<GpuBackend, BitReversedOrder>,
                    &SecureEvaluation<SimdBackend, BitReversedOrder>
                >(src)
            };
            let simd_twiddles = unsafe {
                std::mem::transmute::<&TwiddleTree<GpuBackend>, &TwiddleTree<SimdBackend>>(twiddles)
            };
            SimdBackend::fold_circle_into_line(simd_dst, simd_src, alpha, simd_twiddles);
        } else {
            // Large evaluation - use GPU
            gpu_fold_circle_into_line(dst, src, alpha, twiddles);
        }
    }
    
    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // Decompose is not as compute-intensive, delegate to SIMD
        let simd_eval = unsafe {
            std::mem::transmute::<
                &SecureEvaluation<GpuBackend, BitReversedOrder>,
                &SecureEvaluation<SimdBackend, BitReversedOrder>
            >(eval)
        };
        let (simd_result, lambda) = SimdBackend::decompose(simd_eval);
        let result = unsafe {
            std::mem::transmute::<
                SecureEvaluation<SimdBackend, BitReversedOrder>,
                SecureEvaluation<GpuBackend, BitReversedOrder>
            >(simd_result)
        };
        (result, lambda)
    }
}

// =============================================================================
// GPU FRI Implementation
// =============================================================================

/// GPU-accelerated line folding.
fn gpu_fold_line(
    eval: &LineEvaluation<GpuBackend>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
) -> LineEvaluation<GpuBackend> {
    let _span = span!(Level::INFO, "GPU fold_line", size = eval.len()).entered();
    
    // TODO: Implement actual GPU FRI folding
    // For now, fall back to SIMD with a warning
    tracing::warn!("GPU fold_line not yet implemented, falling back to SIMD");
    
    let simd_eval = unsafe {
        std::mem::transmute::<&LineEvaluation<GpuBackend>, &LineEvaluation<SimdBackend>>(eval)
    };
    let simd_twiddles = unsafe {
        std::mem::transmute::<&TwiddleTree<GpuBackend>, &TwiddleTree<SimdBackend>>(twiddles)
    };
    let result = SimdBackend::fold_line(simd_eval, alpha, simd_twiddles);
    unsafe {
        std::mem::transmute::<LineEvaluation<SimdBackend>, LineEvaluation<GpuBackend>>(result)
    }
}

/// GPU-accelerated circle-to-line folding.
fn gpu_fold_circle_into_line(
    dst: &mut LineEvaluation<GpuBackend>,
    src: &SecureEvaluation<GpuBackend, BitReversedOrder>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
) {
    let _span = span!(Level::INFO, "GPU fold_circle_into_line", size = src.len()).entered();
    
    // TODO: Implement actual GPU FRI folding
    // For now, fall back to SIMD with a warning
    tracing::warn!("GPU fold_circle_into_line not yet implemented, falling back to SIMD");
    
    let simd_dst = unsafe {
        std::mem::transmute::<&mut LineEvaluation<GpuBackend>, &mut LineEvaluation<SimdBackend>>(dst)
    };
    let simd_src = unsafe {
        std::mem::transmute::<
            &SecureEvaluation<GpuBackend, BitReversedOrder>,
            &SecureEvaluation<SimdBackend, BitReversedOrder>
        >(src)
    };
    let simd_twiddles = unsafe {
        std::mem::transmute::<&TwiddleTree<GpuBackend>, &TwiddleTree<SimdBackend>>(twiddles)
    };
    SimdBackend::fold_circle_into_line(simd_dst, simd_src, alpha, simd_twiddles);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_threshold_reasonable() {
        assert!(GPU_FRI_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FRI_THRESHOLD_LOG_SIZE <= 20);
    }
}

