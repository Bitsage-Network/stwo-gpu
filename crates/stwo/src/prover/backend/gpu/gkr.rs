//! GPU-accelerated GKR operations.
//!
//! This module implements [`GkrOps`] and [`MleOps`] for [`GpuBackend`].
//!
//! GKR (Goldwasser-Kalai-Rothblum) is used for lookup arguments.
//! The operations are complex and have irregular access patterns,
//! so we delegate to SIMD for now.

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::lookups::gkr_prover::{
    GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::prover::lookups::utils::UnivariatePoly;
use crate::prover::lookups::mle::{Mle, MleOps};

use super::GpuBackend;

// =============================================================================
// MleOps Implementation
// =============================================================================

impl MleOps<BaseField> for GpuBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        // Convert to SimdBackend, process, convert back
        let simd_mle = unsafe {
            std::mem::transmute::<Mle<GpuBackend, BaseField>, Mle<SimdBackend, BaseField>>(mle)
        };
        let result = SimdBackend::fix_first_variable(simd_mle, assignment);
        unsafe {
            std::mem::transmute::<Mle<SimdBackend, SecureField>, Mle<GpuBackend, SecureField>>(result)
        }
    }
}

impl MleOps<SecureField> for GpuBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let simd_mle = unsafe {
            std::mem::transmute::<Mle<GpuBackend, SecureField>, Mle<SimdBackend, SecureField>>(mle)
        };
        let result = SimdBackend::fix_first_variable(simd_mle, assignment);
        unsafe {
            std::mem::transmute::<Mle<SimdBackend, SecureField>, Mle<GpuBackend, SecureField>>(result)
        }
    }
}

// =============================================================================
// GkrOps Implementation
// =============================================================================

impl GkrOps for GpuBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let result = SimdBackend::gen_eq_evals(y, v);
        unsafe {
            std::mem::transmute::<Mle<SimdBackend, SecureField>, Mle<GpuBackend, SecureField>>(result)
        }
    }
    
    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        let simd_layer = unsafe {
            std::mem::transmute::<&Layer<GpuBackend>, &Layer<SimdBackend>>(layer)
        };
        let result = SimdBackend::next_layer(simd_layer);
        unsafe {
            std::mem::transmute::<Layer<SimdBackend>, Layer<GpuBackend>>(result)
        }
    }
    
    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        // This is complex and has irregular access patterns
        // For now, we need to handle this carefully
        // TODO: Proper implementation
        let simd_h = unsafe {
            std::mem::transmute::<
                &GkrMultivariatePolyOracle<'_, GpuBackend>,
                &GkrMultivariatePolyOracle<'_, SimdBackend>
            >(h)
        };
        SimdBackend::sum_as_poly_in_first_variable(simd_h, claim)
    }
}

