//! GPU-accelerated GKR operations.
//!
//! This module implements [`GkrOps`] and [`MleOps`] for [`GpuBackend`].
//!
//! GKR (Goldwasser-Kalai-Rothblum) is used for lookup arguments.
//! The operations are complex and have irregular access patterns,
//! so we delegate to SIMD for now.
//!
//! # Note on Conversions
//!
//! The GKR types (Mle, Layer, GkrMultivariatePolyOracle) contain backend-specific
//! column types internally. Since GpuBackend and SimdBackend share the same
//! underlying column types, we use the conversion module for type-safe conversions.

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

/// Convert GpuBackend Mle to SimdBackend.
///
/// # Safety
/// This is safe because both backends use identical column types internally.
#[inline]
fn mle_base_to_simd(mle: Mle<GpuBackend, BaseField>) -> Mle<SimdBackend, BaseField> {
    // Mle<B, F> wraps a column. Since GpuBackend::Column = SimdBackend::Column,
    // we can safely transmute.
    unsafe { std::mem::transmute(mle) }
}

/// Convert SimdBackend Mle to GpuBackend.
#[inline]
fn mle_secure_to_gpu(mle: Mle<SimdBackend, SecureField>) -> Mle<GpuBackend, SecureField> {
    unsafe { std::mem::transmute(mle) }
}

/// Convert GpuBackend Mle (SecureField) to SimdBackend.
#[inline]
fn mle_secure_to_simd(mle: Mle<GpuBackend, SecureField>) -> Mle<SimdBackend, SecureField> {
    unsafe { std::mem::transmute(mle) }
}

/// Convert GpuBackend Layer reference to SimdBackend.
#[inline]
fn layer_ref_to_simd<'a>(layer: &'a Layer<GpuBackend>) -> &'a Layer<SimdBackend> {
    unsafe { std::mem::transmute(layer) }
}

/// Convert SimdBackend Layer to GpuBackend.
#[inline]
fn layer_to_gpu(layer: Layer<SimdBackend>) -> Layer<GpuBackend> {
    unsafe { std::mem::transmute(layer) }
}

/// Convert GpuBackend GkrMultivariatePolyOracle reference to SimdBackend.
#[inline]
fn gkr_oracle_ref_to_simd<'a>(
    oracle: &'a GkrMultivariatePolyOracle<'a, GpuBackend>
) -> &'a GkrMultivariatePolyOracle<'a, SimdBackend> {
    unsafe { std::mem::transmute(oracle) }
}

impl MleOps<BaseField> for GpuBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        // Convert to SimdBackend, process, convert back
        let simd_mle = mle_base_to_simd(mle);
        let result = SimdBackend::fix_first_variable(simd_mle, assignment);
        mle_secure_to_gpu(result)
    }
}

impl MleOps<SecureField> for GpuBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let simd_mle = mle_secure_to_simd(mle);
        let result = SimdBackend::fix_first_variable(simd_mle, assignment);
        mle_secure_to_gpu(result)
    }
}

// =============================================================================
// GkrOps Implementation
// =============================================================================

impl GkrOps for GpuBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let result = SimdBackend::gen_eq_evals(y, v);
        mle_secure_to_gpu(result)
    }
    
    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        let simd_layer = layer_ref_to_simd(layer);
        let result = SimdBackend::next_layer(simd_layer);
        layer_to_gpu(result)
    }
    
    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        // This is complex and has irregular access patterns
        // GKR operations are not highly parallelizable, so SIMD is appropriate
        let simd_h = gkr_oracle_ref_to_simd(h);
        SimdBackend::sum_as_poly_in_first_variable(simd_h, claim)
    }
}
