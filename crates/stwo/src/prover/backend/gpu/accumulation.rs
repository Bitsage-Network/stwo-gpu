//! GPU-accelerated accumulation operations.
//!
//! This module implements [`AccumulationOps`] for [`GpuBackend`].

use crate::core::fields::qm31::SecureField;
use crate::prover::AccumulationOps;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::secure_column::SecureColumnByCoords;

use super::GpuBackend;

impl AccumulationOps for GpuBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        // Accumulation is a simple element-wise addition
        // GPU could help for very large columns, but overhead usually not worth it
        let simd_column = unsafe {
            std::mem::transmute::<
                &mut SecureColumnByCoords<GpuBackend>,
                &mut SecureColumnByCoords<SimdBackend>
            >(column)
        };
        let simd_other = unsafe {
            std::mem::transmute::<
                &SecureColumnByCoords<GpuBackend>,
                &SecureColumnByCoords<SimdBackend>
            >(other)
        };
        SimdBackend::accumulate(simd_column, simd_other);
    }
    
    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        // Power generation is sequential, no GPU benefit
        SimdBackend::generate_secure_powers(felt, n_powers)
    }
}

