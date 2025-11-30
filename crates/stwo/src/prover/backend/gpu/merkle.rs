//! GPU-accelerated Merkle tree operations.
//!
//! This module implements [`MerkleOps`] for [`GpuBackend`].
//!
//! Merkle hashing is highly parallelizable and can benefit from GPU,
//! but the speedup is modest (1.3-2x) compared to FFT (50-100x).
//! We delegate to SIMD for now.

use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::core::fields::m31::BaseField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::vcs::ops::MerkleOps;

use super::GpuBackend;

impl MerkleOps<Blake2sMerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        // Merkle hashing - delegate to SIMD for now
        // TODO: Implement GPU Merkle hashing for very large trees
        let simd_prev = prev_layer.map(|p| unsafe {
            std::mem::transmute::<&Col<GpuBackend, Blake2sHash>, &Col<SimdBackend, Blake2sHash>>(p)
        });
        
        let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns.iter().map(|c| unsafe {
            std::mem::transmute::<&&Col<GpuBackend, BaseField>, &Col<SimdBackend, BaseField>>(c)
        }).collect();
        
        let result = <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, 
            simd_prev, 
            &simd_columns
        );
        
        unsafe {
            std::mem::transmute::<Col<SimdBackend, Blake2sHash>, Col<GpuBackend, Blake2sHash>>(result)
        }
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        // Merkle hashing - delegate to SIMD for now
        let simd_prev = prev_layer.map(|p| unsafe {
            std::mem::transmute::<&Col<GpuBackend, Blake2sHash>, &Col<SimdBackend, Blake2sHash>>(p)
        });
        
        let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns.iter().map(|c| unsafe {
            std::mem::transmute::<&&Col<GpuBackend, BaseField>, &Col<SimdBackend, BaseField>>(c)
        }).collect();
        
        let result = <SimdBackend as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
            log_size, 
            simd_prev, 
            &simd_columns
        );
        
        unsafe {
            std::mem::transmute::<Col<SimdBackend, Blake2sHash>, Col<GpuBackend, Blake2sHash>>(result)
        }
    }
}

