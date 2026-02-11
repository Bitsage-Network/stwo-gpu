//! TEE (Trusted Execution Environment) integration for GPU proving.
//!
//! Provides interfaces for running GPU proofs within confidential compute
//! environments (NVIDIA CC-On, Intel TDX, etc.)
//!
//! # Note
//!
//! This module is a stub. TEE attestation is handled at the application level
//! (stwo-ml's receipt module), not within the core prover library.

/// TEE attestation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeeStatus {
    /// TEE is available and active.
    Active,
    /// TEE is not available on this hardware.
    Unavailable,
    /// TEE is available but not enabled.
    Disabled,
}

/// Check if the current GPU environment supports TEE.
pub fn tee_status() -> TeeStatus {
    #[cfg(feature = "cuda-runtime")]
    {
        // Check for NVIDIA Confidential Computing support
        // This requires H100/H200/B200 with CC-On firmware
        if let Ok(executor) = super::cuda_executor::get_cuda_executor() {
            let (major, _minor) = executor.device_info.compute_capability;
            if major >= 9 {
                // Hopper+ architecture supports CC
                return TeeStatus::Disabled; // Available but needs CC-On firmware
            }
        }
    }
    TeeStatus::Unavailable
}

/// Check if the GPU is running in confidential compute mode.
pub fn is_confidential_compute_active() -> bool {
    matches!(tee_status(), TeeStatus::Active)
}
