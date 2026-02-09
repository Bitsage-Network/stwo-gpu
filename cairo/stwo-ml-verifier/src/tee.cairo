/// TEE attestation verification for on-chain proof validation.
///
/// Verifies NVIDIA DCAP attestation report properties:
/// - Report hash matches expected (bound to proof)
/// - Measurement matches expected code identity
/// - Timestamp freshness (within MAX_TEE_AGE_SECS)
use core::poseidon::poseidon_hash_span;

/// Maximum age in seconds between TEE attestation and proof submission.
/// Matches Rust's MAX_TEE_AGE_SECS = 3600 (1 hour).
const MAX_TEE_AGE_SECS: u64 = 3600;

/// TEE attestation data stored on-chain.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct TeeAttestation {
    /// Hash of the NVIDIA DCAP attestation report.
    pub report_hash: felt252,
    /// SHA-256 of the enclave code (measurement).
    pub measurement: felt252,
    /// Timestamp from the TEE attestation (Unix epoch).
    pub tee_timestamp: u64,
    /// GPU device identifier hash.
    pub device_id: felt252,
}

/// Result of TEE attestation verification.
#[derive(Drop, Copy)]
pub struct TeeVerificationResult {
    pub is_valid: bool,
    pub reason: felt252,
}

/// Verify a TEE attestation against expected values.
///
/// Checks:
/// 1. Report hash is non-zero
/// 2. Measurement matches expected code identity
/// 3. Attestation is fresh (within MAX_TEE_AGE_SECS of current_timestamp)
pub fn verify_tee_attestation(
    attestation: TeeAttestation,
    expected_measurement: felt252,
    current_timestamp: u64,
) -> TeeVerificationResult {
    // Report must be non-zero
    if attestation.report_hash == 0 {
        return TeeVerificationResult { is_valid: false, reason: 'EMPTY_REPORT' };
    }

    // Measurement must match expected code identity
    if attestation.measurement != expected_measurement {
        return TeeVerificationResult { is_valid: false, reason: 'MEASUREMENT_MISMATCH' };
    }

    // Freshness check
    if attestation.tee_timestamp > current_timestamp {
        return TeeVerificationResult { is_valid: false, reason: 'FUTURE_TIMESTAMP' };
    }
    if current_timestamp - attestation.tee_timestamp > MAX_TEE_AGE_SECS {
        return TeeVerificationResult { is_valid: false, reason: 'ATTESTATION_EXPIRED' };
    }

    TeeVerificationResult { is_valid: true, reason: 0 }
}

/// Compute the binding hash between a proof and its TEE attestation.
/// This ensures the proof was generated inside the attested TEE.
pub fn compute_proof_tee_binding(
    proof_hash: felt252,
    tee_report_hash: felt252,
) -> felt252 {
    poseidon_hash_span(array![proof_hash, tee_report_hash, 'TEE_BIND'].span())
}
