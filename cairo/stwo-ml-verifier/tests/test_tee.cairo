use stwo_ml_verifier::tee::{
    TeeAttestation,
    verify_tee_attestation, compute_proof_tee_binding,
};

fn make_valid_attestation() -> TeeAttestation {
    TeeAttestation {
        report_hash: 0x1234,
        measurement: 0xABCD,
        tee_timestamp: 1000,
        device_id: 0xDEF0,
    }
}

// ============================================================================
// verify_tee_attestation
// ============================================================================

#[test]
fn test_valid_attestation() {
    let att = make_valid_attestation();
    let result = verify_tee_attestation(att, 0xABCD, 2000);
    assert!(result.is_valid, "valid attestation should pass");
    assert!(result.reason == 0, "valid reason should be 0");
}

#[test]
fn test_zero_report_hash() {
    let att = TeeAttestation {
        report_hash: 0, // zero
        measurement: 0xABCD,
        tee_timestamp: 1000,
        device_id: 0xDEF0,
    };
    let result = verify_tee_attestation(att, 0xABCD, 2000);
    assert!(!result.is_valid, "zero report hash should fail");
    assert!(result.reason == 'EMPTY_REPORT', "reason should be EMPTY_REPORT");
}

#[test]
fn test_measurement_mismatch() {
    let att = make_valid_attestation();
    let result = verify_tee_attestation(att, 0x9999, 2000); // wrong measurement
    assert!(!result.is_valid, "measurement mismatch should fail");
    assert!(result.reason == 'MEASUREMENT_MISMATCH', "reason should be MEASUREMENT_MISMATCH");
}

#[test]
fn test_future_timestamp() {
    let att = TeeAttestation {
        report_hash: 0x1234,
        measurement: 0xABCD,
        tee_timestamp: 3000, // future
        device_id: 0xDEF0,
    };
    let result = verify_tee_attestation(att, 0xABCD, 2000);
    assert!(!result.is_valid, "future timestamp should fail");
    assert!(result.reason == 'FUTURE_TIMESTAMP', "reason should be FUTURE_TIMESTAMP");
}

#[test]
fn test_expired_attestation() {
    let att = TeeAttestation {
        report_hash: 0x1234,
        measurement: 0xABCD,
        tee_timestamp: 1000,
        device_id: 0xDEF0,
    };
    // current_timestamp - tee_timestamp = 5000 - 1000 = 4000 > 3600
    let result = verify_tee_attestation(att, 0xABCD, 5000);
    assert!(!result.is_valid, "expired attestation should fail");
    assert!(result.reason == 'ATTESTATION_EXPIRED', "reason should be ATTESTATION_EXPIRED");
}

#[test]
fn test_exactly_3600s() {
    let att = TeeAttestation {
        report_hash: 0x1234,
        measurement: 0xABCD,
        tee_timestamp: 1000,
        device_id: 0xDEF0,
    };
    // current - tee = 4600 - 1000 = 3600, exactly MAX_TEE_AGE_SECS
    // The check is: `current - tee > 3600` → 3600 > 3600 is false → passes
    let result = verify_tee_attestation(att, 0xABCD, 4600);
    assert!(result.is_valid, "exactly 3600s should pass");
}

#[test]
fn test_3601s() {
    let att = TeeAttestation {
        report_hash: 0x1234,
        measurement: 0xABCD,
        tee_timestamp: 1000,
        device_id: 0xDEF0,
    };
    // current - tee = 4601 - 1000 = 3601 > 3600 → expired
    let result = verify_tee_attestation(att, 0xABCD, 4601);
    assert!(!result.is_valid, "3601s should fail");
}

// ============================================================================
// compute_proof_tee_binding
// ============================================================================

#[test]
fn test_proof_tee_binding_deterministic() {
    let b1 = compute_proof_tee_binding(0x1111, 0x2222);
    let b2 = compute_proof_tee_binding(0x1111, 0x2222);
    assert!(b1 == b2, "same inputs should produce same binding");
    assert!(b1 != 0, "binding should be non-zero");
}

#[test]
fn test_proof_tee_binding_different_inputs() {
    let b1 = compute_proof_tee_binding(0x1111, 0x2222);
    let b2 = compute_proof_tee_binding(0x1111, 0x3333);
    assert!(b1 != b2, "different report hashes should produce different bindings");

    let b3 = compute_proof_tee_binding(0x4444, 0x2222);
    assert!(b1 != b3, "different proof hashes should produce different bindings");
}
