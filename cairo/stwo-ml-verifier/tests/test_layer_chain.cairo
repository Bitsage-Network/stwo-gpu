use stwo_ml_verifier::layer_chain::{
    LayerProofHeader,
    verify_layer_chain, compute_chain_commitment,
};

// ============================================================================
// verify_layer_chain
// ============================================================================

#[test]
fn test_empty_chain() {
    let headers: Array<LayerProofHeader> = array![];
    let result = verify_layer_chain(headers.span(), 0, 0);
    assert!(result.is_valid, "empty chain should be valid");
    assert!(result.num_layers == 0, "empty chain has 0 layers");
    assert!(result.broken_at == 0, "no broken link");
}

#[test]
fn test_single_layer_valid() {
    let headers = array![
        LayerProofHeader {
            layer_index: 0,
            input_commitment: 0x111,
            output_commitment: 0x222,
        },
    ];
    // model_input_commitment=0 skips first check, model_output_commitment=0 skips last
    let result = verify_layer_chain(headers.span(), 0, 0);
    assert!(result.is_valid, "single layer with no constraints should be valid");
    assert!(result.num_layers == 1, "should have 1 layer");
}

#[test]
fn test_two_layers_valid() {
    let headers = array![
        LayerProofHeader {
            layer_index: 0,
            input_commitment: 0x111,
            output_commitment: 0x222,
        },
        LayerProofHeader {
            layer_index: 1,
            input_commitment: 0x222, // matches output of layer 0
            output_commitment: 0x333,
        },
    ];
    let result = verify_layer_chain(headers.span(), 0, 0);
    assert!(result.is_valid, "two connected layers should be valid");
    assert!(result.num_layers == 2, "should have 2 layers");
}

#[test]
fn test_chain_break_middle() {
    let headers = array![
        LayerProofHeader {
            layer_index: 0,
            input_commitment: 0x111,
            output_commitment: 0x222,
        },
        LayerProofHeader {
            layer_index: 1,
            input_commitment: 0x999, // BREAK: doesn't match 0x222
            output_commitment: 0x333,
        },
    ];
    let result = verify_layer_chain(headers.span(), 0, 0);
    assert!(!result.is_valid, "broken chain should be invalid");
    assert!(result.broken_at == 1, "break should be at index 1");
}

#[test]
fn test_first_layer_input_mismatch() {
    let headers = array![
        LayerProofHeader {
            layer_index: 0,
            input_commitment: 0x111,
            output_commitment: 0x222,
        },
    ];
    // model_input_commitment = 0xAAA, doesn't match 0x111
    let result = verify_layer_chain(headers.span(), 0xAAA, 0);
    assert!(!result.is_valid, "first layer input mismatch should fail");
    assert!(result.broken_at == 0, "break at layer 0");
}

#[test]
fn test_last_layer_output_mismatch() {
    let headers = array![
        LayerProofHeader {
            layer_index: 0,
            input_commitment: 0x111,
            output_commitment: 0x222,
        },
    ];
    // model_output_commitment = 0xBBB, doesn't match 0x222
    let result = verify_layer_chain(headers.span(), 0, 0xBBB);
    assert!(!result.is_valid, "last layer output mismatch should fail");
    assert!(result.broken_at == 0, "break at last layer (index 0)");
}

#[test]
fn test_zero_commitment_skips_first_check() {
    let headers = array![
        LayerProofHeader {
            layer_index: 0,
            input_commitment: 0x999, // any value
            output_commitment: 0x222,
        },
    ];
    // model_input_commitment = 0, should skip check
    let result = verify_layer_chain(headers.span(), 0, 0);
    assert!(result.is_valid, "zero model_input should skip first check");
}

#[test]
fn test_three_layers_valid() {
    let headers = array![
        LayerProofHeader { layer_index: 0, input_commitment: 0xA, output_commitment: 0xB },
        LayerProofHeader { layer_index: 1, input_commitment: 0xB, output_commitment: 0xC },
        LayerProofHeader { layer_index: 2, input_commitment: 0xC, output_commitment: 0xD },
    ];
    let result = verify_layer_chain(headers.span(), 0xA, 0xD);
    assert!(result.is_valid, "3 connected layers with matching endpoints should be valid");
    assert!(result.num_layers == 3, "should have 3 layers");
}

// ============================================================================
// compute_chain_commitment
// ============================================================================

#[test]
fn test_compute_chain_commitment_empty() {
    let headers: Array<LayerProofHeader> = array![];
    let result = compute_chain_commitment(headers.span());
    assert!(result == 0, "empty chain commitment should be 0");
}

#[test]
fn test_compute_chain_commitment_deterministic() {
    let headers = array![
        LayerProofHeader { layer_index: 0, input_commitment: 0x111, output_commitment: 0x222 },
    ];
    let c1 = compute_chain_commitment(headers.span());
    let headers2 = array![
        LayerProofHeader { layer_index: 0, input_commitment: 0x111, output_commitment: 0x222 },
    ];
    let c2 = compute_chain_commitment(headers2.span());
    assert!(c1 == c2, "same input should produce same commitment");
    assert!(c1 != 0, "non-empty chain should produce non-zero commitment");
}

#[test]
fn test_compute_chain_commitment_ordering() {
    let headers_ab = array![
        LayerProofHeader { layer_index: 0, input_commitment: 0xA, output_commitment: 0xB },
        LayerProofHeader { layer_index: 1, input_commitment: 0xC, output_commitment: 0xD },
    ];
    let headers_cd = array![
        LayerProofHeader { layer_index: 1, input_commitment: 0xC, output_commitment: 0xD },
        LayerProofHeader { layer_index: 0, input_commitment: 0xA, output_commitment: 0xB },
    ];
    let c1 = compute_chain_commitment(headers_ab.span());
    let c2 = compute_chain_commitment(headers_cd.span());
    assert!(c1 != c2, "different order should produce different commitment");
}
