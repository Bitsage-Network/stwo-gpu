/// Cross-language hash consistency tests.
///
/// These tests compute Poseidon hashes with known inputs and assert the exact
/// output values. The Rust tests in stwo-ml/src/recursive.rs compute the same
/// hashes — if both sides produce the same values, the implementations are consistent.

use core::poseidon::poseidon_hash_span;
use stwo_ml_verify_core::layer_chain::{
    LayerProofHeader, compute_chain_commitment,
};

/// Domain separator used by stwo-ml-recursive aggregate hash.
const AGGREGATE_DOMAIN: felt252 = 'OBELYSK_RECURSIVE_V1';

// ============================================================================
// Domain separator encoding
// ============================================================================

#[test]
fn test_domain_separator_value() {
    // Cairo short string 'OBELYSK_RECURSIVE_V1' should equal the hex encoding
    // of the ASCII bytes packed big-endian.
    // O=0x4f B=0x42 E=0x45 L=0x4c Y=0x59 S=0x53 K=0x4b _=0x5f
    // R=0x52 E=0x45 C=0x43 U=0x55 R=0x52 S=0x53 I=0x49 V=0x56
    // E=0x45 _=0x5f V=0x56 1=0x31
    // = 0x4f42454c59534b5f5245435552534956455f5631
    assert!(
        AGGREGATE_DOMAIN == 0x4f42454c59534b5f5245435552534956455f5631,
        "Domain separator encoding mismatch",
    );
}

// ============================================================================
// Chain commitment
// ============================================================================

#[test]
fn test_chain_commitment_empty() {
    let headers: Array<LayerProofHeader> = array![];
    let commitment = compute_chain_commitment(headers.span());
    assert!(commitment == 0, "Empty chain should have 0 commitment");
}

#[test]
fn test_chain_commitment_two_layers_cross_language() {
    // Input: [(0, 100, 200), (1, 200, 300)]
    // = poseidon_hash_span([0, 100, 200, 1, 200, 300])
    let headers = array![
        LayerProofHeader { layer_index: 0, input_commitment: 100, output_commitment: 200 },
        LayerProofHeader { layer_index: 1, input_commitment: 200, output_commitment: 300 },
    ];
    let commitment = compute_chain_commitment(headers.span());

    // Rust value: 0x0177f5c4aa4b7c74b43770aa0282670d57619387f4faed70e8b412987046d0e1
    assert!(
        commitment == 0x0177f5c4aa4b7c74b43770aa0282670d57619387f4faed70e8b412987046d0e1,
        "Chain commitment mismatch with Rust",
    );
}

// ============================================================================
// Aggregate hash (simple: 0 proofs, 0 layers)
// ============================================================================

#[test]
fn test_aggregate_hash_simple_cross_language() {
    // Inputs: domain=OBELYSK_RECURSIVE_V1, model_id=1, model_commitment=42,
    //         io_commitment=99, chain_commitment=0, tee_hash=0, proof_hashes=[]
    let hash_inputs: Array<felt252> = array![
        AGGREGATE_DOMAIN, // domain
        1,                // model_id
        42,               // model_commitment
        99,               // io_commitment
        0,                // chain_commitment (empty)
        0,                // tee_report_hash
    ];
    let hash = poseidon_hash_span(hash_inputs.span());

    // Rust value: 0x06abb9af4d18d8b49300457471e4a93dcd485013cb4df8a7fed92231501224b1
    assert!(
        hash == 0x06abb9af4d18d8b49300457471e4a93dcd485013cb4df8a7fed92231501224b1,
        "Aggregate hash (simple) mismatch with Rust",
    );
}

// ============================================================================
// Aggregate hash (with 2-layer chain)
// ============================================================================

#[test]
fn test_aggregate_hash_with_chain_cross_language() {
    // Chain commitment from 2-layer chain
    let chain_commitment: felt252 = 0x0177f5c4aa4b7c74b43770aa0282670d57619387f4faed70e8b412987046d0e1;

    let hash_inputs: Array<felt252> = array![
        AGGREGATE_DOMAIN,
        1,                  // model_id
        42,                 // model_commitment
        99,                 // io_commitment
        chain_commitment,
        0,                  // tee_report_hash
    ];
    let hash = poseidon_hash_span(hash_inputs.span());

    // Rust value: 0x03c17dffe67711249279c241bbcb2117b59d84e65cf454d28de758ad31b1c9a0
    assert!(
        hash == 0x03c17dffe67711249279c241bbcb2117b59d84e65cf454d28de758ad31b1c9a0,
        "Aggregate hash (chain) mismatch with Rust",
    );
}
