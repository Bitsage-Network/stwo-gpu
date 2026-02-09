/// STWO ML Verifier — On-chain verification of ML inference proofs.
///
/// Verifies:
/// - MatMul sumcheck proofs (Poseidon Fiat-Shamir)
/// - Activation STARK proofs (via stwo-cairo-verifier)
/// - Poseidon commitment chain across layers
/// - TEE attestation freshness
///
/// Architecture:
///   TEE (GPU) → per-layer proofs → recursive aggregation → single on-chain proof

// Re-export pure verification logic from shared core
pub use stwo_ml_verify_core::sumcheck;
pub use stwo_ml_verify_core::layer_chain;

pub mod tee;
pub mod privacy;
pub mod contract;
