/// STWO ML Verify Core — Pure verification logic shared between
/// the on-chain verifier contract and the recursive executable.
///
/// Contains:
/// - Sumcheck verification (Poseidon Fiat-Shamir)
/// - Layer commitment chain verification
///
/// Zero starknet dependencies — uses only core::poseidon.
pub mod sumcheck;
pub mod layer_chain;
