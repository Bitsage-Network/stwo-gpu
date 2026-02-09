/// Privacy and access control for ML inference verification.
///
/// On-chain visibility:
/// - Public: "model X verified at timestamp T" (yes/no)
/// - Private: commitment roots, detailed proof data (model_owner only)
///
/// Access control:
/// - model_owner: registers model, can read detailed results
/// - authorized_verifier: can submit proofs for verification
/// - public: can query is_verified(proof_hash) → bool
use starknet::ContractAddress;

/// Access roles for the verifier contract.
#[derive(Drop, Copy, Serde, PartialEq)]
pub enum Role {
    /// Can register models, read detailed results, manage authorized verifiers.
    ModelOwner,
    /// Can submit proofs for verification.
    AuthorizedVerifier,
    /// Can only query public verification status.
    Public,
}

/// Verification record stored on-chain.
/// Minimal data exposure — only commitment roots and boolean result.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct VerificationRecord {
    /// Was the proof valid?
    pub is_valid: bool,
    /// Unix timestamp of verification.
    pub verified_at: u64,
    /// Poseidon commitment of model weights.
    pub model_commitment: felt252,
    /// Poseidon(input || output) — meaningless without original data.
    pub io_commitment: felt252,
    /// TEE report hash (if TEE was used).
    pub tee_report_hash: felt252,
}

/// Check if an address has a given role.
/// In production this would check a role mapping in storage.
pub fn check_role(
    caller: ContractAddress,
    owner: ContractAddress,
    _role: Role,
) -> bool {
    // Owner has all roles
    if caller == owner {
        return true;
    }
    // For now, only owner can do protected operations
    // Phase D will add a full role registry
    false
}
