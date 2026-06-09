// pub mod vm31_verifier;   // stripped for lean v32
// pub mod vm31_pool;       // stripped for lean v32
pub mod aggregated_binding;
pub mod channel;
pub mod conversation_statement_stage_verifier;
pub mod field;
pub mod firewall;

// General-purpose STWO verifier (v1.2.2 — uses verify_cairo from stwo_cairo_air)
// Statement-bound STARK-in-STARK verifier. Build with `--features general_stwo_poseidon`.
#[cfg(feature: "general_stwo_poseidon")]
pub mod general_stwo_verifier;
pub mod hades_logup_stage_verifier;
// pub mod gkr;  // stripped for lean v18b (not imported by contract)
// Stripped for lean deploy — not used by GKR verifier contract:
// pub mod ml_air;
// pub mod logup;
pub mod layer_verifiers;
pub mod mle;
pub mod mock_hades_logup_fact_source;
pub mod mock_recursive_statement_fact_source;
pub mod mock_statement_verifier;
pub mod model_verifier;
// Recursive ZKML verifier — production pipeline
pub mod recursive_air;
pub mod recursive_hades_air;
pub mod recursive_statement_stage_verifier;
pub mod recursive_verifier;
pub mod registry;
pub mod statement_fact_registry;
pub mod statement_verification_session;
pub mod sumcheck;
pub mod types;
pub mod verifier;
// pub mod audit;           // stripped for lean v18b
// pub mod access_control;  // stripped for lean v18b
// pub mod view_key;        // stripped for lean v18b
pub mod vm31_merkle;
