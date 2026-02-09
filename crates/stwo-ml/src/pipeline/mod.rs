//! End-to-end ML inference proving pipeline.
//!
//! Chains per-layer proofs (matmul sumcheck + activation STARK) with
//! Poseidon commitment linking, producing a verifiable
//! `ModelPipelineProof` and optional `ComputeReceipt`.
//!
//! # Architecture
//!
//! ```text
//! TEE (H100 GPU)
//! ┌─────────────────────────────────────────────┐
//! │  For each transformer layer:                │
//! │    1. MatMul sumcheck (Poseidon channel)     │
//! │    2. Activation STARK (LogUp)               │
//! │    3. Poseidon commit: layer output          │
//! │  Chain: layer[i].output_commit ==            │
//! │         layer[i+1].input_commit              │
//! └─────────────────┬───────────────────────────┘
//!                   │
//!                   ▼
//! ModelPipelineProof
//! ├── model_commitment   (Poseidon root of weights)
//! ├── io_commitment      (Poseidon of input || output)
//! ├── layer_proofs[]     (matmul + activation per layer)
//! └── receipt            (optional ComputeReceipt)
//! ```

pub mod types;
pub mod prover;
pub mod verifier;
