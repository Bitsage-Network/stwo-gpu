//! ML-specific AIR components for STWO.
//!
//! Each component defines constraints that verify a specific ML operation
//! (matrix multiplication, activation functions, attention) using STWO's
//! constraint framework with LogUp lookups and sumcheck verification.

pub mod activation;
#[cfg(test)]
mod adversarial_tests;
pub mod attention;
pub mod conv2d;
pub mod dequantize;
pub mod elementwise;
pub mod embedding;
pub mod f32_ops;
pub mod integer_math;
pub mod layernorm;
pub mod matmul;
pub mod poseidon2_air;
pub mod quantize;
pub mod qwen35_delta_recurrence;
pub mod qwen35_depthwise_conv1d;
pub mod qwen35_norm_and_z_gate;
pub mod range_check;
pub mod rmsnorm;
pub mod rope;
#[cfg(test)]
mod tamper_tests;
pub mod tiled_matmul;
pub mod topk;
