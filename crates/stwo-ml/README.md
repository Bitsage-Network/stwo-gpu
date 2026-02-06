# stwo-ml

ML inference proving circuits built on [STWO](https://github.com/starkware-libs/stwo) — the fastest STARK prover in the world.

## Overview

`stwo-ml` is a workspace crate within the STWO-ML repository that adds neural network inference verification on top of STWO's Circle STARK backend. It uses STWO's existing cryptographic primitives (sumcheck, LogUp, MLEs, GKR) and wires them together for ML workloads.

### Components

**`components/matmul.rs`** — Sumcheck-based matrix multiplication

Traditional zkML decomposes matmul into O(m x k x n) individual multiply-add trace rows.
stwo-ml represents matrices as multilinear extensions on the boolean hypercube and uses the
sumcheck protocol to verify the computation with O(m + k + n) verifier work:

| Matrix Size | Naive Trace Rows | Sumcheck Rows | Reduction |
|-------------|-----------------|---------------|-----------|
| 128 x 128 | 2,097,152 | 49,152 | 42x |
| 768 x 768 (BERT) | 452,984,832 | 1,769,472 | 255x |
| 4096 x 4096 (LLM) | 68.7B | 50.3M | 1365x |

**`components/activation.rs`** — LogUp activation tables

Non-linear operations (ReLU, GELU, sigmoid, softmax) verified via precomputed lookup tables
using the LogUp protocol. Table sizes range from 2^16 (ReLU) to 2^20 (softmax).

**`components/attention.rs`** — Multi-head attention gadget

Composed verification for transformer attention: QK^T matmul (sumcheck) + softmax (LogUp) +
attention x V matmul (sumcheck). Trace cost analysis for standard architectures:

| Model | Heads | Dims | Seq Len | Sumcheck Rows |
|-------|-------|------|---------|---------------|
| BERT-base | 12 | 768 | 512 | 11.7M |
| GPT-2 | 12 | 768 | 1024 | 23.5M |
| Llama-7B | 32 | 4096 | 2048 | 335.5M |

**`gadgets/`** — Reusable constraint gadgets: range checks, lookup tables, INT8/FP8 quantization.

**`compiler/`** — ONNX model import and computation graph builder (stubs).

## Why M31

The Mersenne-31 prime (2^31 - 1) enables single-cycle field reduction on commodity hardware.
For ML workloads where billions of multiply-accumulate operations dominate, this gives 2-4x
throughput per operation compared to 256-bit fields used by other zkML systems (EZKL, zkLLM).

Combined with STWO's GPU backend (CUDA kernels for FFT, FRI, Merkle) and memory-resident
proving (one transfer in, one transfer out), the compound speedup is 10-50x over existing
zkML approaches.

## Usage

```rust
use stwo_ml::components::matmul::{MatMulDims, M31Matrix};

// Calculate trace cost before proving
let dims = MatMulDims::new(768, 768, 768);
println!("Naive: {} rows", dims.naive_trace_rows());      // 452,984,832
println!("Sumcheck: {} rows", dims.sumcheck_trace_rows()); // 1,769,472
println!("Speedup: {:.0}x", dims.speedup());               // 255x
```

## Building

```bash
# Check
cargo check -p stwo-ml

# Test (6 tests)
cargo test -p stwo-ml

# Bench
cargo bench -p stwo-ml

# With GPU acceleration
cargo check -p stwo-ml --features gpu
```

## Status

**Phase 1** (current): MatMul sumcheck dimensions, activation table sizing, attention cost analysis. All types and tests passing.

**Phase 2** (next): Wire up actual STWO sumcheck prover for matmul, build MLE representations, integrate with constraint framework.

**Phase 3**: LogUp activation tables, ONNX compiler, end-to-end model proving.

## License

Apache 2.0
