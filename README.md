<div align="center">

# 🚀 Stwo-GPU

**GPU-Accelerated Fork of StarkWare's Stwo Prover**

[![Based on Stwo](https://img.shields.io/badge/Based%20on-Stwo-29296E?style=for-the-badge)](https://github.com/starkware-libs/stwo)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Rust](https://img.shields.io/badge/Rust-nightly-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)

*60-117x faster Circle STARK proving with CUDA acceleration*

</div>

---

## 🎯 What is This?

This is **Bitsage Network's fork** of [StarkWare's Stwo prover](https://github.com/starkware-libs/stwo) with full GPU acceleration for the **Obelysk Protocol**.

### Key Features
- **GPU Backend** - Full `GpuBackend` trait implementation
- **CUDA Kernels** - Optimized FFT, FRI folding, Merkle hashing
- **TEE Integration** - Data stays encrypted on GPU
- **Production Ready** - 127 proofs/second throughput

## 🔥 Performance (Verified on A100)

### Obelysk Production Mode
*Data stays on GPU, only 32-byte proof returned*

| Proof Size | Input Data | GPU Time | SIMD Estimate | **Speedup** |
|------------|------------|----------|---------------|-------------|
| 2^18 (256K) | 8 MB | 2.17ms | 132ms | **60.7x** |
| 2^20 (1M) | 32 MB | 6.53ms | 560ms | **85.7x** |
| 2^22 (4M) | 64 MB | 19.02ms | 2.22s | **116.7x** |

### Throughput Metrics

| Metric | Value |
|--------|-------|
| **Proofs/Second** | 127 |
| **Daily Capacity** | ~11 million |
| **Cost per Proof** | $0.000003 (A100) |

## 📦 Installation

### Prerequisites
- Rust nightly
- CUDA Toolkit 12.x
- NVIDIA GPU (Compute Capability 7.0+)

### Add to Cargo.toml

```toml
[dependencies]
# CPU only (SIMD)
stwo = { git = "https://github.com/Bitsage-Network/stwo-gpu", features = ["prover"] }

# With GPU acceleration
stwo = { git = "https://github.com/Bitsage-Network/stwo-gpu", features = ["prover", "cuda-runtime"] }
```

### Build

```bash
# CPU only (SIMD)
cargo build --release --features prover

# With GPU acceleration
cargo build --release --features prover,cuda-runtime
```

## 🚀 Quick Start

### Run Obelysk Production Benchmark

```bash
cargo run --example obelysk_production_benchmark --features cuda-runtime --release
```

### Run GPU vs SIMD Comparison

```bash
cargo run --example gpu_vs_simd_real_benchmark --features cuda-runtime,prover --release
```

### Use GPU Backend in Code

```rust
use stwo::prover::backend::gpu::GpuBackend;
use stwo::prover::backend::gpu::pipeline::GpuProofPipeline;

// Create GPU pipeline
let mut pipeline = GpuProofPipeline::new(log_size)?;

// Upload data (one-time)
pipeline.upload_polynomial(&data)?;

// Compute on GPU (data stays on GPU)
pipeline.ifft(0)?;
pipeline.fft(0)?;

// Get proof (only 32 bytes returned)
let proof = pipeline.merkle_tree_full(&indices, n_leaves)?;
```

## 🏗️ Architecture

```
crates/stwo/src/prover/backend/gpu/
├── mod.rs              # GpuBackend struct
├── pipeline.rs         # GPU proof pipeline (data stays on GPU)
├── fft.rs              # CUDA FFT kernels (optimized)
├── cuda_executor.rs    # CUDA runtime integration
├── poly_ops.rs         # GPU polynomial operations
├── fri.rs              # GPU FRI folding
└── secure_session.rs   # TEE session management
```

### GPU Kernel Optimizations
- **FFT**: Shared memory, vectorized loads, twiddle caching
- **FRI**: uint4 vectorized loads, shared memory alpha broadcast
- **Merkle**: Fully unrolled Blake2s, ping-pong GPU buffers

## 📊 Examples

| Example | Description |
|---------|-------------|
| `obelysk_production_benchmark` | Full production throughput test |
| `gpu_vs_simd_real_benchmark` | Real GPU vs SIMD comparison |
| `gpu_real_stark_proof` | Integration with Stwo proof system |

```bash
# List all examples
ls crates/stwo/examples/
```

## 🔧 Features

| Feature | Description |
|---------|-------------|
| `prover` | Enable proving capabilities |
| `gpu` | Enable GPU backend (kernel source only) |
| `cuda-runtime` | Enable CUDA execution (requires NVIDIA GPU) |

## 🧪 Testing on Cloud GPU

### Using Brev (Recommended)

```bash
# Create A100 instance
brev create my-gpu --gpu a100

# SSH in
brev shell my-gpu

# Clone and test
git clone https://github.com/Bitsage-Network/stwo-gpu.git
cd stwo-gpu
cargo run --example obelysk_production_benchmark --features cuda-runtime --release
```

## 🔄 Changes from Upstream Stwo

### GPU Additions
- Full `GpuBackend` implementing all required traits
- `GpuProofPipeline` for persistent GPU memory
- CUDA kernels for FFT, FRI, Merkle operations
- TEE-aware secure session management

### Stability Fixes
- Replaced `array_chunks` with stable `chunks_exact`
- Compiles on stable Rust (nightly recommended for performance)

### Performance Optimizations
- Twiddle factor caching
- Batch memory transfers
- Vectorized GPU operations
- Shared memory optimizations

## 📄 License

Apache 2.0 - Same as upstream Stwo

## 🔗 Links

- **Upstream Stwo**: [github.com/starkware-libs/stwo](https://github.com/starkware-libs/stwo)
- **Circle STARKs Paper**: [eprint.iacr.org/2024/278](https://eprint.iacr.org/2024/278)
- **BitSage Network**: [github.com/Bitsage-Network](https://github.com/Bitsage-Network)
- **Obelysk Protocol**: [github.com/Bitsage-Network/rust-node](https://github.com/Bitsage-Network/rust-node)

---

<div align="center">

**Built by [BitSage Network](https://github.com/Bitsage-Network) for the Obelysk Protocol**

*Powering verifiable computation with GPU-accelerated ZK proofs*

</div>
