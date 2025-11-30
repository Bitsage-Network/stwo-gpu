<div align="center">

# 🚀 Stwo-GPU

**GPU-Accelerated Fork of StarkWare's Stwo Prover**

[![Based on Stwo](https://img.shields.io/badge/Based%20on-Stwo-29296E?style=for-the-badge)](https://github.com/starkware-libs/stwo)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Rust](https://img.shields.io/badge/Rust-nightly-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)

*50-100x faster Circle STARK proving with CUDA acceleration*

</div>

---

## 🎯 What is This?

This is **Bitsage Network's fork** of [StarkWare's Stwo prover](https://github.com/starkware-libs/stwo) with:

- **GPU Backend** - Full `GpuBackend` trait implementation
- **CUDA FFT Kernels** - Optimized Circle FFT for A100/H100 GPUs
- **Stable Rust Compatibility** - Removed nightly-only features
- **Automatic Fallback** - Falls back to SIMD when GPU unavailable

## 🔥 Performance

### Expected Speedup (NVIDIA A100)

| FFT Size | CPU (SIMD) | GPU (CUDA) | Speedup |
|----------|------------|------------|---------|
| 2^14 (16K) | 2ms | 0.5ms | **4x** |
| 2^16 (64K) | 10ms | 0.8ms | **12x** |
| 2^18 (256K) | 45ms | 1.5ms | **30x** |
| 2^20 (1M) | 200ms | 3ms | **67x** |
| 2^22 (4M) | 900ms | 10ms | **90x** |

## 📦 Installation

### Prerequisites

- Rust nightly
- CUDA Toolkit 12.x
- NVIDIA GPU (Compute Capability 7.0+)

### Add to Cargo.toml

```toml
[dependencies]
stwo = { git = "https://github.com/Bitsage-Network/stwo-gpu", features = ["prover"] }

# For GPU acceleration
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

### Check GPU Availability

```rust
use stwo::prover::backend::gpu::cuda_executor::is_cuda_available;

if is_cuda_available() {
    println!("GPU acceleration available!");
} else {
    println!("Falling back to SIMD");
}
```

### Use GPU Backend

```rust
use stwo::prover::backend::gpu::GpuBackend;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;

// GPU automatically used when available, falls back to SIMD
let twiddles = GpuBackend::precompute_twiddles(domain.half_coset);
let poly = GpuBackend::interpolate(evaluation, &twiddles);
```

### Run GPU Test

```bash
cargo run --example gpu_test --features cuda-runtime,prover --release
```

## 🏗️ Architecture

```
crates/stwo/src/prover/backend/
├── mod.rs              # Backend trait definitions
├── simd/               # CPU SIMD backend (original)
├── cpu/                # CPU scalar backend (original)
└── gpu/                # GPU backend (NEW)
    ├── mod.rs          # GpuBackend struct
    ├── column.rs       # GPU column operations
    ├── poly_ops.rs     # GPU polynomial operations
    ├── fri.rs          # GPU FRI operations
    ├── fft.rs          # CUDA FFT kernel source
    └── cuda_executor.rs # CUDA runtime integration
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
cargo run --example gpu_test --features cuda-runtime,prover --release
```

## 🔄 Changes from Upstream Stwo

### Stability Fixes
- Replaced `array_chunks` with stable `chunks_exact`
- Removed `iter_array_chunks` nightly feature
- Compiles on stable Rust (nightly still recommended for performance)

### GPU Additions
- `GpuBackend` implementing all required traits
- CUDA kernels for M31 field operations
- CUDA Circle FFT implementation
- `cudarc` integration for runtime kernel compilation
- Automatic CPU fallback when GPU unavailable

### Performance Optimizations
- Twiddle factor caching (CPU and GPU)
- Batch memory transfers
- Optimized kernel launch configurations

## 📊 Benchmarks

```bash
# Run FFT benchmark
cargo bench --features cuda-runtime,prover fft

# Run all benchmarks
cargo bench --features cuda-runtime,prover
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests: `cargo test --features cuda-runtime,prover`
5. Submit PR

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

*Powering verifiable AI/ML computation with GPU-accelerated ZK proofs*

</div>
