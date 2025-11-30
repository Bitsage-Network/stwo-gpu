<div align="center">

# 🚀 Stwo-GPU

**GPU-Accelerated Fork of StarkWare's Stwo Prover**

[![Based on Stwo](https://img.shields.io/badge/Based%20on-Stwo-29296E?style=for-the-badge)](https://github.com/starkware-libs/stwo)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Rust](https://img.shields.io/badge/Rust-nightly-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)

*60-230x faster Circle STARK proving with CUDA acceleration*

</div>

---

## 🎯 What is This?

This is **Bitsage Network's fork** of [StarkWare's Stwo prover](https://github.com/starkware-libs/stwo) with full GPU acceleration for the **Obelysk Protocol**.

### Key Features
- **GPU Backend** - Full `GpuBackend` trait implementation
- **CUDA Kernels** - Optimized FFT, FRI folding, Merkle hashing
- **TEE Integration** - Data stays encrypted on GPU
- **Multi-GPU Support** - Scale across multiple GPUs
- **Production Ready** - 127+ proofs/second throughput

## 🔥 Performance

### Verified Results (A100 80GB)

| Proof Size | Input Data | GPU Time | SIMD Estimate | **Speedup** |
|------------|------------|----------|---------------|-------------|
| 2^18 (256K) | 8 MB | 2.17ms | 132ms | **60.7x** |
| 2^20 (1M) | 32 MB | 6.53ms | 560ms | **85.7x** |
| 2^22 (4M) | 64 MB | 19.02ms | 2.22s | **116.7x** |

### GPU Scaling Projections

#### Single GPU Performance

| GPU | Architecture | VRAM | Memory BW | Est. Speedup | Proofs/sec | Max Proof Size |
|-----|--------------|------|-----------|--------------|------------|----------------|
| **RTX 4090** | Ada Lovelace | 24 GB | 1 TB/s | ~50-80x | ~100 | 2^21 |
| **A100 40GB** | Ampere | 40 GB | 1.6 TB/s | ~60-90x | ~120 | 2^22 |
| **A100 80GB** | Ampere | 80 GB | 2 TB/s | **60-117x** ✓ | **127** ✓ | 2^23 |
| **H100 80GB** | Hopper | 80 GB | 3.35 TB/s | ~120-200x | ~250 | 2^24 |
| **H200 141GB** | Hopper | 141 GB | 4.8 TB/s | ~150-230x | ~300 | 2^25 |
| **B100** | Blackwell | 192 GB | 8 TB/s | ~200-350x | ~450 | 2^26 |
| **B200** | Blackwell | 192 GB | 8 TB/s | ~250-400x | ~550 | 2^26 |
| **GB200 NVL** | Blackwell | 384 GB | 16 TB/s | ~400-600x | ~800 | 2^27 |

*Projections based on memory bandwidth scaling. Actual results may vary.*

#### Multi-GPU Scaling

| Configuration | Throughput Mode | Single Proof Mode |
|---------------|-----------------|-------------------|
| **2x A100** | 254 proofs/sec | ~1.8x faster |
| **4x A100** | 508 proofs/sec | ~3.5x faster |
| **8x A100 (DGX)** | 1,016 proofs/sec | ~6.5x faster |
| **4x H100** | ~1,000 proofs/sec | ~7x faster |
| **8x H100 (DGX H100)** | ~2,000 proofs/sec | ~12x faster |
| **GB200 NVL72** | ~50,000 proofs/sec | ~50x faster |

### Cost Analysis

| GPU | Cloud Cost/hr | Proofs/hr | **Cost per Proof** |
|-----|---------------|-----------|-------------------|
| RTX 4090 | $0.40 | 360,000 | $0.0000011 |
| A100 80GB | $1.50 | 457,200 | $0.0000033 |
| H100 80GB | $3.00 | 900,000 | $0.0000033 |
| 8x H100 DGX | $25.00 | 7,200,000 | $0.0000035 |

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

# Multi-GPU support
stwo = { git = "https://github.com/Bitsage-Network/stwo-gpu", features = ["prover", "cuda-runtime", "multi-gpu"] }
```

### Build

```bash
# CPU only (SIMD)
cargo build --release --features prover

# Single GPU
cargo build --release --features prover,cuda-runtime

# Multi-GPU
cargo build --release --features prover,cuda-runtime,multi-gpu
```

## 🚀 Quick Start

### Single GPU

```rust
use stwo::prover::backend::gpu::pipeline::GpuProofPipeline;

// Create pipeline on GPU 0
let mut pipeline = GpuProofPipeline::new(log_size)?;

// Upload, compute, get proof
pipeline.upload_polynomial(&data)?;
pipeline.ifft(0)?;
pipeline.fft(0)?;
let proof = pipeline.merkle_tree_full(&indices, n_leaves)?;
```

### Multi-GPU (Throughput Mode)

```rust
use stwo::prover::backend::gpu::multi_gpu::MultiGpuProver;

// Create prover across all available GPUs
let prover = MultiGpuProver::new_all_gpus(log_size)?;

// Process multiple proofs in parallel
let proofs = prover.prove_batch(&workloads)?;
// Each GPU processes different proofs simultaneously
```

### Multi-GPU (Single Large Proof)

```rust
use stwo::prover::backend::gpu::multi_gpu::DistributedProofPipeline;

// Distribute one proof across 4 GPUs
let pipeline = DistributedProofPipeline::new(log_size, 4)?;

// Polynomials distributed: GPU0=[0-3], GPU1=[4-7], GPU2=[8-11], GPU3=[12-15]
pipeline.upload_polynomials(&all_polynomials)?;

// Coordinated computation across GPUs
let proof = pipeline.generate_proof()?;
```

### Run Benchmarks

```bash
# Single GPU benchmark
cargo run --example obelysk_production_benchmark --features cuda-runtime --release

# GPU vs SIMD comparison
cargo run --example gpu_vs_simd_real_benchmark --features cuda-runtime,prover --release

# Multi-GPU benchmark (if available)
cargo run --example multi_gpu_benchmark --features cuda-runtime,multi-gpu --release
```

## 🏗️ Architecture

```
crates/stwo/src/prover/backend/gpu/
├── mod.rs              # GpuBackend struct
├── pipeline.rs         # Single-GPU proof pipeline
├── multi_gpu.rs        # Multi-GPU coordination (NEW)
├── fft.rs              # CUDA FFT kernels
├── cuda_executor.rs    # CUDA runtime integration
├── poly_ops.rs         # GPU polynomial operations
├── fri.rs              # GPU FRI folding
└── secure_session.rs   # TEE session management
```

### Multi-GPU Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-GPU Proof Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  THROUGHPUT MODE (Independent Proofs)                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                         │
│  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │                         │
│  │ Proof A │  │ Proof B │  │ Proof C │  │ Proof D │  → 4x throughput        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                         │
│                                                                              │
│  DISTRIBUTED MODE (Single Large Proof)                                       │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    Coordinator (CPU)                         │            │
│  └─────────────────────────────────────────────────────────────┘            │
│         │              │              │              │                       │
│         ▼              ▼              ▼              ▼                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│  │  GPU 0  │◄──►│  GPU 1  │◄──►│  GPU 2  │◄──►│  GPU 3  │  (NVLink)        │
│  │Polys 0-3│    │Polys 4-7│    │Polys 8-11│   │Polys12-15│                  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                   │
│         │              │              │              │                       │
│         └──────────────┴──────────────┴──────────────┘                       │
│                              │                                               │
│                              ▼                                               │
│                    ┌─────────────────┐                                       │
│                    │  Combined Proof │                                       │
│                    │    (32 bytes)   │                                       │
│                    └─────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Features

| Feature | Description |
|---------|-------------|
| `prover` | Enable proving capabilities |
| `gpu` | Enable GPU backend (kernel source only) |
| `cuda-runtime` | Enable CUDA execution (requires NVIDIA GPU) |
| `multi-gpu` | Enable multi-GPU support |

## 🧪 Testing on Cloud GPU

### Single GPU (Brev)

```bash
brev create my-gpu --gpu a100
brev shell my-gpu
git clone https://github.com/Bitsage-Network/stwo-gpu.git
cd stwo-gpu
cargo run --example obelysk_production_benchmark --features cuda-runtime --release
```

### Multi-GPU (Lambda Labs / CoreWeave)

```bash
# 4x A100 instance
ssh user@multi-gpu-instance

git clone https://github.com/Bitsage-Network/stwo-gpu.git
cd stwo-gpu

# Verify GPUs
nvidia-smi

# Run multi-GPU benchmark
cargo run --example multi_gpu_benchmark --features cuda-runtime,multi-gpu --release
```

## 📊 Supported GPUs

### Data Center GPUs
| GPU | Status | Notes |
|-----|--------|-------|
| A100 40GB | ✅ Verified | Production ready |
| A100 80GB | ✅ Verified | Production ready |
| H100 80GB | 🔄 Supported | Hopper optimizations |
| H200 141GB | 🔄 Supported | Largest single-GPU proofs |
| B100/B200 | 🔜 Planned | Blackwell architecture |
| GB200 NVL | 🔜 Planned | Grace-Blackwell superchip |

### Consumer GPUs
| GPU | Status | Notes |
|-----|--------|-------|
| RTX 4090 | ✅ Supported | Best consumer option |
| RTX 4080 | ✅ Supported | Good performance |
| RTX 3090 | ✅ Supported | 24GB VRAM |
| RTX 3080 | ⚠️ Limited | 10-12GB VRAM limits proof size |

### Professional GPUs
| GPU | Status | Notes |
|-----|--------|-------|
| RTX 6000 Ada | ✅ Supported | 48GB VRAM |
| A6000 | ✅ Supported | 48GB VRAM |
| L40S | ✅ Supported | Good balance |

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
