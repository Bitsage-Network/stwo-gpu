#!/bin/bash
# GPU Test Script for Obelysk Stwo Fork
#
# This script tests the GPU backend on a system with NVIDIA GPU and CUDA.
#
# Prerequisites:
#   - NVIDIA GPU (A100, H100, etc.)
#   - CUDA 12.x installed
#   - Rust toolchain with nightly
#
# Usage:
#   ./scripts/test_gpu.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Obelysk GPU Backend Test Script                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo

# Check for CUDA
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  nvcc not found in PATH. NVRTC should still work with CUDA drivers."
else
    nvcc --version | head -4
fi
echo

# Check Rust toolchain
echo "Checking Rust toolchain..."
rustc --version
cargo --version
echo

# Navigate to stwo directory
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo

# Build with GPU features
echo "┌────────────────────────────────────────────────────────────────┐"
echo "│ Building with cuda-runtime feature...                         │"
echo "└────────────────────────────────────────────────────────────────┘"
cargo build --features cuda-runtime,prover --release -p stwo
echo "✅ Build successful!"
echo

# Run unit tests
echo "┌────────────────────────────────────────────────────────────────┐"
echo "│ Running unit tests...                                          │"
echo "└────────────────────────────────────────────────────────────────┘"
cargo test --features cuda-runtime,prover --release -p stwo -- --test-threads=1
echo "✅ Unit tests passed!"
echo

# Run GPU test example
echo "┌────────────────────────────────────────────────────────────────┐"
echo "│ Running GPU FFT test...                                        │"
echo "└────────────────────────────────────────────────────────────────┘"
cargo run --example gpu_test --features cuda-runtime,prover --release
echo

# Run comprehensive GPU test
echo "┌────────────────────────────────────────────────────────────────┐"
echo "│ Running comprehensive GPU test...                              │"
echo "└────────────────────────────────────────────────────────────────┘"
cargo run --example gpu_comprehensive_test --features cuda-runtime,prover --release
echo

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    All GPU Tests Complete!                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

