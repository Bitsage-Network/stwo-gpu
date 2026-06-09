#!/usr/bin/env bash
# benchmark_decode.sh — Run decode-step latency benchmarks and report scaling.
#
# Usage:
#   bash scripts/benchmark_decode.sh [N_STEPS] [LAYERS] [MODEL_DIR]
#
# Examples:
#   bash scripts/benchmark_decode.sh 10
#   bash scripts/benchmark_decode.sh 20 all /path/to/qwen3.5-35b-a3b
#   SKIP_BUILD=1 bash scripts/benchmark_decode.sh 5
set -euo pipefail

N_STEPS="${1:-10}"
LAYERS="${2:-${LAYERS:-all}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKIP_BUILD="${SKIP_BUILD:-0}"

case "${LAYERS}" in
    all|full|0|"")
        LAYERS_LABEL="all"
        LAYER_FLAGS=()
        ;;
    *)
        LAYERS_LABEL="${LAYERS}L"
        LAYER_FLAGS=(--layers "${LAYERS}")
        ;;
esac

# Auto-detect model directory
if [[ -n "${3:-}" ]]; then
    MODEL_DIR="$3"
elif [[ -n "${MODEL_DIR:-}" ]]; then
    : # already set
elif [[ -d "/home/shadeform/.obelyzk/models/qwen3.5-35b-a3b" ]]; then
    MODEL_DIR="/home/shadeform/.obelyzk/models/qwen3.5-35b-a3b"
elif [[ -d "$HOME/.obelyzk/models/qwen3.5-35b-a3b" ]]; then
    MODEL_DIR="$HOME/.obelyzk/models/qwen3.5-35b-a3b"
elif [[ -d "$HOME/models/qwen3.5-35b-a3b" ]]; then
    MODEL_DIR="$HOME/models/qwen3.5-35b-a3b"
elif [[ -d "/home/shadeform/.obelysk/models/qwen3-14b" ]]; then
    MODEL_DIR="/home/shadeform/.obelysk/models/qwen3-14b"
elif [[ -d "$HOME/.obelysk/models/qwen3-14b" ]]; then
    MODEL_DIR="$HOME/.obelysk/models/qwen3-14b"
else
    echo "ERROR: No model directory found. Set MODEL_DIR or pass as 3rd arg."
    exit 1
fi

# Determine features
FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit"
if command -v nvidia-smi &>/dev/null; then
    FEATURES="${FEATURES},cuda-runtime"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'detected')"
else
    GPU_NAME="none (CPU only)"
fi

KV_CACHE_DIR="/tmp/decode_bench_kv_$$"
mkdir -p "${KV_CACHE_DIR}"
OUTPUT="/tmp/decode_bench_${LAYERS_LABEL}_${N_STEPS}s_$(date +%s).json"

echo "=== ZKML Decode-Step Benchmark ==="
echo "  Model     : ${MODEL_DIR}"
echo "  Layers    : ${LAYERS_LABEL}"
echo "  Steps     : ${N_STEPS}"
echo "  GPU       : ${GPU_NAME}"
echo "  KV-cache  : ${KV_CACHE_DIR}"
echo "  Output    : ${OUTPUT}"
if [[ "${LAYERS}" == "1" ]]; then
    echo "  WARNING   : 1-layer decode mode is diagnostic only; production H100 runs should use all layers."
fi
echo ""

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo "Building prove-model (release)..."
    cd "${REPO_ROOT}"
    cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -3
    echo ""
fi

echo "Running decode benchmark (${N_STEPS} steps, ${LAYERS_LABEL} layers)..."
echo ""

cd "${REPO_ROOT}"
time target/release/prove-model \
    --model-dir "${MODEL_DIR}" \
    "${LAYER_FLAGS[@]}" \
    --gpu \
    --format ml_gkr \
    --output "${OUTPUT}" \
    --kv-cache-dir "${KV_CACHE_DIR}" \
    --decode-bench "${N_STEPS}" \
    --profile \
    2>&1 | tee /tmp/decode_bench_latest.log

echo ""
echo "=== Results ==="
BENCH_FILE="${OUTPUT%.json}.decode_bench.json"
if [[ -f "${BENCH_FILE}" ]] && command -v jq &>/dev/null; then
    echo ""
    echo "Prefill:"
    jq '.prefill' "${BENCH_FILE}"
    echo ""
    echo "Scaling analysis:"
    jq '.scaling_analysis' "${BENCH_FILE}"
    echo ""
    echo "Per-step summary:"
    jq -r '.decode_steps[] | "  Step \(.step): \(.elapsed_ms)ms (attn=\(.attention_proofs_ms)ms kv=\(.commitments_kv_cache_ms)ms) cache=\(.cache_len)"' "${BENCH_FILE}"
elif [[ -f "${BENCH_FILE}" ]]; then
    cat "${BENCH_FILE}"
fi

# Cleanup
rm -rf "${KV_CACHE_DIR}"
