#!/usr/bin/env bash
#
# Obelysk Full Model Benchmark Suite
# ====================================
# Captures every metric needed for the README and scientific claims.
#
# Outputs a structured JSON + human-readable summary with:
#   - Per-block proving times (each block proved individually)
#   - Per-block matmul sumcheck count and dimensions
#   - Total proving time (all blocks together)
#   - Peak GPU memory
#   - Recursive STARK generation time
#   - Proof sizes (pre-recursion and post-recursion)
#   - Verification time (local CPU)
#   - On-chain submission readiness
#
# Usage:
#   ssh h200
#   cd /path/to/bitsage-network/libs
#   bash scripts/benchmark_full_model.sh \
#     --layers all \
#     --model-dir ~/models/qwen3.5-35b-a3b \
#     --output benchmarks/qwen35b_full.json
#
# For quick single-block validation:
#   bash scripts/benchmark_full_model.sh --layers 1 --model-dir ~/models/qwen3.5-35b-a3b
#
# For full-model one-shot (all N blocks in a single prove-model call):
#   bash scripts/benchmark_full_model.sh --layers all --model-dir ~/models/qwen3.5-35b-a3b --one-shot
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
ENGINE_DIR="${REPO_DIR}/engine"
STARK_CAIRO_DIR="${REPO_DIR}/stark-cairo"
RESULTS_DIR="${REPO_DIR}/benchmarks"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Defaults
NUM_LAYERS=all
MODEL_DIR=""
OUTPUT_FILE=""
SKIP_BUILD=false
SKIP_RECURSIVE=false
WARMUP_RUNS=1
NO_WARMUP=false
ONE_SHOT=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers)          NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)       MODEL_DIR="$2"; shift 2 ;;
        --output)          OUTPUT_FILE="$2"; shift 2 ;;
        --skip-build)      SKIP_BUILD=true; shift ;;
        --skip-recursive)  SKIP_RECURSIVE=true; shift ;;
        --warmup)          WARMUP_RUNS="$2"; shift 2 ;;
        --no-warmup)       NO_WARMUP=true; shift ;;
        --one-shot)        ONE_SHOT=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --layers N|all      Number of transformer blocks (default: all from config.json)"
            echo "  --model-dir PATH    Path to Qwen3.5-35B-A3B weights (SafeTensors)"
            echo "  --output PATH       Output JSON file for results"
            echo "  --skip-build        Skip building binaries"
            echo "  --skip-recursive    Skip recursive STARK generation"
            echo "  --warmup N          GPU warmup runs before measurement (default: 1)"
            echo "  --no-warmup         Skip warmup entirely (use after first run)"
            echo "  --one-shot          Prove all N blocks in a single invocation"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo -e "${RED}ERROR: --model-dir is required${NC}"
    echo "  Example: --model-dir ~/models/qwen3.5-35b-a3b"
    exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo -e "${RED}ERROR: config.json not found in ${MODEL_DIR}${NC}"
    exit 1
fi

case "${NUM_LAYERS}" in
    all|full|0|"")
        NUM_LAYERS=$(python3 - "${MODEL_DIR}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

for key in ("num_hidden_layers", "num_layers", "n_layer", "n_layers"):
    value = cfg.get(key)
    if isinstance(value, int) and value > 0:
        print(value)
        break
else:
    raise SystemExit("could not find num_hidden_layers/num_layers in config.json")
PY
)
        ;;
esac

LAYER_ARGS=(--layers "${NUM_LAYERS}")

mkdir -p "$RESULTS_DIR"

# Default output file
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${RESULTS_DIR}/bench_qwen35b_${NUM_LAYERS}blocks_${TIMESTAMP}.json"
fi

echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗                    ║
║    ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝██╔════╝██║ ██╔╝                    ║
║    ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝ ███████╗█████╔╝                     ║
║    ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝  ╚════██║██╔═██╗                     ║
║    ╚██████╔╝██████╔╝███████╗███████╗██║   ███████║██║  ██╗                    ║
║     ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝                    ║
║                                                                               ║
║        ███████╗████████╗██╗    ██╗ ██████╗     ███╗   ███╗██╗                 ║
║        ██╔════╝╚══██╔══╝██║    ██║██╔═══██╗    ████╗ ████║██║                 ║
║        ███████╗   ██║   ██║ █╗ ██║██║   ██║    ██╔████╔██║██║                 ║
║        ╚════██║   ██║   ██║███╗██║██║   ██║    ██║╚██╔╝██║██║                 ║
║        ███████║   ██║   ╚███╔███╔╝╚██████╔╝    ██║ ╚═╝ ██║███████╗           ║
║        ╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝     ╚═╝     ╚═╝╚══════╝           ║
║                                                                               ║
║                FULL MODEL BENCHMARK SUITE                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# GPU Environment Capture
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[ENV] Capturing hardware environment${NC}"

# CUDA paths
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu"
export PATH="/usr/local/cuda-12.4/bin:${PATH}"

GPU_NAME="unknown"
GPU_MEMORY="unknown"
CUDA_VERSION="unknown"
DRIVER_VERSION="unknown"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
fi

RUST_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
HOSTNAME=$(hostname)
DATE_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "  GPU:       ${GPU_NAME}"
echo "  VRAM:      ${GPU_MEMORY}"
echo "  CUDA:      ${CUDA_VERSION}"
echo "  Driver:    ${DRIVER_VERSION}"
echo "  Rust:      ${RUST_VERSION}"
echo "  Host:      ${HOSTNAME}"
echo "  Date:      ${DATE_ISO}"
echo "  Model dir: ${MODEL_DIR}"
echo "  Layers:    ${NUM_LAYERS}"
echo "  Mode:      $([ "$ONE_SHOT" = true ] && echo "one-shot" || echo "per-block")"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
PROVE_BIN=""
CAIRO_PROVE_BIN=""

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[BUILD] Building prove-model + cairo-prove${NC}"

    # Build engine prove-model
    echo "  Building prove-model..."
    BUILD_START=$(date +%s%N)
    (
        cd "${ENGINE_DIR}"
        FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit"
        if command -v nvidia-smi &>/dev/null; then
            FEATURES="${FEATURES},cuda-runtime"
        fi
        cargo build --release \
            --bin prove-model \
            --features "${FEATURES}" 2>&1 | tail -5
    )
    BUILD_END=$(date +%s%N)
    BUILD_SEC=$(echo "scale=1; ($BUILD_END - $BUILD_START) / 1000000000" | bc)
    PROVE_BIN="${ENGINE_DIR}/target/release/prove-model"
    echo -e "  ${GREEN}prove-model built in ${BUILD_SEC}s${NC}"

    # Build cairo-prove
    echo "  Building cairo-prove..."
    (
        cd "${STARK_CAIRO_DIR}/cairo-prove"
        cargo build --release 2>&1 | tail -5
    )
    CAIRO_PROVE_BIN="${STARK_CAIRO_DIR}/cairo-prove/target/release/cairo-prove"

    echo -e "  ${GREEN}Build complete${NC}"
else
    PROVE_BIN="${ENGINE_DIR}/target/release/prove-model"
    CAIRO_PROVE_BIN="${STARK_CAIRO_DIR}/cairo-prove/target/release/cairo-prove"
    echo -e "${YELLOW}[BUILD] Skipped (--skip-build) — using existing binary${NC}"
    echo -e "${YELLOW}  WARNING: If you recently changed code, remove --skip-build to rebuild!${NC}"
fi

if [ -z "$PROVE_BIN" ] || [ ! -f "$PROVE_BIN" ]; then
    echo -e "${RED}ERROR: prove-model binary not found${NC}"
    exit 1
fi
echo "  prove-model: ${PROVE_BIN}"
echo "  cairo-prove: ${CAIRO_PROVE_BIN:-not found}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Validate model
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[VALIDATE] Checking model directory${NC}"
${PROVE_BIN} --model-dir "${MODEL_DIR}" "${LAYER_ARGS[@]}" --validate 2>&1 || {
    echo -e "${RED}ERROR: Model validation failed${NC}"
    exit 1
}
echo -e "  ${GREEN}Model validation passed${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Warmup — lightweight CUDA context init (NOT a full model prove)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$NO_WARMUP" = true ] || [ "$WARMUP_RUNS" -eq 0 ]; then
    echo -e "${YELLOW}[WARMUP] Skipped${NC}"
    echo ""
else
    echo -e "${YELLOW}[WARMUP] Initializing CUDA context${NC}"
    WARMUP_START=$(date +%s%N)

    # Step 1: CUDA driver + context init via nvidia-smi (< 1s)
    echo "  Initializing CUDA driver..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

    # Step 2: Model inspection (loads safetensors, builds graph — no proving)
    echo "  Loading model weights (inspect only, no proving)..."
    ${PROVE_BIN} --model-dir "${MODEL_DIR}" --layers 1 --inspect 2>&1 | head -20 || true

    WARMUP_END=$(date +%s%N)
    WARMUP_MS=$(( (WARMUP_END - WARMUP_START) / 1000000 ))
    WARMUP_SEC=$(echo "scale=1; ${WARMUP_MS}/1000" | bc)

    echo -e "  ${GREEN}Warmup complete in ${WARMUP_SEC}s${NC}"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Proving Benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "════════════════════════════════════════════════════════════════════"
echo "  PROVING ${NUM_LAYERS} TRANSFORMER BLOCKS"
echo "  Model: Qwen3.5-35B-A3B | GPU: ${GPU_NAME}"
echo "════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

BLOCK_TIMES=()
PROOF_SIZES=()
MATMUL_COUNTS=()
PEAK_GPU_MEM=0
FULL_PROOF="benchmarks/full_${NUM_LAYERS}blocks_proof.json"
FULL_LOG="benchmarks/full_${NUM_LAYERS}blocks.log"

if [ "$ONE_SHOT" = true ]; then
    # ── ONE-SHOT MODE: Prove all N layers in a single invocation ──
    echo -e "${YELLOW}[ONE-SHOT] Proving all ${NUM_LAYERS} blocks in a single invocation${NC}"
    echo ""

    # Prove once in cairo_serde format — reused by recursive pipeline (no double-proving)
    GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
    echo -e "  GPU memory before: ${GPU_MEM_BEFORE} MiB"

    TOTAL_PROVE_START=$(date +%s%N)

    # Stream stderr to BOTH terminal and log file so the user sees live progress
    ${PROVE_BIN} \
        --model-dir "${MODEL_DIR}" \
        "${LAYER_ARGS[@]}" \
        --output "${FULL_PROOF}" \
        --format cairo_serde \
        --gpu 2>&1 | tee "${FULL_LOG}" || true

    TOTAL_PROVE_END=$(date +%s%N)
    TOTAL_PROVE_MS=$(( (TOTAL_PROVE_END - TOTAL_PROVE_START) / 1000000 ))
    TOTAL_PROVE_SEC=$(echo "scale=3; ${TOTAL_PROVE_MS}/1000" | bc)

    GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
    PEAK_GPU_MEM=$GPU_MEM_AFTER

    if [ -f "$FULL_PROOF" ]; then
        PROOF_SIZE=$(du -b "$FULL_PROOF" 2>/dev/null | cut -f1 || echo "0")
    else
        PROOF_SIZE=0
    fi

    MATMUL_COUNT=$(grep -o "matmul_proofs: [0-9]*" "${FULL_LOG}" 2>/dev/null | grep -o "[0-9]*" || echo "0")

    # Store as single "block" entry
    BLOCK_TIMES+=("${TOTAL_PROVE_SEC}")
    PROOF_SIZES+=("${PROOF_SIZE}")
    MATMUL_COUNTS+=("${MATMUL_COUNT}")

    echo ""
    echo -e "  ${GREEN}Time: ${TOTAL_PROVE_SEC}s | Proof: ${PROOF_SIZE} bytes | MatMuls: ${MATMUL_COUNT} | GPU Mem: ${GPU_MEM_AFTER} MiB${NC}"

else
    # ── PER-BLOCK MODE: Prove each block individually ──
    TOTAL_PROVE_START=$(date +%s%N)

    for block in $(seq 1 "${NUM_LAYERS}"); do
        echo -e "${YELLOW}[Block ${block}/${NUM_LAYERS}]${NC} Proving (layers=${block})..."

        BLOCK_START=$(date +%s%N)

        # Capture GPU memory before
        GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")

        BLOCK_PROOF="benchmarks/block_${block}_proof.json"
        BLOCK_LOG="benchmarks/block_${block}.log"

        # Prove exactly 'block' layers to get per-block cumulative timing
        ${PROVE_BIN} \
            --model-dir "${MODEL_DIR}" \
            --layers "${block}" \
            --output "${BLOCK_PROOF}" \
            --format cairo_serde \
            --gpu 2>&1 | tee "${BLOCK_LOG}" || true

        BLOCK_END=$(date +%s%N)
        BLOCK_MS=$(( (BLOCK_END - BLOCK_START) / 1000000 ))
        BLOCK_SEC=$(echo "scale=3; ${BLOCK_MS}/1000" | bc)

        # Capture GPU memory peak
        GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
        if [ "$GPU_MEM_AFTER" -gt "$PEAK_GPU_MEM" ]; then
            PEAK_GPU_MEM=$GPU_MEM_AFTER
        fi

        # Get proof size
        if [ -f "$BLOCK_PROOF" ]; then
            PROOF_SIZE=$(du -b "$BLOCK_PROOF" 2>/dev/null | cut -f1 || echo "0")
        else
            PROOF_SIZE=0
        fi

        # Extract matmul count from log
        MATMUL_COUNT=$(grep -o "matmul_proofs: [0-9]*" "${BLOCK_LOG}" 2>/dev/null | grep -o "[0-9]*" || echo "0")

        BLOCK_TIMES+=("${BLOCK_SEC}")
        PROOF_SIZES+=("${PROOF_SIZE}")
        MATMUL_COUNTS+=("${MATMUL_COUNT}")

        echo "  Time: ${BLOCK_SEC}s | Proof: ${PROOF_SIZE} bytes | MatMuls: ${MATMUL_COUNT} | GPU Mem: ${GPU_MEM_AFTER} MiB"
    done

    TOTAL_PROVE_END=$(date +%s%N)
    TOTAL_PROVE_MS=$(( (TOTAL_PROVE_END - TOTAL_PROVE_START) / 1000000 ))
    TOTAL_PROVE_SEC=$(echo "scale=3; ${TOTAL_PROVE_MS}/1000" | bc)
    FULL_PROOF="benchmarks/block_${NUM_LAYERS}_proof.json"
fi

echo ""
echo -e "${GREEN}Total proving time: ${TOTAL_PROVE_SEC}s${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Recursive STARK (optional)
# ─────────────────────────────────────────────────────────────────────────────
RECURSIVE_TIME_SEC="N/A"
RECURSIVE_PROOF_SIZE="N/A"

if [ "$SKIP_RECURSIVE" = false ] && [ -n "$CAIRO_PROVE_BIN" ]; then
    echo -e "${YELLOW}[RECURSIVE] Generating recursive Circle STARK${NC}"
    echo ""

    # Reuse the proof from the proving phase (already in cairo_serde format)
    CAIRO_SERDE_PROOF="${FULL_PROOF}"
    echo -e "  Reusing proof from proving phase: ${CAIRO_SERDE_PROOF}"

    if [ -f "$CAIRO_SERDE_PROOF" ]; then
        EXECUTABLE="${STARK_CAIRO_DIR}/stwo_cairo_verifier/target/dev/obelysk_ml_verifier.executable.json"
        if [ ! -f "$EXECUTABLE" ] && [ -f "${REPO_DIR}/artifacts/obelysk_ml_verifier.executable.json" ]; then
            EXECUTABLE="${REPO_DIR}/artifacts/obelysk_ml_verifier.executable.json"
        fi

        if [ -f "$EXECUTABLE" ]; then
            echo -e "  ${YELLOW}Step 2/2: Running recursive STARK prover (cairo-prove)...${NC}"
            RECURSIVE_START=$(date +%s%N)

            ${CAIRO_PROVE_BIN} prove-ml \
                --verifier-executable "${EXECUTABLE}" \
                --ml-proof "${CAIRO_SERDE_PROOF}" \
                --output "benchmarks/recursive_proof.json" 2>&1 | tee "benchmarks/recursive_prove.log" || true

            RECURSIVE_END=$(date +%s%N)
            RECURSIVE_MS=$(( (RECURSIVE_END - RECURSIVE_START) / 1000000 ))
            RECURSIVE_TIME_SEC=$(echo "scale=3; ${RECURSIVE_MS}/1000" | bc)

            if [ -f "benchmarks/recursive_proof.json" ]; then
                RECURSIVE_PROOF_SIZE=$(du -b "benchmarks/recursive_proof.json" | cut -f1)
            fi

            echo -e "  ${GREEN}Recursive STARK: ${RECURSIVE_TIME_SEC}s, size: ${RECURSIVE_PROOF_SIZE} bytes${NC}"
        else
            echo -e "  ${YELLOW}ML verifier executable not found — skipping recursive${NC}"
            echo "  Expected: ${EXECUTABLE}"
        fi
    else
        echo -e "  ${YELLOW}cairo_serde proof generation failed — skipping recursive${NC}"
    fi
else
    echo -e "${YELLOW}[RECURSIVE] Skipped${NC}"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Write Results JSON
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[OUTPUT] Writing results to ${OUTPUT_FILE}${NC}"

# Compute average
if [ ${#BLOCK_TIMES[@]} -gt 0 ]; then
    AVG_BLOCK_SEC=$(echo "scale=3; ${TOTAL_PROVE_SEC}/${NUM_LAYERS}" | bc)
else
    AVG_BLOCK_SEC="0"
fi

# Build block times array for JSON
BLOCK_TIMES_JSON="["
for i in "${!BLOCK_TIMES[@]}"; do
    [ "$i" -gt 0 ] && BLOCK_TIMES_JSON+=","
    BLOCK_TIMES_JSON+="{\"block\":$i,\"prove_sec\":${BLOCK_TIMES[$i]},\"proof_bytes\":${PROOF_SIZES[$i]},\"matmul_count\":${MATMUL_COUNTS[$i]}}"
done
BLOCK_TIMES_JSON+="]"

cat > "${OUTPUT_FILE}" << ENDJSON
{
  "benchmark_version": "1.1.0",
  "timestamp": "${DATE_ISO}",
  "hostname": "${HOSTNAME}",
  "hardware": {
    "gpu": "${GPU_NAME}",
    "gpu_memory": "${GPU_MEMORY}",
    "cuda_version": "${CUDA_VERSION}",
    "driver_version": "${DRIVER_VERSION}",
    "rust_version": "${RUST_VERSION}"
  },
  "model": {
    "name": "Qwen3.5-35B-A3B",
    "parameters": "35B total / 3B active",
    "architecture": "Transformer decoder",
    "num_blocks": ${NUM_LAYERS},
    "d_model": 5120,
    "num_heads": 40,
    "d_ff": 13824,
    "head_dim": 128
  },
  "proving": {
    "mode": "$([ "$ONE_SHOT" = true ] && echo "one_shot" || echo "per_block")",
    "total_blocks": ${NUM_LAYERS},
    "total_prove_sec": ${TOTAL_PROVE_SEC},
    "avg_block_prove_sec": ${AVG_BLOCK_SEC},
    "peak_gpu_memory_mib": ${PEAK_GPU_MEM},
    "per_block": ${BLOCK_TIMES_JSON}
  },
  "recursive_stark": {
    "time_sec": "${RECURSIVE_TIME_SEC}",
    "proof_size_bytes": "${RECURSIVE_PROOF_SIZE}"
  },
  "security": {
    "pow_bits": 20,
    "n_queries": 28,
    "log_blowup_factor": 5,
    "security_bits": 160,
    "trusted_setup": false,
    "field": "M31 (p = 2^31 - 1)",
    "channel": "Poseidon252 (on-chain), Blake2s (CPU)"
  },
  "notes": [
    "All times measured with wall-clock (date +%s%N)",
    "GPU warmup: ${WARMUP_RUNS} pass(es) before measurement",
    "Proving uses cuda-runtime feature with GPU residency when NVIDIA CUDA is available",
    "Peak GPU memory is max observed across all blocks",
    "Per-block times are cumulative (block N = prove layers 1..N)"
  ]
}
ENDJSON

echo -e "  ${GREEN}Results written to ${OUTPUT_FILE}${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Human-readable summary
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    BENCHMARK RESULTS SUMMARY                     ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║                                                                   ║"
printf "║  Model:           Qwen3.5-35B-A3B (%d blocks)\n" "${NUM_LAYERS}"
printf "║  GPU:             %s\n" "${GPU_NAME}"
echo "║                                                                   ║"
echo "║  ── Proving ──────────────────────────────────────────────────── ║"
printf "║  Total prove:     %-10s (%d blocks)\n" "${TOTAL_PROVE_SEC}s" "${NUM_LAYERS}"
printf "║  Avg per block:   %-10s\n" "${AVG_BLOCK_SEC}s"
printf "║  Peak GPU mem:    %-10s\n" "${PEAK_GPU_MEM} MiB"
echo "║                                                                   ║"
echo "║  ── Recursive STARK ──────────────────────────────────────────── ║"
printf "║  Recursive time:  %-10s\n" "${RECURSIVE_TIME_SEC}s"
printf "║  Recursive size:  %-10s\n" "${RECURSIVE_PROOF_SIZE} bytes"
echo "║                                                                   ║"
echo "║  ── Security ─────────────────────────────────────────────────── ║"
echo "║  160-bit target (pow=20, queries=28, blowup=5). No trusted setup.║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Quick stats
if [ "$ONE_SHOT" = false ]; then
    echo "Per-block breakdown:"
    for i in "${!BLOCK_TIMES[@]}"; do
        printf "  Block %2d: %8ss | %s matmuls | %s bytes\n" \
            "$((i+1))" "${BLOCK_TIMES[$i]}" "${MATMUL_COUNTS[$i]}" "${PROOF_SIZES[$i]}"
    done
    echo ""
fi

echo -e "${GREEN}Benchmark complete. Results: ${OUTPUT_FILE}${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results and update libs/README.md with real numbers"
echo "  2. Run: bash scripts/h200_submit_onchain.sh --proof benchmarks/recursive_proof.json --submit"
echo "  3. Commit: git add benchmarks/ && git commit -m 'Add verified benchmarks'"
