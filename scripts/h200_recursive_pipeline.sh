#!/usr/bin/env bash
#
# H200 Recursive ML Proof Pipeline
# ==================================
# Full pipeline: GPU proof gen -> Cairo recursive verification -> Circle STARK proof
#
# Pipeline (two-step):
#   Step A: prove-model (GPU, Rust)
#     -> N matmul sumcheck proofs (Qwen3-14B blocks)
#     -> serialize as felt252 hex array (cairo_serde format)
#     -> ml_proof.json
#
#   Step B: cairo-prove prove-ml
#     -> Execute ML verifier in Cairo VM
#     -> Generate Circle STARK proof of that execution
#     -> recursive_proof.json (~17 MB -> ~1 KB on-chain)
#
# Usage:
#   ssh h200
#   cd /path/to/bitsage-network/libs
#   bash scripts/h200_recursive_pipeline.sh --layers 1 --model-dir ~/models/qwen3-14b
#   bash scripts/h200_recursive_pipeline.sh --layers 40 --model-dir ~/models/qwen3-14b
#   bash scripts/h200_recursive_pipeline.sh --skip-build --layers 1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Defaults
SKIP_BUILD=false
NUM_LAYERS=1
MODEL_DIR=""
ML_PROOF_OUTPUT="ml_proof.json"
PROOF_OUTPUT="recursive_proof.json"
MODEL_ID="0x1"
USE_GPU=true

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)  SKIP_BUILD=true; shift ;;
        --layers)      NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --output)      PROOF_OUTPUT="$2"; shift 2 ;;
        --ml-output)   ML_PROOF_OUTPUT="$2"; shift 2 ;;
        --model-id)    MODEL_ID="$2"; shift 2 ;;
        --no-gpu)      USE_GPU=false; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --layers N       Number of transformer blocks (default: 1)"
            echo "  --model-dir PATH Path to HuggingFace model directory (SafeTensors)"
            echo "  --output PATH    Output recursive proof file (default: recursive_proof.json)"
            echo "  --ml-output PATH Output ML proof file (default: ml_proof.json)"
            echo "  --model-id ID    Model ID for on-chain claim (default: 0x1)"
            echo "  --skip-build     Skip building binaries"
            echo "  --no-gpu         Disable GPU acceleration"
            exit 0
            ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Obelysk Protocol — H200 Recursive STARK Pipeline   ║"
echo "║  Qwen3-14B -> Circle STARK Proof                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Layers:      ${NUM_LAYERS}"
echo "  Model dir:   ${MODEL_DIR:-<not set>}"
echo "  ML proof:    ${ML_PROOF_OUTPUT}"
echo "  Recursive:   ${PROOF_OUTPUT}"
echo "  GPU:         ${USE_GPU}"
echo ""

# ----------------------------------------------------------------
# Step 0: Environment checks
# ----------------------------------------------------------------
echo -e "${YELLOW}[Step 0] Environment${NC}"

# CUDA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu"
export PATH="/usr/local/cuda-12.4/bin:${PATH}"

if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  CUDA:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "  Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
    echo -e "  ${YELLOW}WARNING: nvidia-smi not found. GPU features will be disabled.${NC}"
    USE_GPU=false
fi

# Check Rust
if ! command -v rustup &>/dev/null; then
    echo -e "${RED}ERROR: rustup not found. Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
    exit 1
fi
echo "  Rust:   $(rustc --version 2>/dev/null || echo 'not found')"

# Check Scarb (for Cairo executable compilation)
if command -v scarb &>/dev/null; then
    echo "  Scarb:  $(scarb --version 2>/dev/null)"
else
    echo -e "  ${YELLOW}Scarb not found — using pre-compiled executable if available${NC}"
fi

echo ""

# ----------------------------------------------------------------
# Step 1: Build prove-model (stwo-ml)
# ----------------------------------------------------------------
PROVE_MODEL_BIN=""

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[Step 1a] Building prove-model (stwo-ml + GPU)${NC}"
    echo "  Directory: ${REPO_DIR}/stwo-ml"

    FEATURES="cli"
    if [ "$USE_GPU" = true ] && command -v nvidia-smi &>/dev/null; then
        FEATURES="cli,cuda-runtime"
        echo "  Features: ${FEATURES} (GPU enabled)"
    else
        echo "  Features: ${FEATURES} (CPU only)"
    fi

    (
        cd "${REPO_DIR}/stwo-ml"
        cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -5
    )

    PROVE_MODEL_BIN=$(find "${REPO_DIR}/stwo-ml" -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)

    if [ -n "$PROVE_MODEL_BIN" ]; then
        echo -e "  ${GREEN}prove-model built successfully${NC}"
        echo "  Binary: ${PROVE_MODEL_BIN}"
        echo "  Size: $(du -h "$PROVE_MODEL_BIN" | cut -f1)"
    else
        echo -e "${RED}  ERROR: prove-model binary not found after build${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Step 1a] Skipping prove-model build (--skip-build)${NC}"
    PROVE_MODEL_BIN=$(find "${REPO_DIR}/stwo-ml" -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
    if [ -z "$PROVE_MODEL_BIN" ]; then
        # Also check workspace root
        PROVE_MODEL_BIN=$(find "${REPO_DIR}" -maxdepth 3 -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
    fi
    if [ -z "$PROVE_MODEL_BIN" ]; then
        echo -e "${RED}  ERROR: prove-model not found. Run without --skip-build.${NC}"
        exit 1
    fi
    echo "  Using existing: ${PROVE_MODEL_BIN}"
fi
echo ""

# ----------------------------------------------------------------
# Step 1b: Build cairo-prove
# ----------------------------------------------------------------
CAIRO_PROVE_BIN="${REPO_DIR}/stwo-cairo/cairo-prove/target/release/cairo-prove"

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[Step 1b] Building cairo-prove${NC}"
    echo "  Directory: ${REPO_DIR}/stwo-cairo/cairo-prove"

    (
        cd "${REPO_DIR}/stwo-cairo/cairo-prove"
        cargo build --release 2>&1 | tail -5
    ) || {
        echo -e "${YELLOW}  Build failed — trying with RUSTUP_TOOLCHAIN=nightly${NC}"
        (
            cd "${REPO_DIR}/stwo-cairo/cairo-prove"
            RUSTUP_TOOLCHAIN=nightly cargo build --release 2>&1 | tail -5
        )
    }

    if [ -f "$CAIRO_PROVE_BIN" ]; then
        echo -e "  ${GREEN}cairo-prove built successfully${NC}"
        echo "  Binary: ${CAIRO_PROVE_BIN}"
        echo "  Size: $(du -h "$CAIRO_PROVE_BIN" | cut -f1)"
    else
        echo -e "${RED}  ERROR: cairo-prove binary not found after build${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Step 1b] Skipping cairo-prove build (--skip-build)${NC}"
    if [ ! -f "$CAIRO_PROVE_BIN" ]; then
        echo -e "${RED}  ERROR: cairo-prove not found at ${CAIRO_PROVE_BIN}${NC}"
        echo "  Run without --skip-build to build it"
        exit 1
    fi
    echo "  Using existing: ${CAIRO_PROVE_BIN}"
fi
echo ""

# ----------------------------------------------------------------
# Step 2: Check/Build Cairo ML verifier executable
# ----------------------------------------------------------------
EXECUTABLE="${REPO_DIR}/stwo-cairo/stwo_cairo_verifier/target/dev/obelysk_ml_verifier.executable.json"
EXECUTABLE_ARTIFACT="${REPO_DIR}/artifacts/obelysk_ml_verifier.executable.json"

echo -e "${YELLOW}[Step 2] Cairo ML verifier executable${NC}"
if [ -f "$EXECUTABLE" ]; then
    echo "  Found: ${EXECUTABLE}"
    echo "  Size: $(du -h "$EXECUTABLE" | cut -f1)"
elif [ -f "$EXECUTABLE_ARTIFACT" ]; then
    echo "  Found pre-built artifact: ${EXECUTABLE_ARTIFACT}"
    echo "  Size: $(du -h "$EXECUTABLE_ARTIFACT" | cut -f1)"
    EXECUTABLE="$EXECUTABLE_ARTIFACT"
else
    echo "  Not found. Building with scarb..."
    if command -v scarb &>/dev/null; then
        (
            cd "${REPO_DIR}/stwo-cairo/stwo_cairo_verifier"
            scarb build --package obelysk_ml_verifier 2>&1 | tail -3
        )
        if [ -f "$EXECUTABLE" ]; then
            echo -e "  ${GREEN}Built successfully${NC}"
            echo "  Size: $(du -h "$EXECUTABLE" | cut -f1)"
        else
            echo -e "${RED}  ERROR: Executable not found after build${NC}"
            exit 1
        fi
    else
        echo -e "${RED}  ERROR: scarb not available and no pre-built artifact found${NC}"
        echo "  Either install scarb or copy the artifact to: ${EXECUTABLE_ARTIFACT}"
        exit 1
    fi
fi
echo ""

# ----------------------------------------------------------------
# Step 3: Validate model directory (if provided)
# ----------------------------------------------------------------
if [ -n "$MODEL_DIR" ]; then
    echo -e "${YELLOW}[Step 3] Validating model directory${NC}"
    echo "  Model dir: ${MODEL_DIR}"

    if [ ! -d "$MODEL_DIR" ]; then
        echo -e "${RED}  ERROR: Model directory not found: ${MODEL_DIR}${NC}"
        exit 1
    fi

    if [ ! -f "${MODEL_DIR}/config.json" ]; then
        echo -e "${RED}  ERROR: config.json not found in ${MODEL_DIR}${NC}"
        exit 1
    fi

    # Quick validation via prove-model --validate
    ${PROVE_MODEL_BIN} --model-dir "${MODEL_DIR}" --layers "${NUM_LAYERS}" --validate 2>&1 || {
        echo -e "${RED}  ERROR: Model validation failed${NC}"
        exit 1
    }
    echo -e "  ${GREEN}Model validation passed${NC}"
else
    echo -e "${YELLOW}[Step 3] No --model-dir specified. prove-model will need --model (ONNX mode).${NC}"
fi
echo ""

# ----------------------------------------------------------------
# Step 4: GPU ML Proof Generation (prove-model)
# ----------------------------------------------------------------
echo -e "${CYAN}${BOLD}"
echo "════════════════════════════════════════════════════════"
echo "  PHASE 1: GPU ML PROOF GENERATION"
echo "  Layers: ${NUM_LAYERS} | GPU: ${USE_GPU}"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

PROVE_CMD="${PROVE_MODEL_BIN}"
PROVE_CMD+=" --output ${ML_PROOF_OUTPUT}"
PROVE_CMD+=" --format cairo_serde"
PROVE_CMD+=" --model-id ${MODEL_ID}"

if [ -n "$MODEL_DIR" ]; then
    PROVE_CMD+=" --model-dir ${MODEL_DIR}"
    PROVE_CMD+=" --layers ${NUM_LAYERS}"
fi

if [ "$USE_GPU" = true ]; then
    PROVE_CMD+=" --gpu"
fi

echo "  Command: ${PROVE_CMD}"
echo ""

PROVE_START=$(date +%s%N)
eval ${PROVE_CMD}
PROVE_END=$(date +%s%N)
PROVE_MS=$(( (PROVE_END - PROVE_START) / 1000000 ))
PROVE_SEC=$(echo "scale=3; ${PROVE_MS}/1000" | bc)

if [ -f "$ML_PROOF_OUTPUT" ]; then
    echo ""
    echo -e "  ${GREEN}ML proof generated in ${PROVE_SEC}s${NC}"
    echo "  Output: ${ML_PROOF_OUTPUT} ($(du -h "${ML_PROOF_OUTPUT}" | cut -f1))"
else
    echo -e "${RED}  ERROR: ML proof not generated${NC}"
    exit 1
fi
echo ""

# ----------------------------------------------------------------
# Step 5: Recursive Circle STARK (cairo-prove prove-ml)
# ----------------------------------------------------------------
echo -e "${CYAN}${BOLD}"
echo "════════════════════════════════════════════════════════"
echo "  PHASE 2: RECURSIVE CIRCLE STARK"
echo "  ML proof -> Cairo VM -> Circle STARK"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

RECURSIVE_CMD="${CAIRO_PROVE_BIN} prove-ml"
RECURSIVE_CMD+=" --verifier-executable ${EXECUTABLE}"
RECURSIVE_CMD+=" --ml-proof ${ML_PROOF_OUTPUT}"
RECURSIVE_CMD+=" --output ${PROOF_OUTPUT}"

echo "  Command: ${RECURSIVE_CMD}"
echo ""

RECURSIVE_START=$(date +%s%N)
eval ${RECURSIVE_CMD}
RECURSIVE_END=$(date +%s%N)
RECURSIVE_MS=$(( (RECURSIVE_END - RECURSIVE_START) / 1000000 ))
RECURSIVE_SEC=$(echo "scale=3; ${RECURSIVE_MS}/1000" | bc)

if [ -f "$PROOF_OUTPUT" ]; then
    echo ""
    echo -e "  ${GREEN}Recursive STARK generated in ${RECURSIVE_SEC}s${NC}"
    echo "  Output: ${PROOF_OUTPUT} ($(du -h "${PROOF_OUTPUT}" | cut -f1))"
else
    echo -e "${RED}  ERROR: Recursive proof not generated${NC}"
    exit 1
fi
echo ""

# ----------------------------------------------------------------
# Step 6: Verify the STARK proof locally
# ----------------------------------------------------------------
echo -e "${YELLOW}[Step 6] Verifying Circle STARK proof locally${NC}"
echo "  Proof file: ${PROOF_OUTPUT}"
echo "  Size: $(du -h "$PROOF_OUTPUT" | cut -f1)"

${CAIRO_PROVE_BIN} verify "${PROOF_OUTPUT}" && {
    echo -e "  ${GREEN}LOCAL VERIFICATION: PASS${NC}"
} || {
    echo -e "  ${RED}LOCAL VERIFICATION: FAIL${NC}"
    exit 1
}

echo ""

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
TOTAL_SEC=$(echo "scale=3; ${PROVE_SEC} + ${RECURSIVE_SEC}" | bc)

echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                                   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
printf "║  GPU prove time:     %-30s ║\n" "${PROVE_SEC}s"
printf "║  Recursive STARK:    %-30s ║\n" "${RECURSIVE_SEC}s"
printf "║  Total pipeline:     %-30s ║\n" "${TOTAL_SEC}s"
printf "║  ML proof:           %-30s ║\n" "$(du -h "${ML_PROOF_OUTPUT}" | cut -f1)"
printf "║  Recursive proof:    %-30s ║\n" "$(du -h "${PROOF_OUTPUT}" | cut -f1)"
echo "║                                                      ║"
echo "║  Next: Submit to Starknet                            ║"
echo "║    bash scripts/h200_submit_onchain.sh \\             ║"
echo "║      --proof ${PROOF_OUTPUT} --submit                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
