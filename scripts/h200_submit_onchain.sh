#!/usr/bin/env bash
#
# Obelysk Protocol — H200 On-Chain STARK Proof Submission
# ========================================================
#
# Complete pipeline for submitting the recursive STARK proof on-chain
# from the H200 GPU worker.
#
# This script:
#   1. Sets up the H200 environment (CUDA, sncast)
#   2. Optionally runs the full recursive pipeline (prove-model -> cairo-prove)
#   3. Parses the proof and submits verify_recursive_output() on-chain
#
# Usage:
#   cd /path/to/bitsage-network/libs
#   bash scripts/h200_submit_onchain.sh --proof recursive_proof.json --dry-run
#   bash scripts/h200_submit_onchain.sh --proof recursive_proof.json --submit
#   bash scripts/h200_submit_onchain.sh --prove --layers 1 --model-dir ~/models/qwen3-14b --submit
#
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

# Contract addresses (Starknet Sepolia)
STARK_VERIFIER="0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba"
OBELYSK_VERIFIER="0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15"

# Paths
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3-14b}"
PROOF_FILE="${PROOF_FILE:-recursive_proof.json}"

# Starknet
ACCOUNT="${SNCAST_ACCOUNT:-deployer}"

# CUDA (critical for H200 — driver 550 needs 12.4 NVRTC)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu"
export PATH="/usr/local/cuda-12.4/bin:$PATH"

# ═══════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════

DO_PROVE=false
DRY_RUN=false
DO_SUBMIT=false
SKIP_BUILD=false
NUM_LAYERS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --prove)       DO_PROVE=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --submit)      DO_SUBMIT=true; shift ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --layers)      NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --proof)       PROOF_FILE="$2"; shift 2 ;;
        --account)     ACCOUNT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--prove] [--submit|--dry-run] [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prove       Run full recursive pipeline first (prove-model -> cairo-prove)"
            echo "  --submit      Submit transactions on-chain (required for actual submission)"
            echo "  --dry-run     Print commands without executing"
            echo "  --skip-build  Skip building prove-model and cairo-prove"
            echo "  --layers N    Number of transformer layers to prove (default: 1)"
            echo "  --model-dir   Path to Qwen3-14B model directory"
            echo "  --proof       Path to recursive_proof.json (default: recursive_proof.json)"
            echo "  --account     sncast account name (default: deployer)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$DRY_RUN" = false ] && [ "$DO_SUBMIT" = false ]; then
    echo "ERROR: Must specify --dry-run or --submit"
    echo "  --dry-run: Print commands without executing"
    echo "  --submit:  Actually submit on-chain"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════

echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Obelysk Protocol — H200 On-Chain STARK Submission Pipeline  ║"
echo "║                                                               ║"
echo "║  prove-model -> cairo-prove -> verify_recursive_output()     ║"
echo "║  Qwen3-14B -> Starknet Sepolia (1 tx, ~20K gas)             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════
# Step 0: Environment
# ═══════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Step 0] Environment${NC}"

# GPU check
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:    ${GPU_NAME} (${GPU_MEM})"
    echo "  Driver: ${DRIVER}"
    echo "  CUDA:   $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',' || echo 'not found')"
else
    echo "  GPU: Not detected (CPU mode)"
fi

echo "  Rust:   $(rustc --version 2>/dev/null || echo 'not found')"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

# sncast check
if ! command -v sncast &>/dev/null; then
    echo -e "${YELLOW}  sncast not found. Installing starknet-foundry...${NC}"
    curl -L https://raw.githubusercontent.com/foundry-rs/starknet-foundry/master/scripts/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v sncast &>/dev/null; then
        echo -e "${RED}  ERROR: sncast still not found after install${NC}"
        echo "  Try: cargo install starknet-foundry"
        exit 1
    fi
fi
echo "  sncast: $(sncast --version 2>/dev/null || echo 'available')"
echo ""

# ═══════════════════════════════════════════════════════════════
# Step 1 (Optional): Full Recursive Pipeline
# ═══════════════════════════════════════════════════════════════

if [ "$DO_PROVE" = true ]; then
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  PHASE 1: FULL RECURSIVE PIPELINE${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════${NC}"
    echo ""

    # Delegate to the recursive pipeline script
    PIPELINE_ARGS=""
    if [ "$SKIP_BUILD" = true ]; then
        PIPELINE_ARGS+=" --skip-build"
    fi
    PIPELINE_ARGS+=" --layers ${NUM_LAYERS}"
    PIPELINE_ARGS+=" --output ${PROOF_FILE}"
    if [ -n "$MODEL_DIR" ]; then
        PIPELINE_ARGS+=" --model-dir ${MODEL_DIR}"
    fi

    bash "${SCRIPT_DIR}/h200_recursive_pipeline.sh" ${PIPELINE_ARGS}

    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Step 2: Parse Proof and Submit On-Chain
# ═══════════════════════════════════════════════════════════════

echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  PHASE 2: ON-CHAIN SUBMISSION${NC}"
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════${NC}"
echo ""

if [ ! -f "$PROOF_FILE" ]; then
    echo -e "${RED}ERROR: Proof file not found: ${PROOF_FILE}${NC}"
    echo "  Run with --prove to generate it, or specify --proof PATH"
    exit 1
fi

echo "  Proof file: ${PROOF_FILE} ($(du -h "${PROOF_FILE}" | cut -f1))"
echo ""

# Determine mode flag
MODE_FLAG="--dry-run"
if [ "$DO_SUBMIT" = true ]; then
    MODE_FLAG="--submit"
fi

# Find the submit script — check multiple locations
SUBMIT_SCRIPT=""

# 1. Same directory as this script (libs/scripts/)
if [ -f "${SCRIPT_DIR}/submit_recursive_proof.py" ]; then
    SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_recursive_proof.py"
fi

# 2. Parent repo's scripts/ directory
if [ -z "$SUBMIT_SCRIPT" ] && [ -f "${REPO_DIR}/../scripts/submit_recursive_proof.py" ]; then
    SUBMIT_SCRIPT="${REPO_DIR}/../scripts/submit_recursive_proof.py"
fi

# 3. Relative to current directory
if [ -z "$SUBMIT_SCRIPT" ] && [ -f "scripts/submit_recursive_proof.py" ]; then
    SUBMIT_SCRIPT="scripts/submit_recursive_proof.py"
fi

# 4. Relative to repo root
if [ -z "$SUBMIT_SCRIPT" ] && [ -f "../scripts/submit_recursive_proof.py" ]; then
    SUBMIT_SCRIPT="../scripts/submit_recursive_proof.py"
fi

if [ -z "$SUBMIT_SCRIPT" ]; then
    echo -e "${RED}ERROR: submit_recursive_proof.py not found${NC}"
    echo "  Searched:"
    echo "    ${SCRIPT_DIR}/submit_recursive_proof.py"
    echo "    ${REPO_DIR}/../scripts/submit_recursive_proof.py"
    echo "    scripts/submit_recursive_proof.py"
    echo "    ../scripts/submit_recursive_proof.py"
    exit 1
fi

echo "  Submit script: ${SUBMIT_SCRIPT}"
echo ""

python3 "${SUBMIT_SCRIPT}" \
    --proof "${PROOF_FILE}" \
    --account "${ACCOUNT}" \
    ${MODE_FLAG}

echo ""
echo -e "${GREEN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════╗"
if [ "$DO_SUBMIT" = true ]; then
echo "║  ON-CHAIN SUBMISSION COMPLETE                                ║"
echo "║                                                               ║"
echo "║  Qwen3-14B inference verified on Starknet Sepolia            ║"
echo "║  via verify_recursive_output() — 1 tx, permanent record     ║"
else
echo "║  DRY RUN COMPLETE                                            ║"
echo "║                                                               ║"
echo "║  Re-run with --submit to execute on-chain                    ║"
fi
echo "║                                                               ║"
echo "║  Contract: ${STARK_VERIFIER}  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
