#!/usr/bin/env bash
# run_h100_qwen35b_full.sh - Production-oriented H100 Qwen3.5-35B-A3B proving run.
#
# This script is intentionally biased toward full-model runs. Passing
# --layers 1 is allowed for smoke diagnostics, but it is not a production proof.
#
# Usage:
#   bash scripts/run_h100_qwen35b_full.sh --model-dir ~/.obelyzk/models/qwen3.5-35b-a3b
#   RUN_RECURSIVE=1 bash scripts/run_h100_qwen35b_full.sh --model-dir ~/models/qwen3.5-35b-a3b
#   bash scripts/run_h100_qwen35b_full.sh --layers all --decode-steps 16
#
# Environment:
#   MODEL_DIR       HuggingFace model directory.
#   LAYERS          all/full/0 or numeric layer count. Default: all.
#   OUT_DIR         Output directory. Default: ~/obelyzk-h100-runs/qwen35b_<ts>.
#   RUN_GKR         Run full GKR proof. Default: 1.
#   RUN_DECODE      Run decode benchmark. Default: 1.
#   RUN_RECURSIVE   Run recursive STARK compression. Default: 0.
#   SKIP_BUILD      Skip release build. Default: 0.
#   DECODE_STEPS    Decode benchmark steps. Default: 16.
#   PREFILL_LEN     Prefill length. Default: 128.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
LAYERS="${LAYERS:-all}"
OUT_DIR="${OUT_DIR:-}"
RUN_GKR="${RUN_GKR:-1}"
RUN_DECODE="${RUN_DECODE:-1}"
RUN_RECURSIVE="${RUN_RECURSIVE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DECODE_STEPS="${DECODE_STEPS:-16}"
PREFILL_LEN="${PREFILL_LEN:-128}"
POLICY="${POLICY:-strict}"

usage() {
  cat <<'USAGE'
Usage: run_h100_qwen35b_full.sh [OPTIONS]

Options:
  --model-dir PATH       HuggingFace model directory.
  --layers N|all         Layers to prove. Default: all from config.json.
  --out-dir PATH         Output directory.
  --decode-steps N       Decode benchmark steps. Default: 16.
  --prefill-len N        Prefill length. Default: 128.
  --skip-build           Reuse target/release/prove-model.
  --no-gkr               Skip full GKR proof.
  --no-decode            Skip decode benchmark.
  --recursive            Run recursive STARK compression after GKR.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
    --prefill-len) PREFILL_LEN="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-gkr) RUN_GKR=0; shift ;;
    --no-decode) RUN_DECODE=0; shift ;;
    --recursive) RUN_RECURSIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  for candidate in \
    "$HOME/.obelyzk/models/qwen3.5-35b-a3b" \
    "$HOME/.obelyzk/models/qwen3.5-35b-a3b-fp8" \
    "$HOME/.obelyzk/models/qwen3.5-35b-a3b-gptq-int4" \
    "$HOME/models/qwen3.5-35b-a3b" \
    "$HOME/models/qwen3.5-35b-a3b-fp8" \
    "$HOME/models/qwen3.5-35b-a3b-gptq-int4" \
    "$HOME/.obelyzk/models/qwen3-14b" \
    "$HOME/.obelyzk/models/qwen2.5-14b" \
    "$HOME/.obelysk/models/qwen3-14b" \
    "$HOME/models/qwen3-14b" \
    "$HOME/models/qwen2.5-14b"; do
    if [[ -f "$candidate/config.json" ]]; then
      MODEL_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: Set --model-dir to a Qwen3.5-35B-A3B HuggingFace directory containing config.json."
  exit 1
fi

case "$LAYERS" in
  all|full|0|"")
    LAYERS_LABEL="all"
    LAYER_FLAGS=()
    ;;
  *)
    LAYERS_LABEL="${LAYERS}L"
    LAYER_FLAGS=(--layers "$LAYERS")
    ;;
esac

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$HOME/obelyzk-h100-runs/qwen35b_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/run.log"
GPU_INFO="$OUT_DIR/gpu.txt"
ENV_INFO="$OUT_DIR/env.txt"
INSPECT_OUT="$OUT_DIR/inspect.txt"
CONTRACT_OUT="$OUT_DIR/qwen35_contract.json"
READINESS_OUT="$OUT_DIR/qwen35_readiness.json"
GKR_OUT="$OUT_DIR/qwen35b_full_gkr.json"
DECODE_OUT="$OUT_DIR/qwen35b_decode.json"
RECURSIVE_OUT="$OUT_DIR/qwen35b_full_recursive.json"
KV_CACHE_DIR="$OUT_DIR/kv_cache"

exec > >(tee -a "$LOG") 2>&1

echo "=== Obelyzk H100 Qwen3.5-35B-A3B Full Proving Run ==="
echo "Start:       $(date -Is)"
echo "Repo:        $REPO_ROOT"
echo "Model:       $MODEL_DIR"
echo "Layers:      $LAYERS_LABEL"
echo "Output dir:  $OUT_DIR"
echo "Policy:      $POLICY"
echo "GKR:         $RUN_GKR"
echo "Decode:      $RUN_DECODE (prefill=$PREFILL_LEN, steps=$DECODE_STEPS)"
echo "Recursive:   $RUN_RECURSIVE"
if [[ "$LAYERS" == "1" ]]; then
  echo "WARNING:     1-layer mode is diagnostic only. Use --layers all for production evidence."
fi
echo ""

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi | tee "$GPU_INFO"
else
  echo "ERROR: nvidia-smi not found. This script is intended for H100/H200 CUDA hosts."
  exit 1
fi

{
  echo "rustc: $(rustc --version 2>/dev/null || echo missing)"
  echo "cargo: $(cargo --version 2>/dev/null || echo missing)"
  echo "nvcc:  $(nvcc --version 2>/dev/null | grep release || echo missing)"
  echo "git:   $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} | tee "$ENV_INFO"
echo ""

FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit,cuda-runtime"
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "=== Build prove-model ==="
  (cd "$REPO_ROOT" && cargo build --release --bin prove-model --features "$FEATURES")
  echo ""
fi

BIN="$REPO_ROOT/target/release/prove-model"
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: prove-model binary not found at $BIN"
  exit 1
fi

echo "=== Inspect / Validate Model ==="
set +e
"$BIN" \
  --model-dir "$MODEL_DIR" \
  "${LAYER_FLAGS[@]}" \
  --qwen35-readiness-json "$READINESS_OUT" \
  --inspect 2>&1 | tee "$INSPECT_OUT"
INSPECT_STATUS="${PIPESTATUS[0]}"
set -e
CONTRACT_HASH="$(grep -o 'contract_hash=0x[0-9a-fA-F]*' "$INSPECT_OUT" | head -n1 | cut -d= -f2 || true)"
{
  echo "{"
  echo "  \"schema\": \"obelyzk.qwen35_contract.v1\","
  echo "  \"model_dir\": \"$(printf '%s' "$MODEL_DIR" | sed 's/\\/\\\\/g; s/"/\\"/g')\","
  echo "  \"layers\": \"$(printf '%s' "$LAYERS_LABEL" | sed 's/\\/\\\\/g; s/"/\\"/g')\","
  echo "  \"execution_contract_hash\": \"${CONTRACT_HASH:-0x0}\","
  echo "  \"readiness_output\": \"$(printf '%s' "$READINESS_OUT" | sed 's/\\/\\\\/g; s/"/\\"/g')\","
  echo "  \"inspect_status\": $INSPECT_STATUS,"
  echo "  \"inspect_output\": \"$(printf '%s' "$INSPECT_OUT" | sed 's/\\/\\\\/g; s/"/\\"/g')\""
  echo "}"
} > "$CONTRACT_OUT"
echo "Qwen3.5 contract metadata: $CONTRACT_OUT"
if [[ "$INSPECT_STATUS" != "0" ]]; then
  echo "Inspect failed with status $INSPECT_STATUS; refusing to continue."
  exit "$INSPECT_STATUS"
fi
"$BIN" --model-dir "$MODEL_DIR" "${LAYER_FLAGS[@]}" --validate
echo ""

if [[ "$RUN_GKR" == "1" ]]; then
  echo "=== Full GKR Proof ==="
  "$BIN" \
    --model-dir "$MODEL_DIR" \
    "${LAYER_FLAGS[@]}" \
    --gpu \
    --gkr \
    --format ml_gkr \
    --policy "$POLICY" \
    --profile \
    --health-check \
    --output "$GKR_OUT"
  echo "GKR proof: $GKR_OUT"
  [[ -f "${GKR_OUT%.json}.profile.json" ]] && echo "GKR profile: ${GKR_OUT%.json}.profile.json"
  echo ""
fi

if [[ "$RUN_DECODE" == "1" ]]; then
  echo "=== Multi-token Decode Benchmark ==="
  mkdir -p "$KV_CACHE_DIR"
  "$BIN" \
    --model-dir "$MODEL_DIR" \
    "${LAYER_FLAGS[@]}" \
    --gpu \
    --format ml_gkr \
    --policy "$POLICY" \
    --output "$DECODE_OUT" \
    --kv-cache-dir "$KV_CACHE_DIR" \
    --decode-bench "$DECODE_STEPS" \
    --prefill-len "$PREFILL_LEN" \
    --profile
  echo "Decode output: $DECODE_OUT"
  [[ -f "${DECODE_OUT%.json}.decode_bench.json" ]] && echo "Decode metrics: ${DECODE_OUT%.json}.decode_bench.json"
  echo ""
fi

if [[ "$RUN_RECURSIVE" == "1" ]]; then
  echo "=== Recursive STARK Compression ==="
  OBELYZK_HADES_AIR="${OBELYZK_HADES_AIR:-1}" \
  OBELYZK_LOGUP="${OBELYZK_LOGUP:-1}" \
  "$BIN" \
    --model-dir "$MODEL_DIR" \
    "${LAYER_FLAGS[@]}" \
    --gpu \
    --gkr \
    --recursive \
    --format ml_gkr \
    --policy "$POLICY" \
    --profile \
    --health-check \
    --output "$RECURSIVE_OUT"
  echo "Recursive proof: $RECURSIVE_OUT"
  echo ""
fi

echo "=== Complete ==="
echo "End:        $(date -Is)"
echo "Run log:    $LOG"
echo "Artifacts:  $OUT_DIR"
