#!/usr/bin/env bash
# benchmark_production_throughput.sh
#
# Production-readiness throughput harness for ZKML proving.
#
# This intentionally measures separate surfaces:
#   1. Batched GKR throughput across sequence lengths.
#   2. Autoregressive decode-step proof latency with KV continuity.
#   3. Single-token recursive overhead for the trustless finalization path.
#
# The three numbers must not be collapsed into one marketing number. Batched
# GKR tok/s answers "how many tokens can one proof amortize?" Decode tok/s
# answers "how fast can we prove generated tokens?" Recursive overhead answers
# "what does trustless finalization add?"
#
# Usage:
#   bash scripts/benchmark_production_throughput.sh \
#     --model-dir ~/.obelysk/models/qwen3-14b \
#     --layers 40 \
#     --seq-lens 1,100,1000,5000,10000 \
#     --decode-steps 16 \
#     --prefill-len 128 \
#     --gpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${ENGINE_DIR}/.." && pwd)"

MODEL_DIR=""
LAYERS="40"
SEQ_LENS="1,100,1000,5000,10000"
DECODE_STEPS="16"
PREFILL_LEN="128"
GPU_FLAG=""
SKIP_BUILD="0"
RUN_RECURSIVE="1"
RUN_DECODE="1"
OUT_DIR="${REPO_DIR}/benchmarks/production_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<EOF
Production-readiness throughput harness for ZKML proving.

Usage:
  bash scripts/benchmark_production_throughput.sh \\
    --model-dir ~/.obelysk/models/qwen3-14b \\
    --layers 40 \\
    --seq-lens 1,100,1000,5000,10000 \\
    --decode-steps 16 \\
    --prefill-len 128 \\
    --gpu

Options:
  --model-dir PATH       HuggingFace model directory, required
  --layers N             Transformer blocks to prove, default 40
  --seq-lens LIST        Comma-separated batch sizes, default ${SEQ_LENS}
  --decode-steps N       Decode tokens to prove, default ${DECODE_STEPS}
  --prefill-len N        Synthetic prefill length, default ${PREFILL_LEN}
  --gpu                  Use CUDA when available
  --skip-build           Use existing target/release/prove-model
  --skip-recursive       Skip recursive overhead run
  --skip-decode          Skip decode/KV benchmark
  --out-dir PATH         Output directory
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --seq-lens) SEQ_LENS="$2"; shift 2 ;;
    --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
    --prefill-len) PREFILL_LEN="$2"; shift 2 ;;
    --gpu) GPU_FLAG="--gpu"; shift ;;
    --skip-build) SKIP_BUILD="1"; shift ;;
    --skip-recursive) RUN_RECURSIVE="0"; shift ;;
    --skip-decode) RUN_DECODE="0"; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "ERROR: --model-dir is required"
  usage
fi

mkdir -p "$OUT_DIR"
PROVE_BIN="${ENGINE_DIR}/target/release/prove-model"

echo "=== Production ZKML Throughput Benchmark ==="
echo "  Model dir:     ${MODEL_DIR}"
echo "  Layers:        ${LAYERS}"
echo "  Seq lens:      ${SEQ_LENS}"
echo "  Decode steps:  ${DECODE_STEPS}"
echo "  Prefill len:   ${PREFILL_LEN}"
echo "  GPU flag:      ${GPU_FLAG:-off}"
echo "  Output dir:    ${OUT_DIR}"
echo ""

GPU_NAME="none"
GPU_MEMORY="n/a"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo unknown)"
  GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo unknown) MiB"
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit"
  if [[ -n "$GPU_FLAG" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    FEATURES="${FEATURES},cuda-runtime"
  fi
  echo "[build] cargo build --release --bin prove-model --features ${FEATURES}"
  (cd "$ENGINE_DIR" && cargo build --release --bin prove-model --features "$FEATURES")
fi

if [[ ! -f "$PROVE_BIN" ]]; then
  echo "ERROR: prove-model binary not found at ${PROVE_BIN}"
  exit 1
fi

cat > "${OUT_DIR}/environment.json" <<JSON
{
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)",
  "gpu": "${GPU_NAME}",
  "gpu_memory": "${GPU_MEMORY}",
  "rust": "$(rustc --version 2>/dev/null || echo unknown)",
  "model_dir": "${MODEL_DIR}",
  "layers": ${LAYERS},
  "security_target_bits": 160
}
JSON

echo "[1/3] Batched GKR throughput matrix"
GKR_OUT="${OUT_DIR}/gkr_batch.json"
GKR_LOG="${OUT_DIR}/gkr_batch.log"
"$PROVE_BIN" \
  --model-dir "$MODEL_DIR" \
  --layers "$LAYERS" \
  $GPU_FLAG \
  --bench \
  --bench-seq-lens "$SEQ_LENS" \
  --bench-warmup 1 \
  --output "$GKR_OUT" \
  2>&1 | tee "$GKR_LOG"

if [[ "$RUN_DECODE" == "1" ]]; then
  echo "[2/3] Decode/KV proof throughput"
  DECODE_OUT="${OUT_DIR}/decode.json"
  DECODE_LOG="${OUT_DIR}/decode.log"
  KV_DIR="${OUT_DIR}/kv_cache"
  mkdir -p "$KV_DIR"
  "$PROVE_BIN" \
    --model-dir "$MODEL_DIR" \
    --layers "$LAYERS" \
    $GPU_FLAG \
    --format ml_gkr \
    --kv-cache-dir "$KV_DIR" \
    --decode-bench "$DECODE_STEPS" \
    --prefill-len "$PREFILL_LEN" \
    --profile \
    --output "$DECODE_OUT" \
    2>&1 | tee "$DECODE_LOG"
else
  echo "[2/3] Decode/KV proof throughput skipped"
fi

if [[ "$RUN_RECURSIVE" == "1" ]]; then
  echo "[3/3] Single-token recursive overhead"
  RECURSIVE_OUT="${OUT_DIR}/recursive_single.json"
  RECURSIVE_LOG="${OUT_DIR}/recursive_single.log"
  "$PROVE_BIN" \
    --model-dir "$MODEL_DIR" \
    --layers "$LAYERS" \
    $GPU_FLAG \
    --format ml_gkr \
    --gkr \
    --recursive \
    --profile \
    --output "$RECURSIVE_OUT" \
    2>&1 | tee "$RECURSIVE_LOG"
else
  echo "[3/3] Recursive overhead skipped"
fi

cat > "${OUT_DIR}/README.txt" <<EOF
Production throughput benchmark outputs

environment.json      Hardware, model, and security target.
gkr_batch.bench.json  Batched GKR seq_len -> tok/s results.
gkr_batch.log         Raw batched GKR log.
decode.decode_bench.json
                      Autoregressive decode-step latency and phase timing.
decode.log            Raw decode benchmark log.
recursive_single.json Single-token recursive proof artifact when enabled.
recursive_single.log  Raw recursive overhead log.

Interpretation:
- Use gkr_batch.bench.json for peak amortized proof throughput.
- Use decode.decode_bench.json for generated-token proving latency.
- Add recursive_single.log timing to decode or batch timing for trustless
  finalization until multi-step recursive session aggregation is measured.
EOF

echo ""
echo "Benchmark artifacts written to ${OUT_DIR}"
