#!/usr/bin/env bash
# Qwen3.5-35B-A3B H100 setup.
#
# Fresh cloud GPU usage:
#   bash scripts/setup_h100_qwen35b.sh
#   bash scripts/setup_h100_qwen35b.sh --model qwen3.5-35b-a3b --run-qwen-smoke
#
# This script intentionally separates setup/smoke from the full proving run.
# Use --run-qwen-smoke for a small model-path run, then run
# scripts/run_h100_qwen35b_full.sh for production evidence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="${REPO_URL:-https://github.com/Bitsage-Network/obelyzk.rs.git}"
BRANCH="${BRANCH:-development}"
INSTALL_DIR="${INSTALL_DIR:-}"
MODEL="${MODEL:-qwen3.5-35b-a3b-fp8}"
MODEL_BASE="${MODEL_BASE:-$HOME/.obelyzk/models}"
MODEL_DIR="${MODEL_DIR:-}"
OUT_DIR="${OUT_DIR:-$HOME/obelyzk-h100-runs/setup_qwen35b_$(date +%Y%m%d_%H%M%S)}"
LAYERS="${LAYERS:-all}"
DECODE_STEPS="${DECODE_STEPS:-2}"
PREFILL_LEN="${PREFILL_LEN:-32}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly-2025-07-14}"
SCARB_VERSION="${SCARB_VERSION:-2.12.2}"

SKIP_DEPS=0
SKIP_REPO=0
SKIP_BUILD=0
SKIP_MODEL=0
SKIP_ACTIVE_SMOKE=0
RUN_QWEN_SMOKE=0
RUN_RECURSIVE=0

usage() {
  cat <<'USAGE'
Usage: setup_h100_qwen35b.sh [OPTIONS]

Options:
  --model NAME          Model target. Default: qwen3.5-35b-a3b-fp8.
                        Supported: qwen3.5-35b-a3b, qwen3.5-35b-a3b-fp8,
                        qwen3.5-35b-a3b-gptq-int4.
  --model-dir PATH      Existing or desired HuggingFace model directory.
  --model-base PATH     Base directory for model downloads. Default: ~/.obelyzk/models.
  --install-dir PATH    Repo checkout directory. Default: current repo when run in-tree,
                        otherwise ~/bitsage-network.
  --repo-url URL        Git URL used when cloning. Default: Bitsage-Network/obelyzk.rs.
  --branch NAME         Git branch to checkout/pull. Default: development.
  --out-dir PATH        Setup artifacts/logs directory.
  --skip-deps           Skip apt/yum/pip/node/rust/scarb dependency install.
  --skip-repo           Do not clone/fetch/pull repo.
  --skip-build          Skip Rust/Cairo builds.
  --skip-model          Skip model download.
  --skip-active-smoke   Skip active Qwen3.5 recursive statement smoke proof.
  --run-qwen-smoke      After setup, run a small Qwen model-path smoke.
  --recursive           Include recursive mode when --run-qwen-smoke is used.
  --layers N|all        Layers for optional --run-qwen-smoke. Default: all.
  --decode-steps N      Decode steps for optional --run-qwen-smoke. Default: 2.
  --prefill-len N       Prefill length for optional --run-qwen-smoke. Default: 32.
  -h, --help            Show help.

Environment overrides:
  REPO_URL, BRANCH, INSTALL_DIR, MODEL, MODEL_BASE, MODEL_DIR, OUT_DIR,
  RUST_TOOLCHAIN, SCARB_VERSION.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-base) MODEL_BASE="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    --skip-repo) SKIP_REPO=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    --skip-active-smoke) SKIP_ACTIVE_SMOKE=1; shift ;;
    --run-qwen-smoke) RUN_QWEN_SMOKE=1; shift ;;
    --recursive) RUN_RECURSIVE=1; shift ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
    --prefill-len) PREFILL_LEN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

case "$MODEL" in
  qwen3.5-35b-a3b|qwen35b|qwen-35b) MODEL_CANONICAL="qwen3.5-35b-a3b"; HF_REPO="Qwen/Qwen3.5-35B-A3B" ;;
  qwen3.5-35b-a3b-fp8|qwen35b-fp8|qwen-35b-fp8) MODEL_CANONICAL="qwen3.5-35b-a3b-fp8"; HF_REPO="Qwen/Qwen3.5-35B-A3B-FP8" ;;
  qwen3.5-35b-a3b-gptq-int4|qwen35b-gptq-int4|qwen-35b-gptq-int4) MODEL_CANONICAL="qwen3.5-35b-a3b-gptq-int4"; HF_REPO="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" ;;
  *) echo "Unsupported Qwen3.5 model target: $MODEL" >&2; exit 1 ;;
esac
MODEL="$MODEL_CANONICAL"
if [[ -z "$MODEL_DIR" ]]; then
  MODEL_DIR="$MODEL_BASE/$MODEL"
fi

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

step() {
  echo ""
  echo "${CYAN}${BOLD}==> $*${NC}"
}

ok() {
  echo "  ${GREEN}OK${NC} $*"
}

warn() {
  echo "  ${YELLOW}WARN${NC} $*" >&2
}

die() {
  echo "  ${RED}ERROR${NC} $*" >&2
  exit 1
}

run() {
  echo "  $*"
  "$@"
}

detect_current_repo() {
  local engine_candidate
  engine_candidate="$(cd "$SCRIPT_DIR/.." && pwd)"
  if [[ -f "$engine_candidate/Cargo.toml" && -f "$engine_candidate/src/bin/prove_model.rs" ]]; then
    CURRENT_ENGINE_DIR="$engine_candidate"
    if [[ "$(basename "$(dirname "$engine_candidate")")" == "libs" ]]; then
      CURRENT_REPO_ROOT="$(cd "$engine_candidate/../.." && pwd)"
    else
      CURRENT_REPO_ROOT="$(cd "$engine_candidate/.." && pwd)"
    fi
  else
    CURRENT_ENGINE_DIR=""
    CURRENT_REPO_ROOT=""
  fi
}

engine_dir_for_repo() {
  local root="$1"
  if [[ -f "$root/libs/engine/Cargo.toml" ]]; then
    printf '%s\n' "$root/libs/engine"
  elif [[ -f "$root/engine/Cargo.toml" ]]; then
    printf '%s\n' "$root/engine"
  elif [[ -f "$root/Cargo.toml" && -f "$root/src/bin/prove_model.rs" ]]; then
    printf '%s\n' "$root"
  else
    return 1
  fi
}

libs_dir_for_engine() {
  local engine="$1"
  if [[ "$(basename "$(dirname "$engine")")" == "libs" ]]; then
    printf '%s\n' "$(cd "$engine/.." && pwd)"
  elif [[ -d "$(cd "$engine/.." && pwd)/libs" ]]; then
    printf '%s\n' "$(cd "$engine/../libs" && pwd)"
  else
    printf '%s\n' "$(cd "$engine/.." && pwd)"
  fi
}

find_cuda() {
  local candidate
  for candidate in \
    "${CUDA_HOME:-}" \
    "${CUDA_PATH:-}" \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.5 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda \
    /opt/cuda; do
    if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

install_system_deps() {
  step "Installing system dependencies"
  if [[ "$SKIP_DEPS" == "1" ]]; then
    warn "Skipping dependency installation"
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    run sudo apt-get update -qq
    run sudo apt-get install -y -qq \
      build-essential cmake pkg-config libssl-dev \
      git git-lfs curl wget python3 python3-pip python3-venv jq bc nodejs npm
  elif command -v yum >/dev/null 2>&1; then
    run sudo yum install -y -q \
      gcc gcc-c++ make cmake openssl-devel pkg-config \
      git git-lfs curl wget python3 python3-pip jq bc nodejs npm
  else
    warn "Unknown package manager; install build-essential/cmake/pkg-config/libssl/git/git-lfs/python3/jq/bc/node manually"
  fi

  git lfs install >/dev/null 2>&1 || true
  python3 -m pip install --quiet --upgrade --user huggingface_hub filelock >/dev/null 2>&1 || true
  ok "system dependencies available"
}

install_rust_and_scarb() {
  step "Installing Rust and Scarb"
  if [[ "$SKIP_DEPS" == "1" ]]; then
    warn "Skipping Rust/Scarb installation"
    return
  fi

  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
  else
    # shellcheck source=/dev/null
    [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
  fi

  run rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
  run rustup component add rust-src --toolchain "$RUST_TOOLCHAIN" >/dev/null 2>&1 || true
  ok "$(RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN" rustc --version)"

  if ! command -v scarb >/dev/null 2>&1; then
    curl -L https://docs.swmansion.com/scarb/install.sh | sh -s -- -v "$SCARB_VERSION"
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if command -v scarb >/dev/null 2>&1; then
    ok "$(scarb --version)"
  else
    die "scarb was not installed; recursive Cairo verifier build will fail"
  fi
}

verify_gpu_cuda() {
  step "Verifying H100/CUDA environment"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Start a GPU image with NVIDIA drivers installed."
  nvidia-smi

  CUDA_DIR="$(find_cuda || true)"
  if [[ -z "$CUDA_DIR" ]]; then
    warn "CUDA toolkit nvcc not found; attempting apt install of cuda-toolkit"
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -qq
      sudo apt-get install -y -qq cuda-toolkit-12-6 || sudo apt-get install -y -qq cuda-toolkit-12-5 || true
    fi
    CUDA_DIR="$(find_cuda || true)"
  fi
  [[ -n "$CUDA_DIR" ]] || die "CUDA toolkit nvcc not found. Install CUDA toolkit or use a CUDA developer image."

  export CUDA_HOME="$CUDA_DIR"
  export CUDA_PATH="$CUDA_DIR"
  export PATH="$CUDA_DIR/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"

  mkdir -p "$HOME/.obelyzk"
  cat > "$HOME/.obelyzk/cuda_env.sh" <<EOF
export CUDA_HOME="$CUDA_DIR"
export CUDA_PATH="$CUDA_DIR"
export PATH="$CUDA_DIR/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:\${LD_LIBRARY_PATH:-}"
EOF
  ok "$("$CUDA_DIR/bin/nvcc" --version | grep release || true)"
}

prepare_repo() {
  step "Preparing repository"
  detect_current_repo

  if [[ -z "$INSTALL_DIR" ]]; then
    if [[ -n "$CURRENT_REPO_ROOT" ]]; then
      INSTALL_DIR="$CURRENT_REPO_ROOT"
    else
      INSTALL_DIR="$HOME/bitsage-network"
    fi
  fi

  if [[ "$SKIP_REPO" == "1" ]]; then
    [[ -d "$INSTALL_DIR" ]] || die "--skip-repo set but install dir does not exist: $INSTALL_DIR"
  elif [[ -d "$INSTALL_DIR/.git" ]]; then
    (
      cd "$INSTALL_DIR"
      git fetch origin || warn "git fetch failed"
      git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH" 2>/dev/null || true
      git pull origin "$BRANCH" --ff-only || warn "git pull --ff-only failed; continuing with current checkout"
      git submodule update --init --recursive || true
    )
  else
    run git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    (cd "$INSTALL_DIR" && git submodule update --init --recursive || true)
  fi

  ENGINE_DIR="$(engine_dir_for_repo "$INSTALL_DIR")" || die "cannot find engine Cargo.toml under $INSTALL_DIR"
  LIBS_DIR="$(libs_dir_for_engine "$ENGINE_DIR")"
  ok "repo=$INSTALL_DIR"
  ok "engine=$ENGINE_DIR"
  ok "libs=$LIBS_DIR"
}

build_stack() {
  step "Building Qwen3.5 proving stack"
  if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "Skipping builds"
    return
  fi

  local features="std,gpu,onnx,safetensors,model-loading,cli,audit,cuda-runtime"
  (
    cd "$ENGINE_DIR"
    RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN" cargo build --release --bin prove-model --features "$features"
  )
  PROVE_MODEL_BIN="$ENGINE_DIR/target/release/prove-model"
  [[ -x "$PROVE_MODEL_BIN" ]] || die "prove-model binary missing after build"
  ok "prove-model=$PROVE_MODEL_BIN"

  if [[ -d "$LIBS_DIR/stark-cairo/cairo-prove" ]]; then
    (cd "$LIBS_DIR/stark-cairo/cairo-prove" && RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN" cargo build --release)
    ok "cairo-prove built"
  else
    warn "cairo-prove directory not found; active smoke script may use cargo run if available"
  fi

  local cairo_crate
  for cairo_crate in \
    "$LIBS_DIR/qwen35-active-statement-verifier" \
    "$LIBS_DIR/conversation-statement-verifier" \
    "$LIBS_DIR/conversation-gkr-statement-verifier"; do
    if [[ -f "$cairo_crate/Scarb.toml" ]]; then
      (cd "$cairo_crate" && scarb build)
      ok "built $(basename "$cairo_crate")"
    fi
  done
}

download_model() {
  step "Preparing model $MODEL"
  if [[ "$SKIP_MODEL" == "1" ]]; then
    warn "Skipping model download"
    return
  fi
  if [[ -f "$MODEL_DIR/config.json" ]]; then
    ok "model already present at $MODEL_DIR"
    return
  fi

  mkdir -p "$(dirname "$MODEL_DIR")"
  if [[ -x "$ENGINE_DIR/scripts/download_model.sh" && "$MODEL_DIR" == "$MODEL_BASE/$MODEL" ]]; then
    OBELYZK_MODEL_DIR="$MODEL_BASE" bash "$ENGINE_DIR/scripts/download_model.sh" "$MODEL"
  else
    python3 -m pip install --quiet --upgrade --user huggingface_hub >/dev/null 2>&1 || true
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "$HF_REPO",
    local_dir="$MODEL_DIR",
    ignore_patterns=["*.bin", "*.pt", "*.onnx", "*.msgpack"],
)
PY
  fi
  [[ -f "$MODEL_DIR/config.json" ]] || die "model download did not produce config.json at $MODEL_DIR"
  ok "model=$MODEL_DIR"
}

inspect_model() {
  step "Inspecting Qwen3.5 contract/readiness"
  PROVE_MODEL_BIN="${PROVE_MODEL_BIN:-$ENGINE_DIR/target/release/prove-model}"
  [[ -x "$PROVE_MODEL_BIN" ]] || die "prove-model binary not found: $PROVE_MODEL_BIN"

  mkdir -p "$OUT_DIR"
  READINESS_OUT="$OUT_DIR/qwen35_readiness.json"
  INSPECT_OUT="$OUT_DIR/qwen35_inspect.txt"
  VALIDATE_OUT="$OUT_DIR/qwen35_validate.txt"

  set +e
  "$PROVE_MODEL_BIN" \
    --model-dir "$MODEL_DIR" \
    --qwen35-readiness-json "$READINESS_OUT" \
    --inspect >"$INSPECT_OUT" 2>&1
  local inspect_status=$?
  set -e
  cat "$INSPECT_OUT"
  [[ "$inspect_status" == "0" ]] || die "Qwen3.5 inspect failed with status $inspect_status"

  "$PROVE_MODEL_BIN" --model-dir "$MODEL_DIR" --validate >"$VALIDATE_OUT" 2>&1
  cat "$VALIDATE_OUT"
  ok "readiness=$READINESS_OUT"
}

write_active_smoke_input() {
  local path="$1"
  cat > "$path" <<'JSON'
{
  "model_id": "0x1001",
  "policy_commitment": "0x1002",
  "security_bits": 160,
  "active_conversations": [
    {
      "architecture_contract_hash": "0x2001",
      "weight_super_root": "0x2002",
      "receipt_hash": "0x3001",
      "conversation": {
        "conversation_index": 0,
        "conversation_id_hash": "0x4001",
        "prompt_commitment": "0x4002",
        "transcript_commitment": "0x4003",
        "action_root": "0x64a88cfe1e6fd9cfc7dafb6bf512fa59a487c956a4313ff95b9e2c8e97320df",
        "initial_kv_commitment": "0x5001",
        "final_kv_commitment": "0x5002",
        "n_turns": 1,
        "n_prefill_tokens": 3,
        "n_generated_tokens": 1,
        "first_step_index": 0,
        "n_steps": 1
      },
      "steps": [
        {
          "global_step_index": 0,
          "conversation_index": 0,
          "turn_index": 0,
          "token_index": 0,
          "generated_token_id": 42,
          "io_commitment": "0x6001",
          "sampling_commitment": "0x0",
          "prev_kv_commitment": "0x5001",
          "kv_commitment": "0x5002",
          "recursive_proof_hash": "0x7001"
        }
      ],
      "span_receipt_hashes": ["0x7001"],
      "actions": []
    }
  ]
}
JSON
}

run_active_smoke() {
  step "Running active Qwen3.5 recursive statement smoke"
  if [[ "$SKIP_ACTIVE_SMOKE" == "1" ]]; then
    warn "Skipping active statement smoke"
    return
  fi

  local smoke_dir="$OUT_DIR/active_statement_smoke"
  mkdir -p "$smoke_dir"
  write_active_smoke_input "$smoke_dir/input.json"

  INPUT="$smoke_dir/input.json" \
  OUT_DIR="$smoke_dir" \
    bash "$ENGINE_DIR/scripts/prove_qwen35_active_conversation_statement.sh"

  ok "active smoke artifacts=$smoke_dir"
}

run_qwen_smoke() {
  if [[ "$RUN_QWEN_SMOKE" != "1" ]]; then
    return
  fi

  step "Running small Qwen3.5 model-path smoke"
  local args=(--model-dir "$MODEL_DIR" --layers "$LAYERS" --decode-steps "$DECODE_STEPS" --prefill-len "$PREFILL_LEN" --out-dir "$OUT_DIR/qwen_smoke")
  if [[ "$RUN_RECURSIVE" == "1" ]]; then
    args+=(--recursive)
  fi
  bash "$ENGINE_DIR/scripts/run_h100_qwen35b_full.sh" "${args[@]}"
}

summary() {
  step "Setup complete"
  echo "  repo:       $INSTALL_DIR"
  echo "  engine:     $ENGINE_DIR"
  echo "  model:      $MODEL_DIR"
  echo "  artifacts:  $OUT_DIR"
  echo ""
  echo "Next production run:"
  echo "  bash $ENGINE_DIR/scripts/run_h100_qwen35b_full.sh --model-dir '$MODEL_DIR' --layers all"
}

echo "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Bitsage / ObelyZK H100 Qwen3.5-35B Setup           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "${NC}"
echo "model:     $MODEL ($HF_REPO)"
echo "branch:    $BRANCH"
echo "out_dir:   $OUT_DIR"

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/setup.log") 2>&1

install_system_deps
install_rust_and_scarb
verify_gpu_cuda
prepare_repo
build_stack
download_model
inspect_model
run_active_smoke
run_qwen_smoke
summary
