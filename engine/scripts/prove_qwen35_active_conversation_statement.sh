#!/usr/bin/env bash
set -euo pipefail

# Build and prove an active Qwen3.5 conversation statement sidecar with STWO Cairo.
#
# Usage:
#   INPUT=qwen35_active_statement.json OUT_DIR=target/qwen35-active-proof \
#     ./engine/scripts/prove_qwen35_active_conversation_statement.sh
#
# Outputs:
#   $OUT_DIR/qwen35_active_statement.artifact.json
#   $OUT_DIR/qwen35_active_statement.args.json
#   $OUT_DIR/qwen35_active_statement.proof.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT="${INPUT:-}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/target/qwen35-active-proof}"
PROOF_FORMAT="${PROOF_FORMAT:-cairo-serde}"

if [[ -z "$INPUT" ]]; then
  echo "Missing INPUT=/path/to/qwen35_active_statement.json" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

ARTIFACT="$OUT_DIR/qwen35_active_statement.artifact.json"
ARGS="$OUT_DIR/qwen35_active_statement.args.json"
PROOF="$OUT_DIR/qwen35_active_statement.proof.json"
EXECUTABLE="$ROOT_DIR/qwen35-active-statement-verifier/target/dev/qwen35_active_statement_verifier.executable.json"

echo "[1/4] Building active Qwen3.5 statement artifact"
cargo run \
  --manifest-path "$ROOT_DIR/engine/Cargo.toml" \
  --bin prove-model \
  --features cli,serde \
  -- statement \
  --qwen35-active \
  --input "$INPUT" \
  --output "$ARTIFACT" \
  --args-output "$ARGS"

echo "[2/4] Compiling Cairo active Qwen3.5 statement verifier"
(cd "$ROOT_DIR/qwen35-active-statement-verifier" && scarb build)

echo "[3/4] Proving Cairo verifier execution with 160-bit Poseidon252 STARK"
cargo run \
  --manifest-path "$ROOT_DIR/stark-cairo/cairo-prove/Cargo.toml" \
  -- prove "$EXECUTABLE" "$PROOF" \
  --proof-format "$PROOF_FORMAT" \
  --recursive-160 \
  --arguments-file "$ARGS"

STATEMENT_HASH="$(
  node -e "const fs=require('fs'); const a=JSON.parse(fs.readFileSync(process.argv[1],'utf8')); console.log(a.statement_hash)" "$ARTIFACT"
)"

echo "[4/4] Done"
echo "artifact=$ARTIFACT"
echo "args=$ARGS"
echo "proof=$PROOF"
echo "statement_hash=$STATEMENT_HASH"
echo "expected_cairo_output_hash=$STATEMENT_HASH"
