#!/usr/bin/env bash
set -euo pipefail

# Build and prove a canonical conversation/action statement with STWO Cairo.
#
# Usage:
#   INPUT=conversation_statement.json OUT_DIR=target/conversation-proof \
#     ./engine/scripts/prove_conversation_statement.sh
#
# Outputs:
#   $OUT_DIR/conversation_statement.artifact.json
#   $OUT_DIR/conversation_statement.args.json
#   $OUT_DIR/conversation_statement.proof.json. With the default proof format
#   this is a JSON array of felt252 strings suitable for on-chain submission.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT="${INPUT:-}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/target/conversation-proof}"
# On-chain submission expects Cairo's flattened Serde felts. Use
# PROOF_FORMAT=json only for local Rust-side debugging.
PROOF_FORMAT="${PROOF_FORMAT:-cairo-serde}"

if [[ -z "$INPUT" ]]; then
  echo "Missing INPUT=/path/to/conversation_statement.json" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

ARTIFACT="$OUT_DIR/conversation_statement.artifact.json"
ARGS="$OUT_DIR/conversation_statement.args.json"
PROOF="$OUT_DIR/conversation_statement.proof.json"
EXECUTABLE="$ROOT_DIR/conversation-statement-verifier/target/dev/conversation_statement_verifier.executable.json"

echo "[1/4] Building canonical conversation statement artifact"
cargo run \
  --manifest-path "$ROOT_DIR/engine/Cargo.toml" \
  --bin prove-model \
  --features cli \
  -- statement \
  --input "$INPUT" \
  --output "$ARTIFACT" \
  --args-output "$ARGS"

echo "[2/4] Compiling Cairo conversation statement verifier"
scarb --manifest-path "$ROOT_DIR/conversation-statement-verifier/Scarb.toml" build

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
