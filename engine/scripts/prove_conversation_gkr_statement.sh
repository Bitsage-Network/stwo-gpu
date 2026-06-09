#!/usr/bin/env bash
set -euo pipefail

# Prove strict full-GKR conversation verifier execution with STWO Cairo.
#
# Typical decode output usage:
#   ARTIFACT=target/decode/qwen.conversation_gkr.artifact.json \
#     ./engine/scripts/prove_conversation_gkr_statement.sh
#
# Direct args usage:
#   ARGS=target/decode/qwen.conversation_gkr.args.json \
#     ./engine/scripts/prove_conversation_gkr_statement.sh
#
# Outputs:
#   $PROOF, defaulting next to the args file as
#   <basename>.conversation_gkr.proof.json. With the default proof format this
#   is a JSON array of felt252 strings suitable for submit_conversation_stwo.mjs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT="${ARTIFACT:-}"
ARGS="${ARGS:-}"
PROOF="${PROOF:-}"
# On-chain submission expects Cairo's flattened Serde felts. Use
# PROOF_FORMAT=json only for local Rust-side debugging.
PROOF_FORMAT="${PROOF_FORMAT:-cairo-serde}"

if [[ -z "$ARGS" && -n "$ARTIFACT" ]]; then
  ARGS="$(
    node -e "
      const fs = require('fs');
      const path = require('path');
      const artifact = process.argv[1];
      const json = JSON.parse(fs.readFileSync(artifact, 'utf8'));
      if (!json.cairo_args_path) {
        throw new Error('artifact is missing cairo_args_path');
      }
      console.log(path.resolve(path.dirname(artifact), json.cairo_args_path));
    " "$ARTIFACT"
  )"
fi

if [[ -z "$ARGS" ]]; then
  echo "Missing ARGS=/path/to/*.conversation_gkr.args.json or ARTIFACT=/path/to/*.conversation_gkr.artifact.json" >&2
  exit 1
fi

if [[ ! -f "$ARGS" ]]; then
  echo "Args file not found: $ARGS" >&2
  exit 1
fi

if [[ -n "$ARTIFACT" && ! -f "$ARTIFACT" ]]; then
  echo "Artifact file not found: $ARTIFACT" >&2
  exit 1
fi

ARGS_DIR="$(cd "$(dirname "$ARGS")" && pwd)"
ARGS_BASE="$(basename "$ARGS")"
DEFAULT_PROOF_BASE="${ARGS_BASE%.args.json}.proof.json"
PROOF="${PROOF:-$ARGS_DIR/$DEFAULT_PROOF_BASE}"
EXECUTABLE="$ROOT_DIR/conversation-gkr-statement-verifier/target/dev/conversation_gkr_statement_verifier.executable.json"

echo "[1/3] Compiling Cairo strict GKR conversation verifier"
scarb --manifest-path "$ROOT_DIR/conversation-gkr-statement-verifier/Scarb.toml" build

echo "[2/3] Proving verifier execution with 160-bit Poseidon252 STARK"
cargo run \
  --manifest-path "$ROOT_DIR/stark-cairo/cairo-prove/Cargo.toml" \
  -- prove "$EXECUTABLE" "$PROOF" \
  --proof-format "$PROOF_FORMAT" \
  --recursive-160 \
  --arguments-file "$ARGS"

echo "[3/3] Done"
echo "args=$ARGS"
echo "proof=$PROOF"

if [[ -n "$ARTIFACT" ]]; then
  STATEMENT_HASH="$(
    node -e "const fs=require('fs'); const a=JSON.parse(fs.readFileSync(process.argv[1],'utf8')); console.log(a.statement_hash)" "$ARTIFACT"
  )"
  echo "artifact=$ARTIFACT"
  echo "statement_hash=$STATEMENT_HASH"
  echo "expected_cairo_output_hash=$STATEMENT_HASH"
fi
