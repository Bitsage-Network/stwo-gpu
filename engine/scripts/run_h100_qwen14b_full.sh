#!/usr/bin/env bash
# Compatibility shim. The current H100 target is Qwen3.5-35B-A3B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_h100_qwen35b_full.sh" "$@"
