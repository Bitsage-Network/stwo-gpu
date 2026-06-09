#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Download Model
#
# Download any supported model for verifiable inference.
#
# Usage:
#   ./scripts/download_model.sh qwen3.5-35b-a3b
#   ./scripts/download_model.sh qwen3.5-35b-a3b-fp8
#   ./scripts/download_model.sh qwen2.5-14b
#   ./scripts/download_model.sh deepseek-v4-flash
#   ./scripts/download_model.sh glm-4-9b
#   ./scripts/download_model.sh minimax-m2.7
#   ./scripts/download_model.sh kimi-k2.6
#   ./scripts/download_model.sh smollm2-135m
#   ./scripts/download_model.sh llama-3.1-8b
#   ./scripts/download_model.sh mistral-7b
#   ./scripts/download_model.sh mixtral-8x7b
#   ./scripts/download_model.sh phi-3-mini
#   ./scripts/download_model.sh gemma-2b
#
# Models are downloaded to ~/.obelyzk/models/<model-name>/
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL_DIR="${OBELYZK_MODEL_DIR:-$HOME/.obelyzk/models}"
MODEL="${1:-}"

if [ -z "$MODEL" ]; then
    echo "ObelyZK Model Downloader"
    echo ""
    echo "Usage: $0 <model-name>"
    echo ""
    echo "Supported model download targets:"
    echo ""
    echo "  Model               Params   Architecture         Size"
    echo "  ─────────────────── ──────── ──────────────────── ─────"
    echo "  qwen3.5-35b-a3b     35B/3B   Qwen3.5 MoE          ~70 GB"
    echo "  qwen3.5-35b-a3b-fp8 35B/3B   Qwen3.5 MoE FP8      ~35 GB"
    echo "  qwen2.5-14b         14B      Qwen2 (GQA)          30 GB  legacy verified target"
    echo "  qwen2.5-7b          7B       Qwen2 (GQA)          15 GB"
    echo "  deepseek-v4-flash   284B/13B DeepSeek V4 MoE      large"
    echo "  deepseek-v4-pro     1.6T/49B DeepSeek V4 MoE      very large"
    echo "  glm-4-9b            9B       ChatGLM (fused QKV)   18 GB"
    echo "  minimax-m2.7        229B MoE MiniMax              very large"
    echo "  kimi-k2.6           1T/32B   Kimi MoE + MLA        very large"
    echo "  llama-3.1-8b        8B       LLaMA (GQA)          16 GB"
    echo "  mistral-7b          7B       Mistral (GQA+SWA)     15 GB"
    echo "  mixtral-8x7b        47B MoE  Mixtral (8 experts)   87 GB"
    echo "  phi-3-mini          3.8B     Phi (fused QKV)        8 GB"
    echo "  gemma-2b            2B       Gemma                  5 GB"
    echo "  smollm2-135m        135M     SmolLM                 0.3 GB  ← fastest for testing"
    echo ""
    echo "Models are saved to: $MODEL_DIR/<model-name>/"
    echo ""
    echo "After downloading, prove with:"
    echo "  echo 'Hello' | OBELYSK_MODEL_DIR=$MODEL_DIR/<model-name> obelyzk chat --model local"
    exit 1
fi

HF_CLI=()

ensure_hf_cli() {
    if command -v hf &>/dev/null; then
        HF_CLI=(hf download)
        return
    fi
    if command -v huggingface-cli &>/dev/null; then
        HF_CLI=(huggingface-cli download)
        return
    fi

    local venv="${OBELYZK_HF_VENV:-$HOME/.obelyzk/hf-venv}"
    if [[ ! -x "$venv/bin/hf" ]]; then
        echo "Installing huggingface_hub into $venv..."
        python3 -m venv "$venv"
        "$venv/bin/python" -m pip install -q --upgrade pip
        "$venv/bin/python" -m pip install -q huggingface_hub
    fi
    HF_CLI=("$venv/bin/hf" download)
}

ensure_hf_cli

download_hf() {
    local repo="$1"
    local dest="$2"
    echo "Downloading $repo → $dest"
    echo "This may take a while for large models..."
    "${HF_CLI[@]}" "$repo" --local-dir "$dest"
    echo "Done: $(du -sh "$dest" | cut -f1) in $dest"
}

download_small() {
    local repo="$1"
    local dest="$2"
    mkdir -p "$dest"
    echo "Downloading $repo → $dest"
    for f in config.json tokenizer.json tokenizer_config.json; do
        curl -sL "https://huggingface.co/$repo/resolve/main/$f" -o "$dest/$f" 2>/dev/null || true
    done
    # SafeTensors — might be single file or sharded
    if curl -sIf "https://huggingface.co/$repo/resolve/main/model.safetensors" >/dev/null 2>&1; then
        curl -L "https://huggingface.co/$repo/resolve/main/model.safetensors" -o "$dest/model.safetensors"
    else
        "${HF_CLI[@]}" "$repo" --local-dir "$dest"
    fi
    echo "Done: $(du -sh "$dest" | cut -f1) in $dest"
}

DEST="$MODEL_DIR/$MODEL"

case "$MODEL" in
    qwen3.5-35b-a3b|qwen35b|qwen-35b)
        download_hf "Qwen/Qwen3.5-35B-A3B" "$DEST"
        ;;
    qwen3.5-35b-a3b-fp8|qwen35b-fp8|qwen-35b-fp8)
        download_hf "Qwen/Qwen3.5-35B-A3B-FP8" "$DEST"
        ;;
    qwen3.5-35b-a3b-gptq-int4|qwen35b-gptq-int4|qwen-35b-gptq-int4)
        download_hf "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" "$DEST"
        ;;
    qwen2.5-14b|qwen-14b)
        download_hf "Qwen/Qwen2.5-14B" "$DEST"
        ;;
    qwen2.5-7b|qwen-7b)
        download_hf "Qwen/Qwen2.5-7B" "$DEST"
        ;;
    deepseek-v4-flash|deepseek-flash)
        echo "WARNING: DeepSeek V4 Flash is much larger than the Qwen3.5 first target."
        download_hf "deepseek-ai/DeepSeek-V4-Flash" "$DEST"
        ;;
    deepseek-v4-pro|deepseek-pro)
        echo "WARNING: DeepSeek V4 Pro is a frontier-scale MoE model. Confirm storage/GPU capacity first."
        download_hf "deepseek-ai/DeepSeek-V4-Pro" "$DEST"
        ;;
    glm-4-9b|glm4|chatglm)
        download_hf "THUDM/glm-4-9b" "$DEST"
        ;;
    minimax-m2.7|minimax)
        echo "WARNING: MiniMax-M2.7 is a very large MoE model."
        echo "         FP8 dequantization is supported in ObelyZK v0.4.0+."
        echo ""
        read -p "Continue? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 0
        download_hf "MiniMaxAI/MiniMax-M2.7" "$DEST"
        ;;
    minimax-m2.5)
        echo "WARNING: MiniMax-M2.5 is a legacy frontier-scale MoE target."
        read -p "Continue? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 0
        download_hf "MiniMaxAI/MiniMax-M2.5" "$DEST"
        ;;
    kimi-k2.6|kimi)
        echo "WARNING: Kimi-K2.6 is frontier-scale and uses MLA attention."
        echo "         MLA attention path is in development."
        echo ""
        read -p "Continue? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 0
        download_hf "moonshotai/Kimi-K2.6" "$DEST"
        ;;
    kimi-k2.5)
        echo "WARNING: Kimi-K2.5 is a legacy frontier-scale MoE target with MLA attention."
        read -p "Continue? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 0
        download_hf "moonshotai/Kimi-K2.5" "$DEST"
        ;;
    llama-3.1-8b|llama-8b|llama)
        download_hf "meta-llama/Llama-3.1-8B" "$DEST"
        ;;
    mistral-7b|mistral)
        download_hf "mistralai/Mistral-7B-v0.3" "$DEST"
        ;;
    mixtral-8x7b|mixtral)
        download_hf "mistralai/Mixtral-8x7B-v0.1" "$DEST"
        ;;
    phi-3-mini|phi3|phi)
        download_hf "microsoft/Phi-3-mini-4k-instruct" "$DEST"
        ;;
    gemma-2b|gemma)
        download_hf "google/gemma-2b" "$DEST"
        ;;
    smollm2-135m|smollm|smol)
        download_small "HuggingFaceTB/SmolLM2-135M" "$DEST"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Run '$0' without arguments to see supported models."
        exit 1
        ;;
esac

echo ""
echo "To prove with this model:"
echo "  echo 'Hello' | OBELYSK_MODEL_DIR=$DEST obelyzk chat --model local"
