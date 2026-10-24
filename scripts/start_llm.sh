#!/usr/bin/env bash
# Start the local Vienna Stay llama.cpp server. Stop with Ctrl+C.
set -euo pipefail

server=llama-server
model_repo=Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
model_file=
port=8080
context_size=16384
gpu_layers=auto
dry_run=false

usage() {
    cat <<'EOF'
Usage: bash scripts/start_llm.sh [options]
  --server PATH       llama-server executable (default: llama-server on PATH)
  --model-repo REPO   Hugging Face GGUF repository:quantization
  --model-file PATH   Use a downloaded GGUF instead of downloading from HF
  --port N            Local port (default: 8080)
  --context-size N    Context tokens, 512-32768 (default: 16384)
  --gpu-layers N      auto, all, or layer count; 0 for CPU (default: auto)
  --dry-run           Print arguments without starting or downloading anything
  --help              Show this help
EOF
}
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=true; shift; continue ;;
        --server|--model-repo|--model-file|--port|--context-size|--gpu-layers)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            case "$1" in
                --server) server=$2 ;;
                --model-repo) model_repo=$2 ;;
                --model-file) model_file=$2 ;;
                --port) port=$2 ;;
                --context-size) context_size=$2 ;;
                --gpu-layers) gpu_layers=$2 ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

# Bound digit counts before arithmetic; 10# avoids interpreting leading zeros as octal.
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
    printf 'Port must be an integer from 1 to 65535.\n' >&2; exit 2
fi
if [[ ! "$context_size" =~ ^[0-9]{1,5}$ ]] || ((10#$context_size < 512 || 10#$context_size > 32768)); then
    printf 'Context size must be an integer from 512 to 32768.\n' >&2; exit 2
fi
if [[ ! "$gpu_layers" =~ ^(auto|all|[0-9]+)$ ]]; then
    printf 'GPU layers must be auto, all, or a nonnegative integer.\n' >&2; exit 2
fi
server_args=(--host 127.0.0.1 --port "$port" --alias vienna-stay
    --ctx-size "$context_size" --parallel 1 --jinja --gpu-layers "$gpu_layers" --log-verbosity 2)
if [[ -n "$model_file" ]]; then
    if [[ ! -f "$model_file" ]]; then
        printf 'Model file not found: %s\n' "$model_file" >&2; exit 2
    fi
    # Prefix relative paths so even a filename beginning with '-' remains a value.
    [[ "$model_file" = /* ]] || model_file="$PWD/$model_file"
    server_args+=(--model "$model_file")
else
    if [[ -z "$model_repo" || "$model_repo" = -* ]]; then
        printf 'Specify a Hugging Face GGUF repository and quantization.\n' >&2; exit 2
    fi
    server_args+=(--hf-repo "$model_repo")
fi
if "$dry_run"; then
    printf '%q ' "$server" "${server_args[@]}"
    printf '\n'
    exit 0
fi
if ! command -v "$server" >/dev/null 2>&1; then
    printf 'llama-server not found. Install/build llama.cpp, then use --server /path/to/llama-server. See Docs/llama_server.md.\n' >&2
    exit 127
fi
printf 'Starting local LLM at http://127.0.0.1:%s (alias: vienna-stay).\n' "$port"
printf 'The first repository-based launch downloads model weights. Stop with Ctrl+C.\n'
exec "$server" "${server_args[@]}"
