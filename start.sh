#!/bin/bash
set -e

# Default settings
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/Users/wennuan/.lmstudio/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ/}"

echo "=========================================================="
echo " Starting Qwen3 Local Embedding Service (MLX)"
echo " Model: $MODEL_PATH"
echo " Listening on: http://$HOST:$PORT"
echo " Google AI Endpoint: http://$HOST:$PORT/v1beta/models/qwen3-embedding-4b:embedContent"
echo " OpenAI Endpoint:    http://$HOST:$PORT/v1/embeddings"
echo "=========================================================="

export HOST
export PORT
export MODEL_PATH

exec uvicorn app.main:app --host "$HOST" --port "$PORT"
