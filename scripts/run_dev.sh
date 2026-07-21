#!/usr/bin/env bash
# Run the P3 pipeline on the development split (100 questions per language).
# Requires the environment variable GROQ_API_KEY.
#
#   bash scripts/run_dev.sh en                       # gpt-oss-120b, EN dev
#   bash scripts/run_dev.sh es llama-3.3-70b-versatile
set -euo pipefail

LANG="${1:-en}"
MODEL="${2:-openai/gpt-oss-120b}"
NAME="${3:-p3}"

python -m src.pipeline \
  --lang "$LANG" \
  --split dev \
  --model "$MODEL" \
  --name "$NAME" \
  --k 3 \
  --n-candidates 5 \
  --pace 6.0
