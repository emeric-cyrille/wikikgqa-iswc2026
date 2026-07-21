#!/usr/bin/env bash
# Run the P3 pipeline on the official test split (75 questions per language).
# Produces a submission ZIP under predictions/<lang>_mentions_test_<name>.zip.
# Requires the environment variable GROQ_API_KEY.
#
#   bash scripts/run_test.sh en llama-3.3-70b-versatile
#   bash scripts/run_test.sh es openai/gpt-oss-120b
set -euo pipefail

LANG="${1:-en}"
MODEL="${2:-openai/gpt-oss-120b}"
NAME="${3:-p3}"

python -m src.pipeline \
  --lang "$LANG" \
  --split test \
  --model "$MODEL" \
  --name "$NAME" \
  --k 3 \
  --n-candidates 5 \
  --pace 6.0 \
  --resume
