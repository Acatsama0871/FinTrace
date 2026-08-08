#!/usr/bin/env bash
# Evaluate the 0806 test batch (dpo8b / base8b / sft8b) with the Sonnet judge
# using the normal per-request API (--no-batch).
# Sequential across models (avoids rate-limit-guard contention between processes).
# Output: results/trained_qwen3_8b/<model>/
set -u
cd "$(dirname "$0")/.."

JUDGE="claude-sonnet-4-6"
OUT="results/trained_qwen3_8b"

for m in dpo8b base8b sft8b; do
  echo "============================================================"
  echo "[RUN] $m  ($(date '+%H:%M:%S'))"
  echo "============================================================"
  uv run python rubrics/judge_rollouts.py evaluate \
    -d "data/trajectory/rollouts/test_eval_0806/$m" \
    -o "$OUT" \
    -m "$JUDGE" \
    --model-name "$m" \
    --no-batch
  echo "[DONE] $m  ($(date '+%H:%M:%S'))"
done

echo "[ALL DONE] $(date '+%H:%M:%S')"