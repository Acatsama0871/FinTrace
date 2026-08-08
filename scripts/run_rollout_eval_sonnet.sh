#!/usr/bin/env bash
# Evaluate all rollout results under data/trajectory/rollouts/seed_variance
# with the Sonnet judge (claude-sonnet-4-6), one batch per seed file.
# Output: results/rollout_5seed/<group>/<seed>/
set -uo pipefail
cd "$(dirname "$0")/.."

JUDGE="claude-sonnet-4-6"
ROOT="data/trajectory/rollouts/seed_variance"
OUT_ROOT="results/rollout_5seed"

for group in base dpo sft; do
  for seed in seed0 seed1 seed2 seed3 seed4; do
    d="$ROOT/$group/$seed"
    if [ ! -f "$d/rollout_results.json" ]; then
      echo "[SKIP] $d (no rollout_results.json)"
      continue
    fi
    echo "============================================================"
    echo "[RUN] group=$group seed=$seed  ($(date '+%H:%M:%S'))"
    echo "============================================================"
    uv run python rubrics/judge_rollouts.py evaluate \
      -d "$d" \
      -o "$OUT_ROOT/$group" \
      -m "$JUDGE"
    echo "[DONE] $group/$seed  ($(date '+%H:%M:%S'))"
  done
done

echo "============================================================"
echo "[AGGREGATE] building per-group summary CSVs"
echo "============================================================"
for group in base dpo sft; do
  uv run python rubrics/judge_rollouts.py aggregate -r "$OUT_ROOT/$group"
done
echo "[ALL DONE] $(date '+%H:%M:%S')"