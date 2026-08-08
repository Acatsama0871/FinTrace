#!/usr/bin/env bash
# Full rubrics evaluation with DeepSeek-V4-Pro as judge across all 13 trajectory files.
# Sequential across models (avoids rate-limit-guard contention between processes).
set -u

OUT_DIR="results/benchmark_by_deepseek_judge"
JUDGE="deepseek-v4-pro"
CONCURRENCY=8
LOG_DIR="eval_logs_deepseek_v4_pro"
mkdir -p "$LOG_DIR"

TRAJ_FILES=(
  selected_800_claude-opus-4-6_trajectories.json
  selected_800_claude-sonnet-4-6_trajectories.json
  selected_800_gpt-5-mini_trajectories.json
  selected_800_gemini_gemini-3.1-pro-preview_trajectories.json
  selected_800_gemini_gemini-3-flash-preview_trajectories.json
  selected_800_deepseek-reasoner_trajectories.json
  selected_800_qwen_qwen3.5-27b_trajectories.json
  selected_800_together_ai_Qwen_Qwen3.5-397B-A17B_trajectories.json
  selected_800_together_ai_Qwen_Qwen3.5-9B_trajectories.json
  selected_800_together_ai_meta-llama_Llama-3.3-70B-Instruct-Turbo_trajectories.json
  selected_800_together_ai_moonshotai_Kimi-K2.5_trajectories.json
  selected_800_together_ai_openai_gpt-oss-120b_trajectories.json
)

START=$(date +%s)
for f in "${TRAJ_FILES[@]}"; do
  echo ""
  echo "================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $f"
  echo "================================================================"
  log_name=$(basename "$f" .json).log
  uv run python rubrics/judge.py evaluate \
    -t "data/trajectory/testset_trajectory/$f" \
    -g "data/testset/golden_label_trajectories.json" \
    --judge-model "$JUDGE" \
    --output-dir "$OUT_DIR" \
    --concurrency "$CONCURRENCY" \
    2>&1 | tee "$LOG_DIR/$log_name"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $f"
done

echo ""
echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 13 models done. Aggregating..."
echo "================================================================"
uv run python rubrics/judge.py aggregate -r "$OUT_DIR" 2>&1 | tee "$LOG_DIR/aggregate.log"

END=$(date +%s)
echo ""
echo "Total elapsed: $((END - START)) seconds"
