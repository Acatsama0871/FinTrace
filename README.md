# FinTrace

Benchmark and post-training pipeline for LLM tool-calling (function-calling) on
financial queries, built on the Financial Modeling Prep (FMP) MCP server.

FinTrace contains 800 expert-curated multi-turn tool-calling trajectories across
30+ financial task categories, evaluated with a nine-metric rubric spanning
action correctness, execution efficiency, process quality, and output quality.
**FinTrace-Training** extends it with the first trajectory-level preference
dataset for financial tool-calling (SFT + DPO on Qwen3-8B/32B).

- Paper: *FinTrace* (COLM 2026)
- Dataset: [FinTrace](https://huggingface.co/datasets/YupengCao/FinTrace) on Hugging Face

## Setup

```bash
uv sync                    # Python >= 3.12, dependencies via pyproject.toml
cp .env.example .env       # set FMP_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, ...
```

The evaluation set (`evaluation/testset.json`) can be downloaded from the
Hugging Face dataset and placed at `data/testset/testset.json`.

## Repository structure

```
├── gen_eval.py                 # Core library: LLM clients, MCP session, agentic loop
├── testset_selection_v3.py     # Selects the 800-query test set from the corpus
├── benchmark/                  # Benchmark construction (800-query test set)
│   ├── generate_traj.py        #   generate trajectories (LiteLLM: GPT/Claude/Gemini/...)
│   ├── generate_openai_compatible.py  # generate via OpenAI-compatible endpoints
│   └── pickup_trajectories.py  #   LLM-as-judge best-of-3 → golden trajectories
├── rubrics/                    # Nine-metric rubric evaluation
│   ├── judge.py                #   judge benchmark trajectories vs golden
│   └── judge_rollouts.py       #   judge trained-model rollouts (prompt/chosen/rejected)
├── scripts/                    # Batch runners and analysis helpers
├── plotting/                   # Paper figure sources (matplotlib)
├── human_validation/           # Human-annotator validation of the LLM judge
├── data/                       # All datasets (git-ignored)
│   ├── corpus/                 #   source queries + corpus trajectories/labels
│   ├── testset/                #   800-query test set + golden trajectories
│   ├── dpo/                    #   DPO preference data (rollout + good/bad split)
│   └── trajectory/             #   model trajectories (benchmark 13-LLM + trained rollouts)
└── results/                    # Rubric evaluation outputs (git-ignored)
```

## Pipeline

**1. Generate benchmark trajectories** (13 LLMs on the 800-query test set):

```bash
python benchmark/generate_traj.py --model claude-opus-4-6
python benchmark/generate_openai_compatible.py generate --model deepseek-reasoner --api-base https://api.deepseek.com
```

**2. Select golden trajectories** (LLM judge picks best-of-3 frontier runs):

```bash
python benchmark/pickup_trajectories.py --judge-model claude-opus-4-6
```

**3. Rubric evaluation** (9 metrics: 3 algorithmic + 6 LLM-judged):

```bash
# benchmark trajectories vs golden
python rubrics/judge.py evaluate -t data/trajectory/testset_trajectory/selected_800_gpt-5.4_trajectories.json

# trained-model rollouts (base/SFT/DPO)
python rubrics/judge_rollouts.py evaluate -d data/trajectory/rollouts/test_eval_0806/dpo8b \
    -o results/trained_qwen3_8b -m claude-sonnet-4-6 --model-name dpo8b --no-batch
```

Batch runners for full sweeps live in `scripts/run_*.sh`; figures are generated
by the scripts in `plotting/` (run from the repo root).

## Citation

Citation entry coming with the camera-ready release.
