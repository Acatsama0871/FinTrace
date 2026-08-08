"""
Select All GOOD Trajectories
==============================
Extracts all GOOD-labeled entries from the two GPT-5.4 trajectory files,
computes task_type_bucket, difficulty_score, difficulty_tier, traj_len_bin,
and outputs in the same format as testset/selected_800.json.

Output: testset/selected_all_good.json
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAJ_DIR = Path("generate_trajectories_2nd_round")
LABEL_DIR = Path("generate_trajectory_label_2nd_round")
OUTPUT_DIR = Path("testset")
OUTPUT_PATH = OUTPUT_DIR / "selected_all_good2.json"

TRAJ_FILES = {
    "finmcp": TRAJ_DIR / "finmcp_queries_gpt-5-mini_trajectories.json",
    "self_generate": TRAJ_DIR / "self_generate_queries_gpt-5-mini_trajectories.json",
}
LABEL_FILES = {
    "finmcp": LABEL_DIR / "judged_finmcp_queries_gpt-5-mini_trajectories_auto_updated.jsonl",
    "self_generate": LABEL_DIR / "judged_self_generate_queries_gpt-5-mini_trajectories_auto_updated.jsonl",
}

# 34 task types → 12 buckets (same as testset_selection_v3.py)
TASK_TYPE_BUCKETS = {
    "Reasoning QA": "Reasoning_QA",
    "Political Trading": "Trading_Market",
    "Insider Trading Analysis": "Trading_Market",
    "Technical Analysis & Backtesting": "Trading_Market",
    "Historical Price & Market Data": "Trading_Market",
    "Valuation & DCF": "Valuation_Modeling",
    "Financial Statement Analysis": "Valuation_Modeling",
    "Financial Modeling": "Valuation_Modeling",
    "Recurrent Forecasting": "Valuation_Modeling",
    "Crowdfunding & IPO": "Corporate_Events",
    "Equity Offering & Corporate Action": "Corporate_Events",
    "Earnings Analysis": "Corporate_Events",
    "Cryptocurrency": "Alt_Assets_Forex",
    "Forex & Commodities": "Alt_Assets_Forex",
    "Index & Benchmark": "Sector_Index_ETF",
    "Sector & Industry Analysis": "Sector_Index_ETF",
    "ETF & Fund Analysis": "Sector_Index_ETF",
    "Peer Comparison": "Comparison_Screening",
    "Dividend & Shareholder Return": "Comparison_Screening",
    "Screening & Ranking": "Comparison_Screening",
    "ESG Analysis": "ESG_Sustainability",
    "Analyst Ratings & Estimates": "Analyst_Research",
    "Transcript & Press Release Analysis": "Analyst_Research",
    "Macro & Economic": "Macro_Economic",
    "Company & Symbol Lookup": "Company_Lookup",
    "Time-Sensitive_Data_Fetching(Global)": "Company_Lookup",
    "Qualitative Retrieval": "FinBench_Specialized",
    "Quantitative Retrieval": "FinBench_Specialized",
    "Numerical Reasoning": "FinBench_Specialized",
    "Beat or Miss": "FinBench_Specialized",
    "Adjustments": "FinBench_Specialized",
    "Market Analysis": "FinBench_Specialized",
    "Trends": "FinBench_Specialized",
    "Complex Retrieval": "FinBench_Specialized",
}


# ---------------------------------------------------------------------------
# 1. Load and Merge
# ---------------------------------------------------------------------------
def load_and_merge() -> list[dict]:
    all_entries = []
    for source_key, path in TRAJ_FILES.items():
        print(f"  Loading {path.name}...")
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            e["_source"] = source_key
        all_entries.extend(entries)
        print(f"    {len(entries)} entries")

    # Load labels
    labels = {}
    for source_key, path in LABEL_FILES.items():
        print(f"  Loading {path.name}...")
        count = 0
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                labels[rec["trajectory_id"]] = rec
                count += 1
        print(f"    {count} labels")

    # Join: keep only GOOD
    good = []
    for e in all_entries:
        label_rec = labels.get(e["id"])
        if label_rec is None:
            continue
        if label_rec["label"] != "GOOD":
            continue
        e["_label"] = "GOOD"
        e["_bad_reasons"] = []
        e["_reason_summary"] = ""
        good.append(e)

    print(f"  GOOD entries after join: {len(good)}")
    return good


# ---------------------------------------------------------------------------
# 2. Difficulty Scoring (same as testset_selection_v3.py)
# ---------------------------------------------------------------------------
def compute_difficulty_score(entry: dict) -> float:
    score = 0.0
    query = entry.get("source_query", "")
    trajectory = entry.get("trajectory", [])
    q_lower = query.lower()

    num_turns = len(trajectory)
    if num_turns >= 6:
        score += 4.0
    elif num_turns >= 4:
        score += 2.5
    elif num_turns >= 2:
        score += 1.0

    endpoints = entry.get("endpoints_called", [])
    num_endpoints = len(set(endpoints))
    score += max(0, num_endpoints - 1) * 2.0

    num_questions = query.count("?")
    if num_questions >= 3:
        score += 2.0
    elif num_questions >= 2:
        score += 1.0

    if len(query) > 400:
        score += 2.0
    elif len(query) > 200:
        score += 1.0

    hard_kw = [
        "correlat", "regression", "decompos", "backtest", "sensitivity",
        "optimize", "compare across", "weighted average", "rolling",
        "volatility", "sharpe", "drawdown", "attribution",
    ]
    medium_kw = [
        "percentage change", "growth rate", "ratio", "margin",
        "year-over-year", "compare", "rank", "trend", "average",
    ]
    if any(kw in q_lower for kw in hard_kw):
        score += 2.0
    elif any(kw in q_lower for kw in medium_kw):
        score += 1.0

    entities = re.findall(r"\b[A-Z]{2,5}\b", query)
    unique_entities = set(entities)
    if len(unique_entities) >= 3:
        score += 1.5
    elif len(unique_entities) >= 2:
        score += 0.5

    years = set(re.findall(r"20[0-2]\d|19\d\d", query))
    if len(years) >= 3:
        score += 1.0

    return score


def assign_difficulty_tiers(entries: list[dict]) -> None:
    scores = [e["_diff_score"] for e in entries]
    p33 = np.percentile(scores, 33)
    p66 = np.percentile(scores, 66)
    print(f"  Difficulty percentiles: p33={p33:.1f}, p66={p66:.1f}")
    for e in entries:
        if e["_diff_score"] <= p33:
            e["_diff_tier"] = "easy"
        elif e["_diff_score"] <= p66:
            e["_diff_tier"] = "medium"
        else:
            e["_diff_tier"] = "hard"


# ---------------------------------------------------------------------------
# 3. Strata Assignment
# ---------------------------------------------------------------------------
def traj_length_bin(entry: dict) -> str:
    num_turns = len(entry.get("trajectory", []))
    if num_turns <= 2:
        return "short"
    elif num_turns <= 4:
        return "medium"
    else:
        return "long"


def assign_strata(entries: list[dict]) -> None:
    for e in entries:
        tt = e.get("task_type", "")
        e["_task_bucket"] = TASK_TYPE_BUCKETS.get(tt, "FinBench_Specialized")
        e["_traj_bin"] = traj_length_bin(e)


# ---------------------------------------------------------------------------
# 4. Output Formatting (same as testset_selection_v3.py)
# ---------------------------------------------------------------------------
def format_output(entry: dict) -> dict:
    return {
        "id": entry.get("id"),
        "source_query": entry.get("source_query", ""),
        "task_type": entry.get("task_type", ""),
        "task_type_bucket": entry.get("_task_bucket", ""),
        "resource": entry.get("resource", ""),
        "data_source": entry.get("_source", ""),
        "label": "GOOD",
        "bad_reasons": [],
        "reason_summary": "",
        "difficulty_tier": entry.get("_diff_tier", ""),
        "difficulty_score": round(entry.get("_diff_score", 0), 2),
        "num_turns": len(entry.get("trajectory", [])),
        "traj_len_bin": entry.get("_traj_bin", ""),
        "endpoints_called": entry.get("endpoints_called", []),
        "reference_answer": entry.get("reference_answer", ""),
        "final_answer": entry.get("final_answer", ""),
        "reasoning": entry.get("reasoning", ""),
        "trajectory": entry.get("trajectory", []),
        "messages": entry.get("messages", []),
    }


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
def print_summary(selected: list[dict]):
    print("\n" + "=" * 60)
    print("SELECTION SUMMARY — ALL GOOD")
    print("=" * 60)
    print(f"Total: {len(selected)}")

    by_source = Counter(e["data_source"] for e in selected)
    print(f"\nBy data_source:")
    for s, c in by_source.most_common():
        print(f"  {s}: {c} ({c/len(selected)*100:.1f}%)")

    by_bucket = Counter(e["task_type_bucket"] for e in selected)
    print(f"\nBy task_type_bucket:")
    for b, c in by_bucket.most_common():
        print(f"  {b:30s}: {c:>5} ({c/len(selected)*100:.1f}%)")

    by_tt = Counter(e["task_type"] for e in selected)
    print(f"\nBy task_type (top 20):")
    for t, c in by_tt.most_common(20):
        print(f"  {t:40s}: {c:>5}")

    by_bin = Counter(e["traj_len_bin"] for e in selected)
    print(f"\nBy traj_len_bin:")
    for b in ["short", "medium", "long"]:
        print(f"  {b}: {by_bin.get(b, 0)} ({by_bin.get(b, 0)/len(selected)*100:.1f}%)")

    by_diff = Counter(e["difficulty_tier"] for e in selected)
    print(f"\nBy difficulty_tier:")
    for d in ["easy", "medium", "hard"]:
        print(f"  {d}: {by_diff.get(d, 0)} ({by_diff.get(d, 0)/len(selected)*100:.1f}%)")

    turns = [e["num_turns"] for e in selected]
    print(f"\nTrajectory turns: min={min(turns)}, median={sorted(turns)[len(turns)//2]}, max={max(turns)}")

    diffs = [e["difficulty_score"] for e in selected]
    print(f"Difficulty score:  min={min(diffs):.1f}, median={sorted(diffs)[len(diffs)//2]:.1f}, max={max(diffs):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Select All GOOD Trajectories")
    print("=" * 60)

    print("\n[1] Loading and merging data...")
    entries = load_and_merge()

    print("\n[2] Scoring difficulty...")
    for e in entries:
        e["_diff_score"] = compute_difficulty_score(e)
    assign_difficulty_tiers(entries)

    print("\n[3] Assigning strata...")
    assign_strata(entries)

    print("\n[4] Formatting output...")
    output = [format_output(e) for e in entries]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print_summary(output)
    print(f"\nSaved {len(output)} entries to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
