"""
Testset Selection Script v3
============================
Selects 800 labeled tool-calling trajectories from data/corpus/trajectories_round1/,
using annotations from generate_trajectory_label/.

Covers:
  - 12 task-type buckets (grouped from 34 raw types)
  - 3 trajectory-length bins (short / medium / long)
  - 3 difficulty tiers (easy / medium / hard) — balanced within strata
  - Both GOOD and BAD examples (55:45 ratio)

Stratification: label × task_type_bucket × traj_length_bin
Difficulty used as within-stratum balancing criterion (not a stratification dim).

Output: testset/selected_800.json
"""

import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
TARGET_BUDGET = 800
GOOD_BUDGET = 440  # 55%
BAD_BUDGET = 360   # 45%
MAX_BUCKET_SHARE = 0.25  # cap any single bucket at 25% of total
DIFFICULTY_RATIOS = {"easy": 0.30, "medium": 0.40, "hard": 0.30}

TRAJ_DIR = Path("data/corpus/trajectories_round1")
LABEL_DIR = Path("data/corpus/labels_round1")
OUTPUT_DIR = Path("data/testset")
OUTPUT_PATH = OUTPUT_DIR / "selected_800.json"

TRAJ_FILES = {
    "finmcp": TRAJ_DIR / "finmcp_queries_gpt-5.4_trajectories.json",
    "self_generate": TRAJ_DIR / "self_generate_queries_gpt-5.4_trajectories.json",
}
LABEL_FILES = {
    "finmcp": LABEL_DIR / "judged_finmcp_queries_gpt-5.4.jsonl",
    "self_generate": LABEL_DIR / "judged_self_generate_queries_gpt-5.4_trajectories.jsonl",
}

# 34 task types → 12 buckets
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
    # FinBench specialized (small types from Finance Agent Bench)
    "Qualitative Retrieval": "FinBench_Specialized",
    "Quantitative Retrieval": "FinBench_Specialized",
    "Numerical Reasoning": "FinBench_Specialized",
    "Beat or Miss": "FinBench_Specialized",
    "Adjustments": "FinBench_Specialized",
    "Market Analysis": "FinBench_Specialized",
    "Trends": "FinBench_Specialized",
    "Complex Retrieval": "FinBench_Specialized",
}

random.seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# 1. Load and Merge
# ===========================================================================

def load_and_merge() -> list[dict]:
    """Load trajectories + labels, join, exclude errors, deduplicate."""
    # Load trajectories
    all_entries = []
    for source_key, path in TRAJ_FILES.items():
        print(f"  Loading {path.name}...")
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            e["_source"] = source_key
        all_entries.extend(entries)
        print(f"    {len(entries)} entries")

    # Load labels into lookup
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

    # Join trajectories with labels
    merged = []
    for e in all_entries:
        label_rec = labels.get(e["id"])
        if label_rec is None:
            continue
        if label_rec["label"] == "error":
            continue
        e["_label"] = label_rec["label"]
        e["_bad_reasons"] = label_rec.get("bad_reasons", [])
        e["_reason_summary"] = label_rec.get("reason_summary", "")
        e["_label_num_turns"] = label_rec.get("num_turns", len(e.get("trajectory", [])))
        merged.append(e)

    print(f"  After joining & excluding errors: {len(merged)} entries")

    # Deduplicate by query (case-insensitive), keep entry with more endpoints
    seen = {}
    for e in merged:
        key = e["source_query"].strip().lower()
        if key not in seen:
            seen[key] = e
        else:
            existing = seen[key]
            if len(e.get("endpoints_called", [])) > len(existing.get("endpoints_called", [])):
                seen[key] = e
    deduped = list(seen.values())
    print(f"  After dedup: {len(deduped)} entries")
    return deduped


# ===========================================================================
# 2. Difficulty Scoring (adapted from testset_selection_v2.py)
# ===========================================================================

def compute_difficulty_score(entry: dict) -> float:
    """Compute difficulty score. Higher = harder. Range roughly 0–15."""
    score = 0.0
    query = entry.get("source_query", "")
    trajectory = entry.get("trajectory", [])
    q_lower = query.lower()

    # Trajectory turns
    num_turns = len(trajectory)
    if num_turns >= 6:
        score += 4.0
    elif num_turns >= 4:
        score += 2.5
    elif num_turns >= 2:
        score += 1.0

    # Number of unique endpoints called
    endpoints = entry.get("endpoints_called", [])
    num_endpoints = len(set(endpoints))
    score += max(0, num_endpoints - 1) * 2.0

    # Multi-part questions
    num_questions = query.count("?")
    if num_questions >= 3:
        score += 2.0
    elif num_questions >= 2:
        score += 1.0

    # Query length
    if len(query) > 400:
        score += 2.0
    elif len(query) > 200:
        score += 1.0

    # Computation keywords
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

    # Multi-entity analysis (ticker-like patterns)
    entities = re.findall(r"\b[A-Z]{2,5}\b", query)
    unique_entities = set(entities)
    if len(unique_entities) >= 3:
        score += 1.5
    elif len(unique_entities) >= 2:
        score += 0.5

    # Time span complexity
    years = set(re.findall(r"20[0-2]\d|19\d\d", query))
    if len(years) >= 3:
        score += 1.0

    return score


def assign_difficulty_tiers(entries: list[dict]) -> None:
    """Assign difficulty tier (easy/medium/hard) via global percentile cuts."""
    scores = [e["_diff_score"] for e in entries]
    p33 = np.percentile(scores, 33)
    p66 = np.percentile(scores, 66)
    for e in entries:
        if e["_diff_score"] <= p33:
            e["_diff_tier"] = "easy"
        elif e["_diff_score"] <= p66:
            e["_diff_tier"] = "medium"
        else:
            e["_diff_tier"] = "hard"


# ===========================================================================
# 3. Strata Assignment
# ===========================================================================

def traj_length_bin(entry: dict) -> str:
    num_turns = len(entry.get("trajectory", []))
    if num_turns <= 2:
        return "short"
    elif num_turns <= 4:
        return "medium"
    else:
        return "long"


def assign_strata(entries: list[dict]) -> None:
    """Tag each entry with _task_bucket, _traj_bin."""
    for e in entries:
        tt = e.get("task_type", "")
        e["_task_bucket"] = TASK_TYPE_BUCKETS.get(tt, "FinBench_Specialized")
        e["_traj_bin"] = traj_length_bin(e)


# ===========================================================================
# 4. Budget Allocation
# ===========================================================================

def allocate_budget(
    strata: dict[tuple, list[dict]],
    budget: int,
    cap_bucket: str | None = "Reasoning_QA",
    cap_max: int | None = None,
) -> dict[tuple, int]:
    """Proportional allocation with floor=1 and optional per-bucket cap.

    strata keys are (task_bucket, traj_bin) tuples.
    """
    if cap_max is None:
        cap_max = int(TARGET_BUDGET * MAX_BUCKET_SHARE)

    # First pass: compute raw proportional allocation
    total_pool = sum(len(v) for v in strata.values())
    raw = {}
    for key, members in strata.items():
        raw[key] = budget * len(members) / total_pool if total_pool > 0 else 0

    # Apply per-bucket cap (sum across bins for the capped bucket)
    if cap_bucket:
        capped_keys = [k for k in strata if k[0] == cap_bucket]
        capped_raw_total = sum(raw[k] for k in capped_keys)
        if capped_raw_total > cap_max:
            scale = cap_max / capped_raw_total
            for k in capped_keys:
                raw[k] *= scale
            # Redistribute excess to other strata proportionally
            excess = capped_raw_total - cap_max
            other_keys = [k for k in strata if k[0] != cap_bucket]
            other_total = sum(raw[k] for k in other_keys)
            if other_total > 0:
                for k in other_keys:
                    raw[k] += excess * (raw[k] / other_total)

    # Largest-remainder method
    allocations = {k: int(v) for k, v in raw.items()}
    # Floor of 1 for non-empty strata
    for k in strata:
        if strata[k] and allocations[k] < 1:
            allocations[k] = 1

    remainders = {k: raw[k] - allocations[k] for k in raw}
    shortfall = budget - sum(allocations.values())

    if shortfall > 0:
        for k in sorted(remainders, key=lambda c: remainders[c], reverse=True):
            if shortfall <= 0:
                break
            if allocations[k] < len(strata[k]):
                allocations[k] += 1
                shortfall -= 1
    elif shortfall < 0:
        # Trim from smallest allocations (but keep ≥1)
        for k in sorted(allocations, key=lambda c: allocations[c]):
            if shortfall >= 0:
                break
            if allocations[k] > 1:
                allocations[k] -= 1
                shortfall += 1

    return allocations


# ===========================================================================
# 5. Within-Stratum Sampling
# ===========================================================================

def sample_stratum(
    entries: list[dict],
    k: int,
    is_bad: bool = False,
    global_bad_reason_counts: Counter | None = None,
) -> list[dict]:
    """Sample k entries from a stratum, balancing difficulty tiers.

    For BAD entries, prefer diversity across bad_reasons.
    """
    if k <= 0 or not entries:
        return []
    if k >= len(entries):
        return list(entries)

    # Split by difficulty tier
    by_tier = defaultdict(list)
    for e in entries:
        by_tier[e["_diff_tier"]].append(e)

    # Allocate k across tiers, capping at pool size and redistributing excess
    tier_budgets = {}
    for tier, ratio in DIFFICULTY_RATIOS.items():
        tier_budgets[tier] = max(0, round(k * ratio)) if by_tier[tier] else 0

    # Iteratively cap at pool size and redistribute until stable
    for _ in range(5):
        excess = 0
        headroom_total = 0
        for tier in tier_budgets:
            pool_size = len(by_tier[tier])
            if tier_budgets[tier] > pool_size:
                excess += tier_budgets[tier] - pool_size
                tier_budgets[tier] = pool_size
            else:
                headroom_total += pool_size - tier_budgets[tier]
        if excess == 0:
            break
        # Redistribute excess to tiers with headroom
        for tier in sorted(tier_budgets, key=lambda t: len(by_tier[t]), reverse=True):
            if excess <= 0:
                break
            headroom = len(by_tier[tier]) - tier_budgets[tier]
            if headroom > 0:
                add = min(excess, headroom)
                tier_budgets[tier] += add
                excess -= add

    # Final adjust to hit k exactly
    current = sum(tier_budgets.values())
    adj = k - current
    if adj > 0:
        for tier in sorted(tier_budgets, key=lambda t: len(by_tier[t]), reverse=True):
            if adj <= 0:
                break
            headroom = len(by_tier[tier]) - tier_budgets[tier]
            if headroom > 0:
                add = min(adj, headroom)
                tier_budgets[tier] += add
                adj -= add
    elif adj < 0:
        for tier in sorted(tier_budgets, key=lambda t: tier_budgets[t]):
            if adj >= 0:
                break
            remove = min(-adj, tier_budgets[tier])
            tier_budgets[tier] -= remove
            adj += remove

    selected = []
    for tier in ["easy", "medium", "hard"]:
        pool = by_tier[tier]
        tb = tier_budgets.get(tier, 0)
        if tb <= 0:
            continue

        random.shuffle(pool)

        if is_bad and global_bad_reason_counts is not None:
            # Greedy: prefer entries whose bad_reasons are least represented
            pool_scored = []
            for e in pool:
                reasons = e.get("_bad_reasons", [])
                if reasons:
                    rarity = sum(
                        1.0 / (global_bad_reason_counts.get(r, 0) + 1)
                        for r in reasons
                    )
                else:
                    rarity = 0.0
                pool_scored.append((rarity, e))
            pool_scored.sort(key=lambda x: x[0], reverse=True)
            tier_selected = [e for _, e in pool_scored[:tb]]
        else:
            tier_selected = pool[:tb]

        selected.extend(tier_selected)

    return selected


# ===========================================================================
# 6. Output Formatting
# ===========================================================================

def format_output(entry: dict) -> dict:
    return {
        "id": entry.get("id"),
        "source_query": entry.get("source_query", ""),
        "task_type": entry.get("task_type", ""),
        "task_type_bucket": entry.get("_task_bucket", ""),
        "resource": entry.get("resource", ""),
        "data_source": entry.get("_source", ""),
        "label": entry.get("_label", ""),
        "bad_reasons": entry.get("_bad_reasons", []),
        "reason_summary": entry.get("_reason_summary", ""),
        "difficulty_tier": entry.get("_diff_tier", ""),
        "difficulty_score": round(entry.get("_diff_score", 0), 2),
        "num_turns": len(entry.get("trajectory", [])),
        "traj_len_bin": entry.get("_traj_bin", ""),
        "endpoints_called": entry.get("endpoints_called", []),
        "reference_answer": entry.get("reference_answer", ""),
        "final_answer": entry.get("final_answer", ""),
        "reasoning": entry.get("reasoning", ""),
        "trajectory": entry.get("trajectory", []),
    }


# ===========================================================================
# 7. Summary
# ===========================================================================

def print_summary(selected: list[dict]):
    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"Total selected: {len(selected)}")

    # By label
    by_label = Counter(e["label"] for e in selected)
    print(f"\nBy label:")
    for l in ["GOOD", "BAD"]:
        print(f"  {l}: {by_label.get(l, 0)} ({by_label.get(l, 0)/len(selected)*100:.1f}%)")

    # By task_type_bucket
    by_bucket = Counter(e["task_type_bucket"] for e in selected)
    print(f"\nBy task_type_bucket:")
    for b, c in by_bucket.most_common():
        print(f"  {b:30s}: {c:>4} ({c/len(selected)*100:.1f}%)")

    # By original task_type (top 20)
    by_tt = Counter(e["task_type"] for e in selected)
    print(f"\nBy task_type (top 20):")
    for t, c in by_tt.most_common(20):
        print(f"  {t:40s}: {c:>4}")

    # By trajectory length bin
    by_bin = Counter(e["traj_len_bin"] for e in selected)
    print(f"\nBy traj_len_bin:")
    for b in ["short", "medium", "long"]:
        print(f"  {b}: {by_bin.get(b, 0)} ({by_bin.get(b, 0)/len(selected)*100:.1f}%)")

    # By difficulty tier
    by_diff = Counter(e["difficulty_tier"] for e in selected)
    print(f"\nBy difficulty_tier:")
    for d in ["easy", "medium", "hard"]:
        print(f"  {d}: {by_diff.get(d, 0)} ({by_diff.get(d, 0)/len(selected)*100:.1f}%)")

    # By data source
    by_src = Counter(e["data_source"] for e in selected)
    print(f"\nBy data_source:")
    for s, c in by_src.most_common():
        print(f"  {s}: {c} ({c/len(selected)*100:.1f}%)")

    # Cross-tab: label × difficulty
    print(f"\nCross-tab (label × difficulty):")
    for label in ["GOOD", "BAD"]:
        counts = Counter(
            e["difficulty_tier"] for e in selected if e["label"] == label
        )
        parts = ", ".join(f"{d}={counts.get(d, 0)}" for d in ["easy", "medium", "hard"])
        print(f"  {label}: {parts}")

    # Bad reasons distribution (top 15)
    bad_entries = [e for e in selected if e["label"] == "BAD"]
    if bad_entries:
        reason_counter = Counter()
        for e in bad_entries:
            for r in e.get("bad_reasons", []):
                reason_counter[r] += 1
        print(f"\nBad reasons (top 15, from {len(bad_entries)} BAD entries):")
        for r, c in reason_counter.most_common(15):
            print(f"  {c:>4}  {r}")

    # Num turns stats
    turns = [e["num_turns"] for e in selected]
    print(f"\nTrajectory turns: min={min(turns)}, median={sorted(turns)[len(turns)//2]}, max={max(turns)}")

    # Difficulty score stats
    diffs = [e["difficulty_score"] for e in selected]
    print(f"Difficulty score:  min={min(diffs):.1f}, median={sorted(diffs)[len(diffs)//2]:.1f}, max={max(diffs):.1f}")


# ===========================================================================
# 8. Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Testset Selection v3 — 800 Labeled Trajectories")
    print("=" * 60)

    # Step 1: Load and merge
    print("\n[1] Loading and merging data...")
    entries = load_and_merge()

    label_counts = Counter(e["_label"] for e in entries)
    print(f"  Label distribution: {dict(label_counts)}")

    # Step 2: Score difficulty
    print("\n[2] Scoring difficulty...")
    for e in entries:
        e["_diff_score"] = compute_difficulty_score(e)
    assign_difficulty_tiers(entries)

    tier_counts = Counter(e["_diff_tier"] for e in entries)
    print(f"  Difficulty tiers: {dict(sorted(tier_counts.items()))}")

    # Step 3: Assign strata
    print("\n[3] Assigning strata...")
    assign_strata(entries)

    bucket_counts = Counter(e["_task_bucket"] for e in entries)
    print(f"  Task buckets:")
    for b, c in bucket_counts.most_common():
        print(f"    {b:30s}: {c:>5}")

    bin_counts = Counter(e["_traj_bin"] for e in entries)
    print(f"  Traj bins: {dict(sorted(bin_counts.items()))}")

    # Step 4: Split by label
    good_pool = [e for e in entries if e["_label"] == "GOOD"]
    bad_pool = [e for e in entries if e["_label"] == "BAD"]
    print(f"\n[4] Pools — GOOD: {len(good_pool)}, BAD: {len(bad_pool)}")

    # Step 5: Build strata and allocate for each label
    print("\n[5] Allocating budgets...")

    # Build bad_reasons frequency for diversity sampling
    global_bad_reason_counts = Counter()
    for e in bad_pool:
        for r in e.get("_bad_reasons", []):
            global_bad_reason_counts[r] += 1

    all_selected = []

    total_cap = int(TARGET_BUDGET * MAX_BUCKET_SHARE)  # 200

    for label, pool, budget in [("GOOD", good_pool, GOOD_BUDGET), ("BAD", bad_pool, BAD_BUDGET)]:
        # Build strata: (task_bucket, traj_bin) -> entries
        strata = defaultdict(list)
        for e in pool:
            key = (e["_task_bucket"], e["_traj_bin"])
            strata[key].append(e)

        print(f"\n  [{label}] {len(strata)} non-empty strata from {len(pool)} entries, budget={budget}")

        # Split Reasoning_QA cap proportionally per label
        label_cap = round(total_cap * budget / TARGET_BUDGET)
        allocations = allocate_budget(strata, budget, cap_max=label_cap)
        alloc_total = sum(allocations.values())
        print(f"  Allocated total: {alloc_total}")

        # Sample from each stratum
        label_selected = []
        for key, alloc in allocations.items():
            members = strata[key]
            sampled = sample_stratum(
                members, alloc,
                is_bad=(label == "BAD"),
                global_bad_reason_counts=global_bad_reason_counts,
            )
            label_selected.extend(sampled)

        print(f"  Selected: {len(label_selected)}")
        all_selected.extend(label_selected)

    # Step 6: Format and save
    print(f"\n[6] Formatting {len(all_selected)} entries...")
    random.shuffle(all_selected)
    output = [format_output(e) for e in all_selected]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print_summary(output)
    print(f"\nSaved {len(output)} entries to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
