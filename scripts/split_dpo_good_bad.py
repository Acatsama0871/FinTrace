"""
Split SFT rollouts into GOOD / BAD using judge rubric scores, for DPO data.

Joins data/dpo/rollout_0804/rollout_results.json (prompt/chosen/rejected/tools)
with results/dpo_data_0804/aggregate.json (9 rubric scores) by id.

A rollout is BAD (usable as DPO rejected, paired with the golden chosen) when
the SFT rollout is clearly worse than the golden reference:
  - no final answer (no non-empty assistant message at the end), or
  - pass_rate <= --bad-pass-rate, or
  - final_answer_quality <= --bad-faq

Everything else is GOOD. Outputs <out>/bad.json and <out>/good.json; each entry
is the original rollout record plus _rubric_scores and (for bad) _bad_reasons.
bad.json keeps prompt/chosen/rejected/tools untouched, so it is directly the
DPO training input format (src/dpo_data.py tokenize_example).
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def has_final_answer(messages: list[dict]) -> bool:
    """Same rule as rubrics_eval_trained._extract_final_answer."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            if (msg.get("content") or "").strip():
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", default="data/dpo/rollout_0804/rollout_results.json")
    ap.add_argument("--rubrics", default="results/dpo_data_0804/aggregate.json")
    ap.add_argument("--out", default="data/dpo/split_0804")
    ap.add_argument("--bad-pass-rate", type=float, default=2, help="pass_rate <= this => bad")
    ap.add_argument("--bad-faq", type=float, default=2, help="final_answer_quality <= this => bad")
    args = ap.parse_args()

    rollouts = json.load(open(args.rollout))
    scores_by_id = {r["id"]: r["scores"] for r in json.load(open(args.rubrics))}

    good, bad = [], []
    reason_counter = Counter()
    missing_scores = 0

    for entry in rollouts:
        scores = scores_by_id.get(entry["_id"])
        if scores is None:
            missing_scores += 1
            continue

        reasons = []
        if not has_final_answer(entry["rejected"]):
            reasons.append("no_final_answer")
        if scores["pass_rate"] <= args.bad_pass_rate:
            reasons.append("low_pass_rate")
        if scores["final_answer_quality"] <= args.bad_faq:
            reasons.append("low_final_answer_quality")

        entry["_rubric_scores"] = scores
        if reasons:
            entry["_bad_reasons"] = reasons
            reason_counter.update(reasons)
            bad.append(entry)
        else:
            good.append(entry)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("good", good), ("bad", bad)):
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[written] {path}  ({len(data)} entries)")

    total = len(good) + len(bad)
    print(f"\ntotal={total}  good={len(good)} ({len(good)/total:.1%})  bad={len(bad)} ({len(bad)/total:.1%})")
    if missing_scores:
        print(f"[WARN] {missing_scores} rollouts had no rubric scores and were skipped")
    print("\nbad reasons (an entry can have several):")
    for reason, n in reason_counter.most_common():
        print(f"  {reason:<26} {n}")

    print("\nmean scores per split:")
    metrics = list(next(iter(scores_by_id.values())))
    print(f"  {'metric':<24} {'good':>6} {'bad':>6}")
    for m in metrics:
        g = sum(e["_rubric_scores"][m] for e in good) / len(good) if good else float("nan")
        b = sum(e["_rubric_scores"][m] for e in bad) / len(bad) if bad else float("nan")
        print(f"  {m:<24} {g:>6.2f} {b:>6.2f}")


if __name__ == "__main__":
    main()
