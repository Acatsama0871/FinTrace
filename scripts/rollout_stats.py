"""
Statistical analysis of rollout rubric evaluation (base vs dpo vs sft).

Design: 3 groups x 5 seeds x the same 626 queries (aligned by _id).

Outputs:
  1. Seed-level descriptive stats per group x metric:
       mean, variance, std, SEM, 95% CI (across the 5 seed means).
  2. Query-level paired tests (queries averaged over the 5 seeds, _id-aligned):
       - Friedman omnibus across the 3 groups (non-parametric, paired)
       - pairwise Wilcoxon signed-rank with Holm correction
       - effect sizes (matched-pairs rank-biserial r; Cliff's delta)
  3. Seed-level one-way ANOVA + Welch ANOVA (seed as replication unit, n=5).

Results are written to results/rollout_5seed/STATS.md and printed.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("results/rollout_5seed")
GROUPS = ["base", "dpo", "sft"]
SEEDS = ["seed0", "seed1", "seed2", "seed3", "seed4"]
METRICS = [
    "tool_calling_f1",
    "step_efficiency",
    "redundancy_score",
    "pass_rate",
    "task_relevance",
    "logical_progression",
    "information_utilization",
    "progress_score",
    "final_answer_quality",
]
ALGO = {"tool_calling_f1", "step_efficiency", "redundancy_score"}  # /1, else /5


def load_scores(group, seed, metric):
    """Return {id: score} for one (group, seed, metric), skipping None."""
    f = ROOT / group / seed / f"{metric}.json"
    recs = json.load(open(f))
    return {r["id"]: r["score"] for r in recs if r.get("score") is not None}


def seed_level_table():
    """Per group x metric: array of 5 seed means."""
    out = {}  # (group, metric) -> np.array of 5 seed means
    for g in GROUPS:
        for m in METRICS:
            means = []
            for s in SEEDS:
                sc = load_scores(g, s, m)
                means.append(np.mean(list(sc.values())))
            out[(g, m)] = np.array(means)
    return out


def query_level_matrix(metric):
    """Return (ids, {group: vector}) where each query's score is averaged
    over the 5 seeds; only ids present in all 3 groups (all seeds) are kept."""
    per_group = {}  # group -> {id: mean_over_seeds}
    for g in GROUPS:
        acc = {}
        for s in SEEDS:
            for qid, sc in load_scores(g, s, metric).items():
                acc.setdefault(qid, []).append(sc)
        # average over available seeds
        per_group[g] = {qid: np.mean(v) for qid, v in acc.items()}
    common = set(per_group[GROUPS[0]])
    for g in GROUPS[1:]:
        common &= set(per_group[g])
    common = sorted(common)
    vecs = {g: np.array([per_group[g][q] for q in common]) for g in GROUPS}
    return common, vecs


def cliffs_delta(a, b):
    """Cliff's delta: P(a>b) - P(a<b). Range [-1,1]. a vs b."""
    a = np.asarray(a)
    b = np.asarray(b)
    # efficient via sorting
    gt = sum(np.sum(a > x) for x in b)
    lt = sum(np.sum(a < x) for x in b)
    n = len(a) * len(b)
    return (gt - lt) / n if n else 0.0


def rank_biserial_wilcoxon(diff):
    """Matched-pairs rank-biserial correlation from signed-rank stats.
    diff = a - b. Returns r in [-1,1] (positive => a>b)."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = r_plus + r_minus
    return (r_plus - r_minus) / total if total else 0.0


def bonferroni(pvals, m=None):
    """Bonferroni-adjusted p-values: p_adj = min(1, p * m).
    m defaults to the number of tests in `pvals` (family size)."""
    pvals = np.asarray(pvals, dtype=float)
    if m is None:
        m = len(pvals)
    return np.minimum(pvals * m, 1.0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values. Input/output keep original order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def fmt_p(p):
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    seed_means = seed_level_table()

    # ---- 1. Seed-level descriptives ----
    emit("# Rollout 评测统计检验 (base / dpo / sft)")
    emit()
    emit("设计: 3 组 × 5 seed × 同一批 626 query。")
    emit()
    emit("## 1. Seed 级描述统计 (n=5 seed 的均值上)")
    emit()
    emit("每格 = 5 个 seed 均值的 `mean ± std (var; 95%CI)`。")
    emit()
    hdr = f"| {'metric':<24} | {'base':<26} | {'dpo':<26} | {'sft':<26} |"
    emit(hdr)
    emit("|" + "-" * 26 + "|" + ("-" * 28 + "|") * 3)
    for m in METRICS:
        scale = "/1" if m in ALGO else "/5"
        cells = []
        for g in GROUPS:
            x = seed_means[(g, m)]
            mean, sd, var = x.mean(), x.std(ddof=1), x.var(ddof=1)
            sem = sd / np.sqrt(len(x))
            ci = stats.t.ppf(0.975, len(x) - 1) * sem
            cells.append(f"{mean:.3f}±{sd:.3f} (v={var:.4f}; ±{ci:.3f})")
        emit(f"| {m+scale:<24} | " + " | ".join(f"{c:<26}" for c in cells) + " |")
    emit()

    # ---- 2. Query-level paired tests ----
    emit("## 2. 组间显著性 — query 级配对检验 (n=626, 每 query 跨 5 seed 取平均)")
    emit()
    emit("- Friedman: 三组配对全局检验 (非参)。")
    emit("- 两两 Wilcoxon 符号秩检验, Holm 校正; 效应量 = matched-pairs rank-biserial r。")
    emit()
    all_pairwise = []  # (metric, label, mdiff, r, p_raw) across all 27 comparisons
    for m in METRICS:
        scale = "/1" if m in ALGO else "/5"
        ids, vecs = query_level_matrix(m)
        n = len(ids)
        b, d, s = vecs["base"], vecs["dpo"], vecs["sft"]
        # Friedman omnibus
        try:
            fr_stat, fr_p = stats.friedmanchisquare(b, d, s)
        except ValueError:
            fr_stat, fr_p = float("nan"), float("nan")
        emit(f"### {m}{scale}  (n={n})")
        emit(f"- 组均值: base={b.mean():.3f}  dpo={d.mean():.3f}  sft={s.mean():.3f}")
        emit(f"- Friedman χ²={fr_stat:.2f}, p={fmt_p(fr_p)}")
        # pairwise wilcoxon
        pairs = [("sft", "base"), ("sft", "dpo"), ("dpo", "base")]
        pvals, rows = [], []
        for a_name, c_name in pairs:
            a, c = vecs[a_name], vecs[c_name]
            diff = a - c
            if np.all(diff == 0):
                w_p, r = 1.0, 0.0
            else:
                try:
                    _, w_p = stats.wilcoxon(a, c, zero_method="wilcox")
                except ValueError:
                    w_p = 1.0
                r = rank_biserial_wilcoxon(diff)
            pvals.append(w_p)
            rows.append((f"{a_name} vs {c_name}", a.mean() - c.mean(), r))
            all_pairwise.append((m, f"{a_name} vs {c_name}", a.mean() - c.mean(), r, w_p))
        adj = holm(np.array(pvals))
        for (label, mdiff, r), p_raw, p_adj in zip(rows, pvals, adj):
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
            emit(
                f"  - {label:<14} Δmean={mdiff:+.3f}  r={r:+.3f}  "
                f"p={fmt_p(p_raw)}  p_holm={fmt_p(p_adj)}  [{sig}]"
            )
        emit()

    # ---- 3. Seed-level ANOVA ----
    emit("## 3. Seed 级方差分析 (以 seed 为重复单元, n=5/组)")
    emit()
    emit("以每组 5 个 seed 均值做单因素 ANOVA + Welch ANOVA。")
    emit()
    emit(f"| {'metric':<24} | {'F (one-way)':<12} | {'p':<10} | {'Welch F':<10} | {'Welch p':<10} |")
    emit("|" + "-" * 26 + "|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    for m in METRICS:
        groups_data = [seed_means[(g, m)] for g in GROUPS]
        try:
            f1, p1 = stats.f_oneway(*groups_data)
        except Exception:
            f1, p1 = float("nan"), float("nan")
        try:
            wr = stats.alexandergovern(*groups_data)  # Welch-type
            wf, wp = wr.statistic, wr.pvalue
        except Exception:
            wf, wp = float("nan"), float("nan")
        emit(f"| {m:<24} | {f1:<12.3f} | {fmt_p(p1):<10} | {wf:<10.3f} | {fmt_p(wp):<10} |")
    emit()

    # ---- 4. Bonferroni correction ----
    emit("## 4. Bonferroni 校正 (query 级两两 Wilcoxon)")
    emit()
    emit("对 section 2 的 27 个两两对比 (9 指标 × 3 对) 做 Bonferroni 校正。")
    emit("- `p_bonf(m=3)`: 每指标族内校正 (×3), 与 Holm 同族, 便于对比。")
    emit("- `p_bonf(m=27)`: 全局族校正 (×27), 控制整个研究的 family-wise error, 最严格。")
    emit()
    raw = np.array([t[4] for t in all_pairwise])
    # per-metric family (m=3): group by metric
    p_bonf3 = np.empty(len(raw))
    for m in METRICS:
        idx = [i for i, t in enumerate(all_pairwise) if t[0] == m]
        p_bonf3[idx] = bonferroni(raw[idx], m=3)
    p_bonf27 = bonferroni(raw, m=27)

    emit(f"| {'metric':<22} | {'pair':<13} | {'p_raw':<10} | {'p_bonf(3)':<11} | {'p_bonf(27)':<11} | sig27 |")
    emit("|" + "-" * 24 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 13 + "|-------|")
    n_sig27 = 0
    for (metric, label, mdiff, r, p), pb3, pb27 in zip(all_pairwise, p_bonf3, p_bonf27):
        sig = "***" if pb27 < 0.001 else "**" if pb27 < 0.01 else "*" if pb27 < 0.05 else "ns"
        if pb27 < 0.05:
            n_sig27 += 1
        emit(
            f"| {metric:<22} | {label:<13} | {fmt_p(p):<10} | {fmt_p(pb3):<11} | "
            f"{fmt_p(pb27):<11} | {sig:<5} |"
        )
    emit()
    emit(
        f"全局 Bonferroni (m=27) 下: {n_sig27}/27 个对比仍显著 (p<0.05)。"
    )
    ns_list = [
        f"{metric} [{label}]"
        for (metric, label, _, _, _), pb27 in zip(all_pairwise, p_bonf27)
        if pb27 >= 0.05
    ]
    if ns_list:
        emit("不显著的对比 (即便最严格的全局校正下也无法拒绝原假设):")
        for x in ns_list:
            emit(f"  - {x}")
    else:
        emit("所有 27 个对比在全局 Bonferroni 下仍显著。")
    emit()
    emit("注: 显著性标记 — *** p<0.001, ** p<0.01, * p<0.05, ns 不显著 (Holm 校正后)。")
    emit("效应量 r (rank-biserial): |r|≈0.1 小, 0.3 中, 0.5 大; 正值表示前者更高。")

    out_path = ROOT / "STATS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()