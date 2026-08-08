import json, os

OUT_DIR = 'metrics_by_tasktype'
DATA = json.load(open(os.path.join(OUT_DIR, 'by_tasktype_all_models.json')))
MODELS = ['base_9b', 'dpo_9b', 'sft_9b']

METRICS = [
    'tool_calling_f1', 'step_efficiency', 'redundancy_score',
    'pass_rate', 'task_relevance', 'logical_progression',
    'information_utilization', 'progress_score', 'final_answer_quality',
]
SHORT = {
    'tool_calling_f1': 'ToolF1', 'step_efficiency': 'StepEff', 'redundancy_score': 'Redund',
    'pass_rate': 'Pass', 'task_relevance': 'Relev', 'logical_progression': 'Logic',
    'information_utilization': 'InfoUtil', 'progress_score': 'Progress', 'final_answer_quality': 'FinalAns',
}
MAXES = {
    'tool_calling_f1': '1', 'step_efficiency': '1', 'redundancy_score': '1',
    'pass_rate': '5', 'task_relevance': '5', 'logical_progression': '5',
    'information_utilization': '5', 'progress_score': '5', 'final_answer_quality': '5',
}


def fmt(v):
    return f'{v:.3f}' if v is not None else '—'


def col_avg(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def all_task_types():
    tts = set()
    for m in MODELS:
        tts.update(DATA[m].keys())
    return sorted(tts)


lines = []
lines.append('# 9B 模型按 Task Type 的 9 项指标评测结果\n')
lines.append('数据来源：`rubrics_results_trained_batch2/{base_9b, dpo_9b, sft_9b}`\n')
lines.append('指标满分：`ToolF1 / StepEff / Redund` = 1.0；其余 6 项 = 5.0\n')
lines.append('| 指标 | 全称 | 满分 |')
lines.append('|---|---|---|')
for m in METRICS:
    lines.append(f'| {SHORT[m]} | {m} | {MAXES[m]} |')
lines.append('')

# ---- Per-model tables ----
for model in MODELS:
    rows = DATA[model]
    lines.append(f'## {model}\n')
    head = '| task_type | n | ' + ' | '.join(SHORT[m] for m in METRICS) + ' |'
    sep = '|' + '---|' * (len(METRICS) + 2)
    lines.append(head)
    lines.append(sep)
    for tt in sorted(rows):
        r = rows[tt]
        cells = ' | '.join(fmt(r[m]) for m in METRICS)
        lines.append(f'| {tt} | {r["_n"]} | {cells} |')
    # average row (simple mean across task types)
    total_n = sum(r['_n'] for r in rows.values())
    avg_cells = ' | '.join(
        fmt(col_avg([rows[tt][m] for tt in rows])) for m in METRICS
    )
    lines.append(f'| **平均(各task_type)** | {total_n} | {avg_cells} |')
    lines.append('')

# ---- Side-by-side comparison per metric ----
lines.append('## 三模型对比（每项指标）\n')
tts = all_task_types()
for m in METRICS:
    lines.append(f'### {SHORT[m]} — {m} (满分 {MAXES[m]})\n')
    lines.append('| task_type | n | base_9b | dpo_9b | sft_9b |')
    lines.append('|---|---|---|---|---|')
    for tt in tts:
        n = next((DATA[mo][tt]['_n'] for mo in MODELS if tt in DATA[mo]), '')
        vals = []
        for mo in MODELS:
            r = DATA[mo].get(tt)
            vals.append(fmt(r[m]) if r else '—')
        lines.append(f'| {tt} | {n} | ' + ' | '.join(vals) + ' |')
    # average row across task types, per model
    avg_vals = ' | '.join(
        fmt(col_avg([DATA[mo][tt][m] for tt in DATA[mo]])) for mo in MODELS
    )
    lines.append(f'| **平均(各task_type)** |  | ' + avg_vals + ' |')
    lines.append('')

# ---- Overall (weighted by n) per model per metric ----
lines.append('## 总体平均（按样本数加权）\n')
lines.append('| 指标 | base_9b | dpo_9b | sft_9b |')
lines.append('|---|---|---|---|')
for m in METRICS:
    row = [f'{SHORT[m]} (/{MAXES[m]})']
    for mo in MODELS:
        num = den = 0.0
        for tt, r in DATA[mo].items():
            if r[m] is not None:
                num += r[m] * r['_n']
                den += r['_n']
        row.append(fmt(num / den if den else None))
    lines.append('| ' + ' | '.join(row) + ' |')
lines.append('')

out_path = os.path.join(OUT_DIR, 'report.md')
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote', out_path)
