import json, os

BASE = 'rubrics_results_trained_batch2'
OUT_DIR = 'metrics_by_tasktype'
MODELS = ['base_9b', 'sft_9b', 'dpo_9b']  # column order: base, SFT, DPO

METRICS = [
    'tool_calling_f1', 'step_efficiency', 'redundancy_score',
    'pass_rate', 'task_relevance', 'logical_progression',
    'information_utilization', 'progress_score', 'final_answer_quality',
]
MAXES = {
    'tool_calling_f1': '1', 'step_efficiency': '1', 'redundancy_score': '1',
    'pass_rate': '5', 'task_relevance': '5', 'logical_progression': '5',
    'information_utilization': '5', 'progress_score': '5', 'final_answer_quality': '5',
}


def overall_mean(model, metric):
    fpath = os.path.join(BASE, model, f'{metric}.json')
    if not os.path.exists(fpath):
        return None
    scores = [it['score'] for it in json.load(open(fpath)) if it.get('score') is not None]
    return sum(scores) / len(scores) if scores else None


def fmt(v):
    return f'{v:.4f}' if v is not None else '—'


# build rows: one per metric
rows = []
for m in METRICS:
    vals = {mo: overall_mean(mo, m) for mo in MODELS}
    present = [v for v in vals.values() if v is not None]
    avg = sum(present) / len(present) if present else None
    rows.append({'metric': m, 'max': MAXES[m], 'vals': vals, 'avg': avg})

# sort by 3-model average, high -> low
rows.sort(key=lambda r: (r['avg'] is not None, r['avg']), reverse=True)

lines = []
lines.append('# 9B 模型各指标的三模型平均（base / SFT / DPO）\n')
lines.append('数据来源：`rubrics_results_trained_batch2/{base_9b, sft_9b, dpo_9b}`\n')
lines.append('- 每个数值 = 该模型在所有 query 上该指标的总体平均分。')
lines.append('- **Avg(3模型)** = base、SFT、DPO 三者的平均。')
lines.append('- 表格按 **Avg(3模型) 从高到低** 排序。')
lines.append('- 满分：`tool_calling_f1 / step_efficiency / redundancy_score` = 1.0；其余 = 5.0'
             '（量纲不同，跨指标排序仅作参考）。\n')
lines.append('| 指标 (Metric) | 满分 | base_9b | sft_9b | dpo_9b | **Avg(3模型)** |')
lines.append('|---|---|---|---|---|---|')
for r in rows:
    lines.append(
        f'| {r["metric"]} | {r["max"]} | '
        + ' | '.join(fmt(r['vals'][mo]) for mo in MODELS)
        + f' | **{fmt(r["avg"])}** |'
    )
lines.append('')

out_path = os.path.join(OUT_DIR, 'report_by_metric.md')
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote', out_path)
print('\n'.join(lines))
