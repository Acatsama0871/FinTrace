import csv, os

OUT_DIR = 'metrics_by_tasktype'
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


def load(model):
    rows = []
    with open(os.path.join(OUT_DIR, f'{model}_by_tasktype.csv')) as f:
        for r in csv.DictReader(f):
            vals = {m: (float(r[m]) if r[m] != '' else None) for m in METRICS}
            present = [v for v in vals.values() if v is not None]
            avg = sum(present) / len(present) if present else None
            rows.append({'task_type': r['task_type'], 'n': int(r['n']),
                         'vals': vals, 'avg': avg})
    # sort by row-average, high -> low
    rows.sort(key=lambda x: (x['avg'] is not None, x['avg']), reverse=True)
    return rows


lines = []
lines.append('# 9B 模型按 Task Type 的 9 项指标评测结果\n')
lines.append('数据来源：`metrics_by_tasktype/{base_9b, dpo_9b, sft_9b}_by_tasktype.csv`\n')
lines.append('- **Average** 列 = 该 task type 上 9 项指标的简单平均值。')
lines.append('- 表格按 **Average 从高到低** 排序。')
lines.append('- 指标满分：`ToolF1 / StepEff / Redund` = 1.0；其余 6 项 = 5.0'
             '（注意各指标量纲不同，Average 仅作粗略综合参考）。\n')
lines.append('| 指标 | 全称 | 满分 |')
lines.append('|---|---|---|')
for m in METRICS:
    lines.append(f'| {SHORT[m]} | {m} | {MAXES[m]} |')
lines.append('')

for model in MODELS:
    rows = load(model)
    lines.append(f'## {model}\n')
    head = '| task_type | n | ' + ' | '.join(SHORT[m] for m in METRICS) + ' | **Average** |'
    sep = '|' + '---|' * (len(METRICS) + 3)
    lines.append(head)
    lines.append(sep)
    for r in rows:
        cells = ' | '.join(fmt(r['vals'][m]) for m in METRICS)
        lines.append(f'| {r["task_type"]} | {r["n"]} | {cells} | **{fmt(r["avg"])}** |')
    # column-average row over all task types
    total_n = sum(r['n'] for r in rows)
    col_avgs = {}
    for m in METRICS:
        vs = [r['vals'][m] for r in rows if r['vals'][m] is not None]
        col_avgs[m] = sum(vs) / len(vs) if vs else None
    overall = [v for v in col_avgs.values() if v is not None]
    overall_avg = sum(overall) / len(overall) if overall else None
    avg_cells = ' | '.join(fmt(col_avgs[m]) for m in METRICS)
    lines.append(f'| **列平均(各task_type)** | {total_n} | {avg_cells} | **{fmt(overall_avg)}** |')
    lines.append('')

out_path = os.path.join(OUT_DIR, 'report.md')
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote', out_path)
