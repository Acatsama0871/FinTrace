import json, os, csv
from collections import defaultdict

BASE = 'rubrics_results_trained_batch2'
MODELS = ['base_9b', 'dpo_9b', 'sft_9b']

METRICS = [
    'tool_calling_f1', 'step_efficiency', 'redundancy_score',
    'pass_rate', 'task_relevance', 'logical_progression',
    'information_utilization', 'progress_score', 'final_answer_quality',
]
MAX_SCORES = {
    'tool_calling_f1': 1.0, 'step_efficiency': 1.0, 'redundancy_score': 1.0,
    'pass_rate': 5, 'task_relevance': 5, 'logical_progression': 5,
    'information_utilization': 5, 'progress_score': 5, 'final_answer_quality': 5,
}

OUT_DIR = 'metrics_by_tasktype'
os.makedirs(OUT_DIR, exist_ok=True)


def compute(model):
    # sums[task_type][metric] = (sum, count)
    sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    counts = defaultdict(int)  # number of queries per task_type (from pass_rate)

    for metric in METRICS:
        fpath = os.path.join(BASE, model, f'{metric}.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        for item in data:
            tt = item.get('metadata', {}).get('task_type', 'Unknown')
            score = item.get('score')
            if score is None:
                continue
            sums[tt][metric][0] += score
            sums[tt][metric][1] += 1

    # query count per task type from first available metric file
    fpath = os.path.join(BASE, model, 'pass_rate.json')
    if os.path.exists(fpath):
        for item in json.load(open(fpath)):
            tt = item.get('metadata', {}).get('task_type', 'Unknown')
            counts[tt] += 1

    rows = {}
    for tt in sums:
        rows[tt] = {}
        for metric in METRICS:
            s, c = sums[tt][metric]
            rows[tt][metric] = (s / c) if c else None
        rows[tt]['_n'] = counts.get(tt, max((sums[tt][m][1] for m in METRICS), default=0))
    return rows


def main():
    all_results = {}
    for model in MODELS:
        rows = compute(model)
        all_results[model] = rows

        # CSV per model
        csv_path = os.path.join(OUT_DIR, f'{model}_by_tasktype.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task_type', 'n'] + METRICS)
            for tt in sorted(rows):
                r = rows[tt]
                w.writerow([tt, r['_n']] + [
                    f'{r[m]:.4f}' if r[m] is not None else '' for m in METRICS
                ])

        # printed table
        print(f'\n{"="*120}\nMODEL: {model}\n{"="*120}')
        header = f'{"task_type":<38}{"n":>4}  ' + '  '.join(f'{m[:10]:>10}' for m in METRICS)
        print(header)
        print('-' * len(header))
        for tt in sorted(rows):
            r = rows[tt]
            cells = '  '.join(
                (f'{r[m]:>10.4f}' if r[m] is not None else f'{"":>10}') for m in METRICS
            )
            print(f'{tt:<38}{r["_n"]:>4}  {cells}')

    # combined JSON
    with open(os.path.join(OUT_DIR, 'by_tasktype_all_models.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nSaved CSVs + JSON to ./{OUT_DIR}/')
    print('Metrics order:', METRICS)
    print('Max scores:', MAX_SCORES)


if __name__ == '__main__':
    main()
