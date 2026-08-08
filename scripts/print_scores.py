import json, os

output_dir = 'results/benchmark_13llm'
model_dirs = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])

metrics_order = [
    'tool_calling_f1', 'step_efficiency', 'redundancy_score',
    'pass_rate', 'task_relevance', 'logical_progression',
    'information_utilization', 'progress_score', 'final_answer_quality'
]
max_scores = {
    'tool_calling_f1': '1.0', 'step_efficiency': '1.0', 'redundancy_score': '1.0',
    'pass_rate': '5', 'task_relevance': '5', 'logical_progression': '5',
    'information_utilization': '5', 'progress_score': '5', 'final_answer_quality': '5'
}

for i, model in enumerate(model_dirs):
    model_path = os.path.join(output_dir, model)
    print(f'[{i+1}/{len(model_dirs)}] {model}')
    print(f'{"Metric":<35} {"Avg":>10}')
    print('-' * 46)
    for metric in metrics_order:
        fpath = os.path.join(model_path, f'{metric}.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        vals = [item['score'] for item in data]
        avg = sum(vals) / len(vals) if vals else 0
        ms = max_scores[metric]
        print(f'{metric:<35} {avg:.4f}/{ms}')
    print()
