import os
import json
import csv
from glob import glob

def extract_metrics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    total = len(data)
    correct = sum(1 for x in data if x.get('is_correct', False))
    robust = sum(1 for x in data if x.get('is_robust', False))
    locality = sum(1 for x in data if x.get('is_locality', False))
    acc = correct / total if total > 0 else 0
    robust_acc = robust / total if total > 0 else 0
    locality_acc = locality / total if total > 0 else 0
    return {
        'file': os.path.basename(json_path),
        'total': total,
        'accuracy': acc,
        'robust_accuracy': robust_acc,
        'locality_accuracy': locality_acc
    }

def main():
    results_dir = './results/agent_to_agent1'
    all_json = glob(os.path.join(results_dir, '**', '*.json'), recursive=True)
    metrics = []
    for json_file in all_json:
        try:
            m = extract_metrics(json_file)
            metrics.append(m)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    # 写入csv
    with open('results_eval_agent1.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'total', 'accuracy', 'robust_accuracy', 'locality_accuracy'])
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    print('Results saved to results_eval.csv')

if __name__ == '__main__':
    main() 