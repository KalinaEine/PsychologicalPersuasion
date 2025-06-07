import os
import json
import csv
from glob import glob
from openai import OpenAI
import time

client = OpenAI(
    api_key="", # Replace with your OpenAI API key
    base_url="" # Replace with your OpenAI API base URL
)

CATEGORIES = ['person', 'geo', 'culture', 'life']
CACHE_PATH = './results/llama3/prompt_category_cache.json'

def classify_prompt_with_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classification assistant. "
                        "Classify the given prompt into one of: 'person', 'geo', 'culture', or 'life'. "
                        "Definitions:\n"
                        "- person: specific individual or historical figure\n"
                        "- geo: cities, countries, or physical places\n"
                        "- culture: topics like media, art, language, history\n"
                        "- life: daily topics, tech, lifestyle, products, education\n\n"
                        "Respond with ONLY one label: person, geo, culture, or life."
                    )
                },
                {"role": "user", "content": f"Prompt: {prompt}"}
            ],
            temperature=0
        )
        label = response.choices[0].message.content.strip().lower()
        print(f"Classified successfully: {prompt} -> {label}")
        if label in CATEGORIES:
            return label
        print(f"Invalid Classification: {prompt} -> {label}")
        return None
    except Exception as e:
        print(f"Classified unsuccessfully: {prompt} -> {e}")
        return None

def load_or_create_cache(results_dir):
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded class cache (total {len(cache)} items)")
    else:
        cache = {}
        print("No cache was detected. Create the cache while processing.")

    all_json = glob(os.path.join(results_dir, '**', '*.json'), recursive=True)

    for path in all_json:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
            except:
                continue

        for item in data:
            prompt = item.get('prompt')
            if prompt and prompt not in cache:
                label = classify_prompt_with_gpt(prompt)
                if label:
                    cache[prompt] = label
                    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)
                    time.sleep(0.5)

    return cache

def extract_metrics(json_path, prompt_label_cache):
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

    category_count = {cat: 0 for cat in CATEGORIES}
    category_correct = {cat: 0 for cat in CATEGORIES}
    
    for x in data:
        prompt = x.get("prompt", "")
        label = prompt_label_cache.get(prompt)
        if label is None:
            continue
        category_count[label] += 1
        if x.get('is_correct', False):
            category_correct[label] += 1

    result = {
        'file': os.path.basename(json_path),
        'total': total,
        'accuracy': acc,
        'robust_accuracy': robust_acc,
        'locality_accuracy': locality_acc
    }

    for cat in CATEGORIES:
        result[f"{cat}_total"] = category_count[cat]
        result[f"{cat}_accuracy"] = (
            category_correct[cat] / category_count[cat]
            if category_count[cat] > 0 else 0
        )
    
    return result

def run_full_evaluation(results_dir):
    cache = load_or_create_cache(results_dir)

    all_json = glob(os.path.join(results_dir, '**', '*.json'), recursive=True)
    metrics = []

    for json_file in all_json:
        try:
            m = extract_metrics(json_file, cache)
            metrics.append(m)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    output_csv = os.path.join(results_dir, 'results_eval_with_categories.csv')
    fieldnames = ['file', 'total', 'accuracy', 'robust_accuracy', 'locality_accuracy']
    for cat in CATEGORIES:
        fieldnames.append(f"{cat}_total")
        fieldnames.append(f"{cat}_accuracy")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    print(f'Results saved to {output_csv}')

if __name__ == '__main__':
    run_full_evaluation('') # Specify the directory containing the results JSON files
