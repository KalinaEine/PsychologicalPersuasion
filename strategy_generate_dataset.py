import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

data_dir = "" # Specify the training data directory containing JSON files
output_path = "" # Specify the output path for the DPO training data

prompt_prefix = (
    "You are chatting with others on one specific topic. You are required to convince others in your response.\n"
    "Please directly and strictly write your response in a natural, coherent, and emotionally effective manner with the persuasion strategy.\n"
    "Please continue chatting with others in a complete and long paragraph based on the topic ```{} {}```."
)

# key: prompt_text -> { "chosen": [evidence], "rejected": [evidence] }
prompt_dict = defaultdict(lambda: {"chosen": [], "rejected": []})

# Step 1: Aggregate data from all JSON files
for filename in tqdm(os.listdir(data_dir)):
    if not filename.endswith(".json"):
        continue
    path = os.path.join(data_dir, filename)
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        for item in data:
            base_prompt = item.get("prompt", "").strip()
            target_new = item.get("target_new", "").strip()
            evidence = item.get("evidence", "").strip()
            is_correct = item.get("is_correct", False)

            if not base_prompt or not target_new or not evidence:
                continue

            prompt_text = prompt_prefix.format(base_prompt, target_new)

            if is_correct:
                prompt_dict[prompt_text]["chosen"].append(evidence)
            else:
                prompt_dict[prompt_text]["rejected"].append(evidence)

# Step 2: Generate DPO training data
dpo_data = []
for prompt_text, buckets in tqdm(prompt_dict.items()):
    chosen_list = buckets["chosen"]
    rejected_list = buckets["rejected"]
    if not chosen_list or not rejected_list:
        continue

    # Shuffle both lists
    random.shuffle(chosen_list)
    random.shuffle(rejected_list)

    # First pair elements one by one to avoid duplication
    pairs = list(zip(chosen_list, rejected_list))
    
    # If fewer than 5 pairs, fill the rest from all possible combinations
    all_pairs = [(c, r) for c in chosen_list for r in rejected_list]
    used_pairs = set(pairs)
    remain = 5 - len(pairs)
    if remain > 0:
        # Remove already used pairs
        unused_pairs = [pair for pair in all_pairs if pair not in used_pairs]
        # If enough unused pairs, sample from them
        if len(unused_pairs) >= remain:
            pairs += random.sample(unused_pairs, remain)
        else:
            pairs += unused_pairs
            # If still not enough, randomly sample from all pairs
            while len(pairs) < 5 and all_pairs:
                pairs.append(random.choice(all_pairs))

    # Keep at most 5 pairs
    for chosen, rejected in pairs[:5]:
        dpo_data.append({
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected
        })

# Step 3: Save as a JSONL file
with open(output_path, "w", encoding="utf-8") as f:
    for item in dpo_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(dpo_data)} DPO examples to {output_path}")
