import gc
import json
import yaml
from tqdm import tqdm
import argparse
from strategy_agent import PersuaderAgent, ListenerAgent
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--listener', type=str, required=True)
    parser.add_argument('--persuader', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    listener_model_type = args.listener
    persuader_model_type = args.persuader
    output_dir = config.get("output_path", "./results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Listener_{listener_model_type}+Persuader_{persuader_model_type}+Strategy_{args.strategy}.json")
    with open(config["dataset_path"]) as f:
        dataset = json.load(f)

    batch_size = args.batch_size
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    # Load existing results to determine how many batches are completed
    results = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except Exception:
                results = []
    finished_batches = len(results) // batch_size
    print(f"{finished_batches} batches completed, total {total_batches} batches")
    if finished_batches >= total_batches:
        print("All completed, skipping!")
        return

    persuader = PersuaderAgent(config, "Persuader")
    listener = ListenerAgent(config, "Listener")
    correct = false = rephrase_correct = rephrase_false = locality_correct = locality_false = 0
    # Calculate historical accuracy
    for r in results:
        if r.get('is_correct'): correct += 1
        else: false += 1
        if r.get('is_robust'): rephrase_correct += 1
        else: rephrase_false += 1
        if r.get('is_locality'): locality_correct += 1
        else: locality_false += 1

    for batch_index, batch_start in enumerate(tqdm(range(0, len(dataset), batch_size))):
        if batch_index < finished_batches:
            continue
        batch = dataset[batch_start:batch_start+batch_size]
        batch_knowledge = [{
            "prompt": d["prompt"],
            "target_true": d["ground_truth"],
            "target_new": d["target_new"],
            "subject": d["subject"],
            "rephrase_prompt": d["rephrase_prompt"],
            "locality_prompt": d["locality_prompt"],
            "locality_ground_truth": d["locality_ground_truth"]
        } for d in batch]

        # Persuader generates evidence in batch
        batch_evidence = persuader.batch_generate_evidence(batch_knowledge, persuader_model_type, args.strategy)
        # Listener generates answers to the main questions in batch
        batch_prompts = [k["prompt"] for k in batch_knowledge]
        batch_answers = listener.batch_generate_answer(batch_prompts, batch_evidence, listener_model_type)
        # Listener generates answers to rephrased questions in batch
        batch_rephrase_prompts = [k["rephrase_prompt"] for k in batch_knowledge]
        batch_rephrase_answers = listener.batch_generate_answer(batch_rephrase_prompts, batch_evidence, listener_model_type)
        # Listener generates answers to locality questions in batch
        batch_locality_prompts = [k["locality_prompt"] for k in batch_knowledge]
        batch_locality_answers = listener.batch_generate_answer(batch_locality_prompts, batch_evidence, listener_model_type)

        for i, k in enumerate(batch_knowledge):
            answer = batch_answers[i]
            rephrase_answer = batch_rephrase_answers[i]
            locality_answer = batch_locality_answers[i]
            is_correct = k["target_new"] in answer
            is_robust = k["target_new"] in rephrase_answer
            is_locality = k["locality_ground_truth"] in locality_answer
            if is_correct: correct += 1
            else: false += 1
            if is_robust: rephrase_correct += 1
            else: rephrase_false += 1
            if is_locality: locality_correct += 1
            else: locality_false += 1
            current_accuracy = correct / (correct + false) if (correct + false) > 0 else 0
            current_rephrase_accuracy = rephrase_correct / (rephrase_correct + rephrase_false) if (rephrase_correct + rephrase_false) > 0 else 0
            current_locality_accuracy = locality_correct / (locality_correct + locality_false) if (locality_correct + locality_false) > 0 else 0
            result = {
                "ground_truth": k["target_true"],
                "target_new": k["target_new"],
                "prompt": k["prompt"],
                "evidence": batch_evidence[i],
                "answer": answer,
                "is_correct": is_correct,
                "current_accuracy": current_accuracy,
                "rephrase_prompt": k["rephrase_prompt"],
                "rephrase_answer": rephrase_answer,
                "is_robust": is_robust,
                "current_rephrase_accuracy": current_rephrase_accuracy,
                "locality_prompt": k["locality_prompt"],
                "locality_answer": locality_answer,
                "is_locality": is_locality,
                "current_locality_accuracy": current_locality_accuracy
            }
            print(result)
            results.append(result)
        gc.collect()
        # Save after each batch to prevent data loss in case of interruption
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
