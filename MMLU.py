from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm

# Load the full MMLU test set (all subjects)
mmlu = load_dataset("cais/mmlu", "all", split="test")

# Define the models to evaluate
models_info = {
    "LLaMA 3.1 8B Instruct": "",
    "Qwen 2.5 7B Instruct": "",
    "Falcon 3 7B Instruct": ""
}

# Loop over each model and evaluate
for model_name, model_id in models_info.items():
    print(f"Evaluating {model_name}...")
    # Load tokenizer and model (using bfloat16 and device_map="auto" for efficiency if supported)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    model.eval()  # set to eval mode
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    correct = 0
    total = len(mmlu)
    for item in tqdm.tqdm(mmlu):
        question = item["question"]
        choices = item["choices"]
        correct_answer = item["answer"]  # e.g. "C"
        
        # Format the prompt with answer options labeled A, B, C, D
        prompt = f"Question: {question}\n"
        option_labels = ["A", "B", "C", "D"]
        for label, choice in zip(option_labels, choices):
            prompt += f"{label}. {choice}\n"
        prompt += "Please directly select the correct option without any other words:"  # ask the model to provide the letter as answer
        
        # Generate model's answer
        if "Instruct" in model_name:
            # Use chat template for the instruct model
            messages = [{"role": "user", "content": prompt}]
            # Format the chat prompt according to LLaMA 3.1's template
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, 
                return_tensors="pt", return_dict=True
            )
            inputs = {k: v.to(model.device) for k,v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=16)
            # The generated sequence includes the prompt + completion; extract only new tokens
            input_len = inputs["input_ids"].shape[1]
        else:
            # For base LLaMA and GPT-J, use standard prompting
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**enc, max_new_tokens=16)
            input_len = enc["input_ids"].shape[1]
        
        # Decode the model's output tokens (excluding the prompt)
        generated = outputs[0][input_len:]
        output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # Determine the first predicted letter (A, B, C, or D) from the output
        pred_letter = None
        for char in output_text:
            if char.upper() in ["A", "B", "C", "D"]:
                pred_letter = char.upper()
                break
        # If no letter found, optionally check if the output contains one of the choice texts
        if pred_letter is None:
            for idx, choice in enumerate(choices):
                if choice.lower() in output_text.lower():
                    pred_letter = option_labels[idx]
                    break
        correct_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_answer = correct_map[correct_answer]
        # Compare with correct answer
        if pred_letter == correct_answer:
            correct += 1
        
        # print(correct)
    
    accuracy = correct / total
    print(f"{model_name} Accuracy: {accuracy:.2%} ({correct} / {total})\n")

        # Save result to a file
    with open("", "a") as f: # Specify your output file path here
        f.write(f"{model_name} Accuracy: {accuracy:.2%} ({correct} / {total})\n")
