from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

max_seq_length = 512

model = AutoModelForCausalLM.from_pretrained("", torch_dtype=torch.float32, device_map='auto') # Specify the path to your model
tokenizer = AutoTokenizer.from_pretrained("") # Specify the path to your tokenizer
train_dataset = load_dataset('json', data_files='', split="train") # Specify the path to your training data

lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0, 
    bias="none",    
    task_type="CAUSAL_LM",  
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False


training_args = DPOConfig(
    output_dir="",  # Specify the output directory for the model
    logging_steps=10,
    save_steps=500,
    num_train_epochs=1,
    per_device_train_batch_size=32,
)

trainer = DPOTrainer(
    model=model, 
    args=training_args, 
    ref_model=None,
    processing_class=tokenizer, 
    train_dataset=train_dataset,
    )
trainer.train()