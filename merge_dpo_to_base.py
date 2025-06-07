from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "" # Specify the path to your base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto",
)

adapter_checkpoint_path = "" # Specify the path to your adapter checkpoint
model = PeftModel.from_pretrained(
    model,
    adapter_checkpoint_path,
    torch_dtype="auto",
)

model = model.merge_and_unload()

save_path = "" # Specify the path where you want to save the merged model
model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(save_path)
