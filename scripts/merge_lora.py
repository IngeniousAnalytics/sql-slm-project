from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "./models/base/defog_sqlcoder-7b-2"
lora_path = "./models/fine_tuned/sqlcoder-7b-2-qlora"
output_path = "./models/merged/sqlcoder-7b-2-merged"

model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # Merge LoRA weights

model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)