
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Paths
MODEL_PATH = "./models/base/defog_sqlcoder-7b-2"
TRAIN_FILE = "./data/processed/training_data.json/train_prompts.jsonl"
VAL_FILE = "./data/processed/training_data.json/validation_prompts.jsonl"
OUTPUT_DIR = "./models/fine_tuned/sqlcoder-7b-2-finetuned"

# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Helper to load JSONL as Hugging Face Dataset
def load_jsonl(file_path):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = None
            try:
                obj = eval(line) if line.strip().startswith("{") else None
            except Exception:
                continue
            if obj and "text" in obj:
                texts.append({"text": obj["text"]})
    return Dataset.from_list(texts)

train_dataset = load_jsonl(TRAIN_FILE)
val_dataset = load_jsonl(VAL_FILE)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete. Model saved to", OUTPUT_DIR)