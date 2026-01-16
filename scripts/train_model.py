"""
train_model.py - QLoRA Fine-Tuning Script (8GB GPU Friendly)

✅ Supports:
- SQLCoder / CodeLlama / Mistral / DeepSeek (7B class models)
- 4-bit loading (QLoRA)
- LoRA fine-tuning only (low VRAM)
- JSONL dataset format: {"text": "..."}
- Dynamic padding (less memory waste)
- Saves LoRA adapters to OUTPUT_DIR

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 3 --max-len 256
"""

import os
import json
import argparse
import gc
import torch
from datasets import Dataset
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


def print_gpu_stats(title="GPU Stats"):
    if not torch.cuda.is_available():
        print(f"{title}: CUDA not available")
        return
    device_id = 0
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
    print(
        f"{title}: GPU={torch.cuda.get_device_name(device_id)} | "
        f"Total={total:.1f}GB | Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB"
    )


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl_as_dataset(file_path: str) -> Dataset:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                rows.append({"text": obj["text"]})
    return Dataset.from_list(rows)


def find_lora_target_modules(model):
    """
    Different models use different internal names.
    Most LLaMA-style models use: q_proj, k_proj, v_proj, o_proj
    """
    common = ["q_proj", "k_proj", "v_proj", "o_proj"]
    names = set()

    for name, _ in model.named_modules():
        # Collect only last component
        last = name.split(".")[-1]
        if last in common:
            names.add(last)

    # If not found, fallback to common list
    if not names:
        return common

    return sorted(list(names))


def main():
    parser = argparse.ArgumentParser(description="QLoRA Trainer for SQL models (8GB GPU safe)")
    parser.add_argument("--train-file", type=str, default=os.getenv(
        "TRAIN_FILE", "./data/processed/training_data.json/train_prompts.jsonl"
    ))
    parser.add_argument("--val-file", type=str, default=os.getenv(
        "VAL_FILE", "./data/processed/training_data.json/validation_prompts.jsonl"
    ))
    parser.add_argument("--model-path", type=str, default=os.getenv(
        "MODEL_PATH", "./models/base/defog_sqlcoder-7b-2"
    ))
    parser.add_argument("--output-dir", type=str, default=os.getenv(
        "OUTPUT_DIR", "./models/fine_tuned/sqlcoder-7b-2-qlora"
    ))

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("✅ QLoRA TRAINING START")
    print("=" * 90)
    print(f"Base Model Path : {args.model_path}")
    print(f"Train File      : {args.train_file}")
    print(f"Val File        : {args.val_file}")
    print(f"Output Dir      : {args.output_dir}")
    print(f"Epochs          : {args.epochs}")
    print(f"Max Length      : {args.max_len}")
    print(f"Batch Size      : {args.batch_size}")
    print(f"Grad Accum      : {args.grad_accum}")
    print(f"Learning Rate   : {args.lr}")
    print("=" * 90 + "\n")

    if torch.cuda.is_available():
        print_gpu_stats("Before loading model")

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Model load in 4-bit
    # -----------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU not available. QLoRA training requires GPU for practical speed.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Important for training stability
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # -----------------------------
    # LoRA Setup
    # -----------------------------
    target_modules = find_lora_target_modules(model)
    print(f"✅ LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        print_gpu_stats("After loading model")

    # -----------------------------
    # Dataset
    # -----------------------------
    train_dataset = load_jsonl_as_dataset(args.train_file)
    val_dataset = load_jsonl_as_dataset(args.val_file)

    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples  : {len(val_dataset)}")

    # Tokenization (dynamic padding later by collator)
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_len
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Dynamic padding (saves memory)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # -----------------------------
    # TrainingArgs
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,

        fp16=True,
        bf16=False,

        optim="paged_adamw_8bit",

        logging_steps=25,
        save_steps=200,
        save_total_limit=2,

        eval_strategy="steps",

        eval_steps=200,

        report_to="none",

        # Extra stability
        warmup_ratio=0.03,
        weight_decay=0.01,

        # Prevent dataloader stalls on Windows sometimes
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("\n🚀 Starting training...\n")
    trainer.train()

    # -----------------------------
    # Save (LoRA Adapter)
    # -----------------------------
    print("\n💾 Saving LoRA adapter + tokenizer...\n")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    cleanup_cuda()

    print("\n" + "=" * 90)
    print("✅ TRAINING COMPLETE")
    print("=" * 90)
    print(f"Saved to: {args.output_dir}")
    print("NOTE: This folder contains LoRA adapter weights (not full model).")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
