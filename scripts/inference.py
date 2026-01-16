"""
inference.py - Run inference with a fine-tuned QLoRA model

Usage:
    python scripts/inference.py --prompt "Your input prompt here"
    python scripts/inference.py --prompt "SELECT * FROM users;" --base-model ./models/base/defog_sqlcoder-7b-2 --lora-adapter ./models/fine_tuned/sqlcoder-7b-2-qlora
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned QLoRA model")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--base-model", type=str, default="./models/base/defog_sqlcoder-7b-2", help="Path to base model")
    parser.add_argument("--lora-adapter", type=str, default="./models/fine_tuned/sqlcoder-7b-2-qlora", help="Path to LoRA adapter directory")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lora_adapter, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Model Output ===\n")
    print(result)

if __name__ == "__main__":
    main()
