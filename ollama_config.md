# Ollama Fine-Tuned Model Setup (WSL2)

This document explains how to upload your **fine-tuned SQL model (QLoRA / LoRA)** into **Ollama running in WSL2**.

✅ Works with: **SQLCoder / CodeLlama / Mistral / DeepSeek Coder**  
✅ GPU: RTX 4060 Laptop (8GB)  
✅ Output: Ollama model you can run using:

```bash
ollama run sqlcoder-domain
```

---

## 1) Overview: What you are doing

Your fine-tuning process produces **LoRA adapters only**, not a full model.

Ollama does NOT directly use adapter-only output.

✅ Correct pipeline:

1. Merge LoRA adapter into base model (Windows)
2. Copy merged model into WSL2
3. Convert merged model → GGUF
4. Create Ollama model using `Modelfile`
5. Run model using Ollama

---

## 2) Prerequisites

### Windows
- Python installed
- Your fine-tuning output folder exists (LoRA adapter)

### WSL2 Ubuntu
- Ollama installed and running
- git + python3 + pip installed

Install tools in WSL2:

```bash
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv
```

---

## 3) Merge LoRA Adapter into Base Model (Windows)

### Why merge?
Your QLoRA training output is adapter-only.  
To convert into GGUF, you must merge adapter into full model.

Create file:

📌 `scripts/merge_lora.py`

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ✅ CHANGE THESE PATHS AS PER YOUR PROJECT
BASE_MODEL = "./models/base/defog_sqlcoder-7b-2"
LORA_MODEL = "./models/fine_tuned/sqlcoder-7b-2-qlora"
OUTPUT_DIR = "./models/final/sqlcoder-7b-2-merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("✅ Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, LORA_MODEL)

print("✅ Merging adapter into base model...")
merged = model.merge_and_unload()

print("✅ Saving merged model...")
merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Done. Merged model saved at:", OUTPUT_DIR)
```

Install requirements (Windows venv):

```bash
pip install -U transformers peft accelerate bitsandbytes safetensors
```

Run merge:

```bash
python scripts/merge_lora.py
```

After success, merged model will exist at:

```
./models/final/sqlcoder-7b-2-merged/
```

---

## 4) Install llama.cpp in WSL2 (for GGUF conversion)

In WSL2:

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip3 install -r requirements.txt
```

---

## 5) Copy merged model (Windows → WSL2)

Create target folder in WSL2:

```bash
mkdir -p ~/models/sqlcoder-merged
```

Copy from Windows drive.

Example if your project is inside **E:\**:

```bash
cp -r /mnt/e/sql-slm-project/sql-slm-project/models/final/sqlcoder-7b-2-merged/* ~/models/sqlcoder-merged/
```

Verify in WSL2:

```bash
ls -lh ~/models/sqlcoder-merged
```

You should see files like:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- etc.

---

## 6) Convert HF merged model → GGUF (WSL2)

Go to llama.cpp directory:

```bash
cd ~/llama.cpp
```

Convert model:

✅ Recommended quantization for 8GB GPU:

```bash
python3 convert_hf_to_gguf.py ~/models/sqlcoder-merged --outtype q4_k_m
```

Check output:

```bash
ls -lh ~/models/sqlcoder-merged/*.gguf
```

Example output:

```
ggml-model-q4_k_m.gguf
```

---

## 7) Create Ollama Model (WSL2)

Create a folder for Ollama model files:

```bash
mkdir -p ~/ollama/sqlcoder-domain
cd ~/ollama/sqlcoder-domain
```

Copy GGUF file:

```bash
cp ~/models/sqlcoder-merged/*.gguf ./model.gguf
```

Create `Modelfile`:

```bash
cat > Modelfile <<'EOF'
FROM ./model.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """
You are a PostgreSQL SQL expert.
Return ONLY SQL.
No explanation.
No markdown.
"""
EOF
```

Build Ollama model:

```bash
ollama create sqlcoder-domain -f Modelfile
```

Check model list:

```bash
ollama list
```

---

## 8) Run & Test Model

Run:

```bash
ollama run sqlcoder-domain
```

Test prompt:

```text
### Database Schema:
Table bookings(book_ref, book_date, total_amount)
Table tickets(ticket_no, book_ref, passenger_name)

### Question:
Show latest 10 bookings with passenger name.

### SQL:
```

---

## 9) Test Ollama using API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "sqlcoder-domain",
  "prompt": "Generate PostgreSQL SQL to show last 10 bookings"
}'
```

---

## 10) Troubleshooting

### A) LoRA not applied
✅ You converted base model only (wrong)

Fix:
1) merge adapter into base model
2) convert merged model to GGUF
3) recreate ollama model

---

### B) Model slow
Use smaller context:

In `Modelfile`:

```md
PARAMETER num_ctx 2048
```

Then rebuild:

```bash
ollama create sqlcoder-domain -f Modelfile
```

---

### C) GGUF not created
Make sure you are inside llama.cpp folder:

```bash
cd ~/llama.cpp
ls convert_hf_to_gguf.py
```

---

## ✅ Final Summary

✅ Train adapter (QLoRA)  
✅ Merge adapter with base model (Windows)  
✅ Copy merged model to WSL2  
✅ Convert to GGUF (llama.cpp)  
✅ Create model in Ollama  
✅ Run and use it
