# SQL SLM Project 🤖

**Build Your Own Small Language Model for Database Queries**

A complete, automated system for creating a domain-specific Small Language Model (SLM) specialized in generating SQL queries from natural language. Perfect for reporting and analytics!

---

## 📋 **What This Project Does**

✅ Extracts complete database schema automatically  
✅ Collects and processes historical SQL queries  
✅ Downloads and fine-tunes specialized AI models  
✅ Generates accurate SQL from natural language questions  
✅ Works with PostgreSQL, MySQL, SQL Server, Oracle, SQLite  
✅ Runs on local hardware (32-64GB RAM, 8GB GPU minimum)

---

## 🚀 **Quick Start (5 Steps)**

### Step 1: Clone and Setup
```bash
git clone <your-repo>
cd sql-slm-project
./setup.sh
```

### Step 2: Configure Database
Edit `.env` file with your database credentials:
```bash
nano .env
```

### Step 3: Extract Schema
```bash
python scripts/schema_extractor.py
```

### Step 4: Download Model
```bash
python scripts/download_model.py --model sqlcoder --verify
```

### Step 5: Collect Training Data
```bash
# Create sample data to start
python scripts/data_collector.py --source sample

# Or load from CSV
python scripts/data_collector.py --source csv --file your_queries.csv
```

---

## 📁 **Project Structure**

```
sql-slm-project/
├── scripts/              # Main automation scripts
│   ├── schema_extractor.py    # Extract DB schema
│   ├── download_model.py      # Download base models
│   ├── data_collector.py      # Collect training data
│   └── train_model.py         # Fine-tune model
├── src/                  # Source code
│   ├── data_processing/
│   ├── training/
│   ├── inference/
│   └── evaluation/
├── data/                 # Data storage
│   ├── raw/             # Raw query files
│   ├── processed/       # Processed datasets
│   └── schemas/         # Database schemas
├── models/              # Model storage
│   ├── base/           # Downloaded base models
│   ├── fine_tuned/     # Your trained models
│   └── checkpoints/    # Training checkpoints
├── docs/               # Documentation
├── logs/               # Log files
├── .env.example        # Configuration template
├── requirements.txt    # Python dependencies
└── setup.sh           # Automated setup
```

---

## 🗄️ **Supported Databases**

| Database | Tested | Auto-Detection |
|----------|--------|----------------|

| MySQL | ✅ | ✅ |
| SQL Server | ✅ | ✅ |
| SQLite | ✅ | ✅ |

---

## 🤖 **Available Base Models**

| Model | Size | Best For | Recommended |
|-------|------|----------|-------------|
| SQLCoder 7B | 7B params | Text-to-SQL | ⭐⭐⭐ |
| CodeLlama 7B | 7B params | General SQL | ⭐⭐⭐ |
| Mistral 7B | 7B params | Instruction following | ⭐⭐ |
| Phi-3 Mini | 3.8B params | Fast inference | ⭐⭐ |
| DeepSeek Coder | 6.7B params | Code generation | ⭐⭐ |

**Recommended:** Start with SQLCoder 7B

---

## 💻 **System Requirements**

### Minimum Requirements
- **RAM:** 32 GB
- **GPU:** 8 GB VRAM (NVIDIA with CUDA)
- **Storage:** 50 GB free space
- **OS:** Linux (Ubuntu 20.04+), Windows with WSL2

### Recommended Requirements
- **RAM:** 64 GB
- **GPU:** 16-24 GB VRAM (RTX 4090, A100, etc.)
- **Storage:** 100 GB SSD
- **OS:** Ubuntu 22.04 LTS

---

## 📖 **Detailed Documentation**

See **DOCUMENTATION.docx** for:
- Complete step-by-step guide
- Beginner-friendly explanations
- Troubleshooting tips
- Advanced configurations
- Best practices

---

## 🎯 **Usage Examples**

### Extract Database Schema
```bash
# Automatically detects database type from .env
python scripts/schema_extractor.py

# Output: data/schemas/schema.json
```

### Download and Verify Model
```bash
# List available models
python scripts/download_model.py --list

# Download SQLCoder

# Download CodeLlama
python scripts/download_model.py --model codellama --verify
```

### Collect Training Data
```bash
# From CSV file
python scripts/data_collector.py --source csv --file queries.csv

# From database query log
python scripts/data_collector.py --source database --table query_history

# Create sample dataset
python scripts/data_collector.py --source sample
```

### Train Your Model
```bash
# Basic training
python scripts/train_model.py

# With custom settings
python scripts/train_model.py --epochs 5 --batch-size 8 --learning-rate 3e-4
```

---

## 🔧 **Configuration**

### Database Configuration (.env)
```env
DB_TYPE=postgresql
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=your_database
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

### Model Configuration
```env
BASE_MODEL_NAME=defog/sqlcoder-7b-2
LORA_R=16
LORA_ALPHA=32
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=4
```

---

## 📊 **Expected Performance**

With proper training data (1000+ examples):
- **Accuracy:** 95%+ on simple queries
- **Accuracy:** 85%+ on complex queries
- **Inference Time:** <2 seconds per query
- **Model Size:** ~4-5 GB (4-bit quantized)

---

## 🤝 **Contributing**

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📝 **License**

MIT License - See LICENSE file for details

---

## 🆘 **Support**

- 📖 Read DOCUMENTATION.docx
- 🐛 Report issues on GitHub
- 💬 Community discussions

---

## 🙏 **Acknowledgments**

Built using:
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [SQLCoder by Defog.ai](https://github.com/defog-ai/sqlcoder)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)

---

## 🦙 Export Fine-Tuned Model to Ollama (GGUF)

To use your fine-tuned model in Ollama (WSL2/Docker), follow these steps:

### 1. Merge LoRA Adapter with Base Model

Create scripts/merge_lora.py:

```python
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
```

Run:
```sh
python scripts/merge_lora.py
```

### 2. Convert the Merged Model to GGUF

Clone llama.cpp and use its convert.py:

```sh
drive root dir i.e. E:\
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
python -m venv venv
pip install -r requirements.txt
python convert_hf_to_gguf.py E:/sql-slm-project/sql-slm-project/models/merged/sqlcoder-7b-2-merged --outfile E:/sql-slm-project/sql-slm-project/models/merged/sqlcoder-7b-2-merged.gguf
```

### 3. Copy GGUF Model to Ollama Environment

Move the .gguf file to a directory accessible by your Ollama Docker/WSL2 instance.

### 4. Create a Modelfile

```
FROM ./sqlcoder-7b-2-merged.gguf
PARAMETER temperature 0.1
```

### 5. Build and Run in Ollama

```sh
ollama create my-sql-model -f Modelfile
ollama run my-sql-model
```

---

**Happy SQL Generation! 🚀**




cp E:/sql-slm-project/sql-slm-project/models/merged/*.gguf ./model.gguf