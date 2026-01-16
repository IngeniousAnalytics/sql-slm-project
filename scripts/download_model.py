"""
Model Downloader - Downloads and sets up base models for fine-tuning

This script:
1. Downloads recommended base models (SQLCoder, CodeLlama, etc.)
2. Verifies model files
3. Sets up local cache
4. Tests model loading and inference (SAFE for 8GB GPU using 4-bit)

Usage:
    python download_model.py --model sqlcoder
    python download_model.py --model codellama
    python download_model.py --model mistral
    python download_model.py --model phi3
    python download_model.py --model deepseek
    python download_model.py --list
    python download_model.py --model sqlcoder --verify --test
"""

import os
import sys
import argparse
import logging
import shutil
import json
import gc
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download, login

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Available models with metadata
AVAILABLE_MODELS = {
    'sqlcoder': {
        'name': 'defog/sqlcoder-7b-2',
        'description': 'SQLCoder 7B - Specialized for text-to-SQL (RECOMMENDED)',
        'size': '~14 GB',
        'parameters': '7B',
        'recommended': True
    },
    'codellama': {
        'name': 'codellama/CodeLlama-7b-Instruct-hf',
        'description': 'CodeLlama 7B Instruct - Excellent for SQL and code',
        'size': '~14 GB',
        'parameters': '7B',
        'recommended': True
    },
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'description': 'Mistral 7B Instruct - General purpose with good SQL',
        'size': '~14 GB',
        'parameters': '7B',
        'recommended': False
    },
    'phi3': {
        'name': 'microsoft/Phi-3-mini-4k-instruct',
        'description': 'Phi-3 Mini - Smaller, faster option',
        'size': '~8 GB',
        'parameters': '3.8B',
        'recommended': False
    },
    'deepseek': {
        'name': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'description': 'DeepSeek Coder 6.7B - Good for code generation',
        'size': '~13 GB',
        'parameters': '6.7B',
        'recommended': False
    }
}


class ModelDownloader:
    """Handles downloading and setting up base models"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.getenv('MODEL_CACHE_DIR', './models/base')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check for HuggingFace token
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if self.hf_token:
            logger.info("Using HuggingFace token from .env")
            login(token=self.hf_token)

    # -----------------------------
    # Utility
    # -----------------------------

    def _print_gpu_stats(self, title="GPU Stats"):
        """Print GPU memory stats if CUDA is available"""
        if not torch.cuda.is_available():
            logger.info(f"{title}: CUDA not available")
            return

        try:
            device_id = 0
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
            logger.info(
                f"{title}: GPU={torch.cuda.get_device_name(device_id)} | "
                f"Total={total:.1f}GB | Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB"
            )
        except Exception as e:
            logger.warning(f"Could not read GPU stats: {e}")

    def _cleanup_cuda(self):
        """Free CUDA memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_local_model_path(self, model_name: str) -> str:
        """Standardize local model path naming"""
        return os.path.join(self.cache_dir, model_name.replace('/', '_'))

    # -----------------------------
    # Main features
    # -----------------------------

    def list_available_models(self):
        """Display all available models"""
        print("\n" + "="*80)
        print("AVAILABLE BASE MODELS FOR SQL SLM")
        print("="*80 + "\n")

        for key, info in AVAILABLE_MODELS.items():
            marker = "⭐ RECOMMENDED" if info['recommended'] else ""
            print(f"Model ID: {key} {marker}")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']} | Parameters: {info['parameters']}")
            print()

        print("="*80)
        print("\nUsage:")
        print("  python download_model.py --model sqlcoder")
        print("  python download_model.py --model codellama")
        print("  python download_model.py --model sqlcoder --verify --test")
        print("="*80 + "\n")

    def check_disk_space(self, required_gb: int = 20):
        """Check if sufficient disk space is available (cross-platform)"""
        try:
            total, used, free = shutil.disk_usage(self.cache_dir)
            available_gb = free / (1024**3)
        except Exception as e:
            logger.warning(f"Could not determine disk space: {e}. Continuing by default.")
            return True

        if available_gb < required_gb:
            logger.warning(f"⚠ Low disk space: {available_gb:.1f} GB available, {required_gb} GB recommended")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == 'y'

        logger.info(f"✓ Sufficient disk space: {available_gb:.1f} GB available")
        return True

    def download_model(self, model_key: str, force: bool = False):
        """Download a specific model"""
        if model_key not in AVAILABLE_MODELS:
            logger.error(f"Unknown model: {model_key}")
            logger.info("Run with --list to see available models")
            return False

        model_info = AVAILABLE_MODELS[model_key]
        model_name = model_info['name']

        print("\n" + "="*80)
        print(f"DOWNLOADING MODEL: {model_key}")
        print("="*80)
        print(f"Name: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Size: {model_info['size']}")
        print(f"Cache Directory: {self.cache_dir}")
        print("="*80 + "\n")

        # Check disk space
        if not self.check_disk_space():
            logger.error("Insufficient disk space. Aborting.")
            return False

        model_path = self._get_local_model_path(model_name)

        # Check if already downloaded
        if os.path.exists(model_path) and not force:
            logger.info(f"✓ Model already exists at: {model_path}")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                logger.info("Using existing model")
                return True

        try:
            logger.info("Downloading model files... This may take several minutes.")
            logger.info("Progress will be shown below:\n")

            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )

            logger.info(f"\n✓ Model downloaded successfully to: {model_path}")
            return True

        except Exception as e:
            logger.error(f"✗ Download failed: {str(e)}")
            return False

    def verify_model(self, model_key: str):
        """Verify model can be loaded (4-bit safe for 8GB GPU)"""
        model_info = AVAILABLE_MODELS[model_key]
        model_name = model_info['name']
        model_path = self._get_local_model_path(model_name)

        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            return False

        print("\n" + "="*80)
        print("VERIFYING MODEL (SAFE LOAD MODE)")
        print("="*80 + "\n")

        self._print_gpu_stats("Before loading")

        try:
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("✓ Tokenizer loaded successfully")

            # ✅ Prefer 4-bit on GPU to fit 7B models
            if torch.cuda.is_available():
                logger.info("CUDA detected -> Loading model in 4-bit (recommended for 8GB GPUs)")

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                logger.info("CUDA not available -> Loading model on CPU (slow but works)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True
                )

            logger.info("✓ Model loaded successfully")

            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✓ Total parameters: {num_params:,}")

            test_text = "SELECT * FROM customers WHERE"
            tokens = tokenizer(test_text, return_tensors="pt")
            logger.info(f"✓ Test tokenization successful ({len(tokens['input_ids'][0])} tokens)")

            self._print_gpu_stats("After loading")

            print("\n" + "="*80)
            print("MODEL VERIFICATION SUCCESSFUL ✅")
            print("="*80)
            print(f"Model: {model_name}")
            print(f"Location: {model_path}")
            print(f"Parameters: {num_params:,}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU Name: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print("="*80 + "\n")

            # cleanup
            del model
            del tokenizer
            self._cleanup_cuda()
            self._print_gpu_stats("After cleanup")

            return True

        except Exception as e:
            logger.error(f"✗ Verification failed: {str(e)}")
            self._cleanup_cuda()
            return False

    def test_inference(self, model_key: str):
        """Test basic inference with the model (4-bit safe for 8GB GPU)"""
        model_info = AVAILABLE_MODELS[model_key]
        model_name = model_info['name']
        model_path = self._get_local_model_path(model_name)

        print("\n" + "="*80)
        print("TESTING MODEL INFERENCE (SAFE MODE)")
        print("="*80 + "\n")

        self._print_gpu_stats("Before inference load")

        try:
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Loading model...")

            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True
                )

            logger.info("✓ Model loaded")

            test_prompt = """### Database Schema:
Table: customers
  - customer_id (INT, PRIMARY KEY)
  - customer_name (VARCHAR)
  - email (VARCHAR)

### Question: Show me all customers

### SQL:"""

            logger.info("Tokenizing prompt...")
            inputs = tokenizer(test_prompt, return_tensors="pt")

            # move inputs to the correct device
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            logger.info("Generating output...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.1,
                    do_sample=False,
                    use_cache=True
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("\n" + "-"*80)
            print("INPUT PROMPT:")
            print("-"*80)
            print(test_prompt)
            print("\n" + "-"*80)
            print("GENERATED OUTPUT:")
            print("-"*80)
            print(generated_text[len(test_prompt):].strip())
            print("-"*80 + "\n")

            logger.info("✓ Inference test completed successfully")

            self._print_gpu_stats("After inference")

            # cleanup
            del model
            del tokenizer
            self._cleanup_cuda()
            self._print_gpu_stats("After cleanup")

            return True

        except Exception as e:
            logger.error(f"✗ Inference test failed: {str(e)}")
            self._cleanup_cuda()
            return False

    def setup_for_training(self, model_key: str):
        """Prepare model for fine-tuning"""
        model_info = AVAILABLE_MODELS[model_key]
        model_name = model_info['name']
        model_path = self._get_local_model_path(model_name)

        print("\n" + "="*80)
        print("SETTING UP MODEL FOR TRAINING")
        print("="*80 + "\n")

        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding="utf-8") as f:
                lines = f.readlines()

            updated_base_name = False
            updated_cache_dir = False

            for i, line in enumerate(lines):
                if line.startswith('BASE_MODEL_NAME='):
                    lines[i] = f'BASE_MODEL_NAME={model_name}\n'
                    updated_base_name = True
                if line.startswith('MODEL_CACHE_DIR='):
                    lines[i] = f'MODEL_CACHE_DIR={self.cache_dir}\n'
                    updated_cache_dir = True

            if not updated_base_name:
                lines.append(f'BASE_MODEL_NAME={model_name}\n')
            if not updated_cache_dir:
                lines.append(f'MODEL_CACHE_DIR={self.cache_dir}\n')

            with open(env_file, 'w', encoding="utf-8") as f:
                f.writelines(lines)

            logger.info("✓ Updated .env file with model configuration")

        logger.info(f"✓ Model ready for training: {model_path}")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Extract your database schema:")
        print("   python scripts/schema_extractor.py")
        print()
        print("2. Collect historical queries:")
        print("   python scripts/data_collector.py")
        print()
        print("3. Start training:")
        print("   python scripts/train_model.py")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Download and setup base models for SQL SLM training'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to download (e.g., sqlcoder, codellama)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify downloaded model can be loaded'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test model inference'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if model exists'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Custom cache directory for models'
    )

    args = parser.parse_args()

    downloader = ModelDownloader(cache_dir=args.cache_dir)

    if args.list:
        downloader.list_available_models()
        return

    if not args.model:
        print("Error: Please specify a model with --model or use --list to see options")
        parser.print_help()
        return

    if downloader.download_model(args.model, force=args.force):

        if args.verify or args.test:
            if downloader.verify_model(args.model):

                if args.test:
                    downloader.test_inference(args.model)

        downloader.setup_for_training(args.model)


if __name__ == "__main__":
    main()
