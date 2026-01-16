#!/bin/bash

# ============================================
# SQL SLM Project - Setup Script
# ============================================
# This script sets up the complete environment

echo "========================================"
echo "SQL SLM PROJECT - AUTOMATED SETUP"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '3\.\d+')
if [ -z "$python_version" ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✓ Python $python_version found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies (this may take several minutes)..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠  IMPORTANT: Please edit .env file with your database credentials"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/{raw,processed,schemas}
mkdir -p models/{base,fine_tuned,checkpoints}
mkdir -p logs
mkdir -p outputs
echo "✓ Directories created"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠  No GPU detected. Training will be slower on CPU.")
EOF

echo ""
echo "========================================"
echo "SETUP COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your database credentials:"
echo "   nano .env"
echo ""
echo "2. Extract your database schema:"
echo "   python scripts/schema_extractor.py"
echo ""
echo "3. Download a base model:"
echo "   python scripts/download_model.py --model sqlcoder"
echo ""
echo "4. Collect training data:"
echo "   python scripts/data_collector.py --source sample"
echo ""
echo "For detailed instructions, see DOCUMENTATION.docx"
echo ""
