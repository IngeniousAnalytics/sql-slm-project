"""
Helper script to install PyTorch with CUDA support.
Detects system and installs appropriate version.
"""
import subprocess
import sys

print("="*80)
print("PyTorch CUDA Installation Helper")
print("="*80)

# Uninstall CPU version first
print("\nStep 1: Uninstalling CPU-only PyTorch...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

# Install CUDA version (CUDA 11.8 - most compatible)
print("\nStep 2: Installing PyTorch with CUDA 11.8...")
print("This will download ~2-3 GB of data. Please wait...")
cmd = [
    sys.executable, "-m", "pip", "install", 
    "torch==2.5.1", 
    "torchvision", 
    "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu118"
]
subprocess.run(cmd)

print("\n" + "="*80)
print("Installation complete!")
print("="*80)
print("\nVerifying CUDA availability...")
subprocess.run([sys.executable, "-c", 
    "import torch; print('CUDA available:', torch.cuda.is_available()); "
    "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"])
print("="*80)
