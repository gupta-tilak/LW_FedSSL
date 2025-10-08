#!/bin/bash
# Setup and Installation Script for Enhanced LW-FedSSL

set -e  # Exit on error

echo "========================================="
echo "🚀 LW-FedSSL Enhanced Setup"
echo "========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if Python 3.8+
required_version="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "   ✅ pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt
echo "   ✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p data
echo "   ✅ Directories created"
echo ""

# Test imports
echo "🧪 Testing imports..."
python3 << EOF
import sys
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    
    import torchvision
    print(f"   ✅ TorchVision {torchvision.__version__}")
    
    import flwr
    print(f"   ✅ Flower {flwr.__version__}")
    
    import numpy as np
    print(f"   ✅ NumPy {np.__version__}")
    
    import matplotlib
    print(f"   ✅ Matplotlib {matplotlib.__version__}")
    
    print("   ✅ All core dependencies OK")
    
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)
EOF

echo ""

# Check CUDA availability
echo "🎮 Checking CUDA..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✅ CUDA version: {torch.version.cuda}")
else:
    print("   ℹ️  CUDA not available (CPU mode)")
EOF

echo ""

# Download CIFAR-10
echo "📊 Downloading CIFAR-10 dataset..."
python3 << EOF
from torchvision import datasets
import os

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

print("   Downloading training set...")
datasets.CIFAR10(root=data_dir, train=True, download=True)

print("   Downloading test set...")
datasets.CIFAR10(root=data_dir, train=False, download=True)

print("   ✅ CIFAR-10 downloaded")
EOF

echo ""

# Verify installation
echo "✅ Verifying installation..."
python3 << EOF
from config import CONFIG
from telemetry import get_telemetry
from metrics import MetricsTracker
from client_selector import get_selector
from data_utils import get_cifar10_partitioned

print("   ✅ Configuration loaded")
print("   ✅ Telemetry system OK")
print("   ✅ Metrics tracker OK")
print("   ✅ Client selector OK")
print("   ✅ Data utilities OK")

print("\n   🎉 All components verified!")
EOF

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "📝 Quick Start:"
echo ""
echo "1. Local Simulation (10 clients):"
echo "   python simulate_clients.py --num-clients 10 --mode lwfedssl"
echo ""
echo "2. Start Server:"
echo "   python enhanced_server.py --mode lwfedssl"
echo ""
echo "3. Start Client:"
echo "   python enhanced_client.py --server localhost:8080 --client-id 0 --mode lwfedssl"
echo ""
echo "4. Visualize Results:"
echo "   python visualize.py --lwfedssl-session <session_id>"
echo ""
echo "📚 For more info, see README_ENHANCED.md"
echo ""
