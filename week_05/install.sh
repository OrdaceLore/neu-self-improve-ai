#!/bin/bash

# Installation script for PAG with A*-PO
echo "Installing PAG with A*-PO system..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Install/upgrade torch to version 2.6+
echo "Installing/upgrading PyTorch to version 2.6+..."
pip install torch>=2.6.0 --upgrade

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Installation completed!"
echo "You can now run: python test_system.py"
