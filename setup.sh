#!/usr/bin/env bash
set -euo pipefail

# Parse optional --cpu-only flag.
# FORCE_CPU must be initialised before the loop (required by set -u).
FORCE_CPU=0
for arg in "$@"; do
    if [ "$arg" = "--cpu-only" ]; then
        FORCE_CPU=1
    fi
done

# Verify Python version (must be >= 3.10).
PYTHON_MINOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f2)
if [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ is required. Found: $(python3 --version)" >&2
    exit 1
fi

# Create .venv only if it does not already exist.
# NOTE: if .venv exists but is broken, this script will skip recreation.
# To fix a broken venv, run: rm -rf .venv/ && bash setup.sh
if [ -d .venv ]; then
    echo ".venv already exists — skipping creation."
else
    python3 -m venv .venv
fi

# Detect CUDA wheel tag.
if [ "$FORCE_CPU" -eq 1 ]; then
    TAG="cpu"
elif command -v nvidia-smi > /dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$DRIVER_VER" ]; then
        TAG="cpu"
    else
        DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
        if [ "$DRIVER_MAJOR" -ge 525 ]; then
            TAG="cu121"
        elif [ "$DRIVER_MAJOR" -ge 450 ]; then
            TAG="cu118"
        else
            TAG="cpu"
        fi
    fi
else
    TAG="cpu"
fi

echo "PyTorch wheel tag: $TAG"

# Install torch from the appropriate index.
if [ "$TAG" = "cpu" ]; then
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
elif [ "$TAG" = "cu118" ]; then
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu118
elif [ "$TAG" = "cu121" ]; then
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# Install remaining dependencies.
.venv/bin/pip install -r requirements.txt

# Smoke tests.
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
.venv/bin/python -c "from kaggle_environments import make; make('orbit_wars'); print('kaggle_environments: OK')"

echo "Setup complete. Activate with: source .venv/bin/activate"
