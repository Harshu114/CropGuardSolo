#!/bin/bash
# CropGuard API startup script

# Model path.
#
# This must default to the v5 11-class checkpoint. It previously defaulted to
# cropguard_model_final.pt (the old 16-class v2 model), and because this script
# exports MODEL_PATH it OVERRODE the default in main.py -- launching via
# start.sh would have served the wrong model no matter what main.py said.
API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-$API_DIR/cropguard_model_v5.pt}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model file not found at $MODEL_PATH"
    echo "The v5 checkpoint ships as cropguard_model_v5.pt next to this script."
    echo ""
    echo "Or set MODEL_PATH environment variable to the correct location."
    echo ""
fi

# Start the server
echo "Starting CropGuard API on port 8000..."
python main.py