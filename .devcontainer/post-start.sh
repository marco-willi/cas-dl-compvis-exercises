#!/bin/bash
# Post-start command: Runs every time the container starts

set -e

echo "Running post-start tasks..."

# Ensure uv is on PATH (installed during post-create)
export PATH="$HOME/.local/bin:$PATH"

# Install CLI tools
bash "$(dirname "$0")/cli-tools.sh"

echo ""
echo "Python : $(which python3) ($(python3 --version 2>&1))"
echo "uv     : $(uv --version)"
echo "Venv   : /workspace/.venv"

echo ""
echo "Container ready!"
