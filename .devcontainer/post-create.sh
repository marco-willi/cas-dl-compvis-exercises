#!/bin/bash
# Post-create command: Runs once after container is created

set -e

echo "Running post-create setup..."

# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Ensure we're in the workspace
cd /workspace

# Remove a stale/dangling .venv symlink (left over from older setups) so uv can create a real dir
if [ -L .venv ] && [ ! -e .venv ]; then
    echo "Removing dangling .venv symlink -> $(readlink .venv)"
    rm .venv
fi

# Create the in-repo .venv on top of the conda Python from the pytorch base image
uv venv --python "$(which python3)" --system-site-packages --allow-existing .venv
uv sync --group dev --python "$(which python3)"

# Venv is now in-project as .venv
VENV_REAL="/workspace/.venv"
VENV_ACTIVATE=$VENV_REAL/bin/activate

# Persist ~/.local/bin on PATH and auto-activate the venv in every new terminal session
for RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$RC" ]; then
        if ! grep -q 'HOME/.local/bin' "$RC"; then
            {
                echo ""
                echo "# Add user-local binaries (uv, CLI tools) to PATH"
                echo 'export PATH="$HOME/.local/bin:$PATH"'
            } >> "$RC"
        fi
        if ! grep -q "source $VENV_ACTIVATE" "$RC"; then
            {
                echo ""
                echo "# Auto-activate uv venv"
                echo "[ -f $VENV_ACTIVATE ] && source $VENV_ACTIVATE"
            } >> "$RC"
        fi
    fi
done

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo ""
    echo "Installing pre-commit hooks..."
    uv run pre-commit install || echo "Warning: Failed to install pre-commit hooks (continuing anyway)"
fi

# Install developer tools from GitHub releases (best-effort — failures are non-fatal)
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"


echo ""
echo "Post-create setup complete!"
echo "Venv   : /workspace/.venv -> $VENV_REAL"
echo "Python : $($VENV_REAL/bin/python --version)"
