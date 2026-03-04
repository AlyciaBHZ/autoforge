#!/usr/bin/env bash
set -euo pipefail

echo "=== AutoForge Setup ==="
echo ""

# ── Check Python ──
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.11+ required"
    echo "Install from: https://python.org"
    exit 1
fi

echo "Python: $($PYTHON_CMD --version)"

# ── Create virtual environment ──
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "Virtual environment: .venv"

# ── Install AutoForge ──
echo "Installing AutoForge..."
pip install -q --upgrade pip
pip install -q -e .

# ── Create workspace ──
mkdir -p workspace

# ── Check Docker (optional) ──
echo ""
if command -v docker &>/dev/null; then
    echo "Docker: $(docker --version 2>/dev/null || echo 'found but version unknown')"
    if docker info &>/dev/null 2>&1; then
        echo "Docker is running"
        if [ -f "docker/Dockerfile.sandbox" ]; then
            echo "Building sandbox image..."
            if docker build -t autoforge-sandbox:latest -f docker/Dockerfile.sandbox docker/ >/dev/null 2>&1; then
                echo "Sandbox image built successfully"
                if ! grep -q "FORGE_DOCKER_ENABLED" .env 2>/dev/null; then
                    echo "FORGE_DOCKER_ENABLED=true" >> .env
                fi
            else
                echo "Warning: Docker build failed. Sandbox will use subprocess mode."
            fi
        fi
    else
        echo "Warning: Docker found but not running. Sandbox will use subprocess mode."
    fi
else
    echo "Docker: not found (optional). Sandbox will use subprocess mode."
fi

# ── Verify installation ──
echo ""
echo "Verifying installation..."
.venv/bin/forgeai --help >/dev/null 2>&1 && echo "  ForgeAI CLI: OK" || echo "  ForgeAI CLI: FAILED"
.venv/bin/python -c "from autoforge.engine.orchestrator import Orchestrator; print('  Engine: OK')" 2>&1 || echo "  Engine: FAILED"

# ── Auto-launch setup wizard if needed ──
echo ""
if .venv/bin/python -c "from autoforge.cli.setup_wizard import needs_setup; exit(0 if needs_setup() else 1)" 2>/dev/null; then
    echo "First-time setup — launching configuration wizard..."
    echo ""
    .venv/bin/forgeai setup
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Usage:"
echo '  forgeai                                      # Interactive session'
echo '  forgeai generate "your prompt"               # Generate a project'
echo '  forgeai review ./my-project                  # Review existing code'
echo '  forgeai setup                                # Reconfigure settings'
echo ""
echo "Or install globally:"
echo '  pip install -e .    # Then "forgeai" works from anywhere'
echo ""
