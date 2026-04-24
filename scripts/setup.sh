#!/bin/bash
set -e

echo "🚀 Setting up Data Forecast Generator..."
echo ""

# Check nix
if ! command -v nix &> /dev/null; then
    echo "❌ nix not found. Please install Nix."
    exit 1
fi

echo "✓ Nix found: $(nix --version)"

# Update flake lock
echo "📦 Updating flake lock file..."
nix flake update

# Install dependencies
echo "📦 Installing Python dependencies..."
nix develop -c bash -c 'uv sync --extra dev'

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Enter dev environment: nix develop"
echo "  2. Copy .env.example to .env and add ANTHROPIC_API_KEY"
echo "  3. Run tests: pytest"
echo "  4. Run the pipeline as described in CURRENT_SYSTEM_DOCUMENTATION.md"
