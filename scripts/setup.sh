#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Data Forecast Generator..."
echo ""

if ! command -v docker &> /dev/null; then
    echo "docker not found. Install Docker or Podman with Docker-compatible compose."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "docker compose not found. Install Docker Compose v2."
    exit 1
fi

echo "docker found: $(docker --version)"
echo "compose found: $(docker compose version)"

echo "Preparing runtime directories..."
mkdir -p output artifacts/ui_uploads

for path in output artifacts artifacts/ui_uploads; do
    if [[ ! -w "$path" ]]; then
        echo "Directory '$path' is not writable by the current user."
        echo "Fix ownership, then rerun setup:"
        echo "  sudo chown -R \"\$USER\":\"\$USER\" artifacts output"
        exit 1
    fi
done

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Add your GH_TOKEN before running the app."
fi

echo "Building Docker image..."
docker compose build

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Add GH_TOKEN to .env"
echo "  2. Start the app: docker compose up"
echo "  3. Open http://localhost:8501 for training and http://localhost:8502 for inference"
