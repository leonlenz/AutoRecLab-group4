#!/bin/bash
set -e

cd /app && uv run python -m cli.embeddings.main generate --all

echo ""
echo "Embeddings generated successfully!"
echo ""

exec "$@"
