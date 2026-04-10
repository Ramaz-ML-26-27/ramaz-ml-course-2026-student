#!/usr/bin/env bash
# Run this whenever a new assignment is released:
#   bash setup.sh
#
# It pulls the latest changes and installs dependencies for any
# new or updated assignment directories.

set -e

echo "Pulling latest assignments..."
git pull

echo "Installing dependencies..."
find . -name "pyproject.toml" \
    -not -path "*/.venv/*" \
    -not -path "*/node_modules/*" \
    -execdir uv sync \;

echo "Done! Check for new folders in the repo."
