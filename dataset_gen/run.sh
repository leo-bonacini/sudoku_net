#!/usr/bin/env bash
#
# Sets up and runs the Sudoku dataset generator.
#
# Usage:
#   ./run.sh --count 100000 --out ./output --concurrency 8 --resume
#
# Any arguments are forwarded as-is to generate.js (see `node generate.js --help`).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d node_modules ]; then
  echo "Installing npm dependencies..."
  npm install
fi

if [ ! -d "$HOME/.cache/ms-playwright" ]; then
  echo "Installing Playwright's Chromium browser..."
  npx playwright install chromium --with-deps
fi

node generate.js "$@"
