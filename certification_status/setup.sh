#!/usr/bin/env bash
# One-shot setup for the certification-status pipeline.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing Python dependencies"
python3 -m pip install -q -r requirements.txt

echo "==> Installing vendored browser-harness (editable)"
python3 -m pip install -q -e ./browser-harness

if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Ollama not found. Install it from https://ollama.com/download"
  echo "    Linux one-liner:  curl -fsSL https://ollama.com/install.sh | sh"
else
  MODEL="${LLM_MODEL:-qwen3:14b}"
  echo "==> Pulling local model: $MODEL (change with LLM_MODEL=...)"
  ollama pull "$MODEL"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example"
fi

echo "==> Extracting company list from data/GNEM_Excel_Data.xlsx"
python3 scripts/extract_companies.py

cat <<'EOF'

Setup complete. Next steps:
  1. Start the LLM server:            ollama serve      (usually already running)
  2. Smoke test on 3 companies:       python3 scripts/check_certifications.py --limit 3 --backend http
  3. Full run with real Chrome:       python3 scripts/check_certifications.py --backend browser
     (Chrome needs remote debugging:  run `browser-harness --doctor` if it can't connect)

Results land in outputs/certification_status.csv
EOF
