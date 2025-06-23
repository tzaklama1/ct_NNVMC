#!/usr/bin/env bash
# Make the script executable:  chmod +x run.sh
set -e

echo "=== Cotrained t-V Transformer demo (JAX) ==="
python - <<'PY'
from train import main
main()
PY
