#!/usr/bin/env bash
# in Bash: chmod +x run.sh
set -e

echo "=== Cotrained t-V Transformer test run (JAX) ==="
python - <<'PY'
from train_JAX import main
main()
PY
