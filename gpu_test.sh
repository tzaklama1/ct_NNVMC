#!/usr/bin/env bash

python - <<'PY'
from train import run, jax
print("JAX sees devices:", jax.devices())
run(epochs=1000)      # short test
PY
