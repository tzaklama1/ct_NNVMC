#!/usr/bin/env bash
# run_demo.sh  –  make executable with:  chmod +x run_demo.sh
# -------------------------------------------------------------------------
#  * creates ./cotrain_env  (if absent)
#  * installs JAX+Flax+Optax
#  * runs demo_wavefunction.py which trains for a few seconds and
#    prints 10 complex amplitudes of the learnt wave-function
# -------------------------------------------------------------------------
set -e

echo ">>> Setting up Python venv (cotrain_env)"
if [ ! -d "cotrain_env" ]; then
  python -m venv cotrain_env
fi
source cotrain_env/bin/activate

pip install --quiet --upgrade pip
# ------- choose CPU build by default; swap in CUDA build if you have a GPU
pip install --quiet "jax[cpu]" flax optax chex

echo ">>> Launching quick demo"
python wavefunction.py
