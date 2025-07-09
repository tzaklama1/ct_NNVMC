# wavefunction.py
"""
Trains the LatticeTransFormer for a *single* (t,V) pair on synthetic targets,
then prints the first 10 complex coefficients ψ(n; t,V).

Swap OCC/TARGET with your exact-diagonalisation data for real physics runs.
"""
import jax, jax.numpy as jnp
from train import (                      # all imported from existing code
    OCC,                    # (H, N_sites)  integer array of Fock states
    create_state,           # helper that initialises model & optimiser
    train_step,             # one SGD step (jit-compiled)
    mse_fn,                 # MSE loss helper
    model                   # the LatticeTransFormer instance
)

print("Detected JAX backend →", jax.default_backend().upper(),
      "with", len(jax.devices()), "device(s)")

# -------- training set: choose a (t,V) and build batch ----------------------
t, V = 0.5, 4.0
TV_BATCH = jnp.tile(jnp.array([t, V], dtype=jnp.float32), (OCC.shape[0], 1))

# synthetic target coefficients (Re, Im)  –– replace with ED data
key    = jax.random.PRNGKey(123)
TARGET = jax.random.normal(key, (OCC.shape[0], 2)) * 0.1

# -------- quick training ----------------------------------------------------
state = create_state(jax.random.PRNGKey(0))

for epoch in range(1200):                
    state, loss = train_step(state, OCC, TV_BATCH, TARGET)
    if epoch % 300 == 0:
        print(f"epoch {epoch:4d} | MSE = {loss:.3e}")

# -------- wave-function output ---------------------------------------------
coeffs = model.apply(state.params, OCC, TV_BATCH, train=False)  # (H, 2)
print("\nFirst 10 coefficients for (t,V)=(", t, ",", V, "):")
for i in range(10):
    n_str = ''.join(str(int(b)) for b in OCC[i])
    re, im = map(float, coeffs[i])
    print(f"|{n_str}⟩  →  {re:+.5f}  {im:+.5f} i")
