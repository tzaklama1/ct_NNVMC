"""GPU-optimized Metropolis-Hastings sampler implemented fully in JAX.

This module provides a drop-in replacement for :mod:`sampler_jax` that
runs efficiently on accelerators.  It avoids NumPy/Python loops and keeps
all data on-device using ``jax.numpy`` operations.

Example
-------
```
occ_batch = sample_occ_batch(model_apply, params, lam_vec,
                             L, N, num_samples=4096)
```
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


@jax.jit
def _coeffs_to_probs(coeff_ri: jnp.ndarray) -> jnp.ndarray:
    """Convert real/imag pairs to probabilities ``|psi|^2``."""
    psi = coeff_ri[..., 0] + 1j * coeff_ri[..., 1]
    return (jnp.abs(psi) ** 2 + 1e-30).real.astype(jnp.float32)


def sample_occ_batch(
    model_apply: Callable,
    params,
    lam_vec: jnp.ndarray,
    L: int,
    N: int,
    num_samples: int = 4096,
    burn_in: int = 1024,
    thin: int = 4,
    n_chains: int = 16,
    rng_seed: int = 0,
) -> jnp.ndarray:
    """Sample occupation configurations on GPU.

    Parameters
    ----------
    model_apply:
        Callable implementing ``model_apply(params, occ, lam, train=False)``.
    params:
        PyTree of model parameters (kept on device).
    lam_vec:
        1D array of Hamiltonian parameters.  This and all model inputs are
        promoted to 32-bit JAX arrays to ensure compatibility with GPUs.
    L, N:
        System size and particle number.
    num_samples, burn_in, thin, n_chains, rng_seed:
        Standard Metropolis–Hastings hyperparameters.

    Returns
    -------
    ``(num_samples, L)`` integer array of sampled occupations.
    """

    assert 0 <= N <= L

    key = jax.random.PRNGKey(rng_seed)
    lam_vec = jnp.asarray(lam_vec, dtype=jnp.float32)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)

    # initialise independent chains with random occupations
    def init_chain(k):
        perm = jax.random.permutation(k, L)
        occ = jnp.zeros((L,), dtype=jnp.int32)
        occ = occ.at[perm[:N]].set(1)
        return occ

    key, subkey = jax.random.split(key)
    init_keys = jax.random.split(subkey, n_chains)
    chains = jax.vmap(init_chain)(init_keys)  # (C, L)

    # probabilites |psi|^2 for a batch of chains
    def eval_probs(occ_batch: jnp.ndarray) -> jnp.ndarray:
        lam_b = jnp.broadcast_to(lam_vec, (occ_batch.shape[0], lam_vec.shape[0]))
        coeff = model_apply(params, occ_batch, lam_b, train=False)
        return _coeffs_to_probs(coeff)

    eval_probs = jax.jit(eval_probs)
    probs = eval_probs(chains)

    per_chain = (num_samples + n_chains - 1) // n_chains
    total_steps = burn_in + thin * per_chain

    idx = jnp.arange(n_chains)

    def mcmc_step(carry, k):
        chains, probs = carry
        k1, k2, k3 = jax.random.split(k, 3)
        i = jax.random.randint(k1, (n_chains,), 0, L)
        j = jax.random.randint(k2, (n_chains,), 0, L)

        occ_i = jnp.take_along_axis(chains, i[:, None], axis=1).squeeze(-1)
        occ_j = jnp.take_along_axis(chains, j[:, None], axis=1).squeeze(-1)

        swap = (i != j) & (occ_i != occ_j)

        temp = chains
        temp = temp.at[idx, i].set(occ_j)
        temp = temp.at[idx, j].set(occ_i)
        chains_prop = jax.lax.select(swap[:, None], temp, chains)

        probs_new = eval_probs(chains_prop)
        ratio = probs_new / probs
        accept = jax.random.uniform(k3, (n_chains,)) < jnp.minimum(1.0, ratio)

        chains = jax.lax.select(accept[:, None], chains_prop, chains)
        probs = jax.lax.select(accept, probs_new, probs)
        return (chains, probs), chains

    keys = jax.random.split(key, total_steps)
    (chains, probs), history = jax.lax.scan(mcmc_step, (chains, probs), keys)

    samples = history[burn_in::thin].reshape(-1, L)
    samples = samples[:num_samples]
    return samples.astype(jnp.int32)
