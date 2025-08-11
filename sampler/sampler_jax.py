# sampler.py
"""
Metropolis–Hastings sampler for spinless fermion Fock states at fixed N.
Targets p(σ) ∝ |ψ(σ; λ)|^2 using your trained JAX model.

API:
  occ_batch = sample_occ_batch(
      model_apply, params, lam_vec, L, N,
      num_samples=4096, burn_in=1024, thin=4, n_chains=16, rng_seed=0)

where
  • model_apply(params, occ, lam, train=False) -> (B, 2) [Re, Im]
  • lam_vec is a 1D array, e.g. [t, V, N]  or  [t, V]
  • returns occ_batch as int32 array of shape (num_samples, L)
"""

from typing import Callable, Tuple
import numpy as np
import jax
import jax.numpy as jnp


# ------------------------- utilities ----------------------------------------
def _random_occ(L: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Random bitstring with exactly N ones among L sites."""
    occ = np.zeros(L, dtype=np.int8)
    occ[rng.choice(L, size=N, replace=False)] = 1
    return occ

def _propose_exchange(occ: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """
    Propose a number-conserving move: pick one occupied site i and one empty site j and swap.
    Returns (new_occ, i, j).  If all 0s or 1s (should not happen), returns copy.
    """
    ones  = np.flatnonzero(occ == 1)
    zeros = np.flatnonzero(occ == 0)
    if len(ones) == 0 or len(zeros) == 0:
        return occ.copy(), -1, -1
    i = int(rng.choice(ones))
    j = int(rng.choice(zeros))
    new_occ = occ.copy()
    new_occ[i] = 0
    new_occ[j] = 1
    return new_occ, i, j

@jax.jit
def _coeffs_to_probs(coeff_ri: jnp.ndarray) -> jnp.ndarray:
    """(B,2)->(B,) probabilities |ψ|^2 with small epsilon for stability."""
    psi = coeff_ri[:, 0] + 1j * coeff_ri[:, 1]
    return (jnp.abs(psi) ** 2 + 1e-30).real

def _eval_probs(model_apply, params, occ_batch: np.ndarray, lam_vec: jnp.ndarray) -> np.ndarray:
    """Evaluate |ψ|^2 for a batch of integer 0/1 arrays (np) at fixed λ."""
    occ_j = jnp.asarray(occ_batch, dtype=jnp.int32)
    lam_b = jnp.tile(lam_vec[None, :], (occ_j.shape[0], 1))
    coeff = model_apply(params, occ_j, lam_b, train=False)        # (B,2)
    probs = _coeffs_to_probs(coeff)                               # (B,)
    # IMPORTANT: return a WRITEABLE NumPy array
    return np.array(probs, dtype=np.float64, copy=True)


# ------------------------- main sampler -------------------------------------
def sample_occ_batch(model_apply: Callable,
                     params,
                     lam_vec: jnp.ndarray,
                     L: int,
                     N: int,
                     num_samples: int = 4096,
                     burn_in: int = 1024,
                     thin: int = 4,
                     n_chains: int = 16,
                     rng_seed: int = 0) -> jnp.ndarray:
    """
    Return an array of shape (num_samples, L) with entries in {0,1},
    distributed approximately as |ψ(σ;λ)|^2, using n_chains independent MH chains.
    """
    assert 0 <= N <= L
    rng = np.random.default_rng(rng_seed)

    # allocate per-chain state
    chains = np.stack([_random_occ(L, N, rng) for _ in range(n_chains)], axis=0)  # (C, L)
    probs  = _eval_probs(model_apply, params, chains, lam_vec)                    # (C,)
    if not probs.flags.writeable:
        probs = probs.copy()

    # how many samples per chain (ceil)
    per_chain = (num_samples + n_chains - 1) // n_chains
    samples = []

    total_steps = burn_in + thin * per_chain
    for step in range(total_steps):
        # Propose for all chains in parallel (Python level)
        props = []
        for c in range(n_chains):
            occ_new, _, _ = _propose_exchange(chains[c], rng)
            props.append(occ_new)
        props = np.stack(props, axis=0)                                # (C, L)

        probs_new = _eval_probs(model_apply, params, props, lam_vec)   # (C,)
        # MH accept
        accept_ratio = probs_new / (probs + 1e-300)
        accept = rng.random(n_chains) < np.minimum(1.0, accept_ratio)

        # update chains & probs
        chains[accept] = props[accept]
        probs[accept]  = probs_new[accept]

        # record (after burn-in) with thinning
        if step >= burn_in and ((step - burn_in) % thin == 0):
            samples.append(chains.copy())

    # collect & trim
    if len(samples) == 0:
        # edge case: too small total_steps
        return jnp.asarray(chains[:num_samples], dtype=jnp.int32)

    samples = np.concatenate(samples, axis=0)           # (~per_chain, C, L)
    samples = samples.reshape(-1, L)                    # (C*~, L)
    samples = samples[:num_samples]                     # exact M
    return jnp.asarray(samples, dtype=jnp.int32)
