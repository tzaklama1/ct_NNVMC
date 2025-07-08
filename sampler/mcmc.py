"""mcmc.py
Markov Chain Monte Carlo sampler for Fock-space configurations using JAX.

The sampler preserves particle number by proposing swaps between an
occupied and an empty lattice site. Acceptance is done using the
Metropolis-Hastings rule with probabilities derived from the provided
log_prob_fn.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable

# Type alias for clarity
PRNGKey = jax.Array
Array = jax.Array


def random_swap(state: Array, key: PRNGKey) -> Array:
    """Propose a new state by moving a particle to a random empty site."""
    occ_sites = jnp.where(state == 1)[0]
    emp_sites = jnp.where(state == 0)[0]
    # randomly choose occupied and empty indices
    key_occ, key_emp = jax.random.split(key)
    i = int(jax.random.choice(key_occ, occ_sites))
    j = int(jax.random.choice(key_emp, emp_sites))
    new_state = state.at[i].set(0).at[j].set(1)
    return new_state


def metropolis_step(state: Array,
                    key: PRNGKey,
                    log_prob_fn: Callable[[Array], Array]) -> tuple[Array, bool]:
    """Single Metropolis-Hastings update preserving particle number."""
    key_prop, key_acc = jax.random.split(key)
    proposal = random_swap(state, key_prop)

    logp_old = float(log_prob_fn(state))
    logp_new = float(log_prob_fn(proposal))

    accept = jnp.log(jax.random.uniform(key_acc)) < (logp_new - logp_old)
    new_state = jax.lax.select(accept, proposal, state)
    return new_state, bool(accept)


def sample_chain(init_state: Array,
                 log_prob_fn: Callable[[Array], Array],
                 key: PRNGKey,
                 n_samples: int,
                 burn_in: int = 100,
                 thin: int = 1) -> Array:
    """Run an MCMC chain and return sampled states.

    Parameters
    ----------
    init_state : Array
        Initial Fock-state occupation (0/1) of shape `(n_sites,)`.
    log_prob_fn : Callable
        Function returning `log(|psi(state)|^2)` for a given state.
    key : PRNGKey
        Source of randomness.
    n_samples : int
        Number of samples to return after burn-in and thinning.
    burn_in : int
        Number of initial steps to discard.
    thin : int
        Keep one sample every `thin` steps.
    """
    state = jnp.array(init_state, dtype=jnp.int32)
    samples = []
    k = key
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        k, subk = jax.random.split(k)
        state, _ = metropolis_step(state, subk, log_prob_fn)
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(state)
    return jnp.stack(samples)
