"""Local Hamiltonian based energy evaluation for lattice wavefunctions.

This module implements the local energy expression using only the few
configurations that are connected to a given Fock state by the local
Hamiltonian terms.  It avoids constructing the full Hamiltonian matrix,
which rapidly becomes intractable as the lattice size grows.  The
implementation follows the prescription of Eq. (5.13) in
Sorella & Becca, *Quantum Monte Carlo Approaches for Correlated Systems*.
"""

from __future__ import annotations

from typing import Sequence, Tuple, List

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Utility helpers (mirrored from energy.py)
# ---------------------------------------------------------------------------

def _to_complex(arr: jnp.ndarray) -> jnp.ndarray:
    """Convert network output to a complex tensor.

    Networks sometimes return a real tensor of shape ``(..., 2)``
    representing the real and imaginary parts.  This helper converts such
    arrays to a complex dtype.  Arrays that are already complex are
    returned unchanged.
    """
    if jnp.iscomplexobj(arr):
        return arr
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(jnp.float32)


def wavefunction_amplitudes(model, occ: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Evaluate complex wavefunction amplitudes for ``occ``."""
    psi = model(occ, params)
    return _to_complex(psi)


def _hop_sign(occ: jnp.ndarray, i: int, j: int) -> int:
    """Fermionic sign for hopping from site ``j`` to ``i``.

    The sign follows the convention used in the exact diagonalisation
    utilities in this repository.  It counts the number of particles to
    the left of each site in the chosen ordering.
    """
    # number of occupied sites to the left of j and i
    n_left_j = int(jnp.sum(occ[:j]))
    n_left_i = int(jnp.sum(occ[:i]))
    n_count = n_left_j + n_left_i
    hop_sign = n_count if j >= i else n_count + 1
    return 1 if hop_sign % 2 == 0 else -1


# ---------------------------------------------------------------------------
#  Local energy
# ---------------------------------------------------------------------------

def local_energy(
    model,
    occ_batch: jnp.ndarray,
    params_batch: jnp.ndarray,
    hopping_terms: Sequence[Tuple[complex, Tuple[int, int]]],
    repulsion_pairs: Sequence[Tuple[int, int]] | None = None,
    V: float = 0.0,
) -> jnp.ndarray:
    """Return local energies for a batch of configurations.

    Parameters
    ----------
    model:
        Wavefunction model.  Should accept ``(occ, params)`` and return
        complex amplitudes or real/imag pairs.
    occ_batch:
        ``(B, N)`` tensor of sampled occupations.
    params_batch:
        ``(B, P)`` tensor with Hamiltonian parameters for each sample.
    hopping_terms:
        Iterable of ``(t, (i, j))`` entries specifying hopping amplitudes
        and the site indices they connect.  Each pair corresponds to a
        term ``t (c^\dagger_i c_j + c^\dagger_j c_i)``.
    repulsion_pairs:
        Optional list of site pairs ``(i, j)`` entering a density-density
        interaction ``V n_i n_j``.
    V:
        Interaction strength used with ``repulsion_pairs``.
    """
    B = occ_batch.shape[0]
    energies: List[jnp.ndarray] = []

    for b in range(B):
        occ = occ_batch[b]
        params = params_batch[b:b + 1]

        psi_sigma = wavefunction_amplitudes(model, occ[None, :], params)[0]

        # ---- diagonal contributions (repulsion) ----
        E_diag = 0.0
        if repulsion_pairs is not None and V != 0.0:
            for i, j in repulsion_pairs:
                E_diag += V * float(occ[i] * occ[j])

        # ---- off-diagonal hopping contributions ----
        new_occ = []
        coeffs = []
        for t, (i, j) in hopping_terms:
            # hop j -> i
            if occ[j] == 1 and occ[i] == 0:
                occ_prime = occ.copy()
                occ_prime = occ_prime.at[i].set(1)
                occ_prime = occ_prime.at[j].set(0)
                sgn = _hop_sign(occ, i, j)
                new_occ.append(occ_prime)
                coeffs.append(t * sgn)
            # hop i -> j
            if occ[i] == 1 and occ[j] == 0:
                occ_prime = occ.copy()
                occ_prime = occ_prime.at[i].set(0)
                occ_prime = occ_prime.at[j].set(1)
                sgn = _hop_sign(occ, j, i)
                new_occ.append(occ_prime)
                coeffs.append(t * sgn)

        if new_occ:
            occ_neighbors = jnp.stack(new_occ)
            params_neighbors = jnp.broadcast_to(params, (len(new_occ), params.shape[-1]))
            psi_neighbors = wavefunction_amplitudes(model, occ_neighbors, params_neighbors)
            E_off = jnp.dot(jnp.array(coeffs, dtype=psi_neighbors.dtype), psi_neighbors) / psi_sigma
        else:
            E_off = 0.0

        energies.append(E_diag + E_off)

    return jnp.stack(energies)


def expectation_local_energy(
    model,
    occ_batch: jnp.ndarray,
    params_batch: jnp.ndarray,
    hopping_terms: Sequence[Tuple[complex, Tuple[int, int]]],
    repulsion_pairs: Sequence[Tuple[int, int]] | None = None,
    V: float = 0.0,
) -> jnp.ndarray:
    """Monte-Carlo estimate of the total energy."""
    E_loc = local_energy(
        model,
        occ_batch,
        params_batch,
        hopping_terms,
        repulsion_pairs,
        V,
    )
    return jnp.mean(jnp.real(E_loc))
