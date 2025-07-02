"""Compute local energies for lattice wavefunctions.

This module provides helpers to evaluate the local energy

    E_loc(\sigma) = \frac{\langle \sigma | H | \Psi\rangle}{\Psi(\sigma)}

for discrete Fock states ``\sigma``.  The implementation follows the
conventions used in FermiNet and ziyanzzhu/HubbardNet where the full
Hamiltonian matrix is applied to the network generated wavefunction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _to_complex(arr: jnp.ndarray) -> jnp.ndarray:
    """Convert network output to a complex tensor.

    The networks in this repository sometimes return a real tensor of
    shape ``(..., 2)`` representing ``(Re, Im)`` components.  This helper
    converts such an array to ``complex``.  If the input already has a
    complex dtype it is returned unchanged.
    """
    if jnp.iscomplexobj(arr):
        return arr
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(jnp.float32)


def occ_to_mask(occ: jnp.ndarray) -> jnp.ndarray:
    """Convert occupation arrays to integer bit masks.

    Parameters
    ----------
    occ: ``(B, N)`` integer tensor with 0/1 occupations.
    """
    occ = occ.astype(jnp.int32)
    powers = (2 ** jnp.arange(occ.shape[-1])).astype(jnp.int32)
    return jnp.sum(occ * powers, axis=-1)


def build_state_index(basis_masks: Sequence[int]) -> Dict[int, int]:
    """Map occupation bit masks to their row index in the Hamiltonian."""
    return {int(m): i for i, m in enumerate(basis_masks)}


def wavefunction_amplitudes(model, occ: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the complex wavefunction amplitudes.

    Parameters
    ----------
    model:  callable ``model(occ, params)`` returning network outputs.
    occ:    ``(H, N)`` occupation tensor enumerating the Hilbert space.
    params: ``(H, P)`` Hamiltonian parameters broadcastable to ``occ``.
    """
    psi = model(occ, params)
    return _to_complex(psi)


# ---------------------------------------------------------------------------
#  Local energy
# ---------------------------------------------------------------------------

def local_energy(
    model,
    occ_batch: jnp.ndarray,
    params_batch: jnp.ndarray,
    H: jnp.ndarray,
    occ_basis: jnp.ndarray,
    state_index: Dict[int, int] | None = None,
) -> jnp.ndarray:
    """Return local energies for a batch of configurations.

    Parameters
    ----------
    model:         Wavefunction model.  Should accept ``(occ, params)`` and
                    return complex amplitudes or real/imag pairs.
    occ_batch:     ``(B, N)`` tensor of sampled occupations.
    params_batch:  ``(B, P)`` tensor with Hamiltonian parameters for each sample.
    H:             ``(H, H)`` complex Hamiltonian matrix expressed in the same
                    basis order as ``occ_basis``.
    occ_basis:     ``(H, N)`` tensor enumerating the full Hilbert space.
    state_index:   Optional ``{mask: index}`` mapping for ``occ_basis``.
    """

    if state_index is None:
        basis_masks = occ_to_mask(occ_basis).tolist()
        state_index = build_state_index(basis_masks)

    B = occ_batch.shape[0]
    energies: List[jnp.ndarray] = []
    for b in range(B):
        occ = occ_batch[b:b + 1]
        params = params_batch[b:b + 1]

        # wavefunction on the full basis for this parameter set
        params_full = jnp.broadcast_to(params, (len(occ_basis), params.shape[-1]))
        psi_all = wavefunction_amplitudes(model, occ_basis, params_full)

        mask = int(occ_to_mask(occ)[0])
        idx = state_index[mask]
        psi_sigma = psi_all[idx]

        # apply Hamiltonian row to |Psi>
        row = H[idx]
        E_num = jnp.vdot(row, psi_all)
        energies.append(E_num / psi_sigma)

    return jnp.stack(energies)


def expectation_local_energy(
    model,
    occ_batch: jnp.ndarray,
    params_batch: jnp.ndarray,
    H: jnp.ndarray,
    occ_basis: jnp.ndarray,
    state_index: Dict[int, int] | None = None,
) -> jnp.ndarray:
    """Monte-Carlo estimate of the total energy.

    This simply averages the local energies of ``occ_batch``.
    """
    E_loc = local_energy(model, occ_batch, params_batch, H, occ_basis, state_index)
    return jnp.mean(jnp.real(E_loc))

