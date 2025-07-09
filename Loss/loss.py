# loss.py
"""
Loss functions for co-trained lattice TNN
----------------------------------------
From the overleaf document:
• overlap_loss  Eq.(2) exact overlap 1 – |⟨Ψθ|Ψ_ED⟩|²
• amp_phase_loss Eq.(3) amplitude + discrete phase-gradient distance
"""
import jax, jax.numpy as jnp
from typing import Tuple
from jax import lax

# --------------------------------------------------------------------------- #
#   Multi-group overlap loss: several (t,V,N) wave-functions at once          #
# --------------------------------------------------------------------------- #
def overlap_loss_multi(pred:   jnp.ndarray,      # (B,2)
                       target: jnp.ndarray,      # (B,2)
                       gIDs:   jnp.ndarray       # (B,) int  group-id
                       ) -> jnp.ndarray:
    """
    Computes   L = 1 - mean_g |⟨Ψθ_g | Ψ_ED_g⟩|²
    for an arbitrary concatenation of batches that belong to
    different Hamiltonian parameter sets g = 0..G-1.
    """
    ψθ = _to_complex(pred)
    ψ  = _to_complex(target)

    num_groups = int(jnp.max(gIDs)) + 1
    eps        = 1e-12

    # segment sums over groups
    dot   = lax.segment_sum(jnp.conj(ψθ) * ψ,        gIDs, num_groups)
    nθ    = lax.segment_sum(jnp.abs(ψθ)**2,          gIDs, num_groups)
    nψ    = lax.segment_sum(jnp.abs(ψ) **2,          gIDs, num_groups)

    overlap = dot / (jnp.sqrt(nθ*nψ) + eps)          # (G,)
    loss_g  = 1.0 - jnp.abs(overlap)**2
    return jnp.mean(loss_g)                          # scalar


def loss_normDiff(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)

def _to_complex(arr: jnp.ndarray) -> jnp.ndarray:
    """arr[...,2] → complex"""
    return arr[...,0] + 1j * arr[...,1]

# --------------------------------------------------------------------------- #
# 1. Exact overlap loss                                                       #
# --------------------------------------------------------------------------- #
def overlap_loss(pred: jnp.ndarray,
                 target: jnp.ndarray) -> jnp.ndarray:
    """
    pred,target : (H,2)  real-imag coefficients for *one* λ
    returns scalar loss ≥ 0      minimise → max overlap
    """
    ψθ = _to_complex(pred)
    ψ  = _to_complex(target)
    # normalise
    ψθ /= jnp.linalg.norm(ψθ) + 1e-12
    ψ  /= jnp.linalg.norm(ψ)  + 1e-12
    overlap = jnp.vdot(ψθ, ψ)               # ⟨Ψθ | Ψ_ED⟩
    return 1.0 - jnp.abs(overlap)**2        # ≤ 1  … best = 0

# --------------------------------------------------------------------------- #
# 2. Amplitude + phase-gradient loss                                          #
# --------------------------------------------------------------------------- #
def _finite_diff(ψ: jnp.ndarray,
                 neighbours: jnp.ndarray) -> jnp.ndarray:
    """
    ψ : (H,) complex
    neighbours : (H, K) indices of K neighbour Fock states for each basis state
    returns (H, K) finite-difference values ψ(n') − ψ(n)
    """
    ψn  = ψ[:, None]
    ψnp = ψ[neighbours]                     # gather neighbours
    return ψnp - ψn

def amp_phase_loss(pred: jnp.ndarray,
                   target: jnp.ndarray,
                   neighbours: jnp.ndarray,
                   alpha: float = 1.0) -> jnp.ndarray:
    """
    pred,target : (H,2)  coefficients
    neighbours  : (H,K)  index array for finite-diff gradient (Eq.5)
    """
    ψθ = _to_complex(pred)
    ψ  = _to_complex(target)

    ampθ = jnp.abs(ψθ); amp  = jnp.abs(ψ)
    # --- amplitude term
    L_amp = jnp.mean((ampθ - amp) ** 2)

    # --- phase term (discrete gradient)
    gradθ = _finite_diff(jnp.angle(ψθ+1e-12), neighbours)
    grad  = _finite_diff(jnp.angle(ψ +1e-12), neighbours)
    L_phi = jnp.mean((gradθ - grad) ** 2)

    return L_amp + alpha * L_phi
