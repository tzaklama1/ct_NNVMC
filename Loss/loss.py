# loss.py
"""
Loss functions for co-trained lattice TNN
----------------------------------------
• overlap_loss  Eq.(2) exact overlap 1 – |⟨Ψθ|Ψ_ED⟩|²
• amp_phase_loss Eq.(3) amplitude + discrete phase-gradient distance
"""
import jax, jax.numpy as jnp
from typing import Tuple

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

def loss_normDiff(pred: jnp.ndarray, target: jnp.ndarray):
    return jnp.mean((pred - target) ** 2)
