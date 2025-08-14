# main_run.py
"""
Tests model for interacting 1d chain of fermions and beyond.

This file is the main file from which all experiments are run.
"""
import jax, jax.numpy as jnp, optax
from flax.training import train_state
import Helper.helper_module as helper


"""
Latest version of the code to train a Lattice-TransFormer on synthetic data for Interacting fermions in 1D
according to t-V model.
"""
# ---------------------------------------------------------------------------
# 1)  Initialize and set parameters
# ---------------------------------------------------------------------------


# --- hyperparameters -------------------------------------------------------
DEPTH = 8
WIDTH = 256
# --- input/output ----------------------------------------------------------
dir = "/Users/zaklama/VS_Code/Physics_Classes/Python/ct_NNVMC/"
out_dir = dir + "Data/run_output"
# --- lattice ---------------------------------------------------------------
LATTICE   = "1d"     # "1d" or "honeycomb"
N_SITES   = 12        # total number of sites (1d ⇒ chain length L = N_SITES)

# --- Hamiltonian parameters ------------------------------------------------
T_LIST = [1.0, 1.0, 0.4, 0.5, 0.8]       # nearest-neighbour hopping      (t)
V_LIST = [1.0, 5.0, 1.5, 0.2, 0.2]       # nearest-neighbour repulsion    (V)
N_LIST = [4, 8, 10, 6, 4]                # particle numbers (can all differ)  

# --- optimisation ----------------------------------------------------------
LOSS_TYPE = "overlap_multi"     # "overlap_multi"  or  "amp_phase" or "original"
EPOCHS    = 200          # SGD steps in the demo script 
# ^ will set this to 4800 after I get it running fully
PRINT_EVERY = 300         # logging cadence (epochs)

# --- demo output -----------------------------------------------------------
N_PRINT   = 10       # how many coefficients to print at the end
SEED      = 0        # RNG seed


# ---------------------------------------------------------------------------
# 2)  Build concatenated training set
# ---------------------------------------------------------------------------


assert len(T_LIST)==len(V_LIST)==len(N_LIST)
G = len(T_LIST)              # number of Hamiltonians

OCC_ALL, LAM_ALL, TARGET_ALL, GID_ALL = [], [], [], []
for gid,(t,v,npart) in enumerate(zip(T_LIST, V_LIST, N_LIST)):
    # basis for this particle number --------------------
    if LATTICE == "1d":
        from phys_system.lattice1D import enumerate_fock, mask_to_array
        basis = enumerate_fock(N_SITES, npart)
        occ   = jnp.array([mask_to_array(m, N_SITES) for m in basis],
                          dtype=jnp.int32)
    else:
        import phys_system.honeycomb as hc
        basis = hc.enumerate_fock(N_SITES, npart)
        occ   = jnp.array([hc.mask_to_array(m, N_SITES) for m in basis],
                          dtype=jnp.int32)

    # λ-vector extended to include N --------------------
    lam_vec = jnp.array([t, v, npart], dtype=jnp.float32)
    lam     = jnp.tile(lam_vec, (len(basis),1))

    
    # synthetic target coefficients replaced by exact non-interacting ones
    #key  = jax.random.PRNGKey(C.SEED + gid)
    #targ = jax.random.normal(key, (len(basis), 2))*0.1
    coeffs, masks, _ = helper.my_readcsv(f"{dir}ED/output/gsWaveFn_Ns{int(N_SITES)}_Np{npart}_Vnn{round(v/t,2)}.csv")
    # TARGET aligned to OCC (Re, Im)
    targ = jnp.stack([jnp.array(coeffs.real), jnp.array(coeffs.imag)], axis=1)

    gid_vec = jnp.full((len(basis),), gid, dtype=jnp.int32)

    OCC_ALL.append(occ)
    LAM_ALL.append(lam)
    TARGET_ALL.append(targ)
    GID_ALL.append(gid_vec)

# concatenate everything --------------------------------
OCC     = jnp.concatenate(OCC_ALL,    axis=0)
LAM     = jnp.concatenate(LAM_ALL,    axis=0)   # shape (B,3)
TARGET  = jnp.concatenate(TARGET_ALL, axis=0)
GIDS    = jnp.concatenate(GID_ALL,    axis=0)   # shape (B,)

print("Total training states:", OCC.shape[0], "| Hamiltonians:", G)

# ---------------------------------------------------------------------------
# 3)  Define Model
# ---------------------------------------------------------------------------
from networks.model import LatticeTransFormer
model = LatticeTransFormer(n_sites=N_SITES, depth=DEPTH, d_model=WIDTH)

# ---------------------------------------------------------------------------
# 4)  Loss & training step
# ---------------------------------------------------------------------------
from Loss.loss import overlap_loss_multi, amp_phase_loss, loss_normDiff   

def create_state(rng):
    params = model.init(rng, OCC, LAM, train=False)
    tx     = optax.adam(1e-3)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

def loss_fn(params):
    preds = model.apply(params, OCC, LAM, train=False)
    if LOSS_TYPE == "overlap_multi":
        return overlap_loss_multi(preds, TARGET, GIDS, num_groups=G)
    elif LOSS_TYPE == "amp_phase":
        return amp_phase_loss(preds, TARGET, neighbours)
    elif LOSS_TYPE == "original":
        return loss_normDiff(preds, TARGET)
    else:
        raise ValueError("Unknown LOSS_TYPE")

@jax.jit
def train_step(state):
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# ---------------------------------------------------------------------------
# 5)  Run training loop
# ---------------------------------------------------------------------------
print("JAX backend:", jax.default_backend().upper(),
      "| lattice:", LATTICE,
      f"| N={N_SITES}, Hamiltonians={G}")

state = create_state(jax.random.PRNGKey(42))
for epoch in range(EPOCHS):
    state, loss = train_step(state)
    if epoch % PRINT_EVERY == 0:
        print(f"epoch {epoch:4d}  loss = {float(loss):.4e}")

# ---------------------------------------------------------------------------
# 6)  Output wavefunction
# ---------------------------------------------------------------------------
coeffs = model.apply(state.params, OCC, LAM, train=False)
print(f"\nFirst {N_PRINT} coefficients for 1d lattice size of ={LATTICE}:")
for i in range(min(N_PRINT, len(basis))):
    n_str = ''.join(str(int(b)) for b in OCC[i])
    re, im = map(float, coeffs[i])
    print(f"|{n_str}⟩ → {re:+.5f} {im:+.5f} i")


# ---------------------------------------------------------------------------
# 7)  Calculate ground state observables
# ---------------------------------------------------------------------------
## Energy
import observables.local_energy as E
import sampler.sampler_jaxGPU as sampler
import config as C
# (Assumes you have (i) a sampler that draws σ ~ |ψ|^2 and (ii) H for the basis.)

model_apply = lambda params, occ, lam, train=False: \
    model.apply(params, occ, lam, train=train)

L, N, t, v = C.N_SITES, C.N_PART, C.T_HOP, C.V_INT

# 1) Basis + index map (must match H ordering!) (For ED and beyond)
basis_masks = enumerate_fock(L, N)
occ_basis   = jnp.array([mask_to_array(m, L) for m in basis_masks], dtype=jnp.int32)
state_index = helper.build_state_index([int(m) for m in basis_masks])

# 2) Hamiltonian for appropriate basis
H = helper.H_builder(L, N, 2*v)

# Larger M for better accuracy, takes far longer and is marginally better
M = 4096
lam_vec   = jnp.array([2*t, 2*v, float(N)], dtype=jnp.float32) # the 2x is to account for t=1 for Daniele's code
lam_batch = jnp.tile(lam_vec, (M, 1))

# 1) Draw configurations σ ~ |ψ|^2
occ_batch = sampler.sample_occ_batch(model_apply, state.params, lam_vec,
                             L=L, N=N,
                             num_samples=M, burn_in=1024, thin=4,
                             n_chains=32, rng_seed=0)

def model_fn(occ, lam):
    coeff = model_apply(state.params, occ, lam, train=False)  # (B,2)
    return (coeff[:, 0] + 1j * coeff[:, 1]).astype(jnp.complex64)

# 4) Monte Carlo energy (module function averages local energies)
E_mc = E.expectation_local_energy(model=model_fn,
                      occ_batch=occ_batch,
                      params_batch=lam_batch,
                      hopping_terms=helper.hopping_terms(2*t, L),
                      repulsion_pairs=helper.repulsion_pairs(L),
                      V=2*v)
print("E0(model, MC) =", float(E_mc))


# 5) Compare with Exact energy (ED)
_, _, E_ed = helper.my_readcsv(f"{dir}ED/output/gsWaveFn_Ns{int(N_SITES)}_Np{npart}_Vnn{round(v/t,2)}.csv")
print("E0(ED) =", float(E_ed))

# ---------------------------------------------------------------------------
# 8)  Output files
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple

import csv
# ---------------------------------------------------------------- write CSV
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"Fullrun_Size{L}_Np{N}_t{2*t}_Vnn{2*v}.csv"
coeff = model_apply(state.params, occ, lam, train=False)  # (B,2)
coeff = (coeff[:, 0] + 1j * coeff[:, 1]).astype(jnp.complex64)

with out_path.open("w", newline="") as f:
    wr = csv.writer(f)
    hdr = ["E0_model_MC"] + ["E0_ED"] \
            + ["GroundStateWaveFunction"] + ["bitmask"]
    wr.writerow(hdr)

    row = [float(E_mc)] + [float(E_ed)] + [c for c in coeff] + [intRep for intRep in state_index.keys()]
    wr.writerow(row)

print(f"Wrote results to {out_path}")


