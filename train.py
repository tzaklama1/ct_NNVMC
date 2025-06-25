# train.py
"""
Minimal training loop for lattice PsiFormer.
 * Works on CPU or GPU (JAX auto-select).
 * Synthetic targets (random complex) – replace with ED values for real runs.
"""
import jax, jax.numpy as jnp, optax, chex
from flax.training import train_state
from functools import partial

from networks.model import LatticeTransFormer
from phys_system.honeycomb import enumerate_fock as enum_hc, mask_to_array as arr_hc
from phys_system.lattice1D import enumerate_fock as enum_1d, mask_to_array as arr_1d
from Loss.loss import overlap_loss, amp_phase_loss

# ----------------- choose lattice & parameters ----------------------------- #
LATTICE = '1d'        # '1d'  or  'honeycomb'
if LATTICE == '1d':
    L = 10
    N_SITES = L
    N_PART  = L // 2
    BASIS   = enum_1d(L, N_PART)
    arr_fn  = lambda m: arr_1d(m, L)
else:
    Lx = Ly = 2
    from phys_system.honeycomb import enumerate_fock as enum_hc, mask_to_array as arr_hc
    N_SITES = 2*Lx*Ly
    N_PART  = N_SITES // 2
    BASIS   = enum_hc(N_SITES, N_PART)
    arr_fn  = lambda m: arr_hc(m, N_SITES)

OCC = jnp.array([arr_fn(m) for m in BASIS], dtype=jnp.int32)

# ----------------- synthetic coefficients (Re,Im) --------------------------- #
key = jax.random.PRNGKey(0)
def syn_coeffs(k):
    return jax.random.normal(k, (len(BASIS), 2))*0.1
COEFF_DB = {(0.5,4.0): syn_coeffs(key),
            (1.2,2.0): syn_coeffs(key^jnp.uint32(11))}

TRAIN_TV = (0.5,4.0)
TEST_TV  = (0.9,3.0)
COEFF_DB.setdefault(TEST_TV, syn_coeffs(key^jnp.uint32(77)))

# ----------------- model & optimiser --------------------------------------- #
model = LatticeTransFormer(n_sites=N_SITES, depth=8, d_model=256)

def create_state(rng):
    params = model.init(rng, OCC, jnp.ones((len(BASIS),2)), train=False)
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnums=0)
def mse_fn(params, occ, tv, target):
    preds = model.apply(params, occ, tv, train=False)
    return jnp.mean((preds-target)**2)

# loss (no jit, params are dynamic)
def loss_fn(params, occ, tv, target):
    preds = model.apply(params, occ, tv, train=False)
    return jnp.mean((preds - target) ** 2)

LOSS_TYPE = "overlap"         # "overlap"  or  "amp_phase"

def loss_fn(params, occ, tv, target, neighbours=None):
    pred = model.apply(params, occ, tv, train=False)
    if LOSS_TYPE == "overlap":
        return overlap_loss(pred, target)
    else:
        return amp_phase_loss(pred, target, neighbours)

# value_and_grad takes care of autodiff
@jax.jit
def train_step(state: train_state.TrainState,
               occ: jnp.ndarray,
               tv:  jnp.ndarray,
               target: jnp.ndarray,
               neighbours: jnp.ndarray = None):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, occ, tv, target, neighbours)
    state = state.apply_gradients(grads=grads)
    return state, loss

def run(epochs=4000):
    rng = jax.random.PRNGKey(42)
    state = create_state(rng)

    occ = OCC
    tv  = jnp.tile(jnp.array(TRAIN_TV,dtype=jnp.float32), (len(BASIS),1))
    target = COEFF_DB[TRAIN_TV]

    for e in range(epochs):
        state, loss = train_step(state, occ, tv, target)
        if e%500==0:
            print(f"epoch {e:4d} | MSE = {float(loss):.3e}")

    # generalisation test
    test_tv  = jnp.tile(jnp.array(TEST_TV,dtype=jnp.float32),
                        (len(BASIS),1))
    pred     = model.apply(state.params, OCC, test_tv, train=False)
    mse_test = jnp.mean((pred - COEFF_DB[TEST_TV])**2)
    print("Generalisation MSE on unseen (t,V) =", float(mse_test))

if __name__ == "__main__":
    print("Running on device:", jax.default_backend().upper())
    run()
