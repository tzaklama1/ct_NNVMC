# train.py
"""
Tiny demonstration:
* Enumerate 2×2 honeycomb (8 sites, half-filling = 4 particles)
* Create random 'true' coefficients for two (t,V) settings
* Train on one (t,V) ; test generalisation on an unseen pair
Replace the synthetic target with ED-generated data for real use-cases.
"""
import jax, jax.numpy as jnp, optax
from flax.training import train_state
from functools import partial
from honeycomb import enumerate_fock, mask_to_array
from model import CotrainTransformer

# ---------------- Parameters & synthetic data -------------------------------
Lx = Ly = 2
N_SITES = 2 * Lx * Ly
N_PART  = N_SITES // 2
BASIS   = enumerate_fock(N_SITES, N_PART)           # full Hilbert space
OCC_ARR = jnp.array([mask_to_array(m, N_SITES) for m in BASIS], dtype=jnp.int32)

key = jax.random.PRNGKey(0)

def random_coeffs(key):
    z = jax.random.normal(key, (len(BASIS), 2)) * 0.1   # (Re,Im)
    return z

DATA = {
    (0.5, 4.0): random_coeffs(key),
    (1.1, 2.5): random_coeffs(key ^ jnp.uint32(123)),
}

# choose one (t,V) for training, one unseen for test
train_tv = (0.5, 4.0)
test_tv  = (0.8, 3.0)        # unseen!

# synthetic target for unseen params   (just for demo)
DATA[test_tv] = random_coeffs(key ^ jnp.uint32(999))

# -------------------  model & optimiser -------------------------------------
model = CotrainTransformer(n_sites=N_SITES)

def create_state(rng):
    params = model.init(rng, OCC_ARR, jnp.ones((len(BASIS),2)), train=False)
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=tx)

@partial(jax.jit, static_argnums=0)
def loss_fn(params, occ, tv, target):
    preds = model.apply(params, occ, tv, train=False)
    return jnp.mean((preds - target) ** 2)

@jax.jit
def train_step(state, occ, tv, target):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, occ, tv, target)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(state, n_epochs=5000):
    occ = OCC_ARR
    tv  = jnp.tile(jnp.array(train_tv, dtype=jnp.float32), (len(BASIS),1))
    target = DATA[train_tv]
    for epoch in range(n_epochs):
        state, l = train_step(state, occ, tv, target)
        if epoch % 500 == 0:
            print(f"epoch {epoch:4d}, MSE {l:.4e}")
    return state

def evaluate(state, tv_pair):
    occ = OCC_ARR
    tv = jnp.tile(jnp.array(tv_pair, dtype=jnp.float32), (len(BASIS),1))
    preds = model.apply(state.params, occ, tv, train=False)
    return preds

def main():
    rng = jax.random.PRNGKey(42)
    state = create_state(rng)
    print("Training on t,V =", train_tv)
    state = train(state, 2000)

    # sanity check on train set
    mse_train = jnp.mean((evaluate(state, train_tv) - DATA[train_tv])**2)
    print("train-set MSE:", mse_train)

    # generalisation to unseen (t,V)
    preds_unseen = evaluate(state, test_tv)
    mse_test = jnp.mean((preds_unseen - DATA[test_tv])**2)
    print("unseen (t,V) =", test_tv, "; generalisation MSE =", mse_test)

if __name__ == "__main__":
    main()
