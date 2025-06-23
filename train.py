# train.py
import torch, torch.nn.functional as F
from lattice import enumerate_fock
from model import CoTrainingTransformer
from julia.api import Julia; jl = Julia(compiled_modules=False)
from julia import Main as jl_main
jl_main.include("fermionED.jl")                          # loads Hamiltonian helpers

def exact_hamiltonian(t, V, Lx, Ly):
    # Build one- and two-body lists in Julia, get sparse CSC back as SciPy matrix
    return jl_main.build_H(t, V, Lx, Ly)                 # you add this in fermionED.jl

def train_epoch(model, opt, states, params, H):
    B = 128
    for _ in range(0, len(states), B):
        batch = states[_:_+B]
        occ = torch.tensor([[ (s >> k) & 1 for k in range(model.N_sites) ] for s in batch])
        p   = params.repeat(len(batch), 1)
        # ---- loss: supervised energy regression (ED)
        with torch.no_grad():
            E_exact = torch.tensor([ jl_main.local_energy_int(s, H) for s in batch ])
        pred = model(occ, p)
        loss = F.mse_loss(pred, E_exact)
        opt.zero_grad(); loss.backward(); opt.step()

def run():
    Lx = Ly = 2
    N_sites = 2*Lx*Ly
    N_particles = Lx*Ly       # half-filling
    states = enumerate_fock(N_sites, N_particles)
    params = torch.tensor([0.5, 5.0])         # example (t,V)
    H = exact_hamiltonian(*params, Lx, Ly)

    model = CoTrainingTransformer(N_sites)
    model.N_sites = N_sites                   # minor convenience
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    for epoch in range(200):
        train_epoch(model, opt, states, params, H)
        if epoch % 20 == 0:
            print(f"epoch {epoch:03d}")

if __name__ == "__main__":
    run()
