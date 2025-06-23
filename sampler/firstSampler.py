# sampler.py
import torch
import random

def flip_site(state, i):
    return state ^ (1 << i)

def metropolis_step(state_int, model, params, N_sites):
    occ = torch.tensor([(state_int >> k) & 1 for k in range(N_sites)]).unsqueeze(0)
    logp_old = model(occ, params.unsqueeze(0))[0]
    # propose single-site flip
    j = random.randrange(N_sites)
    new_state = flip_site(state_int, j)
    occ_new = torch.tensor([(new_state >> k) & 1 for k in range(N_sites)]).unsqueeze(0)
    logp_new = model(occ_new, params.unsqueeze(0))[0]
    if torch.rand(1) < torch.exp(2*(logp_new - logp_old)):   # ratio of |ψ|^2
        return new_state
    return state_int
