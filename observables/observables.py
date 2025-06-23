# observables.py
import torch

def local_energy(model, occ_batch, params_batch, H_exact):
    """
    Monte-Carlo (or full) expectation  ⟨σ|H|ψ⟩ / ψ(σ)
    For a small Hilbert space we can enumerate all σ
    and reuse H_exact from fermionED.jl compiled to a sparse tensor.
    """
    log_psi = model(occ_batch, params_batch)          # (B,)
    with torch.no_grad():
        psi = torch.exp(log_psi)
        # matrix-vector multiply in log-space is omitted for brevity
    raise NotImplementedError  # fill in with your preferred scheme
