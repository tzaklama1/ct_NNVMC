# lattice.py
import itertools
import numpy as np
from typing import List, Tuple, Dict

# ---- basic honeycomb geometry (two-site unit cell) ----
a1 = np.array([1.0, 0.0])          # Bravais vectors (units: lattice spacing)
a2 = np.array([0.5, np.sqrt(3)/2])
delta = np.stack([                 # vectors A → B
    [0.0, 0.0],
    a1,
    a2
])

def neighbours(idx: int, Lx: int, Ly: int) -> List[int]:
    """Return indices of the 3 n.n. sites for site `idx` on an Lx×Ly torus."""
    cell, subl = divmod(idx, 2)            # 0=A, 1=B
    cx, cy = divmod(cell, Lx)
    out = []
    for d in delta:
        nx, ny = cx + d[0], cy + d[1]
        nx %= Lx; ny %= Ly
        neighbour_cell = int(ny)*Lx + int(nx)
        out.append(2*neighbour_cell + 1 - subl)  # hop A↔B
    return out

# ---- Fock-basis enumeration (bit-mask encoding) ----
def enumerate_fock(Nsites: int, Nparticles: int) -> List[int]:
    """Return state list encoded as bit-masks of length Nsites."""
    basis = []
    for occ in itertools.combinations(range(Nsites), Nparticles):
        mask = 0
        for i in occ:
            mask |= 1 << i
        basis.append(mask)
    return basis

# ---- Hamiltonian matrix elements (t, V, Δ) ----
def hopping_terms(t: float, Lx: int, Ly: int) -> List[Tuple[complex, Tuple[int,int]]]:
    terms = []
    Nsites = 2*Lx*Ly
    for i in range(Nsites):
        for j in neighbours(i, Lx, Ly):
            if i < j:                       # avoid duplicates
                terms.append((-t, (i, j)))
    return terms

def repulsion_pairs(Lx: int, Ly: int) -> List[Tuple[int,int]]:
    pairs = []
    Nsites = 2*Lx*Ly
    for i in range(Nsites):
        for j in neighbours(i, Lx, Ly):
            if i < j:
                pairs.append((i, j))
    return pairs
