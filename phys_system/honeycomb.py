# honeycomb.py
"""
Honeycomb lattice & Fock-basis utilities (periodic torus).

Sites are labelled   idx = 2*(cx + Lx*cy) + subl,
where subl=0 is A and 1 is B.
"""
import itertools
from typing import List, Tuple

def neighbour_vectors():
    # a1 = (1,0), a2=(1/2,√3/2) in lattice units
    return [(0, 0, 1), (1, 0, 1), (0, 1, 1)]  # (dx,dy, flipAB)

def neighbours(idx: int, Lx: int, Ly: int) -> List[int]:
    cell, subl = divmod(idx, 2)
    cx, cy = divmod(cell, Lx)
    outs = []
    for dx, dy, flip in neighbour_vectors():
        nx, ny = (cx + dx) % Lx, (cy + dy) % Ly
        outs.append(2 * (ny * Lx + nx) + (subl ^ flip))
    return outs

# -------- Hilbert-space enumeration (bit masks) -----------------------------
def enumerate_fock(n_sites: int, n_particles: int) -> List[int]:
    """Return list of integer bit-masks with exactly n_particles bits set."""
    basis = []
    for occ in itertools.combinations(range(n_sites), n_particles):
        m = 0
        for i in occ:
            m |= 1 << i
        basis.append(m)
    return basis

def mask_to_array(mask: int, n_sites: int):
    return [(mask >> k) & 1 for k in range(n_sites)]

# small smoke-test
if __name__ == "__main__":
    Lx = Ly = 2
    N = 2 * Lx * Ly
    print("site 0 n.n. =", neighbours(0, Lx, Ly))
    print("2-particle basis sample:", enumerate_fock(N, 2)[:4])
