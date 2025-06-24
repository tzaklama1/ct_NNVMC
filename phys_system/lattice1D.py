# lattice_1d.py
"""
1-D spinless fermion chain with PBC (nearest-neighbour t-V model).
"""
import itertools
from typing import List, Tuple

# ---------- geometry -------------------------------------------------------- #
def neighbours(i: int, L: int) -> Tuple[int, int]:
    return ((i - 1) % L, (i + 1) % L)

# ---------- Hilbert-space enumeration -------------------------------------- #
def enumerate_fock(L: int, Np: int) -> List[int]:
    """All bit-masks with Np particles on L sites."""
    masks = []
    for occ in itertools.combinations(range(L), Np):
        m = 0
        for j in occ:
            m |= 1 << j
        masks.append(m)
    return masks

def mask_to_array(mask: int, L: int):
    return [(mask >> k) & 1 for k in range(L)]

# quick check
if __name__ == "__main__":
    print("N = 4, half-fill:", enumerate_fock(4,2)[:3])
    print("Neighbours of site 0 in PBC L=4:", neighbours(0,4))
