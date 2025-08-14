# helper_module.py
"""
This module is for all helper functions used in the project.

For reading and writing input and output files.

For small functions that do not belong anywhere else.
"""

import jax, jax.numpy as jnp
import csv, re
import numpy as np
from pathlib import Path
from math import comb as _comb
import itertools 
from scipy.sparse import lil_matrix, csr_matrix 
import itertools
from typing import List, Tuple, Dict, Sequence

def my_readcsv(path: str):
    """
    Parse a 'wide' ED CSV with columns like:
      delta, Eigenvalue_0, Eigenvalue_1, Eigenvalue_2, Eigenvalue_3,
      GroundStateWaveFunction_* , bitmask, ...

    File layout (as in your upload):
      - row[0]           -> delta (string int)
      - row[1]           -> a single cell containing the 4 eigenvalues as a bracketed string
                            e.g. "[-8.25  -8.44  -8.44  -8.22]"
                            (we'll parse and take E0 = min of these)
      - from some index  -> Nstates complex coeff cells, each like "(a+bj)"
      - next Nstates     -> integer bitmasks

    Returns:
      coeffs : np.ndarray[complex]   shape (Nstates,)
      masks  : np.ndarray[np.uint64] shape (Nstates,)
      E0     : float                 ground-state energy
    """
    p = Path(path)
    with p.open('r', newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError("CSV appears empty or header-only.")

    header = rows[0]
    row = rows[1]

    # --- infer Ns, Np -> Nstates from filename e.g. ..._Ns12_Np4_...
    m_ns = re.search(r"Ns(\d+)", p.stem)
    m_np = re.search(r"Np(\d+)", p.stem)
    if not (m_ns and m_np):
        raise ValueError("Filename must include Ns<sites> and Np<particles> (e.g. ..._Ns12_Np4_...).")
    Ns = int(m_ns.group(1))
    Np = int(m_np.group(1))
    Nstates = _comb(2*Ns, Np)

    # --- parse ground-state energy from the eigenvalue fields
    # Try 1: the typical "one cell holds all 4 eigenvalues in brackets"
    E0 = None
    if len(row) > 1 and row[1].strip().startswith('['):
        inner = row[1].strip().strip('[]')
        # split on whitespace; allow no commas
        try:
            evals = [float(tok) for tok in re.split(r'\s+', inner.strip()) if tok]
            if evals:
                E0 = float(min(evals))
        except Exception:
            E0 = None

    # Try 2: if each eigenvalue is in its own column and is a number
    if E0 is None:
        # find indices of columns named Eigenvalue_*
        ev_idxs = [i for i, h in enumerate(header) if h.startswith("Eigenvalue_")]
        evals = []
        for i in ev_idxs:
            try:
                evals.append(float(row[i]))
            except Exception:
                pass
        if evals:
            E0 = float(min(evals))

    # If still None, leave E0 as None (not present in CSV)
    # --- find the start of the complex coefficient block
    # A complex token like "(<real>+<imag>j)" or "(<real>-<imag>j)"
    cx_pat = re.compile(
        r"^\(\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*[-+]\s*\d*\.?\d+(?:e[-+]?\d+)?j\s*\)$",
        re.I
    )
    start = None
    for i, cell in enumerate(row):
        if cx_pat.match(cell.replace(" ", "")):
            start = i
            break
    if start is None:
        raise ValueError("Could not detect start of complex coefficients in the CSV row.")

    if len(row) < start + 2 * Nstates:
        raise ValueError(
            f"Row too short: need {2*Nstates} cells after coeff start={start}, "
            f"have {len(row)-start}."
        )

    coeff_cells = row[start:start + Nstates]
    mask_cells  = row[start + Nstates:start + 2 * Nstates]

    # --- parse complex coeffs and integer masks
    def parse_cx(s: str) -> complex:
        s2 = s.strip().strip("()").replace(" ", "")
        return complex(s2)

    coeffs = np.array([parse_cx(s) for s in coeff_cells], dtype=np.complex128)
    masks  = np.array([int(x) for x in mask_cells], dtype=np.uint64)

    # sanity check
    if coeffs.shape[0] != Nstates or masks.shape[0] != Nstates:
        raise ValueError("Parsed lengths do not match Nstates.")

    return coeffs, masks, E0

# For local energy calculation => hopping terms in 1d
def hopping_terms(t: float, L: int) -> List[Tuple[complex, Tuple[int, int]]]:
    """Nearest-neighbour hopping on an L‑site ring with PBC."""
    terms = []
    for i in range(L):
        j = (i + 1) % L               # right neighbour (wraps around)
        pair = (min(i, j), max(i, j))  # avoid duplicates
        terms.append((-t, pair))
    return terms

def repulsion_pairs(L: int) -> List[Tuple[int, int]]:
    """Site pairs entering V·n_i n_j for the same 1D ring."""
    pairs = []
    for i in range(L):
        j = (i + 1) % L
        pairs.append((min(i, j), max(i, j)))
    return pairs

# General build state index function
def build_state_index(basis_masks: Sequence[int]) -> Dict[int, int]:
    """Map occupation bit masks to their row index in the Hamiltonian."""
    return {int(m): i for i, m in enumerate(basis_masks)}

# Build the Hamiltonian for the non-interacting 1D chain of fermions
Array = jax.Array
def H_builder(N_SITES, N_PART, Vnn) -> Array:
    Ns              = N_SITES          # number of sites of the 1D chain
    Nparticelle     = N_PART    # number of particles

    # basis vectors  
    klattice = 2*np.pi/Ns * np.arange(Ns)                                               # k lattice points  
    linking_table = {jr: [np.mod(jr-1, Ns), np.mod(jr+1, Ns)] for jr in range(Ns)}      # 1D linking table 
    # generate Hilbert space 
    def n2occ(n,Nsize):
        binr=np.binary_repr(n,width=Nsize) 
        return binr
    def generate_combinations(N,Np):
        """ Generate all combinations of Np particles from N available positions, returning them as binary numbers. """
        combinations = itertools.combinations(range(N), Np)
        results = []
        for comb in combinations:
            binary_num = 0  # Start with an empty binary number (all zeros)
            for i in comb:
                binary_num |= (1 << i)  # Set the bit at position i
            results.append(binary_num)  # Append the binary number to the results
        return results
    ##### Nparticelle number of particles specified previously >> N=2*Ns 
    combinations = generate_combinations(N=Ns,Np=Nparticelle)
    Nstates = len(combinations)
    states = np.arange(Nstates)
    state2ind = dict(zip(combinations,states))
    print("Number of states =",Nstates)  
    ###### hopping 
    def hop(string,j1,j2): # hopping 
        Ncount = sum( int(string[j]) for j in range(j2) ) + sum( int(string[j]) for j in range(j1) )  
        if j2 >= j1: 
            hop_sign = Ncount
        else: 
            hop_sign = 1 + Ncount 
        # distruggo in j2 
        tmp = string 
        tmp = "o"+tmp+"o" 
        tmp = tmp[:1+j2]+"0"+tmp[2+j2:] 
        tmp_p = tmp[1:-1] 
        # creo in j1 
        tmp_p = "o"+tmp_p+"o" 
        tmp_p = tmp_p[:1+j1]+"1"+tmp_p[2+j1:]  
        out = tmp_p[1:-1]
        return out, (-1)**hop_sign
    # Build the Hamiltonian
    Htunnel = lil_matrix((Nstates, Nstates), dtype=np.float64)   # LIL format for easy assignment
    Hint    = lil_matrix((Nstates, Nstates), dtype=np.float64)   # LIL format for easy assignment 
    #Hloc    = lil_matrix((Nstates, Nstates), dtype=np.float64)   # LIL format for easy assignment
    for key, index in state2ind.items():
        string = n2occ(n=key,Nsize=Ns)  
        occ_vals = np.array([int(bit) for bit in string])  
        posAH = np.where(occ_vals == 1)[0]
        #posAH_odd = posAH[posAH % 2 == 1]
        #Hloc[index,index] = len(posAH_odd)  # number of odd particles in the state 
        for xx in posAH: 
            jxx = linking_table[xx]
            Pxx = [ occ_vals[j] for j in jxx]
            # print( xx , f'linking table, {jxx}', f'Pxx {Pxx}' , string , Pxx.count(1) )
            Hint[index,index] += Pxx.count(1)/2 
            for yy in jxx:  
                if occ_vals[yy] == 0: 
                    string_new, segno = hop(string=string,j1=yy,j2=xx)
                    # print( xx , yy , f'new string {string_new}, sign {segno}' )
                    occ_new = np.array([int(bit) for bit in string_new])  
                    key_new = int(string_new,2) 
                    index_new = state2ind[key_new] 
                    Htunnel[index_new,index] += -segno
    # Convert LIL matrix to dense numpy array, then to jax array
    Ham = Htunnel + Vnn*Hint #+ delta*Hloc 
    H_dense = np.array(Ham.todense(), dtype=np.float32)
    H_jax = jnp.array(H_dense)
    return H_jax