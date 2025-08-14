"""
Exact-diagonalisation solver wrapped in a reusable function.

Usage
-----
from ed_solver import run_ed
run_ed("ED/config_test.csv")                # single run
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import itertools
import time

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _load_params(cfg: Path) -> Dict[str, float]:
    params: Dict[str, float] = {}
    with cfg.open() as f:
        rdr = csv.reader(f)
        next(rdr)                                      # skip header
        for name, val in rdr:
            params[name] = float(val)
    return params


def prodvec(z1: complex, z2: complex) -> float:
    return np.real(z1.conjugate() * z2)


def mod_between_neg_half_and_half(x: float) -> float:
    return np.mod(x + 0.5, 1) - 0.5


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run_ed(config_file: str | Path,
           *,
           out_dir: str | Path = "ED/output",
           delta_list: Tuple[int, ...] = (0,)
           ) -> str:
    """
    Run ED once, using *only* values in `config_file`.

    Parameters
    ----------
    config_file : path-like
        CSV with header exactly ``param,value`` (see example).
    out_dir : path-like, optional
        Where to write results. Directory is created if needed.
    delta_list : tuple[int], optional
        Values of \\Delta that define different Hamiltonians.

    Returns
    -------
    str
        Path of the CSV that was written.

    The CSV columns are::

        delta, Eigenvalue_0, Eigenvalue_1, …, GroundStateWaveFunction
    """
    cfg_path = Path(config_file)
    params = _load_params(cfg_path)

    # ------------------------------------------------------------------ set-up
    Ns             = int(params["Nsites"])  
    Nparticelle    = int(params["Nparticelle"])
    Vnn            = float(params["Vnn"])

    print(f"Parameters read from {cfg_path.name}: {params}")
 
    klattice = 2*np.pi/Ns * np.arange(Ns)                                               # k lattice points  
    linking_table = {jr: [np.mod(jr-1, Ns), np.mod(jr+1, Ns)] for jr in range(Ns)}      # 1D linking table 

    # binary representation
    def n2occ(n,Nsize):
        binr=np.binary_repr(n,width=Nsize) 
        return binr

    # build the Hilbert space
    def generate_combinations(N, Np):
        """ Generate all combinations of Np particles from N available positions, returning them as binary numbers. """
        combinations = itertools.combinations(range(N), Np)
        results = []
        for comb in combinations:
            binary_num = 0  # Start with an empty binary number (all zeros)
            for i in comb:
                binary_num |= (1 << i)  # Set the bit at position i
            results.append(binary_num) 
        return results

    ##### Nparticelle number of particles specified previously >> N=2*Ns 
    combinations = generate_combinations(N=Ns,Np=Nparticelle)
    Nstates = len(combinations)
    states = np.arange(Nstates)
    state2ind = dict(zip(combinations,states)) 
    print("Number of states =",Nstates)  

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

    start_time = time.time()

    # Build the Hamiltonian
    Htunnel = lil_matrix((Nstates, Nstates), dtype=np.complex128)   # LIL format for easy assignment
    Hint    = lil_matrix((Nstates, Nstates), dtype=np.complex128)   # LIL format for easy assignment
    #Hloc    = lil_matrix((Nstates, Nstates), dtype=np.complex128)   # LIL format for easy assignment
    for key, index in state2ind.items():
        string = n2occ(n=key,Nsize=Ns)  
        occ_vals = np.array([int(bit) for bit in string])
        posAH = np.where(occ_vals == 1)[0]
        #part_A = string[:Ns]
        #part_B = string[Ns:]
        #pA = part_A.count('1')
        #pB = part_B.count('1')
        #Hloc[index,index] = pB 
        for xx in posAH: 
            jxx = linking_table[xx]
            # print( f'linking table, {jxx}' ) 
            Pxx = [ occ_vals[j] for j in jxx]
            Hint[index,index] += Pxx.count(1)/2 
            # print( f'Hint, {Hint[index,index]}' ) 
            for yy in jxx:  
                if occ_vals[yy] == 0: 
                    string_new, segno = hop(string=string,j1=yy,j2=xx)
                    # print( xx , yy , f'new string {string_new}, sign {segno}' )
                    occ_new = np.array([int(bit) for bit in string_new])  
                    key_new = int(string_new,2) 
                    index_new = state2ind[key_new] 
                    Htunnel[index_new,index] += -segno

    Htunnel = Htunnel.tocsr()
    #Hloc = Hloc.tocsr()
    Hint = Hint.tocsr()

    # ---------------------------------------------------------------- solve
    int_energy_dict: Dict[int, List[np.ndarray]] = {d: [] for d in delta_list}
    int_gs_vecs = []

    t0 = time.time()
    for delta in delta_list:
        Ham = Htunnel + Vnn*Hint #+ delta*Hloc 
        evals, evecs = eigsh(Ham, k=4, which="SA", tol=1e-10)
        int_energy_dict[delta].append(evals)
        int_gs_vecs.append(evecs[:, 0])                # ground state
        print(f"Δ={delta:<2}: E₀ = {evals[0]:.6f}")

    print(f"Diagonalisation finished in {time.time() - t0:.2f} s")

    # ---------------------------------------------------------------- write CSV
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gsWaveFn_Ns{Ns}_Np{Nparticelle}_Vnn{Vnn}.csv"

    with out_path.open("w", newline="") as f:
        wr = csv.writer(f)
        hdr = ["delta"] + [f"Eigenvalue_{i}"
               for i in range(len(next(iter(int_energy_dict.values()))[0]))] \
             + ["GroundStateWaveFunction_real"] + ["bitmask"]
        wr.writerow(hdr)

        for delta, evals_list in int_energy_dict.items():
            row = [delta] + list(evals_list)  + [c for c in int_gs_vecs[0]] + [intRep for intRep in combinations]
            wr.writerow(row)

    print(f"Wrote results to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# allow "python ed_solver.py <cfg>" for a one-off run
# ---------------------------------------------------------------------------

if __name__ == "__main__":          # executed only when run directly
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ed_solver.py path/to/config.csv")
        sys.exit(1)
    run_ed(sys.argv[1])
