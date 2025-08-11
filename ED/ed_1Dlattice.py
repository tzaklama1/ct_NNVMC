import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylanczos import PyLanczos
import os
import csv 
import itertools 
from scipy.sparse import lil_matrix, csr_matrix 
from scipy.sparse.linalg import eigsh 
import pandas as pd  
import time

# read from file the parameters of the model: define Vpot, Uint, UV cutoff 
params = {} 
with open("config1D.csv", mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        params[row[0]] = float(row[1])          # Convert values to float
Ns              = int(params['size'])           # number of sites of the 1D chain
Nparticelle     = int(params['Nparticelle'])    # number of particles
Vnn             = float(params['Vnn'])          # nearest neighbor interaction
print(f"Parameters: {params}") 

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
Hloc    = lil_matrix((Nstates, Nstates), dtype=np.float64)   # LIL format for easy assignment
for key, index in state2ind.items():
    string = n2occ(n=key,Nsize=Ns)  
    occ_vals = np.array([int(bit) for bit in string])  
    posAH = np.where(occ_vals == 1)[0]
    posAH_odd = posAH[posAH % 2 == 1]
    Hloc[index,index] = len(posAH_odd)  # number of odd particles in the state 
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
Htunnel = Htunnel.tocsr()
Hloc = Hloc.tocsr()
Hint = Hint.tocsr() 
# Consider a set of delta values
delta_list = np.array([0,1,2]) 
int_energy_dict = {delta: [] for delta in delta_list}  # Initialize energy dictionary for all Qcom
int_eigenvec_dict = {delta: [] for delta in delta_list}  # Initialize eigenvector dictionary
for jdelta, delta in enumerate(delta_list): 
    Ham = Htunnel + delta * Hloc + Vnn * Hint   
    eigenvalues, eigenvectors = eigsh(Ham,k=4,which='SA',tol=1.e-10)
    int_energy_dict[delta].append(eigenvalues)  # Store eigenvalues for each delta
    int_eigenvec_dict[delta].append(eigenvectors[:, 0])  # Store ground state eigenvector    
    print(f"delta={delta}, lowest eigenvalue={eigenvalues[0]}",flush=True)
# Store ground state energies for different delta values 
output_energy_file = f"int_energy_dict_Ns{Ns}_Np{Nparticelle}.csv"
with open(output_energy_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    header = ["delta"] + [f"Eigenvalue_{i}" for i in range(len(next(iter(int_energy_dict.values()))))]
    writer.writerow(header)
    # Write the data rows
    for delta, eigenvalues in int_energy_dict.items():
        row = [delta] + list(eigenvalues)  # Combine Qcom and eigenvalues into one row
        writer.writerow(row)
# Save ground states to separate files for each delta
for delta in delta_list:
    output_eigenvec_file = f"int_eigenvec_dict_delta{delta}_Ns{Ns}_Np{Nparticelle}.csv"
    with open(output_eigenvec_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["config_integer", "amplitude"])
        
        # Get the ground state eigenvector for this delta
        ground_state = int_eigenvec_dict[delta][0]  # First (and only) eigenvector stored
        
        # Write data: configuration integer (key) and corresponding complex amplitude
        for (config_integer, config_index), amplitude in zip(state2ind.items(), ground_state):
            writer.writerow([config_integer, amplitude])
    
    print(f"Saved eigenvector for delta={delta} to {output_eigenvec_file}")

