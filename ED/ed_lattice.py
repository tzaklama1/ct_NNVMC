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
with open("ED/config.csv", mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        params[row[0]] = float(row[1])  # Convert values to float
Xa              = int(params['a'])
Xb              = int(params['b'])
Xc              = int(params['c'])
Xd              = int(params['d'])
Nparticelle     = int(params['Nparticelle'])                 # number of particles
Vnn             = float(params['Vnn'])                   # nearest neighbor interaction
print(f"Parameters: {params}") 

# scalar product 
def prodvec(z1,z2):                           
    return np.real(z1.conjugate()*z2) 

# basis vectors  
a_vec = np.array([ np.sqrt(3)/2 - 1j/2 , 1j , -np.sqrt(3)/2 - 1j/2 ])   
ag = np.array([-a_vec[0],a_vec[1]])
uvec = np.array([1,+1j*np.sqrt(3)/2-1/2,-1j*np.sqrt(3)/2-1/2])/np.sqrt(3)
g_vec = (4*np.pi/np.sqrt(3))*np.array([ -1 , 1/2+1j*np.sqrt(3)/2])

# Torus
ff = np.array([[Xa,Xb],[Xc,Xd]])
Ns = int(round(np.abs(np.linalg.det(ff)),5)) 
print("number of sites =",2*Ns," hole-doping =",Nparticelle," filling factor =",Nparticelle/Ns)
T = [] # torus 
T.append(ff[0,0] * ag[0] + ff[0,1] * ag[1] )
T.append(ff[1,0] * ag[0] + ff[1,1] * ag[1] ) 
T = np.asarray(T)
ff_k = np.linalg.inv(ff).T
bb = [] # minimum momentum resolution 
bb.append( ff_k[0,0]*g_vec[0] + ff_k[0,1]*g_vec[1] ) 
bb.append( ff_k[1,0]*g_vec[0] + ff_k[1,1]*g_vec[1] )
bb = np.asarray(bb)

# module  
def mod_between_neg_half_and_half(x):
    return np.mod(x + 0.5, 1) - 0.5

# PBC position
def PBC_position( R, bvec, avec, ft): 
    f = np.zeros(2,dtype=float)
    for s in range(2):  # project in the basis
        f[s] = mod_between_neg_half_and_half(round(prodvec(R,bvec[s])/(2*np.pi),6))
    vec =  np.round( avec @ ft @ f , 6 )
    return vec 

# PBC momentum
def PBC_momentum( kk, bvec, avec, ft): 
    f = np.zeros(2,dtype=float) 
    for s in range(2):  # project in the basis
        f[s] = mod_between_neg_half_and_half(round(prodvec(kk,avec[s])/(2*np.pi),6))
    vec =  np.round( bvec @ ft @ f , 6 )
    return vec

# build the lattice in real and momentum space 
num = 20
lattice = {}
klattice = {}
count = 0 
countK = 0 
ft = np.linalg.inv(ff_k)
pr = np.array([1,1j])
for j in range(num): 
    for l in range(num): 
        R = l*ag[0] + j*ag[1] - sum(T)/2
        k = l*bb[0] + j*bb[1]
        vec = PBC_position( R=R, bvec=bb, avec=ag, ft=ft)
        vecK = PBC_momentum( kk=k, bvec=bb, avec=ag, ft=ft)
        if not any(np.isclose(vec, existing_vec, atol=1e-8) for existing_vec in lattice.values()):
            lattice[count] = vec
            count += 1
        if not any(np.isclose(vecK, existing_vec, atol=1e-8) for existing_vec in klattice.values()):
            klattice[countK] = vecK
            countK += 1 

connectAB = [[0,0],[1,0],[1,-1]]
linking_table = {} 
for jr,r in enumerate(lattice.values()):
    match = []
    for js,dv in enumerate(connectAB):
        rp = r + dv@ag 
        rp = PBC_position( R=rp, bvec=bb, avec=ag, ft=ft)
        match.append(next((k for k, v in lattice.items() if np.isclose(rp, v, atol=1e-5)), None))
    linking_table[jr] = [m + Ns for m in match] 
    match = [] 
    for js,dv in enumerate(connectAB):
        rp = r - dv@ag 
        rp = PBC_position( R=rp, bvec=bb, avec=ag, ft=ft)
        match.append(next((k for k, v in lattice.items() if np.isclose(rp, v, atol=1e-5)), None))   
    linking_table[jr+Ns] = match

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
combinations = generate_combinations(N=2*Ns,Np=Nparticelle)
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
Hloc    = lil_matrix((Nstates, Nstates), dtype=np.complex128)   # LIL format for easy assignment
for key, index in state2ind.items():
    string = n2occ(n=key,Nsize=2*Ns)  
    occ_vals = np.array([int(bit) for bit in string])
    posAH = np.where(occ_vals == 1)[0]
    part_A = string[:Ns]
    part_B = string[Ns:]
    pA = part_A.count('1')
    pB = part_B.count('1')
    Hloc[index,index] = pB 
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
Hloc = Hloc.tocsr()
Hint = Hint.tocsr()

delta_list = np.array([0])#,1,2]) 
int_energy_dict = {delta: [] for delta in delta_list}  # Initialize energy dictionary for all Qcom
int_gsWaveFn = []
for jdelta, delta in enumerate(delta_list): 
    Ham = Htunnel + delta * Hloc + Vnn * Hint   
    eigenvalues, eigenvectors = eigsh(Ham,k=4,which='SA',tol=1.e-10)
    int_energy_dict[delta].append(eigenvalues)  # Store eigenvalues for each delta
    int_gsWaveFn.append(eigenvectors)  # Store eigenvectors for each delta
    print(f"delta={delta}, lowest eigenvalue={eigenvalues[0]}",flush=True)
    print(f"delta={delta}, ground state wave function (first 3 entries)={eigenvectors[:,0][:3]}",flush=True)
    print(f"delta={delta}, ground state wave function (last 3 entries)={eigenvectors[:,0][-3:]}",flush=True)

#output_energy_file = f"int_energy_dict_Ns{Ns}_Np{Nparticelle}.csv"
output_file = f"gsWaveFn_Ns{Ns}_Np{Nparticelle}_Vnn{Vnn}.csv"
with open("ED/output/"+output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    header = ["delta"] + [f"Eigenvalue_{i}" for i in range(len(next(iter(int_energy_dict.values()))))] + ["GroundStateWaveFunction"]
    writer.writerow(header)
    # Write the data rows
    for delta, eigenvalues in int_energy_dict.items():
        row = [delta] + list(eigenvalues)  + [vector0[:,0] for vector0 in int_gsWaveFn] # Combine Qcom and eigenvalues into one row
        writer.writerow(row)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")