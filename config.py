# --------------------------------------------------------------------------
#  Editable configuration for the lattice co-trained Transformer
# --------------------------------------------------------------------------

# --- lattice ---------------------------------------------------------------
LATTICE   = "1d"     # "1d" or "honeycomb"
N_SITES   = 8        # total number of sites (1d ⇒ chain length L = N_SITES)
N_PART    = 4        # number of fermions (half-filling ⇒ N_SITES//2)

# --- Hamiltonian parameters ------------------------------------------------
T_HOP     = 0.5      # nearest-neighbour hopping      (t)
V_INT     = 4.0      # nearest-neighbour repulsion    (V)

# --- optimisation ----------------------------------------------------------
LOSS_TYPE = "amp_phase"     # "overlap"  or  "amp_phase" or "original"
EPOCHS    = 1200          # SGD steps in the demo script
PRINT_EVERY = 300         # logging cadence (epochs)

# --- demo output -----------------------------------------------------------
N_PRINT   = 10       # how many coefficients to print at the end
SEED      = 0        # RNG seed
