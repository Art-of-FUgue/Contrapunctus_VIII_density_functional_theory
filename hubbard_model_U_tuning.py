import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETUP PARAMETERS & MANY-BODY BASIS
# =====================================================================
N_SITES = 4
N_UP = 2
N_DOWN = 2
t = 1.0  # Hopping amplitude scale

def generate_subsector(n_particles, n_sites):
    """Generates all configurations of particles represented as bitmasks."""
    states = []
    for i in range(1 << n_sites):
        if bin(i).count('1') == n_particles:
            states.append(i)
    return states

# Generate separate up/down spin basis states
basis_up = generate_subsector(N_UP, N_SITES)
basis_down = generate_subsector(N_DOWN, N_SITES)

dim_up = len(basis_up)
dim_down = len(basis_down)
TOTAL_DIM = dim_up * dim_down

print(f"Lattice: {N_SITES} sites at half-filling.")
print(f"Hilbert Space Dimensions: Spin-Up={dim_up}, Spin-Down={dim_down}. Total Many-Body Dim={TOTAL_DIM}")

# Lookup dictionaries to map a configuration bitmask to its index in the basis array
up_lookup = {state: idx for idx, state in enumerate(basis_up)}
down_lookup = {state: idx for idx, state in enumerate(basis_down)}

# =====================================================================
# 2. HAMILTONIAN MATRIX CONSTRUCTION
# =====================================================================
def build_hopping_matrix():
    """Constructs the non-interacting kinetic energy (tight-binding) matrix."""
    H_kin = np.zeros((TOTAL_DIM, TOTAL_DIM))
    
    # Define periodic boundary connections (1D Ring topology)
    neighbors = [(i, (i + 1) % N_SITES) for i in range(N_SITES)]
    
    for i_up, s_up in enumerate(basis_up):
        for i_down, s_down in enumerate(basis_down):
            row_idx = i_up * dim_down + i_down
            
            # 1. Hop Spin-Up Electrons
            for src, dst in neighbors:
                # Check if a particle exists at 'src' and space is empty at 'dst' (or vice versa)
                for s, d in [(src, dst), (dst, src)]:
                    if (s_up & (1 << s)) and not (s_up & (1 << d)):
                        # Perform the hop by flipping bits
                        new_s_up = s_up ^ (1 << s) ^ (1 << d)
                        # Determine fermionic sign (count particles crossed between s and d)
                        # For nearest-neighbors on a simple line, the sign remains positive (+1)
                        # unless wrapping around the ring boundaries.
                        sign = -1 if (s == N_SITES - 1 or d == N_SITES - 1) and (N_UP % 2 == 0) else 1
                        
                        target_i_up = up_lookup[new_s_up]
                        col_idx = target_i_up * dim_down + i_down
                        H_kin[row_idx, col_idx] += -t * sign

            # 2. Hop Spin-Down Electrons
            for src, dst in neighbors:
                for s, d in [(src, dst), (dst, src)]:
                    if (s_down & (1 << s)) and not (s_down & (1 << d)):
                        new_s_down = s_down ^ (1 << s) ^ (1 << d)
                        sign = -1 if (s == N_SITES - 1 or d == N_SITES - 1) and (N_DOWN % 2 == 0) else 1
                        
                        target_i_down = down_lookup[new_s_down]
                        col_idx = i_up * dim_down + target_i_down
                        H_kin[row_idx, col_idx] += -t * sign
                        
    return H_kin

def get_interaction_diagonal(U):
    """Computes the diagonal elements corresponding to the local U interactions."""
    diag_int = np.zeros(TOTAL_DIM)
    for i_up, s_up in enumerate(basis_up):
        for i_down, s_down in enumerate(basis_down):
            idx = i_up * dim_down + i_down
            # Count how many sites have BOTH up and down bits active
            double_occupancies = bin(s_up & s_down).count('1')
            diag_int[idx] = U * double_occupancies
    return diag_int

# =====================================================================
# 3. OBSERVABLE CALCULATION
# =====================================================================
def compute_observables(ground_state):
    """Calculates average double occupancy and local magnetic moment squared."""
    avg_double_occ = 0.0
    avg_m_z_sq = 0.0
    
    probs = np.abs(ground_state) ** 2
    
    for i_up, s_up in enumerate(basis_up):
        for i_down, s_down in enumerate(basis_down):
            idx = i_up * dim_down + i_down
            p = probs[idx]
            
            # Double occupancy: counts sites with both spins present
            d_occ = bin(s_up & s_down).count('1')
            avg_double_occ += d_occ * p
            
            # Magnetization squared on each site: (n_up - n_down)^2
            for site in range(N_SITES):
                n_u = 1 if (s_up & (1 << site)) else 0
                n_d = 1 if (s_down & (1 << site)) else 0
                avg_m_z_sq += ((n_u - n_d) ** 2) * p
                
    return avg_double_occ / N_SITES, avg_m_z_sq / N_SITES

# =====================================================================
# 4. SWEEP COUPLING CONSTANT U
# =====================================================================
U_values = np.linspace(-8.0, 8.0, 50)
double_occupancies = []
magnetic_moments = []

H_kin = build_hopping_matrix()

for U in U_values:
    H_int_diag = get_interaction_diagonal(U)
    H_total = H_kin + np.diag(H_int_diag)
    
    # Find the lowest energy state (ground state)
    eigenvalues, eigenvectors = spla.eigsh(H_total, k=1, which='SA')
    psi_0 = eigenvectors[:, 0]
    
    docc, mz2 = compute_observables(psi_0)
    double_occupancies.append(docc)
    magnetic_moments.append(mz2)

# =====================================================================
# 5. VISUALIZATION OF TRANSITIONS
# =====================================================================
plt.figure(figsize=(10, 6))
plt.plot(U_values, double_occupancies, 'bo-', label=r'Double Occupancy $\langle n_{i\uparrow}n_{i\downarrow}\rangle$ (Pairing)')
plt.plot(U_values, magnetic_moments, 'ro-', label=r'Local Moment $\langle (n_{i\uparrow} - n_{i\downarrow})^2 \rangle$ (Mott Localized)')

plt.axvline(x=0, color='gray', linestyle='--', label='Tight-Binding Limit (U=0)')
plt.title('Ground State Phase Crossings in the 4-Site Hubbard Model', fontsize=14)
plt.xlabel('Interaction Strength U / t', fontsize=12)
plt.ylabel('Expectation Value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.show()