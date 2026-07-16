"""
Hubbard dimer: exact diagonalization vs self-consistent mean-field (Hartree/UHF)
2 sites, 1 up-electron + 1 down-electron (half filling).

H = -t * sum_sigma (c1s^dag c2s + h.c.) + U * sum_i n_i,up n_i,down
"""
import numpy as np

t = 1.0  # hopping, sets the energy scale

# ---------------------------------------------------------------
# EXACT DIAGONALIZATION
# Basis: |up_site, down_site>  ->  (1,1),(1,2),(2,1),(2,2)  (1-indexed sites)
# ---------------------------------------------------------------
def exact_ground_state(U, t=1.0):
    H = np.array([
        [U,  -t,  -t,   0],
        [-t,  0,   0,  -t],
        [-t,  0,   0,  -t],
        [0,  -t,  -t,   U],
    ])
    evals, evecs = np.linalg.eigh(H)
    return evals[0], evecs[:, 0]  # ground energy, ground state vector


# ---------------------------------------------------------------
# SELF-CONSISTENT MEAN FIELD (Hartree / "poor man's Kohn-Sham")
# Each spin channel sees an effective on-site potential U*<n_{i,-sigma}>
# from the OTHER spin species. Solve single-particle 2x2 problem per spin,
# occupy its single lowest orbital, update densities, iterate to self-consistency.
# ---------------------------------------------------------------
def meanfield_ground_state(U, t=1.0, n_up_guess=(0.5, 0.5), n_dn_guess=(0.5, 0.5),
                            tol=1e-10, max_iter=500):
    n_up = np.array(n_up_guess, dtype=float)   # (n_1up, n_2up)
    n_dn = np.array(n_dn_guess, dtype=float)   # (n_1dn, n_2dn)

    for _ in range(max_iter):
        # effective single-particle Hamiltonian for the UP electron sees the DOWN density
        h_up = np.array([[U * n_dn[0], -t],
                          [-t,          U * n_dn[1]]])
        h_dn = np.array([[U * n_up[0], -t],
                          [-t,          U * n_up[1]]])

        e_up, v_up = np.linalg.eigh(h_up)
        e_dn, v_dn = np.linalg.eigh(h_dn)

        # occupy the single lowest orbital in each spin channel (1 up e-, 1 down e-)
        new_n_up = v_up[:, 0] ** 2
        new_n_dn = v_dn[:, 0] ** 2

        if (np.max(np.abs(new_n_up - n_up)) < tol and
                np.max(np.abs(new_n_dn - n_dn)) < tol):
            n_up, n_dn = new_n_up, new_n_dn
            break
        n_up, n_dn = new_n_up, new_n_dn

    # total mean-field energy = sum of occupied orbital energies MINUS double-counted
    # Hartree interaction (standard DFT/HF double-counting correction)
    e_up_occ = np.linalg.eigvalsh(np.array([[U * n_dn[0], -t], [-t, U * n_dn[1]]]))[0]
    e_dn_occ = np.linalg.eigvalsh(np.array([[U * n_up[0], -t], [-t, U * n_up[1]]]))[0]
    E_mf = e_up_occ + e_dn_occ - U * (n_up[0] * n_dn[0] + n_up[1] * n_dn[1])
    return E_mf, n_up, n_dn


# ---------------------------------------------------------------
# SCAN OVER U/t
# ---------------------------------------------------------------
U_over_t = np.linspace(0, 10, 41)
E_exact_list, E_mf_sym_list, E_mf_broken_list = [], [], []
mag_broken_list = []

for U in U_over_t:
    E_exact, _ = exact_ground_state(U, t)
    E_exact_list.append(E_exact)

    # symmetric (paramagnetic) mean-field start -> stays symmetric
    E_mf_sym, n_up_s, n_dn_s = meanfield_ground_state(U, t, (0.5, 0.5), (0.5, 0.5))
    E_mf_sym_list.append(E_mf_sym)

    # symmetry-broken start -> can fall into a spurious antiferromagnetic solution
    E_mf_b, n_up_b, n_dn_b = meanfield_ground_state(U, t, (0.9, 0.1), (0.1, 0.9))
    E_mf_broken_list.append(E_mf_b)
    mag_broken_list.append(n_up_b[0] - n_dn_b[0])  # local moment on site 1

E_exact_arr = np.array(E_exact_list)
E_mf_sym_arr = np.array(E_mf_sym_list)
E_mf_broken_arr = np.array(E_mf_broken_list)

# mean-field always relaxes to whichever solution has lower energy
E_mf_best = np.minimum(E_mf_sym_arr, E_mf_broken_arr)
E_corr = E_exact_arr - E_mf_best  # the "correlation energy" DFT/HF misses

# ---------------------------------------------------------------
# PRINT TABLE
# ---------------------------------------------------------------
print(f"{'U/t':>6} {'E_exact':>10} {'E_MF(sym)':>10} {'E_MF(broken)':>13} {'E_corr':>9} {'site1 moment':>13}")
for i in [0, 4, 8, 12, 16, 20, 25, 30, 35, 40]:
    print(f"{U_over_t[i]:6.1f} {E_exact_arr[i]:10.4f} {E_mf_sym_arr[i]:10.4f} "
          f"{E_mf_broken_arr[i]:13.4f} {E_corr[i]:9.4f} {mag_broken_list[i]:13.4f}")

# ---------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(U_over_t, E_exact_arr, 'k-', lw=2.5, label='Exact diagonalization')
ax.plot(U_over_t, E_mf_sym_arr, 'b--', lw=2, label='Mean-field (symmetric)')
ax.plot(U_over_t, E_mf_broken_arr, 'r:', lw=2, label='Mean-field (symmetry-broken)')
ax.set_xlabel('U / t')
ax.set_ylabel('Ground state energy / t')
ax.set_title('Exact vs. mean-field ground state energy')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(U_over_t, E_corr, 'g-', lw=2.5)
ax2.set_xlabel('U / t')
ax2.set_ylabel('E_exact - E_MF  (correlation energy)')
ax2.set_title('Missing correlation energy')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hubbard_comparison.png', dpi=150)
print("\nPlot saved.")
