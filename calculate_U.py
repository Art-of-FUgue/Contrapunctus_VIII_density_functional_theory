import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# =====================================================================
# 1. PHYSICAL CONSTANTS & SYSTEM SETUP
# =====================================================================
me = 1.0       # Effective electron mass
kf = 1.2       # Fermi momentum (determines filling)
ef = (kf**2) / (2 * me)  # Fermi energy
e_charge = 1.4 # Bare electronic charge coupling
alpha = 0.1    # Small imaginary dampening for Green's functions

q_grid = np.linspace(0.01, 4.0, 100)
k_grid = np.linspace(-3.0, 3.0, 300)

# =====================================================================
# 2. COMPUTING LINDHARD POLARIZATION P(q, 0) VIA WICK CONTRACTION
# =====================================================================
def compute_polarization_loop(q):
    """
    Computes the static polarization bubble P(q, w=0).
    Analytically equivalent to the 1D Lindhard loop derived from Wick contractions.
    """
    integrand = []
    for k in k_grid:
        # Energy dispersion e(k) = k^2 / 2m
        en_k = (k**2) / (2 * me)
        en_kq = ((k + q)**2) / (2 * me)
        
        # Fermi-Dirac occupation at T=0
        f_k = 1.0 if en_k <= ef else 0.0
        f_kq = 1.0 if en_kq <= ef else 0.0
        
        # Lindhard factor: (f_k - f_kq) / (en_k - en_kq)
        if abs(en_k - en_kq) > 1e-6:
            val = (f_k - f_kq) / (en_k - en_kq)
        else:
            val = 0.0
        integrand.append(val)
        
    # Integrate over k-space
    return simpson(integrand, k_grid) / (2 * np.pi)

print("Calculating polarization loops and Dyson screening...")
P_q = np.array([compute_polarization_loop(q) for q in q_grid])

# =====================================================================
# 3. DYSON SERIES SCREENING: V_0 -> W
# =====================================================================
# Bare 1D Coulomb interaction regularized at short distance: V0(q) = 4*pi*e^2 / (q^2 + gamma^2)
gamma = 0.5
V0_q = (4 * np.pi * e_charge**2) / (q_grid**2 + gamma**2)

# RPA Dyson Equation: W = V0 / (1 - V0 * P)
epsilon_q = 1.0 - V0_q * P_q
W_q = V0_q / epsilon_q

# =====================================================================
# 4. WANNIER PROJECTION TO FIND LATTICE HUBBARD U
# =====================================================================
def wannier_orbital_k(k, sigma=0.8):
    """Localized spatial Wannier function represented in momentum space (Gaussian approximation)."""
    return np.exp(-(k * sigma)**2 / 2.0)

# Project bare V0 and screened W onto the localized orbital
U_bare_integrand = V0_q * (wannier_orbital_k(q_grid)**2)
U_screened_integrand = W_q * (wannier_orbital_k(q_grid)**2)

U_bare = simpson(U_bare_integrand, q_grid)
U_screened = simpson(U_screened_integrand, q_grid)

print(f"\n--- Results ---")
print(f"Bare On-site U (Unscreened): {U_bare:.4f}")
print(f"RPA Interacting Hubbard U (Screened): {U_screened:.4f}")
print(f"Screening Reduction Factor: {U_screened / U_bare:.2%}")

# =====================================================================
# 5. VISUALIZATION
# =====================================================================
fig, ax1 = plt.subplots(figsize=(9, 5))

color = 'tab:blue'
ax1.set_xlabel('Momentum Transfer q', fontsize=12)
ax1.set_ylabel('Interaction Matrix Element', color=color, fontsize=12)
ax1.plot(q_grid, V0_q, '--', color=color, alpha=0.6, label='Bare Coulomb $V_0(q)$')
ax1.plot(q_grid, W_q, '-', color=color, linewidth=2, label='Screened Coulomb $W(q)$ [RPA]')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Dielectric Function $\epsilon(q)$', color=color, fontsize=12)
ax2.plot(q_grid, epsilon_q, color=color, linestyle=':', linewidth=2, label='Dielectric Function $\epsilon(q)$')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('Dyson Screening Analysis: From Bare Coulomb to Hubbard $U$', fontsize=13)
fig.legend(loc="upper right", bbox_to_anchor=(0.85,0.85))
plt.show()