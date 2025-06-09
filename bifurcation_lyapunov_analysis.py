# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:18:15 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.linalg import eigvals

# ==================== CONFIGURATION ====================
# Parameter sweep settings
eta_vals  = np.linspace(0.1, 1.0, 40)
beta_vals = np.linspace(0.0, 0.5, 40)
V0, I0 = 0.05, 0.10  # frozen variables for reduced 3D system

# Integration settings for Lyapunov
dt_lyap = 1e-2
T_lyap  = 500.0

# Model parameters
kappa, gamma_m, epsilon = 0.10, 0.03, 0.01
alpha_A, gamma_A        = 0.20, 0.05
lambda_w, mu            = 0.01, 0.01

# ==================== MODEL DEFINITION ====================
def F3(x, eta_M, beta_AP):
    m, w, A = x
    dm = kappa*A - gamma_m*m*(1 + np.tanh(w)) - epsilon*V0*m
    dw = eta_M - mu*w - lambda_w*I0*w
    dA = alpha_A*A + beta_AP*A*np.tanh(m) - gamma_A*w*A
    return np.array([dm, dw, dA])

def J3(x, eta_M, beta_AP):
    m, w, A = x
    J = np.zeros((3,3))
    J[0,0] = -gamma_m*(1+np.tanh(w)) - epsilon*V0
    J[0,1] = -gamma_m*m*(1 - np.tanh(w)**2)
    J[0,2] = kappa
    J[1,1] = -mu - lambda_w*I0
    J[2,0] = beta_AP*A*(1 - np.tanh(m)**2)
    J[2,1] = -gamma_A*A
    J[2,2] = alpha_A + beta_AP*np.tanh(m) - gamma_A*w
    return J

# ==================== FIXED-POINT & BIFURCATION SCAN ====================
fixed_points = np.full((len(eta_vals), len(beta_vals), 3), np.nan)
bif_hopf     = []
x_prev = np.array([0.1, 0.2, 0.1])

for i, eta in enumerate(eta_vals):
    for j, beta in enumerate(beta_vals):
        def F_wrap(x):
            return F3(x, eta, beta)

        sol = None
        # Try continuation initial guess, then defaults
        for guess in (x_prev, [0.1, 0.2, 0.1], [0.5, 0.5, 0.5]):
            try:
                sol_candidate, info, ier, msg = fsolve(
                    F_wrap, guess, full_output=True, xtol=1e-12, maxfev=2000
                )
                if ier == 1:
                    sol = sol_candidate
                    x_prev = sol_candidate
                    break
            except Exception:
                continue

        # Fallback to root() if needed
        if sol is None:
            try:
                res = root(F_wrap, x_prev, tol=1e-12, options={'maxiter':2000})
                if res.success:
                    sol = res.x
                    x_prev = res.x
            except Exception:
                pass

        # Record fixed point and check for Hopf
        if sol is not None:
            fixed_points[i, j] = sol
            ev = eigvals(J3(sol, eta, beta))
            re = np.real(ev)
            if np.any(re > 0) and np.any(re < 0):
                bif_hopf.append((eta, beta))

# ==================== LYAPUNOV EXPONENT CALCULATION ====================
def max_lyapunov(F, J, x0, dt=dt_lyap, T=T_lyap):
    x = x0.copy()
    delta = np.random.randn(3)
    delta /= np.linalg.norm(delta)
    sum_log = 0.0
    n_steps = int(T / dt)
    for k in range(n_steps):
        # integrate state
        x += F(x) * dt
        # integrate perturbation
        delta += J(x) @ delta * dt
        if (k + 1) % 100 == 0:
            norm = np.linalg.norm(delta)
            delta /= norm
            sum_log += np.log(norm)
    return sum_log / (T)

lyap_grid = np.full((len(eta_vals), len(beta_vals)), np.nan)
for i, eta in enumerate(eta_vals):
    for j, beta in enumerate(beta_vals):
        sol = fixed_points[i, j]
        if not np.any(np.isnan(sol)):
            F_fixed = lambda x, en=eta, bp=beta: F3(x, en, bp)
            J_fixed = lambda x, en=eta, bp=beta: J3(x, en, bp)
            lyap_grid[i, j] = max_lyapunov(
                lambda x, en=eta, bp=beta: F3(x, en, bp),
                lambda x, en=eta, bp=beta: J3(x, en, bp),
                sol
            )

# ==================== PLOTTING ====================
# 1. Bifurcation diagram
plt.figure()
etas, betas = zip(*bif_hopf)
plt.scatter(etas, betas, s=10, c='red')
plt.xlabel('eta_M'); plt.ylabel('beta_AP')
plt.title('Hopf Bifurcation Points')
plt.tight_layout(); plt.show()

# 2. Lyapunov heatmap
plt.figure()
plt.imshow(lyap_grid, origin='lower',
           extent=(beta_vals.min(), beta_vals.max(), eta_vals.min(), eta_vals.max()),
           aspect='auto', cmap='viridis')
plt.colorbar(label='Max Lyapunov Exponent')
plt.xlabel('beta_AP'); plt.ylabel('eta_M')
plt.title('Lyapunov Exponent Heatmap')
plt.tight_layout(); plt.show()

# 3. Sample phase portrait at a chaotic point
eta_c, beta_c = eta_vals[-1], beta_vals[-1]
x0 = fixed_points[-1, -1]
traj = np.zeros((5000, 3))
traj[0] = x0
for t in range(1, traj.shape[0]):
    traj[t] = traj[t-1] + F3(traj[t-1], eta_c, beta_c) * dt_lyap

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(traj[:,0], traj[:,1], traj[:,2], lw=0.5)
ax.set_xlabel('m'); ax.set_ylabel('w'); ax.set_zlabel('A')
ax.set_title('Phase Portrait (m, w, A)')
plt.tight_layout(); plt.show()
