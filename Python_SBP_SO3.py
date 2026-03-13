import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import shutil
import os

# ============================================================
# System Configuration & LaTeX Safety
# ============================================================
# Check if 'latex' is installed on the system path
latex_exists = shutil.which("latex") is not None

plt.rcParams.update({
    "font.size": 14,
    "text.usetex": latex_exists,
})

if not latex_exists:
    print("!!! LaTeX not found. Using standard Matplotlib fonts (Computer Modern style).")
    # Mimic LaTeX look as closely as possible
    plt.rcParams["mathtext.fontset"] = "cm" 
    plt.rcParams["font.family"] = "STIXGeneral"
else:
    print("--- LaTeX detected. Professional rendering enabled.")

# ============================================================
# Parameters
# ============================================================
ell_max = 60
sigma = 0.5
Nw = 450
T = 1.0

omega = np.linspace(1e-6, np.pi, Nw)
domega = omega[1] - omega[0]

maxIter = 2000
tolChange = 1e-14
epsfloor = 1e-300

# Haar weight (after integrating axis)
w = np.sin(omega/2)**2
w = w / np.sum(w * domega)

# ------------------------------------------------------------
# Character matrix χ_ell(ω) for SO(3)
# ------------------------------------------------------------
Chi = np.zeros((ell_max+1, Nw))
small = 1e-8

for ell in range(ell_max+1):
    numerator = np.sin((ell + 0.5)*omega)
    denominator = np.sin(omega/2)
    
    ratio = np.empty_like(omega)
    mask = np.abs(omega) < small
    
    ratio[mask] = 2*ell + 1
    ratio[~mask] = numerator[~mask] / denominator[~mask]
    
    Chi[ell,:] = ratio

# ------------------------------------------------------------
# Spectral transforms & Helpers
# ------------------------------------------------------------
def zonal_forward(f):
    return np.array([
        np.sum(f * Chi[ell,:] * w * domega)
        for ell in range(ell_max+1)
    ])

def zonal_inverse(fhat):
    return np.sum(fhat[:,None] * Chi, axis=0)

def heat_apply(f, t):
    fhat = zonal_forward(f)
    for ell in range(ell_max+1):
        fhat[ell] *= np.exp(-ell*(ell+1)*sigma**2*t)
    return zonal_inverse(fhat)

def normalize_pdf(f):
    return f / np.sum(f * w * domega)

def safe_log(x):
    return np.log(np.maximum(x, epsfloor))

def zonal_von_mises(omega, mu, kappa):
    return np.exp(kappa*np.cos(omega - mu))

def hilbert_projective_metric(p, q):
    p, q = np.asarray(p), np.asarray(q)
    ratio = p / q
    return safe_log(np.max(ratio) / np.min(ratio))

# ------------------------------------------------------------
# Initial / Terminal Densities
# ------------------------------------------------------------
p0 = normalize_pdf(zonal_von_mises(omega, mu=1.0, kappa=30))
p1 = normalize_pdf(zonal_von_mises(omega, mu=2.0, kappa=30))

# ------------------------------------------------------------
# Log-domain Sinkhorn (spectral operator)
# ------------------------------------------------------------
a, b = np.zeros_like(p0), np.zeros_like(p1)
d_H_phi_0_hat, d_H_phi_1 = [], []

print("Starting Sinkhorn iterations...")

for it in range(maxIter):
    a_old, b_old = a.copy(), b.copy()

    # update a
    sb = np.max(b)
    vb = np.exp(b - sb)
    Kv = np.maximum(heat_apply(vb, T), epsfloor)
    a = safe_log(p0) - (safe_log(Kv) + sb)

    # Metrics for Convergence
    phi_0_hat_next, phi_0_hat = np.exp(a), np.exp(a_old)
    d_H_phi_0_hat.append(hilbert_projective_metric(phi_0_hat_next, phi_0_hat))

    # update b
    sa = np.max(a)
    ua = np.exp(a - sa)
    Ku = np.maximum(heat_apply(ua, T), epsfloor)
    b = safe_log(p1) - (safe_log(Ku) + sa)

    phi_1_next, phi_1 = np.exp(b), np.exp(b_old)
    d_H_phi_1.append(hilbert_projective_metric(phi_1_next, phi_1))

    da, db = np.max(np.abs(a - a_old)), np.max(np.abs(b - b_old))

    if it % 1 == 0:
        print(f"iter {it:4d}: da={da:.3e}, db={db:.3e}")

    if max(da, db) < tolChange:
        print(f"Converged at iteration {it}")
        break

# ------------------------------------------------------------
# Plot 1: Hilbert Metric Convergence
# ------------------------------------------------------------
d_H_phi_0_hat_array = np.array(d_H_phi_0_hat)
d_H_phi_1_array = np.array(d_H_phi_1)

i = 0
while i < len(d_H_phi_0_hat_array) and i < len(d_H_phi_1_array):
    if d_H_phi_0_hat_array[i] < tolChange and d_H_phi_1_array[i] < tolChange:
        break
    i += 1

x_idx = np.arange(i)
fig, ax = plt.subplots(figsize=(7,5))

plt.semilogy(x_idx, d_H_phi_1_array[:i], 'o--', color='blue', 
             label=r'$d_{\mathrm{Hilbert}}(\varphi_1, (\varphi_1)_{\mathrm{next}})$')
plt.semilogy(x_idx, d_H_phi_0_hat_array[:i], 'd-', color='blue', 
             label=r'$d_{\mathrm{Hilbert}}(\widehat{\varphi}_0, (\widehat{\varphi}_0)_{\mathrm{next}})$')

ax.set_xlabel('Recursion index', color='blue')
ax.set_ylabel('Hilbert projective metric', color='blue')
ax.xaxis.set_label_position('top')
ax.yaxis.set_label_position('right')
ax.xaxis.tick_top()
ax.yaxis.tick_right()
ax.tick_params(axis='both', colors='blue')

for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_color('blue')

ax.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("Hilbert_metric_semilogy_plot.png", format="png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Plot 2: Reconstruct Time Marginals
# ------------------------------------------------------------
u, v = np.exp(a), np.exp(b)
Nt = 50
tgrid = np.linspace(0, T, Nt)
rho_t = np.zeros((Nt, Nw))

for k, t in enumerate(tgrid):
    left  = heat_apply(u, t)
    right = heat_apply(v, T-t)
    rho_t[k,:] = normalize_pdf(left * right)

print("Reconstruction complete.")

plt.figure(figsize=(8,5))
Num_lines = 10
for idx in np.linspace(0, Nt-1, Num_lines, dtype=int):
    t_val = tgrid[idx]
    plt.plot(omega, rho_t[idx,:], linewidth=2, label=f"t={t_val:.2f}")

plt.plot(omega, p0, 'k--', linewidth=3, label=r"$\rho_0$ at $t=0$")
plt.plot(omega, p1, 'r--', linewidth=3, label=r"$\rho_1$ at $t=1$")

plt.xlabel(r"Rotation angle magnitude $\|\omega\|_2$")
plt.ylabel(r"Density $\rho$")
plt.legend(loc="upper left", fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.savefig("schrodinger_bridge_SO3.png", format="png", bbox_inches="tight")
plt.show()