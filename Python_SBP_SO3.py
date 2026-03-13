# ============================================================
# Spectral Zonal Schrödinger Bridge on SO(3)
# Single-file version with plot/animation toggles
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# USER TOGGLES
# ============================================================
SAVE_EPS = False
SAVE_PNG = True
DO_ANIMATION = True

PLOT_HILBERT = True
PLOT_DENSITIES = True
SHOW_PLOTS = True
USE_TEX = False   # Set True only if LaTeX is installed and working
DARK_THEME = True

# ============================================================
# OUTPUT FOLDER
# ============================================================
ASSET_DIR = "assets"
os.makedirs(ASSET_DIR, exist_ok=True)

# ============================================================
# MATPLOTLIB SETUP
# ============================================================
plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams.update({
    "font.size": 14
})

# ============================================================
# THEME SETUP
# ============================================================
if DARK_THEME:
    FIG_FACE = "black"
    AX_FACE = "black"
    TEXT_COLOR = "white"
    GRID_COLOR = (0.7, 0.7, 0.7)
    SPINE_COLOR = "white"
else:
    FIG_FACE = "white"
    AX_FACE = "white"
    TEXT_COLOR = "black"
    GRID_COLOR = (0.7, 0.7, 0.7)
    SPINE_COLOR = "black"

# ============================================================
# PARAMETERS
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
w = np.sin(omega / 2) ** 2
w = w / np.sum(w * domega)

# ============================================================
# CHARACTER MATRIX chi_ell(omega)
# ============================================================
Chi = np.zeros((ell_max + 1, Nw))
small = 1e-8

for ell in range(ell_max + 1):
    numerator = np.sin((ell + 0.5) * omega)
    denominator = np.sin(omega / 2)

    ratio = np.empty_like(omega)
    mask = np.abs(omega) < small

    ratio[mask] = 2 * ell + 1
    ratio[~mask] = numerator[~mask] / denominator[~mask]

    Chi[ell, :] = ratio

# ============================================================
# SPECTRAL TRANSFORMS
# ============================================================
def zonal_forward(f):
    return np.array([
        np.sum(f * Chi[ell, :] * w * domega)
        for ell in range(ell_max + 1)
    ])


def zonal_inverse(fhat):
    return np.sum(fhat[:, None] * Chi, axis=0)

# ============================================================
# HEAT SEMIGROUP (SPECTRAL)
# ============================================================
def heat_apply(f, t):
    fhat = zonal_forward(f)
    for ell in range(ell_max + 1):
        fhat[ell] *= np.exp(-ell * (ell + 1) * sigma**2 * t)
    return zonal_inverse(fhat)

# ============================================================
# NORMALIZATION AND SAFE LOG
# ============================================================
def normalize_pdf(f):
    return f / np.sum(f * w * domega)


def safe_log(x):
    return np.log(np.maximum(x, epsfloor))

# ============================================================
# INITIAL / TERMINAL DENSITIES
# ============================================================
def zonal_von_mises(omega, mu, kappa):
    return np.exp(kappa * np.cos(omega - mu))


def hilbert_projective_metric(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    ratio = p / q
    return safe_log(np.max(ratio) / np.min(ratio))


# For the graph in the draft
p0 = zonal_von_mises(omega, mu=1.0, kappa=30)
p1 = zonal_von_mises(omega, mu=2.0, kappa=30)

p0 = normalize_pdf(p0)
p1 = normalize_pdf(p1)

# ============================================================
# LOG-DOMAIN SINKHORN
# ============================================================
a = np.zeros_like(p0)
b = np.zeros_like(p1)

d_H_phi_0_hat = []
d_H_phi_1 = []

da_list = []
db_list = []

print("Starting Sinkhorn...")

for it in range(maxIter):
    a_old = a.copy()
    b_old = b.copy()

    # Update a
    sb = np.max(b)
    vb = np.exp(b - sb)

    Kv = heat_apply(vb, T)
    Kv = np.maximum(Kv, epsfloor)

    a = safe_log(p0) - (safe_log(Kv) + sb)

    phi_0_hat_next = np.exp(a)
    phi_0_hat = np.exp(a_old)
    d_H_phi_0_hat.append(hilbert_projective_metric(phi_0_hat_next, phi_0_hat))

    # Update b
    sa = np.max(a)
    ua = np.exp(a - sa)

    Ku = heat_apply(ua, T)
    Ku = np.maximum(Ku, epsfloor)

    b = safe_log(p1) - (safe_log(Ku) + sa)

    phi_1_next = np.exp(b)
    phi_1 = np.exp(b_old)
    d_H_phi_1.append(hilbert_projective_metric(phi_1_next, phi_1))

    da = np.max(np.abs(a - a_old))
    db = np.max(np.abs(b - b_old))

    da_list.append(da)
    db_list.append(db)

    print(f"iter {it:4d}: da={da:.3e}, db={db:.3e}")

    if max(da, db) < tolChange:
        print(f"Converged at iteration {it}")
        break

d_H_phi_0_hat_array = np.array(d_H_phi_0_hat)
d_H_phi_1_array = np.array(d_H_phi_1)

# ============================================================
# HILBERT METRIC PLOT
# ============================================================
if PLOT_HILBERT:
    tol = tolChange
    i = 0
    while i < len(d_H_phi_0_hat_array) and i < len(d_H_phi_1_array):
        if d_H_phi_0_hat_array[i] < tol and d_H_phi_1_array[i] < tol:
            break
        i += 1

    x = np.arange(i)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=FIG_FACE)
    ax.set_facecolor(AX_FACE)

    plt.semilogy(
        x,
        d_H_phi_1_array[:i],
        'o--',
        color='red' if DARK_THEME else 'blue',
        label=r'$d_{\mathrm{Hilbert}}(\varphi_1,(\varphi_1)_{\mathrm{next}})$'
    )

    plt.semilogy(
        x,
        d_H_phi_0_hat_array[:i],
        'd-',
        color='cyan' if DARK_THEME else 'blue',
        label=r'$d_{\mathrm{Hilbert}}(\widehat{\varphi}_0,(\widehat{\varphi}_0)_{\mathrm{next}})$'
    )

    ax.set_xlabel('Recursion index', color=TEXT_COLOR)
    ax.set_ylabel('Hilbert projective metric', color=TEXT_COLOR)

    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()

    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color=GRID_COLOR)
    leg = ax.legend()
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
    leg.get_frame().set_facecolor(AX_FACE)
    leg.get_frame().set_edgecolor(SPINE_COLOR)

    plt.tight_layout()

    if SAVE_EPS:
        plt.savefig(
            os.path.join(ASSET_DIR, "Hilbert_metric_semilogy_plot.eps"),
            format="eps",
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    if SAVE_PNG:
        plt.savefig(
            os.path.join(ASSET_DIR, "Hilbert_metric_semilogy_plot.png"),
            format="png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# ============================================================
# RECONSTRUCT TIME MARGINALS
# ============================================================
u = np.exp(a)
v = np.exp(b)

Nt = 50
tgrid = np.linspace(0, T, Nt)
rho_t = np.zeros((Nt, Nw))

for k, t in enumerate(tgrid):
    left = heat_apply(u, t)
    right = heat_apply(v, T - t)

    rho = left * right
    rho = normalize_pdf(rho)
    rho_t[k, :] = rho

print("Reconstruction complete.")

# ============================================================
# DENSITY PLOT
# ============================================================
if PLOT_DENSITIES:
    fig = plt.figure(figsize=(8, 5), facecolor=FIG_FACE)
    ax = plt.gca()
    ax.set_facecolor(AX_FACE)

    Num = 10
    for idx in np.linspace(0, Nt - 1, Num, dtype=int):
        t_val = tgrid[idx]
        plt.plot(
            omega,
            rho_t[idx, :],
            linewidth=2,
            label=r"$\rho^{\mathrm{opt}}$ at" + f" t={t_val:.2f}"
        )

    plt.plot(omega, p0, 'k--' if not DARK_THEME else '--', color='white' if DARK_THEME else 'k', linewidth=3, label=r"$\rho_0$ at $t=0.00$")
    plt.plot(omega, p1, 'r--', linewidth=3, label=r"$\rho_1$ at $t=1.00$")

    plt.xlabel(r"Rotation angle magnitude $\|\omega\|_2$", color=TEXT_COLOR)
    plt.ylabel("Density", color=TEXT_COLOR)
    plt.xticks(color=TEXT_COLOR)
    plt.yticks(color=TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)

    plt.grid(True, color=GRID_COLOR)
    leg = plt.legend(loc="upper left", fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
    leg.get_frame().set_facecolor(AX_FACE)
    leg.get_frame().set_edgecolor(SPINE_COLOR)

    plt.tight_layout()

    if SAVE_EPS:
        plt.savefig(
            os.path.join(ASSET_DIR, "schrodinger_bridge_SO3.eps"),
            format="eps",
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    if SAVE_PNG:
        plt.savefig(
            os.path.join(ASSET_DIR, "schrodinger_bridge_SO3.png"),
            format="png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# ============================================================
# ANIMATION OF DENSITIES
# ============================================================
if DO_ANIMATION:
    fig_anim, ax_anim = plt.subplots(figsize=(8, 5), facecolor=FIG_FACE)
    ax_anim.set_facecolor(AX_FACE)

    line_rho, = ax_anim.plot([], [], linewidth=2, label=r"$\rho^{\mathrm{opt}}(\omega,t)$")
    line_p0, = ax_anim.plot(omega, p0, 'k--' if not DARK_THEME else '--', color='white' if DARK_THEME else 'k', linewidth=2.5, label=r"$\rho_0$")
    line_p1, = ax_anim.plot(omega, p1, 'r--', linewidth=2.5, label=r"$\rho_1$")

    ax_anim.set_xlim(omega[0], omega[-1])

    ymin = min(np.min(rho_t), np.min(p0), np.min(p1))
    ymax = max(np.max(rho_t), np.max(p0), np.max(p1))
    pad = 0.05 * (ymax - ymin + 1e-15)
    ax_anim.set_ylim(ymin - pad, ymax + pad)

    ax_anim.set_xlabel(r"Rotation angle magnitude $\|\omega\|_2$", color=TEXT_COLOR)
    ax_anim.set_ylabel("Density", color=TEXT_COLOR)
    ax_anim.tick_params(axis='x', colors=TEXT_COLOR)
    ax_anim.tick_params(axis='y', colors=TEXT_COLOR)

    for spine in ax_anim.spines.values():
        spine.set_color(SPINE_COLOR)

    title = ax_anim.set_title(
        r"Schrödinger Bridge on $\mathsf{SO}(3)$, $t=0.00$",
        color=TEXT_COLOR
    )
    ax_anim.grid(True, color=GRID_COLOR)

    leg = ax_anim.legend(loc="upper left", fontsize=10)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
    leg.get_frame().set_facecolor(AX_FACE)
    leg.get_frame().set_edgecolor(SPINE_COLOR)

    def init():
        line_rho.set_data([], [])
        title.set_text(r"Schrödinger Bridge on $\mathsf{SO}(3)$, $t=0.00$")
        return line_rho, title

    def update(frame):
        line_rho.set_data(omega, rho_t[frame, :])
        title.set_text(
            r"Schrödinger Bridge on $\mathsf{SO}(3)$"
            + f", t={tgrid[frame]:.2f}"
        )
        return line_rho, title

    anim = FuncAnimation(
        fig_anim,
        update,
        frames=Nt,
        init_func=init,
        blit=False,
        interval=120
    )

    anim.save(
        os.path.join(ASSET_DIR, "schrodinger_bridge_SO3_animation.gif"),
        writer=PillowWriter(fps=10)
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_anim)