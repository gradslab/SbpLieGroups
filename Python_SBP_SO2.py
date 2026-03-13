# SB_SO2_fft_log_ribbons_clean_dark.py
# Schrödinger Bridge on SO(2) via FFT + log-domain Sinkhorn with
# Hilbert semilogy plot + polar ribbons + animation
# All outputs are saved in the assets/ folder

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def main():
    # ============================================================
    # USER TOGGLES
    # ============================================================
    SAVE_JPG = True
    SAVE_EPS = False
    DO_ANIMATION = True
    SHOW_PLOTS = True
    USE_TEX = False   # Set True only if LaTeX is installed

    # ============================================================
    # MATPLOTLIB SETUP
    # ============================================================
    plt.rcParams["text.usetex"] = USE_TEX
    plt.rcParams.update({
        "font.size": 14
    })

    # ============================================================
    # PARAMETERS
    # ============================================================
    N = 1024
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    T = 1.0
    sigma = 0.43
    maxIter = 6000
    tolChange = 1e-10
    epsfloor = 1e-300

    # ============================================================
    # TEST CASE
    # If testCase = k then draws all figures for Test k, k = 1,2,3,4
    # ============================================================
    testCase = 1

    # ============================================================
    # OUTPUT FOLDER
    # ============================================================
    codeFolder = os.path.dirname(os.path.abspath(__file__))
    assetsFolder = os.path.join(codeFolder, "assets")
    os.makedirs(assetsFolder, exist_ok=True)

    # ============================================================
    # VON MISES ENDPOINTS
    # ============================================================
    def von_mises(theta_vals, mu, kappa):
        return np.exp(kappa * np.cos(theta_vals - mu)) / (2.0 * np.pi * np.i0(kappa))

    if testCase == 1:
        p0 = von_mises(theta, np.pi / 6.0, 40.0)
        p1 = von_mises(theta, 2.0 * np.pi - np.pi / 6.0, 40.0)

    elif testCase == 2:
        p0 = von_mises(theta, np.pi / 4.0, 40.0)
        p1 = von_mises(theta, np.pi / 4.0 + np.pi, 40.0)

    elif testCase == 3:
        p0 = von_mises(theta, np.pi, 30.0)
        p1 = (
            (1.0 / 3.0) * von_mises(theta, 5.0 * np.pi / 12.0, 50.0)
            + (1.0 / 3.0) * von_mises(theta, 0.0, 50.0)
            + (1.0 / 3.0) * von_mises(theta, -5.0 * np.pi / 12.0, 50.0)
        )

    elif testCase == 4:
        p0 = (
            (1.0 / 3.0) * von_mises(theta, np.pi / 6.0, 70.0)
            + (1.0 / 3.0) * von_mises(theta, 0.0, 70.0)
            + (1.0 / 3.0) * von_mises(theta, -np.pi / 6.0, 70.0)
        )
        p1 = (
            0.5 * von_mises(theta, 5.0 * np.pi / 6.0, 50.0)
            + 0.5 * von_mises(theta, -5.0 * np.pi / 6.0, 50.0)
        )

    else:
        raise ValueError("testCase must be 1, 2, 3, or 4.")

    p0 = p0 / np.trapz(p0, theta)
    p1 = p1 / np.trapz(p1, theta)

    # ============================================================
    # SPECTRAL MULTIPLIERS
    # ============================================================
    k = np.arange(N)
    k[k > N // 2] -= N
    lambda_T = np.exp(-0.5 * (sigma ** 2) * T * (k ** 2))

    def applyK(v, lam):
        return np.real(np.fft.ifft(lam * np.fft.fft(v)))

    # ============================================================
    # HILBERT PROJECTIVE METRIC
    # ============================================================
    def safe_log(x):
        return np.log(np.maximum(x, epsfloor))

    def hilbert_projective_metric(p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        ratio = p / q
        return safe_log(np.max(ratio) / np.min(ratio))

    # ============================================================
    # LOG-DOMAIN SINKHORN
    # ============================================================
    a = np.zeros(N)
    b = np.zeros(N)

    d_H_phi_0_hat = []
    d_H_phi_1 = []

    print("Starting stabilized (log-domain) Sinkhorn ...")
    for it in range(maxIter):
        a_old = a.copy()
        b_old = b.copy()

        # Update a
        sb = np.max(b)
        vb_shift = np.exp(b - sb)
        Kv_shift = applyK(vb_shift, lambda_T)
        Kv_shift = np.maximum(Kv_shift, epsfloor)
        logKv = np.log(Kv_shift) + sb
        a = np.log(np.maximum(p0, epsfloor)) - logKv

        phi_0_hat_next = np.exp(a)
        phi_0_hat = np.exp(a_old)
        d_H_phi_0_hat.append(hilbert_projective_metric(phi_0_hat_next, phi_0_hat))

        # Update b
        sa = np.max(a)
        ua_shift = np.exp(a - sa)
        Ku_shift = applyK(ua_shift, lambda_T)
        Ku_shift = np.maximum(Ku_shift, epsfloor)
        logKu = np.log(Ku_shift) + sa
        b = np.log(np.maximum(p1, epsfloor)) - logKu

        phi_1_next = np.exp(b)
        phi_1 = np.exp(b_old)
        d_H_phi_1.append(hilbert_projective_metric(phi_1_next, phi_1))

        da = np.max(np.abs(a - a_old))
        db = np.max(np.abs(b - b_old))

        if da < tolChange and db < tolChange:
            print(f"Converged at iter {it + 1}")
            break
    else:
        print("Reached maxIter without meeting tolerance.")

    # ============================================================
    # SEMILOGY PLOT (HILBERT METRIC)
    # ============================================================
    d_H_phi_0_hat_array = np.array(d_H_phi_0_hat)
    d_H_phi_1_array = np.array(d_H_phi_1)

    tol = 2e-6
    i = 0
    while i < len(d_H_phi_0_hat_array) and i < len(d_H_phi_1_array):
        if d_H_phi_0_hat_array[i] < tol and d_H_phi_1_array[i] < tol:
            break
        i += 1

    if i == 0:
        i = min(len(d_H_phi_0_hat_array), len(d_H_phi_1_array))

    x = np.arange(i)

    fig0, ax0 = plt.subplots(figsize=(8, 5))
    ax0.semilogy(
        x,
        d_H_phi_1_array[:i],
        'o-',
        color='red',
        linewidth=1.5,
        label=r'$d_{\mathrm{Hilbert}}(\varphi_1,(\varphi_1)_{\mathrm{next}})$'
    )

    ax0.semilogy(
        x,
        d_H_phi_0_hat_array[:i],
        'd-',
        color='blue',
        linewidth=1.5,
        label=r'$d_{\mathrm{Hilbert}}(\widehat{\varphi}_0,(\widehat{\varphi}_0)_{\mathrm{next}})$'
    )

    ax0.set_xlabel('Recursion index')
    ax0.set_ylabel('Hilbert projective metric')
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax0.legend()
    plt.tight_layout()

    hilbert_eps = os.path.join(assetsFolder, f"Hilbert_metric_semilogy_plot_Test{testCase}.eps")
    hilbert_jpg = os.path.join(assetsFolder, f"Hilbert_metric_semilogy_plot_Test{testCase}.jpg")

    if SAVE_EPS:
        fig0.savefig(hilbert_eps, format="eps", transparent=True, bbox_inches="tight")
    if SAVE_JPG:
        fig0.savefig(hilbert_jpg, dpi=1000, bbox_inches="tight")

    # ============================================================
    # RECONSTRUCT rho_t
    # ============================================================
    timeGrid = np.linspace(0.0, T, 35)
    rho_t = np.zeros((N, len(timeGrid)))

    ua_shift = np.exp(a - np.max(a))
    vb_shift = np.exp(b - np.max(b))

    for idx, tcur in enumerate(timeGrid):
        lambda_t = np.exp(-0.5 * sigma ** 2 * tcur * (k ** 2))
        lambda_Tmt = np.exp(-0.5 * sigma ** 2 * (T - tcur) * (k ** 2))

        A = applyK(ua_shift, lambda_t)
        B = applyK(vb_shift, lambda_Tmt)

        pt = A * B
        pt = np.maximum(pt, 0.0)
        denom = np.trapz(pt, theta)
        rho_t[:, idx] = pt / np.maximum(denom, epsfloor)

    # ============================================================
    # FIGURE SHOWING ALL INTERMEDIATE PDFs OF ALL TIMES
    # ============================================================
    fig1 = plt.figure(figsize=(9, 9), facecolor='black')
    ax1 = plt.subplot(111, projection='polar', facecolor='black')

    ax1.set_theta_direction(1)
    ax1.set_theta_zero_location("E")
    ax1.tick_params(colors='white', labelsize=14)
    ax1.grid(color=(0.6, 0.6, 0.6), alpha=0.6)
    ax1.spines['polar'].set_color('white')
    ax1.set_title(
        'Initial, Intermediate, and Terminal PDFs on SO(2)',
        color='white',
        fontsize=20,
        pad=20
    )

    r_offset = 1.0

    ax1.plot(theta, p0 + r_offset, '--', color=(0.4, 0.7, 1.0), linewidth=1.7)
    ax1.plot(theta, p1 + r_offset, '--', color=(1.0, 0.4, 0.4), linewidth=1.7)

    cmap_full = plt.cm.jet(np.linspace(0.0, 1.0, rho_t.shape[1]))
    for idx in range(rho_t.shape[1]):
        if idx != 0 and idx != rho_t.shape[1] - 1:
            ax1.plot(theta, rho_t[:, idx] + r_offset, color=cmap_full[idx], linewidth=1.3)

    sm1 = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap='jet')
    sm1.set_array([])
    cbar1 = fig1.colorbar(sm1, ax=ax1, pad=0.12)
    cbar1.set_label('time (sec)', color='white')
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar1.ax.get_yticklabels(), color='white')

    figure1_jpg = os.path.join(assetsFolder, f"Figure_Test{testCase}.jpg")
    figure1_eps = os.path.join(assetsFolder, f"Figure_Test{testCase}.eps")

    if SAVE_JPG:
        fig1.savefig(figure1_jpg, dpi=1000, facecolor=fig1.get_facecolor(), bbox_inches='tight')
    if SAVE_EPS:
        fig1.savefig(figure1_eps, format='eps', facecolor=fig1.get_facecolor(), bbox_inches='tight')

    # ============================================================
    # BIG LONG FIGURE WITH 5 SNAPSHOTS
    # ============================================================
    numSnapshots = 5
    snapshotIdx = np.round(np.linspace(0, len(timeGrid) - 1, numSnapshots)).astype(int)

    fig2 = plt.figure(figsize=(18, 4.5), facecolor='black')
    r_offset = 1.0
    cmap_snap = plt.cm.jet(np.linspace(0.0, 1.0, numSnapshots))

    for i_snap in range(numSnapshots):
        xPos = 0.02 + 0.18 * i_snap
        width = 0.17
        ax = fig2.add_axes([xPos, 0.14, width, 0.76], facecolor='black')

        ax.set_aspect('equal')
        ax.axis('off')

        R = rho_t[:, snapshotIdx[i_snap]] + r_offset
        R0 = r_offset * np.ones_like(theta)

        x_curve = R * np.cos(theta)
        y_curve = R * np.sin(theta)
        x_base = R0 * np.cos(theta)
        y_base = R0 * np.sin(theta)

        ax.fill(
            np.concatenate([x_base, x_curve[::-1]]),
            np.concatenate([y_base, y_curve[::-1]]),
            color=cmap_snap[i_snap],
            alpha=0.6,
            edgecolor='none'
        )

        ax.plot(x_curve, y_curve, color=cmap_snap[i_snap], linewidth=4.2)

        x0p = (p0 + r_offset) * np.cos(theta)
        y0p = (p0 + r_offset) * np.sin(theta)
        x1p = (p1 + r_offset) * np.cos(theta)
        y1p = (p1 + r_offset) * np.sin(theta)

        ax.plot(x0p, y0p, '--', color='blue', linewidth=2.5)
        ax.plot(x1p, y1p, '--', color='red', linewidth=2.5)

        ax.set_title(
            f't = {timeGrid[snapshotIdx[i_snap]]:.2f}',
            color='white',
            fontweight='bold',
            fontsize=22,
            pad=10
        )

    sm2 = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap='jet')
    sm2.set_array([])

    # Manual colorbar axis to control length and thickness independently
    cbar_ax = fig2.add_axes([0.10, 0.06, 0.80, 0.02])
    cbar2 = fig2.colorbar(sm2, cax=cbar_ax, orientation='horizontal')
    cbar2.set_label('Time t', color='white')
    cbar2.ax.xaxis.set_tick_params(color='white')
    plt.setp(cbar2.ax.get_xticklabels(), color='white')

    figure2_jpg = os.path.join(assetsFolder, f"LongFigure_Test{testCase}.jpg")
    figure2_eps = os.path.join(assetsFolder, f"LongFigure_Test{testCase}.eps")

    if SAVE_JPG:
        fig2.savefig(figure2_jpg, dpi=1000, facecolor=fig2.get_facecolor(), bbox_inches='tight')
    if SAVE_EPS:
        fig2.savefig(figure2_eps, format='eps', facecolor=fig2.get_facecolor(), bbox_inches='tight')

    # ============================================================
    # ANIMATION SETUP
    # ============================================================
    gif_path = os.path.join(assetsFolder, f"Animation_Test{testCase}.gif")

    if DO_ANIMATION:
        fig3 = plt.figure(figsize=(9, 7), facecolor='black')
        ax3 = plt.subplot(111, projection='polar', facecolor='black')

        ax3.tick_params(colors='white', labelsize=14)
        ax3.grid(color=(0.7, 0.7, 0.7), alpha=0.3)
        ax3.spines['polar'].set_color('white')

        numFrames = rho_t.shape[1]
        cmap_anim = plt.cm.jet(np.linspace(0.0, 1.0, numFrames))
        r_offset = 1.0

        sm3 = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap='jet')
        sm3.set_array([])
        cbar3 = fig3.colorbar(sm3, ax=ax3, pad=0.12)
        cbar3.set_label('Time t', color='white')
        cbar3.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar3.ax.get_yticklabels(), color='white')

        line_main, = ax3.plot([], [], linewidth=3)
        line_p0, = ax3.plot(theta, p0 + r_offset, '--', color='blue', linewidth=2)
        line_p1, = ax3.plot(theta, p1 + r_offset, '--', color='red', linewidth=2)

        time_text = ax3.text(
            0.5, 1.07, '',
            transform=ax3.transAxes,
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            color='white'
        )

        rmax = max(np.max(rho_t + r_offset), np.max(p0 + r_offset), np.max(p1 + r_offset))
        ax3.set_ylim(0.0, 1.05 * rmax)

        def init():
            line_main.set_data([], [])
            time_text.set_text('')
            return line_main, line_p0, line_p1, time_text

        def update(frame):
            line_main.set_data(theta, rho_t[:, frame] + r_offset)
            line_main.set_color(cmap_anim[frame])
            time_text.set_text(f'Time t = {timeGrid[frame]:.2f}')
            return line_main, line_p0, line_p1, time_text

        anim = FuncAnimation(
            fig3,
            update,
            frames=numFrames,
            init_func=init,
            interval=80,
            blit=False,
            repeat=True
        )

        anim.save(gif_path, writer=PillowWriter(fps=12))

    # ============================================================
    # PRINT SAVED FILES
    # ============================================================
    print("Saved files:")
    if SAVE_EPS:
        print(f"  {hilbert_eps}")
    if SAVE_JPG:
        print(f"  {hilbert_jpg}")
    if SAVE_JPG:
        print(f"  {figure1_jpg}")
        print(f"  {figure2_jpg}")
    if SAVE_EPS:
        print(f"  {figure1_eps}")
        print(f"  {figure2_eps}")
    if DO_ANIMATION:
        print(f"  {gif_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()