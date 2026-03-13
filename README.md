# Spectral Schrödinger Bridge on SO(3)

A **spectral implementation of the Schrödinger Bridge problem on the rotation group $SO(3)$** using zonal harmonic expansions and log-domain Sinkhorn iterations.

This project computes **entropic optimal transport on $SO(3)$** using the heat kernel as the reference process.

The implementation is designed for **research and visualization of Schrödinger bridges on compact Lie groups**.

---

# Overview

The Schrödinger bridge solves the stochastic control problem

$$
\min_{\rho_t}
\; \mathrm{KL}\!\left(\rho_t \,\|\, \text{Brownian motion}\right)
$$

subject to

$$
\rho_0 = p_0, \qquad
\rho_T = p_1
$$

On $SO(3)$, the heat semigroup admits a **spectral representation**

$$
K_t f
=
\sum_{\ell=0}^{\infty}
e^{-\ell(\ell+1)\sigma^2 t}
\hat f_\ell
\chi_\ell(\omega)
$$

where

- $\chi_\ell$ are the **characters of $SO(3)$**
- $\hat f_\ell$ are spectral coefficients

The Schrödinger bridge density is reconstructed as

$$
\rho_t(\omega)
=
(K_t u)(\omega)
\,
(K_{T-t} v)(\omega)
$$

where $u,v$ are obtained via **Sinkhorn iterations**.

---

# Features

- Spectral representation of the **$SO(3)$ heat kernel**
- **Log-domain Sinkhorn algorithm**
- Hilbert projective metric convergence analysis
- Time evolution of optimal density
- Publication-quality figures
- Density **animation**

---

# Example Outputs

## Schrödinger Bridge Density Evolution

![Density evolution](assets/schrodinger_bridge_SO3.png)

The plot shows the optimal probability density interpolating between the initial and terminal distributions on $SO(3)$.

---

## Sinkhorn Convergence (Hilbert Metric)

![Hilbert metric convergence](assets/Hilbert_metric_semilogy_plot.png)

The Hilbert projective metric demonstrates the contraction of the Sinkhorn updates.

---

## Density Animation

![Bridge animation](assets/schrodinger_bridge_SO3_animation.gif)

The animation shows the time evolution of the optimal density

$$
\rho_t(\omega)
$$

between $t=0$ and $t=T$.

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/schrodinger-bridge-so3.git
cd schrodinger-bridge-so3
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Code

Execute the script

```bash
python schrodinger_bridge_so3.py
```

The program will:

1. Solve the Schrödinger bridge using Sinkhorn iterations
2. Compute the time-marginal densities
3. Generate plots
4. Generate an animation

All outputs are saved in

```
assets/
```

---

# User Controls

The script includes simple toggles:

```python
SAVE_EPS = True
SAVE_PNG = True
DO_ANIMATION = True
```

These allow you to enable or disable:

- figure exports
- animation generation

---

# Repository Structure

```
.
├── schrodinger_bridge_so3.py
├── requirements.txt
├── README.md
└── assets
    ├── schrodinger_bridge_SO3.png
    ├── Hilbert_metric_semilogy_plot.png
    └── schrodinger_bridge_SO3_animation.gif
```

---

# Mathematical Background

The algorithm is based on ideas from

- **Schrödinger bridges**
- **Entropic optimal transport**
- **Heat kernels on compact Lie groups**
- **Sinkhorn scaling**
- **Hilbert projective metric contraction**

Key references include

- Schrödinger (1931)
- Sinkhorn (1967)
- Léonard (2014)
- Peyré & Cuturi (2019)

---

# Potential Extensions

Possible extensions of this project include

- Schrödinger bridges on **$SE(3)$**
- Non-zonal densities on $SO(3)$
- Higher resolution spectral expansions
- GPU acceleration
- Wasserstein comparison with classical optimal transport

---

# License

MIT License
