"""Figuras de la portada de las notas: una ilustración por parte, con los colores del IC y del DM.

* portada-I.pdf   — van der Pol: órbitas que se acercan al ciclo límite (Parte I).
* portada-II.pdf  — sumas parciales de Fourier de una onda cuadrada, en cascada (Parte II).
* portada-III.pdf — un pulso en una cuerda con extremos fijos: se divide en dos y se refleja invertido (Parte III).

Correr desde la raíz del repositorio: ``python tools/fig_portada.py`` (escribe en ``figuras/``).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp

CARPETA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figuras")
cm = LinearSegmentedColormap.from_list("ic", ["#A8000C", "#013274", "#029BDF", "#E2017B"])
W, H = 3.0, 2.6


def guardar(fig, ax, nombre):
    ax.set_axis_off()
    fig.savefig(os.path.join(CARPETA, nombre), transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# I: van der Pol, coloreado por el tiempo
lam = 0.8
def vdp(t, u):
    return [u[1], -u[0] - lam * (u[0] ** 2 - 1) * u[1]]

fig, ax = plt.subplots(figsize=(W, H))
for x0, T in [((0.05, 0.0), 30), ((3.3, 0.0), 12), ((-3.3, 0.0), 12)]:
    s = solve_ivp(vdp, (0, T), x0, max_step=0.01, rtol=1e-9)
    x, y = s.y
    n = len(x)
    for i in range(0, n - 1, 5):
        ax.plot(x[i:i + 6], y[i:i + 6], color=cm(0.95 * i / n), lw=1.0, solid_capstyle="round")
ax.set_aspect("equal")
guardar(fig, ax, "portada-I.pdf")

# II: sumas parciales de la onda cuadrada (atrás la más suave, adelante la función)
t = np.linspace(-1, 1, 2000)
f = np.where(np.abs(t) < 0.5, 1.0, -1.0)
def S(N):
    s = np.zeros_like(t)
    for k in range(1, N + 1, 2):
        s += 4 / np.pi * (-1) ** ((k - 1) // 2) * np.cos(np.pi * k * t) / k
    return s

capas = [S(N) for N in range(1, 24, 2)] + [f]
n = len(capas)
fig, ax = plt.subplots(figsize=(W, H))
for j, y in enumerate(capas):
    d = n - 1 - j
    ax.plot(t + 0.035 * d, 0.6 * y + 0.11 * d, color=cm(d / (n - 1)), lw=0.9 if d else 1.2, zorder=n - d)
guardar(fig, ax, "portada-II.pdf")

# III: pulso en una cuerda con extremos fijos (d'Alembert); adelante t = 0, hacia atrás el tiempo avanza
x = np.linspace(0, 1, 1200)
def g(z):
    """Extensión impar y 2-periódica de un pulso gaussiano centrado en 0.4."""
    z = np.mod(z + 1, 2) - 1
    return np.exp(-((z - 0.4) / 0.05) ** 2) - np.exp(-((z + 0.4) / 0.05) ** 2)

tiempos = np.linspace(0, 0.64, 9)
n = len(tiempos)
fig, ax = plt.subplots(figsize=(W, H))
for j, tt in enumerate(tiempos[::-1]):
    d = n - 1 - j
    u = 0.5 * (g(x - tt) + g(x + tt))
    ax.plot(x + 0.045 * d, 0.3 * u + 0.16 * d, color=cm(d / (n - 1)), lw=0.9 if d else 1.2, zorder=n - d)
guardar(fig, ax, "portada-III.pdf")
print("portada-I.pdf, portada-II.pdf, portada-III.pdf escritas en", CARPETA)
