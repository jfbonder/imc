"""Figuras de la portada de las notas: una por parte, cada una con datos reales de un problema conductor.

* portada-I.pdf   — gripe en un internado (1978): el SIR ajustado con los datos de los primeros 6, 7, ..., 14 días.
* portada-II.pdf  — temperatura horaria de 2023 (Aeroparque): espectro de amplitudes (DFT).
* portada-III.pdf — temperatura del suelo a cuatro profundidades, diez días de enero de 2023.

Usa LaTeX para el texto de las figuras (text.usetex). Correr desde cualquier lado:
``python tools/fig_portada.py`` (escribe en ``figuras/``).
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CAR = os.path.join(RAIZ, "figuras")
NAVY, CYAN, MAG, RED = "#013274", "#029BDF", "#E2017B", "#A8000C"
cm = LinearSegmentedColormap.from_list("ic", [NAVY, CYAN, MAG])
W, H = 2.3, 2.0
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}", "font.family": "serif", "font.size": 7.5, "axes.edgecolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY, "axes.labelcolor": NAVY})
def limpio(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=2.5, labelsize=6)
def guardar(fig, nombre):
    fig.savefig(os.path.join(CAR, nombre), transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# I: gripe del internado (1978): ajustes del SIR a medida que llegan los datos
d = np.genfromtxt(f"{RAIZ}/datos/gripe_internado_1978.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
t_d, I_d, N = d["dia"].astype(float), d["en_cama"].astype(float), 763
def I_mod(t, b, g, i0):
    s = solve_ivp(lambda t, u: [-b*u[0]*u[1], b*u[0]*u[1] - g*u[1]], (1, t[-1]), [1-i0, i0], t_eval=t, rtol=1e-8, atol=1e-10)
    return N*s.y[1] if s.success and s.y.shape[1] == len(t) else np.full(len(t), 1e6)
tt = np.linspace(1, 14, 300)
fig, ax = plt.subplots(figsize=(W, H)); limpio(ax)
ms = list(range(6, 15))
for j, m in enumerate(ms):
    a = least_squares(lambda p: I_mod(t_d[:m], *p) - I_d[:m], [1.8, 0.45, 0.003], bounds=([0.02, 0.02, 0], [20, 20, 0.5]))
    ax.plot(tt, I_mod(tt, *a.x), color=cm(j/(len(ms)-1)), lw=0.9)
ax.plot(t_d, I_d, "o", ms=3.2, color=RED, zorder=5)
ax.set_xticks(range(2, 15, 2)); ax.set_xlabel("día (desde el 22/1/1978)"); ax.set_ylabel("alumnos en cama")
guardar(fig, "portada-I.pdf")

# II: temperatura horaria de 2023 (Aeroparque): espectro
T = np.genfromtxt(f"{RAIZ}/datos/temperatura_horaria.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
x = T["temp"][-8760:].astype(float)
A = np.abs(np.fft.rfft(x - x.mean())) * 2 / len(x)
k = np.arange(len(A))
fig, ax = plt.subplots(figsize=(W, H)); limpio(ax)
ax.loglog(k[1:], A[1:], color=NAVY, lw=0.5)
for kk, txt in [(1, "1 año"), (365, "1 día"), (730, "12 h")]:
    ax.plot(kk, A[kk], "o", ms=3.5, color=RED, zorder=5)
    ax.annotate(txt, (kk, A[kk]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=RED)
ax.set_xlabel("frecuencia (ciclos por año)"); ax.set_ylabel("amplitud ($^\\circ$C)")
guardar(fig, "portada-II.pdf")

# III: temperatura del suelo, diez días de enero de 2023, a cuatro profundidades
S = np.genfromtxt(f"{RAIZ}/datos/suelo_horaria.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
cols = ["t_suelo_0_7", "t_suelo_7_28", "t_suelo_28_100", "t_suelo_100_255"]
nombres = ["0–7 cm", "7–28 cm", "28–100 cm", "100–255 cm"]
i0, i1 = 24*9, 24*19
dias = np.arange(i0, i1) / 24 + 1
fig, ax = plt.subplots(figsize=(W, H)); limpio(ax)
ys, etiquetas = [], []
for j, (c, n_) in enumerate(zip(cols, nombres)):
    y = S[c][i0:i1].astype(float)
    ax.plot(dias, y, color=cm(j/3), lw=1.0)
    ys.append(y[-24:].mean())
    etiquetas.append((n_, cm(j/3)))
orden = np.argsort(ys)[::-1]; pos = {}; ult = None
for o in orden:                      # separar las etiquetas al menos 1.4 grados
    p = ys[o] if ult is None else min(ys[o], ult - 1.4)
    pos[o] = p; ult = p
for o, (n_, col) in enumerate(etiquetas):
    ax.text(dias[-1] + 0.25, pos[o], n_, fontsize=6, color=col, va="center")
ax.set_xlabel("día (enero de 2023)"); ax.set_ylabel("temperatura del suelo ($^\\circ$C)")
guardar(fig, "portada-III.pdf")

print("portada-I.pdf, portada-II.pdf, portada-III.pdf escritas en", CAR)
