"""Figura ranura-placas.png: esquema del ejercicio de las dos placas a tierra (adaptado de Griffiths, ej. 3.3),
dibujado con matplotlib en proyección oblicua."""
import numpy as np, matplotlib.pyplot as plt
from imc import estilo
from imc.estilo import COLORES

estilo.activar(fuente=14)
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.set_aspect("equal"); ax.axis("off")

# proyección oblicua: (x, y, z) -> (x + 0.5 z, y + 0.35 z)
def P(x, y, z): return x + 0.5 * z, y + 0.35 * z
a, X, Z = 1.0, 3.2, 1.6
def poly(pts, **kw):
    ax.add_patch(plt.Polygon([P(*p) for p in pts], closed=True, **kw))

# placa inferior (y=0) y superior (y=a)
poly([(0, 0, 0), (X, 0, 0), (X, 0, Z), (0, 0, Z)], fc="0.85", ec="0.3", lw=1.2)
poly([(0, a, 0), (X, a, 0), (X, a, Z), (0, a, Z)], fc="0.92", ec="0.3", lw=1.2)
# franja en x=0 a potencial V0(y)
poly([(0, 0, 0), (0, a, 0), (0, a, Z), (0, 0, Z)], fc=COLORES["modelo"], ec="0.2", lw=1.2, alpha=0.55)
# ejes
o = P(0, 0, 0)
for (v, name, dx, dy) in [((X + 0.5, 0, 0), "$x$", 0.05, -0.02), ((0, a + 0.55, 0), "$y$", 0.03, 0.03), ((0, 0, Z + 0.9), "$z$", 0.06, -0.02)]:
    e = P(*v); ax.annotate("", xy=e, xytext=o, arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    ax.text(e[0] + dx, e[1] + dy, name, fontsize=15)
# etiquetas
ax.text(*P(X * 0.55, -0.02, Z * 0.5), "$V=0$", ha="center", va="top", fontsize=14)
ax.text(*P(X * 0.55, a + 0.03, Z * 0.5), "$V=0$", ha="center", va="bottom", fontsize=14)
ax.text(*P(0, a * 0.5, Z * 0.55), "$V_0(y)$", ha="center", va="center", fontsize=14, color="0.15", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
ax.text(*P(-0.08, a, 0), "$a$", ha="right", va="center", fontsize=14)
ax.text(*P(X + 0.12, 0.0, Z), r"$x\to\infty$", ha="left", va="center", fontsize=12, color="0.35")
ax.set_xlim(-0.6, X + 1.4); ax.set_ylim(-0.35, a + 1.1)
print(estilo.guardar(fig, "ranura-placas"))
