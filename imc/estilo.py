"""Estilo común de las figuras.

Todas las figuras de las notas se generan con ``estilo.activar()`` al comienzo del
notebook y se guardan con ``estilo.guardar(fig, "nombre")``, que escribe
``figuras/nombre.png`` (o ``nombre.pdf``) con la resolución adecuada para LaTeX.

Convenciones
------------
* Ancho de figura: 0.6\\textwidth de las notas ~ 9.3 cm; usamos 6 x 4 pulgadas y
  fuente 11 pt, de modo que al escalar quede legible.
* Colores: una paleta corta y consistente (``COLORES``): azul para trayectorias,
  naranja/verde para nulclinas, negro para equilibrios, gris para el campo.
* Los nombres de los ejes son los del texto (``h``, ``p``; ``s``, ``i``; ``x``,
  ``y``), no ``x, y`` genéricos.
"""

from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORES = {
    "traj": "#1f4e9c",     # trayectorias
    "traj2": "#7aa6d8",    # trayectorias secundarias
    "nul_h": "#d9541e",    # nulclina de la primera variable
    "nul_p": "#2a9d4b",    # nulclina de la segunda variable
    "eq": "black",         # equilibrios
    "campo": "0.70",       # campo de direcciones
    "dato": "#1f4e9c",
    "modelo": "#d9541e",
    "gris": "0.5",
}

CICLO = ["#1f4e9c", "#d9541e", "#2a9d4b", "#8e44ad", "#c9a227", "#16a2b8"]

ANCHO = 6.0   # pulgadas
ALTO = 4.0

CARPETA_FIGURAS = "figuras"


def activar(tamano=(ANCHO, ALTO), fuente=11):
    """Fija los parámetros de matplotlib para todas las figuras del notebook."""
    mpl.rcParams.update({
        "figure.figsize": tamano,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": fuente,
        "axes.labelsize": fuente,
        "axes.titlesize": fuente,
        "legend.fontsize": fuente - 1,
        "xtick.labelsize": fuente - 1,
        "ytick.labelsize": fuente - 1,
        "axes.prop_cycle": mpl.cycler(color=CICLO),
        "lines.linewidth": 1.6,
        "axes.grid": False,
        "legend.frameon": False,
        "mathtext.fontset": "cm",
    })


def _carpeta_figuras():
    """``figuras/`` del repositorio si el notebook corre dentro de él; si no, ``./figuras``."""
    aqui = os.getcwd()
    for _ in range(4):
        cand = os.path.join(aqui, CARPETA_FIGURAS)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(aqui, "imc")):
            return cand
        aqui = os.path.dirname(aqui)
    return CARPETA_FIGURAS


def guardar(fig, nombre, carpeta=None, formatos=("png",)):
    """Guarda ``fig`` como ``carpeta/nombre.<fmt>``; crea la carpeta si no existe.
    Por defecto la carpeta es ``figuras/`` en la raíz del repositorio."""
    carpeta = carpeta or _carpeta_figuras()
    os.makedirs(carpeta, exist_ok=True)
    rutas = []
    for fmt in formatos:
        ruta = os.path.join(carpeta, f"{nombre}.{fmt}")
        fig.savefig(ruta)
        rutas.append(ruta)
    return rutas


def parametros(ax, texto, loc="upper right", fontsize=9):
    """Escribe los valores de los parámetros en una esquina del gráfico.
    ``fontsize`` conviene subirlo (10--11) en figuras que el texto incluye chicas."""
    xy = {"upper right": (0.98, 0.98), "upper left": (0.02, 0.98),
          "lower right": (0.98, 0.02), "lower left": (0.02, 0.02)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, texto, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"), zorder=10)
