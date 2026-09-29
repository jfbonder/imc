"""Retratos de fase, rectas de fase, nulclinas y campos de direcciones.

Las funciones reciben el campo como ``F(t, X, *args)`` con ``X = (x1, x2)``,
que es la convención de ``scipy.integrate.solve_ivp``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .estilo import COLORES


# ---------------------------------------------------------------------------
# Dimensión uno: recta de fase
# ---------------------------------------------------------------------------

def recta_de_fase(f, xmin, xmax, ax=None, n=400, equilibrios=None, nombre="p",
                  mostrar_f=True):
    """Dibuja la recta de fase de ``x' = f(x)`` en ``[xmin, xmax]``.

    Si ``mostrar_f`` es True, se dibuja además el gráfico de ``f`` arriba de la
    recta. Los equilibrios se calculan buscando cambios de signo de ``f`` (o se
    pasan explícitamente en ``equilibrios``). Devuelve los equilibrios hallados.
    """
    xs = np.linspace(xmin, xmax, n)
    fs = f(xs)
    if equilibrios is None:
        equilibrios = []
        for a, b in zip(xs[:-1], xs[1:]):
            if f(a) == 0:
                equilibrios.append(a)
            elif f(a) * f(b) < 0:
                equilibrios.append(brentq(f, a, b))
    if ax is None:
        fig, ax = plt.subplots()
    if mostrar_f:
        ax.plot(xs, fs, color=COLORES["traj"])
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel(rf"$f({nombre})$")
    else:
        ax.axhline(0, color="black", lw=1)
        ax.set_yticks([])
    # flechas sobre la recta
    puntos = [xmin] + list(equilibrios) + [xmax]
    for a, b in zip(puntos[:-1], puntos[1:]):
        m = 0.5 * (a + b)
        s = np.sign(f(m))
        if s != 0:
            ax.annotate("", xy=(m + 0.12 * s * (b - a), 0), xytext=(m - 0.12 * s * (b - a), 0),
                        arrowprops=dict(arrowstyle="-|>", color=COLORES["traj"], lw=2, mutation_scale=18))
    for e in equilibrios:
        # estable si f cambia de + a -
        eps = 1e-6 * (xmax - xmin)
        estable = f(e - eps) > 0 > f(e + eps)
        ax.plot(e, 0, "o", ms=9, color="black" if estable else "white",
                mec="black", mew=1.5, zorder=5)
    ax.set_xlabel(rf"${nombre}$")
    ax.set_xlim(xmin, xmax)
    return equilibrios


# ---------------------------------------------------------------------------
# Dimensión dos
# ---------------------------------------------------------------------------

def campo(F, xlim, ylim, ax=None, n=20, args=(), normalizar=True, color=None):
    """Campo de direcciones de ``X' = F(t, X)`` en el rectángulo ``xlim x ylim``."""
    if ax is None:
        fig, ax = plt.subplots()
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    U, V = F(0.0, np.array([X, Y]), *args)
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float)
    if normalizar:
        M = np.hypot(U, V)
        M[M == 0] = 1.0
        U, V = U / M, V / M
    ax.quiver(X, Y, U, V, color=color or COLORES["campo"], angles="xy",
              pivot="mid", scale=1.3 * n, width=0.003, headwidth=4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    return ax


def trayectoria(F, X0, T, ax=None, args=(), color=None, flecha=True, lw=None,
                atras=False, **kw):
    """Integra ``X' = F(t, X)`` desde ``X0`` durante ``T`` y dibuja la curva.

    Con ``atras=True`` integra también hacia el pasado (útil para sillas).
    Devuelve la solución de ``solve_ivp``.
    """
    if ax is None:
        fig, ax = plt.subplots()
    color = color or COLORES["traj"]
    sol = solve_ivp(F, (0, T), X0, args=args, rtol=1e-8, atol=1e-10,
                    dense_output=True, max_step=T / 400)
    ax.plot(sol.y[0], sol.y[1], color=color, lw=lw, **kw)
    if flecha and sol.y.shape[1] > 10:
        k = sol.y.shape[1] // 3
        ax.annotate("", xy=sol.y[:, k + 1], xytext=sol.y[:, k],
                    arrowprops=dict(arrowstyle="-|>", color=color, mutation_scale=14))
    if atras:
        solb = solve_ivp(F, (0, -T), X0, args=args, rtol=1e-8, atol=1e-10,
                         max_step=T / 400)
        ax.plot(solb.y[0], solb.y[1], color=color, lw=lw, **kw)
    return sol


def retrato(F, xlim, ylim, inicios, T, ax=None, args=(), n_campo=20,
            equilibrios=(), nulclinas=None, xlabel="$x$", ylabel="$y$",
            color=None, flechas=True, **kw):
    """Retrato de fase completo: campo, trayectorias desde ``inicios``, equilibrios
    y, opcionalmente, nulclinas.

    ``nulclinas``: lista de pares ``(g, color)`` donde ``g(X, Y)`` es una función
    cuyo nivel cero se dibuja con ``contour`` (por ejemplo la primera y la segunda
    componente de ``F``).
    ``equilibrios``: lista de ``(x, y, tipo)`` con ``tipo`` en {"estable",
    "inestable", "silla", "centro"}; se dibujan lleno (estable), vacío
    (inestable/silla) o con un punto (centro).
    """
    if ax is None:
        fig, ax = plt.subplots()
    campo(F, xlim, ylim, ax=ax, n=n_campo, args=args)
    if nulclinas:
        x = np.linspace(*xlim, 300); y = np.linspace(*ylim, 300)
        X, Y = np.meshgrid(x, y)
        for g, c in nulclinas:
            ax.contour(X, Y, g(X, Y), levels=[0], colors=[c], linewidths=1.4,
                       linestyles="--")
    for X0 in inicios:
        trayectoria(F, X0, T, ax=ax, args=args, color=color, flecha=flechas, **kw)
    marcar_equilibrios(ax, equilibrios)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return ax


def marcar_equilibrios(ax, equilibrios):
    for eq in equilibrios:
        x, y = eq[0], eq[1]
        tipo = eq[2] if len(eq) > 2 else "estable"
        if tipo == "estable":
            ax.plot(x, y, "o", ms=8, color="black", zorder=6)
        elif tipo == "centro":
            ax.plot(x, y, "o", ms=8, color="white", mec="black", mew=1.5, zorder=6)
            ax.plot(x, y, ".", ms=4, color="black", zorder=7)
        else:
            ax.plot(x, y, "o", ms=8, color="white", mec="black", mew=1.5, zorder=6)


def clasificar(J):
    """Clasifica el equilibrio de un sistema lineal plano por traza y determinante.

    Devuelve una cadena: 'silla', 'nodo estable', 'nodo inestable', 'foco estable',
    'foco inestable', 'centro' o 'degenerado'.
    """
    J = np.asarray(J, dtype=float)
    tr, det = np.trace(J), np.linalg.det(J)
    if det < 0:
        return "silla"
    if det == 0:
        return "degenerado"
    if tr == 0:
        return "centro"
    disc = tr ** 2 - 4 * det
    tipo = "nodo" if disc >= 0 else "foco"
    return f"{tipo} {'estable' if tr < 0 else 'inestable'}"


def jacobiano(F, X, args=(), h=1e-6):
    """Matriz diferencial numérica de ``F(0, X)`` en ``X`` (diferencias centradas)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (np.asarray(F(0.0, X + e, *args)) - np.asarray(F(0.0, X - e, *args))) / (2 * h)
    return J
