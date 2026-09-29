"""Esquemas numéricos elementales usados en el laboratorio.

Se escriben de la manera más transparente posible (sin optimizar) para que el
código pueda leerse junto con las guías. Para uso serio, ``scipy.integrate``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# EDOs: métodos de un paso
# ---------------------------------------------------------------------------

def euler(f, t0, y0, h, n, args=()):
    """Método de Euler para ``y' = f(t, y)``. Devuelve ``(t, y)`` con ``y`` de
    forma ``(n+1, dim)``."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for k in range(n):
        y[k + 1] = y[k] + h * np.asarray(f(t[k], y[k], *args))
    return t, y


def rk2(f, t0, y0, h, n, args=()):
    """Euler modificado (Runge--Kutta de orden 2, punto medio)."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for k in range(n):
        k1 = np.asarray(f(t[k], y[k], *args))
        k2 = np.asarray(f(t[k] + h / 2, y[k] + h / 2 * k1, *args))
        y[k + 1] = y[k] + h * k2
    return t, y


def rk4(f, t0, y0, h, n, args=()):
    """Runge--Kutta clásico de orden 4."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for k in range(n):
        k1 = np.asarray(f(t[k], y[k], *args))
        k2 = np.asarray(f(t[k] + h / 2, y[k] + h / 2 * k1, *args))
        k3 = np.asarray(f(t[k] + h / 2, y[k] + h / 2 * k2, *args))
        k4 = np.asarray(f(t[k] + h, y[k] + h * k3, *args))
        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


# ---------------------------------------------------------------------------
# Diferencias finitas
# ---------------------------------------------------------------------------

def matriz_laplaciano_1d(m, h, periodico=False):
    """Matriz (m x m) de la segunda derivada centrada con paso ``h``.

    Sin ``periodico`` corresponde a condiciones de Dirichlet homogéneas en los
    extremos (los nodos del borde no se incluyen)."""
    A = (np.diag(-2 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / h ** 2
    if periodico:
        A[0, -1] = A[-1, 0] = 1 / h ** 2
    return A


def calor_explicito(u0, D, dx, dt, pasos, borde=(0.0, 0.0)):
    """Esquema explícito ``u_j^{n+1} = u_j^n + r (u_{j+1}^n - 2u_j^n + u_{j-1}^n)``
    para la ecuación del calor con Dirichlet ``borde``. Devuelve ``u`` de forma
    ``(pasos+1, len(u0))``. Estable si ``r = D dt/dx^2 <= 1/2``."""
    r = D * dt / dx ** 2
    u = np.zeros((pasos + 1, len(u0)))
    u[0] = u0
    for n in range(pasos):
        v = u[n]
        w = v.copy()
        w[1:-1] = v[1:-1] + r * (v[2:] - 2 * v[1:-1] + v[:-2])
        w[0], w[-1] = borde
        u[n + 1] = w
    return u


def calor_implicito(u0, D, dx, dt, pasos, borde=(0.0, 0.0)):
    """Esquema implícito (Euler hacia atrás) para la ecuación del calor."""
    r = D * dt / dx ** 2
    m = len(u0) - 2
    A = np.eye(m) - r * dx ** 2 * matriz_laplaciano_1d(m, dx)
    u = np.zeros((pasos + 1, len(u0)))
    u[0] = u0
    for n in range(pasos):
        b = u[n, 1:-1].copy()
        b[0] += r * borde[0]; b[-1] += r * borde[1]
        w = np.empty_like(u0); w[0], w[-1] = borde
        w[1:-1] = np.linalg.solve(A, b)
        u[n + 1] = w
    return u


def ondas_leapfrog(g, h, c, dx, dt, pasos):
    """Esquema centrado (leapfrog) para ``u_tt = c^2 u_xx`` con extremos fijos.

    ``g``: posición inicial en los nodos, ``h``: velocidad inicial.
    Estable si ``r = c dt/dx <= 1`` (condición CFL)."""
    r = c * dt / dx
    u = np.zeros((pasos + 1, len(g)))
    u[0] = g
    # primer paso con Taylor de segundo orden
    u1 = g + dt * h
    u1[1:-1] += 0.5 * r ** 2 * (g[2:] - 2 * g[1:-1] + g[:-2])
    u1[0] = u1[-1] = 0.0
    u[1] = u1
    for n in range(1, pasos):
        w = np.zeros_like(g)
        w[1:-1] = 2 * u[n, 1:-1] - u[n - 1, 1:-1] + r ** 2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
        u[n + 1] = w
    return u
