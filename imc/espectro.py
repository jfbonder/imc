"""Análisis espectral con las convenciones de las notas (Parte II).

* ``dft(x, dt)`` devuelve las frecuencias físicas ``xi_k = k/T`` y los coeficientes
  ``hat f[k]`` de la definición de la transformada discreta (Sección 14.2), sin normalizar.
* ``amplitudes(x, dt)`` devuelve, para señales reales, la amplitud ``A_k`` de cada
  componente ``A_k cos(2 pi xi_k t + phi_k)`` (es decir ``2|hat f[k]|/N`` para
  ``0<k<N/2``), que es lo que se compara con los coeficientes de Fourier.
* ``periodograma(x, dt, ventana=None)`` devuelve el espectro de potencia unilateral,
  normalizado para que su suma sea la energía ``sum(x**2)``.

Series de Fourier (Capítulos 11 y 12): ``coeficientes(x, L, N)`` y
``coeficientes_funcion(f, L, N)`` calculan los coeficientes ``c_k`` de la
definición de la Sección 11.4 aproximando la integral por la suma de Riemann
``<f, e_k>_n`` sobre una grilla uniforme del período; ``coeficientes_reales``
da los ``a_k, b_k`` de la serie en senos y cosenos y ``suma_parcial`` evalúa
``S_N[f](t)``.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Series de Fourier
# ----------------------------------------------------------------------------

def coeficientes(x, L=1.0, N=None):
    """Coeficientes de Fourier ``c_k``, ``|k| <= N``, de una señal ``L``-periódica
    muestreada uniformemente en un período: ``x[j] = f(j L / n)``, ``j = 0..n-1``.

    Aproxima ``c_k = (1/L) int_0^L f(t) e^{-i w_k t} dt`` por la suma de Riemann
    ``(1/n) sum_j x[j] e^{-2 pi i k j / n}`` (el producto interno ``<f, e_k>_n`` de
    las notas), que es exacta para polinomios trigonométricos de grado ``< n/2``.
    Por defecto ``N = n // 2 - 1``.  Devuelve ``(k, c)`` con ``k = -N..N``.
    """
    x = np.asarray(x)
    n = len(x)
    if N is None:
        N = n // 2 - 1
    if 2 * N + 1 > n:
        raise ValueError(f"hacen falta al menos 2N+1 = {2 * N + 1} muestras (hay {n})")
    X = np.fft.fft(x) / n           # X[k] = (1/n) sum_j x[j] e^{-2 pi i k j/n}
    k = np.arange(-N, N + 1)
    return k, X[k % n]


def coeficientes_funcion(f, L=1.0, N=50, n=2 ** 16):
    """``coeficientes`` para una función ``f`` dada como callable, muestreada en
    ``n`` puntos ``t_j = j L / n`` del período ``[0, L)``.  Devuelve ``(k, c)``."""
    t = np.arange(n) * L / n
    return coeficientes(f(t), L, N)


def coeficientes_reales(x, L=1.0, N=None):
    """Coeficientes ``a_k`` (``k = 0..N``) y ``b_k`` (``b_0 = 0``) de la serie de
    Fourier real ``a_0/2 + sum_k a_k cos(w_k t) + b_k sin(w_k t)`` de una señal
    real muestreada como en ``coeficientes``.  Usa ``a_k = 2 Re c_k``,
    ``b_k = -2 Im c_k``.  Devuelve ``(k, a, b)``."""
    k, c = coeficientes(x, L, N)
    c = c[k >= 0]
    k = k[k >= 0]
    return k, 2 * c.real, -2 * c.imag


def suma_parcial(k, c, L, t):
    """Suma parcial ``S_N[f](t) = sum_k c_k e^{i 2 pi k t / L}`` en los puntos ``t``
    (``k`` y ``c`` como los devuelve ``coeficientes``).  Devuelve la parte real."""
    t = np.asarray(t, dtype=float)
    S = np.zeros(t.shape, dtype=complex)
    for kk, ck in zip(k, c):
        S += ck * np.exp(2j * np.pi * kk * t / L)
    return S.real


def dft(x, dt=1.0):
    """DFT de ``x`` muestreada con paso ``dt``. Devuelve ``(xi, X)`` con las
    frecuencias en ciclos por unidad de tiempo (sólo la mitad no negativa)."""
    x = np.asarray(x)
    N = len(x)
    X = np.fft.rfft(x)
    xi = np.fft.rfftfreq(N, d=dt)
    return xi, X


def amplitudes(x, dt=1.0):
    """Amplitud y fase de cada componente armónica de una señal real.

    Devuelve ``(xi, A, phi)`` tales que
    ``x(t) ~ A[0] + sum_k A[k] cos(2 pi xi[k] t + phi[k])``.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    xi, X = dft(x, dt)
    A = 2 * np.abs(X) / N
    A[0] = np.abs(X[0]) / N
    if N % 2 == 0:
        A[-1] = np.abs(X[-1]) / N
    phi = np.angle(X)
    return xi, A, phi


def periodograma(x, dt=1.0, ventana=None, quitar_media=True):
    """Espectro de potencia unilateral (frecuencias ``0 <= xi_k <= 1/(2 dt)``).

    Convención: ``P[k] = 2 |hat x[k]|^2 / N`` para ``0 < k < N/2`` (cada frecuencia
    positiva junta su potencia con la de ``-k``) y ``P[k] = |hat x[k]|^2 / N`` en
    ``k = 0`` y, si ``N`` es par, en la de Nyquist ``k = N/2``. Así, por la identidad
    de Plancherel discreta, ``sum(P) == sum(x**2)``: la suma es la energía de la
    señal (con la media restada si ``quitar_media``). Un coseno de amplitud ``A``
    con frecuencia en la grilla da un pico ``P[k] = A**2 N / 2``, y un ruido blanco
    de varianza ``sigma**2`` un piso de altura media ``2 sigma**2``.

    ``ventana``: None, "hann", "hamming", "blackman" o un vector de longitud ``len(x)``.
    Con ventana se divide además por ``mean(w**2)``, de modo que la suma conserva
    (en promedio) la energía de la señal sin ventana.
    """
    x = np.asarray(x, dtype=float)
    if quitar_media:
        x = x - x.mean()
    N = len(x)
    if ventana is None:
        w = np.ones(N)
    elif isinstance(ventana, str):
        w = {"hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}[ventana](N)
    else:
        w = np.asarray(ventana)
    xw = x * w
    xi, X = dft(xw, dt)
    P = np.abs(X) ** 2 / (N * np.mean(w ** 2))
    P[1:(N + 1) // 2] *= 2          # 0 < k < N/2: se suman las frecuencias k y -k (no DC ni Nyquist)
    return xi, P


def welch(x, dt=1.0, tramo=None, ventana="hann", solapamiento=0.5):
    """Periodograma promediado (método de Welch) sobre tramos de longitud ``tramo``."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    tramo = tramo or N // 8
    paso = int(tramo * (1 - solapamiento))
    Ps = []
    for i in range(0, N - tramo + 1, paso):
        xi, P = periodograma(x[i:i + tramo], dt, ventana=ventana)
        Ps.append(P)
    return xi, np.mean(Ps, axis=0)


def filtrar(x, dt, mascara):
    """Filtra ``x`` multiplicando su espectro por ``mascara(xi)`` (0/1 o real) y
    antitransformando. Devuelve la señal filtrada (real)."""
    x = np.asarray(x, dtype=float)
    xi, X = dft(x, dt)
    return np.fft.irfft(X * mascara(xi), n=len(x))


def espectrograma(x, dt, tramo, solapamiento=0.5, ventana="hann"):
    """Espectrograma: matriz ``S[k, j]`` = periodograma del tramo ``j``.

    Devuelve ``(t_centros, xi, S)``.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    paso = int(tramo * (1 - solapamiento))
    ts, Ss = [], []
    for i in range(0, N - tramo + 1, paso):
        xi, P = periodograma(x[i:i + tramo], dt, ventana=ventana)
        Ss.append(P); ts.append((i + tramo / 2) * dt)
    return np.array(ts), xi, np.array(Ss).T
