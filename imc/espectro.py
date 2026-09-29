"""Análisis espectral con las convenciones de las notas (Parte II).

* ``dft(x, dt)`` devuelve las frecuencias físicas ``xi_k = k/T`` y los coeficientes
  ``hat f[k]`` de la definición (14.2.1), sin normalizar.
* ``amplitudes(x, dt)`` devuelve, para señales reales, la amplitud ``A_k`` de cada
  componente ``A_k cos(2 pi xi_k t + phi_k)`` (es decir ``2|hat f[k]|/N`` para
  ``0<k<N/2``), que es lo que se compara con los coeficientes de Fourier.
* ``periodograma(x, dt, ventana=None)`` devuelve el espectro de potencia.
"""

from __future__ import annotations

import numpy as np


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
    """Espectro de potencia ``|hat f[k]|^2 / N`` (normalizado para que su suma sea
    la energía de la señal, por la identidad de Plancherel discreta).

    ``ventana``: None, "hann" o un vector de longitud ``len(x)``.
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
