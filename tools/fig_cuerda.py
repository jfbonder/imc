"""Figura espectro-cuerda.png: espectro de una cuerda de guitarra real (problema conductor de la Parte III).
(Se reutiliza en notebooks/lab-cuerda.ipynb.)"""
import wave
import numpy as np, matplotlib.pyplot as plt
from imc import estilo, datos
from imc.estilo import COLORES

estilo.activar(fuente=16)
w = wave.open(datos.obtener("cuerda_guitarra.wav"))
fs = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
t = np.arange(len(x)) / fs

# espectro de los primeros 2 s (ventana de Hann)
T = 2.0
seg = x[: int(T * fs)]
F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg)
f = np.fft.rfftfreq(len(seg), 1 / fs)

# fundamental: pico más alto entre 60 y 400 Hz, refinado con interpolación parabólica
m = (f > 60) & (f < 400)
k0 = np.argmax(F * m)
a, b, c = np.log(F[k0 - 1 : k0 + 2])
f1 = f[k0] + (f[1] - f[0]) * 0.5 * (a - c) / (a - 2 * b + c)
# picos medidos cerca de cada k f1
picos = []
for k in range(1, 16):
    mk = (f > (k - 0.3) * f1) & (f < (k + 0.3) * f1)
    j = np.argmax(F * mk)
    picos.append((k, f[j], F[j]))
picos = np.array(picos)
print(f"f1 = {f1:.2f} Hz")
for k, fk, A in picos:
    print(f"k={int(k):2d}: medido {fk:8.1f} Hz, ideal {k*f1:8.1f}, desvío {(fk/(k*f1)-1)*100:+.2f} %")

fig, axs = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1, 1.6]})
ax = axs[0]
ax.plot(t, x, lw=0.4, color=COLORES["dato"])
ax.set_xlabel("$t$ (s)"); ax.set_ylabel("señal"); ax.set_xlim(0, t[-1])
ax.set_title("cuerda de guitarra pulsada (grabación)", fontsize=15)
ax = axs[1]
ax.semilogy(f, F, lw=0.8, color=COLORES["dato"], label="espectro (primeros 2 s)")
for k in range(1, 16):
    ax.axvline(k * f1, color=COLORES["modelo"], lw=1, ls="--", alpha=0.8)
ax.plot([], [], color=COLORES["modelo"], ls="--", label=f"$k f_1$, $f_1 = {f1:.1f}$ Hz")
ax.plot(picos[:, 1], picos[:, 2], "o", color=COLORES["modelo"], ms=5, label="picos medidos")
ax.set_xlim(0, 16.5 * f1); ax.set_ylim(F[f < 16.5 * f1].max() * 1e-5, F.max() * 3)
ax.set_xlabel(r"$\xi$ (Hz)"); ax.set_ylabel("amplitud"); ax.legend(loc="upper right", fontsize=13)
fig.tight_layout()
print(estilo.guardar(fig, "espectro-cuerda"))
