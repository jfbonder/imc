"""Figura casos-internado-1978.png: la curva de casos del problema conductor de la Parte I.
(Se reutiliza en notebooks/lab-SIR.ipynb.)"""
import numpy as np, matplotlib.pyplot as plt
from imc import estilo, datos
from imc.estilo import COLORES

estilo.activar(fuente=14)
ruta = datos.obtener("gripe_internado_1978.csv")
d = np.genfromtxt(ruta, delimiter=",", names=True, dtype=None, encoding="utf-8")
t, y = d["dia"], d["en_cama"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(t, y, color=COLORES["dato"], alpha=0.35, width=0.8)
ax.plot(t, y, "o-", color=COLORES["dato"], ms=6, lw=2, label="alumnos en cama")
ax.set_xlabel("día (1 = 22 de enero de 1978)"); ax.set_ylabel("alumnos en cama")
ax.set_xticks(range(1, 15)); ax.set_ylim(0, 320)
ax.legend(loc="upper right", fontsize=12)
estilo.parametros(ax, r"$N=763$ alumnos", loc="upper left", fontsize=12)
print(estilo.guardar(fig, "casos-internado-1978"))
