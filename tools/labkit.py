"""Herramientas para generar los notebooks de laboratorio en dos versiones:

* ``notebooks/lab-XX.ipynb``          — versión para estudiantes: consignas, esqueletos con ``# TODO``
                                        y celdas de verificación.
* ``notebooks/docente/lab-XX.ipynb``  — guía del docente: el mismo notebook con las soluciones
                                        completas en lugar de los esqueletos (se ejecuta y se guarda con salidas).

Uso en un generador ``tools/make_lab_XX.py``::

    from labkit import Lab
    lab = Lab("lab-SIR", "Laboratorio: ajuste del modelo SIR a datos", REPO)
    lab.md("...")                      # markdown común a las dos versiones
    lab.code("...")                    # código común (configuración, carga de datos, gráficos dados)
    lab.tarea(consigna="...",          # markdown de la consigna (común)
              esqueleto="...",         # código con firma y # TODO (solo estudiantes)
              solucion="...",          # código completo (solo docente)
              verificacion="...")      # código de chequeo (común; opcional)
    lab.escribir()                     # escribe las dos versiones

La celda de configuración estándar (instalación de ``imc`` en Colab, imports, ``estilo.activar()``)
se agrega sola con ``lab.configuracion(extra=...)``.
"""

from __future__ import annotations

import os
import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Lab:
    def __init__(self, nombre, titulo, repo="jfbonder/imc"):
        self.nombre = nombre
        self.titulo = titulo
        self.repo = repo
        self.celdas = []   # (tipo, contenido, destino) con destino en {"ambos", "estudiante", "docente"}
        self.n_tareas = 0
        self._encabezado()

    # ---- celdas básicas -------------------------------------------------
    def md(self, texto, destino="ambos"):
        self.celdas.append(("md", texto.strip("\n"), destino))

    def code(self, codigo, destino="ambos"):
        self.celdas.append(("code", codigo.strip("\n"), destino))

    def tarea(self, consigna, esqueleto, solucion, verificacion=None, titulo=None):
        self.n_tareas += 1
        enc = f"### Tarea {self.n_tareas}" + (f": {titulo}" if titulo else "")
        self.md(enc + "\n\n" + consigna.strip("\n"))
        self.code(esqueleto, destino="estudiante")
        self.code(solucion, destino="docente")
        if verificacion:
            self.code(verificacion)

    # ---- bloques estándar ----------------------------------------------
    def _encabezado(self):
        nb_est = f"notebooks/{self.nombre}.ipynb"
        nb_doc = f"notebooks/docente/{self.nombre}.ipynb"
        badge = "[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{repo}/blob/main/{nb})"
        self.md(f"# {self.titulo}\n\n" + badge.format(repo=self.repo, nb=nb_est), destino="estudiante")
        self.md(f"# {self.titulo} — guía del docente\n\n" + badge.format(repo=self.repo, nb=nb_doc)
                + "\n\n> Versión con las soluciones completas. La versión para estudiantes, con esqueletos en lugar de las soluciones, es `" + nb_est + "`.",
                destino="docente")

    def configuracion(self, extra=""):
        self.code(f'''# Configuración (funciona en Colab y en una copia local del repositorio)
try:
    import imc
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/{self.repo}.git"], check=True)
    import imc

import numpy as np
import matplotlib.pyplot as plt
from imc import estilo, datos
from imc.estilo import COLORES
{extra.strip()}
estilo.activar()''')

    def interpretacion(self, preguntas):
        """Sección final de interpretación escrita (lista de preguntas)."""
        texto = ("## Interpretación\n\nEsta es la parte que se evalúa. Respondé en las celdas de texto que siguen, "
                 "con oraciones completas y citando los números o gráficos que obtuviste.\n")
        self.md(texto)
        for i, p in enumerate(preguntas, 1):
            self.md(f"**{i}.** {p}")
            self.md("*(escribí tu respuesta acá)*", destino="estudiante")
        # el docente ve las preguntas; las respuestas modelo se agregan con lab.md(..., destino="docente")

    # ---- escritura ------------------------------------------------------
    def _armar(self, version):
        nb = nbf.v4.new_notebook()
        cells = []
        for tipo, contenido, destino in self.celdas:
            if destino != "ambos" and destino != version:
                continue
            cells.append(nbf.v4.new_markdown_cell(contenido) if tipo == "md" else nbf.v4.new_code_cell(contenido))
        nb["cells"] = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        return nb

    def escribir(self):
        os.makedirs(os.path.join(RAIZ, "notebooks", "docente"), exist_ok=True)
        rutas = {}
        for version, ruta in [("estudiante", f"notebooks/{self.nombre}.ipynb"),
                              ("docente", f"notebooks/docente/{self.nombre}.ipynb")]:
            ruta_abs = os.path.join(RAIZ, ruta)
            nbf.write(self._armar(version), ruta_abs)
            rutas[version] = ruta_abs
            print("escrito", ruta)
        return rutas
