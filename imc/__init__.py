"""imc: utilidades para los notebooks de *Introducción al Modelado Continuo*.

Uso típico en un notebook (Colab o local):

    from imc import estilo, fases, espectro
    estilo.activar()

Submódulos
----------
estilo    : estilo común de las figuras (tamaño, colores, fuentes).
fases     : retratos de fase, rectas de fase, nulclinas, campos de direcciones.
espectro  : DFT con la grilla de frecuencias en unidades físicas, periodograma.
numerico  : esquemas numéricos elementales (Euler, RK, diferencias finitas).
datos     : rutas y descarga de los archivos de ``datos/`` (funciona en Colab).
"""

from . import estilo, fases, espectro, numerico, datos  # noqa: F401

__version__ = "0.1.0"
