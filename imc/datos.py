"""Acceso a los archivos de ``datos/``, tanto en una copia local del repositorio
como en Colab (donde se descargan desde GitHub).

    from imc import datos
    ruta = datos.obtener("temperatura_horaria.csv")

``REPO`` se fija al crear el repositorio; ``RAMA`` es la rama publicada.
"""

from __future__ import annotations

import os
import urllib.request

REPO = "USUARIO/imc-notas"   # <- completar con el usuario de GitHub
RAMA = "main"

_RAW = "https://raw.githubusercontent.com/{repo}/{rama}/datos/{archivo}"


def _raiz_local():
    """Devuelve la carpeta ``datos/`` si el notebook corre dentro del repo."""
    aqui = os.getcwd()
    for _ in range(4):
        cand = os.path.join(aqui, "datos")
        if os.path.isdir(cand):
            return cand
        aqui = os.path.dirname(aqui)
    return None


def obtener(archivo, destino="datos"):
    """Devuelve la ruta local de ``archivo``; si no está, lo descarga del repo."""
    raiz = _raiz_local()
    if raiz and os.path.exists(os.path.join(raiz, archivo)):
        return os.path.join(raiz, archivo)
    os.makedirs(destino, exist_ok=True)
    ruta = os.path.join(destino, archivo)
    if not os.path.exists(ruta):
        url = _RAW.format(repo=REPO, rama=RAMA, archivo=archivo)
        print(f"Descargando {url}")
        urllib.request.urlretrieve(url, ruta)
    return ruta
