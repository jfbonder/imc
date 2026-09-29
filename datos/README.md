# Datos

Cada archivo de esta carpeta debe figurar acá con su fuente y su licencia.

| Archivo | Contenido | Fuente | Licencia |
| --- | --- | --- | --- |
| `temperatura_horaria.csv` | temperatura del aire a 2 m cada hora, del 1/1/2021 al 31/12/2023 (26 280 filas, sin huecos); columnas `fecha_hora` (ISO, hora local, sin cambio de horario) y `temp` (°C, un decimal). Son datos de **reanálisis** (un modelo que asimila observaciones), no la medición directa de la estación | Open-Meteo Historical Weather API (reanálisis ERA5/ERA5-Land, Copernicus), punto -34.56, -58.42 (Aeroparque), hora local | CC BY 4.0 (Open-Meteo) / Copernicus |
| `suelo_horaria.csv` | año 2023 completo cada hora (8 760 filas, sin huecos): temperatura del aire (`t_aire`) y del suelo en las capas 0–7 cm, 7–28 cm, 28–100 cm y 100–255 cm (`t_suelo_0_7`, `t_suelo_7_28`, `t_suelo_28_100`, `t_suelo_100_255`, °C). Son datos de **reanálisis** (modelo que asimila observaciones), no mediciones directas de sensores enterrados | Open-Meteo Historical Weather API (reanálisis ERA5/ERA5-Land, Copernicus), punto -34.56, -58.42 (Aeroparque), hora local | CC BY 4.0 (Open-Meteo) / Copernicus |
| `gripe_internado_1978.csv` | alumnos en cama por día durante la epidemia de gripe en un internado inglés (763 alumnos, 22/1 al 4/2 de 1978); columnas `dia`, `fecha`, `en_cama` | Anónimo, "Influenza in a boarding school", *British Medical Journal*, 4 de marzo de 1978, p. 587; tabla reproducida en J. D. Murray, *Mathematical Biology I*, §10.2 | datos de dominio público (14 números); la cita es la fuente |
| `astronauta.png` | fotografía de la astronauta Eileen Collins (NASA), 512 × 512, en escala de grises (8 bits); es `skimage.data.astronaut()` convertida con `skimage.color.rgb2gray` y guardada por `notebooks/08-aplicaciones.ipynb`. Se usa en el Capítulo 15 (compresión de imágenes) | NASA, distribuida con scikit-image (`skimage.data.astronaut`) | dominio público (obra del gobierno de EE. UU.) |
| (pendiente) `cuerda_guitarra.wav` | grabación de una cuerda pulsada | propia | CC BY 4.0 |
| `vanderpol_registro.csv` | corriente simulada con ruido (se genera con `notebooks/lab-vanderPol.ipynb`) | propia | CC BY 4.0 |
