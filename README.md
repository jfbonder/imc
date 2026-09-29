# Introducción al Modelado Continuo

Sitio de la materia: https://jfbonder.github.io/imc/

Notas de clase, guías y notebooks de la materia *Introducción al Modelado Continuo* (Licenciatura en Ciencia de Datos, FCEN–UBA). Julián Fernández Bonder, Departamento de Matemática e Instituto de Cálculo, FCEN–UBA / CONICET.

## Contenido del repositorio

| Carpeta | Qué hay |
| --- | --- |
| `notas/` | `Notas-IMC.tex` (fuente LaTeX de las notas), `Notas-IMC.pdf` (última compilación), `biblio.bib`, `CAMBIOS.md` (registro de la revisión), `FIGURAS-parte-I.md`, `FIGURAS-parte-II.md`, `FIGURAS-parte-III.md` (figuras generadas por los notebooks y dónde van) |
| `figuras/` | Figuras de las notas. Las genera cada notebook con `estilo.guardar(fig, "nombre")` |
| `imc/` | Módulo Python común a los notebooks (estilo, retratos de fase, espectros, esquemas numéricos, datos) |
| `notebooks/` | Notebooks del texto (uno por capítulo) y de laboratorio (uno por problema conductor, con su guía del docente en `notebooks/docente/`) |
| `datos/` | Datos usados en los laboratorios, con fuente y licencia en `datos/README.md` |

Para compilar las notas: `cd notas && pdflatex Notas-IMC && bibtex Notas-IMC && pdflatex Notas-IMC && pdflatex Notas-IMC` (las figuras se buscan en `../figuras`, ver el `\graphicspath` del preámbulo).

## Notebooks

Cada notebook se abre directamente en Google Colab con el botón; la primera celda instala el módulo `imc` desde este repositorio y descarga los datos que necesite. Para trabajar en una copia local, `pip install -e .` en la raíz del repositorio y abrir los notebooks con Jupyter.

### Del texto (acompañan a los capítulos)

| Notebook | Capítulo | Colab |
| --- | --- | --- |
| `01-poblaciones` | 2. Modelos poblacionales | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/01-poblaciones.ipynb) |
| `02-mecanicos` | 3. Sistemas mecánicos y eléctricos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/02-mecanicos.ipynb) |
| `03-lineales-HG` | 4–5. Flujo, sistemas lineales, Hartman–Grobman | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/03-lineales-HG.ipynb) |
| `04-lyapunov-global` | 6–7. Lyapunov, Poincaré–Bendixson, competencia, Lorenz | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/04-lyapunov-global.ipynb) |
| `05-bifurcaciones` | 8. Bifurcaciones | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/05-bifurcaciones.ipynb) |
| `06-series` | 11–12. Series de Fourier y convergencia | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/06-series.ipynb) |
| `07-transformada-DFT` | 13–14. Transformada de Fourier, DFT, FFT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/07-transformada-DFT.ipynb) |
| `08-aplicaciones` | 15. Filtrado, ventanas, compresión | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/08-aplicaciones.ipynb) |
| `09-calor` | 18. Ecuación de difusión | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/09-calor.ipynb) |
| `10-laplace` | 19. Laplace/Poisson | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/10-laplace.ipynb) |
| `11-ondas` | 20. Ecuación de ondas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/11-ondas.ipynb) |

### De laboratorio (uno por problema conductor)

Cada laboratorio viene en dos versiones generadas por el mismo script (`tools/make_lab_*.py`, con `tools/labkit.py`): la de estudiantes, con la explicación de los métodos, las consignas, esqueletos con `# TODO` y celdas de verificación, y la guía del docente en `notebooks/docente/`, con las soluciones completas ejecutadas, respuestas modelo de la interpretación y notas para la clase.

| Notebook | Problema | Estudiantes | Guía del docente |
| --- | --- | --- | --- |
| `lab-EDO-numerico` | Euler, Runge–Kutta, orden de convergencia | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-EDO-numerico.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-EDO-numerico.ipynb) |
| `lab-SIR` | Epidemia: simulación, SEIR, ajuste a datos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-SIR.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-SIR.ipynb) |
| `lab-vanderPol` | Circuito: forma de onda, período, espectro, problema inverso | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-vanderPol.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-vanderPol.ipynb) |
| `lab-senal-temperatura` | Serie horaria de temperatura: espectro, filtrado, compresión | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-senal-temperatura.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-senal-temperatura.ipynb) |
| `lab-EDP-numerico` | Diferencias finitas, esquema explícito e implícito, $r\le 1/2$ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-EDP-numerico.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-EDP-numerico.ipynb) |
| `lab-suelo` | Temperatura del suelo: ajuste de la difusividad | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-suelo.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-suelo.ipynb) |
| `lab-cuerda` | Cuerda vibrante: espectro real, inarmonicidad, leapfrog | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-cuerda.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-cuerda.ipynb) |
| `lab-laplace-datos` | Cinco puntos vs. Monte Carlo, inpainting | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/lab-laplace-datos.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/imc/blob/main/notebooks/docente/lab-laplace-datos.ipynb) |

## Convenciones

* Los nombres de las variables en las figuras son los del texto ($h,p$; $s,i$; $x,y$), y cada figura muestra los parámetros con que fue hecha.
* Los notebooks son cortos (un capítulo o un laboratorio cada uno). Lo repetido vive en `imc/`.
* Los laboratorios terminan con celdas de interpretación escrita; ésa es la parte que se evalúa.

## Licencia

Texto y figuras: [CC BY-NC-SA 4.0](LICENSE-texto). Código (`imc/`, `tools/`, `notebooks/`): [MIT](LICENSE). Los datos tienen la licencia que indica `datos/README.md`.

Algunas partes de las notas siguen presentaciones clásicas: la organización de la Parte I debe mucho a S. Lynch, *Dynamical Systems with Applications using Python*, y un ejercicio de la Parte II está adaptado de D. J. Griffiths, *Introduction to Electrodynamics*.
