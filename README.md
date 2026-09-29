# Introducción al Modelado Continuo

Notas de clase, guías y notebooks de la materia *Introducción al Modelado Continuo* (Licenciatura en Ciencia de Datos, FCEN–UBA). Julián Fernández Bonder, Departamento de Matemática e Instituto de Cálculo, FCEN–UBA / CONICET.

## Contenido del repositorio

| Carpeta | Qué hay |
| --- | --- |
| `notas/` | `Notas-IMC.tex` (fuente LaTeX de las notas), `biblio.bib`, `CAMBIOS.md` (registro de la revisión) |
| `figuras/` | Figuras de las notas. Las genera cada notebook con `estilo.guardar(fig, "nombre")` |
| `imc/` | Módulo Python común a los notebooks (estilo, retratos de fase, espectros, esquemas numéricos, datos) |
| `notebooks/` | Notebooks del texto (uno por capítulo) y de laboratorio (uno por problema conductor) |
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
| `06-series` | 11–12. Series de Fourier y convergencia | (pendiente) |
| `07-transformada-DFT` | 13–14. Transformada de Fourier, DFT, FFT | (pendiente) |
| `08-aplicaciones` | 15. Filtrado, ventanas, compresión | (pendiente) |
| `09-calor` | 18. Ecuación de difusión | (pendiente) |
| `10-laplace` | 19. Laplace/Poisson | (pendiente) |
| `11-ondas` | 20. Ecuación de ondas | (pendiente) |

### De laboratorio (uno por problema conductor)

| Notebook | Problema | Colab |
| --- | --- | --- |
| `lab-EDO-numerico` | Euler, Runge–Kutta, orden de convergencia | (pendiente) |
| `lab-SIR` | Epidemia: simulación, SEIR, ajuste a datos | (pendiente) |
| `lab-vanderPol` | Circuito: forma de onda, período, espectro, problema inverso | (pendiente) |
| `lab-senal-temperatura` | Serie horaria de temperatura: espectro, filtrado, compresión | (pendiente) |
| `lab-EDP-numerico` | Diferencias finitas, esquema explícito e implícito, $r\le 1/2$ | (pendiente) |
| `lab-suelo` | Temperatura del suelo: ajuste de la difusividad | (pendiente) |
| `lab-cuerda` | Cuerda vibrante: espectro real, inarmonicidad, leapfrog | (pendiente) |
| `lab-laplace-datos` | Cinco puntos vs. Monte Carlo, inpainting | (pendiente) |

## Convenciones

* Los nombres de las variables en las figuras son los del texto ($h,p$; $s,i$; $x,y$), y cada figura muestra los parámetros con que fue hecha.
* Los notebooks son cortos (un capítulo o un laboratorio cada uno). Lo repetido vive en `imc/`.
* Los laboratorios terminan con celdas de interpretación escrita; ésa es la parte que se evalúa.

## Licencia

Texto y figuras: CC BY-NC-SA 4.0. Código: MIT. Los datos tienen la licencia que indica `datos/README.md`.
