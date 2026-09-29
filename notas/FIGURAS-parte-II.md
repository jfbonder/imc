# Figuras de la Parte II generadas por los notebooks (2026-09-29)

Las figuras de los capítulos 11–15 salen de `notebooks/06-series`, `07-transformada-DFT` y `08-aplicaciones`. **Todo lo de este archivo ya está aplicado en `Notas-IMC.tex`**; queda como registro y para saber de qué celda sale cada figura.

## Datos

- `datos/temperatura_horaria.csv`: temperatura del aire horaria, 2021–2023, punto de Aeroparque (Buenos Aires); reanálisis ERA5 vía Open-Meteo (CC BY 4.0). Es el problema conductor de la parte.
- `datos/astronauta.png`: imagen de dominio público (Eileen Collins, NASA, `skimage.data.astronaut`) que reemplaza la foto de perros sin fuente.

## Figuras regeneradas (mismo nombre)

| Figura | Notebook | Qué cambió |
| --- | --- | --- |
| `ondas-fundamentales` | 06 | tres paneles (A, f, φ) con leyendas; fuente 18 |
| `sombrero`, `sombrero-aproximacion` | 06 | mismos N; ejes con nombre |
| `nucleo-dirichlet` | 06 | **ahora con el factor 1/L** ($D_N(0)=(2N+1)/L$); caption del tex corregido |
| `signo`, `signo-aproximacion`, `signo-aproximacion2`, `signo-detalle` | 06 | mismos N; en el detalle, recta $1+2\cdot0{,}0895$ (sobreimpulso de Gibbs medido: 8,95 %) |
| `cuadratica`, `cuadratica-aproximaciones` | 06 | ídem |
| `diente`, `diente-gibbs`, `diente-detalle` | 06 | ídem; referencia $f(t)+2\cdot0{,}0895$ |
| `transformadas` (nueva; reemplaza `transformada1/2/3`, borradas) | 07 | panel 3×2, convención $e^{-2\pi i\xi x}$ del texto, 0.9\textwidth |
| `aliasing` | 07 | ejemplo del texto ($f_s=16$, $13\to3$) con caja explicativa |
| `señal1`, `espectro-señal`, `espectro-filtrado`, `señal-filtrada` | 08 | leyenda "umbral" (era "Threshold"); en `espectro-filtrado` las etiquetas original/filtrado estaban **invertidas** en la vieja; `default_rng(0)` |
| `imagen-original`, `imagen-3`, `imagen-1`, `imagen-01`, `imagen-001` | 08 | nueva imagen (NASA); reconstrucciones con 3 %, 1 %, 0,1 %, 0,01 % de los coeficientes de mayor módulo |

## Figuras nuevas (todas insertadas en el tex)

| Figura | Notebook | Dónde | Ancho |
| --- | --- | --- | --- |
| `serie-temperatura` | 06 | intro de la Parte II (era `\figpendiente`) | 0.95 |
| `decaimiento-coeficientes` | 06 | §12.1 tras la Prop. de convergencia uniforme | 0.7 |
| `temperatura-serie-fourier` | 06 | "Volvemos" del cap. 11 | 0.9 |
| `grilla-frecuencias` | 07 | §14.7, grilla y Nyquist | 0.7 |
| `temperatura-muestreo` | 07 | "Volvemos" del cap. 14 | 0.9 |
| `fft-mariposa`, `fft-costo` | 07 | §14.8 FFT | 0.85 / 0.6 |
| `temperatura-filtrado`, `temperatura-compresion` | 08 | "Volvemos" del cap. 15 | 0.95 (página aparte) / 0.7 |
| `imagen-espectro` | 08 | §15.2.1 contenido frecuencial | 0.95 |

## Cambios de texto hechos a partir de las cuentas

- Cap. 11, "Volvemos": se agregaron los números de 2023 (media 18,2 °C; $A_1=6{,}9$ °C, máximo el 25 de enero; $A_{365}=2{,}8$ °C, máximo a las 15:45; 70 % de la varianza con tres términos).
- Cap. 14, "Volvemos": la diferencia entre la media de las 15 h y la media horaria es 3,3 °C (el primer armónico diario aporta 2,7; el resto son los armónicos de 12 y 8 h, que también se pliegan sobre $k=0$).
- Cap. 15, "Volvemos": 71 % de la energía en los cinco picos; residuo con desvío 3,4 °C y espectro $\sim\xi^{-1{,}7}$ (no blanco); 462 coeficientes para un error del 5 %.
- Cap. 15, compresión: la afirmación "del orden del $10^{-2}\%$ o menos ... claramente reconocible" **no se cumple** con el criterio del texto (27 coeficientes en 512×512 no reconstruyen nada); se reescribió: reconocible con ~1 %, distorsiones por debajo del 0,1 %, y la observación de que la energía (82 % en el 0,01 %) no mide la calidad visual. Captions con errores relativos y atribución NASA.
- Caption de `nucleo-dirichlet` sin la aclaración "(sin el factor 1/L)".

## Observaciones que quedaron sin tocar (para decidir)

- §14.5 (interpolación trigonométrica): el polinomio $\sum_{j=0}^{N-1}c_je^{2\pi ijx/T}$ interpola en los nodos pero entre los nodos hay que leer $j>N/2$ como $j-N$; podría agregarse una frase.
- §14.8: una etapa de la FFT para $N=8$ cuesta 4 productos y 8 sumas (no 8 productos).
- Para el signo, $\sup|f-S_N|=1$ para todo $N$ (la suma parcial es continua); el 9 % es el sobreimpulso, no el error uniforme.
