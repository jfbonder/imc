# Figuras de la Parte III generadas por los notebooks (2026-09-29)

Las figuras de los capítulos 18–20 salen de `notebooks/09-calor`, `10-laplace` y `11-ondas`. **Todo lo de este archivo ya está aplicado en `Notas-IMC.tex`.** Quedan como esquemas dibujados a mano (no los genera ningún script): `QT`, `ondas-resorte`, `ondas-tension`. El escaneo del Griffiths se reemplazó por `ranura-placas` (`tools/fig_ranura.py`) y el espectro de la cuerda real por `espectro-cuerda` (`tools/fig_cuerda.py`, con `datos/cuerda_guitarra.wav`): ya no queda ninguna `\figpendiente` en el texto.

## Datos

- `datos/suelo_horaria.csv`: 2023, aire + temperatura media de cuatro capas de suelo (0–7, 7–28, 28–100, 100–255 cm), reanálisis ERA5-Land vía Open-Meteo. Son promedios por capa, no mediciones puntuales; en los ajustes se usa la profundidad media de la capa y el texto lo aclara.

## Figuras regeneradas (mismo nombre)

| Figura | Notebook | Qué cambió |
| --- | --- | --- |
| `Browniano` | 09 | estilo común, parámetros en la figura |
| `V-heat`, `u-heat` | 09 | mismos ejemplos; `u-heat` ahora sí con $D=0.1$ y tiempos hasta $\tau=160$ (la vieja correspondía a $D\approx0.4$); primer modo superpuesto en `V-heat` |
| `sol-fundamental` | 09 | $D=0.4$, cinco tiempos |
| `errores-heat-log` | 09 | pendiente ajustada $-7.90$, recta teórica $2\pi^2D$ y, en gris, el esquema implícito con $\Delta t=0.01$ (pendiente $-7.60$, que era la de la figura vieja) |
| `sup-min1`, `sup-min2` | 10 | ahora son distintas: mismo alambre, superficie arrugada (área 4.20) vs armónica (3.41); única excepción 3D |
| `juego` | 10 | rehecha en matplotlib |
| `sol-fundamental-lap`, `superposicion`, `convolucion`, `poisson-fourier` | 10 | mapas de color 2D con contornos en lugar de superficies 3D |
| `sol-ondas-20nodos`, `energia-ondas` | 11 | dato identificado ($g=\sin\pi x+\frac12\sin3\pi x$, $h=0$, $L=c=1$) y escrito en el caption; energía por modos verificada |

## Figuras fusionadas (archivos viejos borrados)

- `estacionaria-heat` + `heat2d-1..4` → `heat2d` (panel 2×3, misma escala de color).
- `errores-heat` (lineal) → eliminada.
- `viajera`, `armonica`, `estacionaria`, `plana`, `esferica` → `catalogo-ondas` (referencias `\ref{fig:catalogo-ondas}(a)`…(e)). En la vieja `estacionaria` los instantes estaban mal rotulados ($T/2$ y $T$ en lugar de $T/4$ y $T/2$).
- `dominio-dependencia` + `rango-influencia` → `caracteristicas` (a)/(b).

## Figuras nuevas

| Figura | Notebook | Dónde | Ancho |
| --- | --- | --- | --- |
| `paseo-azar` | 09 | §18.2, tras el browniano | 0.9 |
| `suelo-datos` | 09 | intro de la Parte III (era `\figpendiente`) | 0.95 |
| `suelo-periodica` | 09 | §18.6 (era `\figpendiente`) | 0.95 |
| `suelo-ajuste`, `suelo-caneria` | 09 | §18.6, final | 0.9 / 0.6 |
| `juego-montecarlo` | 10 | §19.2 | 0.95 |
| `principio-maximo` | 10 | §19.3 | 0.6 |
| `valor-medio` | 10 | §19.3, tras el teorema | 0.85 |
| `ondas-esquemas` | 11 | §20.1 (era `\figpendiente` de fotos) | 0.95 |
| `modos-cuerda` | 11 | "Volvemos" de la cuerda | 0.95 |
| `dalembert-pulso` | 11 | tras la fórmula de d'Alembert | 0.9 |
| `reflexion-pulso` | 11 | §20.7 (era `\figpendiente`) | 0.95 |

## Cambios de texto hechos a partir de las cuentas

- §18.5, tasa de decaimiento: la pendiente medida es $7.90=0.8\pi^2$ (coincide con la teoría); el $7.6$ de la versión anterior era la tasa del esquema de Euler implícito con $\Delta t=0.01$, $\ln(1+D\lambda_1\Delta t)/\Delta t$. Se reescribió el párrafo (ya no se atribuye a "error de discretización y transitorio").
- §18.6: se agregó el resultado del ajuste de $D$ con los datos (diario: $3$–$5\cdot10^{-7}$; anual: retraso consistente, atenuación menor por ser capas promedio) y la respuesta a la cañería.
- Caption de `poisson-fourier`: $-\Delta u=f$ (no $\Delta u=f$) y "20 modos" (no "nodos").
- Caption de `sup-min`: sin la nota "(Figura a rehacer)".
- Caption de `reflexion-pulso`: se menciona el instante en que la cuerda queda plana (toda la energía cinética).

## Observaciones (aplicadas el 2026-09-29)

- §19.5: frase sobre la convergencia rápida de la serie de $u$ frente al Gibbs de la serie de $f$.
- §18.6: frase sobre el forzado: la capa superficial oscila más que el aire (radiación); el forzado correcto es la temperatura de la superficie.
- `ejemplo3-3-griffiths.png` (escaneo) reemplazado por `ranura-placas.png` (`tools/fig_ranura.py`, matplotlib).
