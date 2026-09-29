# Figuras de la Parte I generadas por los notebooks (2026-09-29)

**Estado (2026-09-29, noche):** ya aplicado al tex todo lo de las secciones 1 y 2 "reemplazan un `\figpendiente`" (anchos ajustados, cuatro figuras pendientes reemplazadas con referencias en el texto). Queda pendiente decidir sobre las figuras *opcionales* de la sección 2 y las observaciones de la sección 3.

Todas las figuras matplotlib de los capítulos 2–8 salen ahora de `notebooks/01`–`05` (`GUARDAR = True` en la celda de configuración las reescribe en `figuras/`). Este archivo lista qué hay que tocar en `Notas-IMC.tex` para aprovecharlas. `Resorte.png` y `Pendulo.png` (esquemas) quedan como están.

## 1. Figuras regeneradas con el mismo nombre (el tex no cambia, salvo el ancho indicado)

| Figura | Notebook | Qué cambió | Ancho recomendado |
| --- | --- | --- | --- |
| `LV1`, `LVk<1`, `LVk=1`, `LVk>1-1`, `LVk>1-2`, `bacteria1`, `bacteria2`, `SIR`, `logistica`, `tasa-logistica`, `maltus-vs-logistico` | 01 | ejes $h,p$ / $s,i$; nulclinas; parámetros en la figura | 0.6 (como está) |
| `silla`, `nodo1`, `nodo2`, `foco1`, `foco2`, `centro` | 03 | flechas de sentido en todas (el centro no tenía); matriz $A$ y autovalores en la figura | 0.4 (como está; 0.45 si se quiere letra más grande) |
| `clasificacion` | 03 | rehecha en matplotlib: parábola $\det=\mathrm{tr}^2/4$, regiones con nombre, los seis ejemplos marcados | **subir a 0.6** |
| `pendulo-amortiguado` | 03 | ejes $x_1=\theta$, $x_2=\dot\theta$; equilibrios rotulados; separatrices de las sillas | 0.6 |
| `sinHG` | 03 | títulos $\lambda>0$ / $\lambda<0$; flechas legibles. Usa $x,y$ (el tex escribe \eqref{eq:sinHG} con $x_1,x_2$: unificar en un sentido u otro) | **subir a 0.9** |
| `energia_pendulo` | 04 | niveles etiquetados, separatriz $E=2\omega^2$ resaltada, centros y sillas marcados | 0.6 |
| `pendulo_potencial`, `potencial` | 04 | paneles alineados en $x$, niveles de energía punteados en el potencial, separatriz | 0.6 |
| `vanderPol` | 04 | $\lambda=0.5$ (ciclo estable) y $\lambda=-0.5$ (ciclo inestable a trazos, que antes no se veía) | **subir a 0.9** |
| `Lorentz` | 04 | atractor 3D con parámetros | 0.6 |
| `ciclo_limite` | 04 | agrega la región anular $r_1\le r\le r_2$ con el campo apuntando hacia adentro (Poincaré–Bendixson) | 0.6 |
| `nulclinas` | 04 | $\mathcal N_1,\mathcal N_2$ etiquetadas, equilibrios con su tipo. Parámetros: $\rho=1$; izq. $\alpha_1=0.6,\alpha_2=0.8$; der. $\alpha_1=1.4,\alpha_2=1.25$ (para el caption) | **subir a 0.9** |
| `c0-cerca` | 05 | **cambia el ejemplo**: antes $x^3$ vs $x^3-\varepsilon x$ (que también son $C^1$-cercanos y con equilibrio no hiperbólico); ahora $F=-x$ y $G=-x+3\varepsilon\,\mathrm{sen}(x/\varepsilon)$: $\|F-G\|_0=3\varepsilon$ pero aparecen dos equilibrios nuevos. Si se prefiere el original, son dos líneas en `make_nb05.py` | 0.6 |
| `saddle-node`, `transcritica`, `pitchfork` | 05 | tres paneles apilados $\mu<0,=0,>0$ con $f(x,\mu)$ y la recta de fase | 0.6 |
| `diagrama-saddle-node`, `diagrama-transcritica`, `diagrama-pitchfork` | 05 | ramas estable (llena) / inestable (trazos), rectas de fase verticales | 0.6 |
| `nulclinas-bif` | 05 | $b=1$, $a=0.4,0.5,0.6$, nulclinas y equilibrios etiquetados | 0.9 (hoy 0.8) |
| `ej-bif1`, `ej-bif1-fases` | 05 | origen marcado; silla vacía con leyenda de la convención; variedades estable/inestable de la silla | 0.6 (fases: mejor 0.7) |
| `diagrama-ej-bif1` | 05 | rama $x^*=0$ visible; punto $(a_c,1)$ marcado | 0.6 |

## 2. Figuras nuevas: reemplazan `\figpendiente` o se agregan

### Reemplazan un `\figpendiente`

| Figura | Dónde (línea aprox. del tex) | Ancho | Caption sugerido |
| --- | --- | --- | --- |
| `recta-fase-logistica` | 570, cap. 2 | 0.6 | Recta de fase de $\dot p=(1-p)p$: los equilibrios $p=0$ (inestable, vacío) y $p=1$ (estable, lleno) y el sentido del movimiento en cada intervalo; debajo, el gráfico de $f(p)$. |
| `SIR-plano-si` | 868, cap. 2 | 0.9 | Plano de fases $(s,i)$ del modelo SIR para $R_0=0.8$, $2$ y $4$: curvas de nivel de $i+s-\frac1{R_0}\ln s$ en el triángulo $s+i\le1$, recta $s=1/R_0$ y sentido del movimiento ($s$ decreciente). |
| `hopf-diagrama` | 2584, cap. 8 | 0.9 | Bifurcación de Hopf. Arriba: amplitud de la órbita periódica en función de $\mu$ en los casos supercrítico ($\ell_1<0$, ciclo estable para $\mu>0$) y subcrítico ($\ell_1>0$, ciclo inestable para $\mu<0$); trazo lleno = estable, a trazos = inestable. Abajo: retratos de fase de la forma normal $\dot r=\mu r-r^3$, $\dot\theta=1$ para $\mu<0$, $\mu=0$ y $\mu>0$. |
| `circuito-vdp-corriente` | 485, cap. 1 (problema conductor; versión simulada, no un registro real) | 0.8 | Corriente en el circuito de van der Pol a partir de una perturbación pequeña del reposo. Arriba, $\lambda=0.1$: la corriente crece lentamente y se estabiliza en una oscilación casi sinusoidal de amplitud $\approx2$ y período $\approx2\pi$. Abajo, $\lambda=5$: el circuito se enciende de inmediato y oscila con forma de relajación, con período mucho mayor que el natural. |

### Se agregan (opcionales)

| Figura | Dónde | Ancho | Caption sugerido |
| --- | --- | --- | --- |
| `oscilador-armonico` | cap. 3, §3.2 o §3.8 | 0.9 | Oscilador armónico con $k=2$, $m=0.5$: soluciones $x(t)$ para tres amplitudes y las órbitas correspondientes en el plano $(x,v)$, que son las elipses $E=$ cte. El período $2\pi\sqrt{m/k}$ no depende de la amplitud. |
| `resorte-friccion` | cap. 3, §3.3 | 0.9 | Resorte con fricción ($k=2$, $m=0.5$, $x_0=1$, $v_0=0$). Izquierda: los tres regímenes según $\gamma$ frente a $2\sqrt{km}$. Derecha: en el caso sub-amortiguado la trayectoria cruza las elipses $E=$ cte hacia adentro ($\dot E=-\gamma v^2\le0$) y termina en el origen. |
| `pendulo-simulacion` | cap. 3, §3.4 | 0.6 | Péndulo simple ($g=9.8$, $\ell=1$): para $\theta_0=0.2$ la solución es casi sinusoidal con período $\approx2\pi\sqrt{\ell/g}=2.007$; para $\theta_0=2.5$ el período medido es $3.30$ y la forma de onda deja de ser sinusoidal. |
| `energia-vdp` | cap. 3, §3.8 ("Volvemos al circuito") | 0.8 | La energía $E=\frac12(x^2+y^2)$ a lo largo de una solución de van der Pol ($\lambda=1$, $x(0)=0.1$): crece mientras $|x|<1$ y decrece cuando $|x|>1$, así que no es monótona; la solución termina oscilando con $E$ entre $1.17$ y $4.00$. |
| `flujo-conjugacion` | cap. 5, junto a la observación sobre HG topológico | 0.9 | Los sistemas $\dot x=x,\ \dot y=2y+x^2$ (izquierda) y su linealización $\dot x=x,\ \dot y=2y$ (derecha). Cerca del origen ambos son nodos inestables y el homeomorfismo $\mathbf h(x,y)=(x,\,y-x^2\ln|x|)$ lleva las trayectorias de uno en las del otro, pero no hay ningún cambio de coordenadas $C^2$ que lo haga. |
| `linealizacion-LV2` | cap. 5, §5.4 después de los autovalores de LV2 | 0.9 (media página de alto; alternativa: una sola fila) | Hartman–Grobman en el modelo \eqref{LV2}: a la izquierda el sistema en una ventana alrededor del equilibrio de coexistencia, a la derecha el sistema linealizado en las mismas coordenadas. Arriba $\rho^2\ge4k(k-1)$ (nodo estable), abajo $\rho^2<4k(k-1)$ (foco estable). |
| `lyapunov-pendulo` | cap. 6, §6.2 | 0.9 | Izquierda: energía $E(t)$ del péndulo amortiguado ($\omega=1$, $a=0.15$, $x(0)=2.6$, $y(0)=0$) y $\dot E=-2ay^2$; los puntos marcan los instantes con $y=0$, donde $\dot E=0$: $E$ no es estricta pero igual decrece a $0$ (LaSalle). Derecha: $V=\frac12(x^2+y^2)$ a lo largo de soluciones de van der Pol: decrece hacia $0$ para $\lambda=-0.5$ partiendo de $(0.9,0)$, y crece para $\lambda=0.5$ desde $(0.1,0)$ hasta que $|x|>1$. |
| `competencia-retratos` | cap. 7, final de §7.3 | 0.9 | Retratos de fase del modelo de competencia ($\rho=1$) en los cuatro casos: $a_1,a_2>1$ (coexistencia asintóticamente estable), $a_1,a_2<1$ (la coexistencia es una silla: biestabilidad, gana la especie que parte con ventaja), $a_1<1<a_2$ (gana la especie 2) y $a_2<1<a_1$ (gana la especie 1). |
| `hopf-generica-vs-vdp` | cap. 8, después del `volvemos{el circuito}` | 0.9 | Hopf genérica frente al caso degenerado. Izquierda: amplitud del ciclo de $\ddot x+(x^2-\mu)\dot x+x=0$, que nace en $\mu=0$ con amplitud $2\sqrt\mu$. Derecha: van der Pol, donde el ciclo aparece con amplitud $\approx2$ para todo $\lambda>0$. |
| `sir-demografia` | cap. 8, §8.6 | 0.9 | SIR con nacimientos ($\gamma=1$, $\mu=0.02$). (a) Diagrama de bifurcación: la rama endémica $i^*=\frac{\mu}{\beta}(R_0-1)$ intercambia estabilidad con la libre de enfermedad en $R_0=1$; el punto lleno es el equilibrio del panel (b). (b) Con $\beta=3$ ($R_0\approx2.94$) el equilibrio endémico es un foco estable: $i(t)$ se acerca con ondas de período $\approx32$. |

## 3. Observaciones sobre el texto que surgieron al hacer las cuentas

Ninguna afirmación numérica del texto resultó incorrecta (autovalores de §5.4, $k_c=2+\beta$, $a_c=1/(2b)$, amplitud $2\sqrt\mu$, SIR con demografía: todo coincide con $10^{-10}$ o mejor). Detalles de redacción:

- §3.2 escribe el oscilador como $\ddot x+kx=0$ (sin $m$) mientras §3.7–3.8 usan $m\ddot x+kx=0$.
- §3.3 no menciona el umbral $\gamma=2\sqrt{km}$ entre los regímenes sub/sobre-amortiguado (el notebook lo usa).
- El título del cap. 3 es "Sistemas mecánicos" aunque §3.6 es de circuitos; el notebook se llama "Sistemas mecánicos y eléctricos".
- Caption de `fig:vanderPol`: el ciclo inestable para $\lambda<0$ es la reflexión $(x,y)\mapsto(x,-y)$ del ciclo de $\lambda>0$ (invertir el tiempo cambia el signo de $y=\dot x$), no "el mismo ciclo recorrido al revés".
- §7.3: "la coexistencia es posible si y sólo si $\mathcal N_1\cap\mathcal N_2\ne\emptyset$": en el caso $a_i<1$ la intersección existe pero es una silla (biestabilidad); convendría "hay un equilibrio de coexistencia".
- `c0-cerca`: ver arriba (el ejemplo original no ilustraba lo que dice el párrafo).
