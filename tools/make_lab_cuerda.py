"""Genera notebooks/lab-cuerda.ipynb (estudiantes) y notebooks/docente/lab-cuerda.ipynb.

Laboratorio: el espectro de una cuerda real. Cubre el ejercicio ``lab:cuerda`` de la sección
"Laboratorio: los problemas conductores" de la Parte III con la grabación ``datos/cuerda_guitarra.wav``
(cuerda Sol, mono, 44.1 kHz, 6 s, CC0).

Con la variable de entorno LAB_REVISION=1 la versión docente guarda además algunas figuras en
/tmp/lab-cuerda-*.png (celdas auxiliares de revisión; no forman parte del notebook final).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Lab  # noqa: E402

REPO = "jfbonder/imc"
REVISION = bool(os.environ.get("LAB_REVISION"))

lab = Lab("lab-cuerda", "Laboratorio: el espectro de una cuerda real", REPO)


def figura_revision(nombre, var="fig"):
    """Celda docente auxiliar que guarda la última figura (solo con LAB_REVISION)."""
    if REVISION:
        lab.code(f'{var}.savefig("/tmp/lab-cuerda-{nombre}.png")  # celda auxiliar de revisión', destino="docente")


# =============================================================================
# Presentación
# =============================================================================
lab.md(r"""
Este laboratorio cierra el segundo problema conductor de la Parte III: la cuerda vibrante. En el texto planteamos el modelo ($u_{tt}=c^2u_{xx}$ con extremos fijos), lo resolvimos con series de Fourier (modos normales $\sin(k\pi x/L)$ de frecuencias $f_k = k f_1$, con $f_1=\frac1{2L}\sqrt{T/\rho}$) y dijimos que el **timbre** es el reparto de energía entre los modos. Quedó pendiente la pregunta 4 del problema: **si grabamos el sonido de una cuerda y calculamos su espectro, ¿qué esperamos ver? ¿Qué vemos en realidad? ¿Qué le falta al modelo?** Eso es lo que hacemos acá, con una grabación real de una cuerda de guitarra.

**Qué vamos a hacer.** Cargar el audio y entender qué son los números (Tarea 1); medir el espectro de un tramo, la frecuencia fundamental $f_1$ y los picos siguientes con una precisión mucho mejor que la resolución de la DFT (Tarea 2); ver que los picos **no** están exactamente en $kf_1$, ajustar el parámetro de **inarmonicidad** $B$ del modelo de cuerda con rigidez y juzgar si su valor es razonable (Tarea 3); medir con un espectrograma cuánto tarda en apagarse cada armónico y comparar dos modelos de fricción (Tarea 4); comparar el reparto de energía entre armónicos con el de una cuerda ideal pulsada en distintos puntos (Tarea 5); **simular** la cuerda con el esquema leapfrog que ya programaron, con fricción, "escuchar" el resultado y compararlo con lo real (Tarea 6); opcionalmente, agregar la rigidez al esquema (Tarea 7); y hacer el balance de lo que el modelo ideal predice bien y de lo que le falta (Tarea 8).

**Lo que las notas no explican y este notebook sí.** Cómo se representa un sonido digital; qué ventana usar y qué resolución tiene el espectro de un tramo, y cómo medir una frecuencia **mejor** que esa resolución con la interpolación parabólica del pico; mínimos cuadrados (lineales y no lineales) aplicados a frecuencias y a logaritmos de amplitudes, con su incertidumbre; el espectrograma como instrumento de medida; la deducción del modelo con rigidez y de los dos modelos de fricción; y los detalles de simular un instrumento con diferencias finitas (paso de tiempo, muestreo del "micrófono", estabilidad con rigidez).

**Una sola grabación.** El enunciado del ejercicio pide "repetir pulsando la cuerda en el medio y cerca del puente". Acá hay una única grabación, así que ese punto (Tarea 5) se hace con la **simulación y las fórmulas**: la cuerda ideal pulsada en $x_0=L/2$, $L/5$ y $L/20$, contra el espectro real. Si tenés una guitarra (o cualquier cuerda) a mano, grabá el mismo sonido pulsando en distintos puntos y repetí las Tareas 2 y 5 con tus archivos: es lo más instructivo que se puede hacer con este laboratorio, y es opcional.

**Herramientas disponibles.** `imc.datos.obtener` (el archivo de audio), `scipy.io.wavfile` (leer y escribir WAV), `imc.espectro.espectrograma` (Tarea 4), `imc.estilo`, `scipy.optimize.curve_fit` y `numpy.polyfit` para los ajustes. La solución en serie de la cuerda pulsada es la del Ejercicio guiado de la cuerda; el esquema leapfrog es el del laboratorio de EDPs (Tarea 9), que acá se extiende. No repetimos el notebook `11-ondas`: allí están los modos, la cuerda pulsada ideal y la reflexión.

**Cómo se evalúa.** La sección final de **interpretación escrita** (las cuatro preguntas del problema conductor, ahora con números). Cada tarea dice qué se espera y trae una celda de verificación. Tiempo estimado: una sesión de 4 h para las Tareas 1 a 6 y 8; la Tarea 7 es opcional (30 minutos).
""")

lab.configuracion(extra="""
import os
import tempfile
from scipy.io import wavfile
from scipy.optimize import curve_fit
from imc import espectro
from imc.estilo import CICLO
""")

# =============================================================================
# 1. El modelo ideal y qué predice
# =============================================================================
lab.md(r"""
## 1. El modelo ideal: qué predice

Sea $u(x,t)$ el desplazamiento transversal de una cuerda de longitud $L$, tensión $T$ y densidad lineal $\rho$, con los extremos fijos. Para desplazamientos chicos la ecuación es
$$u_{tt}=c^2u_{xx},\qquad c^2=\frac T\rho,\qquad u(0,t)=u(L,t)=0,$$
y su solución es la superposición de **modos normales**
$$u(x,t)=\sum_{k\ge1}\bigl(A_k\cos\omega_kt+B_k\sin\omega_kt\bigr)\sin\frac{k\pi x}{L},\qquad \omega_k=\frac{ck\pi}{L},\quad f_k=\frac{\omega_k}{2\pi}=k\,f_1,\quad f_1=\frac1{2L}\sqrt{\frac T\rho}.$$
Para una cuerda **pulsada** (soltada desde el reposo con la forma triangular de altura $a$ y vértice en $x_0$), $B_k=0$ y (Ejercicio guiado de la cuerda)
$$A_k=\frac{2aL^2}{\pi^2x_0(L-x_0)}\,\frac{\sin(k\pi x_0/L)}{k^2}.$$
El **timbre** es la lista de los $A_k$ (o de la energía $E_k=\frac L4\omega_k^2A_k^2$ de cada modo): la misma nota, con la misma $f_1$, suena distinto según cuánta energía tenga en cada armónico. Tocar cerca del puente ($x_0\ll L$) da un sonido brillante; tocar en el medio da un sonido "redondo" (sin armónicos pares, porque $\sin(k\pi/2)=0$ para $k$ par).

### Qué predice el modelo, y qué vamos a chequear

1. **Frecuencias:** picos en $f_k=kf_1$, exactamente equiespaciados (un espectro "de peine").
2. **Amplitudes:** $A_k\propto\sin(k\pi x_0/L)/k^2$: decaen como $1/k^2$, con ceros en los $k$ múltiplos de $L/x_0$.
3. **Duración:** ¡el modelo no tiene fricción! La energía se conserva y la cuerda vibra para siempre, todos los modos con la misma amplitud constante.

Una cuerda real contradice el punto 3 de entrada (el sonido se apaga), y el laboratorio muestra que también matiza los puntos 1 y 2. **Una advertencia sobre lo que se mide:** el micrófono no registra el desplazamiento $u(x_m,t)$ de un punto de la cuerda, sino una presión sonora producida por el cuerpo de la guitarra, que amplifica algunas frecuencias más que otras. Las **frecuencias** y las **tasas de decaimiento** de los modos sí se transmiten sin cambio (el cuerpo es lineal e independiente del tiempo); las **amplitudes relativas** entre armónicos, en cambio, quedan multiplicadas por la respuesta del cuerpo y sólo se pueden comparar con el modelo de manera cualitativa. Tenelo presente en la Tarea 5.
""")

# =============================================================================
# 2. Sonido digital y espectro de un tramo
# =============================================================================
lab.md(r"""
## 2. Sonido digital y espectro de un tramo

### Qué hay en un archivo WAV

Un archivo WAV sin comprimir guarda la señal muestreada: una lista de enteros (acá `int16`, de $-32768$ a $32767$) y la **frecuencia de muestreo** $f_s$ (muestras por segundo). `scipy.io.wavfile.read` devuelve `(fs, datos)`. Conviene **normalizar** dividiendo por $32768$: así la señal `x` tiene valores en $[-1,1)$ y los espectros no dependen de la codificación. El instante de la muestra $j$ es $t_j=j/f_s$; la duración es $N/f_s$; y por el teorema de muestreo (Parte II) sólo hay información hasta la **frecuencia de Nyquist** $f_s/2$: con $f_s=44100$ Hz, hasta $22050$ Hz, que cubre el rango audible.

### El espectro de un tramo: ventana y resolución

Si tomamos un tramo de duración $T$ (con $M=Tf_s$ muestras), la DFT da las frecuencias $\xi_j=j/T$: la **resolución** es $\Delta\xi=1/T$. Para $T=0.5$ s son $2$ Hz; para $T=2$ s, $0.5$ Hz. Dos picos más cercanos que unas pocas resoluciones no se distinguen: **para ver mejor en frecuencia hay que mirar más tiempo**, y el precio es promediar sobre un tramo en el que la señal cambia (acá, se apaga). Ése es el compromiso que hay que elegir.

Cortar el tramo de golpe equivale a multiplicar por una ventana rectangular y "derrama" energía de cada pico sobre las frecuencias vecinas (*fuga espectral*). Multiplicar por una **ventana de Hann** $w_j=\frac12\bigl(1-\cos\frac{2\pi j}{M-1}\bigr)$ baja la fuga (los lóbulos laterales quedan $31$ dB debajo del pico, y decaen rápido) a cambio de ensanchar el lóbulo principal (de ancho $\pm2/T$ en vez de $\pm1/T$). Como los armónicos de una cuerda están separados por $f_1\approx196$ Hz y $2/T\ll196$ Hz, el ensanchamiento no molesta y la fuga baja permite ver armónicos de $-60$ dB o menos sin que los tape la cola del fundamental. Normalizamos para que un seno de amplitud $A$ dé un pico de altura $A$:
$$A(\xi_j)=\frac{2\,\bigl|\sum_m w_mx_m\,e^{-2\pi i\,jm/M}\bigr|}{\sum_m w_m}.$$
*(Rellenar con ceros, `np.fft.rfft(seg, n=4*M)`, interpola visualmente el gráfico pero **no** agrega resolución: la información sigue siendo la de $T$ segundos. Lo que sí permite medir mejor un pico es lo que sigue.)*

### Medir una frecuencia mejor que la resolución: interpolación parabólica

Si el pico más alto está en el índice $j$, la frecuencia verdadera está entre $\xi_{j-1/2}$ y $\xi_{j+1/2}$: el error de "tomar el máximo" es de hasta $\Delta\xi/2$ (1 Hz si $T=0.5$ s), mucho más que los corrimientos que queremos medir (¡el corrimiento de $k=2$ por inarmonicidad es de unos $0.3$ Hz!). Pero el lóbulo principal de la ventana de Hann tiene la forma de una campana casi gaussiana y **el logaritmo de una gaussiana es una parábola**. Entonces ajustamos una parábola por los tres puntos $(\xi_{j-1},a),(\xi_j,b),(\xi_{j+1},c)$, con $a,b,c=\ln A_{j-1},\ln A_j,\ln A_{j+1}$, y tomamos su vértice:
$$\delta=\frac12\,\frac{a-c}{a-2b+c}\in\bigl(-\tfrac12,\tfrac12\bigr),\qquad \xi_{pico}\approx\xi_j+\delta\,\Delta\xi,\qquad A_{pico}\approx\exp\bigl(b-\tfrac14(a-c)\,\delta\bigr).$$
En la práctica esto da la frecuencia con un error de una pequeña fracción de $\Delta\xi$ (en la Tarea 2 lo medís con una señal de frecuencia conocida). **Errores típicos:** interpolar sobre la amplitud en vez de su logaritmo (funciona peor); aplicarlo a un pico que no es el máximo local (el vértice sale fuera de $(-\frac12,\frac12)$: chequealo); y olvidar que si dos picos están a menos de $\sim4/T$ el vértice es un promedio sin sentido (la cuerda real tiene *dobletes* de ese tipo, como vas a ver).
""")

lab.tarea(
    titulo="Cargar el audio, graficar la señal y su envolvente",
    consigna=r"""
Cargá `datos.obtener("cuerda_guitarra.wav")` con `wavfile.read` y construí: `fs` (frecuencia de muestreo), `x` (la señal **normalizada** a $[-1,1)$, en `float`), `N` (cantidad de muestras) y `t` (el tiempo en segundos).

1. Imprimí `fs`, `N`, la duración, el tipo de dato original, el máximo de $|x|$ y el promedio de `x` (¿está centrada?). Comprobá que no hay saturación (valores en el tope).
2. Escribí `envolvente(x, fs, ventana_ms=20)`: la raíz de la media móvil de $x^2$ en ventanas de `ventana_ms` milisegundos (`np.convolve` con `mode="same"`). Graficá, en dos paneles, **(a)** la señal $x(t)$ y **(b)** la envolvente en escala **logarítmica** (`semilogy`).
3. Ajustá una recta a $\ln(\text{env})$ entre $t=1$ s y $t=4$ s: la pendiente es $-\gamma_{glob}$ (la **tasa global de decaimiento de la amplitud**, en s$^{-1}$). Calculá también la duración $T_{40}$: el tiempo, medido desde el máximo de la envolvente, que tarda en caer $40$ dB (un factor $100$ en amplitud).

**Qué se espera.** La señal es un ataque brusco y un decaimiento suave (una "campana"); en escala log la envolvente no es una recta: cae más rápido al principio y después sigue una recta bastante buena (decaimiento en dos etapas, típico de las guitarras). $\gamma_{glob}$ es del orden de $1$ s$^{-1}$ (la amplitud se divide por $e$ cada segundo, más o menos) y $T_{40}$ de unos 3.5 s. Esa recta tardía la domina el armónico fundamental, que es el que dura más; en la Tarea 4 vas a ver que los demás se apagan más rápido. Fijate también si hay algo raro cerca de $t=5.5$ s (un chasquido) y anotá la hora: hay que evitarlo en las mediciones.
""",
    esqueleto=r'''
fs, x_int = ...   # TODO: wavfile.read(datos.obtener("cuerda_guitarra.wav"))
x = ...           # TODO: señal normalizada a [-1, 1)
N = ...           # TODO
t = ...           # TODO: tiempo en segundos

def envolvente(x, fs, ventana_ms=20):
    """Raíz de la media móvil de x^2 en ventanas de ventana_ms milisegundos (misma longitud que x)."""
    pass  # TODO

env = envolvente(x, fs)
gamma_glob = ...  # TODO: -pendiente del ajuste lineal de ln(env) entre 1 y 4 s
T40 = ...         # TODO: segundos desde el máximo hasta que la envolvente cae 40 dB

# TODO: imprimir fs, N, duración, tipo, máx |x|, promedio; figura de dos paneles (señal y envolvente en semilogy)
''',
    solucion=r'''
fs, x_int = wavfile.read(datos.obtener("cuerda_guitarra.wav"))
x = x_int / 32768.0
N = len(x)
t = np.arange(N) / fs

def envolvente(x, fs, ventana_ms=20):
    """Raíz de la media móvil de x^2 en ventanas de ventana_ms milisegundos (misma longitud que x)."""
    n = int(ventana_ms * 1e-3 * fs)
    return np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same"))

env = envolvente(x, fs)
m = (t >= 1) & (t <= 4)
gamma_glob = -np.polyfit(t[m], np.log(env[m]), 1)[0]
i_max = np.argmax(env)
T40 = t[i_max + np.argmax(env[i_max:] < env[i_max] * 10 ** (-40 / 20))] - t[i_max]

print(f"fs = {fs} Hz, N = {N}, duración = {N / fs:.2f} s, dtype original = {x_int.dtype}, Nyquist = {fs / 2:.0f} Hz")
print(f"max|x| = {np.abs(x).max():.3f} (tope 1.0), promedio = {x.mean():.2e}")
print(f"tasa global gamma_glob = {gamma_glob:.2f} 1/s;  T40 = {T40:.2f} s (desde t = {t[i_max]:.3f} s)")

fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
axs[0].plot(t, x, lw=0.4, color=COLORES["dato"])
axs[0].set_ylabel("$x(t)$ (señal normalizada)")
axs[1].semilogy(t, env, color=COLORES["dato"], label="envolvente (RMS de 20 ms)")
axs[1].semilogy(t[m], np.exp(np.polyval(np.polyfit(t[m], np.log(env[m]), 1), t[m])), color=COLORES["modelo"], lw=2.2,
                label=rf"ajuste en [1, 4] s: $\gamma_{{glob}}={gamma_glob:.2f}$ s$^{{-1}}$")
axs[1].axhline(env[i_max] * 1e-2, color="0.6", ls=":", lw=1)
axs[1].text(0.05, env[i_max] * 1.3e-2, "$-40$ dB", color="0.4", fontsize=9)
axs[1].set_xlabel("$t$ (s)"); axs[1].set_ylabel("envolvente"); axs[1].set_xlim(0, t[-1]); axs[1].legend(loc="upper right")
fig.tight_layout()
''',
    verificacion=r'''
# Verificación
assert fs == 44100 and N == 264600 and abs(x).max() <= 1.0 and abs(x).max() > 0.5, "señal mal cargada o sin normalizar"
assert 0.8 < gamma_glob < 1.5, "la tasa global debería ser del orden de 1 1/s"
assert 2.5 < T40 < 5.0, "T40 debería ser de unos 3.5 s"
assert env.shape == x.shape and np.all(env >= 0)
print("señal y envolvente: OK")
''')
figura_revision("senal")

# ---- Tarea 2 -----------------------------------------------------------------
lab.tarea(
    titulo="El espectro de un tramo, la fundamental y los armónicos",
    consigna=r"""
Escribí las tres funciones siguientes (las vas a reutilizar en todo el laboratorio):

* `espectro_tramo(x, fs, t0, t1)`: espectro de amplitudes (con la normalización $2|\text{DFT}|/\sum w$ de la Sección 2) del tramo $t_0\le t<t_1$ con ventana de Hann. Devuelve `(f, A)`.
* `interp_parabolica(f, A, j)`: frecuencia y altura del pico interpolado alrededor del índice `j`.
* `medir_picos(f, A, f1, K=15, umbral=20)`: para $k=1,\dots,K$ busca el máximo de `A` en $(k-0.3)f_1<f<(k+0.3)f_1$, exige que sea un máximo **interior** a la ventana y que supere `umbral` veces el **piso de ruido** (la mediana de `A` entre $6$ y $9$ kHz), e interpola. Devuelve un arreglo con columnas `(k, f_k, A_k)` sólo de los picos aceptados.

Con eso:

1. Calculá el espectro del **primer medio segundo** (`fa, Aa`, tramo $[0,0.5]$ s) y el del tramo $[0.5,2.5]$ s (`fb, Ab`); graficalos en `semilogy` hasta $4$ kHz. Definí `medir_f1(f, A)`, que interpola el mayor pico entre $150$ y $250$ Hz, y calculá `f1a` y `f1b`. Los picos aceptados del primer tramo van a `picos_a`, con `f1 = f1a`, y los del segundo tramo a `picos_b`; marcalos en el gráfico.
2. **Validá tu interpolación** con una señal de frecuencia conocida: un seno de $1234.567$ Hz y amplitud $0.3$ durante $0.5$ s a $f_s=44100$ Hz. Comparalo con el error de tomar sólo el máximo de la DFT.
3. **¿Es constante la fundamental?** Calculá `f1` en tramos de $0.5$ s desplazados de a $0.25$ s ($t_0=0,0.25,\dots,4$ s) y graficala en función del tiempo. Anotá cuánto cambia.

**Qué se espera.** (1) Una fundamental en $f_1\approx196$ Hz y una sucesión de picos casi equiespaciados: en el primer tramo se aceptan los $15$ (con el umbral por defecto); en el segundo, unos $12$ (los armónicos altos ya se apagaron). La fundamental es por lejos el pico más alto; los otros pueden ser $10$ a $1000$ veces más bajos. (2) El error de la interpolación es de unas centésimas de hertz ($0.03$ Hz), contra hasta $1$ Hz de "tomar el máximo" ($0.57$ Hz en este caso). (3) $f_1$ **baja** unos $0.4$ a $0.5$ Hz (un $0.2$–$0.3$ %) durante el primer segundo y después queda casi quieta: cuando la cuerda se pulsa fuerte, el estiramiento aumenta la tensión y sube el tono; al apagarse vuelve. Es un efecto **no lineal** que el modelo lineal no tiene; y tiene una consecuencia práctica: $f_1$ depende del tramo que mires. Por eso (y por qué no importa mucho ahora) cada análisis de este laboratorio usa la $f_1$ **de su propio tramo**.
""",
    esqueleto=r'''
def espectro_tramo(x, fs, t0, t1):
    """Espectro de amplitudes con ventana de Hann del tramo t0 <= t < t1. Devuelve (f, A)."""
    pass  # TODO

def interp_parabolica(f, A, j):
    """Frecuencia y altura del pico interpolado (parábola en ln A por j-1, j, j+1). Devuelve (f_pico, A_pico)."""
    pass  # TODO

def medir_f1(f, A):
    """f1 interpolada: mayor pico entre 150 y 250 Hz."""
    pass  # TODO

def medir_picos(f, A, f1, K=15, umbral=20):
    """Picos k = 1..K cerca de k*f1. Devuelve un arreglo con columnas (k, f_k, A_k) de los aceptados."""
    pass  # TODO

fa, Aa = espectro_tramo(x, fs, 0.0, 0.5)
fb, Ab = espectro_tramo(x, fs, 0.5, 2.5)
f1a, f1b = ..., ...              # TODO
picos_a, picos_b = ..., ...      # TODO (picos_a con f1a, picos_b con f1b)
f1 = f1a

# TODO: figura con los dos espectros (semilogy, hasta 4 kHz) y los picos
# TODO: validación con el seno sintético de 1234.567 Hz
# TODO: f1 en tramos de 0.5 s cada 0.25 s, y su gráfico
''',
    solucion=r'''
def espectro_tramo(x, fs, t0, t1):
    """Espectro de amplitudes con ventana de Hann del tramo t0 <= t < t1. Devuelve (f, A)."""
    seg = x[int(round(t0 * fs)):int(round(t1 * fs))]
    w = np.hanning(len(seg))
    return np.fft.rfftfreq(len(seg), 1 / fs), 2 * np.abs(np.fft.rfft(seg * w)) / w.sum()

def interp_parabolica(f, A, j):
    """Frecuencia y altura del pico interpolado (parábola en ln A por j-1, j, j+1). Devuelve (f_pico, A_pico)."""
    a, b, c = np.log(A[j - 1:j + 2])
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    return f[j] + delta * (f[1] - f[0]), np.exp(b - 0.25 * (a - c) * delta)

def medir_f1(f, A):
    """f1 interpolada: mayor pico entre 150 y 250 Hz."""
    j = np.argmax(A * ((f > 150) & (f < 250)))
    return interp_parabolica(f, A, j)[0]

def medir_picos(f, A, f1, K=15, umbral=20):
    """Picos k = 1..K cerca de k*f1. Devuelve un arreglo con columnas (k, f_k, A_k) de los aceptados."""
    piso = np.median(A[(f > 6000) & (f < 9000)])
    salida = []
    for k in range(1, K + 1):
        idx = np.nonzero((f > (k - 0.3) * f1) & (f < (k + 0.3) * f1))[0]
        j = idx[np.argmax(A[idx])]
        if j in (idx[0], idx[-1]) or A[j] < umbral * piso:
            continue
        fk, Ak = interp_parabolica(f, A, j)
        salida.append((k, fk, Ak))
    return np.array(salida)

fa, Aa = espectro_tramo(x, fs, 0.0, 0.5)
fb, Ab = espectro_tramo(x, fs, 0.5, 2.5)
f1a, f1b = medir_f1(fa, Aa), medir_f1(fb, Ab)
picos_a, picos_b = medir_picos(fa, Aa, f1a), medir_picos(fb, Ab, f1b)
f1 = f1a
print(f"tramo [0, 0.5] s:   resolución {fa[1]:.1f} Hz, f1 = {f1a:.2f} Hz, {len(picos_a)} picos aceptados, piso de ruido {np.median(Aa[(fa > 6000) & (fa < 9000)]):.1e}")
print(f"tramo [0.5, 2.5] s: resolución {fb[1]:.1f} Hz, f1 = {f1b:.2f} Hz, {len(picos_b)} picos aceptados")

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, f_, A_, P, titulo in [(axs[0], fa, Aa, picos_a, "primer medio segundo"), (axs[1], fb, Ab, picos_b, "tramo 0.5 a 2.5 s")]:
    m_ = f_ < 4000
    ax.semilogy(f_[m_], A_[m_], lw=0.8, color=COLORES["dato"])
    ax.plot(P[:, 1], P[:, 2], "o", color=COLORES["modelo"], ms=4, label="picos medidos")
    ax.set_title(titulo); ax.set_xlabel(r"$\xi$ (Hz)")
axs[0].set_ylabel("amplitud"); axs[0].legend(loc="upper right"); axs[0].set_ylim(1e-6, 1)
fig.tight_layout()

# validación: seno de frecuencia conocida
f_ver, tt = 1234.567, np.arange(int(0.5 * fs)) / fs
fv, Av = espectro_tramo(0.3 * np.sin(2 * np.pi * f_ver * tt), fs, 0, 0.5)
jv = np.argmax(Av); fi, Ai = interp_parabolica(fv, Av, jv)
print(f"\nseno de {f_ver} Hz: máximo de la DFT en {fv[jv]:.1f} Hz (error {fv[jv] - f_ver:+.3f}), interpolado en {fi:.3f} Hz (error {fi - f_ver:+.4f}); altura {Ai:.4f} (verdadera 0.3)")

# la fundamental en el tiempo
t0s = np.arange(0, 4.01, 0.25)
f1_t = np.array([medir_f1(*espectro_tramo(x, fs, a0, a0 + 0.5)) for a0 in t0s])
print(f"f1 en el tiempo: de {f1_t[0]:.2f} Hz (t0 = 0) a un mínimo de {f1_t.min():.2f} Hz; variación relativa {(f1_t.max() - f1_t.min()) / f1_t.mean():.2%}")
fig2, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(t0s + 0.25, f1_t, "o-", color=COLORES["dato"], ms=4)
ax.set_xlabel("$t$ (centro del tramo de 0.5 s) (s)"); ax.set_ylabel("$f_1$ (Hz)")
fig2.tight_layout()
''',
    verificacion=r'''
# Verificación
assert abs(f1a - 195.8) < 0.5 and abs(f1b - 195.8) < 0.5, "f1 debería ser 195.8 +- 0.5 Hz"
assert len(picos_a) >= 12 and len(picos_b) >= 8, "deberías identificar al menos 12 picos en el primer tramo"
k_, fk_ = picos_a[:, 0], picos_a[:, 1]
assert np.all(np.abs(fk_ / (k_ * f1a) - 1) < 0.02), "los picos deberían estar a menos del 2 % de k*f1"
fv_, Av_ = espectro_tramo(0.3 * np.sin(2 * np.pi * 1234.567 * np.arange(22050) / 44100), 44100, 0, 0.5)
assert abs(interp_parabolica(fv_, Av_, np.argmax(Av_))[0] - 1234.567) < 0.05, "la interpolación debería dar el seno con error < 0.05 Hz"
print("espectro y picos: OK")
''')
figura_revision("espectro")
figura_revision("f1-tiempo", "fig2")

# =============================================================================
# 3. Inarmonicidad
# =============================================================================
lab.md(r"""
## 3. Inarmonicidad: la cuerda con rigidez

### El modelo y su deducción

Una cuerda real no es un hilo perfectamente flexible: tiene **rigidez a la flexión** (curvarla cuesta energía, como a una varilla). La fuerza de restitución extra es proporcional a $-u_{xxxx}$ y la ecuación pasa a ser
$$u_{tt}=c^2u_{xx}-\kappa\,u_{xxxx},\qquad \kappa>0,$$
con $\kappa=EI/\rho$ ($E$: módulo de Young; $I$: momento de inercia de la sección). Sigue habiendo extremos "apoyados" ($u=0$ y $u_{xx}=0$ en $x=0,L$, es decir, sin momento flector), que es lo que permite seguir usando los mismos modos. **Separación de variables:** con $u=X(x)\,q(t)$,
$$\frac{q''}{q}=\frac{c^2X''-\kappa X''''}{X}=-\omega^2\quad\Longrightarrow\quad c^2X''-\kappa X''''=-\omega^2X,\qquad X(0)=X(L)=X''(0)=X''(L)=0.$$
Como $X=\sin(k\pi x/L)$ cumple las cuatro condiciones y tiene $X''=-\lambda_kX$, $X''''=\lambda_k^2X$ con $\lambda_k=(k\pi/L)^2$, la ecuación da $\omega^2=c^2\lambda_k+\kappa\lambda_k^2=c^2\lambda_k\bigl(1+\tfrac{\kappa}{c^2}\lambda_k\bigr)$:
$$\boxed{f_k=k\,f_0\sqrt{1+B\,k^2},\qquad f_0=\frac{c}{2L},\qquad B=\frac{\kappa\pi^2}{c^2L^2}=\frac{\pi^2EI}{TL^2}}$$
(el $\beta$ del Ejercicio de amortiguamiento). Los **modos siguen siendo los mismos senos**, pero las frecuencias ya no son múltiplos enteros: los armónicos se corren **hacia arriba**, y más cuanto mayor es $k$. Notá que $f_1=f_0\sqrt{1+B}$ no es exactamente $f_0$, pero como $B\sim10^{-4}$ la diferencia es de una parte en $10^4$ (unos $0.01$ Hz).

**Qué predice para el corrimiento relativo.** Con $B\ll1$ y $\delta_k=(f_k-kf_1)/(kf_1)$,
$$\delta_k=\sqrt{\frac{1+Bk^2}{1+B}}-1\approx\frac B2\,(k^2-1),$$
es decir, crece como $k^2$ (y vale $0$ en $k=1$ por definición).

### Cómo ajustar $B$

Los datos son las frecuencias $f_k$ medidas para $k=1,\dots,K$, y el error de medición está en $f_k$. Hay dos maneras de estimar $f_0$ y $B$ por mínimos cuadrados:

* **Linealizada.** $(f_k/k)^2=f_0^2+f_0^2B\,k^2$ es una **recta** en la variable $k^2$: $y_k=\alpha+\beta k^2$ con $\alpha=f_0^2$, $\beta=f_0^2B$. Se ajusta con `np.polyfit(k**2, (fk/k)**2, 1)` y $B=\beta/\alpha$, $f_0=\sqrt\alpha$. Es rápida y sirve de control, pero minimiza el error en una cantidad transformada (los errores de $(f_k/k)^2$ no tienen la misma varianza para todos los $k$).
* **No lineal.** `curve_fit(lambda k, f0, B: k*f0*np.sqrt(1+B*k**2), k, fk, p0=[f1, 1e-4])` minimiza directamente $\sum_k\bigl(f_k-kf_0\sqrt{1+Bk^2}\bigr)^2$. Devuelve la matriz de covarianza de los parámetros (aproximada, del jacobiano en el óptimo): su diagonal da los **errores estándar** $\text{se}(f_0)$ y $\text{se}(B)$, con los que informás $B\pm\text{se}$. Esa incertidumbre supone que los residuos son independientes y de igual varianza; miralos (¿tienen estructura?) antes de creerla.

### ¿Es razonable el valor de $B$?

Para una cuerda de acero liso de diámetro $d$, $I=\pi d^4/64$, $\kappa=EI/\rho_{lin}$ y $T=\rho_{lin}c^2$ con $\rho_{lin}=\rho_{ac}\pi d^2/4$ ($\rho_{ac}=7850$ kg/m$^3$, $E=2\times10^{11}$ Pa), y se obtiene $B=\pi^2EI/(TL^2)=\dfrac{\pi^2E\,d^2}{16\,\rho_{ac}\,c^2L^2}$. Con $L=0.65$ m y $f_1=196$ Hz ($c=2Lf_1\approx255$ m/s) y una cuerda de $0.4$ mm sale $B\approx10^{-4}$: el orden de magnitud típico de las cuerdas de guitarra es $10^{-4}$–$10^{-3}$ (más grande para cuerdas gruesas o cortas y tensas, como las graves del piano). $B$ es adimensional y depende de $d^2$, y no de la tensión ni de la longitud por separado más que a través de $c$ y $L$: al pisar un traste (acortar $L$ a $L'$, manteniendo $T$) $B$ crece como $1/L'^2$, y la inarmonicidad aumenta.
""")

lab.tarea(
    titulo="Corrimiento relativo y ajuste de $B$",
    consigna=r"""
Con `picos_a` y `f1a` de la Tarea 2 (primer medio segundo):

1. Calculá `delta`: el corrimiento relativo $\delta_k=(f_k-kf_1)/(kf_1)$ y graficalo contra $k$ junto con la curva $\frac B2(k^2-1)$ del ajuste de abajo. Estimá, además, la **potencia** $p$ de $|\delta_k|\propto k^p$ con un ajuste lineal en escala log-log para $k\ge3$: ¿da algo cerca de $2$?
2. Ajustá $B$ y $f_0$ de las dos maneras (linealizada: `B_lin, f0_lin`; no lineal con `curve_fit`: `B_nl, f0_nl, se_B`). Graficá también la recta $(f_k/k)^2$ contra $k^2$.
3. Miralo con crítica: ¿qué pasa con los residuos $f_k-kf_0\sqrt{1+Bk^2}$ (¿hay algún $k$ que se aparte mucho?, ¿por qué?)? ¿Cambia $B$ (y su incertidumbre) si sacás el $k=1$ (cuya frecuencia se movió por el efecto de la Tarea 2), o el armónico que más se aparta? ¿Y si usás el otro tramo (`picos_b`), con todos sus picos y sólo con $k\le8$?
4. Con tu $B$ y $c=2Lf_1$ ($L=0.65$ m), calculá el diámetro $d$ de una cuerda de acero liso que daría ese $B$ con la fórmula de arriba, y compará con las cuerdas comerciales (una Sol de guitarra eléctrica ronda los $0.43$ mm). Calculá también el corrimiento en **cents** ($1200\log_2(1+\delta)$) del armónico $k=10$: ¿se oiría?

**Qué se espera.** $\delta_k$ crece con $k$ y es positivo: de $\sim0.06$ % en $k=2$ a $\sim1.5$ % en $k=15$. La potencia da cerca de $2$ (entre $1.6$ y $2.3$ según el rango). $B\approx1.4$–$1.5\times10^{-4}$ con las dos formas de ajuste (dentro de $\sim10$ % entre sí; el error estándar del ajuste no lineal es de $\sim0.1\times10^{-4}$): positivo, chico y del orden que se espera. El armónico $k=9$ se aparta del resto ($\sim6$ Hz por debajo del modelo, contra $\lesssim1.5$ Hz los demás): su pico es un **doblete** (dos picos a unos $6$ Hz, por las dos polarizaciones de la vibración de la cuerda) y la interpolación cae en uno de ellos. Sin él, $B=1.40\times10^{-4}$ con un error cuatro veces menor: **un solo dato malo infla la incertidumbre**. Sin $k=1$, $B$ casi no cambia. Con el tramo largo y todos sus picos, $B$ sale $\approx0$ (o negativo): los armónicos $k\ge9$ de ese tramo están enterrados en el ruido y arruinan el ajuste; con $k\le8$, en cambio, da $\approx1.2\times10^{-4}$, consistente. El orden de magnitud $10^{-4}$ es robusto, el segundo dígito no. Informá $B$ con su tramo, los $k$ usados y su incertidumbre, y escribí en una línea cuánto confiás.
""",
    esqueleto=r'''
k_a, fk_a = picos_a[:, 0], picos_a[:, 1]

delta = ...        # TODO: corrimiento relativo (f_k - k f1) / (k f1)
p_delta = ...      # TODO: pendiente log-log de |delta| vs k, para k >= 3

# TODO: ajuste linealizado: (f_k/k)^2 = alfa + beta k^2  ->  f0_lin, B_lin
f0_lin, B_lin = ..., ...
# TODO: ajuste no lineal con curve_fit  ->  f0_nl, B_nl y su error estándar se_B
f0_nl, B_nl, se_B = ..., ..., ...

# TODO: figura (delta vs k con la curva ajustada; (f_k/k)^2 vs k^2 con la recta)
# TODO: residuos; B sin k=1; B con el otro tramo (picos_b, f1b)
# TODO: diámetro de la cuerda de acero que daría ese B; corrimiento de k = 10 en cents
''',
    solucion=r'''
k_a, fk_a = picos_a[:, 0], picos_a[:, 1]

delta = (fk_a - k_a * f1a) / (k_a * f1a)
m3 = k_a >= 3
p_delta = np.polyfit(np.log(k_a[m3]), np.log(np.abs(delta[m3])), 1)[0]

def ajustar_B(k, fk, f1):
    """Devuelve (f0_lin, B_lin, f0_nl, B_nl, se_B) con los dos ajustes."""
    beta, alfa = np.polyfit(k ** 2, (fk / k) ** 2, 1)
    modelo = lambda k, f0, B: k * f0 * np.sqrt(1 + B * k ** 2)
    p, cov = curve_fit(modelo, k, fk, p0=[f1, 1e-4])
    return np.sqrt(alfa), beta / alfa, p[0], p[1], np.sqrt(cov[1, 1])

f0_lin, B_lin, f0_nl, B_nl, se_B = ajustar_B(k_a, fk_a, f1a)
print(f"potencia de |delta_k| ~ k^p (k >= 3): p = {p_delta:.2f}")
print(f"linealizado:  f0 = {f0_lin:.2f} Hz, B = {B_lin:.2e}")
print(f"no lineal:    f0 = {f0_nl:.2f} Hz, B = {B_nl:.2e} +- {se_B:.1e}")

res = fk_a - k_a * f0_nl * np.sqrt(1 + B_nl * k_a ** 2)
print("residuos (Hz):", np.round(res, 2), f" -> el mayor en k = {int(k_a[np.argmax(np.abs(res))])}")
print(f"sin k = 1:           B = {ajustar_B(k_a[1:], fk_a[1:], f1a)[3]:.2e}")
sin9 = k_a != 9
_, _, _, B_sin9, se_sin9 = ajustar_B(k_a[sin9], fk_a[sin9], f1a)
print(f"sin k = 9 (doblete): B = {B_sin9:.2e} +- {se_sin9:.0e}")
print(f"tramo [0.5, 2.5] s:  B = {ajustar_B(picos_b[:, 0], picos_b[:, 1], f1b)[3]:.2e} (con {len(picos_b)} picos)")
print(f"tramo [0.5, 2.5] s, k <= 8: B = {ajustar_B(picos_b[picos_b[:, 0] <= 8, 0], picos_b[picos_b[:, 0] <= 8, 1], f1b)[3]:.2e}")

L_g = 0.65; c_g = 2 * L_g * f1a; rho_ac, E_ac = 7850.0, 2e11
d_ac = 4 * c_g * L_g / np.pi * np.sqrt(rho_ac * B_nl / E_ac)
print(f"cuerda de acero liso equivalente: d = {1e3 * d_ac:.2f} mm (c = {c_g:.0f} m/s)")
print(f"corrimiento de k = 10: {100 * (np.sqrt((1 + B_nl * 100) / (1 + B_nl)) - 1):.2f} % = {1200 * np.log2(np.sqrt((1 + B_nl * 100) / (1 + B_nl))):.1f} cents")

kk = np.linspace(1, 15.5, 200)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(k_a, 100 * delta, "o", color=COLORES["dato"], label="medido")
axs[0].plot(kk, 100 * (np.sqrt((1 + B_nl * kk ** 2) / (1 + B_nl)) - 1), color=COLORES["modelo"], label=rf"$B={B_nl:.2e}$")
axs[0].axhline(0, color="0.7", lw=0.8)
axs[0].set_xlabel("$k$"); axs[0].set_ylabel(r"$(f_k-kf_1)/(kf_1)$ (%)"); axs[0].legend(loc="upper left")
axs[1].plot(k_a ** 2, (fk_a / k_a) ** 2, "o", color=COLORES["dato"], label="medido")
axs[1].plot(k_a ** 2, f0_lin ** 2 * (1 + B_lin * k_a ** 2), color=COLORES["modelo"], label="recta ajustada")
axs[1].set_xlabel("$k^2$"); axs[1].set_ylabel("$(f_k/k)^2$ (Hz$^2$)"); axs[1].legend(loc="upper left")
fig.tight_layout()
''',
    verificacion=r'''
# Verificación
assert np.all(delta[1:] > -0.002) and delta[-1] > 0.008, "los armónicos altos deberían correrse hacia arriba (~1 % en k = 15)"
assert 1.4 < p_delta < 2.6, "el corrimiento debería crecer casi como k^2"
assert 0 < B_lin < 1e-3 and 0 < B_nl < 1e-3, "B debería ser positivo y chico (1e-4 a 1e-3)"
assert 0.5 < B_lin / B_nl < 2 and 0 < se_B < B_nl, "los dos ajustes deberían coincidir en el orden de magnitud"
assert abs(f0_nl - 195.8) < 1.0, "f0 debería ser cercana a f1"
print("inarmonicidad: OK")
''')
figura_revision("inarmonicidad")

# =============================================================================
# 4. Decaimiento por armónico
# =============================================================================
lab.md(r"""
## 4. Cuánto tarda en apagarse cada armónico

### Dos modelos de fricción

Para que el modelo tenga pérdidas hay que agregarle un término disipativo. Vamos a comparar dos.

**(A) Fricción con el aire, igual para todos.** $u_{tt}+2\gamma u_t=c^2u_{xx}$. Con $u=X(x)q(t)$ y los mismos modos, cada coeficiente cumple $q_k''+2\gamma q_k'+\omega_k^2q_k=0$, cuya solución es
$$q_k(t)=e^{-\gamma t}\cos(\tilde\omega_kt-\phi_k),\qquad\tilde\omega_k=\sqrt{\omega_k^2-\gamma^2}.$$
**Todos los modos decaen con la misma tasa $\gamma$.** La frecuencia casi no cambia: $\gamma\sim1$–$10$ s$^{-1}$ contra $\omega_1\approx1200$ s$^{-1}$, es decir, un cambio relativo de $\gamma^2/(2\omega_k^2)\lesssim10^{-5}$. (Por eso los picos siguen en el mismo lugar y sólo cambia su ancho.)

**(B) Fricción que crece con la frecuencia (viscoelástica).** $u_{tt}+2\gamma u_t-\nu\,u_{xxt}=c^2u_{xx}$ con $\nu>0$ (en el texto aparece como $+\nu u_{xxt}$ a la derecha; es lo mismo). El nuevo término es una fricción proporcional a la velocidad **de la curvatura**: frena más lo que se dobla más rápido, es decir, las oscilaciones de longitud de onda corta; físicamente es la disipación interna del material. Con $X=\sin(k\pi x/L)$, $X''=-\lambda_kX$ ($\lambda_k=(k\pi/L)^2$), el modo cumple $q_k''+(2\gamma+\nu\lambda_k)q_k'+\omega_k^2q_k=0$ y por lo tanto
$$q_k\propto e^{-\gamma_kt},\qquad\boxed{\gamma_k=\gamma+\frac\nu2\lambda_k=\gamma+b\,k^2},\qquad b=\frac{\nu\pi^2}{2L^2}.$$
El modelo (A) es el caso $b=0$. Ambos tienen *dos* parámetros a lo sumo, pero sólo $\gamma$ y $b$ son **identificables** con datos de audio; $\nu$ se obtiene de $b$ si se conoce $L$.

Como la energía de un modo es proporcional al cuadrado de su amplitud, $E_k\propto e^{-2\gamma_kt}$.

### Cómo medir $\gamma_k$

El espectrograma (Parte II) es la DFT de tramos sucesivos: para cada instante $t_i$ da el espectro de un tramo corto centrado en $t_i$. Elegimos tramos de $50$ ms con ventana de Hann y $50$ % de solapamiento (un tramo cada $25$ ms). El compromiso es el mismo de la Sección 2: la resolución en frecuencia es $\Delta\xi=1/T=20$ Hz (lóbulo de $\pm40$ Hz), mucho menor que los $196$ Hz entre armónicos, y la resolución temporal es de $50$ ms, mucho menor que los tiempos de decaimiento (del orden de $0.1$ a $1$ s). Para cada armónico $k$ definimos su amplitud instantánea como la raíz de la potencia sumada en una banda de $\pm50$ Hz alrededor de $f_k$ (sumar sobre la banda, en lugar de tomar el máximo, evita los saltos cuando el pico se parte en un doblete). Si $A_k(t)\propto e^{-\gamma_kt}$, entonces $\ln A_k(t)$ es una **recta** de pendiente $-\gamma_k$, y $\gamma_k$ sale de un ajuste lineal.

**Qué hay que cuidar en el ajuste.**

* **El ataque.** Los primeros $\sim0.3$ s tienen transitorios de la pulsación; no se ajustan.
* **El piso de ruido.** Cuando $A_k(t)$ llega al ruido de fondo, $\ln A_k$ deja de bajar y se aplana: si se incluye esa parte, $\gamma_k$ sale subestimada. Se ajusta hasta el primer instante en que $A_k$ cae por debajo de `umbral` veces el piso (medido en una banda vacía, $6$–$9$ kHz, con $t>3$ s para evitar el ataque).
* **El decaimiento no es exactamente exponencial** (dos etapas, batidos entre las dos polarizaciones): el valor de $\gamma_k$ depende algo de la ventana temporal de ajuste. Hay que **medir esa dependencia** (cambiá la ventana y mirá cuánto se mueve) y mirarla como parte de la incertidumbre.
* **La incertidumbre del ajuste** (de `np.polyfit(..., cov=True)`) es *optimista*: los residuos no son independientes (tienen ondulaciones lentas), por eso subestima el error real.
* **No hay que usar los armónicos que quedaron enterrados en el ruido:** sólo los que tengan suficientes puntos de ajuste.
""")

lab.tarea(
    titulo="Espectrograma y tasas de decaimiento por armónico",
    consigna=r"""
Escribí `tasas_decaimiento(x, fs, fk, ta=0.3, tb=2.0, tramo_ms=50, semiancho=50.0, umbral=4.0)`, que recibe las frecuencias `fk` de los armónicos ($k=1,2,\dots$) y devuelve un arreglo con columnas `(k, gamma_k, error, npuntos)`:

* construye el espectrograma con `espectro.espectrograma(x, 1/fs, tramo, 0.5, "hann")` (devuelve `(t_centros, xi, S)` con `S[i, j]` la potencia en la frecuencia `xi[i]` y el tramo `j`);
* estima el piso de potencia por bin como la mediana de `S` entre $6$ y $9$ kHz para $t>3$ s;
* para cada $k$: $A_k(t)=\sqrt{\sum_{|\xi-f_k|<\text{semiancho}}S}$; ajusta $\ln A_k$ contra $t$ en $t_a\le t\le t_b$, cortando en el primer instante con $A_k<\text{umbral}\cdot\text{piso}_k$ (donde $\text{piso}_k$ es el piso de amplitud de esa banda); si quedan menos de $6$ puntos, devuelve `nan`.

Con eso:

1. Graficá el espectrograma (en dB, hasta $4$ kHz) y, en otro panel, $\ln A_k(t)$ para $k=1,2,4,8$ con sus rectas de ajuste.
2. Armá la tabla de `gamma_k` para los armónicos medidos en la Tarea 2 (`picos_a[:, 1]`, así los $k$ coinciden), descartando los que tienen menos de $30$ puntos, y graficá $\gamma_k$ contra $k$ (con barras de error, y marcando la sensibilidad a la ventana: repetí con $t_b=1.5$).
3. Ajustá los dos modelos con mínimos cuadrados (sin pesos): **(A)** $\gamma_k=\gamma$ (constante) y **(B)** $\gamma_k=\gamma+bk^2$. Informá los parámetros, el desvío de los residuos de cada uno, y superponé ambos en el gráfico. Calculá $\nu=2bL^2/\pi^2$ con $L=0.65$ m.

**Qué se espera.** $\gamma_k$ **crece con $k$**: de $\sim1.4$ s$^{-1}$ en el fundamental a $\sim5$–$6$ s$^{-1}$ hacia $k=10$–$12$ (el armónico 12 tiene sólo $\sim30$ puntos: se apaga en menos de un segundo). Hay dispersión (los armónicos $5$ y $8$, por ejemplo, salen más bajos que sus vecinos: son armónicos con batidos fuertes), la sensibilidad a la ventana es de unas décimas de s$^{-1}$ en los primeros armónicos, y el error estándar del ajuste subestima esa dispersión. El modelo (A), con una sola tasa ($\gamma\approx3.4$ s$^{-1}$), no puede reproducir un factor 4 entre el primero y el último: sus residuos ($\sim1.6$ s$^{-1}$) son unas tres veces los de (B) ($\sim0.5$ s$^{-1}$, con $\gamma\approx1.6$ s$^{-1}$ y $b\approx0.03$ s$^{-1}$, es decir $\nu\approx3\times10^{-3}$ m$^2$/s). Como extra, el ajuste con una ley **lineal** $\gamma+bk$ (que no hemos deducido de ningún modelo) da residuos **iguales** a los de (B): con $k\le12$ los datos no distinguen una ley lineal de una cuadrática. Decí qué implica esto sobre lo que se puede afirmar del mecanismo de pérdida.
""",
    esqueleto=r'''
def tasas_decaimiento(x, fs, fk, ta=0.3, tb=2.0, tramo_ms=50, semiancho=50.0, umbral=4.0):
    """Tasas gamma_k de decaimiento de la amplitud de cada armónico.
    Devuelve un arreglo con columnas (k, gamma_k, error, npuntos)."""
    # TODO: espectrograma; piso; para cada k: A_k(t), ventana de ajuste, polyfit de ln A_k
    pass

G = tasas_decaimiento(x, fs, picos_a[:, 1])
ok = ...   # TODO: máscara de los armónicos con al menos 30 puntos
k_g, gam_k, se_g = ..., ..., ...    # TODO

# TODO: modelo (A) constante y modelo (B) gamma0 + b k^2 (mínimos cuadrados sin pesos)
gam_A = ...                       # TODO
gam0_B, b_B = ..., ...            # TODO
L = 0.65
nu = ...                          # TODO: 2 b L^2 / pi^2

# TODO: espectrograma en dB; ln A_k(t) con sus rectas; gamma_k vs k con los dos modelos
''',
    solucion=r'''
def tasas_decaimiento(x, fs, fk, ta=0.3, tb=2.0, tramo_ms=50, semiancho=50.0, umbral=4.0):
    """Tasas gamma_k de decaimiento de la amplitud de cada armónico.
    Devuelve un arreglo con columnas (k, gamma_k, error, npuntos)."""
    tc, xi, S = espectro.espectrograma(x, 1 / fs, int(tramo_ms * 1e-3 * fs), 0.5, "hann")
    piso_bin = np.median(S[(xi > 6000) & (xi < 9000)][:, tc > 3])
    salida = []
    for k, f in enumerate(fk, 1):
        banda = np.abs(xi - f) < semiancho
        Ak = np.sqrt(S[banda].sum(axis=0))
        piso = np.sqrt(piso_bin * banda.sum())
        ok = (tc >= ta) & (tc <= tb)
        bajo = np.nonzero((Ak < umbral * piso) & (tc > ta))[0]
        if len(bajo):
            ok &= tc < tc[bajo[0]]
        if ok.sum() < 6:
            salida.append((k, np.nan, np.nan, ok.sum())); continue
        p, cov = np.polyfit(tc[ok], np.log(Ak[ok]), 1, cov=True)
        salida.append((k, -p[0], np.sqrt(cov[0, 0]), ok.sum()))
    return np.array(salida)

G = tasas_decaimiento(x, fs, picos_a[:, 1])
ok = (G[:, 3] >= 30) & ~np.isnan(G[:, 1])
k_g, gam_k, se_g = G[ok, 0], G[ok, 1], G[ok, 2]
G15 = tasas_decaimiento(x, fs, picos_a[:, 1], tb=1.5)
gam_k15 = G15[ok, 1]

print("  k   gamma_k (1/s)   error   npuntos   gamma_k con tb=1.5")
for fila, g15 in zip(G[ok], gam_k15):
    print(f"{int(fila[0]):3d}   {fila[1]:8.2f}     {fila[2]:.3f}   {int(fila[3]):5d}       {g15:.2f}")

# modelo (A): constante;  modelo (B): gamma0 + b k^2
gam_A = gam_k.mean()
MB = np.c_[np.ones_like(k_g), k_g ** 2]
gam0_B, b_B = np.linalg.lstsq(MB, gam_k, rcond=None)[0]
MC = np.c_[np.ones_like(k_g), k_g]
gam0_C, b_C = np.linalg.lstsq(MC, gam_k, rcond=None)[0]
L = 0.65
nu = 2 * b_B * L ** 2 / np.pi ** 2
rms = lambda r: np.sqrt(np.mean(r ** 2))
print(f"\n(A) gamma constante:      gamma = {gam_A:.2f} 1/s;   desvío de los residuos {rms(gam_k - gam_A):.2f}")
print(f"(B) gamma + b k^2:        gamma = {gam0_B:.2f} 1/s, b = {b_B:.4f} 1/s;   desvío {rms(gam_k - MB @ [gam0_B, b_B]):.2f};   nu = {nu:.2e} m^2/s (con L = {L} m)")
print(f"(C) gamma + b k (extra):  gamma = {gam0_C:.2f} 1/s, b = {b_C:.3f} 1/s;   desvío {rms(gam_k - MC @ [gam0_C, b_C]):.2f}")

tc_, xi_, S_ = espectro.espectrograma(x, 1 / fs, int(0.05 * fs), 0.5, "hann")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
mf = xi_ < 4000
im = axs[0].imshow(10 * np.log10(S_[mf] + 1e-14), origin="lower", aspect="auto", extent=[tc_[0], tc_[-1], 0, xi_[mf][-1]], vmin=-90, cmap="viridis")
axs[0].set_xlabel("$t$ (s)"); axs[0].set_ylabel(r"$\xi$ (Hz)"); axs[0].set_title("espectrograma (dB)")
fig.colorbar(im, ax=axs[0], pad=0.02, label="dB")
pis = np.median(S_[(xi_ > 6000) & (xi_ < 9000)][:, tc_ > 3])
for k, col in zip([1, 2, 4, 8], CICLO):
    f = picos_a[k - 1, 1]; banda = np.abs(xi_ - f) < 50; Ak = np.sqrt(S_[banda].sum(axis=0))
    axs[1].semilogy(tc_, Ak, color=col, lw=1, label=f"$k={k}$")
    ventana = (tc_ >= 0.3) & (tc_ <= 2.0)
    bajo = np.nonzero((Ak < 4 * np.sqrt(pis * banda.sum())) & (tc_ > 0.3))[0]
    if len(bajo):
        ventana &= tc_ < tc_[bajo[0]]
    p_ = np.polyfit(tc_[ventana], np.log(Ak[ventana]), 1)
    axs[1].semilogy(tc_[ventana], np.exp(np.polyval(p_, tc_[ventana])), "--", color=col, lw=2)
axs[1].set_xlabel("$t$ (s)"); axs[1].set_ylabel("$A_k(t)$ (banda de 100 Hz)"); axs[1].set_xlim(0, 3.5); axs[1].set_ylim(1e-5, 5); axs[1].legend(loc="upper right")
axs[1].set_title("decaimiento y ajuste")
kk = np.linspace(0.5, 12.5, 100)
axs[2].errorbar(k_g, gam_k, yerr=se_g, fmt="o", color=COLORES["dato"], ms=4, capsize=2, label=r"medido ($t_b=2$ s)")
axs[2].plot(k_g, gam_k15, "x", color=COLORES["dato"], ms=6, label=r"medido ($t_b=1.5$ s)")
axs[2].axhline(gam_A, color=CICLO[2], label=f"(A) constante: {gam_A:.1f}")
axs[2].plot(kk, gam0_B + b_B * kk ** 2, color=COLORES["modelo"], label=rf"(B) ${gam0_B:.1f}+{b_B:.3f}\,k^2$")
axs[2].set_xlabel("$k$"); axs[2].set_ylabel(r"$\gamma_k$ (1/s)"); axs[2].legend(loc="upper left", fontsize=8); axs[2].set_ylim(0, 8)
fig.tight_layout()
''',
    verificacion=r'''
# Verificación
assert ok.sum() >= 9 and np.all(gam_k > 0), "deberías tener al menos 9 armónicos con tasas positivas"
assert np.corrcoef(k_g, gam_k)[0, 1] > 0.7, "gamma_k debería crecer con k"
assert gam_k[k_g >= 8].mean() > 2 * gam_k[k_g <= 2].mean(), "los armónicos altos deberían apagarse bastante más rápido"
assert gam0_B > 0 and b_B > 0 and rms(gam_k - MB @ [gam0_B, b_B]) < 0.8 * rms(gam_k - gam_A), "el modelo (B) debería ajustar mejor que el (A)"
print("decaimiento: OK")
''')
figura_revision("decaimiento")

# =============================================================================
# 5. Timbre
# =============================================================================
lab.md(r"""
## 5. El timbre: el reparto de energía entre los modos

Para la cuerda pulsada ideal, con el vértice en $x_0=\theta L$ (con $0<\theta<1$), las amplitudes de los modos son
$$A_k=\frac{2a}{\pi^2\theta(1-\theta)}\,\frac{\sin(k\pi\theta)}{k^2},$$
(la fórmula de la Sección 1 con $x_0=\theta L$). Dos propiedades para tener presentes: (i) los $A_k$ **normalizados** por $A_1$ no dependen de $a$, y (ii) se anulan los $k$ con $k\theta\in\mathbb Z$: para $\theta=\frac12$ faltan los armónicos pares; para $\theta=\frac15$ faltan $k=5,10,15,\dots$; y para $\theta=\frac1{20}$ el primer cero está en $k=20$, así que casi ninguno falta y los $A_k$ decaen como $1/k^2$ desde $k=1$ hasta $\sim20$.

La energía de cada modo es $E_k=\frac L4\omega_k^2A_k^2\propto k^2A_k^2\propto\sin^2(k\pi\theta)/k^2$: **decae como $1/k^2$**, más lentamente que la de $A_k^2$ (que decae como $1/k^4$). El *timbre* es esa distribución de energía entre los modos.

### Cómo comparar dos espectros

Si $\mathbf a$ y $\mathbf m$ son los vectores de amplitudes medidas y del modelo, normalizados por su primer elemento ($k=1$) para que no importe la escala (ni el volumen de la grabación, ni la altura $a$ del pulsado), hay muchas maneras de medir su parecido, y **dan resultados distintos**:

* la **correlación** (o similitud coseno) $\cos\varphi=\frac{\mathbf a\cdot\mathbf m}{\|\mathbf a\|\|\mathbf m\|}$, y la distancia euclídea entre los vectores normalizados a norma $1$: son medidas **lineales**, dominadas por los $k$ de mayor amplitud (el fundamental y los primeros);
* la **distancia en decibeles**, $d_{dB}=\sqrt{\frac1K\sum_k\bigl(20\log_{10}(a_k/a_1)-20\log_{10}(m_k/m_1)\bigr)^2}$ (el desvío cuadrático medio de la diferencia de los espectros en dB), que pesa igual un error de un factor $2$ en un armónico de amplitud $10^{-2}$ que en uno de amplitud $1$. Es más cercana a cómo se oye un timbre (la audición es logarítmica), pero explota cuando el modelo tiene un cero exacto: hay que fijar un **piso** (no dejar que $a_k/a_1$ ni $m_k/m_1$ bajen de, por ejemplo, $10^{-2}$, es decir, $-40$ dB). El piso es una decisión de modelado que **puede cambiar el resultado**: lo vas a comprobar.

Comparamos con $k=1,\dots,12$ (los armónicos de la grabación bien por encima del ruido).
""")

lab.tarea(
    titulo="Timbre: la grabación contra la cuerda ideal pulsada en $x_0=L/2$, $L/5$ y $L/20$",
    consigna=r"""
Con las amplitudes $A_k$ de `picos_a[:, 2]` (primer medio segundo):

1. Escribí `A_ideal(k, theta)`: las amplitudes de los modos de la cuerda pulsada en $x_0=\theta L$, normalizadas de modo que $A_1=1$ (valor absoluto de la fórmula de arriba dividido por su valor en $k=1$). Verificá **la fórmula** contra una cuenta numérica: los coeficientes de Fourier seno del triángulo (por ejemplo con la suma de Riemann `np.sum(g * np.sin(k*pi*x/L)) * 2/N` sobre una grilla fina) para $\theta=0.2$ y $k=1,\dots,6$.
2. Calculá la **fracción de energía** $E_k/\sum_jE_j$ en los primeros modos ($k=1,2,3,4$) para $\theta=\frac12,\frac15,\frac1{20}$ (sumando hasta $k=2000$) y comprobá que la del fundamental es $8/\pi^2\approx0.81$ en el medio.
3. Armá `a_med` (las amplitudes medidas para $k=1..12$, normalizadas a $a_1=1$) y, para los tres valores de $\theta$, calculá la **similitud coseno** y la **distancia en dB** con dos pisos: $-40$ dB (`piso=1e-2`, el valor por defecto) y $-60$ dB (`piso=1e-3`), contra `a_med`. Graficá, en escala log, `a_med` contra los tres modelos.
4. Barré $\theta\in[0.02,0.5]$ y graficá las dos medidas (con piso $-40$ dB) en función de $\theta$. ¿Cuál es el $x_0$ que "se parece más" a la grabación? Elegí `theta_elegido` (lo vas a usar en la simulación; por razones prácticas conviene que $1/\theta$ no sea un entero $\le15$: si no, el modelo tiene ceros exactos en algún $k\le15$ y la Tarea 6 se complica; por ejemplo, $\theta=0.15$).

**Qué se espera.** La grabación tiene el fundamental dominante y los armónicos $2$ a $6$ entre $-16$ y $-25$ dB, con caídas irregulares, y llega a $-48$ dB en $k=12$. Los armónicos **pares** están presentes ($k=2$ y $k=4$ a $-16$ dB): eso descarta a $\theta=\frac12$ **cualitativamente**, aunque la similitud coseno (dominada por el fundamental) lo favorezca ($0.975$ contra $0.964$ y $0.927$). Con piso $-40$ dB, la distancia en dB da $\approx11$ dB para $\theta=\frac12$, $\approx7$ para $\frac15$ y $\approx11$ para $\frac1{20}$: gana $\theta=\frac15$. Con piso $-60$ dB el orden **cambia** ($23$, $14$ y $12$ dB: $\frac1{20}$ gana), porque los ceros exactos del modelo cuestan mucho más. Hay que entender por qué: $\theta=\frac1{20}$ predice un espectro demasiado parejo, con los armónicos altos mucho más fuertes que los de la grabación ($-27$ dB contra $-48$ dB en $k=12$), pero no tiene ceros; $\theta=\frac15$ tiene ceros en $k=5,10$ que la grabación no muestra (aunque $k=5$ es un valle: $-25$ dB). El barrido muestra que la similitud coseno tiene su mínimo de distancia cerca de $\theta\approx0.41$ y la distancia en dB (con $-40$ dB) cerca de $\theta\approx0.14$, con muchas irregularidades: **las dos medidas no coinciden**, y la primera casi no ve los armónicos pequeños. Ninguna cuerda ideal reproduce la grabación en detalle: el micrófono ve la respuesta del cuerpo de la guitarra, la pulsación real no es un triángulo perfecto (la púa o el dedo tienen ancho y velocidad), y la vibración tiene dos polarizaciones. Anotá tu elección y por qué.
""",
    esqueleto=r'''
def A_ideal(k, theta):
    """Amplitudes A_k de la cuerda pulsada en x0 = theta L, normalizadas con A_1 = 1 (valor absoluto)."""
    pass  # TODO

K_cmp = 12
ks = np.arange(1, K_cmp + 1)
a_med = ...            # TODO: picos_a[:K_cmp, 2] / picos_a[0, 2]

def sim_coseno(a, m):
    pass  # TODO: similitud coseno entre a y m

def dist_dB(a, m, piso=1e-2):
    pass  # TODO: desvío cuadrático medio de la diferencia en dB (con piso en las amplitudes)

thetas = [0.5, 0.2, 0.05]
# TODO: verificación de A_ideal contra coeficientes de Fourier numéricos; fracciones de energía
# TODO: tabla de similitud y distancia para los tres theta; figura en log; barrido en theta
theta_elegido = ...    # TODO
''',
    solucion=r'''
def A_ideal(k, theta):
    """Amplitudes A_k de la cuerda pulsada en x0 = theta L, normalizadas con A_1 = 1 (valor absoluto)."""
    k = np.asarray(k, dtype=float)
    A = np.sin(k * np.pi * theta) / k ** 2
    return np.abs(A / np.sin(np.pi * theta))

K_cmp = 12
ks = np.arange(1, K_cmp + 1)
a_med = picos_a[:K_cmp, 2] / picos_a[0, 2]

def sim_coseno(a, m):
    return a @ m / (np.linalg.norm(a) * np.linalg.norm(m))

def dist_dB(a, m, piso=1e-2):
    return np.sqrt(np.mean((20 * np.log10(np.maximum(a, piso)) - 20 * np.log10(np.maximum(m, piso))) ** 2))

thetas = [0.5, 0.2, 0.05]

# (1) la fórmula contra los coeficientes de Fourier numéricos (theta = 0.2, L = 1, a = 1)
xg = np.linspace(0, 1, 20001)[:-1]
g_tri = np.where(xg <= 0.2, xg / 0.2, (1 - xg) / 0.8)
A_num = np.array([2 * np.mean(g_tri * np.sin(k * np.pi * xg)) for k in range(1, 7)])
A_for = 2 / (np.pi ** 2 * 0.2 * 0.8) * np.sin(np.arange(1, 7) * np.pi * 0.2) / np.arange(1, 7) ** 2
print("A_k numérico :", np.round(A_num, 5))
print("A_k fórmula  :", np.round(A_for, 5), f"  (máx. diferencia {np.abs(A_num - A_for).max():.1e})")

# (2) fracción de energía por modo:  E_k ~ k^2 A_k^2 ~ sin^2(k pi theta)/k^2
kk = np.arange(1, 2001)
print("\nfracción de la energía en los modos 1..4:")
for th in thetas:
    E = np.sin(kk * np.pi * th) ** 2 / kk ** 2
    print(f"  theta = {th:<5}: " + "  ".join(f"{100 * E[j] / E.sum():5.1f} %" for j in range(4)))
print(f"  (para theta = 1/2: 8/pi^2 = {8 / np.pi ** 2:.3f})")

# (3) comparación con la grabación
print("\n   theta    coseno   dist. dB (piso -40 dB)   dist. dB (piso -60 dB)")
for th in thetas:
    m = A_ideal(ks, th)
    print(f"  {th:6.2f}   {sim_coseno(a_med, m):.4f}          {dist_dB(a_med, m):6.2f}                  {dist_dB(a_med, m, 1e-3):6.2f}")

# (4) barrido
ths = np.linspace(0.02, 0.5, 481)
cos_th = np.array([sim_coseno(a_med, A_ideal(ks, th)) for th in ths])
dB_th = np.array([dist_dB(a_med, A_ideal(ks, th)) for th in ths])
print(f"\nbarrido: mejor coseno en theta = {ths[cos_th.argmax()]:.3f} ({cos_th.max():.4f});  mínimo de dB en theta = {ths[dB_th.argmin()]:.3f} ({dB_th.min():.2f} dB)")
print("dB medidos (k = 1..12):", np.round(20 * np.log10(a_med), 1))
theta_elegido = 0.15

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(ks, a_med, "ko-", label="grabación", lw=2)
for th, col in zip(thetas, CICLO[1:]):
    axs[0].semilogy(ks, np.maximum(A_ideal(ks, th), 1e-4), "s--", color=col, ms=4, label=rf"ideal, $x_0=L\cdot{th}$")
axs[0].set_xlabel("$k$"); axs[0].set_ylabel("$A_k/A_1$"); axs[0].set_ylim(1e-4, 2); axs[0].legend(fontsize=8, loc="lower left")
axs[1].plot(ths, 1 - cos_th, color=CICLO[0], label=r"$1-\cos\varphi$")
ax2 = axs[1].twinx()
ax2.plot(ths, dB_th, color=CICLO[1], label="distancia en dB")
axs[1].set_xlabel(r"$\theta=x_0/L$"); axs[1].set_ylabel(r"$1-$ coseno", color=CICLO[0]); ax2.set_ylabel("distancia (dB, piso $-40$ dB)", color=CICLO[1])
axs[1].set_yscale("log")
fig.tight_layout()
''',
    verificacion=r'''
# Verificación
assert np.allclose(A_ideal([1, 2, 3], 0.5), [1, 0, 1 / 9], atol=1e-12), "A_ideal(k, 1/2) = 1, 0, 1/9, ..."
assert abs(A_ideal([2], 0.2)[0] - np.sin(0.4 * np.pi) / np.sin(0.2 * np.pi) / 4) < 1e-12
assert a_med.shape == (12,) and a_med[0] == 1.0 and 0.05 < a_med[1] < 0.3
assert 0.8 < sim_coseno(a_med, A_ideal(ks, 0.2)) < 1.0
assert dist_dB(a_med, A_ideal(ks, 0.2)) < min(dist_dB(a_med, A_ideal(ks, 0.5)), dist_dB(a_med, A_ideal(ks, 0.05))), "con piso -40 dB, x0 = L/5 debería ganar entre los tres"
assert 0.02 <= theta_elegido <= 0.5 and not any(abs(theta_elegido * k - round(theta_elegido * k)) < 1e-9 for k in range(1, 16)), "theta_elegido: sin ceros exactos para k <= 15"

print("timbre: OK")
''')
figura_revision("timbre")

# =============================================================================
# 6. Simulación
# =============================================================================
lab.md(r"""
## 6. Simular la cuerda y "escucharla"

### El esquema

Resolvemos, con el esquema leapfrog del laboratorio de EDPs, la ecuación con los dos términos de fricción
$$u_{tt}=c^2u_{xx}-2\gamma\,u_t+\nu\,u_{xxt}\qquad(\text{y, en la Tarea 7, }-\kappa\,u_{xxxx}).$$
Con $h=\Delta x=L/N$, $k=\Delta t$ (¡no confundir con el índice de armónico!), $r=ck/h$ y $\delta^2v_j=v_{j+1}-2v_j+v_{j-1}$, las derivadas se aproximan por $u_{tt}\approx\frac{u^{n+1}-2u^n+u^{n-1}}{k^2}$, $u_{xx}\approx\frac{\delta^2u^n}{h^2}$, $u_t\approx\frac{u^n-u^{n-1}}{k}$ (diferencia hacia atrás: mantiene el esquema explícito) y $u_{xxt}\approx\frac{\delta^2(u^n-u^{n-1})}{h^2k}$. Despejando $u^{n+1}$:
$$u_j^{n+1}=2u_j^n-u_j^{n-1}+r^2\,\delta^2u^n_j-2\gamma k\,(u_j^n-u_j^{n-1})+\frac{\nu k}{h^2}\,\delta^2(u^n-u^{n-1})_j .$$
El **primer paso** es el de Taylor del laboratorio anterior (con velocidad inicial nula): $u^1=u^0+\frac{r^2}{2}\delta^2u^0$. Los términos de fricción usan diferencias hacia atrás, de orden 1 en $k$: eso introduce un error relativo de orden $\gamma k$ en las tasas de decaimiento, absolutamente despreciable acá ($\gamma k\sim10^{-5}$); la estabilidad del esquema sigue gobernada por $r\le1$ (los términos de fricción son chicos, $\frac{\nu k}{h^2}\ll1$: chequealo con tus números).

### El micrófono y la frecuencia de muestreo

Queremos una señal a $f_s=44100$ Hz **en un punto** $x_m$ de la cuerda (el "micrófono"; recordá que un punto sobre un nodo de un modo no lo registra: $\sin(k\pi x_m/L)=0$; conviene un $x_m/L$ irracional-ish, como $0.27$). Pero un paso de tiempo $k=1/f_s$ obligaría a $h\ge c/f_s\approx5.8$ mm, es decir, sólo $N\le112$ nodos, y con $N$ chico los modos altos se propagan con la velocidad equivocada. En la ecuación discretizada, el modo $k$ tiene frecuencia numérica $f_k^{num}=\frac{1}{\pi k_t}\arcsin\bigl(r\sin\frac{k\pi}{2N}\bigr)$ y el error relativo respecto de $kf_1$ es $\approx-\frac{(1-r^2)}{24}\bigl(\frac{k\pi}{N}\bigr)^2$: con $N=400$ y $r\approx0.9$ es de $10^{-4}$ ($0.01$ %) en $k=20$ y de $4\times10^{-4}$ en $k=40$; con $N=100$ sería 16 veces más. Para que hayan $\ge20$ modos bien resueltos elegimos $N=400$ y **avanzamos con un paso $k=1/(mf_s)$, con $m=4$ pasos por muestra**, guardando el valor del punto $x_m$ cada $m$ pasos. El costo es proporcional a $N\cdot m\cdot f_sT$.

Todo lo que está por encima de la frecuencia de Nyquist ($22050$ Hz, es decir, $k\gtrsim112$) no debería estar en la señal muestreada: tomar una de cada $m$ muestras **sin filtrar** los pliega (*aliasing*) sobre las frecuencias audibles. Acá es despreciable porque los $A_k$ decaen como $1/k^2$ y, sobre todo, la fricción viscoelástica apaga esos modos en milisegundos ($\gamma_{112}\sim b\cdot112^2\gtrsim100$ s$^{-1}$): fijate en el gráfico que no hay basura visible arriba de los armónicos.

### Qué parámetros usar

$L=0.65$ m (la longitud de una guitarra; sólo importa a través de $c=2Lf_1$ y de la conversión $\nu=2bL^2/\pi^2$); $f_1$ medida en la Tarea 2; la altura $a$ del pulsado es irrelevante (el problema es lineal; sólo cambia el volumen) y se elige $a=1$ mm; $x_0=\theta L$ con el $\theta$ de la Tarea 5; $\gamma$ y $\nu$ del modelo (B) de la Tarea 4.
""")

lab.tarea(
    titulo="Simulación con fricción, el WAV y la comparación con la grabación",
    consigna=r"""
Escribí `cuerda_sim(N, m, theta, xm, gam0, nu, kappa=0.0, T=1.0, a=1e-3)` que avanza el esquema de arriba con paso $k=1/(mf_s)$ durante $T$ segundos, desde la cuerda pulsada en $x_0=\theta L$ (triángulo de altura $a$, velocidad inicial nula), y devuelve la señal `y` del punto $x_m=x_{m,rel}L$ ($x_m$ dado como fracción de $L$) muestreada a $f_s$ (un valor de cada $m$ pasos; `len(y) = int(T*fs)`). Usá `L`, `c = 2*L*f1` y `fs` como variables globales. Dejá el argumento `kappa` (rigidez) para la Tarea 7: por ahora es $0$ y no se usa.

1. **Comprobación sin fricción.** Con `gam0=nu=0`, $N=400$, $m=4$, $\theta=0.2$, $x_m=0.27$, compará $y(t)$ con la solución en serie ($\sum_kA_k\cos(2\pi kf_1t)\sin(k\pi x_m/L)$ con los $A_k$ del triángulo, $a=1$ mm) para $t\le0.05$ s. Calculá el número $r$ del esquema y el error máximo relativo.
2. **La simulación completa.** Con $\theta=$ `theta_elegido`, `gam0 = gam0_B`, `nu = nu` (los de la Tarea 4), $T=6$ s. Escribí `/tmp/...` (en `tempfile.gettempdir()`), no en el repositorio, el archivo `cuerda_sim.wav` con `wavfile.write(ruta, fs, señal_int16)`, escalando el pico a $0.9$ del fondo de escala. (Si estás en tu máquina, abrilo y escuchalo: ¿se parece a la guitarra? ¿Qué le falta al oído?)
3. **Comparación con lo real.** Usando las funciones de las Tareas 2 y 4 sobre `y_sim`: (a) el espectro del primer medio segundo, superpuesto con el real (ambos normalizados a la altura del fundamental), hasta $3.5$ kHz; (b) las frecuencias de los picos (`picos_s`, con la $f_1$ de la simulación `f1s`) y `delta_sim` $=(f_k-kf_1)/(kf_1)$; (c) las tasas $\gamma_k$ medidas **con la misma función** `tasas_decaimiento` sobre la simulación (pasale `np.arange(1, 13) * f1s` como frecuencias de los armónicos), contra las impuestas $\gamma+bk^2$ y las de la grabación real; (d) las envolventes (RMS) de real y simulación en escala log.

**Qué se espera.** (1) $r\approx0.89$; el error de la simulación contra la serie es de $\sim1$–$2$ % de la amplitud en $t\le0.05$ s (queda una diferencia de fase por la dispersión numérica, que crece con $k$ y con $t$). (2) La simulación corre en unos 15–30 s. (3) Los picos de la simulación están (casi) en $kf_1$: `delta_sim` $\lesssim3\times10^{-4}$, negativo y creciente en módulo con $k$ (**dispersión numérica**, el signo contrario a la inarmonicidad real), y vale $\sim10^{-2}$ en la grabación: el modelo ideal no da el corrimiento. Las $\gamma_k$ medidas sobre la simulación **coinciden con las impuestas** (es la prueba de que la cadena de medición funciona: si medís $\gamma+bk^2$ sobre algo que sabés que decae con esa ley, y no lo recuperás, hay un error en el análisis), y se parecen a las reales sólo en tendencia. El espectro simulado tiene los armónicos con las amplitudes del modelo ideal (con el factor $\sin(k\pi x_m/L)$ del micrófono): no reproduce las de la grabación en detalle, y la simulación **no tiene piso de ruido**. La envolvente simulada es casi una recta en escala log (la domina el fundamental con $\gamma_1+\ldots\approx1.6$ s$^{-1}$), mientras que la real tiene dos etapas y a partir de $t\approx2$ s decae más despacio ($\sim1$ s$^{-1}$).
""",
    esqueleto=r'''
L = 0.65
c = 2 * L * f1

def cuerda_sim(N, m, theta, xm, gam0, nu, kappa=0.0, T=1.0, a=1e-3):
    """Cuerda de longitud L con fricción, pulsada en theta*L, muestreada en xm*L a fs.
    Paso de tiempo 1/(m fs); N tramos; extremos fijos. Devuelve y de longitud int(T*fs)."""
    dx, dt = L / N, 1 / (fs * m)
    r2 = (c * dt / dx) ** 2
    x_nodos = np.arange(N + 1) * dx
    # TODO: dato inicial (triángulo), primer paso, bucle leapfrog con fricción, guardar cada m pasos el nodo del micrófono
    y = np.zeros(int(T * fs))
    return y

# TODO: 1. comprobación sin fricción contra la serie (t <= 0.05 s)
# TODO: 2. simulación de 6 s, WAV en tempfile.gettempdir()
y_sim = ...
# TODO: 3. comparación: espectro, delta_sim, gamma_k de la simulación, envolventes
''',
    solucion=r'''
L = 0.65
c = 2 * L * f1

def cuerda_sim(N, m, theta, xm, gam0, nu, kappa=0.0, T=1.0, a=1e-3):
    """Cuerda de longitud L con fricción, pulsada en theta*L, muestreada en xm*L a fs.
    Paso de tiempo 1/(m fs); N tramos; extremos fijos. Devuelve y de longitud int(T*fs).
    kappa: rigidez (Tarea 7); extremos apoyados u = u_xx = 0 (extensión impar en los bordes)."""
    dx, dt = L / N, 1 / (fs * m)
    r2, q, s = (c * dt / dx) ** 2, nu * dt / dx ** 2, kappa * dt ** 2 / dx ** 4
    xn = np.arange(N + 1) * dx
    x0 = theta * L
    def lap(v):   # segunda diferencia con extensión impar en los bordes (v = 0 en los extremos)
        ve = np.concatenate(([-v[1]], v, [-v[-2]]))
        return ve[2:] - 2 * ve[1:-1] + ve[:-2]
    u_ant = np.where(xn <= x0, a * xn / x0, a * (L - xn) / (L - x0))        # u^0 (velocidad inicial nula)
    u = u_ant + 0.5 * r2 * lap(u_ant)                                        # u^1 (Taylor)
    jm = int(round(xm * N))
    pasos = int(T * fs) * m
    y = np.zeros(int(T * fs))
    y[0] = u_ant[jm]
    for n in range(1, pasos):
        if n % m == 0:
            y[n // m] = u[jm]
        d = u - u_ant
        nuevo = 2 * u - u_ant + r2 * lap(u) - 2 * gam0 * dt * d + q * lap(d)
        if kappa:
            nuevo -= s * lap(lap(u))
        u_ant, u = u, nuevo
    return y

# 1. comprobación sin fricción contra la serie
N_s, m_s, th_s, xm_s = 400, 4, theta_elegido, 0.27
r_num = c * (1 / (fs * m_s)) / (L / N_s)
y0 = cuerda_sim(N_s, m_s, th_s, xm_s, 0.0, 0.0, T=0.05)
tt = np.arange(len(y0)) / fs
kk = np.arange(1, 2001)
A_k = 1e-3 * 2 / (np.pi ** 2 * th_s * (1 - th_s)) * np.sin(kk * np.pi * th_s) / kk ** 2
y_serie = (A_k[:, None] * np.cos(2 * np.pi * kk[:, None] * f1 * tt[None, :]) * np.sin(kk[:, None] * np.pi * xm_s)).sum(axis=0)
print(f"r = {r_num:.3f};  error máximo contra la serie (t <= 0.05 s): {np.abs(y0 - y_serie).max():.1e} (amplitud máxima {np.abs(y_serie).max():.1e}) -> relativo {np.abs(y0 - y_serie).max() / np.abs(y_serie).max():.1e}")
print(f"nu k / h^2 = {nu * (1 / (fs * m_s)) / (L / N_s) ** 2:.1e},  gamma k = {gam0_B / (fs * m_s):.1e}   (deben ser << 1)")

# 2. simulación completa
y_sim = cuerda_sim(N_s, m_s, theta_elegido, xm_s, gam0_B, nu, T=6.0)
ruta_wav = os.path.join(tempfile.gettempdir(), "cuerda_sim.wav")
wavfile.write(ruta_wav, fs, (0.9 * y_sim / np.abs(y_sim).max() * 32767).astype(np.int16))
print(f"WAV escrito en {ruta_wav} ({os.path.getsize(ruta_wav) / 1e6:.1f} MB)")

# 3. comparación
f_s_, A_s_ = espectro_tramo(y_sim, fs, 0.0, 0.5)
f1s = medir_f1(f_s_, A_s_)
picos_s = medir_picos(f_s_, A_s_, f1s, K=15)
delta_sim = (picos_s[:, 1] - picos_s[:, 0] * f1s) / (picos_s[:, 0] * f1s)
Gs = tasas_decaimiento(y_sim, fs, np.arange(1, 13) * f1s)
oks = (Gs[:, 3] >= 30) & ~np.isnan(Gs[:, 1])
print(f"\nf1 simulada = {f1s:.3f} Hz (real {f1a:.3f}); picos simulados aceptados: {len(picos_s)} de 15")
print(f"delta_sim en k = 5, 10, 15: {delta_sim[4]:.1e}, {delta_sim[9]:.1e}, {delta_sim[14]:.1e}   (real: {delta[4]:.1e}, {delta[9]:.1e}, {delta[14]:.1e})")
print("  k   gamma medido en la simulación   impuesto   real")
for kk_, gs_ in zip(Gs[oks, 0], Gs[oks, 1]):
    kk_ = int(kk_); real = G[kk_ - 1, 1] if kk_ - 1 < len(G) else np.nan
    print(f"{kk_:3d}          {gs_:6.2f}                  {gam0_B + b_B * kk_ ** 2:6.2f}     {real:6.2f}")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
mf_ = fa < 3500
axs[0].semilogy(fa[mf_], Aa[mf_] / picos_a[0, 2], color=COLORES["dato"], lw=0.9, label="grabación")
axs[0].semilogy(f_s_[f_s_ < 3500], A_s_[f_s_ < 3500] / picos_s[0, 2], color=COLORES["modelo"], lw=0.9, label="simulación")
axs[0].set_xlabel(r"$\xi$ (Hz)"); axs[0].set_ylabel("amplitud (normalizada)"); axs[0].set_ylim(1e-6, 3); axs[0].legend(loc="upper right")
axs[0].set_title("espectro, primer medio segundo")
kkk = np.linspace(0.5, 12.5, 100)
axs[1].errorbar(k_g, gam_k, yerr=se_g, fmt="o", color=COLORES["dato"], ms=4, capsize=2, label="grabación")
axs[1].plot(Gs[oks, 0], Gs[oks, 1], "s", color=COLORES["modelo"], ms=5, label="simulación (medido)")
axs[1].plot(kkk, gam0_B + b_B * kkk ** 2, color=COLORES["modelo"], lw=1, label="impuesto")
axs[1].set_xlabel("$k$"); axs[1].set_ylabel(r"$\gamma_k$ (1/s)"); axs[1].legend(loc="upper left", fontsize=8); axs[1].set_ylim(0, 9)
axs[2].semilogy(t, env, color=COLORES["dato"], label="grabación")
e_sim = envolvente(y_sim, fs)
axs[2].semilogy(t, e_sim * env[int(0.05 * fs):int(0.15 * fs)].max() / e_sim[int(0.05 * fs):int(0.15 * fs)].max(), color=COLORES["modelo"], label="simulación (misma altura inicial)")
axs[2].set_xlabel("$t$ (s)"); axs[2].set_ylabel("envolvente"); axs[2].set_ylim(1e-5, 1); axs[2].legend(loc="upper right", fontsize=8)
fig.tight_layout()
''',
    verificacion=r'''
# Verificación
assert callable(cuerda_sim) and len(y_sim) == 6 * fs, "cuerda_sim debe devolver len(y) = int(T*fs)"
assert 0.8 < r_num < 1.0
_y = cuerda_sim(400, 4, theta_elegido, 0.27, 0.0, 0.0, T=0.05)
_k = np.arange(1, 2001)
_A = 1e-3 * 2 / (np.pi ** 2 * theta_elegido * (1 - theta_elegido)) * np.sin(_k * np.pi * theta_elegido) / _k ** 2
_s = (_A[:, None] * np.cos(2 * np.pi * _k[:, None] * f1 * (np.arange(len(_y)) / fs)[None, :]) * np.sin(_k[:, None] * np.pi * 0.27)).sum(axis=0)
assert np.abs(_y - _s).max() < 3e-2 * np.abs(_s).max(), "sin fricción, la simulación debería coincidir con la serie a ~1 % (t <= 0.05 s)"
assert os.path.exists(ruta_wav) and not ruta_wav.startswith(os.getcwd() + os.sep + "datos")
assert abs(f1s - f1) < 1.0, "la fundamental simulada debería ser f1"
assert np.all(np.abs(delta_sim) < 1e-3), "la simulación no tiene inarmonicidad: los picos deberían estar en k f1 (a 1e-3)"
_imp = gam0_B + b_B * Gs[oks, 0] ** 2
assert np.all(np.abs(Gs[oks, 1] / _imp - 1) < 0.25), "las tasas medidas sobre la simulación deberían recuperar las impuestas"
print("simulación: OK")
''')
figura_revision("simulacion")

# =============================================================================
# 7. Rigidez en el esquema (opcional)
# =============================================================================
lab.md(r"""
## 7. (Opcional) La rigidez en el esquema

Para $u_{tt}=c^2u_{xx}-\kappa u_{xxxx}$ agregamos al leapfrog el término $-\kappa k^2\,\delta^4u^n/h^4$, con $\delta^4=\delta^2\delta^2$ (la diferencia centrada de cuarto orden, con los coeficientes $1,-4,6,-4,1$). Dos detalles que **no** son triviales:

* **Los bordes.** Para la condición $u=u_{xx}=0$ hay que evaluar $\delta^2u$ y $\delta^2(\delta^2u)$ cerca del borde, donde faltan nodos. La *extensión impar* ($u_{-1}=-u_1$, y lo mismo para $\delta^2u$) los pone en su lugar: el mismo `lap` de la Tarea 6, que ya la usa, sirve de nuevo para $\delta^2$ y para $\delta^4=\texttt{lap(lap(u))}$. Así, la simulación conserva los mismos modos $\sin(k\pi x/L)$ que la teoría de la Sección 3, con las frecuencias $\omega_k^2=c^2\lambda_k+\kappa\lambda_k^2$ de esa sección (más el error numérico).
* **La estabilidad es más exigente.** Con Fourier discreta, cada modo cumple $z^2-(2-\alpha-\beta)z+1=0$ con $\alpha=4r^2\sin^2\frac\theta2$ y $\beta=\frac{16\kappa k^2}{h^4}\sin^4\frac\theta2$ ($\theta=k\pi/N$); las raíces tienen módulo $1$ si y sólo si $0\le\alpha+\beta\le4$, y el peor caso es $\theta=\pi$: $\boxed{\,r^2+\dfrac{4\kappa k^2}{h^4}\le1\,}$, es decir, $r^2\bigl(1+\frac{4\kappa}{c^2h^2}\bigr)\le1$. La rigidez **exige un $r$ menor que 1**, y más exigente cuanto más fina es la grilla: el paso $k$ tiene que ser $O(h^2)$ y no $O(h)$. Con $\kappa/c^2=BL^2/\pi^2\approx5.6\times10^{-6}$ m$^2$ y $h=1.6$ mm (la grilla de la Tarea 6), $\frac{4\kappa}{c^2h^2}\approx8.7$, y la condición pide $r\le0.32$, mientras que en la Tarea 6 el paso tenía $r=0.89$. Hay que **engrosar la grilla** ($N=200$, $h=3.25$ mm: el factor baja a $2.1$ y alcanza $r\le0.57$, y con $m=4$ pasos por muestra $r=0.44$).

El $\kappa$ sale de $B$: $\kappa=Bc^2L^2/\pi^2$. En lo que sigue, $c$ es la de la Tarea 6 (con $f_0=f_1$ en lugar de $f_0=f_1/\sqrt{1+B}$: la diferencia relativa es $B/2\sim10^{-4}$, del orden del error numérico).
""")

lab.tarea(
    titulo="(Opcional) Agregar la rigidez y ver el corrimiento de los picos",
    consigna=r"""
La función `cuerda_sim` de la Tarea 6 ya trae el argumento `kappa`: completá (en esa celda) el término $-\kappa k^2\,\delta^4u^n/h^4$ usando la función `lap` dos veces.

1. Calculá $\kappa=B c^2L^2/\pi^2$ con `B_nl` de la Tarea 3. Comprobá la condición de estabilidad $r^2+4\kappa k^2/h^4\le1$ para $(N,m)=(400,4)$ y $(200,4)$, y mostrá (simulando $0.002$ s, con la cuerda pulsada a $10^{-3}$) que el primer caso explota (`np.abs(y).max()` enorme o `nan`) y el segundo no.
2. Con $(N,m)=(200,4)$, $T=1$ s, la misma fricción y los mismos $\theta,x_m$ de antes, simulá **con** rigidez (`y_rig`) y **sin** ella (`y_ctrl`, `kappa=0`, misma grilla: sólo pone en evidencia la dispersión numérica). Medí los picos de ambas con `medir_picos` (tramo $[0,0.5]$ s) y calculá el corrimiento relativo $\delta_k$ de cada una.
3. Graficá los $\delta_k$ de la grabación, de `y_rig`, de `y_ctrl` y la curva $\frac B2(k^2-1)$. Comprobá que $\delta_k^{rig}-\delta_k^{ctrl}$ reproduce la curva teórica y que, restando el control, $B$ recuperado por el ajuste de la Tarea 3 coincide con el que le impusiste.

**Qué se espera.** El primer caso ($h=1.6$ mm, $r=0.89$) tiene $r^2+4\kappa k^2/h^4\approx8$: explota en pocas decenas de pasos (la amplitud pasa a `inf`/`nan` antes de los $0.002$ s); el segundo ($\approx0.7$) es estable. Con rigidez, los picos de la simulación se corren hacia arriba como los de la grabación (de $\sim1.5$ % en $k=15$), con el mismo $B$ ajustado: es la comprobación de que el modelo con rigidez **explica** el corrimiento, con un solo parámetro. Sin rigidez, la dispersión numérica corre los picos hacia **abajo** $\sim2\times10^{-3}$ en $k=15$ con esta grilla gruesa, un orden de magnitud menos que el efecto de la rigidez: por eso no se pierde la comparación, pero hay que restarlo.
""",
    esqueleto=r'''
# TODO: en la celda de la Tarea 6, completar el término de rigidez de cuerda_sim (usando lap dos veces)

kappa = ...   # TODO: B_nl c^2 L^2 / pi^2

def condicion_estabilidad(N, m):
    """r^2 + 4 kappa dt^2 / dx^4 (debe ser <= 1)."""
    pass  # TODO

# TODO: 1. estabilidad para (400, 4) y (200, 4); explosión en 0.02 s
# TODO: 2. simulaciones y_rig y y_ctrl con (N, m) = (200, 4), T = 1 s; picos y delta de cada una
# TODO: 3. figura y ajuste de B sobre (delta_rig - delta_ctrl)
''',
    solucion=r'''
kappa = B_nl * c ** 2 * L ** 2 / np.pi ** 2

def condicion_estabilidad(N, m):
    """r^2 + 4 kappa dt^2 / dx^4 (debe ser <= 1)."""
    dx, dt = L / N, 1 / (fs * m)
    return (c * dt / dx) ** 2 + 4 * kappa * dt ** 2 / dx ** 4

for N_, m_ in [(400, 4), (200, 4)]:
    print(f"(N, m) = ({N_}, {m_}): r = {c / (fs * m_) / (L / N_):.2f}, r^2 + 4 kappa k^2/h^4 = {condicion_estabilidad(N_, m_):.2f}")
with np.errstate(all="ignore"):
    print("  amplitud máxima tras 0.002 s con (400, 4):", f"{np.abs(cuerda_sim(400, 4, theta_elegido, 0.27, gam0_B, nu, kappa=kappa, T=0.002)).max():.1e}", "(la cuerda se pulsó con 1e-3)")
    print("  amplitud máxima tras 0.002 s con (200, 4):", f"{np.abs(cuerda_sim(200, 4, theta_elegido, 0.27, gam0_B, nu, kappa=kappa, T=0.002)).max():.1e}")

y_rig = cuerda_sim(200, 4, theta_elegido, 0.27, gam0_B, nu, kappa=kappa, T=1.0)
y_ctrl = cuerda_sim(200, 4, theta_elegido, 0.27, gam0_B, nu, kappa=0.0, T=1.0)
dd = {}
for nombre, y_ in [("rig", y_rig), ("ctrl", y_ctrl)]:
    f_, A_ = espectro_tramo(y_, fs, 0.0, 0.5)
    f1_ = medir_f1(f_, A_)
    P_ = medir_picos(f_, A_, f1_, K=15)
    dd[nombre] = (P_[:, 0], (P_[:, 1] - P_[:, 0] * f1_) / (P_[:, 0] * f1_))
k_r, d_rig = dd["rig"]; k_c, d_ctrl = dd["ctrl"]
comun = np.intersect1d(k_r, k_c)                       # los k medidos en las dos simulaciones
d_rig_c = d_rig[np.isin(k_r, comun)]; d_ctrl_c = d_ctrl[np.isin(k_c, comun)]
print(f"\ncorrimiento en k = 15: grabación {delta[14]:.2e},  simulación con rigidez {d_rig[k_r == 15][0]:.2e},  control sin rigidez {d_ctrl[k_c == 15][0]:.2e}")
dif = d_rig_c - d_ctrl_c
B_rec = 2 * np.polyfit(comun ** 2 - 1, dif, 1)[0]
print(f"B impuesto = {B_nl:.2e};  B recuperado de (delta_rig - delta_ctrl) = {B_rec:.2e}")

fig, ax = plt.subplots(figsize=(6.5, 4))
kk = np.linspace(1, 15.5, 100)
ax.plot(k_a, 100 * delta, "o", color=COLORES["dato"], label="grabación")
ax.plot(k_r, 100 * d_rig, "s", color=COLORES["modelo"], ms=5, label="simulación con rigidez")
ax.plot(k_c, 100 * d_ctrl, "^", color=CICLO[2], ms=5, label="control (sin rigidez)")
ax.plot(kk, 100 * (np.sqrt((1 + B_nl * kk ** 2) / (1 + B_nl)) - 1), "k--", lw=1, label=rf"teoría, $B={B_nl:.1e}$")
ax.axhline(0, color="0.7", lw=0.8); ax.set_xlabel("$k$"); ax.set_ylabel(r"$(f_k-kf_1)/(kf_1)$ (%)"); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
''',
    verificacion=r'''
# Verificación
assert condicion_estabilidad(400, 4) > 1 and condicion_estabilidad(200, 4) < 1
assert np.abs(y_rig).max() < 1 and np.abs(y_ctrl).max() < 1, "con (200, 4) la simulación debe ser estable"
assert d_rig[k_r == 15][0] > 0.008, "con rigidez, k = 15 debería correrse ~ 1.5 % hacia arriba"
assert abs(B_rec / B_nl - 1) < 0.15, "restando el control, B se debería recuperar a ~15 %"
print("rigidez en el esquema: OK")
''')
figura_revision("rigidez")

# =============================================================================
# 8. Balance
# =============================================================================
lab.md(r"""
## 8. Balance: lo que el modelo ideal predice y lo que le falta

Con todo lo medido, armá una tabla que ponga lado a lado, para cada característica de la cuerda: lo que predice el modelo ideal, lo que mediste en la grabación y qué término del modelo (o qué fenómeno *no incluido*) lo explica. Es el insumo directo de la interpretación escrita.
""")

lab.tarea(
    titulo="Tabla resumen",
    consigna=r"""
Imprimí una tabla de texto con (al menos) estas filas, con **tus** números y sus incertidumbres o rangos: (i) frecuencia fundamental $f_1$ (y su variación en el tiempo); (ii) desvío de los armónicos respecto de $kf_1$ (a $k=5$ y $k=15$) y el parámetro $B$; (iii) tasa de decaimiento del fundamental y de los armónicos altos ($\gamma_1$, $\gamma_{10}$); (iv) reparto de energía entre armónicos ($A_2/A_1$, $A_5/A_1$, $x_0$ elegido); (v) duración total ($T_{40}$). Para cada fila, poné la predicción del modelo ideal en una columna (frecuencias $kf_1$; $\gamma=0$; $A_k$ del triángulo) y la del modelo con los términos agregados (rigidez, fricción viscoelástica) en otra.

**Qué se espera.** Una tabla corta y legible: con una fila por característica y una nota (columna "qué falta") como *rigidez*, *fricción*, *forma real de la pulsación / respuesta del cuerpo*, *no linealidad (cambio de $f_1$ con la amplitud)* o *dos polarizaciones (dobletes, batidos)*. Cada afirmación de tu interpretación tiene que poder rastrearse a una fila de esta tabla.
""",
    esqueleto=r'''
# TODO: imprimir la tabla resumen (variables de las tareas anteriores)
''',
    solucion=r'''
filas = [
    ("f1 (Hz)",                   "constante",                           f"{f1a:.2f} ({f1_t.min():.2f}..{f1_t.max():.2f} según el tramo)", "constante (aprox.)",                        "no linealidad: tensión depende de la amplitud"),
    ("delta_5 (%)",               "0",                                   f"{100 * delta[4]:.2f}",                                          f"{100 * (np.sqrt((1 + B_nl * 25) / (1 + B_nl)) - 1):.2f} (con B)", "rigidez a la flexión"),
    ("delta_15 (%)",              "0",                                   f"{100 * delta[14]:.2f}",                                         f"{100 * (np.sqrt((1 + B_nl * 225) / (1 + B_nl)) - 1):.2f}", "rigidez a la flexión"),
    ("B",                         "0",                                   f"{B_nl:.1e} +- {se_B:.0e}",                                      "parámetro ajustado",                           "orden típico 1e-4 a 1e-3"),
    ("gamma_1 (1/s)",             "0",                                   f"{gam_k[0]:.2f}",                                                f"{gam0_B + b_B:.2f} (gamma + b)",             "fricción (aire, cuerpo)"),
    ("gamma_10 (1/s)",            "0",                                   f"{gam_k[k_g == 10][0]:.2f}",                                     f"{gam0_B + 100 * b_B:.2f} (gamma + 100 b)",   "fricción viscoelástica (crece con k)"),
    ("A_2/A_1",                   f"{A_ideal(2, theta_elegido):.3f} (x0 = {theta_elegido} L)", f"{a_med[1]:.3f}",                            "igual que el ideal",                           "pulsación real, cuerpo, polarizaciones"),
    ("A_5/A_1",                   f"{A_ideal(5, theta_elegido):.3f}",    f"{a_med[4]:.3f}",                                                "igual que el ideal",                           "idem"),
    ("T40 (s)",                   "infinita",                            f"{T40:.1f}",                                                     "según gamma_k",                                "dos etapas de decaimiento"),
]
anchos = [max(len(f[i]) for f in filas + [("cantidad", "modelo ideal", "grabación", "con rigidez y fricción", "qué falta")]) for i in range(5)]
formato = "  ".join(f"{{:<{a}}}" for a in anchos)
print(formato.format("cantidad", "modelo ideal", "grabación", "con rigidez y fricción", "qué falta"))
print("  ".join("-" * a for a in anchos))
for f in filas:
    print(formato.format(*f))
''',
    verificacion=None)

# =============================================================================
# Interpretación
# =============================================================================
lab.interpretacion([
    r"**La ecuación y sus hipótesis (pregunta 1 del problema).** ¿Qué ecuación de movimiento usaste para la cuerda ideal y en qué hipótesis físicas se apoya (tensión uniforme, desplazamientos chicos, cuerda perfectamente flexible, sin pérdidas)? Para cada hipótesis, decí si tus mediciones la confirman o la contradicen, y citá el número (o el gráfico) que lo muestra: al menos las de flexibilidad y sin pérdidas.",
    r"**Las frecuencias (pregunta 2).** Con tu $f_1$ medida y $L=0.65$ m: ¿cuánto vale $c$ y, si la densidad lineal es $\rho=3.5$ g/m, cuánto $T$? ¿Qué pasa con $f_1$ si acortás la cuerda a la mitad, y si duplicás la tensión? ¿Están los picos en $kf_1$? Citá tu $B\pm$ su incertidumbre, el corrimiento a $k=10$ en cents y decí si un afinador que trabajara con el armónico 10 daría la misma nota que con el fundamental. ¿Por qué el corrimiento va hacia arriba y no hacia abajo?",
    r"**El timbre (pregunta 3).** ¿Qué determina que una misma nota suene distinta según dónde se la pulse, y por qué un piano (cuerda golpeada) suena más rico en armónicos altos que una guitarra (pulsada)? Con la Tarea 5: ¿qué $x_0$ elegiste, qué medida usaste y por qué la otra medida no daba lo mismo? ¿Por qué una grabación real no puede reproducir *exactamente* los $A_k$ del modelo?",
    r"**Qué esperamos ver, qué vemos, qué le falta (pregunta 4).** Escribí un párrafo (máximo 12 líneas) con la conclusión que pide el enunciado: qué predice bien el modelo de la cuerda ideal (¿qué medida lo confirma?), qué le falta (con el número y el nombre del fenómeno: rigidez, fricción que depende de $k$, ¿algo más?), qué términos le agregarías al modelo y cuáles de esos efectos reproduce tu simulación (Tareas 6 y 7) y cuáles no. Incluí una observación sobre el decaimiento: ¿son todos los armónicos iguales? ¿Cuál de los dos modelos de fricción se parece más a lo medido, con qué residuo, y qué sugiere sobre el mecanismo de pérdida?",
])

lab.md(r"""
### Respuestas modelo (para el docente)

**1.** Ecuación de ondas $u_{tt}=c^2u_{xx}$, $c^2=T/\rho$, de la segunda ley de Newton para un elemento de cuerda con tensión $T$ uniforme y pendientes chicas ($|u_x|\ll1$: la componente vertical de la tensión es $T u_x$). Hipótesis y evidencia: (a) *flexibilidad perfecta*: falsa, $B\approx1.4\times10^{-4}$ (positivo, significativo: $\delta_{15}\approx1.5$ %, contra $\sim0.05$ % de error de medición), la rigidez agrega $-\kappa u_{xxxx}$; (b) *sin pérdidas*: falsa, la envolvente cae $40$ dB en $\sim4$ s ($\gamma_{glob}\approx1.1$ s$^{-1}$) y la tasa depende de $k$ ($\gamma_1\approx1.4$, $\gamma_{10}\approx5.4$ s$^{-1}$); (c) *desplazamientos chicos, tensión constante*: falla débilmente: $f_1$ baja $0.4$ Hz ($0.2$ %) en el primer segundo, el estiramiento de la cuerda con la amplitud (no linealidad); (d) *extremos fijos*: se ve indirectamente en que los picos son agudos y bien definidos (el puente y el cejuelo son casi rígidos).

**2.** $c=2Lf_1=2\times0.65\times196\approx255$ m/s; $T=\rho c^2=3.5\times10^{-3}\times6.5\times10^4\approx227$ N (una cuerda **de acero liso** de $0.43$ mm tendría $\rho\approx1.1$ g/m y $T\approx71$ N; el $3.5$ g/m del texto corresponde a una cuerda entorchada gruesa: no hay contradicción, sólo otra cuerda). $f_1\propto1/L$: la mitad de la longitud duplica $f_1$ (una octava: el traste 12); $f_1\propto\sqrt T$: duplicar la tensión multiplica $f_1$ por $\sqrt2$ ($+6$ semitonos). Los picos están *casi* en $kf_1$: hasta el 1.5 % a $k=15$. Con $B=(1.4\pm0.1)\times10^{-4}$: $\delta_{10}\approx0.7$ % $\approx11$ cents (audible: el umbral de discriminación de altura ronda los 5 cents), así que el armónico 10 suena más agudo que $10f_1$; los afinadores de piano ("estiran" las octavas para compensar la inarmonicidad) lo tienen en cuenta. El corrimiento va hacia arriba porque la rigidez agrega una fuerza de restitución ($-\kappa u_{xxxx}$ suma a $c^2u_{xx}$) que crece con la curvatura, o sea con $k$: $\omega_k^2=c^2\lambda_k+\kappa\lambda_k^2$.

**3.** El timbre es el reparto de energía entre los modos; la amplitud del modo $k$ depende del punto de pulsado por $\sin(k\pi x_0/L)$: pulsar en un nodo del modo $k$ lo anula, pulsar cerca del puente ($x_0\ll L$) da un espectro rico (sin ceros hasta $k\approx L/x_0$) y en el medio uno pobre (sólo armónicos impares). En el piano el martillo golpea (velocidad inicial concentrada, $B_k\propto\sin(k\pi x_0/L)/k$, decae como $1/k$) y la guitarra se pulsa (desplazamiento inicial triangular, $A_k\propto\sin/k^2$): la energía $E_k\propto\omega_k^2B_k^2\sim k^0$ contra $\sim k^{-2}$, mucho más brillante. Con la Tarea 5: $x_0=L/5$ (o cualquier $\theta$ entre $0.12$ y $0.2$; se acepta $L/5$ entre los tres y una justificación): la distancia en dB penaliza los ceros y los errores en los armónicos chicos y descarta $L/2$ (los pares están presentes a $-17$ dB) y $L/20$ (predice demasiada energía arriba); la similitud coseno está dominada por el fundamental y no discrimina ($>0.92$ en los tres casos, incluso prefiere $L/2$). No puede reproducir los $A_k$ exactamente porque: el micrófono/cuerpo aplica una respuesta en frecuencia (resonancias del cuerpo), la pulsación real no es un triángulo con velocidad nula, hay dos polarizaciones, y la cuerda pulsada por un dedo tiene un ancho de contacto.

**4.** *Párrafo modelo.* El modelo ideal predice bien que el espectro es un peine de picos separados por $f_1=\frac1{2L}\sqrt{T/\rho}$ (picos a menos de $1.5$ % de $kf_1$ hasta $k=15$), que el timbre es el reparto de energía entre modos y que depende del punto de pulsado (aunque sólo cualitativamente). Le faltan: (1) *inarmonicidad*: los picos se corren hacia arriba $\delta_k\approx\frac B2(k^2-1)$ con $B\approx1.4\times10^{-4}$, que se explica agregando la rigidez $-\kappa u_{xxxx}$ (con $\kappa=Bc^2L^2/\pi^2$) y con la que la simulación reproduce el corrimiento; (2) *pérdidas*: la señal se apaga (40 dB en 4 s) y los armónicos altos más rápido, $\gamma_k$ de $\sim1.4$ a $\sim5.4$ s$^{-1}$ entre $k=1$ y $k=10$, lo que descarta la fricción constante (residuo $1.5$ s$^{-1}$ contra $1.1$ del modelo (B) $\gamma+bk^2$, con $\gamma\approx2.4$ s$^{-1}$ y $b\approx0.013$ s$^{-1}$); pero el modelo (B) sobreestima $\gamma_k$ en $k$ grande y subestima en los intermedios: los datos crecen casi linealmente en $k$, lo que sugiere que la pérdida real no es sólo interna viscoelástica (contribuyen la radiación por el cuerpo y el puente, la fricción con el aire, que depende de la frecuencia de otra manera) y que habría que proponer un término distinto; (3) los efectos no lineales (la fundamental baja $0.2$ %) y las dos polarizaciones (dobletes y batidos) quedan fuera del modelo lineal de una polarización. La simulación con (B) reproduce las tasas impuestas (con error de unos 10 %), no reproduce la inarmonicidad salvo que se agregue la rigidez (Tarea 7), ni el piso de ruido, ni la respuesta del cuerpo, ni el ataque real.

**Tiempos.** Tarea 1: 20 min; Tarea 2: 40; Tarea 3: 35; Tarea 4: 50 (la más difícil: el espectrograma y los criterios de ajuste); Tarea 5: 30; Tarea 6: 50 (la simulación de 6 s tarda 15–30 s); Tarea 7 (opcional): 30; Tarea 8: 10; interpretación: 40 minutos en casa. **Dificultades típicas.** (a) No normalizar a `float`: los espectros salen 32768 veces más grandes y nada falla, pero el piso de ruido y los umbrales se rompen. (b) Interpolar sobre `A` en lugar de `ln A`. (c) Fijar la ventana de ajuste de $\ln A_k(t)$ sin excluir el ataque o sin cortar en el piso de ruido: el $\gamma_k$ de los armónicos altos sale entre $2$ y $3$ veces menor. (d) En la Tarea 4, usar $k=1$ como $f_1$ de otro tramo. (e) En la Tarea 6, olvidar que $r$ depende de $m$ (usar $m=1$: $N$ máximo de $112$ nodos, error $\sim16\times$ mayor en $k=20$); pasos con $r>1$ (explota) y, en la Tarea 7, usar la misma grilla fina (explota por la rigidez). (f) Confundir el índice de armónico $k$ con el paso de tiempo $k$ (por eso en la Tarea 6 se dice $r=ck/h$ con $k=\Delta t$ y en el código aparece `dt`). **Con una sola sesión de 4 h**, dar hecha la Tarea 1 y pedir 2, 3, 4, 6 y la interpretación; la Tarea 5 puede darse resuelta en la parte de la fórmula.
""", destino="docente")

lab.md(r"""
### Cambios sugeridos para el texto (`lab:cuerda`), para el docente

1. **"Repetir pulsando la cuerda en el medio y cerca del puente"** (ítem 4) requiere una grabación por posición; hay una sola en `datos/`. En el notebook se hace con la fórmula del Ejercicio guiado ($x_0=L/2$, $L/5$, $L/20$) contra la grabación existente, y se deja como *opcional* que quien tenga una guitarra grabe la suya. Si se quiere que el ítem se pueda hacer siempre, conviene reformularlo ("Comparar el reparto de energía entre armónicos con la predicción del Ejercicio guiado para distintas posiciones de pulsado; si se dispone de una cuerda, repetir la grabación pulsando en el medio y cerca del puente").
2. **"Calcular el espectro de potencia del primer medio segundo"**: con una resolución de $2$ Hz sólo se mide bien la inarmonicidad si se interpola el pico; conviene nombrarlo en el enunciado ("con una precisión mejor que la resolución de la DFT, interpolando el pico"), y advertir que con un tramo largo el corrimiento se confunde con el descenso de $f_1$ (la no linealidad): el enunciado presupone que $f_k-kf_1$ es una propiedad de la cuerda, y no es del todo cierto con una grabación real.
3. **"¿crece como $k^2$?"**: con esta grabación y $k\le15$ el exponente ajustado sale $\sim1.9$ (bien), pero el resultado depende de un doblete en $k=9$; sugiero decir "compararlo con $\frac B2(k^2-1)$ ajustando $B$".
4. **Ejercicio de amortiguamiento, ítem 3** ("$-2\gamma u_t+\nu u_{xxt}$"): la tasa de decaimiento resulta $\gamma_k=\gamma+\frac\nu2(k\pi/L)^2$; con los datos, el crecimiento medido es casi *lineal* en $k$ y no cuadrático. Puede valer la pena agregar un ítem ("comparar con la ley medida y discutir otros mecanismos") en la guía, o suavizar el ítem 3 del laboratorio a "proponer un término que haga crecer la tasa con $k$".
5. **Ejercicio de amortiguamiento, ítem 4** define $\beta=\kappa\pi^2/(c^2L^2)$ y el laboratorio pide comparar "con la predicción de inarmonicidad": la fórmula $\omega_k=\frac{ck\pi}{L}\sqrt{1+\beta k^2}$ vale para extremos *apoyados* ($u=u_{xx}=0$); conviene decirlo en el ítem (para extremos empotrados, $u=u_x=0$, no son senos).
6. **Item 5** ("con el esquema del Ejercicio `lab:leapfrog`, agregando el término de fricción"): con $f_s=44100$ Hz, un solo paso por muestra da como máximo $N\approx112$ nodos; la sugerencia de 4 pasos por muestra (que dobla el $N$ al costo de repetir) merece una línea en el enunciado del ejercicio de leapfrog o en el del laboratorio, y la rigidez requiere $k=O(h^2)$.
""", destino="docente")

lab.md(r"""
## Cobertura del ejercicio del texto y ejercicios adicionales

Los ítems del ejercicio `lab:cuerda` y dónde se resuelven:

* Ítem 1 (cargar el audio): Tarea 1 (sobre `cuerda_guitarra.wav`; grabar el propio es opcional).
* Ítem 2 (espectro del primer medio segundo; $f_1$, picos, $kf_1$; corrimiento y $k^2$): Tareas 2 y 3.
* Ítem 3 (espectrograma; tasas de decaimiento; término de fricción): Tarea 4.
* Ítem 4 (pulsar en distintos puntos; reparto de energía): Tarea 5 (con la fórmula del Ejercicio guiado; ver el cambio sugerido 1).
* Ítem 5 (simulación con fricción, escribir el audio, comparar; párrafo de conclusión): Tarea 6 (rigidez: Tarea 7 opcional; balance: Tarea 8; párrafo: pregunta 4 de la interpretación).

**Para experimentar.**

* Grabá tu propia cuerda (o la de un ukelele, o una regla de acero sujeta en un extremo) y repetí las Tareas 2, 3 y 5 con pulsados en el medio y cerca del puente. ¿Coincide la posición de los ceros de los $A_k$ con la fórmula?
* En la Tarea 6 cambiá $\gamma$ y $\nu$ y *escuchá* la diferencia: ¿qué se oye cuando $\nu=0$ (todos los armónicos decaen igual)? ¿Y con $\nu$ 10 veces mayor?
* Cambiá el pulsado por una **cuerda golpeada** (velocidad inicial concentrada en $x_0$, desplazamiento inicial nulo: `u_ant = 0` y primer paso con $\eta$) y compará los $A_k$ con los del Ejercicio guiado: ¿cuánto más brillante es?
* Simulá con $\kappa$ diez veces mayor: ¿desde qué $k$ los armónicos ya no son "armónicos" (pasan a ser una campana)?
""")

rutas = lab.escribir()
