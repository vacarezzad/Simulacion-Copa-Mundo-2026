# ⚽ Simulador Monte Carlo — Mundial 2026

Simulador estadístico del **FIFA World Cup 2026** (USA · Canadá · México) basado en Monte Carlo. Corre decenas de miles de torneos completos para estimar probabilidades de clasificación, avance por ronda y campeonato para cada uno de los 48 equipos clasificados.

---

## Demo

```
streamlit run app.py
```

---

## Características

| Pestaña | Contenido |
|---|---|
| 🏆 Campeones | Probabilidad de ganar el torneo por equipo y confederación |
| 🗺️ Grupos | Clasificación esperada dentro de cada grupo |
| 📊 Avance | Probabilidad de llegar a cada ronda (R32 → Final) |
| ⚽ Partidos frecuentes | Resultados y marcadores más probables, fase de grupos y eliminatorias |
| 🔎 Equipo | Análisis individual: grupo, rivales y camino al título |
| 📝 Resumen | Texto exportable con todos los resultados |

- **Paralelismo multicore** — usa todos los núcleos disponibles vía `ProcessPoolExecutor`
- **Parámetros ajustables en tiempo real** desde el sidebar de Streamlit
- **Semilla fija** opcional para reproducibilidad exacta

---

## Modelo probabilístico

El modelo sigue la estructura **Dixon-Coles simplificada**:

```
λ_A = base_goles × (ataque_A / μ_gf) × (μ_gc / defensa_B) × factor_local_A
```

Los goles de cada equipo en cada partido se sampean de una distribución de **Poisson independiente** con medias `λ_A` y `λ_B`.

### Construcción de `ataque` y `defensa`

1. **Blend histórico + reciente** (`alpha_reciente`): pondera el promedio de goles de las últimas 3 ediciones (2014-2022) contra todo el historial (1930-2022).
2. **Imputación de debutantes**: equipos sin historial mundialista (Curacao, Jordan, Cape Verde, Uzbekistán) se imputan por promedio de confederación, mínimo global o interpolación desde puntos FIFA.
3. **Blend con ranking FIFA** (`beta_fifa`): corrige los promedios históricos con el rendimiento actual según los puntos FIFA.

### Resolución de partidos KO

```
90 min  →  ¿empate?
               ├── No  → ganador en 90'
               └── Sí  → Tiempo extra (30 min, λ × 0.35)
                             ├── gol → ganador en ET
                             └── sin gol → Penales
                                   p_A = 0.5 + w × tanh(Δpuntos_FIFA / k)
```

### Parámetros configurables

| Parámetro | Descripción | Default |
|---|---|---|
| `alpha_reciente` | Peso de datos 2014-2022 vs historial completo | 0.60 |
| `beta_fifa` | Peso del ranking FIFA vs estadísticas históricas | 0.55 |
| `min_partidos_recientes` | Umbral mínimo para usar datos recientes | 6 |
| `base_goles` | Media de goles esperados por equipo/partido | 1.35 |
| `factor_local` | Bonus de goles para sedes (USA, Canadá, México) | 1.08 |
| `penalty_weight` | Influencia del ranking FIFA en penales | 0.15 |
| `penalty_k` | Escala de diferencia de puntos en función tanh | 400 |
| `simular_tiempo_extra` | Tiempo extra antes de penales (True/False) | True |
| `bonus_campeon_vigente` | Multiplicador para Argentina (campeón 2022) | 1.08 |
| `debutante_mode` | Método de imputación para debutantes | `conf_avg` |
| `n_simulaciones` | Cantidad de torneos a simular | 100 000 |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/simulacion-mundial-2026.git
cd simulacion-mundial-2026

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar la app
streamlit run app.py
```

**Python requerido:** 3.10+

---

## Estructura del proyecto

```
simulacion-mundial-2026/
│
├── app.py                      # Interfaz Streamlit
├── requirements.txt
│
├── src/
│   ├── config.py               # Dataclass SimConfig con todos los parámetros
│   ├── modelo.py               # Núcleo probabilístico (Poisson + KO)
│   ├── preparacion.py          # Construcción de stats por equipo
│   ├── torneo.py               # Simulación de grupos y fases KO
│   └── montecarlo.py           # Loop Monte Carlo (secuencial y paralelo)
│
├── Data/
│   ├── clasificados_2026.csv   # Dataset final: 48 equipos con stats
│   ├── ranking_fifa.csv        # Ranking FIFA (extraído de la web)
│   ├── equipos_completo.csv    # Historial consolidado por equipo
│   └── participaciones_mundial/ # CSVs por edición (1930-2022)
│
└── Scripts de datos (ejecutar en orden):
    ├── 01_extraccion.py        # Scraping del ranking FIFA
    ├── 02_consolidacion.py     # Consolida historial de mundiales
    ├── 03_ranking_fifa.py      # Procesa el ranking FIFA
    └── 04_clasificados_2026.py # Arma el dataset final de 48 equipos
```

---

## Pipeline de datos

Los datos ya vienen incluidos en `Data/`, pero si querés regenerarlos desde cero:

```bash
python 01_extraccion.py      # Descarga ranking FIFA actual
python 02_consolidacion.py   # Consolida resultados históricos
python 03_ranking_fifa.py    # Procesa y limpia el ranking
python 04_clasificados_2026.py  # Genera clasificados_2026.csv
```

---

## Rendimiento

| Configuración | Tiempo (aprox.) |
|---|---|
| 10 000 sims, 1 core | ~3.5 s |
| 10 000 sims, 4 cores | ~1.5 s |
| 100 000 sims, 4 cores | ~15 s |
| 500 000 sims, 8 cores | ~60 s |

El número de workers se configura desde el sidebar (`⚡ Rendimiento`).

---

## Stack tecnológico

- **Python 3.10+**
- [Streamlit](https://streamlit.io/) — interfaz web
- [NumPy](https://numpy.org/) — generación de números aleatorios y cálculo vectorizado
- [Pandas](https://pandas.pydata.org/) — carga y manipulación del dataset
- [Plotly](https://plotly.com/python/) — visualizaciones interactivas
- [SciPy](https://scipy.org/) — distribuciones estadísticas

---

## Licencia

MIT
