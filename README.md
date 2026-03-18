# ⚽ Simulador Monte Carlo — Copa del Mundo 2026

Simulador estadístico del **FIFA World Cup 2026** (USA · Canadá · México) basado en Monte Carlo. Corre decenas de miles de torneos completos para estimar probabilidades de clasificación, avance por ronda y campeonato para cada uno de los **48 equipos** clasificados.

---

## Demo

```bash
streamlit run app.py
```

Abrir en: [http://localhost:8501](http://localhost:8501)

---

## Pestañas de la aplicación

| Pestaña | Contenido |
|---|---|
| 🏆 Probabilidades | Probabilidad de ser campeón del mundo, métricas globales y top equipos |
| 🎯 Escenarios frecuentes | Los caminos al título más probables (qué equipos aparecen más en cada ronda) |
| ⚽ Partidos frecuentes | Resultados y marcadores más probables en fase de grupos y eliminatorias |
| 🗂️ Bracket | Bracket visual R32 → Final con el campeón más probable; scrollable horizontalmente |
| 📊 Etapas | Heatmap de probabilidades por ronda (Cuartos → Campeón) para los top equipos |
| 🌍 Por confederación | Probabilidad de que el campeón provenga de cada confederación |
| 🔬 Datos | Tabla completa con todas las probabilidades y estadísticas del dataset |

---

## Características

- **Dixon-Coles** — corrección de baja puntuación (ρ) sobre modelo Poisson independiente
- **Paralelismo multicore** — usa todos los núcleos disponibles vía `ProcessPoolExecutor`
- **Parámetros ajustables en tiempo real** desde el sidebar de Streamlit
- **Semilla fija** opcional para reproducibilidad exacta
- **Bracket interactivo** con scroll horizontal y zoom (Plotly + `streamlit.components.v1`)

---

## Modelo probabilístico

### Goles esperados (Dixon-Coles simplificado)

```
λ_A = base_goles × (ataque_A / μ_gf) × (μ_gc / defensa_B) × factor_local_A
λ_B = base_goles × (ataque_B / μ_gf) × (μ_gc / defensa_A) × factor_local_B
```

Los goles se sampean de una distribución de **Poisson independiente** con corrección de baja puntuación Dixon-Coles (parámetro `rho_dc`).

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
                                   p_A = 0.5 + penalty_weight × tanh(Δpuntos_FIFA / penalty_k)
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
| `rho_dc` | Corrección Dixon-Coles de baja puntuación (0 = Poisson puro) | 0.08 |
| `simular_tiempo_extra` | Tiempo extra antes de penales (True/False) | True |
| `bonus_campeon_vigente` | Multiplicador para Argentina (campeón 2022) | 1.08 |
| `debutante_mode` | Método de imputación para debutantes | `conf_avg` |
| `n_simulaciones` | Cantidad de torneos a simular | 100 000 |
| `seed` | Semilla aleatoria (None = no reproducible) | 42 |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/vacarezzad/Simulacion-Copa-Mundo-2026.git
cd Simulacion-Copa-Mundo-2026

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar la app
streamlit run app.py
```

**Python requerido:** 3.9+

---

## Estructura del proyecto

```
Simulacion-Copa-Mundo-2026/
│
├── app.py                          # Interfaz Streamlit (UI, tabs, sidebar, bracket)
├── requirements.txt                # Dependencias Python
├── CLAUDE.md                       # Guía para agentes IA que trabajen en el repo
│
├── src/
│   ├── config.py                   # Dataclass SimConfig con todos los parámetros
│   ├── modelo.py                   # Núcleo probabilístico (Poisson + Dixon-Coles + KO)
│   ├── preparacion.py              # Construcción de stats por equipo desde los CSVs
│   ├── torneo.py                   # Simulación de grupos y fases KO; GRUPOS_2026_REALES
│   └── montecarlo.py               # Loop Monte Carlo (secuencial y paralelo)
│
├── Data/
│   ├── clasificados_2026.csv       # Dataset final: 48 equipos con stats y ranking FIFA
│   ├── ranking_fifa.csv            # Ranking FIFA (puntos y posición)
│   ├── equipos_completo.csv        # Historial consolidado por equipo (1930-2022)
│   ├── equipos_historico.csv       # Historial crudo antes de consolidar
│   ├── world_cup_last_50_years.csv # Resumen últimas 50 años
│   └── participaciones_mundial/    # CSVs por edición (1930-2022)
│
└── Scripts de datos (ejecutar en orden para regenerar Data/):
    ├── 01_extraccion.py            # Scraping del ranking FIFA
    ├── 02_consolidacion.py         # Consolida historial de mundiales
    ├── 03_ranking_fifa.py          # Procesa el ranking FIFA
    └── 04_clasificados_2026.py     # Arma el dataset final de 48 equipos
```

---

## Pipeline de datos

Los datos ya vienen incluidos en `Data/`. Para regenerarlos desde cero:

```bash
python 01_extraccion.py         # Descarga ranking FIFA actual (requiere internet)
python 02_consolidacion.py      # Consolida resultados históricos 1930-2022
python 03_ranking_fifa.py       # Procesa y limpia el ranking
python 04_clasificados_2026.py  # Genera clasificados_2026.csv (dataset principal)
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

- **Python 3.9+**
- [Streamlit](https://streamlit.io/) — interfaz web
- [NumPy](https://numpy.org/) — generación de números aleatorios y cálculo vectorizado
- [Pandas](https://pandas.pydata.org/) — carga y manipulación del dataset
- [Plotly](https://plotly.com/python/) — visualizaciones interactivas (barras, heatmaps, bracket)
- [Matplotlib](https://matplotlib.org/) — requerido por Pandas Styler (background_gradient)
- [SciPy](https://scipy.org/) — distribuciones estadísticas

---

## Licencia

MIT
