# CLAUDE.md — Guía para agentes IA

Este archivo orienta a cualquier agente IA (Claude, Copilot, Gemini, etc.) que trabaje en este repositorio.

---

## ¿Qué es este proyecto?

Simulador Monte Carlo del **FIFA World Cup 2026** (48 equipos, formato USA/Canadá/México).
La app es una interfaz **Streamlit** (`app.py`) que expone todos los parámetros del modelo y muestra los resultados en 7 pestañas interactivas.

---

## Cómo levantar el servidor local

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Python requerido: 3.9+**
> Si el servidor ya está corriendo, matalo con `pkill -f streamlit` antes de reiniciar.

---

## Arquitectura del código

```
app.py          ← punto de entrada; sidebar + 7 tabs + bracket visual
src/
  config.py     ← SimConfig (dataclass con todos los parámetros del modelo)
  preparacion.py← construye ataque/defensa por equipo desde Data/
  modelo.py     ← simula un partido (Poisson + Dixon-Coles + KO/penales)
  torneo.py     ← simula un torneo completo (grupos + llaves KO); contiene GRUPOS_2026_REALES
  montecarlo.py ← corre N torneos (secuencial o paralelo con ProcessPoolExecutor)
```

### Flujo de datos

```
Data/clasificados_2026.csv
        │
        ▼
src/preparacion.py  →  df_clasif  (DataFrame con ataque/defensa/puntos_fifa por equipo)
        │
        ▼
src/montecarlo.py   →  (df_result, escenarios)
        │                ├── df_result: probabilidades por equipo y ronda
        │                └── escenarios: bracket_slots, campeón_freq, matchup_freq, etc.
        ▼
app.py              →  visualizaciones en las 7 pestañas
```

---

## Archivos clave

| Archivo | Qué toca cuando… |
|---|---|
| `src/config.py` | Agregás/modificás un parámetro del modelo |
| `src/modelo.py` | Cambiás la lógica de simulación de un partido |
| `src/torneo.py` | Cambiás el formato del torneo (grupos, llaves KO) o los equipos clasificados |
| `src/preparacion.py` | Cambiás cómo se calculan ataque/defensa desde los datos históricos |
| `src/montecarlo.py` | Cambiás el loop principal o el output de la simulación |
| `app.py` | Cambiás la UI, agregás una pestaña, modificás el bracket visual |

---

## Agregar un nuevo parámetro al modelo

1. **`src/config.py`** — agregarlo como atributo de `SimConfig` con su valor default y comentario de rango.
2. **`src/modelo.py` o `src/preparacion.py`** — usarlo donde corresponda.
3. **`app.py` → sidebar** — agregar el widget de Streamlit (slider, checkbox, selectbox) en la sección del sidebar. Los parámetros se construyen como `SimConfig(param=valor, ...)` en la línea ~187.

---

## Bracket visual

- Función: `_crear_bracket_full(escenarios, df_prob, COLORES_CONF, conf_map)` en `app.py` (~línea 630).
- Devuelve un `go.Figure` de Plotly con coordenadas cartesianas (no axes).
- Se renderiza con `streamlit.components.v1.html(full_html=True, scrolling=True)` para habilitar scroll horizontal y zoom.
- El ancho se calcula automáticamente: `FIG_W = int(FIG_H * x_span / y_span)` (~2400 px para 48 equipos).
- Para ajustar el espaciado: modificar `STEP` (distancia entre rondas) y `BOX_W` (ancho de cada caja) cerca de la línea 716.

---

## Convenciones del código

- **Todo en español** (variables, comentarios, UI, mensajes de error).
- Los DataFrames de resultados tienen columnas: `equipo`, `confederacion`, `puntos_fifa`, `p_grupos`, `p_r32`, `p_cuartos`, `p_semis`, `p_final`, `p_campeon`.
- `escenarios` es un dict con claves: `n`, `campeon_freq`, `bracket_slots`, `match_freq_grupos`, `match_freq_ko`, `score_freq_grupos`, `score_freq_ko`.
- Los colores de confederación se definen en `COLORES_CONF` al inicio de `app.py` (~línea 35).

---

## Dataset principal

`Data/clasificados_2026.csv` — columnas clave:

| Columna | Descripción |
|---|---|
| `equipo` | Nombre del equipo (en inglés, consistente con los CSVs históricos) |
| `confederacion` | UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC |
| `puntos_fifa` | Puntos FIFA al momento de la clasificación |
| `gf_hist` / `gc_hist` | Promedio histórico de goles a favor/en contra |
| `gf_rec` / `gc_rec` | Promedio reciente (2014-2022) de goles a favor/en contra |
| `partidos_rec` | Partidos jugados en las últimas 3 ediciones |
| `es_local` | 1 si es sede (USA, Canada, Mexico) |

---

## Dependencias

```
pandas>=2.0
numpy>=1.24
requests>=2.31
streamlit>=1.32
plotly>=5.18
matplotlib>=3.7   # requerido por pandas Styler (background_gradient)
scipy>=1.11
```

---

## Gotchas / cosas a tener en cuenta

- **`matplotlib` es requerido** aunque no se importa directamente: `st.dataframe(df.style.background_gradient(...))` lo necesita internamente.
- **El bracket usa `components.v1.html` con `full_html=True`**: incluye Plotly.js vía CDN (`include_plotlyjs='cdn'`). Requiere conexión a internet la primera vez que se carga.
- **`ProcessPoolExecutor` en Windows**: si se corre en Windows, hay que proteger el entry point con `if __name__ == '__main__':` o usar `n_workers=1`.
- **`GRUPOS_2026_REALES`** en `src/torneo.py` es un dict con los 12 grupos del Mundial 2026 (formato real FIFA). Cualquier cambio de equipos debe hacerse ahí.
- **Semilla**: con `seed=42` los resultados son reproducibles. Con `seed=None` cada corrida es diferente.
- **Debutantes**: Curacao, Jordan, Cape Verde, Uzbekistán no tienen historial en los CSVs históricos y son imputados según `debutante_mode`.
