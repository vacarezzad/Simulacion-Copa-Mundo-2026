"""
src/config.py
-------------
Dataclass central con todos los parámetros del modelo.
Se instancia en app.py (Streamlit) desde los widgets del sidebar
y se pasa a todas las funciones del pipeline de simulación.

Cada parámetro tiene:
  - valor default razonable
  - rango recomendado en el comentario (para los sliders de Streamlit)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SimConfig:

    # ─── Blend histórico vs reciente ──────────────────────────────────────────
    # ¿Cuánto peso tienen las últimas 3 ediciones (2014-2022) vs todo el historial?
    # Streamlit: st.slider(0.0, 1.0, step=0.05)
    alpha_reciente: float = 0.6

    # ─── Blend estadísticas vs puntos FIFA ────────────────────────────────────
    # ¿Cuánto corrige el ranking FIFA actual los promedios históricos de goles?
    # 0.0 → solo historial  |  1.0 → solo ranking FIFA
    # Default sube a 0.55 para corregir el sesgo histórico de equipos con pocos
    # Mundiales recientes (ej: Netherlands no clasificó 2018 → muestra chica).
    # Streamlit: st.slider(0.0, 1.0, step=0.05)
    beta_fifa: float = 0.55

    # ─── Muestra mínima para datos recientes ──────────────────────────────────
    # Equipos con menos de N partidos recientes (2014-2022) usan el historial
    # completo en lugar de los datos recientes, para evitar extremos estadísticos.
    # Netherlands no clasificó en 2018: solo tiene 2 ediciones recientes.
    # Streamlit: st.slider(3, 15, step=1)
    min_partidos_recientes: int = 6

    # ─── Parámetro base de goles ──────────────────────────────────────────────
    # Media de goles por equipo por partido en mundiales (~1.35 en datos reales).
    # Sube → más goles en general → más varianza → mayor incertidumbre.
    # Streamlit: st.slider(1.0, 1.8, step=0.05)
    base_goles: float = 1.35

    # ─── Ventaja de sede ──────────────────────────────────────────────────────
    # Multiplicador sobre λ para USA, Canadá y México.
    # 1.0 → sin ventaja  |  1.15 → +15 % en goles esperados
    # Streamlit: st.slider(1.0, 1.20, step=0.01)
    factor_local: float = 1.08
    equipos_locales: tuple = ('USA', 'Canada', 'Mexico')

    # ─── Penales ──────────────────────────────────────────────────────────────
    # ¿Cuánto influye la diferencia de puntos FIFA en la prob. de ganar penales?
    # 0.0 → moneda al aire (50-50)
    # 0.3 → ventaja notable al más fuerte
    # Fórmula: p_A = 0.5 + penalty_weight × tanh(Δpuntos / penalty_k)
    # Streamlit: st.slider(0.0, 0.40, step=0.05)
    penalty_weight: float = 0.15

    # Escala de la diferencia de puntos FIFA en la función tanh.
    # Mayor k → la diferencia de puntos impacta menos en penales.
    # Streamlit: st.slider(200, 600, step=50)
    penalty_k: float = 400.0

    # ─── Tiempo extra ─────────────────────────────────────────────────────────
    # True  → simular 30 min adicionales con λ reducido (~35 %) antes de penales
    # False → empate en 90' va directo a penales
    # Streamlit: st.checkbox
    simular_tiempo_extra: bool = True

    # ─── Bonus campeón vigente ────────────────────────────────────────────────
    # Multiplicador adicional sobre λ_ataque del campeón vigente (Argentina 2022).
    # 1.0 → sin bonus  |  1.10 → +10% en goles esperados por partido
    # Refleja la confianza y rodaje de un equipo recién campeón del mundo.
    # Streamlit: st.slider(1.0, 1.20, step=0.01)
    bonus_campeon_vigente: float = 1.08
    campeon_vigente: str = 'Argentina'

    # ─── Imputación para debutantes ───────────────────────────────────────────
    # Equipos sin historial en Mundiales (Curacao, Jordan, Cape Verde, Uzbekistan)
    # 'conf_avg'  → promedio de ataque/defensa de su confederación
    # 'global_min'→ el ataque más débil y la defensa más porosa del dataset
    # 'puntaje'   → interpolación lineal desde puntos FIFA
    # Streamlit: st.selectbox
    debutante_mode: Literal['conf_avg', 'global_min', 'puntaje'] = 'conf_avg'

    # ─── Monte Carlo ──────────────────────────────────────────────────────────
    # Número de torneos completos a simular.
    # 10k → ~3 seg  |  100k → ~30 seg  |  500k → ~2.5 min
    # Streamlit: st.select_slider([10_000, 50_000, 100_000, 250_000, 500_000])
    n_simulaciones: int = 100_000

    # Semilla para reproducibilidad (None → aleatorio en cada corrida)
    # Streamlit: st.number_input(0, 9999) + checkbox "Usar semilla fija"
    seed: Optional[int] = 42

    # ─── Dixon-Coles ──────────────────────────────────────────────────────────
    # ρ > 0 → más resultados 1-0/0-1, menos 0-0/1-1 (empírico en mundiales)
    # ρ = 0 → Poisson puro (sin corrección, comportamiento anterior)
    # Rango recomendado: 0.00 – 0.20. Default empírico: ~0.08 para mundiales.
    # Streamlit: st.slider(0.00, 0.20, step=0.01)
    rho_dc: float = 0.08
