"""
src/modelo.py
-------------
Núcleo probabilístico del simulador.

Funciones principales
---------------------
  calcular_lambda()    → goles esperados de un equipo vs otro
  simular_partido()    → resultado en 90' (goles A, goles B)
  simular_tiempo_extra()→ resultado en ET (30 min adicionales)
  simular_penales()    → ganador en penales (sesgado por FIFA pts)
  resolver_ko()        → ganador de un partido KO completo
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .config import SimConfig


# ─── Tipos ───────────────────────────────────────────────────────────────────
Stats     = Dict[str, Any]   # stats de un equipo (del dict preparado en preparacion.py)
StatsDict = Dict[str, Stats]


# ─── Lambda (goles esperados) ─────────────────────────────────────────────────

def calcular_lambda(
    stats_a: Stats,
    stats_b: Stats,
    config: SimConfig,
    es_local_a: bool = False,
) -> float:
    """
    Calcula λ_A (goles esperados del equipo A contra B) usando la fórmula
    Dixon-Coles simplificada:

        λ_A = base_goles × (ataque_A / μ_gf) × (μ_gc / defensa_B) × factor_local_A

    Parámetros
    ----------
    stats_a      : stats del equipo atacante (ataque, defensa, mu_gf, mu_gc)
    stats_b      : stats del equipo defensor
    config       : SimConfig
    es_local_a   : True si A juega de local (USA, CAN, MEX)

    Retorna
    -------
    float → λ ≥ 0.05 (clamp para evitar Poisson degenerado)
    """
    mu_gf   = stats_a['mu_gf']
    mu_gc   = stats_a['mu_gc']

    ataque_a  = stats_a['ataque']
    defensa_b = stats_b['defensa']

    # defensa_b = gc promedio del equipo B (más alto → peor defensa → más fácil anotar)
    lam = (
        config.base_goles
        * (ataque_a  / mu_gf)
        * (defensa_b / mu_gc)
    )

    if es_local_a:
        lam *= config.factor_local

    return max(lam, 0.05)


def aplicar_bonus_campeon(
    equipo: str,
    lam: float,
    config: SimConfig,
) -> float:
    """Aplica el bonus de campeón vigente al lambda de ataque."""
    if equipo == config.campeon_vigente:
        return lam * config.bonus_campeon_vigente
    return lam


# ─── Corrección Dixon-Coles ──────────────────────────────────────────────────

def _simular_dc(
    lam_a: float,
    lam_b: float,
    rho: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    Muestrea (ga, gb) de la distribución Poisson conjunta corregida por
    Dixon-Coles (1997) mediante rejection sampling.

    La corrección τ(ga, gb) ajusta solo los marcadores bajos (ga, gb ≤ 1):
        τ(0,0) = 1 − λ_a × λ_b × ρ
        τ(1,0) = 1 + λ_b × ρ
        τ(0,1) = 1 + λ_a × ρ
        τ(1,1) = 1 − ρ
        τ(x,y) = 1   si x ≥ 2 o y ≥ 2

    Para ρ ≈ 0.08 y λ ≈ 1.35, M ≈ 1.10 → eficiencia > 90 %.
    """
    tau_00 = 1.0 - lam_a * lam_b * rho
    tau_10 = 1.0 + lam_b * rho
    tau_01 = 1.0 + lam_a * rho
    tau_11 = 1.0 - rho
    M = max(tau_00, tau_10, tau_01, tau_11, 1.0)

    while True:
        ga = int(rng.poisson(lam_a))
        gb = int(rng.poisson(lam_b))

        if ga > 1 or gb > 1:
            tau = 1.0
        elif ga == 0 and gb == 0:
            tau = tau_00
        elif ga == 1 and gb == 0:
            tau = tau_10
        elif ga == 0 and gb == 1:
            tau = tau_01
        else:                        # ga == 1 and gb == 1
            tau = tau_11

        if rng.random() < tau / M:
            return ga, gb


# ─── Simulación de 90 minutos ─────────────────────────────────────────────────

def simular_partido(
    equipo_a: str,
    equipo_b: str,
    stats: StatsDict,
    config: SimConfig,
    rng: np.random.Generator,
    local: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Simula 90 minutos. Retorna (goles_a, goles_b).

    Parámetros
    ----------
    equipo_a / equipo_b : nombres de los equipos
    stats               : dict global de stats (salida de preparar_stats)
    config              : SimConfig
    rng                 : generador de números aleatorios (para reproducibilidad)
    local               : nombre del equipo local (o None si cancha neutral)

    Retorna
    -------
    (int, int) → (goles_A, goles_B)
    """
    sa = stats[equipo_a]
    sb = stats[equipo_b]

    es_local_a = (local == equipo_a)
    es_local_b = (local == equipo_b)

    lam_a = calcular_lambda(sa, sb, config, es_local_a)
    lam_b = calcular_lambda(sb, sa, config, es_local_b)

    lam_a = aplicar_bonus_campeon(equipo_a, lam_a, config)
    lam_b = aplicar_bonus_campeon(equipo_b, lam_b, config)

    if config.rho_dc == 0.0:
        goles_a = int(rng.poisson(lam_a))
        goles_b = int(rng.poisson(lam_b))
    else:
        goles_a, goles_b = _simular_dc(lam_a, lam_b, config.rho_dc, rng)

    return goles_a, goles_b


# ─── Tiempo extra (30 min) ────────────────────────────────────────────────────

def simular_tiempo_extra(
    equipo_a: str,
    equipo_b: str,
    stats: StatsDict,
    config: SimConfig,
    rng: np.random.Generator,
    local: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Simula 30 minutos de tiempo extra.
    Usa λ escalado a 30/90 ≈ 0.333 del partido regular.

    Retorna
    -------
    (int, int) → goles adicionales en ET
    """
    sa = stats[equipo_a]
    sb = stats[equipo_b]

    escala_et = 30 / 90  # tiempo extra = 1/3 del partido

    es_local_a = (local == equipo_a)
    es_local_b = (local == equipo_b)

    lam_a = calcular_lambda(sa, sb, config, es_local_a) * escala_et
    lam_b = calcular_lambda(sb, sa, config, es_local_b) * escala_et

    lam_a = aplicar_bonus_campeon(equipo_a, lam_a, config)
    lam_b = aplicar_bonus_campeon(equipo_b, lam_b, config)

    goles_a = int(rng.poisson(lam_a))
    goles_b = int(rng.poisson(lam_b))

    return goles_a, goles_b


# ─── Penales ──────────────────────────────────────────────────────────────────

def simular_penales(
    equipo_a: str,
    equipo_b: str,
    stats: StatsDict,
    config: SimConfig,
    rng: np.random.Generator,
) -> str:
    """
    Determina el ganador de una tanda de penales.

    Si penalty_weight = 0  → 50-50 puro.
    Si penalty_weight > 0  → sesgo hacia el equipo con más puntos FIFA,
                              usando: p_A = 0.5 + w × tanh(Δ_puntos / k)

    Retorna
    -------
    str → nombre del equipo ganador
    """
    sa = stats[equipo_a]
    sb = stats[equipo_b]

    pts_a = sa.get('puntos_fifa', np.nan)
    pts_b = sb.get('puntos_fifa', np.nan)

    if config.penalty_weight > 0 and not (np.isnan(pts_a) or np.isnan(pts_b)):
        delta = pts_a - pts_b
        p_a   = 0.5 + config.penalty_weight * np.tanh(delta / config.penalty_k)
        p_a   = np.clip(p_a, 0.05, 0.95)  # nunca 0 ni 1
    else:
        p_a = 0.5

    ganador = equipo_a if rng.random() < p_a else equipo_b
    return ganador


# ─── Resolver partido KO completo ─────────────────────────────────────────────

def resolver_ko(
    equipo_a: str,
    equipo_b: str,
    stats: StatsDict,
    config: SimConfig,
    rng: np.random.Generator,
    local: Optional[str] = None,
) -> Tuple[str, str, Dict]:
    """
    Simula un partido de eliminación directa:
      1. 90 minutos
      2. Si hay empate y config.simular_tiempo_extra: 30 min ET
      3. Si sigue empatado: penales

    Retorna
    -------
    ganador   : str → nombre del equipo que avanza
    eliminado : str → nombre del equipo que sale
    detalle   : dict → {goles_a, goles_b, metodo: '90min' | 'ET' | 'Penales'}
    """
    # ─ 90 minutos ─
    g_a, g_b = simular_partido(equipo_a, equipo_b, stats, config, rng, local)

    if g_a != g_b:
        ganador   = equipo_a if g_a > g_b else equipo_b
        eliminado = equipo_b if g_a > g_b else equipo_a
        return ganador, eliminado, {'goles_a': g_a, 'goles_b': g_b, 'metodo': '90min'}

    # ─ Tiempo extra ─
    if config.simular_tiempo_extra:
        et_a, et_b = simular_tiempo_extra(equipo_a, equipo_b, stats, config, rng, local)
        g_a += et_a
        g_b += et_b

        if g_a != g_b:
            ganador   = equipo_a if g_a > g_b else equipo_b
            eliminado = equipo_b if g_a > g_b else equipo_a
            return ganador, eliminado, {'goles_a': g_a, 'goles_b': g_b, 'metodo': 'ET'}

    # ─ Penales ─
    ganador   = simular_penales(equipo_a, equipo_b, stats, config, rng)
    eliminado = equipo_b if ganador == equipo_a else equipo_a
    return ganador, eliminado, {'goles_a': g_a, 'goles_b': g_b, 'metodo': 'Penales'}
