"""
src/preparacion.py
------------------
Transforma clasificados_2026.csv en un diccionario de stats listas para
el simulador de partidos. Aplica los blends definidos en SimConfig.

Flujo:
  1. Blend histórico + reciente → gf_base, gc_base por equipo
  2. Imputar debutantes (sin historial WC) según debutante_mode
  3. Blend con puntos FIFA según beta_fifa
  4. Normalizar a ataque/defensa relativo al promedio del grupo
  5. Retornar dict {equipo: stats_dict}
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import SimConfig


# ─── Tipo de retorno ──────────────────────────────────────────────────────────
StatsDict = Dict[str, Dict[str, Any]]


def preparar_stats(df: pd.DataFrame, config: SimConfig) -> StatsDict:
    """
    Convierte el DataFrame de 48 clasificados en un dict de stats por equipo.

    Parámetros
    ----------
    df : pd.DataFrame
        Cargado desde Data/clasificados_2026.csv
    config : SimConfig
        Configuración del modelo

    Retorna
    -------
    dict  →  { 'Spain': {'ataque': 1.42, 'defensa': 0.91, 'puntos_fifa': 1877.18,
                          'confederacion': 'UEFA', 'es_local': False}, ... }
    """
    stats: StatsDict = {}

    # ── Paso 1: Blend histórico + reciente ────────────────────────────────────
    for _, row in df.iterrows():
        equipo = row['equipo']

        gf_hist = _safe(row, 'gf_prom_hist')
        gc_hist = _safe(row, 'gc_prom_hist')
        gf_rec  = _safe(row, 'gf_prom_rec')
        gc_rec  = _safe(row, 'gc_prom_rec')

        # Verificar muestra mínima: equipos con pocos partidos recientes
        # usan el historial completo para evitar extremos estadísticos
        # (ej: Netherlands no clasificó en 2018 → solo 2 ediciones recientes)
        partidos_rec = _safe(row, 'partidos_rec')  # columna si existe
        muestra_insuficiente = (
            not np.isnan(partidos_rec) and partidos_rec < config.min_partidos_recientes
        ) if not np.isnan(partidos_rec) else False

        if _ambos_validos(gf_hist, gc_hist, gf_rec, gc_rec) and not muestra_insuficiente:
            gf = config.alpha_reciente * gf_rec  + (1 - config.alpha_reciente) * gf_hist
            gc = config.alpha_reciente * gc_rec  + (1 - config.alpha_reciente) * gc_hist
        elif _ambos_validos(gf_hist, gc_hist):
            gf, gc = gf_hist, gc_hist
        else:
            gf, gc = np.nan, np.nan  # debutante sin historial

        stats[equipo] = {
            'gf_raw'       : gf,
            'gc_raw'       : gc,
            'puntos_fifa'  : _safe(row, 'puntos_fifa'),
            'ranking_fifa' : _safe(row, 'ranking_fifa'),
            'confederacion': str(row.get('confederacion', '')),
            'es_local'     : equipo in config.equipos_locales,
            'debut'        : bool(row.get('debut_2026', False)),
        }

    # ── Paso 2: Imputar debutantes ────────────────────────────────────────────
    _imputar_debutantes(stats, config)

    # ── Paso 3: Blend con puntos FIFA ─────────────────────────────────────────
    if config.beta_fifa > 0:
        _blend_fifa(stats, config)

    # ── Paso 4: Normalizar (ataque / defensa relativo al promedio) ────────────
    valores_gf = [s['gf_raw'] for s in stats.values() if not np.isnan(s['gf_raw'])]
    valores_gc = [s['gc_raw'] for s in stats.values() if not np.isnan(s['gc_raw'])]
    mu_gf = np.mean(valores_gf) if valores_gf else 1.35
    mu_gc = np.mean(valores_gc) if valores_gc else 1.35

    for equipo, s in stats.items():
        gf = s['gf_raw'] if not np.isnan(s['gf_raw']) else mu_gf
        gc = s['gc_raw'] if not np.isnan(s['gc_raw']) else mu_gc

        # Clamp mínimo para evitar λ = 0 en Poisson
        s['ataque']  = max(gf, 0.10)
        s['defensa'] = max(gc, 0.10)
        s['mu_gf']   = mu_gf
        s['mu_gc']   = mu_gc

    return stats


# ─── Helpers internos ─────────────────────────────────────────────────────────

def _safe(row: pd.Series, col: str) -> float:
    """Devuelve float o np.nan si el campo falta o es NaN."""
    val = row.get(col, np.nan)
    return float(val) if pd.notna(val) else np.nan


def _ambos_validos(*vals: float) -> bool:
    return all(not np.isnan(v) for v in vals)


def _imputar_debutantes(stats: StatsDict, config: SimConfig) -> None:
    """
    Rellena gf_raw / gc_raw para equipos sin historial WC.
    Modifica stats in-place.
    """
    tiene_historial = {e: not np.isnan(s['gf_raw']) for e, s in stats.items()}
    debutantes      = [e for e, ok in tiene_historial.items() if not ok]

    if not debutantes:
        return

    mode = config.debutante_mode

    # Promedio global (para fallback)
    todos_gf = [s['gf_raw'] for s in stats.values() if not np.isnan(s['gf_raw'])]
    todos_gc = [s['gc_raw'] for s in stats.values() if not np.isnan(s['gc_raw'])]
    global_gf = np.mean(todos_gf)
    global_gc = np.mean(todos_gc)

    # Promedios por confederación
    conf_stats: Dict[str, Dict] = {}
    for equipo, s in stats.items():
        if not np.isnan(s['gf_raw']):
            conf = s['confederacion']
            if conf not in conf_stats:
                conf_stats[conf] = {'gf': [], 'gc': []}
            conf_stats[conf]['gf'].append(s['gf_raw'])
            conf_stats[conf]['gc'].append(s['gc_raw'])
    conf_avg = {
        c: {'gf': np.mean(v['gf']), 'gc': np.mean(v['gc'])}
        for c, v in conf_stats.items()
    }

    # Parámetros para modo 'puntaje'
    puntos_con_hist = [(s['puntos_fifa'], s['gf_raw'], s['gc_raw'])
                       for s in stats.values()
                       if not np.isnan(s['gf_raw']) and not np.isnan(s['puntos_fifa'])]
    if puntos_con_hist:
        p_arr = np.array([x[0] for x in puntos_con_hist])
        gf_arr = np.array([x[1] for x in puntos_con_hist])
        gc_arr = np.array([x[2] for x in puntos_con_hist])
        # coef. de regresión lineal simple puntos → gf / gc
        coef_gf = np.polyfit(p_arr, gf_arr, 1)
        coef_gc = np.polyfit(p_arr, gc_arr, 1)
    else:
        coef_gf = coef_gc = None

    for equipo in debutantes:
        s    = stats[equipo]
        conf = s['confederacion']
        pts  = s['puntos_fifa']

        if mode == 'conf_avg':
            if conf in conf_avg:
                s['gf_raw'] = conf_avg[conf]['gf']
                s['gc_raw'] = conf_avg[conf]['gc']
            else:
                s['gf_raw'] = global_gf
                s['gc_raw'] = global_gc

        elif mode == 'global_min':
            s['gf_raw'] = min(todos_gf)   # peor ataque del dataset
            s['gc_raw'] = max(todos_gc)   # peor defensa del dataset

        elif mode == 'puntaje':
            if coef_gf is not None and not np.isnan(pts):
                s['gf_raw'] = float(np.polyval(coef_gf, pts))
                s['gc_raw'] = float(np.polyval(coef_gc, pts))
            elif conf in conf_avg:
                s['gf_raw'] = conf_avg[conf]['gf']
                s['gc_raw'] = conf_avg[conf]['gc']
            else:
                s['gf_raw'] = global_gf
                s['gc_raw'] = global_gc

        # Clamp valores negativos que podrían surgir de regresión
        s['gf_raw'] = max(s['gf_raw'], 0.10)
        s['gc_raw'] = max(s['gc_raw'], 0.10)


def _blend_fifa(stats: StatsDict, config: SimConfig) -> None:
    """
    Corrige gf_raw / gc_raw usando los puntos FIFA como señal de fuerza actual.
    La fuerza relativa FIFA escala el ataque y comprime la defensa.
    Modifica stats in-place.
    """
    puntos = [s['puntos_fifa'] for s in stats.values() if not np.isnan(s['puntos_fifa'])]
    if not puntos:
        return
    mu_pts = np.mean(puntos)

    gf_vals = [s['gf_raw'] for s in stats.values()]
    gc_vals = [s['gc_raw'] for s in stats.values()]
    global_gf = np.mean(gf_vals)
    global_gc = np.mean(gc_vals)

    beta = config.beta_fifa

    for equipo, s in stats.items():
        if np.isnan(s['puntos_fifa']):
            continue
        fuerza = s['puntos_fifa'] / mu_pts   # >1 para equipos fuertes, <1 para débiles

        # Equipo más fuerte que el promedio FIFA → más ataque, menos defensa (menos goles concedidos)
        gf_fifa = global_gf * fuerza
        gc_fifa = global_gc / fuerza

        s['gf_raw'] = (1 - beta) * s['gf_raw'] + beta * gf_fifa
        s['gc_raw'] = (1 - beta) * s['gc_raw'] + beta * gc_fifa

        # Clamp
        s['gf_raw'] = max(s['gf_raw'], 0.10)
        s['gc_raw'] = max(s['gc_raw'], 0.10)
