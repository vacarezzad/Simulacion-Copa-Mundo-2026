"""
src/montecarlo.py
-----------------
Loop principal del Monte Carlo. Corre N torneos completos y agrega
los resultados en probabilidades por equipo, etapa y escenarios frecuentes.

Funciones
---------
  correr_simulacion()          → (df_prob, dict_escenarios)   [secuencial]
  correr_simulacion_paralelo() → (df_prob, dict_escenarios)   [multi-core]
  resumen_texto()              → str con top 10 favoritos

Qué se guarda en escenarios
---------------------------
  'finales'            → Counter {(campeon, finalista): n}
  'semifinales'        → Counter {frozenset(4 semis): n}
  'ganadores_grupo'    → {letra: Counter {equipo: n}}
  'segundos_grupo'     → {letra: Counter {equipo: n}}
  'campeon_por_conf'   → Counter {confederacion: n}
  'resultados_grupos'  → {(ta, tb): Counter {(ga, gb): n}}   ← NUEVO
                         72 duelos fijos; ta < tb (orden alfabético)
  'resultados_ko'      → {ronda: {(ta,tb): Counter {(ga,gb,metodo): n}}}
                         r32/r16/cuartos/semis/final; ta < tb ← NUEVO
  'n'                  → total simulaciones

Normalización de clave de partido
----------------------------------
Para comparar resultados sin importar el orden en que se pasan los equipos,
siempre se ordena (ta, tb) alfabéticamente y el score se ajusta en consecuencia:
  ta = min(ea, eb), ga = goles del equipo min
  tb = max(ea, eb), gb = goles del equipo max
"""

import os
import pickle
import dataclasses
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List

from .config import SimConfig
from .preparacion import preparar_stats
from .torneo import simular_torneo, asignar_grupos, ETAPAS, GRUPOS, RONDAS_KO


# ─── Orden de etapas (de peor a mejor) ───────────────────────────────────────
ORDEN_ETAPAS = {e: i for i, e in enumerate(ETAPAS)}

_ETAPAS_HASTA: Dict[str, list] = {
    etapa_max: [e for e, o in ORDEN_ETAPAS.items() if o <= ORDEN_ETAPAS[etapa_max]]
    for etapa_max in ETAPAS
}


# ─── Helpers de normalización ─────────────────────────────────────────────────

def _partido_key(ta: str, tb: str, ga: int, gb: int):
    """Retorna (key, ga_norm, gb_norm) con key=(ta,tb) en orden alfabético."""
    if ta <= tb:
        return (ta, tb), ga, gb
    return (tb, ta), gb, ga


def _inicializar_res_grupos(grupos_dict: Dict) -> Dict:
    """Pre-crea los 72 Counters para los partidos de la fase de grupos."""
    res: Dict[Tuple, Counter] = {}
    from itertools import combinations
    for eqs in grupos_dict.values():
        for ta, tb in combinations(eqs, 2):
            key = (ta, tb) if ta <= tb else (tb, ta)
            res[key] = Counter()
    return res


def _inicializar_res_ko() -> Dict:
    return {ronda: defaultdict(Counter) for ronda in RONDAS_KO}


def _merge_res_grupos(dst: Dict, src: Dict) -> None:
    """Suma Counter de resultados de grupos in-place (dst += src)."""
    for key, cnt in src.items():
        if key in dst:
            dst[key] += cnt
        else:
            dst[key] = Counter(cnt)


def _merge_res_ko(dst: Dict, src: Dict) -> None:
    """Suma Counter de resultados KO in-place."""
    for ronda in RONDAS_KO:
        for key, cnt in src[ronda].items():
            dst[ronda][key] += cnt


def _inicializar_bracket_slots() -> Dict:
    """Crea listas de Counters para rastrear qué equipos aparecen en cada slot KO."""
    sizes = [('r32', 16), ('r16', 8), ('cuartos', 4), ('semis', 2), ('final', 1)]
    return {ronda: [Counter() for _ in range(n)] for ronda, n in sizes}


def _merge_bracket_slots(dst: Dict, src: Dict) -> None:
    """Suma bracket_slots in-place (dst += src)."""
    for ronda in dst:
        for i in range(len(dst[ronda])):
            dst[ronda][i] += src[ronda][i]


# ─── Worker (nivel de módulo → picklable para ProcessPoolExecutor) ────────────

def _worker_chunk(packed_args: tuple) -> dict:
    """
    Función de nivel de módulo que corre n_chunk torneos y devuelve
    contadores parciales (incluyendo resultados de partidos).
    """
    df_pickle, config_dict, n_chunk, seed = packed_args

    df     = pickle.loads(df_pickle)
    config = SimConfig(**config_dict)
    config.n_simulaciones = n_chunk
    config.seed           = seed

    stats    = preparar_stats(df, config)
    rng      = np.random.default_rng(seed)
    conf_map = dict(zip(df['equipo'], df['confederacion']))
    equipos  = df['equipo'].tolist()

    grupos_dict = asignar_grupos(equipos)

    # Contadores de etapas
    conteos                 = {eq: {e: 0 for e in ETAPAS} for eq in equipos}
    finales_counter         = Counter()
    semifinales_counter     = Counter()
    campeon_conf_counter    = Counter()
    ganadores_grupo_counter = {g: Counter() for g in GRUPOS}
    segundos_grupo_counter  = {g: Counter() for g in GRUPOS}

    # Contadores de resultados de partidos
    res_grupos    = _inicializar_res_grupos(grupos_dict)
    res_ko        = _inicializar_res_ko()
    bracket_slots = _inicializar_bracket_slots()

    for _ in range(n_chunk):
        torneo = simular_torneo(df, stats, config, rng, grupos_dict=grupos_dict)

        # Etapas
        for eq, etapa_max in torneo['etapas'].items():
            for etapa in _ETAPAS_HASTA[etapa_max]:
                conteos[eq][etapa] += 1

        campeon   = torneo['campeon']
        finalista = torneo['finalista']
        finales_counter[(campeon, finalista)] += 1
        semifinales_counter[frozenset([campeon, finalista] + torneo['semifinalistas'])] += 1
        campeon_conf_counter[conf_map.get(campeon, '?')] += 1

        for letra, eq in torneo['ganadores_grupo'].items():
            ganadores_grupo_counter[letra][eq] += 1
        for letra, eq in torneo['segundos_grupo'].items():
            segundos_grupo_counter[letra][eq] += 1

        # Resultados de partidos — fase de grupos
        rp = torneo['resultados_partidos']
        for ta, tb, ga, gb in rp['grupos']:
            key, ga_n, gb_n = _partido_key(ta, tb, ga, gb)
            res_grupos[key][(ga_n, gb_n)] += 1

        # Resultados de partidos — fases KO
        for ronda in RONDAS_KO:
            for i, (ta, tb, ga, gb, metodo, ganador) in enumerate(rp[ronda]):
                key, ga_n, gb_n = _partido_key(ta, tb, ga, gb)
                ta_wins = (ganador == key[0])
                res_ko[ronda][key][(ga_n, gb_n, metodo, ta_wins)] += 1
                # Bracket slots: qué equipos aparecen en cada slot
                bracket_slots[ronda][i][ta] += 1
                bracket_slots[ronda][i][tb] += 1

    return {
        'conteos'        : conteos,
        'finales'        : finales_counter,
        'semifinales'    : semifinales_counter,
        'campeon_conf'   : campeon_conf_counter,
        'ganadores_grupo': ganadores_grupo_counter,
        'segundos_grupo' : segundos_grupo_counter,
        'res_grupos'     : res_grupos,
        'res_ko'         : res_ko,
        'bracket_slots'  : bracket_slots,
    }


# ─── Helper: construir DataFrame de resultados ────────────────────────────────

def _construir_df_resultado(
    equipos: list,
    conteos: dict,
    n: int,
    clasificados_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for eq in equipos:
        row = {'equipo': eq}
        for etapa in ETAPAS:
            row[f'p_{etapa}'] = conteos[eq][etapa] / n
        rows.append(row)

    df_result = pd.DataFrame(rows)
    meta_cols = ['equipo', 'confederacion', 'puntos_fifa', 'ranking_fifa',
                 'veces_campeon', 'debut_2026']
    meta_cols = [c for c in meta_cols if c in clasificados_df.columns]
    df_result = df_result.merge(clasificados_df[meta_cols], on='equipo', how='left')
    df_result = df_result.sort_values('p_campeon', ascending=False).reset_index(drop=True)
    df_result.index += 1
    return df_result


# ─── Versión secuencial ───────────────────────────────────────────────────────

def correr_simulacion(
    clasificados_df: pd.DataFrame,
    config: SimConfig,
    callback=None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Corre `config.n_simulaciones` torneos (modo secuencial).
    Retorna (df_prob, escenarios).
    """
    stats    = preparar_stats(clasificados_df, config)
    conf_map = dict(zip(clasificados_df['equipo'], clasificados_df['confederacion']))
    equipos  = clasificados_df['equipo'].tolist()

    grupos_dict = asignar_grupos(equipos)

    conteos                 = {eq: {e: 0 for e in ETAPAS} for eq in equipos}
    finales_counter         = Counter()
    semifinales_counter     = Counter()
    campeon_conf_counter    = Counter()
    ganadores_grupo_counter = {g: Counter() for g in GRUPOS}
    segundos_grupo_counter  = {g: Counter() for g in GRUPOS}

    res_grupos    = _inicializar_res_grupos(grupos_dict)
    res_ko        = _inicializar_res_ko()
    bracket_slots = _inicializar_bracket_slots()

    rng = np.random.default_rng(config.seed)
    n   = config.n_simulaciones

    for i in range(n):
        torneo = simular_torneo(clasificados_df, stats, config, rng, grupos_dict=grupos_dict)

        for eq, etapa_max in torneo['etapas'].items():
            for etapa in _ETAPAS_HASTA[etapa_max]:
                conteos[eq][etapa] += 1

        campeon   = torneo['campeon']
        finalista = torneo['finalista']
        finales_counter[(campeon, finalista)] += 1
        semifinales_counter[frozenset([campeon, finalista] + torneo['semifinalistas'])] += 1
        campeon_conf_counter[conf_map.get(campeon, '?')] += 1

        for letra, eq in torneo['ganadores_grupo'].items():
            ganadores_grupo_counter[letra][eq] += 1
        for letra, eq in torneo['segundos_grupo'].items():
            segundos_grupo_counter[letra][eq] += 1

        rp = torneo['resultados_partidos']
        for ta, tb, ga, gb in rp['grupos']:
            key, ga_n, gb_n = _partido_key(ta, tb, ga, gb)
            res_grupos[key][(ga_n, gb_n)] += 1

        for ronda in RONDAS_KO:
            for j, (ta, tb, ga, gb, metodo, ganador) in enumerate(rp[ronda]):
                key, ga_n, gb_n = _partido_key(ta, tb, ga, gb)
                ta_wins = (ganador == key[0])
                res_ko[ronda][key][(ga_n, gb_n, metodo, ta_wins)] += 1
                bracket_slots[ronda][j][ta] += 1
                bracket_slots[ronda][j][tb] += 1

        if callback and (i + 1) % max(1, n // 100) == 0:
            callback(i + 1, n)

    df_result = _construir_df_resultado(equipos, conteos, n, clasificados_df)

    escenarios = {
        'finales'          : finales_counter,
        'semifinales'      : semifinales_counter,
        'ganadores_grupo'  : ganadores_grupo_counter,
        'segundos_grupo'   : segundos_grupo_counter,
        'campeon_por_conf' : campeon_conf_counter,
        'resultados_grupos': res_grupos,
        'resultados_ko'    : res_ko,
        'bracket_slots'    : bracket_slots,
        'n'                : n,
    }
    return df_result, escenarios


# ─── Versión paralela (multi-core) ────────────────────────────────────────────

def correr_simulacion_paralelo(
    clasificados_df: pd.DataFrame,
    config: SimConfig,
    callback=None,
    n_workers: int = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Corre `config.n_simulaciones` torneos en paralelo.
    Misma interfaz que correr_simulacion().
    """
    if n_workers is None or n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 8)

    n = config.n_simulaciones

    base      = n // n_workers
    remainder = n % n_workers
    chunks    = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    df_pickle   = pickle.dumps(clasificados_df, protocol=pickle.HIGHEST_PROTOCOL)
    master_rng  = np.random.default_rng(config.seed)
    seeds       = master_rng.integers(0, 2**31, size=n_workers).tolist()
    config_dict = dataclasses.asdict(config)

    packed_args = [
        (df_pickle, config_dict, chunk_n, int(seed))
        for chunk_n, seed in zip(chunks, seeds)
    ]

    equipos = clasificados_df['equipo'].tolist()
    grupos_dict = asignar_grupos(equipos)

    # Acumuladores globales
    conteos_total         = {eq: {e: 0 for e in ETAPAS} for eq in equipos}
    finales_total         = Counter()
    semifinales_total     = Counter()
    campeon_conf_total    = Counter()
    ganadores_grupo_total = {g: Counter() for g in GRUPOS}
    segundos_grupo_total  = {g: Counter() for g in GRUPOS}
    res_grupos_total      = _inicializar_res_grupos(grupos_dict)
    res_ko_total          = _inicializar_res_ko()
    bracket_slots_total   = _inicializar_bracket_slots()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures     = [executor.submit(_worker_chunk, args) for args in packed_args]
        completadas = 0

        for future in as_completed(futures):
            result = future.result()

            for eq in equipos:
                for etapa in ETAPAS:
                    conteos_total[eq][etapa] += result['conteos'][eq][etapa]

            finales_total        += result['finales']
            semifinales_total    += result['semifinales']
            campeon_conf_total   += result['campeon_conf']

            for g in GRUPOS:
                ganadores_grupo_total[g] += result['ganadores_grupo'][g]
                segundos_grupo_total[g]  += result['segundos_grupo'][g]

            _merge_res_grupos(res_grupos_total, result['res_grupos'])
            _merge_res_ko(res_ko_total, result['res_ko'])
            _merge_bracket_slots(bracket_slots_total, result['bracket_slots'])

            completadas += 1
            if callback:
                callback(completadas, n_workers)

    df_result = _construir_df_resultado(equipos, conteos_total, n, clasificados_df)

    escenarios = {
        'finales'          : finales_total,
        'semifinales'      : semifinales_total,
        'ganadores_grupo'  : ganadores_grupo_total,
        'segundos_grupo'   : segundos_grupo_total,
        'campeon_por_conf' : campeon_conf_total,
        'resultados_grupos': res_grupos_total,
        'resultados_ko'    : res_ko_total,
        'bracket_slots'    : bracket_slots_total,
        'n'                : n,
    }
    return df_result, escenarios


# ─── Resumen de texto ─────────────────────────────────────────────────────────

def resumen_texto(df: pd.DataFrame) -> str:
    """Top 10 favoritos en texto plano."""
    lineas = ["🏆 TOP 10 FAVORITOS — Mundial 2026\n"]
    for i, row in df.head(10).iterrows():
        lineas.append(
            f"  {i:>2}. {row['equipo']:<15} "
            f"Campeón: {row['p_campeon']:>6.2%}  "
            f"Final: {row['p_final']:>6.2%}  "
            f"Semis: {row['p_semis']:>6.2%}"
        )
    return "\n".join(lineas)
