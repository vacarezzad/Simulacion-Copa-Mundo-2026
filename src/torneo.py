"""
src/torneo.py
-------------
Simula un torneo completo del Mundial 2026.

Formato 2026
------------
  · 48 equipos → 12 grupos (A-L) de 4 equipos
  · Round-robin por grupo (6 partidos): W=3, D=1, L=0
  · Clasifican: 1° y 2° de cada grupo (24) + 8 mejores 3° (8) = 32
  · Round of 32  → 16 ganadores
  · Round of 16  → 8 ganadores
  · Cuartos      → 4 ganadores
  · Semifinales  → 2 ganadores
  · Final        → 1 campeón

Resultados de partidos
----------------------
  simular_grupo()     devuelve (standings, partidos_grupo)
  simular_ronda_ko()  devuelve (ganadores, partidos_ko)
  simular_torneo()    recopila todo en 'resultados_partidos':
    {
      'grupos': [(ta, tb, ga, gb), ...],         # 72 partidos
      'r32':    [(ta, tb, ga, gb, metodo), ...],  # 16 partidos
      'r16':    [...],                            # 8 partidos
      'cuartos':[(...)],                          # 4 partidos
      'semis':  [(...)],                          # 2 partidos
      'final':  [(...)],                          # 1 partido
    }

Optimizaciones de rendimiento
------------------------------
  - simular_grupo() usa sorted() con clave de tuplas en lugar de
    pandas.sort_values(). Elimina el 77 % de overhead detectado por cProfile.
  - seleccionar_mejores_terceros() ídem: sorted() puro.
  - simular_torneo() acepta grupos_dict opcional → pre-computable en MC loop.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional

from .config import SimConfig
from .modelo import simular_partido, resolver_ko


# ─── Constantes ───────────────────────────────────────────────────────────────
GRUPOS = list('ABCDEFGHIJKL')          # 12 grupos
ETAPAS = ['grupos', 'r32', 'r16', 'cuartos', 'semis', 'final', 'campeon']
RONDAS_KO = ['r32', 'r16', 'cuartos', 'semis', 'final']

# ─── Grupos reales del sorteo FIFA (5 dic 2024, Washington D.C.) ──────────────
GRUPOS_2026_REALES: Dict[str, List[str]] = {
    'A': ['Mexico',      'South Africa', 'South Korea', 'Denmark'],
    'B': ['Canada',      'Italy',        'Qatar',       'Switzerland'],
    'C': ['Brazil',      'Morocco',      'Haiti',       'Scotland'],
    'D': ['USA',         'Paraguay',     'Australia',   'Serbia'],
    'E': ['Germany',     'Curacao',      'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan',        'Ukraine',     'Tunisia'],
    'G': ['Belgium',     'Egypt',        'Iran',        'New Zealand'],
    'H': ['Spain',       'Cape Verde',   'Saudi Arabia','Uruguay'],
    'I': ['France',      'Senegal',      'Jamaica',     'Norway'],
    'J': ['Argentina',   'Algeria',      'Austria',     'Jordan'],
    'K': ['Portugal',    'Iraq',         'Uzbekistan',  'Colombia'],
    'L': ['England',     'Croatia',      'Ghana',       'Panama'],
}


# ─── Asignación de grupos ─────────────────────────────────────────────────────

def asignar_grupos(equipos: List[str]) -> Dict[str, List[str]]:
    """
    Devuelve los grupos según el sorteo oficial FIFA 2026.
    Valida que todos los equipos del fixture existan en el dataset.
    """
    equipos_set = set(equipos)
    grupos_out: Dict[str, List[str]] = {}

    for letra, eqs in GRUPOS_2026_REALES.items():
        faltantes = [e for e in eqs if e not in equipos_set]
        if faltantes:
            raise ValueError(
                f"Grupo {letra}: equipos no encontrados en el dataset → {faltantes}\n"
                f"Verificá que los nombres en GRUPOS_2026_REALES coincidan exactamente "
                f"con los de clasificados_2026.csv."
            )
        grupos_out[letra] = list(eqs)

    return grupos_out


# ─── Fase de grupos ───────────────────────────────────────────────────────────

def simular_grupo(
    equipos_grupo: List[str],
    stats: Dict[str, Any],
    config: SimConfig,
    rng: np.random.Generator,
) -> Tuple[List[Dict], List[Tuple[str, str, int, int]]]:
    """
    Simula el round-robin de un grupo (6 partidos).

    Retorna
    -------
    standings     : List[Dict] ordenada [1°, 2°, 3°, 4°]
                    Cada dict: {equipo, pts, gf, gc, gd, v, e, d}
    partidos_grupo: List[Tuple(ta, tb, ga, gb)]
                    Lista de los 6 resultados en orden de juego.
    """
    tabla: Dict[str, Dict] = {
        eq: {'pts': 0, 'gf': 0, 'gc': 0, 'gd': 0, 'v': 0, 'e': 0, 'd': 0}
        for eq in equipos_grupo
    }
    partidos_grupo: List[Tuple[str, str, int, int]] = []

    for eq_a, eq_b in combinations(equipos_grupo, 2):
        g_a, g_b = simular_partido(eq_a, eq_b, stats, config, rng)
        partidos_grupo.append((eq_a, eq_b, g_a, g_b))

        tabla[eq_a]['gf'] += g_a;  tabla[eq_a]['gc'] += g_b
        tabla[eq_b]['gf'] += g_b;  tabla[eq_b]['gc'] += g_a

        if g_a > g_b:
            tabla[eq_a]['pts'] += 3;  tabla[eq_a]['v'] += 1
            tabla[eq_b]['d'] += 1
        elif g_b > g_a:
            tabla[eq_b]['pts'] += 3;  tabla[eq_b]['v'] += 1
            tabla[eq_a]['d'] += 1
        else:
            tabla[eq_a]['pts'] += 1;  tabla[eq_a]['e'] += 1
            tabla[eq_b]['pts'] += 1;  tabla[eq_b]['e'] += 1

    for eq in equipos_grupo:
        tabla[eq]['gd'] = tabla[eq]['gf'] - tabla[eq]['gc']

    rnd = {eq: rng.random() for eq in equipos_grupo}
    ordered = sorted(
        equipos_grupo,
        key=lambda eq: (tabla[eq]['pts'], tabla[eq]['gd'], tabla[eq]['gf'], rnd[eq]),
        reverse=True,
    )

    return [{'equipo': eq, **tabla[eq]} for eq in ordered], partidos_grupo


# ─── Selección de mejores terceros ───────────────────────────────────────────

def seleccionar_mejores_terceros(
    terceros: List[Dict],
    rng: np.random.Generator,
    n: int = 8,
) -> List[str]:
    """De los 12 terceros, elige los 8 mejores. Criterios: pts → gd → gf → sorteo."""
    rnd = {t['equipo']: rng.random() for t in terceros}
    ordenados = sorted(
        terceros,
        key=lambda t: (t['pts'], t['gd'], t['gf'], rnd[t['equipo']]),
        reverse=True,
    )
    return [t['equipo'] for t in ordenados[:n]]


# ─── Bracket del Round of 32 ─────────────────────────────────────────────────

def construir_bracket_r32(
    primeros: List[str],
    segundos: List[str],
    mejores_terceros: List[str],
    grupos_por_equipo: Dict[str, str],
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    """Genera los 16 enfrentamientos del R32 sin equipos del mismo grupo."""
    no_cabezas = mejores_terceros + segundos
    rng.shuffle(no_cabezas)

    disponibles = list(no_cabezas)
    asignados: List[Tuple[str, str]] = []

    for primero in primeros:
        grupo_p    = grupos_por_equipo[primero]
        candidatos = [e for e in disponibles if grupos_por_equipo.get(e) != grupo_p]
        if not candidatos:
            candidatos = disponibles
        rival = candidatos[0]
        disponibles.remove(rival)
        asignados.append((primero, rival))

    assert len(disponibles) == 8
    rng.shuffle(disponibles)
    for i in range(0, 8, 2):
        asignados.append((disponibles[i], disponibles[i + 1]))

    return asignados


# ─── Ronda KO genérica ───────────────────────────────────────────────────────

def simular_ronda_ko(
    enfrentamientos: List[Tuple[str, str]],
    stats: Dict[str, Any],
    config: SimConfig,
    rng: np.random.Generator,
) -> Tuple[List[str], List[Tuple[str, str, int, int, str, str]]]:
    """
    Simula una ronda de eliminación directa.

    Retorna
    -------
    ganadores    : List[str]
    partidos_ko  : List[Tuple(ta, tb, ga, gb, metodo, ganador)]
                   metodo  = '90min' | 'ET' | 'Penales'
                   ganador = nombre del equipo ganador (necesario para penales,
                             donde ga == gb y no se puede inferir del score)
    """
    ganadores:   List[str]                               = []
    partidos_ko: List[Tuple[str, str, int, int, str, str]] = []

    for eq_a, eq_b in enfrentamientos:
        ganador, _, detalle = resolver_ko(eq_a, eq_b, stats, config, rng)
        ganadores.append(ganador)
        partidos_ko.append((
            eq_a, eq_b,
            detalle['goles_a'], detalle['goles_b'],
            detalle['metodo'], ganador,
        ))

    return ganadores, partidos_ko


# ─── Torneo completo ─────────────────────────────────────────────────────────

def simular_torneo(
    clasificados_df: pd.DataFrame,
    stats: Dict[str, Any],
    config: SimConfig,
    rng: np.random.Generator,
    grupos_dict: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Simula un Mundial 2026 completo.

    Parámetros
    ----------
    grupos_dict : pre-computado con asignar_grupos() para evitar validar N veces.

    Retorna
    -------
    dict con:
      'etapas'            → {equipo: etapa_max}
      'campeon'           → str
      'finalista'         → str
      'semifinalistas'    → [str, str]
      'cuartofinalistas'  → [str, str, str, str]
      'ganadores_grupo'   → {letra: equipo}
      'segundos_grupo'    → {letra: equipo}
      'resultados_partidos' → {
            'grupos': [(ta, tb, ga, gb), ...],        # 72 partidos
            'r32':    [(ta, tb, ga, gb, metodo), ...], # 16 partidos
            'r16':    [...], 'cuartos': [...],
            'semis':  [...], 'final':   [...],
        }
    """
    equipos = clasificados_df['equipo'].tolist()
    etapas: Dict[str, str] = {eq: 'grupos' for eq in equipos}

    if grupos_dict is None:
        grupos_dict = asignar_grupos(equipos)

    grupos_por_equipo = {eq: g for g, eqs in grupos_dict.items() for eq in eqs}

    primeros:        List[str]  = []
    segundos:        List[str]  = []
    terceros_info:   List[Dict] = []
    ganadores_grupo: Dict[str, str] = {}
    segundos_grupo:  Dict[str, str] = {}

    # Recolector de resultados de partidos
    all_partidos_grupos: List[Tuple[str, str, int, int]] = []

    # ── Fase de grupos ────────────────────────────────────────────────────────
    for letra, eqs_grupo in grupos_dict.items():
        tabla, partidos_grupo = simular_grupo(eqs_grupo, stats, config, rng)
        all_partidos_grupos.extend(partidos_grupo)

        primero = tabla[0]['equipo']
        segundo = tabla[1]['equipo']
        tercero = tabla[2]['equipo']

        primeros.append(primero)
        segundos.append(segundo)
        ganadores_grupo[letra] = primero
        segundos_grupo[letra]  = segundo

        terceros_info.append({
            'equipo': tercero,
            'pts'   : tabla[2]['pts'],
            'gd'    : tabla[2]['gd'],
            'gf'    : tabla[2]['gf'],
            'grupo' : letra,
        })

        etapas[primero] = 'r32'
        etapas[segundo] = 'r32'

    # ── Mejores 8 terceros ────────────────────────────────────────────────────
    mejores_terceros = seleccionar_mejores_terceros(terceros_info, rng, n=8)
    for eq in mejores_terceros:
        etapas[eq] = 'r32'

    # ── Round of 32 ───────────────────────────────────────────────────────────
    bracket_r32 = construir_bracket_r32(
        primeros, segundos, mejores_terceros, grupos_por_equipo, rng
    )
    ganadores_r32, partidos_r32 = simular_ronda_ko(bracket_r32, stats, config, rng)
    for eq in ganadores_r32:
        etapas[eq] = 'r16'

    # ── Round of 16 ───────────────────────────────────────────────────────────
    rng.shuffle(ganadores_r32)
    bracket_r16 = [(ganadores_r32[i], ganadores_r32[i + 1])
                   for i in range(0, 16, 2)]
    ganadores_r16, partidos_r16 = simular_ronda_ko(bracket_r16, stats, config, rng)
    for eq in ganadores_r16:
        etapas[eq] = 'cuartos'

    # ── Cuartos de final ─────────────────────────────────────────────────────
    rng.shuffle(ganadores_r16)
    bracket_qf = [(ganadores_r16[i], ganadores_r16[i + 1])
                  for i in range(0, 8, 2)]
    ganadores_qf, partidos_qf = simular_ronda_ko(bracket_qf, stats, config, rng)
    perdedores_qf = [eq for eq in ganadores_r16 if eq not in ganadores_qf]
    for eq in ganadores_qf:
        etapas[eq] = 'semis'

    # ── Semifinales ───────────────────────────────────────────────────────────
    bracket_sf = [(ganadores_qf[0], ganadores_qf[1]),
                  (ganadores_qf[2], ganadores_qf[3])]
    ganadores_sf,  partidos_sf = simular_ronda_ko(bracket_sf, stats, config, rng)
    perdedores_sf = [eq for eq in ganadores_qf if eq not in ganadores_sf]
    for eq in ganadores_sf:
        etapas[eq] = 'final'

    # ── Final ────────────────────────────────────────────────────────────────
    ganador_final, finalista, detalle_final = resolver_ko(
        ganadores_sf[0], ganadores_sf[1], stats, config, rng
    )
    etapas[ganador_final] = 'campeon'
    partidos_final = [(
        ganadores_sf[0], ganadores_sf[1],
        detalle_final['goles_a'], detalle_final['goles_b'],
        detalle_final['metodo'], ganador_final,
    )]

    return {
        'etapas'           : etapas,
        'campeon'          : ganador_final,
        'finalista'        : finalista,
        'semifinalistas'   : perdedores_sf,
        'cuartofinalistas' : perdedores_qf,
        'ganadores_grupo'  : ganadores_grupo,
        'segundos_grupo'   : segundos_grupo,
        'resultados_partidos': {
            'grupos'  : all_partidos_grupos,
            'r32'     : partidos_r32,
            'r16'     : partidos_r16,
            'cuartos' : partidos_qf,
            'semis'   : partidos_sf,
            'final'   : partidos_final,
        },
    }
