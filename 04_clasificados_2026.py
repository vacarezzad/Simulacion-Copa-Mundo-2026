"""
04_clasificados_2026.py
-----------------------
Filtra el dataset maestro a los 48 clasificados al Mundial 2026
y agrega equipos debutantes que no tienen historial previo.

Fuente de clasificados
----------------------
  Wikipedia / CBS Sports / FIFA.com (marzo 2026)
  42 clasificados confirmados + 6 estimados de playoffs pendientes.

Lógica para debutantes
----------------------
  Equipos sin historial en FIFA - YYYY.csv (primer Mundial) reciben
  métricas históricas en 0/NaN y se identifican con debut_2026 = True.
  Sus puntos FIFA y ranking son los únicos inputs del modelo.

Salida
------
  Data/clasificados_2026.csv   → 48 filas, todas las columnas del maestro

Ejecutar
--------
  python 04_clasificados_2026.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE          = Path(__file__).parent / 'Data'
COMPLETO_PATH = BASE / 'equipos_completo.csv'
RANKING_PATH  = BASE / 'ranking_fifa.csv'
OUTPUT_PATH   = BASE / 'clasificados_2026.csv'

# ─── Lista de 48 clasificados ─────────────────────────────────────────────────
# Estado: confirmados (✓) o estimados de playoffs pendientes (~)

CLASIFICADOS_2026 = [
    # ── CONCACAF (6) ─────────────────────────────────────────────────────────
    {'equipo': 'USA',        'confederacion': 'CONCACAF', 'estado': 'confirmado', 'nota': 'Sede'},
    {'equipo': 'Canada',     'confederacion': 'CONCACAF', 'estado': 'confirmado', 'nota': 'Sede'},
    {'equipo': 'Mexico',     'confederacion': 'CONCACAF', 'estado': 'confirmado', 'nota': 'Sede'},
    {'equipo': 'Panama',     'confederacion': 'CONCACAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Haiti',      'confederacion': 'CONCACAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Curacao',    'confederacion': 'CONCACAF', 'estado': 'confirmado', 'nota': 'Debut - menor nación clasificada'},
    {'equipo': 'Jamaica',    'confederacion': 'CONCACAF', 'estado': 'estimado',   'nota': 'Playoff interconfederal'},

    # ── CONMEBOL (6) ─────────────────────────────────────────────────────────
    {'equipo': 'Argentina',  'confederacion': 'CONMEBOL', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Brazil',     'confederacion': 'CONMEBOL', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Colombia',   'confederacion': 'CONMEBOL', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Ecuador',    'confederacion': 'CONMEBOL', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Uruguay',    'confederacion': 'CONMEBOL', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Paraguay',   'confederacion': 'CONMEBOL', 'estado': 'confirmado', 'nota': ''},

    # ── UEFA (16) ─────────────────────────────────────────────────────────────
    {'equipo': 'Spain',       'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'England',     'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'France',      'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Germany',     'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Portugal',    'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Netherlands', 'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Belgium',     'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Croatia',     'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Switzerland', 'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Austria',     'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Norway',      'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Scotland',    'confederacion': 'UEFA', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Italy',       'confederacion': 'UEFA', 'estado': 'estimado',   'nota': 'Playoff UEFA'},
    {'equipo': 'Denmark',     'confederacion': 'UEFA', 'estado': 'estimado',   'nota': 'Playoff UEFA'},
    {'equipo': 'Ukraine',     'confederacion': 'UEFA', 'estado': 'estimado',   'nota': 'Playoff UEFA'},
    {'equipo': 'Serbia',      'confederacion': 'UEFA', 'estado': 'estimado',   'nota': 'Playoff UEFA'},

    # ── AFC (8) ───────────────────────────────────────────────────────────────
    {'equipo': 'Japan',        'confederacion': 'AFC', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'South Korea',  'confederacion': 'AFC', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Iran',         'confederacion': 'AFC', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Australia',    'confederacion': 'AFC', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Saudi Arabia', 'confederacion': 'AFC', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Qatar',        'confederacion': 'AFC', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Jordan',       'confederacion': 'AFC', 'estado': 'confirmado', 'nota': 'Debut'},
    {'equipo': 'Uzbekistan',   'confederacion': 'AFC', 'estado': 'confirmado', 'nota': 'Debut'},

    # ── CAF (9) ───────────────────────────────────────────────────────────────
    {'equipo': 'Morocco',      'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Senegal',      'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Egypt',        'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Algeria',      'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Ivory Coast',  'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Ghana',        'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Tunisia',      'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'South Africa', 'confederacion': 'CAF', 'estado': 'confirmado', 'nota': ''},
    {'equipo': 'Cape Verde',   'confederacion': 'CAF', 'estado': 'confirmado', 'nota': 'Debut'},

    # ── OFC (1) ───────────────────────────────────────────────────────────────
    {'equipo': 'New Zealand',  'confederacion': 'OFC', 'estado': 'confirmado', 'nota': ''},

    # ── Inter-confederal restante ─────────────────────────────────────────────
    {'equipo': 'Iraq',         'confederacion': 'AFC',      'estado': 'estimado', 'nota': 'Playoff interconfederal'},
]

assert len(CLASIFICADOS_2026) == 48, f'Se esperaban 48 equipos, hay {len(CLASIFICADOS_2026)}'

# ─── Normalización de nombres para merge ─────────────────────────────────────
# Algunos equipos pueden tener nombres distintos en el dataset histórico
NOMBRE_MAP_2026 = {
    'Cape Verde': 'Cape Verde',   # No está en histórico → debutante
    'Jordan':     'Jordan',       # No está en histórico → debutante
    'Uzbekistan': 'Uzbekistan',   # No está en histórico → debutante
    'Haiti':      'Haiti',
    'Iraq':       'Iraq',
    'Jamaica':    'Jamaica',
    'Panama':     'Panama',
}


def construir_fila_debutante(equipo: str, confederacion: str,
                              df_ranking: pd.DataFrame) -> dict:
    """
    Construye una fila con valores base para equipos sin historial en Mundiales.
    Los únicos datos reales son los del ranking FIFA.
    """
    rk = df_ranking[df_ranking['equipo'] == equipo]
    ranking = int(rk['ranking_fifa'].values[0]) if len(rk) > 0 else 999
    puntos  = float(rk['puntos_fifa'].values[0]) if len(rk) > 0 else 1000.0

    return {
        'equipo'              : equipo,
        'confederacion'       : confederacion,
        'ranking_fifa'        : ranking,
        'puntos_fifa'         : puntos,
        'mundiales_jugados'   : 0,
        'partidos_totales'    : 0,
        'victorias_totales'   : 0,
        'empates_totales'     : 0,
        'derrotas_totales'    : 0,
        'gf_total'            : 0,
        'gc_total'            : 0,
        'win_rate_historico'  : 0.0,
        'gf_prom_hist'        : 0.0,
        'gc_prom_hist'        : 0.0,
        'mejor_posicion'      : np.nan,
        'ultimo_mundial'      : np.nan,
        'veces_campeon'       : 0,
        'veces_final'         : 0,
        'veces_semifinal'     : 0,
        'veces_cuartos'       : 0,
        'mundiales_recientes' : 0,
        'partidos_recientes'  : 0,
        'victorias_recientes' : 0,
        'empates_recientes'   : 0,
        'derrotas_recientes'  : 0,
        'gf_reciente'         : 0,
        'gc_reciente'         : 0,
        'win_rate_reciente'   : np.nan,
        'gf_prom_rec'         : np.nan,
        'gc_prom_rec'         : np.nan,
        'summary_campeon'     : 0,
        'summary_sub'         : 0,
        'summary_tercero'     : 0,
        'debut_2026'          : True,
    }


if __name__ == '__main__':
    print('\n=== FILTRO CLASIFICADOS MUNDIAL 2026 ===\n')

    # ── Cargar fuentes ────────────────────────────────────────────────────────
    print('1. Cargando dataset maestro y ranking FIFA...')
    df_maestro = pd.read_csv(COMPLETO_PATH)
    df_ranking = pd.read_csv(RANKING_PATH)
    print(f'  Dataset maestro : {len(df_maestro)} equipos, {len(df_maestro.columns)} cols')
    print(f'  Ranking FIFA    : {len(df_ranking)} equipos')

    # ── Marcar debut_2026 en maestro ─────────────────────────────────────────
    df_maestro['debut_2026'] = False

    # ── Construir df de clasificados ──────────────────────────────────────────
    print('\n2. Filtrando y completando 48 clasificados...')
    df_lista   = pd.DataFrame(CLASIFICADOS_2026)
    equipos_en_maestro = set(df_maestro['equipo'].values)

    filas_ok      = []
    filas_debut   = []
    no_encontrados = []

    for _, row in df_lista.iterrows():
        nombre = row['equipo']
        if nombre in equipos_en_maestro:
            filas_ok.append(nombre)
        else:
            filas_debut.append(nombre)
            no_encontrados.append(nombre)

    print(f'  Con historial   : {len(filas_ok)} equipos')
    print(f'  Debutantes      : {len(filas_debut)} → {filas_debut}')

    # ── Filtrar maestro a clasificados con historial ──────────────────────────
    df_con_hist = df_maestro[df_maestro['equipo'].isin(filas_ok)].copy()

    # ── Construir filas para debutantes ──────────────────────────────────────
    filas_nuevas = []
    for nombre in filas_debut:
        conf = df_lista[df_lista['equipo'] == nombre]['confederacion'].values[0]
        filas_nuevas.append(construir_fila_debutante(nombre, conf, df_ranking))

    df_debut = pd.DataFrame(filas_nuevas)

    # ── Concatenar ────────────────────────────────────────────────────────────
    df_final = pd.concat([df_con_hist, df_debut], ignore_index=True)

    # ── Agregar columnas de estado del clasificado ────────────────────────────
    df_lista_meta = df_lista[['equipo', 'estado', 'nota']].rename(
        columns={'estado': 'estado_clasificacion', 'nota': 'nota_clasificacion'}
    )
    df_final = df_final.merge(df_lista_meta, on='equipo', how='left')

    # ── Asegurar columna debut_2026 ───────────────────────────────────────────
    if 'debut_2026' not in df_final.columns:
        df_final['debut_2026'] = False
    df_final['debut_2026'] = df_final['debut_2026'].fillna(False)

    # ── Ordenar por puntos FIFA (proxy de fuerza actual) ──────────────────────
    df_final = df_final.sort_values('puntos_fifa', ascending=False,
                                    na_position='last').reset_index(drop=True)
    df_final['seed_ranking'] = df_final.index + 1

    # ── Guardar ───────────────────────────────────────────────────────────────
    print(f'\n3. Guardando clasificados_2026.csv...')
    df_final.to_csv(OUTPUT_PATH, index=False, float_format='%.4f')
    print(f'  [OK] {OUTPUT_PATH}')
    print(f'  Filas: {len(df_final)} | Columnas: {len(df_final.columns)}')

    # ── Reporte ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('48 CLASIFICADOS MUNDIAL 2026 — ordenados por puntos FIFA')
    print('=' * 72)
    cols_show = ['seed_ranking', 'equipo', 'confederacion', 'estado_clasificacion',
                 'puntos_fifa', 'ranking_fifa', 'win_rate_historico',
                 'veces_campeon', 'debut_2026']
    print(df_final[cols_show].to_string(index=False))

    print('\nResumen por confederación:')
    resumen = (df_final.groupby('confederacion')
               .agg(equipos=('equipo', 'count'),
                    confirmados=('estado_clasificacion',
                                 lambda x: (x == 'confirmado').sum()),
                    estimados=('estado_clasificacion',
                               lambda x: (x == 'estimado').sum()))
               .reset_index())
    print(resumen.to_string(index=False))
    print('=' * 72)
    print('\n[✓] Proceso completado.\n')
