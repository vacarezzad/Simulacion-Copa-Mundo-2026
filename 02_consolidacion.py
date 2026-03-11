"""
02_consolidacion.py
-------------------
Consolida todas las fuentes de datos en un único archivo maestro con
todos los KPIs por selección.

Fuentes
-------
  Data/equipos_historico.csv              → stats históricas (output de 01_extraccion.py)
  Data/world_cup_last_50_years.csv        → resultados partido a partido (1974-2022)

Salida
------
  Data/equipos_completo.csv              → dataset maestro, una fila por selección

KPIs generados
--------------
  Historial completo · Logros · Por fase (grupos vs KO) · Forma reciente
  · Ratios ofensivos/defensivos · Índices compuestos · Tendencia

Ejecutar
--------
  python 02_consolidacion.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent / 'Data'
HIST_PATH    = BASE / 'equipos_historico.csv'
MATCHES_PATH = BASE / 'world_cup_last_50_years.csv'
OUTPUT_PATH  = BASE / 'equipos_completo.csv'

# ─── Categorización de fases ──────────────────────────────────────────────────
FASES_GRUPO = {'Group Stage'}
FASES_KO    = {'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final', 'Third Place'}
FASES_FINAL = {'Final'}
FASES_SEMI  = {'Semi-finals'}

# ─── Normalización de nombres (mismo criterio que 01_extraccion.py) ───────────
NOMBRE_MAP_MATCHES = {
    'West Germany'  : 'Germany',
    'United States' : 'USA',
    'Soviet Union'  : 'Russia',
    'FR Yugoslavia' : 'Serbia',
    'Yugoslavia'    : 'Serbia',
    'South Korea'   : 'South Korea',
    'Ivory Coast'   : 'Ivory Coast',
    'IR Iran'       : 'Iran',
}


def normalizar(nombre: str) -> str:
    return NOMBRE_MAP_MATCHES.get(str(nombre).strip(), str(nombre).strip())


# ─── Carga de datos ───────────────────────────────────────────────────────────

def cargar_historico() -> pd.DataFrame:
    df = pd.read_csv(HIST_PATH)
    print(f'  [OK] Histórico: {len(df)} equipos, {len(df.columns)} columnas')
    return df


def cargar_partidos() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_PATH)

    # Normalizar nombres
    df['home_team'] = df['home_team'].apply(normalizar)
    df['away_team'] = df['away_team'].apply(normalizar)
    df['winner']    = df['winner'].apply(lambda x: normalizar(x) if x != 'Draw' else 'Draw')

    print(f'  [OK] Partidos: {len(df)} registros | '
          f'años: {df["year"].min()}–{df["year"].max()} | '
          f'fases: {sorted(df["stage"].unique())}')
    return df


# ─── Estadísticas desde archivo de partidos ──────────────────────────────────

def stats_desde_partidos(partidos: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada equipo y cada fase (grupos / KO / total), calcula:
      partidos, victorias, empates, derrotas, GF, GC.
    Retorna un DataFrame con una fila por equipo.
    """
    equipos_set = set(partidos['home_team'].unique()) | set(partidos['away_team'].unique())
    rows = []

    for equipo in sorted(equipos_set):

        # Partidos como local y visitante
        como_local    = partidos[partidos['home_team'] == equipo].copy()
        como_visitante= partidos[partidos['away_team'] == equipo].copy()

        def calcular_stats(df_local, df_visit):
            """Agrega stats para un subconjunto de partidos."""
            n    = len(df_local) + len(df_visit)
            gf   = df_local['home_goals'].sum() + df_visit['away_goals'].sum()
            gc   = df_local['away_goals'].sum() + df_visit['home_goals'].sum()
            wins = (df_local['winner'] == equipo).sum() + \
                   (df_visit['winner'] == equipo).sum()
            draws= (df_local['winner'] == 'Draw').sum() + \
                   (df_visit['winner'] == 'Draw').sum()
            loss = n - wins - draws
            return n, int(wins), int(draws), int(loss), int(gf), int(gc)

        # ── Fase de grupos ───────────────────────────────────────────────────
        loc_g  = como_local[como_local['stage'].isin(FASES_GRUPO)]
        vis_g  = como_visitante[como_visitante['stage'].isin(FASES_GRUPO)]
        pg, vg, eg, dg, gfg, gcg = calcular_stats(loc_g, vis_g)

        # ── Fase eliminatoria ────────────────────────────────────────────────
        loc_ko = como_local[como_local['stage'].isin(FASES_KO)]
        vis_ko = como_visitante[como_visitante['stage'].isin(FASES_KO)]
        pko, vko, eko, dko, gfko, gcko = calcular_stats(loc_ko, vis_ko)

        # ── Finales y semis ──────────────────────────────────────────────────
        loc_f  = como_local[como_local['stage'].isin(FASES_FINAL)]
        vis_f  = como_visitante[como_visitante['stage'].isin(FASES_FINAL)]
        finales_jugadas = len(loc_f) + len(vis_f)

        loc_s  = como_local[como_local['stage'].isin(FASES_SEMI)]
        vis_s  = como_visitante[como_visitante['stage'].isin(FASES_SEMI)]
        semis_jugadas = len(loc_s) + len(vis_s)

        # ── Total (partidos + KO) ────────────────────────────────────────────
        pt, vt, et, dt, gft, gct = calcular_stats(
            pd.concat([loc_g, loc_ko]),
            pd.concat([vis_g, vis_ko])
        )

        rows.append({
            'equipo'           : equipo,
            # Grupos
            'ptd_grupos_match' : pg,
            'v_grupos_match'   : vg,
            'e_grupos_match'   : eg,
            'd_grupos_match'   : dg,
            'gf_grupos_match'  : gfg,
            'gc_grupos_match'  : gcg,
            'wr_grupos_match'  : round(vg / pg, 4) if pg > 0 else np.nan,
            'gfp_grupos_match' : round(gfg / pg, 4) if pg > 0 else np.nan,
            'gcp_grupos_match' : round(gcg / pg, 4) if pg > 0 else np.nan,
            # Eliminatorias
            'ptd_ko_match'     : pko,
            'v_ko_match'       : vko,
            'e_ko_match'       : eko,
            'd_ko_match'       : dko,
            'gf_ko_match'      : gfko,
            'gc_ko_match'      : gcko,
            'wr_ko_match'      : round(vko / pko, 4) if pko > 0 else np.nan,
            'gfp_ko_match'     : round(gfko / pko, 4) if pko > 0 else np.nan,
            'gcp_ko_match'     : round(gcko / pko, 4) if pko > 0 else np.nan,
            # Instancias de alto impacto
            'finales_jugadas'  : finales_jugadas,
            'semis_jugadas'    : semis_jugadas,
            # Total archivo partidos
            'ptd_total_match'  : pt,
            'wr_total_match'   : round(vt / pt, 4) if pt > 0 else np.nan,
            'gfp_total_match'  : round(gft / pt, 4) if pt > 0 else np.nan,
            'gcp_total_match'  : round(gct / pt, 4) if pt > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ─── KPIs compuestos ──────────────────────────────────────────────────────────

def calcular_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas de KPIs derivados sobre el DataFrame ya consolidado.

    KPI                     Fórmula / Lógica
    ─────────────────────── ──────────────────────────────────────────────────
    ratio_goles_hist        GF_hist / GC_hist  → equilibrio ataque-defensa
    ratio_goles_rec         GF_rec  / GC_rec   → ídem en forma reciente
    consistencia_mundial    mundiales_jugados / 22  → % Mundiales asistidos
    indice_titulo           campeon×5 + final×3 + semi×2 + cuartos×1
    tendencia               win_rate_reciente - win_rate_historico (↑ mejora)
    brecha_grupos_ko        wr_grupos - wr_ko  (↑ = baja en KO, ↓ = crece)
    indice_presion          wr_ko_match  → rendimiento bajo presión eliminatoria
    eficiencia_goleadora    gf_prom_hist / max(gc_prom_hist, 0.1)
    solidez_defensiva_hist  1 / max(gc_prom_hist, 0.1)  (normalizado)
    indice_rendimiento      promedio ponderado de varios KPIs (0-100)
    """

    # ── Ratios goles ─────────────────────────────────────────────────────────
    df['ratio_goles_hist'] = (
        df['gf_prom_hist'] / df['gc_prom_hist'].replace(0, np.nan)
    ).round(4)

    df['ratio_goles_rec'] = (
        df['gf_prom_rec'] / df['gc_prom_rec'].replace(0, np.nan)
    ).round(4)

    # ── Consistencia ─────────────────────────────────────────────────────────
    df['consistencia_mundial'] = (df['mundiales_jugados'] / 22).round(4)

    # ── Índice de títulos ─────────────────────────────────────────────────────
    df['indice_titulo'] = (
        df['summary_campeon'] * 5 +
        df['summary_sub']     * 3 +
        df['summary_tercero'] * 2 +
        df['veces_semifinal'] * 1
    )

    # ── Tendencia ─────────────────────────────────────────────────────────────
    df['tendencia'] = (
        df['win_rate_reciente'] - df['win_rate_historico']
    ).round(4)

    # ── Brecha grupos vs KO ───────────────────────────────────────────────────
    df['brecha_grupos_ko'] = (
        df['wr_grupos_match'] - df['wr_ko_match']
    ).round(4)

    # ── Índice de presión (rendimiento KO) ───────────────────────────────────
    df['indice_presion'] = df['wr_ko_match'].round(4)

    # ── Eficiencia goleadora ─────────────────────────────────────────────────
    df['eficiencia_goleadora'] = (
        df['gf_prom_hist'] / df['gc_prom_hist'].replace(0, np.nan)
    ).round(4)

    # ── Solidez defensiva (invertir gc: menor gc → mayor solidez) ────────────
    gc_max = df['gc_prom_hist'].max()
    df['solidez_defensiva'] = (
        (gc_max - df['gc_prom_hist']) / gc_max
    ).round(4)

    # ── Índice de rendimiento compuesto (0–100) ───────────────────────────────
    # Normalización min-max de cada componente, luego promedio ponderado
    def norm(serie):
        mn, mx = serie.min(), serie.max()
        return (serie - mn) / (mx - mn + 1e-9)

    df['indice_rendimiento'] = (
        norm(df['win_rate_historico'])   * 30 +
        norm(df['win_rate_reciente'].fillna(df['win_rate_historico'])) * 25 +
        norm(df['indice_titulo'])        * 20 +
        norm(df['ratio_goles_hist'].fillna(1)) * 15 +
        norm(df['wr_ko_match'].fillna(df['win_rate_historico'])) * 10
    ).round(2)

    return df


# ─── Reporte final ────────────────────────────────────────────────────────────

def reporte(df: pd.DataFrame) -> None:
    print('\n' + '=' * 70)
    print('DATASET MAESTRO — RESUMEN')
    print('=' * 70)
    print(f'  Equipos  : {len(df)}')
    print(f'  Columnas : {len(df.columns)}')
    print()

    cols_show = [
        'equipo', 'confederacion',
        'mundiales_jugados', 'win_rate_historico',
        'win_rate_reciente', 'veces_campeon',
        'wr_ko_match', 'indice_titulo',
        'indice_rendimiento', 'tendencia'
    ]
    top = df.sort_values('indice_rendimiento', ascending=False).head(20)
    print('Top 20 por índice de rendimiento:')
    print(top[cols_show].to_string(index=False))
    print()

    print('Columnas generadas:')
    for i, col in enumerate(df.columns, 1):
        print(f'  {i:>2}. {col}')
    print('=' * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n=== CONSOLIDACIÓN DE KPIs — MUNDIALES ===\n')

    print('1. Cargando datos base...')
    df_hist    = cargar_historico()
    df_matches = cargar_partidos()

    print('\n2. Calculando stats por fase desde partidos...')
    df_fase = stats_desde_partidos(df_matches)
    print(f'  [OK] {len(df_fase)} equipos procesados desde archivo de partidos')

    print('\n3. Mergeando fuentes...')
    df = df_hist.merge(df_fase, on='equipo', how='left')
    print(f'  [OK] {len(df)} filas | {len(df.columns)} columnas tras merge')

    # Equipos del historial sin datos en el archivo de partidos → NaN en cols de fase
    sin_datos_fase = df[df['ptd_grupos_match'].isna()]['equipo'].tolist()
    if sin_datos_fase:
        print(f'  [!] Sin datos de partidos (históricos no cubiertos): {sin_datos_fase}')

    print('\n4. Calculando KPIs compuestos...')
    df = calcular_kpis(df)
    print(f'  [OK] KPIs calculados | Total columnas: {len(df.columns)}')

    print('\n5. Guardando dataset maestro...')
    df.to_csv(OUTPUT_PATH, index=False, float_format='%.4f')
    print(f'  [OK] {OUTPUT_PATH}')
    print(f'  Filas: {len(df)} | Columnas: {len(df.columns)}')

    print('\n6. Reporte:')
    reporte(df)

    print('\n[✓] Proceso completado.\n')
