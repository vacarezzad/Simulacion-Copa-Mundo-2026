"""
01_extraccion.py
----------------
Extrae, limpia y consolida las estadísticas históricas de todos los
Mundiales (1930-2022) para construir el dataset base del modelo.

Fuentes
-------
  Data/participaciones_mundial/FIFA - YYYY.csv  → stats por equipo por torneo
  Data/participaciones_mundial/FIFA - World Cup Summary.csv → campeones/sub/3ro

Salida
------
  Data/equipos_historico.csv   → una fila por selección, columnas detalladas abajo

Ejecutar
--------
  python 01_extraccion.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE              = Path(__file__).parent / 'Data'
PART_DIR          = BASE / 'participaciones_mundial'
SUMMARY_PATH      = PART_DIR / 'FIFA - World Cup Summary.csv'
OUTPUT_PATH       = BASE / 'equipos_historico.csv'

# ─── Torneos recientes (para métricas de forma) ───────────────────────────────
TORNEOS_RECIENTES = {2014, 2018, 2022}

# ─── Mapa de normalización de nombres ─────────────────────────────────────────
# Equipos que cambiaron de nombre o son sucesores reconocidos por FIFA.
# None = excluir del dataset (equipos extintos sin sucesor claro en 2026).
NOMBRE_MAP = {
    # Reunificaciones / sucesores
    'West Germany'              : 'Germany',
    'German Democratic Republic': 'Germany',
    'East Germany'              : 'Germany',
    'Soviet Union'              : 'Russia',
    'FR Yugoslavia'             : 'Serbia',
    'Yugoslavia'                : 'Serbia',
    # Extintos sin sucesor claro → excluir
    'Czechoslovakia'            : None,
    'Serbia and Montenegro'     : None,
    'Dutch East Indies'         : None,
    'Bohemia'                   : None,
    'Chinese Taipei'            : None,
    # Renombres / estandarización
    'United States'             : 'USA',
    'Republic of Ireland'       : 'Ireland',
    'Zaire'                     : 'DR Congo',
    'United Arab Emirates'      : 'UAE',
    'China PR'                  : 'China',
    'IR Iran'                   : 'Iran',
    'Ivory Coast'               : 'Ivory Coast',
    'North Korea'               : 'North Korea',
    'South Korea'               : 'South Korea',
    'Trinidad and Tobago'       : 'Trinidad and Tobago',
}

# Confederación por equipo (para las 2026 y equipos históricos relevantes)
CONFEDERACION_MAP = {
    'Argentina': 'CONMEBOL', 'Brazil': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL', 'Chile': 'CONMEBOL',
    'Paraguay': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Bolivia': 'CONMEBOL',
    'Venezuela': 'CONMEBOL',
    'France': 'UEFA', 'Germany': 'UEFA', 'Spain': 'UEFA', 'Italy': 'UEFA',
    'England': 'UEFA', 'Portugal': 'UEFA', 'Netherlands': 'UEFA',
    'Belgium': 'UEFA', 'Croatia': 'UEFA', 'Switzerland': 'UEFA',
    'Denmark': 'UEFA', 'Sweden': 'UEFA', 'Poland': 'UEFA', 'Serbia': 'UEFA',
    'Austria': 'UEFA', 'Turkey': 'UEFA', 'Russia': 'UEFA', 'Hungary': 'UEFA',
    'Romania': 'UEFA', 'Bulgaria': 'UEFA', 'Czech Republic': 'UEFA',
    'Czechoslovakia': 'UEFA', 'Scotland': 'UEFA', 'Ukraine': 'UEFA',
    'Slovakia': 'UEFA', 'Slovenia': 'UEFA', 'Norway': 'UEFA',
    'Wales': 'UEFA', 'Ireland': 'UEFA', 'Northern Ireland': 'UEFA',
    'Iceland': 'UEFA', 'Albania': 'UEFA', 'Georgia': 'UEFA',
    'Japan': 'AFC', 'South Korea': 'AFC', 'Australia': 'AFC',
    'Iran': 'AFC', 'Saudi Arabia': 'AFC', 'Qatar': 'AFC',
    'Iraq': 'AFC', 'Uzbekistan': 'AFC', 'China': 'AFC',
    'North Korea': 'AFC', 'Kuwait': 'AFC', 'UAE': 'AFC',
    'Indonesia': 'AFC', 'Jordan': 'AFC',
    'Morocco': 'CAF', 'Senegal': 'CAF', 'Nigeria': 'CAF',
    'Egypt': 'CAF', 'Ivory Coast': 'CAF', 'Ghana': 'CAF',
    'Cameroon': 'CAF', 'Algeria': 'CAF', 'Tunisia': 'CAF',
    'South Africa': 'CAF', 'DR Congo': 'CAF', 'Togo': 'CAF',
    'Angola': 'CAF', 'Mali': 'CAF',
    'USA': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Costa Rica': 'CONCACAF', 'Jamaica': 'CONCACAF', 'Panama': 'CONCACAF',
    'Honduras': 'CONCACAF', 'Trinidad and Tobago': 'CONCACAF',
    'El Salvador': 'CONCACAF', 'Cuba': 'CONCACAF', 'Haiti': 'CONCACAF',
    'New Zealand': 'OFC',
    # Históricos sin categoría clara
    'Israel*': 'AFC',
}


def normalizar_nombre(nombre: str) -> Optional[str]:
    """
    Limpia asteriscos/marcadores y devuelve el nombre normalizado.
    Retorna None si el equipo debe excluirse del dataset.
    """
    nombre = nombre.strip().replace('*', '').replace('**', '').strip()
    return NOMBRE_MAP.get(nombre, nombre)


# ─── Lectura de archivos por torneo ───────────────────────────────────────────

def leer_torneo(path: Path, year: int) -> pd.DataFrame:
    """
    Lee un archivo FIFA - YYYY.csv y devuelve un DataFrame limpio con columna 'year'.
    Maneja el guión especial '−' en Goal Difference.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Limpiar Goal Difference: reemplazar guión especial (U+2212) por '-' normal
    if 'Goal Difference' in df.columns:
        df['Goal Difference'] = (
            df['Goal Difference']
            .astype(str)
            .str.replace('\u2212', '-', regex=False)  # em-dash → minus
            .str.replace('−', '-', regex=False)        # variante
        )
        df['Goal Difference'] = pd.to_numeric(df['Goal Difference'], errors='coerce')

    # Asegurar tipos numéricos
    num_cols = ['Games Played', 'Win', 'Draw', 'Loss', 'Goals For', 'Goals Against', 'Points']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['year'] = year

    # Normalizar nombres de equipos
    df['Team'] = df['Team'].apply(normalizar_nombre)
    df = df[df['Team'].notna()].copy()  # eliminar equipos extintos

    return df


def cargar_todos_los_torneos() -> pd.DataFrame:
    """Lee todos los CSV de torneos y los concatena en un único DataFrame."""
    archivos = sorted(PART_DIR.glob('FIFA - [0-9]*.csv'))

    if not archivos:
        raise FileNotFoundError(f'No se encontraron archivos en {PART_DIR}')

    frames = []
    for path in archivos:
        year = int(path.stem.split('-')[-1].strip())
        df = leer_torneo(path, year)
        frames.append(df)
        print(f'  [OK] {year} — {len(df)} equipos')

    todos = pd.concat(frames, ignore_index=True)
    print(f'\n  Total registros cargados: {len(todos)}\n')
    return todos


# ─── Carga del Summary ────────────────────────────────────────────────────────

def cargar_summary() -> pd.DataFrame:
    """Lee el resumen histórico de campeones, sub-campeones y terceros."""
    df = pd.read_csv(SUMMARY_PATH)
    df.columns = df.columns.str.strip()

    # Normalizar nombres en las columnas de posiciones
    for col in ['CHAMPION', 'RUNNER UP', 'THIRD PLACE']:
        if col in df.columns:
            df[col] = df[col].apply(normalizar_nombre)

    return df


# ─── Construcción del dataset por equipo ─────────────────────────────────────

def construir_dataset(todos: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos por equipo y calcula todas las métricas.

    Métricas históricas (todos los torneos):
      mundiales_jugados, partidos_totales, victorias, empates, derrotas
      gf_total, gc_total, win_rate_historico, gf_prom_hist, gc_prom_hist
      mejor_posicion, ultimo_mundial
      veces_campeon, veces_final, veces_semifinal, veces_cuartos

    Métricas recientes (2014, 2018, 2022):
      mundiales_recientes, partidos_recientes, victorias_recientes
      gf_reciente, gc_reciente, win_rate_reciente, gf_prom_rec, gc_prom_rec
    """

    equipos = []

    for equipo, grupo in todos.groupby('Team'):

        # ── Histórico completo ───────────────────────────────────────────────
        hist = grupo.copy()
        rec  = grupo[grupo['year'].isin(TORNEOS_RECIENTES)].copy()

        mundiales_jugados  = hist['year'].nunique()
        partidos_totales   = hist['Games Played'].sum()
        victorias_totales  = hist['Win'].sum()
        empates_totales    = hist['Draw'].sum()
        derrotas_totales   = hist['Loss'].sum()
        gf_total           = hist['Goals For'].sum()
        gc_total           = hist['Goals Against'].sum()
        mejor_posicion     = hist['Position'].min()
        ultimo_mundial     = hist['year'].max()

        win_rate_historico = victorias_totales / partidos_totales if partidos_totales > 0 else 0
        gf_prom_hist       = gf_total / partidos_totales if partidos_totales > 0 else 0
        gc_prom_hist       = gc_total / partidos_totales if partidos_totales > 0 else 0

        # Logros históricos por posición final
        veces_campeon    = (hist['Position'] == 1).sum()
        veces_final      = (hist['Position'] <= 2).sum()
        veces_semifinal  = (hist['Position'] <= 4).sum()
        veces_cuartos    = (hist['Position'] <= 8).sum()

        # ── Forma reciente (2014-2022) ───────────────────────────────────────
        mundiales_recientes  = rec['year'].nunique()
        partidos_recientes   = rec['Games Played'].sum()
        victorias_recientes  = rec['Win'].sum()
        empates_recientes    = rec['Draw'].sum()
        derrotas_recientes   = rec['Loss'].sum()
        gf_reciente          = rec['Goals For'].sum()
        gc_reciente          = rec['Goals Against'].sum()

        win_rate_reciente = (
            victorias_recientes / partidos_recientes
            if partidos_recientes > 0 else np.nan
        )
        gf_prom_rec = (
            gf_reciente / partidos_recientes
            if partidos_recientes > 0 else np.nan
        )
        gc_prom_rec = (
            gc_reciente / partidos_recientes
            if partidos_recientes > 0 else np.nan
        )

        equipos.append({
            'equipo'              : equipo,
            # Historial
            'mundiales_jugados'   : mundiales_jugados,
            'partidos_totales'    : int(partidos_totales),
            'victorias_totales'   : int(victorias_totales),
            'empates_totales'     : int(empates_totales),
            'derrotas_totales'    : int(derrotas_totales),
            'gf_total'            : int(gf_total),
            'gc_total'            : int(gc_total),
            'win_rate_historico'  : round(win_rate_historico, 4),
            'gf_prom_hist'        : round(gf_prom_hist, 4),
            'gc_prom_hist'        : round(gc_prom_hist, 4),
            'mejor_posicion'      : int(mejor_posicion),
            'ultimo_mundial'      : int(ultimo_mundial),
            # Logros
            'veces_campeon'       : int(veces_campeon),
            'veces_final'         : int(veces_final),
            'veces_semifinal'     : int(veces_semifinal),
            'veces_cuartos'       : int(veces_cuartos),
            # Forma reciente
            'mundiales_recientes' : int(mundiales_recientes),
            'partidos_recientes'  : int(partidos_recientes),
            'victorias_recientes' : int(victorias_recientes),
            'empates_recientes'   : int(empates_recientes),
            'derrotas_recientes'  : int(derrotas_recientes),
            'gf_reciente'         : int(gf_reciente),
            'gc_reciente'         : int(gc_reciente),
            'win_rate_reciente'   : round(win_rate_reciente, 4) if not np.isnan(win_rate_reciente) else None,
            'gf_prom_rec'         : round(gf_prom_rec, 4) if not np.isnan(gf_prom_rec) else None,
            'gc_prom_rec'         : round(gc_prom_rec, 4) if not np.isnan(gc_prom_rec) else None,
        })

    df_out = pd.DataFrame(equipos)

    # ── Agregar confederación ────────────────────────────────────────────────
    df_out['confederacion'] = df_out['equipo'].map(CONFEDERACION_MAP).fillna('OTROS')

    # ── Cruzar con Summary: campeones y subcampeones ─────────────────────────
    camp_counts = summary['CHAMPION'].value_counts().rename('summary_campeon')
    sub_counts  = summary['RUNNER UP'].value_counts().rename('summary_sub')
    ter_counts  = summary['THIRD PLACE'].value_counts().rename('summary_tercero')

    df_out = df_out.merge(camp_counts, left_on='equipo', right_index=True, how='left')
    df_out = df_out.merge(sub_counts,  left_on='equipo', right_index=True, how='left')
    df_out = df_out.merge(ter_counts,  left_on='equipo', right_index=True, how='left')

    df_out['summary_campeon']  = df_out['summary_campeon'].fillna(0).astype(int)
    df_out['summary_sub']      = df_out['summary_sub'].fillna(0).astype(int)
    df_out['summary_tercero']  = df_out['summary_tercero'].fillna(0).astype(int)

    # ── Consolidar duplicados por normalización (ej. Germany de dos entradas) ─
    num_cols = [
        'mundiales_jugados', 'partidos_totales', 'victorias_totales',
        'empates_totales', 'derrotas_totales', 'gf_total', 'gc_total',
        'veces_campeon', 'veces_final', 'veces_semifinal', 'veces_cuartos',
        'mundiales_recientes', 'partidos_recientes', 'victorias_recientes',
        'empates_recientes', 'derrotas_recientes', 'gf_reciente', 'gc_reciente',
        'summary_campeon', 'summary_sub', 'summary_tercero',
    ]
    agg_dict = {c: 'sum' for c in num_cols}
    agg_dict['mejor_posicion']  = 'min'
    agg_dict['ultimo_mundial']  = 'max'
    agg_dict['confederacion']   = 'first'

    df_out = df_out.groupby('equipo', as_index=False).agg(agg_dict)

    # Recalcular ratios después de consolidar
    df_out['win_rate_historico'] = (
        df_out['victorias_totales'] / df_out['partidos_totales']
    ).round(4)
    df_out['gf_prom_hist'] = (
        df_out['gf_total'] / df_out['partidos_totales']
    ).round(4)
    df_out['gc_prom_hist'] = (
        df_out['gc_total'] / df_out['partidos_totales']
    ).round(4)
    df_out['win_rate_reciente'] = (
        df_out['victorias_recientes'] / df_out['partidos_recientes'].replace(0, np.nan)
    ).round(4)
    df_out['gf_prom_rec'] = (
        df_out['gf_reciente'] / df_out['partidos_recientes'].replace(0, np.nan)
    ).round(4)
    df_out['gc_prom_rec'] = (
        df_out['gc_reciente'] / df_out['partidos_recientes'].replace(0, np.nan)
    ).round(4)

    # Ordenar columnas y filas
    col_order = [
        'equipo', 'confederacion',
        'mundiales_jugados', 'partidos_totales',
        'victorias_totales', 'empates_totales', 'derrotas_totales',
        'gf_total', 'gc_total',
        'win_rate_historico', 'gf_prom_hist', 'gc_prom_hist',
        'mejor_posicion', 'ultimo_mundial',
        'veces_campeon', 'veces_final', 'veces_semifinal', 'veces_cuartos',
        'mundiales_recientes', 'partidos_recientes',
        'victorias_recientes', 'empates_recientes', 'derrotas_recientes',
        'gf_reciente', 'gc_reciente',
        'win_rate_reciente', 'gf_prom_rec', 'gc_prom_rec',
        'summary_campeon', 'summary_sub', 'summary_tercero',
    ]
    df_out = df_out[col_order]
    df_out = df_out.sort_values('win_rate_historico', ascending=False).reset_index(drop=True)

    return df_out


# ─── Reporte de calidad ────────────────────────────────────────────────────────

def reporte(df: pd.DataFrame) -> None:
    """Imprime un resumen del dataset generado."""
    print('=' * 65)
    print('DATASET GENERADO — RESUMEN')
    print('=' * 65)
    print(f'  Total selecciones únicas : {len(df)}')
    print(f'  Columnas                 : {len(df.columns)}')
    print(f'  Equipos sin datos rec.   : {df["win_rate_reciente"].isna().sum()}')
    print()

    print('Top 15 por win_rate histórico:')
    cols_show = ['equipo', 'mundiales_jugados', 'win_rate_historico',
                 'gf_prom_hist', 'gc_prom_hist', 'veces_campeon',
                 'win_rate_reciente', 'ultimo_mundial']
    print(df[cols_show].head(15).to_string(index=False))
    print()

    print('Equipos sin participación reciente (2014-2022):')
    sin_rec = df[df['mundiales_recientes'] == 0]['equipo'].tolist()
    print(f'  {sin_rec}')
    print('=' * 65)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n=== EXTRACCIÓN Y LIMPIEZA DE DATOS MUNDIALES ===\n')

    print('1. Cargando torneos individuales...')
    todos = cargar_todos_los_torneos()

    print('2. Cargando World Cup Summary...')
    summary = cargar_summary()

    print('3. Construyendo dataset por equipo...')
    df_final = construir_dataset(todos, summary)

    print('4. Guardando CSV...')
    df_final.to_csv(OUTPUT_PATH, index=False, float_format='%.4f')
    print(f'  [OK] Guardado en: {OUTPUT_PATH}')
    print(f'  Filas: {len(df_final)} | Columnas: {len(df_final.columns)}')

    print('\n5. Reporte de calidad:')
    reporte(df_final)

    print('\n[✓] Proceso completado.\n')
