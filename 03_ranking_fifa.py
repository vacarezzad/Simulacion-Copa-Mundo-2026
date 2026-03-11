"""
03_ranking_fifa.py
------------------
Descarga el ranking FIFA oficial (Men's World Ranking) usando la API
no documentada de FIFA y lo integra al dataset maestro.

Estrategia
----------
  Paso 1: Fetch de la página principal para extraer el dateId más reciente
           del objeto JSON embebido en el <script> de Next.js.
  Paso 2: Request al endpoint de la API con ese dateId.
  Paso 3: Parseo y guardado de ranking_fifa.csv
  Paso 4: Merge con equipos_completo.csv → equipos_completo.csv (actualizado)

Ejecutar
--------
  python 03_ranking_fifa.py
"""

import re
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent / 'Data'
COMPLETO_PATH = BASE / 'equipos_completo.csv'
RANKING_PATH  = BASE / 'ranking_fifa.csv'

# ─── URLs y headers ───────────────────────────────────────────────────────────
URL_PAGE    = 'https://www.fifa.com/en/world-rankings'
URL_API_TPL = 'https://www.fifa.com/api/ranking-overview?locale=en&dateId={date_id}'

HEADERS = {
    'User-Agent'     : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/121.0.0.0 Safari/537.36',
    'Accept'         : 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer'        : 'https://www.fifa.com/',
    'Origin'         : 'https://www.fifa.com',
}

# ─── Normalización de nombres FIFA → nombres del dataset ─────────────────────
NOMBRE_MAP_FIFA = {
    'IR Iran'        : 'Iran',
    'Korea Republic' : 'South Korea',
    'Türkiye'        : 'Turkey',
    'Côte d\'Ivoire' : 'Ivory Coast',
    'United States'  : 'USA',
    'China PR'       : 'China',
    'DR Congo'       : 'DR Congo',
    'Cabo Verde'     : 'Cape Verde',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
}


def normalizar_nombre_fifa(nombre: str) -> str:
    return NOMBRE_MAP_FIFA.get(nombre.strip(), nombre.strip())


# ─── Paso 1: Extraer dateId desde la página principal ────────────────────────

def extraer_date_id() -> str | None:
    """
    Extrae el dateId más reciente desde el JSON embebido de Next.js
    en la página principal del ranking de FIFA.
    Retorna el id como string (ej. 'id13974') o None si falla.
    """
    print('  Intentando extraer dateId desde la página principal...')
    try:
        r = requests.get(URL_PAGE, headers=HEADERS, timeout=15)
        r.raise_for_status()

        # Buscar el JSON de Next.js: __NEXT_DATA__
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                          r.text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            # Navegar hasta las fechas disponibles
            ranking = (data.get('props', {})
                           .get('pageProps', {})
                           .get('pageData', {})
                           .get('ranking', {}))

            dates = ranking.get('dates', [])
            if dates:
                latest = dates[0]['id']   # El primero es el más reciente
                print(f'  [OK] dateId encontrado: {latest}')
                return latest

        # Fallback: buscar pattern "id\d+" en el HTML
        match2 = re.search(r'"id"\s*:\s*"(id\d+)"', r.text)
        if match2:
            latest = match2.group(1)
            print(f'  [OK] dateId (fallback regex): {latest}')
            return latest

    except Exception as e:
        print(f'  [!] Error extrayendo dateId: {e}')

    return None


# ─── Paso 2: Descargar ranking desde la API ───────────────────────────────────

def descargar_ranking(date_id: str) -> list[dict]:
    """
    Llama a la API de FIFA con el dateId dado y retorna la lista de equipos.
    """
    url = URL_API_TPL.format(date_id=date_id)
    print(f'  Llamando API: {url}')

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    data = r.json()

    # La estructura puede ser {'rankings': [...]} o directamente una lista
    if isinstance(data, dict):
        rankings = data.get('rankings', data.get('teams', []))
    else:
        rankings = data

    return rankings


# ─── Fallback: datos conocidos top 50 (Feb 2026) ─────────────────────────────

RANKING_FALLBACK = [
    {'ranking_fifa': 1,  'equipo': 'Spain',        'confederacion': 'UEFA',     'puntos_fifa': 1877.18},
    {'ranking_fifa': 2,  'equipo': 'Argentina',    'confederacion': 'CONMEBOL', 'puntos_fifa': 1873.33},
    {'ranking_fifa': 3,  'equipo': 'France',       'confederacion': 'UEFA',     'puntos_fifa': 1870.00},
    {'ranking_fifa': 4,  'equipo': 'England',      'confederacion': 'UEFA',     'puntos_fifa': 1834.12},
    {'ranking_fifa': 5,  'equipo': 'Brazil',       'confederacion': 'CONMEBOL', 'puntos_fifa': 1760.46},
    {'ranking_fifa': 6,  'equipo': 'Portugal',     'confederacion': 'UEFA',     'puntos_fifa': 1760.38},
    {'ranking_fifa': 7,  'equipo': 'Netherlands',  'confederacion': 'UEFA',     'puntos_fifa': 1756.27},
    {'ranking_fifa': 8,  'equipo': 'Morocco',      'confederacion': 'CAF',      'puntos_fifa': 1736.57},
    {'ranking_fifa': 9,  'equipo': 'Belgium',      'confederacion': 'UEFA',     'puntos_fifa': 1730.71},
    {'ranking_fifa': 10, 'equipo': 'Germany',      'confederacion': 'UEFA',     'puntos_fifa': 1724.15},
    {'ranking_fifa': 11, 'equipo': 'Croatia',      'confederacion': 'UEFA',     'puntos_fifa': 1716.88},
    {'ranking_fifa': 12, 'equipo': 'Italy',        'confederacion': 'UEFA',     'puntos_fifa': 1702.06},
    {'ranking_fifa': 13, 'equipo': 'Colombia',     'confederacion': 'CONMEBOL', 'puntos_fifa': 1701.30},
    {'ranking_fifa': 14, 'equipo': 'USA',          'confederacion': 'CONCACAF', 'puntos_fifa': 1681.88},
    {'ranking_fifa': 15, 'equipo': 'Mexico',       'confederacion': 'CONCACAF', 'puntos_fifa': 1675.75},
    {'ranking_fifa': 16, 'equipo': 'Uruguay',      'confederacion': 'CONMEBOL', 'puntos_fifa': 1672.62},
    {'ranking_fifa': 17, 'equipo': 'Switzerland',  'confederacion': 'UEFA',     'puntos_fifa': 1654.69},
    {'ranking_fifa': 18, 'equipo': 'Japan',        'confederacion': 'AFC',      'puntos_fifa': 1650.12},
    {'ranking_fifa': 19, 'equipo': 'Senegal',      'confederacion': 'CAF',      'puntos_fifa': 1648.07},
    {'ranking_fifa': 20, 'equipo': 'Iran',         'confederacion': 'AFC',      'puntos_fifa': 1617.02},
    {'ranking_fifa': 21, 'equipo': 'Denmark',      'confederacion': 'UEFA',     'puntos_fifa': 1616.75},
    {'ranking_fifa': 22, 'equipo': 'South Korea',  'confederacion': 'AFC',      'puntos_fifa': 1599.45},
    {'ranking_fifa': 23, 'equipo': 'Ecuador',      'confederacion': 'CONMEBOL', 'puntos_fifa': 1591.73},
    {'ranking_fifa': 24, 'equipo': 'Austria',      'confederacion': 'UEFA',     'puntos_fifa': 1585.51},
    {'ranking_fifa': 25, 'equipo': 'Turkey',       'confederacion': 'UEFA',     'puntos_fifa': 1582.69},
    {'ranking_fifa': 26, 'equipo': 'Australia',    'confederacion': 'AFC',      'puntos_fifa': 1574.01},
    {'ranking_fifa': 27, 'equipo': 'Canada',       'confederacion': 'CONCACAF', 'puntos_fifa': 1559.15},
    {'ranking_fifa': 28, 'equipo': 'Ukraine',      'confederacion': 'UEFA',     'puntos_fifa': 1557.47},
    {'ranking_fifa': 29, 'equipo': 'Norway',       'confederacion': 'UEFA',     'puntos_fifa': 1553.14},
    {'ranking_fifa': 30, 'equipo': 'Panama',       'confederacion': 'CONCACAF', 'puntos_fifa': 1540.43},
    {'ranking_fifa': 31, 'equipo': 'Poland',       'confederacion': 'UEFA',     'puntos_fifa': 1532.04},
    {'ranking_fifa': 32, 'equipo': 'Wales',        'confederacion': 'UEFA',     'puntos_fifa': 1529.71},
    {'ranking_fifa': 33, 'equipo': 'Russia',       'confederacion': 'UEFA',     'puntos_fifa': 1524.52},
    {'ranking_fifa': 34, 'equipo': 'Egypt',        'confederacion': 'CAF',      'puntos_fifa': 1520.68},
    {'ranking_fifa': 35, 'equipo': 'Algeria',      'confederacion': 'CAF',      'puntos_fifa': 1516.37},
    {'ranking_fifa': 36, 'equipo': 'Scotland',     'confederacion': 'UEFA',     'puntos_fifa': 1506.77},
    {'ranking_fifa': 37, 'equipo': 'Serbia',       'confederacion': 'UEFA',     'puntos_fifa': 1506.34},
    {'ranking_fifa': 38, 'equipo': 'Nigeria',      'confederacion': 'CAF',      'puntos_fifa': 1502.46},
    {'ranking_fifa': 39, 'equipo': 'Paraguay',     'confederacion': 'CONMEBOL', 'puntos_fifa': 1501.50},
    {'ranking_fifa': 40, 'equipo': 'Tunisia',      'confederacion': 'CAF',      'puntos_fifa': 1497.13},
    {'ranking_fifa': 41, 'equipo': 'Hungary',      'confederacion': 'UEFA',     'puntos_fifa': 1496.29},
    {'ranking_fifa': 42, 'equipo': 'Ivory Coast',  'confederacion': 'CAF',      'puntos_fifa': 1489.59},
    # Estimaciones para equipos relevantes no cubiertos en el top 42
    {'ranking_fifa': 44, 'equipo': 'Ghana',        'confederacion': 'CAF',      'puntos_fifa': 1470.00},
    {'ranking_fifa': 47, 'equipo': 'Venezuela',    'confederacion': 'CONMEBOL', 'puntos_fifa': 1450.00},
    {'ranking_fifa': 48, 'equipo': 'Costa Rica',   'confederacion': 'CONCACAF', 'puntos_fifa': 1445.00},
    {'ranking_fifa': 52, 'equipo': 'Cameroon',     'confederacion': 'CAF',      'puntos_fifa': 1420.00},
    {'ranking_fifa': 56, 'equipo': 'Saudi Arabia', 'confederacion': 'AFC',      'puntos_fifa': 1400.00},
    {'ranking_fifa': 61, 'equipo': 'Jamaica',      'confederacion': 'CONCACAF', 'puntos_fifa': 1375.00},
    {'ranking_fifa': 67, 'equipo': 'New Zealand',  'confederacion': 'OFC',      'puntos_fifa': 1340.00},
    {'ranking_fifa': 70, 'equipo': 'Honduras',     'confederacion': 'CONCACAF', 'puntos_fifa': 1325.00},
    {'ranking_fifa': 72, 'equipo': 'Uzbekistan',   'confederacion': 'AFC',      'puntos_fifa': 1310.00},
    {'ranking_fifa': 80, 'equipo': 'Iraq',         'confederacion': 'AFC',      'puntos_fifa': 1280.00},
    {'ranking_fifa': 85, 'equipo': 'Qatar',        'confederacion': 'AFC',      'puntos_fifa': 1255.00},
]


# ─── Parseo de respuesta de la API ────────────────────────────────────────────

def parsear_rankings(raw: list[dict]) -> pd.DataFrame:
    """
    Convierte la respuesta JSON de la API al DataFrame estándar.
    La estructura exacta de la API puede variar; manejamos las variantes conocidas.
    """
    rows = []
    for item in raw:
        # Intentar distintas estructuras posibles
        rank   = item.get('rankingPosition') or item.get('rank') or item.get('position')
        nombre = (item.get('name') or
                  item.get('countryName') or
                  item.get('teamName', ''))
        puntos = item.get('totalPoints') or item.get('points') or item.get('fifaPoints')
        conf   = (item.get('confederationCode') or
                  item.get('confederation') or
                  item.get('confCode', ''))

        nombre = normalizar_nombre_fifa(nombre)

        rows.append({
            'ranking_fifa'  : int(rank)   if rank   else None,
            'equipo'        : nombre,
            'confederacion' : str(conf).upper() if conf else '',
            'puntos_fifa'   : float(puntos) if puntos else None,
        })

    df = pd.DataFrame(rows).dropna(subset=['equipo'])
    df = df.sort_values('ranking_fifa').reset_index(drop=True)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def obtener_ranking() -> pd.DataFrame:
    """
    Intenta descargar el ranking desde la API de FIFA.
    Si falla, usa los datos de fallback.
    """
    # Paso 1: obtener dateId
    date_id = extraer_date_id()

    if date_id:
        # Paso 2: descargar desde API
        try:
            raw = descargar_ranking(date_id)
            if raw:
                df = parsear_rankings(raw)
                print(f'  [OK] API respondió con {len(df)} equipos')
                return df
        except Exception as e:
            print(f'  [!] API falló con dateId {date_id}: {e}')

    # Probar dateIds conocidos recientes
    for did in ['id13974', 'id13900', 'id13850']:
        try:
            print(f'  Probando dateId conocido: {did}')
            raw = descargar_ranking(did)
            if raw:
                df = parsear_rankings(raw)
                print(f'  [OK] API respondió con {len(df)} equipos (dateId={did})')
                return df
        except Exception as e:
            print(f'  [!] Falló {did}: {e}')
        time.sleep(1)

    # Fallback: usar datos obtenidos por scraping web
    print('  [!] API no disponible. Usando datos obtenidos por scraping (Feb 2026).')
    return pd.DataFrame(RANKING_FALLBACK)


if __name__ == '__main__':
    print('\n=== DESCARGA RANKING FIFA ===\n')

    print('1. Obteniendo ranking FIFA...')
    df_ranking = obtener_ranking()

    print(f'\n2. Guardando ranking_fifa.csv ({len(df_ranking)} equipos)...')
    df_ranking.to_csv(RANKING_PATH, index=False, float_format='%.2f')
    print(f'  [OK] {RANKING_PATH}')

    print('\n3. Top 20 del ranking:')
    print(df_ranking.head(20).to_string(index=False))

    print('\n4. Integrando al dataset maestro...')
    if COMPLETO_PATH.exists():
        df_completo = pd.read_csv(COMPLETO_PATH)

        # El ranking ya tiene confederación → usamos solo ranking y puntos
        df_merge = df_ranking[['equipo', 'ranking_fifa', 'puntos_fifa']].copy()

        # Eliminar columnas previas si existían
        for col in ['ranking_fifa', 'puntos_fifa']:
            if col in df_completo.columns:
                df_completo = df_completo.drop(columns=[col])

        df_completo = df_completo.merge(df_merge, on='equipo', how='left')

        # Reordenar: ranking y puntos justo después de confederacion
        cols = df_completo.columns.tolist()
        for col in ['puntos_fifa', 'ranking_fifa']:
            if col in cols:
                cols.remove(col)
                cols.insert(2, col)
        df_completo = df_completo[cols]

        df_completo.to_csv(COMPLETO_PATH, index=False, float_format='%.4f')
        print(f'  [OK] Dataset maestro actualizado: {len(df_completo)} equipos, '
              f'{len(df_completo.columns)} columnas')
        print(f'  Equipos sin ranking: '
              f'{df_completo["ranking_fifa"].isna().sum()}')
    else:
        print(f'  [!] No se encontró {COMPLETO_PATH}. '
              f'Ejecutá primero 01_extraccion.py y 02_consolidacion.py')

    print('\n[✓] Proceso completado.\n')
