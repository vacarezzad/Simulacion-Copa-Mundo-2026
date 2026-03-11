"""
app.py — Streamlit
------------------
Interfaz para el simulador Monte Carlo del Mundial 2026.

Ejecutar:
    streamlit run app.py
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
from itertools import combinations

from src.config import SimConfig
from src.montecarlo import correr_simulacion, correr_simulacion_paralelo, resumen_texto
from src.torneo import GRUPOS_2026_REALES

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Simulador Mundial 2026",
    page_icon   = "⚽",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

DATA_PATH = Path(__file__).parent / "Data" / "clasificados_2026.csv"

COLORES_CONF = {
    "UEFA"    : "#003399",
    "CONMEBOL": "#009B3A",
    "CONCACAF": "#EF3340",
    "AFC"     : "#FF6600",
    "CAF"     : "#FFD700",
    "OFC"     : "#00AACC",
}

# ─── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Parámetros del modelo
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parámetros del modelo")

    st.subheader("📊 Fuente de datos")
    alpha_reciente = st.slider(
        "Peso de datos recientes (2014–2022)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="0 = solo historial completo (1930-2022) | 1 = solo últimas 3 ediciones"
    )
    beta_fifa = st.slider(
        "Corrección por ranking FIFA actual",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="Ajusta los promedios históricos según los puntos FIFA vigentes (Feb 2026)"
    )
    debutante_mode = st.selectbox(
        "Imputación de debutantes",
        options=['conf_avg', 'puntaje', 'global_min'],
        format_func=lambda x: {
            'conf_avg'  : '📐 Promedio de confederación',
            'puntaje'   : '📈 Interpolación por puntos FIFA',
            'global_min': '📉 Peor del dataset (pesimista)',
        }[x],
        help="Curacao, Jordan, Cape Verde y Uzbekistán no tienen historial en Mundiales"
    )

    st.divider()

    st.subheader("⚽ Modelo de goles")
    base_goles = st.slider(
        "Goles base por equipo por partido",
        min_value=1.00, max_value=1.80, value=1.35, step=0.05,
        help="Media global histórica en Mundiales. Sube → más goles → más incertidumbre"
    )
    factor_local = st.slider(
        "Ventaja de sede (USA / Canadá / México)",
        min_value=1.00, max_value=1.20, value=1.08, step=0.01,
        help="Multiplicador sobre los goles esperados para los 3 países anfitriones"
    )
    min_partidos_rec = st.slider(
        "Muestra mínima de partidos recientes",
        min_value=3, max_value=15, value=6, step=1,
        help=(
            "Equipos con menos partidos en 2014–2022 usan solo el historial completo. "
            "Evita que promedios de 2 torneos (ej: Netherlands no clasificó 2018) dominen el modelo."
        )
    )
    bonus_campeon = st.slider(
        "Bonus campeón vigente (Argentina 2022)",
        min_value=1.00, max_value=1.20, value=1.08, step=0.01,
        help="Multiplicador extra sobre el λ de ataque del actual campeón del mundo"
    )

    st.divider()

    st.subheader("🔴 Fases eliminatorias")
    simular_et = st.checkbox(
        "Simular tiempo extra (30 min)",
        value=True,
        help="Si está desmarcado, el empate en 90' va directo a penales"
    )
    penalty_weight = st.slider(
        "Influencia del ranking en penales",
        min_value=0.00, max_value=0.40, value=0.15, step=0.05,
        help="0 = 50-50 puro | 0.4 = fuerte ventaja al equipo mejor rankeado"
    )
    penalty_k = st.select_slider(
        "Escala de diferencia en penales",
        options=[200, 300, 400, 500, 600],
        value=400,
        help="Mayor valor → diferencias de puntos FIFA impactan menos en penales"
    )

    st.divider()

    st.subheader("🎲 Simulación")
    n_sims = st.select_slider(
        "Número de simulaciones",
        options=[10_000, 50_000, 100_000, 250_000, 500_000],
        value=100_000,
        format_func=lambda x: f"{x:,}",
        help="Más simulaciones = más precisión pero más tiempo de cómputo"
    )
    usar_semilla = st.checkbox("Usar semilla fija (reproducible)", value=True)
    seed = st.number_input("Semilla", min_value=0, max_value=99999, value=42,
                           disabled=not usar_semilla)

    st.divider()

    st.subheader("⚡ Rendimiento")
    _n_cpu = os.cpu_count() or 4
    n_workers = st.select_slider(
        "Núcleos para paralelismo",
        options=[1, 2, 4, 8],
        value=min(4, _n_cpu),
        format_func=lambda x: f"{'🔵 Secuencial' if x == 1 else f'🟢 {x} núcleos'}",
        help=(
            f"Tu equipo tiene {_n_cpu} núcleo(s) disponibles. "
            "Modo secuencial (1) muestra barra de progreso fina. "
            "Modo paralelo divide el trabajo entre núcleos → 3-6× más rápido. "
            "Recomendado ≥ 50.000 simulaciones para amortizar el arranque."
        ),
    )

    st.divider()
    simular_btn = st.button("▶  Simular torneo", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚽ Simulador Monte Carlo — Copa del Mundo 2026")
st.caption("Modelo Poisson (Dixon-Coles simplificado) | 48 equipos | 12 grupos | R32 → Final")

df_clasif = cargar_datos()

# ── Estado de sesión ──────────────────────────────────────────────────────────
for key in ['resultados', 'escenarios', 'config_usada']:
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# Correr simulación
# ─────────────────────────────────────────────────────────────────────────────
if simular_btn:
    config = SimConfig(
        alpha_reciente         = alpha_reciente,
        beta_fifa              = beta_fifa,
        base_goles             = base_goles,
        factor_local           = factor_local,
        min_partidos_recientes = min_partidos_rec,
        bonus_campeon_vigente  = bonus_campeon,
        simular_tiempo_extra   = simular_et,
        penalty_weight         = penalty_weight,
        penalty_k              = float(penalty_k),
        debutante_mode         = debutante_mode,
        n_simulaciones         = n_sims,
        seed                   = int(seed) if usar_semilla else None,
    )

    usar_paralelo = n_workers > 1
    modo_txt = f"{n_workers} núcleos en paralelo" if usar_paralelo else "modo secuencial"

    with st.spinner(f"Corriendo {n_sims:,} simulaciones ({modo_txt})..."):
        progress_bar = st.progress(0.0, text="Iniciando...")
        t0 = time.time()

        if usar_paralelo:
            # ── Modo paralelo: callback por worker completado ──────────────────
            def cb_par(done: int, total: int):
                pct = done / total
                progress_bar.progress(
                    pct,
                    text=f"Workers completados: {done}/{total} ({pct:.0%}) — "
                         f"aprox. {(time.time()-t0)*(total-done)/max(done,1):.0f} s restantes",
                )

            try:
                df_result, escenarios = correr_simulacion_paralelo(
                    df_clasif, config,
                    callback  = cb_par,
                    n_workers = n_workers,
                )
            except Exception as exc:
                # Fallback robusto al modo secuencial si el paralelismo falla
                st.warning(
                    f"⚠️ Paralelismo no disponible ({exc}). "
                    "Usando modo secuencial..."
                )
                progress_bar.progress(0.0, text="Modo secuencial (fallback)...")

                def cb(actual, total):
                    pct = actual / total
                    progress_bar.progress(
                        pct, text=f"{actual:,} / {total:,} torneos ({pct:.0%})"
                    )

                df_result, escenarios = correr_simulacion(df_clasif, config, callback=cb)

        else:
            # ── Modo secuencial: progress bar fina (por 1 % de avance) ────────
            def cb(actual, total):
                pct = actual / total
                progress_bar.progress(
                    pct, text=f"{actual:,} / {total:,} torneos ({pct:.0%})"
                )

            df_result, escenarios = correr_simulacion(df_clasif, config, callback=cb)

        elapsed = time.time() - t0

    progress_bar.empty()
    st.session_state.resultados   = df_result
    st.session_state.escenarios   = escenarios
    st.session_state.config_usada = config

    sims_por_seg = n_sims / elapsed
    st.success(
        f"✅ {n_sims:,} simulaciones completadas en **{elapsed:.1f} s** "
        f"({sims_por_seg:,.0f} torneos/seg) — {modo_txt}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 helper — Partidos frecuentes
# ─────────────────────────────────────────────────────────────────────────────

def _stats_grupo_partido(cnt: Counter, n: int):
    """
    A partir del Counter {(ga, gb): count} de un duelo de fase de grupos,
    calcula P(A gana), P(Empate), P(B gana) y el resultado más frecuente.
    ga = goles del equipo ta (alfabéticamente primero), gb del tb.
    """
    p_a   = sum(c for (ga, gb), c in cnt.items() if ga > gb) / n
    p_d   = sum(c for (ga, gb), c in cnt.items() if ga == gb) / n
    p_b   = sum(c for (ga, gb), c in cnt.items() if ga < gb) / n
    top   = cnt.most_common(1)
    if top:
        (ga_t, gb_t), cnt_t = top[0]
        score = f"{ga_t}-{gb_t}"
        freq  = cnt_t / n
    else:
        score, freq = "N/A", 0.0
    return p_a, p_d, p_b, score, freq


def _stats_ko_partido(cnt: Counter, n: int):
    """
    A partir del Counter {(ga, gb, metodo, ta_wins): count} de un duelo KO,
    calcula P(ta gana), P(tb gana) por método y el resultado más frecuente.
    """
    p_ta_90 = sum(c for (ga, gb, m, tw), c in cnt.items() if tw and m == '90min') / n
    p_tb_90 = sum(c for (ga, gb, m, tw), c in cnt.items() if not tw and m == '90min') / n
    p_ta_et = sum(c for (ga, gb, m, tw), c in cnt.items() if tw and m == 'ET') / n
    p_tb_et = sum(c for (ga, gb, m, tw), c in cnt.items() if not tw and m == 'ET') / n
    p_ta_pk = sum(c for (ga, gb, m, tw), c in cnt.items() if tw and m == 'Penales') / n
    p_tb_pk = sum(c for (ga, gb, m, tw), c in cnt.items() if not tw and m == 'Penales') / n
    p_ta = p_ta_90 + p_ta_et + p_ta_pk
    p_tb = p_tb_90 + p_tb_et + p_tb_pk

    top = cnt.most_common(1)
    if top:
        (ga_t, gb_t, m_t, tw_t), cnt_t = top[0]
        label_m = {'90min': '', 'ET': ' (ET)', 'Penales': ' (Pen)'}[m_t]
        if m_t == 'Penales':
            score = f"{ga_t}-{gb_t}{label_m}"
        else:
            score = f"{ga_t}-{gb_t}{label_m}"
        freq = cnt_t / n
    else:
        score, freq = "N/A", 0.0
    return p_ta, p_tb, p_ta_90, p_ta_et, p_ta_pk, p_tb_90, p_tb_et, p_tb_pk, score, freq


def _mostrar_tab_partidos(escenarios, n_total, df_clasif, COLORES_CONF, GRUPOS_2026_REALES):
    """Contenido completo del tab '⚽ Partidos frecuentes'."""
    rg = escenarios.get('resultados_grupos', {})
    rk = escenarios.get('resultados_ko', {})

    # ── Selector de sección ───────────────────────────────────────────────────
    seccion = st.radio(
        "Ver:",
        ["🏆 Eliminatorias", "📋 Fase de grupos"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECCIÓN: ELIMINATORIAS
    # ═══════════════════════════════════════════════════════════════════════════
    if seccion == "🏆 Eliminatorias":
        st.subheader("Partidos más frecuentes por ronda KO")
        st.caption(
            "Probabilidad de que cada duelo ocurra y su resultado más habitual. "
            "En penales el score refleja el tiempo reglamentario + extra."
        )

        ronda_labels = {
            'final'  : '🏅 Final',
            'semis'  : '4️⃣ Semifinales',
            'cuartos': '8️⃣ Cuartos de final',
            'r16'    : '16️⃣ Round of 16',
            'r32'    : '32️⃣ Round of 32',
        }
        ronda_sel = st.select_slider(
            "Ronda",
            options=list(ronda_labels.keys()),
            format_func=lambda x: ronda_labels[x],
            value='final',
        )

        matchups_ronda = rk.get(ronda_sel, {})
        n_mostrar_ko   = st.slider(
            "Duelos a mostrar", 5, 40,
            {'final': 20, 'semis': 20, 'cuartos': 15, 'r16': 12, 'r32': 10}[ronda_sel],
            key="sl_ko_n",
        )

        # Ordenar matchups por frecuencia de aparición
        ranked_ko = sorted(
            matchups_ronda.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        )[:n_mostrar_ko]

        if not ranked_ko:
            st.info("No hay datos para esta ronda aún. Ejecutá una simulación.")
        else:
            # Build DataFrame para el gráfico y la tabla
            rows_ko = []
            for (ta, tb), cnt in ranked_ko:
                total = sum(cnt.values())
                freq  = total / n_total
                p_ta, p_tb, p_ta_90, p_ta_et, p_ta_pk, p_tb_90, p_tb_et, p_tb_pk, score, freq_score = \
                    _stats_ko_partido(cnt, n_total)

                conf_ta = df_clasif.loc[df_clasif.equipo == ta, 'confederacion'].values
                conf_ta = conf_ta[0] if len(conf_ta) else ''
                conf_tb = df_clasif.loc[df_clasif.equipo == tb, 'confederacion'].values
                conf_tb = conf_tb[0] if len(conf_tb) else ''

                rows_ko.append({
                    'Duelo'             : f"{ta} 🆚 {tb}",
                    'ta'                : ta, 'tb': tb,
                    'P(ocurre)'         : freq,
                    f'P({ta} gana)'     : p_ta,
                    f'P({tb} gana)'     : p_tb,
                    'Score más común'   : score,
                    'Frec. score'       : freq_score,
                    'P(90min)'          : (p_ta_90 + p_tb_90) / max(freq, 1e-9),
                    'P(ET)'             : (p_ta_et + p_tb_et) / max(freq, 1e-9),
                    'P(Penales)'        : (p_ta_pk + p_tb_pk) / max(freq, 1e-9),
                    'conf_ta'           : conf_ta,
                })

            df_ko = pd.DataFrame(rows_ko)

            # ── Gráfico: P(ocurre) con color del ganador más probable ────────
            df_ko['Ganador probable'] = df_ko.apply(
                lambda r: r['ta'] if r[f"P({r['ta']} gana)"] >= r[f"P({r['tb']} gana)"]
                          else r['tb'],
                axis=1,
            )
            df_ko['% ganador'] = df_ko.apply(
                lambda r: max(r[f"P({r['ta']} gana)"], r[f"P({r['tb']} gana)"]) / max(r['P(ocurre)'], 1e-9),
                axis=1,
            )

            fig_ko = go.Figure()
            for _, row in df_ko.iterrows():
                ta, tb = row['ta'], row['tb']
                p_ta_val = row[f'P({ta} gana)']
                p_tb_val = row[f'P({tb} gana)']
                p_tot    = row['P(ocurre)']
                duelo    = row['Duelo']
                score_lbl = f"  {row['Score más común']} ({row['Frec. score']:.0%})"

                c_ta = COLORES_CONF.get(row['conf_ta'], '#888')

                fig_ko.add_trace(go.Bar(
                    name=ta, x=[p_ta_val], y=[duelo],
                    orientation='h', marker_color=c_ta,
                    showlegend=False,
                    hovertemplate=f"<b>{ta}</b> gana: {p_ta_val:.1%}<extra></extra>",
                ))
                fig_ko.add_trace(go.Bar(
                    name=tb, x=[p_tb_val], y=[duelo],
                    orientation='h', marker_color='#CCCCCC',
                    showlegend=False,
                    text=[score_lbl] if _ == df_ko.index[-1] else [''],
                    hovertemplate=f"<b>{tb}</b> gana: {p_tb_val:.1%}<extra></extra>",
                ))

            fig_ko.update_layout(
                barmode='stack',
                title=f"{ronda_labels[ronda_sel]} — Top {len(df_ko)} duelos",
                xaxis=dict(title='Probabilidad', tickformat='.0%'),
                yaxis=dict(categoryorder='total ascending'),
                height=max(350, len(df_ko) * 34),
                margin=dict(l=10, r=20, t=40, b=20),
                legend_title='Equipo',
            )
            st.plotly_chart(fig_ko, use_container_width=True)

            # ── Tabla detallada ───────────────────────────────────────────────
            st.caption("Tabla detallada — haz clic en una columna para ordenar")
            tabla_cols = ['Duelo', 'P(ocurre)', 'Score más común', 'Frec. score',
                          'P(90min)', 'P(ET)', 'P(Penales)']
            # Añadir columnas de victoria dinámica
            tabla_cols += [f'P({r["ta"]} gana)' for _, r in df_ko.iterrows()][:1]

            df_tabla_ko = df_ko[['Duelo', 'P(ocurre)', 'Score más común', 'Frec. score',
                                  'P(90min)', 'P(ET)', 'P(Penales)']].copy()

            # Añadir columnas de victoria por equipo
            for _, row in df_ko.iterrows():
                ta, tb = row['ta'], row['tb']
                df_tabla_ko.loc[_, f'P({ta})'] = row[f'P({ta} gana)']
                df_tabla_ko.loc[_, f'P({tb})'] = row[f'P({tb} gana)']

            st.dataframe(
                df_tabla_ko.style.format({
                    'P(ocurre)' : '{:.1%}',
                    'Frec. score': '{:.1%}',
                    'P(90min)'  : '{:.0%}',
                    'P(ET)'     : '{:.0%}',
                    'P(Penales)': '{:.0%}',
                    **{c: '{:.1%}' for c in df_tabla_ko.columns
                       if c.startswith('P(') and c not in ('P(ocurre)',)}
                }),
                use_container_width=True,
                height=min(600, len(df_ko) * 38 + 60),
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECCIÓN: FASE DE GRUPOS
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        st.subheader("Resultados más frecuentes — Fase de grupos")
        st.caption(
            "Cada partido del round-robin se jugó en todas las simulaciones. "
            "Las probabilidades reflejan la distribución empírica de resultados."
        )

        col_gs, col_info = st.columns([1, 3])
        with col_gs:
            grupo_sel = st.radio(
                "Grupo",
                options=list(GRUPOS_2026_REALES.keys()),
                format_func=lambda g: f"Grupo {g}  ({' · '.join(GRUPOS_2026_REALES[g])})",
            )

        with col_info:
            eqs = GRUPOS_2026_REALES[grupo_sel]
            st.markdown(f"### Grupo {grupo_sel}")
            st.caption(f"{' · '.join(eqs)}")

            # ── Stacked bar chart W/D/L para los 6 partidos ─────────────────
            rows_g = []
            for ta, tb in combinations(eqs, 2):
                key = (ta, tb) if ta <= tb else (tb, ta)
                cnt = rg.get(key, Counter())
                # normalize: if key is (tb, ta) we need to swap
                if ta > tb:  # key is (tb, ta), ga=goles de tb, gb=goles de ta
                    cnt_adj = Counter({(gb, ga): c for (ga, gb), c in cnt.items()})
                else:
                    cnt_adj = cnt

                p_a, p_d, p_b, score, freq_s = _stats_grupo_partido(cnt_adj, n_total)
                rows_g.append({
                    'Partido'          : f"{ta} - {tb}",
                    'ta': ta, 'tb': tb,
                    f'P({ta})'         : p_a,
                    'P(Empate)'        : p_d,
                    f'P({tb})'         : p_b,
                    'Score más común'  : score,
                    'Frecuencia score' : freq_s,
                })

            df_g = pd.DataFrame(rows_g)

            # Gráfico stacked bar (ta / empate / tb)
            fig_g = go.Figure()
            color_wins = '#1f77b4'
            color_draw = '#aec7e8'
            color_loss = '#d62728'

            for i, row in df_g.iterrows():
                ta, tb = row['ta'], row['tb']
                partido = row['Partido']
                p_a_v = row[f'P({ta})']
                p_d_v = row['P(Empate)']
                p_b_v = row[f'P({tb})']
                score_txt = f"  Score más común: {row['Score más común']} ({row['Frecuencia score']:.0%})"

                fig_g.add_trace(go.Bar(
                    name=ta, x=[p_a_v], y=[partido], orientation='h',
                    marker_color=color_wins, showlegend=(i == 0),
                    legendgroup='wins',
                    hovertemplate=f"<b>{ta} gana</b>: {p_a_v:.1%}<extra></extra>",
                ))
                fig_g.add_trace(go.Bar(
                    name='Empate', x=[p_d_v], y=[partido], orientation='h',
                    marker_color=color_draw, showlegend=(i == 0),
                    legendgroup='draw',
                    hovertemplate=f"<b>Empate</b>: {p_d_v:.1%}<extra></extra>",
                ))
                fig_g.add_trace(go.Bar(
                    name=tb, x=[p_b_v], y=[partido], orientation='h',
                    marker_color=color_loss, showlegend=(i == 0),
                    legendgroup='loss',
                    hovertemplate=f"<b>{tb} gana</b>: {p_b_v:.1%}<extra></extra>",
                ))

            fig_g.update_layout(
                barmode='stack',
                xaxis=dict(title='Probabilidad', tickformat='.0%', range=[0, 1]),
                yaxis=dict(categoryorder='array',
                           categoryarray=[r['Partido'] for r in rows_g][::-1]),
                height=320,
                margin=dict(l=10, r=10, t=10, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # ── Tabla de detalle ─────────────────────────────────────────────
            df_tabla_g = pd.DataFrame([{
                'Partido'         : r['Partido'],
                f'P({r["ta"]} gana)': r[f'P({r["ta"]})'],
                'P(Empate)'       : r['P(Empate)'],
                f'P({r["tb"]} gana)': r[f'P({r["tb"]})'],
                'Score más común' : r['Score más común'],
                'Frecuencia'      : r['Frecuencia score'],
            } for _, r in df_g.iterrows()])

            pct_cols = [c for c in df_tabla_g.columns
                        if c.startswith('P(') or c == 'Frecuencia']
            st.dataframe(
                df_tabla_g.style.format({c: '{:.1%}' for c in pct_cols}),
                use_container_width=True,
                hide_index=True,
            )

            # ── Top 5 scores más comunes para el partido más frecuente ───────
            st.divider()
            st.markdown("##### 📊 Distribución de scores por partido")
            partido_sel_idx = st.selectbox(
                "Seleccionar partido",
                options=list(range(len(rows_g))),
                format_func=lambda i: rows_g[i]['Partido'],
                key="sel_partido_grupo",
            )
            row_sel = rows_g[partido_sel_idx]
            ta_s, tb_s = row_sel['ta'], row_sel['tb']
            key_s = (ta_s, tb_s) if ta_s <= tb_s else (tb_s, ta_s)
            cnt_s = rg.get(key_s, Counter())
            if ta_s > tb_s:
                cnt_s = Counter({(gb, ga): c for (ga, gb), c in cnt_s.items()})

            top_scores = cnt_s.most_common(12)
            if top_scores:
                df_scores = pd.DataFrame([
                    {'Score': f"{ga}-{gb}", 'Ocurrencias': c, 'Probabilidad': c / n_total}
                    for (ga, gb), c in top_scores
                ])
                fig_sc = px.bar(
                    df_scores, x='Score', y='Probabilidad',
                    text=df_scores['Probabilidad'].apply(lambda x: f"{x:.1%}"),
                    labels={'Probabilidad': 'P(score)', 'Score': ''},
                    title=f"{ta_s} vs {tb_s} — distribución de marcadores",
                    color='Probabilidad',
                    color_continuous_scale='Blues',
                    height=320,
                )
                fig_sc.update_traces(textposition='outside')
                fig_sc.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=40, b=20),
                    yaxis_tickformat='.0%',
                )
                st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Resultados
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.resultados is not None:
    df          = st.session_state.resultados
    escenarios  = st.session_state.escenarios
    config_usada = st.session_state.config_usada
    n_total     = escenarios['n']

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Probabilidades",
        "🎯 Escenarios frecuentes",
        "⚽ Partidos frecuentes",
        "📊 Etapas",
        "🌍 Por confederación",
        "🔬 Datos",
    ])

    # ── Tab 1: Gráfico principal ──────────────────────────────────────────────
    with tab1:
        st.subheader("Probabilidad de ser campeón del mundo")

        fig = px.bar(
            df.head(20),
            x='p_campeon', y='equipo',
            orientation='h',
            color='confederacion',
            color_discrete_map=COLORES_CONF,
            text=df.head(20)['p_campeon'].apply(lambda x: f"{x:.1%}"),
            labels={'p_campeon': 'P(Campeón)', 'equipo': ''},
            height=620,
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat='.0%',
            showlegend=True,
            legend_title="Confederación",
            margin=dict(l=10, r=80, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        top = df.iloc[0]
        c1.metric("🥇 Máximo favorito", top['equipo'], f"{top['p_campeon']:.1%} de ganar")
        uefa_top = df[df['confederacion'] == 'UEFA'].iloc[0]
        c2.metric("📈 Mejor europeo", uefa_top['equipo'], f"{uefa_top['p_campeon']:.1%}")
        csur_top = df[df['confederacion'] == 'CONMEBOL'].iloc[0]
        c3.metric("🌎 Mejor CONMEBOL", csur_top['equipo'], f"{csur_top['p_campeon']:.1%}")
        c4.metric("🎯 Equipos con > 5% de ganar", int((df['p_campeon'] > 0.05).sum()))

    # ── Tab 2: Escenarios frecuentes ─────────────────────────────────────────
    with tab2:

        # ── Sección A: Finals más frecuentes ─────────────────────────────────
        st.subheader("🏟️ Finals más frecuentes")
        st.caption("Cada barra = un escenario de final concreta. El equipo en primer lugar es el campeón.")

        finales_raw = escenarios['finales']
        top_n_finales = st.slider("Mostrar top N finales", 5, 30, 15, key="sl_finales")

        filas_final = [
            {
                'Final'    : f"{camp} 🆚 {final}",
                'Campeón'  : camp,
                'Finalista': final,
                'Frec.'    : cnt,
                'Prob.'    : cnt / n_total,
            }
            for (camp, final), cnt in finales_raw.most_common(top_n_finales)
        ]
        df_finales = pd.DataFrame(filas_final)

        fig_f = px.bar(
            df_finales,
            x='Prob.', y='Final',
            orientation='h',
            text=df_finales['Prob.'].apply(lambda x: f"{x:.2%}"),
            color='Campeón',
            color_discrete_map={eq: COLORES_CONF.get(
                df_clasif.loc[df_clasif['equipo'] == eq, 'confederacion'].values[0]
                if eq in df_clasif['equipo'].values else '', '#888'
            ) for eq in df_finales['Campeón'].unique()},
            labels={'Prob.': 'Probabilidad', 'Final': ''},
            height=max(350, top_n_finales * 32),
        )
        fig_f.update_traces(textposition='outside')
        fig_f.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat='.1%',
            showlegend=False,
            margin=dict(l=10, r=80, t=20, b=20),
        )
        st.plotly_chart(fig_f, use_container_width=True)

        # ── Sección B: Ganadores de grupo ─────────────────────────────────────
        st.divider()
        st.subheader("📋 Ganadores más probables por grupo")
        st.caption("Para cada grupo: los 4 equipos ordenados por P(ganar el grupo).")

        cols_grupos = st.columns(4)
        grupos_letras = list(GRUPOS_2026_REALES.keys())

        for idx, letra in enumerate(grupos_letras):
            col = cols_grupos[idx % 4]
            counter = escenarios['ganadores_grupo'][letra]
            equipos_grupo = GRUPOS_2026_REALES[letra]

            rows_g = []
            for eq in equipos_grupo:
                cnt = counter.get(eq, 0)
                conf = df_clasif.loc[df_clasif['equipo'] == eq, 'confederacion'].values
                conf = conf[0] if len(conf) > 0 else ''
                rows_g.append({'Equipo': eq, 'P(1°)': cnt / n_total, 'Conf': conf})

            df_g = pd.DataFrame(rows_g).sort_values('P(1°)', ascending=False)

            with col:
                st.markdown(f"**Grupo {letra}**")
                for _, r in df_g.iterrows():
                    color = COLORES_CONF.get(r['Conf'], '#888')
                    pct   = r['P(1°)']
                    bar   = int(pct * 20)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
                        f"<span style='width:90px;font-size:12px'>{r['Equipo']}</span>"
                        f"<div style='background:{color};width:{bar*8}px;height:10px;"
                        f"border-radius:3px;min-width:4px'></div>"
                        f"<span style='font-size:12px;color:#aaa'>{pct:.0%}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.write("")

        # ── Sección C: Semifinalistas más frecuentes ──────────────────────────
        st.divider()
        st.subheader("🔝 Grupos de semifinalistas más frecuentes")
        st.caption("Cada fila = los 4 equipos que llegaron a semis juntos en la misma simulación.")

        top_n_semis = st.slider("Mostrar top N combinaciones", 5, 20, 10, key="sl_semis")
        semis_raw   = escenarios['semifinales']

        filas_semis = []
        for combo, cnt in semis_raw.most_common(top_n_semis):
            equipos_sorted = sorted(combo)
            filas_semis.append({
                'Semifinalistas': ' · '.join(equipos_sorted),
                'Veces'         : cnt,
                'Prob.'         : cnt / n_total,
            })

        df_semis = pd.DataFrame(filas_semis)
        fig_s = px.bar(
            df_semis,
            x='Prob.', y='Semifinalistas',
            orientation='h',
            text=df_semis['Prob.'].apply(lambda x: f"{x:.2%}"),
            color_discrete_sequence=['#5588CC'],
            labels={'Prob.': 'Probabilidad', 'Semifinalistas': ''},
            height=max(300, top_n_semis * 34),
        )
        fig_s.update_traces(textposition='outside')
        fig_s.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat='.2%',
            showlegend=False,
            margin=dict(l=10, r=80, t=20, b=20),
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # ── Sección D: Campeón por confederación ─────────────────────────────
        st.divider()
        st.subheader("🌍 Probabilidad de que el campeón sea de cada confederación")

        conf_cnt = escenarios['campeon_por_conf']
        df_conf_camp = pd.DataFrame([
            {'Confederación': k, 'Prob.': v / n_total}
            for k, v in conf_cnt.most_common()
        ])
        fig_cc = px.pie(
            df_conf_camp,
            names='Confederación', values='Prob.',
            color='Confederación',
            color_discrete_map=COLORES_CONF,
            hole=0.4,
        )
        fig_cc.update_traces(textinfo='label+percent', textposition='outside')
        fig_cc.update_layout(height=380, showlegend=False,
                              margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cc, use_container_width=True)

    # ── Tab 3: Partidos frecuentes ───────────────────────────────────────────
    with tab3:
        _mostrar_tab_partidos(escenarios, n_total, df_clasif, COLORES_CONF, GRUPOS_2026_REALES)

    # ── Tab 4: Heatmap por etapa ──────────────────────────────────────────────
    with tab4:
        st.subheader("Probabilidad de alcanzar cada etapa")

        cols_etapa = {
            'p_r32'    : 'R32',
            'p_r16'    : 'R16',
            'p_cuartos': 'Cuartos',
            'p_semis'  : 'Semis',
            'p_final'  : 'Final',
            'p_campeon': 'Campeón',
        }
        n_mostrar = st.slider("Equipos a mostrar", 10, 48, 24, key="heatmap_n")
        df_heat   = df.head(n_mostrar)[['equipo'] + list(cols_etapa.keys())].copy()
        df_heat   = df_heat.rename(columns=cols_etapa)

        fig2 = px.imshow(
            df_heat.set_index('equipo')[list(cols_etapa.values())].values,
            x=list(cols_etapa.values()),
            y=df_heat['equipo'].tolist(),
            color_continuous_scale='Blues',
            zmin=0, zmax=1,
            text_auto='.0%',
            aspect='auto',
            height=max(400, n_mostrar * 22),
        )
        fig2.update_coloraxes(colorbar_tickformat='.0%')
        fig2.update_layout(margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 5: Por confederación ──────────────────────────────────────────────
    with tab5:
        st.subheader("Probabilidad acumulada por confederación")

        df_conf = (
            df.groupby('confederacion')[['p_campeon', 'p_final', 'p_semis', 'p_r16']]
            .sum().reset_index().sort_values('p_campeon', ascending=False)
        )
        fig3 = px.bar(
            df_conf, x='confederacion',
            y=['p_campeon', 'p_final', 'p_semis', 'p_r16'],
            barmode='group',
            labels={'value': 'Prob. acumulada', 'variable': 'Etapa'},
            color_discrete_map={
                'p_campeon': '#FFD700', 'p_final': '#C0C0C0',
                'p_semis'  : '#CD7F32', 'p_r16'  : '#6699CC',
            },
        )
        fig3.update_layout(yaxis_tickformat='.0%', height=420)
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(
            df_conf.style.format({
                'p_campeon': '{:.1%}', 'p_final': '{:.1%}',
                'p_semis'  : '{:.1%}', 'p_r16'  : '{:.1%}',
            }),
            use_container_width=True,
        )

    # ── Tab 6: Datos del modelo ───────────────────────────────────────────────
    with tab6:
        st.subheader("Tabla completa de probabilidades")

        cols_show = {
            'equipo'       : 'Equipo',
            'confederacion': 'Conf.',
            'ranking_fifa' : 'Ranking FIFA',
            'puntos_fifa'  : 'Puntos FIFA',
            'p_r32'        : 'R32',
            'p_r16'        : 'R16',
            'p_cuartos'    : 'Cuartos',
            'p_semis'      : 'Semis',
            'p_final'      : 'Final',
            'p_campeon'    : 'Campeón',
        }
        available = {k: v for k, v in cols_show.items() if k in df.columns}
        df_show   = df[list(available.keys())].rename(columns=available).copy()

        fmt = {
            'R32': '{:.1%}', 'R16': '{:.1%}', 'Cuartos': '{:.1%}',
            'Semis': '{:.1%}', 'Final': '{:.1%}', 'Campeón': '{:.1%}',
            'Puntos FIFA': '{:.0f}', 'Ranking FIFA': '{:.0f}',
        }
        try:
            styled = df_show.style.format(fmt).background_gradient(
                subset=['Campeón'], cmap='YlOrRd'
            )
        except Exception:
            styled = df_show.style.format(fmt)

        st.dataframe(styled, use_container_width=True, height=600)

        st.code(resumen_texto(df), language=None)

        st.subheader("Parámetros usados")
        st.json({
            "alpha_reciente"      : config_usada.alpha_reciente,
            "beta_fifa"           : config_usada.beta_fifa,
            "base_goles"          : config_usada.base_goles,
            "factor_local"        : config_usada.factor_local,
            "simular_tiempo_extra": config_usada.simular_tiempo_extra,
            "penalty_weight"      : config_usada.penalty_weight,
            "penalty_k"           : config_usada.penalty_k,
            "debutante_mode"      : config_usada.debutante_mode,
            "n_simulaciones"      : config_usada.n_simulaciones,
            "seed"                : config_usada.seed,
        })

else:
    st.info("👈 Configurá los parámetros en el panel izquierdo y presioná **▶ Simular torneo**.")

    with st.expander("📋 Ver los 48 clasificados y sus grupos", expanded=True):
        # Tabla con grupo asignado
        grupo_map = {eq: g for g, eqs in GRUPOS_2026_REALES.items() for eq in eqs}
        df_prev = df_clasif.copy()
        df_prev['Grupo'] = df_prev['equipo'].map(grupo_map)

        cols_preview = ['Grupo', 'equipo', 'confederacion', 'ranking_fifa',
                        'puntos_fifa', 'win_rate_historico', 'veces_campeon']
        cols_preview = [c for c in cols_preview if c in df_prev.columns]

        st.dataframe(
            df_prev[cols_preview]
            .sort_values(['Grupo', 'puntos_fifa'], ascending=[True, False])
            .style.format({
                'ranking_fifa'      : '{:.0f}',
                'puntos_fifa'       : '{:.0f}',
                'win_rate_historico': '{:.1%}',
            }),
            use_container_width=True,
            height=600,
        )
