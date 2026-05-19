"""
app.py — Interface Streamlit pour le Fire Risk Project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR   = BASE_DIR / "data" / "figures"

LAT_MIN, LAT_MAX = 35.0, 47.0
LON_MIN, LON_MAX = -5.0, 28.0
RES = 0.25
LATS = np.arange(LAT_MIN, LAT_MAX + RES / 2, RES)
LONS = np.arange(LON_MIN, LON_MAX + RES / 2, RES)
N_LAT, N_LON = len(LATS), len(LONS)

FEATURE_LABELS = {
    "ndvi":               "NDVI",
    "temp_max":           "Température max (°C)",
    "humidity":           "Humidité relative (%)",
    "wind_speed":         "Vitesse vent (m/s)",
    "precip":             "Précipitations (mm)",
    "drought_index":      "Indice sécheresse",
    "past_hotspot_count": "Hotspots passés",
    "past_frp":           "FRP moyen (MW)",
}

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(15,17,26,0)",
    plot_bgcolor="rgba(15,17,26,0)",
    font=dict(color="#E0E0E0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0c14 0%, #0f1420 40%, #12101a 100%);
    color: #E0E0E0;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0a00 0%, #2d0f00 30%, #1a0520 70%, #0a0c14 100%);
    border: 1px solid rgba(255,80,0,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(255,80,0,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(ellipse, rgba(255,80,0,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF6B35, #FF4500, #FF8C42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: rgba(220,220,220,0.7);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,80,0,0.12);
    border: 1px solid rgba(255,80,0,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    color: #FF8C42;
    font-weight: 500;
}

/* ── Tab Bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 500;
    font-size: 0.9rem;
    color: rgba(220,220,220,0.6);
    background: transparent;
    transition: all 0.2s ease;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,80,0,0.25), rgba(255,140,66,0.15)) !important;
    color: #FF8C42 !important;
    border: 1px solid rgba(255,80,0,0.3) !important;
    box-shadow: 0 0 20px rgba(255,80,0,0.1);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(255,80,0,0.3);
    box-shadow: 0 4px 20px rgba(255,80,0,0.08);
}
[data-testid="metric-container"] label {
    color: rgba(200,200,200,0.65) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #FF4500, #FF6B35) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(255,69,0,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255,69,0,0.45) !important;
    background: linear-gradient(135deg, #FF5500, #FF7B45) !important;
}

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div,
.stDateInput > div > div > input,
.stSlider > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E0E0E0 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #FF4500 !important;
    border: 2px solid #FF6B35 !important;
}
.stSlider [data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #FF4500, #FF6B35) !important;
}

/* ── Info / Warning / Success boxes ── */
.stInfo {
    background: rgba(30,136,229,0.1) !important;
    border: 1px solid rgba(30,136,229,0.3) !important;
    border-radius: 12px !important;
    color: #90CAF9 !important;
}
.stSuccess {
    background: rgba(67,160,71,0.1) !important;
    border: 1px solid rgba(67,160,71,0.3) !important;
    border-radius: 12px !important;
    color: #A5D6A7 !important;
}
.stWarning {
    background: rgba(255,152,0,0.1) !important;
    border: 1px solid rgba(255,152,0,0.3) !important;
    border-radius: 12px !important;
}
.stError {
    background: rgba(229,57,53,0.1) !important;
    border: 1px solid rgba(229,57,53,0.3) !important;
    border-radius: 12px !important;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 1rem 0;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F0F0F0;
    margin: 0;
}
.section-accent {
    height: 3px;
    width: 36px;
    background: linear-gradient(90deg, #FF4500, transparent);
    border-radius: 3px;
}

/* ── Stat card ── */
.stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: rgba(255,80,0,0.25);
    background: linear-gradient(135deg, rgba(255,80,0,0.05), rgba(255,255,255,0.02));
}
.stat-value { font-size: 1.8rem; font-weight: 800; color: #FF8C42; }
.stat-label { font-size: 0.78rem; color: rgba(200,200,200,0.6); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border-radius: 50px;
    padding: 0.5rem 1.4rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 0 auto;
    width: fit-content;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── DataFrame ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,80,0,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,80,0,0.5); }

/* ── Spinner ── */
.stSpinner > div > div { border-top-color: #FF4500 !important; }

/* ── Caption / small text ── */
.stCaption { color: rgba(200,200,200,0.5) !important; }

/* ── subheader override ── */
h2, h3 { color: #F0F0F0 !important; }
</style>
"""


def apply_styles():
    st.markdown(CSS, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(
        f"""<div class="section-header">
              <p class="section-title">{title}</p>
              <div class="section-accent"></div>
            </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_parquet(year: int) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"dataset_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_history(model_name: str) -> dict | None:
    path = PROCESSED_DIR / f"history_{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics(model_name: str) -> dict | None:
    for d in [FIGURES_DIR, PROCESSED_DIR]:
        path = d / f"metrics_{model_name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


@st.cache_resource(show_spinner=False)
def load_model_cached(model_type: str):
    try:
        from inference import load_model
        return load_model(model_type, PROCESSED_DIR)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_scaler_cached():
    try:
        from inference import load_scaler
        return load_scaler(PROCESSED_DIR)
    except Exception:
        return None


def has_real_data(year: int) -> bool:
    path = PROCESSED_DIR / f"dataset_{year}.parquet"
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["past_hotspot_count"])
        return bool(df["past_hotspot_count"].sum() > 0)
    except Exception:
        return False


def has_2023_data() -> bool:
    return has_real_data(2023)


def has_2024_data() -> bool:
    return has_real_data(2024)


def best_available_year() -> int:
    if has_2024_data():
        return 2024
    if has_2023_data():
        return 2023
    return 2022


# ---------------------------------------------------------------------------
# TAB 1 — Prédiction
# ---------------------------------------------------------------------------

def tab_prediction():
    examples = {
        2022: "**Gironde (France)** : lat=44.50, lon=-0.75, date=2022-08-10 → Critique 1.00",
        2023: "**Penteli/Attique (Grèce)** : lat=38.25, lon=24.00, date=2023-07-31 → Élevé 0.52",
        2024: "**Attique/Athènes** : lat=38.25, lon=23.75, date=2024-08-14 → Critique 0.99",
    }
    best_year = best_available_year()
    st.info(f"**Exemple à fort risque ({best_year})** — {examples[best_year]}")

    col_side, col_main = st.columns([1, 3], gap="large")

    with col_side:
        st.markdown(
            """<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:16px;padding:1.4rem 1.2rem;margin-bottom:1rem;">
            <p style="font-size:0.85rem;font-weight:600;color:#FF8C42;text-transform:uppercase;
            letter-spacing:0.08em;margin:0 0 1rem 0;">⚙️ Paramètres</p>""",
            unsafe_allow_html=True,
        )

        model_choice = st.selectbox("Modèle IA", ["LSTM", "TCN"], key="pred_model")

        if has_2024_data():
            default_date = pd.Timestamp("2024-08-14").date()
            min_d = pd.Timestamp("2022-01-15").date()
            max_d = pd.Timestamp("2024-12-31").date()
        elif has_2023_data():
            default_date = pd.Timestamp("2023-07-31").date()
            min_d = pd.Timestamp("2022-01-15").date()
            max_d = pd.Timestamp("2023-12-31").date()
        else:
            default_date = pd.Timestamp("2022-08-10").date()
            min_d = pd.Timestamp("2022-01-15").date()
            max_d = pd.Timestamp("2022-12-31").date()

        date_sel = st.date_input("Date", value=default_date, min_value=min_d, max_value=max_d, key="pred_date")
        lat_sel  = st.slider("Latitude",  LAT_MIN, LAT_MAX, 43.3, step=0.25, key="pred_lat")
        lon_sel  = st.slider("Longitude", LON_MIN, LON_MAX, 5.4,  step=0.25, key="pred_lon")

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🔍 Analyser le risque", use_container_width=True)

    with col_main:
        # Location map
        fig_loc = go.Figure(go.Scattergeo(
            lat=[lat_sel], lon=[lon_sel],
            mode="markers",
            marker=dict(size=16, color="#FF4500", symbol="circle",
                        line=dict(width=2, color="#FF8C42")),
            name="Point sélectionné",
        ))
        fig_loc.update_geos(
            lataxis_range=[LAT_MIN - 1, LAT_MAX + 1],
            lonaxis_range=[LON_MIN - 1, LON_MAX + 1],
            showland=True,    landcolor="#1a1f2e",
            showcoastlines=True, coastlinecolor="#3a4060",
            showocean=True,   oceancolor="#0d1520",
            showlakes=True,   lakecolor="#0d1520",
            showrivers=True,  rivercolor="#1a2540",
            showcountries=True, countrycolor="#2a3050",
            projection_type="equirectangular",
            bgcolor="rgba(0,0,0,0)",
        )
        fig_loc.update_layout(
            **PLOTLY_DARK,
            margin=dict(l=0, r=0, t=30, b=0),
            height=240,
            title=dict(
                text=f"<b>📍 {lat_sel:.2f}°N, {lon_sel:.2f}°E</b>",
                font=dict(size=13, color="#FF8C42"),
                x=0.5,
            ),
            geo=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_loc, use_container_width=True)

        if predict_btn:
            with st.spinner("Analyse en cours..."):
                try:
                    from inference import predict as infer_predict

                    result  = infer_predict(
                        lat=lat_sel, lon=lon_sel,
                        date=str(date_sel),
                        model_type=model_choice.lower(),
                        processed_dir=PROCESSED_DIR,
                    )
                    score   = result["risk_score"]
                    niveau  = result["niveau"]
                    couleur = result["couleur"]

                    if result.get("fallback_2022"):
                        st.info("ℹ️ Données 2023 absentes — utilisation proxy 2022")

                    # ── Gauge + badge ──
                    g_col, b_col = st.columns([3, 2])
                    with g_col:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score * 100,
                            number={"suffix": "%", "font": {"size": 40, "color": "#FFFFFF"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#666", "tickwidth": 1,
                                         "tickfont": {"color": "#888"}},
                                "bar": {"color": couleur, "thickness": 0.28},
                                "bgcolor": "rgba(0,0,0,0)",
                                "borderwidth": 0,
                                "steps": [
                                    {"range": [0,  20], "color": "rgba(76,175,80,0.15)"},
                                    {"range": [20, 40], "color": "rgba(255,235,59,0.12)"},
                                    {"range": [40, 60], "color": "rgba(255,152,0,0.15)"},
                                    {"range": [60, 100],"color": "rgba(244,67,54,0.15)"},
                                ],
                                "threshold": {"line": {"color": couleur, "width": 3},
                                              "thickness": 0.8, "value": score * 100},
                            },
                            title={"text": f"Score de risque", "font": {"size": 14, "color": "#AAA"}},
                        ))
                        fig_gauge.update_layout(**PLOTLY_DARK, height=230,
                                                margin=dict(t=30, b=0, l=10, r=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with b_col:
                        badges_map = {
                            "Faible":   ("🟢", "rgba(76,175,80,0.15)",  "rgba(76,175,80,0.5)",  "#81C784"),
                            "Modéré":   ("🟡", "rgba(255,235,59,0.15)", "rgba(255,235,59,0.5)", "#FFF176"),
                            "Élevé":    ("🟠", "rgba(255,152,0,0.15)",  "rgba(255,152,0,0.5)",  "#FFB74D"),
                            "Critique": ("🔴", "rgba(244,67,54,0.15)",  "rgba(244,67,54,0.5)",  "#EF9A9A"),
                        }
                        emoji, bg, border, txt = badges_map.get(niveau, ("🔥","rgba(255,80,0,0.15)","rgba(255,80,0,0.5)","#FF8C42"))
                        st.markdown(
                            f"""<div style="background:{bg};border:1px solid {border};
                            border-radius:16px;padding:1.5rem;text-align:center;margin-top:1rem;">
                            <div style="font-size:2.5rem;margin-bottom:0.5rem;">{emoji}</div>
                            <div style="font-size:1.5rem;font-weight:800;color:{txt};">{niveau}</div>
                            <div style="font-size:0.85rem;color:rgba(200,200,200,0.6);margin-top:0.3rem;">
                            Score : {score:.3f}</div>
                            <div style="font-size:0.8rem;color:rgba(200,200,200,0.5);margin-top:0.2rem;">
                            Modèle : {model_choice}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    # ── Time series ──
                    section_header("Série temporelle — 15 derniers jours")
                    series = result["serie_temporelle"]
                    x = series["dates"]

                    fig_ts = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("🌡️ Température max (°C)", "💧 Humidité (%)",
                                        "🌿 NDVI", "🔥 Risque prédit"),
                        horizontal_spacing=0.1, vertical_spacing=0.15,
                    )
                    traces = [
                        (series["temp_max"],                             "#FF6B35", 1, 1),
                        (series["humidity"],                             "#42A5F5", 1, 2),
                        (series["ndvi"],                                 "#66BB6A", 2, 1),
                        ([r if r else 0 for r in series["risk_predit"]], couleur,   2, 2),
                    ]
                    for vals, clr, r, c in traces:
                        fig_ts.add_trace(
                            go.Scatter(x=x, y=vals, mode="lines+markers",
                                       line=dict(color=clr, width=2),
                                       marker=dict(size=5, color=clr),
                                       showlegend=False),
                            row=r, col=c,
                        )

                    fig_ts.update_layout(
                        **PLOTLY_DARK,
                        height=370,
                        margin=dict(t=50, b=10, l=10, r=10),
                    )
                    for ann in fig_ts.layout.annotations:
                        ann.font.color = "#AAAAAA"
                        ann.font.size  = 12
                    st.plotly_chart(fig_ts, use_container_width=True)

                    # ── Features table ──
                    st.divider()
                    section_header("Features utilisées (moyenne 15 jours)")
                    feats   = result["features_utilisees"]
                    feat_df = pd.DataFrame(
                        [{"Feature": FEATURE_LABELS.get(k, k), "Valeur": f"{v:.3f}"}
                         for k, v in feats.items()]
                    )

                    # Horizontal bar chart for features
                    fig_feat = go.Figure(go.Bar(
                        x=list(feats.values()),
                        y=[FEATURE_LABELS.get(k, k) for k in feats],
                        orientation="h",
                        marker=dict(
                            color=list(feats.values()),
                            colorscale=[[0,"#1E4D8C"],[0.5,"#FF8C42"],[1,"#FF4500"]],
                            showscale=False,
                        ),
                        text=[f"{v:.3f}" for v in feats.values()],
                        textposition="outside",
                        textfont=dict(color="#CCCCCC", size=11),
                    ))
                    _feat_layout = {k: v for k, v in PLOTLY_DARK.items() if k != "xaxis"}
                    fig_feat.update_layout(
                        **_feat_layout,
                        height=280,
                        margin=dict(t=10, b=10, l=10, r=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                   zerolinecolor="rgba(255,255,255,0.1)"),
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)

                except FileNotFoundError as e:
                    st.error(f"❌ Fichier manquant : {e}")
                    st.info("Lancez : `python src/pipeline.py` puis `python src/train.py`")
                except ValueError as e:
                    st.warning(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# TAB 2 — Carte de risque
# ---------------------------------------------------------------------------

def tab_risk_map():
    col1, col2, col3 = st.columns([2, 2, 1], gap="medium")
    with col1:
        if has_2024_data():
            default_d = pd.Timestamp("2024-08-14").date()
            min_d, max_d = pd.Timestamp("2022-01-16").date(), pd.Timestamp("2024-12-31").date()
        elif has_2023_data():
            default_d = pd.Timestamp("2023-07-31").date()
            min_d, max_d = pd.Timestamp("2022-01-16").date(), pd.Timestamp("2023-12-31").date()
        else:
            default_d = pd.Timestamp("2022-08-10").date()
            min_d, max_d = pd.Timestamp("2022-01-16").date(), pd.Timestamp("2022-12-31").date()
        map_date = st.date_input("Date", value=default_d, min_value=min_d, max_value=max_d, key="map_date")
    with col2:
        map_model = st.selectbox("Modèle", ["LSTM", "TCN"], key="map_model")
    with col3:
        st.write("")
        st.write("")
        gen_btn = st.button("🗺️ Générer", use_container_width=True)

    if not has_2023_data():
        st.warning("⏳ Données 2023 absentes → affichage données 2022")

    if gen_btn:
        with st.spinner("Calcul de la carte méditerranéenne..."):
            try:
                from inference import predict_grid

                risk_grid = predict_grid(
                    str(map_date), model_type=map_model.lower(), processed_dir=PROCESSED_DIR,
                )

                grid_records = [
                    {"lat": float(LATS[li]), "lon": float(LONS[lj]), "risk": float(risk_grid[li, lj])}
                    for li in range(len(LATS)) for lj in range(len(LONS))
                ]
                grid_df = pd.DataFrame(grid_records)

                fig_map = px.density_mapbox(
                    grid_df, lat="lat", lon="lon", z="risk",
                    radius=12,
                    color_continuous_scale=[
                        [0.0,  "#1a5c1a"],
                        [0.25, "#8bc34a"],
                        [0.5,  "#ffc107"],
                        [0.75, "#ff5722"],
                        [1.0,  "#b71c1c"],
                    ],
                    range_color=[0, 1],
                    mapbox_style="carto-darkmatter",
                    zoom=4,
                    center={"lat": 41, "lon": 11},
                    labels={"risk": "Risque"},
                )
                fig_map.update_layout(
                    **PLOTLY_DARK,
                    height=560,
                    margin=dict(t=10, b=0, l=0, r=0),
                    coloraxis_colorbar=dict(
                        title=dict(text="Risque", font=dict(color="#FF8C42")),
                        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                        ticktext=["Nul", "Faible", "Modéré", "Élevé", "Critique"],
                        bgcolor="rgba(15,17,26,0.8)",
                        bordercolor="rgba(255,255,255,0.1)",
                        tickfont=dict(color="#CCCCCC"),
                    ),
                )
                st.plotly_chart(fig_map, use_container_width=True)

                # FIRMS overlay
                year_data = load_parquet(2023 if has_2023_data() else 2022)
                if year_data is not None:
                    hotspot_day = year_data[
                        (year_data["date"].dt.date == map_date) &
                        (year_data["past_hotspot_count"] > 0)
                    ]
                    if not hotspot_day.empty:
                        st.caption(f"⚫ {len(hotspot_day)} cellules avec hotspots FIRMS réels ({map_date})")
                        fig_map2 = go.Figure(fig_map)
                        fig_map2.add_trace(go.Scattermapbox(
                            lat=hotspot_day["lat"].tolist(),
                            lon=hotspot_day["lon"].tolist(),
                            mode="markers",
                            marker=dict(size=7, color="#FFFFFF", opacity=0.9),
                            name="Hotspots réels FIRMS",
                        ))
                        st.plotly_chart(fig_map2, use_container_width=True)

                # Stats
                st.divider()
                section_header("Statistiques de la carte")
                stats = [
                    ("Score moyen",    f"{risk_grid.mean():.3f}", "📊"),
                    ("Score max",      f"{risk_grid.max():.3f}",  "🔺"),
                    ("Cellules > 0.5", str(int((risk_grid > 0.5).sum())), "⚠️"),
                    ("Cellules > 0.8", str(int((risk_grid > 0.8).sum())), "🔴"),
                ]
                s_cols = st.columns(4)
                for col, (label, val, icon) in zip(s_cols, stats):
                    col.markdown(
                        f"""<div class="stat-card">
                        <div style="font-size:1.5rem;">{icon}</div>
                        <div class="stat-value">{val}</div>
                        <div class="stat-label">{label}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                st.info("Lancez `python src/pipeline.py` puis `python src/train.py`")
            except Exception as e:
                st.error(f"Erreur : {e}")
                import traceback
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# TAB 3 — Évaluation
# ---------------------------------------------------------------------------

def tab_evaluation():
    st.markdown(
        """<div style="background:linear-gradient(135deg,rgba(30,136,229,0.08),rgba(30,136,229,0.03));
        border:1px solid rgba(30,136,229,0.2);border-radius:14px;padding:1rem 1.4rem;margin-bottom:1.5rem;">
        <b style="color:#90CAF9;">Split temporel</b><br>
        <span style="color:rgba(200,200,200,0.7);font-size:0.88rem;">
        🔵 Entraînement : 01/01/2022 → 15/12/2022 &nbsp;|&nbsp;
        🟡 Validation : 16/12/2022 → 31/12/2022 &nbsp;|&nbsp;
        🔴 Test : 01/01/2023 → 31/12/2023
        </span></div>""",
        unsafe_allow_html=True,
    )

    section_header("Métriques sur le Test set (2023)")
    m_lstm = load_metrics("lstm")
    m_tcn  = load_metrics("tcn")

    if m_lstm or m_tcn:
        metric_names = ["MAE", "RMSE", "Spearman", "AUC"]
        cols = st.columns(len(metric_names))
        for i, mname in enumerate(metric_names):
            with cols[i]:
                v_lstm = m_lstm.get(mname, float("nan")) if m_lstm else float("nan")
                v_tcn  = m_tcn.get(mname,  float("nan")) if m_tcn  else float("nan")
                v_lstm_str = f"{v_lstm:.4f}" if not np.isnan(v_lstm) else "—"
                v_tcn_str  = f"{v_tcn:.4f}"  if not np.isnan(v_tcn)  else "—"
                st.markdown(
                    f"""<div class="stat-card">
                    <div style="font-size:0.75rem;font-weight:600;color:#FF8C42;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">{mname}</div>
                    <div style="display:flex;justify-content:space-around;">
                      <div style="text-align:center;">
                        <div style="font-size:1.2rem;font-weight:700;color:#42A5F5;">{v_lstm_str}</div>
                        <div style="font-size:0.72rem;color:rgba(180,180,180,0.6);">LSTM</div>
                      </div>
                      <div style="width:1px;background:rgba(255,255,255,0.08);"></div>
                      <div style="text-align:center;">
                        <div style="font-size:1.2rem;font-weight:700;color:#EF5350;">{v_tcn_str}</div>
                        <div style="font-size:0.72rem;color:rgba(180,180,180,0.6);">TCN</div>
                      </div>
                    </div></div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Métriques non disponibles. Lance `python src/evaluate.py`.")

    st.divider()

    # Learning curves
    section_header("Courbes d'apprentissage")
    h_lstm = load_history("lstm")
    h_tcn  = load_history("tcn")

    if h_lstm or h_tcn:
        fig_loss = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss (Weighted MSE)", "MAE de validation"),
            horizontal_spacing=0.12,
        )
        for h, name, color in [(h_lstm, "LSTM", "#42A5F5"), (h_tcn, "TCN", "#EF5350")]:
            if h is None:
                continue
            epochs = list(range(1, len(h["train_loss"]) + 1))
            fig_loss.add_trace(
                go.Scatter(x=epochs, y=h["train_loss"], name=f"{name} Train",
                           line=dict(color=color, dash="solid", width=2),
                           fill="tozeroy", fillcolor=color.replace(")", ",0.05)").replace("rgb", "rgba") if color.startswith("rgb") else f"rgba(66,165,245,0.05)" if name=="LSTM" else "rgba(239,83,80,0.05)"),
                row=1, col=1,
            )
            fig_loss.add_trace(
                go.Scatter(x=epochs, y=h["val_loss"], name=f"{name} Val",
                           line=dict(color=color, dash="dash", width=1.5)),
                row=1, col=1,
            )
            if "val_mae" in h:
                fig_loss.add_trace(
                    go.Scatter(x=epochs, y=h["val_mae"], name=f"{name} MAE",
                               line=dict(color=color, width=2)),
                    row=1, col=2,
                )

        fig_loss.update_xaxes(title_text="Époque", **PLOTLY_DARK.get("xaxis", {}))
        fig_loss.update_yaxes(title_text="Loss", row=1, col=1, **PLOTLY_DARK.get("yaxis", {}))
        fig_loss.update_yaxes(title_text="MAE",  row=1, col=2, **PLOTLY_DARK.get("yaxis", {}))
        fig_loss.update_layout(
            **PLOTLY_DARK,
            height=370,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                        bgcolor="rgba(0,0,0,0)", font=dict(color="#CCC")),
            margin=dict(t=50, b=10),
        )
        for ann in fig_loss.layout.annotations:
            ann.font.color = "#AAAAAA"
        st.plotly_chart(fig_loss, use_container_width=True)

    # Scatter predicted vs real
    st.divider()
    section_header("Risque prédit vs réel (Test 2023)")
    y_test_path = PROCESSED_DIR / "y_test.npy"
    if y_test_path.exists():
        y_test = np.load(y_test_path)
        scatter_cols = st.columns(2)
        for col_idx, (model_type, color) in enumerate([("lstm", "#42A5F5"), ("tcn", "#EF5350")]):
            ckpt = PROCESSED_DIR / f"best_model_{model_type}.pt"
            if not ckpt.exists():
                scatter_cols[col_idx].warning(f"{model_type.upper()} : modèle non entraîné")
                continue
            try:
                import torch
                from model import build_model as bm

                model = bm(model_type, n_features=8)
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model_state"])
                model.eval()

                X_test = np.load(PROCESSED_DIR / "X_test.npy")
                T, NL, NG, F = X_test.shape
                SEQ, N_cells = 15, NL * NG
                n_win  = T - SEQ + 1
                X_flat = X_test.reshape(T, N_cells, F)
                y_flat = y_test.reshape(T, N_cells)
                seqs   = np.zeros((N_cells, n_win, SEQ, F), dtype=np.float32)
                ytrue  = np.zeros((N_cells, n_win), dtype=np.float32)
                for t in range(n_win):
                    seqs[:, t] = X_flat[t:t+SEQ].transpose(1, 0, 2)
                    ytrue[:, t] = y_flat[t+SEQ-1]

                X_all = torch.from_numpy(seqs.reshape(-1, SEQ, F))
                BATCH, preds_list = 4096, []
                with torch.no_grad():
                    for s in range(0, len(X_all), BATCH):
                        preds_list.append(model(X_all[s:s+BATCH]).numpy())
                y_pred = np.clip(np.concatenate(preds_list), 0, 1)
                y_true = ytrue.reshape(-1)
                idx    = np.random.choice(len(y_true), size=min(3000, len(y_true)), replace=False)

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=y_true[idx], y=y_pred[idx],
                    mode="markers",
                    marker=dict(color=color, opacity=0.25, size=4,
                                line=dict(width=0)),
                    name=model_type.upper(),
                ))
                fig_sc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=1.5),
                    name="Parfait",
                ))
                fig_sc.update_layout(
                    **PLOTLY_DARK,
                    title=dict(text=f"<b>{model_type.upper()}</b> — Réel vs Prédit",
                               font=dict(color="#CCC", size=13), x=0.5),
                    xaxis_title="Risk score réel",
                    yaxis_title="Risk score prédit",
                    height=350,
                    xaxis_range=[0, 1], yaxis_range=[0, 1],
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CCC")),
                    margin=dict(t=40, b=40),
                )
                scatter_cols[col_idx].plotly_chart(fig_sc, use_container_width=True)
            except Exception as e:
                scatter_cols[col_idx].error(f"Erreur scatter {model_type}: {e}")
    else:
        st.info("y_test.npy absent. Lance `python src/pipeline.py`.")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="FireRisk — Méditerranée",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_styles()

    # Hero banner
    st.markdown(
        """<div class="hero-banner">
          <h1 class="hero-title">🔥 FireRisk — Bassin Méditerranéen</h1>
          <p class="hero-subtitle">
            Prédiction du risque d'incendie par deep learning (LSTM &amp; TCN)<br>
            Données ERA5 · MODIS NDVI · NASA FIRMS · Résolution 0.25°
          </p>
          <div class="hero-badges">
            <span class="hero-badge">🧠 LSTM</span>
            <span class="hero-badge">⚡ TCN</span>
            <span class="hero-badge">🛰️ ERA5</span>
            <span class="hero-badge">🌿 MODIS NDVI</span>
            <span class="hero-badge">🔥 NASA FIRMS</span>
            <span class="hero-badge">🗺️ Méditerranée</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "🔥  Prédiction",
        "🗺️  Carte de risque",
        "📊  Évaluation",
    ])

    with tabs[0]:
        tab_prediction()
    with tabs[1]:
        tab_risk_map()
    with tabs[2]:
        tab_evaluation()


if __name__ == "__main__":
    main()
