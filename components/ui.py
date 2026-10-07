from __future__ import annotations

import streamlit as st


def render_metric_card(
    title: str,
    value: str,
    context: str = "",
    delta: str | None = None,
    help_text: str | None = None,
    delta_color: str = "normal",
):
    """KPI tile: native metric (already a card) + one-line context."""
    label = title if not context else f"{title} • {context}"
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )
def render_status_badge(label: str, status: str):
    """Native status badge using st.badge (fallback to caption)."""
    color_map = {
        "healthy": "green",
        "low": "green",
        "comfortable": "green",
        "balanced": "green",
        "informational": "blue",
        "moderate": "blue",
        "moderately busy": "blue",
        "warning": "orange",
        "moderate impact": "orange",
        "busy": "orange",
        "high": "orange",
        "critical": "red",
        "near capacity": "red",
        "high impact": "red",
        "off": "gray",
        "low impact": "gray",
    }
    color = color_map.get(status.lower(), "gray")
    try:
        # Streamlit >= 1.48
        st.badge(label, color=color)
    except Exception:
        st.caption(f"{label} — {status}")


def apply_styles():
    # ParkSmart "Night Ops" design system: dark enterprise, calm semantics.
    # Green = healthy/available, red = occupied/critical, amber = warning, blue = info.
    # One global <style> only. All content uses native Streamlit + Plotly.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');
        :root {
            --ps-bg: #0b1526;
            --ps-bg2: #0e1b31;
            --ps-card: #13233c;
            --ps-card2: #16263f;
            --ps-border: #22354f;
            --ps-text: #e8eef6;
            --ps-muted: #8ea0b8;
            --ps-green: #34d399;
            --ps-green-dim: rgba(52,211,153,0.14);
            --ps-red: #f87171;
            --ps-red-dim: rgba(248,113,113,0.14);
            --ps-amber: #fbbf24;
            --ps-blue: #60a5fa;
        }
        /* Hide Streamlit auto pages nav (we render our own menu) + chrome */
        [data-testid="stSidebarNav"], [data-testid="stSidebarNavSeparator"] { display: none !important; }
        [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
        footer { visibility: hidden; }
        .stApp { background: radial-gradient(1200px 500px at 80% -5%, #16305a 0%, var(--ps-bg) 55%) fixed, var(--ps-bg); color: var(--ps-text); }
        .block-container { padding-top: 1rem; padding-bottom: 3rem; padding-left: 1.6rem; padding-right: 1.6rem; max-width: 1400px; }
        /* Top bar */
        [data-testid="stHeader"] { background: rgba(11,21,38,0.85); }
        [data-testid="stHeader"] *, [data-testid="stToolbar"] * { color: #fff !important; fill: #fff !important; }
        /* Typography: display font for headings, clean sans for body */
        h1 { font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif !important;
             font-size: 1.5rem !important; font-weight: 700 !important; margin: 0 !important;
             letter-spacing: -0.02em !important; color: #f2f6fb !important; }
        h2 { font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif !important;
             font-size: 1.12rem !important; font-weight: 700 !important; color: #e6edf6 !important;
             letter-spacing: -0.01em !important; }
        h3 { font-size: 0.98rem !important; font-weight: 700 !important; color: #e6edf6 !important; }
        .stCaption, small, p { color: var(--ps-muted); }
        [data-testid="stMarkdownContainer"] p { color: #c6d2e2; }
        /* Sidebar */
        [data-testid="stSidebar"] { background: #0c1728; border-right: 1px solid var(--ps-border); }
        [data-testid="stSidebar"] * { color: var(--ps-text) !important; }
        [data-testid="stSidebar"] .stCaption { color: var(--ps-muted) !important; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: 3px; }
        [data-testid="stSidebar"] [role="radiogroup"] > label {
            border-radius: 8px; padding: 9px 12px !important; margin: 0 !important;
            border: 1px solid transparent;
        }
        [data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: #14263f; }
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
            background: #14263f !important; border: 1px solid var(--ps-border);
            border-left: 3px solid var(--ps-green);
        }
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: #fff !important; font-weight: 700; }
        [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
        /* KPI cards: dark glass tiles, big numbers */
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, var(--ps-card2), var(--ps-card));
            border: 1px solid var(--ps-border);
            border-radius: 14px; padding: 16px 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.35);
            text-align: center;
        }
        [data-testid="stMetric"] label {
            font-size: 0.72rem !important; font-weight: 600 !important;
            color: var(--ps-muted) !important;
        }
        [data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 800 !important; color: #fff !important; }
        [data-testid="stMetricDelta"] { font-size: 0.75rem !important; justify-content: center; }
        /* Cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px !important; border-color: var(--ps-border) !important;
            background: linear-gradient(180deg, rgba(22,38,63,0.9), rgba(19,35,60,0.9));
            box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        }
        /* Progress */
        [data-testid="stProgressBar"] > div > div { border-radius: 4px; background-color: var(--ps-blue); }
        [data-testid="stProgressBar"] { background-color: #1d2f49; }
        /* Tabs */
        [data-testid="stTabs"] [role="tablist"] { gap: 16px; border-bottom: 1px solid var(--ps-border); }
        [data-testid="stTabs"] [role="tab"] { border-radius: 0; padding: 8px 2px; color: var(--ps-muted); border-bottom: 2px solid transparent; font-weight: 600; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #fff !important; border-bottom: 2px solid var(--ps-green); font-weight: 700; }
        /* Charts + tables */
        [data-testid="stPlotlyChart"] { background: var(--ps-card); border: 1px solid var(--ps-border); border-radius: 14px; padding: 8px; }
        [data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid var(--ps-border); }
        [data-testid="stExpander"] { border-radius: 12px; background: var(--ps-card); border: 1px solid var(--ps-border); }
        /* Buttons */
        .stButton > button { border-radius: 10px !important; font-weight: 600 !important; border: 1px solid var(--ps-border) !important; background: #fff !important; }
        .stButton > button * { color: #0b1526 !important; }
        .stButton > button:hover { border-color: var(--ps-green) !important; }
        .stButton > button:hover * { color: #06281d !important; }
        .stButton > button[kind="primary"] { background: var(--ps-green) !important; border-color: var(--ps-green) !important; }
        .stButton > button[kind="primary"] * { color: #06281d !important; }
        .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div > input { border-radius: 8px !important; }
        .stAlert { border-radius: 12px !important; }
        :focus-visible { outline: 2px solid var(--ps-green); outline-offset: 1px; }
        /* Tables scroll instead of overflowing */
        [data-testid="stDataFrame"] { overflow-x: auto; }
        @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
        @media (max-width: 900px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            h1 { font-size: 1.25rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
        }
        @media (max-width: 640px) {
            /* Stack columns: 2 KPI cards per row instead of squeezed 4 */
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.5rem !important; }
            [data-testid="stHorizontalBlock"] [data-testid="stColumn"] { flex: 1 1 44% !important; min-width: 140px !important; }
            [data-testid="stMetric"] { padding: 12px 8px; }
            [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
            h1 { font-size: 1.2rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
