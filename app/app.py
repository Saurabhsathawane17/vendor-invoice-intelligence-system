"""
Vendor Invoice Intelligence System
Improved Streamlit App
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import logging

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lazy import with friendly error ──────────────────────────────────────────
try:
    from src.predict import predict_invoice_risk
except ImportError as e:
    predict_invoice_risk = None
    _import_error = str(e)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* App background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="metric-container"] label {
        color: #8892a4 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem !important;
        color: #e8ecf4 !important;
    }

    /* Section headers */
    h2 { color: #c9d1e0 !important; font-weight: 300 !important; letter-spacing: 0.04em; }
    h3 { color: #8892a4 !important; font-weight: 400 !important; }

    /* Divider */
    hr { border-color: #2a2f3e !important; }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    .risk-high  { background: rgba(255,70,70,0.15); color: #ff4646; border: 1px solid #ff464640; }
    .risk-low   { background: rgba(0,200,140,0.12); color: #00c88c; border: 1px solid #00c88c40; }

    /* Expander */
    details summary { color: #8892a4 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:0.5rem">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
              color:#4f6bff;letter-spacing:0.14em;text-transform:uppercase;">
            VENDOR INTELLIGENCE · v2.0
        </span>
        <h1 style="margin:0;font-size:2rem;font-weight:300;color:#e8ecf4;
                   font-family:'IBM Plex Sans',sans-serif;letter-spacing:0.02em;">
            Invoice Risk Analyzer
        </h1>
        <p style="color:#5a6478;margin:0;font-size:0.9rem;">
            Predict freight cost · Detect anomalies · Explain model decisions
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Guard: missing module ─────────────────────────────────────────────────────
if predict_invoice_risk is None:
    st.error(
        f"⚠️ Could not import `src.predict`. Make sure the module exists and "
        f"dependencies are installed.\n\n`{_import_error}`"
    )
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
        "color:#4f6bff;letter-spacing:0.12em;'>INVOICE PARAMETERS</p>",
        unsafe_allow_html=True,
    )

    quantity = st.number_input(
        "Quantity", min_value=1, value=50, step=1,
        help="Number of units on the invoice."
    )
    unit_price = st.number_input(
        "Unit Price ($)", min_value=0.01, value=12.50, step=0.01, format="%.2f",
        help="Price per unit in USD."
    )
    freight_cost = st.number_input(
        "Reported Freight Cost ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f",
        help="Leave as 0 to let the model predict expected freight."
    )

    st.markdown("<hr style='border-color:#2a2f3e;margin:1rem 0'>", unsafe_allow_html=True)

    # Quick summary in sidebar
    total_val = quantity * unit_price
    st.markdown(
        f"<p style='color:#5a6478;font-size:0.78rem;margin:0'>Order value preview</p>"
        f"<p style='font-family:IBM Plex Mono,monospace;color:#e8ecf4;"
        f"font-size:1.1rem;margin:0'>${total_val:,.2f}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Invoice", use_container_width=True, type="primary")

# ── Main logic ────────────────────────────────────────────────────────────────
if not analyze_btn:
    # Placeholder / welcome state
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 0;color:#2a3040;">
            <div style="font-size:3.5rem">📦</div>
            <p style="font-size:1rem;color:#3a4257;margin-top:0.5rem;">
                Fill in invoice details on the left and click <strong>Analyze Invoice</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Run prediction ────────────────────────────────────────────────────────────
input_data = {
    "Quantity": quantity,
    "UnitPrice": unit_price,
    "freight_cost": freight_cost if freight_cost > 0 else None,
}

models_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)

try:
    with st.spinner("Running model inference…"):
        results = predict_invoice_risk(input_data, models_dir)
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Check your `models/` directory.")
    st.stop()
except Exception as e:
    logger.exception("Prediction failed")
    st.error(f"Prediction failed: {e}")
    st.stop()

# ── Unpack results safely ─────────────────────────────────────────────────────
expected_freight  = results.get("expected_freight_cost", 0.0)
provided_freight  = results.get("provided_freight_cost", freight_cost)
risk_flag         = results.get("risk_flag", 0)
shap_values       = results.get("shap_values")
features          = results.get("features", {})

freight_delta = provided_freight - expected_freight
delta_pct     = (freight_delta / expected_freight * 100) if expected_freight else 0

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Order Value",       f"${quantity * unit_price:,.2f}")
col2.metric("Expected Freight",  f"${expected_freight:,.2f}")
col3.metric(
    "Provided Freight",
    f"${provided_freight:,.2f}",
    delta=f"{freight_delta:+.2f} ({delta_pct:+.1f}%)",
    delta_color="inverse",
)
col4.metric("Risk Score",        "HIGH ⚠️" if risk_flag == 1 else "LOW ✅")

st.markdown("<br>", unsafe_allow_html=True)

# ── Risk banner ───────────────────────────────────────────────────────────────
if risk_flag == 1:
    st.markdown(
        '<div class="risk-badge risk-high">🚨 RISKY INVOICE — Manual review recommended</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="risk-badge risk-low">✅ SAFE INVOICE — Within expected parameters</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col, gauge_col = st.columns([3, 2])

with chart_col:
    st.markdown("#### Freight Cost Comparison")
    fig = go.Figure()
    colors = ["#4f6bff", "#ff4646" if risk_flag else "#00c88c"]
    for label, val, color in zip(
        ["Expected", "Provided"],
        [expected_freight, provided_freight],
        colors,
    ):
        fig.add_trace(
            go.Bar(
                x=[label], y=[val],
                name=label,
                marker_color=color,
                text=[f"${val:,.2f}"],
                textposition="outside",
                width=0.35,
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", family="IBM Plex Mono"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4")),
        margin=dict(t=20, b=20, l=0, r=0),
        yaxis=dict(gridcolor="#1e2535", tickprefix="$", tickfont=dict(color="#5a6478")),
        xaxis=dict(tickfont=dict(color="#8892a4")),
        bargap=0.4,
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

with gauge_col:
    st.markdown("#### Freight Deviation")
    max_gauge = max(abs(delta_pct) * 2, 100)
    gauge_val = min(abs(delta_pct), max_gauge)
    gauge_color = "#ff4646" if risk_flag else "#00c88c"
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=abs(delta_pct),
            number={"suffix": "%", "font": {"color": gauge_color, "family": "IBM Plex Mono"}},
            delta={"reference": 0, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, max_gauge], "tickcolor": "#2a2f3e"},
                "bar": {"color": gauge_color},
                "bgcolor": "#161b27",
                "bordercolor": "#2a2f3e",
                "steps": [
                    {"range": [0, 10],         "color": "#0d1120"},
                    {"range": [10, 30],         "color": "#111827"},
                    {"range": [30, max_gauge],  "color": "#181f30"},
                ],
                "threshold": {
                    "line": {"color": "#ff4646", "width": 2},
                    "thickness": 0.75,
                    "value": 30,
                },
            },
            title={"text": "Deviation %", "font": {"color": "#5a6478", "family": "IBM Plex Sans"}},
        )
    )
    fig_g.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4"),
        height=320,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_g, use_container_width=True)

# ── SHAP ──────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### 🧠 Model Explainability (SHAP)")

if shap_values is not None:
    try:
        fig_shap, ax = plt.subplots(figsize=(9, 4))
        fig_shap.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        plt.rcParams.update({
            "text.color": "#8892a4",
            "axes.labelcolor": "#8892a4",
            "xtick.color": "#5a6478",
            "ytick.color": "#5a6478",
        })
        
        if hasattr(shap_values, "values") and len(shap_values.shape) >= 2:
            shap_exp = shap_values[0]
        else:
            shap_exp = shap_values
            
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
    except Exception as e:
        st.warning(f"Could not render SHAP waterfall chart: {e}")

    with st.expander("📋 Raw SHAP feature values"):
        st.json(features)
else:
    st.info("SHAP values were not returned by the model. Check `predict_invoice_risk` output.")

# ── Audit log / debug expander ────────────────────────────────────────────────
with st.expander("🔬 Full prediction payload (debug)"):
    safe_results = {k: v for k, v in results.items() if k != "shap_values"}
    st.json(safe_results)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr><p style='text-align:center;color:#2a3040;font-size:0.75rem;"
    "font-family:IBM Plex Mono,monospace;'>VENDOR INVOICE INTELLIGENCE · INTERNAL USE ONLY</p>",
    unsafe_allow_html=True,
)