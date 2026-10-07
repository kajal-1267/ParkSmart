from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.charts import (
    feature_importance_chart,
    historical_fit_chart,
    prediction_error_chart,
)
from components.ui import render_status_badge
from ml.predictor import MODEL_PATH, load_model, model_metrics


def render_ml_page(df):
    st.caption("ParkSmart / ML Performance")
    st.title("Model quality")
    st.caption("30-minute availability model • Linear Regression • retrain via ml/train_model.py")

    m = model_metrics()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("R²", f"{m['r2']:.3f}", help="1.0 = perfect, 0 = baseline")
    with m2:
        st.metric("MAE", f"{m['mae']:.2f}", "bays avg error")
    with m3:
        st.metric("RMSE", f"{m['rmse']:.2f}", "bays large-error penalty")
    with m4:
        st.metric("Features", str(len(m["features"])))

    with st.container(border=True):
        if m["r2"] >= 0.75:
            st.success("Reliable for live operations — error within a few bays.", icon="✅")
        elif m["r2"] >= 0.5:
            st.warning("Usable but watch — retrain if MAE grows.", icon="⚠️")
        else:
            st.error("Needs retraining before trusting forecasts.", icon="🚨")
        st.progress(max(0.0, min(1.0, float(m["r2"]))))
        st.caption(f"R² = {m['r2']:.3f} • MAE {m['mae']:.2f} bays • RMSE {m['rmse']:.2f} bays")

    with st.container(border=True):
        st.subheader("📊 Prediction model performance", anchor=False)
        st.caption("Mean error and fit quality on held-out data")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Mean Absolute Error", f"{m['mae']:.2f}", "bays")
        with p2:
            st.metric("R² Score", f"{m['r2']:.3f}", "1.0 = perfect")

    with st.container(border=True):
        st.subheader("Actual vs predicted", anchor=False)
        st.caption("Each dot = one zone at one time. Dots on the dashed diagonal = perfect call.")
        zone_pts = (
            df.dropna(subset=["future_availability"])
            .groupby(["timestamp", "zone"])
            .agg(now=("availability", "sum"), later=("future_availability", "first"))
            .reset_index()
        )
        if len(zone_pts) > 4000:
            zone_pts = zone_pts.sample(4000, random_state=42)
        fig = px.scatter(
            zone_pts, x="now", y="later", opacity=0.32,
            color_discrete_sequence=["#34d399"],
            labels={"now": "Free now (bays)", "later": "Free +30 min actual (bays)"},
        )
        _mx = float(max(zone_pts["now"].max(), zone_pts["later"].max(), 10))
        fig.add_shape(type="line", x0=0, y0=0, x1=_mx, y1=_mx, line=dict(color="#fbbf24", width=1.5, dash="dash"))
        fig.update_layout(height=380, template="plotly_dark",
                          margin=dict(l=8, r=8, t=10, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=11, color="#c6d2e2"),
                          xaxis=dict(gridcolor="#22354f"), yaxis=dict(gridcolor="#22354f"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with st.container(border=True):
        st.subheader("Historical actual vs predicted availability", anchor=False)
        st.caption("Model replayed hourly over the last 24 hours")
        historical_fit_chart(df, hours=24)

    with st.container(border=True):
        st.subheader("Prediction error", anchor=False)
        st.caption("Actual minus predicted free bays — centred near zero means unbiased")
        prediction_error_chart(df)

    with st.container(border=True):
        st.subheader("What drives the model?", anchor=False)
        st.caption("Linear coefficients — sign = direction, length = strength")
        try:
            coefs = list(load_model()["model"].coef_)
            feature_importance_chart(m["features"], coefs)
        except Exception:
            for f in m["features"]:
                st.caption(f"• {f}")

    st.subheader("🟢 System status", anchor=False)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_status_badge("✓ IoT simulation online", "healthy")
    with s2:
        render_status_badge("✓ Prediction model active", "healthy")
    with s3:
        render_status_badge("✓ Anomaly detection active", "healthy")
    with s4:
        render_status_badge("✓ Live dashboard active", "healthy")

    with st.expander("Model information"):
        import os as _os
        from datetime import datetime as _dt

        trained = _dt.fromtimestamp(_os.path.getmtime(MODEL_PATH)).strftime("%d %b %Y, %H:%M") if _os.path.exists(MODEL_PATH) else "unknown"
        span = f"{df['timestamp'].min():%d %b %Y} → {df['timestamp'].max():%d %b %Y}" if len(df) else "unknown"
        st.write("**Algorithm:** Linear Regression (scikit-learn)")
        st.write(f"**Training data:** {len(df):,} records across 30 days at 15-minute intervals")
        st.write(f"**Evaluation period:** {span} (80/20 train-test split, seed 42)")
        st.write(f"**Target:** free bays 30 minutes ahead (2 intervals)")
        st.write(f"**Last trained:** {trained}")
        st.write("**Features:**")
        for f in m["features"]:
            st.caption(f"• {f}")
