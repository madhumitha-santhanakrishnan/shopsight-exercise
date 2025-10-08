# frontend/app.py
"""
ShopSight – Agentic E-commerce Analytics Prototype UI (Streamlit)
"""

import os
import time
import requests
import streamlit as st


st.set_page_config(
    page_title="ShopSight",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load backend URL from env 
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Sidebar for settings and info

st.sidebar.title("ShopSight Settings")
st.sidebar.caption("Tweak runtime options for the demo.")

demo_mode = st.sidebar.toggle("Demo mode", value=True, help="Use cached/mocked responses when possible.")
model_name = st.sidebar.text_input("LLM model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
backend_url = st.sidebar.text_input("Backend URL", BACKEND_URL)

st.sidebar.divider()
st.sidebar.write("**About:** ShopSight is a search-driven analytics prototype for e-commerce.")
st.sidebar.write("Ask for sales, forecasts, and insights using natural language.")


def ping_health(url: str) -> bool:
    """
    Ping the FastAPI /health endpoint so users see connectivity status.
    """
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🛍️ ShopSight")
    st.caption("Ask questions like a colleague. Get charts, forecasts, and insights.")
with col2:
    ok = ping_health(backend_url)
    st.metric(
        label="API Status",
        value="Online" if ok else "Offline",
        help="FastAPI /health endpoint check"
    )

tab_search, tab_sales, tab_forecast, tab_insights, tab_assistant = st.tabs(
    ["Search", "Sales", "Forecast", "Insights", "Assistant"]
)

with tab_search:
    st.subheader("Search for products")
    st.caption("Try: *“Nike running shoes”*, *“Adidas sneakers last quarter”*")
    query = st.text_input("What are you looking for?", value="", placeholder="e.g., Nike running shoes")
    colA, colB = st.columns([0.2, 0.8])
    with colA:
        st.button("Search", type="primary", key="btn_search")
    with colB:
        st.toggle("Trust mode: NL→SQL", value=True, key="trust_toggle",
                  help="If off, the app uses structured filters only.")
    st.info("Results will appear here once we wire the /search endpoint.", icon="ℹ️")

with tab_sales:
    st.subheader("Sales")
    st.caption("Historical sales chart for the selected product.")

    c1, c2, c3, c4 = st.columns([0.35, 0.2, 0.2, 0.25]) # columns for input rows
    with c1:
        product_id_input = st.text_input(
            "Product ID",
            value=st.session_state.get("sales_product", ""),
            placeholder="e.g., 508929006",
            help="Paste a product_id from Search.",
        )
    with c2:
        grain = st.selectbox("Grain", ["day", "week", "month"], index=1)
    with c3:
        metric = st.radio("Metric", ["units", "revenue"], horizontal=True)
    with c4:
        st.write("")  # spacing
        fetch_btn = st.button("Load sales", type="primary")

    c5, c6 = st.columns(2)
    with c5:
        start_date = st.date_input("Start (optional)", value=None)
    with c6:
        end_date = st.date_input("End (optional)", value=None)

    st.divider()

    if fetch_btn and product_id_input.strip():
        params = {
            "product_id": product_id_input.strip(),
            "grain": grain,
        }
        if start_date:
            params["start"] = start_date.isoformat()
        if end_date:
            params["end"] = end_date.isoformat()
        
        meta = None
        try:
            rmeta = requests.get(f"{backend_url}/product", params={"product_id": product_id_input.strip()}, timeout=10)
            if rmeta.ok and rmeta.json():
                meta = rmeta.json()
        except Exception:
            meta = None

        # Show a product card (if it exists)
        if meta:
            with st.container(border=True):
                cA, cB, cC = st.columns([0.6, 0.2, 0.2])
                with cA:
                    st.metric("Product Name", meta.get("product_name") or "—")
                    st.caption(f"ID: `{meta['product_id']}`")
                with cB:
                    st.metric("Brand", meta.get("brand") or "—")
                with cC:
                    st.metric("Type", meta.get("product_type") or "—")

        with st.spinner("Fetching sales…"):
            try:
                r = requests.get(f"{backend_url}/sales", params=params, timeout=20)
                if not r.ok:
                    st.error(f"API error {r.status_code}: {r.text}")
                else:
                    payload = r.json()
                    rows = payload.get("rows", [])
                    if not rows:
                        st.info("No sales found for this selection. Try another product_id or adjust dates.")
                    else:
                        import pandas as pd
                        import plotly.express as px

                        df = pd.DataFrame(rows)
                        df["period_start"] = pd.to_datetime(df["period_start"])

                        y = metric
                        fig = px.line(
                            df,
                            x="period_start",
                            y=y,
                            markers=True,
                            title=f"{metric.title()} over time • {product_id_input}",
                        )
                        fig.update_layout(xaxis_title="Period start", yaxis_title=metric.title())
                        st.plotly_chart(fig, use_container_width=True)

                        # Downloads
                        st.download_button(
                            "Download CSV",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name=f"sales_{product_id_input}_{grain}.csv",
                            mime="text/csv",
                        )

                        # Show the exact SQL that ran, for transparency
                        with st.expander("Show SQL"):
                            st.code(payload.get("display_sql", "--"), language="sql")

                        # Store selection so other tabs can use it
                        st.session_state["sales_product"] = product_id_input.strip()

            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.info("Enter a product_id and click **Load sales** to see the chart.", icon="ℹ️")


with tab_forecast:
    st.subheader("Forecast")
    st.caption("Demand forecast with confidence intervals.")
    st.slider("Horizon (weeks)", min_value=4, max_value=16, value=8, step=1, key="horizon")
    st.toggle("Show 80% / 95% bands", value=True, key="bands")
    st.warning("We’ll compute and plot forecasts once /forecast is ready.", icon="🧮")

with tab_insights:
    st.subheader("Insights")
    st.caption("AI-generated summary of trends and risks.")
    st.button("Generate insights", key="btn_insights")
    st.info("We’ll call /insights to produce a concise narrative here.", icon="💡")
    with st.expander("Prompt Inspector (transparency)"):
        st.code("<!-- prompt and variables will show here once wired -->", language="markdown")

with tab_assistant:
    st.subheader("Assistant (Agent)")
    st.caption("Ask end-to-end: ‘Show Nike running shoes last quarter and forecast next month.’")
    agent_q = st.text_area("Your request", value="", height=80, placeholder="Type a natural-language request…")
    st.button("Run", key="btn_agent_run")
    st.info("The /agent/run endpoint will plan tool calls and return results here.", icon="🤖")

st.divider()
st.caption("© ShopSight – Prototype for take-home assessment.")
