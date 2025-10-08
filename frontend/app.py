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
    st.selectbox("Product", options=["(pick from Search later)"], index=0, key="sales_product")
    st.warning("We’ll plot sales here after we build the /sales endpoint.", icon="🧩")

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
