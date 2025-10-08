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

demo_mode = st.sidebar.toggle("Demo mode", value=True, help="Use mocked responses when possible.")
model_name = st.sidebar.text_input("LLM model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
backend_url = st.sidebar.text_input("Backend URL", BACKEND_URL)

def _select_pid(pid: str):
    st.session_state["sales_product"] = pid
    st.session_state["fc_pid"] = pid            
    st.session_state["ins_pid"] = pid
    st.session_state["just_selected_pid"] = pid

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

tab_search, tab_sales, tab_forecast, tab_insights= st.tabs(
    ["Search", "Sales", "Forecast", "Insights"]
)

with tab_search:
    st.subheader("Search for products")
    st.caption("Try: *“Nike running shoes”*, *“Adidas sneakers”*")

    colq, coll, colm = st.columns([0.6, 0.15, 0.25])
    with colq:
        query = st.text_input("What are you looking for?", value="", placeholder="e.g., Nike running shoes")
    with coll:
        limit = st.number_input("Limit", min_value=1, max_value=50, value=10, step=1)
    with colm:
        trust_nl = st.toggle(
            "Trust mode: NL→SQL",
            value=True,
            help="If off, uses a structured ILIKE search."
        )

    run_search = st.button("Search", type="primary", key="btn_search")

    st.divider()

    if run_search and query.strip():
        try:
            params = {
                "q": query.strip(),
                "limit": int(limit),
                "nl": bool(trust_nl),
                "demo_mode": demo_mode,
                "model": model_name,
            }
            with st.spinner("Searching…"):
                r = requests.get(f"{backend_url}/search", params=params, timeout=30)
            if not r.ok:
                st.error(f"/search error {r.status_code}: {r.text}")
            else:
                payload = r.json()
                rows = payload.get("rows", [])
                mode = payload.get("mode", "structured")

                if not rows:
                    st.info("No matches found. Try fewer words or different terms.")
                else:
                    st.caption(f"Mode: **{mode}**")
                    # st.markdown(f"**{len(rows)} products found.**")
                    for i, row in enumerate(rows):
                        with st.container(border=True):
                            cA, cB, cC, cD = st.columns([0.45, 0.2, 0.2, 0.15])
                            with cA:
                                st.markdown(f"**{row.get('product_name','')}**")
                                st.caption(f"ID: `{row.get('product_id','')}`")
                            with cB:
                                st.markdown("**Brand**")
                                st.caption(row.get("brand","") or "—")
                            with cC:
                                st.markdown("**Type**")
                                st.caption(row.get("product_type","") or "—")
                            with cD:
                                pid = row.get("product_id","")
                                st.button(
                                    "Use",
                                    key=f"use_{i}_{pid}",
                                    on_click=_select_pid,
                                    args=(pid,),
                                    disabled=(pid == "")
                                )

                    # Show the exact SQL used
                    with st.expander("Show SQL"):
                        st.code(payload.get("display_sql", "--"), language="sql")

        except Exception as e:
            st.error(f"Search failed: {e}")

    if "just_selected_pid" in st.session_state:
        sel = st.session_state.pop("just_selected_pid")
        st.info(f"Selected product_id `{sel}` for Sales/Forecast/Insights tabs.")
        st.toast(f"Selected product_id `{sel}` for Sales/Forecast/Insights tabs.", icon="✅")

    else:
        st.info("Enter a query and click **Search**.", icon="🔎")


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
    st.caption("Demand forecast with confidence intervals overlaid on history.")

    # Controls
    c1, c2, c3 = st.columns([0.35, 0.30, 0.35])
    with c1:
        product_id_input_fc = st.text_input(
            "Product ID",
            value=st.session_state.get("sales_product", ""),
            placeholder="e.g., 508929006",
            help="Pick a product_id that has sales (copy from Sales tab).",
            key="fc_pid",
        )
    with c2:
        grain_fc = st.selectbox("Grain", ["day", "week", "month"], index=1, key="fc_grain")
    with c3:
        metric_fc = st.radio("Metric", ["units", "revenue"], horizontal=True, key="fc_metric")

    c4, c5, c6 = st.columns([0.33, 0.33, 0.34])
    with c4:
        horizon = st.slider("Horizon (periods)", min_value=4, max_value=16, value=8, step=1, key="fc_h")
    with c5:
        show_bands = st.toggle("Show 80% / 95% bands", value=True, key="fc_bands")
    with c6:
        overlay_history = st.toggle("Overlay history", value=True, key="fc_overlay")

    run_btn = st.button("Compute forecast", type="primary", key="btn_fc")

    st.divider()

    import pandas as pd
    import plotly.graph_objects as go
    # Action
    if run_btn and product_id_input_fc.strip():
        try:
            # 1) Pull aggregated sales for history
            sales_params = {"product_id": product_id_input_fc.strip(), "grain": grain_fc}
            with st.spinner("Fetching sales…"):
                r1 = requests.get(f"{backend_url}/sales", params=sales_params, timeout=30)
            if not r1.ok:
                st.error(f"/sales error {r1.status_code}: {r1.text}")
                st.stop()

            payload = r1.json()
            rows = payload.get("rows", [])
            if not rows:
                st.info("No sales found. Try another product_id or different grain.")
                st.stop()


            hist = pd.DataFrame(rows)
            hist["period_start"] = pd.to_datetime(hist["period_start"])
            hist = hist.sort_values("period_start")

            if len(hist) < 6:
                st.info("Not enough history to forecast reliably. Try a different product or grain.")
                st.stop()
            
            if grain_fc == "day" and len(hist) > 300:
                st.warning("Daily data is dense — aggregating to weekly for clarity.", icon="⚠️")
                hist = (
                    hist.set_index("period_start")
                        .resample("W")
                        .agg({metric_fc: "sum"}) # Aggregate to weekly sums for readability
                        .reset_index()
                )
                grain_fc = "week"

            # show_markers = len(hist) < 200
            # mode_hist = "lines+markers" if show_markers else "lines"

            # 2) Prepare series for /forecast as [{ds, y}]
            series = [{"ds": d.strftime("%Y-%m-%d"), "y": float(v)}
                      for d, v in zip(hist["period_start"], hist[metric_fc])]

            # 3) Call /forecast
            req_body = {"series": series, "horizon": horizon, "level": [80, 95], "grain": grain_fc}
            with st.spinner("Computing forecast…"):
                r2 = requests.post(f"{backend_url}/forecast", json=req_body, timeout=60)
            if not r2.ok:
                st.error(f"/forecast error {r2.status_code}: {r2.text}")
                st.stop()

            fc = pd.DataFrame(r2.json().get("points", []))
            if fc.empty:
                st.info("Forecast returned no points.")
                st.stop()
            fc["ds"] = pd.to_datetime(fc["ds"])
            fc = fc.sort_values("ds")
            # mode_fc = "lines+markers" if len(fc) < 200 else "lines"

            # Plotly chart with history + forecast + confidence bands
            color_hist = "#2563EB"   # blue-600
            color_fc   = "#0EA5E9"   # sky-500
            color80    = "rgba(14,165,233,0.20)"  # 20% opacity
            color95    = "rgba(14,165,233,0.10)"  # 10% opacity

            fig = go.Figure()

            # Bands (95% and 80%)
            if show_bands and {"lo95","hi95"}.issubset(fc.columns):
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                    y=pd.concat([fc["hi95"], fc["lo95"][::-1]]),
                    fill="toself", fillcolor=color95, line=dict(width=0),
                    hoverinfo="skip", showlegend=True, name="95% PI", legendgroup="pi",
                ))
            if show_bands and {"lo80","hi80"}.issubset(fc.columns):
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                    y=pd.concat([fc["hi80"], fc["lo80"][::-1]]),
                    fill="toself", fillcolor=color80, line=dict(width=0),
                    hoverinfo="skip", showlegend=True, name="80% PI", legendgroup="pi",
                ))

            # History line
            if overlay_history:
                fig.add_trace(go.Scatter(
                    x=hist["period_start"], y=hist[metric_fc],
                    mode="lines+markers",
                    name=f"History ({metric_fc})",
                    line=dict(width=2, color=color_hist),
                    marker=dict(size=5, color=color_hist),
                    legendgroup="main",
                    hovertemplate="%{x|%b %d, %Y}<br>"+metric_fc.title()+": %{y:.0f}<extra></extra>",
                ))

            # Forecast mean
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat"],
                mode="lines+markers",
                name="Forecast",
                line=dict(width=2, color=color_fc, dash="dash"),
                marker=dict(size=5, color=color_fc),
                legendgroup="main",
                hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:.0f}<extra></extra>",
            ))

            # Vertical guide at forecast start (last history point)
            split_x = hist["period_start"].max()
            fig.add_vline(x=split_x, line_dash="dot", line_color="#9CA3AF", opacity=0.7)
            fig.add_annotation(x=split_x, y=1, yref="paper", yanchor="bottom",
                               text="Forecast starts", showarrow=False, xanchor="left",
                               font=dict(size=11, color="#6B7280"))

            fig.update_layout(
                title=f"Forecast • {product_id_input_fc}",
                xaxis_title="Period start",
                yaxis_title=metric_fc.title(),
                hovermode="x unified",
                yaxis=dict(rangemode="tozero"),
                legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(255,255,255,0)"),
                margin=dict(t=60, r=110, b=40, l=50),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Downloads + SQL
            hist_out = hist.rename(columns={"period_start":"ds", metric_fc:"y"})
            st.download_button(
                "Download history (CSV)",
                hist_out.to_csv(index=False).encode("utf-8"),
                file_name=f"history_{product_id_input_fc}_{grain_fc}_{metric_fc}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download forecast (CSV)",
                fc.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{product_id_input_fc}_{grain_fc}_{metric_fc}.csv",
                mime="text/csv",
            )
            with st.expander("Show SQL used for history"):
                st.code(payload.get("display_sql", "--"), language="sql")

        except Exception as e:
            st.error(f"Forecast flow failed: {e}")
    else:
        st.info("Enter a product_id and click **Compute forecast** to see the overlay.", icon="ℹ️")


with tab_insights:
    st.subheader("Insights")
    st.caption("AI-generated summary of recent trends, forecast, and next actions.")

    c1, c2, c3, c4 = st.columns([0.34, 0.22, 0.22, 0.22])
    with c1:
        pid_i = st.text_input("Product ID", value=st.session_state.get("sales_product", ""), placeholder="e.g., 508929006")
    with c2:
        grain_i = st.selectbox("Grain", ["day", "week", "month"], index=1, key="ins_grain")
    with c3:
        metric_i = st.selectbox("Metric", ["units", "revenue"], index=0, key="ins_metric")
    with c4:
        horizon_i = st.slider("Horizon", min_value=4, max_value=16, value=8, step=1, key="ins_h")

    run_i = st.button("Generate insights", type="primary", key="btn_insights")
    st.divider()

    if run_i and pid_i.strip():
        body = {
            "product_id": pid_i.strip(),
            "grain": grain_i,
            "metric": metric_i,
            "horizon": horizon_i,
            "model": model_name,
            "demo_mode": demo_mode,
        }
        with st.spinner("Thinking…"):
            try:
                r = requests.post(f"{backend_url}/insights", json=body, timeout=90)
                if not r.ok:
                    st.error(f"/insights error {r.status_code}: {r.text}")
                else:
                    payload = r.json()
                    st.markdown(payload.get("insight_markdown", "_No insight returned._"))
                    with st.expander("Inspect Prompt"):
                        st.code(payload.get("prompt_used", ""), language="markdown")
                    st.caption(f"Model: {payload.get('model_used', 'unknown')}")
            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.info("Pick a product and click **Generate insights**.", icon="💡")
    
    st.divider()

    with st.expander("What-if Simulator", expanded=False):
        st.caption("Simulate the impact of a pricing or promotion change on sales.")
        scenario = st.text_input(
            "Describe a scenario:",
            key="whatif_scenario",
            placeholder="e.g., Increase price by 10% or Run a 20% off promo next month",
        )
        simulate = st.button("Simulate", key="btn_whatif")

        if simulate:
            if not pid_i.strip():
                st.warning("Please select a Product ID first (from Search/Sales).", icon="⚠️")
            elif not scenario.strip():
                st.warning("Please describe a scenario.", icon="⚠️")
            else:
                body = {
                    "product_id": pid_i.strip(),
                    "scenario": scenario.strip(),
                    "metric": metric_i,
                    "demo_mode": demo_mode,
                    "model": model_name,
                }
                with st.spinner("Simulating scenario…"):
                    try:
                        r = requests.post(f"{backend_url}/llm/whatif", json=body, timeout=60)
                        if r.ok:
                            p = r.json()
                            st.markdown(p.get("answer_md", "_No answer returned._"))
                            with st.expander("Prompt used"):
                                st.code(p.get("prompt_used", ""), language="markdown")
                            st.caption(f"Model: {p.get('model_used', 'unknown')}")
                        else:
                            st.error(f"What-if error {r.status_code}: {r.text}")
                    except Exception as e:
                        st.error(f"What-if request failed: {e}")
        else:
            st.info("Describe a scenario and click **Simulate**.", icon="🔍")


st.divider()
st.caption("© ShopSight 2025")
