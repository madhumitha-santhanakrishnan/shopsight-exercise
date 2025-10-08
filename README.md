# ShopSight – An E-commerce Analytics Prototype

ShopSight is a small end-to-end prototype I built to explore how an LLM-powered analytics assistant could help business users make sense of sales and demand patterns.
It connects real data (DuckDB over public S3 parquet files) with a simple FastAPI + Streamlit stack and a light forecasting/LLM layer on top.

The goal wasn’t to build a huge product that does everything but to show a clean and working loop: Search → Sales history → Forecast → Insights (LLM)

Everything is intentionally kept simple, transparent, and fast to run locally.

## What works today
- Data + Backend
  - DuckDB + httpfs connector to read directly from S3
  - SQL templates in `core/queries.py` for searching products & aggregating sales by day/week/month.
  - FastAPI backend with endpoints: /health, /sales, /forecast, /insights, /search
- Frontend (Streamlit)
  - 4 tabs: Search, Sales, Forecast, Insights (planned 5, but removed agents due to the lack of time)
  - Each tab is self contained & easy to follow. 
  - Has 'Show SQL' expands everywhere for transparency reg what's going on in the backend, and confirms the actual data usage from the S3 url. 
  - Product ID automatically gets carried over across tabs. 
  - Forecast view overlays history + forecast, and also shows 80/95% confidence bands. 
  - Added a simple 'What-if simulator' for quick scenarios. 

## How to run locally

```bash
# 1) Create environment
python -m venv .venv 
pip install -r requirements.txt
```

```bash
# 2) Add OpenAI key (optional)
cp .env.example .env
# Then edit .env and add your OPENAI_API_KEY
```

```bash
# 3) Start backend
uvicorn backend.main:app --reload --port 8000
```

```bash
# 4) Start frontend (in new shell)
streamlit run frontend/app.py
```

Note: If you don’t set an OpenAI key, the app runs in demo/mock mode: all LLM features return deterministic mock summaries.

## Design Choices
- DuckDB: Simple & fast. Most importantly, perfect for querying parquet over HTTP.
- FastAPI: Backend for RESTful endpoints. 
- Streamlit: Fastest way to protype a full UI without focusing too much on the nitty gritty of HTML/CSS/JS. 
- StatsForecast (Naive + SeasonalNaive): Works well with sparse data (like ours)
- OpenAI/Mock: For LLM summaries. And added mock/demo mode as a safe fallback. 

## Scope Decisions
- I focused on a clean flow that works end-to-end
- Avoided adding mutliple LLM endpoints until the basics were polished. 
- Some features (like the assistant/agents tab) were cut off intentionally due to dataset limits & time constraints -> "last month/this month" type of queries wouldn't make sense on a static dataset. 
- I chose a simple, classical forecasting model (SeasonalNaive) over a more complex one (like Auto ARIMA, Prophet etc). This was a deliberate choice, as in the real world, I feel one should establish a robust & understandable baseline first, then explore others to avoid the "black-box" issues in more complex models. 

## Assumptions
- The dataset on S3 is static and covers a fixed historical window so “this month” or “next month” queries are illustrative, not live.
- Each record represents a single product’s sales (units and revenue are aggregated).
- LLM and forecasting outputs are meant for demo-level interpretability, not production forecasting accuracy.

## What’s real vs mocked
- **Real:** DuckDB queries over S3 parquet (via httpfs), FastAPI endpoints, SQL templates, StatsForecast, real product search and aggregation.
- **Mocked:** LLM summaries (when DEMO_MODE=True or no API key). What-if simulator uses heuristic text rules.

## Time spent
Roughly 3.5–4 hours across setup, data connection, testing connection, sales + forecast flow, LLM integration, and polish. I completed this exercise over a few focused work sessions.

## What I would build next
- Explain forecast: Button on the forecast tab that calls `/llm/explain_forecast` to describe the chart in plain english ("For example, sales have stabilized after winter peaks...")
- Ask the Chart: Conversational interface to query patterns ("When was the biggest dip?")
- Agent Orchestrator: An agent planner to run Search → Sales history → Forecast → Insights autonomously. This is the ultimate goal: An orchestrator that can take a high-level prompt like "Find my top 3 riskiest products for next quarter," and autonomously run the chain for multiple products and synthesize a final report for the user. 
  - Also, an agent that connects product trends directly to specific customer segments. This would help transform the dashboard from viz/reporting tool to revenue-generation engine. 

## Notes / Tradeoffs
- Kept forecast models basic as they are fast to compute and explainable. 
- Added deterministic mocked responses to make the demo stable even without API keys. 
- Kept UI design minimal but readable as I prioritized clarity/ease of use over polish for the prototype. 
