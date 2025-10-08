# backend/main.py
"""
FastAPI backend for ShopSight.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from typing import List, Optional, Literal

import pandas as pd
from pydantic import BaseModel, Field, validator

from core.duck import run_sql, pretty_sql
from core.queries import sales_history

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

import os
import json
from openai import OpenAI
from textwrap import dedent
import re

from dotenv import load_dotenv 
load_dotenv() 

app = FastAPI(title="ShopSight API", version="0.1.0")

# Allow CORS for all origins (for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str

class SalesPoint(BaseModel):
    period_start: date
    units: int = Field(0, ge=0)
    revenue: float = Field(0.0, ge=0)

class SalesResponse(BaseModel):
    product_id: str
    grain: str #eg: 'day', 'week', 'month'
    display_sql: str
    rows: List[SalesPoint]

class ProductInfo(BaseModel):
    product_id: str
    product_name: str
    brand: Optional[str] = ""
    product_type: Optional[str] = ""

class TSPoint(BaseModel):
    """
    A single time series data point with a date string (ISO) (ds) and a value (y).
    """
    ds: date
    y: float

class ForecastRequest(BaseModel):
    series: List[TSPoint] = Field(..., description="Historic series: list of {ds, y}")
    horizon: int = Field(8, ge=1, le=52, description="Number of future periods to forecast")
    level: List[int] = Field(default_factory=lambda: [80, 95], description="Prediction interval levels")
    grain: Literal["day", "week", "month"] = "week"

    @validator("series")
    def non_empty_series(cls, v):
        if not v:
            raise ValueError("series must be a non-empty list")
        return v

class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    lo80: Optional[float] = None
    hi80: Optional[float] = None
    lo95: Optional[float] = None
    hi95: Optional[float] = None

class ForecastResponse(BaseModel):
    horizon: int
    level: List[int]
    points: List[ForecastPoint]

class InsightsRequest(BaseModel):
    product_id: str
    grain: Literal["day", "week", "month"] = "week"
    metric: Literal["units", "revenue"] = "units"
    horizon: int = 8
    model: Optional[str] = None
    demo_mode: bool = False # if true, or no API key -> mock summary

class InsightsResponse(BaseModel):
    model_used: str
    insight_markdown: str
    prompt_used: str

class SearchResponseRow(BaseModel):
    product_id: str
    product_name: str
    brand: Optional[str] = ""
    product_type: Optional[str] = ""

class SearchResponse(BaseModel):
    mode: Literal["nlsql", "structured"]
    display_sql: str
    rows: list[SearchResponseRow]

def _season_length(grain: str) -> int:
    g = (grain or "week").lower()
    if g == "day":
        return 7     
    if g == "month":
        return 12 
    return 52 # default to weekly

def _summarize_series_for_prompt(hist_df: pd.DataFrame, fc_df: pd.DataFrame, metric: str) -> dict:
    """Return small dict of stats + compact tables to keep the prompt lean."""
    hist_tail = (
        hist_df.sort_values("period_start")
               .tail(12)[["period_start", metric]]
               .rename(columns={"period_start": "ds", metric: "y"})
    )
    fc_tail = (
        fc_df[["ds", "yhat", "lo80", "hi80", "lo95", "hi95"]]
        .head(8)
    )
    summary = {
        "history_tail": hist_tail.assign(ds=hist_tail["ds"].astype(str)).to_dict(orient="records"),
        "forecast_head": fc_tail.assign(ds=fc_tail["ds"].astype(str)).to_dict(orient="records"),
        "history_n": int(len(hist_df)),
        "metric": metric,
        "recent_mean": float(hist_df[metric].tail(12).mean()) if len(hist_df) else 0.0,
        "recent_std": float(hist_df[metric].tail(12).std(ddof=0)) if len(hist_df) else 0.0,
        "overall_mean": float(hist_df[metric].mean()) if len(hist_df) else 0.0,
    }
    return summary

def _build_insights_prompt(product_meta: dict | None, grain: str, metric: str, stats: dict) -> str:
    name = (product_meta or {}).get("product_name") or ""
    brand = (product_meta or {}).get("brand") or ""
    ptype = (product_meta or {}).get("product_type") or ""
    header = f"Product: {name}".strip() or "Selected product"
    brand_line = f"Brand: {brand}" if brand else ""
    type_line = f"Type: {ptype}" if ptype else ""

    return dedent(f"""
    You are an e-commerce analytics assistant. Write a concise, executive-friendly insight
    about the time series below.

    {header}
    {brand_line}
    {type_line}

    Grain: {grain}
    Metric: {metric}

    DATA (compact JSON):
    {json.dumps(stats, ensure_ascii=False, indent=2)}

    INSTRUCTIONS:
    - First, 1 short paragraph (2–4 sentences) describing recent trend, volatility, and any seasonality.
    - Then, exactly 3 bullet points:
      • one on the forecast level & direction (mention the horizon),
      • one on risk/uncertainty using the prediction intervals,
      • one recommended action (clear and non-technical).
    - Avoid repeating raw numbers excessively; pick only the most salient.
    - Audience is business users; keep it plain and crisp.
    """).strip()

def _generate_schema() -> str:
    return dedent("""
    You are generating DuckDB SQL for a read-only e-commerce dataset.

    Tables:
      products(product_id VARCHAR, product_name TEXT, product_type TEXT, brand TEXT)
      transactions(product_id VARCHAR, t_dat DATE, price DOUBLE)

    Requirements:
    - Output a single SELECT statement only.
    - Return columns: product_id, product_name, brand, product_type.
    - Prefer ranking by recent popularity when it helps (e.g., last 90 days transactions).
    - Put a LIMIT :limit at the end (use named parameters :q and :limit).
    - Never use DDL/DML (no CREATE/ALTER/INSERT/DELETE/UPDATE/ATTACH/PRAGMA/etc.).
    - Keep it simple and robust.

    Examples:

    -- Query: "Nike running shoes"
    SELECT p.product_id, p.product_name, p.brand, p.product_type
    FROM products p
    WHERE (p.product_name ILIKE :q OR p.brand ILIKE :q OR p.product_type ILIKE :q)
    ORDER BY p.product_name
    LIMIT :limit;

    -- Query: "top adidas sneakers recently"
    WITH recent AS (
      SELECT product_id, COUNT(*) AS n
      FROM transactions
      WHERE t_dat >= CURRENT_DATE - INTERVAL 90 DAY
      GROUP BY 1
    )
    SELECT p.product_id, p.product_name, p.brand, p.product_type
    FROM products p
    LEFT JOIN recent r USING(product_id)
    WHERE (p.brand ILIKE '%adidas%' OR p.product_name ILIKE '%adidas%')
      AND (p.product_type ILIKE '%sneaker%' OR p.product_name ILIKE '%sneaker%')
    ORDER BY COALESCE(r.n, 0) DESC, p.product_name
    LIMIT :limit;
    """).strip()

_BLOCKLIST = re.compile(
    r"\b(create|alter|drop|insert|update|delete|truncate|merge|replace|grant|revoke|vacuum|pragma|load|install|set|attach|copy)\b",
    re.IGNORECASE,
)

def _is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select") and not s.lower().startswith("with"):
        return False
    if _BLOCKLIST.search(s):
        return False
    # No multiple statements
    if ";" in sql.strip():
        return False
    # Restrict table names
    if re.search(r"\binformation_schema\b|\bpg_\w+\b", s, flags=re.IGNORECASE):
        return False
    return True

def _structured_search_sql() -> tuple[str, dict]:
    sql = dedent("""
        SELECT
          product_id, product_name, brand, product_type
        FROM products
        WHERE product_name ILIKE :q
           OR brand        ILIKE :q
           OR product_type ILIKE :q
        ORDER BY product_name
        LIMIT :limit;
    """).strip()
    params = {}  # filled as per request
    return sql, params

_PARAM_RE = re.compile(r":(q|limit)\b", flags=re.IGNORECASE)

def _to_positional(sql: str, q_like: str, limit: int) -> tuple[str, list]:
    """
    Replace :q / :limit with positional ? placeholders, building a params list in the exact textual order of appearance.
    """
    params: list = []
    out = []
    i = 0
    for m in _PARAM_RE.finditer(sql):
        out.append(sql[i:m.start()])
        out.append("?")
        name = m.group(1).lower()
        params.append(q_like if name == "q" else limit)
        i = m.end()
    out.append(sql[i:])
    return ("".join(out), params)

@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """
    Health endpoint for the UI to ping.
    """
    return HealthResponse(status="ok")

@app.get("/sales", response_model=SalesResponse)
def get_sales(
    product_id: str,
    grain: str = "week",
    start: Optional[date] = None,
    end: Optional[date] = None,
):
    sql, params = sales_history(product_id=product_id, grain=grain)
    df = run_sql(sql, params) # store the result in a DataFrame

    display_sql = pretty_sql(sql, params)

    if not df.empty:
        df["period_start"] = pd.to_datetime(df["period_start"]).dt.date
        if start:
            df = df[df["period_start"] >= start]
        if end:
            df = df[df["period_start"] <= end]

    
    rows: List[SalesPoint] = []
    for _, r in df.iterrows():
        rows.append(
            SalesPoint(
                period_start=r["period_start"],
                units=int(r.get("units", 0) or 0),
                revenue=float(r.get("revenue", 0.0) or 0.0),
            )
        )

    return SalesResponse(
        product_id=str(product_id),
        grain=grain,
        display_sql=display_sql,
        rows=rows,
    )

@app.get("/product", response_model=Optional[ProductInfo])
def get_product(product_id: str):
    """
    Fetch a single product's basic metadata for display.
    """
    df = run_sql(
        """
        SELECT product_id, product_name, brand, product_type
        FROM products
        WHERE product_id = ?
        LIMIT 1;
        """,
        (product_id,),
    )
    if df.empty:
        return None
    r = df.iloc[0]
    return ProductInfo(
        product_id=str(r["product_id"]),
        product_name=str(r["product_name"]),
        brand=str(r.get("brand", "") or ""),
        product_type=str(r.get("product_type", "") or ""),
    )

@app.post("/forecast", response_model=ForecastResponse)
def post_forecast(req: ForecastRequest):
    """
    Generate a forecast for the given: univariate time series with SeasonalNaive model.
    """
    df = pd.DataFrame([{"ds": p.ds, "y": p.y} for p in req.series])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")
    df["unique_id"] = "series-1"

    m = _season_length(req.grain)
    models = [SeasonalNaive(season_length=m)]

    # If we have too little data, we fall back to Naive model
    if len(df) < m:
        models = [Naive()]
    
    frequency_map = {"day": "D", "week": "W", "month": "MS"}
    sf = StatsForecast(models=models, freq=frequency_map[req.grain])

    fcst = sf.forecast(df=df, h=req.horizon, level=req.level).reset_index(drop=True)

    model_name = models[0].__class__.__name__                
    mean_col = model_name  

    points: List[ForecastPoint] = []
    for _, row in fcst.iterrows():
        obj = ForecastPoint(
            ds=pd.to_datetime(row["ds"]).date().isoformat(),
            yhat=float(row[mean_col]),
        )
        
        for lvl in req.level:
            lo_col = f"{model_name}-lo-{lvl}"
            hi_col = f"{model_name}-hi-{lvl}"
            if lo_col in fcst.columns:
                setattr(obj, f"lo{lvl}", float(row[lo_col]))
            if hi_col in fcst.columns:
                setattr(obj, f"hi{lvl}", float(row[hi_col]))
        points.append(obj)

    return ForecastResponse(horizon=req.horizon, level=req.level, points=points)

@app.post("/insights", response_model=InsightsResponse)
def post_insights(req: InsightsRequest):
    """
    Generate a short narrative about history + forecast using OpenAI.
    """
    # 1) History (reuse sales query)
    sql, params = sales_history(product_id=req.product_id, grain=req.grain)
    hist = run_sql(sql, params)
    if hist.empty:
        return InsightsResponse(
            model_used="mock",
            insight_markdown="No history found for this selection.",
            prompt_used="(no prompt – empty history)",
        )
    hist["period_start"] = pd.to_datetime(hist["period_start"])
    hist = hist.sort_values("period_start")

    # 2) Forecast (reuse in-process call)
    series = [{"ds": d.strftime("%Y-%m-%d"), "y": float(v)}
              for d, v in zip(hist["period_start"], hist[req.metric])]
    # Build a tiny request DataFrame then reuse our forecast logic
    fc_req = ForecastRequest(
        series=[TSPoint(ds=pd.to_datetime(s["ds"]).date(), y=s["y"]) for s in series],
        horizon=req.horizon,
        level=[80, 95],
        grain=req.grain,
    )
    fc_resp: ForecastResponse = post_forecast(fc_req)  # call our own endpoint function
    fc = pd.DataFrame([p.dict() for p in fc_resp.points])
    fc["ds"] = pd.to_datetime(fc["ds"])

    # 3) Product metadata (nice touch for context)
    meta_df = run_sql(
        "SELECT product_id, product_name, brand, product_type FROM products WHERE product_id = ? LIMIT 1;",
        (req.product_id,),
    )
    meta = None if meta_df.empty else meta_df.iloc[0].to_dict()

    # 4) Build prompt materials
    stats = _summarize_series_for_prompt(hist_df=hist, fc_df=fc, metric=req.metric)
    prompt = _build_insights_prompt(product_meta=meta, grain=req.grain, metric=req.metric, stats=stats)

    # 5) Decide model / fallback
    model = req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")

    use_mock = req.demo_mode or not api_key or OpenAI is None
    reason = []
    if req.demo_mode: reason.append("demo_mode")
    if not api_key:   reason.append("no_api_key")
    if OpenAI is None: reason.append("openai_sdk_missing")
    print("[/insights] mode:", "mock" if use_mock else "live", "| reasons:", ", ".join(reason) or "—")

    if use_mock:
        # Deterministic, no-network summary
        recent = stats["recent_mean"]
        direction = "stable" if stats["recent_std"] < 0.1 * max(recent, 1.0) else "variable"
        mock = dedent(f"""
        Recent performance is **{direction}** with average {req.metric} ≈ {recent:.1f} per {req.grain}.
        The short-term forecast remains within historical ranges.

        - **Outlook:** Next {req.horizon} {req.grain}s are expected to track recent levels with modest variation.
        - **Uncertainty:** Prediction intervals widen slightly; watch for weekly swings around promotions.
        - **Action:** Align inventory to the forecast range and test a small price/promo to tighten variability.
        """).strip()
        return InsightsResponse(model_used="mock", insight_markdown=mock, prompt_used=prompt)

    # 6) Live call to OpenAI
    try:
        client = OpenAI(api_key=api_key)
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a crisp analytics writer for business users."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = chat.choices[0].message.content.strip()
        return InsightsResponse(model_used=model, insight_markdown=text, prompt_used=prompt)
    except Exception as e:
        # Safe fallback if the model call fails
        fallback = f"_LLM call failed ({e}). Showing mock summary instead._\n\n"
        recent = stats["recent_mean"]
        mock = dedent(f"""
        Recent performance is **steady** with average {req.metric} ≈ {recent:.1f} per {req.grain}.
        - **Outlook:** Next {req.horizon} {req.grain}s are expected to follow recent levels.
        - **Uncertainty:** Prediction intervals indicate moderate variability.
        - **Action:** Keep inventory aligned to the median forecast and monitor anomalies.
        """).strip()
        return InsightsResponse(model_used="mock", insight_markdown=fallback + mock, prompt_used=prompt)
    
@app.get("/search", response_model=SearchResponse)
def search_products_endpoint(
    q: str,
    limit: int = 10,
    nl: bool = True,           # when False → force structured fallback
    demo_mode: bool = False,
    model: Optional[str] = None,
):
    """
    NL→SQL product search with guardrails. Falls back to structured ILIKE.
    """
    limit = max(1, min(limit, 50))
    q_like = f"%{q}%"

    # If NL disabled, go straight to fallback
    if not nl:
        sql, params = _structured_search_sql()
        params = {"q": q_like, "limit": limit}
        df = run_sql(sql, params)
        rows = [SearchResponseRow(**{k: str(v) if v is not None else "" for k, v in r.items()})
                for r in df.to_dict("records")]
        return SearchResponse(mode="structured", display_sql=pretty_sql(sql, params), rows=rows)

    # NL→SQL path (OpenAI with fallback)
    api_key = os.getenv("OPENAI_API_KEY", "")
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    use_mock = demo_mode or not api_key or OpenAI is None

    # 1) Make a candidate SQL (LLM or heuristic mock)
    if use_mock:
        # Heuristic: same as fallback, but we still call it "nlsql" to show the flow
        draft_sql = dedent("""
            SELECT product_id, product_name, brand, product_type
            FROM products
            WHERE (product_name ILIKE :q OR brand ILIKE :q OR product_type ILIKE :q)
            ORDER BY product_name
            LIMIT :limit
        """).strip()
    else:
        sys = "You convert short product search queries into safe DuckDB SQL."
        user = dedent(f"""
            Query: {q}

            { _generate_schema() }

            Output only the SQL, no extra text:
        """).strip()

        try:
            client = OpenAI(api_key=api_key)
            chat = client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                temperature=0.1,
            )
            draft_sql = chat.choices[0].message.content.strip().strip("`")
            # If the model returned fenced code, strip it
            if draft_sql.lower().startswith("sql"):
                draft_sql = re.sub(r"^sql\s*", "", draft_sql, flags=re.IGNORECASE).strip()
        except Exception as e:
            # LLM failed → use heuristic mock
            draft_sql = dedent("""
                SELECT product_id, product_name, brand, product_type
                FROM products
                WHERE (product_name ILIKE :q OR brand ILIKE :q OR product_type ILIKE :q)
                ORDER BY product_name
                LIMIT :limit
            """).strip()

    # 2) Validate & normalize the SQL
    candidate = draft_sql.rstrip(";")
    if not _is_safe_select(candidate):
        # Guardrail tripped → fallback to structured
        sql, params = _structured_search_sql()
        params = {"q": q_like, "limit": limit}
        mode = "structured"
    else:
        # Ensure it returns the required columns & has a limit param; if missing, append
        if "limit" not in candidate.lower():
            candidate += " LIMIT :limit"
        sql = candidate
        params = {"q": q_like, "limit": limit}
        mode = "nlsql"

    # 3) Execute
    try:
        sql_exec, param_list = _to_positional(sql, q_like, limit)
        df = run_sql(sql_exec, param_list)
    except Exception:
        # Execution failed → fallback to structured
        fb_sql, _ = _structured_search_sql() 
        sql_exec, param_list = _to_positional(fb_sql, q_like, limit)
        df = run_sql(sql_exec, param_list)
        mode = "structured"
        display_sql = pretty_sql(sql_exec, param_list)

    # 4) Serialize
    rows = [SearchResponseRow(**{k: str(v) if v is not None else "" for k, v in r.items()})
            for r in df.to_dict("records")]
    return SearchResponse(mode=mode, display_sql=pretty_sql(sql, params), rows=rows)
