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

def _season_length(grain: str) -> int:
    g = (grain or "week").lower()
    if g == "day":
        return 7     
    if g == "month":
        return 12 
    return 52 # default to weekly

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