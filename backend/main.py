# backend/main.py
"""
FastAPI backend for ShopSight.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from core.duck import run_sql, pretty_sql
from core.queries import sales_history

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