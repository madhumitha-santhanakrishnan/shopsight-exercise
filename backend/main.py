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

@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """
    Health endpoint for the UI to ping.
    """
    return HealthResponse(status="ok")

