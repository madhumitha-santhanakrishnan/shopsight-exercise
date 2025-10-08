# core/duck.py
import os
import duckdb
import pandas as pd
from typing import Iterable, Mapping, Any

S3_BASE = os.getenv("SHOPSIGHT_S3_BASE_URL", "s3://kumo-public-datasets/hm_with_images/")
_CONN: duckdb.DuckDBPyConnection | None = None

def get_conn() -> duckdb.DuckDBPyConnection:
    global _CONN
    if _CONN is not None:
        return _CONN

    # In-memory DuckDB instance
    con = duckdb.connect(database=":memory:")

    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    con.execute("SET s3_region='us-west-2';")            
    con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';") 
    con.execute("SET s3_use_ssl=true;")
    con.execute("SET s3_url_style='path';")
    # con.execute("SET s3_allow_unsigned=true;") 
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    con.execute("SET s3_session_token='';") 

    # To read: get_conn().execute("SELECT * FROM read_parquet('<s3-path>') LIMIT 5").df()

    articles_path = f"{S3_BASE.rstrip('/')}/articles/*.parquet"
    txns_path     = f"{S3_BASE.rstrip('/')}/transactions/*.parquet"

    art_cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{articles_path}')").df()["column_name"].str.lower().tolist()
    txn_cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{txns_path}')").df()["column_name"].str.lower().tolist()

    def pick_first(cands, cols, default=None):
        for c in cands:
            if c in cols:
                return c
        return default

    # Products / Articles
    art_id_col   = pick_first(["article_id", "product_id", "sku", "id"], art_cols)
    name_col     = pick_first(["prod_name", "product_name", "name", "title"], art_cols)
    brand_col    = pick_first(["brand", "index_name", "brand_name"], art_cols)
    type_col     = pick_first(["product_type_name", "product_type", "product_group_name", "category"], art_cols)

    if not art_id_col:
        art_id_expr = "ROW_NUMBER() OVER ()"
    else:
        art_id_expr = f"CAST(src.{art_id_col} AS VARCHAR)"

    name_expr  = f"COALESCE(src.{name_col}, 'Unknown')" if name_col else "'Unknown'"
    brand_expr = f"COALESCE(src.{brand_col}, '')"       if brand_col else "''"
    type_expr  = f"COALESCE(src.{type_col}, '')"        if type_col else "''"

    con.execute(f"""
        CREATE OR REPLACE VIEW products AS
        SELECT
            {art_id_expr} AS product_id,
            {name_expr}  AS product_name,
            {brand_expr} AS brand,
            {type_expr}  AS product_type
        FROM read_parquet('{articles_path}') AS src;
    """)

    # Transactions
    txn_id_col  = pick_first(["article_id", "product_id", "sku", "id"], txn_cols)
    date_col    = pick_first(["t_dat", "date", "datetime", "order_date"], txn_cols)
    price_col   = pick_first(["price", "sales_price", "unit_price"], txn_cols)

    if not txn_id_col:
        txn_id_expr = "ROW_NUMBER() OVER ()"
    else:
        txn_id_expr = f"CAST(src.{txn_id_col} AS VARCHAR)"
    
    date_expr  = f"CAST(src.{date_col} AS DATE)" if date_col else "CAST(CURRENT_DATE AS DATE)"
    price_expr = f"CAST(COALESCE(src.{price_col}, 0) AS DOUBLE)" if price_col else "CAST(0 AS DOUBLE)"

    con.execute(f"""
        CREATE OR REPLACE VIEW transactions AS
        SELECT
            {txn_id_expr}  AS product_id,
            {date_expr}    AS t_dat,
            {price_expr}   AS price
        FROM read_parquet('{txns_path}') AS src;
    """)

    _CONN = con
    return con

def run_sql(sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> pd.DataFrame:
    con = get_conn()
    if isinstance(params, dict):
        res = con.execute(sql, parameters=params)
    else:
        res = con.execute(sql, params)
    return res.df()

def _sql_literal(value: Any) -> str:
    """
    Smalll helper to pretty-print bound parameters as SQL literals
    for the "Show SQL" read-only display.
    """
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    
    text = str(value).replace("'", "''")
    return f"'{text}'"

def pretty_sql(sql: str, params: Iterable[Any] | Mapping[str, Any]) -> str:
    """
    Render a human-readable SQL string with parameters embedded. Only for display.
    """
    if isinstance(params, dict):
        pretty = sql
        for k, v in params.items():
            pretty = pretty.replace(f":{k}", _sql_literal(v))
        return pretty

    pretty = sql
    for v in params:
        pretty = pretty.replace("?", _sql_literal(v), 1)
    return pretty
