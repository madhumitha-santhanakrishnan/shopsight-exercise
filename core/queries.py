# core/queries.py
from typing import Tuple, Iterable

_VALID_GRAINS = {
    "day": "day",
    "week": "week",
    "month": "month",
}


def search_products(name_query: str, limit: int = 20) -> Tuple[str, Iterable]:
    """
    Build a product search query.
    """
    # Build the LIKE pattern safely on the Python side
    pattern = f"%{name_query.strip()}%" if name_query else "%"

    sql = """
        SELECT
            product_id,
            product_name,
            brand,
            product_type
        FROM products
        WHERE product_name ILIKE ?
           OR brand        ILIKE ?
           OR product_type ILIKE ?
        ORDER BY product_name ASC
        LIMIT ?;
    """

    params = (pattern, pattern, pattern, int(limit))
    return sql, params


def sales_history(product_id: str, grain: str = "week") -> Tuple[str, Iterable]:
    """
    Build a sales history query.
    """
    g = _VALID_GRAINS.get(grain.lower(), "week")

    sql = f"""
        SELECT
            date_trunc('{g}', t_dat)::DATE AS period_start,
            COUNT(*)                       AS units,
            SUM(price)                     AS revenue
        FROM transactions
        WHERE product_id = ?
        GROUP BY 1
        ORDER BY 1;
    """

    params = (str(product_id),)
    return sql, params
