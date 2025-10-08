from core.duck import run_sql, pretty_sql, get_conn
from core import queries as Q

get_conn()

# 1) Try product search
sql, params = Q.search_products("Nike", 5)
print(pretty_sql(sql, params))
print(run_sql(sql, params).head())

# 2.1) Try top products
cands = run_sql("""
  SELECT t.product_id,
         COUNT(*) AS n,
         MIN(t_dat) AS first_date,
         MAX(t_dat) AS last_date
  FROM transactions t
  JOIN products p USING (product_id)
  GROUP BY 1
  ORDER BY n DESC
  LIMIT 10;
""")
print(cands)

# 2.2) Try sales history for a known product_id from above
pid = "372860001"  
sql, params = Q.sales_history(pid, "week")
print(pretty_sql(sql, params))
print(run_sql(sql, params).head())


