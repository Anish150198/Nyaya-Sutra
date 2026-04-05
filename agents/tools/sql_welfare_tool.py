"""
Agentic tool: translates user demographics into deterministic PySpark/SQL
queries against the schemes_curated Delta Table.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_scheme_filter_query(
    age: Optional[int] = None,
    gender: Optional[str] = None,
    state_code: Optional[str] = None,
    income: Optional[float] = None,
    caste: Optional[str] = None,
    disability: bool = False,
) -> str:
    """
    Build a SQL WHERE clause for filtering schemes_curated based on demographics.

    Returns
    -------
    str  SQL query string ready for Spark SQL execution.
    """
    conditions = []

    if age is not None:
        conditions.append(f"(age_min IS NULL OR age_min <= {age})")
        conditions.append(f"(age_max IS NULL OR age_max >= {age})")

    if gender:
        conditions.append(f"(gender IS NULL OR gender = 'all' OR gender = '{gender}')")

    if state_code:
        conditions.append(
            f"(state_codes IS NULL OR state_codes LIKE '%all%' OR state_codes LIKE '%{state_code}%')"
        )

    if income is not None:
        conditions.append(f"(income_max IS NULL OR income_max >= {income})")

    if caste:
        conditions.append(f"(caste_flags IS NULL OR caste_flags LIKE '%{caste}%')")

    if disability:
        conditions.append("(caste_flags LIKE '%disability%' OR caste_flags LIKE '%pwd%')")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
    SELECT scheme_id, name, category, description, eligibility_text,
           benefits_text, apply_url
    FROM schemes_curated
    WHERE {where_clause}
    ORDER BY name
    """
    return query.strip()


def run_scheme_query(spark, query: str) -> list[dict]:
    """
    Execute the scheme filter query on a Spark session.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session (from Databricks or local).
    query : str
        SQL query from build_scheme_filter_query().

    Returns
    -------
    list[dict]  Matching scheme records.
    """
    try:
        df = spark.sql(query)
        results = [row.asDict() for row in df.collect()]
        logger.info("Scheme query returned %d results", len(results))
        return results
    except Exception as exc:
        logger.error("Scheme SQL query failed: %s", exc)
        return []


def filter_schemes_local(
    schemes: list[dict],
    age: Optional[int] = None,
    gender: Optional[str] = None,
    state_code: Optional[str] = None,
    income: Optional[float] = None,
    caste: Optional[str] = None,
) -> list[dict]:
    """
    Local (non-Spark) fallback: filter a list of scheme dicts in memory.
    Useful when developing without a Spark session.
    """
    filtered = []
    for s in schemes:
        if age is not None:
            if s.get("age_min") and age < s["age_min"]:
                continue
            if s.get("age_max") and age > s["age_max"]:
                continue
        if gender and s.get("gender") and s["gender"] not in ("all", gender):
            continue
        if state_code and s.get("state_codes"):
            codes = s["state_codes"]
            if "all" not in codes and state_code not in codes:
                continue
        if income is not None and s.get("income_max") and income > s["income_max"]:
            continue
        if caste and s.get("caste_flags") and caste not in s["caste_flags"]:
            continue
        filtered.append(s)
    return filtered
