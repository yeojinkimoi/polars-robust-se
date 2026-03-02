from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import polars as pl
from polars.plugins import register_plugin_function

ExprOrStr = Union[pl.Expr, str]


def _parse(e: ExprOrStr) -> pl.Expr:
    return pl.col(e) if isinstance(e, str) else e


def ols_hc1(
    target: ExprOrStr,
    *features: ExprOrStr,
    add_intercept: bool = False,
) -> pl.Expr:
    """Compute OLS with HC1 robust standard errors.

    Args:
        target: Target (dependent variable) column.
        *features: Feature (independent variable) columns.
        add_intercept: Whether to add a constant intercept term.

    Returns:
        Polars expression yielding a struct with:
            feature_names, coefficients, standard_errors,
            t_values, p_values, r2, mse, mae
    """
    target_expr = _parse(target)
    feature_exprs = [_parse(f) for f in features]

    if add_intercept:
        feature_exprs.append(
            target_expr.fill_null(0.0).mul(0.0).add(1.0).alias("const")
        )

    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="ols_hc1",
        args=[target_expr, *feature_exprs],
        kwargs={"null_policy": "zero"},
        is_elementwise=False,
        returns_scalar=True,
        input_wildcard_expansion=True,
    ).alias("hc1_statistics")
