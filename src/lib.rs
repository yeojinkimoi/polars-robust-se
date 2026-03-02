#![allow(clippy::unit_arg, clippy::unused_unit)]

use ndarray::{Array1, Array2, Axis};
use polars::datatypes::{DataType, Field};
use polars::error::{polars_err, PolarsResult};
use polars::frame::DataFrame;
use polars::prelude::{
    FillNullStrategy, IntoSeries, NamedFrom, NamedFromOwned, Series,
};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::distribution::{ContinuousCDF, StudentsT};

// ---------------------------------------------------------------------------
// Kwargs deserialized from Python
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct OLSKwargs {
    null_policy: Option<String>,
    feature_names: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Helpers: Polars Series -> ndarray
// ---------------------------------------------------------------------------

fn series_to_f64_array(s: &Series) -> Array1<f64> {
    let s = s
        .cast(&DataType::Float64)
        .expect("cast to f64")
        .fill_null(FillNullStrategy::Zero)
        .unwrap();
    let ca = s.f64().expect("f64 chunked array").rechunk();
    let mut out = Array1::<f64>::zeros(ca.len());
    for (i, val) in ca.into_no_null_iter().enumerate() {
        out[i] = val;
    }
    out
}

fn build_features(inputs: &[Series]) -> Array2<f64> {
    let n = inputs[0].len();
    let m = inputs.len();
    let mut x = Array2::<f64>::zeros((n, m));
    for (j, s) in inputs.iter().enumerate() {
        let col = series_to_f64_array(s);
        x.column_mut(j).assign(&col);
    }
    x
}

// ---------------------------------------------------------------------------
// Linear algebra: Gauss-Jordan matrix inversion with partial pivoting
// ---------------------------------------------------------------------------

fn invert_matrix(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());

    // Augmented matrix [A | I]
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
        }
        aug[(i, n + i)] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[(col, col)].abs();
        for row in (col + 1)..n {
            let val = aug[(row, col)].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for k in 0..(2 * n) {
                let tmp = aug[(col, k)];
                aug[(col, k)] = aug[(max_row, k)];
                aug[(max_row, k)] = tmp;
            }
        }

        let pivot = aug[(col, col)];
        assert!(pivot.abs() > 1e-15, "Matrix is singular");

        // Scale pivot row
        for k in 0..(2 * n) {
            aug[(col, k)] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[(row, col)];
                for k in 0..(2 * n) {
                    aug[(row, k)] -= factor * aug[(col, k)];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = aug[(i, n + j)];
        }
    }
    inv
}

// ---------------------------------------------------------------------------
// OLS + HC1
// ---------------------------------------------------------------------------

fn solve_ols(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    let xtx_inv = invert_matrix(&xtx);
    xtx_inv.dot(&xty)
}

fn t_value_to_p_value(t_value: f64, df: f64) -> f64 {
    let t_dist =
        StudentsT::new(0.0, 1.0, df).expect("Invalid StudentT parameters");
    2.0 * (1.0 - t_dist.cdf(t_value.abs()))
}

struct HC1Result {
    coefficients: Vec<f64>,
    std_errors: Vec<f64>,
    t_values: Vec<f64>,
    p_values: Vec<f64>,
    r2: f64,
    mse: f64,
    mae: f64,
}

fn compute_ols_hc1(x: &Array2<f64>, y: &Array1<f64>) -> HC1Result {
    let n = x.nrows() as f64;
    let p = x.ncols() as f64;

    let beta = solve_ols(x, y);

    let y_hat = x.dot(&beta);
    let residuals = y - &y_hat;

    let mean_y = y.mean().unwrap_or(0.0);
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut sum_abs = 0.0;
    for i in 0..y.len() {
        let e = residuals[i];
        ss_res += e * e;
        ss_tot += (y[i] - mean_y).powi(2);
        sum_abs += e.abs();
    }

    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { f64::NAN };
    let mse = ss_res / n;
    let mae = sum_abs / n;

    let xtx = x.t().dot(x);
    let bread = invert_matrix(&xtx);

    let mut xe = x.to_owned();
    for (mut row, &e) in xe.axis_iter_mut(Axis(0)).zip(residuals.iter()) {
        row *= e;
    }
    let meat = xe.t().dot(&xe);
    let sandwich = bread.dot(&meat).dot(&bread);
    let scale = n / (n - p);

    let df_resid = n - p;
    let k = beta.len();
    let mut std_errors = Vec::with_capacity(k);
    let mut t_values = Vec::with_capacity(k);
    let mut p_values = Vec::with_capacity(k);

    for i in 0..k {
        let se = (scale * sandwich[(i, i)].abs()).sqrt();
        let t = beta[i] / se;
        let pv = t_value_to_p_value(t, df_resid);
        std_errors.push(se);
        t_values.push(t);
        p_values.push(pv);
    }

    HC1Result {
        coefficients: beta.to_vec(),
        std_errors,
        t_values,
        p_values,
        r2,
        mse,
        mae,
    }
}

// ---------------------------------------------------------------------------
// Polars expression plugin
// ---------------------------------------------------------------------------

fn hc1_struct_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("feature_names", DataType::List(Box::new(DataType::String))),
        Field::new("coefficients", DataType::List(Box::new(DataType::Float64))),
        Field::new("standard_errors", DataType::List(Box::new(DataType::Float64))),
        Field::new("t_values", DataType::List(Box::new(DataType::Float64))),
        Field::new("p_values", DataType::List(Box::new(DataType::Float64))),
        Field::new("r2", DataType::Float64),
        Field::new("mse", DataType::Float64),
        Field::new("mae", DataType::Float64),
    ];
    Ok(Field::new("hc1_statistics", DataType::Struct(fields)))
}

#[polars_expr(output_type_func=hc1_struct_dtype)]
fn ols_hc1(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let _ = kwargs;

    let y = series_to_f64_array(&inputs[0]);
    let x = build_features(&inputs[1..]);

    let result = compute_ols_hc1(&x, &y);

    let feature_names: Vec<String> = match &kwargs.feature_names {
        Some(names) if names.len() == inputs.len() - 1 => names.clone(),
        _ => inputs[1..]
            .iter()
            .map(|s| s.name().to_string())
            .collect(),
    };
    let feature_names_series = Series::new("feature_names", &feature_names);

    let df = DataFrame::new(vec![
        Series::new("feature_names", [feature_names_series]),
        Series::new("coefficients", [Series::from_vec("coefficients", result.coefficients)]),
        Series::new("standard_errors", [Series::from_vec("standard_errors", result.std_errors)]),
        Series::new("t_values", [Series::from_vec("t_values", result.t_values)]),
        Series::new("p_values", [Series::from_vec("p_values", result.p_values)]),
        Series::new("r2", &[result.r2]),
        Series::new("mse", &[result.mse]),
        Series::new("mae", &[result.mae]),
    ])?;

    Ok(df.into_struct("hc1_statistics").into_series())
}
