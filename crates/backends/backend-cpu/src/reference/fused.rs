// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Fused composite reference kernels.
//!
//! A fused kernel computes what would otherwise be a short chain of primitive
//! operators. It shares the guarantees of the other reference kernels: dense
//! row-major tensors, no heap allocation, exhaustive validation before the
//! first write, and a fixed accumulation order for reproducible results.

use super::{expect_len, shape_len, KernelError};

/// Computes `y = (x * rsqrt(mean(x^2) + eps) * weight) · wᵀ`.
///
/// Shapes:
///
/// - `x`: `[rows, hidden]`
/// - `weight`: `[hidden]`
/// - `w`: `[out_features, hidden]`
/// - `out`: `[rows, out_features]`
///
/// This is the fused form of [`rms_norm`](super::rms_norm) followed by
/// [`matmul_nt`](super::matmul_nt). For every row the inverse RMS factor is
/// computed exactly once, then each output column is the left-to-right sum
///
/// ```text
/// out[row, column] = Σ_i (x[row, i] * inverse_rms * weight[i]) * w[column, i]
/// ```
///
/// The kernel allocates no temporary buffer: the normalized row is never
/// materialized. `rows == 0` is accepted as an empty tensor. `hidden` and
/// `out_features` must both be nonzero.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `hidden` or `out_features` is
/// zero, and [`KernelError::LengthMismatch`] when any slice does not match its
/// shape. [`KernelError::ShapeOverflow`] is returned when a shape product does
/// not fit in `usize`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_matmul(
    x: &[f32],
    weight: &[f32],
    w: &[f32],
    out: &mut [f32],
    rows: usize,
    hidden: usize,
    out_features: usize,
    eps: f32,
) -> Result<(), KernelError> {
    if hidden == 0 {
        return Err(KernelError::InvalidShape(
            "rms_norm_matmul requires a nonzero hidden size",
        ));
    }
    if out_features == 0 {
        return Err(KernelError::InvalidShape(
            "rms_norm_matmul requires a nonzero out_features",
        ));
    }

    let x_len = shape_len(&[rows, hidden])?;
    let out_len = shape_len(&[rows, out_features])?;
    expect_len(x.len(), x_len)?;
    expect_len(weight.len(), hidden)?;
    expect_len(w.len(), shape_len(&[out_features, hidden])?)?;
    expect_len(out.len(), out_len)?;

    for row_index in 0..rows {
        let x_row = &x[row_index * hidden..(row_index + 1) * hidden];
        let mut sum_of_squares = 0.0_f32;
        for &value in x_row {
            sum_of_squares += value * value;
        }
        let mean_square = sum_of_squares / hidden as f32;
        let inverse_rms = 1.0_f32 / (mean_square + eps).sqrt();

        let out_row = &mut out[row_index * out_features..(row_index + 1) * out_features];
        for (column, slot) in out_row.iter_mut().enumerate() {
            let w_row = &w[column * hidden..(column + 1) * hidden];
            let mut sum = 0.0_f32;
            for index in 0..hidden {
                sum += x_row[index] * inverse_rms * weight[index] * w_row[index];
            }
            *slot = sum;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    /// Independent fused oracle evaluated entirely in `f64`.
    fn oracle_rms_norm_matmul(
        x: &[f32],
        weight: &[f32],
        w: &[f32],
        rows: usize,
        hidden: usize,
        out_features: usize,
        eps: f32,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows * out_features];
        for row in 0..rows {
            let mut sum_of_squares = 0.0_f64;
            for index in 0..hidden {
                let value = f64::from(x[row * hidden + index]);
                sum_of_squares += value * value;
            }
            let mean_square = sum_of_squares / hidden as f64;
            let inverse_rms = 1.0_f64 / (mean_square + f64::from(eps)).sqrt();
            for column in 0..out_features {
                let mut sum = 0.0_f64;
                for index in 0..hidden {
                    let normalized =
                        f64::from(x[row * hidden + index]) * inverse_rms * f64::from(weight[index]);
                    sum += normalized * f64::from(w[column * hidden + index]);
                }
                out[row * out_features + column] = sum;
            }
        }
        out
    }

    fn check_rms_norm_matmul(rows: usize, hidden: usize, out_features: usize, eps: f32, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; rows * hidden];
        let mut weight = vec![0.0_f32; hidden];
        let mut w = vec![0.0_f32; out_features * hidden];
        let mut out = vec![0.0_f32; rows * out_features];
        lcg.fill(&mut x, -2.0, 2.0);
        lcg.fill(&mut weight, 0.5, 1.5);
        lcg.fill(&mut w, -1.0, 1.0);

        let expected = oracle_rms_norm_matmul(&x, &weight, &w, rows, hidden, out_features, eps);
        rms_norm_matmul(&x, &weight, &w, &mut out, rows, hidden, out_features, eps)
            .expect("rms_norm_matmul should accept valid shapes");
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn rms_norm_matmul_matches_f64_oracle() {
        // Non-power-of-two extents exercise the column-major-free inner loop.
        check_rms_norm_matmul(1, 1, 1, 1e-5, 1);
        check_rms_norm_matmul(3, 7, 5, 1e-6, 2);
        check_rms_norm_matmul(2, 33, 17, 1e-5, 3);
        check_rms_norm_matmul(4, 16, 9, 1e-6, 4);
    }

    #[test]
    fn rms_norm_matmul_accepts_zero_rows() {
        let weight = [1.0_f32, 0.5, 2.0];
        let w = [1.0_f32; 6];
        let empty: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        rms_norm_matmul(&empty, &weight, &w, &mut out, 0, 3, 2, 1e-5)
            .expect("zero rows should be accepted");
    }

    #[test]
    fn rms_norm_matmul_rejects_invalid_shapes() {
        let x = [1.0_f32; 8];
        let weight = [1.0_f32; 4];
        let w = [1.0_f32; 12];
        let mut out = [0.0_f32; 6];
        let empty: [f32; 0] = [];
        let mut empty_out: [f32; 0] = [];

        let error = rms_norm_matmul(&empty, &empty, &empty, &mut empty_out, 2, 0, 3, 1e-5)
            .expect_err("zero hidden must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = rms_norm_matmul(&empty, &empty, &empty, &mut empty_out, 2, 4, 0, 1e-5)
            .expect_err("zero out_features must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = rms_norm_matmul(&x[..7], &weight, &w, &mut out, 2, 4, 3, 1e-5)
            .expect_err("short x must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = rms_norm_matmul(&x, &weight[..3], &w, &mut out, 2, 4, 3, 1e-5)
            .expect_err("short weight must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = rms_norm_matmul(&x, &weight, &w[..11], &mut out, 2, 4, 3, 1e-5)
            .expect_err("short w must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = rms_norm_matmul(&x, &weight, &w, &mut out[..5], 2, 4, 3, 1e-5)
            .expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = rms_norm_matmul(&empty, &weight, &w, &mut empty_out, usize::MAX, 4, 3, 1e-5)
            .expect_err("overflowing rows must be rejected");
        assert_eq!(error, KernelError::ShapeOverflow);
    }
}
