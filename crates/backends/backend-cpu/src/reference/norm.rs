// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! RMS normalization.

use super::{expect_len, shape_len, KernelError};

/// Applies RMS normalization over the last dimension.
///
/// Shapes:
///
/// - `x`: `[rows, hidden]`
/// - `weight`: `[hidden]`
/// - `out`: `[rows, hidden]`
///
/// For each row the kernel computes `sum(x^2) / hidden`, adds `eps`, takes the
/// reciprocal square root once, and scales every element exactly once:
/// `out = x * rsqrt(mean(x^2) + eps) * weight`. The squared sum is accumulated
/// left to right in `f32`.
///
/// `rows == 0` is accepted as an empty tensor. `hidden` must be nonzero.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `hidden` is zero, and
/// [`KernelError::LengthMismatch`] when any slice does not match its shape.
pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    out: &mut [f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<(), KernelError> {
    if hidden == 0 {
        return Err(KernelError::InvalidShape(
            "rms_norm requires a nonzero hidden size",
        ));
    }
    let tensor_len = shape_len(&[rows, hidden])?;
    expect_len(x.len(), tensor_len)?;
    expect_len(weight.len(), hidden)?;
    expect_len(out.len(), tensor_len)?;

    for row_index in 0..rows {
        let x_row = &x[row_index * hidden..(row_index + 1) * hidden];
        let out_row = &mut out[row_index * hidden..(row_index + 1) * hidden];
        let mut sum_of_squares = 0.0_f32;
        for &value in x_row {
            sum_of_squares += value * value;
        }
        let mean_square = sum_of_squares / hidden as f32;
        let inverse_rms = 1.0_f32 / (mean_square + eps).sqrt();
        for ((slot, &value), &scale) in out_row.iter_mut().zip(x_row).zip(weight) {
            *slot = value * inverse_rms * scale;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn oracle_rms_norm(
        x: &[f32],
        weight: &[f32],
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows * hidden];
        for row_index in 0..rows {
            let mut sum_of_squares = 0.0_f64;
            for column in 0..hidden {
                let value = f64::from(x[row_index * hidden + column]);
                sum_of_squares += value * value;
            }
            let mean_square = sum_of_squares / hidden as f64;
            let inverse_rms = 1.0_f64 / (mean_square + f64::from(eps)).sqrt();
            for column in 0..hidden {
                out[row_index * hidden + column] = f64::from(x[row_index * hidden + column])
                    * inverse_rms
                    * f64::from(weight[column]);
            }
        }
        out
    }

    fn check_rms_norm(rows: usize, hidden: usize, eps: f32, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; rows * hidden];
        let mut weight = vec![0.0_f32; hidden];
        lcg.fill(&mut x, -2.0, 2.0);
        lcg.fill(&mut weight, 0.5, 1.5);
        let mut out = vec![0.0_f32; rows * hidden];
        let expected = oracle_rms_norm(&x, &weight, rows, hidden, eps);
        rms_norm(&x, &weight, &mut out, rows, hidden, eps)
            .expect("rms_norm should accept valid shapes");
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn rms_norm_matches_f64_oracle() {
        check_rms_norm(1, 1, 1e-5, 1);
        check_rms_norm(3, 7, 1e-6, 2);
        check_rms_norm(4, 16, 1e-5, 3);
        check_rms_norm(2, 33, 1e-6, 4);
    }

    #[test]
    fn rms_norm_accepts_zero_rows() {
        let weight = [1.0_f32; 4];
        let empty: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        rms_norm(&empty, &weight, &mut out, 0, 4, 1e-5).expect("zero rows should be accepted");
    }

    #[test]
    fn rms_norm_rejects_invalid_shapes() {
        let empty: [f32; 0] = [];
        let mut empty_out: [f32; 0] = [];
        let error = rms_norm(&empty, &empty, &mut empty_out, 2, 0, 1e-5)
            .expect_err("zero hidden must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let x = [1.0_f32; 8];
        let weight = [1.0_f32; 4];
        let mut out = [0.0_f32; 8];
        let error = rms_norm(&x, &weight[..3], &mut out, 2, 4, 1e-5)
            .expect_err("short weight must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error =
            rms_norm(&x[..7], &weight, &mut out, 2, 4, 1e-5).expect_err("short x must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = rms_norm(&x, &weight, &mut out[..7], 2, 4, 1e-5)
            .expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }
}
