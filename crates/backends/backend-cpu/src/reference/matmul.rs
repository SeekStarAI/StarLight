// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Batched matrix multiplication in `NT` and `NN` layouts.
//!
//! Both kernels use the same fixed accumulation order inside every batch:
//! output row `m`, then output column `n`, then the reduction dimension `k`
//! from `0` upwards. This order is part of the reference contract and must be
//! preserved by any golden baseline derived from these functions.

use super::{dot, expect_len, shape_len, KernelError};

/// Computes `out = a · bᵀ` for batched `NT` operands.
///
/// Shapes:
///
/// - `a`: `[batch, m, k]`
/// - `b`: `[batch, n, k]`
/// - `out`: `[batch, m, n]`
///
/// An empty batch or any zero-sized `m`/`n`/`k` is accepted. For `k == 0` the
/// output is filled with zeros.
///
/// # Errors
///
/// Returns [`KernelError::LengthMismatch`] when a slice does not match the
/// requested shape and [`KernelError::ShapeOverflow`] when a shape product
/// does not fit in `usize`.
pub fn matmul_nt(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), KernelError> {
    let a_len = shape_len(&[batch, m, k])?;
    let b_len = shape_len(&[batch, n, k])?;
    let out_len = shape_len(&[batch, m, n])?;
    expect_len(a.len(), a_len)?;
    expect_len(b.len(), b_len)?;
    expect_len(out.len(), out_len)?;

    for batch_index in 0..batch {
        let a_batch = batch_index * m * k;
        let b_batch = batch_index * n * k;
        let out_batch = batch_index * m * n;
        for row in 0..m {
            let a_row = &a[a_batch + row * k..a_batch + row * k + k];
            for column in 0..n {
                let b_row = &b[b_batch + column * k..b_batch + column * k + k];
                out[out_batch + row * n + column] = dot(a_row, b_row);
            }
        }
    }

    Ok(())
}

/// Computes `out = a · b` for batched `NN` operands.
///
/// Shapes:
///
/// - `a`: `[batch, m, k]`
/// - `b`: `[batch, k, n]`
/// - `out`: `[batch, m, n]`
///
/// An empty batch or any zero-sized `m`/`n`/`k` is accepted. For `k == 0` the
/// output is filled with zeros.
///
/// # Errors
///
/// Returns [`KernelError::LengthMismatch`] when a slice does not match the
/// requested shape and [`KernelError::ShapeOverflow`] when a shape product
/// does not fit in `usize`.
pub fn matmul_nn(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), KernelError> {
    let a_len = shape_len(&[batch, m, k])?;
    let b_len = shape_len(&[batch, k, n])?;
    let out_len = shape_len(&[batch, m, n])?;
    expect_len(a.len(), a_len)?;
    expect_len(b.len(), b_len)?;
    expect_len(out.len(), out_len)?;

    for batch_index in 0..batch {
        let a_batch = batch_index * m * k;
        let b_batch = batch_index * k * n;
        let out_batch = batch_index * m * n;
        for row in 0..m {
            let a_row = &a[a_batch + row * k..a_batch + row * k + k];
            for column in 0..n {
                let mut sum = 0.0_f32;
                for (depth, &value) in a_row.iter().enumerate() {
                    sum += value * b[b_batch + depth * n + column];
                }
                out[out_batch + row * n + column] = sum;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn oracle_matmul_nt(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; batch * m * n];
        for batch_index in 0..batch {
            for row in 0..m {
                for column in 0..n {
                    let mut sum = 0.0_f64;
                    for depth in 0..k {
                        let a_index = (batch_index * m + row) * k + depth;
                        let b_index = (batch_index * n + column) * k + depth;
                        sum += f64::from(a[a_index]) * f64::from(b[b_index]);
                    }
                    out[(batch_index * m + row) * n + column] = sum;
                }
            }
        }
        out
    }

    fn oracle_matmul_nn(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; batch * m * n];
        for batch_index in 0..batch {
            for row in 0..m {
                for column in 0..n {
                    let mut sum = 0.0_f64;
                    for depth in 0..k {
                        let a_index = (batch_index * m + row) * k + depth;
                        let b_index = (batch_index * k + depth) * n + column;
                        sum += f64::from(a[a_index]) * f64::from(b[b_index]);
                    }
                    out[(batch_index * m + row) * n + column] = sum;
                }
            }
        }
        out
    }

    fn check_matmul_nt(batch: usize, m: usize, k: usize, n: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut a = vec![0.0_f32; batch * m * k];
        let mut b = vec![0.0_f32; batch * n * k];
        lcg.fill(&mut a, -1.0, 1.0);
        lcg.fill(&mut b, -1.0, 1.0);
        let mut out = vec![0.0_f32; batch * m * n];
        matmul_nt(&a, &b, &mut out, batch, m, k, n).expect("matmul_nt should accept valid shapes");
        let expected = oracle_matmul_nt(&a, &b, batch, m, k, n);
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    fn check_matmul_nn(batch: usize, m: usize, k: usize, n: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut a = vec![0.0_f32; batch * m * k];
        let mut b = vec![0.0_f32; batch * k * n];
        lcg.fill(&mut a, -1.0, 1.0);
        lcg.fill(&mut b, -1.0, 1.0);
        let mut out = vec![0.0_f32; batch * m * n];
        matmul_nn(&a, &b, &mut out, batch, m, k, n).expect("matmul_nn should accept valid shapes");
        let expected = oracle_matmul_nn(&a, &b, batch, m, k, n);
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn matmul_nt_matches_f64_oracle() {
        for (seed, (batch, m, k, n)) in [(1, (1, 1, 1, 1)), (2, (2, 3, 5, 7)), (3, (2, 4, 9, 6))] {
            check_matmul_nt(batch, m, k, n, seed);
        }
    }

    #[test]
    fn matmul_nn_matches_f64_oracle() {
        for (seed, (batch, m, k, n)) in [(11, (1, 1, 1, 1)), (12, (3, 4, 7, 5)), (13, (2, 5, 3, 9))]
        {
            check_matmul_nn(batch, m, k, n, seed);
        }
    }

    #[test]
    fn matmul_accepts_empty_dimensions() {
        let empty: [f32; 0] = [];
        let mut empty_out: [f32; 0] = [];
        let empty_b = [0.0_f32; 2 * 4 * 3];

        matmul_nt(&empty, &empty, &mut empty_out, 0, 2, 3, 4)
            .expect("empty batch should be accepted");
        matmul_nn(&empty, &empty, &mut empty_out, 0, 2, 3, 4)
            .expect("empty batch should be accepted");
        matmul_nt(&empty, &empty_b, &mut empty_out, 2, 0, 3, 4)
            .expect("empty rows should be accepted");
        matmul_nn(&empty, &empty_b, &mut empty_out, 2, 0, 3, 4)
            .expect("empty rows should be accepted");

        let mut nt_out = [1.0_f32; 6];
        matmul_nt(&empty, &empty, &mut nt_out, 1, 2, 0, 3).expect("zero k should be accepted");
        assert_eq!(nt_out, [0.0; 6]);

        let mut nn_out = [1.0_f32; 6];
        matmul_nn(&empty, &empty, &mut nn_out, 1, 2, 0, 3).expect("zero k should be accepted");
        assert_eq!(nn_out, [0.0; 6]);
    }

    #[test]
    fn matmul_rejects_wrong_lengths() {
        let a = vec![0.0_f32; 2 * 3 * 4];
        let b_nt = vec![0.0_f32; 2 * 5 * 4];
        let b_nn = vec![0.0_f32; 2 * 4 * 5];
        let mut out = vec![0.0_f32; 2 * 3 * 5];

        let short_a = &a[..a.len() - 1];
        let error =
            matmul_nt(short_a, &b_nt, &mut out, 2, 3, 4, 5).expect_err("short a must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let short_b = &b_nt[..b_nt.len() - 1];
        let error =
            matmul_nt(&a, short_b, &mut out, 2, 3, 4, 5).expect_err("short b must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let out_len = out.len();
        let short_out = &mut out[..out_len - 1];
        let error =
            matmul_nn(&a, &b_nn, short_out, 2, 3, 4, 5).expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let wrong_b = vec![0.0_f32; b_nn.len() - 1];
        let error = matmul_nn(&a, &wrong_b, &mut out, 2, 3, 4, 5)
            .expect_err("wrong b length must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }

    #[test]
    fn matmul_reports_shape_overflow() {
        let empty: [f32; 0] = [];
        let mut empty_out: [f32; 0] = [];
        let error = matmul_nt(&empty, &empty, &mut empty_out, usize::MAX, 2, 1, 1)
            .expect_err("overflowing shape must be rejected");
        assert_eq!(error, KernelError::ShapeOverflow);
    }
}
