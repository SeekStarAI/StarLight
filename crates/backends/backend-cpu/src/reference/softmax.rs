// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! In-place softmax kernels.

use super::{expect_len, shape_len, KernelError};

/// Applies a numerically stable softmax to every contiguous row of `x`.
///
/// `x` is interpreted as `[rows, cols]`. Each row subtracts its maximum before
/// exponentiating, then divides by the sum of the exponentials. The reduction
/// order is fixed: left to right over the columns.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `rows` or `cols` is zero, and
/// [`KernelError::LengthMismatch`] when `x` does not contain `rows * cols`
/// elements.
pub fn softmax(x: &mut [f32], rows: usize, cols: usize) -> Result<(), KernelError> {
    if rows == 0 || cols == 0 {
        return Err(KernelError::InvalidShape(
            "softmax requires nonzero rows and cols",
        ));
    }
    expect_len(x.len(), shape_len(&[rows, cols])?)?;

    for row_index in 0..rows {
        let row = &mut x[row_index * cols..(row_index + 1) * cols];
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let inverse_sum = 1.0_f32 / sum;
        for value in row.iter_mut() {
            *value *= inverse_sum;
        }
    }

    Ok(())
}

/// Applies a causal, in-place softmax with a KV-cache prefix.
///
/// `x` is interpreted as `[rows, cols]` where `rows = batch * heads * seq` and
/// `cols = past_len + seq`. For row `r`, the query position is `r % seq` and
/// the last visible key position is `past_len + (r % seq)`. Columns after that
/// position are written as zero; the remaining columns are normalized with a
/// numerically stable softmax.
///
/// This is the score softmax used by single-token (decode) and chunked
/// (prefill) causal attention. It does not read a KV cache itself.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `rows`, `cols` or `seq` is zero,
/// when `cols != past_len + seq`, or when `rows` is not a multiple of `seq`.
/// Returns [`KernelError::LengthMismatch`] when `x` does not contain
/// `rows * cols` elements.
pub fn softmax_causal(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    seq: usize,
    past_len: usize,
) -> Result<(), KernelError> {
    if rows == 0 || cols == 0 {
        return Err(KernelError::InvalidShape(
            "softmax_causal requires nonzero rows and cols",
        ));
    }
    if seq == 0 {
        return Err(KernelError::InvalidShape(
            "softmax_causal requires a nonzero seq",
        ));
    }
    let expected_cols = past_len
        .checked_add(seq)
        .ok_or(KernelError::ShapeOverflow)?;
    if cols != expected_cols {
        return Err(KernelError::InvalidShape(
            "softmax_causal requires cols == past_len + seq",
        ));
    }
    if rows % seq != 0 {
        return Err(KernelError::InvalidShape(
            "softmax_causal requires rows to be a multiple of seq",
        ));
    }
    expect_len(x.len(), shape_len(&[rows, cols])?)?;

    for row_index in 0..rows {
        let query = row_index % seq;
        let last_visible = past_len + query;
        let row = &mut x[row_index * cols..(row_index + 1) * cols];
        let maximum = row[..=last_visible]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for value in &mut row[..=last_visible] {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let inverse_sum = 1.0_f32 / sum;
        for value in &mut row[..=last_visible] {
            *value *= inverse_sum;
        }
        row[last_visible + 1..].fill(0.0);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn oracle_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows * cols];
        for row_index in 0..rows {
            let row = &x[row_index * cols..(row_index + 1) * cols];
            let maximum = row
                .iter()
                .map(|&value| f64::from(value))
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0_f64;
            for (slot, &value) in out[row_index * cols..(row_index + 1) * cols]
                .iter_mut()
                .zip(row)
            {
                let exponential = (f64::from(value) - maximum).exp();
                *slot = exponential;
                sum += exponential;
            }
            for slot in &mut out[row_index * cols..(row_index + 1) * cols] {
                *slot /= sum;
            }
        }
        out
    }

    fn oracle_softmax_causal(
        x: &[f32],
        rows: usize,
        cols: usize,
        seq: usize,
        past_len: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows * cols];
        for row_index in 0..rows {
            let query = row_index % seq;
            let last_visible = past_len + query;
            let row = &x[row_index * cols..(row_index + 1) * cols];
            let maximum = row[..=last_visible]
                .iter()
                .map(|&value| f64::from(value))
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0_f64;
            for column in 0..=last_visible {
                let exponential = (f64::from(row[column]) - maximum).exp();
                out[row_index * cols + column] = exponential;
                sum += exponential;
            }
            for column in 0..=last_visible {
                out[row_index * cols + column] /= sum;
            }
        }
        out
    }

    fn check_softmax(rows: usize, cols: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; rows * cols];
        lcg.fill(&mut x, -4.0, 4.0);
        let expected = oracle_softmax(&x, rows, cols);
        softmax(&mut x, rows, cols).expect("softmax should accept valid shapes");
        assert_close(&x, &expected, 1e-4, 1e-5);
    }

    fn check_softmax_causal(rows: usize, seq: usize, past_len: usize, seed: u64) {
        let cols = past_len + seq;
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; rows * cols];
        lcg.fill(&mut x, -3.0, 3.0);
        let expected = oracle_softmax_causal(&x, rows, cols, seq, past_len);
        softmax_causal(&mut x, rows, cols, seq, past_len)
            .expect("softmax_causal should accept valid shapes");
        assert_close(&x, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn softmax_matches_f64_oracle() {
        for (seed, (rows, cols)) in [(1, (1, 1)), (2, (3, 5)), (3, (6, 17))] {
            check_softmax(rows, cols, seed);
        }
    }

    #[test]
    fn softmax_is_stable_for_extreme_values() {
        let mut x = vec![1000.0_f32, 999.0, -1000.0, 0.0];
        let expected = oracle_softmax(&x, 2, 2);
        softmax(&mut x, 2, 2).expect("softmax should accept valid shapes");
        assert_close(&x, &expected, 1e-4, 1e-5);
        assert!(x.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn softmax_causal_matches_f64_oracle() {
        for (seed, (rows, seq, past_len)) in [
            (11, (2, 2, 0)),
            (12, (12, 4, 1)),
            (13, (4, 1, 7)),
            (14, (6, 3, 2)),
        ] {
            check_softmax_causal(rows, seq, past_len, seed);
        }
    }

    #[test]
    fn softmax_rejects_invalid_shapes() {
        let mut empty: [f32; 0] = [];
        let error = softmax(&mut empty, 0, 4).expect_err("zero rows must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = softmax(&mut empty, 4, 0).expect_err("zero cols must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let mut short = [0.0_f32; 5];
        let error = softmax(&mut short, 2, 3).expect_err("short input must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let mut mismatched_cols = [0.0_f32; 6];
        let error = softmax_causal(&mut mismatched_cols, 2, 3, 2, 0)
            .expect_err("cols must equal past_len + seq");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let mut zero_seq = [0.0_f32; 4];
        let error =
            softmax_causal(&mut zero_seq, 2, 2, 0, 2).expect_err("zero seq must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let mut bad_rows = [0.0_f32; 6];
        let error =
            softmax_causal(&mut bad_rows, 3, 2, 2, 0).expect_err("rows must be a multiple of seq");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let mut short_causal = [0.0_f32; 5];
        let error = softmax_causal(&mut short_causal, 2, 3, 2, 1)
            .expect_err("short input must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }
}
