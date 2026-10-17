// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Elementwise kernels.

use super::{expect_len, KernelError};

/// Scalar SiLU activation: `x * sigmoid(x)`.
#[inline]
pub(crate) fn silu_scalar(value: f32) -> f32 {
    value * (1.0_f32 / (1.0_f32 + (-value).exp()))
}

/// Applies SiLU (`x * sigmoid(x)`) to every element in place.
///
/// # Errors
///
/// This kernel has no shape parameters and never fails; the `Result` matches
/// the rest of the reference API.
pub fn silu(x: &mut [f32]) -> Result<(), KernelError> {
    for value in x.iter_mut() {
        *value = silu_scalar(*value);
    }
    Ok(())
}

/// Multiplies `a` and `b` elementwise into `out`.
///
/// All three slices must contain exactly `len` elements.
///
/// # Errors
///
/// Returns [`KernelError::LengthMismatch`] when any slice length differs from
/// `len`.
pub fn mul(a: &[f32], b: &[f32], out: &mut [f32], len: usize) -> Result<(), KernelError> {
    expect_len(a.len(), len)?;
    expect_len(b.len(), len)?;
    expect_len(out.len(), len)?;
    for ((slot, &left), &right) in out.iter_mut().zip(a).zip(b) {
        *slot = left * right;
    }
    Ok(())
}

/// Adds `a` and `b` elementwise into `out`.
///
/// All three slices must contain exactly `len` elements.
///
/// # Errors
///
/// Returns [`KernelError::LengthMismatch`] when any slice length differs from
/// `len`.
pub fn add(a: &[f32], b: &[f32], out: &mut [f32], len: usize) -> Result<(), KernelError> {
    expect_len(a.len(), len)?;
    expect_len(b.len(), len)?;
    expect_len(out.len(), len)?;
    for ((slot, &left), &right) in out.iter_mut().zip(a).zip(b) {
        *slot = left + right;
    }
    Ok(())
}

/// Multiplies every element of `x` by `factor` in place.
///
/// # Errors
///
/// This kernel has no shape parameters and never fails; the `Result` matches
/// the rest of the reference API.
pub fn scale(x: &mut [f32], factor: f32) -> Result<(), KernelError> {
    for value in x.iter_mut() {
        *value *= factor;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn oracle_silu(x: &[f32]) -> Vec<f64> {
        x.iter()
            .map(|&value| {
                let value = f64::from(value);
                value / (1.0 + (-value).exp())
            })
            .collect()
    }

    #[test]
    fn silu_matches_f64_oracle() {
        for (seed, len) in [(1, 1_usize), (2, 7), (3, 33)] {
            let mut lcg = Lcg::from_seed(seed);
            let mut x = vec![0.0_f32; len];
            lcg.fill(&mut x, -8.0, 8.0);
            let expected = oracle_silu(&x);
            silu(&mut x).expect("silu cannot fail");
            assert_close(&x, &expected, 1e-4, 1e-5);
        }

        let mut extreme = [-100.0_f32, -10.0, 0.0, 10.0, 100.0];
        let expected = oracle_silu(&extreme);
        silu(&mut extreme).expect("silu cannot fail");
        assert_close(&extreme, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn mul_add_scale_match_f64_oracle() {
        for (seed, len) in [(11, 0_usize), (12, 1), (13, 5), (14, 33)] {
            let mut lcg = Lcg::from_seed(seed);
            let mut a = vec![0.0_f32; len];
            let mut b = vec![0.0_f32; len];
            lcg.fill(&mut a, -2.0, 2.0);
            lcg.fill(&mut b, -2.0, 2.0);

            let expected_mul: Vec<f64> = a
                .iter()
                .zip(&b)
                .map(|(&left, &right)| f64::from(left) * f64::from(right))
                .collect();
            let mut mul_out = vec![0.0_f32; len];
            mul(&a, &b, &mut mul_out, len).expect("mul should accept equal lengths");
            assert_close(&mul_out, &expected_mul, 1e-4, 1e-5);

            let expected_add: Vec<f64> = a
                .iter()
                .zip(&b)
                .map(|(&left, &right)| f64::from(left) + f64::from(right))
                .collect();
            let mut add_out = vec![0.0_f32; len];
            add(&a, &b, &mut add_out, len).expect("add should accept equal lengths");
            assert_close(&add_out, &expected_add, 1e-4, 1e-5);

            let factor = -1.5_f32;
            let expected_scale: Vec<f64> = a
                .iter()
                .map(|&value| f64::from(value) * f64::from(factor))
                .collect();
            let mut scaled = a.clone();
            scale(&mut scaled, factor).expect("scale cannot fail");
            assert_close(&scaled, &expected_scale, 1e-4, 1e-5);
        }
    }

    #[test]
    fn elementwise_rejects_wrong_lengths() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let mut out = [0.0_f32; 3];

        let error = mul(&a, &b[..2], &mut out, 3).expect_err("short b must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = mul(&a[..2], &b, &mut out, 3).expect_err("short a must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = add(&a, &b, &mut out[..2], 3).expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = add(&a, &b[..2], &mut out, 3).expect_err("short b must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let empty: [f32; 0] = [];
        let mut empty_out: [f32; 0] = [];
        silu(&mut empty_out).expect("empty silu cannot fail");
        scale(&mut empty_out, 2.0).expect("empty scale cannot fail");
        mul(&empty, &empty, &mut empty_out, 0).expect("empty mul cannot fail");
        add(&empty, &empty, &mut empty_out, 0).expect("empty add cannot fail");
    }
}
