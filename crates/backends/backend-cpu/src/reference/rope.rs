// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Rotary position embedding (RoPE), NeoX and GPT-J layouts.

use super::{expect_len, shape_len, KernelError};

/// Validates the shared RoPE shapes.
fn validate_rope(
    x_len: usize,
    positions_len: usize,
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
) -> Result<(), KernelError> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(
            "head_dim must be a positive even number",
        ));
    }
    expect_len(x_len, shape_len(&[batch, heads, seq, head_dim])?)?;
    expect_len(positions_len, shape_len(&[batch, seq])?)?;
    Ok(())
}

/// Returns the angle for pair `pair` of a head at `position`.
///
/// The frequency is `theta^(2 * pair / head_dim)`; the angle is computed in
/// `f64` and only the final sine/cosine are rounded to `f32`.
#[inline]
fn rope_angle(position: f64, pair: usize, head_dim: usize, theta: f64) -> f64 {
    let exponent = 2.0 * pair as f64 / head_dim as f64;
    position / theta.powf(exponent)
}

/// Applies NeoX-style (split-half) rotary position embedding in place.
///
/// Shapes:
///
/// - `x`: `[batch, heads, seq, head_dim]`
/// - `positions`: `[batch, seq]`
///
/// Within each head the pairs are `(x[i], x[i + head_dim / 2])`. For pair `i`
/// and position `p` the kernel computes the angle `p / theta^(2i/head_dim)` in
/// `f64`, rounds sine and cosine to `f32`, and rotates the pair.
///
/// `head_dim` must be positive and even. Empty batches, heads or sequence
/// lengths are accepted.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] for an odd or zero `head_dim`, and
/// [`KernelError::LengthMismatch`] when a slice does not match its shape.
pub fn rope_neox(
    x: &mut [f32],
    positions: &[u32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    theta: f32,
) -> Result<(), KernelError> {
    validate_rope(x.len(), positions.len(), batch, heads, seq, head_dim)?;
    let half = head_dim / 2;
    let theta = f64::from(theta);

    for batch_index in 0..batch {
        for token in 0..seq {
            let position = f64::from(positions[batch_index * seq + token]);
            for pair in 0..half {
                let angle = rope_angle(position, pair, head_dim, theta);
                let (sine, cosine) = angle.sin_cos();
                let sine = sine as f32;
                let cosine = cosine as f32;
                for head in 0..heads {
                    let base = ((batch_index * heads + head) * seq + token) * head_dim;
                    let first = x[base + pair];
                    let second = x[base + pair + half];
                    x[base + pair] = first * cosine - second * sine;
                    x[base + pair + half] = first * sine + second * cosine;
                }
            }
        }
    }

    Ok(())
}

/// Applies GPT-J-style (interleaved) rotary position embedding in place.
///
/// Shapes:
///
/// - `x`: `[batch, heads, seq, head_dim]`
/// - `positions`: `[batch, seq]`
///
/// Within each head the pairs are `(x[2i], x[2i + 1])`. For pair `i` and
/// position `p` the kernel uses the same angle `p / theta^(2i/head_dim)` as
/// [`rope_neox`], computed in `f64` and rounded to `f32`.
///
/// `head_dim` must be positive and even. Empty batches, heads or sequence
/// lengths are accepted.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] for an odd or zero `head_dim`, and
/// [`KernelError::LengthMismatch`] when a slice does not match its shape.
pub fn rope_gptj(
    x: &mut [f32],
    positions: &[u32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    theta: f32,
) -> Result<(), KernelError> {
    validate_rope(x.len(), positions.len(), batch, heads, seq, head_dim)?;
    let half = head_dim / 2;
    let theta = f64::from(theta);

    for batch_index in 0..batch {
        for token in 0..seq {
            let position = f64::from(positions[batch_index * seq + token]);
            for pair in 0..half {
                let angle = rope_angle(position, pair, head_dim, theta);
                let (sine, cosine) = angle.sin_cos();
                let sine = sine as f32;
                let cosine = cosine as f32;
                for head in 0..heads {
                    let base = ((batch_index * heads + head) * seq + token) * head_dim;
                    let first = x[base + 2 * pair];
                    let second = x[base + 2 * pair + 1];
                    x[base + 2 * pair] = first * cosine - second * sine;
                    x[base + 2 * pair + 1] = first * sine + second * cosine;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn make_positions(lcg: &mut Lcg, batch: usize, seq: usize, past_offset: u32) -> Vec<u32> {
        let mut positions = vec![0_u32; batch * seq];
        for (index, position) in positions.iter_mut().enumerate() {
            *position = past_offset + index as u32 * 2 + lcg.below(5) as u32;
        }
        positions
    }

    fn oracle_rope_neox(
        x: &[f32],
        positions: &[u32],
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
    ) -> Vec<f64> {
        let mut out: Vec<f64> = x.iter().map(|&value| f64::from(value)).collect();
        let half = head_dim / 2;
        for batch_index in 0..batch {
            for token in 0..seq {
                let position = f64::from(positions[batch_index * seq + token]);
                for pair in 0..half {
                    let angle = f64::from(theta).powf(2.0 * pair as f64 / head_dim as f64);
                    let (sine, cosine) = (position / angle).sin_cos();
                    for head in 0..heads {
                        let base = ((batch_index * heads + head) * seq + token) * head_dim;
                        let first = f64::from(x[base + pair]);
                        let second = f64::from(x[base + pair + half]);
                        out[base + pair] = first * cosine - second * sine;
                        out[base + pair + half] = first * sine + second * cosine;
                    }
                }
            }
        }
        out
    }

    fn oracle_rope_gptj(
        x: &[f32],
        positions: &[u32],
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
    ) -> Vec<f64> {
        let mut out: Vec<f64> = x.iter().map(|&value| f64::from(value)).collect();
        let half = head_dim / 2;
        for batch_index in 0..batch {
            for token in 0..seq {
                let position = f64::from(positions[batch_index * seq + token]);
                for pair in 0..half {
                    let angle = f64::from(theta).powf(2.0 * pair as f64 / head_dim as f64);
                    let (sine, cosine) = (position / angle).sin_cos();
                    for head in 0..heads {
                        let base = ((batch_index * heads + head) * seq + token) * head_dim;
                        let first = f64::from(x[base + 2 * pair]);
                        let second = f64::from(x[base + 2 * pair + 1]);
                        out[base + 2 * pair] = first * cosine - second * sine;
                        out[base + 2 * pair + 1] = first * sine + second * cosine;
                    }
                }
            }
        }
        out
    }

    fn check_neox(batch: usize, heads: usize, seq: usize, head_dim: usize, theta: f32, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; batch * heads * seq * head_dim];
        lcg.fill(&mut x, -2.0, 2.0);
        let positions = make_positions(&mut lcg, batch, seq, 3);
        let expected = oracle_rope_neox(&x, &positions, batch, heads, seq, head_dim, theta);
        rope_neox(&mut x, &positions, batch, heads, seq, head_dim, theta)
            .expect("rope_neox should accept valid shapes");
        assert_close(&x, &expected, 1e-4, 1e-5);
    }

    fn check_gptj(batch: usize, heads: usize, seq: usize, head_dim: usize, theta: f32, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; batch * heads * seq * head_dim];
        lcg.fill(&mut x, -2.0, 2.0);
        let positions = make_positions(&mut lcg, batch, seq, 1);
        let expected = oracle_rope_gptj(&x, &positions, batch, heads, seq, head_dim, theta);
        rope_gptj(&mut x, &positions, batch, heads, seq, head_dim, theta)
            .expect("rope_gptj should accept valid shapes");
        assert_close(&x, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn rope_neox_matches_f64_oracle() {
        for (seed, (batch, heads, seq, head_dim, theta)) in [
            (1, (1, 1, 1, 2, 10_000.0)),
            (2, (2, 3, 5, 4, 10_000.0)),
            (3, (2, 2, 4, 8, 500_000.0)),
            (4, (1, 3, 3, 6, 10_000.0)),
        ] {
            check_neox(batch, heads, seq, head_dim, theta, seed);
        }
    }

    #[test]
    fn rope_gptj_matches_f64_oracle() {
        for (seed, (batch, heads, seq, head_dim, theta)) in [
            (11, (1, 1, 1, 2, 10_000.0)),
            (12, (2, 3, 5, 4, 10_000.0)),
            (13, (2, 2, 4, 8, 500_000.0)),
            (14, (1, 2, 3, 10, 1_000_000.0)),
        ] {
            check_gptj(batch, heads, seq, head_dim, theta, seed);
        }
    }

    #[test]
    fn rope_rejects_invalid_shapes() {
        let mut x = [0.0_f32; 8];
        let positions = [0_u32; 2];

        let error = rope_neox(&mut x, &positions, 1, 1, 2, 3, 10_000.0)
            .expect_err("odd head_dim must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = rope_gptj(&mut x, &positions, 1, 1, 2, 0, 10_000.0)
            .expect_err("zero head_dim must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = rope_neox(&mut x, &positions[..1], 1, 1, 2, 4, 10_000.0)
            .expect_err("short positions must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = rope_gptj(&mut x[..7], &positions, 1, 1, 2, 4, 10_000.0)
            .expect_err("short x must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }
}
