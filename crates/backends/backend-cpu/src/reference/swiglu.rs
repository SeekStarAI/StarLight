// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Fused SwiGLU feed-forward kernel.

use super::elementwise::silu_scalar;
use super::{dot, expect_len, shape_len, KernelError};

/// Computes a SwiGLU feed-forward block without temporary buffers.
///
/// Shapes:
///
/// - `x`: `[rows, hidden]`
/// - `w_gate`: `[inter, hidden]`
/// - `w_up`: `[inter, hidden]`
/// - `w_down`: `[hidden, inter]`
/// - `out`: `[rows, hidden]`
///
/// Semantics:
///
/// ```text
/// gate = silu(x · w_gateᵀ)
/// up   = x · w_upᵀ
/// h    = gate * up
/// out  = h · w_downᵀ
/// ```
///
/// The kernel processes one output row at a time. For each intermediate index
/// `i` it computes `g_i = silu(dot(x[r], w_gate[i]))` and
/// `u_i = dot(x[r], w_up[i])`, then immediately accumulates
/// `out[r, j] += g_i * u_i * w_down[j, i]`. This keeps the memory footprint
/// constant and avoids an `inter`-sized temporary. Both `hidden` and `inter`
/// must be nonzero.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `hidden` or `inter` is zero, and
/// [`KernelError::LengthMismatch`] when a slice does not match its shape.
#[allow(clippy::too_many_arguments)]
pub fn swiglu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    out: &mut [f32],
    rows: usize,
    hidden: usize,
    inter: usize,
) -> Result<(), KernelError> {
    if hidden == 0 {
        return Err(KernelError::InvalidShape(
            "swiglu requires a nonzero hidden size",
        ));
    }
    if inter == 0 {
        return Err(KernelError::InvalidShape(
            "swiglu requires a nonzero inter size",
        ));
    }
    let tensor_len = shape_len(&[rows, hidden])?;
    expect_len(x.len(), tensor_len)?;
    expect_len(w_gate.len(), shape_len(&[inter, hidden])?)?;
    expect_len(w_up.len(), shape_len(&[inter, hidden])?)?;
    expect_len(w_down.len(), shape_len(&[hidden, inter])?)?;
    expect_len(out.len(), tensor_len)?;

    for row_index in 0..rows {
        let x_row = &x[row_index * hidden..(row_index + 1) * hidden];
        let out_row = &mut out[row_index * hidden..(row_index + 1) * hidden];
        out_row.fill(0.0);
        for intermediate in 0..inter {
            let gate_row = &w_gate[intermediate * hidden..(intermediate + 1) * hidden];
            let up_row = &w_up[intermediate * hidden..(intermediate + 1) * hidden];
            let gate = silu_scalar(dot(x_row, gate_row));
            let up = dot(x_row, up_row);
            let weight = gate * up;
            for (column, slot) in out_row.iter_mut().enumerate() {
                *slot += weight * w_down[column * inter + intermediate];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn oracle_swiglu(
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        rows: usize,
        hidden: usize,
        inter: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows * hidden];
        for row in 0..rows {
            let mut intermediate_values = vec![0.0_f64; inter];
            for intermediate in 0..inter {
                let mut gate = 0.0_f64;
                let mut up = 0.0_f64;
                for column in 0..hidden {
                    let x_value = f64::from(x[row * hidden + column]);
                    gate += x_value * f64::from(w_gate[intermediate * hidden + column]);
                    up += x_value * f64::from(w_up[intermediate * hidden + column]);
                }
                let silu = gate / (1.0 + (-gate).exp());
                intermediate_values[intermediate] = silu * up;
            }
            for column in 0..hidden {
                let mut sum = 0.0_f64;
                for intermediate in 0..inter {
                    sum += intermediate_values[intermediate]
                        * f64::from(w_down[column * inter + intermediate]);
                }
                out[row * hidden + column] = sum;
            }
        }
        out
    }

    fn check_swiglu(rows: usize, hidden: usize, inter: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut x = vec![0.0_f32; rows * hidden];
        let mut w_gate = vec![0.0_f32; inter * hidden];
        let mut w_up = vec![0.0_f32; inter * hidden];
        let mut w_down = vec![0.0_f32; hidden * inter];
        lcg.fill(&mut x, -1.0, 1.0);
        lcg.fill(&mut w_gate, -1.0, 1.0);
        lcg.fill(&mut w_up, -1.0, 1.0);
        lcg.fill(&mut w_down, -1.0, 1.0);

        let expected = oracle_swiglu(&x, &w_gate, &w_up, &w_down, rows, hidden, inter);
        let mut out = vec![0.0_f32; rows * hidden];
        swiglu(&x, &w_gate, &w_up, &w_down, &mut out, rows, hidden, inter)
            .expect("swiglu should accept valid shapes");
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    #[test]
    fn swiglu_matches_f64_oracle() {
        check_swiglu(1, 1, 1, 1);
        check_swiglu(3, 5, 7, 2);
        check_swiglu(2, 8, 4, 3);
        check_swiglu(4, 6, 13, 4);
    }

    #[test]
    fn swiglu_accepts_zero_rows() {
        let empty: [f32; 0] = [];
        let w_gate = [0.5_f32; 6];
        let w_up = [0.25_f32; 6];
        let w_down = [0.75_f32; 6];
        let mut out: [f32; 0] = [];
        swiglu(&empty, &w_gate, &w_up, &w_down, &mut out, 0, 3, 2)
            .expect("zero rows should be accepted");
    }

    #[test]
    fn swiglu_rejects_invalid_shapes() {
        let empty: [f32; 0] = [];
        let mut empty_out: [f32; 0] = [];
        let error = swiglu(&empty, &empty, &empty, &empty, &mut empty_out, 0, 0, 2)
            .expect_err("zero hidden must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = swiglu(&empty, &empty, &empty, &empty, &mut empty_out, 0, 2, 0)
            .expect_err("zero inter must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let x = [1.0_f32; 4];
        let w_gate = [1.0_f32; 8];
        let w_up = [1.0_f32; 8];
        let w_down = [1.0_f32; 8];
        let mut out = [0.0_f32; 4];

        let error = swiglu(&x[..3], &w_gate, &w_up, &w_down, &mut out, 2, 2, 4)
            .expect_err("short x must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = swiglu(&x, &w_gate[..7], &w_up, &w_down, &mut out, 2, 2, 4)
            .expect_err("short w_gate must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = swiglu(&x, &w_gate, &w_up[..7], &w_down, &mut out, 2, 2, 4)
            .expect_err("short w_up must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = swiglu(&x, &w_gate, &w_up, &w_down[..7], &mut out, 2, 2, 4)
            .expect_err("short w_down must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = swiglu(&x, &w_gate, &w_up, &w_down, &mut out[..3], 2, 2, 4)
            .expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }
}
