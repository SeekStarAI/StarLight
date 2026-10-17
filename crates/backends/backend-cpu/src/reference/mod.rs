// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Scalar reference kernels.
//!
//! Every kernel in this module is a straightforward scalar implementation
//! written against dense, row-major (`C`-contiguous) tensors. The kernels:
//!
//! - take `&[f32]` / `&mut [f32]` plus explicit shape parameters,
//! - never allocate heap memory,
//! - validate lengths and shapes before touching any output buffer,
//! - accumulate in a fixed order so results are reproducible across targets.
//!
//! They intentionally ignore performance. The SIMD backend and the golden
//! baselines use these results as their numerical oracle.

use core::fmt;

pub mod attention;
pub mod elementwise;
pub mod layout;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod softmax;
pub mod swiglu;

pub use attention::{causal_attention, kv_append};
pub use elementwise::{add, mul, scale, silu};
pub use layout::{gather, last_token, merge_heads, split_heads};
pub use matmul::{matmul_nn, matmul_nt};
pub use norm::rms_norm;
pub use rope::{rope_gptj, rope_neox};
pub use softmax::{softmax, softmax_causal};
pub use swiglu::swiglu;

/// Error returned by a reference kernel.
///
/// The variant set is intentionally small and allocation-free so the enum can
/// cross an FFI boundary or be embedded in a larger backend error type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelError {
    /// A slice has the wrong number of elements for the requested shape.
    LengthMismatch {
        /// Number of elements implied by the shape parameters.
        expected: usize,
        /// Number of elements in the slice supplied by the caller.
        actual: usize,
    },
    /// A shape product does not fit in `usize`.
    ShapeOverflow,
    /// The shape parameters violate a kernel invariant.
    ///
    /// The payload statically describes the violated invariant, for example
    /// `"head_dim must be a positive even number"`.
    InvalidShape(&'static str),
    /// The requested operation or variant is not implemented.
    Unsupported(&'static str),
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "length mismatch: expected {expected} elements, got {actual}"
                )
            }
            Self::ShapeOverflow => formatter.write_str("tensor shape product overflows usize"),
            Self::InvalidShape(reason) => write!(formatter, "invalid shape: {reason}"),
            Self::Unsupported(operation) => write!(formatter, "unsupported: {operation}"),
        }
    }
}

impl core::error::Error for KernelError {}

/// Returns the product of `dimensions`, rejecting overflow.
pub(crate) fn shape_len(dimensions: &[usize]) -> Result<usize, KernelError> {
    dimensions.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(KernelError::ShapeOverflow)
    })
}

/// Validates that a slice contains exactly `expected` elements.
pub(crate) fn expect_len(actual: usize, expected: usize) -> Result<(), KernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(KernelError::LengthMismatch { expected, actual })
    }
}

/// Sequential `f32` dot product with a fixed left-to-right accumulation order.
///
/// Callers must pass slices of equal length; every kernel using this helper
/// validates its shape parameters before slicing.
#[inline]
pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (&left_value, &right_value) in left.iter().zip(right) {
        sum += left_value * right_value;
    }
    sum
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Deterministic 64-bit LCG used by the kernel tests.
    ///
    /// The multiplier and increment are the constants used by PCG; the output
    /// is the high bits of the state. No external `rand` dependency is used.
    pub(crate) struct Lcg(u64);

    impl Lcg {
        pub(crate) fn from_seed(seed: u64) -> Self {
            Self(seed)
        }

        pub(crate) fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        pub(crate) fn uniform(&mut self, low: f32, high: f32) -> f32 {
            let unit = f64::from(self.next_u32()) / f64::from(u32::MAX);
            low + (high - low) * unit as f32
        }

        pub(crate) fn fill(&mut self, values: &mut [f32], low: f32, high: f32) {
            for value in values {
                *value = self.uniform(low, high);
            }
        }

        pub(crate) fn below(&mut self, upper: usize) -> usize {
            assert!(upper > 0, "upper bound must be nonzero");
            (self.next_u32() as usize) % upper
        }
    }

    /// Asserts that `actual` matches an `f64` oracle within a mixed tolerance.
    ///
    /// A value passes when the absolute error is within `abs_tol` *or* the
    /// relative error is within `rel_tol`, i.e. the looser of the two bounds.
    pub(crate) fn assert_close(actual: &[f32], expected: &[f64], abs_tol: f64, rel_tol: f64) {
        assert_eq!(actual.len(), expected.len(), "output length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let difference = (f64::from(actual) - expected).abs();
            let tolerance = abs_tol.max(rel_tol * expected.abs());
            assert!(
                difference <= tolerance,
                "mismatch at index {index}: actual={actual}, expected={expected}, \
                 difference={difference}, tolerance={tolerance}"
            );
        }
    }
}
