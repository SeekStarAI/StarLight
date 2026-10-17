// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! CPU compute backend for StarLight.
//!
//! The crate currently ships a scalar, allocation-free reference
//! implementation of the compute kernels needed by the first model family.
//! The reference kernels live in [`reference`] and serve as the numerical
//! oracle for SIMD implementations and golden baselines. They are deliberately
//! simple and must not be used for performance evaluation.

#![forbid(unsafe_code)]

pub mod reference;

pub use reference::KernelError;
