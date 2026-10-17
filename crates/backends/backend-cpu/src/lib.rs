// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! CPU compute backend for StarLight.
//!
//! The crate ships a scalar, allocation-free reference implementation of the
//! compute kernels needed by the first model family. The kernels live in
//! [`mod@reference`] and serve as the numerical oracle for SIMD implementations
//! and golden baselines. They are deliberately simple and must not be used for
//! performance evaluation.
//!
//! [`executor`] contains a reference interpreter that runs an
//! [`sl_ir::CommandBuffer`] through those kernels. It is a semantic baseline,
//! not a device backend. [`device::CpuDevice`] is the `sl-device` backend: it
//! owns buffers and dispatches one operator at a time, also through these
//! kernels.

#![forbid(unsafe_code)]

pub mod device;
pub mod executor;
pub mod reference;

pub use device::CpuDevice;
pub use executor::{execute, BindingTable, BufferStore, ExecutionError, ScalarStore, SCRATCHPAD};
pub use reference::KernelError;
