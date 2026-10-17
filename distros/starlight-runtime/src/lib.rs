// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Server runtime composition root for StarLight.
//!
//! This crate wires the graph compiler, core engine, device and backend into a
//! single executable. It owns configuration, `describe`/`validate`/`run` and
//! (later) `serve`; it does not implement kernels, tokenization or HTTP.

pub mod config;
pub mod fixture;
pub mod runtime;

/// Runtime crate version, reported by `version` and `describe`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
