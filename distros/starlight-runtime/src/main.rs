// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! `starlight-runtime` executable entry point.

fn main() {
    std::process::exit(starlight_runtime::runtime::entry(std::env::args().skip(1)));
}
