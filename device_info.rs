// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

use std::env;
use std::thread;

fn main() {
    println!("Device information");
    println!("  Hostname: {}", hostname().unwrap_or_else(|| "unavailable".to_owned()));
    println!("  OS: {} ({})", env::consts::OS, env::consts::FAMILY);
    println!("  Architecture: {}", env::consts::ARCH);

    match thread::available_parallelism() {
        Ok(count) => println!("  Available parallelism: {}", count),
        Err(_) => println!("  Available parallelism: unavailable"),
    }
}

fn hostname() -> Option<String> {
    ["HOSTNAME", "COMPUTERNAME"].iter().find_map(|name| {
        env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}