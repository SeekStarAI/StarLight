// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Black-box tests for the `starlight-runtime` executable.
//!
//! These tests only cover paths that do not need a working graph compiler,
//! device or engine, so they stay meaningful while the parallel `sl-graph` and
//! `sl-core` work is landing.

use std::path::PathBuf;
use std::process::{Command, Output};

const CONFIG: &str = r#"{
    "model":  { "kind": "fixture", "architecture": "fixture" },
    "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
    "server": { "bind": "127.0.0.1", "port": 8000 }
}"#;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_starlight-runtime"))
}

fn write_config(name: &str, contents: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("starlight-runtime-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write config");
    path
}

fn exit_code(output: &Output) -> i32 {
    output
        .status
        .code()
        .unwrap_or_else(|| panic!("process terminated by signal: {output:?}"))
}

#[test]
fn version_prints_crate_and_ir_versions() {
    let output = binary().arg("version").output().expect("spawn binary");
    assert_eq!(exit_code(&output), 0);
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains(starlight_runtime::VERSION));
    assert!(stdout.contains(&format!("ir_version {}", sl_ir::IR_VERSION)));
}

#[test]
fn help_succeeds_and_lists_subcommands() {
    let output = binary().arg("--help").output().expect("spawn binary");
    assert_eq!(exit_code(&output), 0);
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    for subcommand in ["describe", "validate", "run", "serve", "version"] {
        assert!(
            stdout.contains(subcommand),
            "help must mention {subcommand}"
        );
    }
}

#[test]
fn missing_config_is_a_usage_error() {
    let output = binary()
        .args(["describe", "--config", "/definitely/not/a/config.json"])
        .output()
        .expect("spawn binary");
    assert_eq!(exit_code(&output), 2);
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("failed to read config"));
}

#[test]
fn bad_arguments_exit_with_two() {
    let cases: &[&[&str]] = &[
        &[],
        &["bogus"],
        &["run", "--config", "x.json"],
        &["run", "--input-ids", "1,2"],
        &["run", "--input-ids", "1,nope", "--config", "x.json"],
        &["describe"],
    ];
    for case in cases {
        let output = binary().args(*case).output().expect("spawn binary");
        assert_eq!(exit_code(&output), 2, "arguments {case:?}");
    }
}

#[test]
fn describe_emits_the_runtime_protocol() {
    let config = write_config("describe.json", CONFIG);
    let output = binary()
        .args(["describe", "--config"])
        .arg(&config)
        .output()
        .expect("spawn binary");
    assert_eq!(exit_code(&output), 0);
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("describe is JSON");
    assert_eq!(value["engine"], "starlight");
    assert_eq!(value["runtime_version"], starlight_runtime::VERSION);
    assert_eq!(value["engine_version"], starlight_runtime::VERSION);
    assert_eq!(value["ir_version"], sl_ir::IR_VERSION);
    assert_eq!(value["backend"], "cpu");
    assert_eq!(value["architectures"], serde_json::json!(["fixture"]));
    assert_eq!(value["max_context"], 16);
    assert_eq!(value["max_batch"], 1);
    assert!(value["capabilities"].is_array());
}

#[test]
fn serve_reports_unimplemented_with_exit_five() {
    let config = write_config("serve.json", CONFIG);
    let output = binary()
        .args(["serve", "--config"])
        .arg(&config)
        .output()
        .expect("spawn binary");
    assert_eq!(exit_code(&output), 5);
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("sl-daemon"), "stderr: {stderr}");
}

#[test]
fn run_rejects_token_ids_outside_the_vocabulary() {
    let config = write_config("bad-id.json", CONFIG);
    let output = binary()
        .args(["run", "--config"])
        .arg(&config)
        .args(["--input-ids", "1,8"])
        .output()
        .expect("spawn binary");
    assert_eq!(exit_code(&output), 2);
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("vocabulary"), "stderr: {stderr}");
}

#[test]
fn run_rejects_sequences_over_max_context() {
    let small = r#"{
        "model":  { "kind": "fixture", "architecture": "fixture" },
        "engine": { "backend": "cpu", "max_context": 1, "max_batch": 1 },
        "server": { "bind": "127.0.0.1", "port": 8000 }
    }"#;
    let config = write_config("small-context.json", small);
    let output = binary()
        .args(["run", "--config"])
        .arg(&config)
        .args(["--input-ids", "1,2,3"])
        .output()
        .expect("spawn binary");
    assert_eq!(exit_code(&output), 2);
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("max_context"), "stderr: {stderr}");
}

#[test]
fn invalid_config_is_rejected() {
    let invalid = r#"{
        "model":  { "kind": "fixture", "architecture": "fixture" },
        "engine": { "backend": "vulkan", "max_context": 16, "max_batch": 1 },
        "server": { "bind": "127.0.0.1", "port": 8000 }
    }"#;
    let config = write_config("invalid-backend.json", invalid);
    let output = binary()
        .args(["describe", "--config"])
        .arg(&config)
        .output()
        .expect("spawn binary");
    assert_eq!(exit_code(&output), 2);
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("unsupported engine.backend"));
}
