// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Real end-to-end test: compile → `CpuDevice` → `Engine::build` → `step`.
//!
//! No mocks: the chain under test is exactly the one `starlight-runtime run`
//! uses. Expected logits are computed by an independent `f64` implementation of
//! gather → RMSNorm → matmul → last token with tolerance `1e-5`.

use std::path::PathBuf;

use sl_core::{CompiledProgram, Engine, EngineConfig, StepInput, StepMode};
use sl_device::Device;
use sl_ir::{BufferId, ValueId};
use starlight_backend_cpu::CpuDevice;
use starlight_runtime::fixture::{Fixture, HIDDEN, MAX_SEQ, VOCAB};
use starlight_runtime::runtime::{cpu_capabilities, dispatch, Command, BUFFER_ALIGNMENT};

const CONFIG: &str = r#"{
    "model":  { "kind": "fixture", "architecture": "fixture" },
    "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
    "server": { "bind": "127.0.0.1", "port": 8000 }
}"#;

fn write_config(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("starlight-runtime-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(name);
    std::fs::write(&path, CONFIG).expect("write config");
    path
}

/// Independent naive implementation in `f64`.
///
/// Gathers embedding rows, applies RMSNorm with the fixture's deterministic
/// weights, projects through the LM head with `MatMulNT`, then selects the last
/// position.
fn expected_logits(token_ids: &[u32]) -> Vec<f64> {
    let embedding: Vec<f64> = (0..(VOCAB * HIDDEN) as usize)
        .map(|index| (index as f64 - 3.0) * 0.1)
        .collect();
    let norm_weights = [1.0_f64, 0.5, 1.5, 0.75];
    let lm_head: Vec<f64> = (0..(VOCAB * HIDDEN) as usize)
        .map(|index| ((index % 5) as f64 - 2.0) * 0.1)
        .collect();

    let hidden = HIDDEN as usize;
    let mut last_rms = Vec::new();
    for &id in token_ids {
        let offset = id as usize * hidden;
        let row = &embedding[offset..offset + hidden];
        let mean_square = row.iter().map(|value| value * value).sum::<f64>() / hidden as f64;
        let scale = 1.0 / (mean_square + 1e-5).sqrt();
        last_rms = row
            .iter()
            .zip(norm_weights)
            .map(|(value, weight)| value * scale * weight)
            .collect();
    }

    (0..VOCAB as usize)
        .map(|vocab| {
            (0..hidden)
                .map(|column| last_rms[column] * lm_head[vocab * hidden + column])
                .sum()
        })
        .collect()
}

fn upload_weights(device: &mut CpuDevice, fixture: &Fixture) -> Vec<(ValueId, BufferId)> {
    fixture
        .weights
        .iter()
        .map(|(value, data)| {
            let buffer = device
                .allocate_f32(data.len(), BUFFER_ALIGNMENT)
                .expect("allocate weight buffer");
            device
                .upload_f32(buffer, data)
                .expect("upload weight buffer");
            (*value, buffer)
        })
        .collect()
}

fn compiled_program(
    fixture: &Fixture,
    ir: sl_ir::CommandBuffer,
    weights: &[(ValueId, BufferId)],
) -> CompiledProgram {
    CompiledProgram {
        ir,
        weights: weights.to_vec(),
        token_ids: fixture.token_ids,
        positions: fixture.positions,
        seq_len: fixture.seq_len,
        past_len: None,
        logits: fixture.logits,
    }
}

fn assert_matches_expected(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (f64::from(*actual) - expected).abs();
        assert!(
            difference <= 1e-5,
            "logit {index}: actual {actual} != expected {expected} (difference {difference})"
        );
    }
}

#[test]
fn runs_the_real_prefill_chain_and_matches_the_f64_oracle() {
    let token_ids = [1_u32, 2];
    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();
    let seq = token_ids.len() as u32;
    let prefill_ir = fixture
        .compile(seq, &capabilities)
        .expect("compile prefill");
    let decode_ir = fixture.compile(1, &capabilities).expect("compile decode");

    let mut device = CpuDevice::new();
    let weights = upload_weights(&mut device, &fixture);
    let config = EngineConfig {
        prefill: compiled_program(&fixture, prefill_ir, &weights),
        decode: compiled_program(&fixture, decode_ir, &weights),
        device: Box::new(device),
        max_batch: 1,
        max_context: 16,
    };
    let mut engine = Engine::build(config).expect("build engine");

    let positions = [0_u32, 1];
    let mut logits = vec![0.0_f32; VOCAB as usize];
    let stats = engine
        .step(
            StepMode::Prefill,
            StepInput {
                token_ids: &token_ids,
                positions: &positions,
                seq_len: seq,
                past_len: 0,
            },
            &mut logits,
        )
        .expect("prefill step");
    assert_eq!(stats.tokens, seq);

    assert_matches_expected(&logits, &expected_logits(&token_ids));
}

#[test]
fn the_run_command_uses_the_same_chain() {
    let config = write_config("run.json");
    let mut output = Vec::new();
    dispatch(
        &Command::Run {
            config,
            input_ids: vec![1, 2],
        },
        &mut output,
    )
    .expect("run command");

    let text = String::from_utf8(output).expect("logits are UTF-8");
    let logits: Vec<f32> = text
        .lines()
        .map(|line| line.parse::<f32>().expect("one f32 logit per line"))
        .collect();
    assert_eq!(logits.len(), VOCAB as usize);
    assert_matches_expected(&logits, &expected_logits(&[1, 2]));
}

#[test]
fn validate_builds_an_engine_without_stepping_it() {
    let config = write_config("validate.json");
    let mut output = Vec::new();
    dispatch(&Command::Validate { config }, &mut output).expect("validate command");
    assert_eq!(String::from_utf8(output).expect("utf8"), "ok\n");
}

#[test]
fn single_token_prefill_is_supported() {
    let token_ids = [7_u32];
    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();
    let prefill_ir = fixture.compile(1, &capabilities).expect("compile prefill");
    let decode_ir = fixture.compile(1, &capabilities).expect("compile decode");

    let mut device = CpuDevice::new();
    let weights = upload_weights(&mut device, &fixture);
    let config = EngineConfig {
        prefill: compiled_program(&fixture, prefill_ir, &weights),
        decode: compiled_program(&fixture, decode_ir, &weights),
        device: Box::new(device),
        max_batch: 1,
        max_context: 16,
    };
    let mut engine = Engine::build(config).expect("build engine");
    let positions = [0_u32];
    let mut logits = vec![0.0_f32; VOCAB as usize];
    engine
        .step(
            StepMode::Prefill,
            StepInput {
                token_ids: &token_ids,
                positions: &positions,
                seq_len: 1,
                past_len: 0,
            },
            &mut logits,
        )
        .expect("single token step");
    assert_matches_expected(&logits, &expected_logits(&token_ids));
}

#[test]
fn fixture_max_sequence_is_exposed_for_validation() {
    assert_eq!(MAX_SEQ, 8);
    assert_eq!(
        Fixture::shape_spec(MAX_SEQ).get(starlight_runtime::fixture::SEQ),
        Some(MAX_SEQ)
    );
}
