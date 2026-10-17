// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Compilation tests for the deterministic fixture (ARCH-0007 §1).

use sl_ir::{ValueId, IR_VERSION};
use starlight_runtime::fixture::{Fixture, MAX_SEQ, VOCAB};
use starlight_runtime::runtime::cpu_capabilities;

#[test]
fn compiles_prefill_and_decode_and_validates_the_plan() {
    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();

    for seq in [2_u32, 1] {
        let ir = fixture
            .compile(seq, &capabilities)
            .unwrap_or_else(|error| panic!("seq {seq} must compile: {error}"));
        assert_eq!(ir.ir_version, IR_VERSION);
        ir.validate(64)
            .unwrap_or_else(|error| panic!("seq {seq} plan must validate: {error}"));

        assert!(
            ir.values.iter().all(|value| !value.shape.has_symbolic()),
            "compiled values must be concrete"
        );

        let logits = ir.value(fixture.logits).expect("logits value");
        assert_eq!(logits.shape.to_consts(), Some(vec![1, VOCAB]));

        let token_ids = ir.value(fixture.token_ids).expect("token id value");
        assert_eq!(token_ids.shape.to_consts(), Some(vec![1, seq]));

        // One command per source node: gather, rms_norm, matmul_nt, last_token.
        let codes: Vec<sl_ir::OpCode> = ir
            .commands
            .iter()
            .map(|command| command.op.code())
            .collect();
        assert_eq!(
            codes,
            vec![
                sl_ir::OpCode::Gather,
                sl_ir::OpCode::RmsNorm,
                sl_ir::OpCode::MatMulNT,
                sl_ir::OpCode::LastToken,
            ]
        );
    }
}

#[test]
fn compilation_is_deterministic() {
    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();
    let first = fixture.compile(2, &capabilities).expect("first compile");
    let second = fixture.compile(2, &capabilities).expect("second compile");
    assert_eq!(format!("{first:?}"), format!("{second:?}"));
}

#[test]
fn rejects_sequence_lengths_above_the_symbolic_bound() {
    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();
    let error = fixture
        .compile(MAX_SEQ + 1, &capabilities)
        .expect_err("seq above the symbolic bound must fail");
    assert!(
        matches!(
            error,
            sl_graph::CompileError::SymbolicDimExceedsMax {
                max: MAX_SEQ,
                requested,
                ..
            } if requested == MAX_SEQ + 1
        ),
        "unexpected compile error: {error}"
    );
}

#[test]
fn fixture_weights_are_bound_to_parameter_values() {
    let fixture = Fixture::build();
    assert_eq!(fixture.weights.len(), 3);
    let parameter_ids: Vec<ValueId> = fixture.weights.iter().map(|(id, _)| *id).collect();
    assert_eq!(parameter_ids, vec![ValueId(1), ValueId(3), ValueId(5)]);
    for (id, data) in &fixture.weights {
        assert_eq!(
            fixture.weight(*id).map(<[f32]>::len),
            Some(data.len()),
            "weight lookup must return the uploaded data"
        );
        assert!(fixture.program.value(*id).expect("parameter").role == sl_ir::ValueRole::Param);
    }
}
