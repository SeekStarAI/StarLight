// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Deterministic tiny fixture used by the end-to-end test.
//!
//! The graph is a minimal decoder layer without attention:
//!
//! ```text
//! token_ids ─ Gather(embed, token_ids) ─ RmsNorm(x, norm_weight)
//!           ─ MatMulNT(rms, lm_head) ─ LastToken(matmul, seq_len) ─ logits
//! ```
//!
//! Shapes are `vocab = 8`, `hidden = 4`, `seq` symbolic with bound `8`
//! (ARCH-0007 §1). Prefill uses `seq = max_context` (2 for the CLI example) and
//! decode uses `seq = 1`; both are compiled from the same [`Program`] with a
//! different [`ShapeSpec`].
//!
//! Every weight is a deterministic constant. There is no randomness, no file
//! access and no dependency on `sl-model-adapters`.

use sl_ir::{
    CommandBuffer, DType, Dim, Op, Program, Shape, ShapeSpec, ShapeSpecEntry, SymDimDecl, SymId,
    Value, ValueId, ValueRole,
};

/// Vocabulary size.
pub const VOCAB: u32 = 8;
/// Hidden width.
pub const HIDDEN: u32 = 4;
/// Largest accepted sequence length, matching the symbolic bound.
pub const MAX_SEQ: u32 = 8;
/// Symbolic sequence dimension.
pub const SEQ: SymId = SymId(0);

/// Number of `f32` elements per embedding row.
pub const EMBED_ELEMENTS: usize = (VOCAB * HIDDEN) as usize;

/// A built fixture: the symbolic program plus the deterministic weights and the
/// value ids the engine needs to bind.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// Symbolic program shared by the prefill and decode compilations.
    pub program: Program,
    /// Deterministic weights, in [`ValueId`] order.
    pub weights: Vec<(ValueId, Vec<f32>)>,
    /// `Input` value carrying token ids.
    pub token_ids: ValueId,
    /// `Input` value carrying positions; the fixture has no RoPE, so `None`.
    pub positions: Option<ValueId>,
    /// `Scalar` value carrying the current sequence length.
    pub seq_len: ValueId,
    /// `Output` value carrying logits.
    pub logits: ValueId,
}

impl Fixture {
    /// Builds the deterministic fixture.
    ///
    /// The returned program passes [`Program::validate`].
    pub fn build() -> Self {
        let token_ids = ValueId(0);
        let embed = ValueId(1);
        let gather = ValueId(2);
        let norm_weight = ValueId(3);
        let rms = ValueId(4);
        let lm_head = ValueId(5);
        let matmul = ValueId(6);
        let seq_len = ValueId(7);
        let logits = ValueId(8);

        let seq_shape = Shape::new(vec![Dim::Const(1), Dim::Sym(SEQ)]);
        let seq_hidden = Shape::new(vec![Dim::Const(1), Dim::Sym(SEQ), Dim::Const(HIDDEN)]);

        let values = vec![
            Value::new(DType::U32, seq_shape, ValueRole::Input),
            Value::new(
                DType::F32,
                Shape::consts(&[VOCAB, HIDDEN]),
                ValueRole::Param,
            ),
            Value::new(DType::F32, seq_hidden.clone(), ValueRole::Temp),
            Value::new(DType::F32, Shape::consts(&[HIDDEN]), ValueRole::Param),
            Value::new(DType::F32, seq_hidden, ValueRole::Temp),
            Value::new(
                DType::F32,
                Shape::consts(&[VOCAB, HIDDEN]),
                ValueRole::Param,
            ),
            Value::new(
                DType::F32,
                Shape::new(vec![Dim::Const(1), Dim::Sym(SEQ), Dim::Const(VOCAB)]),
                ValueRole::Temp,
            ),
            Value::new(DType::U32, Shape::default(), ValueRole::Scalar),
            Value::new(DType::F32, Shape::consts(&[1, VOCAB]), ValueRole::Output),
        ];

        let nodes = vec![
            sl_ir::Node {
                op: Op::Gather,
                inputs: vec![embed, token_ids],
                outputs: vec![gather],
                effect: sl_ir::Effect::None,
            },
            sl_ir::Node {
                op: Op::RmsNorm { eps: 1e-5 },
                inputs: vec![gather, norm_weight],
                outputs: vec![rms],
                effect: sl_ir::Effect::None,
            },
            sl_ir::Node {
                op: Op::MatMulNT,
                inputs: vec![rms, lm_head],
                outputs: vec![matmul],
                effect: sl_ir::Effect::None,
            },
            sl_ir::Node {
                op: Op::LastToken,
                inputs: vec![matmul, seq_len],
                outputs: vec![logits],
                effect: sl_ir::Effect::None,
            },
        ];

        let program = Program {
            sym_dims: vec![SymDimDecl {
                id: SEQ,
                name: "seq",
                max: MAX_SEQ,
            }],
            values,
            nodes,
            inputs: vec![token_ids],
            outputs: vec![logits],
            states: Vec::new(),
        };
        program
            .validate()
            .expect("the deterministic fixture program is valid");

        let weights = vec![
            (embed, embedding_weights()),
            (norm_weight, norm_weights()),
            (lm_head, lm_head_weights()),
        ];

        Self {
            program,
            weights,
            token_ids,
            positions: None,
            seq_len,
            logits,
        }
    }

    /// Returns the prefill/decode shape specification for `seq`.
    ///
    /// # Errors
    ///
    /// This function cannot fail for the fixture; the caller passes the result
    /// to `sl_graph::compile`, which rejects values above [`MAX_SEQ`].
    pub fn shape_spec(seq: u32) -> ShapeSpec {
        ShapeSpec::new(vec![ShapeSpecEntry {
            id: SEQ,
            max: MAX_SEQ,
            value: seq,
        }])
    }

    /// Returns the deterministic weight for `value`, if the value is a
    /// parameter.
    pub fn weight(&self, value: ValueId) -> Option<&[f32]> {
        self.weights
            .iter()
            .find(|(id, _)| *id == value)
            .map(|(_, data)| data.as_slice())
    }

    /// Returns the number of logits (the vocabulary size).
    pub const fn logits_len(&self) -> usize {
        VOCAB as usize
    }

    /// Convenience wrapper mirroring `sl_graph::compile` for one sequence
    /// length.
    ///
    /// # Errors
    ///
    /// Propagates [`sl_graph::CompileError`].
    pub fn compile(
        &self,
        seq: u32,
        capabilities: &sl_ir::DeviceCapabilities,
    ) -> Result<CommandBuffer, sl_graph::CompileError> {
        sl_graph::compile(&self.program, &Self::shape_spec(seq), capabilities)
    }
}

/// Deterministic embedding table, `embed[i] = (i as f32 - 3.0) * 0.1`.
pub fn embedding_weights() -> Vec<f32> {
    (0..EMBED_ELEMENTS)
        .map(|index| (index as f32 - 3.0) * 0.1)
        .collect()
}

/// Deterministic RMSNorm scales.
pub fn norm_weights() -> Vec<f32> {
    vec![1.0, 0.5, 1.5, 0.75]
}

/// Deterministic LM head, `lm_head[i] = ((i % 5) as f32 - 2.0) * 0.1`.
pub fn lm_head_weights() -> Vec<f32> {
    (0..EMBED_ELEMENTS)
        .map(|index| ((index % 5) as f32 - 2.0) * 0.1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        embedding_weights, lm_head_weights, norm_weights, Fixture, HIDDEN, MAX_SEQ, VOCAB,
    };
    use sl_ir::{Dim, Op, ValueId, ValueRole};

    #[test]
    fn program_is_valid_and_has_the_expected_shape() {
        let fixture = Fixture::build();
        let program = &fixture.program;
        assert_eq!(program.validate(), Ok(()));
        assert_eq!(program.inputs, vec![fixture.token_ids]);
        assert_eq!(program.outputs, vec![fixture.logits]);
        assert!(program.states.is_empty());
        assert_eq!(program.sym_dims.len(), 1);
        assert_eq!(program.sym_dims[0].max, MAX_SEQ);
        assert_eq!(program.nodes.len(), 4);
        assert!(matches!(program.nodes[0].op, Op::Gather));
        assert!(matches!(program.nodes[1].op, Op::RmsNorm { .. }));
        assert!(matches!(program.nodes[2].op, Op::MatMulNT));
        assert!(matches!(program.nodes[3].op, Op::LastToken));

        let logits = program.value(fixture.logits).expect("logits value");
        assert_eq!(logits.role, ValueRole::Output);
        assert_eq!(logits.dtype, sl_ir::DType::F32);
        assert_eq!(logits.shape.to_consts(), Some(vec![1, VOCAB]));

        let token_ids = program.value(fixture.token_ids).expect("token id value");
        assert_eq!(token_ids.role, ValueRole::Input);
        assert_eq!(
            token_ids.shape.as_slice(),
            [Dim::Const(1), Dim::Sym(super::SEQ)].as_slice()
        );
        assert_eq!(fixture.positions, None);
        assert_eq!(fixture.seq_len, ValueId(7));
        assert_eq!(fixture.weight(ValueId(1)).map(<[f32]>::len), Some(32));
        assert_eq!(fixture.logits_len(), VOCAB as usize);
    }

    #[test]
    fn weights_are_deterministic_constants() {
        assert_eq!(embedding_weights().len(), 32);
        assert_eq!(embedding_weights()[0], -0.3);
        assert_eq!(embedding_weights()[3], 0.0);
        assert_eq!(embedding_weights()[31], 2.8);
        assert_eq!(norm_weights(), vec![1.0, 0.5, 1.5, 0.75]);
        assert_eq!(lm_head_weights().len(), 32);
        assert_eq!(lm_head_weights()[0], -0.2);
        assert_eq!(lm_head_weights()[4], 0.2);
        assert_eq!(lm_head_weights()[5], -0.2);
        assert_eq!(HIDDEN, 4);
    }

    #[test]
    fn shape_specs_resolve_the_symbolic_dimension() {
        for seq in [1, 2, MAX_SEQ] {
            let spec = Fixture::shape_spec(seq);
            let shape = sl_ir::Shape::new(vec![Dim::Const(1), Dim::Sym(super::SEQ)]);
            assert_eq!(
                spec.resolve(&shape).expect("resolve").to_consts(),
                Some(vec![1, seq])
            );
        }
    }
}
