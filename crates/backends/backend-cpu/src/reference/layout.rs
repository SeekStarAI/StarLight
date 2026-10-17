// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Data movement kernels: embedding gather, last-token selection, and the
//! split/merge-head permutations.

use super::{expect_len, shape_len, KernelError};

/// Gathers embedding rows by integer id.
///
/// Shapes:
///
/// - `table`: `[vocab, hidden]`
/// - `ids`: `[rows]`
/// - `out`: `[rows, hidden]`
///
/// Row `r` of `out` is a copy of `table[ids[r]]`. All ids are validated before
/// any output is written, so a failing call leaves `out` unchanged.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] with the message `"id out of range"`
/// when an id is not smaller than `vocab`, and
/// [`KernelError::LengthMismatch`] when a slice does not match its shape.
pub fn gather(
    table: &[f32],
    ids: &[u32],
    out: &mut [f32],
    vocab: usize,
    hidden: usize,
    rows: usize,
) -> Result<(), KernelError> {
    expect_len(table.len(), shape_len(&[vocab, hidden])?)?;
    expect_len(ids.len(), rows)?;
    expect_len(out.len(), shape_len(&[rows, hidden])?)?;

    for &id in ids {
        let index =
            usize::try_from(id).map_err(|_| KernelError::InvalidShape("id out of range"))?;
        if index >= vocab {
            return Err(KernelError::InvalidShape("id out of range"));
        }
    }

    for (row, &id) in ids.iter().enumerate() {
        let source = id as usize * hidden;
        let destination = row * hidden;
        out[destination..destination + hidden].copy_from_slice(&table[source..source + hidden]);
    }

    Ok(())
}

/// Copies the last real token of every batch row from `input` into `out`.
///
/// Shapes:
///
/// - `input`: `[batch, seq, hidden]`
/// - `out`: `[batch, hidden]`
///
/// Only the first `seq_len` positions of each sequence are considered, and
/// position `seq_len - 1` is copied. `seq_len` must satisfy
/// `1 <= seq_len <= seq`; this covers decode (`seq_len == 1`) as well as
/// variable-length batches stored in a padded buffer.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `seq_len` is zero or greater
/// than `seq`, and [`KernelError::LengthMismatch`] when a slice does not match
/// its shape.
pub fn last_token(
    input: &[f32],
    out: &mut [f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    seq_len: usize,
) -> Result<(), KernelError> {
    if !(1..=seq).contains(&seq_len) {
        return Err(KernelError::InvalidShape(
            "last_token requires 1 <= seq_len <= seq",
        ));
    }
    expect_len(input.len(), shape_len(&[batch, seq, hidden])?)?;
    expect_len(out.len(), shape_len(&[batch, hidden])?)?;

    let position = seq_len - 1;
    for batch_index in 0..batch {
        let source = (batch_index * seq + position) * hidden;
        let destination = batch_index * hidden;
        out[destination..destination + hidden].copy_from_slice(&input[source..source + hidden]);
    }

    Ok(())
}

/// Converts `[batch, seq, heads * head_dim]` into
/// `[batch, heads, seq, head_dim]`.
///
/// This is an explicit data movement kernel, not a view.
///
/// # Errors
///
/// Returns [`KernelError::LengthMismatch`] when either slice does not contain
/// `batch * seq * heads * head_dim` elements, and
/// [`KernelError::ShapeOverflow`] when the shape product overflows.
pub fn split_heads(
    input: &[f32],
    out: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), KernelError> {
    let total = shape_len(&[batch, seq, heads, head_dim])?;
    expect_len(input.len(), total)?;
    expect_len(out.len(), total)?;

    for batch_index in 0..batch {
        for token in 0..seq {
            for head in 0..heads {
                let source = ((batch_index * seq + token) * heads + head) * head_dim;
                let destination = ((batch_index * heads + head) * seq + token) * head_dim;
                out[destination..destination + head_dim]
                    .copy_from_slice(&input[source..source + head_dim]);
            }
        }
    }

    Ok(())
}

/// Converts `[batch, heads, seq, head_dim]` back into
/// `[batch, seq, heads * head_dim]`.
///
/// This is the inverse of [`split_heads`], implemented as an explicit data
/// movement kernel.
///
/// # Errors
///
/// Returns [`KernelError::LengthMismatch`] when either slice does not contain
/// `batch * seq * heads * head_dim` elements, and
/// [`KernelError::ShapeOverflow`] when the shape product overflows.
pub fn merge_heads(
    input: &[f32],
    out: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), KernelError> {
    let total = shape_len(&[batch, seq, heads, head_dim])?;
    expect_len(input.len(), total)?;
    expect_len(out.len(), total)?;

    for batch_index in 0..batch {
        for token in 0..seq {
            for head in 0..heads {
                let source = ((batch_index * heads + head) * seq + token) * head_dim;
                let destination = ((batch_index * seq + token) * heads + head) * head_dim;
                out[destination..destination + head_dim]
                    .copy_from_slice(&input[source..source + head_dim]);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    fn oracle_gather(table: &[f32], ids: &[u32], hidden: usize, rows: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows * hidden];
        for row in 0..rows {
            let id = ids[row] as usize;
            for column in 0..hidden {
                out[row * hidden + column] = f64::from(table[id * hidden + column]);
            }
        }
        out
    }

    fn check_gather(vocab: usize, hidden: usize, rows: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut table = vec![0.0_f32; vocab * hidden];
        lcg.fill(&mut table, -1.0, 1.0);
        let mut ids = vec![0_u32; rows];
        for id in &mut ids {
            *id = lcg.below(vocab) as u32;
        }
        let expected = oracle_gather(&table, &ids, hidden, rows);
        let mut out = vec![0.0_f32; rows * hidden];
        gather(&table, &ids, &mut out, vocab, hidden, rows)
            .expect("gather should accept valid shapes");
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    fn check_last_token(batch: usize, seq: usize, hidden: usize, seq_len: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut input = vec![0.0_f32; batch * seq * hidden];
        lcg.fill(&mut input, -1.0, 1.0);
        let mut expected = vec![0.0_f64; batch * hidden];
        for batch_index in 0..batch {
            let source = (batch_index * seq + seq_len - 1) * hidden;
            for column in 0..hidden {
                expected[batch_index * hidden + column] = f64::from(input[source + column]);
            }
        }
        let mut out = vec![0.0_f32; batch * hidden];
        last_token(&input, &mut out, batch, seq, hidden, seq_len)
            .expect("last_token should accept valid shapes");
        assert_close(&out, &expected, 1e-4, 1e-5);
    }

    fn oracle_split(
        input: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; batch * seq * heads * head_dim];
        for batch_index in 0..batch {
            for token in 0..seq {
                for head in 0..heads {
                    for column in 0..head_dim {
                        let source =
                            ((batch_index * seq + token) * heads + head) * head_dim + column;
                        let destination =
                            ((batch_index * heads + head) * seq + token) * head_dim + column;
                        out[destination] = f64::from(input[source]);
                    }
                }
            }
        }
        out
    }

    fn oracle_merge(
        input: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; batch * seq * heads * head_dim];
        for batch_index in 0..batch {
            for token in 0..seq {
                for head in 0..heads {
                    for column in 0..head_dim {
                        let source =
                            ((batch_index * heads + head) * seq + token) * head_dim + column;
                        let destination =
                            ((batch_index * seq + token) * heads + head) * head_dim + column;
                        out[destination] = f64::from(input[source]);
                    }
                }
            }
        }
        out
    }

    fn check_split_merge(batch: usize, seq: usize, heads: usize, head_dim: usize, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut input = vec![0.0_f32; batch * seq * heads * head_dim];
        lcg.fill(&mut input, -1.0, 1.0);

        let mut split = vec![0.0_f32; input.len()];
        let expected_split = oracle_split(&input, batch, seq, heads, head_dim);
        split_heads(&input, &mut split, batch, seq, heads, head_dim)
            .expect("split_heads should accept valid shapes");
        assert_close(&split, &expected_split, 1e-4, 1e-5);

        let mut merged = vec![0.0_f32; input.len()];
        let expected_merge = oracle_merge(&split, batch, seq, heads, head_dim);
        merge_heads(&split, &mut merged, batch, seq, heads, head_dim)
            .expect("merge_heads should accept valid shapes");
        assert_close(&merged, &expected_merge, 1e-4, 1e-5);

        let original: Vec<f64> = input.iter().map(|&value| f64::from(value)).collect();
        assert_close(&merged, &original, 1e-4, 1e-5);
    }

    #[test]
    fn gather_matches_f64_oracle() {
        for (seed, (vocab, hidden, rows)) in [
            (1, (10, 5, 3)),
            (2, (1, 1, 1)),
            (3, (4, 7, 0)),
            (4, (13, 9, 6)),
        ] {
            check_gather(vocab, hidden, rows, seed);
        }
    }

    #[test]
    fn gather_rejects_bad_ids_and_lengths() {
        let table = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 2];
        let error =
            gather(&table, &[4], &mut out, 2, 2, 1).expect_err("out-of-range id must be rejected");
        assert_eq!(error, KernelError::InvalidShape("id out of range"));
        assert_eq!(out, [0.0; 2]);

        let error =
            gather(&table[..3], &[0], &mut out, 2, 2, 1).expect_err("short table must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error =
            gather(&table, &[0, 1], &mut out, 2, 2, 1).expect_err("extra ids must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error =
            gather(&table, &[0], &mut out[..1], 2, 2, 1).expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }

    #[test]
    fn last_token_matches_f64_oracle() {
        for (seed, (batch, seq, hidden, seq_len)) in
            [(11, (1, 1, 3, 1)), (12, (3, 5, 4, 5)), (13, (2, 4, 7, 2))]
        {
            check_last_token(batch, seq, hidden, seq_len, seed);
        }
    }

    #[test]
    fn last_token_rejects_bad_seq_len_and_lengths() {
        let input = [0.0_f32; 16];
        let mut out = [0.0_f32; 4];

        let error =
            last_token(&input, &mut out, 2, 4, 2, 0).expect_err("zero seq_len must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = last_token(&input, &mut out, 2, 4, 2, 5)
            .expect_err("seq_len beyond seq must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = last_token(&input[..15], &mut out, 2, 4, 2, 4)
            .expect_err("short input must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error =
            last_token(&input, &mut out[..3], 2, 4, 2, 4).expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }

    #[test]
    fn split_and_merge_heads_match_f64_oracle() {
        for (seed, (batch, seq, heads, head_dim)) in [
            (21, (1, 1, 1, 2)),
            (22, (2, 3, 4, 4)),
            (23, (2, 5, 3, 6)),
            (24, (1, 4, 2, 8)),
        ] {
            check_split_merge(batch, seq, heads, head_dim, seed);
        }
    }

    #[test]
    fn split_and_merge_reject_wrong_lengths() {
        let input = [0.0_f32; 16];
        let mut out = [0.0_f32; 16];

        let error = split_heads(&input[..15], &mut out, 2, 2, 2, 2)
            .expect_err("short input must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = split_heads(&input, &mut out[..15], 2, 2, 2, 2)
            .expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = merge_heads(&input[..15], &mut out, 2, 2, 2, 2)
            .expect_err("short input must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = merge_heads(&input, &mut out[..15], 2, 2, 2, 2)
            .expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }
}
