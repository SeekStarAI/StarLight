// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! KV cache append and fused causal attention.

use super::{dot, expect_len, shape_len, KernelError};

/// Appends new K/V rows to a layer of the KV cache.
///
/// Cache layout (all dimensions row-major, innermost last):
///
/// ```text
/// cache[batch, layers, 2, max_seq, kv_heads, head_dim]
/// ```
///
/// The `2` dimension is the K/V selector: index `0` stores keys, index `1`
/// stores values. Shapes:
///
/// - `k`: `[batch, kv_heads, seq, head_dim]`
/// - `v`: `[batch, kv_heads, seq, head_dim]`
/// - `cache`: `[batch, layers, 2, max_seq, kv_heads, head_dim]`
///
/// New tokens are written to positions `past_len .. past_len + seq` of
/// `layer`. Existing entries outside that range are left untouched.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `layer >= layers` or when
/// `past_len + seq > max_seq`, and [`KernelError::LengthMismatch`] when a
/// slice does not match its shape. All checks run before the first write, so a
/// rejected call leaves the cache unchanged.
#[allow(clippy::too_many_arguments)]
pub fn kv_append(
    k: &[f32],
    v: &[f32],
    cache: &mut [f32],
    batch: usize,
    layers: usize,
    layer: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    past_len: usize,
    seq: usize,
) -> Result<(), KernelError> {
    let kv_len = shape_len(&[batch, kv_heads, seq, head_dim])?;
    let cache_len = shape_len(&[batch, layers, 2, max_seq, kv_heads, head_dim])?;
    expect_len(k.len(), kv_len)?;
    expect_len(v.len(), kv_len)?;
    expect_len(cache.len(), cache_len)?;
    if layer >= layers {
        return Err(KernelError::InvalidShape("layer out of range"));
    }
    let end = past_len
        .checked_add(seq)
        .ok_or(KernelError::ShapeOverflow)?;
    if end > max_seq {
        return Err(KernelError::InvalidShape("past_len + seq exceeds max_seq"));
    }

    let time_stride = shape_len(&[kv_heads, head_dim])?;
    let parity_stride = shape_len(&[max_seq, kv_heads, head_dim])?;
    let layer_stride = shape_len(&[2, max_seq, kv_heads, head_dim])?;
    let batch_stride = shape_len(&[layers, 2, max_seq, kv_heads, head_dim])?;

    for batch_index in 0..batch {
        for head in 0..kv_heads {
            for token in 0..seq {
                let source = ((batch_index * kv_heads + head) * seq + token) * head_dim;
                let destination = batch_index * batch_stride
                    + layer * layer_stride
                    + (past_len + token) * time_stride
                    + head * head_dim;
                cache[destination..destination + head_dim]
                    .copy_from_slice(&k[source..source + head_dim]);
                cache[destination + parity_stride..destination + parity_stride + head_dim]
                    .copy_from_slice(&v[source..source + head_dim]);
            }
        }
    }

    Ok(())
}

/// Computes fused causal attention with a KV cache.
///
/// Shapes:
///
/// - `q`: `[batch, heads, seq, head_dim]`
/// - `k_new`: `[batch, kv_heads, seq, head_dim]`
/// - `v_new`: `[batch, kv_heads, seq, head_dim]`
/// - `cache`: `[batch, layers, 2, max_seq, kv_heads, head_dim]`
/// - `out`: `[batch, heads, seq, head_dim]`
///
/// `k_new`/`v_new` are first appended to `cache` at `layer` with
/// [`kv_append`], starting at `past_len`. Then, for every `(batch, head,
/// token)`, the kernel computes
///
/// ```text
/// score[j] = scale * dot(q[b, h, t, :], cache[b, layer, 0, j, kv_head, :])
/// ```
///
/// for `j in 0..=past_len + t`, normalizes the scores with a numerically
/// stable softmax, and accumulates
/// `out[b, h, t, :] = Σ_j p[j] * cache[b, layer, 1, j, kv_head, :]`.
///
/// `layer` selects the cache layer and must be smaller than `layers`; it is a
/// parameter (rather than a fixed zero) so the same kernel serves every
/// transformer layer. `scale` is supplied by the caller, normally
/// `1 / sqrt(head_dim)`.
///
/// GQA/MQA is expressed by `heads` and `kv_heads`: `heads` must be a nonzero
/// multiple of `kv_heads`, `group = heads / kv_heads`, and query head `h`
/// reads KV head `h / group`. `heads == kv_heads` is ordinary MHA.
///
/// The kernel allocates no memory and stores no score vector. It scans the
/// keys three times (maximum, denominator, weighted values), which is slower
/// than a buffered implementation but keeps the stack footprint constant.
/// `seq == 0` is a no-op apart from shape validation.
///
/// # Errors
///
/// Returns [`KernelError::InvalidShape`] when `kv_heads` or `heads` is zero,
/// when `heads` is not a multiple of `kv_heads`, when `layer` is out of range,
/// or when the appended range exceeds `max_seq`; and
/// [`KernelError::LengthMismatch`] when a slice does not match its shape. All
/// validation happens before the cache is modified.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention(
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    cache: &mut [f32],
    out: &mut [f32],
    batch: usize,
    layers: usize,
    layer: usize,
    heads: usize,
    kv_heads: usize,
    seq: usize,
    past_len: usize,
    head_dim: usize,
    max_seq: usize,
    scale: f32,
) -> Result<(), KernelError> {
    if kv_heads == 0 {
        return Err(KernelError::InvalidShape("kv_heads must be nonzero"));
    }
    if heads == 0 {
        return Err(KernelError::InvalidShape("heads must be nonzero"));
    }
    if heads % kv_heads != 0 {
        return Err(KernelError::InvalidShape(
            "heads must be a multiple of kv_heads",
        ));
    }
    let q_len = shape_len(&[batch, heads, seq, head_dim])?;
    expect_len(q.len(), q_len)?;
    expect_len(out.len(), q_len)?;

    kv_append(
        k_new, v_new, cache, batch, layers, layer, kv_heads, head_dim, max_seq, past_len, seq,
    )?;

    let group = heads / kv_heads;
    let time_stride = shape_len(&[kv_heads, head_dim])?;
    let parity_stride = shape_len(&[max_seq, kv_heads, head_dim])?;
    let layer_stride = shape_len(&[2, max_seq, kv_heads, head_dim])?;
    let batch_stride = shape_len(&[layers, 2, max_seq, kv_heads, head_dim])?;

    for batch_index in 0..batch {
        for head in 0..heads {
            let kv_head = head / group;
            for token in 0..seq {
                let key_count = past_len + token + 1;
                let query_base = ((batch_index * heads + head) * seq + token) * head_dim;
                let query_row = &q[query_base..query_base + head_dim];
                let key_base =
                    batch_index * batch_stride + layer * layer_stride + kv_head * head_dim;
                let value_base = key_base + parity_stride;

                let mut maximum = f32::NEG_INFINITY;
                for key in 0..key_count {
                    let offset = key_base + key * time_stride;
                    let key_row = &cache[offset..offset + head_dim];
                    let score = dot(query_row, key_row) * scale;
                    if score > maximum {
                        maximum = score;
                    }
                }

                let mut denominator = 0.0_f32;
                for key in 0..key_count {
                    let offset = key_base + key * time_stride;
                    let key_row = &cache[offset..offset + head_dim];
                    let score = dot(query_row, key_row) * scale;
                    denominator += (score - maximum).exp();
                }
                let inverse_denominator = 1.0_f32 / denominator;

                let output_row = &mut out[query_base..query_base + head_dim];
                output_row.fill(0.0);
                for key in 0..key_count {
                    let key_offset = key_base + key * time_stride;
                    let key_row = &cache[key_offset..key_offset + head_dim];
                    let score = dot(query_row, key_row) * scale;
                    let probability = (score - maximum).exp() * inverse_denominator;
                    let value_offset = value_base + key * time_stride;
                    let value_row = &cache[value_offset..value_offset + head_dim];
                    for (slot, &value) in output_row.iter_mut().zip(value_row) {
                        *slot += probability * value;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::test_support::{assert_close, Lcg};

    #[derive(Clone, Copy, Debug)]
    struct AttentionShape {
        batch: usize,
        layers: usize,
        layer: usize,
        heads: usize,
        kv_heads: usize,
        seq: usize,
        past_len: usize,
        head_dim: usize,
        max_seq: usize,
    }

    impl AttentionShape {
        fn query_len(self) -> usize {
            self.batch * self.heads * self.seq * self.head_dim
        }

        fn kv_len(self) -> usize {
            self.batch * self.kv_heads * self.seq * self.head_dim
        }

        fn cache_len(self) -> usize {
            self.batch * self.layers * 2 * self.max_seq * self.kv_heads * self.head_dim
        }
    }

    fn oracle_kv_append(
        k: &[f32],
        v: &[f32],
        cache_seed: &[f32],
        shape: AttentionShape,
    ) -> Vec<f64> {
        let mut cache: Vec<f64> = cache_seed.iter().map(|&value| f64::from(value)).collect();
        let time_stride = shape.kv_heads * shape.head_dim;
        let parity_stride = shape.max_seq * time_stride;
        let layer_stride = 2 * parity_stride;
        let batch_stride = shape.layers * layer_stride;
        for batch_index in 0..shape.batch {
            for head in 0..shape.kv_heads {
                for token in 0..shape.seq {
                    let source = ((batch_index * shape.kv_heads + head) * shape.seq + token)
                        * shape.head_dim;
                    let destination = batch_index * batch_stride
                        + shape.layer * layer_stride
                        + (shape.past_len + token) * time_stride
                        + head * shape.head_dim;
                    for column in 0..shape.head_dim {
                        cache[destination + column] = f64::from(k[source + column]);
                        cache[destination + parity_stride + column] = f64::from(v[source + column]);
                    }
                }
            }
        }
        cache
    }

    fn oracle_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        cache_seed: &[f32],
        shape: AttentionShape,
        scale: f32,
    ) -> (Vec<f64>, Vec<f64>) {
        let cache = oracle_kv_append(k, v, cache_seed, shape);
        let time_stride = shape.kv_heads * shape.head_dim;
        let parity_stride = shape.max_seq * time_stride;
        let layer_stride = 2 * parity_stride;
        let batch_stride = shape.layers * layer_stride;
        let group = shape.heads / shape.kv_heads;
        let scale = f64::from(scale);
        let mut out = vec![0.0_f64; shape.query_len()];
        for batch_index in 0..shape.batch {
            for head in 0..shape.heads {
                let kv_head = head / group;
                for token in 0..shape.seq {
                    let key_count = shape.past_len + token + 1;
                    let query_base =
                        ((batch_index * shape.heads + head) * shape.seq + token) * shape.head_dim;
                    let key_base = batch_index * batch_stride
                        + shape.layer * layer_stride
                        + kv_head * shape.head_dim;
                    let value_base = key_base + parity_stride;

                    let mut scores = vec![0.0_f64; key_count];
                    for (key, score) in scores.iter_mut().enumerate() {
                        let mut sum = 0.0_f64;
                        for column in 0..shape.head_dim {
                            sum += f64::from(q[query_base + column])
                                * cache[key_base + key * time_stride + column];
                        }
                        *score = sum * scale;
                    }
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut denominator = 0.0_f64;
                    for score in &scores {
                        denominator += (*score - maximum).exp();
                    }
                    for (key, score) in scores.iter().enumerate() {
                        let probability = (*score - maximum).exp() / denominator;
                        for column in 0..shape.head_dim {
                            out[query_base + column] +=
                                probability * cache[value_base + key * time_stride + column];
                        }
                    }
                }
            }
        }
        (out, cache)
    }

    fn run_kv_append_case(shape: AttentionShape, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut k = vec![0.0_f32; shape.kv_len()];
        let mut v = vec![0.0_f32; shape.kv_len()];
        let mut cache = vec![0.0_f32; shape.cache_len()];
        lcg.fill(&mut k, -1.0, 1.0);
        lcg.fill(&mut v, -1.0, 1.0);
        lcg.fill(&mut cache, -1.0, 1.0);
        let cache_seed = cache.clone();
        let expected = oracle_kv_append(&k, &v, &cache_seed, shape);
        kv_append(
            &k,
            &v,
            &mut cache,
            shape.batch,
            shape.layers,
            shape.layer,
            shape.kv_heads,
            shape.head_dim,
            shape.max_seq,
            shape.past_len,
            shape.seq,
        )
        .expect("kv_append should accept valid shapes");
        assert_close(&cache, &expected, 1e-4, 1e-5);
    }

    fn run_attention_case(shape: AttentionShape, seed: u64) {
        let mut lcg = Lcg::from_seed(seed);
        let mut q = vec![0.0_f32; shape.query_len()];
        let mut k = vec![0.0_f32; shape.kv_len()];
        let mut v = vec![0.0_f32; shape.kv_len()];
        let mut cache = vec![0.0_f32; shape.cache_len()];
        lcg.fill(&mut q, -1.0, 1.0);
        lcg.fill(&mut k, -1.0, 1.0);
        lcg.fill(&mut v, -1.0, 1.0);
        lcg.fill(&mut cache, -1.0, 1.0);
        let cache_seed = cache.clone();
        let mut out = vec![0.0_f32; shape.query_len()];
        let scale = 1.0 / (shape.head_dim as f64).sqrt();
        let scale = scale as f32;

        causal_attention(
            &q,
            &k,
            &v,
            &mut cache,
            &mut out,
            shape.batch,
            shape.layers,
            shape.layer,
            shape.heads,
            shape.kv_heads,
            shape.seq,
            shape.past_len,
            shape.head_dim,
            shape.max_seq,
            scale,
        )
        .expect("causal_attention should accept valid shapes");

        let (expected_out, expected_cache) =
            oracle_attention(&q, &k, &v, &cache_seed, shape, scale);
        assert_close(&out, &expected_out, 1e-4, 1e-5);
        assert_close(&cache, &expected_cache, 1e-4, 1e-5);
    }

    #[test]
    fn kv_append_matches_f64_oracle() {
        for (seed, shape) in [
            (
                1,
                AttentionShape {
                    batch: 1,
                    layers: 1,
                    layer: 0,
                    heads: 1,
                    kv_heads: 1,
                    seq: 2,
                    past_len: 0,
                    head_dim: 2,
                    max_seq: 4,
                },
            ),
            (
                2,
                AttentionShape {
                    batch: 2,
                    layers: 2,
                    layer: 1,
                    heads: 2,
                    kv_heads: 2,
                    seq: 3,
                    past_len: 2,
                    head_dim: 3,
                    max_seq: 5,
                },
            ),
            (
                3,
                AttentionShape {
                    batch: 1,
                    layers: 1,
                    layer: 0,
                    heads: 2,
                    kv_heads: 2,
                    seq: 1,
                    past_len: 0,
                    head_dim: 8,
                    max_seq: 1,
                },
            ),
            (
                4,
                AttentionShape {
                    batch: 1,
                    layers: 1,
                    layer: 0,
                    heads: 1,
                    kv_heads: 1,
                    seq: 0,
                    past_len: 2,
                    head_dim: 4,
                    max_seq: 4,
                },
            ),
        ] {
            run_kv_append_case(shape, seed);
        }
    }

    #[test]
    fn kv_append_rejects_invalid_shapes() {
        let shape = AttentionShape {
            batch: 1,
            layers: 1,
            layer: 0,
            heads: 1,
            kv_heads: 1,
            seq: 2,
            past_len: 1,
            head_dim: 4,
            max_seq: 3,
        };
        let k = vec![0.0_f32; shape.kv_len()];
        let v = vec![0.0_f32; shape.kv_len()];
        let mut cache = vec![0.0_f32; shape.cache_len()];

        let error = kv_append(&k[..k.len() - 1], &v, &mut cache, 1, 1, 0, 1, 4, 3, 1, 2)
            .expect_err("short k must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = kv_append(&k, &v[..v.len() - 1], &mut cache, 1, 1, 0, 1, 4, 3, 1, 2)
            .expect_err("short v must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let cache_len = cache.len();
        let error = kv_append(&k, &v, &mut cache[..cache_len - 1], 1, 1, 0, 1, 4, 3, 1, 2)
            .expect_err("short cache must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let mut full_cache = vec![0.0_f32; shape.cache_len()];
        let error = kv_append(&k, &v, &mut full_cache, 1, 1, 1, 1, 4, 3, 1, 2)
            .expect_err("out-of-range layer must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = kv_append(&k, &v, &mut full_cache, 1, 1, 0, 1, 4, 3, 2, 2)
            .expect_err("past_len + seq beyond max_seq must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));
    }

    #[test]
    fn kv_append_writes_documented_layout() {
        let k = [1.0_f32, 2.0];
        let v = [3.0_f32, 4.0];
        // [batch=1, layers=2, 2, max_seq=3, kv_heads=1, head_dim=2].
        let mut cache = vec![0.0_f32; 24];
        kv_append(&k, &v, &mut cache, 1, 2, 1, 1, 2, 3, 1, 1)
            .expect("kv_append should accept the documented layout");

        // [batch=0, layer=1, parity=0, t=1, head=0, :] and the value slot
        // parity=1 at the same coordinates.
        assert_eq!(cache[14..16], [1.0, 2.0]);
        assert_eq!(cache[20..22], [3.0, 4.0]);
        for (index, &value) in cache.iter().enumerate() {
            if !matches!(index, 14 | 15 | 20 | 21) {
                assert_eq!(value, 0.0, "unexpected write at index {index}");
            }
        }
    }

    #[test]
    fn causal_attention_matches_hand_computed_decode() {
        let q = [1.0_f32, 0.0];
        let k = [1.0_f32, 0.0];
        let v = [3.0_f32, 4.0];
        // [batch=1, layers=1, 2, max_seq=1, kv_heads=1, head_dim=2].
        let mut cache = [0.0_f32; 4];
        let mut out = [0.0_f32; 2];
        causal_attention(
            &q, &k, &v, &mut cache, &mut out, 1, 1, 0, 1, 1, 1, 0, 2, 1, 1.0,
        )
        .expect("single-token attention should succeed");
        assert_close(&out, &[3.0, 4.0], 1e-4, 1e-5);
    }

    #[test]
    fn causal_attention_matches_f64_oracle() {
        for (seed, shape) in [
            (
                11,
                AttentionShape {
                    batch: 1,
                    layers: 1,
                    layer: 0,
                    heads: 2,
                    kv_heads: 2,
                    seq: 3,
                    past_len: 0,
                    head_dim: 4,
                    max_seq: 4,
                },
            ),
            (
                12,
                AttentionShape {
                    batch: 2,
                    layers: 1,
                    layer: 0,
                    heads: 2,
                    kv_heads: 2,
                    seq: 1,
                    past_len: 3,
                    head_dim: 8,
                    max_seq: 6,
                },
            ),
            (
                13,
                AttentionShape {
                    batch: 2,
                    layers: 2,
                    layer: 1,
                    heads: 4,
                    kv_heads: 2,
                    seq: 2,
                    past_len: 2,
                    head_dim: 8,
                    max_seq: 5,
                },
            ),
            (
                14,
                AttentionShape {
                    batch: 2,
                    layers: 1,
                    layer: 0,
                    heads: 2,
                    kv_heads: 1,
                    seq: 1,
                    past_len: 5,
                    head_dim: 2,
                    max_seq: 7,
                },
            ),
            (
                15,
                AttentionShape {
                    batch: 1,
                    layers: 1,
                    layer: 0,
                    heads: 6,
                    kv_heads: 3,
                    seq: 2,
                    past_len: 1,
                    head_dim: 4,
                    max_seq: 4,
                },
            ),
            (
                16,
                AttentionShape {
                    batch: 1,
                    layers: 1,
                    layer: 0,
                    heads: 3,
                    kv_heads: 1,
                    seq: 1,
                    past_len: 2,
                    head_dim: 4,
                    max_seq: 3,
                },
            ),
        ] {
            run_attention_case(shape, seed);
        }
    }

    #[test]
    fn causal_attention_accepts_empty_sequence() {
        let shape = AttentionShape {
            batch: 1,
            layers: 1,
            layer: 0,
            heads: 1,
            kv_heads: 1,
            seq: 0,
            past_len: 2,
            head_dim: 4,
            max_seq: 4,
        };
        let mut lcg = Lcg::from_seed(21);
        let mut cache = vec![0.0_f32; shape.cache_len()];
        lcg.fill(&mut cache, -1.0, 1.0);
        let cache_seed = cache.clone();
        let mut out = vec![0.0_f32; shape.query_len()];
        causal_attention(
            &[],
            &[],
            &[],
            &mut cache,
            &mut out,
            shape.batch,
            shape.layers,
            shape.layer,
            shape.heads,
            shape.kv_heads,
            shape.seq,
            shape.past_len,
            shape.head_dim,
            shape.max_seq,
            0.5,
        )
        .expect("empty sequence should be accepted");
        let expected: Vec<f64> = cache_seed.iter().map(|&value| f64::from(value)).collect();
        assert_close(&cache, &expected, 1e-4, 1e-5);
        assert!(out.is_empty());
    }

    #[test]
    fn causal_attention_rejects_invalid_shapes() {
        let shape = AttentionShape {
            batch: 1,
            layers: 1,
            layer: 0,
            heads: 2,
            kv_heads: 2,
            seq: 2,
            past_len: 1,
            head_dim: 4,
            max_seq: 3,
        };
        let q = vec![0.0_f32; shape.query_len()];
        let k = vec![0.0_f32; shape.kv_len()];
        let v = vec![0.0_f32; shape.kv_len()];
        let mut cache = vec![0.0_f32; shape.cache_len()];
        let mut out = vec![0.0_f32; shape.query_len()];
        let out_len = out.len();
        let cache_len = cache.len();

        let error = causal_attention(
            &q, &k, &v, &mut cache, &mut out, 1, 1, 0, 3, 2, 2, 1, 4, 3, 0.5,
        )
        .expect_err("heads not divisible by kv_heads must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = causal_attention(
            &q, &k, &v, &mut cache, &mut out, 1, 1, 0, 0, 0, 2, 1, 4, 3, 0.5,
        )
        .expect_err("zero kv_heads must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = causal_attention(
            &q, &k, &v, &mut cache, &mut out, 1, 1, 0, 0, 1, 2, 1, 4, 3, 0.5,
        )
        .expect_err("zero heads must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = causal_attention(
            &q, &k, &v, &mut cache, &mut out, 1, 1, 1, 2, 2, 2, 1, 4, 3, 0.5,
        )
        .expect_err("out-of-range layer must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = causal_attention(
            &q, &k, &v, &mut cache, &mut out, 1, 1, 0, 2, 2, 2, 2, 4, 3, 0.5,
        )
        .expect_err("past_len + seq beyond max_seq must be rejected");
        assert!(matches!(error, KernelError::InvalidShape(_)));

        let error = causal_attention(
            &q[..q.len() - 1],
            &k,
            &v,
            &mut cache,
            &mut out,
            1,
            1,
            0,
            2,
            2,
            2,
            1,
            4,
            3,
            0.5,
        )
        .expect_err("short q must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = causal_attention(
            &q,
            &k,
            &v,
            &mut cache,
            &mut out[..out_len - 1],
            1,
            1,
            0,
            2,
            2,
            2,
            1,
            4,
            3,
            0.5,
        )
        .expect_err("short out must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));

        let error = causal_attention(
            &q,
            &k,
            &v,
            &mut cache[..cache_len - 1],
            &mut out,
            1,
            1,
            0,
            2,
            2,
            2,
            1,
            4,
            3,
            0.5,
        )
        .expect_err("short cache must be rejected");
        assert!(matches!(error, KernelError::LengthMismatch { .. }));
    }
}
