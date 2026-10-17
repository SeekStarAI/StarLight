# backend-cpu

StarLight 的纯 CPU 算力后端。后续目标是以 Rust `core::arch` 提供 AVX2、AVX-512 和 ARM NEON 算子，并实现 `sl-device` 的 VDI 契约。

当前 crate 提供的是 `src/reference/` 下的**标量参考实现**。它是后续 SIMD 内核和 `tests/golden-baselines/` 的数值 oracle，不包含任何 SIMD 路径。

> **不得用于性能评估。** 这些内核以可读性和确定性为第一目标，循环顺序、重复点积和三遍 attention 扫描都是刻意选择的。任何吞吐、延迟或内存带宽结论都必须来自后续 SIMD 实现，而不是这里的标量代码。

## 标量参考实现

- 仅使用 `f32` 数据，所有张量按**行主序连续**（C-contiguous，最后一维连续）解释。
- **无 stride、无 bias、无广播**。唯一的批量前缀是 matmul 的 `batch` 维；GQA 的 KV head 映射是显式索引，不是广播。
- 内核函数只接收 `&[f32]` / `&mut [f32]` 和显式形状参数，返回 `Result<(), KernelError>`，**不做堆分配**，也不返回 `Vec`。
- `lib.rs` 启用 `#![forbid(unsafe_code)]`，crate 不新增依赖。
- 所有公开函数在入口校验长度与形状；非法输入返回 `KernelError`，不使用 `assert!` 做输入校验，不 panic。
- 没有别名输入/输出：Rust 借用规则已经保证 `&[f32]` 与 `&mut [f32]` 不重叠。

### `KernelError`

```rust
pub enum KernelError {
    LengthMismatch { expected: usize, actual: usize },
    ShapeOverflow,
    InvalidShape(&'static str),
    Unsupported(&'static str),
}
```

该枚举是 `Copy + Debug + Display + core::error::Error`，不携带堆分配的字符串。

### 内核入口

| 模块 | 函数 | 形状 / 语义 |
| --- | --- | --- |
| `reference::matmul` | `matmul_nt(a, b, out, batch, m, k, n)` | `a[batch,m,k] · b[batch,n,k]ᵀ -> out[batch,m,n]` |
| `reference::matmul` | `matmul_nn(a, b, out, batch, m, k, n)` | `a[batch,m,k] · b[batch,k,n] -> out[batch,m,n]` |
| `reference::softmax` | `softmax(x, rows, cols)` | 逐行原地数值稳定 softmax |
| `reference::softmax` | `softmax_causal(x, rows, cols, seq, past_len)` | 逐行因果 mask 后原地 softmax |
| `reference::norm` | `rms_norm(x, weight, out, rows, hidden, eps)` | `out = x * rsqrt(mean(x²)+eps) * weight` |
| `reference::rope` | `rope_neox(x, positions, batch, heads, seq, head_dim, theta)` | NeoX split-half RoPE，原地 |
| `reference::rope` | `rope_gptj(x, positions, batch, heads, seq, head_dim, theta)` | GPT-J 交错 RoPE，原地 |
| `reference::elementwise` | `silu(x)`、`scale(x, factor)` | 原地逐元素 |
| `reference::elementwise` | `mul(a, b, out, len)`、`add(a, b, out, len)` | 逐元素二元运算 |
| `reference::layout` | `gather(table, ids, out, vocab, hidden, rows)` | 按 id 拷贝 embedding 行 |
| `reference::layout` | `last_token(input, out, batch, seq, hidden, seq_len)` | 每批取第 `seq_len-1` 个 token |
| `reference::layout` | `split_heads(input, out, batch, seq, heads, head_dim)` | `[B,S,H·D] -> [B,H,S,D]` |
| `reference::layout` | `merge_heads(input, out, batch, seq, heads, head_dim)` | `split_heads` 的逆变换 |
| `reference::attention` | `kv_append(k, v, cache, batch, layers, layer, kv_heads, head_dim, max_seq, past_len, seq)` | 写入 KV cache |
| `reference::attention` | `causal_attention(q, k_new, v_new, cache, out, batch, layers, layer, heads, kv_heads, seq, past_len, head_dim, max_seq, scale)` | KV 写入 + GQA causal attention |
| `reference::swiglu` | `swiglu(x, w_gate, w_up, w_down, out, rows, hidden, inter)` | 无临时缓冲的 SwiGLU FFN |

`reference/mod.rs` 同时重新导出以上函数。

## 确定性累加顺序

跨平台可复现是 reference 的硬性要求，因此固定以下顺序，且**不使用** FMA、`mul_add`、fast-math 或并行归约：

- `matmul_nt` / `matmul_nn`：batch → 输出行 `m` → 输出列 `n` → 规约维 `k` 从 0 递增。
- `softmax` / `softmax_causal`：每行从左到右求和。
- `rms_norm`：每行平方和从左到右累加，`rsqrt` 只计算一次，然后对每个元素只做一次缩放。
- `causal_attention`：对每个 `(b, h, t)`，key 下标 `j` 从 0 到 `past_len + t` 递增；先求最大 score，再求指数和，最后累加 value。
- `swiglu`：每行按中间维 `i` 从 0 递增累加输出。
- RoPE 的角度、sin、cos 用 `f64` 计算，最后一步才转成 `f32`。

RoPE / attention 的测试容差为绝对 `1e-4`；其余内核使用“绝对 `1e-4` 或相对 `1e-5`，取较宽松者”。测试中的 oracle 是独立的 `f64` 朴素实现。

## KV cache 布局与 GQA 约定

KV cache 的完整形状为：

```text
cache[batch, layers, 2, max_seq, kv_heads, head_dim]
```

索引 `[b, layer, 0, t, h, d]` 存 key，`[b, layer, 1, t, h, d]` 存 value。`layer` 是显式参数，因此同一份缓存可以服务所有 transformer 层；`kv_append` 只写 `past_len .. past_len + seq`，超出 `max_seq` 或 `layer >= layers` 会在写入前返回 `InvalidShape`。

`causal_attention` 先调用 `kv_append` 写入当前 chunk，再计算：

```text
group   = heads / kv_heads
kv_head = h / group
score[j] = scale * dot(q[b,h,t,:], cache[b,layer,0,j,kv_head,:])
out[b,h,t,:] = Σ_{j=0}^{past_len+t} softmax(score)[j] * cache[b,layer,1,j,kv_head,:]
```

- 要求 `heads % kv_heads == 0`、`heads > 0` 且 `kv_heads > 0`；`heads == kv_heads` 是普通 MHA，`heads == 2 * kv_heads` 是 2 倍 GQA，`kv_heads == 1` 是 MQA。
- `scale` 由调用方传入，通常是 `1 / sqrt(head_dim)`。
- 注意力用三遍扫描（max、denominator、加权 value）实现，不保存 score 向量；`head_dim` 是运行期值，不使用变长栈数组。
- `seq == 0` 的 attention 是合法空操作（只做形状校验和可选的空 KV 写入）。

## 边界与语义选择

- `matmul`：`batch == 0` 或 `m/n/k == 0` 合法；`k == 0` 时输出全零。
- `softmax` / `softmax_causal`：`rows == 0` 或 `cols == 0` 返回 `InvalidShape`；`softmax_causal` 额外要求 `cols == past_len + seq`、`rows % seq == 0`、`seq > 0`。mask 掉的列最终为 0，归一化只覆盖可见列。
- `rms_norm`：`hidden == 0` 返回 `InvalidShape`；`rows == 0` 合法。
- `rope_*`：`head_dim` 必须为正偶数；`positions` 为 `u32`，形状 `[batch, seq]`。
- `gather`：id 按 `u32` 接收，必须小于 `vocab`，否则返回 `InvalidShape("id out of range")`；所有 id 在写入前统一校验。
- `last_token`：要求 `1 <= seq_len <= seq`。
- `swiglu`：`hidden` 与 `inter` 都必须非零；`rows == 0` 合法。
- 不做 paged KV、变长 batch、量化权重、FP16/BF16、融合 Pass 或 bias；这些属于后续能力。
- 调用方应保证输入为有限值；参考内核不额外做 NaN/Inf 清洗。

## 验证

```sh
cargo test -p starlight-backend-cpu
cargo clippy -p starlight-backend-cpu --all-targets -- -D warnings
```

测试使用自带的确定性 LCG（`struct Lcg(u64)`，不引入 `rand`）生成输入，并为每个内核维护独立的 `f64` oracle。覆盖 MHA / GQA / MQA、`seq == 1` 的 decode、prefill、零批量、零尺寸、`head_dim` 为 2/4/8、非 2 的幂尺寸以及错误长度/形状。
