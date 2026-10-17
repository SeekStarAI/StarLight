# backend-cpu

StarLight 的纯 CPU 算力后端。后续目标是以 Rust `core::arch` 提供 AVX2、AVX-512 和 ARM NEON 算子，并实现 `sl-device` 的 VDI 契约。

当前 crate 提供 `src/reference/` 下的**标量参考实现**，以及一个把 `sl_ir::CommandBuffer` 顺序跑起来的**参考执行器**（`src/executor.rs`）。参考内核是后续 SIMD 内核和 `tests/golden-baselines/` 的数值 oracle，不包含任何 SIMD 路径。

> **不得用于性能评估。** 这些内核以可读性和确定性为第一目标，循环顺序、重复点积和三遍 attention 扫描都是刻意选择的。任何吞吐、延迟或内存带宽结论都必须来自后续 SIMD 实现，而不是这里的标量代码。

## 标量参考实现

- 仅使用 `f32` 数据，所有张量按**行主序连续**（C-contiguous，最后一维连续）解释。
- **无 stride、无 bias**。matmul 的前缀维支持 equal-or-one 广播；GQA 的 KV head 映射是显式索引，不走广播。
- 内核函数只接收 `&[f32]` / `&mut [f32]` 和显式形状参数，返回 `Result<(), KernelError>`，**不做堆分配**，也不返回 `Vec`。
- `lib.rs` 启用 `#![forbid(unsafe_code)]`。内核本身不依赖第三方 crate；`executor` 额外使用 workspace 内的 `sl-ir`。
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
| `reference::fused` | `rms_norm_matmul(x, weight, w, out, rows, hidden, out_features, eps)` | `y = (x * rsqrt(mean(x²)+eps) * weight) · wᵀ`，无临时缓冲 |

`reference/mod.rs` 同时重新导出以上函数。

## 确定性累加顺序

跨平台可复现是 reference 的硬性要求，因此固定以下顺序，且**不使用** FMA、`mul_add`、fast-math 或并行归约：

- `matmul_nt` / `matmul_nn`：batch → 输出行 `m` → 输出列 `n` → 规约维 `k` 从 0 递增。
- `softmax` / `softmax_causal`：每行从左到右求和。
- `rms_norm`：每行平方和从左到右累加，`rsqrt` 只计算一次，然后对每个元素只做一次缩放。
- `causal_attention`：对每个 `(b, h, t)`，key 下标 `j` 从 0 到 `past_len + t` 递增；先求最大 score，再求指数和，最后累加 value。
- `swiglu`：每行按中间维 `i` 从 0 递增累加输出。
- `rms_norm_matmul`：每行只计算一次 `inverse_rms`；输出列 `j` 的规约维 `i` 从 0 递增，累加顺序为 `(x[i] * inverse_rms * weight[i]) * w[j, i]`。
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
- `rms_norm_matmul`：`hidden` 与 `out_features` 都必须非零；`rows == 0` 合法。
- 不做 paged KV、变长 batch、量化权重、FP16/BF16、融合 Pass 或 bias；这些属于后续能力。
- 调用方应保证输入为有限值；参考内核不额外做 NaN/Inf 清洗。

## 参考执行器（`executor`）

`src/executor.rs` 是一个把 `sl_ir::CommandBuffer` 真正跑起来的**参考执行器**。它按命令顺序调用 `reference::*` 内核，用于端到端验证 IR、内存计划与内核语义的配合。

> **这是参考执行器，不是设备后端。** 它同步运行在 host 上，不做图优化、算子融合、队列提交、异步完成或设备内存管理；性能路径属于后续 SIMD / Vulkan 后端。

### 入口

```rust
pub fn execute(
    ir: &CommandBuffer,
    bindings: &BindingTable,
    buffers: &mut BufferStore,
    scalars: &ScalarStore,
) -> Result<(), ExecutionError>;
```

### 三张 build 阶段表

- `BindingTable`：`ValueId -> BufferId`。供 `Param`、`Input`、`Output`、`State` 使用。`Scalar` 与 `Temp` 不进入该表。
- `BufferStore`：`BufferId -> Vec<f32>`（`insert` / `get`）。也提供 `insert_u32` / `get_u32`，用于 `positions`、embedding `ids` 这类 `U32` 张量。相同 id 再次插入会替换旧数据。
- `ScalarStore`：`ValueId -> u32`，供 `ValueRole::Scalar` 使用。`seq_len`、`past_len` 按命令的端口顺序读取。

所有表均在 build 阶段填充；`execute` 只更新缓冲区内容，不插入新表项。

### scratchpad 约定

- `Temp` 不拥有独立缓冲区。所有 `Temp` 共享绑定到 `SCRATCHPAD`（`pub const SCRATCHPAD: BufferId = BufferId(0)`）的一个 `f32` 缓冲区。
- `ir.plan.slots` 的 `offset` 与 `bytes` 是**字节**，执行器要求二者都是 4 的倍数，并换算成 `f32` 下标。
- 每个 slot 的 `bytes` 必须等于对应 `Value::byte_size()`；`offset + bytes` 必须同时位于 `plan.scratchpad_bytes` 与实际 scratchpad 缓冲区之内。
- `BufferId(0)` 专用于 scratchpad；其他角色的 binding 不得指向它。
- 无 `Temp` 时可以不提供 scratchpad；有 `Temp` 时缺失会返回 `MissingBuffer(SCRATCHPAD)`。

### 执行顺序与借用

- 严格按 `ir.commands` 顺序执行；`deps` 被忽略（v1 命令流是全序）。
- 每个命令先解析并校验端口、形状、binding、scalar 与 plan，再写入任何输出。
- 借用管理采用 **take/put**：执行命令前把输入、输出与 state 缓冲区从 `BufferStore` 中 `take` 出来，执行完放回；scratchpad 取出后用 `split_at_mut` 切成互不重叠的 `Temp` 切片。执行路径不使用 `unsafe`，也不创建 `Vec` / `String` / `format!`。
- 同一命令的多个端口不得指向同一个 value 或同一个 buffer（与 ARCH-0004「禁止原地写入」/「不做别名」一致）；违反时返回 `InvalidPlan`。

### 算子与形状

支持 v1 的全部 `OpCode`：`MatMulNT`、`MatMulNN`、`Softmax`、`SoftmaxCausal`、`RmsNorm`、`Rope`、`Silu`、`Mul`、`Add`、`Scale`、`Gather`、`LastToken`、`KvAppend`、`SplitHeads`、`MergeHeads`、`CausalAttention`、`SwiGlu`、`RmsNormMatMul`。关键形状规则：

- **MatMulNT / MatMulNN**：取最后两维得到 `m`、`k`、`n`，其余前缀维相乘得到各自的 `batch`。前缀维按 equal-or-one 广播：2D 权重 `[N, K]` 可以直接乘 3D 激活 `[B, M, K]`；输出前缀取两者中较大的那个，任一为 0 时结果为 0。
- **Softmax / SoftmaxCausal**：把最后一维之外的维度相乘得到 `rows`，最后一维是 `cols`。
- **SoftmaxCausal**：端口 1、2 依次是 `seq_len`、`past_len`；内核额外要求 `cols == past_len + seq_len` 且 `rows % seq_len == 0`。
- **Rope**：rank 4 输入按 `[batch, heads, seq, head_dim]`，rank 3 输入按 `[batch, seq, head_dim]`（`heads = 1`）；`positions` 为 `[batch, seq]`。
- **Gather**：`table` 为 `[vocab, hidden]`，`ids` 的每个维度作为输出前缀，最后一维为 `hidden`。
- **LastToken**：`[B, M, H] -> [B, H]`；端口 1 是 `seq_len`。
- **KvAppend**：`K [B, Hkv, M, D]`、`V` 同形；`seq = M`，端口 2 是 `past_len`；`layer` 取自 `Op::layer()`。
- **CausalAttention**：`Q/B/Hkv/V` 必须是 `[B, Hq, M, D]` 与 `[B, Hkv, M, D]`；`seq = M`，端口 3 的单个 scalar 解释为 `past_len`；`layer` 取自 `Op::layer()`，`scale` 取自 attr。
- **SplitHeads / MergeHeads**：`[B, M, H·D] <-> [B, H, M, D]`，`heads * head_dim` 必须等于通道数。
- **SwiGlu / RmsNormMatMul**：把最后一维之外的维度相乘得到 `rows`，最后一维是 `hidden`；权重形状分别为 `[inter, hidden]`、`[hidden, inter]` 与 `[out_features, hidden]`。
- 形状不满足内核契约时返回 `ExecutionError::Kernel(KernelError::InvalidShape(..))`。

### state 解析

- `KvAppend` 与 `CausalAttention` 必须带 `Effect::WriteState(state_value)`。
- state 的 buffer 由 `BindingTable` 解析；`layers`、`max_seq`、`kv_heads`、`head_dim` 取自 `ir.states` 中的 `StateKind::KvCache`，并与算子输入形状交叉校验。
- 其他算子携带 state effect，或 stateful 算子缺少 `WriteState`，都会返回 `InvalidPlan`。

### `U32` 张量

- 推荐用 `BufferStore::insert_u32` 写入精确的 `u32` 数据。
- 兼容路径：如果某个 `u32` 端口只绑定了 `f32` 缓冲区，执行器会把它解释为非负整数，并复用 build 阶段预分配的转换向量完成转换；同一命令最多一个这样的张量。非整数、负数或超出 `u32` 的值返回 `InvalidShape`。转换向量最多预分配 `2^20` 个元素；更大的整数张量请使用 `insert_u32`，否则返回 `InvalidPlan`。

### 错误类型

```rust
pub enum ExecutionError {
    UnsupportedOp(OpCode),
    MissingBinding(ValueId),
    MissingBuffer(BufferId),
    MissingScalar(ValueId),
    InvalidPlan(&'static str),
    Kernel(KernelError),
}
```

`InvalidPlan` 用于 scratchpad 越界/尺寸不匹配、端口别名、缺少 state 声明等 IR/plan 级问题；`Kernel` 包装参考内核返回的长度/形状错误。执行器不 panic。

### 边界

- 只接受全具体形状（`CommandBuffer` 本就不允许 `Dim::Sym`）；rank 上限由算子签名与执行器的固定数组决定。
- 不支持输入/输出原地别名、paged KV、变长 batch、FP16/BF16。
- 空张量合法：`MatMul` 的 `batch == 0`、`RmsNorm` / `SwiGLU` / `RmsNormMatMul` 的 `rows == 0` 都会正常执行；`Softmax` 的 `rows == 0` 由内核按契约返回 `InvalidShape`，不会 panic。

## `CpuDevice`（`sl-device` 实现）

`src/device.rs` 提供 `CpuDevice`，实现 `sl_device::Device`：

- 缓冲区：`Vec<Option<Vec<f32>>>` / `Vec<Option<Vec<u32>>>`；
- 提供分配、上传、下载、同步；
- `dispatch` 按 `Op` 分发到 `reference::*`，覆盖全部 18 个算子；
- `BufferRef.offset` 是元素偏移，按 `input_values` / `output_values` 的形状切片；
- 执行路径无 `unsafe`、无堆分配；
- `executor` 保留为 IR 级参考执行器；`CpuDevice` 是设备契约实现，两者共享同一组 reference 内核。

`starlight-runtime` 通过 `CpuDevice::new()` 使用它完成端到端 fixture 执行。

## 验证

```sh
cargo test -p starlight-backend-cpu
cargo clippy -p starlight-backend-cpu --all-targets -- -D warnings
cargo fmt --check
```

测试使用自带的确定性 LCG（`struct Lcg(u64)`，不引入 `rand`）生成输入，并为每个内核维护独立的 `f64` oracle。覆盖 MHA / GQA / MQA、`seq == 1` 的 decode、prefill、零批量、零尺寸、`head_dim` 为 2/4/8、非 2 的幂尺寸以及错误长度/形状。执行器测试使用 `sl-ir` 手工构造 `CommandBuffer`（含 `MemoryPlan` / slots / `StateDecl`），覆盖最小图、prefill + decode、scratchpad 复用、全部算子、错误路径与空张量。
