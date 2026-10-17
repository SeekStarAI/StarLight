# StarLight 模型家族 Lowering 与 LoweredModel 契约

**文档编号：** ARCH-0006
**状态：** 设计提案，待首个模型家族验证
**范围：** 模型家族 Adapter 的输入输出、`LoweredModel` 结构、参数角色命名、prefill/decode 切分、v1 支持矩阵与验收
**前置文档：** [ARCH-0002](<ARCH-0002: 模块边界与发行版业务管线契约.md>)、[ARCH-0003](<ARCH-0003: 模型制品与多模态推理数据流设计.md>)、[ARCH-0004](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)、[RFC-0002](<RFC-0002: 采纳 sl-ir 作为图 IR 与设备命令 ABI.md>)

本文定义 `sl-model-adapters` 的交付物 `LoweredModel`，即"模型包 + 架构配置"经过家族 lowering 之后、交给 `sl-graph` 编译之前的数据形态。文中类型已由 `sl-model-adapters` 的 `llama` / `dense` 首个家族落地；`ProgramIo`、`ParamRef` 与 `WeightSpec.scales` 等细节以实现为准。

## 0. 定位

三份文档的分工：

| 文档 | 回答的问题 |
| --- | --- |
| ARCH-0003 | 模型文件如何被读取、权重如何表示、请求如何流动 |
| **ARCH-0006（本文）** | 某个模型家族如何变成 `Program` + 权重绑定，以及这份产物长什么样 |
| ARCH-0004 | `Program` 与 `CommandBuffer` 的类型、算子、能力与内存契约 |

`LoweredModel` 不是可执行图，也不绑定设备。它是**家族语义的中间产物**：说清楚有哪些参数、哪些状态、prefill 与 decode 各是什么程序。

## 1. 设计结论

1. `LoweredModel` 由 Adapter 产出，包含 **prefill 与 decode 两个 `Program`** 和**一张共享权重表**。
2. 参数用**角色路径**标识（如 `layers.3.attn.q`），而不是源 tensor 名；源名只在 `WeightSpec.tensor` 中出现。
3. 架构标识与配置版本必须显式，不得从 tensor 名猜测。
4. v1 只支持无 bias 的 decoder-only dense 架构，权重为 F32 行主序；量化权重与设备布局推迟。
5. 支持矩阵按“容器 + 家族 + 精度 + 后端 + 模态 + 特性”组合记录，不用“支持某格式”概括。
6. 未支持的变体、缺失或多余张量、形状不符，必须返回可诊断错误，不得静默降级。

## 2. 输入

Adapter 的输入是：

```rust
pub struct LoweringInput<'a> {
    pub bundle: &'a ModelBundle,
    pub architecture: ArchitectureId,
    pub config: &'a serde_json::Value,
    pub limits: ShapeSpec,
}
```

要求：

- `ModelBundle` 来自 `sl-model`，含显式 architecture 与 config；`sl-model` 不推断架构，Adapter 也不从 tensor 名推断。
- `config` 是模型自带的架构配置（如 Hugging Face `config.json`）。Adapter 负责把它解析为**类型化配置**并校验必需字段。
- `limits` 给出 `batch` 与 `seq` 的上界，用于确定 `Program` 中符号维度的声明。
- 未知 config 字段允许存在，但必须被记录到诊断信息里，便于排查“看起来支持、其实忽略”的情况。

## 2.1 仓库内布局

`sl-model-adapters` 按“家族 / 版本”组织：

```text
src/
├── error.rs              # 跨家族共享
├── model.rs              # LoweredModel / WeightSpec / ProgramIo
└── families/
    ├── mod.rs
    └── llama/
        ├── mod.rs        # 家族元数据与版本分发
        └── v1/           # 配置 schema v1 + lowering
```

家族特有的张量名、配置字段与建图逻辑留在 `families/<family>/<version>/`；只有跨家族契约放在 crate 根。新增家族是**新增目录**，不修改已有家族；新增版本是同一家族下的新版本目录。

## 3. 输出

```rust
pub struct ArchitectureId {
    /// 家族名，例如 "llama"。
    pub family: &'static str,
    /// 变体名，例如 "qwen2"；无变体时为 None。
    pub variant: Option<&'static str>,
    /// Adapter 支持的配置 schema 版本。
    pub config_version: u32,
}

pub struct LoweredModel {
    pub ir_version: u32,
    pub architecture: ArchitectureId,
    pub programs: Programs,
    pub weights: Vec<WeightSpec>,
    pub states: Vec<StateSpec>,
    pub inputs: Vec<IoSpec>,
    pub outputs: Vec<IoSpec>,
}

pub struct Programs {
    /// 一次处理 N 个 token 的 prefill 程序。
    pub prefill: Program,
    /// 每步处理 1 个 token 的 decode 程序。
    pub decode: Program,
}

pub struct WeightSpec {
    /// 稳定角色路径，例如 "layers.0.attn.q"。
    pub role: String,
    /// 源张量名，例如 "model.layers.0.self_attn.q_proj.weight"。
    pub tensor: String,
    /// 逻辑形状，例如 [n_heads * head_dim, hidden]。
    pub shape: Vec<u32>,
    pub dtype: DType,
    /// v1 只允许 DenseF32RowMajor。
    pub layout: WeightLayout,
}

pub enum WeightLayout {
    /// 行主序、连续、64 字节对齐。
    DenseF32RowMajor,
}

pub struct StateSpec {
    pub name: &'static str,
    pub value: ValueId,
    pub kind: StateKind,
}

pub struct IoSpec {
    pub name: &'static str,
    pub value: ValueId,
    pub role: ValueRole,
}
```

说明：

- `ValueId` 指向对应 `Program` 的 value 表。`prefill` 与 `decode` 各自有独立 value 表，因此 `StateSpec`/`IoSpec` 中的 id 必须在两个程序中分别解析；实现时应以 `(program, value)` 成对存储，或约定 id 在两个程序中一致。
- `WeightSpec` 不含 `ValueId`：同一个权重会在两个程序里各有一个 `Param` value，物理缓冲只有一份，由 `sl-core` 在 `ParamBinding` 阶段合并。
- 这里的 `ParamBinding` 是 `sl-ir` 中的"value → buffer"设备绑定，不要与本文的 `WeightSpec` 混淆。

## 3.1 与 `sl-ir` 的编码映射

Adapter 必须把 `sl-model` 的 `TensorEncoding` 显式映射为 `sl-ir::TensorEncoding`：

| 源编码 | 目标 `Value` |
| --- | --- |
| `Dense` F32/F16/BF16 | `Value::new(dtype, logical_shape, Param)` |
| `GroupwiseQuantized { Int4/Int8, group_size }` | `Value::with_encoding(F32, logical_shape, Param, GroupwiseQuantized { .. })` |
| `OpaqueQuantized` | 拒绝：`UnsupportedEncoding`，不得静默当作 dense |

量化参数的 scale 张量是独立的 `Param`（例如 `<role>.__scales`），不在 `TensorEncoding` 中隐式携带。`sl-core` 当前只执行稠密参数，遇到非稠密 `Param` 会在 `Engine::build` 返回 `UnsupportedEncoding`；因此量化的可执行支持必须先由 backend 提供量化算子，再把该类组合加入认证矩阵。

## 4. 参数角色命名

角色路径是 Adapter 与 `sl-core` 之间唯一的稳定标识，必须满足：

1. **与源 tensor 名解耦**：换一个 checkpoint 命名风格，角色路径不变。
2. **可机械生成**：`layers.{i}.…` 中的 `{i}` 从 0 开始，连续无空洞。
3. **与算子语义对应**：角色描述"这个权重在计算里扮演什么"，而不是"它叫什么"。

v1 的规范角色集合：

| 角色 | 形状 | 说明 |
| --- | --- | --- |
| `embed` | `[vocab, hidden]` | 词嵌入表 |
| `final_norm` | `[hidden]` | 最终 RMSNorm 权重 |
| `lm_head` | `[vocab, hidden]` | 输出投影；`tie_word_embeddings` 时与 `embed` 共享缓冲 |
| `layers.{i}.attn_norm` | `[hidden]` | 注意力前归一化 |
| `layers.{i}.attn.q` | `[heads * head_dim, hidden]` | 查询投影 |
| `layers.{i}.attn.k` | `[kv_heads * head_dim, hidden]` | 键投影 |
| `layers.{i}.attn.v` | `[kv_heads * head_dim, hidden]` | 值投影 |
| `layers.{i}.attn.o` | `[hidden, heads * head_dim]` | 输出投影 |
| `layers.{i}.mlp_norm` | `[hidden]` | MLP 前归一化 |
| `layers.{i}.mlp.gate` | `[inter, hidden]` | SwiGLU 门控 |
| `layers.{i}.mlp.up` | `[inter, hidden]` | SwiGLU 上投影 |
| `layers.{i}.mlp.down` | `[hidden, inter]` | SwiGLU 下投影 |

规则：

- 所有矩阵权重按"输出维在前、输入维在后"存储，与 `MatMulNT` 的右操作数约定一致。
- 角色集合由 Adapter 的 `required_roles()` 声明，缺失即报错；源 checkpoint 中未被任何角色消费的张量必须报告，不能静默忽略。
- `tie_word_embeddings = true` 时，`lm_head` 与 `embed` 指向同一源张量，`sl-core` 必须只分配一份缓冲。

## 5. prefill 与 decode

两者是**不同的 `Program`**，共享同一张权重表：

| 差异 | prefill | decode |
| --- | --- | --- |
| 输入 token 数 | `seq > 1` | `seq = 1` |
| 位置 | `0..seq` | `past_len` |
| KV 写入 | 写入 `0..seq` | 写入 `past_len` |
| 输出 | 取最后一个位置的 logits | 取当前位置的 logits |
| attention | 因果掩码覆盖 `0..=past_len+i` | 覆盖 `0..=past_len` |

选择两个程序而不是"一个程序带 mode 标志"的理由：两者的 attention 掩码、KV 写入区间和 `LastToken` 行为不同，硬塞进一个程序会让算子语义含糊，也会让 backend 的分支变多。

约束：

- 两个程序必须使用同一组权重角色与同一份 KV 状态布局；
- 两个程序各自的 `Param` value 形状必须一致；
- decode 程序的 `seq` 固定为 1，不再符号化。

## 6. v1 家族约束

Adapter 在 v1 必须拒绝以下配置，而不是尝试兼容：

- 非 decoder-only（encoder、encoder-decoder）；
- 任何 bias（`attention_bias`、`mlp_bias`）；
- MoE、专家路由；
- 滑动窗口、ALiBi、位置插值以外的未支持 RoPE 变体；
- 量化权重（`GroupwiseQuantized` 或 `OpaqueQuantized` 编码）；
- 未声明的架构或变体。

v1 允许：GQA（`heads % kv_heads == 0`）、`tie_word_embeddings`、RMSNorm、RoPE（Neox 风格）、SwiGLU、F32 权重。

## 7. 首个家族的选择标准

不按"最容易拿到 checkpoint"选，而按"能最快暴露契约缺陷"选：

1. 结构是标准 decoder-only，dense，无 MoE；
2. 组件恰好覆盖 v1 算子集：RMSNorm + RoPE + SwiGLU + GQA；
3. `config.json` 字段明确，tensor 命名稳定；
4. 词表可以很小，便于构造 tiny fixture；
5. RoPE 变体明确（rotate_half 还是交错），不能含糊；
6. 有可对照的公开实现或数值基线。

建议先用**自造 tiny fixture**（2 层、`hidden = 32`、`heads = 4`、`kv_heads = 2`、`head_dim = 8`、`vocab = 64`、`seq = 16`）验证 lowering 与执行，再接真实 checkpoint。tiny fixture 的权重可随机生成，预期 logits 由独立 f64 参考实现产生。

## 8. 支持矩阵

不记录"支持 Safetensors"这种粒度，而记录组合状态：

| 容器 | 家族 | 变体 | 权重精度 | 后端 | 模态 | 特性 | 状态 | 测试 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Safetensors | 待定 | 待定 | F32 | CPU | text | prefill+decode | 未开始 | — |

规则：

- 每一行必须有对应的 fixture 与基线，否则标为"未验证"；
- 新增格式、精度或后端时新增行，不修改已有行来"继承"能力；
- 矩阵随 Adapter 的 `README` 一起维护，并在 `sl-model` 的格式支持表中交叉引用。

## 9. 错误模型

Adapter 必须返回结构化错误，至少区分：

```rust
pub enum LoweringError {
    UnsupportedArchitecture { family: String, found: String },
    MissingConfigField(&'static str),
    InvalidConfigField { field: &'static str, reason: &'static str },
    MissingTensor(String),
    UnexpectedTensor(String),
    ShapeMismatch { role: String, expected: Vec<u32>, actual: Vec<u32> },
    DTypeMismatch { role: String, expected: DType, actual: DType },
    UnsupportedEncoding { tensor: String, encoding: String },
    UnsupportedFeature(&'static str),
    ProgramValidation(sl_ir::IrError),
}
```

要求：

- 错误里必须带角色名与源张量名，便于用户定位；
- 缺失张量一次性报告全部，不要遇到第一个就返回；
- `ProgramValidation` 直接透传 `sl-ir` 的错误，不做二次包装。

## 10. 验收

当前状态（2026-10-18）：第 1–4 项已有测试覆盖；第 5 项在 `sl-model-adapters` README 建立支持矩阵；第 6 项尚未打通（缺真实 checkpoint CLI 与 `Engine` 执行验证）。

首个家族完成的标准：

1. `describe()` 能在不加载权重字节的情况下，从 config 推导出角色表与形状；
2. tiny fixture 的 lowering 产物通过 `sl_ir::Program::validate()`；
3. prefill 与 decode 的程序结构有快照测试（角色表、算子序列、value 数量）；
4. 负例测试覆盖：缺张量、多张量、形状不符、bias、MoE、未知架构；
5. 支持矩阵新增一行并标注测试路径；
6. `sl-core` 能用该产物构建 Engine 并跑出与 f64 参考一致的 logits（容差见后端文档）。

## 11. 当前明确不承诺

- 具体家族名称已确定为 `llama` / `dense`；真实 checkpoint 支持尚未验证；
- 量化权重、多模态输入、MoE、滑动窗口；
- 多设备切分；
- 模型配置的自动推断；
- 与 Hugging Face `transformers` 的逐字段兼容。

## 12. 待决问题

1. **`IoSpec` 的跨程序 id 表示**：约定 id 一致，还是用 `(program, value)` 二元组？
2. **`tie_word_embeddings` 的绑定方式**：在 `WeightSpec` 中标记共享，还是在 `sl-core` 中按角色去重？
3. **RoPE 变体的配置来源**：`config.json` 是否有显式字段，还是由家族 + 变体固定？
4. **支持矩阵的存储格式**：Markdown 表格，还是机器可读的 TOML（供 CI 校验）？
5. **tiny fixture 的权重生成脚本**：放 `sl-model-adapters` 仓库还是 `tests/golden-baselines`？
