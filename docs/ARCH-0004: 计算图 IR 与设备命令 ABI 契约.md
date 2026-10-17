# StarLight 计算图 IR 与设备命令 ABI 契约

**文档编号：** ARCH-0004
**状态：** 设计提案，待首条端到端管线验证
**范围：** IR 分层、ABI 类型归属、算子契约、能力协商、静态内存规划、绑定语义、错误模型与版本策略
**前置文档：** [ARCH-0002](<ARCH-0002: 模块边界与发行版业务管线契约.md>)、[ARCH-0003](<ARCH-0003: 模型制品与多模态推理数据流设计.md>)、[RFC-0001](<RFC-0001: StarLight 系统工程与架构规划草案 (2026-10-16).md>)

本文把 ARCH-0002 中仍是概念名的交接对象（`GraphPlan`、`CompiledGraph`、`Device`、`Engine`）细化成可评审的数据契约，并引入一个新的 ABI crate `sl-ir`。文中出现的 Rust 类型部分已由 `sl-ir` 0.1.0 的 M0 交付（§3–§9 的数据类型与不变量校验），但编译流水线（§10）、能力协商 pass 与内存着色仍属设计目标。

## 0. 效力与冲突处理

- 本文**补充** ARCH-0002 §2/§3 的数据契约，不改变其调度分层（图内并发、设备提交、单模型运行时、请求编排四层职责）。
- 本文与 ARCH-0002 存在两处依赖方向差异，已由 [RFC-0002](<RFC-0002: 采纳 sl-ir 作为图 IR 与设备命令 ABI.md>) 接受并完成对 ARCH-0002 的修订：
  1. 新增独立 crate `sl-ir`，`sl-graph` 与 `sl-device` 均依赖它；
  2. `sl-model-adapters` 不再依赖 `sl-graph`，只依赖 `sl-model` 与 `sl-ir`。
- 本文与 RFC-0001 冲突时以本文为准，但 RFC-0001 的 trait 变更同样需要走 RFC 流程。
- 标为“目标”“后续”“v1 不承诺”的内容不是已交付能力。

## 1. 设计结论

1. **新增 ABI crate `sl-ir`**：零依赖、纯数据、无 I/O、无优化算法，具备 `no_std + alloc` 可行性。
2. **IR 分两层**：`Program` 是符号程序（形状含符号维，由模型 Adapter 构造），`CommandBuffer` 是物理命令流（形状全部具体化，由 `sl-graph` 编译产出，由 backend 消费）。
3. **算子分 core 与 composite**：core 是 IR 允许出现的语义原语词汇表，backend 必须逐项声明支持或拒绝，由编译期校验；composite 表达可整体优化的意图，backend 不支持时由 `sl-graph` 在编译期分解（v1 对 `CausalAttention` 例外，见 §5.3）。
4. **形状按上界具体化**：编译期用 `≤ 上界` 的具体形状做静态规划，运行期的真实长度通过标量 value 传入，不做运行期形状推导。
5. **副作用显式建模**：KV 写入标记为对 State value 的 `Write`，禁止重排，且 State 永不进入临时内存着色池。
6. **能力协商在编译期完成**：`DeviceCapabilities` 描述 backend 支持的算子、dtype 与形状约束，缺失能力在 build 阶段报错，不留到 `step()`。
7. **v1 范围收紧**：F32 dense、单设备、单队列、线性调度、无算子融合、无 paged KV、无量化权重。
8. **`IR_VERSION` 独立于 crate semver**：前者约束 ABI 语义，后者约束 Rust API。

## 2. 分层与 crate 归属

### 2.1 依赖方向

```text
                        ┌───────────────────────────────┐
                        │  sl-ir                        │  纯数据 ABI，零依赖
                        │  DType / Shape / Op / Plan    │  no_std 可行
                        └────┬───────────────────┬──────┘
                             │                   │
                    ┌────────┴──────┐     ┌──────┴────────┐
                    │  sl-graph     │     │  sl-device    │  traits + 资源管理
                    │  编译器 pass  │     └──────┬────────┘
                    └────────┬──────┘            │
                             │            ┌──────┴───────┐
                    ┌────────┴──────┐     │  backend-*   │
                    │ adapters      │     └──────┬───────┘
                    └────────┬──────┘            │
                             └────────┬──────────┘
                                ┌─────┴─────┐
                                │  sl-core  │
                                └───────────┘
```

关键点：**`sl-graph` 不再依赖 `sl-device`**。因为 `DeviceCapabilities` 是纯数据、放在 `sl-ir`，图编译器只需要认识 ABI，不需要认识设备 trait。这条边是旧方案（ABI 放 `sl-device`）无法避免的。

### 2.2 职责划分

| crate | 拥有 | 不得拥有 |
| --- | --- | --- |
| `sl-ir` | IR 数据模型、算子签名、内存计划不变量、`DeviceCapabilities`、绑定结构、`IrError`、`IR_VERSION` | 任何优化、调度、生命周期分析、I/O、日志、线程、`serde` |
| `sl-graph` | 形状具体化、能力校验、composite 分解、生命周期分析、内存着色、`CommandBuffer` 生成 | 解析模型文件、定义算子语义、持有设备句柄 |
| `sl-device` | `BackendDevice` trait、真实缓冲/队列/管线句柄、能力查询、提交与同步语义 | 图优化、内存着色、模型加载 |
| `backend-*` | core/composite 算子内核、设备自有的内核选择与融合 | 通用图优化、请求队列、模型格式 |
| `sl-model-adapters` | 家族规则、tensor 角色映射、构造 `Program`、声明 State 与 IO | 编译、内存规划、设备交互 |
| `sl-core` | 装配 `Program` + 权重 + 设备，驱动 `step` | 解析模型文件、实现算子、请求调度 |
| `sl-daemon` | 请求生命周期、队列、批处理、IPC | 算子、图优化、IR 定义 |

### 2.3 为什么不是 `sl-graph` 或 `sl-device`

- 放 `sl-device`：`BackendDevice::compile_pipeline` 需要引用 `CommandBuffer`，而 `sl-graph` 需要产出它，形成 `sl-device ↔ sl-graph` 循环；只能靠把 backend trait 上移到 `sl-core` 解决，但那违反“backend 不依赖高层”的边界。
- 放 `sl-graph`：则 backend 必须依赖编译器才能拿到 ABI 类型，第三方后端接入成本变高，且 ABI 版本会被编译器版本绑架。
- 新建 `sl-ir`：多一个 submodule 是唯一代价，换来无环依赖、独立版本化和“第三方后端只依赖两个小 crate”。

## 3. 类型系统

### 3.1 元素类型

```rust
pub enum DType { F32, F16, BF16, U32, I32, U8, I8 }
```

v1 的计算与权重逻辑 dtype 只要求 `F32`；`U32` 用于 token、位置与长度标量；`U8`/`I8` 用于打包后的量化载荷。其他变体为后续扩展预留，backend 必须在能力中声明实际支持集合。

### 3.2 维度与形状

```rust
pub struct SymId(u32);

pub enum Dim { Const(u32), Sym(SymId) }

pub struct Shape(Vec<Dim>);

pub struct SymDimDecl {
    pub id: SymId,
    pub name: &'static str,
    pub max: u32,
}
```

规则：

- 元素数用 `u64` 计算并对溢出报错；单维上限为 `u32::MAX`。
- 标量是空 `Shape`，元素数为 1。
- 任一维为 0 表示空张量，必须被允许（空序列）。
- `Dim::Sym` 在 `Program` 中只允许出现在 `batch` 与 `seq` 两个语义维度上；`CommandBuffer` 中禁止出现任何 `Dim::Sym`。
- 编译期的具体化上界来自 `ShapeSpec`（`SymId → u32` 的映射），由运行时配置提供。

### 3.3 值

```rust
pub struct ValueId(u32);

pub enum ValueRole {
    Input,    // 每步变化的输入：token、位置
    Scalar,   // 每步变化的小整数标量：seq_len、past_len
    Output,   // 每步结果：logits
    Param,    // 持久权重
    State,    // 持久状态：KV cache
    Temp,     // 静态 scratchpad
}

pub struct Value {
    pub dtype: DType,
    pub shape: Shape,
    pub role: ValueRole,
    pub encoding: TensorEncoding,
}

pub enum QuantizationFormat { Int4, Int8 }

pub enum TensorEncoding {
    Dense,
    GroupwiseQuantized { format: QuantizationFormat, group_size: u32 },
    OpaqueQuantized { scheme: String },
}
```

`dtype` 是**逻辑**元素类型，`encoding` 描述物理存储。只有 `Param` 允许非 `Dense` 编码；`Input`、`Output`、`Scalar`、`State`、`Temp` 必须为 `Dense`。量化参数按逻辑形状声明（例如 `[N, K]`），`storage_bytes` 返回打包载荷字节数；scale 是独立的 `Param` value，不由 `TensorEncoding` 隐式携带。`OpaqueQuantized` 没有可计算的存储大小，编译器与运行时必须显式拒绝执行。

规则：

- v1 所有 value 均为**行主序连续**，不引入 stride、不做别名。因此不需要 `Transpose`/`Reshape` 这类视图算子；需要转置语义的地方由 `MatMulNT` 表达，多头布局切换由 `SplitHeads`/`MergeHeads` 两个显式搬运算子承担（它们写新的 value，不构成别名）。
- **禁止原地写入**：每条命令输出到独立 value，输入与输出不得指向同一 value。
- `Param` 只读；任何命令不得将其作为输出。
- `Output` 可以直接作为命令的输出写入，不需要额外的拷贝命令。

### 3.4 对齐

- scratchpad 内每个 value 起始地址按 `max(64, caps.min_buffer_align)` 对齐。
- `Param` 缓冲由 `sl-device` 分配，保证同样的对齐下界。
- 对齐值属于契约而非实现细节：backend 可以据此省略运行期对齐检查。

## 4. 符号阶段：Program

```rust
pub struct Program {
    pub sym_dims: Vec<SymDimDecl>,
    pub values:   Vec<Value>,      // 允许 Dim::Sym
    pub nodes:    Vec<Node>,
    pub inputs:   Vec<ValueId>,
    pub outputs:  Vec<ValueId>,
    pub states:   Vec<StateDecl>,
}

pub struct Node {
    pub op: Op,
    pub inputs:  Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub effect:  Effect,
}

pub enum Effect {
    None,
    ReadState(ValueId),   // 目标必须 role == State
    WriteState(ValueId),
}

pub enum StateKind {
    KvCache {
        batch: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        max_seq: Dim,
        dtype: DType,
    },
}

pub struct StateDecl {
    pub value: ValueId,
    pub kind: StateKind,
}
```

约束：

- `Program` 不含架构标识。家族信息属于 `LoweredModel`（见 ARCH-0006），编译器不需要它。
- 一个 `Program` 对应一种执行形态。prefill 与 decode 是两个 `Program`，共享同一份参数表。
- `StateDecl` 只描述语义布局；其物理缓冲由 `sl-core` 通过 `sl-device` 分配后绑定。
- 同一 `State` 的读写顺序由 `effect` 与节点顺序共同决定，`sl-graph` 必须校验。

## 5. 算子契约

### 5.1 编码原则

**能影响内核是否存在的差异 → 独立 `OpCode`；纯数值参数 → `Op` 携带的 attr。**

反例：若用 `MatMul { transpose_b: bool }` 一个 op，`DeviceCapabilities` 就无法表达“支持 NT 但不支持 NN”，只能退化成运行期查询。

```rust
#[repr(u16)]
pub enum OpCode {
    MatMulNT, MatMulNN,
    Softmax, SoftmaxCausal,
    RmsNorm, Rope,
    Silu, Mul, Add, Scale,
    Gather, LastToken, KvAppend,
    SplitHeads, MergeHeads,
    CausalAttention, SwiGlu, RmsNormMatMul,
}

pub enum Op {
    MatMulNT,
    MatMulNN,
    Softmax,
    SoftmaxCausal,
    RmsNorm { eps: f32 },
    Rope { variant: RopeVariant, theta: f32 },
    Silu,
    Mul,
    Add,
    Scale { factor: f32 },
    Gather,
    LastToken,
    KvAppend { layer: u32 },
    SplitHeads { heads: u32, head_dim: u32 },
    MergeHeads { heads: u32, head_dim: u32 },
    CausalAttention { layer: u32, heads: u32, kv_heads: u32, head_dim: u32, scale: f32 },
    SwiGlu,
    RmsNormMatMul { eps: f32 },
}

pub enum RopeVariant { Neox, GptJ }
```

`Op::code()` 返回 `OpCode`，二者一一对应。`OpCode` 的判别值是 ABI 的一部分，永不复用（见 §12）。

### 5.2 core 算子

记法：`[...]` 表示可广播的前缀维。广播规则：前缀维**右对齐**后逐维比较，相等或其中一个为 1 即广播；缺失的维视为 1。因此三维激活 `[B, M]` 与二维权重 `[N, K]` 的 `[]` 前缀相容，而 GQA 的 `[B, Hq]` 与 `[B, 1]` 也相容。

| OpCode | 输入 | 输出 | 语义与约束 | v1 首个家族 |
| --- | --- | --- | --- | --- |
| `MatMulNT` | A `[…, M, K]`，B `[…, N, K]` | `[…, M, N]` | 前缀维广播；`K` 必须相等 | 使用 |
| `MatMulNN` | A `[…, M, K]`，B `[…, K, N]` | `[…, M, N]` | 前缀维广播 | 保留 |
| `Softmax` | X `[…, N]` | 同形 | 沿最后一维 | 保留 |
| `SoftmaxCausal` | X `[…, M, N]`，`seq_len: U32[]`，`past_len: U32[]` | 同形 | 位置 `past_len + i` 之后置零；`M == 1` 时为纯 decode | 保留（分解路径） |
| `RmsNorm` | X `[…, H]`，W `[H]` | `[…, H]` | attr: `eps` | 使用 |
| `Rope` | X `[…, M, D]`，`positions: U32[B, M]` | 同形 | attr: `variant`、`theta`；`positions` 沿 heads 维广播，且已含 `past_len` 偏移 | 使用 |
| `Silu` | X | 同形 | `x * sigmoid(x)` | 仅分解 |
| `Mul` | A、B 同形 | 同形 | v1 不做广播 | 仅分解 |
| `Add` | A、B 同形 | 同形 | 残差 | 使用 |
| `Scale` | X | 同形 | attr: `factor` | 保留 |
| `Gather` | `table[V, H]`，`ids: U32[..]` | `[…, H]` | 词表查表 | 使用 |
| `LastToken` | X `[B, M, H]`，`seq_len: U32[]` | `[B, H]` | 取每批最后一个位置 | 使用 |
| `KvAppend` | K `[B, Hkv, M, D]`，V 同形，`past_len: U32[]` | 无输出 | attr: `layer`；写区间 `past_len..past_len+M` | 保留（分解路径） |
| `SplitHeads` | X `[B, M, H·D]` | `[B, H, M, D]` | attr: `heads`、`head_dim`；显式数据搬运算子，不是视图 | 使用 |
| `MergeHeads` | X `[B, H, M, D]` | `[B, M, H·D]` | `SplitHeads` 的逆变换 | 使用 |

“v1 首个家族”一列的三种取值含义：

- **使用**：模型图会直接发出该算子，backend 必须实现；
- **仅分解**：只有当 backend 不支持对应 composite 时才会出现在图中；
- **保留**：v1 的图不会发出，backend 在能力中声明为不支持即可，但编译器必须认识它（用于未来的分解路径与第二后端）。

因此 v1 的 CPU reference 实际需要实现的是：`MatMulNT`、`RmsNorm`、`Rope`、`Add`、`Gather`、`LastToken`、`SplitHeads`、`MergeHeads`，外加 `CausalAttention` 与 `SwiGlu` 两个 composite；若它原生支持 `SwiGlu`，则 `Silu`/`Mul` 也不必实现。

副作用方面：v1 中只有原生 `CausalAttention` 与分解路径上的 `KvAppend` 带 `WriteState`；`KvAppend` 没有输出，因此命令的 `outputs` 允许为空。

### 5.3 composite 算子与分解

| composite | v1 是否提供分解 | 说明 |
| --- | --- | --- |
| `CausalAttention` | **不提供** | 分解需要对每层读取 KV 并展开 GQA，这要求额外的 core 算子（暂记作 `KvRead`）。v1 要求 backend 原生实现该 composite |
| `SwiGlu` | 提供 | 分解为 `MatMulNT`(gate)、`MatMulNT`(up)、`Silu`、`Mul`、`MatMulNT`(down) |
| `RmsNormMatMul` | 提供 | 分解为 `RmsNorm`、`MatMulNT`；v1 的图不会主动发出它，保留给后续融合 Pass |

**这是 v1 的一处显式例外**：§1 的“composite 不支持则分解”规则对 `CausalAttention` 不成立。选择它的理由是——直接实现 attention 内核与“先造 `KvRead` 再做分解”工作量相当，而前者路径更短、更容易通过数值基线；等出现第二个 backend 或第二类 attention 变体时，再决定 `KvRead` 的形态。

分解只对有对应规则的 composite 进行，递归直到全部为 core 算子；若某个 core 算子仍不被支持，则编译失败。分解必须保持可观察语义一致：副作用顺序、数值顺序、输出 binding 均不得改变。backend 既不原生支持某个 composite、编译器又没有对应分解规则时，返回 `UnsupportedOp`，不得在运行期降级。

### 5.4 签名与端口约束

```rust
pub struct Signature {
    pub inputs:  &'static [PortSpec],
    pub outputs: &'static [PortSpec],
}

pub struct PortSpec {
    pub dtypes:  DTypeSet,           // 允许的 dtype 集合
    pub roles:   ValueRoleSet,       // 允许的 value 角色集合
    pub min_rank: u8,
    pub max_rank: u8,
    pub allow_symbolic: bool,
}

impl OpCode {
    pub const fn signature(self) -> Signature;
}

pub fn validate_program(program: &Program) -> Result<(), IrError>;
```

一个端口通常接受多种角色（例如 `MatMulNT` 的 A 可以是 `Param`、`Temp` 或 `Input`），所以用集合而非单一 `ValueRole`。`DTypeSet` 与 `ValueRoleSet` 使用固定宽度位掩码（元素类型与角色集合稳定且很小）；`OpSet` 使用有序 `Vec` + 二分查找，因为 `OpCode` 判别值永不复用，位图宽度会随演进耗尽。

签名校验属于 `sl-ir`（“这个图是否良构”）；能力校验属于 `sl-graph`（“这个设备能否执行”）。两者错误类型分开，便于诊断。

### 5.5 v1 明确不提供的算子

`Concat`、`Slice`、`Reshape`、`Transpose`、`Reduce`、`ArgMax`、`Copy`、`Cast`、`MatMul` 的非转置权重变体、任何带 bias 的算子。

对应约束：**v1 支持矩阵要求模型无 bias、无偏置项、无 MoE、无滑动窗口**。采样与 argmax 属于生成控制层，不进入图。

## 6. 物理阶段：CommandBuffer

```rust
pub struct CommandId(u32);

pub struct Command {
    pub op: Op,
    pub inputs:  Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub deps:    Vec<CommandId>,   // v1 恒为 vec![i - 1]
    pub effect:  Effect,
}

pub struct CommandBuffer {
    pub ir_version: u32,
    pub values:   Vec<Value>,      // 禁止 Dim::Sym
    pub commands: Vec<Command>,
    pub plan:     MemoryPlan,
    pub inputs:   Vec<ValueId>,
    pub outputs:  Vec<ValueId>,
    pub states:   Vec<StateDecl>,
}
```

`deps` 在 v1 全部指向序号减一的前驱，形成全序。保留该字段是为了让并发调度成为**语义放宽**而不是接口破坏：后续版本可以填入真实依赖边，并在 `MemoryPlan` 中引入 schedule 参数。

确定性与顺序规则：

1. `commands` 的编号即执行顺序；`deps` 中每个 id 必须小于自身。
2. 对同一 `State` 的任意两条命令，只要其中一条为 `Write`，相对顺序必须保持。
3. 编译输出必须对同一 `(Program, ShapeSpec, DeviceCapabilities)` 逐字节一致；实现不得依赖 `HashMap`/`HashSet` 的迭代顺序。
4. `StepBinding` 的完成语义为“已入队”：调用方读取输出前必须显式同步。单设备同步推理可以立即返回，但契约按异步定义，避免将来改签名。

## 7. 静态内存规划

```rust
pub struct Slot {
    pub value:  ValueId,
    pub offset: u64,
    pub bytes:  u64,
}

pub struct MemoryPlan {
    pub scratchpad_bytes: u64,
    pub slots: Vec<Slot>,          // 按 ValueId 升序
}
```

生命周期定义为命令序号上的闭区间 `[first_use, last_use]`：

- `first_use`：该 value 首次作为任一命令的输入或输出的序号；
- `last_use`：该 value 最后一次作为输入的序号；
- 若 value 是 `Output`，其生命周期延伸到缓冲末尾，因此不参与复用。

不变量（必须由 `MemoryPlan::validate` 检查，并作为测试断言）：

1. 每个 slot 的 `offset` 是 `max(64, caps.min_buffer_align)` 的整数倍；
2. `offset + bytes ≤ scratchpad_bytes`；
3. 复用同一段字节的两个 value，其生命周期区间不相交；
4. `Param`、`State`、`Input`、`Output` 角色不得出现在 `slots` 中；
5. `slots` 覆盖全部 `Temp` 角色 value，且每个 value 至多出现一次。

算法要求：按 `first_use` 排序后做贪心区间着色，复杂度不超过 `O(N log N)`；不得使用依赖随机化的启发式，以保证编译确定性。

与 RFC-0001 WP-02 验收项的对应关系：构造 100 个节点、生命周期存在重叠的图，验证任意两个共享字节区间的 value 其区间无交集。

## 8. 能力协商

```rust
pub struct BackendTag(pub &'static str);

pub struct DeviceCapabilities {
    pub backend: BackendTag,
    pub ir_version: u32,
    pub core_ops: OpSet,
    pub composite_ops: OpSet,
    pub dtypes: DTypeSet,
    pub min_buffer_align: u32,
    pub max_scratchpad_bytes: u64,
    pub max_value_bytes: u64,
    pub constraints: ShapeConstraints,
}

pub struct ShapeConstraints {
    pub head_dim_multiple_of: u32,   // 例如 8
    pub matmul_k_multiple_of: u32,   // 例如 8
    pub max_heads: u32,
}
```

语义：

- `OpSet` 是**有序 `Vec<OpCode>` + 二分查找**，而不是位图。原因是 `OpCode` 判别值永不复用，位图宽度会在长期演进中被烧穿。
- `composite_ops` 只影响是否分解，不影响正确性；`core_ops` 缺失则编译失败。
- `constraints` 任一不满足即返回 `CapabilityMismatch`，错误中必须带上约束名、上限与实际值。
- `ir_version` 与编译器的 `IR_VERSION` 不一致时，`compile_pipeline` 入口直接拒绝，不做尽力兼容。

## 9. 绑定

```rust
pub struct BufferId(u32);   // 不透明索引，由 backend 解释
pub struct QueueId(u32);
pub struct PipelineId(u32);

pub struct ParamBinding {
    pub value:  ValueId,
    pub buffer: BufferId,
}

pub struct StepBinding {
    pub inputs:     Vec<(ValueId, BufferId)>,
    pub outputs:    Vec<(ValueId, BufferId)>,
    pub scalars:    Vec<(ValueId, u32)>,
    pub states:     Vec<(ValueId, BufferId)>,
    pub scratchpad: Option<BufferId>,
}
```

设计要点：

- **权重在编译期绑定**，进入 `PipelineHandle`；`StepBinding` 不含权重，避免每步重复传递指针。这是对 RFC-0001 §2.3 trait 签名的修正建议。
- `BufferId` 是间接层：backend 自己维护 `BufferId → 真实句柄` 的表。代价是每步若干次查表，收益是 ABI 与 Rust 所有权模型解耦，将来可以映射到 C ABI。
- 所有 `Vec` 在 `build` 阶段一次性分配并填满，`step()` 只更新 `scalars` 中的值，不新增分配。
- `PageTableDescriptor` 与 paged KV 属于 v1 之后的能力，本文不定义其结构，`bind_sparse_pages` 也不在 v1 必需接口内。

### 9.1 v1 decoder 的 IO 契约

| 名称 | dtype | shape | 角色 |
| --- | --- | --- | --- |
| `token_ids` | U32 | `[batch, seq]` | Input |
| `positions` | U32 | `[batch, seq]` | Input |
| `seq_len` | U32 | 标量 | Scalar |
| `past_len` | U32 | 标量 | Scalar |
| `kv_cache` | F32 | `[batch, layers, 2, max_seq, kv_heads, head_dim]` | State |
| `logits` | F32 | `[batch, vocab]` | Output |

约束：v1 要求同一 batch 内各序列长度一致；变长批量、paged KV 与连续批处理属于后续能力。

## 10. 编译流水线

```text
compile(program, shape_spec, caps) -> Result<CommandBuffer, CompileError>
```

| 步骤 | 输入 | 输出 | 失败条件 |
| --- | --- | --- | --- |
| 1. validate | `Program` | 良构性检查通过 | `IrError`：未知 value、环、arity、dtype/shape 不符 |
| 2. specialize | `ShapeSpec` | 全部 `Dim::Const` | `SymbolicDimExceedsMax` |
| 3. capability | `caps` | 已分解的算子集合 | `UnsupportedOp`、`CapabilityMismatch` |
| 4. effects | 副作用与 State 顺序 | 校验通过 | `StateEffectConflict` |
| 5. plan | 生命周期区间 | `MemoryPlan` | `PlanOverflow` |
| 6. emit | 全部 | `CommandBuffer` | 断言无 `Dim::Sym` 残留 |

要求：

- 步骤 3 必须在步骤 5 之前：分解会改变 value 数量，先规划再分解会导致偏移失效。
- 每个错误都必须携带节点序号（或 value id）与期望/实际值，禁止只返回字符串。
- `compile` 是纯函数，不读文件、不访问设备、不加锁。

## 11. 错误模型

```rust
pub enum IrError {
    UnknownValue(ValueId),
    DuplicateValue(ValueId),
    Cycle,
    ShapeMismatch { node: usize, input: usize, expected: Shape, actual: Shape },
    DTypeMismatch { node: usize, expected: DType, actual: DType },
    InvalidRole { value: ValueId, expected: ValueRole, actual: ValueRole },
    InvalidEffect { value: ValueId },
    StateNotDeclared(ValueId),
    SymbolicDimInCommandBuffer(ValueId),
    PlanInvariant(&'static str),
}

pub enum CompileError {
    Ir(IrError),
    UnsupportedOp { node: usize, op: OpCode },
    UnsupportedDType { value: ValueId, dtype: DType },
    CapabilityMismatch { constraint: ConstraintKind, limit: u32, actual: u32 },
    SymbolicDimExceedsMax { dim: SymId, max: u32, requested: u32 },
    StateEffectConflict { state: ValueId, first: usize, second: usize },
    PlanOverflow { required: u64, limit: u64 },
    VersionMismatch { device: u32, compiler: u32 },
}
```

分工：

- `IrError` 描述“图本身不合法”，与设备无关，放在 `sl-ir`；
- `CompileError` 描述“这台设备跑不了这张图”，放在 `sl-graph`；
- 运行期错误（队列、设备丢失、超时）由 `sl-device` 的 `DeviceError` 承担，不得混入编译期错误。

## 12. 版本与兼容

1. `IR_VERSION: u32` 约束 ABI 语义；crate semver 约束 Rust API。二者独立升级。当前 `IR_VERSION = 2`（v1 初始 ABI；v2 增加 `U8`/`I8`、`Value::encoding`、`TensorEncoding` 与 state/编码校验）。
2. 以下任一变化必须提升 `IR_VERSION`：增删 `OpCode`；修改 `Signature`；修改 `DType`、`TensorEncoding` 或 `Value` 字段；修改 `MemoryPlan` 不变量；修改绑定结构。
3. `OpCode` 使用 `#[repr(u16)]`，判别值永不复用。删除算子时保留其判别值作为墓碑，不重新分配。
4. backend 必须声明 `ir_version`，且 match 必须有 catch-all 返回 `UnsupportedOp`，以保证旧 backend 面对新算子时安全失败。
5. 编译缓存、序列化计划、C ABI 均以 `IR_VERSION` 作为缓存键的一部分。
6. `sl-ir` 从第一版即按 `no_std + alloc` 设计，`IrError` 不携带 `String`；`sl-graph` 的 `CompileError` 同样建议保持纯数据，便于日志、序列化与跨 ABI 传递。需要文本描述时由最上层格式化。

## 13. 演进路线

| 阶段 | 交付 | 验收 |
| --- | --- | --- |
| M0 | 建立 `sl-ir` 仓库，接入 workspace 与 `[patch.crates-io]` | `cargo check --workspace` 通过；crate 零依赖 |
| M1 | `OpCode::signature` 的 `Program` 级校验、属性测试 | 非法图全部被拒；属性测试覆盖 shape/arity（签名校验已实现，属性测试待补） |
| M2 | `specialize` + 能力校验 + 分解 | 缺算子的 backend 在编译期失败；分解前后语义等价 |
| M3 | 生命周期分析与 `MemoryPlan` | 100 节点图无区间交集；同输入两次编译字节一致 |
| M4 | `sl-device` trait + `backend-cpu` scalar reference | reference 实现首个家族所需的 core 子集与 `CausalAttention`/`SwiGlu`；与 golden 基线在容差内一致 |
| M5 | `sl-core` build/step + `starlight-nano` CLI | token IDs 输入到 logits 输出；零分配门禁通过 |

每个阶段都必须先更新对应模块的 README/DEVELOPMENT 与支持矩阵，再合并实现。M4 起引入的数值基线格式、容差与跨 ISA 一致性策略由后端文档定义，不写入本文。

M0 已完成：`sl-ir` 已建立独立仓库、接入 workspace 与 `[patch.crates-io]`，并提供本文 §3–§9 的数据类型、错误模型与不变量校验。算子签名校验（M1）已随 M0 一并落地；属性测试与 `specialize`/能力协商/内存着色仍待补。`[workspace].members` 与 `[patch.crates-io]` 只有在 crate 具备 Cargo manifest 后才能添加，否则 `cargo check --workspace` 会因找不到包而失败。

## 14. 当前明确不承诺

- 图内并发执行、多队列、真实依赖调度（`deps` 仅为预留）。
- 算子融合 Pass（`RmsNormMatMul` 只是提示，不保证被融合）。
- paged KV cache、变长批量、连续批处理。
- 量化权重、混合精度、F16/BF16 计算。
- 多设备切分、集合通信、CUDA/NPU 后端。
- IR 的序列化格式、编译缓存、C ABI 导出。
- 动态形状（运行期改变物理形状）；v1 只接受“上界具体化 + 标量长度”。

## 15. 待决问题

1. **paged KV 的 State 表示**：是扩展 `StateKind`，还是引入独立的 `PageTable` value？建议等第一个真实变长批量需求出现后再定。
2. **bias 支持**：v1 要求无 bias；若首个家族带 bias，是增加 `AddBias` 算子，还是把 bias 并入 `RmsNormMatMul` 之类的 composite？
3. **`Scalar` 的设备可见性**：CPU 后端可直接读 host 内存；Vulkan 后端需要 uniform buffer。是否需要在 `StepBinding` 中区分 host/device 标量，留待第二后端验证。
4. **`OpSet` 与位图的取舍**：本文选择有序 `Vec`，若实测编译期开销明显，再评估位图 + 墓碑计数的方案。
5. **`Program` 是否需要 builder API**：若 Adapter 手写 `Vec` 容易构造非法图，可在 `sl-ir` 增加受限 builder，但必须保持“不藏算法”。
6. **`no_std` 是否作为硬性门禁**：本文按可 `no_std` 设计，是否加 CI 检查待定。

## 16. 与现有文档的差异汇总

| 文档 | 需要的修订 |
| --- | --- |
| ARCH-0002 §2 | 新增 `sl-ir` 行；`sl-model-adapters` 改为依赖 `sl-model` + `sl-ir` |
| ARCH-0002 §3 | 依赖图插入 `sl-ir`；`sl-graph` 不再依赖 `sl-device` |
| ARCH-0002 §7 | 缺口列表增加“`sl-ir` 尚未建立” |
| RFC-0001 §2.3 | `compile_pipeline` 增加 `params` 参数；`StepBinding` 明确不含权重；`CommandBuffer` 来源改为 `sl-ir` |
| RFC-0001 §2.2 | “线性指令队列”改为“带依赖边与副作用标记的命令流，v1 调度为全序” |
| RFC-0001 §4.1 | 仓库拓扑增加 `crates/sl-ir` |
| 主 README | 组件表增加 `sl-ir`，并注明其为 ABI crate |

以上修订已由 RFC-0002 完成（2026-10-17）。
