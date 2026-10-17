# ARCH-0007：Graph / Core / Runtime 三线施工方案

**文档编号：** ARCH-0007
**状态：** 施工方案 v1.0
**日期：** 2026-10-17
**范围：** `sl-graph`、`sl-device`、`sl-core`、`distros/starlight-runtime` 的冻结接口、文件归属、并行顺序与验收
**前置文档：** [ARCH-0004](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)、[ARCH-0005](<ARCH-0005: 服务面契约 sl-core sl-daemon 与 InfrArray.md>)、[ARCH-0006](<ARCH-0006: 模型家族 Lowering 与 LoweredModel 契约.md>)

## 1. 目标与验收

> **落地状态（2026-10-18）：** `sl-graph`、`sl-device`、`sl-core`、`CpuDevice` 与 `starlight-runtime` 的 token-ID 端到端链路已跑通，workspace 共 190 个测试；`serve` 与真实模型 Adapter 仍未实现。

本轮三条线合并后必须跑通**真实端到端**，不允许 mock：

```text
token IDs
  → sl-graph::compile(Program, ShapeSpec, DeviceCapabilities)
  → CommandBuffer
  → sl-core::Engine::build(CommandBuffer + 权重 + CpuDevice)
  → Engine::step(Prefill)
  → logits
```

验收用一个 **tiny fixture**（不依赖真实模型、不依赖 Adapter）：

- 计算图：`Gather → RmsNorm → MatMulNT → LastToken`；
- 形状：`vocab = 8`、`hidden = 4`、`seq = 2`（prefill）/ `1`（decode）；
- 权重：确定性常量，不加载文件；
- 期望值：测试内用独立 `f64` 朴素实现计算；
- 入口：`starlight-runtime run --config <fixture> --input-ids 1,2` 输出 logits。

`last_token` 之后是 `[1, vocab]` logits；测试逐元素比较，容差 `1e-5`。

## 2. 并行策略

| 线 | 可立即开始 | 依赖 |
| --- | --- | --- |
| `sl-graph` | 是 | 仅 `sl-ir` |
| `sl-device` + `sl-core` | 是 | `sl-ir`；CPU 设备实现复用 `backend-cpu` 参考内核 |
| `starlight-runtime` | 是（CLI/config/describe/fixture） | 真实 `sl-graph`/`sl-core` API；合龙时完成 `run` |

`sl-core` **只接收 `CommandBuffer`，不依赖 `sl-graph`**。三条线共用本文 §3 的签名，不得自行改接口。

`sl-ir` 当前为 `IR_VERSION = 2`（v2 引入量化存储编码）；三线按该版本实现。任何一条线发现契约缺口，必须报告，不得单方面修改 `sl-ir`。

## 3. 冻结接口

### 3.1 `sl-device`

细粒度设备契约：设备负责缓冲区、上传/下载与**单算子分发**；`sl-core` 负责命令顺序与内存偏移。

```rust
use sl_ir::{BufferId, Op, Value};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferRef {
    pub buffer: BufferId,
    pub offset: usize, // 元素偏移（f32 或 u32，由 value.dtype 决定）
    pub len: usize,    // 元素个数
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceError {
    OutOfMemory { requested: usize },
    InvalidBuffer(BufferId),
    InvalidRange { buffer: BufferId, offset: usize, len: usize },
    UnsupportedOp(sl_ir::OpCode),
    ShapeOverflow,
    Execution(&'static str),
    Unimplemented(&'static str),
}

pub trait Device: Send {
    fn allocate_f32(&mut self, elements: usize, align: usize) -> Result<BufferId, DeviceError>;
    fn allocate_u32(&mut self, elements: usize, align: usize) -> Result<BufferId, DeviceError>;
    fn upload_f32(&mut self, buffer: BufferId, data: &[f32]) -> Result<(), DeviceError>;
    fn upload_u32(&mut self, buffer: BufferId, data: &[u32]) -> Result<(), DeviceError>;
    fn download_f32(&mut self, buffer: BufferId, out: &mut [f32]) -> Result<(), DeviceError>;
    fn download_u32(&mut self, buffer: BufferId, out: &mut [u32]) -> Result<(), DeviceError>;
    fn dispatch(
        &mut self,
        op: &Op,
        inputs: &[BufferRef],
        input_values: &[&Value],
        outputs: &[BufferRef],
        output_values: &[&Value],
        scalars: &[u32],
    ) -> Result<(), DeviceError>;
    fn synchronize(&mut self) -> Result<(), DeviceError>;
}
```

约束：

- `dispatch` 不得分配；`inputs.len() == input_values.len()`，`outputs.len() == output_values.len()`；
- `BufferRef.len` 必须等于对应 value 的元素数；
- 设备只执行单算子，不解释依赖、不持有请求、不做调度。

CPU 实现 `CpuDevice` 放在 `crates/backends/backend-cpu`，构造函数固定为 `CpuDevice::new()`，复用现有 `reference::*` 内核；现有 `executor` 保持不变，作为 IR 级参考执行器。

### 3.2 `sl-core`

```rust
pub struct CompiledProgram {
    pub ir: sl_ir::CommandBuffer,
    pub weights: Vec<(sl_ir::ValueId, sl_ir::BufferId)>,
    /// `Input` value carrying token ids.
    pub token_ids: sl_ir::ValueId,
    /// `Input` value carrying positions; `None` when the graph has no RoPE.
    pub positions: Option<sl_ir::ValueId>,
    /// `Scalar` value carrying the current sequence length.
    pub seq_len: sl_ir::ValueId,
    /// `Scalar` value carrying the KV prefix length; `None` without KV cache.
    pub past_len: Option<sl_ir::ValueId>,
    /// `Output` value carrying logits.
    pub logits: sl_ir::ValueId,
}

pub struct EngineConfig {
    pub prefill: CompiledProgram,
    pub decode: CompiledProgram,
    pub device: Box<dyn sl_device::Device>,
    pub max_batch: u32,
    pub max_context: u32,
}

pub enum StepMode { Prefill, Decode }

pub struct StepInput<'a> {
    pub token_ids: &'a [u32],
    pub positions: &'a [u32],
    pub seq_len: u32,
    pub past_len: u32,
}

pub struct StepStats { pub tokens: u32 }

pub struct EngineInfo {
    pub ir_version: u32,
    pub max_context: u32,
    pub max_batch: u32,
}

impl Engine {
    pub fn build(config: EngineConfig) -> Result<Self, BuildError>;
    pub fn step(&mut self, mode: StepMode, input: StepInput<'_>, output: &mut [f32]) -> Result<StepStats, StepError>;
    pub fn info(&self) -> EngineInfo;
}
```

`CompiledProgram` 是 `Engine` 与 IR 之间的绑定元数据：IR 只有角色，没有 `token_ids` / `seq_len` 这类语义名，必须由装配方显式给出；`weights` 中的 buffer 由调用方在构建前上传。

构建期职责：

1. 校验 `CommandBuffer` 与 `IR_VERSION`；
2. 用 `device` 分配 scratchpad、KV state、输入、输出缓冲；
3. 建立 `ValueId → BufferRef` 表（Param 来自 `weights`，State/Input/Output 来自分配，Temp 来自 `MemoryPlan`）；
4. 上传权重（由调用方先上传，或 `Engine::build` 接收已上传的 buffer）。

步进职责：

1. 校验 `seq_len`/batch 与缓冲区容量；
2. 上传 `token_ids`、`positions`；
3. 按命令顺序解析 `BufferRef`，组织 `scalars`，调用 `device.dispatch`；
4. `device.synchronize`；
5. 下载 logits 到调用方 `output`。

约束：`step` 不得分配、不得加锁、不得做 token 循环或采样。

### 3.3 `sl-graph`

```rust
pub fn compile(
    program: &sl_ir::Program,
    shapes: &sl_ir::ShapeSpec,
    capabilities: &sl_ir::DeviceCapabilities,
) -> Result<sl_ir::CommandBuffer, CompileError>;
```

`CompileError` 至少包含：`Ir(sl_ir::IrError)`、`SymbolicDimExceedsMax`、`UnsupportedOp`、`CapabilityMismatch`、`StateEffectConflict`、`PlanOverflow`。

流水线顺序固定：

```text
validate → specialize → capability check → decompose → effects → liveness/coloring → emit
```

要求：

- 输出无 `Dim::Sym`；
- 同输入编译结果逐字节一致；
- `SwiGlu`、`RmsNormMatMul` 可分解；`CausalAttention` 不分解，缺能力直接失败；
- 内存着色为贪心区间复用，`O(N log N)`，不变量由 `sl-ir` 校验。

### 3.4 `starlight-runtime`

位置：`distros/starlight-runtime`（主仓库，workspace member）。

```text
starlight-runtime describe --config <path>
starlight-runtime validate --config <path>
starlight-runtime run --config <path> --input-ids 1,2
starlight-runtime serve --config <path>
starlight-runtime version
```

配置使用 JSON（避免额外 TOML 依赖），字段：

```json
{
  "model":  { "kind": "fixture", "architecture": "fixture" },
  "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
  "server": { "bind": "127.0.0.1", "port": 8000 }
}
```

`describe` 输出按 ARCH-0005 §5.2，`engine` 为 `starlight`。`serve` 在 `sl-daemon` 就绪前返回 `Unsupported`，不得自行实现 HTTP。

## 4. 文件归属

| 线 | 可写路径 | 不得触碰 |
| --- | --- | --- |
| graph | `crates/sl-graph/**` | 其他一切 |
| core | `crates/sl-device/**`、`crates/sl-core/**`、`crates/backends/backend-cpu/**` | 其他一切 |
| runtime | `distros/starlight-runtime/**`、根 `Cargo.toml` 的 member 行、`Cargo.lock` | 其他 crate、submodule |

三条线都不修改 `sl-ir`；都不执行 git 写操作。

## 5. 合龙顺序

1. graph 与 core 各自完成后，runtime 的 `run` 直接调用真实 `compile` 与 `Engine`；
2. 若 runtime 先完成，其 `run` 可能因 API 未落地而无法编译；此时只报告缺口，不造 stub；
3. 合并后统一跑：

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo run -p starlight-runtime -- run --config <fixture> --input-ids 1,2
```

## 6. 风险

| 风险 | 对策 |
| --- | --- |
| `sl-device` 粒度定错导致 core/backend 返工 | 本文冻结 `dispatch` 单算子接口；本轮不引入队列/命令录制 |
| runtime 与 graph/core 接口漂移 | 三方共用本文签名；发现不一致立即停止并报告 |
| CPU 设备实现与现有 executor 逻辑重复 | `CpuDevice` 只做算子分发；executor 保持为 IR 级参考，不强行合并 |
| 端到端 fixture 依赖 Adapter | fixture 直接在测试/运行时代码里构造 `Program`，不依赖 `sl-model-adapters` |
| 热路径分配 | `step`/`dispatch` 只使用预分配数组与切片；`Engine::build` 允许分配 |
