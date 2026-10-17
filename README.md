# StarLight

StarLight 是 SeekStarAI 面向模型推理构建的 AI Infra 基础设施项目。它不是把模型格式、算子、图执行、设备后端、请求服务和分布式通信封装在一起的单体引擎，而是一组边界清晰、可独立演进、可按场景组合的基础组件。

项目由契约层、模型与精度工具、图编译器、运行时、设备接口、计算后端和服务组件构成；发行版负责把这些组件组装成可交付形态。

> **项目阶段：早期。**
> 当前可运行：模型读取、离线量化、IR 校验、图编译、CPU 执行器与 `Engine::step`、`starlight-runtime` 的 token-ID 端到端 fixture，以及 `sl-model-adapters` 的 llama-dense lowering（测试中贯通 `ModelBundle → Program → CommandBuffer`）。
> **尚不能以服务形态加载真实模型**：tokenizer、`sl-daemon` 服务端、CUDA/Vulkan 内核仍未实现；真实 checkpoint 的端到端 CLI 尚未提供。
> 本 README 中的“目标”“规划”“后续”均不代表已交付能力。

## 1. 项目定位

StarLight 解决的是一组相互独立、但必须协同的问题：

| 问题 | 负责组件 |
| --- | --- |
| 模型文件如何进入系统 | `sl-model` |
| 权重如何量化、转换和布局 | `sl-precision` |
| 模型家族如何变成计算图 | `sl-model-adapters` |
| 计算如何表达成可执行的图与命令 | `sl-ir`、`sl-graph` |
| 计算如何落到设备 | `sl-device`、`backend-*` |
| 一次推理如何被装配和驱动 | `sl-core` |
| 请求、会话、批处理与 API 如何组织 | `sl-daemon`、`starlight-runtime` |
| 多模型、多节点如何被编排 | 外部控制面（如 InfrArray） |

任何单个 crate 都不代表整个 StarLight。

## 2. 设计原则

1. **契约先行。** 模块之间只通过版本化的数据结构与接口协作。`sl-ir` 是图、算子、静态内存计划与设备命令的公共 ABI，独立于任何后端。
2. **机制与策略分离。** `sl-graph`、`sl-core` 只提供执行机制；请求队列、批处理、优先级、租户与生命周期属于运行时外围。
3. **控制面与计算面分离。** 内核不关心部署、节点、租户与 UI；编排器不关心算子、图与内核。
4. **可替换性。** 后端、运行时、控制面都应可替换；替换某一层不应要求修改其他层。
5. **编译期决定，运行期少思考。** 形状、内存布局、算子能力在构建期确定；运行期通过显式绑定执行。
6. **按场景组装。** 端侧、服务器与集群对进程模型、调度和通信的需求不同，通过发行版与运行时组合，而不是宏开关。

## 3. 系统分层

```text
┌────────────────────────────────────────────────────────────┐
│ 控制面（本仓库之外）                                        │
│ InfrArray / 外部编排器：节点、放置、多服务、网关、UI          │
└───────────────────────┬────────────────────────────────────┘
                        │ runtime 进程协议（describe/serve/health/metrics）
┌───────────────────────▼────────────────────────────────────┐
│ 服务层                                                      │
│ starlight-runtime（已实现 token-ID fixture）                  │
└───────────────────────┬────────────────────────────────────┘
                        │ Rust API
┌───────────────────────▼────────────────────────────────────┐
│ 运行时层                                                    │
│ sl-core：Engine / Session / step，无 HTTP、无队列、无策略     │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│ 编译与契约层                                                │
│ sl-ir：Program / CommandBuffer / Op / MemoryPlan            │
│ sl-graph：具体化、能力协商、分解、内存规划（已实现）           │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│ 设备与内核层                                                │
│ sl-device（契约）+ backend-cpu / backend-vulkan / backend-*  │
└────────────────────────────────────────────────────────────┘
```

数据流：

```text
模型文件
  │
  ├── sl-model ──> ModelBundle ──> sl-model-adapters ──> Program（sl-ir）
  │                                        │
  ├── sl-precision（离线量化）              │
  │                                        ▼
  └──────────────────────────────> sl-graph ──> CommandBuffer（sl-ir）
                                                   │
                                          sl-core Engine
                                                   │
                                          sl-device + backend-*
                                                   │
                                          sl-daemon / starlight-runtime
                                                   │
                                          InfrArray / CLI / SDK
```

图中 `sl-graph`、`sl-device`、`sl-core`、`backend-cpu` 与 `starlight-runtime` 的 token-ID 路径已实现；模型 Adapter、`sl-daemon`、tokenizer、CUDA/Vulkan 仍处于骨架或设计阶段。

## 4. 组件与职责

| 组件 | 仓库形态 | 职责 | 当前状态 |
| --- | --- | --- | --- |
| `sl-ir` | submodule | 图 IR、算子契约、量化存储编码、静态内存计划、设备能力与绑定 ABI | **已实现**：数据类型、签名/state/编码校验、`ProgramBuilder`；32 个测试；零依赖、可 `no_std` |
| `sl-model` | submodule | 格式无关的模型包、张量元数据与格式读取 | **已实现 Safetensors reader**；6 个测试；GGUF/ONNX/写入能力未实现 |
| `sl-precision` | submodule | 量化、反量化与精度转换工具 | **已实现**对称分组 INT4/INT8 与 `sl-quantize` CLI；5 个测试 |
| `sl-model-adapters` | submodule | 模型家族 lowering 为 `Program` 与权重绑定 | **已实现 llama-dense**；16 个测试；无 CLI/服务 |
| `sl-graph` | submodule | 将 `Program` 具体化为 `CommandBuffer`：能力校验、分解、内存规划 | **已实现 `compile` 流水线**；22 个测试 |
| `sl-device` | submodule | 设备、缓冲区、上传/下载与单算子分发契约 | **已实现细粒度 `Device` 契约**；3 个测试 |
| `sl-core` | submodule | 装配 `CommandBuffer`、权重与设备，提供 `Engine::build/step` | **已实现**；9 个测试 |
| `backend-cpu` | 主仓库 | CPU 参考内核、IR 执行器与 `CpuDevice` | **已实现**；71 个测试；`CpuDevice` 实现 `sl-device::Device`；无 SIMD |
| `backend-vulkan` | 主仓库 | Vulkan 计算后端 | **骨架** |
| `sl-daemon` | submodule | 请求生命周期、批处理、OpenAI 协议、流式 | **骨架**：空 `main` |
| `starlight-nano` | 主仓库 | 端侧发行版装配入口 | **空壳**：`main` 为空 |
| `starlight-cluster` | 主仓库 | 集群发行版装配入口 | **空壳**：`main` 为空 |
| `starlight-runtime` | 主仓库 | 服务运行时组合根：装配 daemon、模型、Adapter、图编译与后端 | **已实现 token-ID 路径**（config/describe/validate/run）；`serve` 未实现；43 个测试 |
| 分布式通信 | 未建立 | 跨设备/跨节点通信 | **未开始** |

## 5. 发行版、运行时与产品

三个概念需要区分：

| 概念 | 含义 | 当前状态 |
| --- | --- | --- |
| 发行版（distro） | 选择并组装组件的交付形态 | `starlight-nano` / `starlight-cluster` 为空壳 |
| 运行时（runtime） | 由 `sl-daemon` + 后端 + 模型组装出的可执行文件，暴露统一进程协议 | `starlight-runtime` 已实现 token-ID fixture；`serve` 待 `sl-daemon` |
| 产品（product） | 面向部署与运营的完整系统，可管理多个运行时 | InfrArray，独立仓库 |

- **StarLight-Nano**：面向端侧、AIPC 与嵌入式环境，目标是轻量、进程内执行。
- **StarLight-Cluster**：面向服务器与集群，目标是组合 `sl-daemon`、设备后端与后续通信组件。
- **StarLight-Runtime**（`distros/starlight-runtime`）：通用服务运行时组合根，装配 `sl-daemon`、模型加载、Adapter、图编译与后端。当前已实现 config/CLI/describe/validate/run 与 token-ID fixture；`serve` 在 `sl-daemon` 就绪前返回未实现。它不是 submodule，也不属于 `sl-daemon`。
- **InfrArray**：以 StarLight 为引擎之一的独立商业产品，负责节点纳管、放置、服务编排、网关与 UI。它**不是本仓库的发行版，也不是 submodule**；首发使用 vLLM / TensorRT-LLM 运行时，StarLight 按认证矩阵逐格替换。详见 [PRODUCT-0001](<docs/PRODUCT-0001: InfrArray 产品规划.md>) 与 [ARCH-0005](<docs/ARCH-0005: 服务面契约 sl-core sl-daemon 与 InfrArray.md>)。

## 6. 仓库结构

```text
StarLight/
├── crates/
│   ├── sl-ir/                # 图 IR 与设备命令 ABI（submodule，已实现 M0/M1）
│   ├── sl-model/             # 模型包与格式读取（submodule）
│   ├── sl-model-adapters/    # 模型家族 lowering（submodule，llama-dense）
│   ├── sl-precision/         # 量化与精度工具（submodule）
│   ├── sl-graph/             # 图编译与内存规划（submodule，已实现）
│   ├── sl-core/              # 运行时装配与 Engine（submodule，已实现）
│   ├── sl-device/            # 设备接口契约（submodule，已实现）
│   └── backends/
│       ├── backend-cpu/      # CPU 参考内核 + 执行器 + CpuDevice
│       └── backend-vulkan/   # Vulkan 后端（骨架）
├── services/
│   └── sl-daemon/            # 服务进程与请求编排（submodule，骨架）
├── distros/
│   ├── starlight-nano/       # 端侧发行版入口（空壳）
│   ├── starlight-cluster/    # 集群发行版入口（空壳）
│   └── starlight-runtime/    # 服务运行时组合根（已实现 token-ID 路径）
├── docs/                     # 架构、决策记录、产品规划
├── tests/
│   └── golden-baselines/     # 数值基准目录（尚无基准数据）
├── device_info.rs            # 独立诊断脚本（非 workspace 成员）
└── Cargo.toml                # workspace 与 [patch.crates-io]
```

仓库形态说明：

- `sl-ir`、`sl-model`、`sl-model-adapters`、`sl-precision`、`sl-graph`、`sl-device`、`sl-core`、`sl-daemon` 是独立 Git 仓库，以 submodule 接入；
- 主仓库只记录 submodule URL 与固定提交，不纳入其源文件；
- 只有具备 Cargo manifest 并列入根 workspace 的组件才参与构建；`sl-model-adapters` 已加入 workspace；
- 根 `Cargo.toml` 的 `[patch.crates-io]` 将 StarLight 组件解析到本地路径，因此 workspace 构建不依赖 crates.io 上的同名包。

## 7. 当前实现状态

### 7.1 已验证可运行的能力

1. **模型读取**：`sl-model` 读取 Safetensors，校验 shape/dtype/编码，并以共享 `Arc<[u8]>` 视图暴露张量数据，避免逐张量复制。
2. **离线量化**：`sl-precision` 对 `f32` 做对称分组 INT4/INT8 量化，载荷 64 字节对齐；`sl-quantize` 可将 Safetensors 中的浮点张量转换为 StarLight 量化格式。
3. **IR 构造与校验**：`sl-ir` 提供 `Program` / `CommandBuffer` / `Op` / `Signature` / `MemoryPlan` / `DeviceCapabilities` / 绑定结构与错误模型，并在 `Program::validate` 中校验端口数量、dtype、角色、秩与状态副作用。
4. **命令流执行**：`backend-cpu` 的参考执行器可以执行手工构造并通过校验的 `CommandBuffer`，覆盖全部 18 个 `OpCode`，包括 `CausalAttention`、`SwiGlu`、`RmsNormMatMul`。执行路径无 `unsafe`、无堆分配。
5. **图编译**：`sl-graph::compile` 完成符号维具体化、能力校验、`SwiGlu`/`RmsNormMatMul` 分解、生命周期着色与 `CommandBuffer` 生成；22 个测试。
6. **端到端 token-ID 路径**：`starlight-runtime run` 把 fixture `Program` 编译后交给 `sl-core::Engine`，通过 `CpuDevice` 执行并输出 logits；与独立 f64 实现比对，容差 `1e-5`。
7. **模型 lowering**：`sl-model-adapters` 的 `llama`/`dense` 家族把 `ModelBundle` 转成 prefill/decode `Program`，覆盖 GQA、NeoX RoPE、`CausalAttention`、SwiGLU 与 KV state；16 个测试。

### 7.2 只有骨架或设计的能力

- `sl-daemon`：尚无 HTTP、队列、OpenAI 协议与流式；
- `sl-tokenizer`：尚未建立；
- `backend-vulkan`：无 Vulkan 实现；
- `starlight-nano` / `starlight-cluster`：无 CLI/SDK 管线；
- 分布式通信：未建立。

### 7.3 已打通与未打通的链路

当前已打通（测试中）：

```text
Safetensors (F32) ──> ModelBundle ──> llama-dense Adapter ──> Program
                                                               │
token IDs ──> ... ──> sl-graph::compile ──> CommandBuffer ──> Engine ──> CpuDevice ──> logits
```

`lower` 已能把 tiny `ModelBundle` 转成 prefill/decode `Program`，并由 `sl-graph::compile` 生成合法 `CommandBuffer`；尚未提供把真实 checkpoint 串起来的 CLI。

尚未打通：

- **文本**：token ↔ 字符串，`sl-tokenizer` 尚未建立；
- **服务**：HTTP / OpenAI / 队列，`sl-daemon` 仍是骨架；
- **真实权重执行**：F32 checkpoint 的端到端 CLI；F16/BF16 需转换或新增算子；量化执行需要 backend 量化内核；
- **硬件**：CUDA / Vulkan 内核尚未实现。

### 7.4 质量基线

| 项目 | 状态 |
| --- | --- |
| 单元测试 | **207 个**：`sl-ir` 32、`sl-graph` 22、`sl-model-adapters` 16、`backend-cpu` 71、`starlight-runtime` 43、`sl-core` 9、`sl-device` 3、`sl-model` 6、`sl-precision` 5 |
| `cargo clippy --workspace --all-targets -D warnings` | 通过 |
| `cargo fmt --all --check` | 通过 |
| `cargo check -p sl-ir --no-default-features` | 通过 |
| CI | **尚未配置** |
| 跨组件 golden baseline | **尚无数据**，仅目录与说明 |

## 8. 快速开始

### 8.1 环境

- Rust stable（本仓库在 1.87 上验证）；
- Git（需要 submodule 支持）。

### 8.2 克隆与构建

```sh
git clone --recurse-submodules https://github.com/SeekStarAI/StarLight.git
cd StarLight

cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

如果克隆时遗漏了 submodule：

```sh
git submodule update --init --recursive
```

### 8.3 运行已实现的工具

量化一个 Safetensors 文件：

```sh
cargo run --release -p sl-precision --bin sl-quantize -- \
  --input model.safetensors \
  --output model-int4.safetensors \
  --format int4 \
  --group-size 128
```

输出文件把量化载荷与 scale 保存为张量，并通过 `starlight.quantization` 元数据记录恢复信息；它不是普通模型加载器可直接读取的权重文件。参数与限制见 [`sl-precision` 文档](crates/sl-precision/README.md)。

### 8.4 组件级测试

```sh
cargo test -p sl-ir
cargo test -p sl-model
cargo test -p sl-precision
cargo test -p starlight-backend-cpu
cargo check -p sl-ir --no-default-features
```

## 9. 文档

| 文档 | 内容 |
| --- | --- |
| [文档索引](docs/README.md) | 全部架构、决策与产品文档 |
| [ARCH-0002](<docs/ARCH-0002: 模块边界与发行版业务管线契约.md>) | 模块职责、依赖方向与调度分层（当前工程基线） |
| [ARCH-0003](<docs/ARCH-0003: 模型制品与多模态推理数据流设计.md>) | 模型制品、权重表示、多模态与流式边界 |
| [ARCH-0004](<docs/ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>) | `sl-ir`、两层 IR、算子契约、能力协商、内存计划 |
| [ARCH-0005](<docs/ARCH-0005: 服务面契约 sl-core sl-daemon 与 InfrArray.md>) | `sl-core` / `sl-daemon` / runtime 协议与控制面配合 |
| [ARCH-0006](<docs/ARCH-0006: 模型家族 Lowering 与 LoweredModel 契约.md>) | 模型家族 Adapter 与 `LoweredModel` |
| [RFC-0002](<docs/RFC-0002: 采纳 sl-ir 作为图 IR 与设备命令 ABI.md>) | 采纳 `sl-ir` 的决策记录 |
| [PRODUCT-0001](<docs/PRODUCT-0001: InfrArray 产品规划.md>) | InfrArray 产品规划与运行时可替换策略 |
| [PHIL-0001](<docs/PHIL-0001: 为什么我们将 StarLight 构建为“推理领域的 GNU Linux”？.md>) | 模块化与发行版设计哲学 |

文档与实现不一致时，以代码、测试和对应模块的最新文档为准；跨仓库契约以 ARCH 文档与 RFC 为准。

## 10. 开发与贡献

### 10.1 仓库与 submodule 工作流

1. 在对应 submodule 仓库内开发、提交；
2. 先推送 submodule，再在主仓库更新指针；
3. 主仓库提交必须包含 `.gitmodules`/指针与必要文档变更；
4. 不要在子模块中提交主仓库文件，也不要在主仓库中直接修改 submodule 源文件。

### 10.2 契约变更

涉及以下内容的修改必须先走 RFC 或更新对应 ARCH 文档：

- `sl-ir` 的 `OpCode`、`Signature`、`MemoryPlan` 不变量或绑定结构（同时提升 `IR_VERSION`）；
- `sl-device` 的公共 trait；
- `sl-core` 的 `Engine`/`step` 签名；
- runtime 进程协议；
- 模块依赖方向。

`OpCode` 判别值永不复用；删除的算子保留墓碑值。backends 必须声明 `ir_version`，并对未知算子安全失败。

### 10.3 质量要求

提交前至少运行：

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

新增能力必须附带测试；跨模块交接应有契约测试；性能声明必须附带基准与硬件信息。

### 10.4 新增组件的放置规则

| 能力 | 放置位置 |
| --- | --- |
| 新设备 | 实现 `sl-device` 契约，新增 `backend-*` |
| 新模型格式 | `sl-model` 的格式适配层 |
| 新模型家族 | `sl-model-adapters` |
| 量化算法 | `sl-precision` |
| 图优化、内存规划 | `sl-graph` |
| 请求队列、批处理、API | `sl-daemon` |
| 部署、节点、租户编排 | 控制面（InfrArray 或外部系统），不进入本仓库内核 |

## 11. 路线图

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | `sl-ir` 数据类型、不变量校验、workspace 接入 | **已完成** |
| M1 | 算子签名校验（端口数量、dtype、角色、秩） | **已完成** |
| M2 | `sl-graph`：`Program → CommandBuffer` 具体化、能力协商、composite 分解 | **已完成** |
| M3 | `sl-graph`：生命周期分析与内存着色 | **已完成** |
| M4 | `sl-device` 契约、`backend-cpu` 接入设备执行路径 | **已完成** |
| M5 | `sl-core` Engine/step、`sl-tokenizer`、`sl-daemon`、`starlight-runtime` | **部分完成**：Engine/step 与 runtime token-ID 已完成；tokenizer/daemon 待做 |
| M6 | 首个模型家族端到端：tiny fixture 起步，逐步覆盖 dense 7B–14B | **部分完成**：llama-dense lowering + compile 已贯通；缺真实 checkpoint CLI 与执行验证 |

路线图以 [ARCH-0004 §13](<docs/ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>) 与 [ARCH-0005 §10](<docs/ARCH-0005: 服务面契约 sl-core sl-daemon 与 InfrArray.md>) 为准。

## 12. 相关项目

| 项目 | 关系 |
| --- | --- |
| [SeekStarAI/infrarray](https://github.com/SeekStarAI/infrarray) | 独立商业产品，控制面与编排器；与 StarLight 平级，不作为 submodule。首发使用 vLLM / TensorRT-LLM 运行时，StarLight 按认证矩阵逐步替换 |

StarLight 不依赖 InfrArray；InfrArray 通过运行时进程协议使用 StarLight，不链接内核 crate。

## 13. 许可

本仓库采用双重许可，可选择 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。

各 submodule 以其仓库内的许可证为准；当前 `sl-ir`、`sl-model`、`sl-model-adapters`、`sl-precision`、`sl-graph`、`sl-device`、`sl-core`、`sl-daemon` 均为 MIT。
