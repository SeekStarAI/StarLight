# ARCH-0005：服务面契约 —— sl-core、sl-daemon 与 InfrArray

**文档编号：** ARCH-0005
**状态：** 设计提案，待首个运行时验证
**范围：** `sl-core` 的 Engine 契约、`sl-daemon` 的服务策略、`starlight-runtime` 的进程协议，以及 InfrArray 如何在不链接内核的前提下接管服务
**前置文档：** [ARCH-0002](<ARCH-0002: 模块边界与发行版业务管线契约.md>)、[ARCH-0003](<ARCH-0003: 模型制品与多模态推理数据流设计.md>)、[ARCH-0004](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)、[PRODUCT-0001](<PRODUCT-0001: InfrArray 产品规划.md>)

本文把“服务”这件事拆成四层，并给出层间契约。目标是让 InfrArray 能完整接管部署与运营，同时**不链接、不侵入 StarLight 的任何执行代码**。

## 1. 设计结论

1. **四层分离**：执行机制（`sl-core`）、服务策略（`sl-daemon`）、进程打包（`starlight-runtime`）、集群控制面（InfrArray）。
2. **OpenAI 协议只实现一次**：在 `sl-daemon` 中；InfrArray 的 Gateway 只做多服务路由、鉴权、配额与计量，不做协议翻译。
3. **InfrArray 不链接 `sl-core`**：两者只通过 `starlight-runtime` 的进程协议交互。
4. **一个 Service = 一个 runtime 进程 = 一个 Engine 实例**（v1）。多模型靠多进程，而不是一个进程内多模型。
5. **`sl-core` 不知道 HTTP**；`sl-daemon` 不知道 GPU 细节；`starlight-runtime` 不知道多节点；InfrArray 不知道算子。
6. **可替换性是验收标准**：换 backend 不影响 daemon，换 daemon 不影响 InfrArray，换控制面不影响 runtime。

## 2. 分层与调用链

```text
┌────────────────────── InfrArray ──────────────────────┐
│ Gateway（路由/鉴权/配额）  Controller  Agent  Web UI   │
└───────────────┬───────────────────────────────────────┘
                │ 进程协议：describe / serve / health / metrics
┌───────────────▼───────────────────────────────────────┐
│ starlight-runtime（可执行文件，发行版形态）             │
│   = sl-daemon + backend + 模型加载 + tokenizer          │
└───────────────┬───────────────────────────────────────┘
                │ Rust API
┌───────────────▼───────────────────────────────────────┐
│ sl-core（Engine / Session / step）                     │
│   sl-graph 编译结果 + sl-device 设备 + 权重             │
└───────────────┬───────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────┐
│ backend-*（CPU / Vulkan / CUDA ...）                    │
└───────────────────────────────────────────────────────┘
```

职责一句话：

| 层 | 只做 | 绝不做 |
| --- | --- | --- |
| `sl-core` | 一次或多次 `step` | HTTP、队列、鉴权、tokenizer、采样策略 |
| `sl-daemon` | 请求生命周期、批处理、OpenAI 协议、流式、取消 | 算子、图编译、设备分配 |
| `starlight-runtime` | 把上述组件组装成一个可执行文件 | 多节点、租户、部署编排 |
| InfrArray | 节点、放置、多服务、UI、网关策略 | 模型解析、算子、协议翻译 |

## 3. `sl-core` 契约

### 3.1 原则

- 同步、无 I/O、无网络、无全局状态；
- `step` 不分配堆内存；
- 不知道 tokenizer、采样、停止条件；
- 不知道请求来自哪里，也不关心。

### 3.2 类型（目标）

```rust
pub struct EngineConfig {
    pub prefill: CommandBuffer,
    pub decode: CommandBuffer,
    pub weights: Vec<ParamBinding>,
    pub device: Arc<dyn Device>,
    pub max_batch: u32,
    pub max_context: u32,
}

pub struct Engine;

impl Engine {
    pub fn build(config: EngineConfig) -> Result<Self, BuildError>;

    /// 执行一次 prefill 或 decode。
    ///
    /// - `mode` 决定使用 prefill 还是 decode 程序；
    /// - `input` 提供 token/position/长度；
    /// - `output` 是预分配的 logits 缓冲；
    /// - 不分配、不阻塞、不回调。
    pub fn step(
        &mut self,
        mode: StepMode,
        input: StepInput<'_>,
        output: &mut [f32],
    ) -> Result<StepStats, StepError>;

    pub fn info(&self) -> EngineInfo;
}

pub enum StepMode { Prefill, Decode }

pub struct StepInput<'a> {
    pub token_ids: &'a [u32],
    pub positions: &'a [u32],
    pub seq_len: u32,
    pub past_len: u32,
}

pub struct StepStats {
    pub tokens: u32,
    pub steps: u32,
}

pub struct EngineInfo {
    pub engine_version: &'static str,
    pub ir_version: u32,
    pub max_context: u32,
    pub max_batch: u32,
    pub architectures: Vec<String>,
}
```

### 3.3 边界

- `Engine` 不做 token 循环；循环在 `sl-daemon`。
- `Engine` 不做采样；logits 原样返回。
- `Engine` 不持有 KV cache 的内存分配策略；KV 是 `sl-device` 上的 State 缓冲，由构建期分配。
- `Engine` 的错误只分两类：构建失败（模型、图、设备、能力）与步进失败（绑定、设备）。
- 热路径必须通过 allocator 门禁；`step` 的签名里不出现 `Vec`/`Box`/`String`。

### 3.4 依赖

```text
sl-core → sl-ir, sl-graph, sl-device
```

`sl-core` **不依赖** `sl-model`、`sl-model-adapters`、`sl-daemon`、任何 backend。模型加载与 lowering 在 runtime 层完成，`EngineConfig` 只接收已编译的 `CommandBuffer` 与权重绑定。

## 4. `sl-daemon` 契约

### 4.1 职责

- 持有一个 `Engine`；
- 接收 HTTP 请求，管理请求生命周期；
- 维护会话与 KV cache 的分配与回收；
- 做 tokenize / detokenize / 采样 / 停止条件；
- 输出 OpenAI 兼容响应与 SSE；
- 暴露健康、就绪、指标；
- 不关心多节点、租户、部署清单。

### 4.2 进程内模型

```text
HTTP 请求
  → 准入（队列长度、上下文长度、并发上限）
  → 组装 StepInput
  → 循环：Engine.step(Decode/Prefill) → 采样 → 停止判断
  → 流式输出 / 一次性响应
  → 释放会话与 KV
```

v1 的并发模型：

- 单 Engine，`step` 串行；
- 有界请求队列，满时返回 429；
- 同一时刻只处理一个 batch；continuous batching 在 v2；
- 请求取消在 step 边界生效；已提交的 step 不中断。

### 4.3 OpenAI 兼容面

```text
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
GET  /v1/models
GET  /healthz
GET  /readyz
GET  /metrics
```

- 协议实现只此一处；InfrArray Gateway 透传，不改语义；
- 响应头 `x-starlight-engine`、`x-starlight-version`；
- 错误映射到 OpenAI 错误结构；
- 流式统一使用 SSE，结束发送 `[DONE]`。

### 4.4 tokenizer 与采样

这两件事需要一个新组件 `sl-tokenizer`：

```rust
pub struct Tokenizer;
impl Tokenizer {
    pub fn from_model_bundle(bundle: &ModelBundle) -> Result<Self, TokenizerError>;
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, TokenizerError>;
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
}
```

- `sl-tokenizer` 是独立 crate，不依赖 `sl-core`；
- 词表来源是 `ModelBundle` 的附属资产（tokenizer.json / vocab）；
- 采样器（temperature/top-p/top-k/惩罚/停止串）放在 `sl-daemon`，不进 `sl-core`；
- 多模态预处理不属于 `sl-daemon` 的 v1 范围。

### 4.5 配置

```toml
[model]
path = "/var/lib/infrarray/models/qwen-14b"
architecture = "qwen2"
format = "safetensors"

[engine]
backend = "cpu"          # cpu | vulkan | cuda（后续）
max_context = 32768
max_batch = 1

[kv]
dir = "/var/lib/infrarray/kv"
mode = "contiguous"

[server]
bind = "127.0.0.1"
port = 8000
api_key = ""             # 为空时只允许本机访问
```

### 4.6 错误分类

| 类别 | 例子 | HTTP |
| --- | --- | --- |
| 准入 | 队列满、上下文超限 | 429 / 400 |
| 请求 | 参数非法、模型不支持 | 400 |
| 引擎 | 步进失败、设备错误 | 500 / 503 |
| 运行时 | 未就绪、正在关闭 | 503 |

## 5. `starlight-runtime` 进程协议

`starlight-runtime` 是由 `sl-daemon` + 一个 backend + 模型加载器组装出的可执行文件，也是 InfrArray 唯一需要认识的 StarLight 产物。

### 5.1 子命令

```text
starlight-runtime describe --config <path>   输出 JSON
starlight-runtime serve --config <path>      启动 HTTP 服务
starlight-runtime run --config <path> --prompt ...   一次性推理
starlight-runtime validate --config <path>   校验配置与模型
starlight-runtime version                    输出版本
```

### 5.2 `describe` 输出

```json
{
  "engine": "starlight",
  "engine_version": "0.1.0",
  "ir_version": 1,
  "runtime_version": "0.1.0",
  "backend": "cpu",
  "architectures": ["qwen2"],
  "max_context": 32768,
  "max_batch": 1,
  "quantization": "int4",
  "capabilities": ["text", "streaming"]
}
```

规则：

- `describe` 不加载权重，只读配置与元数据；
- 字段只增不改；未知字段消费方必须忽略；
- `engine` 是 `starlight` 或 `compatible`，由构建时的 runtime 决定；
- `ir_version` 用于控制面做兼容检查。

### 5.3 生命周期

```text
启动 → 读配置 → 加载模型 → 构建 Engine → 绑定端口
     → /readyz 200 → 可接收请求
     → SIGTERM → 停止接新请求 → 等待在途 → 退出
```

- `serve` 在前台运行，日志写 stdout/stderr；
- `--config` 是唯一输入，禁止通过环境变量偷偷改变语义；
- 退出码：0 正常，2 配置错误，3 模型错误，4 设备错误，5 端口占用。

## 6. InfrArray 如何配合

### 6.1 Agent 侧

1. Agent 收到 Service 的期望状态；
2. 生成 `runtime.toml`；
3. 调用 `starlight-runtime describe` 获取能力与版本；
4. 调用 `starlight-runtime serve` 拉起进程；
5. 轮询 `/readyz`，通过后把 `host:port` 注册给 Controller；
6. 转发 `/metrics` 到本地采集；
7. 进程退出时按退避策略重启，并上报原因。

Agent **不解析** OpenAI 请求，也不做 tokenize。

### 6.2 Gateway 侧

- 只做三件事：按 service 路由、校验 API key、按租户/密钥限额；
- 请求体与响应体**不解析、不重写**（除必要的头）；
- 从 runtime 的 `describe` 结果中取 `engine` 与版本，附加 `x-infrarray-engine` 响应头；
- 服务不健康时直接 503，不重试到其他节点（v1 无跨节点副本）。

### 6.3 Controller 侧

- 认证矩阵检查：`describe` 返回的 `engine_version`、`backend`、`architectures` 必须命中矩阵；
- 未命中时，Service 的 `engine: starlight` 被拒绝，返回 `EngineNotCertified`；
- `engine: auto` 时允许回落到 `compatible`，并在 Service 状态中记录实际引擎。

### 6.4 版本与升级

- Agent 启动前比对 `runtime_version` 与 Controller 期望版本；
- 升级采用“先拉新二进制 → 启动新进程 → 就绪后摘旧进程”，不原地替换；
- Service 运行期间不自动升级；由 Deployment 变更触发。

## 7. 模块化的验收方式

用一个产品需求验证四层是否真的解耦：

| 变更 | 应改动 | 不应改动 |
| --- | --- | --- |
| 新增 CUDA backend | 只新增 `backend-cuda` | `sl-core`、`sl-daemon`、InfrArray |
| 换 CPU 内核实现（标量→SIMD） | 只改 `backend-cpu` | 同上 |
| 新增模型家族 | `sl-model-adapters` + 支持矩阵 | `sl-core`、InfrArray |
| 换调度策略（批处理） | `sl-daemon` | `sl-core`、InfrArray |
| 换控制面（InfrArray → 自研） | 只换控制面 | runtime 与内核 |
| 脱离 InfrArray 单独用 | 直接 `starlight-runtime serve` | 不需要任何控制面 |

只要其中任何一行需要跨层改动，就说明边界破了，应回到本文修正契约。

## 8. 对 InfrArray 的约束

InfrArray 可以做：节点纳管、放置、生命周期、网关策略、UI、计量、审计。

InfrArray 不可以做：

- 解析 Safetensors 或 tokenizer；
- 直接链接 `sl-core`、`sl-graph`、`sl-device` 或 backend；
- 实现第二套 OpenAI 协议；
- 在 Gateway 里做 tokenize 或采样；
- 修改 runtime 的配置文件之外的状态。

## 9. 与现有文档的差异

本文对 PRODUCT-0001 做一处修正：

- PRODUCT-0001 原先把 OpenAI 协议放在 InfrArray Gateway；本文改为**由 `sl-daemon` 实现，Gateway 透传**。理由是避免两套协议实现，并让 StarLight 脱离 InfrArray 也能作为独立服务使用。

需要在 PRODUCT-0001 中同步修改 §8（架构图与组件表）与 §11（API 归属）。

## 10. 实施顺序

| 步骤 | 交付 | 依赖 |
| --- | --- | --- |
| 1 | `sl-core` Engine/Session 与 `step` | `sl-graph` 编译结果、`sl-device` |
| 2 | `sl-tokenizer` | `sl-model` 附属资产 |
| 3 | `sl-daemon` HTTP + OpenAI + 队列 | 1、2 |
| 4 | `starlight-runtime` 打包与 `describe/serve` | 3 |
| 5 | InfrArray Agent 对接进程协议 | 4 |

每一步都必须有独立可运行的验证：`cargo test`、`curl`、或 `describe` 输出快照。

## 11. 验收

1. 不装 InfrArray，只用 `starlight-runtime serve` + curl 能完成一次对话；
2. InfrArray 不链接任何 StarLight 执行 crate，能完成拉起、健康、路由、重启；
3. 把 `runtime-compatible` 换成 `runtime-starlight` 时，控制面零改动；
4. 换 backend 时，`sl-daemon` 与 InfrArray 零改动；
5. `step` 的 allocator 门禁通过。

## 12. 待决问题

1. `sl-tokenizer` 是自研还是封装 `tokenizers` crate？
2. v1 是否支持并发请求（单 Engine + 队列）还是严格串行？
3. continuous batching 放到 v2 还是 v1 后段？
4. `describe` 用 JSON 还是带版本的 protobuf？
5. 多模型是否允许一个进程内共存（v1 明确不允许，v2 再评估）？
6. Gateway 是否需要支持非 OpenAI 的私有协议？
