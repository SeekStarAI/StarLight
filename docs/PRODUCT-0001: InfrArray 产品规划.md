# PRODUCT-0001：InfrArray 产品规划

**文档编号：** PRODUCT-0001
**状态：** 产品规划 v1.0（已定稿，进入执行）
**日期：** 2026-10-17
**产品名：** **InfrArray**
**内核：** StarLight（SeekStarAI）
**一句话：** InfrArray 把一台或多台 GPU 机器，变成一个开箱即用的多模型推理平台。
**品牌语：** InfrArray, powered by StarLight.
**前置文档：** [ARCH-0002](<ARCH-0002: 模块边界与发行版业务管线契约.md>)、[ARCH-0004](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)、[ARCH-0005](<ARCH-0005: 服务面契约 sl-core sl-daemon 与 InfrArray.md>)、[RFC-0002](<RFC-0002: 采纳 sl-ir 作为图 IR 与设备命令 ABI.md>)

> 本文是产品规划，不是能力清单。StarLight 内核的当前实现状态以主 README 和各 crate 文档为准；本文中“v1 必须”“v1 不做”是约束，不是愿望。

## 0. 定稿决定

| # | 决定 | 结论 |
| --- | --- | --- |
| D1 | 交付形态 | **私有化 + 托管云双形态**，同一套代码库 |
| D2 | 运行时 | **运行时契约可插拔：首发 vLLM / TensorRT-LLM，StarLight 按认证矩阵逐块替换** |
| D3 | 开闭源 | **核心全开源（Apache-2.0）**，无功能阉割版 |
| D4 | 首发硬件 | **NVIDIA + Linux 单平台** |
| D5 | 多机 | **v1 即支持多机纳管**；但 v1 不做跨节点并行 |
| D6 | 画布与服务 | 同时交付，**manifest-first** |
| D7 | 首个演示 | **dense 7B–14B + INT4/INT8** |

## 1. 产品定义

### 1.1 一句话

InfrArray 是一个**图形化的推理基础设施控制面**：启动后自动发现设备，拉取模型，用节点画布把一个或多个推理服务编排到指定节点与 GPU 组，选择量化与运行时，并对外导出 OpenAI 兼容 API。

### 1.2 目标客户（ICP）

| 客户类型 | 规模 | 痛点 |
| --- | --- | --- |
| 私有化部署企业 | 1–8 台 GPU 服务器，2–64 卡 | 没有专职 MLOps，模型上线靠 shell 与文档 |
| 系统集成商 / 行业 ISV | 多客户交付 | 每个客户现场重搭一遍，交付不可复制 |
| 托管云 / SaaS 提供商 | 多租户 | 需要按租户隔离、计量、限额的推理平台 |

### 1.3 价值主张

1. **15 分钟上线**：裸机到第一个 token，不需要写 YAML、不需要 K8s。
2. **一张画布管所有模型**：多模型、多节点、多 GPU 组，服务之间不打架。
3. **可导出的部署清单**：画布产出的是版本化 `DeploymentSpec`，可 diff、可进 Git、可重放。
4. **引擎可替换**：同一份清单可以跑在 vLLM、TensorRT-LLM 或 StarLight 上；切换引擎不改部署形态，并在 UI/API 中如实标注。

### 1.4 非目标（v1 明确不做）

- 训练与微调；
- Kubernetes Operator（v2 再评估）；
- 第三方可执行插件 / 自定义节点代码；
- 非 NVIDIA 硬件、非 Linux 系统；
- 跨节点张量并行或流水并行；
- 模型市场与计费市场；
- 自托管版本的用量计费（云版本才有计费）。

## 2. 产品形态

### 2.1 两种交付形态

| 形态 | 组成 | 数据面 | 目标 |
| --- | --- | --- | --- |
| **InfrArray Self-hosted** | `infrarray` 控制面二进制（内嵌 Web UI）+ 每节点 `infrarray-agent` | 客户机房，SQLite 元数据，本地模型库 | 私有化交付 |
| **InfrArray Cloud** | 同一控制面 + Postgres + 对象存储 + 多租户网关 | 托管区域，按租户隔离 | 订阅制云服务 |

两种形态共用同一份代码与同一个 `DeploymentSpec`；云版本只增加租户、计量、计费和隔离能力，不改变调度语义。

### 2.2 版本与授权

核心全部以 **Apache-2.0** 开源，**不存在功能阉割的社区版**。商业差异只体现在服务，不体现在代码：

| 层级 | 内容 | 收费方式 |
| --- | --- | --- |
| Self-hosted | 全部功能，自行运维 | 免费 |
| Enterprise Support | SLA、7×24、合规材料、升级保障、培训 | 年费订阅 |
| InfrArray Cloud | 托管运行、弹性、按量计费 | 用量计费 |
| Certified Appliance | 预装认证硬件的一体机 | 硬件 + 服务 |

这条决定意味着：**销售不能靠 license 卡功能，只能靠交付、运维与云服务赚钱。**

### 2.3 安装与启动

```sh
curl -fsSL https://get.infrarray.com | sh
infrarray up
# InfrArray is running at http://localhost:8080
```

- 单机：一条命令启动控制面，UI 内嵌，无需额外服务；
- 多机：控制面节点执行 `infrarray up`，其余节点执行 `infrarray join <token>`；
- 离线：提供 tarball 与 OCI 镜像，支持 air-gapped 导入模型；
- 容器：官方镜像 `infrarray/infrarray`，支持 Docker Compose 单机与多机。

### 2.4 Web UI 信息架构

| 页面 | 内容 |
| --- | --- |
| 首页 | 集群健康、GPU 利用率、服务状态、告警 |
| 画布 | 节点式编排、连线、校验、应用 |
| 资产 | 模型、tokenizer、量化配置、下载任务 |
| 服务 | 服务详情、副本、日志、指标、重启、API 信息 |
| 节点 | 设备清单、驱动、拓扑、模型缓存 |
| 设置 | 用户、密钥、存储路径、网关、审计 |

首发中文优先，英文同版本；所有页面支持只读链接与深链接。

### 2.5 CLI

```text
infrarray up                         启动控制面
infrarray join <token>               加入集群
infrarray apply -f deployment.yaml   应用清单
infrarray get services|nodes|assets  查看状态
infrarray logs <service> [-f]        查看日志
infrarray port-forward <service>     本地转发
infrarray export <deployment>        导出清单
infrarray doctor                     环境诊断
```

CLI 与 UI 操作的是同一份清单和同一套 API。

## 3. 与 StarLight 的关系（定稿）

### 3.1 三层结构

```text
InfrArray（产品层，Apache-2.0）
    │  控制面、画布、网关、资产、调度
    ▼
runtime 插槽（统一 describe / serve / health / metrics 协议）
    ├── runtime-vllm           首发
    ├── runtime-tensorrt-llm   首发（NVIDIA 可选）
    └── starlight-runtime      按认证矩阵逐块替换
            ▼
        StarLight（引擎上游，MIT/Apache-2.0）
        sl-model / sl-ir / sl-graph / sl-core / sl-device / backend-*
```

### 3.2 关系定义

1. **InfrArray 不是 StarLight 的发行版**。InfrArray 是独立产品；它通过统一的 runtime 协议驱动推理进程，StarLight 只是其中一种实现。
2. **运行时是插槽，不是绑定**：InfrArray 首发用 vLLM / TensorRT-LLM 交付；StarLight 是目标一等引擎，按认证矩阵逐块替换。替换的是“某个模型家族 × 某类硬件 × 某种精度”的组合，不是一次性切换全部。
3. **依赖方向单向**：InfrArray 依赖 StarLight 的公开进程协议；StarLight 的任何 crate 不得依赖 InfrArray。
4. **不 fork**：引擎侧的问题一律向上游提交；InfrArray 仓库不维护 StarLight 的私有分支。
5. **版本锁定**：每个 InfrArray 版本声明每个 runtime 实现的版本范围与认证矩阵（模型家族 × 硬件 × 精度 × runtime 版本）。
6. **能力门禁**：只有进入认证矩阵的组合才允许标注 `engine: starlight` / `vllm` / `tensorrt`；`auto` 按矩阵优先级回落，并在 UI、日志、API 响应头中标注实际实现。
7. **性能口径**：对外性能数字必须注明 runtime 与版本；vLLM / TensorRT-LLM 的成绩不得归因于 StarLight，反之亦然。
8. **贡献分工**：内核、算子、图编译、量化算法进 StarLight；控制面、调度、UI、网关、runtime 适配进 InfrArray。
9. **发布节奏独立**：StarLight 按内核里程碑发布；InfrArray 按月发布；两者通过 runtime 协议版本与认证矩阵解耦。

### 3.3 运行时适配层

`infrarray-runtime` 是 Agent 拉起的一个受监管进程，接口固定为：

```text
runtime describe                 -> JSON: engine, version, capabilities, models
runtime serve --config <path>    -> 启动 HTTP 推理服务
runtime health                   -> 200 / 503
runtime metrics                  -> Prometheus 文本
```

三个实现，共用同一套协议：

| 实现 | 何时使用 | 对外表述 |
| --- | --- | --- |
| `runtime-vllm` | v1 默认；dense 文本模型 | “vLLM 运行时” |
| `runtime-tensorrt-llm` | NVIDIA + 追求吞吐的客户 | “TensorRT-LLM 运行时” |
| `runtime-starlight` | 命中 StarLight 认证矩阵的组合 | “StarLight 引擎” |

认证矩阵是唯一开关。矩阵之外，不允许手工把某个组合标成 `starlight`；切换必须经过矩阵与契约测试。

### 3.5 迁移策略：vLLM / TensorRT-LLM → StarLight

迁移的单位是**认证矩阵的一格**，不是整个产品：

```text
1. 契约测试：同一请求在 runtime-vllm 与 runtime-starlight 上输出一致（容差内）
2. 影子流量：同一 Service 双跑，主流量仍走旧实现，对比延迟与输出
3. 灰度：按 Service 或租户切 1% → 10% → 100%
4. 回滚：清单里把 engine 改回 vllm/tensorrt 即可，无需改架构
5. 下线：旧实现只在没有认证组合依赖时移除
```

四条硬规则：

1. **协议先于实现**：`describe/serve/health/metrics` 的字段与语义在 vLLM 适配时就要冻结，不能等 StarLight 再补。
2. **适配层薄**：`runtime-vllm` 只做“启动参数映射 + 协议转换”，不允许把 vLLM 的私有能力泄漏进清单或 UI。
3. **能力用矩阵表达**：`describe` 必须报告 `engine`、版本、`architectures`、`max_context`、`quantization`、`parallelism`（tp/pp/是否跨节点）、`topology` 要求；控制面据此放置，而不是假设。
4. **性能分开记账**：迁移前后分别记录 TTFT/TPOT/吞吐，作为是否扩大认证矩阵的依据。

### 3.4 与内核的契约

InfrArray 只通过以下稳定接口使用 StarLight：

- `starlight-runtime` 可执行文件与上述 `describe/serve/health/metrics` 协议；
- `sl-model` 的模型包读取；
- `sl-model-adapters` 的家族描述与内存估算元数据；
- `sl-precision` 的量化产物格式（经 `starlight-runtime` 消费，而不是 InfrArray 直接调用）；
- `sl-ir` 的版本号，用于兼容性检查。

InfrArray 不直接链接 `sl-core`、`sl-graph`、`sl-device` 或任何 backend；这些由 `starlight-runtime` 内部组装。

## 4. 用户与场景

### 4.1 角色

| 角色 | 关心 | 权限 |
| --- | --- | --- |
| 平台管理员 | 节点、GPU、模型、密钥 | 全部 |
| 业务开发者 | 自己的服务与 API | 命名空间内 |
| 运维 | 告警、日志、重启 | 只读 + 运维操作 |

### 4.2 场景 A：单机 8 卡跑两个模型

- 0–3 卡：Qwen 14B，INT4，OpenAI API `chat`；
- 4–7 卡：嵌入模型，INT8，OpenAI API `embeddings`；
- 两个服务独立进程、独立端口、独立 KV cache 目录；
- 画布上就是两个 Server 节点 + 两个 Device Pool 节点。

### 4.3 场景 B：三台机器纳管

- 控制面跑在 node-a；
- node-b、node-c 通过 `infrarray join` 加入；
- 服务可以落在任意节点的 GPU 组，但**一个服务的所有 GPU 必须位于同一节点**；
- 网关统一入口，按 service 路由。

### 4.4 场景 C：托管云租户

- 租户创建 Workspace，获得独立命名空间与 API key；
- 模型缓存按租户隔离；
- 用量按 token 计量，出账单；
- 平台管理员可看到全局 GPU 利用率和租户配额。

## 5. 核心概念

| 概念 | 层级 | 含义 |
| --- | --- | --- |
| Cluster | 全局 | 一组被纳管的节点 |
| Node | 集群 | 一台 NVIDIA + Linux 机器，运行 Agent |
| Device Pool | 节点 | 节点内 GPU 的逻辑分组 |
| Workspace | 租户/项目 | 服务与资产的隔离边界 |
| Asset | 工作区 | 模型、tokenizer、量化配置 |
| Service | 工作区 | 一个模型服务：模型 + 放置 + 量化 + 运行时 + API |
| Deployment | 工作区 | 一组 Service 的期望状态 |
| Endpoint | 服务 | 对外暴露的 API |
| Run | 服务 | 一次具体执行，有健康、日志、指标 |

## 6. 节点体系

节点是**部署资源**，不是计算算子。边分两类：数据流边与放置约束边。

| 类别 | 节点 | 输入 → 输出 | 说明 |
| --- | --- | --- | --- |
| Asset | Model | — → model | HF repo/revision 或本地路径 |
| Asset | Tokenizer | model → tokenizer | 可与模型绑定 |
| Asset | Quantization | model → quantized model | INT4/INT8；后续 FP8/FP4 |
| Infra | Device Pool | — → devices | 节点 + GPU 集合 |
| Infra | Placement | devices + model → placement | 副本、节点选择、显存上限 |
| Service | **Inference Server** | model + placement → endpoint | 唯一可导出 API 的节点 |
| IO | Prompt Preset | — → prompt | 预置提示词/系统提示词 |
| IO | File Input | — → document | 文本/Embedding 文件 |
| IO | Multimodal Input | — → image/audio | v2 |
| Ops | Gateway | endpoints → endpoint | 路由、鉴权、配额 |
| Ops | Health Check | endpoint → status | 存活/就绪 |
| Ops | Autoscaler | endpoint → replicas | v2 |

画布校验规则（不通过不允许 Apply）：

1. 无环；
2. 端口类型匹配；
3. GPU 不超卖；
4. 显存估算不超预算；
5. Server 节点必须有唯一的 model 与 placement；
6. 一个服务不得跨节点使用 GPU。

## 7. 部署清单（v1 冻结）

```yaml
apiVersion: infrarray.com/v1
kind: Deployment
metadata:
  name: prod-a
  workspace: default
spec:
  assets:
    models:
      - name: qwen-14b
        source:
          type: huggingface
          repo: Qwen/Qwen2.5-14B-Instruct
          revision: main
        files: ["*.safetensors", "config.json", "tokenizer*"]
  services:
    - name: chat
      model: qwen-14b
      placement:
        node: node-a
        gpus: [0, 1, 2, 3]
        replicas: 1
        tensor_parallel: 1      # v1 必须为 1
        pipeline_parallel: 1    # v1 必须为 1
      quantization:
        scheme: int4
        group_size: 128
        keep: ["model.embed_tokens.weight"]
      runtime:
        engine: auto            # auto | vllm | tensorrt | starlight
        max_context: 32768
        kv_cache:
          dir: /var/lib/infrarray/kv
          mode: contiguous      # paged 为 reserve
        batching:
          max_batch: 8
      api:
        protocol: openai
        port: 8000
        auth:
          type: api_key
          secret: chat-keys
```

规则：

- `tensor_parallel`、`pipeline_parallel` 在 v1 只接受 1；大于 1 返回 `UnsupportedPlacement`，不静默串行；
- `engine: starlight` 只在认证矩阵内被接受；其余情况返回 `EngineNotCertified`，`auto` 按矩阵优先级回落到 vllm/tensorrt；
- schema 版本随字段变化升版；未知字段拒绝，不做静默忽略；
- 清单可导出、可导入、可 diff，UI 不产生清单之外的私有状态。

## 8. 架构

```text
┌────────────────────────── InfrArray ──────────────────────────┐
│  Web UI（内嵌 SPA）                                            │
│  Control API  ── 清单校验 / 版本 / 鉴权 / 审计                  │
│  Controller   ── 调度 / 期望状态收敛 / 漂移检测                 │
│  Hub          ── HF 下载 / 校验 / 缓存 / 量化派生               │
│  Gateway      ── 路由 / key / 配额 / 用量（协议透传）            │
│  Metadata     ── SQLite（self-hosted）/ Postgres（cloud）       │
└───────────────┬───────────────────────────────┬───────────────┘
                │ mTLS，Agent 主动外连            │
      ┌─────────▼─────────┐          ┌──────────▼────────┐
      │ Agent（node-a）    │          │ Agent（node-b）    │
      │ 设备扫描 / 进程监管 │          │ ...               │
      │ 端口分配 / 模型缓存 │          │                   │
      │ runtime-vllm       │          │                   │
      │ runtime-tensorrt   │          │                   │
      │ runtime-starlight  │          │                   │
      └────────────────────┘          └───────────────────┘
```

### 8.1 组件职责

| 组件 | 职责 | 不做什么 |
| --- | --- | --- |
| `infrarray-web` | 画布、向导、服务面板、日志 | 不直接调用运行时 |
| `infrarray-api` | 清单校验、版本、鉴权、审计 | 不实现调度 |
| `infrarray-controller` | 放置、收敛、重启、漂移检测 | 不解析模型文件 |
| `infrarray-agent` | NVML 扫描、进程监管、端口、缓存、健康 | 不做跨节点调度 |
| `infrarray-hub` | HF 下载、断点续传、哈希、许可证检查 | 不做推理 |
| `infrarray-gateway` | 多服务路由、密钥、配额、计量；透传 OpenAI 协议 | 不实现 OpenAI 协议、不做 tokenize/采样 |
| `infrarray-runtime` | 启动并监管引擎进程 | 不含模型逻辑 |

### 8.2 控制循环

```text
用户 Apply → API 校验清单 → Controller 计算放置
          → Agent 拉取模型/量化产物 → 启动 runtime
          → 健康通过 → 注册 Endpoint → Gateway 可路由
          → 持续上报状态 → 漂移则重建
```

所有操作幂等；重复 Apply 不产生副作用。删除 Deployment 采用优雅停止：先摘除路由，再等待在途请求，最后停止进程。

### 8.3 多机模型

- **v1 是管理面多机**：Controller 管多个 Node，但一个 Service 的全部 GPU 必须在同一节点；
- Agent 只主动外连 Controller（mTLS），不需要入站端口，适配 NAT 与防火墙；
- Controller 在 v1 是单实例 + SQLite；HA（3 节点 + Postgres）在 v2；
- 模型分发：Agent 直接对 HF 下载或从 Controller 的 Hub 缓存拉取；v2 增加节点间 P2P；
- 跨节点推理流量在 v1 不存在；Gateway 只做 `service → 节点地址` 的 HTTP 代理。

### 8.4 存储

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| 元数据 | SQLite / Postgres | Deployment、Service、Run、审计 |
| 模型 | 节点本地目录（可配 NFS） | 由 Hub 管理，哈希校验 |
| 量化产物 | 同模型目录，派生资产 | 记录来源哈希与参数 |
| KV cache | 节点本地 NVMe（可配） | 每服务独立目录 |
| 日志 | 本地滚动 + 可选 Loki | 默认保留 7 天 |

## 9. 资源与调度

### 9.1 v1 调度模型

- 一个 Service = 一个受监管进程 = 一个节点上的一个 GPU 组；
- 副本可以跨节点，但单个副本不跨节点；
- 不做抢占；资源不足时进入 Pending 并给出可操作原因（差多少显存、缺哪类卡）；
- 端口由 Agent 分配，避免冲突。

### 9.2 放置算法

1. 过滤：节点在线、驱动版本、GPU 型号、显存、端口、模型缓存；
2. 估算：`weights + kv_cache + activations + overhead`；
3. 约束：`gpus.len() == 1`（v1），TP/PP 必须为 1；
4. 打分：显存余量、缓存命中、负载；
5. 失败时返回结构化原因，UI 直接展示“需要 42 GiB，最大可用 38 GiB”。

### 9.3 显存估算

```text
weights   = params * bits / 8
kv_cache  = 2 * layers * kv_heads * head_dim * max_context * batch * bytes_per_elem
activation ≈ batch * max_context * hidden * bytes_per_elem * k
total     = weights + kv_cache + activation + overhead
```

`layers`、`kv_heads`、`head_dim`、`k` 由 `sl-model-adapters` 的家族描述提供；未认证家族使用保守系数，并在 UI 标注“估算”。

### 9.4 并行路线

| 版本 | 能力 | 范围 |
| --- | --- | --- |
| v1 | 进程隔离 + GPU 组 | 单节点 |
| v2 | 流水并行（按层切分） | 单节点 |
| v3 | 张量并行（权重分片 + 集合通信） | 单节点 |
| v4 | 跨节点并行 | 多节点 |

v1 的 UI 必须明确写出“当前为单卡执行”，避免用户误以为已经做了多卡并行。

## 10. 模型资产与量化

### 10.1 资产获取

- 支持 Hugging Face repo + revision + 文件 glob；
- 仅接受 Safetensors；拒绝 pickle；
- 断点续传、SHA-256 校验、大小与并发限制；
- 记录许可证元数据；商用前提示确认模型许可；
- 离线导入：`infrarray asset import --path /media/models/qwen-14b`。

### 10.2 量化

| 方案 | v1 | 说明 |
| --- | --- | --- |
| INT4 groupwise | 支持 | 基于 `sl-precision` 的对称分组量化 |
| INT8 groupwise | 支持 | 同上 |
| FP8 / MXFP8 / NVFP4 | 不支持 | v3 之后，依赖新 kernel 与硬件 |

- 量化产物是派生资产，必须记录：来源哈希、scheme、group size、keep 列表、工具版本；
- 量化失败或冒烟推理失败时，回退未量化资产并标注原因；
- v1 不做自动精度评估，只做加载 + 冒烟生成。

## 11. 服务与 API

### 11.1 OpenAI 兼容面

**协议由 `starlight-runtime`（即 `sl-daemon`）实现，InfrArray Gateway 只做路由与鉴权后转发**，详见 ARCH-0005。

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
```

- 支持 SSE 流式输出、`finish_reason`、usage；
- 错误码与 OpenAI 对齐；
- 响应头 `x-infrarray-engine: vllm|tensorrt|starlight`，客户端可感知引擎；
- 多模态输入在 v2 加入，v1 只接受文本与 token 数组。

### 11.2 管理面

```text
/api/v1/deployments
/api/v1/services
/api/v1/assets
/api/v1/nodes
/api/v1/runs
/api/v1/keys
```

### 11.3 服务配置

- `max_context`：服务级，受模型上限约束；
- `kv_cache.dir`：每服务独立目录；
- `batching.max_batch`：v1 为静态批上限；
- `engine`：`auto | vllm | tensorrt | starlight`；
- `replicas`：同一 Service 的副本数，可跨节点。

## 12. 安全模型

1. **v1 不执行第三方代码**：没有自定义节点、没有插件、没有任意脚本；
2. Agent 只外连 Controller，mTLS，双向证书；
3. API key 加密存储，日志脱敏，支持按 key 的配额与速率限制；
4. 模型供应链：Safetensors only、哈希固定、许可证记录、来源可追溯；
5. 发布物：签名二进制、SBOM、校验和；
6. 云多租户：命名空间隔离、模型缓存隔离、网关按租户限额、审计日志；
7. 后续插件模型：签名包 + 能力清单（可访问设备/目录/网络）+ 沙箱，三者缺一不可。

## 13. 可观测性

- 指标：GPU 利用率/显存、TTFT、TPOT、tokens/s、队列深度、错误率、重启次数；
- 日志：按服务聚合，支持下载；
- 运行历史：每次 Run 的引擎、版本、配置摘要、失败原因；
- 告警：服务不健康、显存不足、模型缺失、证书过期；
- 导出：Prometheus `/metrics`，后续接 OpenTelemetry。

## 14. 版本与兼容

| 维度 | 策略 |
| --- | --- |
| InfrArray | SemVer，按月发布 |
| Manifest | `apiVersion: infrarray.com/v1`，字段只增不改；不兼容变更升 v2 |
| 引擎 | 每个 InfrArray 版本声明 `starlight-runtime` 版本范围 |
| 认证矩阵 | 模型家族 × GPU 型号 × 精度 × 引擎版本 |
| 升级 | 先升级控制面，再滚动升级 Agent；服务不中断 |

## 15. 商业模式

核心全开源，收入来自四类服务：

1. **InfrArray Cloud**：托管推理，按 token 或 GPU 小时计费；
2. **Enterprise Support**：SLA、7×24、合规材料、升级保障；
3. **交付与集成**：私有化实施、网络与存储对接、模型迁移；
4. **认证一体机**：与硬件厂商联合认证，预装交付。

不做：功能阉割、license 锁、按 GPU 收费的自托管授权。

## 16. 路线图（12 个月）

| 时间 | 里程碑 | 交付 | 退出标准 |
| --- | --- | --- | --- |
| 2026 Q4 | M0 单机闭环 | Controller + Agent、设备扫描、单服务、vLLM 运行时、OpenAI API、向导式 UI、CLI apply | 裸机到首 token ≤ 15 分钟 |
| 2027 Q1 | M1 多机 + 画布 | `join` 多节点、自由画布、HF Hub、INT4/INT8、清单导入导出 | 3 节点纳管；双模型双卡组 ≤ 10 分钟 |
| 2027 Q2 | M2 StarLight 认证 | 首个 dense 7B–14B + NVIDIA 组合接入 `runtime-starlight`；指标与告警；Controller HA | 认证组合数值一致；StarLight 引擎可切换 |
| 2027 Q3 | M3 流水并行 + 云 alpha | 单节点流水并行、paged KV、云多租户 alpha、离线包 | 2 卡吞吐 ≥ 1.6× |
| 2027 Q4 | M4 张量并行 + 云 GA | 单节点张量并行、FP8、企业支持开售 | 与单卡基线数值一致；云 SLA 达标 |

关键依赖：M2 依赖 StarLight 的 M5（`sl-core` build/step + 首个家族端到端）。若内核延期，M2 顺延，M0/M1 不受影响。

## 17. 成功指标

| 指标 | 目标 |
| --- | --- |
| 单机首 token 时间 | ≤ 15 分钟 |
| 3 节点纳管时间 | ≤ 10 分钟 |
| 双模型 8 卡配置时间 | ≤ 10 分钟 |
| 重启后恢复 | ≤ 30 秒 |
| 网关 p95 开销 | < 5% |
| 无需 shell 的日常任务占比 | ≥ 90% |
| 认证组合中 StarLight 引擎占比（2027 Q4） | ≥ 50% |

## 18. 团队假设

该范围（双形态 + 多机 + 画布 + 开源）需要至少：

- 控制面/后端 3 人；
- 前端/画布 2 人；
- 平台与交付 2 人；
- 测试与 SRE 1 人；
- 产品/设计 1 人。

低于 6 人时必须砍范围：优先砍云形态，其次砍自由画布（保留模板）。

## 19. 风险与对策

| 风险 | 对策 |
| --- | --- |
| StarLight 不能推理真实模型 | 首发用 vLLM / TensorRT-LLM 交付；M2 用认证矩阵逐格切换，不阻塞 M0/M1 |
| 多机 + 双形态 + 画布同时做导致范围失控 | v1 冻结清单字段；不做跨节点并行；画布先模板后自由 |
| 全开源导致商业化压力 | 收入绑定云、支持、交付；不靠 license |
| 首发运行时是外部引擎，StarLight 价值被稀释 | 认证矩阵与性能对比公开；适配层保持薄；每季度扩大 StarLight 认证范围 |
| 自定义节点安全事件 | v1 不开放；后续签名 + 能力清单 + 沙箱 |
| 仅 NVIDIA 限制市场 | 明确支持矩阵；v3 评估 ROCm/国产卡 |
| 商标风险 | 发布前完成正式检索；先占名 |

## 20. 对 StarLight 的硬性需求

优先级是相对 **StarLight 接管第一个认证组合** 而言，不是 InfrArray 首发的前置条件——首发由 vLLM / TensorRT-LLM 交付。

| 需求 | 用途 | 优先级 |
| --- | --- | --- |
| `starlight-runtime` 可执行文件与 describe/serve/health/metrics 协议 | Agent 拉起引擎 | P0 |
| `describe` 扩展字段：`parallelism`、`topology`、`quantization`、`device` | 放置与认证矩阵 | P0 |
| `sl-daemon` 的 OpenAI 兼容服务（协议只实现一次） | 推理 API | P0 |
| `sl-tokenizer`（encode/decode，基于模型附属资产） | 文本请求 | P0 |
| `Engine`/`Session`/`step` 稳定 API | 运行时内部 | P0 |
| 首个 dense 7B–14B 家族端到端 | M2 认证 | P0 |
| 本地设备枚举（NVML 封装） | runtime 能力上报 | P1 |
| 家族内存估算元数据（层数、KV 维度、激活系数） | 放置算法 | P1 |
| 量化产物格式与校验 | INT4/INT8 资产 | P1 |
| 设备能力（显存、算力、拓扑） | 调度 | P1 |
| 指标端点 | 可观测性 | P2 |
| 单节点流水并行 | M3 | P3 |
| 单节点张量并行 | M4 | P4 |

## 21. 仓库、命名与占名

**产品仓库：** [SeekStarAI/infrarray](https://github.com/SeekStarAI/infrarray)。它与 StarLight 平级，**不作为 StarLight 的 submodule**；StarLight 不依赖该仓库，两者只通过 runtime 协议交互。

产品名 **InfrArray** 已定。`InfraFlow` 因存在活跃产品与商标申请弃用；`StarLight Canvas` 因已有同名节点式工作流项目弃用。`InfrArray` 在 npm、PyPI、crates.io、GitHub 与 `infrarray.com` 上均未被占用。

对外发布前完成：域名（`.com/.ai/.io/.dev`）、GitHub Org、npm scope、PyPI、crates.io、社交账号、Docker Hub、中国商标第 9/42 类。

## 22. 决策记录

| 日期 | 决定 |
| --- | --- |
| 2026-10-17 | 产品定位：以 StarLight 为内核的独立商业产品 |
| 2026-10-17 | MVP：画布与服务同时，manifest-first |
| 2026-10-17 | 首个演示：dense 7B–14B + INT4/INT8 |
| 2026-10-17 | 命名：InfrArray |
| 2026-10-17 | 交付：私有化 + 托管云双形态 |
| 2026-10-17 | 引擎：运行时契约可插拔，首发 vLLM / TensorRT-LLM，StarLight 逐格替换 |
| 2026-10-17 | 开源：核心全开源 Apache-2.0，无功能阉割 |
| 2026-10-17 | 硬件：NVIDIA + Linux |
| 2026-10-17 | 多机：v1 管理面多机，不做跨节点并行 |
