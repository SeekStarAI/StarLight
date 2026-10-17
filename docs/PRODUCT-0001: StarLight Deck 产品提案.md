# PRODUCT-0001：StarLight Deck 产品提案

**文档编号：** PRODUCT-0001
**状态：** 产品提案，待立项验证
**日期：** 2026-10-17
**产品名：** StarLight Deck（primary）；候选 `StarLight Studio`
**内部代号：** InfraFlow（仅内部使用，不对外）
**定位：** 以 StarLight 为推理内核的独立商业产品
**前置文档：** [ARCH-0002](<ARCH-0002: 模块边界与发行版业务管线契约.md>)、[ARCH-0004](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)、[RFC-0002](<RFC-0002: 采纳 sl-ir 作为图 IR 与设备命令 ABI.md>)

> 本文是产品方向的落地文档，不是已交付能力。文中所有“目标”“规划”“候选”均不代表当前实现。StarLight 内核现状见主 README 与各 crate 文档。

## 0. 摘要

StarLight Deck 是一个**面向单机多卡私有化部署的图形化推理基础设施控制面**：启动后自动扫描本机设备，从 Hugging Face 拉取模型，用节点画布把一个或多个推理服务编排到指定 GPU 组上，选择量化与并行策略，并导出 OpenAI 兼容 API。

它不是 StarLight 的发行版，而是**建立在 StarLight 之上的独立产品**：StarLight 提供推理内核，Deck 提供设备纳管、模型资产、放置调度、服务生命周期、API 网关和 UI。

首个可交付版本的边界（已确认）：

- **画布与服务同时做**，但两者共用同一份部署清单（manifest-first），避免两套逻辑；
- 首个演示场景：**dense 7B–14B + INT4/INT8**；
- 并行只做到**进程隔离 + GPU 绑定**，不做张量并行；
- 名字不用 `InfraFlow`，改用 `StarLight Deck`。

## 1. 名称核查

### 1.1 `InfraFlow` 不可用

| 冲突类型 | 证据 | 风险 |
| --- | --- | --- |
| 活跃商业产品 | `infraflow.app` 的 npm CLI `infraflow`（[npm registry](https://registry.npmjs.org/infraflow)，0.1.20，创建于 2026-10-10），功能是部署目录、看日志、隧道暴露本地服务 | 极高：同赛道、同语义 |
| 商标申请 | `INFRAFLOW` 美国商标申请，序列号 99210377，申请人 Foshan Jiuzhou Technology（[检索结果](https://furm.com/trademarks/infraflow-99210377)） | 高 |
| 开源项目 | GitHub 搜索 `infraflow` 共 **83** 个仓库，含 AWS 基础设施可视化、AI 基础设施看板、CI/CD（[GitHub 搜索](https://api.github.com/search/repositories?q=infraflow)） | 中高：搜索污染 |
| 其他产品 | InfraFlow ZeroCloud（AI mesh）、InfraFlow 矿机集装箱、InfraFlow SMTP 系统、AWS→Terraform 可视化工具（[Product Hunt](https://www.producthunt.com/products/infraflow)、[报道](https://devbytes.co.in/news/infraflow-lets-you-visually-design-aws-infrastructure-and-export-it-as-terraform)） | 中 |

结论：**不作为公开产品名**。即使法务允许，搜索与品牌识别成本也不划算。

### 1.2 `StarLight Canvas` 也不建议

GitHub 已存在中文节点式工作流项目 `nayanjohri-star/starlight-canvas`（“星盘智能画布”，节点式图片/视频工作流，AGPL-3.0），与“节点画布”语义直接冲突（[仓库](https://github.com/nayanjohri-star/starlight-canvas)）。

### 1.3 推荐名

| 候选 | npm | GitHub 精确匹配 | 评价 |
| --- | --- | --- | --- |
| **StarLight Deck** | 未占用（404） | 0 | 推荐：“控制台/甲板”语义贴合多服务编排；与 StarLight 品牌一致 |
| StarLight Studio | 未占用（404） | 17（多为同名小项目） | 备选：更通用，但重名更多 |
| StarLight Flow | 未核查 | 未核查 | 不推荐：“Flow”过于通用 |
| **InfrArray** | 未占用（404） | 0 | 候选：商标独特性最好，但拼写有歧义，见 §1.4 |

仓库与 crate 前缀建议：`sl-deck`（例如 `deck-api`、`deck-agent`、`deck-gateway`）。

> 以上为工程师级别的公开检索，不构成法律意见。对外发布前必须做正式商标检索（USPTO/EUIPO/CNIPA）、域名、应用商店与包管理器核查。

### 1.4 `InfrArray` 评估（2026-10-17）

可用性核查结果：

| 渠道 | 结果 |
| --- | --- |
| npm `infrarray` | 404，未占用 |
| PyPI `infrarray` | 404，未占用 |
| crates.io `infrarray` | 未占用 |
| GitHub 精确搜索 | 0 个仓库 |
| `infrarray.com` | Verisign RDAP 返回 404，未注册 |
| `infrarray.ai` / `.io` / `.dev` | 无 A 记录（未确认注册状态，需 registrar 查询） |
| 既有语义 | 生物医疗领域有 `Infrarray SNAP`（SBIR 项目），属不同类别，软件类风险低 |

评估：

| 维度 | 评价 | 说明 |
| --- | --- | --- |
| 可用性 | 优 | 包名、GitHub、主域名都干净 |
| 商标独特性 | 优 | 生造词，比 `Deck`/`Studio` 更容易注册 |
| 可拼写性 | **差** | `Infr` + `Array` 的接缝少一个 `a`，大量用户会写成 `InfraArray` 或 `InfraRay` |
| 可发音 | 中 | “in-fra-RAY” 与 “in-fra-AR-ray” 都说得通，口头传播会分叉 |
| 语义 | 中上 | “GPU/服务阵列”贴合多卡多模型，但也可能被理解为存储阵列 |
| 中文市场 | 中 | 中文音译不自然；好处是用户更多点击而非手输 |
| 品牌架构 | 好 | 中性的平台品牌，与 StarLight 内核品牌分离，便于将来接外部运行时 |

结论：

- **作为防御性占名，值得做**：成本低，且这是目前唯一一个全渠道干净的名字；
- **作为公开主品牌，可用但需接受拼写成本**：所有对外材料必须固定大小写，并同时持有 `infraarray` 变体作为重定向；
- **推荐的品牌结构**：`InfrArray` 是平台产品，`StarLight` 是推理内核，对外一句话 “InfrArray, powered by StarLight”。这同时满足“独立商业产品”的定位和内核品牌沉淀。

### 1.5 占名清单（若采用 InfrArray）

1. 域名：`infrarray.com`、`.ai`、`.io`、`.dev`、`.app`；防御性注册 `infraarray.*`；
2. npm：组织 `infrarray` 与 scope `@infrarray/*`，同时占 `infraarray` 包名；
3. PyPI：`infrarray`、`infraarray`；
4. crates.io：`infrarray`、`infraarray`；
5. GitHub Organization：`infrarray`；
6. 社交与内容：X、LinkedIn、YouTube、Bilibili、微信公众号；
7. 容器：Docker Hub `infrarray`；
8. 商标：在中国（第 9、42 类）与目标市场提交意向使用申请，由代理机构完成；
9. 注意：占名不产生商标权，也不能替代正式检索；商标需要在商业使用中维持。

## 2. 问题与机会

### 2.1 目标客户

- 有 2–8 张 GPU 的私有化部署客户：制造业、医疗、金融、政务、教育；
- 系统集成商与行业 ISV：需要把模型交付到客户机房，但不想养一支 MLOps 团队；
- 中小型 AI 应用公司：要同时跑多个模型（对话、嵌入、多模态），但买不起 K8s 平台团队。

共同痛点：模型下载、量化、显存估算、端口与进程、API 鉴权、日志与重启，全靠人肉 shell 脚本；换一个模型就要重来一遍。

### 2.2 现有方案与空白

| 类别 | 代表 | 空白 |
| --- | --- | --- |
| 单机模型运行器 | Ollama、LM Studio | 多模型多卡编排弱，缺少服务治理与 API 网关 |
| 集群推理平台 | KServe、BentoML、Ray Serve | 面向 K8s，私有化单机太重 |
| GPU 纳管 + 部署 | GPUStack 等 | 有 UI，但节点化编排、量化策略与“导出即服务”的体验仍可差异化；需逐一核查 |
| 工作流画布 | ComfyUI | UX 启发，但它是纯函数 DAG，不处理资源、进程与失败恢复 |
| 推理引擎 | vLLM、TensorRT-LLM、llama.cpp | 只解决“怎么算”，不解决“怎么部署和运营” |

Deck 的差异化：**节点化编排 + 多模型多卡放置 + 量化策略 + 一条命令导出 API + 中文优先的本地化体验**。

## 3. 产品定义

### 3.1 是什么

- **部署控制面**：把“想跑什么、跑在哪、怎么量化、暴露什么 API”变成一份可版本化的部署清单；
- **节点画布**：用图形方式编辑这份清单，而不是写 YAML；
- **模型资产管家**：从 Hugging Face 下载、校验、缓存、派生量化版本；
- **API 网关**：把服务以 OpenAI 兼容协议暴露出去，带鉴权、配额和用量统计；
- **运维面板**：设备、显存、服务健康、日志、重启。

### 3.2 不是什么

- 不是训练/微调平台；
- 不是 Kubernetes 替代品（第一版不假设有 K8s）；
- 不是模型市场；
- 不是 StarLight 的发行版：Deck 是独立产品，StarLight 是它的推理内核依赖。

### 3.3 用户旅程

```text
启动 → 扫描设备 → 登录 Hugging Face → 选模型/下载 → 量化配置
     → 画布编排（模型 + GPU 组 + 服务 + 输入/输出）
     → 应用部署清单 → 服务启动 → 健康检查
     → 导出 API（key/配额/上下文/缓存目录）→ 观测与重启
```

目标：**裸机到第一个 token ≤ 15 分钟**，双模型双卡组配置 ≤ 10 分钟。

### 3.4 核心概念

| 概念 | 含义 |
| --- | --- |
| Workspace | 一个部署环境（通常一台机器） |
| Device Pool | 本机 GPU 的逻辑分组，如 `gpu-0-3`、`gpu-4-7` |
| Asset | 模型、tokenizer、量化配置等可复用资产 |
| Service | 一个模型服务实例：模型 + 放置 + 量化 + 运行时配置 |
| Deployment | 一组 Service 的期望状态 |
| Endpoint | Service 对外暴露的 API |
| Run | 一次具体执行，有健康、日志、指标和失败原因 |

## 4. 关键设计决定：Manifest-first

画布与服务同时做，最容易失控的方式是“画布一套逻辑、API 一套逻辑”。因此第一原则是：

**画布、向导、CLI 都只产出同一份 `DeploymentSpec`；执行器只认这份清单。**

```text
Canvas ─┐
Wizard ─┼─> DeploymentSpec (YAML/JSON) ─> Reconciler ─> Agent ─> Runtime
CLI ────┘                                      │
                                          Gateway / Metrics
```

好处：

- 画布只是清单的**视图与编辑器**，不做业务逻辑；
- 同一份清单可导出、可 diff、可进 Git、可被 CI 应用；
- 服务端能力增加时，画布自动获得，不需要双写；
- 失败恢复与漂移检测集中在 reconciler，不在 UI 里。

## 5. 节点模型

节点是**部署资源**，不是计算算子。边分两类：数据流边（typed）与放置约束边（placement）。

| 类别 | 节点 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| Asset | Model | → model | HF repo/revision/文件清单；或本地路径 |
| Asset | Tokenizer | → tokenizer | 可与 Model 绑定或独立 |
| Asset | Quantization Profile | model → quantized model | INT4/INT8 分组大小、keep 列表；后续 FP8/FP4 |
| Infra | Device Pool | → devices | GPU 选择、显存预算、NVLink 域 |
| Infra | Placement | devices + model → placement | 副本数、TP/PP 度、每卡显存上限 |
| Service | **Inference Server** | model + placement + io → endpoint | **唯一可导出 API 的节点** |
| IO | Prompt Preset | → prompt | 预置提示词/系统提示词 |
| IO | File Input | → document | 文本/Embedding 文件输入 |
| IO | Multimodal Input | → image/audio | 多模态输入端口（后续） |
| Ops | Gateway | endpoints → endpoint | 路由、鉴权、配额 |
| Ops | Health Check | endpoint → status | 存活/就绪探针 |
| Ops | Scaler | endpoint → replicas | 后续：按队列长度扩缩容 |

约束：

- `Inference Server` 是唯一有副作用且唯一能导出 endpoint 的节点；
- 画布必须能检测环、端口类型不匹配、GPU 超卖、显存不足；
- 每个 Server 节点在清单里对应一个运行进程（v1），因此端口、显存、日志天然隔离。

## 6. 部署清单（目标格式）

```yaml
apiVersion: starlight.seekstar.ai/v1
kind: Deployment
metadata:
  name: prod-a
spec:
  assets:
    models:
      - name: qwen-14b
        source: { type: huggingface, repo: Qwen/Qwen2.5-14B-Instruct, revision: main }
        files: ["*.safetensors", "config.json", "tokenizer*"]
  device_pools:
    - name: gpu-0-3
      devices: [0, 1, 2, 3]
  services:
    - name: chat
      model: qwen-14b
      placement: { pool: gpu-0-3, replicas: 1, tensor_parallel: 1, pipeline_parallel: 1 }
      quantization: { scheme: int4, group_size: 128, keep: ["model.embed_tokens.weight"] }
      runtime:
        max_context: 32768
        kv_cache: { dir: /var/lib/starlight/kv, mode: contiguous }
        batching: { max_batch: 8 }
      api:
        protocol: openai
        port: 8000
        auth: { type: api_key, keys_from: secret:chat-keys }
```

状态标注：

- `tensor_parallel` / `pipeline_parallel` 字段**先保留但只接受 1**；>1 在 v1 返回明确的“未支持”错误，而不是静默串行；
- `paged`、`mxfp8`、`nvfp4` 等在 schema 中标记为 reserved，校验器拒绝；
- 清单必须有 `apiVersion`，并随能力演进升版本。

## 7. 架构

```text
┌──────────────────────────── Deck ────────────────────────────┐
│  Web UI (canvas / wizard / dashboards)                       │
│        │                                                     │
│  Control API  ──  DeploymentSpec 校验 / 版本 / 鉴权           │
│        │                                                     │
│  Reconciler  ── 期望 vs 实际 / 幂等 apply / 漂移检测          │
│        │                                                     │
│  Agent（每台机器）── 设备清单 / 进程监管 / 日志 / 健康         │
│        │                                                     │
│  Runtime Adapter ── starlight | external（过渡）              │
│        │                                                     │
│  Gateway ── OpenAI 兼容 / key / 配额 / 用量                   │
│  Hub ── HF 下载 / 校验 / 缓存 / 量化派生                      │
└──────────────────────────────┬───────────────────────────────┘
                               │ 公开 API（只读依赖）
                     ┌─────────▼──────────┐
                     │      StarLight      │
                     │ sl-model / adapters │
                     │ sl-graph / sl-core  │
                     │ sl-device / backend │
                     └────────────────────┘
```

### 7.1 组件职责

| 组件 | 职责 | 不做什么 |
| --- | --- | --- |
| `deck-web` | 画布、向导、服务面板、日志 | 不直接调运行时 |
| `deck-api` | 清单校验、版本、鉴权、审计 | 不实现调度算法 |
| `deck-reconciler` | 期望状态收敛、重启、漂移检测 | 不解析模型文件 |
| `deck-agent` | NVML 设备扫描、进程监督、端口分配、健康 | 不做跨机调度 |
| `deck-hub` | HF 下载、断点续传、哈希、缓存、许可证检查 | 不执行模型 |
| `deck-gateway` | OpenAI 协议、路由、key、配额、用量 | 不做推理 |
| `deck-runtime` | 把清单翻译成运行时调用 | 不含模型逻辑 |

### 7.2 与 StarLight 的边界

- Deck **只依赖 StarLight 的公开 API**；StarLight 的任何 crate 不得依赖 Deck；
- Deck 不修改 `sl-core`、`sl-graph`、`sl-device` 的语义；
- 设备扫描、放置、网关、UI 都是 Deck 自己的组件，不塞进内核；
- 画布的 DAG 是**部署 DAG**，不是 `sl-graph` 的计算图，严禁混用。

### 7.3 过渡期 Runtime Adapter

现实问题：StarLight 目前还不能推理一个 7B 模型（`sl-core`/`sl-graph`/CUDA 后端仍未实现）。如果 Deck 只支持 StarLight，产品在 6–9 个月内无法交付。

因此 v1 采用 **Runtime Adapter 接口 + 两个实现**：

| 实现 | 用途 | 对外表述 |
| --- | --- | --- |
| `runtime-starlight` | 战略路径；随内核成熟逐步接管 | “StarLight 内核” |
| `runtime-external` | 过渡期可用（可接 vLLM/llama.cpp 等） | “兼容运行时”，不宣称 StarLight |

退出条件：当 StarLight 对某个“家族 + 精度 + 硬件”组合通过数值基线后，Deck 将该组合的默认运行时切到 `runtime-starlight`。在此之前，市场材料不得把外部运行时的性能算作 StarLight 的成绩。

## 8. 并行与放置

### 8.1 三个层级

| 层级 | 做法 | 是否需要通信 | Deck 版本 |
| --- | --- | --- | --- |
| 进程隔离 | 每个服务一个进程，`CUDA_VISIBLE_DEVICES` 绑定 GPU 组 | 否 | **v1** |
| 流水并行 | 按层切分到多卡，激活值跨卡传输 | 是 | v2 |
| 张量并行 | 权重分片，all-reduce/all-gather | 是（NVLink/NCCL） | v3 |

v1 支持“第一个模型放 0–3、第二个放 4–7”，但每个模型内部仍跑在**单卡**上；GPU 组只是隔离与显存预算单位。这一点必须在 UI 上说清楚，否则用户会以为已经做了张量并行。

### 8.2 放置算法

v1 用约束 + 首次适应（first-fit）即可：

1. 过滤：显存、算力架构、驱动版本、NVLink 域、可用端口；
2. 估算内存：`weights + kv_cache + activations + overhead`；
3. 约束检查：TP 度整除 heads/kv_heads，PP 度整除层数；
4. 装箱：按显存降序放置，失败时给出“差多少 GiB”的可操作提示。

内存估算公式（v1 用保守估计，后续用实测校准）：

```text
weights   = params * bits / 8
kv_cache  = 2 * layers * kv_heads * head_dim * max_context * batch * bytes_per_elem
activation ≈ batch * max_context * hidden * bytes_per_elem * k   // k 由家族给出
```

`k`、`head_dim` 等由 `sl-model-adapters` 的家族描述提供——这是 Deck 需要 StarLight 补充的能力之一（见 §13）。

## 9. 量化

| 方案 | 状态 | 说明 |
| --- | --- | --- |
| INT4/INT8 groupwise | 已有（`sl-precision`） | v1 唯一支持的量化 |
| FP8 | 未实现 | 需要新 dtype、校准和 kernel |
| MXFP8 | 未实现 | 需要 block scaling 与硬件支持 |
| NVFP4 | 未实现 | 依赖 Blackwell 级硬件与 FP4 kernel |

设计：

- 量化配置是 Asset 节点，产物是**派生资产**，必须记录来源哈希、scheme、group size、工具版本；
- 量化失败或精度评估不通过时，回退到未量化资产，并在 UI 上明确标注；
- v1 不做自动精度评估，只做“可加载 + 冒烟推理”。

## 10. 安全与多租户

- **模型下载**：只接受 Safetensors 与显式允许的格式；拒绝 pickle；限制大小与并发；记录来源与哈希；
- **自定义节点**：v1 **不开放**第三方节点。ComfyUI 的自定义节点生态是前车之鉴——任意代码 + GPU + 网络 + 模型下载；
- **后续开放**：签名节点包 + 能力清单（能访问哪些设备/目录/网络）+ 沙箱；
- **API 安全**：每服务独立 key、配额、速率限制、审计日志；
- **隔离**：一个服务一个进程；不共享 KV cache 目录；禁止跨服务读取工作目录；
- **密钥**：HF token、API key 加密存储，日志脱敏。

## 11. 商业模式

| 层级 | 内容 | 价格假设 |
| --- | --- | --- |
| Community | 单机 ≤ 2 GPU，全部核心功能 | 免费 |
| Team | 单机 8 GPU，多用户、审计、SSO | 按节点年费 |
| Enterprise | 多机、私有化支持、定制节点、合规 | 年费 + 支持 |

开源策略：Agent 与清单格式开源（便于集成商接入），控制面与 UI 采用 open-core。StarLight 本身是 MIT/Apache，Deck 的商业价值在控制面、UX、交付与支持，而不是内核许可证。

## 12. 路线图

| 阶段 | 时间 | 交付 | 退出标准 |
| --- | --- | --- | --- |
| P0 | 0–6 周 | 设备扫描、HF 下载、单服务启动、OpenAI API、日志、CLI apply | 裸机到首 token ≤ 15 分钟 |
| P1 | 6–12 周 | 自由画布、多服务、GPU 组、清单导入导出、reconciler 重启 | 双模型双卡组 ≤ 10 分钟；重启后自动收敛 |
| P2 | 3–6 月 | 流水并行、paged KV、指标面板、按队列扩缩容 | 单模型跨 2 卡，吞吐线性度 ≥ 1.6× |
| P3 | 6–12 月 | 张量并行、多机、MoE/FP8/FP4、K8s 部署 | 与单卡基线数值一致，扩展效率达标 |

P0/P1 同时做画布与服务，风险在于范围翻倍。缓解手段：画布先只支持**固定模板 + 有限节点**，清单格式先冻结，复杂节点在 P1 后半段开放。

## 13. 产品对 StarLight 内核的需求

这是本文对内核路线图最有价值的部分。Deck 要交付，内核必须提供：

| 需求 | 当前状态 | 优先级 |
| --- | --- | --- |
| 可调用的 `Engine`/`Session` 与 `step` | 未实现 | P0 阻断 |
| 一个真实模型家族的端到端推理 | 未实现 | P0 阻断 |
| OpenAI 兼容服务端（可放 `sl-daemon`） | 未实现 | P0 |
| 设备能力查询（显存、算力、拓扑） | `sl-ir::DeviceCapabilities` 只有形状/算子 | P1 |
| 家族内存估算（层数、KV 维度、激活系数） | 未定义 | P1 |
| 量化产物的加载与校验 | `sl-model` 有编码描述，无 writer | P1 |
| 指标钩子（队列、延迟、显存） | 未定义 | P2 |
| 多卡通信原语 | 未实现 | P2 |
| 张量并行图切分 | 未实现 | P3 |

建议：把“OpenAI 兼容服务端”作为 `sl-daemon` 的第一个真实交付，而不是在 Deck 里重写一遍协议。

## 14. 风险

| 风险 | 级别 | 缓解 |
| --- | --- | --- |
| 内核不能推理 7B，产品无内容可编排 | **高** | 过渡期 runtime adapter；对内明确里程碑依赖 |
| 画布 + 服务同时做导致范围失控 | 高 | manifest-first；P0 只做模板画布 |
| 自定义节点带来安全与生态负担 | 中高 | v1 不开放；后续签名 + 沙箱 |
| 竞品（GPUStack 等）已覆盖基础能力 | 中 | 差异化在节点编排、量化策略、中文交付 |
| 名字仍有商标风险 | 中 | 正式检索后再对外 |
| 只支持 NVIDIA，客户用 Apple/国产卡 | 中 | 明确支持矩阵；后续接 Vulkan/国产后端 |
| 量化精度问题导致客户不信任 | 中 | 保留未量化回退；记录 provenance |

## 15. 成功指标

- 裸机到首个 token ≤ 15 分钟；
- 双模型双卡组配置 ≤ 10 分钟，无需 shell；
- 服务重启后 30 秒内自动恢复；
- 网关相对直连的 p95 开销 < 5%；
- 每周人工干预次数 < 1；
- 画布构建的部署清单 100% 可导出、可重放。

## 16. 开放问题

1. `runtime-external` 是否进入 v1 对外版本，还是只作为内部过渡？
2. 自定义节点何时开放？开放到什么粒度（仅配置 vs 可执行代码）？
3. 首发是否中文优先？英文 UI 是否同版本？
4. 多机（P3）是否值得投入，还是先做单机极致体验？
5. 定价按 GPU 数还是按节点？社区版上限设在几卡？
6. 是否提供托管云版本，还是只做私有化？
7. 最终命名选 `InfrArray`（全渠道干净、商标独特性好、但有拼写歧义）还是 `StarLight Deck`（零歧义、品牌一致、商标独特性较弱）？

## 17. 已确认的产品决定（2026-10-17）

| 决定 | 选择 |
| --- | --- |
| 与 StarLight 的关系 | 独立商业产品，以 StarLight 为内核依赖 |
| MVP 范围 | 画布与服务同时做，manifest-first |
| 首个演示场景 | dense 7B–14B + INT4/INT8，单机多卡进程隔离 |
| 命名 | 放弃 InfraFlow；现候选 `InfrArray`（§1.4）与 `StarLight Deck`，待最终确认 |

## 18. 名称核查清单（对外发布前）

1. USPTO TESS / EUIPO / CNIPA 商标检索（第 9、42 类）；
2. 域名：`starlightdeck.com`、`.ai`、`.io`、`.dev`；
3. npm / PyPI / crates.io / GitHub Organization；
4. App Store / Microsoft Store / 应用市场；
5. 搜索引擎与社交媒体 handle；
6. 竞品与同名的开源项目；
7. 由法务出具书面意见。
