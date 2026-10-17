# StarLight 文档索引

- [ARCH-0002：模块边界与发行版业务管线契约](<ARCH-0002: 模块边界与发行版业务管线契约.md>)：模块内职责、模块间依赖与调度分层，以及 CLI/SDK 端到端集成验收标准。与早期规划冲突时，以本文作为当前工程基线。
- [ARCH-0003：模型制品与多模态推理数据流设计](<ARCH-0003: 模型制品与多模态推理数据流设计.md>)：模型格式、逻辑权重表示、家族 Adapter、图/设备表示，以及输入预处理和流式输出边界。
- [ARCH-0004：计算图 IR 与设备命令 ABI 契约](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)：`sl-ir` 的职责边界、`Program`/`CommandBuffer` 两层 IR、算子分层与能力协商、静态内存规划不变量、绑定语义与版本策略。
- [ARCH-0005：服务面契约 sl-core、sl-daemon 与 InfrArray](<ARCH-0005: 服务面契约 sl-core sl-daemon 与 InfrArray.md>)：四层服务职责、Engine 与 daemon 契约、`starlight-runtime` 进程协议，以及 InfrArray 如何在不链接内核的前提下接管服务。
- [ARCH-0006：模型家族 Lowering 与 LoweredModel 契约](<ARCH-0006: 模型家族 Lowering 与 LoweredModel 契约.md>)：模型家族 Adapter 的输入输出、`LoweredModel` 结构、参数角色命名、prefill/decode 切分与 v1 支持矩阵。
- [PHIL-0001：为什么将 StarLight 构建为“推理领域的 GNU/Linux”？](<PHIL-0001: 为什么我们将 StarLight 构建为“推理领域的 GNU Linux”？.md>)：模块化与发行版设计哲学。
- [RFC-0001：系统工程与架构规划草案](<RFC-0001: StarLight 系统工程与架构规划草案 (2026-10-16).md>)：早期架构目标与任务拆分；其中的规划不代表当前实现状态。
- [RFC-0002：采纳 sl-ir 作为图 IR 与设备命令 ABI](<RFC-0002: 采纳 sl-ir 作为图 IR 与设备命令 ABI.md>)：新增 ABI crate、修订模块依赖与后端 trait 的决策记录。
- [PRODUCT-0001：InfrArray 产品规划](<PRODUCT-0001: InfrArray 产品规划.md>)：以 StarLight 为默认引擎的图形化推理基础设施控制面，含产品形态、节点体系、部署清单、架构、路线图与内核需求。
