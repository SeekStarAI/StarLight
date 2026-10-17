# RFC-0002：采纳 sl-ir 作为图 IR 与设备命令 ABI

**文档编号：** RFC-0002
**状态：** 已接受
**日期：** 2026-10-17
**决策依据：** [ARCH-0004：计算图 IR 与设备命令 ABI 契约](<ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)
**影响范围：** ARCH-0002 §2/§3/§7、RFC-0001 §2.2/§2.3/§4.1、根 Cargo workspace、`sl-ir`/`sl-graph`/`sl-device`/`sl-model-adapters` 的依赖声明

## 1. 摘要

本 RFC 接受 ARCH-0004 的设计，并做出以下具有约束力的决定：

1. 新增独立 crate `sl-ir`，承载图 IR 与设备命令 ABI 的全部纯数据类型；
2. `sl-graph` 与 `sl-device` 都依赖 `sl-ir`，两者之间不再存在依赖边；
3. `sl-model-adapters` 改为依赖 `sl-model` 与 `sl-ir`，不再依赖 `sl-graph`；
4. IR 分为符号层 `Program` 与物理层 `CommandBuffer`；
5. 后端 trait 的签名按 `sl-ir` 的绑定结构修订。

本 RFC 通过后，ARCH-0002 的相关章节按 §4 修订；RFC-0001 的相应条款按 §5 视为被取代。

## 2. 背景与问题

在 ARCH-0002 与 RFC-0001 中，模块之间的交接对象（`ModelBundle`、`GraphPlan`、`CompiledGraph`、`Device`、`Engine`）只有概念名，没有可评审的数据结构。由此产生三个具体问题：

1. **循环依赖。** 若 ABI 类型随 backend trait 放在 `sl-device`，`sl-device` 需要引用 `sl-graph` 的命令类型，而 `sl-graph` 又需要产出设备句柄，形成环；若放在 `sl-graph`，backend 又被追依赖编译器。
2. **契约不可版本化。** 没有独立 crate 就没有独立的 ABI 版本边界，算子或命令布局的改动只能靠各仓库同步发版。
3. **Adapter 依赖倒置。** Adapter 的职责是"构造程序"，不是"编译程序"。让它依赖 `sl-graph` 会迫使它拉入整个编译器，也让 IR 的定义权落在优化器手里。

## 3. 决策

### D1：新增 `sl-ir`

- 零依赖，`no_std + alloc` 可行，`#![deny(unsafe_code)]`；
- 只含数据与不变量校验，不含优化、调度、着色、I/O 或设备访问；
- `IR_VERSION` 与 crate semver 独立演进。

**代价：** 多一个 submodule 与一套变更流程。**收益：** 无环依赖、独立版本化、第三方后端只依赖两个小 crate。

### D2：ABI 类型清单

`sl-ir` 拥有：`DType`/`DTypeSet`、`Dim`/`Shape`/`ShapeSpec`/`SymDimDecl`/`ShapeConstraints`、`Value`/`ValueId`/`ValueRole`/`ValueRoleSet`、`OpCode`/`Op`/`Signature`/`PortSpec`/`OpSet`、`Program`/`Node`/`Effect`/`StateDecl`、`Command`/`CommandBuffer`、`MemoryPlan`/`Slot`、`DeviceCapabilities`、`BufferId`/`ParamBinding`/`StepBinding`、`IrError`。

`sl-ir` 不拥有：模型容器、权重加载、图优化、内存着色、设备句柄、队列、错误文本格式化。

### D3：两层 IR

- `Program`：符号程序，形状可含 `Dim::Sym`，由 Adapter 构造；
- `CommandBuffer`：物理命令流，形状全部具体化，含执行顺序与静态内存布局，由 `sl-graph` 产出、backend 消费。

`Program` 不携带架构标识；家族信息留在 `LoweredModel`（见 ARCH-0006，待编写）。

### D4：算子分层

- **core**：IR 允许出现的语义原语词汇表，backend 必须逐项声明支持或拒绝；
- **composite**：可整体优化的意图，backend 不支持时由 `sl-graph` 分解。

v1 的例外：`CausalAttention` **不提供分解**，要求 backend 原生实现。原因是分解需要"按层读取并展开 GQA"的额外 core 算子（`KvRead`），工作量与直接实现 attention 相当。该决定记录在 ARCH-0004 §5.3，后续出现第二后端或第二类 attention 变体时重新评估。

### D5：形状按上界具体化

编译期用不超过上界的**具体形状**做静态规划；运行期真实长度通过标量 value（`seq_len`、`past_len`）传入。v1 不做运行期形状推导，也不引入符号维度算术。

### D6：能力协商在编译期完成

`DeviceCapabilities` 描述 backend 支持的算子、dtype、对齐与形状约束。缺失能力必须在 build 阶段以 `UnsupportedOp` / `CapabilityMismatch` 失败，不得留到 `step()`。

### D7：后端 trait 修订

`sl-device` 的 `BackendDevice` 按下列语义定义（签名以 ARCH-0004 §9 为准）：

```rust
fn compile_pipeline(
    &self,
    ir: &CommandBuffer,
    plan: &MemoryPlan,
    params: &[ParamBinding],
) -> Result<PipelineHandle, DeviceError>;

fn submit_step(
    &self,
    queue: QueueId,
    pipeline: &PipelineHandle,
    io: &StepBinding,
) -> Result<(), DeviceError>;
```

- **权重在编译期绑定**并进入 `PipelineHandle`，`StepBinding` 只含每步动态量；
- `submit_step` 的完成语义是**已入队**，读取输出前必须显式同步；
- `bind_sparse_pages` 与 paged KV 不在 v1 必需接口内。

### D8：版本策略

- `IR_VERSION = 1`；
- 增删 `OpCode`、修改 `Signature` 或字段含义、修改 `MemoryPlan` 不变量、修改绑定结构，都要提升 `IR_VERSION`；
- `OpCode` 判别值永不复用，删除算子保留墓碑；
- backend 必须声明 `ir_version`，且 match 必须带 catch-all，保证新算子对旧 backend 安全失败。

## 4. 对 ARCH-0002 的修订

| 位置 | 修订 |
| --- | --- |
| 文档头 | 增加"已按 RFC-0002 修订"说明 |
| §2 模块表 | 新增 `sl-ir` 行；`sl-graph` 行改为"将 `Program` 具体化为 `CommandBuffer`"；`sl-model-adapters` 行改为输出 `sl-ir::Program` |
| §2.1 | Adapter 依赖改为 `sl-model` + `sl-ir`；明确 Adapter 不依赖 `sl-graph` |
| §3 依赖图 | 插入 `sl-ir`；删除 `sl-graph → sl-device` 的隐含依赖；补充说明 |
| §7 缺口 | 增加 `sl-ir` 的当前状态 |

## 5. 对 RFC-0001 的取代

| RFC-0001 条款 | 处理 |
| --- | --- |
| §2.2 `sl-graph` 输出"融合后的线性指令队列" | 被取代：命令流带依赖边与副作用标记，v1 调度为全序，见 ARCH-0004 §6 |
| §2.3 `BackendDevice` trait | 被取代：`compile_pipeline` 增加 `params`，`submit_step` 使用 `sl-ir` 绑定，完成语义明确为"已入队" |
| §4.1 仓库拓扑 | 被取代：新增 `crates/sl-ir` |

RFC-0001 的其余内容（分层目标、任务包划分、CI 门禁原则）继续有效。

## 6. 交付与迁移

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | `sl-ir` 仓库建立、submodule 接入、workspace member 与 `[patch.crates-io]`、数据类型与不变量校验 | **已完成**（`sl-ir` 0.1.0，26 个单测） |
| M1 | `OpCode::signature` 的 `Program` 级校验、属性测试 | 待开始 |
| M2 | `specialize`、能力校验、composite 分解（`sl-graph`） | 待开始 |
| M3 | 生命周期分析与 `MemoryPlan` 着色 | 待开始 |
| M4 | `sl-device` trait、`backend-cpu` 标量参考实现 | 进行中 |
| M5 | `sl-core` build/step、`starlight-nano` CLI | 待开始 |

迁移顺序：**先建立并锁定 `sl-ir`，再让 `sl-graph`/`sl-device` 声明依赖，最后修改 Adapter 与 backend。** 在 `sl-ir` 的提交被推送到远端之前，主仓库的 submodule 指针不得指向该提交，否则其他协作者无法初始化子模块。

## 7. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 纸面设计过度、字段无消费者 | 每个字段必须能指出生产方、消费方与测试；无消费者的字段推迟加入；用 tiny fixture 纵切验证 |
| 早期 ABI 频繁变更 | `IR_VERSION` + 判别值墓碑 + backend 显式声明版本 |
| 新 crate 变成"第二个 sl-graph" | 在 `DEVELOPMENT.md` 中禁止算法；评审时以"是否包含优化/调度"为否决条件 |
| Adapter 与 IR 同时演进导致返工 | 先冻结 v1 所需算子（8 个 core + 2 个 composite），其余标记为"保留" |
| 版本不一致导致运行期失败 | 能力校验在编译期完成；`compile_pipeline` 入口检查 `ir_version` |

## 8. 验收标准

1. ARCH-0002 按 §4 完成修订，且与 ARCH-0004 无冲突；
2. `sl-ir` 在 workspace 中可独立构建：`cargo test -p sl-ir`、`cargo check -p sl-ir --no-default-features`、`cargo clippy -p sl-ir --all-targets -D warnings` 全部通过；
3. 根 `cargo test --workspace` 与 `cargo clippy --workspace --all-targets -D warnings` 通过；
4. 后续任何修改 `OpCode`/`Signature`/`MemoryPlan`/绑定的 PR，都必须同时更新 `IR_VERSION` 与 ARCH-0004。

## 9. 未决问题

1. paged KV cache 的 State 表示与 `bind_sparse_pages` 的形态；
2. v1 是否接受带 bias 的模型（当前要求无 bias）；
3. `Scalar` 在 GPU 后端的可见性（uniform buffer）；
4. `Program` 是否需要受控 builder API；
5. `no_std` 是否作为 CI 硬门禁；
6. 首个模型家族与支持矩阵（ARCH-0006）。

## 10. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-17 | 初版，接受 ARCH-0004 并完成 M0 |
