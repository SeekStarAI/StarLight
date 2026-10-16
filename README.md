# StarLight

StarLight 是 SeekStarAI 面向模型推理构建的 AI Infra 基础设施项目。它不是一个把模型格式、算子实现、执行引擎、请求服务和分布式通信全部封装在一起的单体引擎，而是一组边界清晰、可独立演进并可按场景组合的基础组件。

项目维护模型格式支持、精度转换工具、并发计算图引擎、设备接口与计算内核、推理运行时和调度服务等组件。围绕主流推理场景，StarLight 再将这些组件组装为不同的发行版。部署方可以按设备与服务形态选用所需组件，也可以替换后端、调度器或通信实现。

## 设计目标

- **组件化，而非单体化：** 模型读写、精度处理、图执行、设备后端、请求编排和发行版入口各有职责边界。组件之间通过明确的数据结构和接口协作。
- **以图组织计算：** 自研 `sl-graph`（StarLight Graph）面向并发图执行，目标是提供计算图构建、依赖管理、编译与执行组织能力，对标 CUDA Graph、CGraph 等图执行方案。它是 StarLight 推理执行链路的核心组件之一，不等同于整个 StarLight 项目。
- **面向不同设备：** `sl-device` 定义设备抽象契约，后端负责将图中的计算映射为目标设备上的内核执行。CPU、Vulkan 以及后续可接入的 GPU、NPU 或第三方实现属于可替换的后端选择；“any device”是接口与生态建设目标，不表示当前已支持所有设备。
- **将控制面与计算面分开：** 图引擎与设备后端组织计算；运行时负责装配模型和驱动执行；调度器负责请求、资源和任务编排。发行版可以包含或省略服务端调度与分布式通信设施。
- **支持按场景组装：** 端侧与集群推理对进程模型、调度、通信和设备的需求不同，StarLight 通过发行版组合这些基础组件，而不是要求所有部署共享同一套完整运行时。

## 推理组件如何协作

典型推理链路由离线准备、执行和服务编排组成。具体发行版可以裁剪步骤或替换其中的组件。

```text
模型文件
   │
   ├── 模型格式读取与适配工具（sl-model 及配套工具）
   │        │
   │        └── 格式无关的模型描述、张量与权重数据
   │
   ├── 精度转换工具（sl-precision）
   │        └── 量化、反量化及面向后端的数据布局准备
   │
   └── StarLight Graph（sl-graph）
            └── 构建、编译并组织并发计算图
                     │
              运行时与调度器
              （sl-core / sl-daemon 或外部调度器）
                     │
              sl-device 设备接口
                     │
              后端与计算内核
              （backend-cpu / backend-vulkan / 后续后端）
                     │
              目标设备上的推理执行

可选：分布式通信组件连接多设备或多节点执行，
      由采用该能力的发行版按需装配。
```

模型格式适配解决“模型如何进入系统”；精度工具解决“权重如何表示、转换和供后端使用”；StarLight Graph 负责“计算如何形成可执行的图”；设备接口与后端负责“计算如何落到硬件”；运行时、调度器以及可选通信组件负责“任务如何被驱动、编排和扩展”。这些组件共同构成推理基础设施，任何单个 crate 都不代表整个 StarLight。

## 组件与职责

| 组件 | 职责 | 在推理链路中的位置 |
| --- | --- | --- |
| `sl-model` | 定义格式无关的模型包、张量元数据和数据视图，并提供模型格式读取适配 | 读取与规范化模型输入 |
| `sl-model-adapters` | 将通用模型包按模型家族 lowering 为计算图描述与权重绑定 | 连接模型表示和通用图；当前仓库已接入但尚无 Cargo 包/API |
| `sl-precision` | 提供量化、反量化及精度相关数据处理工具 | 离线或加载阶段的权重准备；不属于 `sl-core` 单步执行热路径 |
| `sl-graph` | StarLight Graph；表达计算拓扑并规划图执行和内存布局 | 形成可供运行时与后端执行的计算图 |
| `sl-device` | 定义设备、缓冲区、队列等虚拟设备接口（VDI）契约 | 隔离图执行与具体硬件实现 |
| `backend-cpu` | CPU 设备后端与计算内核实现入口 | 将计算映射至 CPU |
| `backend-vulkan` | Vulkan 设备后端与计算内核实现入口 | 将计算映射至支持 Vulkan 的设备 |
| `sl-core` | 装配模型、图与设备，并提供核心推理执行路径 | 驱动模型计算 |
| `sl-daemon` | 集群场景的服务进程与请求编排入口 | 在运行时外围管理会话、调度和服务生命周期 |
| `starlight-nano` | 面向端侧场景的发行版装配入口 | 选择适合端侧的运行时与后端组件 |
| `starlight-cluster` | 面向服务器与集群场景的发行版装配入口 | 组合服务端调度、运行时和集群所需扩展 |
| 分布式通信扩展 | 为跨设备、跨卡或跨节点执行提供通信能力 | 由需要分布式推理的发行版选择性接入 |

通信能力是可选的外围设施，不是所有推理任务的必需依赖，也不应与图引擎或设备内核混为一体。调度器同样可以由 StarLight 服务组件提供，或由部署环境中的外部系统承担。

## 发行版

发行版是针对目标环境选择和组装组件的交付形态，不是另一个独立的推理引擎。当前 workspace 提供两个发行版入口：

- **StarLight-Nano**（`distros/starlight-nano`）：面向端侧、AIPC 和嵌入式环境，优先考虑轻量部署与进程内执行。具体支持的设备和绑定能力取决于后端与发行版实现进度。
- **StarLight-Cluster**（`distros/starlight-cluster`）：面向服务器及集群推理，可组合 `sl-daemon`、设备后端和后续的分布式通信组件。集群调度与跨节点能力按需引入，不是 `sl-core` 的职责。

这两个目录目前是装配入口骨架，不能据此推断端侧或集群功能已经完整可用。发行版的目标是把经过验证的组件组合成便于使用和部署的方案。

## 仓库结构

```text
StarLight/
├── crates/
│   ├── sl-model/             # 模型数据结构与格式适配
│   ├── sl-model-adapters/    # 模型家族到通用图的适配（独立 submodule）
│   ├── sl-precision/         # 量化、反量化与精度工具
│   ├── sl-graph/             # StarLight Graph 图执行与规划组件
│   ├── sl-device/            # 虚拟设备接口契约
│   └── backends/             # 设备后端与计算内核
├── services/
│   └── sl-daemon/            # 集群服务与请求编排入口
├── distros/
│   ├── starlight-nano/       # 端侧发行版入口
│   └── starlight-cluster/    # 集群发行版入口
├── docs/                     # 架构说明、设计哲学与工程规划
└── tests/
    └── golden-baselines/     # 跨组件数值基准数据
```

其中 `sl-core`、`sl-device`、`sl-graph`、`sl-model`、`sl-model-adapters`、`sl-precision` 和 `sl-daemon` 是独立 Git 仓库，作为 Git submodule 接入。StarLight 主仓库只记录各 submodule 的仓库 URL 与固定提交，不纳入它们的源文件；递归克隆后，submodule 内容会检出到对应目录。只有具备 Cargo manifest 并列入根 workspace 的组件才参与 Cargo workspace 构建；`sl-model-adapters` 当前尚无 Cargo package，因此暂未列入 workspace。其他路径是否为独立仓库，以 `.gitmodules` 和 Git 索引为准。各组件通过 Rust crate 边界组织；具体发布、依赖和开发约定以对应组件文档为准。

## 当前实现状态

StarLight 仍处于基础能力建设阶段。README 描述的是组件定位和目标协作方式，不应视为所有架构能力均已实现。

- `sl-model` 已提供模型包与张量数据契约，并有 Safetensors reader；格式支持仍在扩展。
- `sl-model-adapters` 已作为独立 submodule 接入，但当前仅有仓库说明和许可证，尚无 Cargo package、公开 API 或适配实现。
- `sl-precision` 已提供对称分组 INT4/INT8 量化与反量化能力，并包含 Safetensors 量化命令行工具。
- `sl-graph`、`sl-core`、`sl-device`、`sl-daemon` 和当前 CPU/Vulkan 后端仍在基线或骨架阶段。StarLight Graph 的并发图执行能力、完整设备执行链路及服务端调度能力需以实际实现和测试为准。
- CUDA、NPU、分布式通信等属于可扩展方向；当前 workspace 不包含已实现的 CUDA 后端或通信组件。
- `starlight-nano` 与 `starlight-cluster` 当前为装配入口骨架，具体可用能力随底层组件实现逐步增加。

如需了解模块边界与发行版管线，请阅读 [ARCH-0002：模块边界与发行版业务管线契约](<docs/ARCH-0002: 模块边界与发行版业务管线契约.md>)；模型制品、多模态输入和流式推理的数据边界见 [ARCH-0003：模型制品与多模态推理数据流设计](<docs/ARCH-0003: 模型制品与多模态推理数据流设计.md>)。其他架构文档见[文档索引](docs/README.md)；早期规划中的设计目标不自动代表已交付功能。

## 开始使用

需要安装 Rust 稳定版工具链。克隆仓库并进入目录：

```sh
git clone https://github.com/SeekStarAI/StarLight.git
cd StarLight
```

检查 workspace 并运行已实现组件的测试：

```sh
cargo check --workspace
cargo test -p sl-model
cargo test -p sl-precision
```

`sl-precision` Safetensors 量化命令示例：

```sh
cargo run --release -p sl-precision --bin sl-quantize -- \
  --input model.safetensors \
  --output model-int4.safetensors \
  --format int4 \
  --group-size 128
```

输出文件会将量化载荷和 scale 作为张量保存，并使用 StarLight 元数据记录恢复信息；它不是可直接交给普通模型加载器的权重文件。更多参数与限制见 [`sl-precision` 文档](crates/sl-precision/README.md)。

## 开发与贡献

请先阅读相关 crate 的 README 和开发文档，再确认接口是否已经稳定。新设备通常通过实现 `sl-device` 契约并提供对应后端接入；新的模型格式由 `sl-model` 的格式适配层负责；精度转换能力应放在 `sl-precision` 或适当的离线工具中；请求队列、服务生命周期和集群策略应留在运行时外围，而不是塞入设备内核或图执行组件。

提交改动时，应使用对应模块的测试和数值基准验证行为。当前架构仍在演进，文档和实现若有差异，应以代码、测试及对应模块的最新开发说明为准。

## 许可

本仓库采用双重许可，可选择 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。