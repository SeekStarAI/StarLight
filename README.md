
---

# 🌟 StarLight (星光推理引擎)

**StarLight** 是由 **SeekStarAI 实验室 (SeekStarAI)** 研发的新一代大模型（LLM）高性能推理引擎底座。

我们拒绝传统深度学习框架“单体巨石（Monolithic）”式的沉重包袱，回归系统级软件工程本质。StarLight 将 **GNU/Linux 的微内核与机制/策略分离哲学**、**Vulkan 的显式无锁管线** 以及 **现代图形学 TBDR（Tile-Based Deferred Rendering）的片上驻留思想** 深度融合，打造出一套既能裁剪为几兆大小塞进家用 AIPC，又能水平扩展至大规模裸金属商业集群的模块化推理基础设施。

---

## ⚡ 核心设计特性 (Key Highlights)

* **🧩 彻底的微内核与发行版组装（Microkernel & Distros）**
核心层（`sl-core`）仅负责虚拟页表寻址与无状态单步推进（`step`），不包含任何网络协议或业务排队逻辑。通过组合不同外围组件，只需几十行装配代码即可构建面向端侧的 `StarLight-Nano` 或面向高并发服务器的 `StarLight-Cluster`。
* **🔥 零运行时堆内存分配（Zero-Allocation at Runtime）**
在图编译期通过区间图着色算法（Interval Coloring）完成全局 `Scratchpad` 激活值内存复用规划。单步推理热路径上严格执行 **`0 malloc`** 门禁，彻底消除运行时内存碎片与延迟毛刺（Jitter）。
* **🛡️ TBDR 启发的片上驻留与预剔除（Tile Residency & Early Culling）**
针对 Decode 阶段极端的 Memory-Bound 瓶颈，引入显式暂存缓冲（`TransientBuffer`）语义，将局部注意力计算约束在硬件片上缓存（SRAM / Shared Memory / L1-L2 Cache）内，结合粗粒度可见性测试提前剪除低贡献 KV 块加载。
* **🗺️ 硬件级稀疏页表映射（Sparse PagedAttention）**
借鉴 Vulkan Sparse Binding 机制，实现逻辑连续、物理离散的 KV-Cache 管理，配合硬件感知的权重交织重排（`sl-precision`），确保底层 SIMD / Tensor Core 始终处于满带宽对齐流式访存状态。
* **🔌 统一虚拟设备接口（VDI - Virtual Device Interface）**
通过零成本的 Rust `Trait` 抽象底层算力。原生支持手写 AVX2/AVX-512/NEON 的纯 CPU 后端、跨平台 Vulkan Compute 后端，以及专有硬件（CUDA / ROCm / 第三方 NPU）的无侵入插件化挂载。

---

## 🏗️ 系统架构与模块拓扑 (Architecture & Modules)

本仓库作为 StarLight 的**主控工作区（Workspace）與发行版装配中心**，通过 Git Submodules 聚合了 SeekStarAI 旗下各个可独立演进、独立开源的核心组件库：

```text
SeekStarAI/StarLight (主控仓库)
├── docs/                         # 📚 架构哲学、RFC 工程规范与决策档案库
├── crates/                       # 🧱 核心微内核与独立开源基座 (Git Submodules)
│   ├── sl-precision/             #   ├─ [工具链] 硬件感知量化、交织重排与精度校准引擎
│   ├── sl-graph/                 #   ├─ [编译器] 静态 DAG 融合、TBDR 标记与内存着色规划器
│   ├── sl-device/                #   ├─ [驱动层] VDI 虚拟硬件抽象接口契约 (Traits)
│   ├── sl-core/                  #   ├─ [微内核] 无状态单步驱动器与 PagedKV 虚拟内存管理
│   └── backends/                 #   └─ [硬后端] 插件式算子实现
│       ├── backend-cpu/          #        ├─ 纯 Rust 手写 SIMD 内核 (AVX2/AVX-512/AMX/NEON)
│       ├── backend-vulkan/       #        ├─ 跨平台 SPIR-V 通用计算后端
│       └── backend-cuda/         #        └─ 面向 NVIDIA GPU 的 Tile 特化后端
├── services/
│   └── sl-daemon/                # ⚙️ Master-Worker 进程监管、IPC 共享内存与动态批处理调度器
├── distros/                      # 📦 官方发行版装配入口 (Distributions)
│   ├── starlight-nano/           #   ├─ 端侧/AIPC 极简直通发行版 (含 PyO3 绑定)
│   └── starlight-cluster/        #   └─ 商业裸金属集群高并发服务发行版
└── tests/
    └── golden-baselines/         # 🎯 跨模块联合调试的标准数值基准快照

```

---

## 📖 架构文档与工程档案 (Documentation Index)

**强烈建议所有团队成员与贡献者在编写代码前，先阅读 `/docs` 目录下的核心档案：**

* 🧭 [文档导航总首页 (docs/README.md)](docs/README.md)
* **💡 架构哲学备忘录**
* **[PHIL-0001: 为什么我们将 StarLight 构建为“推理领域的 GNU/Linux”？](<docs/PHIL-0001: 为什么我们将 StarLight 构建为“推理领域的 GNU Linux”？.md>)** — 理解微内核解耦思想与“50 行代码造发行版”的组装逻辑。


* **📐 系统工程规范与任务拆解 (RFCs)**
* **[RFC-0001: StarLight 系统工程与架构规划草案 (2026-10-16)](<docs/RFC-0001: StarLight 系统工程与架构规划草案 (2026-10-16).md>)** — 核心契约定义、零分配原则及第一阶段并行任务包（WP-01 ~ WP-05）验收标准。



---

## 🚀 快速开始与开发者工作流 (Quick Start & Workflow)

### 1. 克隆完整工作区

由于核心子模块采用独立仓库管理，克隆主仓库时请务必带上 `--recurse-submodules` 参数：

```bash
git clone --recurse-submodules https://github.com/SeekStarAI/StarLight.git
cd StarLight

```

*(如果已经克隆了主仓库但子目录为空，请执行：`git submodule update --init --recursive`)*

### 2. 构建与运行基准测试

确保本地已安装最新稳定版 Rust 工具链（`rustup update stable`）：

```bash
# 检查全局 Workspace 接口契约编译
cargo check --workspace

# 运行指定模块（如内存规划器或精度打包器）的单元测试
cargo test --package sl-graph
cargo test --package sl-precision

# 编译端侧发行版 StarLight-Nano (Release 优化模式)
cargo build --release --package starlight-nano

```

### 3. 多模块并行开发守则

* **独立提交，互不污染：** 若你认领了某个子模块的任务包（例如 `crates/sl-precision`），请直接进入该子目录切分支、开发并推送到该子模块的远端仓库，**严禁在主控仓库直接夹带子模块的业务代码修改**。
* **严守红线：** 在 `sl-core` 与 `backend-*` 的单步执行路径中，严禁任何运行期动态堆分配；所有新增算子必须通过 `tests/golden-baselines/` 下的浮点精度对齐测试。

---

## 🤝 参与贡献与任务认领 (Contributing)

StarLight 目前正处于 **Phase 1（核心通路打通与基线建设）** 的并行冲刺阶段。请查看 **[RFC-0001 任务包清单](https://www.google.com/search?q=./docs/01-rfcs/0001-core-architecture-and-wbs-202610.md%235-%E7%AC%AC%E4%B8%80%E9%98%B6%E6%AE%B5%E5%B9%B6%E8%A1%8C%E4%BB%BB%E5%8A%A1%E5%8C%85work-packages%E5%88%86%E5%8F%91%E4%B8%8E%E9%AA%8C%E6%94%B6%E6%A0%87%E5%87%86)** 了解当前开放的 Work Packages，并在各子模块下的 `DEVELOPMENT.md` 中查阅具体的开发边界与测试指令。

---

## 📜 开源许可 (License)

本项目采用双重许可，您可以选择 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。

Copyright (c) 2026 SeekStarAI and StarLight Contributors.