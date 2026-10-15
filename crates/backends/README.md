# Backend 插件

本目录承载 StarLight 的具体算力后端。后端实现 `sl-device` 定义的 VDI 契约，并按设备能力调用 `sl-precision` 提供的数据排布工具；算法实现不进入 `sl-core`。

当前 workspace 提供 CPU 与 Vulkan 两个骨架。CUDA 和第三方 NPU 后端待对应仓库与构建条件明确后接入。
