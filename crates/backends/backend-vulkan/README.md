# backend-vulkan

StarLight 的 Vulkan/SPIR-V 通用计算后端，目标是实现 `sl-device` 的 VDI 契约，并将可分块算子映射到 GPU 计算管线。

临时资源应遵循图规划给出的生命周期与片上驻留约束。当前目录仅为 workspace crate 骨架；实际 Vulkan 依赖和设备能力将在后续实现阶段加入。
