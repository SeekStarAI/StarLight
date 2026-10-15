# backend-cpu

StarLight 的纯 CPU 算力后端，目标是以 Rust `core::arch` 提供 AVX2、AVX-512 和 ARM NEON 算子，并实现 `sl-device` 的 VDI 契约。

实现需遵守零运行期临时堆分配约束，并以 `tests/golden-baselines/` 中的数值样例验证结果。当前目录仅为 workspace crate 骨架。
