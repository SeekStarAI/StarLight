# starlight-cluster

面向裸金属服务器与多卡集群的发行版装配入口。按 RFC-0001 规划组合 `sl-core` 与 `sl-daemon`，并在后续接入 CUDA 和通信后端。

调度、IPC 与跨机通信属于发行版外围服务，不进入核心微内核。当前目录仅包含 workspace 启动骨架。
