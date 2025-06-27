# TaskScheduling

本项目用于调度和执行MOE模型（Mixture of Experts）任务，支持多GPU资源管理和批量任务分发。

## 主要功能
- MOE模型的任务拆分与调度
- GPU资源管理与多核并行
- WASI-NN适配

## 目录结构
- crates/scheduler/src/error.rs: 错误类型定义
- crates/scheduler/src/config.rs: 配置结构体
- crates/scheduler/src/task.rs: 任务结构体
- crates/scheduler/src/scheduler.rs: 任务调度器
- crates/scheduler/lib.rs: 模块导出

## 依赖
- rustacuda
- serde
- uuid
- rand
- anyhow