# TaskScheduling

本项目用于调度和执行MOE模型（Mixture of Experts）任务，支持多GPU资源管理和批量任务分发。

## 主要功能
- MOE模型的任务拆分与调度
- GPU资源管理与多核并行
- WASI-NN适配

## 目录结构
- crates/scheduler/src/
  - task_splitter.rs      // 任务拆分器
  - data_preparator.rs    // 数据准备器
  - result_merger.rs      // 结果合并器
  - task_executor.rs      // 任务执行器
  - types.rs              // 通用类型
  - mod.rs                // 统一导出
## 依赖
- rustacuda
- serde
- uuid
- rand
- anyhow