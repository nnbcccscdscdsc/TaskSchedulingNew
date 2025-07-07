# TaskScheduling

本项目用于调度和执行MOE模型（Mixture of Experts）任务，支持多GPU资源管理和批量任务分发。

## 主要功能
- MOE模型的任务拆分与调度
- 单个GPU资源管理与多核并行
- WASI-NN适配

## 构建和运行

### 构建项目
```bash
cargo build
```

### 运行示例

#### 1. 模型下载与任务拆分示例
```bash
cargo run --example download_and_split_example
```
这个示例演示了：
- 下载Switch Transformer模型
- 验证模型完整性
- 使用按专家策略拆分任务
- 表格化输出拆分结果

#### 2. 验证任务拆分逻辑
```bash
cargo run --example verify_split_logic
```
这个示例用于验证TaskSplitter的核心逻辑：
- 加载真实的PyTorch Switch Transformer模型
- 获取模型真实的门控权重和路由决策
- 比较拆分结果与模型内部路由结果

#### 3. 任务拆分测试
```bash
cargo run --example split_task_test
```
这个示例测试任务拆分功能：
- 从本地加载模型信息
- 创建模拟输入数据
- 执行任务拆分并输出结果

#### 4. 模型下载
```bash
cargo run --example download_model
```
这个示例专门用于下载模型：
- 下载指定的Switch Transformer模型
- 支持国内镜像加速
- 验证下载结果

#### 5. 综合测试（推荐）
```bash
cargo run --example comprehensive_test
```
这个示例是完整的系统测试：
- 测试所有拆分策略（专家、层、批次、混合）
- 验证参数有效性检查
- 测试任务执行和内存管理
- 验证负载均衡功能
- 测试边缘情况和错误处理

## 目录结构
- crates/scheduler/src/
  - task_splitter.rs      // 任务拆分器 支持多种拆分策略（按专家、按层、按批次或混合策略）, 生成带有依赖关系的子任务
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
- prettytable
- serde_json

## 环境要求
- Rust 1.70+
- CUDA支持（用于GPU加速）
- Python 3.7+（用于模型下载）
- transformers、torch、sentencepiece库（用于模型处理）

## 最新改进

### 代码质量提升
- ✅ 完善了SplitStrategy的参数验证
- ✅ 修复了result_merger.rs中的模式匹配问题
- ✅ 添加了内存池管理和负载均衡功能
- ✅ 实现了完整的错误处理机制
- ✅ 添加了拆分结果验证功能

### 功能增强
- ✅ 支持混合拆分策略的比例控制
- ✅ 添加了任务依赖关系分析
- ✅ 实现了GPU内存管理和复用
- ✅ 支持批量任务执行
- ✅ 添加了边缘情况测试

### 测试覆盖
- ✅ 创建了综合测试示例
- ✅ 添加了参数验证测试
- ✅ 实现了错误处理测试
- ✅ 支持多种拆分策略测试