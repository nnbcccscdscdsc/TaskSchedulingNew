// task_splitter.rs
// 任务拆分器，负责将MOE任务按专家、层、批次等策略拆分为多个子任务。
use crate::config::ModelInfo;
use crate::error::{Error, Result};
use crate::task::{MoeTask, TaskPriority, TaskStatus};
use crate::types::*;
use crate::data_preparator::DataPreparator;
use crate::result_merger::ResultMerger;
use crate::task_executor::TaskExecutor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use std::path::Path;
use std::fs::File;
use crate::config::ModelConfigJson;
use std::io::Read;

// 常量定义，避免硬编码
const EXPERT_ID_SIZE: usize = 4;
const LAYER_ID_SIZE: usize = 4;
const GATE_WEIGHT_SIZE: usize = 4;

/// MOE任务拆分策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// 按专家拆分：每个专家一个任务
    ByExpert,
    /// 按层拆分：每个MOE层一个任务
    ByLayer,
    /// 按批次拆分：将输入分批处理
    ByBatch { batch_size: usize },
    /// 混合策略：结合多种拆分方式
    Hybrid { expert_split: bool, layer_split: bool, batch_size: usize },
}

/// 任务拆分器，负责将MOE模型推理任务拆分为多个子任务
/// 模型信息：用于标识模型类型、专家数量、隐藏层大小、中间层大小、层数等。
/// 拆分策略：用于标识拆分策略，如按专家、按层、按批次、混合策略等。
/// 数据准备器：用于准备数据，如专家数据、层数据、批次数据等。
/// 结果合并器：用于合并结果，如专家结果、层结果、批次结果等。
pub struct TaskSplitter {
    /// 模型信息
    pub model_info: ModelInfo,
    /// 拆分策略
    pub strategy: SplitStrategy,
    /// 数据准备器
    pub data_preparator: Arc<DataPreparator>,
    /// 结果合并器
    pub result_merger: Arc<ResultMerger>,
}

/// 任务拆分器实现
impl TaskSplitter {
    /// 创建新的任务拆分器
    pub fn new(model_info: ModelInfo, strategy: SplitStrategy) -> Self {
        let data_preparator = Arc::new(DataPreparator::new(model_info.clone()));
        let result_merger = Arc::new(ResultMerger::new(model_info.clone()));
        
        Self {
            model_info,
            strategy,
            data_preparator,
            result_merger,
        }
    }

    /// 从模型目录自动读取 config.json 并初始化 ModelInfo
    /// 如果 config.json 不存在则返回错误
    pub fn new_from_model_dir(model_dir: &str, strategy: SplitStrategy) -> Result<Self> {
        // 拼接 config.json 路径
        let config_path = Path::new(model_dir).join("config.json");
        if !config_path.exists() {
            return Err(Error::ConfigError(format!("模型目录 {} 下未找到 config.json", model_dir)));
        }
        // 读取 config.json 文件
        let mut file = File::open(&config_path)
            .map_err(|e| Error::ConfigError(format!("打开 config.json 失败: {}", e)))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| Error::ConfigError(format!("读取 config.json 失败: {}", e)))?;
        // 解析 json
        let config_json: ModelConfigJson = serde_json::from_str(&contents)
            .map_err(|e| Error::ConfigError(format!("解析 config.json 失败: {}", e)))?;
        // 转换为 ModelInfo
        let model_info = ModelInfo::from(config_json);
        // 调用原有构造方法
        Ok(Self::new(model_info, strategy))
    }

    /// 拆分MOE任务
    pub fn split_task(&self, input_data: &[u8], task_id: &str, priority: TaskPriority) -> Result<Vec<MoeTask>> {
        // 验证输入数据格式
        self.validate_input_data(input_data)?;
        
        match &self.strategy {
            SplitStrategy::ByExpert => self.split_by_expert(input_data, task_id, priority),
            SplitStrategy::ByLayer => self.split_by_layer(input_data, task_id, priority),
            SplitStrategy::ByBatch { batch_size } => self.split_by_batch(input_data, task_id, priority, *batch_size),
            SplitStrategy::Hybrid { expert_split, layer_split, batch_size } => {
                self.split_hybrid(input_data, task_id, priority, *expert_split, *layer_split, *batch_size)
            }
        }
    }

    /// 按专家拆分任务
    fn split_by_expert(&self, input_data: &[u8], parent_task_id: &str, priority: TaskPriority) -> Result<Vec<MoeTask>> {
        let mut tasks = Vec::new();
        
        for expert_id in 0..self.model_info.num_experts {
            let task_id = self.generate_task_id(parent_task_id, "expert", expert_id);
            
            // 为每个专家创建专门的任务数据
            let expert_data = self.data_preparator.prepare_expert_data(input_data, expert_id)?;
            
            let task = MoeTask {
                task_id,
                input_data: expert_data,
                status: crate::task::TaskStatus::Pending,
                result: None,
                priority,
                stream_id: Some(expert_id),
                parent_task_id: Some(parent_task_id.to_string()),
            };
            
            tasks.push(task);
        }
        
        println!("按专家拆分为 {} 个任务", tasks.len());
        Ok(tasks)
    }

    /// 按层拆分任务
    fn split_by_layer(&self, input_data: &[u8], parent_task_id: &str, priority: TaskPriority) -> Result<Vec<MoeTask>> {
        let mut tasks = Vec::new();
        
        for layer_id in 0..self.model_info.num_layers {
            let task_id = self.generate_task_id(parent_task_id, "layer", layer_id);
            
            // 为每个层创建专门的任务数据
            let layer_data = self.data_preparator.prepare_layer_data(input_data, layer_id)?;
            
            let task = MoeTask {
                task_id,
                input_data: layer_data,
                status: crate::task::TaskStatus::Pending,
                result: None,
                priority,
                stream_id: Some(layer_id),
                parent_task_id: Some(parent_task_id.to_string()),
            };
            
            tasks.push(task);
        }
        
        println!("按层拆分为 {} 个任务", tasks.len());
        Ok(tasks)
    }

    /// 按批次拆分任务
    fn split_by_batch(&self, input_data: &[u8], parent_task_id: &str, priority: TaskPriority, batch_size: usize) -> Result<Vec<MoeTask>> {
        let mut tasks = Vec::new();
        
        // 计算需要多少个批次，考虑填充
        let total_size = input_data.len();
        let num_batches = (total_size + batch_size - 1) / batch_size; // 向上取整
        
        for batch_id in 0..num_batches {
            let task_id = self.generate_task_id(parent_task_id, "batch", batch_id);
            
            let start = batch_id * batch_size;
            let end = std::cmp::min(start + batch_size, total_size);
            let mut batch_data = input_data[start..end].to_vec();
            
            // 如果最后一个批次不足，进行填充
            if batch_data.len() < batch_size {
                let padding_size = batch_size - batch_data.len();
                batch_data.extend(vec![0u8; padding_size]);
            }
            
            let task = MoeTask {
                task_id,
                input_data: batch_data,
                status: crate::task::TaskStatus::Pending,
                result: None,
                priority,
                stream_id: Some(batch_id),
                parent_task_id: Some(parent_task_id.to_string()),
            };
            
            tasks.push(task);
        }
        
        println!("按批次拆分为 {} 个任务", tasks.len());
        Ok(tasks)
    }

    /// 混合拆分策略
    fn split_hybrid(
        &self, 
        input_data: &[u8], 
        parent_task_id: &str, 
        priority: TaskPriority,
        expert_split: bool, 
        layer_split: bool, 
        batch_size: usize
    ) -> Result<Vec<MoeTask>> {
        let mut tasks = Vec::new();
        
        if expert_split && layer_split {
            // 先按层拆分，再按专家拆分
            for layer_id in 0..self.model_info.num_layers {
                for expert_id in 0..self.model_info.num_experts {
                    let task_id = self.generate_task_id(parent_task_id, &format!("layer_{}_expert", layer_id), expert_id);
                    
                    let layer_expert_data = self.data_preparator.prepare_layer_expert_data(input_data, layer_id, expert_id)?;
                    
                    let task = MoeTask {
                        task_id,
                        input_data: layer_expert_data,
                        status: crate::task::TaskStatus::Pending,
                        result: None,
                        priority,
                        stream_id: Some(layer_id * self.model_info.num_experts + expert_id),
                        parent_task_id: Some(parent_task_id.to_string()),
                    };
                    
                    tasks.push(task);
                }
            }
        } else if expert_split && batch_size > 0 {
            // 专家拆分 + 批次拆分
            let expert_tasks = self.split_by_expert(input_data, parent_task_id, priority)?;
            for expert_task in expert_tasks {
                let batch_tasks = self.split_by_batch(&expert_task.input_data, &expert_task.task_id, priority, batch_size)?;
                tasks.extend(batch_tasks);
            }
        } else if layer_split && batch_size > 0 {
            // 层拆分 + 批次拆分
            let layer_tasks = self.split_by_layer(input_data, parent_task_id, priority)?;
            for layer_task in layer_tasks {
                let batch_tasks = self.split_by_batch(&layer_task.input_data, &layer_task.task_id, priority, batch_size)?;
                tasks.extend(batch_tasks);
            }
        } else if expert_split {
            return self.split_by_expert(input_data, parent_task_id, priority);
        } else if layer_split {
            return self.split_by_layer(input_data, parent_task_id, priority);
        } else {
            return self.split_by_batch(input_data, parent_task_id, priority, batch_size);
        }
        
        println!("混合拆分为 {} 个任务", tasks.len());
        Ok(tasks)
    }

    /// 生成任务ID
    fn generate_task_id(&self, parent_id: &str, prefix: &str, id: usize) -> String {
        format!("{}_{}_{}", parent_id, prefix, id)
    }

    /// 验证输入数据格式
    fn validate_input_data(&self, input_data: &[u8]) -> Result<()> {
        if input_data.is_empty() {
            return Err(Error::InferenceError("输入数据为空".to_string()));
        }
        
        // 检查数据大小是否合理
        let min_size = self.model_info.hidden_size * 4; // 假设每个元素4字节
        if input_data.len() < min_size {
            return Err(Error::InferenceError(format!(
                "输入数据大小 {} 小于最小要求 {}", input_data.len(), min_size
            )));
        }
        
        Ok(())
    }

    /// 获取任务依赖关系
    pub fn get_task_dependencies(&self, tasks: &[MoeTask]) -> Result<HashMap<String, Vec<String>>> {
        let mut dependencies = HashMap::new();
        
        // 根据拆分策略确定依赖关系
        match &self.strategy {
            SplitStrategy::ByExpert => {
                // 专家任务之间没有依赖关系，可以并行执行
                for task in tasks {
                    dependencies.insert(task.task_id.clone(), Vec::new());
                }
            }
            SplitStrategy::ByLayer => {
                // 层任务有顺序依赖关系，考虑残差连接
                for (i, task) in tasks.iter().enumerate() {
                    let mut deps = Vec::new();
                    if i > 0 {
                        deps.push(tasks[i-1].task_id.clone());
                    }
                    // 如果有残差连接，可能需要依赖更早的层
                    if i >= 2 {
                        deps.push(tasks[i-2].task_id.clone());
                    }
                    dependencies.insert(task.task_id.clone(), deps);
                }
            }
            SplitStrategy::ByBatch { .. } => {
                // 批次任务可以并行执行
                for task in tasks {
                    dependencies.insert(task.task_id.clone(), Vec::new());
                }
            }
            SplitStrategy::Hybrid { expert_split, layer_split, .. } => {
                // 混合策略的依赖关系
                if *expert_split && *layer_split {
                    // 层内专家并行，层间顺序
                    let experts_per_layer = self.model_info.num_experts;
                    let layers = self.model_info.num_layers;
                    
                    for layer_id in 0..layers {
                        for expert_id in 0..experts_per_layer {
                            let task_idx = layer_id * experts_per_layer + expert_id;
                            let mut deps = Vec::new();
                            
                            // 同一层内的专家任务没有依赖
                            // 不同层之间有依赖关系
                            if layer_id > 0 {
                                for prev_expert in 0..experts_per_layer {
                                    let prev_task_idx = (layer_id - 1) * experts_per_layer + prev_expert;
                                    if prev_task_idx < tasks.len() {
                                        deps.push(tasks[prev_task_idx].task_id.clone());
                                    }
                                }
                            }
                            
                            if task_idx < tasks.len() {
                                dependencies.insert(tasks[task_idx].task_id.clone(), deps);
                            }
                        }
                    }
                } else {
                    // 其他混合策略的依赖关系
                    for task in tasks {
                        dependencies.insert(task.task_id.clone(), Vec::new());
                    }
                }
            }
        }
        
        Ok(dependencies)
    }

    /// 合并任务结果
    pub fn merge_results(&self, results: &[Vec<u8>], gate_weights: Option<GateWeights>) -> Result<Vec<u8>> {
        self.result_merger.merge_results(results, gate_weights, &self.strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_splitter_creation() {
        let model_info = ModelInfo {
            model_type: "switch_transformer".to_string(),
            num_experts: 8,
            hidden_size: 512,
            intermediate_size: 2048,
            num_layers: 12,
        };
        
        let strategy = SplitStrategy::ByExpert;
        let splitter = TaskSplitter::new(model_info, strategy);
        
        assert_eq!(splitter.data_preparator.read().unwrap().len(), 0);
    }

    #[test]
    fn test_data_preparator() {
        let model_info = ModelInfo {
            model_type: "switch_transformer".to_string(),
            num_experts: 4,
            hidden_size: 256,
            intermediate_size: 1024,
            num_layers: 6,
        };
        
        let preparator = DataPreparator::new(model_info);
        let input_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        
        let expert_data = preparator.prepare_expert_data(&input_data, 1).unwrap();
        assert!(expert_data.len() > input_data.len());
        
        let layer_data = preparator.prepare_layer_data(&input_data, 2).unwrap();
        assert!(layer_data.len() > input_data.len());
    }

    #[test]
    fn test_result_merger() {
        let model_info = ModelInfo {
            model_type: "switch_transformer".to_string(),
            num_experts: 2,
            hidden_size: 128,
            intermediate_size: 512,
            num_layers: 4,
        };
        
        let merger = ResultMerger::new(model_info);
        
        // 创建模拟结果
        let mut results = Vec::new();
        for i in 0..2 {
            let mut result = Vec::new();
            for j in 0..32 { // 8个f32值
                let value = (i * 10 + j) as f32;
                result.extend_from_slice(&value.to_le_bytes());
            }
            results.push(result);
        }
        
        let gate_weights = GateWeights {
            weights: vec![0.7, 0.3],
            top_k: 2,
        };
        
        let merged = merger.merge_expert_results(&results, Some(gate_weights)).unwrap();
        assert!(!merged.is_empty());
    }

    #[test]
    fn test_task_executor() {
        let model_info = ModelInfo {
            model_type: "switch_transformer".to_string(),
            num_experts: 4,
            hidden_size: 256,
            intermediate_size: 1024,
            num_layers: 6,
        };
        
        let executor = TaskExecutor::new(model_info);
        
        let mut task = MoeTask {
            task_id: "test_expert_1".to_string(),
            input_data: vec![1, 2, 3, 4],
            status: crate::task::TaskStatus::Pending,
            result: None,
            priority: TaskPriority::Normal,
            stream_id: Some(0),
            parent_task_id: Some("parent".to_string()),
        };
        
        let result = executor.execute_task(&mut task);
        assert!(result.is_ok());
        assert!(matches!(task.status, crate::task::TaskStatus::Completed));
        assert!(task.result.is_some());
    }
}
