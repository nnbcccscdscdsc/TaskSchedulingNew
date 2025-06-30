// config.rs
// 调度器全局配置结构体及其默认实现，包含最大并发任务数、批处理大小和可用GPU列表。
use serde::{Deserialize, Serialize};

/// 模型信息，包含模型类型、专家数、隐藏层大小等关键参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_type: String,
    pub num_experts: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
}

/// 用于直接反序列化模型目录中 config.json 的结构体
/// 使用 serde 属性来处理字段名不匹配的问题 (e.g., "d_model" -> hidden_size)
#[derive(Debug, Deserialize)]
pub(crate) struct ModelConfigJson {
    model_type: String,
    num_experts: usize,
    #[serde(rename = "d_model")]
    hidden_size: usize,
    #[serde(rename = "d_ff")]
    intermediate_size: usize,
    num_layers: usize,
}

// 为 ModelConfigJson 实现一个转换方法，使其可以轻松地转为 ModelInfo
impl From<ModelConfigJson> for ModelInfo {
    fn from(config_json: ModelConfigJson) -> Self {
        Self {
            model_type: config_json.model_type,
            num_experts: config_json.num_experts,
            hidden_size: config_json.hidden_size,
            intermediate_size: config_json.intermediate_size,
            num_layers: config_json.num_layers,
        }
    }
}

/// 调度器全局配置，控制任务并发、批大小和可用GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// 最大并发任务数
    pub max_concurrent_tasks: usize,
    /// 默认批处理大小
    pub default_batch_size: usize,
    /// 可用GPU设备ID列表
    pub gpu_ids: Vec<i32>,
}

impl Default for SchedulerConfig {
    /// 默认配置：最大4个并发任务，批大小为1，仅使用0号GPU
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 4,
            default_batch_size: 1,
            gpu_ids: vec![0],
        }
    }
}