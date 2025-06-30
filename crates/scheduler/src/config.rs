// config.rs
// 调度器全局配置结构体及其默认实现，包含最大并发任务数、批处理大小和可用GPU列表。
use serde::{Deserialize, Serialize};

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