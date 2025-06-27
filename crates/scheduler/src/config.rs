use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_concurrent_tasks: usize,
    pub default_batch_size: usize,
    pub gpu_ids: Vec<i32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 4,
            default_batch_size: 1,
            gpu_ids: vec![0],
        }
    }
}