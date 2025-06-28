use crate::error::Result;
use serde::{Deserialize, Serialize};
use crate::task::{MoeTask, TaskStatus};
use crate::config::SchedulerConfig;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub model_path: String,
    pub batch_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub device_type: String,
    pub device_id: i32,
    pub use_quantization: bool,
    pub quantization_bits: u8,
}

pub trait MoeAdapter {
    fn load_model(&mut self, model_path: &str) -> Result<()>;
    fn compute(&self, input_data: &[u8]) -> Result<Vec<u8>>;
    fn release_model(&mut self) -> Result<()>;
    fn get_model_id(&self) -> Option<&str>;
}

pub struct TaskScheduler {
    pub config: SchedulerConfig,
    pub queue: Arc<Mutex<VecDeque<MoeTask>>>,
}

impl TaskScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn submit_task(&self, task: MoeTask) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(task);
    }

    pub fn fetch_next_task(&self) -> Option<MoeTask> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }
}

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
