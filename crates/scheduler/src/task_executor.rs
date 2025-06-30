// task_executor.rs
// 任务执行器，负责MOE任务的执行、重试、超时等处理。
use crate::model_downloader::ModelInfo;
use crate::error::{Error, Result};
use crate::task::{MoeTask, TaskStatus, TaskPriority};

pub struct TaskExecutor {
    pub model_info: ModelInfo,
    pub timeout_ms: u64,
    pub max_retries: u32,
}

impl TaskExecutor {
    pub fn new(model_info: ModelInfo) -> Self {
        Self {
            model_info,
            timeout_ms: 30000,
            max_retries: 3,
        }
    }

    pub fn set_timeout(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms;
    }

    pub fn set_max_retries(&mut self, max_retries: u32) {
        self.max_retries = max_retries;
    }

    pub fn execute_task(&self, task: &mut MoeTask) -> Result<()> {
        let start_time = std::time::Instant::now();
        task.status = TaskStatus::Running;
        let mut retry_count = 0;
        loop {
            match self.execute_single_task(task) {
                Ok(()) => {
                    task.status = TaskStatus::Completed;
                    return Ok(());
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= self.max_retries {
                        task.status = TaskStatus::Failed(e.to_string());
                        return Err(e);
                    }
                    if start_time.elapsed().as_millis() > self.timeout_ms as u128 {
                        task.status = TaskStatus::Failed("任务超时".to_string());
                        return Err(Error::InferenceError("任务执行超时".to_string()));
                    }
                    std::thread::sleep(std::time::Duration::from_millis((100 * retry_count).into()));
                }
            }
        }
    }

    fn execute_single_task(&self, task: &mut MoeTask) -> Result<()> {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let result = self.generate_mock_result(task)?;
        task.result = Some(result);
        Ok(())
    }

    fn generate_mock_result(&self, task: &MoeTask) -> Result<Vec<u8>> {
        let output_size = self.model_info.hidden_size * 4;
        let mut result = Vec::new();
        result.extend_from_slice(&(0u32).to_le_bytes());
        result.extend_from_slice(&(0u32).to_le_bytes());
        for i in 0..(output_size - 8) / 4 {
            let value = (i % 100) as f32 / 100.0;
            result.extend_from_slice(&value.to_le_bytes());
        }
        Ok(result)
    }
} 