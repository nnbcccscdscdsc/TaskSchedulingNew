use crate::task::{MoeTask, TaskStatus};
use crate::config::SchedulerConfig;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// 简单的任务调度器，支持任务队列的提交与获取
pub struct TaskScheduler {
    /// 调度器配置
    pub config: SchedulerConfig,
    /// 任务队列，线程安全
    pub queue: Arc<Mutex<VecDeque<MoeTask>>>,
}

impl TaskScheduler {
    /// 创建新的调度器实例
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// 提交一个新任务到队列
    pub fn submit_task(&self, task: MoeTask) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(task);
    }

    /// 获取下一个待执行任务（FIFO）
    pub fn fetch_next_task(&self) -> Option<MoeTask> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }
}