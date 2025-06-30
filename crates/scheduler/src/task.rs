// task.rs
// 定义MOE任务结构体、任务状态枚举、任务优先级等。
use serde::{Deserialize, Serialize};

/// 任务状态枚举，描述任务的生命周期
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    /// 等待执行
    Pending,
    /// 正在执行
    Running,
    /// 已完成
    Completed,
    /// 执行失败，包含失败原因
    Failed(String),
}

/// 任务优先级
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// MOE任务结构体，包含任务ID、输入数据、状态和结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeTask {
    /// 任务唯一ID
    pub task_id: String,
    /// 输入数据（通常为序列化后的张量）
    pub input_data: Vec<u8>,
    /// 当前任务状态
    pub status: TaskStatus,
    /// 推理结果（字节流），仅在Completed时有值
    pub result: Option<Vec<u8>>,
    /// 任务优先级
    pub priority: TaskPriority,
    /// 分配的逻辑流ID（可用于CUDA Stream或并发任务标识）
    pub stream_id: Option<usize>,
    /// 父任务ID（用于子任务）
    pub parent_task_id: Option<String>,
}