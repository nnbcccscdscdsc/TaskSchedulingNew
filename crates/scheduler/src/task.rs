use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeTask {
    pub task_id: String,
    pub input_data: Vec<u8>,
    pub status: TaskStatus,
    pub result: Option<Vec<u8>>,
}