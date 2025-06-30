// types.rs
// 定义通用类型，如专家到GPU的映射、门控权重、常量等辅助类型。
use serde::{Deserialize, Serialize};

/// 专家到GPU的映射信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertGpuMapping {
    pub expert_id: usize,
    pub gpu_id: i32,
    pub memory_required: u64, // MB
}

/// 门控权重信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateWeights {
    pub weights: Vec<f32>,
    pub top_k: usize,
}

// 常量定义，避免硬编码
pub const EXPERT_ID_SIZE: usize = 4;
pub const LAYER_ID_SIZE: usize = 4;
pub const GATE_WEIGHT_SIZE: usize = 4; 