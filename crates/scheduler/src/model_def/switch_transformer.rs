//! crates/scheduler/src/model_def/switch_transformer.rs
//!
//! 在这里我们用 `tch` 来重新定义 Switch Transformer 的关键部分，
//! 以便能加载预训练权重并验证我们的任务拆分逻辑。

use tch::{nn, Tensor};

/// 定义单个专家网络。
/// 它通常是一个简单的两层前馈网络。
#[derive(Debug)]
pub struct Expert {
    wi: nn::Linear,
    wo: nn::Linear,
}

impl Expert {
    /// 创建一个新的 Expert。
    /// `p` 是 `VarStore` 的路径，例如 "encoder.block.0.layer.1.mlp.experts.expert_0"
    pub fn new(p: nn::Path, hidden_size: i64, intermediate_size: i64) -> Self {
        let wi = nn::linear(&p / "wi", hidden_size, intermediate_size, Default::default());
        let wo = nn::linear(&p / "wo", intermediate_size, hidden_size, Default::default());
        Self { wi, wo }
    }
}

/// 定义稀疏MLP层，这是MoE的核心。
#[derive(Debug)]
pub struct SwitchTransformersSparseMLP {
    router: nn::Linear,
    experts: nn::ModuleList<Expert>,
}

impl SwitchTransformersSparseMLP {
    /// 创建一个新的 SparseMLP 层。
    /// `p` 是 `VarStore` 的路径，例如 "encoder.block.0.layer.1.mlp"
    pub fn new(p: nn::Path, config: &crate::config::ModelInfo) -> Self {
        let router = nn::linear(&p / "router", config.hidden_size as i64, config.num_experts as i64, Default::default());
        let mut experts = nn::ModuleList::new();
        for i in 0..config.num_experts {
            let expert_path = &p / "experts" / format!("expert_{}", i);
            let expert = Expert::new(expert_path, config.hidden_size as i64, config.intermediate_size as i64);
            experts.add(expert);
        }
        Self { router, experts }
    }

    /// 执行前向传播，但我们只关心 router 的输出。
    ///
    /// 返回:
    /// * `router_logits`: 一个形状为 [batch_size, seq_len, num_experts] 的张量，
    ///   包含了每个 token 被分配到每个 expert 的原始分数。
    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        // 只计算并返回 router 的输出，用于验证
        hidden_states.apply(&self.router)
    }
} 