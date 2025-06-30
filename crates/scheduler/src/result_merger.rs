// result_merger.rs
// 结果合并器，负责合并各子任务（如专家、层、批次等）的推理结果。
use crate::model_downloader::ModelInfo;
use crate::error::{Error, Result};
use crate::types::*;
use crate::task_splitter::SplitStrategy;
 
/// 结果合并器，负责合并各子任务（如专家、层、批次等）的推理结果。
pub struct ResultMerger {
    pub model_info: ModelInfo,
}

/// 结果合并器实现
impl ResultMerger {
    // 创建结果合并器
    pub fn new(model_info: ModelInfo) -> Self {
        Self { model_info }
    }

    // 合并结果
    pub fn merge_results(&self, results: &[Vec<u8>], gate_weights: Option<GateWeights>, strategy: &SplitStrategy) -> Result<Vec<u8>> {
        match strategy {
            SplitStrategy::ByExpert => self.merge_expert_results(results, gate_weights),
            SplitStrategy::ByLayer => self.merge_layer_results(results),
            SplitStrategy::ByBatch { .. } => self.merge_batch_results(results),
            SplitStrategy::Hybrid { .. } => self.merge_hybrid_results(results, gate_weights),
        }
    }

    // 合并专家结果 加权求和
    fn merge_expert_results(&self, results: &[Vec<u8>], gate_weights: Option<GateWeights>) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有专家结果可合并".to_string()));
        }
        let gate_weights = gate_weights.ok_or_else(|| Error::InferenceError("专家结果合并需要门控权重".to_string()))?;
        if results.len() != self.model_info.num_experts {
            return Err(Error::InferenceError(format!("专家结果数量 {} 与专家数量 {} 不匹配", results.len(), self.model_info.num_experts)));
        }
        if gate_weights.weights.len() != self.model_info.num_experts {
            return Err(Error::InferenceError(format!("门控权重数量 {} 与专家数量 {} 不匹配", gate_weights.weights.len(), self.model_info.num_experts)));
        }
        let expert_output_size = results[0].len();
        let mut merged_result = vec![0.0f32; expert_output_size / 4];
        for (expert_id, result) in results.iter().enumerate() {
            if result.len() != expert_output_size {
                return Err(Error::InferenceError(format!("专家 {} 的输出大小 {} 与其他专家不一致 {}", expert_id, result.len(), expert_output_size)));
            }
            let weight = gate_weights.weights[expert_id];
            for (i, chunk) in result.chunks_exact(4).enumerate() {
                if i < merged_result.len() {
                    if let Ok(bytes) = chunk.try_into() {
                        let value = f32::from_le_bytes(bytes);
                        merged_result[i] += weight * value;
                    }
                }
            }
        }
        let mut final_result = Vec::new();
        for &value in &merged_result {
            final_result.extend_from_slice(&value.to_le_bytes());
        }
        Ok(final_result)
    }

    // 合并层结果 残差相加
    fn merge_layer_results(&self, results: &[Vec<u8>]) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有层结果可合并".to_string()));
        }
        if results.len() != self.model_info.num_layers {
            return Err(Error::InferenceError(format!("层结果数量 {} 与层数 {} 不匹配", results.len(), self.model_info.num_layers)));
        }
        let mut merged_result = Vec::new();
        for (layer_id, result) in results.iter().enumerate() {
            if layer_id == 0 {
                merged_result = result.clone();
            } else {
                if merged_result.len() != result.len() {
                    return Err(Error::InferenceError(format!("层 {} 的输出大小 {} 与前一层的残差大小 {} 不匹配", layer_id, result.len(), merged_result.len())));
                }
                for (i, (residual, current)) in merged_result.chunks_exact(4).zip(result.chunks_exact(4)).enumerate() {
                    if let (Ok(residual_bytes), Ok(current_bytes)) = (residual.try_into(), current.try_into()) {
                        let residual_val = f32::from_le_bytes(residual_bytes);
                        let current_val = f32::from_le_bytes(current_bytes);
                        let sum = residual_val + current_val;
                        let start_idx = i * 4;
                        if start_idx + 4 <= merged_result.len() {
                            merged_result[start_idx..start_idx + 4].copy_from_slice(&sum.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(merged_result)
    }

    // 合并批次结果 直接拼接
    fn merge_batch_results(&self, results: &[Vec<u8>]) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有批次结果可合并".to_string()));
        }
        let mut merged_result = Vec::new();
        for (batch_id, result) in results.iter().enumerate() {
            let actual_result = if batch_id == results.len() - 1 {
                self.remove_padding(result)?
            } else {
                result.clone()
            };
            merged_result.extend_from_slice(&actual_result);
        }
        Ok(merged_result)
    }

    // 合并混合策略结果 先合并专家结果，再合并层结果
    fn merge_hybrid_results(&self, results: &[Vec<u8>], gate_weights: Option<GateWeights>) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有混合策略结果可合并".to_string()));
        }
        let num_layers = self.model_info.num_layers;
        let num_experts = self.model_info.num_experts;
        if results.len() != num_layers * num_experts {
            return Err(Error::InferenceError(format!("混合策略结果数量 {} 与期望数量 {} 不匹配", results.len(), num_layers * num_experts)));
        }
        let mut layer_results = Vec::new();
        for layer_id in 0..num_layers {
            let layer_start = layer_id * num_experts;
            let layer_end = layer_start + num_experts;
            let layer_expert_results = &results[layer_start..layer_end];
            let layer_result = self.merge_expert_results(layer_expert_results, gate_weights.clone())?;
            layer_results.push(layer_result);
        }
        self.merge_layer_results(&layer_results)
    }

    // 移除填充
    fn remove_padding(&self, result: &[u8]) -> Result<Vec<u8>> {
        Ok(result.to_vec())
    }
} 