// result_merger.rs
// 结果合并器，负责合并各子任务（如专家、层、批次等）的推理结果。
use crate::config::ModelInfo;
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

    /// 合并多个子任务的结果
    pub fn merge_results(
        &self, 
        results: &[Vec<u8>], 
        gate_weights: Option<GateWeights>, 
        strategy: &SplitStrategy
    ) -> Result<Vec<u8>> {
        match strategy {
            SplitStrategy::ByExpert => {
                // 如果是按专家拆分，必须有门控权重才能进行有意义的合并
                // 在模拟场景下，如果权重为 None，我们可以采取一种简化的合并策略，例如拼接
                if gate_weights.is_none() {
                    println!("警告：缺少门控权重，将使用简单的拼接策略合并专家结果。");
                    return self.concatenate_results(results);
                }
                self.merge_expert_results(results, gate_weights.unwrap())
            },
            SplitStrategy::ByLayer => self.merge_layer_results(results),
            SplitStrategy::ByBatch { .. } => self.merge_batch_results(results),
            SplitStrategy::Hybrid { expert_split, layer_split, expert_ratio, layer_ratio, .. } => {
                self.merge_hybrid_results(results, gate_weights, *expert_split, *layer_split, *expert_ratio, *layer_ratio)
            }
        }
    }

    /// 将所有结果简单地拼接在一起
    fn concatenate_results(&self, results: &[Vec<u8>]) -> Result<Vec<u8>> {
        Ok(results.concat())
    }

    /// 合并专家结果
    fn merge_expert_results(&self, results: &[Vec<u8>], gate_weights: GateWeights) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有专家结果可合并".to_string()));
        }
        
        if results.len() != gate_weights.weights.len() {
            return Err(Error::InferenceError(format!(
                "专家结果数量 {} 与门控权重数量 {} 不匹配", 
                results.len(), 
                gate_weights.weights.len()
            )));
        }
        
        // 检查所有结果的大小是否一致
        let result_size = results[0].len();
        for (i, result) in results.iter().enumerate() {
            if result.len() != result_size {
                return Err(Error::InferenceError(format!(
                    "专家 {} 的结果大小 {} 与其他专家不一致 {}", 
                    i, result.len(), result_size
                )));
            }
        }
        
        // 按门控权重合并结果
        let mut merged_result = vec![0u8; result_size];
        
        for (i, (result, weight)) in results.iter().zip(gate_weights.weights.iter()).enumerate() {
            if *weight > 0.0 {
                // 将结果按权重累加
                for (merged_chunk, result_chunk) in merged_result.chunks_exact_mut(4).zip(result.chunks_exact(4)) {
                    let current_val = f32::from_le_bytes(merged_chunk.try_into().unwrap());
                    let expert_val = f32::from_le_bytes(result_chunk.try_into().unwrap());
                    let weighted_sum = current_val + expert_val * weight;
                    merged_chunk.copy_from_slice(&weighted_sum.to_le_bytes());
                }
            }
        }
        
        Ok(merged_result)
    }

    fn merge_layer_results(&self, results: &[Vec<u8>]) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有层结果可合并".to_string()));
        }
        let mut merged_result = vec![0u8; results[0].len()];
        let mut is_first = true;

        for result in results {
            if is_first {
                merged_result.copy_from_slice(result);
                is_first = false;
            } else {
                if merged_result.len() != result.len() {
                    return Err(Error::InferenceError("层输出大小与残差大小不匹配".to_string()));
                }
                for (merged_chunk, result_chunk) in merged_result.chunks_exact_mut(4).zip(result.chunks_exact(4)) {
                    let residual_val = f32::from_le_bytes(merged_chunk.try_into().unwrap());
                    let current_val = f32::from_le_bytes(result_chunk.try_into().unwrap());
                    let sum = residual_val + current_val;
                    merged_chunk.copy_from_slice(&sum.to_le_bytes());
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

    // 合并混合策略结果
    fn merge_hybrid_results(
        &self, 
        results: &[Vec<u8>], 
        gate_weights: Option<GateWeights>,
        expert_split: bool,
        layer_split: bool,
        expert_ratio: f32,
        layer_ratio: f32,
    ) -> Result<Vec<u8>> {
        if results.is_empty() {
            return Err(Error::InferenceError("没有混合策略结果可合并".to_string()));
        }

        if expert_split && layer_split {
            // 先按层合并专家结果，再合并层结果
            let num_experts_to_use = (self.model_info.num_experts as f32 * expert_ratio).round() as usize;
            let num_layers_to_use = (self.model_info.num_layers as f32 * layer_ratio).round() as usize;
            
            if results.len() != num_layers_to_use * num_experts_to_use {
                return Err(Error::InferenceError(format!(
                    "混合策略结果数量 {} 与期望数量 {} 不匹配", 
                    results.len(), 
                    num_layers_to_use * num_experts_to_use
                )));
            }

            let mut layer_results = Vec::new();
            for layer_id in 0..num_layers_to_use {
                let layer_start = layer_id * num_experts_to_use;
                let layer_end = layer_start + num_experts_to_use;
                let layer_expert_results = &results[layer_start..layer_end];
                
                // 为每层创建门控权重
                let layer_gate_weights = if let Some(ref weights) = gate_weights {
                    GateWeights {
                        weights: weights.weights.iter().take(num_experts_to_use).cloned().collect(),
                        top_k: std::cmp::min(weights.top_k, num_experts_to_use),
                    }
                } else {
                    // 如果没有门控权重，使用均匀权重
                    GateWeights {
                        weights: vec![1.0 / num_experts_to_use as f32; num_experts_to_use],
                        top_k: num_experts_to_use,
                    }
                };
                
                let layer_result = self.merge_expert_results(layer_expert_results, layer_gate_weights)?;
                layer_results.push(layer_result);
            }
            self.merge_layer_results(&layer_results)
        } else if expert_split {
            // 只按专家拆分
            let num_experts_to_use = (self.model_info.num_experts as f32 * expert_ratio).round() as usize;
            if results.len() != num_experts_to_use {
                return Err(Error::InferenceError(format!(
                    "专家拆分结果数量 {} 与期望数量 {} 不匹配", 
                    results.len(), 
                    num_experts_to_use
                )));
            }
            
            let expert_gate_weights = if let Some(ref weights) = gate_weights {
                GateWeights {
                    weights: weights.weights.iter().take(num_experts_to_use).cloned().collect(),
                    top_k: std::cmp::min(weights.top_k, num_experts_to_use),
                }
            } else {
                GateWeights {
                    weights: vec![1.0 / num_experts_to_use as f32; num_experts_to_use],
                    top_k: num_experts_to_use,
                }
            };
            
            self.merge_expert_results(results, expert_gate_weights)
        } else if layer_split {
            // 只按层拆分
            let num_layers_to_use = (self.model_info.num_layers as f32 * layer_ratio).round() as usize;
            if results.len() != num_layers_to_use {
                return Err(Error::InferenceError(format!(
                    "层拆分结果数量 {} 与期望数量 {} 不匹配", 
                    results.len(), 
                    num_layers_to_use
                )));
            }
            
            self.merge_layer_results(results)
        } else {
            // 只按批次拆分
            self.merge_batch_results(results)
        }
    }

    // 移除填充
    fn remove_padding(&self, result: &[u8]) -> Result<Vec<u8>> {
        Ok(result.to_vec())
    }
} 