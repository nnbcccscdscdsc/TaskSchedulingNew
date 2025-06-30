// data_preparator.rs
// 数据准备器，负责为专家、层等准备输入数据，包含数据格式转换和辅助信息生成。
use crate::model_downloader::ModelInfo;
use crate::error::{Error, Result};
use crate::types::*;

pub struct DataPreparator {
    pub model_info: ModelInfo,
}

impl DataPreparator {
    pub fn new(model_info: ModelInfo) -> Self {
        Self { model_info }
    }

    /// 为专家准备数据
    pub fn prepare_expert_data(&self, input_data: &[u8], expert_id: usize) -> Result<Vec<u8>> {
        if expert_id >= self.model_info.num_experts {
            return Err(Error::InferenceError(format!(
                "专家ID {} 超出范围 [0, {})", expert_id, self.model_info.num_experts
            )));
        }
        let mut expert_data = Vec::new();
        expert_data.extend_from_slice(&(expert_id as u32).to_le_bytes());
        let gate_info = self.generate_gate_info(expert_id)?;
        expert_data.extend_from_slice(&gate_info);
        expert_data.extend_from_slice(input_data);
        Ok(expert_data)
    }

    /// 为层准备数据
    pub fn prepare_layer_data(&self, input_data: &[u8], layer_id: usize) -> Result<Vec<u8>> {
        if layer_id >= self.model_info.num_layers {
            return Err(Error::InferenceError(format!(
                "层ID {} 超出范围 [0, {})", layer_id, self.model_info.num_layers
            )));
        }
        let mut layer_data = Vec::new();
        layer_data.extend_from_slice(&(layer_id as u32).to_le_bytes());
        let layer_config = self.generate_layer_config(layer_id)?;
        layer_data.extend_from_slice(&layer_config);
        layer_data.extend_from_slice(input_data);
        Ok(layer_data)
    }

    /// 为层和专家准备数据
    pub fn prepare_layer_expert_data(&self, input_data: &[u8], layer_id: usize, expert_id: usize) -> Result<Vec<u8>> {
        if layer_id >= self.model_info.num_layers {
            return Err(Error::InferenceError(format!(
                "层ID {} 超出范围 [0, {})", layer_id, self.model_info.num_layers
            )));
        }
        if expert_id >= self.model_info.num_experts {
            return Err(Error::InferenceError(format!(
                "专家ID {} 超出范围 [0, {})", expert_id, self.model_info.num_experts
            )));
        }
        let mut layer_expert_data = Vec::new();
        layer_expert_data.extend_from_slice(&(layer_id as u32).to_le_bytes());
        layer_expert_data.extend_from_slice(&(expert_id as u32).to_le_bytes());
        let gate_info = self.generate_gate_info(expert_id)?;
        layer_expert_data.extend_from_slice(&gate_info);
        let layer_config = self.generate_layer_config(layer_id)?;
        layer_expert_data.extend_from_slice(&layer_config);
        layer_expert_data.extend_from_slice(input_data);
        Ok(layer_expert_data)
    }

    /// 生成门控信息
    fn generate_gate_info(&self, expert_id: usize) -> Result<Vec<u8>> {
        let mut gate_info = Vec::new();
        for i in 0..self.model_info.num_experts {
            let weight = if i == expert_id { 1.0 } else { 0.0 };
            gate_info.extend_from_slice(&weight.to_le_bytes());
        }
        Ok(gate_info)
    }

    /// 生成层配置信息
    fn generate_layer_config(&self, layer_id: usize) -> Result<Vec<u8>> {
        let mut layer_config = Vec::new();
        layer_config.extend_from_slice(&(layer_id as u32).to_le_bytes());
        layer_config.extend_from_slice(&(self.model_info.hidden_size as u32).to_le_bytes());
        layer_config.extend_from_slice(&(self.model_info.intermediate_size as u32).to_le_bytes());
        layer_config.extend_from_slice(&(self.model_info.num_experts as u32).to_le_bytes());
        Ok(layer_config)
    }
} 