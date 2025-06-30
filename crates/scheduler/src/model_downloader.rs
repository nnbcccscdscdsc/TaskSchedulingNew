// model_downloader.rs
// 模型下载器，支持从Hugging Face等平台下载Switch Transformer模型及其配置信息。
use crate::error::{Error, Result};
use std::path::Path;
use std::fs;
use std::process::Command;

/// 模型下载器，支持从Hugging Face下载Switch Transformer模型
pub struct ModelDownloader {
    /// 缓存目录
    cache_dir: String,
    /// 是否使用镜像源
    use_mirror: bool,
}

impl ModelDownloader {
    /// 创建新的模型下载器
    pub fn new(cache_dir: String) -> Self {
        Self {
            cache_dir,
            use_mirror: false,
        }
    }

    /// 设置是否使用镜像源
    pub fn use_mirror(&mut self, use_mirror: bool) {
        self.use_mirror = use_mirror;
    }

    /// 下载Switch Transformer模型
    pub fn download_switch_transformer(&self, model_name: &str) -> Result<String> {
        println!("开始下载Switch Transformer模型: {}", model_name);
        
        // 创建缓存目录
        let model_dir = format!("{}/{}", self.cache_dir, model_name);
        fs::create_dir_all(&model_dir)?;
        
        // 使用Python脚本下载模型
        let python_script = self.generate_download_script(model_name, &model_dir)?;
        let script_path = format!("{}/download_model.py", model_dir);
        fs::write(&script_path, python_script)?;
        
        // 执行下载脚本
        let output = Command::new("python3")
            .arg(&script_path)
            .output()
            .map_err(|e| Error::Other(format!("执行Python脚本失败: {}", e)))?;
        
        if !output.status.success() {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(Error::ModelLoadError(format!("模型下载失败: {}", error_msg)));
        }
        
        println!("Switch Transformer模型下载完成: {}", model_dir);
        Ok(model_dir)
    }

    /// 生成Python下载脚本
    fn generate_download_script(&self, model_name: &str, model_dir: &str) -> Result<String> {
        let mirror_url = if self.use_mirror {
            "https://hf-mirror.com"
        } else {
            "https://huggingface.co"
        };
        
        let script = format!(
            r#"
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import torch

def download_model(model_name, save_dir):
    print(f"正在下载模型: {{model_name}}")
    print(f"保存目录: {{save_dir}}")
    
    try:
        # 设置镜像源（如果需要）
        os.environ['HF_ENDPOINT'] = '{}'
        
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # 下载模型配置
        print("下载模型配置...")
        config = AutoConfig.from_pretrained(model_name, cache_dir=save_dir)
        config.save_pretrained(save_dir)
        
        # 下载模型权重
        print("下载模型权重...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            cache_dir=save_dir,
            torch_dtype=torch.float16,  # 使用半精度节省内存
            device_map="auto"  # 自动设备映射
        )
        model.save_pretrained(save_dir)
        
        print("模型下载完成！")
        
        # 打印模型信息
        print(f"模型类型: {{type(model).__name__}}")
        print(f"参数数量: {{sum(p.numel() for p in model.parameters()):,}}")
        print(f"专家数量: {{getattr(config, 'num_experts', 'N/A')}}")
        
    except Exception as e:
        print(f"下载失败: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    model_name = "{}"
    save_dir = "{}"
    download_model(model_name, save_dir)
"#,
            mirror_url, model_name, model_dir
        );
        
        Ok(script)
    }

    /// 验证下载的模型
    pub fn verify_model(&self, model_dir: &str) -> Result<bool> {
        let required_files = [
            "config.json",
            "tokenizer.json", 
            "pytorch_model.bin",
            "special_tokens_map.json"
        ];
        
        for file in &required_files {
            let file_path = format!("{}/{}", model_dir, file);
            if !Path::new(&file_path).exists() {
                return Err(Error::ModelLoadError(format!("缺少必要文件: {}", file)));
            }
        }
        
        Ok(true)
    }

    /// 获取模型配置信息
    pub fn get_model_info(&self, model_dir: &str) -> Result<ModelInfo> {
        let config_path = format!("{}/config.json", model_dir);
        let config_content = fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_content)?;
        
        Ok(ModelInfo {
            model_type: config["model_type"].as_str().unwrap_or("unknown").to_string(),
            num_experts: config["num_experts"].as_u64().unwrap_or(0) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(0) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(0) as usize,
            num_layers: config["num_layers"].as_u64().unwrap_or(0) as usize,
        })
    }
}

/// 模型信息结构体
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: String,
    pub num_experts: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
}

/// 常用的Switch Transformer模型列表
pub const SWITCH_TRANSFORMER_MODELS: &[&str] = &[
    "google/switch-base-8",           // 8个专家，基础版本
    "google/switch-base-16",          // 16个专家，基础版本
    "google/switch-base-32",          // 32个专家，基础版本
    "google/switch-base-64",          // 64个专家，基础版本
    "google/switch-base-128",         // 128个专家，基础版本
    "google/switch-large-8",          // 8个专家，大版本
    "google/switch-large-16",         // 16个专家，大版本
    "google/switch-large-32",         // 32个专家，大版本
    "google/switch-large-64",         // 64个专家，大版本
    "google/switch-large-128",        // 128个专家，大版本
    "google/switch-xxl-8",            // 8个专家，超大版本
    "google/switch-xxl-16",           // 16个专家，超大版本
    "google/switch-xxl-32",           // 32个专家，超大版本
    "google/switch-xxl-64",           // 64个专家，超大版本
    "google/switch-xxl-128",          // 128个专家，超大版本
]; 