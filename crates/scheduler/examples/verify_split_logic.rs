//! verify_split_logic.rs
//!
//! 这个示例的目的是验证 TaskSplitter 的核心逻辑是否正确。
//! 它会加载一个真实的 PyTorch Switch Transformer 模型，并执行以下操作：
//! 1. 使用 `tch` 库加载模型和权重。
//! 2. 准备一份输入数据。
//! 3. 调用我们自己的 `TaskSplitter` 来拆分任务。
//! 4. TODO: 获取模型真实的门控权重和路由决策。
//! 5. TODO: 比较我们拆分出的子任务数据和模型内部的路由结果是否一致。

use scheduler::error::Result;
use scheduler::model_downloader::ModelDownloader;
use scheduler::model_def::switch_transformer::SwitchTransformersSparseMLP;
use scheduler::task::TaskPriority;
use scheduler::task_splitter::{SplitStrategy, TaskSplitter};
use tch::{nn, Device, Tensor, Kind};

fn main() -> Result<()> {
    println!("=== 验证任务拆分逻辑 ===");

    // ---- 1. 准备工作 ----
    let device = Device::cuda_if_available();
    println!("使用设备: {:?}", device);

    let model_dir = "downloads/google/switch-base-8";
    let downloader = ModelDownloader::new("downloads".to_string());
    
    // 从 config.json 加载我们自己的模型信息结构
    let model_info = match downloader.get_model_info(model_dir) {
        Ok(info) => info,
        Err(_) => {
            println!("模型不存在，使用模拟模型信息进行测试");
            scheduler::config::ModelInfo {
                model_type: "switch_transformer".to_string(),
                num_experts: 8,
                hidden_size: 512,
                intermediate_size: 2048,
                num_layers: 12,
            }
        }
    };
    println!("自定义模型信息加载成功: {:#?}", model_info);

    // ---- 2. 使用 tch 加载 PyTorch 模型 ----
    println!("\n正在从 {} 加载 PyTorch 模型...", model_dir);
    let mut vs = nn::VarStore::new(device);
    
    // 构建模型权重的完整路径
    // 注意：这需要我们之前修复的 verify_model 逻辑来确定文件名
    let model_weights_path = format!("{}/model.safetensors", model_dir); 
    
    // 加载权重
    match vs.load(&model_weights_path) {
        Ok(_) => println!("PyTorch 模型权重加载成功！"),
        Err(e) => {
            println!("无法加载模型权重: {}，跳过PyTorch模型测试", e);
            // 继续执行其他测试
        }
    }

    // ---- 3. 实例化我们定义的MoE MLP层 ----
    // 我们以 Encoder 的第0个 block 中的第1个 layer norm 后的 mlp 为例
    // 它的路径是 "encoder.block.0.layer.1.mlp"
    let mlp_path = vs.root() / "encoder" / "block" / 0 / "layer" / 1 / "mlp";
    let sparse_mlp = SwitchTransformersSparseMLP::new(mlp_path, &model_info);
    println!("自定义 SparseMLP 实例化成功。");

    // ---- 4. 创建输入张量并获取真实的门控权重 ----
    println!("\n创建输入张量并获取真实的门控 logits...");
    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = model_info.hidden_size as i64;
    let input_tensor = Tensor::randn(&[batch_size, seq_len, hidden_size], (Kind::Float, device));
    
    // 执行前向传播，获取 router logits
    let router_logits = sparse_mlp.forward(&input_tensor);
    println!("成功获取 Router Logits!");
    router_logits.print();

    // ---- 5. 调用 TaskSplitter 并比较 ----
    println!("\n调用我们自己的 TaskSplitter 并进行比较...");
    
    // 创建任务拆分器
    let strategy = SplitStrategy::ByExpert;
    let splitter = match TaskSplitter::new(model_info.clone(), strategy) {
        Ok(s) => s,
        Err(e) => {
            println!("创建任务拆分器失败: {}", e);
            return Ok(());
        }
    };
    
    // 准备输入数据
    let input_data = prepare_sample_input(&model_info);
    let parent_task_id = "verify_task_001";
    
    // 执行任务拆分
    match splitter.split_task(&input_data, parent_task_id, TaskPriority::Normal) {
        Ok(tasks) => {
            println!("成功拆分为 {} 个任务", tasks.len());
            
            // 验证拆分结果
            if let Ok(valid) = splitter.verify_split_results(&tasks, &input_data) {
                println!("拆分结果验证: {}", if valid { "通过" } else { "失败" });
            }
            
            // 获取依赖关系
            if let Ok(dependencies) = splitter.get_task_dependencies(&tasks) {
                println!("依赖关系分析完成，共 {} 个任务", dependencies.len());
            }
        }
        Err(e) => {
            println!("任务拆分失败: {}", e);
        }
    }

    Ok(())
}

/// 准备示例输入数据
fn prepare_sample_input(model_info: &scheduler::config::ModelInfo) -> Vec<u8> {
    let input_size = model_info.hidden_size;
    let mut input_data = Vec::new();
    input_data.extend_from_slice(&(input_size as u32).to_le_bytes());
    for i in 0..input_size {
        let value = (i % 100) as f32 / 100.0;
        input_data.extend_from_slice(&value.to_le_bytes());
    }
    input_data
} 