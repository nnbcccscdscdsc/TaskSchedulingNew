use scheduler::{
    model_downloader::{ModelDownloader, SWITCH_TRANSFORMER_MODELS},
    task_splitter::{TaskSplitter, SplitStrategy},
    config::SchedulerConfig,
    scheduler::TaskScheduler,
    task::MoeTask,
    error::Result,
};
use std::collections::HashMap;
use uuid::Uuid;

/// 下载Switch Transformer模型并进行任务拆分的完整示例
fn main() -> Result<()> {
    println!("=== Switch Transformer模型下载与任务拆分示例 ===");
    
    // 1. 创建模型下载器
    let mut downloader = ModelDownloader::new("./models".to_string());
    
    // 使用镜像源加速下载（可选）
    downloader.use_mirror(true);
    
    // 2. 选择要下载的模型
    let model_name = "google/switch-base-8"; // 8个专家的基础版本，适合测试
    println!("选择模型: {}", model_name);
    
    // 3. 下载模型
    let model_dir = downloader.download_switch_transformer(model_name)?;
    println!("模型下载完成，保存在: {}", model_dir);
    
    // 4. 验证模型
    downloader.verify_model(&model_dir)?;
    println!("模型验证通过");
    
    // 5. 获取模型信息
    let model_info = downloader.get_model_info(&model_dir)?;
    println!("模型信息:");
    println!("  类型: {}", model_info.model_type);
    println!("  专家数量: {}", model_info.num_experts);
    println!("  隐藏层大小: {}", model_info.hidden_size);
    println!("  中间层大小: {}", model_info.intermediate_size);
    println!("  层数: {}", model_info.num_layers);
    
    // 6. 创建任务拆分器
    let strategy = SplitStrategy::ByExpert; // 按专家拆分
    let mut splitter = TaskSplitter::new(model_info.clone(), strategy);
    
    // 7. 设置专家到GPU的映射（如果有多个GPU）
    let mut expert_gpu_mapping = HashMap::new();
    for expert_id in 0..model_info.num_experts {
        expert_gpu_mapping.insert(expert_id, (expert_id % 2) as i32); // 分配到2个GPU
    }
    splitter.set_expert_gpu_mapping(expert_gpu_mapping);
    
    // 8. 准备输入数据
    let input_data = prepare_sample_input(&model_info);
    println!("准备输入数据，大小: {} 字节", input_data.len());
    
    // 9. 拆分任务
    let parent_task_id = format!("moe_task_{}", Uuid::new_v4());
    let sub_tasks = splitter.split_task(&input_data, &parent_task_id)?;
    println!("任务拆分完成，共生成 {} 个子任务", sub_tasks.len());
    
    // 10. 创建调度器
    let config = SchedulerConfig::default();
    let scheduler = TaskScheduler::new(config);
    
    // 11. 提交所有子任务
    for task in sub_tasks {
        scheduler.submit_task(task);
    }
    println!("所有子任务已提交到调度器");
    
    // 12. 模拟任务执行和结果收集
    simulate_task_execution(&scheduler, &splitter)?;
    
    println!("=== 示例执行完成 ===");
    Ok(())
}

/// 准备示例输入数据
fn prepare_sample_input(model_info: &scheduler::model_downloader::ModelInfo) -> Vec<u8> {
    // 创建一个简单的输入张量（序列化为字节）
    let input_size = model_info.hidden_size;
    let mut input_data = Vec::new();
    
    // 添加输入维度信息（4字节）
    input_data.extend_from_slice(&(input_size as u32).to_le_bytes());
    
    // 添加示例数据（浮点数）
    for i in 0..input_size {
        let value = (i % 100) as f32 / 100.0;
        input_data.extend_from_slice(&value.to_le_bytes());
    }
    
    input_data
}

/// 模拟任务执行过程
fn simulate_task_execution(
    scheduler: &TaskScheduler, 
    splitter: &TaskSplitter
) -> Result<()> {
    println!("开始模拟任务执行...");
    
    let mut completed_tasks = Vec::new();
    let mut results = Vec::new();
    
    // 模拟获取和执行任务
    while let Some(mut task) = scheduler.fetch_next_task() {
        println!("执行任务: {}", task.task_id);
        
        // 更新任务状态
        task.status = scheduler::task::TaskStatus::Running;
        
        // 模拟计算过程
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        // 生成模拟结果
        let result_data = generate_mock_result(&task);
        task.result = Some(result_data.clone());
        task.status = scheduler::task::TaskStatus::Completed;
        
        completed_tasks.push(task);
        results.push(result_data);
        
        println!("任务完成: {}", task.task_id);
    }
    
    // 合并结果
    let final_result = splitter.merge_results(&results)?;
    println!("结果合并完成，最终结果大小: {} 字节", final_result.len());
    
    // 验证结果
    validate_final_result(&final_result)?;
    
    Ok(())
}

/// 生成模拟结果
fn generate_mock_result(task: &MoeTask) -> Vec<u8> {
    // 从任务ID中提取专家ID
    let expert_id = if task.task_id.contains("expert_") {
        let parts: Vec<&str> = task.task_id.split("expert_").collect();
        if parts.len() > 1 {
            parts[1].parse::<u32>().unwrap_or(0)
        } else {
            0
        }
    } else {
        0
    };
    
    // 生成模拟输出
    let output_size = 1024; // 假设输出大小
    let mut result = Vec::new();
    
    // 添加专家ID
    result.extend_from_slice(&expert_id.to_le_bytes());
    
    // 添加模拟输出数据
    for i in 0..output_size {
        let value = ((i + expert_id as usize) % 100) as f32 / 100.0;
        result.extend_from_slice(&value.to_le_bytes());
    }
    
    result
}

/// 验证最终结果
fn validate_final_result(result: &[u8]) -> Result<()> {
    if result.len() < 4 {
        return Err(scheduler::error::Error::Other("结果数据太小".to_string()));
    }
    
    println!("结果验证通过");
    println!("前10个输出值:");
    for i in 0..10.min(result.len() / 4) {
        let start = i * 4;
        let end = start + 4;
        if end <= result.len() {
            if let Ok(bytes) = result[start..end].try_into() {
                let value = f32::from_le_bytes(bytes);
                println!("  [{}]: {:.6}", i, value);
            }
        }
    }
    
    Ok(())
}

/// 列出所有可用的Switch Transformer模型
fn list_available_models() {
    println!("可用的Switch Transformer模型:");
    for (i, model) in SWITCH_TRANSFORMER_MODELS.iter().enumerate() {
        println!("  {}. {}", i + 1, model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_downloader_creation() {
        let downloader = ModelDownloader::new("./test_models".to_string());
        assert_eq!(downloader.cache_dir, "./test_models");
    }
    
    #[test]
    fn test_task_splitter_creation() {
        let model_info = scheduler::model_downloader::ModelInfo {
            model_type: "switch_transformer".to_string(),
            num_experts: 8,
            hidden_size: 512,
            intermediate_size: 2048,
            num_layers: 12,
        };
        
        let strategy = SplitStrategy::ByExpert;
        let splitter = TaskSplitter::new(model_info, strategy);
        
        assert_eq!(splitter.expert_gpu_mapping.len(), 0);
    }
} 