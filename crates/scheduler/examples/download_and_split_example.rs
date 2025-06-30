use scheduler::{
    model_downloader::{ModelDownloader, SWITCH_TRANSFORMER_MODELS},
    task_splitter::{TaskSplitter, SplitStrategy},
    config::SchedulerConfig,
    scheduler::TaskScheduler,
    task_executor::TaskExecutor,
    task::{MoeTask, TaskPriority},
    error::Result,
};
use uuid::Uuid;
use std::fs;
use tempfile::tempdir;

/// 下载Switch Transformer模型并进行任务拆分的完整示例
fn main() -> Result<()> {
    println!("=== Switch Transformer模型下载与任务拆分示例 ===");
    
    // 1. 创建任务执行器
    println!("初始化任务执行器...");
    let executor = TaskExecutor::new(0)?;
    println!("任务执行器初始化成功！");
    
    // 1. 创建模型下载器，使用正确的模型缓存目录
    let mut downloader = ModelDownloader::new("downloads".to_string());
    
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
    let splitter = TaskSplitter::new(model_info.clone(), strategy);
    
    // 7. 准备输入数据
    let input_data = prepare_sample_input(&model_info);
    println!("准备输入数据，大小: {} 字节", input_data.len());
    
    // 8. 拆分任务
    let parent_task_id = format!("moe_task_{}", Uuid::new_v4());
    let sub_tasks = splitter.split_task(&input_data, &parent_task_id, TaskPriority::Normal)?;
    println!("任务拆分完成，共生成 {} 个子任务", sub_tasks.len());
    
    // 10. 创建调度器
    let config = SchedulerConfig::default();
    let scheduler = TaskScheduler::new(config);
    
    // 11. 提交所有子任务
    for task in sub_tasks {
        scheduler.submit_task(task);
    }
    println!("所有子任务已提交到调度器");
    
    // 12. 执行真实的任务并收集结果
    run_real_execution(&scheduler, &splitter, &executor)?;
    
    println!("=== 示例执行完成 ===");
    Ok(())
}

/// 准备示例输入数据
fn prepare_sample_input(model_info: &scheduler::config::ModelInfo) -> Vec<u8> {
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

/// 运行真实的执行过程，并收集结果
fn run_real_execution(
    scheduler: &TaskScheduler, 
    splitter: &TaskSplitter,
    executor: &TaskExecutor,
) -> Result<()> {
    println!("\n开始执行真实任务...");
    
    let mut completed_tasks = Vec::new();
    let mut results = Vec::new();
    
    // 模拟获取和执行任务
    while let Some(mut task) = scheduler.fetch_next_task() {
        // 使用 executor 执行真实的任务
        match executor.execute_task(&task) {
            Ok(result_data) => {
                task.status = scheduler::task::TaskStatus::Completed;
                task.result = Some(result_data.clone());
                
                println!("任务完成: {}", task.task_id);
                completed_tasks.push(task);
                results.push(result_data);
            }
            Err(e) => {
                task.status = scheduler::task::TaskStatus::Failed(e.to_string());
                eprintln!("任务 {} 执行失败: {}", task.task_id, e);
                // 可以在这里实现任务失败后的重试或错误处理逻辑
            }
        }
    }
    
    // 合并结果
    let final_result = splitter.result_merger.merge_results(&results, None, &splitter.strategy)?;
    println!("结果合并完成，最终结果大小: {} 字节", final_result.len());
    
    // 验证结果
    validate_final_result(&final_result)?;
    
    Ok(())
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
    fn test_task_splitter_creation() -> Result<()> {
        // 创建一个临时目录来模拟模型文件夹
        let dir = tempdir()?;
        let model_dir = dir.path();

        // 在临时目录中创建一个模拟的 config.json
        let config_content = r#"
        {
          "model_type": "switch_transformers",
          "num_experts": 8,
          "d_model": 512,
          "d_ff": 2048,
          "num_layers": 12
        }
        "#;
        fs::write(model_dir.join("config.json"), config_content)?;

        // 使用 downloader 从这个模拟的 config.json 加载信息
        let downloader = ModelDownloader::new(model_dir.to_str().unwrap().to_string());
        let model_info = downloader.get_model_info(model_dir.to_str().unwrap())?;
        
        let strategy = SplitStrategy::ByExpert;
        let splitter = TaskSplitter::new(model_info, strategy);
        
        assert_eq!(splitter.expert_gpu_mapping.len(), 0);
        assert_eq!(splitter.strategy, strategy);
        Ok(())
    }
} 