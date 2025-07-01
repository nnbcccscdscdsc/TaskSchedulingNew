use scheduler::{
    model_downloader::{ModelDownloader, SWITCH_TRANSFORMER_MODELS},
    task_splitter::{TaskSplitter, SplitStrategy},
    task::{MoeTask, TaskPriority},
    error::Result,
};
use uuid::Uuid;
use prettytable::{Table, row, cell};

/// 只验证任务拆分器功能的示例
fn main() -> Result<()> {
    println!("=== Switch Transformer模型下载与任务拆分和表格化输出示例 ===");
    
    // 1. 创建模型下载器
    let mut downloader = ModelDownloader::new("downloads".to_string());
    downloader.use_mirror(true);
    let model_name = "google/switch-base-8";
    println!("选择模型: {}", model_name);
    let model_dir = downloader.download_switch_transformer(model_name)?;
    println!("模型下载完成，保存在: {}", model_dir);
    downloader.verify_model(&model_dir)?;
    println!("模型验证通过");
    let model_info = downloader.get_model_info(&model_dir)?;
    println!("模型信息:");
    println!("  类型: {}", model_info.model_type);
    println!("  专家数量: {}", model_info.num_experts);
    println!("  隐藏层大小: {}", model_info.hidden_size);
    println!("  中间层大小: {}", model_info.intermediate_size);
    println!("  层数: {}", model_info.num_layers);

    // 2. 创建任务拆分器
    let strategy = SplitStrategy::ByExpert;
    let splitter = TaskSplitter::new_from_model_dir(&model_dir, strategy)?;
    let input_data = prepare_sample_input(&model_info);
    println!("准备输入数据，大小: {} 字节", input_data.len());
    let parent_task_id = format!("moe_task_{}", Uuid::new_v4());
    let sub_tasks = splitter.split_task(&input_data, &parent_task_id, TaskPriority::Normal)?;
    println!("任务拆分完成，共生成 {} 个子任务", sub_tasks.len());
    print_tasks_table(&sub_tasks);
    println!("=== 拆分器验证完成 ===");
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

/// 表格化打印任务列表
fn print_tasks_table(tasks: &[MoeTask]) {
    let mut table = Table::new();
    table.add_row(row!["任务ID", "父任务ID", "优先级", "状态", "流ID"]);
    for task in tasks {
        table.add_row(row![
            &task.task_id,
            task.parent_task_id.as_deref().unwrap_or("-"),
            format!("{:?}", task.priority),
            format!("{:?}", task.status),
            task.stream_id.map(|id| id.to_string()).unwrap_or("-".to_string())
        ]);
    }
    table.printstd();
} 