use scheduler::{
    model_downloader::{ModelDownloader, SWITCH_TRANSFORMER_MODELS},
    task_splitter::{TaskSplitter, SplitStrategy},
    task::{MoeTask, TaskPriority},
    error::Result,
};
use uuid::Uuid;
use prettytable::{Table, row, cell};
use std::collections::HashMap;

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
    let splitter = TaskSplitter::new_from_model_dir(&model_dir, strategy.clone())?;
    let input_data = prepare_sample_input(&model_info);
    println!("准备输入数据，大小: {} 字节", input_data.len());
    let parent_task_id = format!("moe_task_{}", Uuid::new_v4());
    let sub_tasks = splitter.split_task(&input_data, &parent_task_id, TaskPriority::Normal)?;
    println!("任务拆分完成，共生成 {} 个子任务", sub_tasks.len());

    // 3. 打印根任务信息
    println!("\n根任务信息：");
    println!("  任务ID: {}", parent_task_id);
    println!("  输入数据大小: {} 字节", input_data.len());
    println!("  拆分策略: {}", format_split_strategy(&strategy, &model_info));
    println!("  优先级: {:?}", TaskPriority::Normal);

    // 4. 获取依赖关系
    let dependencies = splitter.get_task_dependencies(&sub_tasks)?;

    // 5. 打印子任务表格
    print_tasks_table(&sub_tasks, &dependencies);

    // 6. 流ID说明
    println!("\n流ID说明：当前流ID对应专家ID（0-7），每个专家任务使用独立GPU流以支持并行执行。");
    println!("=== 拆分器验证完成 ===");
    Ok(())
}

/// 拆分策略详细描述
fn format_split_strategy(strategy: &SplitStrategy, model_info: &scheduler::config::ModelInfo) -> String {
    match strategy {
        SplitStrategy::ByExpert => format!("按专家拆分（使用全部{}个专家）", model_info.num_experts),
        SplitStrategy::ByLayer => format!("按层拆分（使用全部{}层）", model_info.num_layers),
        SplitStrategy::ByBatch { batch_size } => format!("按批次拆分（批次大小={}）", batch_size),
        SplitStrategy::Hybrid { expert_split, layer_split, batch_size, expert_ratio, layer_ratio } => {
            let mut desc = String::from("混合拆分：");
            if *expert_split {
                desc += &format!("专家比例={:.2} ", expert_ratio);
            }
            if *layer_split {
                desc += &format!("层比例={:.2} ", layer_ratio);
            }
            desc += &format!("批次大小={}", batch_size);
            desc
        }
    }
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

/// 表格化打印任务列表，增加依赖任务ID和子任务数据大小
fn print_tasks_table(tasks: &[MoeTask], dependencies: &HashMap<String, Vec<String>>) {
    let mut table = Table::new();
    table.add_row(row!["任务ID", "父任务ID", "优先级", "状态", "流ID", "依赖任务ID", "子任务数据大小"]);
    for task in tasks {
        let deps = dependencies.get(&task.task_id)
            .map(|v| if v.is_empty() { "[]".to_string() } else { format!("{:?}", v) })
            .unwrap_or("-".to_string());
        table.add_row(row![
            &task.task_id,
            task.parent_task_id.as_deref().unwrap_or("-"),
            format!("{:?}", task.priority),
            format!("{:?}", task.status),
            task.stream_id.map(|id| id.to_string()).unwrap_or("-".to_string()),
            deps,
            format!("{} 字节", task.input_data.len())
        ]);
    }
    table.printstd();
} 