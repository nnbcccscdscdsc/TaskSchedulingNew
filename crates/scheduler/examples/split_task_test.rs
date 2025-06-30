use scheduler::model_downloader::{ModelDownloader, ModelInfo};
use scheduler::task_splitter::{SplitStrategy, TaskSplitter};
use scheduler::task::TaskPriority;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. 加载模型信息 ----
    let model_dir = "downloads/google/switch-base-8";
    println!("从 {} 加载模型信息...", model_dir);

    // 假设 model_downloader 能够从本地路径获取信息
    // 注意：这需要 ModelDownloader 有一个相应的功能，我们暂时手动构建 ModelInfo
    let model_info = ModelInfo {
        model_type: "switch_transformer".to_string(),
        num_experts: 8,
        hidden_size: 512,
        intermediate_size: 2048,
        num_layers: 12,
    };
    println!("模型信息加载成功: {:#?}", model_info);


    // ---- 2. 实例化任务拆分器 ----
    // 我们选择"按专家拆分"策略进行测试
    let strategy = SplitStrategy::ByExpert;
    let task_splitter = TaskSplitter::new(model_info.clone(), strategy);
    println!("\n任务拆分器创建成功，使用策略: {:?}", task_splitter.strategy);


    // ---- 3. 创建模拟输入数据 ----
    // 假设输入数据是一个包含 512 个 f32 浮点数的向量
    let input_size = model_info.hidden_size;
    let mock_input_data: Vec<u8> = vec![0; input_size * 4]; // 4 bytes per f32
    println!("创建模拟输入数据，大小: {} 字节", mock_input_data.len());


    // ---- 4. 执行任务拆分 ----
    println!("\n开始执行任务拆分...");
    let parent_task_id = "main_task_001";
    let priority = TaskPriority::Normal;

    match task_splitter.split_task(&mock_input_data, parent_task_id, priority) {
        Ok(sub_tasks) => {
            // ---- 5. 打印拆分结果 ----
            println!("\n🎉 任务拆分成功！🎉");
            println!("总共拆分出 {} 个子任务。", sub_tasks.len());

            for (i, task) in sub_tasks.iter().enumerate() {
                println!("\n--- 子任务 #{} ---", i + 1);
                println!("  任务 ID: {}", task.task_id);
                println!("  父任务 ID: {:?}", task.parent_task_id);
                println!("  输入数据大小: {} 字节", task.input_data.len());
                println!("  分配的 GPU ID: {:?}", task.gpu_id);
                println!("  状态: {:?}", task.status);
            }
        }
        Err(e) => {
            eprintln!("\n❌ 任务拆分失败: {}", e);
        }
    }

    Ok(())
} 