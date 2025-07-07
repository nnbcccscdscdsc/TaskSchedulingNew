use scheduler::model_downloader::{ModelDownloader, ModelInfo};
use scheduler::task_splitter::{SplitStrategy, TaskSplitter};
use scheduler::task::TaskPriority;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. 加载模型信息 ----
    let model_dir = "downloads/google/switch-base-8";
    println!("从 {} 加载模型信息...", model_dir);

    // 使用 ModelDownloader 从本地路径动态获取模型信息
    let downloader = ModelDownloader::new("downloads".to_string());
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
    
    println!("模型信息加载成功: {:#?}", model_info);


    // ---- 2. 实例化任务拆分器 ----
    // 我们选择"按专家拆分"策略进行测试
    let strategy = SplitStrategy::ByExpert;
    let task_splitter = match TaskSplitter::new(model_info.clone(), strategy) {
        Ok(splitter) => splitter,
        Err(e) => {
            eprintln!("创建任务拆分器失败: {}", e);
            return Err(e.into());
        }
    };
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

            // 验证拆分结果
            if let Ok(valid) = task_splitter.verify_split_results(&sub_tasks, &mock_input_data) {
                println!("拆分结果验证: {}", if valid { "通过" } else { "失败" });
            }

            for (i, task) in sub_tasks.iter().enumerate() {
                println!("\n--- 子任务 #{} ---", i + 1);
                println!("  任务 ID: {}", task.task_id);
                println!("  父任务 ID: {:?}", task.parent_task_id);
                println!("  输入数据大小: {} 字节", task.input_data.len());
                println!("  分配的流 ID: {:?}", task.stream_id);
                println!("  状态: {:?}", task.status);
            }
        }
        Err(e) => {
            eprintln!("\n❌ 任务拆分失败: {}", e);
        }
    }

    Ok(())
} 