use scheduler::{
    model_downloader::ModelDownloader,
    task_splitter::{TaskSplitter, SplitStrategy},
    task::{MoeTask, TaskPriority},
    task_executor::TaskExecutor,
    types::GateWeights,
    error::Result,
};
use uuid::Uuid;
use prettytable::{Table, row, cell};

/// 综合测试示例，验证所有修复的功能
fn main() -> Result<()> {
    println!("=== 综合测试：验证任务拆分与执行系统 ===");
    
    // 1. 创建模型下载器并获取模型信息
    let downloader = ModelDownloader::new("downloads".to_string());
    let model_dir = "downloads/google/switch-base-8";
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
    
    println!("模型信息: {:?}", model_info);

    // 2. 测试不同的拆分策略
    let strategies = vec![
        SplitStrategy::ByExpert,
        SplitStrategy::ByLayer,
        SplitStrategy::ByBatch { batch_size: 1024 },
        SplitStrategy::Hybrid { 
            expert_split: true, 
            layer_split: false, 
            batch_size: 512,
            expert_ratio: 0.5,
            layer_ratio: 1.0,
        },
        SplitStrategy::Hybrid { 
            expert_split: true, 
            layer_split: true, 
            batch_size: 256,
            expert_ratio: 0.25,
            layer_ratio: 0.5,
        },
    ];

    for (i, strategy) in strategies.iter().enumerate() {
        println!("\n--- 测试策略 {}: {} ---", i + 1, strategy.description());
        
        // 验证策略参数
        if let Err(e) = strategy.validate(&model_info) {
            println!("策略验证失败: {}", e);
            continue;
        }

        // 创建任务拆分器
        let splitter = match TaskSplitter::new(model_info.clone(), strategy.clone()) {
            Ok(s) => s,
            Err(e) => {
                println!("创建拆分器失败: {}", e);
                continue;
            }
        };

        // 准备输入数据
        let input_data = prepare_test_input(&model_info);
        let parent_task_id = format!("test_task_{}", Uuid::new_v4());

        // 执行任务拆分
        match splitter.split_task(&input_data, &parent_task_id, TaskPriority::Normal) {
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

                // 打印任务表格
                print_tasks_table(&tasks);

                // 测试任务执行（如果任务数量合理）
                if tasks.len() <= 10 {
                    test_task_execution(&tasks)?;
                } else {
                    println!("任务数量过多 ({}), 跳过执行测试", tasks.len());
                }
            }
            Err(e) => {
                println!("任务拆分失败: {}", e);
            }
        }
    }

    // 3. 测试边缘情况
    println!("\n--- 测试边缘情况 ---");
    test_edge_cases(&model_info)?;

    println!("\n=== 综合测试完成 ===");
    Ok(())
}

/// 准备测试输入数据
fn prepare_test_input(model_info: &scheduler::config::ModelInfo) -> Vec<u8> {
    let input_size = model_info.hidden_size;
    let mut input_data = Vec::new();
    
    // 添加输入大小信息
    input_data.extend_from_slice(&(input_size as u32).to_le_bytes());
    
    // 添加模拟的输入数据
    for i in 0..input_size {
        let value = (i % 100) as f32 / 100.0;
        input_data.extend_from_slice(&value.to_le_bytes());
    }
    
    input_data
}

/// 测试任务执行
fn test_task_execution(tasks: &[MoeTask]) -> Result<()> {
    println!("开始测试任务执行...");
    
    // 创建任务执行器
    let executor = TaskExecutor::new(0)?;
    
    // 复制任务以便修改
    let mut tasks_copy: Vec<MoeTask> = tasks.iter().cloned().collect();
    
    // 执行前几个任务作为示例
    let num_to_execute = std::cmp::min(3, tasks_copy.len());
    let tasks_to_execute = &mut tasks_copy[..num_to_execute];
    
    match executor.execute_tasks(tasks_to_execute) {
        Ok(results) => {
            println!("成功执行 {} 个任务", results.len());
            
            // 验证执行结果
            for (i, (task, result)) in tasks_to_execute.iter().zip(results.iter()).enumerate() {
                println!("任务 {}: ID={}, 状态={:?}, 结果大小={}", 
                    i + 1, task.task_id, task.status, result.len());
            }
            
            // 获取执行器状态
            if let Ok((allocated, max)) = executor.get_memory_status() {
                println!("内存使用: {}/{} 字节 ({:.1}%)", 
                    allocated, max, (allocated as f32 / max as f32) * 100.0);
            }
            
            if let Ok(loads) = executor.get_load_status() {
                println!("GPU负载: {:?}", loads);
            }
        }
        Err(e) => {
            println!("任务执行失败: {}", e);
        }
    }
    
    Ok(())
}

/// 测试边缘情况
fn test_edge_cases(model_info: &scheduler::config::ModelInfo) -> Result<()> {
    println!("测试边缘情况...");
    
    // 1. 测试无效的拆分策略
    let invalid_strategies = vec![
        SplitStrategy::ByBatch { batch_size: 0 }, // 批次大小为0
        SplitStrategy::Hybrid { 
            expert_split: false, 
            layer_split: false, 
            batch_size: 100,
            expert_ratio: 0.0,
            layer_ratio: 0.0,
        }, // 没有启用任何拆分
        SplitStrategy::Hybrid { 
            expert_split: true, 
            layer_split: true, 
            batch_size: 100,
            expert_ratio: 1.5, // 比例超过1.0
            layer_ratio: 0.5,
        },
    ];
    
    for (i, strategy) in invalid_strategies.iter().enumerate() {
        println!("测试无效策略 {}: {}", i + 1, strategy.description());
        match strategy.validate(model_info) {
            Ok(_) => println!("  意外通过验证"),
            Err(e) => println!("  正确拒绝: {}", e),
        }
    }
    
    // 2. 测试空输入数据
    println!("测试空输入数据...");
    let splitter = TaskSplitter::new(model_info.clone(), SplitStrategy::ByExpert)?;
    match splitter.split_task(&[], "test", TaskPriority::Normal) {
        Ok(_) => println!("  意外接受空数据"),
        Err(e) => println!("  正确拒绝空数据: {}", e),
    }
    
    // 3. 测试过小的输入数据
    println!("测试过小的输入数据...");
    let small_input = vec![1u8, 2, 3, 4]; // 远小于要求的最小大小
    match splitter.split_task(&small_input, "test", TaskPriority::Normal) {
        Ok(_) => println!("  意外接受过小数据"),
        Err(e) => println!("  正确拒绝过小数据: {}", e),
    }
    
    Ok(())
}

/// 表格化打印任务列表
fn print_tasks_table(tasks: &[MoeTask]) {
    let mut table = Table::new();
    table.add_row(row!["序号", "任务ID", "父任务ID", "优先级", "状态", "流ID", "输入大小"]);
    
    for (i, task) in tasks.iter().enumerate() {
        table.add_row(row![
            i + 1,
            &task.task_id,
            task.parent_task_id.as_deref().unwrap_or("-"),
            format!("{:?}", task.priority),
            format!("{:?}", task.status),
            task.stream_id.map(|id| id.to_string()).unwrap_or("-".to_string()),
            task.input_data.len()
        ]);
    }
    
    table.printstd();
} 