use scheduler::model_downloader::ModelDownloader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 定义缓存目录
    let cache_dir = "downloads".to_string();
    let mut downloader = ModelDownloader::new(cache_dir);

    // 如果在国内环境，可以开启镜像加速
    // downloader.use_mirror(true);

    // 定义要下载的模型
    let model_name = "google/switch-base-8";

    println!("开始下载模型: {}...", model_name);

    // 执行下载
    match downloader.download_switch_transformer(model_name) {
        Ok(model_path) => {
            println!("\n模型下载成功！");
            println!("模型保存在: {}", model_path);
            println!("你可以查看 {} 目录下的文件。", model_path);
        }
        Err(e) => {
            eprintln!("\n模型下载失败: {}", e);
            eprintln!("请检查你的网络连接或Python环境配置。");
            eprintln!("你需要安装 Python3 和 `transformers`、`torch` 库。");
            eprintln!("可以尝试运行: pip install transformers torch sentencepiece");
        }
    }

    Ok(())
} 