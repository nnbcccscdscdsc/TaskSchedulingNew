// lib.rs
// 调度器模块入口，声明并导出各子模块。
pub mod config;
pub mod data_preparator;
pub mod error;
pub mod model_downloader;
pub mod model_def;
pub mod result_merger;
pub mod scheduler;
pub mod task;
pub mod task_executor;
pub mod task_splitter;
pub mod types; 