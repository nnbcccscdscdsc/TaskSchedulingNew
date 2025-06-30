// task_executor.rs
// 任务执行器，负责实际执行单个MoE子任务，例如调用CUDA核函数进行专家计算。
use crate::error::{Error, Result};
use crate::task::MoeTask;
use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, CopyDestination};
use std::error::Error as StdError;

/// 任务执行器，管理CUDA上下文和设备
pub struct TaskExecutor {
    // 这个 context 必须存在，以确保 CUDA API 的调用在此上下文中执行。
    // 我们用 _ 开头是因为我们不会直接使用它，但需要它来管理生命周期。
    _context: Context,
}

impl TaskExecutor {
    /// 创建一个新的 TaskExecutor
    ///
    /// 这会初始化 Rustacuda 并设置当前的 CUDA 上下文。
    pub fn new(device_id: usize) -> Result<Self> {
        // 初始化CUDA驱动API
        rustacuda::init(CudaFlags::empty())
            .map_err(Error::CudaError)?;

        // 获取指定ID的设备
        let device = Device::get_device(device_id as u32)
            .map_err(Error::CudaError)?;

        // 为该设备创建上下文
        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
            .map_err(Error::CudaError)?;

        Ok(Self { _context: context })
    }

    /// 执行一个任务，将数据拷贝到GPU再拷贝回来
    ///
    /// 这是真实计算的第一步，用于验证数据通路。
    pub fn execute_task(&self, task: &MoeTask) -> Result<Vec<u8>> {
        println!("  [Executor] 开始执行任务: {}", task.task_id);

        // 1. 将输入数据的切片从CPU内存拷贝到GPU设备内存
        let mut device_buffer = DeviceBuffer::from_slice(&task.input_data)
            .map_err(|e| Error::CudaError(e))?;
        println!("  [Executor] 已将 {} 字节数据拷贝到 GPU。", task.input_data.len());
        
        // --- 此处未来将插入真实的CUDA核函数调用 ---
        
        // 2. 将结果从GPU设备内存拷贝回CPU内存
        let mut host_result = vec![0u8; task.input_data.len()];
        device_buffer.copy_to(&mut host_result)
            .map_err(|e| Error::CudaError(e))?;
        println!("  [Executor] 已将 {} 字节结果传回 CPU。", host_result.len());

        Ok(host_result)
    }
} 