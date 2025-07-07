// task_executor.rs
// 任务执行器，负责实际执行单个MoE子任务，例如调用CUDA核函数进行专家计算。
use crate::error::{Error, Result};
use crate::task::{MoeTask, TaskStatus};
use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, CopyDestination};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 内存池管理
#[derive(Debug)]
struct MemoryPool {
    available_buffers: HashMap<usize, Vec<DeviceBuffer<u8>>>,
    total_allocated: usize,
    max_memory: usize,
}

impl MemoryPool {
    fn new(max_memory_mb: usize) -> Self {
        Self {
            available_buffers: HashMap::new(),
            total_allocated: 0,
            max_memory: max_memory_mb * 1024 * 1024, // 转换为字节
        }
    }

    fn get_buffer(&mut self, size: usize) -> Result<DeviceBuffer<u8>> {
        // 检查是否有合适大小的可用缓冲区
        if let Some(buffers) = self.available_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }

        // 检查内存限制
        if self.total_allocated + size > self.max_memory {
            return Err(Error::CudaError(rustacuda::error::CudaError::InvalidValue));
        }

        // 创建新的缓冲区
        let buffer = unsafe { DeviceBuffer::uninitialized(size) }
            .map_err(|e| Error::CudaError(e))?;
        self.total_allocated += size;
        Ok(buffer)
    }

    fn return_buffer(&mut self, buffer: DeviceBuffer<u8>) {
        let size = buffer.len();
        self.available_buffers.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}

/// 负载均衡器
#[derive(Debug)]
struct LoadBalancer {
    gpu_loads: HashMap<usize, f32>, // GPU ID -> 当前负载 (0.0-1.0)
    task_distribution: HashMap<String, usize>, // 任务ID -> GPU ID
}

impl LoadBalancer {
    fn new() -> Self {
        Self {
            gpu_loads: HashMap::new(),
            task_distribution: HashMap::new(),
        }
    }

    fn select_gpu(&mut self, available_gpus: &[usize]) -> Result<usize> {
        if available_gpus.is_empty() {
            return Err(Error::CudaError(rustacuda::error::CudaError::InvalidValue));
        }

        // 找到负载最低的GPU
        let mut best_gpu = available_gpus[0];
        let mut min_load = self.gpu_loads.get(&best_gpu).unwrap_or(&0.0);

        for &gpu_id in available_gpus {
            let load = self.gpu_loads.get(&gpu_id).unwrap_or(&0.0);
            if load < min_load {
                best_gpu = gpu_id;
                min_load = load;
            }
        }

        // 更新负载
        let current_load = self.gpu_loads.get(&best_gpu).unwrap_or(&0.0);
        self.gpu_loads.insert(best_gpu, current_load + 0.1); // 增加负载

        Ok(best_gpu)
    }

    fn release_gpu(&mut self, gpu_id: usize) {
        if let Some(load) = self.gpu_loads.get_mut(&gpu_id) {
            *load = (*load - 0.1).max(0.0);
        }
    }

    fn assign_task(&mut self, task_id: &str, gpu_id: usize) {
        self.task_distribution.insert(task_id.to_string(), gpu_id);
    }
}

/// 任务执行器，管理CUDA上下文和设备
pub struct TaskExecutor {
    // 这个 context 必须存在，以确保 CUDA API 的调用在此上下文中执行。
    // 我们用 _ 开头是因为我们不会直接使用它，但需要它来管理生命周期。
    _context: Context,
    memory_pool: Arc<Mutex<MemoryPool>>,
    load_balancer: Arc<Mutex<LoadBalancer>>,
    device_id: usize,
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

        // 获取设备内存信息
        let total_memory = device.total_memory()
            .map_err(Error::CudaError)?;
        let max_memory_mb = (total_memory / 1024 / 1024 * 80) / 100; // 使用80%的显存

        let memory_pool = Arc::new(Mutex::new(MemoryPool::new(max_memory_mb as usize)));
        let load_balancer = Arc::new(Mutex::new(LoadBalancer::new()));

        Ok(Self { 
            _context: context,
            memory_pool,
            load_balancer,
            device_id,
        })
    }

    /// 执行一个任务，将数据拷贝到GPU再拷贝回来
    ///
    /// 这是真实计算的第一步，用于验证数据通路。
    pub fn execute_task(&self, task: &mut MoeTask) -> Result<Vec<u8>> {
        println!("  [Executor] 开始执行任务: {}", task.task_id);

        // 更新任务状态
        task.status = TaskStatus::Running;

        // 选择GPU进行负载均衡
        let gpu_id = {
            let mut balancer = self.load_balancer.lock()
                .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
            let selected_gpu = balancer.select_gpu(&[self.device_id])?;
            balancer.assign_task(&task.task_id, selected_gpu);
            selected_gpu
        };

        // 从内存池获取缓冲区
        let mut device_buffer = {
            let mut pool = self.memory_pool.lock()
                .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
            pool.get_buffer(task.input_data.len())?
        };

        // 1. 将输入数据的切片从CPU内存拷贝到GPU设备内存
        device_buffer.copy_from(&task.input_data)
            .map_err(|e| Error::CudaError(e))?;
        println!("  [Executor] 已将 {} 字节数据拷贝到 GPU {}。", task.input_data.len(), gpu_id);
        
        // --- 此处未来将插入真实的CUDA核函数调用 ---
        // 模拟计算延迟
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        // 2. 将结果从GPU设备内存拷贝回CPU内存
        let mut host_result = vec![0u8; task.input_data.len()];
        device_buffer.copy_to(&mut host_result)
            .map_err(|e| Error::CudaError(e))?;
        println!("  [Executor] 已将 {} 字节结果传回 CPU。", host_result.len());

        // 将缓冲区归还给内存池
        {
            let mut pool = self.memory_pool.lock()
                .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
            pool.return_buffer(device_buffer);
        }

        // 释放GPU负载
        {
            let mut balancer = self.load_balancer.lock()
                .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
            balancer.release_gpu(gpu_id);
        }

        // 更新任务状态和结果
        task.status = TaskStatus::Completed;
        task.result = Some(host_result.clone());

        Ok(host_result)
    }

    /// 批量执行任务
    pub fn execute_tasks(&self, tasks: &mut [MoeTask]) -> Result<Vec<Vec<u8>>> {
        let mut results = Vec::new();
        
        for task in tasks.iter_mut() {
            match self.execute_task(task) {
                Ok(result) => results.push(result),
                Err(e) => {
                    task.status = TaskStatus::Failed(e.to_string());
                    return Err(e);
                }
            }
        }
        
        Ok(results)
    }

    /// 获取内存池状态
    pub fn get_memory_status(&self) -> Result<(usize, usize)> {
        let pool = self.memory_pool.lock()
            .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
        Ok((pool.total_allocated, pool.max_memory))
    }

    /// 获取负载均衡状态
    pub fn get_load_status(&self) -> Result<HashMap<usize, f32>> {
        let balancer = self.load_balancer.lock()
            .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
        Ok(balancer.gpu_loads.clone())
    }

    /// 清理资源
    pub fn cleanup(&self) -> Result<()> {
        // 清理内存池
        {
            let mut pool = self.memory_pool.lock()
                .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
            pool.available_buffers.clear();
            pool.total_allocated = 0;
        }

        // 清理负载均衡器
        {
            let mut balancer = self.load_balancer.lock()
                .map_err(|_| Error::CudaError(rustacuda::error::CudaError::InvalidValue))?;
            balancer.gpu_loads.clear();
            balancer.task_distribution.clear();
        }

        println!("  [Executor] 资源清理完成");
        Ok(())
    }
}

impl Drop for TaskExecutor {
    fn drop(&mut self) {
        // 自动清理资源
        let _ = self.cleanup();
    }
} 