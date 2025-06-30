// error.rs
// 定义项目通用的错误类型（如IO、CUDA、模型加载、推理等）和Result类型。
use std::fmt;
use std::io;

/// 项目通用错误类型，涵盖IO、CUDA、模型加载、推理、GPU等错误
#[derive(Debug)]
pub enum Error {
    /// IO错误
    Io(io::Error),
    /// CUDA相关错误
    CudaError(rustacuda::error::CudaError),
    /// 模型加载错误
    ModelLoadError(String),
    /// 推理阶段错误
    InferenceError(String),
    /// GPU资源相关错误
    GpuError(String),
    /// 其他类型错误
    Other(String),
}

/// 通用结果类型
pub type Result<T> = std::result::Result<T, Error>;

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<rustacuda::error::CudaError> for Error {
    fn from(e: rustacuda::error::CudaError) -> Self {
        Error::CudaError(e)
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Error::Other(format!("NulError: {}", e))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO错误: {}", e),
            Error::CudaError(e) => write!(f, "CUDA错误: {:?}", e),
            Error::ModelLoadError(msg) => write!(f, "模型加载错误: {}", msg),
            Error::InferenceError(msg) => write!(f, "推理错误: {}", msg),
            Error::GpuError(msg) => write!(f, "GPU错误: {}", msg),
            Error::Other(msg) => write!(f, "其他错误: {}", msg),
        }
    }
}

impl std::error::Error for Error {}