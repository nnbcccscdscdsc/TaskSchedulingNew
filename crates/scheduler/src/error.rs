use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    CudaError(rustacuda::error::CudaError),
    ModelLoadError(String),
    InferenceError(String),
    GpuError(String),
    Other(String),
}

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