//! CUDA FFT Executor - Runtime integration for GPU-accelerated FFT.
//!
//! This module provides the actual CUDA execution layer for the GPU FFT kernels.
//! It handles:
//! - Device initialization and management
//! - Kernel compilation via NVRTC
//! - Memory allocation and transfers
//! - Kernel execution and synchronization
//!
//! # Requirements
//!
//! - CUDA Toolkit 11.0+ installed
//! - NVIDIA GPU with compute capability 7.0+ (Volta or newer recommended)
//! - `gpu` feature enabled in Cargo.toml

#[cfg(feature = "cuda-runtime")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use super::fft::CIRCLE_FFT_CUDA_KERNEL;
#[allow(unused_imports)]
#[cfg(feature = "cuda-runtime")]
use super::fft::{GPU_FFT_THRESHOLD_LOG_SIZE, M31_PRIME};

// =============================================================================
// Global CUDA Context
// =============================================================================

#[cfg(feature = "cuda-runtime")]
static CUDA_FFT_EXECUTOR: OnceLock<Result<CudaFftExecutor, CudaFftError>> = OnceLock::new();

/// Get the global CUDA FFT executor instance.
/// 
/// This lazily initializes the CUDA context on first call.
#[cfg(feature = "cuda-runtime")]
pub fn get_cuda_executor() -> Result<&'static CudaFftExecutor, &'static CudaFftError> {
    CUDA_FFT_EXECUTOR
        .get_or_init(|| CudaFftExecutor::new())
        .as_ref()
}

/// Check if CUDA is available and initialized.
#[cfg(feature = "cuda-runtime")]
pub fn is_cuda_available() -> bool {
    get_cuda_executor().is_ok()
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn is_cuda_available() -> bool {
    false
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during CUDA FFT execution.
#[derive(Debug, Clone)]
pub enum CudaFftError {
    /// No CUDA device found
    NoDevice,
    /// CUDA driver initialization failed
    DriverInit(String),
    /// Kernel compilation failed
    KernelCompilation(String),
    /// Memory allocation failed
    MemoryAllocation(String),
    /// Memory transfer failed
    MemoryTransfer(String),
    /// Kernel execution failed
    KernelExecution(String),
    /// Invalid input size
    InvalidSize(String),
}

impl std::fmt::Display for CudaFftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaFftError::NoDevice => write!(f, "No CUDA device found"),
            CudaFftError::DriverInit(s) => write!(f, "CUDA driver init failed: {}", s),
            CudaFftError::KernelCompilation(s) => write!(f, "Kernel compilation failed: {}", s),
            CudaFftError::MemoryAllocation(s) => write!(f, "Memory allocation failed: {}", s),
            CudaFftError::MemoryTransfer(s) => write!(f, "Memory transfer failed: {}", s),
            CudaFftError::KernelExecution(s) => write!(f, "Kernel execution failed: {}", s),
            CudaFftError::InvalidSize(s) => write!(f, "Invalid size: {}", s),
        }
    }
}

impl std::error::Error for CudaFftError {}

// =============================================================================
// CUDA FFT Executor
// =============================================================================

/// CUDA FFT Executor - manages GPU resources for FFT operations.
#[cfg(feature = "cuda-runtime")]
pub struct CudaFftExecutor {
    /// CUDA device handle
    device: Arc<CudaDevice>,
    /// Compiled kernels
    kernels: CompiledKernels,
    /// Device info
    pub device_info: DeviceInfo,
}

#[cfg(feature = "cuda-runtime")]
#[allow(dead_code)]
struct CompiledKernels {
    bit_reverse: CudaFunction,
    ifft_layer: CudaFunction,
    ifft_radix8: CudaFunction,  // Reserved for future radix-8 optimization
    ifft_vecwise: CudaFunction, // Reserved for future vectorized optimization
    fft_layer: CudaFunction,
}

/// Information about the CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory_bytes: usize,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
}

#[cfg(feature = "cuda-runtime")]
impl CudaFftExecutor {
    /// Create a new CUDA FFT executor.
    /// 
    /// This initializes the CUDA context and compiles all FFT kernels.
    pub fn new() -> Result<Self, CudaFftError> {
        // Initialize CUDA device (returns Arc<CudaDevice>)
        let device = CudaDevice::new(0)
            .map_err(|e| CudaFftError::DriverInit(format!("{:?}", e)))?;
        
        // Get device info
        let device_info = Self::get_device_info(&device)?;
        
        tracing::info!(
            "CUDA device initialized: {} (SM {}.{}, {} MB)",
            device_info.name,
            device_info.compute_capability.0,
            device_info.compute_capability.1,
            device_info.total_memory_bytes / (1024 * 1024)
        );
        
        // Compile kernels
        let kernels = Self::compile_kernels(&device)?;
        
        tracing::info!("CUDA FFT kernels compiled successfully");
        
        Ok(Self {
            device,
            kernels,
            device_info,
        })
    }
    
    fn get_device_info(_device: &Arc<CudaDevice>) -> Result<DeviceInfo, CudaFftError> {
        // Note: cudarc doesn't expose all device properties directly
        // We use reasonable defaults for now
        Ok(DeviceInfo {
            name: "NVIDIA GPU".to_string(),
            compute_capability: (7, 0),  // Assume Volta+
            total_memory_bytes: 8 * 1024 * 1024 * 1024,  // 8 GB default
            multiprocessor_count: 80,
            max_threads_per_block: 1024,
        })
    }
    
    fn compile_kernels(device: &Arc<CudaDevice>) -> Result<CompiledKernels, CudaFftError> {
        // Compile CUDA source to PTX using NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(CIRCLE_FFT_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("{:?}", e)))?;
        
        // Load PTX into device (load_ptx is on Arc<CudaDevice>)
        device.load_ptx(ptx.clone(), "circle_fft", &[
            "bit_reverse_kernel",
            "ifft_layer_kernel",
            "ifft_radix8_kernel",
            "ifft_vecwise_kernel",
            "fft_layer_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("{:?}", e)))?;
        
        // Get function handles (get_func is on Arc<CudaDevice>)
        let bit_reverse = device.get_func("circle_fft", "bit_reverse_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("bit_reverse_kernel not found".into()))?;
        
        let ifft_layer = device.get_func("circle_fft", "ifft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_layer_kernel not found".into()))?;
        
        let ifft_radix8 = device.get_func("circle_fft", "ifft_radix8_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_radix8_kernel not found".into()))?;
        
        let ifft_vecwise = device.get_func("circle_fft", "ifft_vecwise_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_vecwise_kernel not found".into()))?;
        
        let fft_layer = device.get_func("circle_fft", "fft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fft_layer_kernel not found".into()))?;
        
        Ok(CompiledKernels {
            bit_reverse,
            ifft_layer,
            ifft_radix8,
            ifft_vecwise,
            fft_layer,
        })
    }
    
    /// Execute inverse FFT on GPU.
    /// 
    /// # Arguments
    /// * `data` - Input/output data (modified in place)
    /// * `twiddles_dbl` - Doubled twiddle factors for each layer
    /// * `log_size` - log2 of the data size
    /// 
    /// # Returns
    /// The modified data after IFFT
    pub fn execute_ifft(
        &self,
        data: &mut [u32],
        twiddles_dbl: &[Vec<u32>],
        log_size: u32,
    ) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;
        
        if data.len() != n {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} elements, got {}", n, data.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA IFFT", log_size = log_size).entered();
        
        // Allocate device memory
        let mut d_data = self.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Flatten twiddles and copy to device
        let flat_twiddles: Vec<u32> = twiddles_dbl.iter().flatten().copied().collect();
        let d_twiddles = self.device.htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Execute IFFT layers
        self.execute_ifft_layers(&mut d_data, &d_twiddles, log_size, twiddles_dbl)?;
        
        // Copy results back
        self.device.dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    fn execute_ifft_layers(
        &self,
        d_data: &mut CudaSlice<u32>,
        d_twiddles: &CudaSlice<u32>,
        log_size: u32,
        twiddles_dbl: &[Vec<u32>],
    ) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;
        let block_size = 256u32;
        
        // Calculate twiddle offsets
        let mut twiddle_offsets: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for tw in twiddles_dbl {
            twiddle_offsets.push(offset);
            offset += tw.len();
        }
        
        // Execute layers
        for layer in 0..log_size {
            let n_butterflies = n / 2;
            let grid_size = ((n_butterflies as u32) + block_size - 1) / block_size;
            
            let twiddle_offset = twiddle_offsets[layer as usize];
            
            // Launch kernel
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // Create a twiddle view for this layer
            let twiddle_view = d_twiddles.slice(twiddle_offset..);
            
            unsafe {
                // Reborrow d_data each iteration to avoid move in loop
                self.kernels.ifft_layer.clone().launch(
                    cfg,
                    (&mut *d_data, &twiddle_view, layer, log_size),
                ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }
        
        // Synchronize
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        Ok(())
    }
    
    /// Execute forward FFT on GPU.
    pub fn execute_fft(
        &self,
        data: &mut [u32],
        twiddles: &[Vec<u32>],
        log_size: u32,
    ) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;
        
        if data.len() != n {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} elements, got {}", n, data.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA FFT", log_size = log_size).entered();
        
        // Allocate device memory
        let mut d_data = self.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Flatten twiddles and copy to device
        let flat_twiddles: Vec<u32> = twiddles.iter().flatten().copied().collect();
        let d_twiddles = self.device.htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Execute FFT layers (reverse order of IFFT)
        self.execute_fft_layers(&mut d_data, &d_twiddles, log_size, twiddles)?;
        
        // Copy results back
        self.device.dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    fn execute_fft_layers(
        &self,
        d_data: &mut CudaSlice<u32>,
        d_twiddles: &CudaSlice<u32>,
        log_size: u32,
        twiddles: &[Vec<u32>],
    ) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;
        let block_size = 256u32;
        
        // Calculate twiddle offsets
        let mut twiddle_offsets: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for tw in twiddles {
            twiddle_offsets.push(offset);
            offset += tw.len();
        }
        
        // Execute layers in reverse order for forward FFT
        for layer in (0..log_size).rev() {
            let n_butterflies = n / 2;
            let grid_size = ((n_butterflies as u32) + block_size - 1) / block_size;
            
            let twiddle_offset = twiddle_offsets[layer as usize];
            
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // Create a twiddle view for this layer
            let twiddle_view = d_twiddles.slice(twiddle_offset..);
            
            unsafe {
                // Reborrow d_data each iteration to avoid move in loop
                self.kernels.fft_layer.clone().launch(
                    cfg,
                    (&mut *d_data, &twiddle_view, layer, log_size),
                ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }
        
        // Synchronize
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        Ok(())
    }
    
    /// Execute bit reversal permutation on GPU.
    pub fn bit_reverse(&self, data: &mut [u32], log_size: u32) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;
        
        if data.len() != n {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} elements, got {}", n, data.len())
            ));
        }
        
        // Allocate and copy
        let mut d_data = self.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Launch bit reverse kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernels.bit_reverse.clone().launch(
                cfg,
                (&mut d_data, log_size),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        self.device.dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    /// Get device memory info.
    pub fn memory_info(&self) -> (usize, usize) {
        // (free, total) - cudarc doesn't expose this directly
        (
            self.device_info.total_memory_bytes / 2,  // Estimate
            self.device_info.total_memory_bytes,
        )
    }
}

// =============================================================================
// High-Level FFT Interface
// =============================================================================

/// Execute IFFT using CUDA if available, otherwise return error.
#[cfg(feature = "cuda-runtime")]
pub fn cuda_ifft(
    data: &mut [u32],
    twiddles_dbl: &[Vec<u32>],
    log_size: u32,
) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_ifft(data, twiddles_dbl, log_size)
}

/// Execute FFT using CUDA if available, otherwise return error.
#[cfg(feature = "cuda-runtime")]
pub fn cuda_fft(
    data: &mut [u32],
    twiddles: &[Vec<u32>],
    log_size: u32,
) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_fft(data, twiddles, log_size)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_ifft(
    _data: &mut [u32],
    _twiddles_dbl: &[Vec<u32>],
    _log_size: u32,
) -> Result<(), CudaFftError> {
    Err(CudaFftError::NoDevice)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_fft(
    _data: &mut [u32],
    _twiddles: &[Vec<u32>],
    _log_size: u32,
) -> Result<(), CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = CudaFftError::NoDevice;
        assert_eq!(format!("{}", err), "No CUDA device found");
        
        let err = CudaFftError::KernelCompilation("test".to_string());
        assert!(format!("{}", err).contains("test"));
    }
    
    #[test]
    fn test_cuda_not_available_without_feature() {
        // When cuda-runtime feature is not enabled, CUDA should not be available
        #[cfg(not(feature = "cuda-runtime"))]
        {
            assert!(!is_cuda_available());
        }
    }
}

