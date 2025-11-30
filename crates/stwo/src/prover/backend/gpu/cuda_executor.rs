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
    /// CUDA device handle (public for memory management)
    pub device: Arc<CudaDevice>,
    /// Compiled kernels
    kernels: CompiledKernels,
    /// Device info
    pub device_info: DeviceInfo,
}

#[cfg(feature = "cuda-runtime")]
struct CompiledKernels {
    // FFT kernels
    bit_reverse: CudaFunction,
    ifft_layer: CudaFunction,
    fft_layer: CudaFunction,
    // FRI folding kernels
    fold_line: CudaFunction,
    fold_circle_into_line: CudaFunction,
    // Quotient accumulation kernel
    accumulate_quotients: CudaFunction,
    // Merkle hashing kernel
    merkle_layer: CudaFunction,
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
        // Compile FFT CUDA source to PTX using NVRTC
        let fft_ptx = cudarc::nvrtc::compile_ptx(CIRCLE_FFT_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("FFT kernel: {:?}", e)))?;
        
        // Load FFT PTX into device
        device.load_ptx(fft_ptx, "circle_fft", &[
            "bit_reverse_kernel",
            "ifft_layer_kernel",
            "fft_layer_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("FFT load: {:?}", e)))?;
        
        // Compile FRI folding CUDA source to PTX
        use super::fft::FRI_FOLDING_CUDA_KERNEL;
        let fri_ptx = cudarc::nvrtc::compile_ptx(FRI_FOLDING_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("FRI kernel: {:?}", e)))?;
        
        // Load FRI PTX into device
        device.load_ptx(fri_ptx, "fri_folding", &[
            "fold_line_kernel",
            "fold_circle_into_line_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("FRI load: {:?}", e)))?;
        
        // Compile Quotient accumulation CUDA source to PTX
        use super::fft::QUOTIENT_CUDA_KERNEL;
        let quotient_ptx = cudarc::nvrtc::compile_ptx(QUOTIENT_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("Quotient kernel: {:?}", e)))?;
        
        // Load Quotient PTX into device
        device.load_ptx(quotient_ptx, "quotient", &[
            "accumulate_quotients_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("Quotient load: {:?}", e)))?;
        
        // Get FFT function handles
        let bit_reverse = device.get_func("circle_fft", "bit_reverse_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("bit_reverse_kernel not found".into()))?;
        
        let ifft_layer = device.get_func("circle_fft", "ifft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_layer_kernel not found".into()))?;
        
        let fft_layer = device.get_func("circle_fft", "fft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fft_layer_kernel not found".into()))?;
        
        // Get FRI function handles
        let fold_line = device.get_func("fri_folding", "fold_line_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fold_line_kernel not found".into()))?;
        
        let fold_circle_into_line = device.get_func("fri_folding", "fold_circle_into_line_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fold_circle_into_line_kernel not found".into()))?;
        
        // Get Quotient function handle
        let accumulate_quotients = device.get_func("quotient", "accumulate_quotients_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("accumulate_quotients_kernel not found".into()))?;
        
        // Compile Blake2s Merkle CUDA source to PTX
        use super::fft::BLAKE2S_MERKLE_CUDA_KERNEL;
        let merkle_ptx = cudarc::nvrtc::compile_ptx(BLAKE2S_MERKLE_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("Merkle kernel: {:?}", e)))?;
        
        // Load Merkle PTX into device
        device.load_ptx(merkle_ptx, "merkle", &[
            "merkle_layer_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("Merkle load: {:?}", e)))?;
        
        // Get Merkle function handle
        let merkle_layer = device.get_func("merkle", "merkle_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("merkle_layer_kernel not found".into()))?;
        
        tracing::info!("Compiled FFT, FRI, Quotient, and Merkle kernels successfully");
        
        Ok(CompiledKernels {
            bit_reverse,
            ifft_layer,
            fft_layer,
            fold_line,
            fold_circle_into_line,
            accumulate_quotients,
            merkle_layer,
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
        let num_layers = twiddles_dbl.len();
        
        // Validate we have the expected number of twiddle layers
        // For a domain of size 2^log_size, we need log_size layers of twiddles:
        // - Layer 0: circle layer (n/4 twiddles)
        // - Layers 1 to log_size-1: line layers
        if num_layers != log_size as usize {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} twiddle layers for log_size={}, got {}",
                log_size, log_size, num_layers
            )));
        }
        
        // Calculate twiddle offsets
        let mut twiddle_offsets: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for tw in twiddles_dbl {
            twiddle_offsets.push(offset);
            offset += tw.len();
        }
        
        // Execute butterfly layers
        // Layer 0 is the circle layer, layers 1+ are line layers
        for layer in 0..num_layers {
            let n_butterflies = n / 2;
            let grid_size = ((n_butterflies as u32) + block_size - 1) / block_size;
            
            let twiddle_offset = twiddle_offsets[layer];
            
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
                    (&mut *d_data, &twiddle_view, layer as u32, log_size),
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
        let num_layers = twiddles.len();
        
        // Validate we have the expected number of twiddle layers
        if num_layers != log_size as usize {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} twiddle layers for log_size={}, got {}",
                log_size, log_size, num_layers
            )));
        }
        
        // Calculate twiddle offsets
        let mut twiddle_offsets: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for tw in twiddles {
            twiddle_offsets.push(offset);
            offset += tw.len();
        }
        
        // Execute layers in reverse order for forward FFT
        // Layer 0 is circle layer, layers 1+ are line layers
        for layer in (0..num_layers).rev() {
            let n_butterflies = n / 2;
            let grid_size = ((n_butterflies as u32) + block_size - 1) / block_size;
            
            let twiddle_offset = twiddle_offsets[layer];
            
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
                    (&mut *d_data, &twiddle_view, layer as u32, log_size),
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
    
    // =========================================================================
    // FRI Folding Operations
    // =========================================================================
    
    /// Execute FRI fold_line on GPU.
    ///
    /// Folds a line evaluation by factor of 2 using the FRI protocol.
    pub fn execute_fold_line(
        &self,
        input: &[u32],
        itwiddles: &[u32],
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        // Validate input
        if input.len() != n * 4 {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} u32 values, got {}", n * 4, input.len())
            ));
        }
        if itwiddles.len() < n / 2 {
            return Err(CudaFftError::InvalidSize(
                format!("Expected at least {} twiddles, got {}", n / 2, itwiddles.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA fold_line", n = n).entered();
        
        let n_output = n / 2;
        
        // Allocate device memory
        let d_input = self.device.htod_sync_copy(input)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_itwiddles = self.device.htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_alpha = self.device.htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut d_output = unsafe {
            self.device.alloc::<u32>(n_output * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_output as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernels.fold_line.clone().launch(
                cfg,
                (&mut d_output, &d_input, &d_itwiddles, &d_alpha, n as u32, log_n),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        let mut output = vec![0u32; n_output * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU fold_line completed: {} -> {} elements", n, n_output);
        
        Ok(output)
    }
    
    /// Execute FRI fold_circle_into_line on GPU.
    ///
    /// Folds circle evaluation into line evaluation (accumulated).
    pub fn execute_fold_circle_into_line(
        &self,
        dst: &mut [u32],
        src: &[u32],
        itwiddles: &[u32],
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<(), CudaFftError> {
        // Validate input
        if src.len() != n * 4 {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} u32 values in src, got {}", n * 4, src.len())
            ));
        }
        let n_dst = n / 2;
        if dst.len() != n_dst * 4 {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} u32 values in dst, got {}", n_dst * 4, dst.len())
            ));
        }
        if itwiddles.len() < n_dst {
            return Err(CudaFftError::InvalidSize(
                format!("Expected at least {} twiddles, got {}", n_dst, itwiddles.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA fold_circle_into_line", n = n).entered();
        
        // Allocate device memory
        let d_src = self.device.htod_sync_copy(src)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut d_dst = self.device.htod_sync_copy(dst)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_itwiddles = self.device.htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_alpha = self.device.htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernels.fold_circle_into_line.clone().launch(
                cfg,
                (&mut d_dst, &d_src, &d_itwiddles, &d_alpha, n as u32, log_n),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        self.device.dtoh_sync_copy_into(&d_dst, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU fold_circle_into_line completed: {} -> {} elements", n, n_dst);
        
        Ok(())
    }
    
    // =========================================================================
    // Quotient Accumulation Operations
    // =========================================================================
    
    /// Execute quotient accumulation on GPU.
    pub fn execute_accumulate_quotients(
        &self,
        columns: &[Vec<u32>],
        line_coeffs: &[[u32; 12]],
        denom_inv: &[u32],
        batch_sizes: &[usize],
        col_indices: &[usize],
        n_points: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        let _span = tracing::span!(tracing::Level::INFO, "CUDA accumulate_quotients", n_points = n_points).entered();
        
        let n_columns = columns.len();
        let n_batches = batch_sizes.len();
        
        // Flatten columns (interleave by point, not by column)
        // Layout: col0[0], col1[0], col2[0], ..., col0[1], col1[1], ...
        let mut flat_columns: Vec<u32> = Vec::with_capacity(n_columns * n_points);
        for col in columns {
            flat_columns.extend_from_slice(col);
        }
        
        // Flatten line coefficients
        let flat_line_coeffs: Vec<u32> = line_coeffs.iter()
            .flat_map(|coeffs| coeffs.iter().copied())
            .collect();
        
        // Convert batch_sizes and col_indices to u32
        let batch_sizes_u32: Vec<u32> = batch_sizes.iter().map(|&s| s as u32).collect();
        let col_indices_u32: Vec<u32> = col_indices.iter().map(|&i| i as u32).collect();
        
        // Allocate device memory
        let d_columns = self.device.htod_sync_copy(&flat_columns)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_line_coeffs = self.device.htod_sync_copy(&flat_line_coeffs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_denom_inv = self.device.htod_sync_copy(denom_inv)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_batch_sizes = self.device.htod_sync_copy(&batch_sizes_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_col_indices = self.device.htod_sync_copy(&col_indices_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut d_output = unsafe {
            self.device.alloc::<u32>(n_points * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_points as u32) + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernels.accumulate_quotients.clone().launch(
                cfg,
                (
                    &mut d_output,
                    &d_columns,
                    &d_line_coeffs,
                    &d_denom_inv,
                    &d_batch_sizes,
                    &d_col_indices,
                    n_batches as u32,
                    n_points as u32,
                    n_columns as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        let mut output = vec![0u32; n_points * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU accumulate_quotients completed: {} points, {} batches", n_points, n_batches);
        
        Ok(output)
    }
    
    // =========================================================================
    // Merkle Hashing Operations
    // =========================================================================
    
    /// Execute Blake2s Merkle hashing on GPU.
    pub fn execute_blake2s_merkle(
        &self,
        columns: &[Vec<u32>],
        prev_layer: Option<&[u8]>,
        n_hashes: usize,
    ) -> Result<Vec<u8>, CudaFftError> {
        let _span = tracing::span!(tracing::Level::INFO, "CUDA blake2s_merkle", n_hashes = n_hashes).entered();
        
        let n_columns = columns.len();
        
        // Flatten columns
        let flat_columns: Vec<u32> = columns.iter()
            .flat_map(|col| col.iter().copied())
            .collect();
        
        // Allocate device memory for columns (if any)
        let d_columns = if n_columns > 0 {
            Some(self.device.htod_sync_copy(&flat_columns)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?)
        } else {
            None
        };
        
        // Allocate device memory for previous layer (if any)
        let d_prev_layer = if let Some(prev) = prev_layer {
            Some(self.device.htod_sync_copy(prev)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?)
        } else {
            None
        };
        
        // Allocate output (32 bytes per hash)
        let mut d_output = unsafe {
            self.device.alloc::<u8>(n_hashes * 32)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let has_prev_layer = if prev_layer.is_some() { 1u32 } else { 0u32 };
        
        unsafe {
            // We need to handle the optional parameters carefully
            // If columns is None, pass a null-like slice
            // If prev_layer is None, pass a null-like slice
            
            match (&d_columns, &d_prev_layer) {
                (Some(cols), Some(prev)) => {
                    self.kernels.merkle_layer.clone().launch(
                        cfg,
                        (
                            &mut d_output,
                            cols,
                            prev,
                            n_columns as u32,
                            n_hashes as u32,
                            has_prev_layer,
                        ),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (Some(cols), None) => {
                    // Need a dummy buffer for prev_layer
                    let dummy_prev = self.device.alloc::<u8>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels.merkle_layer.clone().launch(
                        cfg,
                        (
                            &mut d_output,
                            cols,
                            &dummy_prev,
                            n_columns as u32,
                            n_hashes as u32,
                            has_prev_layer,
                        ),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, Some(prev)) => {
                    // Need a dummy buffer for columns
                    let dummy_cols = self.device.alloc::<u32>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels.merkle_layer.clone().launch(
                        cfg,
                        (
                            &mut d_output,
                            &dummy_cols,
                            prev,
                            n_columns as u32,
                            n_hashes as u32,
                            has_prev_layer,
                        ),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, None) => {
                    return Err(CudaFftError::InvalidSize(
                        "Merkle hashing requires either columns or prev_layer".to_string()
                    ));
                }
            }
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        let mut output = vec![0u8; n_hashes * 32];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU blake2s_merkle completed: {} hashes", n_hashes);
        
        Ok(output)
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
// High-Level FRI Folding Interface
// =============================================================================

/// Execute FRI fold_line using CUDA.
///
/// # Arguments
/// * `input` - Input SecureField values as flat u32 array (4 u32 per element)
/// * `itwiddles` - Inverse twiddle factors
/// * `alpha` - Folding random challenge (4 u32 for QM31)
/// * `n` - Number of input elements
///
/// # Returns
/// Output SecureField values (n/2 elements, 4 u32 each)
#[cfg(feature = "cuda-runtime")]
pub fn cuda_fold_line(
    input: &[u32],
    itwiddles: &[u32],
    alpha: &[u32; 4],
    n: usize,
) -> Result<Vec<u32>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_fold_line(input, itwiddles, alpha, n)
}

/// Execute FRI fold_circle_into_line using CUDA.
///
/// # Arguments
/// * `dst` - Destination line evaluation (modified in place)
/// * `src` - Source circle evaluation
/// * `itwiddles` - Inverse twiddle factors (y-coordinates)
/// * `alpha` - Folding random challenge (4 u32 for QM31)
/// * `n` - Number of source elements
#[cfg(feature = "cuda-runtime")]
pub fn cuda_fold_circle_into_line(
    dst: &mut [u32],
    src: &[u32],
    itwiddles: &[u32],
    alpha: &[u32; 4],
    n: usize,
) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_fold_circle_into_line(dst, src, itwiddles, alpha, n)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_fold_line(
    _input: &[u32],
    _itwiddles: &[u32],
    _alpha: &[u32; 4],
    _n: usize,
) -> Result<Vec<u32>, CudaFftError> {
    Err(CudaFftError::NoDevice)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_fold_circle_into_line(
    _dst: &mut [u32],
    _src: &[u32],
    _itwiddles: &[u32],
    _alpha: &[u32; 4],
    _n: usize,
) -> Result<(), CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// High-Level Quotient Accumulation Interface
// =============================================================================

/// Execute quotient accumulation using CUDA.
///
/// # Arguments
/// * `columns` - Column data (each column is a Vec<u32> of M31 values)
/// * `line_coeffs` - Line coefficients (a, b, c as QM31, 12 u32 each)
/// * `denom_inv` - Denominator inverses (CM31, 2 u32 each)
/// * `batch_sizes` - Number of columns per sample batch
/// * `col_indices` - Column indices for each coefficient
/// * `n_points` - Number of domain points
///
/// # Returns
/// Output QM31 values (4 u32 per element)
#[cfg(feature = "cuda-runtime")]
pub fn cuda_accumulate_quotients(
    columns: &[Vec<u32>],
    line_coeffs: &[[u32; 12]],
    denom_inv: &[u32],
    batch_sizes: &[usize],
    col_indices: &[usize],
    n_points: usize,
) -> Result<Vec<u32>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_accumulate_quotients(
        columns, line_coeffs, denom_inv, batch_sizes, col_indices, n_points
    )
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_accumulate_quotients(
    _columns: &[Vec<u32>],
    _line_coeffs: &[[u32; 12]],
    _denom_inv: &[u32],
    _batch_sizes: &[usize],
    _col_indices: &[usize],
    _n_points: usize,
) -> Result<Vec<u32>, CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// High-Level Merkle Hashing Interface
// =============================================================================

/// Execute Blake2s Merkle hashing using CUDA.
///
/// # Arguments
/// * `columns` - Column data (each column is a Vec<u32> of M31 values)
/// * `prev_layer` - Previous layer hashes (64 bytes per pair, or None for leaves)
/// * `n_hashes` - Number of hashes to compute
///
/// # Returns
/// Output hashes (32 bytes each)
#[cfg(feature = "cuda-runtime")]
pub fn cuda_blake2s_merkle(
    columns: &[Vec<u32>],
    prev_layer: Option<&[u8]>,
    n_hashes: usize,
) -> Result<Vec<u8>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_blake2s_merkle(columns, prev_layer, n_hashes)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_blake2s_merkle(
    _columns: &[Vec<u32>],
    _prev_layer: Option<&[u8]>,
    _n_hashes: usize,
) -> Result<Vec<u8>, CudaFftError> {
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

