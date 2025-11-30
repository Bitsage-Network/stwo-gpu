//! Thread-Safe Multi-GPU Executor
//!
//! This module provides a thread-safe executor that can manage multiple GPUs
//! with proper CUDA context handling for parallel execution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    MultiGpuExecutorPool                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │   │ GpuExecutor  │  │ GpuExecutor  │  │ GpuExecutor  │  ...     │
//! │   │   (GPU 0)    │  │   (GPU 1)    │  │   (GPU 2)    │          │
//! │   │  Mutex<...>  │  │  Mutex<...>  │  │  Mutex<...>  │          │
//! │   └──────────────┘  └──────────────┘  └──────────────┘          │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each GPU has its own executor wrapped in a Mutex, allowing thread-safe
//! access from multiple threads.

#[cfg(feature = "cuda-runtime")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::{CudaFftError, CudaFftExecutor, DeviceInfo};
#[cfg(feature = "cuda-runtime")]
use super::fft::{compute_itwiddle_dbls_cpu, compute_twiddle_dbls_cpu};

// =============================================================================
// Global Multi-GPU Executor Pool
// =============================================================================

#[cfg(feature = "cuda-runtime")]
static MULTI_GPU_POOL: OnceLock<MultiGpuExecutorPool> = OnceLock::new();

/// Get or initialize the global multi-GPU executor pool.
#[cfg(feature = "cuda-runtime")]
pub fn get_multi_gpu_pool() -> Result<&'static MultiGpuExecutorPool, CudaFftError> {
    MULTI_GPU_POOL.get_or_init(|| {
        MultiGpuExecutorPool::new_all_gpus().unwrap_or_else(|_| {
            // Fallback to single GPU
            MultiGpuExecutorPool::new_with_devices(&[0]).unwrap()
        })
    });
    
    MULTI_GPU_POOL.get().ok_or(CudaFftError::NoDevice)
}

// =============================================================================
// Multi-GPU Executor Pool
// =============================================================================

/// Pool of GPU executors for multi-GPU operations.
#[cfg(feature = "cuda-runtime")]
pub struct MultiGpuExecutorPool {
    /// Per-GPU executors (thread-safe)
    executors: Vec<Arc<Mutex<GpuExecutorContext>>>,
    /// Device IDs
    device_ids: Vec<usize>,
}

/// Context for a single GPU with all resources needed for proof generation.
#[cfg(feature = "cuda-runtime")]
pub struct GpuExecutorContext {
    /// The CUDA executor
    pub executor: CudaFftExecutor,
    /// Cached twiddles for common sizes
    pub twiddle_cache: std::collections::HashMap<u32, TwiddleCache>,
}

/// Cached twiddles for a specific log_size.
#[cfg(feature = "cuda-runtime")]
pub struct TwiddleCache {
    pub itwiddles: CudaSlice<u32>,
    pub twiddles: CudaSlice<u32>,
    pub twiddle_offsets: CudaSlice<u32>,
    pub itwiddles_cpu: Vec<Vec<u32>>,
    pub twiddles_cpu: Vec<Vec<u32>>,
}

#[cfg(feature = "cuda-runtime")]
impl MultiGpuExecutorPool {
    /// Create a pool with all available GPUs.
    pub fn new_all_gpus() -> Result<Self, CudaFftError> {
        let device_count = Self::get_device_count()?;
        if device_count == 0 {
            return Err(CudaFftError::NoDevice);
        }
        
        let device_ids: Vec<usize> = (0..device_count).collect();
        Self::new_with_devices(&device_ids)
    }
    
    /// Create a pool with specific GPUs.
    pub fn new_with_devices(device_ids: &[usize]) -> Result<Self, CudaFftError> {
        let mut executors = Vec::new();
        let mut valid_ids = Vec::new();
        
        for &device_id in device_ids {
            match CudaFftExecutor::new_on_device(device_id) {
                Ok(executor) => {
                    let context = GpuExecutorContext {
                        executor,
                        twiddle_cache: std::collections::HashMap::new(),
                    };
                    executors.push(Arc::new(Mutex::new(context)));
                    valid_ids.push(device_id);
                    tracing::info!("Initialized GPU {} for multi-GPU pool", device_id);
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize GPU {}: {:?}", device_id, e);
                }
            }
        }
        
        if executors.is_empty() {
            return Err(CudaFftError::NoDevice);
        }
        
        tracing::info!("Multi-GPU pool initialized with {} GPUs", executors.len());
        
        Ok(Self {
            executors,
            device_ids: valid_ids,
        })
    }
    
    /// Get the number of GPUs in the pool.
    pub fn gpu_count(&self) -> usize {
        self.executors.len()
    }
    
    /// Get device IDs.
    pub fn device_ids(&self) -> &[usize] {
        &self.device_ids
    }
    
    /// Get executor for a specific GPU (by pool index, not device ID).
    pub fn get_executor(&self, pool_index: usize) -> Option<Arc<Mutex<GpuExecutorContext>>> {
        self.executors.get(pool_index).cloned()
    }
    
    /// Execute a function on a specific GPU.
    /// 
    /// This locks the GPU's executor for the duration of the function.
    pub fn with_gpu<F, R>(&self, pool_index: usize, f: F) -> Result<R, CudaFftError>
    where
        F: FnOnce(&mut GpuExecutorContext) -> Result<R, CudaFftError>,
    {
        let executor = self.executors.get(pool_index)
            .ok_or_else(|| CudaFftError::InvalidSize(format!("Invalid GPU index: {}", pool_index)))?;
        
        let mut guard = executor.lock()
            .map_err(|_| CudaFftError::KernelExecution("Failed to lock GPU executor".into()))?;
        
        f(&mut guard)
    }
    
    /// Execute functions on all GPUs in parallel.
    /// 
    /// Returns results from all GPUs.
    pub fn parallel_execute<F, R>(&self, f: F) -> Vec<Result<R, CudaFftError>>
    where
        F: Fn(usize, &mut GpuExecutorContext) -> Result<R, CudaFftError> + Send + Sync,
        R: Send,
    {
        use std::thread;
        
        let f = Arc::new(f);
        let mut handles = Vec::new();
        
        for (idx, executor) in self.executors.iter().enumerate() {
            let executor = Arc::clone(executor);
            let f = Arc::clone(&f);
            
            let handle = thread::spawn(move || {
                let mut guard = executor.lock()
                    .map_err(|_| CudaFftError::KernelExecution("Failed to lock GPU executor".into()))?;
                f(idx, &mut guard)
            });
            
            handles.push(handle);
        }
        
        handles.into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(CudaFftError::KernelExecution("Thread panicked".into()))))
            .collect()
    }
    
    fn get_device_count() -> Result<usize, CudaFftError> {
        let mut count = 0;
        for i in 0..16 {
            if CudaDevice::new(i).is_ok() {
                count = i + 1;
            } else {
                break;
            }
        }
        Ok(count)
    }
}

#[cfg(feature = "cuda-runtime")]
impl GpuExecutorContext {
    /// Get or create twiddles for a specific log_size.
    pub fn get_or_create_twiddles(&mut self, log_size: u32) -> Result<&TwiddleCache, CudaFftError> {
        if !self.twiddle_cache.contains_key(&log_size) {
            let cache = self.create_twiddle_cache(log_size)?;
            self.twiddle_cache.insert(log_size, cache);
        }
        
        Ok(self.twiddle_cache.get(&log_size).unwrap())
    }
    
    fn create_twiddle_cache(&self, log_size: u32) -> Result<TwiddleCache, CudaFftError> {
        let itwiddles_cpu = compute_itwiddle_dbls_cpu(log_size);
        let twiddles_cpu = compute_twiddle_dbls_cpu(log_size);
        
        let flat_itwiddles: Vec<u32> = itwiddles_cpu.iter().flatten().copied().collect();
        let flat_twiddles: Vec<u32> = twiddles_cpu.iter().flatten().copied().collect();
        
        let itwiddles = self.executor.device.htod_sync_copy(&flat_itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let twiddles = self.executor.device.htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut offsets: Vec<u32> = Vec::new();
        let mut offset = 0u32;
        for tw in &itwiddles_cpu {
            offsets.push(offset);
            offset += tw.len() as u32;
        }
        let twiddle_offsets = self.executor.device.htod_sync_copy(&offsets)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        Ok(TwiddleCache {
            itwiddles,
            twiddles,
            twiddle_offsets,
            itwiddles_cpu,
            twiddles_cpu,
        })
    }
    
    /// Allocate GPU memory for polynomial data.
    pub fn allocate_poly(&self, log_size: u32) -> Result<CudaSlice<u32>, CudaFftError> {
        let n = 1usize << log_size;
        unsafe {
            self.executor.device.alloc::<u32>(n)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))
    }
    
    /// Upload polynomial to GPU.
    pub fn upload_poly(&self, data: &[u32]) -> Result<CudaSlice<u32>, CudaFftError> {
        self.executor.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))
    }
    
    /// Download polynomial from GPU.
    pub fn download_poly(&self, d_data: &CudaSlice<u32>) -> Result<Vec<u32>, CudaFftError> {
        self.executor.device.dtoh_sync_copy(d_data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))
    }
    
    /// Synchronize the device.
    pub fn sync(&self) -> Result<(), CudaFftError> {
        self.executor.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))
    }
}

// =============================================================================
// Thread-Safe Proof Pipeline
// =============================================================================

/// A proof pipeline that can run on any GPU from the pool.
#[cfg(feature = "cuda-runtime")]
pub struct ThreadSafeProofPipeline {
    /// GPU pool index this pipeline is assigned to
    gpu_index: usize,
    /// Polynomial data on GPU
    poly_data: Vec<CudaSlice<u32>>,
    /// Log size
    log_size: u32,
}

#[cfg(feature = "cuda-runtime")]
impl ThreadSafeProofPipeline {
    /// Create a new pipeline on a specific GPU from the pool.
    pub fn new(gpu_index: usize, log_size: u32) -> Result<Self, CudaFftError> {
        let pool = get_multi_gpu_pool()?;
        
        if gpu_index >= pool.gpu_count() {
            return Err(CudaFftError::InvalidSize(
                format!("GPU index {} out of range (pool has {} GPUs)", gpu_index, pool.gpu_count())
            ));
        }
        
        // Initialize twiddles for this log_size
        pool.with_gpu(gpu_index, |ctx| {
            ctx.get_or_create_twiddles(log_size)?;
            Ok(())
        })?;
        
        Ok(Self {
            gpu_index,
            poly_data: Vec::new(),
            log_size,
        })
    }
    
    /// Upload polynomial data.
    pub fn upload_polynomial(&mut self, data: &[u32]) -> Result<usize, CudaFftError> {
        let pool = get_multi_gpu_pool()?;
        
        let d_data = pool.with_gpu(self.gpu_index, |ctx| {
            ctx.upload_poly(data)
        })?;
        
        let idx = self.poly_data.len();
        self.poly_data.push(d_data);
        Ok(idx)
    }
    
    /// Execute IFFT on a polynomial.
    pub fn ifft(&mut self, poly_idx: usize) -> Result<(), CudaFftError> {
        if poly_idx >= self.poly_data.len() {
            return Err(CudaFftError::InvalidSize(format!("Invalid poly index: {}", poly_idx)));
        }
        
        let pool = get_multi_gpu_pool()?;
        let log_size = self.log_size;
        
        // We need to move the data out temporarily due to borrow checker
        let mut d_poly = std::mem::take(&mut self.poly_data[poly_idx]);
        
        pool.with_gpu(self.gpu_index, |ctx| {
            let twiddles = ctx.get_or_create_twiddles(log_size)?;
            ctx.executor.execute_ifft_on_device(
                &mut d_poly,
                &twiddles.itwiddles,
                &twiddles.twiddle_offsets,
                &twiddles.itwiddles_cpu,
                log_size,
            )
        })?;
        
        self.poly_data[poly_idx] = d_poly;
        Ok(())
    }
    
    /// Execute FFT on a polynomial.
    pub fn fft(&mut self, poly_idx: usize) -> Result<(), CudaFftError> {
        if poly_idx >= self.poly_data.len() {
            return Err(CudaFftError::InvalidSize(format!("Invalid poly index: {}", poly_idx)));
        }
        
        let pool = get_multi_gpu_pool()?;
        let log_size = self.log_size;
        
        let mut d_poly = std::mem::take(&mut self.poly_data[poly_idx]);
        
        pool.with_gpu(self.gpu_index, |ctx| {
            let twiddles = ctx.get_or_create_twiddles(log_size)?;
            ctx.executor.execute_fft_on_device(
                &mut d_poly,
                &twiddles.twiddles,
                &twiddles.twiddle_offsets,
                &twiddles.twiddles_cpu,
                log_size,
            )
        })?;
        
        self.poly_data[poly_idx] = d_poly;
        Ok(())
    }
    
    /// Synchronize the GPU.
    pub fn sync(&self) -> Result<(), CudaFftError> {
        let pool = get_multi_gpu_pool()?;
        pool.with_gpu(self.gpu_index, |ctx| ctx.sync())
    }
    
    /// Get the GPU index this pipeline is running on.
    pub fn gpu_index(&self) -> usize {
        self.gpu_index
    }
}

// =============================================================================
// True Multi-GPU Prover
// =============================================================================

/// Multi-GPU prover that truly uses all GPUs in parallel.
#[cfg(feature = "cuda-runtime")]
pub struct TrueMultiGpuProver {
    log_size: u32,
}

#[cfg(feature = "cuda-runtime")]
impl TrueMultiGpuProver {
    /// Create a new multi-GPU prover.
    pub fn new(log_size: u32) -> Result<Self, CudaFftError> {
        // Initialize the pool
        let _ = get_multi_gpu_pool()?;
        Ok(Self { log_size })
    }
    
    /// Get the number of available GPUs.
    pub fn gpu_count(&self) -> Result<usize, CudaFftError> {
        Ok(get_multi_gpu_pool()?.gpu_count())
    }
    
    /// Process proofs in parallel across all GPUs.
    /// 
    /// Each GPU processes a subset of the workloads.
    pub fn prove_parallel<F, R>(&self, workloads: Vec<Vec<u32>>, process_fn: F) -> Vec<Result<R, CudaFftError>>
    where
        F: Fn(usize, &mut GpuExecutorContext, &[u32], u32) -> Result<R, CudaFftError> + Send + Sync + 'static,
        R: Send + 'static,
    {
        use std::thread;
        
        let pool = match get_multi_gpu_pool() {
            Ok(p) => p,
            Err(e) => return vec![Err(e)],
        };
        
        let num_gpus = pool.gpu_count();
        let log_size = self.log_size;
        
        // Distribute workloads across GPUs
        let mut gpu_workloads: Vec<Vec<(usize, Vec<u32>)>> = vec![Vec::new(); num_gpus];
        for (i, workload) in workloads.into_iter().enumerate() {
            let gpu_idx = i % num_gpus;
            gpu_workloads[gpu_idx].push((i, workload));
        }
        
        let process_fn = Arc::new(process_fn);
        let mut handles = Vec::new();
        
        for (gpu_idx, workloads) in gpu_workloads.into_iter().enumerate() {
            let executor = match pool.get_executor(gpu_idx) {
                Some(e) => e,
                None => continue,
            };
            let process_fn = Arc::clone(&process_fn);
            
            let handle = thread::spawn(move || {
                let mut results = Vec::new();
                
                let mut guard = executor.lock()
                    .map_err(|_| CudaFftError::KernelExecution("Lock failed".into()))?;
                
                for (_orig_idx, workload) in workloads {
                    let result = process_fn(gpu_idx, &mut guard, &workload, log_size);
                    results.push(result);
                }
                
                Ok::<Vec<Result<R, CudaFftError>>, CudaFftError>(results)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut all_results = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(results)) => all_results.extend(results),
                Ok(Err(e)) => all_results.push(Err(e)),
                Err(_) => all_results.push(Err(CudaFftError::KernelExecution("Thread panicked".into()))),
            }
        }
        
        all_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_multi_gpu_pool_creation() {
        // This test will only pass on a system with CUDA GPUs
        if let Ok(pool) = MultiGpuExecutorPool::new_all_gpus() {
            assert!(pool.gpu_count() > 0);
            println!("Found {} GPUs", pool.gpu_count());
        }
    }
}

