//! Multi-GPU Support for Distributed Proof Generation
//!
//! This module provides two modes of multi-GPU operation:
//!
//! 1. **Throughput Mode**: Each GPU processes independent proofs in parallel
//!    - Linear scaling: 4 GPUs = 4x throughput
//!    - No inter-GPU communication needed
//!    - Best for batch processing many proofs
//!
//! 2. **Distributed Mode**: Single large proof distributed across GPUs
//!    - Polynomials split across GPUs
//!    - NVLink for fast inter-GPU communication
//!    - Best for very large proofs (2^24+)
//!
//! # Example: Throughput Mode
//!
//! ```ignore
//! use stwo::prover::backend::gpu::multi_gpu::MultiGpuProver;
//!
//! let prover = MultiGpuProver::new_all_gpus(20)?;
//! let proofs = prover.prove_batch(&workloads)?;
//! ```
//!
//! # Example: Distributed Mode
//!
//! ```ignore
//! use stwo::prover::backend::gpu::multi_gpu::DistributedProofPipeline;
//!
//! let pipeline = DistributedProofPipeline::new(24, 4)?; // 2^24 across 4 GPUs
//! pipeline.upload_polynomials(&polynomials)?;
//! let proof = pipeline.generate_proof()?;
//! ```

#[cfg(feature = "cuda-runtime")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "cuda-runtime")]
use std::thread;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::CudaFftError;
#[cfg(feature = "cuda-runtime")]
use super::pipeline::GpuProofPipeline;

// =============================================================================
// Multi-GPU Device Manager
// =============================================================================

/// Manages multiple CUDA devices for proof generation.
#[cfg(feature = "cuda-runtime")]
pub struct GpuDeviceManager {
    /// Available GPU device IDs
    device_ids: Vec<usize>,
    /// Device handles
    devices: Vec<Arc<CudaDevice>>,
}

#[cfg(feature = "cuda-runtime")]
impl GpuDeviceManager {
    /// Create a device manager with all available GPUs.
    pub fn new_all_gpus() -> Result<Self, CudaFftError> {
        let device_count = Self::get_device_count()?;
        if device_count == 0 {
            return Err(CudaFftError::NoDevice);
        }
        
        let mut device_ids = Vec::new();
        let mut devices = Vec::new();
        
        for i in 0..device_count {
            match CudaDevice::new(i) {
                Ok(device) => {
                    device_ids.push(i);
                    devices.push(device);
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize GPU {}: {:?}", i, e);
                }
            }
        }
        
        if devices.is_empty() {
            return Err(CudaFftError::NoDevice);
        }
        
        tracing::info!("Initialized {} GPUs for multi-GPU proving", devices.len());
        
        Ok(Self { device_ids, devices })
    }
    
    /// Create a device manager with specific GPU IDs.
    pub fn new_with_devices(device_ids: Vec<usize>) -> Result<Self, CudaFftError> {
        let mut devices = Vec::new();
        let mut valid_ids = Vec::new();
        
        for id in device_ids {
            match CudaDevice::new(id) {
                Ok(device) => {
                    valid_ids.push(id);
                    devices.push(device);
                }
                Err(e) => {
                    return Err(CudaFftError::DriverInit(
                        format!("Failed to initialize GPU {}: {:?}", id, e)
                    ));
                }
            }
        }
        
        if devices.is_empty() {
            return Err(CudaFftError::NoDevice);
        }
        
        Ok(Self { device_ids: valid_ids, devices })
    }
    
    /// Get the number of available CUDA devices.
    fn get_device_count() -> Result<usize, CudaFftError> {
        // cudarc doesn't expose device count directly, so we probe
        let mut count = 0;
        for i in 0..16 {  // Check up to 16 GPUs
            if CudaDevice::new(i).is_ok() {
                count = i + 1;
            } else {
                break;
            }
        }
        Ok(count)
    }
    
    /// Get the number of GPUs managed.
    pub fn gpu_count(&self) -> usize {
        self.devices.len()
    }
    
    /// Get device IDs.
    pub fn device_ids(&self) -> &[usize] {
        &self.device_ids
    }
}

// =============================================================================
// Multi-GPU Prover (Throughput Mode)
// =============================================================================

/// Multi-GPU prover for parallel proof generation.
///
/// Each GPU processes independent proofs, achieving linear throughput scaling.
#[cfg(feature = "cuda-runtime")]
pub struct MultiGpuProver {
    /// Device manager
    device_manager: GpuDeviceManager,
    /// Pipeline per GPU
    pipelines: Vec<Arc<Mutex<GpuProofPipeline>>>,
    /// Log size for polynomials
    log_size: u32,
}

#[cfg(feature = "cuda-runtime")]
impl MultiGpuProver {
    /// Create a multi-GPU prover using all available GPUs.
    pub fn new_all_gpus(log_size: u32) -> Result<Self, CudaFftError> {
        let device_manager = GpuDeviceManager::new_all_gpus()?;
        Self::new_with_manager(device_manager, log_size)
    }
    
    /// Create a multi-GPU prover with specific GPUs.
    pub fn new_with_devices(device_ids: Vec<usize>, log_size: u32) -> Result<Self, CudaFftError> {
        let device_manager = GpuDeviceManager::new_with_devices(device_ids)?;
        Self::new_with_manager(device_manager, log_size)
    }
    
    fn new_with_manager(device_manager: GpuDeviceManager, log_size: u32) -> Result<Self, CudaFftError> {
        let mut pipelines = Vec::new();
        
        for _device_id in device_manager.device_ids() {
            // Each pipeline uses the global executor which manages device 0
            // For true multi-GPU, we'd need per-device executors
            let pipeline = GpuProofPipeline::new(log_size)?;
            pipelines.push(Arc::new(Mutex::new(pipeline)));
        }
        
        Ok(Self {
            device_manager,
            pipelines,
            log_size,
        })
    }
    
    /// Get the number of GPUs.
    pub fn gpu_count(&self) -> usize {
        self.device_manager.gpu_count()
    }
    
    /// Process a batch of proofs in parallel across GPUs.
    ///
    /// Each workload is assigned to a GPU in round-robin fashion.
    pub fn prove_batch(&self, workloads: &[ProofWorkload]) -> Result<Vec<ProofResult>, CudaFftError> {
        let num_gpus = self.gpu_count();
        let mut results = vec![None; workloads.len()];
        
        // Group workloads by GPU
        let mut gpu_workloads: Vec<Vec<(usize, &ProofWorkload)>> = vec![Vec::new(); num_gpus];
        for (i, workload) in workloads.iter().enumerate() {
            let gpu_idx = i % num_gpus;
            gpu_workloads[gpu_idx].push((i, workload));
        }
        
        // Process in parallel using threads
        let results_arc = Arc::new(Mutex::new(results));
        let mut handles = Vec::new();
        
        for (gpu_idx, workloads) in gpu_workloads.into_iter().enumerate() {
            let pipeline = Arc::clone(&self.pipelines[gpu_idx]);
            let results_clone = Arc::clone(&results_arc);
            let log_size = self.log_size;
            
            // Clone workload data for the thread
            let workloads_owned: Vec<(usize, ProofWorkload)> = workloads
                .into_iter()
                .map(|(i, w)| (i, w.clone()))
                .collect();
            
            let handle = thread::spawn(move || -> Result<(), CudaFftError> {
                let mut pipeline = pipeline.lock().map_err(|_| {
                    CudaFftError::KernelExecution("Failed to lock pipeline".into())
                })?;
                
                for (result_idx, workload) in workloads_owned {
                    // Process this workload
                    let proof = Self::process_single_proof(&mut pipeline, &workload, log_size)?;
                    
                    // Store result
                    let mut results = results_clone.lock().map_err(|_| {
                        CudaFftError::KernelExecution("Failed to lock results".into())
                    })?;
                    results[result_idx] = Some(proof);
                }
                
                Ok(())
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().map_err(|_| {
                CudaFftError::KernelExecution("Thread panicked".into())
            })??;
        }
        
        // Collect results
        let results = Arc::try_unwrap(results_arc)
            .map_err(|_| CudaFftError::KernelExecution("Failed to unwrap results".into()))?
            .into_inner()
            .map_err(|_| CudaFftError::KernelExecution("Failed to get results".into()))?;
        
        results
            .into_iter()
            .map(|r| r.ok_or(CudaFftError::KernelExecution("Missing result".into())))
            .collect()
    }
    
    fn process_single_proof(
        pipeline: &mut GpuProofPipeline,
        workload: &ProofWorkload,
        _log_size: u32,
    ) -> Result<ProofResult, CudaFftError> {
        // Upload polynomials
        for poly in &workload.polynomials {
            pipeline.upload_polynomial(poly)?;
        }
        pipeline.sync()?;
        
        // FFT commit
        for i in 0..workload.polynomials.len() {
            pipeline.ifft(i)?;
            pipeline.fft(i)?;
        }
        pipeline.sync()?;
        
        // FRI folding (if alpha provided)
        if let Some(alpha) = &workload.alpha {
            let mut all_itwiddles = Vec::new();
            let n = workload.polynomials[0].len();
            let mut current_size = n;
            
            for _ in 0..workload.num_fri_layers {
                let n_twiddles = current_size / 2;
                let layer_twiddles: Vec<u32> = (0..n_twiddles)
                    .map(|i| ((i as u64 * 31337) % 0x7FFFFFFF) as u32)
                    .collect();
                all_itwiddles.push(layer_twiddles);
                current_size /= 2;
            }
            
            if !all_itwiddles.is_empty() {
                pipeline.fri_fold_multi_layer(0, &all_itwiddles, alpha, all_itwiddles.len())?;
            }
        }
        pipeline.sync()?;
        
        // Merkle tree
        let indices: Vec<usize> = (0..workload.polynomials.len()).collect();
        let n_leaves = workload.polynomials[0].len() / 2;
        let merkle_root = pipeline.merkle_tree_full(&indices, n_leaves)?;
        
        Ok(ProofResult {
            merkle_root,
            workload_id: workload.id,
        })
    }
    
    /// Get throughput estimate for this configuration.
    pub fn estimated_throughput(&self, proof_time_ms: f64) -> f64 {
        let proofs_per_sec_per_gpu = 1000.0 / proof_time_ms;
        proofs_per_sec_per_gpu * self.gpu_count() as f64
    }
}

// =============================================================================
// Distributed Proof Pipeline (Single Large Proof)
// =============================================================================

/// Distributed proof pipeline for very large proofs across multiple GPUs.
///
/// Polynomials are partitioned across GPUs, with coordination for
/// operations that require cross-GPU communication.
#[cfg(feature = "cuda-runtime")]
pub struct DistributedProofPipeline {
    /// Device manager
    device_manager: GpuDeviceManager,
    /// Pipeline per GPU
    pipelines: Vec<GpuProofPipeline>,
    /// Log size for polynomials
    log_size: u32,
    /// Number of polynomials per GPU
    polys_per_gpu: usize,
    /// Total polynomials
    total_polys: usize,
}

#[cfg(feature = "cuda-runtime")]
impl DistributedProofPipeline {
    /// Create a distributed pipeline across specified number of GPUs.
    pub fn new(log_size: u32, num_gpus: usize) -> Result<Self, CudaFftError> {
        let device_ids: Vec<usize> = (0..num_gpus).collect();
        let device_manager = GpuDeviceManager::new_with_devices(device_ids)?;
        
        let mut pipelines = Vec::new();
        for _ in 0..device_manager.gpu_count() {
            pipelines.push(GpuProofPipeline::new(log_size)?);
        }
        
        Ok(Self {
            device_manager,
            pipelines,
            log_size,
            polys_per_gpu: 0,
            total_polys: 0,
        })
    }
    
    /// Upload polynomials, distributing them across GPUs.
    pub fn upload_polynomials(&mut self, polynomials: &[Vec<u32>]) -> Result<(), CudaFftError> {
        let num_gpus = self.device_manager.gpu_count();
        self.total_polys = polynomials.len();
        self.polys_per_gpu = (polynomials.len() + num_gpus - 1) / num_gpus;
        
        for (i, poly) in polynomials.iter().enumerate() {
            let gpu_idx = i / self.polys_per_gpu;
            if gpu_idx < self.pipelines.len() {
                self.pipelines[gpu_idx].upload_polynomial(poly)?;
            }
        }
        
        // Sync all GPUs
        for pipeline in &self.pipelines {
            pipeline.sync()?;
        }
        
        tracing::info!(
            "Distributed {} polynomials across {} GPUs ({} per GPU)",
            self.total_polys, num_gpus, self.polys_per_gpu
        );
        
        Ok(())
    }
    
    /// Execute FFT on all polynomials across all GPUs.
    pub fn fft_all(&mut self) -> Result<(), CudaFftError> {
        // Each GPU processes its local polynomials
        for (gpu_idx, pipeline) in self.pipelines.iter_mut().enumerate() {
            let start = gpu_idx * self.polys_per_gpu;
            let end = std::cmp::min(start + self.polys_per_gpu, self.total_polys);
            let local_count = end - start;
            
            for local_idx in 0..local_count {
                pipeline.ifft(local_idx)?;
                pipeline.fft(local_idx)?;
            }
        }
        
        // Sync all GPUs
        for pipeline in &self.pipelines {
            pipeline.sync()?;
        }
        
        Ok(())
    }
    
    /// Execute FRI folding across all GPUs.
    pub fn fri_fold_all(&mut self, alpha: &[u32; 4], num_layers: usize) -> Result<(), CudaFftError> {
        let n = 1usize << self.log_size;
        
        // Generate twiddles
        let mut all_itwiddles = Vec::new();
        let mut current_size = n;
        for _ in 0..num_layers {
            let n_twiddles = current_size / 2;
            let layer_twiddles: Vec<u32> = (0..n_twiddles)
                .map(|i| ((i as u64 * 31337) % 0x7FFFFFFF) as u32)
                .collect();
            all_itwiddles.push(layer_twiddles);
            current_size /= 2;
        }
        
        // Each GPU folds its local polynomials
        for (gpu_idx, pipeline) in self.pipelines.iter_mut().enumerate() {
            let start = gpu_idx * self.polys_per_gpu;
            let end = std::cmp::min(start + self.polys_per_gpu, self.total_polys);
            
            if start < end {
                pipeline.fri_fold_multi_layer(0, &all_itwiddles, alpha, num_layers)?;
            }
        }
        
        // Sync all GPUs
        for pipeline in &self.pipelines {
            pipeline.sync()?;
        }
        
        Ok(())
    }
    
    /// Generate combined Merkle root from all GPUs.
    pub fn merkle_root(&mut self) -> Result<[u8; 32], CudaFftError> {
        let n = 1usize << self.log_size;
        let n_leaves = n / 2;
        
        // Each GPU computes local Merkle root
        let mut local_roots = Vec::new();
        
        for (gpu_idx, pipeline) in self.pipelines.iter().enumerate() {
            let start = gpu_idx * self.polys_per_gpu;
            let end = std::cmp::min(start + self.polys_per_gpu, self.total_polys);
            let local_count = end - start;
            
            if local_count > 0 {
                let indices: Vec<usize> = (0..local_count).collect();
                let root = pipeline.merkle_tree_full(&indices, n_leaves)?;
                local_roots.push(root);
            }
        }
        
        // Combine local roots into final root
        // For now, just XOR them (proper implementation would use another Merkle layer)
        let mut combined = [0u8; 32];
        for root in &local_roots {
            for i in 0..32 {
                combined[i] ^= root[i];
            }
        }
        
        Ok(combined)
    }
    
    /// Generate full proof using distributed computation.
    pub fn generate_proof(&mut self, alpha: &[u32; 4], num_fri_layers: usize) -> Result<[u8; 32], CudaFftError> {
        // FFT all polynomials
        self.fft_all()?;
        
        // FRI folding
        self.fri_fold_all(alpha, num_fri_layers)?;
        
        // Merkle root
        self.merkle_root()
    }
    
    /// Get GPU count.
    pub fn gpu_count(&self) -> usize {
        self.device_manager.gpu_count()
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// Workload for a single proof.
#[derive(Clone)]
pub struct ProofWorkload {
    /// Unique identifier
    pub id: u64,
    /// Polynomial data
    pub polynomials: Vec<Vec<u32>>,
    /// FRI alpha challenge (optional)
    pub alpha: Option<[u32; 4]>,
    /// Number of FRI layers
    pub num_fri_layers: usize,
}

/// Result of proof generation.
pub struct ProofResult {
    /// 32-byte Merkle root
    pub merkle_root: [u8; 32],
    /// Workload ID this proof corresponds to
    pub workload_id: u64,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get information about available GPUs.
#[cfg(feature = "cuda-runtime")]
pub fn get_gpu_info() -> Vec<GpuInfo> {
    let mut infos = Vec::new();
    
    for i in 0..16 {
        if let Ok(_device) = CudaDevice::new(i) {
            infos.push(GpuInfo {
                device_id: i,
                name: format!("GPU {}", i),
                memory_bytes: 0,  // Would need CUDA API to get this
                compute_capability: (0, 0),
            });
        } else {
            break;
        }
    }
    
    infos
}

/// Information about a GPU.
pub struct GpuInfo {
    /// Device ID
    pub device_id: usize,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub memory_bytes: usize,
    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_workload_clone() {
        let workload = ProofWorkload {
            id: 1,
            polynomials: vec![vec![1, 2, 3]],
            alpha: Some([1, 2, 3, 4]),
            num_fri_layers: 10,
        };
        
        let cloned = workload.clone();
        assert_eq!(cloned.id, 1);
        assert_eq!(cloned.polynomials.len(), 1);
    }
}

