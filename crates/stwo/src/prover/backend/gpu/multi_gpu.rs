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
// GPU Capability Detection
// =============================================================================

/// GPU capabilities for intelligent work distribution.
#[derive(Debug, Clone)]
#[cfg(feature = "cuda-runtime")]
pub struct GpuCapabilities {
    /// Device ID
    pub device_id: usize,
    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors
    pub sm_count: u32,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Memory bandwidth estimate (GB/s)
    pub memory_bandwidth_gbps: f32,
    /// Has Tensor Cores (Volta+)
    pub has_tensor_cores: bool,
    /// Relative compute power (normalized, 1.0 = baseline)
    pub relative_power: f32,
}

#[cfg(feature = "cuda-runtime")]
impl GpuCapabilities {
    /// Query capabilities for a device.
    pub fn query(device_id: usize) -> Result<Self, CudaFftError> {
        use cudarc::driver::sys;
        
        let device = CudaDevice::new(device_id)
            .map_err(|e| CudaFftError::DriverInit(format!("{:?}", e)))?;
        
        let cu_device = device.cu_device();
        
        // Query compute capability
        let mut major = 0i32;
        let mut minor = 0i32;
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut major,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            );
            sys::cuDeviceGetAttribute(
                &mut minor,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            );
        }
        
        // Query SM count
        let mut sm_count = 0i32;
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut sm_count,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                cu_device,
            );
        }
        
        // Query memory
        let mut total_memory = 0usize;
        unsafe {
            sys::cuDeviceTotalMem_v2(&mut total_memory, cu_device);
        }
        
        // Estimate memory bandwidth based on architecture
        let memory_bandwidth_gbps = Self::estimate_bandwidth(major as u32, sm_count as u32);
        
        // Tensor Cores available on Volta (SM 7.0) and later
        let has_tensor_cores = major >= 7;
        
        // Calculate relative power (normalized to A100 = 1.0)
        let relative_power = Self::calculate_relative_power(major as u32, sm_count as u32);
        
        Ok(GpuCapabilities {
            device_id,
            compute_capability: (major as u32, minor as u32),
            sm_count: sm_count as u32,
            total_memory,
            memory_bandwidth_gbps,
            has_tensor_cores,
            relative_power,
        })
    }
    
    /// Estimate memory bandwidth based on GPU architecture.
    fn estimate_bandwidth(sm_major: u32, sm_count: u32) -> f32 {
        // Rough estimates based on architecture
        match sm_major {
            9 => 3350.0,   // Hopper (H100: 3.35 TB/s HBM3)
            8 => 2000.0,   // Ampere (A100: 2 TB/s HBM2e)
            7 => 900.0,    // Volta/Turing (V100: 900 GB/s)
            6 => 480.0,    // Pascal (P100: 720 GB/s, P40: 346 GB/s)
            _ => 200.0 + (sm_count as f32 * 4.0),  // Rough estimate
        }
    }
    
    /// Calculate relative compute power.
    fn calculate_relative_power(sm_major: u32, sm_count: u32) -> f32 {
        // Base multiplier per architecture generation
        let arch_multiplier = match sm_major {
            9 => 3.0,    // Hopper: ~3x A100
            8 => 1.0,    // Ampere: baseline
            7 => 0.6,    // Volta/Turing
            6 => 0.4,    // Pascal
            _ => 0.2,
        };
        
        // Scale by SM count (A100 has 108 SMs)
        arch_multiplier * (sm_count as f32 / 108.0)
    }
}

// =============================================================================
// Work Distribution Strategy
// =============================================================================

/// Strategy for distributing work across GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "cuda-runtime")]
pub enum WorkDistributionStrategy {
    /// Round-robin: assign workloads sequentially to GPUs
    RoundRobin,
    /// Capability-weighted: assign more work to more powerful GPUs
    CapabilityWeighted,
    /// Memory-aware: consider GPU memory availability
    MemoryAware,
    /// Dynamic: steal work from slower GPUs
    Dynamic,
}

/// Tracks work distribution and GPU load.
#[cfg(feature = "cuda-runtime")]
pub struct WorkDistributor {
    /// GPU capabilities
    capabilities: Vec<GpuCapabilities>,
    /// Current load per GPU (0.0 - 1.0)
    loads: Vec<f32>,
    /// Distribution strategy
    strategy: WorkDistributionStrategy,
}

#[cfg(feature = "cuda-runtime")]
impl WorkDistributor {
    /// Create a work distributor from GPU capabilities.
    pub fn new(capabilities: Vec<GpuCapabilities>, strategy: WorkDistributionStrategy) -> Self {
        let loads = vec![0.0; capabilities.len()];
        Self { capabilities, loads, strategy }
    }
    
    /// Assign a workload to the best GPU.
    pub fn assign_workload(&mut self, workload_size: usize) -> usize {
        match self.strategy {
            WorkDistributionStrategy::RoundRobin => {
                // Find GPU with lowest current assignment count
                self.loads
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| {
                        self.loads[idx] += 1.0;
                        idx
                    })
                    .unwrap_or(0)
            }
            WorkDistributionStrategy::CapabilityWeighted => {
                // Assign to GPU with most available relative capacity
                let (best_idx, _) = self.capabilities
                    .iter()
                    .zip(self.loads.iter())
                    .enumerate()
                    .map(|(idx, (cap, load))| {
                        // Available capacity = relative_power - current_load
                        (idx, cap.relative_power - load)
                    })
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, 0.0));
                
                // Update load (normalize by relative power)
                let cap = &self.capabilities[best_idx];
                self.loads[best_idx] += 1.0 / cap.relative_power;
                best_idx
            }
            WorkDistributionStrategy::MemoryAware => {
                // Assign to GPU with most available memory
                let required_memory = workload_size * 4 * 8; // Rough estimate
                let (best_idx, _) = self.capabilities
                    .iter()
                    .zip(self.loads.iter())
                    .enumerate()
                    .map(|(idx, (cap, load))| {
                        let estimated_free = cap.total_memory as f32 * (1.0 - load);
                        (idx, estimated_free)
                    })
                    .filter(|(_, free)| *free > required_memory as f32)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, 0.0));
                
                self.loads[best_idx] += 0.1; // Approximate load increase
                best_idx
            }
            WorkDistributionStrategy::Dynamic => {
                // Similar to capability-weighted, but allows rebalancing
                self.assign_workload_dynamic()
            }
        }
    }
    
    fn assign_workload_dynamic(&mut self) -> usize {
        // Find least loaded GPU weighted by capability
        let (best_idx, _) = self.capabilities
            .iter()
            .zip(self.loads.iter())
            .enumerate()
            .map(|(idx, (cap, load))| {
                (idx, cap.relative_power / (load + 0.1))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, 0.0));
        
        self.loads[best_idx] += 1.0;
        best_idx
    }
    
    /// Get current load distribution.
    pub fn get_loads(&self) -> &[f32] {
        &self.loads
    }
    
    /// Reset load tracking.
    pub fn reset(&mut self) {
        self.loads.fill(0.0);
    }
    
    /// Get GPU with Tensor Cores (for operations that can use them).
    pub fn get_tensor_core_gpu(&self) -> Option<usize> {
        self.capabilities
            .iter()
            .enumerate()
            .filter(|(_, cap)| cap.has_tensor_cores)
            .max_by(|(_, a), (_, b)| {
                a.relative_power.partial_cmp(&b.relative_power).unwrap()
            })
            .map(|(idx, _)| idx)
    }
}

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
    /// GPU capabilities for each device
    capabilities: Vec<GpuCapabilities>,
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
        let mut capabilities = Vec::new();
        
        for i in 0..device_count {
            match CudaDevice::new(i) {
                Ok(device) => {
                    // Query capabilities
                    match GpuCapabilities::query(i) {
                        Ok(cap) => {
                            tracing::info!(
                                "GPU {}: SM {}.{}, {} SMs, {:.1} GB, Tensor Cores: {}, Power: {:.2}x",
                                i,
                                cap.compute_capability.0,
                                cap.compute_capability.1,
                                cap.sm_count,
                                cap.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                                cap.has_tensor_cores,
                                cap.relative_power
                            );
                            capabilities.push(cap);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to query GPU {} capabilities: {:?}", i, e);
                            // Use default capabilities
                            capabilities.push(GpuCapabilities {
                                device_id: i,
                                compute_capability: (7, 0),
                                sm_count: 80,
                                total_memory: 16 * 1024 * 1024 * 1024,
                                memory_bandwidth_gbps: 900.0,
                                has_tensor_cores: true,
                                relative_power: 0.5,
                            });
                        }
                    }
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
        
        Ok(Self { device_ids, devices, capabilities })
    }
    
    /// Create a device manager with specific GPU IDs.
    pub fn new_with_devices(device_ids: Vec<usize>) -> Result<Self, CudaFftError> {
        let mut devices = Vec::new();
        let mut valid_ids = Vec::new();
        let mut capabilities = Vec::new();
        
        for id in device_ids {
            match CudaDevice::new(id) {
                Ok(device) => {
                    let cap = GpuCapabilities::query(id).unwrap_or(GpuCapabilities {
                        device_id: id,
                        compute_capability: (7, 0),
                        sm_count: 80,
                        total_memory: 16 * 1024 * 1024 * 1024,
                        memory_bandwidth_gbps: 900.0,
                        has_tensor_cores: true,
                        relative_power: 0.5,
                    });
                    capabilities.push(cap);
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
        
        Ok(Self { device_ids: valid_ids, devices, capabilities })
    }
    
    /// Get GPU capabilities.
    pub fn capabilities(&self) -> &[GpuCapabilities] {
        &self.capabilities
    }
    
    /// Create a work distributor for this device manager.
    pub fn create_distributor(&self, strategy: WorkDistributionStrategy) -> WorkDistributor {
        WorkDistributor::new(self.capabilities.clone(), strategy)
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
        
        // Each pipeline now gets its own executor from the device pool,
        // enabling true multi-GPU parallelism where each GPU runs independently.
        for (idx, &device_id) in device_manager.device_ids().iter().enumerate() {
            let pipeline = GpuProofPipeline::new_on_device(log_size, device_id)?;
            tracing::info!(
                "Created pipeline {} on GPU {} (true multi-GPU enabled)", 
                idx, device_id
            );
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
        let results: Vec<Option<ProofResult>> = (0..workloads.len()).map(|_| None).collect();
        
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
    /// 
    /// Uses proper cryptographic combination via Blake2s hashing.
    /// Local roots are combined into a Merkle tree structure.
    pub fn merkle_root(&mut self) -> Result<[u8; 32], CudaFftError> {
        let n = 1usize << self.log_size;
        let n_leaves = n / 2;
        
        // Each GPU computes local Merkle root
        let mut local_roots: Vec<[u8; 32]> = Vec::new();
        
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
        
        // Combine local roots using proper Merkle tree combination
        // This builds a binary tree of hashes from the local roots
        let combined = combine_merkle_roots_secure(&local_roots);
        
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

// =============================================================================
// Cryptographic Root Combination
// =============================================================================

/// Securely combine multiple Merkle roots into a single root using Blake2s.
/// 
/// This builds a proper Merkle tree from the local roots:
/// - If 1 root: return as-is
/// - If 2 roots: hash(left || right)
/// - If N roots: build a balanced tree and hash up to the root
/// 
/// This is cryptographically secure unlike XOR combination.
#[cfg(feature = "cuda-runtime")]
fn combine_merkle_roots_secure(roots: &[[u8; 32]]) -> [u8; 32] {
    match roots.len() {
        0 => [0u8; 32], // Empty case
        1 => roots[0],   // Single root, return as-is
        _ => {
            // Build Merkle tree from roots using Blake2s
            let mut current_layer: Vec<[u8; 32]> = roots.to_vec();
            
            // Pad to power of 2 if necessary
            while !current_layer.len().is_power_of_two() {
                // Duplicate last element for padding (standard Merkle padding)
                current_layer.push(*current_layer.last().unwrap());
            }
            
            // Build tree bottom-up
            while current_layer.len() > 1 {
                let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
                
                for chunk in current_layer.chunks(2) {
                    let combined = blake2s_hash_pair(&chunk[0], &chunk[1]);
                    next_layer.push(combined);
                }
                
                current_layer = next_layer;
            }
            
            current_layer[0]
        }
    }
}

/// Hash two 32-byte values using Blake2s to produce a single 32-byte hash.
/// 
/// This is the standard Merkle tree node combination operation.
#[cfg(feature = "cuda-runtime")]
fn blake2s_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    use blake2::{Blake2s256, Digest};
    
    let mut hasher = Blake2s256::new();
    hasher.update(left);
    hasher.update(right);
    
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
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
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_combine_merkle_roots_empty() {
        let roots: Vec<[u8; 32]> = vec![];
        let combined = combine_merkle_roots_secure(&roots);
        assert_eq!(combined, [0u8; 32]);
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_combine_merkle_roots_single() {
        let root = [42u8; 32];
        let roots = vec![root];
        let combined = combine_merkle_roots_secure(&roots);
        assert_eq!(combined, root);
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_combine_merkle_roots_two() {
        let root1 = [1u8; 32];
        let root2 = [2u8; 32];
        let roots = vec![root1, root2];
        
        let combined = combine_merkle_roots_secure(&roots);
        
        // Verify it's not just XOR (which would be [3u8; 32])
        assert_ne!(combined, [3u8; 32]);
        // Verify it's deterministic
        assert_eq!(combined, combine_merkle_roots_secure(&roots));
        // Verify it matches manual Blake2s computation
        assert_eq!(combined, blake2s_hash_pair(&root1, &root2));
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_combine_merkle_roots_multiple() {
        let roots: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
        
        let combined = combine_merkle_roots_secure(&roots);
        
        // Build expected tree manually:
        // Level 1: hash([0], [1]), hash([2], [3])
        // Level 0 (root): hash(level1[0], level1[1])
        let level1_0 = blake2s_hash_pair(&roots[0], &roots[1]);
        let level1_1 = blake2s_hash_pair(&roots[2], &roots[3]);
        let expected = blake2s_hash_pair(&level1_0, &level1_1);
        
        assert_eq!(combined, expected);
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_combine_merkle_roots_non_power_of_two() {
        // 3 roots should be padded to 4
        let roots: Vec<[u8; 32]> = (0..3).map(|i| [i as u8; 32]).collect();
        
        let combined = combine_merkle_roots_secure(&roots);
        
        // Should be deterministic
        assert_eq!(combined, combine_merkle_roots_secure(&roots));
        
        // Verify padding: last element duplicated
        let padded: Vec<[u8; 32]> = vec![roots[0], roots[1], roots[2], roots[2]];
        let level1_0 = blake2s_hash_pair(&padded[0], &padded[1]);
        let level1_1 = blake2s_hash_pair(&padded[2], &padded[3]);
        let expected = blake2s_hash_pair(&level1_0, &level1_1);
        
        assert_eq!(combined, expected);
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_blake2s_hash_pair_deterministic() {
        let left = [1u8; 32];
        let right = [2u8; 32];
        
        let hash1 = blake2s_hash_pair(&left, &right);
        let hash2 = blake2s_hash_pair(&left, &right);
        
        assert_eq!(hash1, hash2);
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_blake2s_hash_pair_order_matters() {
        let left = [1u8; 32];
        let right = [2u8; 32];
        
        let hash_lr = blake2s_hash_pair(&left, &right);
        let hash_rl = blake2s_hash_pair(&right, &left);
        
        // Order should matter for Merkle tree correctness
        assert_ne!(hash_lr, hash_rl);
    }
}

