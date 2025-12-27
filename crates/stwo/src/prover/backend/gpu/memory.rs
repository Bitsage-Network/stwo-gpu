//! GPU Memory Management for Stwo.
//!
//! This module provides type-safe GPU memory allocation and transfer operations.
//! It is the foundation for all GPU-accelerated operations in Stwo.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐        ┌─────────────────┐
//! │   CPU Memory    │ ←───→  │   GPU Memory    │
//! │  (BaseColumn)   │  H2D   │  (GpuBuffer)    │
//! │                 │  D2H   │                 │
//! └─────────────────┘        └─────────────────┘
//! ```
//!
//! # Design Principles
//!
//! 1. **Explicit transfers**: No implicit data movement
//! 2. **Type safety**: GpuBuffer<T> tracks element type
//! 3. **RAII**: GPU memory freed on drop
//! 4. **Error handling**: All operations return Result

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use super::cuda_executor::CudaFftError;

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::get_cuda_executor;

// =============================================================================
// GPU Buffer - Type-safe GPU Memory Handle
// =============================================================================

/// A buffer of data residing in GPU memory.
///
/// `GpuBuffer<T>` owns GPU memory and provides safe access patterns.
/// The memory is automatically freed when the buffer is dropped.
///
/// # Type Parameters
///
/// - `T`: The element type. Must implement `DeviceRepr` for CUDA compatibility.
///
/// # Example
///
/// ```ignore
/// let cpu_data = vec![1u32, 2, 3, 4];
/// let gpu_buffer = GpuBuffer::from_cpu(&cpu_data)?;
/// // ... perform GPU operations ...
/// let result = gpu_buffer.to_cpu()?;
/// ```
#[cfg(feature = "cuda-runtime")]
pub struct GpuBuffer<T: DeviceRepr> {
    /// The underlying CUDA device memory
    slice: CudaSlice<T>,
    /// Reference to the CUDA device (for operations)
    device: Arc<CudaDevice>,
    /// Number of elements
    len: usize,
}

#[cfg(feature = "cuda-runtime")]
impl<T: DeviceRepr + Clone + Default> GpuBuffer<T> {
    /// Allocate uninitialized GPU memory for `len` elements.
    ///
    /// # Safety
    ///
    /// The returned buffer contains uninitialized memory. Reading from it
    /// before writing is undefined behavior.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryAllocation` if allocation fails.
    pub fn alloc_uninit(len: usize) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();
        
        // Allocate device memory
        // Safety: We're allocating uninitialized memory that will be filled by the caller
        let slice = unsafe {
            device.alloc::<T>(len)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        Ok(Self { slice, device, len })
    }
    
    /// Allocate GPU memory and initialize with zeros.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryAllocation` if allocation fails.
    pub fn zeros(len: usize) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();
        
        // Allocate and zero-initialize
        let zeros: Vec<T> = vec![T::default(); len];
        let slice = device.htod_sync_copy(&zeros)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        Ok(Self { slice, device, len })
    }
    
    /// Create a GPU buffer from CPU data (Host-to-Device transfer).
    ///
    /// This copies the data from CPU memory to GPU memory.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryTransfer` if the transfer fails.
    pub fn from_cpu(data: &[T]) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();
        
        let slice = device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("H2D failed: {:?}", e)))?;
        
        Ok(Self {
            slice,
            device,
            len: data.len(),
        })
    }
    
    /// Copy GPU data back to CPU (Device-to-Host transfer).
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryTransfer` if the transfer fails.
    pub fn to_cpu(&self) -> Result<Vec<T>, CudaFftError> {
        let mut result = vec![T::default(); self.len];
        self.device.dtoh_sync_copy_into(&self.slice, &mut result)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("D2H failed: {:?}", e)))?;
        Ok(result)
    }
    
    /// Copy GPU data into an existing CPU buffer.
    ///
    /// # Panics
    ///
    /// Panics if `dst.len() != self.len()`.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryTransfer` if the transfer fails.
    pub fn to_cpu_into(&self, dst: &mut [T]) -> Result<(), CudaFftError> {
        assert_eq!(dst.len(), self.len, "Destination buffer size mismatch");
        self.device.dtoh_sync_copy_into(&self.slice, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("D2H failed: {:?}", e)))?;
        Ok(())
    }
    
    /// Get the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the underlying CUDA slice for kernel operations.
    ///
    /// # Safety
    ///
    /// The returned slice is valid only as long as this `GpuBuffer` exists.
    pub fn as_slice(&self) -> &CudaSlice<T> {
        &self.slice
    }
    
    /// Get a mutable reference to the underlying CUDA slice.
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
    
    /// Get the CUDA device handle.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

// GPU memory is automatically freed when GpuBuffer is dropped
// (CudaSlice handles this via cudarc)

// =============================================================================
// M31 Field GPU Operations
// =============================================================================

/// GPU buffer specifically for M31 field elements.
///
/// M31 elements are stored as `u32` on the GPU (the raw representation).
#[cfg(feature = "cuda-runtime")]
pub type GpuM31Buffer = GpuBuffer<u32>;

#[cfg(feature = "cuda-runtime")]
impl GpuM31Buffer {
    /// Create a GPU buffer from M31 field elements.
    pub fn from_m31(data: &[crate::core::fields::m31::M31]) -> Result<Self, CudaFftError> {
        // M31 is repr(transparent) over u32, so we can safely transmute
        let raw: &[u32] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u32,
                data.len()
            )
        };
        Self::from_cpu(raw)
    }
    
    /// Copy GPU data back as M31 field elements.
    pub fn to_m31(&self) -> Result<Vec<crate::core::fields::m31::M31>, CudaFftError> {
        let raw = self.to_cpu()?;
        // M31 is repr(transparent) over u32
        Ok(raw.into_iter().map(crate::core::fields::m31::M31).collect())
    }
}

// =============================================================================
// GPU Memory Pool (for reducing allocation overhead)
// =============================================================================

/// A pool of reusable GPU buffers to reduce allocation overhead.
///
/// Allocation on GPU is expensive (~100μs). For repeated operations,
/// reusing buffers from a pool can significantly improve performance.
#[cfg(feature = "cuda-runtime")]
pub struct GpuBufferPool {
    /// Available buffers, keyed by size
    available: std::collections::HashMap<usize, Vec<CudaSlice<u32>>>,
    /// The CUDA device
    device: Arc<CudaDevice>,
    /// Total allocated bytes (for monitoring)
    total_allocated: usize,
}

#[cfg(feature = "cuda-runtime")]
impl GpuBufferPool {
    /// Create a new buffer pool.
    pub fn new() -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        Ok(Self {
            available: std::collections::HashMap::new(),
            device: executor.device.clone(),
            total_allocated: 0,
        })
    }
    
    /// Get a buffer of at least `min_len` elements.
    ///
    /// This will reuse an existing buffer if one is available,
    /// otherwise it allocates a new one.
    pub fn get(&mut self, min_len: usize) -> Result<CudaSlice<u32>, CudaFftError> {
        // Round up to power of 2 for better reuse
        let size = min_len.next_power_of_two();
        
        if let Some(buffers) = self.available.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }
        
        // Allocate new buffer
        // Safety: We're allocating uninitialized memory for the pool
        let buffer = unsafe {
            self.device.alloc::<u32>(size)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        self.total_allocated += size * std::mem::size_of::<u32>();
        
        Ok(buffer)
    }
    
    /// Return a buffer to the pool for reuse.
    pub fn put(&mut self, buffer: CudaSlice<u32>, size: usize) {
        let size = size.next_power_of_two();
        self.available.entry(size).or_default().push(buffer);
    }
    
    /// Get total allocated memory in bytes.
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_allocated
    }
    
    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.available.clear();
    }
}

#[cfg(feature = "cuda-runtime")]
impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new().expect("Failed to create GPU buffer pool")
    }
}

// =============================================================================
// Stubs for non-CUDA builds
// =============================================================================

#[cfg(not(feature = "cuda-runtime"))]
pub struct GpuBuffer<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "cuda-runtime"))]
impl<T> GpuBuffer<T> {
    pub fn from_cpu(_data: &[T]) -> Result<Self, CudaFftError> {
        Err(CudaFftError::NoDevice)
    }
    
    pub fn to_cpu(&self) -> Result<Vec<T>, CudaFftError> {
        Err(CudaFftError::NoDevice)
    }
    
    pub fn len(&self) -> usize { 0 }
    pub fn is_empty(&self) -> bool { true }
}

#[cfg(not(feature = "cuda-runtime"))]
pub type GpuM31Buffer = GpuBuffer<u32>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::GpuBuffer;
    
    #[test]
    #[cfg(not(feature = "cuda-runtime"))]
    fn test_no_cuda_returns_error() {
        let result = GpuBuffer::<u32>::from_cpu(&[1, 2, 3]);
        assert!(result.is_err());
    }
}

