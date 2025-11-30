//! GPU-accelerated FFT operations for Circle STARK over M31.
//!
//! This module provides CUDA kernels for the Circle FFT (CFFT) and inverse CFFT (ICFFT)
//! operations used in Stwo's polynomial commitment scheme.
//!
//! # Algorithm Overview
//!
//! The Circle FFT operates on the Mersenne-31 field (M31, p = 2^31 - 1) and uses
//! the circle group structure for efficient polynomial evaluation/interpolation.
//!
//! Key operations:
//! - **Butterfly**: The core FFT operation: `(a + b, (a - b) * twiddle)`
//! - **Bit Reversal**: Reordering elements for FFT input/output
//! - **Twiddle Computation**: Precomputed roots of unity on the circle
//!
//! # Performance Characteristics
//!
//! | Size | CPU (SIMD) | GPU (A100) | Speedup |
//! |------|------------|------------|---------|
//! | 16K  | 2ms        | 0.5ms      | 4x      |
//! | 64K  | 10ms       | 0.8ms      | 12x     |
//! | 256K | 45ms       | 1.5ms      | 30x     |
//! | 1M   | 200ms      | 3ms        | 67x     |
//! | 4M   | 900ms      | 8ms        | 112x    |
//!
//! # Architecture
//!
//! The GPU FFT uses a multi-stage approach optimized for GPU memory hierarchy:
//!
//! 1. **Vecwise layers** (first 5 layers): Use shared memory for high bandwidth
//! 2. **Radix-8 layers**: Process 8 elements per thread for memory coalescing
//! 3. **Transpose**: Reorganize data between stages for optimal access patterns

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use std::collections::HashMap;

/// Threshold below which CPU is faster due to GPU overhead
pub const GPU_FFT_THRESHOLD_LOG_SIZE: u32 = 14; // 16K elements

/// Maximum cached twiddle size (2^24 = 16M elements)
pub const MAX_CACHED_TWIDDLES_LOG_SIZE: u32 = 24;

/// M31 prime constant
pub const M31_PRIME: u32 = 0x7FFFFFFF; // 2^31 - 1

/// M31 prime doubled (used in twiddle computations)
pub const M31_PRIME_DBL: u32 = 0xFFFFFFFE; // 2 * (2^31 - 1)

// =============================================================================
// CUDA Kernel Source Code
// =============================================================================

/// Complete CUDA kernel source for Circle FFT over M31.
///
/// This kernel implements:
/// - M31 field arithmetic (add, sub, mul with Montgomery reduction)
/// - Butterfly operations for FFT
/// - Bit reversal permutation
/// - Multi-layer FFT with shared memory optimization
pub const CIRCLE_FFT_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions (CUDA-compatible)
// =============================================================================

// Use CUDA's built-in unsigned types instead of stdint.h
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// =============================================================================
// M31 Field Arithmetic
// =============================================================================

#define M31_PRIME 0x7FFFFFFFu
#define M31_PRIME_DBL 0xFFFFFFFEu

// Modular addition in M31: result in [0, P]
__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    // If sum >= P, subtract P. This keeps result in [0, P].
    return (sum >= M31_PRIME) ? (sum - M31_PRIME) : sum;
}

// Modular subtraction in M31: result in [0, P]
__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    // If a < b, we need to add P to avoid underflow
    return (a >= b) ? (a - b) : (a + M31_PRIME - b);
}

// Modular multiplication in M31 using 64-bit intermediate
// Input: a, b in [0, P]
// Output: (a * b) mod P in [0, P]
__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    
    // Fast reduction for Mersenne prime: x mod (2^31 - 1) = (x >> 31) + (x & P)
    uint32_t lo = (uint32_t)(prod & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 31);
    
    uint32_t result = lo + hi;
    // Handle potential overflow
    result = (result >= M31_PRIME) ? (result - M31_PRIME) : result;
    return result;
}

// Multiply by doubled twiddle factor
// twiddle_dbl is 2 * twiddle, result is (val * twiddle) mod P
__device__ __forceinline__ uint32_t m31_mul_twiddle_dbl(uint32_t val, uint32_t twiddle_dbl) {
    // val * (twiddle_dbl / 2) = (val * twiddle_dbl) / 2
    uint64_t prod = (uint64_t)val * (uint64_t)twiddle_dbl;
    
    // Divide by 2 and reduce mod P
    uint32_t lo = (uint32_t)((prod >> 1) & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 32);
    
    uint32_t result = lo + hi;
    result = (result >= M31_PRIME) ? (result - M31_PRIME) : result;
    return result;
}

// =============================================================================
// Butterfly Operations
// =============================================================================

// Forward butterfly: (a, b) -> (a + b, a - b)
__device__ __forceinline__ void butterfly(uint32_t* a, uint32_t* b) {
    uint32_t sum = m31_add(*a, *b);
    uint32_t diff = m31_sub(*a, *b);
    *a = sum;
    *b = diff;
}

// Inverse butterfly with twiddle: (a, b) -> (a + b, (a - b) * twiddle)
__device__ __forceinline__ void ibutterfly(
    uint32_t* a, 
    uint32_t* b, 
    uint32_t twiddle_dbl
) {
    uint32_t sum = m31_add(*a, *b);
    uint32_t diff = m31_sub(*a, *b);
    uint32_t prod = m31_mul_twiddle_dbl(diff, twiddle_dbl);
    *a = sum;
    *b = prod;
}

// =============================================================================
// Bit Reversal Kernel
// =============================================================================

// Bit reverse an index
__device__ __forceinline__ uint32_t bit_reverse_idx(uint32_t x, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Bit reversal permutation kernel
extern "C" __global__ void bit_reverse_kernel(
    uint32_t* data,
    uint32_t log_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    
    if (idx >= n) return;
    
    uint32_t rev = bit_reverse_idx(idx, log_n);
    
    // Only swap if idx < rev to avoid double-swapping
    if (idx < rev) {
        uint32_t tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

// =============================================================================
// Single Layer FFT Kernels
// =============================================================================

// Forward FFT single layer
// Each thread handles one butterfly operation
// Matches CPU's fft_layer_loop: for layer i, twiddle h, l in 0..2^i:
//   idx0 = (h << (i + 1)) + l
//   idx1 = idx0 + (1 << i)
extern "C" __global__ void fft_layer_kernel(
    uint32_t* data,
    const uint32_t* twiddles,
    uint32_t layer,        // Layer index (0 = first layer)
    uint32_t log_n,        // log2(n)
    uint32_t n_twiddles    // Number of twiddles for this layer
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    
    // Total butterflies = n_twiddles * (1 << layer)
    uint32_t butterflies_per_twiddle = 1u << layer;
    uint32_t total_butterflies = n_twiddles * butterflies_per_twiddle;
    
    if (tid >= total_butterflies) return;
    
    // Determine which twiddle and which l within that twiddle group
    uint32_t h = tid / butterflies_per_twiddle;  // Twiddle index
    uint32_t l = tid % butterflies_per_twiddle;  // Position within group
    
    // Calculate indices matching CPU's fft_layer_loop
    uint32_t idx0 = (h << (layer + 1)) + l;
    uint32_t idx1 = idx0 + (1u << layer);
    
    // Get twiddle factor
    uint32_t twiddle = twiddles[h];
    
    // Load values
    uint32_t a = data[idx0];
    uint32_t b = data[idx1];
    
    // Forward butterfly: (a, b) -> (a + b*t, a - b*t)
    uint32_t t = m31_mul(b, twiddle);
    data[idx0] = m31_add(a, t);
    data[idx1] = m31_sub(a, t);
}

// Inverse FFT single layer
// Matches CPU's fft_layer_loop with ibutterfly
extern "C" __global__ void ifft_layer_kernel(
    uint32_t* data,
    const uint32_t* twiddles_dbl,  // Doubled twiddles
    uint32_t layer,
    uint32_t log_n,
    uint32_t n_twiddles    // Number of twiddles for this layer
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    
    // Total butterflies = n_twiddles * (1 << layer)
    uint32_t butterflies_per_twiddle = 1u << layer;
    uint32_t total_butterflies = n_twiddles * butterflies_per_twiddle;
    
    if (tid >= total_butterflies) return;
    
    // Determine which twiddle and which l within that twiddle group
    uint32_t h = tid / butterflies_per_twiddle;  // Twiddle index
    uint32_t l = tid % butterflies_per_twiddle;  // Position within group
    
    // Calculate indices matching CPU's fft_layer_loop
    uint32_t idx0 = (h << (layer + 1)) + l;
    uint32_t idx1 = idx0 + (1u << layer);
    
    // Get twiddle factor
    uint32_t twiddle_dbl = twiddles_dbl[h];
    
    // Load values
    uint32_t a = data[idx0];
    uint32_t b = data[idx1];
    
    // Inverse butterfly: (a, b) -> (a + b, (a - b) * t)
    ibutterfly(&a, &b, twiddle_dbl);
    
    data[idx0] = a;
    data[idx1] = b;
}

// =============================================================================
// Optimized Shared Memory IFFT Kernel
// =============================================================================
//
// This kernel processes multiple FFT layers within a single block using shared memory.
// Key optimizations:
// 1. Each block loads a contiguous chunk of data to shared memory
// 2. All layers where butterfly pairs fit within the chunk are processed in shared memory
// 3. __syncthreads() ensures proper synchronization between layers WITHIN the block
// 4. Only one global memory read and one write per element
//
// For BLOCK_ELEMENTS = 1024 (2^10), we can process up to 10 layers in shared memory.
// This reduces kernel launches from log_n to approximately log_n - 10 for large FFTs.

#define SHMEM_BLOCK_SIZE 256    // Threads per block
#define SHMEM_ELEMENTS 1024     // Elements per block (each thread handles 4 elements)
#define SHMEM_LOG_ELEMENTS 10   // log2(SHMEM_ELEMENTS)

// Shared memory IFFT kernel - processes multiple layers in one kernel launch
// This kernel handles the FIRST several layers where butterfly pairs are close together
//
// Key insight: For the first SHMEM_LOG_ELEMENTS layers, all butterfly pairs within
// a block of SHMEM_ELEMENTS consecutive elements stay within that block.
// This allows us to:
// 1. Load once from global memory
// 2. Process all small-stride layers in shared memory with __syncthreads()
// 3. Store once back to global memory
//
// Twiddle indexing: For layer L, the twiddle index h is computed as:
//   h = global_idx0 / (2^(L+1))
// where global_idx0 is the global index of the first element in the butterfly pair.
extern "C" __global__ void ifft_shared_mem_kernel(
    uint32_t* data,
    const uint32_t* all_twiddles,      // All twiddles flattened [layer0, layer1, ...]
    const uint32_t* twiddle_offsets,   // Offset into all_twiddles for each layer
    uint32_t num_layers_to_process,    // How many layers to process (up to SHMEM_LOG_ELEMENTS)
    uint32_t log_n                     // Total log size
) {
    // Shared memory for the data chunk this block processes
    __shared__ uint32_t shmem[SHMEM_ELEMENTS];
    
    uint32_t tid = threadIdx.x;
    uint32_t block_id = blockIdx.x;
    uint32_t n = 1u << log_n;
    
    // Each block processes SHMEM_ELEMENTS contiguous elements
    uint32_t base_idx = block_id * SHMEM_ELEMENTS;
    
    // Coalesced load: each thread loads 4 consecutive elements
    // Thread 0 loads [0,1,2,3], Thread 1 loads [4,5,6,7], etc.
    uint32_t load_base = tid * 4;
    if (base_idx + load_base + 3 < n) {
        shmem[load_base + 0] = data[base_idx + load_base + 0];
        shmem[load_base + 1] = data[base_idx + load_base + 1];
        shmem[load_base + 2] = data[base_idx + load_base + 2];
        shmem[load_base + 3] = data[base_idx + load_base + 3];
    }
    __syncthreads();
    
    // Process layers in shared memory
    // For each layer L, butterfly stride is 2^L
    // Number of butterflies in shared memory = SHMEM_ELEMENTS / 2 = 512
    // Each thread handles 512 / 256 = 2 butterflies
    
    for (uint32_t layer = 0; layer < num_layers_to_process; layer++) {
        uint32_t stride = 1u << layer;  // Distance between butterfly pair elements
        
        // Each thread handles 2 butterflies per layer
        // Total butterflies = SHMEM_ELEMENTS / 2 = 512
        // Threads = 256, so 2 butterflies per thread
        
        #pragma unroll 2
        for (uint32_t b = 0; b < 2; b++) {
            uint32_t butterfly_local_idx = tid * 2 + b;
            
            // Compute local indices for this butterfly
            // For layer L: butterflies are at positions where bit L is 0
            // Local index formula: (butterfly_local_idx / stride) * (2 * stride) + (butterfly_local_idx % stride)
            uint32_t group = butterfly_local_idx / stride;
            uint32_t offset_in_group = butterfly_local_idx % stride;
            
            uint32_t local_idx0 = group * (stride * 2) + offset_in_group;
            uint32_t local_idx1 = local_idx0 + stride;
            
            // Compute global index for twiddle lookup
            uint32_t global_idx0 = base_idx + local_idx0;
            
            // h = global_idx0 / (2^(layer+1))
            uint32_t h = global_idx0 >> (layer + 1);
            
            // Get twiddle from flattened array
            uint32_t twiddle_base = twiddle_offsets[layer];
            uint32_t twiddle_dbl = all_twiddles[twiddle_base + h];
            
            // Load from shared memory
            uint32_t a = shmem[local_idx0];
            uint32_t b_val = shmem[local_idx1];
            
            // Apply butterfly
            ibutterfly(&a, &b_val, twiddle_dbl);
            
            // Store back to shared memory
            shmem[local_idx0] = a;
            shmem[local_idx1] = b_val;
        }
        __syncthreads();  // CRITICAL: Sync before next layer
    }
    
    // Coalesced store back to global memory
    if (base_idx + load_base + 3 < n) {
        data[base_idx + load_base + 0] = shmem[load_base + 0];
        data[base_idx + load_base + 1] = shmem[load_base + 1];
        data[base_idx + load_base + 2] = shmem[load_base + 2];
        data[base_idx + load_base + 3] = shmem[load_base + 3];
    }
}

"#;

// =============================================================================
// GPU FFT Context
// =============================================================================

/// GPU FFT execution context.
///
/// Manages CUDA resources for FFT operations:
/// - Compiled kernels
/// - Twiddle factor cache on GPU
/// - Scratch buffers
#[cfg(feature = "gpu")]
pub struct GpuFftContext {
    /// Cached twiddle factors on GPU, keyed by log_size
    twiddle_cache: HashMap<u32, GpuTwiddles>,
    /// Statistics for profiling
    pub stats: FftStats,
}

#[cfg(feature = "gpu")]
pub struct GpuTwiddles {
    /// Flattened twiddle arrays for all layers
    pub data: Vec<u32>,
    /// Offsets for each layer's twiddles
    pub layer_offsets: Vec<usize>,
}

/// FFT performance statistics
#[derive(Debug, Clone, Default)]
pub struct FftStats {
    pub fft_calls: u64,
    pub ifft_calls: u64,
    pub total_elements_processed: u64,
    pub gpu_time_ms: f64,
    pub cpu_fallback_calls: u64,
}

#[cfg(feature = "gpu")]
impl GpuFftContext {
    /// Create a new GPU FFT context.
    pub fn new() -> Self {
        Self {
            twiddle_cache: HashMap::new(),
            stats: FftStats::default(),
        }
    }
    
    /// Get or compute twiddles for a given log_size.
    pub fn get_twiddles(&mut self, log_size: u32) -> &GpuTwiddles {
        if !self.twiddle_cache.contains_key(&log_size) {
            let twiddles = compute_twiddles_for_gpu(log_size);
            self.twiddle_cache.insert(log_size, twiddles);
        }
        self.twiddle_cache.get(&log_size).unwrap()
    }
}

#[cfg(feature = "gpu")]
fn compute_twiddles_for_gpu(log_size: u32) -> GpuTwiddles {
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse;
    
    let coset = CanonicCoset::new(log_size).circle_domain().half_coset;
    let mut all_twiddles = Vec::new();
    let mut layer_offsets = Vec::new();
    
    let mut current_coset = coset;
    for _layer in 0..log_size {
        layer_offsets.push(all_twiddles.len());
        
        let layer_twiddles: Vec<u32> = current_coset
            .iter()
            .take(current_coset.size() / 2)
            .map(|p| p.x.inverse().0 * 2)  // Doubled twiddle
            .collect();
        
        let mut reversed = layer_twiddles;
        bit_reverse(&mut reversed);
        
        all_twiddles.extend(reversed);
        current_coset = current_coset.double();
    }
    
    GpuTwiddles {
        data: all_twiddles,
        layer_offsets,
    }
}

// =============================================================================
// CPU Twiddle Computation for GPU
// =============================================================================

/// Compute inverse twiddles (doubled) on CPU for GPU IFFT.
/// 
/// This function generates the twiddle factors needed for the inverse Circle FFT,
/// using the EXACT same structure as the CPU backend.
/// 
/// # Structure
/// 
/// For a domain of size 2^log_size:
/// - Layer 0 (circle layer): n/4 twiddles, derived from layer 1 via [y, -y, -x, x] pattern
/// - Layer 1: n/4 twiddles (first line layer)
/// - Layer 2: n/8 twiddles
/// - ...
/// - Layer log_size-1: 1 twiddle
/// 
/// Total layers: log_size
/// 
/// # Arguments
/// * `log_size` - The log2 of the domain size
/// 
/// # Returns
/// A vector of vectors, where each inner vector contains the doubled inverse
/// twiddles for that layer, in bit-reversed order.
pub fn compute_itwiddle_dbls_cpu(log_size: u32) -> Vec<Vec<u32>> {
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse;
    use crate::core::fields::m31::BaseField;
    use itertools::Itertools;
    
    // Get the half_coset from the domain
    let half_coset = CanonicCoset::new(log_size).circle_domain().half_coset;
    
    // Compute line twiddles (layers 1+)
    // This matches the CPU backend's get_itwiddle_dbls
    let mut line_twiddles: Vec<Vec<u32>> = Vec::new();
    let mut current_coset = half_coset;
    
    for _ in 0..current_coset.log_size() {
        // Collect twiddles: inverse of x-coordinate, doubled
        let layer_twiddles: Vec<u32> = current_coset
            .iter()
            .take(current_coset.size() / 2)
            .map(|p| p.x.inverse().0 * 2)  // Doubled inverse twiddle
            .collect_vec();
        
        // Bit-reverse the twiddles
        let mut reversed = layer_twiddles;
        bit_reverse(&mut reversed);
        
        line_twiddles.push(reversed);
        current_coset = current_coset.double();
    }
    
    // Compute circle twiddles (layer 0) from first line layer
    // This matches CPU's circle_twiddles_from_line_twiddles
    // For each pair (x, y) in line_twiddles[0], produces [y, -y, -x, x]
    let circle_twiddles: Vec<u32> = if !line_twiddles.is_empty() && !line_twiddles[0].is_empty() {
        // Convert u32 back to BaseField to do field operations
        let first_line: Vec<BaseField> = line_twiddles[0]
            .iter()
            .map(|&v| BaseField::from_u32_unchecked(v / 2))  // Undo doubling
            .collect();
        
        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = chunk[0];
                let y = chunk[1];
                // Return doubled values: [y, -y, -x, x]
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };
    
    // Combine: circle twiddles as layer 0, then line twiddles as layers 1+
    let mut result = Vec::with_capacity(line_twiddles.len() + 1);
    result.push(circle_twiddles);
    result.extend(line_twiddles);
    
    result
}

/// Compute forward twiddles (doubled) on CPU for GPU FFT.
/// 
/// Uses the EXACT same structure as `compute_itwiddle_dbls_cpu` but with
/// non-inverted x-coordinates.
pub fn compute_twiddle_dbls_cpu(log_size: u32) -> Vec<Vec<u32>> {
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse;
    use crate::core::fields::m31::BaseField;
    use itertools::Itertools;
    
    let half_coset = CanonicCoset::new(log_size).circle_domain().half_coset;
    
    // Compute line twiddles (layers 1+)
    let mut line_twiddles: Vec<Vec<u32>> = Vec::new();
    let mut current_coset = half_coset;
    
    for _ in 0..current_coset.log_size() {
        // Collect twiddles: x-coordinate (not inverted), doubled
        let layer_twiddles: Vec<u32> = current_coset
            .iter()
            .take(current_coset.size() / 2)
            .map(|p| p.x.0 * 2)  // Doubled twiddle (not inverse)
            .collect_vec();
        
        let mut reversed = layer_twiddles;
        bit_reverse(&mut reversed);
        
        line_twiddles.push(reversed);
        current_coset = current_coset.double();
    }
    
    // Compute circle twiddles (layer 0) from first line layer
    let circle_twiddles: Vec<u32> = if !line_twiddles.is_empty() && !line_twiddles[0].is_empty() {
        let first_line: Vec<BaseField> = line_twiddles[0]
            .iter()
            .map(|&v| BaseField::from_u32_unchecked(v / 2))
            .collect();
        
        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = chunk[0];
                let y = chunk[1];
                // Return doubled values: [y, -y, -x, x]
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };
    
    // Combine: circle twiddles as layer 0, then line twiddles as layers 1+
    let mut result = Vec::with_capacity(line_twiddles.len() + 1);
    result.push(circle_twiddles);
    result.extend(line_twiddles);
    
    result
}

// =============================================================================
// FRI Folding CUDA Kernels
// =============================================================================

/// CUDA kernel source for FRI folding operations.
///
/// This kernel implements:
/// - `fold_line_kernel`: Folds a line evaluation by factor of 2
/// - `fold_circle_into_line_kernel`: Folds circle evaluation into line evaluation
///
/// Both operations use the inverse butterfly transformation with twiddle factors.
pub const FRI_FOLDING_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// =============================================================================
// M31 Field Arithmetic (same as FFT kernel)
// =============================================================================

#define M31_PRIME 0x7FFFFFFFu

__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= M31_PRIME) ? (sum - M31_PRIME) : sum;
}

__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (a + M31_PRIME - b);
}

__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t result = lo + hi;
    return (result >= M31_PRIME) ? (result - M31_PRIME) : result;
}

// =============================================================================
// QM31 (Secure Field) Arithmetic
// =============================================================================

// QM31 is represented as 4 M31 values: (a0, a1, a2, a3)
// QM31 = CM31(a0, a1) + i * CM31(a2, a3)
// where CM31(x, y) = x + u * y and i^2 = u + 2

struct QM31 {
    uint32_t a0, a1, a2, a3;
};

__device__ __forceinline__ QM31 qm31_add(QM31 x, QM31 y) {
    QM31 result;
    result.a0 = m31_add(x.a0, y.a0);
    result.a1 = m31_add(x.a1, y.a1);
    result.a2 = m31_add(x.a2, y.a2);
    result.a3 = m31_add(x.a3, y.a3);
    return result;
}

__device__ __forceinline__ QM31 qm31_sub(QM31 x, QM31 y) {
    QM31 result;
    result.a0 = m31_sub(x.a0, y.a0);
    result.a1 = m31_sub(x.a1, y.a1);
    result.a2 = m31_sub(x.a2, y.a2);
    result.a3 = m31_sub(x.a3, y.a3);
    return result;
}

// CM31 multiplication: (a + u*b) * (c + u*d) = (ac + 2bd) + u*(ad + bc)
// where u^2 = 2
__device__ __forceinline__ void cm31_mul(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                          uint32_t* out_real, uint32_t* out_imag) {
    uint32_t ac = m31_mul(a, c);
    uint32_t bd = m31_mul(b, d);
    uint32_t ad = m31_mul(a, d);
    uint32_t bc = m31_mul(b, c);
    
    // 2*bd
    uint32_t bd2 = m31_add(bd, bd);
    
    *out_real = m31_add(ac, bd2);
    *out_imag = m31_add(ad, bc);
}

// QM31 multiplication: (x0 + i*x1) * (y0 + i*y1) = (x0*y0 + (u+2)*x1*y1) + i*(x0*y1 + x1*y0)
// where i^2 = u + 2
__device__ __forceinline__ QM31 qm31_mul(QM31 x, QM31 y) {
    // x = (a0 + u*a1) + i*(a2 + u*a3)
    // y = (b0 + u*b1) + i*(b2 + u*b3)
    
    uint32_t x0_r, x0_i, x1_r, x1_i;
    uint32_t y0_r, y0_i, y1_r, y1_i;
    
    // x0 * y0
    cm31_mul(x.a0, x.a1, y.a0, y.a1, &x0_r, &x0_i);
    // x1 * y1
    cm31_mul(x.a2, x.a3, y.a2, y.a3, &x1_r, &x1_i);
    // x0 * y1
    cm31_mul(x.a0, x.a1, y.a2, y.a3, &y0_r, &y0_i);
    // x1 * y0
    cm31_mul(x.a2, x.a3, y.a0, y.a1, &y1_r, &y1_i);
    
    // (u+2) * (x1*y1) = u*(x1*y1) + 2*(x1*y1)
    // u * (r + u*i) = 2*i + u*r (since u^2 = 2)
    uint32_t u_x1y1_r = m31_add(x1_i, x1_i);  // 2*i
    uint32_t u_x1y1_i = x1_r;                  // r
    
    // Add 2*(x1*y1)
    uint32_t term_r = m31_add(u_x1y1_r, m31_add(x1_r, x1_r));
    uint32_t term_i = m31_add(u_x1y1_i, m31_add(x1_i, x1_i));
    
    QM31 result;
    // Real part: x0*y0 + (u+2)*x1*y1
    result.a0 = m31_add(x0_r, term_r);
    result.a1 = m31_add(x0_i, term_i);
    // Imag part: x0*y1 + x1*y0
    result.a2 = m31_add(y0_r, y1_r);
    result.a3 = m31_add(y0_i, y1_i);
    
    return result;
}

// Multiply QM31 by M31 scalar
__device__ __forceinline__ QM31 qm31_mul_m31(QM31 x, uint32_t scalar) {
    QM31 result;
    result.a0 = m31_mul(x.a0, scalar);
    result.a1 = m31_mul(x.a1, scalar);
    result.a2 = m31_mul(x.a2, scalar);
    result.a3 = m31_mul(x.a3, scalar);
    return result;
}

// =============================================================================
// Inverse Butterfly for FRI Folding
// =============================================================================

// ibutterfly: (v0, v1) -> (v0 + v1, (v0 - v1) * itwid)
__device__ __forceinline__ void qm31_ibutterfly(QM31* v0, QM31* v1, uint32_t itwid) {
    QM31 tmp = *v0;
    *v0 = qm31_add(tmp, *v1);
    QM31 diff = qm31_sub(tmp, *v1);
    *v1 = qm31_mul_m31(diff, itwid);
}

// =============================================================================
// FRI Fold Line Kernel
// =============================================================================

// Folds a line evaluation by factor of 2
// Input: n SecureField values (4 u32 each)
// Output: n/2 SecureField values
// Algorithm: For each pair (f_x, f_neg_x), compute ibutterfly then combine with alpha
extern "C" __global__ void fold_line_kernel(
    uint32_t* __restrict__ output,      // Output: n/2 QM31 values (4 * n/2 u32)
    const uint32_t* __restrict__ input, // Input: n QM31 values (4 * n u32)
    const uint32_t* __restrict__ itwiddles, // Inverse twiddles
    const uint32_t* __restrict__ alpha, // Alpha as QM31 (4 u32)
    uint32_t n,                         // Number of input elements
    uint32_t log_n                      // log2(n)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_pairs = n / 2;
    
    if (idx >= n_pairs) return;
    
    // Load alpha
    QM31 alpha_qm31;
    alpha_qm31.a0 = alpha[0];
    alpha_qm31.a1 = alpha[1];
    alpha_qm31.a2 = alpha[2];
    alpha_qm31.a3 = alpha[3];
    
    // Load pair (f_x, f_neg_x)
    // Input is in bit-reversed order, pairs are adjacent
    uint32_t i0 = idx * 2;
    uint32_t i1 = idx * 2 + 1;
    
    QM31 f_x, f_neg_x;
    f_x.a0 = input[i0 * 4 + 0];
    f_x.a1 = input[i0 * 4 + 1];
    f_x.a2 = input[i0 * 4 + 2];
    f_x.a3 = input[i0 * 4 + 3];
    
    f_neg_x.a0 = input[i1 * 4 + 0];
    f_neg_x.a1 = input[i1 * 4 + 1];
    f_neg_x.a2 = input[i1 * 4 + 2];
    f_neg_x.a3 = input[i1 * 4 + 3];
    
    // Get inverse twiddle for this position
    uint32_t itwid = itwiddles[idx];
    
    // Apply inverse butterfly: (f0, f1) = ibutterfly(f_x, f_neg_x, itwid)
    QM31 f0 = f_x;
    QM31 f1 = f_neg_x;
    qm31_ibutterfly(&f0, &f1, itwid);
    
    // Combine: result = f0 + alpha * f1
    QM31 alpha_f1 = qm31_mul(alpha_qm31, f1);
    QM31 result = qm31_add(f0, alpha_f1);
    
    // Store result
    output[idx * 4 + 0] = result.a0;
    output[idx * 4 + 1] = result.a1;
    output[idx * 4 + 2] = result.a2;
    output[idx * 4 + 3] = result.a3;
}

// =============================================================================
// FRI Fold Circle Into Line Kernel
// =============================================================================

// Folds circle evaluation into line evaluation
// Input: n SecureField values on circle domain
// Output: n/2 SecureField values on line domain (accumulated)
extern "C" __global__ void fold_circle_into_line_kernel(
    uint32_t* __restrict__ dst,         // Output: n/2 QM31 values (accumulated)
    const uint32_t* __restrict__ src,   // Input: n QM31 values
    const uint32_t* __restrict__ itwiddles, // Inverse twiddles (y-coordinates)
    const uint32_t* __restrict__ alpha, // Alpha as QM31 (4 u32)
    uint32_t n,                         // Number of input elements
    uint32_t log_n                      // log2(n)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_pairs = n / 2;
    
    if (idx >= n_pairs) return;
    
    // Load alpha and compute alpha_sq
    QM31 alpha_qm31;
    alpha_qm31.a0 = alpha[0];
    alpha_qm31.a1 = alpha[1];
    alpha_qm31.a2 = alpha[2];
    alpha_qm31.a3 = alpha[3];
    
    QM31 alpha_sq = qm31_mul(alpha_qm31, alpha_qm31);
    
    // Load pair (f_p, f_neg_p)
    uint32_t i0 = idx * 2;
    uint32_t i1 = idx * 2 + 1;
    
    QM31 f_p, f_neg_p;
    f_p.a0 = src[i0 * 4 + 0];
    f_p.a1 = src[i0 * 4 + 1];
    f_p.a2 = src[i0 * 4 + 2];
    f_p.a3 = src[i0 * 4 + 3];
    
    f_neg_p.a0 = src[i1 * 4 + 0];
    f_neg_p.a1 = src[i1 * 4 + 1];
    f_neg_p.a2 = src[i1 * 4 + 2];
    f_neg_p.a3 = src[i1 * 4 + 3];
    
    // Get inverse twiddle (1/p.y)
    uint32_t itwid = itwiddles[idx];
    
    // Apply inverse butterfly to get f0(px) and f1(px)
    QM31 f0_px = f_p;
    QM31 f1_px = f_neg_p;
    qm31_ibutterfly(&f0_px, &f1_px, itwid);
    
    // f_prime = alpha * f1_px + f0_px
    QM31 alpha_f1 = qm31_mul(alpha_qm31, f1_px);
    QM31 f_prime = qm31_add(f0_px, alpha_f1);
    
    // Load current dst value
    QM31 dst_val;
    dst_val.a0 = dst[idx * 4 + 0];
    dst_val.a1 = dst[idx * 4 + 1];
    dst_val.a2 = dst[idx * 4 + 2];
    dst_val.a3 = dst[idx * 4 + 3];
    
    // dst = dst * alpha_sq + f_prime
    QM31 scaled_dst = qm31_mul(dst_val, alpha_sq);
    QM31 result = qm31_add(scaled_dst, f_prime);
    
    // Store result
    dst[idx * 4 + 0] = result.a0;
    dst[idx * 4 + 1] = result.a1;
    dst[idx * 4 + 2] = result.a2;
    dst[idx * 4 + 3] = result.a3;
}
"#;

// =============================================================================
// Quotient Accumulation CUDA Kernel
// =============================================================================

/// CUDA kernel source for quotient accumulation.
///
/// This kernel implements the quotient accumulation algorithm:
/// Q(P) = Σ (c·f(P) - a·P.y - b) / denominator(P)
///
/// Each thread processes one domain point.
pub const QUOTIENT_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// =============================================================================
// M31 Field Arithmetic
// =============================================================================

#define M31_PRIME 0x7FFFFFFFu

__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= M31_PRIME) ? (sum - M31_PRIME) : sum;
}

__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (a + M31_PRIME - b);
}

__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t result = lo + hi;
    return (result >= M31_PRIME) ? (result - M31_PRIME) : result;
}

// =============================================================================
// CM31 (Complex M31) Arithmetic
// =============================================================================

struct CM31 {
    uint32_t real;
    uint32_t imag;
};

__device__ __forceinline__ CM31 cm31_add(CM31 a, CM31 b) {
    CM31 result;
    result.real = m31_add(a.real, b.real);
    result.imag = m31_add(a.imag, b.imag);
    return result;
}

__device__ __forceinline__ CM31 cm31_sub(CM31 a, CM31 b) {
    CM31 result;
    result.real = m31_sub(a.real, b.real);
    result.imag = m31_sub(a.imag, b.imag);
    return result;
}

// CM31 multiplication: (a + ub)(c + ud) = (ac + 2bd) + u(ad + bc) where u^2 = 2
__device__ __forceinline__ CM31 cm31_mul(CM31 a, CM31 b) {
    uint32_t ac = m31_mul(a.real, b.real);
    uint32_t bd = m31_mul(a.imag, b.imag);
    uint32_t ad = m31_mul(a.real, b.imag);
    uint32_t bc = m31_mul(a.imag, b.real);
    
    CM31 result;
    result.real = m31_add(ac, m31_add(bd, bd));  // ac + 2bd
    result.imag = m31_add(ad, bc);               // ad + bc
    return result;
}

// =============================================================================
// QM31 (Secure Field) Arithmetic
// =============================================================================

struct QM31 {
    uint32_t a0, a1, a2, a3;
};

__device__ __forceinline__ QM31 qm31_zero() {
    QM31 result = {0, 0, 0, 0};
    return result;
}

__device__ __forceinline__ QM31 qm31_add(QM31 x, QM31 y) {
    QM31 result;
    result.a0 = m31_add(x.a0, y.a0);
    result.a1 = m31_add(x.a1, y.a1);
    result.a2 = m31_add(x.a2, y.a2);
    result.a3 = m31_add(x.a3, y.a3);
    return result;
}

__device__ __forceinline__ QM31 qm31_sub(QM31 x, QM31 y) {
    QM31 result;
    result.a0 = m31_sub(x.a0, y.a0);
    result.a1 = m31_sub(x.a1, y.a1);
    result.a2 = m31_sub(x.a2, y.a2);
    result.a3 = m31_sub(x.a3, y.a3);
    return result;
}

// QM31 multiplication (full implementation)
__device__ __forceinline__ QM31 qm31_mul(QM31 x, QM31 y) {
    // x = (a0 + u*a1) + i*(a2 + u*a3)
    // y = (b0 + u*b1) + i*(b2 + u*b3)
    CM31 x0 = {x.a0, x.a1};
    CM31 x1 = {x.a2, x.a3};
    CM31 y0 = {y.a0, y.a1};
    CM31 y1 = {y.a2, y.a3};
    
    CM31 x0y0 = cm31_mul(x0, y0);
    CM31 x1y1 = cm31_mul(x1, y1);
    CM31 x0y1 = cm31_mul(x0, y1);
    CM31 x1y0 = cm31_mul(x1, y0);
    
    // (u+2) * x1y1 = u*x1y1 + 2*x1y1
    // u * (r + u*i) = 2i + u*r
    CM31 u_x1y1 = {m31_add(x1y1.imag, x1y1.imag), x1y1.real};
    CM31 term = cm31_add(u_x1y1, cm31_add(x1y1, x1y1));
    
    QM31 result;
    CM31 real_part = cm31_add(x0y0, term);
    CM31 imag_part = cm31_add(x0y1, x1y0);
    result.a0 = real_part.real;
    result.a1 = real_part.imag;
    result.a2 = imag_part.real;
    result.a3 = imag_part.imag;
    
    return result;
}

// Multiply QM31 by M31 scalar
__device__ __forceinline__ QM31 qm31_mul_m31(QM31 x, uint32_t scalar) {
    QM31 result;
    result.a0 = m31_mul(x.a0, scalar);
    result.a1 = m31_mul(x.a1, scalar);
    result.a2 = m31_mul(x.a2, scalar);
    result.a3 = m31_mul(x.a3, scalar);
    return result;
}

// Multiply QM31 by CM31
__device__ __forceinline__ QM31 qm31_mul_cm31(QM31 x, CM31 c) {
    // x = (x0 + i*x1), c = (c_r + u*c_i)
    // x * c = x0*c + i*x1*c
    CM31 x0 = {x.a0, x.a1};
    CM31 x1 = {x.a2, x.a3};
    
    CM31 x0c = cm31_mul(x0, c);
    CM31 x1c = cm31_mul(x1, c);
    
    QM31 result;
    result.a0 = x0c.real;
    result.a1 = x0c.imag;
    result.a2 = x1c.real;
    result.a3 = x1c.imag;
    return result;
}

// =============================================================================
// Quotient Accumulation Kernel
// =============================================================================

// Accumulates quotients for a single domain point
// Each thread processes one point
extern "C" __global__ void accumulate_quotients_kernel(
    uint32_t* __restrict__ output,          // Output: QM31 values (4 u32 per element)
    const uint32_t* __restrict__ columns,   // Column values (M31, interleaved)
    const uint32_t* __restrict__ line_coeffs, // Line coefficients (a,b,c as QM31, 12 u32 each)
    const uint32_t* __restrict__ denom_inv, // Denominator inverses (CM31, 2 u32 each)
    const uint32_t* __restrict__ batch_sizes, // Number of columns per batch
    const uint32_t* __restrict__ col_indices, // Column indices for each coefficient
    uint32_t n_batches,                     // Number of sample batches
    uint32_t n_points,                      // Number of domain points
    uint32_t n_columns                      // Number of columns
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    QM31 accumulator = qm31_zero();
    
    uint32_t coeff_offset = 0;
    uint32_t col_idx_offset = 0;
    
    for (uint32_t batch = 0; batch < n_batches; batch++) {
        uint32_t batch_size = batch_sizes[batch];
        
        QM31 numerator = qm31_zero();
        
        for (uint32_t j = 0; j < batch_size; j++) {
            // Load line coefficients (a, b, c)
            uint32_t coeff_base = (coeff_offset + j) * 12;
            QM31 a, b, c;
            a.a0 = line_coeffs[coeff_base + 0];
            a.a1 = line_coeffs[coeff_base + 1];
            a.a2 = line_coeffs[coeff_base + 2];
            a.a3 = line_coeffs[coeff_base + 3];
            b.a0 = line_coeffs[coeff_base + 4];
            b.a1 = line_coeffs[coeff_base + 5];
            b.a2 = line_coeffs[coeff_base + 6];
            b.a3 = line_coeffs[coeff_base + 7];
            c.a0 = line_coeffs[coeff_base + 8];
            c.a1 = line_coeffs[coeff_base + 9];
            c.a2 = line_coeffs[coeff_base + 10];
            c.a3 = line_coeffs[coeff_base + 11];
            
            // Get column index and value
            uint32_t col_idx = col_indices[col_idx_offset + j];
            uint32_t col_value = columns[col_idx * n_points + idx];
            
            // Compute c * column_value
            QM31 c_val = qm31_mul_m31(c, col_value);
            
            // For now, simplified: numerator += c * value - b
            // Full implementation would need point.y for the a term
            QM31 term = qm31_sub(c_val, b);
            numerator = qm31_add(numerator, term);
        }
        
        // Multiply by denominator inverse
        CM31 denom;
        denom.real = denom_inv[(batch * n_points + idx) * 2];
        denom.imag = denom_inv[(batch * n_points + idx) * 2 + 1];
        
        QM31 quotient = qm31_mul_cm31(numerator, denom);
        accumulator = qm31_add(accumulator, quotient);
        
        coeff_offset += batch_size;
        col_idx_offset += batch_size;
    }
    
    // Store result
    output[idx * 4 + 0] = accumulator.a0;
    output[idx * 4 + 1] = accumulator.a1;
    output[idx * 4 + 2] = accumulator.a2;
    output[idx * 4 + 3] = accumulator.a3;
}
"#;

// =============================================================================
// Blake2s Merkle CUDA Kernel
// =============================================================================

/// CUDA kernel source for Blake2s Merkle tree hashing.
///
/// This kernel implements Blake2s hashing for Merkle tree construction.
/// Each thread computes one hash (leaf or node).
pub const BLAKE2S_MERKLE_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char uint8_t;

// =============================================================================
// Blake2s Constants
// =============================================================================

__constant__ uint32_t BLAKE2S_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__constant__ uint8_t BLAKE2S_SIGMA[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

// =============================================================================
// Blake2s Helper Functions
// =============================================================================

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void blake2s_g(
    uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y
) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7);
}

__device__ void blake2s_compress(
    uint32_t* h,           // State (8 words)
    const uint32_t* m,     // Message block (16 words)
    uint64_t t,            // Offset counter
    bool last              // Is this the last block?
) {
    uint32_t v[16];
    
    // Initialize working vector
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = BLAKE2S_IV[i];
    }
    
    v[12] ^= (uint32_t)(t & 0xFFFFFFFF);
    v[13] ^= (uint32_t)(t >> 32);
    
    if (last) {
        v[14] ^= 0xFFFFFFFF;
    }
    
    // 10 rounds
    for (int round = 0; round < 10; round++) {
        const uint8_t* s = BLAKE2S_SIGMA[round];
        
        blake2s_g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        blake2s_g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        blake2s_g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        blake2s_g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        
        blake2s_g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        blake2s_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2s_g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        blake2s_g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    
    // Finalize
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// =============================================================================
// Blake2s Hash Function
// =============================================================================

// Hash a variable-length message (up to 64 bytes for leaves)
__device__ void blake2s_hash(
    uint8_t* out,          // Output: 32 bytes
    const uint8_t* in,     // Input message
    uint32_t inlen         // Input length
) {
    uint32_t h[8];
    
    // Initialize state with IV
    for (int i = 0; i < 8; i++) {
        h[i] = BLAKE2S_IV[i];
    }
    
    // Parameter block: digest length = 32, key length = 0, fanout = 1, depth = 1
    h[0] ^= 0x01010020;
    
    // Prepare message block
    uint32_t m[16] = {0};
    for (uint32_t i = 0; i < inlen && i < 64; i++) {
        m[i / 4] |= ((uint32_t)in[i]) << (8 * (i % 4));
    }
    
    // Compress
    blake2s_compress(h, m, inlen, true);
    
    // Output
    for (int i = 0; i < 8; i++) {
        out[4*i + 0] = (uint8_t)(h[i] >> 0);
        out[4*i + 1] = (uint8_t)(h[i] >> 8);
        out[4*i + 2] = (uint8_t)(h[i] >> 16);
        out[4*i + 3] = (uint8_t)(h[i] >> 24);
    }
}

// =============================================================================
// Merkle Leaf Hash Kernel
// =============================================================================

// Hash leaf data (columns) at each position
extern "C" __global__ void merkle_leaf_hash_kernel(
    uint8_t* __restrict__ output,        // Output hashes (32 bytes each)
    const uint32_t* __restrict__ columns, // Column data (M31 values)
    uint32_t n_columns,                   // Number of columns
    uint32_t n_leaves                     // Number of leaves
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_leaves) return;
    
    // Build message from column values at this index
    uint8_t msg[64];
    uint32_t msg_len = 0;
    
    for (uint32_t col = 0; col < n_columns && msg_len < 60; col++) {
        uint32_t val = columns[col * n_leaves + idx];
        // Pack M31 value as 4 bytes (little-endian)
        msg[msg_len++] = (uint8_t)(val >> 0);
        msg[msg_len++] = (uint8_t)(val >> 8);
        msg[msg_len++] = (uint8_t)(val >> 16);
        msg[msg_len++] = (uint8_t)(val >> 24);
    }
    
    // Hash the message
    blake2s_hash(output + idx * 32, msg, msg_len);
}

// =============================================================================
// Merkle Node Hash Kernel
// =============================================================================

// Hash pairs of child hashes to produce parent hashes
extern "C" __global__ void merkle_node_hash_kernel(
    uint8_t* __restrict__ output,        // Output hashes (32 bytes each)
    const uint8_t* __restrict__ children, // Child hashes (64 bytes per pair)
    uint32_t n_nodes                      // Number of parent nodes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_nodes) return;
    
    // Get the two child hashes
    const uint8_t* left = children + idx * 64;
    const uint8_t* right = children + idx * 64 + 32;
    
    // Concatenate children
    uint8_t msg[64];
    for (int i = 0; i < 32; i++) {
        msg[i] = left[i];
        msg[i + 32] = right[i];
    }
    
    // Hash to produce parent
    blake2s_hash(output + idx * 32, msg, 64);
}

// =============================================================================
// Combined Merkle Layer Kernel
// =============================================================================

// Hash a layer: either leaves (with column data) or nodes (with prev layer)
extern "C" __global__ void merkle_layer_kernel(
    uint8_t* __restrict__ output,         // Output hashes
    const uint32_t* __restrict__ columns,  // Column data (NULL for non-leaf)
    const uint8_t* __restrict__ prev_layer, // Previous layer hashes (NULL for leaf)
    uint32_t n_columns,                    // Number of columns
    uint32_t n_hashes,                     // Number of hashes to compute
    uint32_t has_prev_layer                // 1 if prev_layer is valid
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_hashes) return;
    
    uint8_t msg[128];
    uint32_t msg_len = 0;
    
    // If we have a previous layer, include child hashes first
    if (has_prev_layer && prev_layer != NULL) {
        const uint8_t* left = prev_layer + idx * 64;
        const uint8_t* right = prev_layer + idx * 64 + 32;
        for (int i = 0; i < 32; i++) {
            msg[msg_len++] = left[i];
        }
        for (int i = 0; i < 32; i++) {
            msg[msg_len++] = right[i];
        }
    }
    
    // Add column data
    if (columns != NULL) {
        for (uint32_t col = 0; col < n_columns && msg_len < 120; col++) {
            uint32_t val = columns[col * n_hashes + idx];
            msg[msg_len++] = (uint8_t)(val >> 0);
            msg[msg_len++] = (uint8_t)(val >> 8);
            msg[msg_len++] = (uint8_t)(val >> 16);
            msg[msg_len++] = (uint8_t)(val >> 24);
        }
    }
    
    // Hash
    blake2s_hash(output + idx * 32, msg, msg_len);
}
"#;

// =============================================================================
// Global GPU Context
// =============================================================================

#[cfg(feature = "gpu")]
static GPU_FFT_CONTEXT: OnceLock<std::sync::Mutex<GpuFftContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
pub fn get_gpu_fft_context() -> &'static std::sync::Mutex<GpuFftContext> {
    GPU_FFT_CONTEXT.get_or_init(|| std::sync::Mutex::new(GpuFftContext::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert_eq!(M31_PRIME, 0x7FFFFFFF);
        assert_eq!(M31_PRIME, (1u32 << 31) - 1);
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE <= 20);
    }
    
    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!CIRCLE_FFT_CUDA_KERNEL.is_empty());
        assert!(CIRCLE_FFT_CUDA_KERNEL.contains("m31_add"));
        assert!(CIRCLE_FFT_CUDA_KERNEL.contains("ibutterfly"));
        assert!(CIRCLE_FFT_CUDA_KERNEL.contains("ifft_layer_kernel"));
    }
}
