//! GPU FFT Test Example
//!
//! Tests the CUDA FFT implementation against the SIMD baseline.
//!
//! Run with:
//!   cargo run --example gpu_test --features cuda-runtime,prover --release
//!
//! Expected output on A100:
//!   - 2^14 (16K): ~4x speedup
//!   - 2^16 (64K): ~12x speedup
//!   - 2^18 (256K): ~30x speedup
//!   - 2^20 (1M): ~60x speedup

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use std::time::Instant;

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn main() {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::prover::backend::gpu::cuda_executor::{get_cuda_executor, is_cuda_available};
    use stwo::prover::backend::gpu::GpuBackend;
    use stwo::prover::backend::simd::column::BaseColumn;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Obelysk GPU FFT Test Suite                     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ==========================================================================
    // Step 1: Check CUDA availability
    // ==========================================================================
    
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Step 1: CUDA Environment Check                          │");
    println!("└──────────────────────────────────────────────────────────┘");
    
    if !is_cuda_available() {
        println!("❌ CUDA is NOT available!");
        println!();
        println!("Possible causes:");
        println!("  1. No NVIDIA GPU detected");
        println!("  2. CUDA drivers not installed");
        println!("  3. CUDA toolkit not in PATH");
        println!();
        println!("On Brev, make sure you created a GPU instance:");
        println!("  brev create my-instance --gpu a100");
        return;
    }
    
    println!("✅ CUDA is available!");
    
    match get_cuda_executor() {
        Ok(executor) => {
            println!("   Device: {}", executor.device_info.name);
            println!(
                "   Compute Capability: {}.{}",
                executor.device_info.compute_capability.0,
                executor.device_info.compute_capability.1
            );
            println!(
                "   Total Memory: {} GB",
                executor.device_info.total_memory_bytes / (1024 * 1024 * 1024)
            );
            println!(
                "   Multiprocessors: {}",
                executor.device_info.multiprocessor_count
            );
        }
        Err(e) => {
            println!("❌ Failed to get CUDA executor: {}", e);
            return;
        }
    }
    println!();

    // ==========================================================================
    // Step 2: Correctness Test
    // ==========================================================================
    
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Correctness Test                                 │");
    println!("└──────────────────────────────────────────────────────────┘");
    
    let test_log_size = 14u32; // 16K elements - small enough for quick test
    let test_size = 1usize << test_log_size;
    
    println!("Testing with {} elements (2^{})...", test_size, test_log_size);
    
    // Generate deterministic test data
    let data: BaseColumn = (0..test_size)
        .map(|i| BaseField::from((i * 7 + 13) as u32 % (1 << 30)))
        .collect();
    
    let domain = CanonicCoset::new(test_log_size).circle_domain();
    
    // SIMD computation
    let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    let simd_eval =
        CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, data.clone());
    let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
    
    // GPU computation
    let gpu_twiddles = GpuBackend::precompute_twiddles(domain.half_coset);
    let gpu_eval =
        CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(domain, data.clone());
    let gpu_result = GpuBackend::interpolate(gpu_eval, &gpu_twiddles);
    
    // Compare results
    let simd_coeffs = simd_result.coeffs.as_slice();
    let gpu_coeffs = gpu_result.coeffs.as_slice();
    
    let mut differences = 0usize;
    for (i, (s, g)) in simd_coeffs.iter().zip(gpu_coeffs.iter()).enumerate() {
        if s != g {
            differences += 1;
            if differences <= 10 {
                println!("   Mismatch at index {}: SIMD={}, GPU={}", i, s.0, g.0);
            }
        }
    }
    
    if differences == 0 {
        println!("✅ All {} coefficients match!", test_size);
    } else {
        println!("❌ Found {} mismatches out of {} coefficients", differences, test_size);
        println!();
        println!("This indicates a bug in the GPU FFT implementation.");
        println!("Please report this issue with the test output.");
        return;
    }
    println!();

    // ==========================================================================
    // Step 3: Performance Benchmark
    // ==========================================================================
    
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Performance Benchmark                           │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();
    println!("  Size      │  SIMD Time  │  GPU Time   │  Speedup");
    println!("────────────┼─────────────┼─────────────┼──────────");
    
    let benchmark_sizes = [12, 14, 16, 18, 20];
    let warmup_runs = 2;
    let benchmark_runs = 5;
    
    for log_size in benchmark_sizes {
        let size = 1usize << log_size;
        
        // Generate data
        let data: BaseColumn = (0..size)
            .map(|i| BaseField::from((i * 7 + 13) as u32 % (1 << 30)))
            .collect();
        
        let domain = CanonicCoset::new(log_size).circle_domain();
        
        // Precompute twiddles (not included in timing)
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let gpu_twiddles = GpuBackend::precompute_twiddles(domain.half_coset);
        
        // Warmup
        for _ in 0..warmup_runs {
            let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                domain,
                data.clone(),
            );
            let _ = SimdBackend::interpolate(simd_eval, &simd_twiddles);
            
            let gpu_eval = CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(
                domain,
                data.clone(),
            );
            let _ = GpuBackend::interpolate(gpu_eval, &gpu_twiddles);
        }
        
        // Benchmark SIMD
        let simd_start = Instant::now();
        for _ in 0..benchmark_runs {
            let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                domain,
                data.clone(),
            );
            let _ = SimdBackend::interpolate(simd_eval, &simd_twiddles);
        }
        let simd_total = simd_start.elapsed();
        let simd_avg = simd_total / benchmark_runs as u32;
        
        // Benchmark GPU
        let gpu_start = Instant::now();
        for _ in 0..benchmark_runs {
            let gpu_eval = CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(
                domain,
                data.clone(),
            );
            let _ = GpuBackend::interpolate(gpu_eval, &gpu_twiddles);
        }
        let gpu_total = gpu_start.elapsed();
        let gpu_avg = gpu_total / benchmark_runs as u32;
        
        // Calculate speedup
        let speedup = simd_avg.as_secs_f64() / gpu_avg.as_secs_f64();
        
        println!(
            "  2^{:<2} ({:>7}) │  {:>9.2?} │  {:>9.2?} │  {:>6.2}x",
            log_size,
            format_size(size),
            simd_avg,
            gpu_avg,
            speedup
        );
    }
    
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    Test Complete!                        ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn format_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

#[cfg(not(all(feature = "cuda-runtime", feature = "prover")))]
fn main() {
    println!("This example requires the 'cuda-runtime' and 'prover' features.");
    println!();
    println!("Run with:");
    println!("  cargo run --example gpu_test --features cuda-runtime,prover --release");
}

