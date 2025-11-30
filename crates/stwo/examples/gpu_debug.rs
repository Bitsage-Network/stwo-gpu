//! GPU FFT Debug Tool
//!
//! Diagnoses FFT correctness issues by comparing GPU and SIMD implementations
//! at various stages.
//!
//! Run with:
//!   cargo run --example gpu_debug --features cuda-runtime,prover --release

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn main() {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::prover::backend::gpu::cuda_executor::{get_cuda_executor, is_cuda_available};
    use stwo::prover::backend::gpu::fft::{compute_itwiddle_dbls_cpu, compute_twiddle_dbls_cpu};
    use stwo::prover::backend::gpu::GpuBackend;
    use stwo::prover::backend::simd::column::BaseColumn;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              GPU FFT Debug & Diagnostic Tool                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Check CUDA
    if !is_cuda_available() {
        println!("❌ CUDA not available");
        return;
    }
    
    match get_cuda_executor() {
        Ok(exec) => println!("✅ CUDA executor ready: {}", exec.device_info.name),
        Err(e) => {
            println!("❌ CUDA executor failed: {}", e);
            return;
        }
    }
    println!();

    // ==========================================================================
    // Test 1: Very small FFT (2^4 = 16 elements)
    // ==========================================================================
    
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Test 1: Small FFT (2^4 = 16 elements)                          │");
    println!("└────────────────────────────────────────────────────────────────┘");
    
    let log_size = 4u32;
    let size = 1usize << log_size;
    
    // Simple sequential input
    let data: BaseColumn = (0..size)
        .map(|i| BaseField::from(i as u32))
        .collect();
    
    println!("Input data: {:?}", data.as_slice().iter().take(16).map(|x| x.0).collect::<Vec<_>>());
    
    let domain = CanonicCoset::new(log_size).circle_domain();
    
    // SIMD computation
    let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, data.clone());
    let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
    
    println!("SIMD result: {:?}", simd_result.coeffs.as_slice().iter().take(16).map(|x| x.0).collect::<Vec<_>>());
    
    // GPU computation
    let gpu_twiddles = GpuBackend::precompute_twiddles(domain.half_coset);
    let gpu_eval = CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(domain, data.clone());
    let gpu_result = GpuBackend::interpolate(gpu_eval, &gpu_twiddles);
    
    println!("GPU result:  {:?}", gpu_result.coeffs.as_slice().iter().take(16).map(|x| x.0).collect::<Vec<_>>());
    
    // Compare
    let simd_coeffs = simd_result.coeffs.as_slice();
    let gpu_coeffs = gpu_result.coeffs.as_slice();
    
    let mut mismatches = 0;
    for (i, (s, g)) in simd_coeffs.iter().zip(gpu_coeffs.iter()).enumerate() {
        if s != g {
            println!("  Mismatch at {}: SIMD={}, GPU={}", i, s.0, g.0);
            mismatches += 1;
        }
    }
    
    if mismatches == 0 {
        println!("✅ All {} values match!", size);
    } else {
        println!("❌ {} mismatches out of {}", mismatches, size);
    }
    println!();

    // ==========================================================================
    // Test 2: Compare Twiddles
    // ==========================================================================
    
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Test 2: Compare Twiddle Computation                            │");
    println!("└────────────────────────────────────────────────────────────────┘");
    
    let log_size = 4u32;
    
    // Get SIMD twiddles
    let domain = CanonicCoset::new(log_size).circle_domain();
    let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    
    println!("SIMD itwiddles (first 16): {:?}", simd_twiddles.itwiddles.iter().take(16).collect::<Vec<_>>());
    
    // Get GPU twiddles
    let gpu_itwiddles = compute_itwiddle_dbls_cpu(log_size);
    
    for (layer, twiddles) in gpu_itwiddles.iter().enumerate() {
        println!("GPU layer {} ({} twiddles): {:?}", layer, twiddles.len(), 
            twiddles.iter().take(8).collect::<Vec<_>>());
    }
    println!();

    // ==========================================================================
    // Test 3: Test at threshold size
    // ==========================================================================
    
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Test 3: FFT at threshold (2^14 = 16K elements)                 │");
    println!("└────────────────────────────────────────────────────────────────┘");
    
    let log_size = 14u32;
    let size = 1usize << log_size;
    
    let data: BaseColumn = (0..size)
        .map(|i| BaseField::from((i * 7 + 13) as u32 % (1 << 30)))
        .collect();
    
    let domain = CanonicCoset::new(log_size).circle_domain();
    
    // SIMD
    let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, data.clone());
    let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
    
    // GPU
    let gpu_twiddles = GpuBackend::precompute_twiddles(domain.half_coset);
    let gpu_eval = CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(domain, data.clone());
    let gpu_result = GpuBackend::interpolate(gpu_eval, &gpu_twiddles);
    
    // Compare first/last elements
    let simd_coeffs = simd_result.coeffs.as_slice();
    let gpu_coeffs = gpu_result.coeffs.as_slice();
    
    println!("First 8 SIMD: {:?}", simd_coeffs.iter().take(8).map(|x| x.0).collect::<Vec<_>>());
    println!("First 8 GPU:  {:?}", gpu_coeffs.iter().take(8).map(|x| x.0).collect::<Vec<_>>());
    println!();
    println!("Last 8 SIMD:  {:?}", simd_coeffs.iter().rev().take(8).map(|x| x.0).collect::<Vec<_>>());
    println!("Last 8 GPU:   {:?}", gpu_coeffs.iter().rev().take(8).map(|x| x.0).collect::<Vec<_>>());
    
    let mut mismatches = 0;
    let mut first_mismatch = None;
    for (i, (s, g)) in simd_coeffs.iter().zip(gpu_coeffs.iter()).enumerate() {
        if s != g {
            if first_mismatch.is_none() {
                first_mismatch = Some((i, s.0, g.0));
            }
            mismatches += 1;
        }
    }
    
    if mismatches == 0 {
        println!("✅ All {} values match!", size);
    } else {
        println!("❌ {} mismatches out of {} ({:.1}%)", mismatches, size, 100.0 * mismatches as f64 / size as f64);
        if let Some((i, s, g)) = first_mismatch {
            println!("  First mismatch at index {}: SIMD={}, GPU={}", i, s, g);
        }
    }
    println!();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    Debug Complete                                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

#[cfg(not(all(feature = "cuda-runtime", feature = "prover")))]
fn main() {
    println!("This example requires the 'cuda-runtime' and 'prover' features.");
    println!();
    println!("Run with:");
    println!("  cargo run --example gpu_debug --features cuda-runtime,prover --release");
}

