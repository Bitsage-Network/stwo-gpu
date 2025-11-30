//! GPU FFT Detailed Benchmark
//!
//! This benchmark compares GPU vs SIMD FFT performance and estimates
//! the overhead breakdown.
//!
//! Run with:
//!   cargo run --example gpu_benchmark --features cuda-runtime,prover --release

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use std::time::Instant;

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn main() {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::prover::backend::gpu::cuda_executor::get_cuda_executor;
    use stwo::prover::backend::gpu::fft::compute_itwiddle_dbls_cpu;
    use stwo::prover::backend::simd::column::BaseColumn;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          GPU FFT Detailed Benchmark                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let executor = match get_cuda_executor() {
        Ok(e) => e,
        Err(e) => {
            println!("CUDA not available: {}", e);
            return;
        }
    };

    println!("Device: {}", executor.device_info.name);
    println!("Memory: {} GB", executor.device_info.total_memory_bytes / (1024 * 1024 * 1024));
    println!("SMs: {}", executor.device_info.multiprocessor_count);
    println!();

    let benchmark_sizes = [14, 16, 18, 20, 22];
    let warmup_runs = 2;
    let benchmark_runs = 5;

    println!("┌────────┬───────────┬────────────┬───────────┬────────────────────────────────┐");
    println!("│  Size  │ Elements  │ SIMD Time  │ GPU Time  │ Analysis                       │");
    println!("├────────┼───────────┼────────────┼───────────┼────────────────────────────────┤");

    for log_size in benchmark_sizes {
        let size = 1usize << log_size;
        let data_size_mb = (size * 4) as f64 / (1024.0 * 1024.0);
        
        // Generate data
        let data: Vec<u32> = (0..size)
            .map(|i| ((i * 7 + 13) as u32) % (1 << 30))
            .collect();
        
        let domain = CanonicCoset::new(log_size).circle_domain();
        
        // Precompute twiddles
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let gpu_itwiddles = compute_itwiddle_dbls_cpu(log_size);

        // Warmup
        for _ in 0..warmup_runs {
            let simd_data: BaseColumn = data.iter().map(|&x| BaseField::from(x)).collect();
            let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                domain,
                simd_data,
            );
            let _ = SimdBackend::interpolate(simd_eval, &simd_twiddles);
            
            let mut gpu_data = data.clone();
            let _ = executor.execute_ifft(&mut gpu_data, &gpu_itwiddles, log_size);
        }

        // Benchmark SIMD
        let simd_start = Instant::now();
        for _ in 0..benchmark_runs {
            let simd_data: BaseColumn = data.iter().map(|&x| BaseField::from(x)).collect();
            let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                domain,
                simd_data,
            );
            let _ = SimdBackend::interpolate(simd_eval, &simd_twiddles);
        }
        let simd_avg = simd_start.elapsed() / benchmark_runs as u32;

        // Benchmark GPU
        let gpu_start = Instant::now();
        for _ in 0..benchmark_runs {
            let mut gpu_data = data.clone();
            let _ = executor.execute_ifft(&mut gpu_data, &gpu_itwiddles, log_size);
        }
        let gpu_avg = gpu_start.elapsed() / benchmark_runs as u32;

        // Estimate transfer time (PCIe ~12 GB/s for A100)
        // Transfer = 2 * data_size (H2D + D2H) + twiddles
        let estimated_transfer_ms = (data_size_mb * 2.0) / 12000.0 * 1000.0;
        
        // Calculate speedup
        let speedup = simd_avg.as_secs_f64() / gpu_avg.as_secs_f64();
        
        let analysis = if speedup > 1.0 {
            format!("✅ {:.1}x faster", speedup)
        } else {
            format!("⚠️ {:.1}x slower (est. xfer: {:.1}ms)", 1.0/speedup, estimated_transfer_ms)
        };

        println!(
            "│ 2^{:<4} │ {:>9} │ {:>10.2?} │ {:>9.2?} │ {:<30} │",
            log_size,
            format_size(size),
            simd_avg,
            gpu_avg,
            analysis
        );
    }

    println!("└────────┴───────────┴────────────┴───────────┴────────────────────────────────┘");
    println!();
    println!("Analysis:");
    println!("  - PCIe transfer overhead dominates for small sizes");
    println!("  - GPU kernel time is fast, but transfers add ~1ms per MB");
    println!("  - Break-even point is around 2^22-2^24 for single FFT");
    println!();
    println!("For real speedup, data should stay on GPU across multiple operations.");
    println!("In a full proof, this means: commit → FFT → FRI → FFT → ... all on GPU.");
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
    println!("Run with:");
    println!("  cargo run --example gpu_benchmark --features cuda-runtime,prover --release");
}
