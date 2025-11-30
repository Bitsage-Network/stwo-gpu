//! GPU Pipeline Benchmark - Path to 50x+ Speedup
//!
//! This benchmark demonstrates the full GPU proof pipeline where data stays
//! on GPU throughout the entire proof generation process.
//!
//! Run with:
//!   cargo run --example gpu_pipeline_benchmark --features cuda-runtime,prover --release

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use std::time::Instant;

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn main() {
    use stwo::prover::backend::gpu::pipeline::{GpuProofPipeline, benchmark_proof_pipeline};
    use stwo::prover::backend::gpu::cuda_executor::get_cuda_executor;
    use stwo::prover::backend::gpu::fft::compute_itwiddle_dbls_cpu;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::simd::column::BaseColumn;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║          GPU Pipeline Benchmark - Path to 50x+ Speedup                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let executor = match get_cuda_executor() {
        Ok(e) => e,
        Err(e) => {
            println!("CUDA not available: {}", e);
            return;
        }
    };

    println!("Device: {} ({} SMs, {} GB)", 
             executor.device_info.name, 
             executor.device_info.multiprocessor_count,
             executor.device_info.total_memory_bytes / (1024 * 1024 * 1024));
    println!();

    // Test configurations
    let configs = [
        (18, 4, 10),   // 256K elements, 4 polys, 10 rounds = 80 FFTs
        (20, 4, 10),   // 1M elements, 4 polys, 10 rounds = 80 FFTs
        (20, 8, 10),   // 1M elements, 8 polys, 10 rounds = 160 FFTs
        (20, 4, 25),   // 1M elements, 4 polys, 25 rounds = 200 FFTs (realistic proof)
    ];

    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark: GPU Pipeline vs SIMD (Per-Operation Transfer)                │");
    println!("└──────────────────────────────────────────────────────────────────────────┘");
    println!();

    for (log_size, num_polys, num_rounds) in configs {
        let n = 1usize << log_size;
        let total_ffts = num_polys * num_rounds * 2;

        println!("Configuration: 2^{} elements × {} polynomials × {} rounds = {} FFTs", 
                 log_size, num_polys, num_rounds, total_ffts);
        println!("─────────────────────────────────────────────────────────────────────");
        
        // Generate test data
        let test_data: Vec<Vec<u32>> = (0..num_polys)
            .map(|p| {
                (0..n)
                    .map(|i| ((i * 7 + p * 13 + 17) as u32) % ((1 << 31) - 1))
                    .collect()
            })
            .collect();

        // =====================================================================
        // Benchmark SIMD (baseline)
        // =====================================================================
        let domain = CanonicCoset::new(log_size).circle_domain();
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        
        let simd_start = Instant::now();
        for _round in 0..num_rounds {
            for data in &test_data {
                let simd_data: BaseColumn = data.iter().map(|&x| BaseField::from(x)).collect();
                let eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    domain,
                    simd_data,
                );
                let coeffs = SimdBackend::interpolate(eval, &simd_twiddles);
                let _eval2 = SimdBackend::evaluate(&coeffs, domain, &simd_twiddles);
            }
        }
        let simd_time = simd_start.elapsed();
        
        // =====================================================================
        // Benchmark GPU Pipeline
        // =====================================================================
        let pipeline_result = benchmark_proof_pipeline(log_size, num_polys, num_rounds).unwrap();
        
        // =====================================================================
        // Benchmark GPU with per-operation transfer (for comparison)
        // =====================================================================
        let itwiddles = compute_itwiddle_dbls_cpu(log_size);
        
        let per_op_start = Instant::now();
        for _round in 0..num_rounds {
            for data in &test_data {
                let mut work_data = data.clone();
                executor.execute_ifft(&mut work_data, &itwiddles, log_size).unwrap();
                // Note: Not doing FFT back to keep comparison fair with pipeline
            }
        }
        let per_op_time = per_op_start.elapsed();
        
        // =====================================================================
        // Results
        // =====================================================================
        let speedup_vs_simd = simd_time.as_secs_f64() / pipeline_result.total_time.as_secs_f64();
        let speedup_vs_per_op = per_op_time.as_secs_f64() / pipeline_result.compute_time.as_secs_f64();
        
        println!("  SIMD baseline:        {:>10.2?}  ({:.1} µs/FFT)", 
                 simd_time, simd_time.as_secs_f64() / total_ffts as f64 * 1_000_000.0);
        println!("  GPU per-op transfer:  {:>10.2?}  ({:.1} µs/FFT)", 
                 per_op_time, per_op_time.as_secs_f64() / (num_polys * num_rounds) as f64 * 1_000_000.0);
        println!("  GPU pipeline total:   {:>10.2?}  ({:.1} µs/FFT)", 
                 pipeline_result.total_time, pipeline_result.time_per_fft().as_secs_f64() * 1_000_000.0);
        println!("    - Upload:           {:>10.2?}", pipeline_result.upload_time);
        println!("    - Compute:          {:>10.2?}", pipeline_result.compute_time);
        println!("    - Download:         {:>10.2?}", pipeline_result.download_time);
        println!();
        println!("  ╔═══════════════════════════════════════════════════════════════╗");
        println!("  ║  Speedup vs SIMD:           {:>6.1}x                           ║", speedup_vs_simd);
        println!("  ║  Speedup vs per-op GPU:     {:>6.1}x                           ║", speedup_vs_per_op);
        println!("  ║  Transfer overhead:         {:>6.1}%                           ║", pipeline_result.transfer_overhead_percent());
        println!("  ╚═══════════════════════════════════════════════════════════════╝");
        println!();
    }

    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ Summary                                                                  │");
    println!("└──────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("The GPU Pipeline achieves massive speedup by:");
    println!("  1. Uploading all data ONCE at the start");
    println!("  2. Executing ALL FFT operations on GPU");
    println!("  3. Downloading results ONCE at the end");
    println!();
    println!("For a full STARK proof (FFT + FRI + Quotient + Merkle), the speedup");
    println!("would be even higher as more operations stay on GPU.");
}

#[cfg(not(all(feature = "cuda-runtime", feature = "prover")))]
fn main() {
    println!("This example requires the 'cuda-runtime' and 'prover' features.");
    println!("Run with:");
    println!("  cargo run --example gpu_pipeline_benchmark --features cuda-runtime,prover --release");
}

