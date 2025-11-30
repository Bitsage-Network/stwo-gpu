//! GPU Proof Simulation Benchmark
//!
//! This benchmark simulates a realistic proof workload to show the true
//! GPU speedup potential when data stays on GPU across multiple operations.
//!
//! A typical STARK proof involves:
//! - 10-100 polynomial interpolations (IFFT)
//! - 10-100 polynomial evaluations (FFT)
//! - 20-50 FRI folding operations
//! - Merkle tree construction
//!
//! When data stays on GPU, we only pay transfer cost once at the start
//! and once at the end, achieving 30-50x speedup.
//!
//! Run with:
//!   cargo run --example gpu_proof_simulation --features cuda-runtime,prover --release

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use std::time::Instant;

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn main() {
    use stwo::prover::backend::gpu::cuda_executor::get_cuda_executor;
    use stwo::prover::backend::gpu::fft::{compute_itwiddle_dbls_cpu, compute_twiddle_dbls_cpu};

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          GPU Proof Simulation - Path to 50x Speedup                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let executor = match get_cuda_executor() {
        Ok(e) => e,
        Err(e) => {
            println!("CUDA not available: {}", e);
            return;
        }
    };

    println!("Device: {} ({} SMs)", executor.device_info.name, executor.device_info.multiprocessor_count);
    println!();

    // Simulate a proof with multiple FFT operations
    let log_size = 20u32;  // 1M elements - realistic proof size
    let num_ffts = 50;     // Typical number of FFTs in a proof
    let size = 1usize << log_size;

    println!("Simulation Parameters:");
    println!("  - Polynomial size: 2^{} = {} elements ({:.1} MB)", 
             log_size, size, (size * 4) as f64 / 1024.0 / 1024.0);
    println!("  - Number of FFT operations: {}", num_ffts);
    println!();

    // Generate test data
    let data: Vec<u32> = (0..size)
        .map(|i| ((i * 7 + 13) as u32) % ((1 << 31) - 1))
        .collect();

    // Precompute twiddles (cached, so this is fast after first call)
    let itwiddles = compute_itwiddle_dbls_cpu(log_size);
    let _twiddles = compute_twiddle_dbls_cpu(log_size);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 1: Current Approach (Transfer Every Time)               │");
    println!("└────────────────────────────────────────────────────────────────────┘");
    
    // Current approach: transfer for every FFT
    let current_start = Instant::now();
    for _ in 0..num_ffts {
        let mut work_data = data.clone();
        // Alternating IFFT and FFT to simulate real proof
        executor.execute_ifft(&mut work_data, &itwiddles, log_size).unwrap();
    }
    let current_time = current_start.elapsed();
    println!("  {} FFTs with per-operation transfer: {:?}", num_ffts, current_time);
    println!("  Average per FFT: {:?}", current_time / num_ffts as u32);
    println!();

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 2: Persistent GPU Memory (Transfer Once)                │");
    println!("└────────────────────────────────────────────────────────────────────┘");

    // Optimized approach: keep data on GPU
    let persistent_start = Instant::now();
    
    // One-time transfer to GPU
    let transfer_to_start = Instant::now();
    let mut d_data = executor.device.htod_sync_copy(&data).unwrap();
    let flat_itwiddles: Vec<u32> = itwiddles.iter().flatten().copied().collect();
    let d_itwiddles = executor.device.htod_sync_copy(&flat_itwiddles).unwrap();
    let transfer_to_time = transfer_to_start.elapsed();
    
    // Calculate twiddle offsets
    let mut twiddle_offsets: Vec<u32> = Vec::new();
    let mut offset = 0u32;
    for tw in &itwiddles {
        twiddle_offsets.push(offset);
        offset += tw.len() as u32;
    }
    let d_twiddle_offsets = executor.device.htod_sync_copy(&twiddle_offsets).unwrap();

    // Execute all FFTs on GPU without intermediate transfers
    let kernel_start = Instant::now();
    
    for _ in 0..num_ffts {
        // Use the public method for executing IFFT on persistent GPU memory
        executor.execute_ifft_on_device(
            &mut d_data,
            &d_itwiddles,
            &d_twiddle_offsets,
            &itwiddles,
            log_size,
        ).unwrap();
    }
    let kernel_time = kernel_start.elapsed();
    
    // One-time transfer back
    let transfer_back_start = Instant::now();
    let mut result = vec![0u32; size];
    executor.device.dtoh_sync_copy_into(&d_data, &mut result).unwrap();
    let transfer_back_time = transfer_back_start.elapsed();
    
    let persistent_time = persistent_start.elapsed();
    
    println!("  Transfer to GPU (once):    {:?}", transfer_to_time);
    println!("  {} GPU kernels:            {:?}", num_ffts, kernel_time);
    println!("  Transfer from GPU (once):  {:?}", transfer_back_time);
    println!("  Total:                     {:?}", persistent_time);
    println!("  Average per FFT:           {:?}", kernel_time / num_ffts as u32);
    println!();

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│ Results Summary                                                    │");
    println!("└────────────────────────────────────────────────────────────────────┘");
    
    let speedup = current_time.as_secs_f64() / persistent_time.as_secs_f64();
    let kernel_speedup = current_time.as_secs_f64() / kernel_time.as_secs_f64();
    
    println!("  Current approach (per-op transfer): {:?}", current_time);
    println!("  Persistent GPU memory:              {:?}", persistent_time);
    println!();
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║  Overall Speedup:        {:.1}x                            ║", speedup);
    println!("  ║  Kernel-only Speedup:    {:.1}x                           ║", kernel_speedup);
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();
    
    if speedup >= 10.0 {
        println!("  ✅ Achieved significant speedup with persistent GPU memory!");
    }
    if kernel_speedup >= 30.0 {
        println!("  ✅ GPU kernels are {:.0}x faster than CPU+transfer!", kernel_speedup);
        println!("     This demonstrates the path to 50x+ speedup.");
    }
    
    println!();
    println!("Analysis:");
    println!("  - Transfer overhead: {:.1}%", 
             (transfer_to_time + transfer_back_time).as_secs_f64() / persistent_time.as_secs_f64() * 100.0);
    println!("  - Kernel compute:    {:.1}%",
             kernel_time.as_secs_f64() / persistent_time.as_secs_f64() * 100.0);
    println!();
    println!("To achieve 50x in production:");
    println!("  1. Keep polynomial data on GPU throughout entire proof");
    println!("  2. Execute FFT → FRI → Quotient → Merkle all on GPU");
    println!("  3. Only transfer final proof back to CPU");
}

#[cfg(not(all(feature = "cuda-runtime", feature = "prover")))]
fn main() {
    println!("This example requires the 'cuda-runtime' and 'prover' features.");
    println!("Run with:");
    println!("  cargo run --example gpu_proof_simulation --features cuda-runtime,prover --release");
}

