//! Large Proof Benchmark
//!
//! This benchmark tests GPU performance with large polynomial sizes (2^24 to 2^28).
//! It automatically detects the maximum proof size that fits in GPU memory and
//! runs benchmarks at various sizes.
//!
//! Run with:
//! ```bash
//! cargo run --example large_proof_benchmark --release --features cuda-runtime
//! ```

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           Large Proof Benchmark - Obelysk GPU                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::large_proofs::{
            MemoryRequirements, detect_max_proof_size, benchmark_large_proof,
        };
        use stwo::prover::backend::gpu::GpuBackend;
        
        // Check GPU availability
        if !GpuBackend::is_available() {
            println!("❌ No CUDA GPU available!");
            return;
        }
        
        // Print GPU info
        println!("🖥️  GPU Information:");
        if let Some(name) = GpuBackend::device_name() {
            println!("    Name: {}", name);
        }
        if let Some(mem) = GpuBackend::available_memory() {
            println!("    Memory: {:.2} GB", mem as f64 / (1024.0 * 1024.0 * 1024.0));
        }
        if let Some((major, minor)) = GpuBackend::compute_capability() {
            println!("    Compute Capability: {}.{}", major, minor);
        }
        println!();
        
        // ═══════════════════════════════════════════════════════════════════════
        // Memory Requirements Analysis
        // ═══════════════════════════════════════════════════════════════════════
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Memory Requirements Analysis");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        
        println!("┌──────────┬──────────────┬─────────────────┬─────────────────┐");
        println!("│ Log Size │ Elements     │ Per Poly (MB)   │ 4 Polys (MB)    │");
        println!("├──────────┼──────────────┼─────────────────┼─────────────────┤");
        
        for log_size in [20, 22, 24, 26, 28] {
            let reqs_1 = MemoryRequirements::calculate(log_size, 1, 2);
            let reqs_4 = MemoryRequirements::calculate(log_size, 4, 2);
            println!("│ {:>8} │ {:>12} │ {:>15} │ {:>15} │",
                     log_size,
                     format!("{}M", reqs_1.elements / 1_000_000),
                     reqs_1.total_bytes / (1024 * 1024),
                     reqs_4.total_bytes / (1024 * 1024));
        }
        println!("└──────────┴──────────────┴─────────────────┴─────────────────┘");
        println!();
        
        // ═══════════════════════════════════════════════════════════════════════
        // Maximum Proof Size Detection
        // ═══════════════════════════════════════════════════════════════════════
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Maximum Proof Size Detection");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        
        match detect_max_proof_size(4) {
            Ok(max_size) => {
                println!("{}", max_size);
            }
            Err(e) => {
                println!("❌ Detection failed: {:?}", e);
                return;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // Benchmark at Various Sizes
        // ═══════════════════════════════════════════════════════════════════════
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Benchmark Results");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        
        // Get max size first
        let max_log = match detect_max_proof_size(4) {
            Ok(m) => m.max_log_size,
            Err(_) => 24,  // Fallback
        };
        
        // Test sizes from 2^18 to max
        let test_sizes: Vec<u32> = (18..=max_log.min(28)).filter(|&s| s % 2 == 0 || s == max_log).collect();
        
        let mut results = Vec::new();
        
        for log_size in test_sizes {
            println!("┌────────────────────────────────────────────────────────────────┐");
            println!("│ Testing 2^{} ({} elements)", log_size, 1usize << log_size);
            println!("└────────────────────────────────────────────────────────────────┘");
            
            match benchmark_large_proof(log_size, 4) {
                Ok(result) => {
                    println!();
                    println!("  ✅ Results:");
                    println!("     Setup:     {:>10.2}ms", result.setup_time.as_secs_f64() * 1000.0);
                    println!("     Upload:    {:>10.2}ms", result.upload_time.as_secs_f64() * 1000.0);
                    println!("     Compute:   {:>10.2}ms", result.compute_time.as_secs_f64() * 1000.0);
                    println!("     Download:  {:>10.2}ms", result.download_time.as_secs_f64() * 1000.0);
                    println!("     Total:     {:>10.2}ms", result.total_time.as_secs_f64() * 1000.0);
                    println!();
                    println!("     Throughput: {:.1} FFTs/sec", result.throughput_ffts_per_sec);
                    println!("     Bandwidth:  {:.2} GB/s", result.bandwidth_gbps);
                    
                    results.push((log_size, result));
                }
                Err(e) => {
                    println!("  ❌ Failed: {:?}", e);
                }
            }
            println!();
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // Summary Table
        // ═══════════════════════════════════════════════════════════════════════
        if !results.is_empty() {
            println!("════════════════════════════════════════════════════════════════════");
            println!("  Summary");
            println!("════════════════════════════════════════════════════════════════════");
            println!();
            
            println!("┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
            println!("│ Log Size │ Elements     │ Compute (ms) │ FFTs/sec     │ Bandwidth    │");
            println!("├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
            
            for (log_size, result) in &results {
                println!("│ {:>8} │ {:>12} │ {:>12.2} │ {:>12.1} │ {:>10.2} GB/s │",
                         log_size,
                         format!("{}M", result.elements / 1_000_000),
                         result.compute_time.as_secs_f64() * 1000.0,
                         result.throughput_ffts_per_sec,
                         result.bandwidth_gbps);
            }
            println!("└──────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
            println!();
            
            // Calculate scaling efficiency
            if results.len() >= 2 {
                let (log1, res1) = &results[0];
                let (log2, res2) = &results[results.len() - 1];
                
                let size_ratio = (1usize << *log2) as f64 / (1usize << *log1) as f64;
                let time_ratio = res2.compute_time.as_secs_f64() / res1.compute_time.as_secs_f64();
                let scaling_efficiency = (size_ratio / time_ratio) * 100.0;
                
                println!("  📊 Scaling Analysis:");
                println!("     From 2^{} to 2^{}: {}x larger", log1, log2, size_ratio as usize);
                println!("     Time increase: {:.1}x", time_ratio);
                println!("     Scaling efficiency: {:.1}%", scaling_efficiency);
                println!();
                println!("     (100% = perfect linear scaling, >100% = super-linear due to amortization)");
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // GPU Tier Recommendations
        // ═══════════════════════════════════════════════════════════════════════
        println!();
        println!("════════════════════════════════════════════════════════════════════");
        println!("  GPU Tier Recommendations");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        println!("  ┌─────────────────┬──────────┬───────────────────────────────────┐");
        println!("  │ GPU             │ Memory   │ Max Proof Size (4 polys)          │");
        println!("  ├─────────────────┼──────────┼───────────────────────────────────┤");
        println!("  │ RTX 4090        │ 24 GB    │ 2^26 (64M elements)               │");
        println!("  │ A100 40GB       │ 40 GB    │ 2^27 (128M elements)              │");
        println!("  │ A100 80GB       │ 80 GB    │ 2^28 (256M elements)              │");
        println!("  │ H100 80GB       │ 80 GB    │ 2^28 (256M elements) + faster     │");
        println!("  │ H200 141GB      │ 141 GB   │ 2^29 (512M elements)              │");
        println!("  └─────────────────┴──────────┴───────────────────────────────────┘");
        println!();
    }
    
    #[cfg(not(feature = "cuda-runtime"))]
    {
        println!("❌ This benchmark requires the 'cuda-runtime' feature.");
        println!("   Run with: cargo run --example large_proof_benchmark --release --features cuda-runtime");
    }
}

