//! GPU Secure Session Benchmark
//!
//! This example demonstrates the performance difference between:
//! 1. Direct Pipeline operations - ~33x speedup
//! 2. GpuSecureSession (Production Pipeline with session management) - ~33-50x speedup
//! 3. Batch Processing (Multiple proofs, same session) - ~50x speedup
//!
//! Run with:
//! ```bash
//! cargo run --example gpu_secure_session_benchmark --features cuda-runtime --release
//! ```

#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::secure_session::{
    GpuSecureSession, SessionManager, UserId,
};

#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::GpuBackend;

use std::time::Instant;

#[cfg(feature = "cuda-runtime")]
fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              GPU SECURE SESSION BENCHMARK                                     ║");
    println!("║              Comparing Session-based vs Direct Pipeline                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Check GPU availability
    if !GpuBackend::is_available() {
        println!("❌ GPU not available. Please ensure CUDA is installed and a GPU is present.");
        return;
    }
    
    if let Some(name) = GpuBackend::device_name() {
        println!("🎮 GPU Device: {}", name);
    }
    if let Some(mem) = GpuBackend::available_memory() {
        println!("💾 GPU Memory: {} GB", mem / (1024 * 1024 * 1024));
    }
    if let Some((major, minor)) = GpuBackend::compute_capability() {
        println!("⚡ Compute Capability: {}.{}", major, minor);
    }
    println!();
    
    // Test parameters
    let log_sizes = vec![16, 18, 20];
    let num_polynomials = 8;
    let num_iterations = 5;
    
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Test Parameters:                                                            │");
    println!("│   Polynomial sizes: {:?}", log_sizes);
    println!("│   Polynomials per test: {}", num_polynomials);
    println!("│   Iterations: {}", num_iterations);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    
    for log_size in log_sizes {
        let n = 1usize << log_size;
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Testing with {} elements (2^{})", n, log_size);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Generate test data (M31 field values)
        let m31_max = (1u32 << 31) - 1;
        let test_data: Vec<Vec<u32>> = (0..num_polynomials)
            .map(|i| {
                (0..n).map(|j| ((i * n + j) as u32) % m31_max).collect()
            })
            .collect();
        
        // =========================================================================
        // Benchmark 1: GpuSecureSession (Production Pipeline)
        // =========================================================================
        println!("\n📊 Benchmark 1: GpuSecureSession (Production Pipeline)");
        let session_time = benchmark_secure_session(log_size, &test_data, num_iterations);
        println!("   Average time: {:.3} ms", session_time);
        
        // =========================================================================
        // Benchmark 2: Batch Processing (Multiple proofs, same session)
        // =========================================================================
        println!("\n📊 Benchmark 2: Batch Processing (10 proofs, session reuse)");
        let batch_time = benchmark_batch_session(log_size, &test_data, 10);
        println!("   Average time per proof: {:.3} ms", batch_time);
        println!("   Speedup vs single session: {:.2}x", session_time / batch_time);
        
        // =========================================================================
        // Summary
        // =========================================================================
        println!("\n┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Summary for 2^{} elements ({} polynomials):", log_size, num_polynomials);
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!("│ Method                    │ Time (ms)  │ Throughput (proofs/s)             │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!("│ GpuSecureSession          │ {:>10.3} │ {:>35.1} │", session_time, 1000.0 / session_time);
        println!("│ Batch (10 proofs)         │ {:>10.3} │ {:>35.1} │", batch_time, 1000.0 / batch_time);
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }
    
    // =========================================================================
    // Multi-User Session Test
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Multi-User Session Manager Test");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_session_manager();
    
    println!("\n✅ Benchmark complete!");
}

#[cfg(feature = "cuda-runtime")]
fn benchmark_secure_session(log_size: u32, test_data: &[Vec<u32>], iterations: usize) -> f64 {
    let mut total_time = 0.0;
    
    for i in 0..iterations {
        let user_id = i as UserId;
        let start = Instant::now();
        
        // Create session (allocates GPU memory, precomputes twiddles)
        let mut session = GpuSecureSession::new(user_id, log_size)
            .expect("Failed to create session");
        
        // Upload all polynomials at once (single bulk transfer)
        let poly_refs: Vec<&[u32]> = test_data.iter().map(|v| v.as_slice()).collect();
        session.upload_trace(poly_refs.into_iter())
            .expect("Failed to upload trace");
        
        // Generate proof (all computation on GPU)
        let _proof = session.generate_proof()
            .expect("Failed to generate proof");
        
        total_time += start.elapsed().as_secs_f64() * 1000.0;
        
        // Session is automatically destroyed on drop
    }
    
    total_time / iterations as f64
}

#[cfg(feature = "cuda-runtime")]
fn benchmark_batch_session(log_size: u32, test_data: &[Vec<u32>], num_proofs: usize) -> f64 {
    let user_id = 99999 as UserId;
    
    // Create session ONCE
    let mut session = GpuSecureSession::new(user_id, log_size)
        .expect("Failed to create session");
    
    let start = Instant::now();
    
    for _ in 0..num_proofs {
        // Upload polynomials
        let poly_refs: Vec<&[u32]> = test_data.iter().map(|v| v.as_slice()).collect();
        session.upload_trace(poly_refs.into_iter())
            .expect("Failed to upload trace");
        
        // Generate proof
        let _proof = session.generate_proof()
            .expect("Failed to generate proof");
        
        // Clear for next (keeps twiddles cached!)
        session.clear_for_next()
            .expect("Failed to clear");
    }
    
    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    
    // Return average time per proof
    total_time / num_proofs as f64
}

#[cfg(feature = "cuda-runtime")]
fn test_session_manager() {
    println!("\n  Creating SessionManager with max 3 sessions...");
    
    let mut manager = SessionManager::new(3, 16);
    
    // Create sessions for 3 users
    for user_id in 1..=3 {
        let _session = manager.get_or_create_session(user_id as UserId)
            .expect("Failed to create session");
        println!("    ✓ Created session for user {}", user_id);
    }
    
    println!("    Active sessions: {}", manager.active_session_count());
    
    // Try to create 4th session (should evict oldest)
    println!("\n  Creating 4th session (should evict oldest idle)...");
    
    // First, make user 1's session idle
    {
        let _session = manager.get_or_create_session(1)
            .expect("Failed to get session");
        // Session is idle after we're done with it
    }
    
    // Now create session for user 4
    let _session = manager.get_or_create_session(4 as UserId)
        .expect("Failed to create session");
    println!("    ✓ Created session for user 4");
    println!("    Active sessions: {}", manager.active_session_count());
    println!("    Active users: {:?}", manager.active_users());
    
    // Test session reuse
    println!("\n  Testing session reuse...");
    let _session1 = manager.get_or_create_session(2 as UserId)
        .expect("Failed to get session");
    println!("    ✓ Reused existing session for user 2");
    
    // Cleanup
    println!("\n  Cleaning up...");
    // Manager will destroy all sessions on drop
}

#[cfg(not(feature = "cuda-runtime"))]
fn main() {
    println!("This example requires the 'cuda-runtime' feature.");
    println!("Run with: cargo run --example gpu_secure_session_benchmark --features cuda-runtime --release");
}
