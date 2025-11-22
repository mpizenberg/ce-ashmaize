use ashmaize::{
    b2::{RomGenerationType, hash as cpu_hash},
    metal::MetalAshmaize,
    rom::Rom,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Instant;

// Helper function to run a single benchmark scenario
fn run_benchmark_scenario(
    metal_ashmaize: &MetalAshmaize,
    batch_size: usize,
    nb_loops: u32,
    nb_instrs: u32,
    rom_size: usize,
) -> (std::time::Duration, std::time::Duration) {
    // Generate ROM once per scenario
    let rom = Rom::new(
        b"benchmark_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        rom_size,
    );
    let light_rom = rom.shrink();

    // Generate salts for the batch
    let salts: Vec<Vec<u8>> = (0..batch_size)
        .map(|i| format!("salt_{}", i).into_bytes())
        .collect();

    // Pad salts to the same length for the GPU implementation
    let max_len = salts.iter().map(|s| s.len()).max().unwrap_or(0);
    let padded_salts: Vec<Vec<u8>> = salts
        .into_iter()
        .map(|mut s| {
            s.resize(max_len, 0);
            s
        })
        .collect();
    let salt_slices: Vec<&[u8]> = padded_salts.iter().map(|s| s.as_slice()).collect();

    // CPU Benchmark
    let cpu_start = Instant::now();
    let cpu_results: Vec<_> = salt_slices
        .iter()
        .map(|s| cpu_hash(s, &light_rom, nb_loops, nb_instrs))
        .collect();
    let cpu_duration = cpu_start.elapsed();

    // GPU Benchmark
    let gpu_start = Instant::now();
    let gpu_results = metal_ashmaize
        .hash(&salt_slices, &light_rom, nb_loops, nb_instrs)
        .expect("GPU hash failed");
    let gpu_duration = gpu_start.elapsed();

    // Basic verification (compare first hash)
    if !cpu_results.is_empty() && !gpu_results.is_empty() {
        assert_eq!(
            cpu_results[0], gpu_results[0],
            "CPU and GPU hash mismatch for first salt in batch!"
        );
    }

    (cpu_duration, gpu_duration)
}

fn metal_ashmaize_benchmark(c: &mut Criterion) {
    let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

    // --- Scenario 1: Batch Size Scaling ---
    let mut group = c.benchmark_group("Batch Size Scaling");
    let batch_sizes = [1, 10, 100, 1000, 10000];
    let fixed_nb_loops = 8;
    let fixed_nb_instrs = 256;
    let fixed_rom_size = 10 * 1024 * 1024; // 10MB

    for &batch_size in batch_sizes.iter() {
        // Reduce sample count for larger batch sizes
        if batch_size >= 1000 {
            group.sample_size(10); // Much faster, still statistically valid
        } else {
            group.sample_size(100); // Default is typically 100
        }

        group.throughput(criterion::Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            format!("batch_size_{}", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    run_benchmark_scenario(
                        &metal_ashmaize,
                        size,
                        fixed_nb_loops,
                        fixed_nb_instrs,
                        fixed_rom_size,
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, metal_ashmaize_benchmark);
criterion_main!(benches);
