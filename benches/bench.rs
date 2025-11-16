use ashmaize::rom::{Rom, RomGenerationType};
use criterion::{Criterion, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    const GB: usize = 1_024 * 1_024 * 1_024;
    const MB: usize = 1_024 * 1_024;

    // ashmaize/initialize is taking a long time, so set the sample size to the minimum
    // Rom::new takes 1.5s
    // let mut group = c.benchmark_group("ashmaize");
    // group.sample_size(10);
    // group.bench_function("Rom::new", |b| {
    //     b.iter(|| {
    //         Rom::new(
    //             b"password",
    //             RomGenerationType::TwoStep {
    //                 pre_size: 16 * MB,
    //                 mixing_numbers: 4,
    //             },
    //             1 * GB,
    //         )
    //     })
    // });
    // group.finish();

    let rom = Rom::new(
        b"password",
        RomGenerationType::TwoStep {
            pre_size: 16 * MB,
            mixing_numbers: 4,
        },
        1 * GB,
    );

    // // hash takes 720 us
    // c.bench_function("original::hash", |b| {
    //     b.iter(|| ashmaize::original::hash(b"salt", &rom, 8, 256))
    // });

    // blake2 crate hash slightly faster than cryptoxide
    c.bench_function("b2::hash", |b| {
        b.iter(|| ashmaize::b2::hash(b"salt", &rom, 8, 256))
    });

    // // simd hash same speed as cryptoxide one
    // c.bench_function("simd::hash", |b| {
    //     b.iter(|| ashmaize::simd::hash(b"salt", &rom, 8, 256))
    // });

    // Benchmark the CPU version running 64 hashes sequentially
    c.bench_function("b2::hash x64 (CPU)", |b| {
        b.iter(|| {
            for i in 0..64 {
                let salt = format!("salt{}", i).into_bytes();
                ashmaize::b2::hash(&salt, &rom, 8, 256);
            }
        })
    });

    // Benchmark the Metal GPU version running 64 hashes in parallel (if available)
    #[cfg(target_os = "macos")]
    {
        use ashmaize::metal::MetalAshmaize;
        if let Some(metal_ashmaize) = MetalAshmaize::new() {
            let metal_instance = metal_ashmaize; // Store the instance
            c.bench_function("metal::hash x64 (GPU)", |b| {
                b.iter(|| {
                    let mut salts = Vec::new();
                    for i in 0..64 {
                        salts.push(format!("salt{}", i).into_bytes());
                    }
                    // Convert to slices for the function
                    let salt_refs: Vec<&[u8]> = salts.iter().map(|v| v.as_slice()).collect();
                    let _ = metal_instance.hash_parallel(&salt_refs, &rom, 8, 256);
                })
            });
        } else {
            // Fallback benchmark that will show a warning
            c.bench_function("metal::hash x64 (GPU) - NOT AVAILABLE", |b| {
                b.iter(|| {
                    // This is just a placeholder since Metal is not available
                    std::hint::black_box(());
                })
            });
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        c.bench_function("metal::hash x64 (GPU) - PLATFORM NOT SUPPORTED", |b| {
            b.iter(|| {
                // This is just a placeholder since Metal is not available on this platform
                std::hint::black_box(());
            })
        });
    }

    // // VM::new takes 2 us
    // c.bench_function("VM::new", |b| {
    //     b.iter(|| ashmaize::original::VM::new(&rom.digest, 256, b"salt"))
    // });

    // // VM::execute takes 100 us (and is executed 8 times) is CLEARLY the thing to optimize
    // // VM::finalize takes 1 us
    // let mut vm = ashmaize::original::VM::new(&rom.digest, 256, b"salt");
    // c.bench_function("VM::execute", |b| b.iter(|| vm.execute(&rom, 256)));
    // c.bench_function("VM::finalize", |b| b.iter(|| vm.clone().finalize()));

    // // Look at inside the VM::execute function.
    // // It seems that all three of shuffle / step / post_instructions
    // // would be worth optimizing equally.
    // //
    // // program.shuffle takes 20 us
    // let mut vm1 = ashmaize::original::VM::new(&rom.digest, 256, b"salt");
    // c.bench_function("program.shuffle", |b| {
    //     b.iter(|| vm1.program.shuffle(&vm1.prog_seed))
    // });

    // // VM.step (x256) takes 27 us
    // let mut vm2 = ashmaize::original::VM::new(&rom.digest, 256, b"salt");
    // vm2.program.shuffle(&vm2.prog_seed);
    // c.bench_function("VM.step (x256)", |b| {
    //     b.iter(|| {
    //         for _ in 0..256 {
    //             vm2.step(&rom)
    //         }
    //     })
    // });

    // // VM.post_instructions takes 35 us
    // let mut vm3 = ashmaize::original::VM::new(&rom.digest, 256, b"salt");
    // c.bench_function("VM.post_instructions", |b| {
    //     b.iter(|| vm3.post_instructions())
    // });

    // c.bench_function("RandomX/initialize", |b| {
    //     b.iter(|| {
    //         RandomXVM::new(
    //             RandomXFlag::FLAG_DEFAULT,
    //             Some(RandomXCache::new(RandomXFlag::FLAG_DEFAULT, b"key").unwrap()),
    //             None,
    //         )
    //     })
    // });

    // let vm = RandomXVM::new(
    //     RandomXFlag::FLAG_DEFAULT,
    //     Some(RandomXCache::new(RandomXFlag::FLAG_DEFAULT, b"key").unwrap()),
    //     None,
    // )
    // .unwrap();
    // c.bench_function("RandomX/hash", |b| b.iter(|| vm.calculate_hash(b"data")));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
