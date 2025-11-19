use ashmaize::b2::hash as cpu_hash;
#[cfg(target_os = "macos")]
use ashmaize::metal::MetalAshmaize;
use ashmaize::{Rom, RomGenerationType};
use clap::Parser;
use rayon::prelude::*;
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

pub const MB: usize = 1024 * 1024;
pub const GB: usize = 1024 * MB;

#[cfg(target_os = "macos")]
const GPU_BATCH_SIZE: usize = 10000;
#[cfg(target_os = "macos")]
const GPU_NONCE_START: u64 = 1 << 52; // Start GPU mining from a high nonce to avoid collision with CPU

mod tests;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    address: String,
    #[arg(long)]
    challenge_id: String,
    #[arg(long)]
    difficulty: String,
    #[arg(long)]
    no_pre_mine: String,
    #[arg(long)]
    latest_submission: String,
    #[arg(long)]
    no_pre_mine_hour: String,
    #[arg(long)]
    cpu_threads: Option<usize>,
}

pub fn hash_structure_good(hash: &[u8], difficulty_mask: u32) -> bool {
    if hash.len() < 4 {
        return false;
    }
    let hash_prefix = u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]);
    (hash_prefix & !difficulty_mask) == 0
}

pub fn init_rom(no_pre_mine_hex: &str) -> Rom {
    Rom::new(
        no_pre_mine_hex.as_bytes(),
        RomGenerationType::TwoStep {
            pre_size: 16 * MB,
            mixing_numbers: 4,
        },
        1 * GB,
    )
}

fn main() {
    let args = Args::parse();

    // --- Common Mining Setup ---
    let light_rom = Arc::new(init_rom(&args.no_pre_mine).shrink());
    let difficulty_mask = u32::from_str_radix(&args.difficulty, 16).unwrap();
    let suffix = Arc::new(format!(
        "{}{}{}{}{}{}",
        args.address,
        args.challenge_id,
        args.difficulty,
        args.no_pre_mine,
        args.latest_submission,
        args.no_pre_mine_hour
    ));

    // This atomic variable holds the winning nonce. It's initialized to MAX (a sentinel value).
    // The first thread to find a solution will `compare_exchange` it. This also acts as the "solution found" flag.
    let winning_nonce = Arc::new(AtomicU64::new(u64::MAX));

    // --- Thread and Concurrency Setup ---
    let num_cpu_threads = args.cpu_threads.unwrap_or_else(|| {
        let num_cores = num_cpus::get();
        ((num_cores as f64 * 0.8).floor() as usize).max(1)
    });

    #[cfg(target_os = "macos")]
    let gpu_available = {
        // --- Check GPU Availability ---
        let gpu = MetalAshmaize::new();
        if gpu.is_none() {
            eprintln!("Warning: Metal GPU not available. Mining will use CPU only.");
        }
        gpu
    };

    #[cfg(not(target_os = "macos"))]
    eprintln!("GPU mining not available on this platform. Using CPU only.");

    thread::scope(|s| {
        #[cfg(target_os = "macos")]
        {
            // --- Spawn GPU Worker Thread (only if GPU is available) ---
            if let Some(metal) = gpu_available {
                s.spawn({
                let winning_nonce = Arc::clone(&winning_nonce);
                let light_rom = Arc::clone(&light_rom);
                let suffix = Arc::clone(&suffix);
                move || {
                    eprintln!("Starting GPU mining from nonce {}...", GPU_NONCE_START);
                    let mut current_nonce = GPU_NONCE_START;

                    loop {
                        // Stop if another thread has found a solution
                        if winning_nonce.load(Ordering::Relaxed) != u64::MAX {
                            break;
                        }

                        let mut preimages: Vec<Vec<u8>> = Vec::with_capacity(GPU_BATCH_SIZE);
                        for i in 0..GPU_BATCH_SIZE {
                            preimages.push(
                                format!("{:016x}{}", current_nonce + i as u64, &*suffix)
                                    .into_bytes(),
                            );
                        }

                        let preimage_slices: Vec<&[u8]> =
                            preimages.iter().map(|p| p.as_slice()).collect();
                        let hash_results = metal
                            .hash(&preimage_slices, &*light_rom, 8, 256)
                            .expect("GPU hashing failed");

                        let mut found_in_batch = false;
                        for (i, hash_result) in hash_results.iter().enumerate() {
                            if hash_structure_good(hash_result, difficulty_mask) {
                                let found_nonce = current_nonce + i as u64;
                                // Atomically try to set the winning nonce.
                                if winning_nonce
                                    .compare_exchange(
                                        u64::MAX,
                                        found_nonce,
                                        Ordering::SeqCst,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    eprintln!(
                                        "\nSolution found by GPU at nonce: {:016x}",
                                        found_nonce
                                    );
                                }
                                // A solution is found (either by us or another thread). Stop work.
                                found_in_batch = true;
                                break;
                            }
                        }

                        if found_in_batch {
                            break; // Exit the main GPU loop
                        }

                        current_nonce += GPU_BATCH_SIZE as u64;
                    }
                }
            });
            }
        }

        // --- Run CPU Workers on Main Thread using Rayon ---
        eprintln!("Starting CPU mining with {} threads...", num_cpu_threads);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpu_threads)
            .build()
            .unwrap();
        pool.install(|| {
            (0..num_cpu_threads as u64)
                .into_par_iter()
                .for_each(|thread_id| {
                    let mut local_nonce = thread_id;
                    let stride = num_cpu_threads as u64;
                    let mut preimage = String::with_capacity(16 + suffix.len());

                    // Loop until a solution is found by any thread
                    while winning_nonce.load(Ordering::Relaxed) == u64::MAX {
                        preimage.clear();
                        write!(&mut preimage, "{:016x}{}", local_nonce, &*suffix).unwrap();
                        let hash_result = cpu_hash(preimage.as_bytes(), &*light_rom, 8, 256);

                        if hash_structure_good(&hash_result, difficulty_mask) {
                            // Atomically try to set the winning nonce. If we succeed, we are the winner.
                            if winning_nonce
                                .compare_exchange(
                                    u64::MAX,
                                    local_nonce,
                                    Ordering::SeqCst,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                            {
                                eprintln!(
                                    "\nSolution found by CPU thread {} at nonce: {:016x}",
                                    thread_id, local_nonce
                                );
                            }
                            // Break the loop whether we were the first or not, since a solution is now found.
                            break;
                        }
                        local_nonce += stride;
                    }
                });
        });
    }); // `thread::scope` waits for all threads to finish

    // --- Final Result ---
    let final_nonce = winning_nonce.load(Ordering::Relaxed);
    if final_nonce != u64::MAX {
        println!("{:016x}", final_nonce);
        std::process::exit(0);
    } else {
        eprintln!("No solution found.");
        std::process::exit(1);
    }
}
