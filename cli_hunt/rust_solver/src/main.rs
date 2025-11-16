// use ashmaize::{hash, Rom, RomGenerationType};
use ashmaize::b2::hash; // Use the blake2 implementation (slightly faster)
use ashmaize::{Rom, RomGenerationType};
use clap::Parser;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const NUM_THREADS: u64 = 5;
pub const MB: usize = 1024 * 1024;
pub const GB: usize = 1024 * MB;

mod tests;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    address: String,
    #[arg(long)]
    challenge_id: String,
    #[arg(long)]
    difficulty: String, // This is a hexadecimal string representing the bitmask for the required zero prefix
    #[arg(long)]
    no_pre_mine: String,
    #[arg(long)]
    latest_submission: String,
    #[arg(long)]
    no_pre_mine_hour: String,
}

pub fn hash_structure_good(hash: &[u8], difficulty_mask: u32) -> bool {
    if hash.len() < 4 {
        return false; // Not enough bytes to apply a u32 mask
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

    // Initialize AshMaize ROM
    let rom = init_rom(&args.no_pre_mine);

    // Parse difficulty from hex string to u32 mask
    let difficulty_mask = u32::from_str_radix(&args.difficulty, 16).unwrap();

    // Compute suffix once
    let suffix = format!(
        "{}{}{}{}{}{}",
        args.address,
        args.challenge_id,
        args.difficulty,
        args.no_pre_mine,
        args.latest_submission,
        args.no_pre_mine_hour
    );

    // Share ROM across threads (read-only, no mutex needed)
    let rom = Arc::new(rom);

    let found = Arc::new(AtomicBool::new(false));
    let result_nonce = Arc::new(AtomicU64::new(0));
    let start_nonce = 0;

    (0..NUM_THREADS).into_par_iter().for_each(|thread_id| {
        let rom = Arc::clone(&rom);
        let mut local_nonce = start_nonce + thread_id as u64;
        let stride = NUM_THREADS as u64;

        // Reuse preimage buffer across iterations
        let mut preimage = String::with_capacity(16 + suffix.len());

        while !found.load(Ordering::Relaxed) {
            preimage.clear();
            use std::fmt::Write;
            write!(&mut preimage, "{:016x}{}", local_nonce, &suffix).unwrap();

            // Each hash call allocates ~15-20KB temporarily
            let hash_result = hash(preimage.as_bytes(), &rom, 8, 256);

            if hash_structure_good(&hash_result, difficulty_mask) {
                found.store(true, Ordering::Relaxed);
                result_nonce.store(local_nonce, Ordering::Relaxed);
                break;
            }

            local_nonce += stride;
        }
    });

    if found.load(Ordering::Relaxed) {
        println!("{:016x}", result_nonce.load(Ordering::Relaxed));
    }
}
