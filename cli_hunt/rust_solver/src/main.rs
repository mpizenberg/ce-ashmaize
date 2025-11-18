use ashmaize::b2::hash as cpu_hash;
use ashmaize::metal::MetalAshmaize;
use ashmaize::rom::RomLike;
use ashmaize::{Rom, RomGenerationType};
use clap::Parser;

pub const MB: usize = 1024 * 1024;
pub const GB: usize = 1024 * MB;
const BATCH_SIZE: usize = 10000; // Number of hashes to compute per GPU batch

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

    // Initialize AshMaize ROM (full version for CPU verification)
    let rom = init_rom(&args.no_pre_mine);

    // Create light ROM for GPU (memory-optimized)
    let light_rom = rom.shrink();
    eprintln!(
        "ROM initialized: full size = {} MB, light size = {} MB",
        rom.data().len() / MB,
        light_rom.data().len() / MB
    );

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

    // Initialize Metal GPU
    let metal = MetalAshmaize::new().expect("Failed to initialize Metal GPU");
    eprintln!("Metal GPU initialized successfully");

    let mut current_nonce = 0u64;
    let mut batch_count = 0u64;

    loop {
        // Generate batch of preimages
        let mut preimages: Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);
        let batch_start_nonce = current_nonce;

        for i in 0..BATCH_SIZE {
            let nonce = batch_start_nonce + i as u64;
            let preimage = format!("{:016x}{}", nonce, &suffix);
            preimages.push(preimage.into_bytes());
        }

        // Convert to slices for Metal API
        let preimage_slices: Vec<&[u8]> = preimages.iter().map(|p| p.as_slice()).collect();

        // Compute hashes on GPU using LightRom
        let hash_results = metal
            .hash(&preimage_slices, &light_rom, 8, 256)
            .expect("GPU hashing failed");

        // Check results
        for (i, hash_result) in hash_results.iter().enumerate() {
            if hash_structure_good(hash_result, difficulty_mask) {
                let winning_nonce = batch_start_nonce + i as u64;
                let winning_preimage = &preimages[i];

                // Verify against CPU implementation
                eprintln!("Found candidate nonce: {:016x}", winning_nonce);
                eprintln!("Verifying with CPU implementation...");
                let cpu_result = cpu_hash(winning_preimage, &light_rom, 8, 256);

                if cpu_result != *hash_result {
                    eprintln!("VERIFICATION FAILED!");
                    eprintln!("GPU hash: {:02x?}", &hash_result[..8]);
                    eprintln!("CPU hash: {:02x?}", &cpu_result[..8]);
                    panic!("GPU and CPU hashes do not match!");
                }

                eprintln!("Verification successful!");
                println!("{:016x}", winning_nonce);
                return;
            }
        }

        current_nonce += BATCH_SIZE as u64;
        batch_count += 1;

        if batch_count % 100 == 0 {
            eprintln!(
                "Processed {} batches ({} hashes)...",
                batch_count, current_nonce
            );
        }
    }
}
