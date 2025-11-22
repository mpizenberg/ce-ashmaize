#[cfg(test)]
mod basic_tests {
    use crate::{hash_structure_good, init_rom}; // Import hash_structure_good, init_rom and constants
    use ashmaize::original::hash;

    #[test]
    fn validate_example_solution() {
        let address = "addr1q84h0q756f6fslk9y3v48kztxug9nk2es3wvw3dyumfy2qwvpuzhn97jay38vh4sspz45ukzavalsm0tf6q4gx39rl8sc7f5rf";
        let challenge_id = "**D01C17";
        let difficulty_str = "00007FFF";
        let no_pre_mine = "e8a195800bae57517c85955a784faa6162051f41ef86bcb93be0c3e01a9b63c8";
        let latest_submission = "2025-10-31T15:59:59.000Z";
        let no_pre_mine_hour = "967125414";
        let nonce_hex = "001af01e65703909";
        let nonce = u64::from_str_radix(nonce_hex, 16).unwrap();

        // Initialize AshMaize ROM
        let rom = init_rom(no_pre_mine);

        // Construct the preimage
        let preimage = format!(
            "{0:016x}{1}{2}{3}{4}{5}{6}",
            nonce,
            address,
            challenge_id,
            difficulty_str,
            no_pre_mine,
            latest_submission,
            no_pre_mine_hour
        );

        // Hash the preimage
        let hash_result = hash(preimage.as_bytes(), &rom, 8, 256);
        println!("DEBUG: Hash result: {:?}", hash_result);

        // Parse difficulty from hex string to u32 mask
        let difficulty_mask = u32::from_str_radix(difficulty_str, 16).unwrap();

        // Validate the hash against the difficulty
        assert!(
            hash_structure_good(&hash_result, difficulty_mask),
            "Hash does not meet difficulty requirements"
        );
    }
}
