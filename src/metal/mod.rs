//! GPU-accelerated Ashmaize hash implementation using Metal
//! This module provides GPU acceleration for the Ashmaize hash algorithm for Apple Silicon

use crate::rom::Rom;
use metal::{CommandQueue, ComputePipelineState, Device, MTLSize};
use std::ffi::c_void;
use std::ptr;

pub struct MetalAshmaize {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
    blake2b_pipeline_state: ComputePipelineState, // Added for blake2b test kernel
    hprime_pipeline_state: ComputePipelineState,  // Added for hprime test kernel
    post_instructions_pipeline_state: ComputePipelineState, // Added for post_instructions test kernel
    vm_init_pipeline_state: ComputePipelineState,           // Added for vm_init test kernel
    execute_program_pipeline_state: ComputePipelineState, // Added for execute_one_instruction test kernel
    execute_one_instruction_pipeline_state: ComputePipelineState,
    finalize_pipeline_state: ComputePipelineState,
}

impl MetalAshmaize {
    /// Get the optimal storage mode for this device
    /// Uses StorageModeShared on Apple Silicon (unified memory) for better performance
    /// Falls back to StorageModeManaged on discrete GPUs
    fn storage_mode(&self) -> metal::MTLResourceOptions {
        if self.device.has_unified_memory() {
            metal::MTLResourceOptions::StorageModeShared
        } else {
            metal::MTLResourceOptions::StorageModeManaged
        }
    }

    pub fn new() -> Option<Self> {
        println!("Attempting to initialize MetalAshmaize...");

        let device = match metal::Device::system_default() {
            Some(device) => {
                println!("Successfully obtained Metal device: {}", device.name());
                device
            }
            None => {
                eprintln!("Failed to get Metal system default device");
                return None;
            }
        };

        // Create library from compiled .metallib file or source
        let library = match device.new_library_with_source(
            include_str!("ashmaize.metal"),
            &metal::CompileOptions::new(),
        ) {
            Ok(lib) => {
                println!("Successfully compiled Metal shader library");
                lib
            }
            Err(e) => {
                eprintln!("Failed to compile Metal shader from source: {}", e);
                // Print the shader source for debugging (first 500 characters)
                let shader_source = include_str!("ashmaize.metal");
                eprintln!(
                    "Shader source (first 500 chars): {}",
                    &shader_source[..std::cmp::min(500, shader_source.len())]
                );
                return None;
            }
        };

        let command_queue = device.new_command_queue();

        let kernel_function = match library.get_function("ashmaize_hash", None) {
            Ok(func) => {
                println!("Successfully retrieved kernel function 'ashmaize_hash'");
                func
            }
            Err(e) => {
                eprintln!("Failed to get function 'ashmaize_hash' from library: {}", e);
                return None;
            }
        };

        let pipeline_state = match device.new_compute_pipeline_state_with_function(&kernel_function)
        {
            Ok(state) => {
                println!("Successfully created compute pipeline state for ashmaize_hash");
                state
            }
            Err(e) => {
                eprintln!(
                    "Failed to create compute pipeline state for ashmaize_hash: {}",
                    e
                );
                return None;
            }
        };

        // --- NEW CODE FOR BLAKE2B TEST KERNEL ---
        let blake2b_kernel_function = match library.get_function("test_blake2b", None) {
            Ok(func) => {
                println!("Successfully retrieved kernel function 'test_blake2b'");
                func
            }
            Err(e) => {
                eprintln!("Failed to get function 'test_blake2b' from library: {}", e);
                return None;
            }
        };

        let blake2b_pipeline_state =
            match device.new_compute_pipeline_state_with_function(&blake2b_kernel_function) {
                Ok(state) => {
                    println!("Successfully created compute pipeline state for test_blake2b");
                    state
                }
                Err(e) => {
                    eprintln!(
                        "Failed to create compute pipeline state for test_blake2b: {}",
                        e
                    );
                    return None;
                }
            };

        // --- NEW CODE FOR HPRIME TEST KERNEL ---
        let hprime_kernel_function = match library.get_function("test_hprime", None) {
            Ok(func) => {
                println!("Successfully retrieved kernel function 'test_hprime'");
                func
            }
            Err(e) => {
                eprintln!("Failed to get function 'test_hprime' from library: {}", e);
                return None;
            }
        };

        let hprime_pipeline_state =
            match device.new_compute_pipeline_state_with_function(&hprime_kernel_function) {
                Ok(state) => {
                    println!("Successfully created compute pipeline state for test_hprime");
                    state
                }
                Err(e) => {
                    eprintln!(
                        "Failed to create compute pipeline state for test_hprime: {}",
                        e
                    );
                    return None;
                }
            };

        // --- NEW CODE FOR POST_INSTRUCTIONS TEST KERNEL ---
        let post_instructions_kernel_function =
            match library.get_function("test_post_instructions", None) {
                Ok(func) => {
                    println!("Successfully retrieved kernel function 'test_post_instructions'");
                    func
                }
                Err(e) => {
                    eprintln!(
                        "Failed to get function 'test_post_instructions' from library: {}",
                        e
                    );
                    return None;
                }
            };

        let post_instructions_pipeline_state = match device
            .new_compute_pipeline_state_with_function(&post_instructions_kernel_function)
        {
            Ok(state) => {
                println!("Successfully created compute pipeline state for test_post_instructions");
                state
            }
            Err(e) => {
                eprintln!(
                    "Failed to create compute pipeline state for test_post_instructions: {}",
                    e
                );
                return None;
            }
        };

        // --- NEW CODE FOR VM_INIT TEST KERNEL ---
        let vm_init_kernel_function = match library.get_function("test_vm_init", None) {
            Ok(func) => {
                println!("Successfully retrieved kernel function 'test_vm_init'");
                func
            }
            Err(e) => {
                eprintln!("Failed to get function 'test_vm_init' from library: {}", e);
                return None;
            }
        };

        let vm_init_pipeline_state =
            match device.new_compute_pipeline_state_with_function(&vm_init_kernel_function) {
                Ok(state) => {
                    println!("Successfully created compute pipeline state for test_vm_init");
                    state
                }
                Err(e) => {
                    eprintln!(
                        "Failed to create compute pipeline state for test_vm_init: {}",
                        e
                    );
                    return None;
                }
            };

        // ---------------------------
        let execute_program_kernel_function =
            match library.get_function("test_execute_program", None) {
                Ok(func) => {
                    println!("Successfully retrieved kernel function 'test_execute_program'");
                    func
                }
                Err(e) => {
                    eprintln!(
                        "Failed to get function 'test_execute_program' from library: {}",
                        e
                    );
                    return None;
                }
            };

        let execute_program_pipeline_state = match device
            .new_compute_pipeline_state_with_function(&execute_program_kernel_function)
        {
            Ok(state) => {
                println!("Successfully created compute pipeline state for test_execute_program");
                state
            }
            Err(e) => {
                eprintln!(
                    "Failed to create compute pipeline state for test_execute_program: {}",
                    e
                );
                return None;
            }
        };
        // ---------------------------

        // --- NEW CODE FOR EXECUTE_ONE_INSTRUCTION TEST KERNEL ---
        let execute_one_instruction_kernel_function = match library
            .get_function("test_execute_one_instruction", None)
        {
            Ok(func) => {
                println!("Successfully retrieved kernel function 'test_execute_one_instruction'");
                func
            }
            Err(e) => {
                eprintln!(
                    "Failed to get function 'test_execute_one_instruction' from library: {}",
                    e
                );
                return None;
            }
        };

        let execute_one_instruction_pipeline_state = match device
            .new_compute_pipeline_state_with_function(&execute_one_instruction_kernel_function)
        {
            Ok(state) => {
                println!(
                    "Successfully created compute pipeline state for test_execute_one_instruction"
                );
                state
            }
            Err(e) => {
                eprintln!(
                    "Failed to create compute pipeline state for test_execute_one_instruction: {}",
                    e
                );
                return None;
            }
        };
        // ---------------------------

        // --- NEW CODE FOR FINALIZE TEST KERNEL ---
        let finalize_kernel_function = match library.get_function("test_vm_finalize", None) {
            Ok(func) => {
                println!("Successfully retrieved kernel function 'test_vm_finalize'");
                func
            }
            Err(e) => {
                eprintln!(
                    "Failed to get function 'test_vm_finalize' from library: {}",
                    e
                );
                return None;
            }
        };

        let finalize_pipeline_state =
            match device.new_compute_pipeline_state_with_function(&finalize_kernel_function) {
                Ok(state) => {
                    println!("Successfully created compute pipeline state for test_vm_finalize");
                    state
                }
                Err(e) => {
                    eprintln!(
                        "Failed to create compute pipeline state for test_vm_finalize: {}",
                        e
                    );
                    return None;
                }
            };
        // ---------------------------

        println!("MetalAshmaize initialized successfully");
        Some(Self {
            device,
            command_queue,
            pipeline_state,
            blake2b_pipeline_state,
            hprime_pipeline_state,
            post_instructions_pipeline_state,
            vm_init_pipeline_state,
            execute_program_pipeline_state,
            execute_one_instruction_pipeline_state,
            finalize_pipeline_state,
        })
    }

    pub fn hash_with_timing<R: crate::rom::RomLike>(
        &self,
        salts: &[&[u8]],
        rom: &R,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> Result<(Vec<[u8; 64]>, std::time::Duration, std::time::Duration, std::time::Duration), Box<dyn std::error::Error>> {
        use std::time::Instant;

        if salts.is_empty() {
            return Ok((Vec::new(), std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO));
        }

        let num_salts = salts.len();
        let salt_len = salts[0].len();
        if salts.iter().any(|s| s.len() != salt_len) {
            return Err("All salts must have the same length for GPU hashing.".into());
        }

        let concatenated_salts: Vec<u8> = salts.iter().flat_map(|s| *s).cloned().collect();
        let storage_mode = self.storage_mode();

        // TIMING: Buffer creation
        let buffer_start = Instant::now();
        let rom_buffer = self.device.new_buffer_with_data(
            rom.data().as_ptr() as *const c_void,
            rom.data().len() as u64,
            storage_mode,
        );
        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom.digest().0.as_ptr() as *const c_void,
            rom.digest().0.len() as u64,
            storage_mode,
        );
        let salt_buffer = self.device.new_buffer_with_data(
            concatenated_salts.as_ptr() as *const c_void,
            concatenated_salts.len() as u64,
            storage_mode,
        );

        // Constant buffers
        let salt_len_data = salt_len as u32;
        let salt_len_buffer = self.device.new_buffer_with_data(
            &salt_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&salt_len_data) as u64,
            storage_mode,
        );

        let rom_size_data = rom.original_len() as u32;
        let rom_size_buffer = self.device.new_buffer_with_data(
            &rom_size_data as *const u32 as *const c_void,
            std::mem::size_of_val(&rom_size_data) as u64,
            storage_mode,
        );

        let nb_instrs_data = nb_instrs;
        let nb_instrs_buffer = self.device.new_buffer_with_data(
            &nb_instrs_data as *const u32 as *const c_void,
            std::mem::size_of_val(&nb_instrs_data) as u64,
            storage_mode,
        );

        let nb_loops_data = nb_loops;
        let nb_loops_buffer = self.device.new_buffer_with_data(
            &nb_loops_data as *const u32 as *const c_void,
            std::mem::size_of_val(&nb_loops_data) as u64,
            storage_mode,
        );

        // Output buffer for the final hashes
        let final_hash_buffer = self.device.new_buffer(
            (num_salts * 64) as u64, // 64 bytes per hash
            storage_mode,
        );

        // Program buffers - one per thread (saves 5KB thread-local memory per thread)
        // Max program size: 256 instructions * 20 bytes/instr = 5120 bytes
        let program_buffer_size = 5120u64;
        let program_buffers = self.device.new_buffer(
            program_buffer_size * num_salts as u64,
            storage_mode,
        );

        // DEBUG: Counters buffer - 7 uint32_t per thread
        let debug_counters = self.device.new_buffer(
            (7 * std::mem::size_of::<u32>() * num_salts) as u64,
            storage_mode,
        );

        // Command encoding
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);

        compute_encoder.set_buffer(0, Some(&rom_buffer), 0);
        compute_encoder.set_buffer(1, Some(&rom_digest_buffer), 0);
        compute_encoder.set_buffer(2, Some(&salt_buffer), 0);
        compute_encoder.set_buffer(3, Some(&salt_len_buffer), 0);
        compute_encoder.set_buffer(4, Some(&rom_size_buffer), 0);
        compute_encoder.set_buffer(5, Some(&nb_instrs_buffer), 0);
        compute_encoder.set_buffer(6, Some(&nb_loops_buffer), 0);
        compute_encoder.set_buffer(7, Some(&final_hash_buffer), 0);
        compute_encoder.set_buffer(8, Some(&program_buffers), 0);
        compute_encoder.set_buffer(9, Some(&debug_counters), 0);

        let grid_size = MTLSize {
            width: num_salts as u64,
            height: 1,
            depth: 1,
        };

        let threadgroup_width = self.pipeline_state.max_total_threads_per_threadgroup();
        let threadgroup_size = MTLSize {
            width: threadgroup_width.min(num_salts as u64),
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();

        let buffer_time = buffer_start.elapsed();

        // TIMING: GPU execution
        let gpu_start = Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let gpu_time = gpu_start.elapsed();

        // TIMING: Result readback
        let readback_start = Instant::now();
        let mut results = Vec::with_capacity(num_salts);
        let result_ptr = final_hash_buffer.contents() as *const u8;
        for i in 0..num_salts {
            let mut hash = [0u8; 64];
            unsafe {
                ptr::copy_nonoverlapping(result_ptr.add(i * 64), hash.as_mut_ptr(), 64);
            }
            results.push(hash);
        }
        let readback_time = readback_start.elapsed();

        // DEBUG: Read counters from debug buffer
        #[derive(Debug, Clone)]
        struct DebugCounters {
            blake2b_calls: u32,
            mem_accesses: u32,
            special1_hits: u32,
            special1_misses: u32,
            special2_hits: u32,
            special2_misses: u32,
            instructions: u32,
        }

        let debug_ptr = debug_counters.contents() as *const u32;
        let mut debug_data = Vec::with_capacity(num_salts);
        for i in 0..num_salts {
            unsafe {
                let offset = i * 7;
                debug_data.push(DebugCounters {
                    blake2b_calls: *debug_ptr.add(offset),
                    mem_accesses: *debug_ptr.add(offset + 1),
                    special1_hits: *debug_ptr.add(offset + 2),
                    special1_misses: *debug_ptr.add(offset + 3),
                    special2_hits: *debug_ptr.add(offset + 4),
                    special2_misses: *debug_ptr.add(offset + 5),
                    instructions: *debug_ptr.add(offset + 6),
                });
            }
        }

        // Calculate averages
        if !debug_data.is_empty() {
            let avg_blake2b = debug_data.iter().map(|d| d.blake2b_calls as u64).sum::<u64>() / debug_data.len() as u64;
            let avg_mem = debug_data.iter().map(|d| d.mem_accesses as u64).sum::<u64>() / debug_data.len() as u64;
            let avg_special1_hits = debug_data.iter().map(|d| d.special1_hits as u64).sum::<u64>() / debug_data.len() as u64;
            let avg_special1_misses = debug_data.iter().map(|d| d.special1_misses as u64).sum::<u64>() / debug_data.len() as u64;
            let avg_special2_hits = debug_data.iter().map(|d| d.special2_hits as u64).sum::<u64>() / debug_data.len() as u64;
            let avg_special2_misses = debug_data.iter().map(|d| d.special2_misses as u64).sum::<u64>() / debug_data.len() as u64;
            let avg_instructions = debug_data.iter().map(|d| d.instructions as u64).sum::<u64>() / debug_data.len() as u64;

            let total_special1 = avg_special1_hits + avg_special1_misses;
            let special1_hit_rate = if total_special1 > 0 {
                (avg_special1_hits as f64 / total_special1 as f64) * 100.0
            } else {
                0.0
            };

            let total_special2 = avg_special2_hits + avg_special2_misses;
            let special2_hit_rate = if total_special2 > 0 {
                (avg_special2_hits as f64 / total_special2 as f64) * 100.0
            } else {
                0.0
            };

            println!("\n=== GPU KERNEL PROFILING (Average per hash) ===");
            println!("Blake2b calls:       {} operations", avg_blake2b);
            println!("Memory accesses:     {} ROM reads", avg_mem);
            println!("Instructions:        {} executed", avg_instructions);
            println!("\nCache Performance:");
            println!("  Special1 hits:     {} ({:.1}% hit rate)", avg_special1_hits, special1_hit_rate);
            println!("  Special1 misses:   {}", avg_special1_misses);
            println!("  Special2 hits:     {} ({:.1}% hit rate)", avg_special2_hits, special2_hit_rate);
            println!("  Special2 misses:   {}", avg_special2_misses);
            println!("\nEstimated Time Breakdown (rough approximation):");
            println!("  Blake2b:     ~{:.1}% of time ({} calls @ ~0.5μs each)",
                (avg_blake2b as f64 / (avg_blake2b + avg_mem + avg_instructions) as f64) * 100.0, avg_blake2b);
            println!("  Memory:      ~{:.1}% of time ({} accesses @ ~0.2μs each)",
                (avg_mem as f64 / (avg_blake2b + avg_mem + avg_instructions) as f64) * 100.0, avg_mem);
            println!("  Other:       ~{:.1}% of time\n",
                (avg_instructions as f64 / (avg_blake2b + avg_mem + avg_instructions) as f64) * 100.0);
        }

        Ok((results, buffer_time, gpu_time, readback_time))
    }

    pub fn hash<R: crate::rom::RomLike>(
        &self,
        salts: &[&[u8]],
        rom: &R,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> Result<Vec<[u8; 64]>, Box<dyn std::error::Error>> {
        let (results, _buffer_time, _gpu_time, _readback_time) =
            self.hash_with_timing(salts, rom, nb_loops, nb_instrs)?;
        Ok(results)
    }

    pub fn test_blake2b_kernel(
        &self,
        input: &[u8],
    ) -> Result<[u8; 64], Box<dyn std::error::Error>> {
        let storage_mode = self.storage_mode();
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            input.len() as u64,
            storage_mode,
        );
        let output_buffer = self.device.new_buffer(
            64, // Blake2b output is 64 bytes
            storage_mode,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_buffer(0, Some(&input_buffer), 0);
        compute_encoder.set_buffer(1, Some(&output_buffer), 0);

        let input_len_data = input.len() as u32;
        let input_len_buffer = self.device.new_buffer_with_data(
            &input_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&input_len_data) as u64,
            storage_mode,
        );
        compute_encoder.set_buffer(2, Some(&input_len_buffer), 0);

        compute_encoder.set_compute_pipeline_state(&self.blake2b_pipeline_state);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        }; // Only one thread needed for this test
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut result = [0u8; 64];
        unsafe {
            ptr::copy_nonoverlapping(
                output_buffer.contents() as *const u8,
                result.as_mut_ptr(),
                64,
            );
        }

        Ok(result)
    }

    pub fn test_hprime_kernel(
        &self,
        input: &[u8],
        output_len: u32,
        input_len: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            input.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let output_buffer = self.device.new_buffer(
            output_len as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_buffer(0, Some(&input_buffer), 0);
        compute_encoder.set_buffer(1, Some(&output_buffer), 0);

        let output_len_data = output_len;
        let output_len_buffer = self.device.new_buffer_with_data(
            &output_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&output_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        compute_encoder.set_buffer(2, Some(&output_len_buffer), 0);

        let input_len_data = input_len;
        let input_len_buffer = self.device.new_buffer_with_data(
            &input_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&input_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        compute_encoder.set_buffer(3, Some(&input_len_buffer), 0);

        compute_encoder.set_compute_pipeline_state(&self.hprime_pipeline_state);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut result_vec = vec![0u8; output_len as usize];
        unsafe {
            ptr::copy_nonoverlapping(
                output_buffer.contents() as *const u8,
                result_vec.as_mut_ptr(),
                output_len as usize,
            );
        }

        Ok(result_vec)
    }

    /// Tests the `vm_init` logic on the GPU
    pub fn test_vm_init_kernel(
        &self,
        rom_digest: &[u8; 64],
        salt: &[u8],
    ) -> Result<([u64; crate::b2::NB_REGS], u32, [u8; 64], u32, u32), Box<dyn std::error::Error>>
    {
        // Input buffers
        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom_digest.as_ptr() as *const c_void,
            rom_digest.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salt_buffer = self.device.new_buffer_with_data(
            salt.as_ptr() as *const c_void,
            salt.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salt_len_data = salt.len() as u32;
        let salt_len_buffer = self.device.new_buffer_with_data(
            &salt_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&salt_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Output buffers
        let output_regs_buffer = self.device.new_buffer(
            (crate::b2::NB_REGS * std::mem::size_of::<u64>()) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let output_ip_buffer = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let output_prog_seed_buffer = self
            .device
            .new_buffer(64 as u64, metal::MTLResourceOptions::StorageModeManaged);
        let output_memory_counter_buffer = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let output_loop_counter_buffer = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_buffer(0, Some(&rom_digest_buffer), 0);
        compute_encoder.set_buffer(1, Some(&salt_buffer), 0);
        compute_encoder.set_buffer(2, Some(&salt_len_buffer), 0);
        compute_encoder.set_buffer(3, Some(&output_regs_buffer), 0);
        compute_encoder.set_buffer(4, Some(&output_prog_seed_buffer), 0);
        compute_encoder.set_buffer(5, Some(&output_ip_buffer), 0);
        compute_encoder.set_buffer(6, Some(&output_memory_counter_buffer), 0);
        compute_encoder.set_buffer(7, Some(&output_loop_counter_buffer), 0);

        compute_encoder.set_compute_pipeline_state(&self.vm_init_pipeline_state);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut final_regs = [0u64; crate::b2::NB_REGS];
        unsafe {
            ptr::copy_nonoverlapping(
                output_regs_buffer.contents() as *const u64,
                final_regs.as_mut_ptr(),
                crate::b2::NB_REGS,
            );
        }

        let mut final_ip = 0u32;
        unsafe {
            ptr::copy_nonoverlapping(
                output_ip_buffer.contents() as *const u32,
                &mut final_ip as *mut u32,
                1,
            );
        }

        let mut final_prog_seed = [0u8; 64];
        unsafe {
            ptr::copy_nonoverlapping(
                output_prog_seed_buffer.contents() as *const u8,
                final_prog_seed.as_mut_ptr(),
                64,
            );
        }

        let mut final_memory_counter = 0u32;
        unsafe {
            ptr::copy_nonoverlapping(
                output_memory_counter_buffer.contents() as *const u32,
                &mut final_memory_counter as *mut u32,
                1,
            );
        }

        let mut final_loop_counter = 0u32;
        unsafe {
            ptr::copy_nonoverlapping(
                output_loop_counter_buffer.contents() as *const u32,
                &mut final_loop_counter as *mut u32,
                1,
            );
        }

        Ok((
            final_regs,
            final_ip,
            final_prog_seed,
            final_memory_counter,
            final_loop_counter,
        ))
    }

    /// Tests the `post_instructions` logic on the GPU
    pub fn test_post_instructions_kernel(
        &self,
        rom_digest: &[u8; 64],
        salt: &[u8],
    ) -> Result<([u64; crate::b2::NB_REGS], [u8; 64], u32), Box<dyn std::error::Error>> {
        // Input buffers
        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom_digest.as_ptr() as *const c_void,
            rom_digest.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let salt_buffer = self.device.new_buffer_with_data(
            salt.as_ptr() as *const c_void,
            salt.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let salt_len_data = salt.len() as u32;
        let salt_len_buffer = self.device.new_buffer_with_data(
            &salt_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&salt_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Output buffers
        let output_regs_buffer = self.device.new_buffer(
            (crate::b2::NB_REGS * std::mem::size_of::<u64>()) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let output_prog_seed_buffer = self
            .device
            .new_buffer(64 as u64, metal::MTLResourceOptions::StorageModeManaged);
        let output_loop_counter_buffer = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // inputs
        compute_encoder.set_buffer(0, Some(&rom_digest_buffer), 0);
        compute_encoder.set_buffer(1, Some(&salt_buffer), 0);
        compute_encoder.set_buffer(2, Some(&salt_len_buffer), 0);
        // outputs
        compute_encoder.set_buffer(3, Some(&output_regs_buffer), 0);
        compute_encoder.set_buffer(4, Some(&output_prog_seed_buffer), 0);
        compute_encoder.set_buffer(5, Some(&output_loop_counter_buffer), 0);

        compute_encoder.set_compute_pipeline_state(&self.post_instructions_pipeline_state);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut final_regs = [0u64; crate::b2::NB_REGS];
        unsafe {
            ptr::copy_nonoverlapping(
                output_regs_buffer.contents() as *const u64,
                final_regs.as_mut_ptr(),
                crate::b2::NB_REGS,
            );
        }

        let mut final_prog_seed = [0u8; 64];
        unsafe {
            ptr::copy_nonoverlapping(
                output_prog_seed_buffer.contents() as *const u8,
                final_prog_seed.as_mut_ptr(),
                64,
            );
        }

        let mut final_loop_counter = 0u32;
        unsafe {
            ptr::copy_nonoverlapping(
                output_loop_counter_buffer.contents() as *const u32,
                &mut final_loop_counter as *mut u32,
                1,
            );
        }

        Ok((final_regs, final_prog_seed, final_loop_counter))
    }

    /// Tests the `execute_program` logic on the GPU
    pub fn test_execute_program_kernel<R: crate::rom::RomLike>(
        &self,
        rom: &R,
        rom_digest: &[u8; 64],
        salt: &[u8],
        nb_instrs: u32,
    ) -> Result<[u64; crate::b2::NB_REGS], Box<dyn std::error::Error>> {
        // Input buffers
        let rom_buffer = self.device.new_buffer_with_data(
            rom.data().as_ptr() as *const c_void,
            rom.data().len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom_digest.as_ptr() as *const c_void,
            rom_digest.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let salt_buffer = self.device.new_buffer_with_data(
            salt.as_ptr() as *const c_void,
            salt.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salt_len_data = salt.len() as u32;
        let salt_len_buffer = self.device.new_buffer_with_data(
            &salt_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&salt_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let rom_size_data = rom.original_len() as u32;
        let rom_size_buffer = self.device.new_buffer_with_data(
            &rom_size_data as *const u32 as *const c_void,
            std::mem::size_of_val(&rom_size_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let nb_instrs_data = nb_instrs;
        let nb_instrs_buffer = self.device.new_buffer_with_data(
            &nb_instrs_data as *const u32 as *const c_void,
            std::mem::size_of_val(&nb_instrs_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Output buffer for final registers
        let output_regs_buffer = self.device.new_buffer(
            (crate::b2::NB_REGS * std::mem::size_of::<u64>()) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Set buffers
        compute_encoder.set_buffer(0, Some(&rom_buffer), 0); // ROM array
        compute_encoder.set_buffer(1, Some(&rom_digest_buffer), 0); // ROM digest array
        compute_encoder.set_buffer(2, Some(&salt_buffer), 0); // Salt array
        compute_encoder.set_buffer(3, Some(&salt_len_buffer), 0); // Salt length
        compute_encoder.set_buffer(4, Some(&rom_size_buffer), 0); // ROM size
        compute_encoder.set_buffer(5, Some(&nb_instrs_buffer), 0); // Number of instructions
        compute_encoder.set_buffer(6, Some(&output_regs_buffer), 0); // Output registers array

        compute_encoder.set_compute_pipeline_state(&self.execute_program_pipeline_state);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut final_regs = [0u64; crate::b2::NB_REGS];
        unsafe {
            ptr::copy_nonoverlapping(
                output_regs_buffer.contents() as *const u64,
                final_regs.as_mut_ptr(),
                crate::b2::NB_REGS,
            );
        }

        Ok(final_regs)
    }

    pub fn test_execute_one_instruction(
        &self,
        rom: &Rom,
        rom_digest: &[u8; 64],
        salt: &[u8],
        nb_instrs: u32,
        prog_chunk: &[u8; 20],
    ) -> Result<TestExecOneResult, Box<dyn std::error::Error>> {
        // Input buffers
        let rom_buffer = self.device.new_buffer_with_data(
            rom.data.as_ptr() as *const c_void,
            rom.data.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom_digest.as_ptr() as *const c_void,
            rom_digest.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let salt_buffer = self.device.new_buffer_with_data(
            salt.as_ptr() as *const c_void,
            salt.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salt_len_data = salt.len() as u32;
        let salt_len_buffer = self.device.new_buffer_with_data(
            &salt_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&salt_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let rom_size_data = rom.data.len() as u32;
        let rom_size_buffer = self.device.new_buffer_with_data(
            &rom_size_data as *const u32 as *const c_void,
            std::mem::size_of_val(&rom_size_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let nb_instrs_data = nb_instrs;
        let nb_instrs_buffer = self.device.new_buffer_with_data(
            &nb_instrs_data as *const u32 as *const c_void,
            std::mem::size_of_val(&nb_instrs_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let prog_chunk_buffer = self.device.new_buffer_with_data(
            prog_chunk.as_ptr() as *const c_void,
            20,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Output buffers
        let final_regs_buffer = self.device.new_buffer(
            (32 * 8) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let final_counters_buffer = self.device.new_buffer(
            (2 * 4) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let final_prog_digest_hash_buffer = self
            .device
            .new_buffer(64, metal::MTLResourceOptions::StorageModeManaged);
        let final_mem_digest_hash_buffer = self
            .device
            .new_buffer(64, metal::MTLResourceOptions::StorageModeManaged);

        // Command submission
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&self.execute_one_instruction_pipeline_state);

        // input buffers
        compute_encoder.set_buffer(0, Some(&rom_buffer), 0); // ROM array
        compute_encoder.set_buffer(1, Some(&rom_digest_buffer), 0); // ROM digest array
        compute_encoder.set_buffer(2, Some(&salt_buffer), 0); // Salt array
        compute_encoder.set_buffer(3, Some(&salt_len_buffer), 0); // Salt length
        compute_encoder.set_buffer(4, Some(&rom_size_buffer), 0); // ROM size
        compute_encoder.set_buffer(5, Some(&nb_instrs_buffer), 0); // Number of instructions
        compute_encoder.set_buffer(6, Some(&prog_chunk_buffer), 0);
        // output buffers
        compute_encoder.set_buffer(7, Some(&final_regs_buffer), 0);
        compute_encoder.set_buffer(8, Some(&final_counters_buffer), 0);
        compute_encoder.set_buffer(9, Some(&final_prog_digest_hash_buffer), 0);
        compute_encoder.set_buffer(10, Some(&final_mem_digest_hash_buffer), 0);

        // Dispatch a single thread
        compute_encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        compute_encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let mut final_regs = [0u64; 32];
        let mut final_counters = [0u32; 2];
        let mut final_prog_digest_hash = [0u8; 64];
        let mut final_mem_digest_hash = [0u8; 64];

        unsafe {
            ptr::copy_nonoverlapping(
                final_regs_buffer.contents() as *const u64,
                final_regs.as_mut_ptr(),
                32,
            );
            ptr::copy_nonoverlapping(
                final_counters_buffer.contents() as *const u32,
                final_counters.as_mut_ptr(),
                2,
            );
            ptr::copy_nonoverlapping(
                final_prog_digest_hash_buffer.contents() as *const u8,
                final_prog_digest_hash.as_mut_ptr(),
                64,
            );
            ptr::copy_nonoverlapping(
                final_mem_digest_hash_buffer.contents() as *const u8,
                final_mem_digest_hash.as_mut_ptr(),
                64,
            );
        }

        Ok(TestExecOneResult {
            final_regs,
            final_ip: final_counters[0],
            final_memory_counter: final_counters[1],
            final_prog_digest_hash,
            final_mem_digest_hash,
        })
    }

    pub fn test_vm_finalize_kernel(
        &self,
        rom_digest: &[u8; 64],
        salt: &[u8],
    ) -> Result<[u8; 64], Box<dyn std::error::Error>> {
        // Input buffers
        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom_digest.as_ptr() as *const c_void,
            rom_digest.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salt_buffer = self.device.new_buffer_with_data(
            salt.as_ptr() as *const c_void,
            salt.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salt_len_data = salt.len() as u32;
        let salt_len_buffer = self.device.new_buffer_with_data(
            &salt_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&salt_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Output buffer
        let output_hash_buffer = self
            .device
            .new_buffer(64, metal::MTLResourceOptions::StorageModeManaged);

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_buffer(0, Some(&rom_digest_buffer), 0);
        compute_encoder.set_buffer(1, Some(&salt_buffer), 0);
        compute_encoder.set_buffer(2, Some(&salt_len_buffer), 0);
        compute_encoder.set_buffer(3, Some(&output_hash_buffer), 0);

        compute_encoder.set_compute_pipeline_state(&self.finalize_pipeline_state);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut final_hash = [0u8; 64];
        unsafe {
            ptr::copy_nonoverlapping(
                output_hash_buffer.contents() as *const u8,
                final_hash.as_mut_ptr(),
                64,
            );
        }

        Ok(final_hash)
    }
}

/// The output of a single instruction execution test on the GPU.
#[derive(Debug, PartialEq, Eq)]
pub struct TestExecOneResult {
    pub final_regs: [u64; 32],
    pub final_ip: u32,
    pub final_memory_counter: u32,
    pub final_prog_digest_hash: [u8; 64],
    pub final_mem_digest_hash: [u8; 64],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b2::RomGenerationType;
    use crate::b2::{VM, argon2};
    use crate::rom::RomLike;
    use blake2::{Blake2b512, Digest}; // Import for CPU Blake2b

    #[test]
    fn test_ashmaize_hash_single() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        // 1. Create a full Rom
        let rom_key = b"comparison_seed";
        let full_rom = Rom::new(
            rom_key,
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        // 2. Create a LightRom from it
        let light_rom = full_rom.shrink();
        assert!(light_rom.data().len() < full_rom.data().len());

        // let salt_ok: [u8; _] = [0x73, 0x61, 0x6c, 0x74, 0x5f, 0x30];
        let salt_ko: [u8; _] = [0x73, 0x61, 0x6c, 0x74, 0x5f, 0x30, 0x00];
        let salt = salt_ko;

        // 3. Calculate the expected hash on the CPU for comparison
        let expected_hash = crate::b2::hash(&salt, &full_rom, 8, 256);

        // 4. Calculate the hash on the GPU using the LightRom
        let gpu_result = metal_ashmaize
            .hash(&[&salt], &light_rom, 8, 256)
            .expect("GPU hash failed")
            .into_iter()
            .next()
            .unwrap();

        // 5. Compare the results
        assert_eq!(
            expected_hash, gpu_result,
            "GPU hash does not match expected hash"
        );
    }

    #[test]
    fn test_ashmaize_hash_multiple() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        let rom = Rom::new(
            b"comparison_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salts: Vec<&[u8]> = vec![b"test_salt_1", b"test_salt_2", b"a_slightly_longer_salt_3"];

        // To test the parallel implementation, all salts must have the same length.
        // Let's find the max length and pad the others.
        let max_len = salts.iter().map(|s| s.len()).max().unwrap_or(0);
        let padded_salts: Vec<Vec<u8>> = salts
            .iter()
            .map(|s| {
                let mut padded = s.to_vec();
                padded.resize(max_len, 0);
                padded
            })
            .collect();

        let salt_slices: Vec<&[u8]> = padded_salts.iter().map(|s| s.as_slice()).collect();

        // CPU computation
        let cpu_results: Vec<[u8; 64]> = salt_slices
            .iter()
            .map(|s| crate::b2::hash(s, &rom, 8, 256))
            .collect();

        // GPU computation
        let gpu_results = metal_ashmaize
            .hash(&salt_slices, &rom, 8, 256)
            .expect("GPU hash failed");

        assert_eq!(
            cpu_results.len(),
            gpu_results.len(),
            "Mismatch in number of hashes returned"
        );

        for i in 0..cpu_results.len() {
            assert_eq!(
                cpu_results[i], gpu_results[i],
                "Mismatch for salt index {}",
                i
            );
        }
    }

    #[test]
    fn diagnose_timing_breakdown() {
        use std::time::Instant;

        println!("\n=== TIMING BREAKDOWN: Where Does The Time Go? ===\n");

        let init_start = Instant::now();
        let metal = MetalAshmaize::new().expect("MetalAshmaize initialization failed");
        println!("Metal initialization: {:?}", init_start.elapsed());

        let rom_start = Instant::now();
        let rom = Rom::new(
            b"diagnostic",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            1024 * 1024,
        );
        println!("ROM creation (1MB): {:?}", rom_start.elapsed());

        // Test with batch of 100
        let salts: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("salt_{}", i).into_bytes())
            .collect();
        let max_len = salts.iter().map(|s| s.len()).max().unwrap();
        let padded: Vec<Vec<u8>> = salts
            .into_iter()
            .map(|mut s| {
                s.resize(max_len, 0);
                s
            })
            .collect();
        let refs: Vec<&[u8]> = padded.iter().map(|s| s.as_slice()).collect();

        println!("\nCalling hash() with batch_size=100...");
        let total_start = Instant::now();
        let _result = metal.hash(&refs, &rom, 8, 256).unwrap();
        let total_time = total_start.elapsed();

        println!("Total GPU time: {:?}", total_time);
        println!("Per-hash time: {:.2} ms", total_time.as_secs_f64() * 1000.0 / 100.0);

        println!("\n=== Analysis ===");
        println!("Most time is likely in buffer creation + GPU execution.");
        println!("We need to instrument hash() to see the breakdown.");
    }

    #[test]
    fn diagnose_bottleneck() {
        use std::time::Instant;

        println!("\n=== DIAGNOSTIC: Finding the Bottleneck ===\n");

        let metal = MetalAshmaize::new().expect("MetalAshmaize initialization failed");
        let rom = Rom::new(
            b"diagnostic",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            1024 * 1024, // 1MB ROM
        );

        let batch_sizes = [1, 10, 100, 1000];

        println!("{:<12} | {:<15} | {:<15} | {:<15} | {:<10}",
                 "Batch Size", "GPU Total", "GPU per Hash", "CPU per Hash", "Speedup");
        println!("{:-<80}", "");

        for &size in &batch_sizes {
            let salts: Vec<Vec<u8>> = (0..size)
                .map(|i| format!("salt_{}", i).into_bytes())
                .collect();
            let max_len = salts.iter().map(|s| s.len()).max().unwrap();
            let padded: Vec<Vec<u8>> = salts
                .into_iter()
                .map(|mut s| {
                    s.resize(max_len, 0);
                    s
                })
                .collect();
            let refs: Vec<&[u8]> = padded.iter().map(|s| s.as_slice()).collect();

            // GPU timing
            let gpu_start = Instant::now();
            let _gpu_result = metal.hash(&refs, &rom, 8, 256).unwrap();
            let gpu_total = gpu_start.elapsed();
            let gpu_per_hash = gpu_total.as_secs_f64() * 1000.0 / size as f64;

            // CPU timing (single hash for fair comparison)
            let cpu_start = Instant::now();
            let _cpu_result = crate::b2::hash(&refs[0], &rom, 8, 256);
            let cpu_per_hash = cpu_start.elapsed().as_secs_f64() * 1000.0;

            let speedup = cpu_per_hash / gpu_per_hash;

            println!(
                "{:<12} | {:>13.2?} | {:>13.2} ms | {:>13.2} ms | {:>9.2}x",
                size, gpu_total, gpu_per_hash, cpu_per_hash, speedup
            );
        }

        println!("\n=== Analysis ===");
        println!("Expected GPU speedup for batch=1000: ~100x (CPU limited, GPU parallel)");
        println!("If speedup is < 10x: GPU has fundamental efficiency problem");
        println!("If speedup is < 2x: GPU is barely working or has massive overhead");
        println!("If GPU is SLOWER: Something is very wrong\n");
    }

    #[test]
    fn profile_gpu_single() {
        use std::time::{Duration, Instant};

        let metal = MetalAshmaize::new().expect("Failed to initialize Metal");
        let rom = Rom::new(
            b"profile",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            1024 * 1024, // 1MB ROM
        );

        // Warm up GPU - get everything initialized
        println!("Warming up GPU...");
        for _ in 0..5 {
            let _ = metal.hash(&[b"warmup"], &rom, 8, 256);
        }

        println!("\n=== READY TO PROFILE ===");
        println!("You have 5 seconds to start Instruments recording...");
        println!("1. Open Instruments.app");
        println!("2. Choose 'Metal System Trace' template");
        println!("3. Select this process (cargo test)");
        println!("4. Click Record button");
        std::thread::sleep(Duration::from_secs(5));

        println!("\n▶ PROFILING WINDOW STARTED - Recording for 10 seconds");
        let start = Instant::now();
        let mut iterations = 0;

        // Run for 10 seconds to get good profiling data
        while start.elapsed() < Duration::from_secs(10) {
            let _ = metal.hash(&[b"salt"], &rom, 8, 256);
            iterations += 1;
        }

        let elapsed = start.elapsed();
        println!("■ PROFILING WINDOW ENDED");
        println!("Completed {} iterations in {:?}", iterations, elapsed);
        println!("Average time per hash: {:.2} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);

        println!("\nYou can stop Instruments recording now.");
        println!("Keeping process alive for 3 seconds to capture final data...");
        std::thread::sleep(Duration::from_secs(3));
    }

    #[test]
    fn profile_gpu_batch() {
        use std::time::{Duration, Instant};

        let metal = MetalAshmaize::new().expect("Failed to initialize Metal");
        let rom = Rom::new(
            b"profile_batch",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            1024 * 1024, // 1MB ROM
        );

        // Prepare batch of 1000 salts
        let salts: Vec<Vec<u8>> = (0..1000)
            .map(|i| format!("salt_{:04}", i).into_bytes())
            .collect();
        let max_len = salts.iter().map(|s| s.len()).max().unwrap();
        let padded: Vec<Vec<u8>> = salts
            .into_iter()
            .map(|mut s| {
                s.resize(max_len, 0);
                s
            })
            .collect();
        let refs: Vec<&[u8]> = padded.iter().map(|s| s.as_slice()).collect();

        // Warm up
        println!("Warming up GPU...");
        let _ = metal.hash(&refs, &rom, 8, 256);

        println!("\n=== READY TO PROFILE (BATCH MODE) ===");
        println!("You have 5 seconds to start Instruments recording...");
        println!("This will run 1000 parallel hashes repeatedly");
        std::thread::sleep(Duration::from_secs(5));

        println!("\n▶ PROFILING WINDOW STARTED - Recording for 10 seconds");
        let start = Instant::now();
        let mut iterations = 0;

        while start.elapsed() < Duration::from_secs(10) {
            let _ = metal.hash(&refs, &rom, 8, 256);
            iterations += 1;
        }

        let elapsed = start.elapsed();
        println!("■ PROFILING WINDOW ENDED");
        println!("Completed {} batch iterations in {:?}", iterations, elapsed);
        println!("Total hashes: {}", iterations * 1000);
        println!("Average time per batch: {:.2} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);

        println!("\nYou can stop Instruments recording now.");
        std::thread::sleep(Duration::from_secs(3));
    }

    #[test]
    fn detailed_timing_breakdown() {
        println!("\n=== DETAILED TIMING BREAKDOWN ===\n");

        let metal = MetalAshmaize::new().expect("MetalAshmaize initialization failed");
        let rom = Rom::new(
            b"timing",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            1024 * 1024, // 1MB ROM
        );

        let batch_sizes = [1, 10, 100, 1000];

        println!("{:<12} | {:<15} | {:<15} | {:<15} | {:<15}",
                 "Batch Size", "Buffer Create", "GPU Execute", "Readback", "Total");
        println!("{:-<95}", "");

        for &size in &batch_sizes {
            let salts: Vec<Vec<u8>> = (0..size)
                .map(|i| format!("salt_{}", i).into_bytes())
                .collect();
            let max_len = salts.iter().map(|s| s.len()).max().unwrap();
            let padded: Vec<Vec<u8>> = salts
                .into_iter()
                .map(|mut s| {
                    s.resize(max_len, 0);
                    s
                })
                .collect();
            let refs: Vec<&[u8]> = padded.iter().map(|s| s.as_slice()).collect();

            // Use timing version
            let (_results, buffer_time, gpu_time, readback_time) =
                metal.hash_with_timing(&refs, &rom, 8, 256).unwrap();

            let total_time = buffer_time + gpu_time + readback_time;

            println!(
                "{:<12} | {:>13.2?} | {:>13.2?} | {:>13.2?} | {:>13.2?}",
                size, buffer_time, gpu_time, readback_time, total_time
            );
        }

        println!("\n=== Analysis ===");
        println!("If Buffer Create dominates: Memory allocation/transfer overhead");
        println!("If GPU Execute dominates: Actual kernel execution inefficiency");
        println!("If Readback dominates: Memory copy-back overhead");
        println!("Look for where the time is actually spent!\n");
    }

    #[test]
    fn benchmark_cpu_vs_gpu() {
        use std::time::Instant;

        println!("\n--- Ashmaize CPU vs. GPU Benchmark ---");

        // 1. Setup
        const NB_LOOPS: u32 = 8;
        const NB_INSTRS: u32 = 256;
        const ROM_SIZE: usize = 1 * 1024 * 1024; // 1MB
        let batch_sizes = [1, 10, 100, 1000, 10000, 20000];

        println!(
            "Parameters: nb_loops={}, nb_instrs={}, rom_size={}MB",
            NB_LOOPS,
            NB_INSTRS,
            ROM_SIZE / (1024 * 1024)
        );
        println!("{:-<55}", "");
        println!(
            "{: >10} | {: >20} | {: >20}",
            "Batch Size", "CPU Time", "GPU Time"
        );
        println!("{:-<55}", "");

        let full_rom = Rom::new(
            b"benchmark_seed",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            ROM_SIZE,
        );
        let light_rom = full_rom.shrink();
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        let max_batch_size = *batch_sizes.iter().max().unwrap_or(&0);
        let salts: Vec<Vec<u8>> = (0..max_batch_size)
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

        // 2. Execution Loop
        for &batch_size in &batch_sizes {
            // CPU Benchmark
            let cpu_start = Instant::now();
            let cpu_results: Vec<_> = salt_slices[..batch_size]
                .iter()
                .map(|s| crate::b2::hash(s, &light_rom, NB_LOOPS, NB_INSTRS))
                .collect();
            let cpu_duration = cpu_start.elapsed();

            // GPU Benchmark
            let gpu_start = Instant::now();
            let gpu_results = metal_ashmaize
                .hash(&salt_slices[..batch_size], &light_rom, NB_LOOPS, NB_INSTRS)
                .expect("GPU hash failed");
            let gpu_duration = gpu_start.elapsed();

            // Verification
            assert_eq!(cpu_results[0], gpu_results[0]);

            // Print Results
            println!(
                "{: >10} | {: >20.2?} | {: >20.2?}",
                batch_size, cpu_duration, gpu_duration
            );
        }
        println!("{:-<55}", "");
    }

    #[test]
    fn test_light_rom_gpu_hashing() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        // 1. Create a full Rom
        let full_rom = Rom::new(
            b"light_rom_gpu_test_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        // 2. Create a LightRom from it
        let light_rom = full_rom.shrink();
        assert!(light_rom.data().len() < full_rom.data().len());

        let salt = b"light_rom_gpu_salt";

        // 3. Calculate the expected hash on the CPU for comparison
        let expected_hash = crate::b2::hash(salt, &full_rom, 8, 256);

        // 4. Calculate the hash on the GPU using the LightRom
        let gpu_result = metal_ashmaize
            .hash(&[salt], &light_rom, 8, 256)
            .expect("GPU hash with LightRom failed")
            .into_iter()
            .next()
            .unwrap();

        // 5. Compare the results
        assert_eq!(
            expected_hash, gpu_result,
            "GPU hash with LightRom does not match expected hash"
        );
    }

    #[test]
    fn test_metal_blake2b_vs_cpu() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        let test_data = b"This is a test string for Blake2b hashing on both CPU and GPU.";
        let mut hasher = Blake2b512::new();
        hasher.update(test_data);
        let cpu_result: [u8; 64] = hasher.finalize().into();

        let gpu_result = metal_ashmaize
            .test_blake2b_kernel(test_data)
            .expect("GPU Blake2b kernel failed");

        assert_eq!(cpu_result, gpu_result);

        // Test with different data
        let test_data_2 = b"Another string, slightly different length and content for Blake2b.";
        let mut hasher_2 = Blake2b512::new();
        hasher_2.update(test_data_2);
        let cpu_result_2: [u8; 64] = hasher_2.finalize().into();

        let gpu_result_2 = metal_ashmaize
            .test_blake2b_kernel(test_data_2)
            .expect("GPU Blake2b kernel 2 failed");

        assert_eq!(cpu_result_2, gpu_result_2);
    }

    #[test]
    fn test_metal_hprime_vs_cpu() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        // Test case 1: output_len <= 64
        let test_input_1 = b"hprime test case 1";
        let output_len_1: u32 = 60;
        let mut cpu_output_1 = vec![0u8; output_len_1 as usize];
        argon2::hprime(&mut cpu_output_1, test_input_1);

        let gpu_output_1 = metal_ashmaize
            .test_hprime_kernel(test_input_1, output_len_1, test_input_1.len() as u32)
            .expect("GPU hprime kernel failed for case 1");

        assert_eq!(cpu_output_1, gpu_output_1);

        // Test case 2: output_len > 64
        let test_input_2 = b"a longer input for hprime to test the multi-block case";
        let output_len_2: u32 = 200;
        let mut cpu_output_2 = vec![0u8; output_len_2 as usize];
        argon2::hprime(&mut cpu_output_2, test_input_2);

        let gpu_output_2 = metal_ashmaize
            .test_hprime_kernel(test_input_2, output_len_2, test_input_2.len() as u32)
            .expect("GPU hprime kernel failed for case 2");
        assert_eq!(cpu_output_2, gpu_output_2);

        // Test case 3: another large output_len
        let test_input_3 = b"short";
        let output_len_3: u32 = 1024;
        let mut cpu_output_3 = vec![0u8; output_len_3 as usize];
        argon2::hprime(&mut cpu_output_3, test_input_3);

        let gpu_output_3 = metal_ashmaize
            .test_hprime_kernel(test_input_3, output_len_3, test_input_3.len() as u32)
            .expect("GPU hprime kernel failed for case 3");

        assert_eq!(cpu_output_3, gpu_output_3);
    }

    #[test]
    fn test_metal_vm_init_vs_cpu() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        let rom = Rom::new(
            b"vm_init_test_seed",
            RomGenerationType::FullRandom, // Using FullRandom to keep RomDigest simple for testing
            10_240,
        );
        let nb_instrs = 256;
        let salt = b"vm_init_test_salt_lets_make_it_long_enough_just_to_be_sure";

        // CPU VM initialization
        let cpu_vm = VM::new(&rom.digest, nb_instrs, salt);

        // GPU VM initialization
        let (gpu_regs, gpu_ip, gpu_prog_seed, gpu_memory_counter, gpu_loop_counter) =
            metal_ashmaize
                .test_vm_init_kernel(&rom.digest.0, salt)
                .expect("GPU vm_init kernel failed");

        // Compare results, excluding prog_digest and mem_digest as requested
        assert_eq!(cpu_vm.ip, gpu_ip);
        assert_eq!(cpu_vm.loop_counter, gpu_loop_counter);
        assert_eq!(cpu_vm.memory_counter, gpu_memory_counter);
        assert_eq!(cpu_vm.regs, gpu_regs);
        assert_eq!(cpu_vm.prog_seed, gpu_prog_seed);
    }

    #[test]
    fn test_metal_instruction() {
        let instructions_to_test: Vec<(&str, [u8; 20])> = vec![
            ("ADD R1, R0, R8", {
                let mut bytes = [0u8; 20];
                bytes[0] = 1; // ADD
                bytes[1] = 0x00; // Reg, Reg
                let rs: u16 = (0 << 10) | (8 << 5) | 1; // r1=0, r2=8, r3=1
                bytes[2..4].copy_from_slice(&rs.to_be_bytes());
                bytes
            }),
            ("MUL R2, R3, 10 (lit)", {
                let mut bytes = [0u8; 20];
                bytes[0] = 40; // MUL
                bytes[1] = 0x09; // Reg, Literal
                let rs: u16 = (3 << 10) | (0 << 5) | 2; // r1=3, r2=unused, r3=2
                bytes[2..4].copy_from_slice(&rs.to_be_bytes());
                bytes[12..20].copy_from_slice(&10u64.to_le_bytes()); // lit2
                bytes
            }),
            ("XOR R4, Mem(0x100), R5", {
                let mut bytes = [0u8; 20];
                bytes[0] = 148; // XOR
                bytes[1] = 0x50; // Mem, Reg
                let rs: u16 = (0 << 10) | (5 << 5) | 4; // r1=unused, r2=5, r3=4
                bytes[2..4].copy_from_slice(&rs.to_be_bytes());
                bytes[4..12].copy_from_slice(&0x100u64.to_le_bytes()); // lit1 for addr
                bytes
            }),
            ("ISQRT R9, R10", {
                let mut bytes = [0u8; 20];
                bytes[0] = 128; // ISQRT
                bytes[1] = 0x00; // Reg, (op2 unused)
                let rs: u16 = (10 << 10) | (0 << 5) | 9; // r1=10, r2=unused, r3=9
                bytes[2..4].copy_from_slice(&rs.to_be_bytes());
                bytes
            }),
        ];

        let metal_ashmaize = MetalAshmaize::new().unwrap();
        let rom = Rom::new(b"seed", RomGenerationType::FullRandom, 16 * 1024);
        let nb_instrs = 256;
        let salt = b"instruction_salt";

        for (name, instruction_bytes) in instructions_to_test {
            println!("Testing instruction: {}", name);

            let mut vm_cpu = VM::new(&rom.digest, nb_instrs, salt);

            vm_cpu.program.get_instructions_mut()[0..20].copy_from_slice(&instruction_bytes);

            // 3. Execute on CPU (Oracle)
            crate::b2::execute_one_instruction(&mut vm_cpu, &rom);
            let expected_regs = vm_cpu.regs;
            let expected_prog_hash = vm_cpu.prog_digest.clone().finalize();
            let expected_mem_hash = vm_cpu.mem_digest.clone().finalize();

            // 4. Execute on GPU
            let result_gpu = metal_ashmaize
                .test_execute_one_instruction(
                    &rom,
                    &rom.digest.0,
                    salt,
                    nb_instrs,
                    &instruction_bytes,
                )
                .unwrap();

            // 5. Compare results
            assert_eq!(
                expected_regs, result_gpu.final_regs,
                "Register states do not match for instruction '{}'!",
                name
            );
            assert_eq!(
                expected_prog_hash[..],
                result_gpu.final_prog_digest_hash[..],
                "Program digests do not match for instruction '{}'!",
                name
            );
            assert_eq!(
                expected_mem_hash[..],
                result_gpu.final_mem_digest_hash[..],
                "Memory digests do not match for instruction '{}'!",
                name
            );
        }
    }

    #[test]
    fn test_metal_post_instructions_vs_cpu() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        let rom = Rom::new(
            b"test_seed_for_post_instr",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );
        let nb_instrs = 256;
        let salt = b"post_instructions_salt";

        // Create a CPU VM and post_instructions
        let mut cpu_vm = VM::new(&rom.digest, nb_instrs, salt);
        cpu_vm.post_instructions();

        // Run GPU kernel with CPU-derived intermediate values
        let (gpu_regs, gpu_prog_seed, gpu_loop_counter) = metal_ashmaize
            .test_post_instructions_kernel(&rom.digest.0, salt)
            .expect("GPU post_instructions kernel failed");

        // Compare results
        assert_eq!(cpu_vm.regs, gpu_regs);
        assert_eq!(cpu_vm.prog_seed, gpu_prog_seed);
        assert_eq!(cpu_vm.loop_counter, gpu_loop_counter);
    }

    #[test]
    fn test_metal_execute_program_vs_cpu() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        // Create a ROM for testing
        let rom = Rom::new(
            b"execute_test_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240, // 10KB ROM
        );

        // Create the salt
        let salt = b"test_salt_execute";

        // CPU execution
        let nb_instrs = 256;
        let mut cpu_vm = VM::new(&rom.digest, nb_instrs, salt);
        cpu_vm.execute(&rom, nb_instrs);

        // GPU execution
        let gpu_final_regs = metal_ashmaize
            .test_execute_program_kernel(&rom, &rom.digest.0, salt, nb_instrs)
            .expect("GPU execute_one_instruction kernel failed");

        // Compare results
        assert_eq!(cpu_vm.regs, gpu_final_regs);
    }

    #[test]
    fn test_metal_finalize_vs_cpu() {
        let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize initialization failed");

        let rom = Rom::new(b"finalize_test_seed", RomGenerationType::FullRandom, 1024);
        let nb_instrs = 256;
        let salt = b"finalize_test_salt";

        // CPU VM finalization
        let cpu_vm = VM::new(&rom.digest, nb_instrs, salt);
        let cpu_hash = cpu_vm.finalize();

        // GPU VM finalization
        let gpu_hash = metal_ashmaize
            .test_vm_finalize_kernel(&rom.digest.0, salt)
            .expect("GPU vm_finalize kernel failed");

        assert_eq!(cpu_hash, gpu_hash);
    }
}
