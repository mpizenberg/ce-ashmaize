//! GPU-accelerated Ashmaize hash implementation using Metal
//! This module provides GPU acceleration for the Ashmaize hash algorithm for Apple Silicon

use crate::b2::VM;
use crate::rom::{Rom, RomDigest};
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
}

impl MetalAshmaize {
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
        })
    }

    pub fn hash_parallel(
        &self,
        salts: &[&[u8]],
        rom: &Rom,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> Result<Vec<[u8; 64]>, Box<dyn std::error::Error>> {
        let num_hashes = salts.len();
        let program_size = nb_instrs as usize * crate::b2::INSTR_SIZE;

        // Prepare ROM data
        let rom_data = &rom.data;
        let rom_digest = &rom.digest.0;

        // Collect initial prog_seed for each salt. The Metal kernel will use this to
        // initialize its own `prog_seed` and shuffle the program for the first loop.
        let mut initial_prog_seeds = Vec::new();
        for salt in salts {
            let vm = VM::new(&RomDigest(*rom_digest), nb_instrs, salt);
            initial_prog_seeds.extend_from_slice(&vm.prog_seed);
        }

        // Create a template program (all zeros) to be shuffled on the GPU.
        // The instructions in this buffer are irrelevant, as they will be
        // overwritten by the shuffled program data computed by hprime on GPU.
        let template_program = crate::b2::Program::new(nb_instrs);
        let template_program_bytes = template_program.get_instructions();

        // Prepare salt data (pad each to 32 bytes)
        let mut all_salts = Vec::new();
        for salt in salts {
            let mut padded_salt = vec![0u8; 32];
            let len = std::cmp::min(salt.len(), 32);
            padded_salt[..len].copy_from_slice(&salt[..len]);
            all_salts.extend_from_slice(&padded_salt);
        }

        // Create buffers
        let rom_buffer = self.device.new_buffer_with_data(
            rom_data.as_ptr() as *const c_void,
            rom_data.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let rom_digest_buffer = self.device.new_buffer_with_data(
            rom_digest.as_ptr() as *const c_void,
            rom_digest.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let salts_buffer = self.device.new_buffer_with_data(
            all_salts.as_ptr() as *const c_void,
            all_salts.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Buffer for initial prog_seeds
        let initial_prog_seeds_buffer = self.device.new_buffer_with_data(
            initial_prog_seeds.as_ptr() as *const c_void,
            initial_prog_seeds.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Pass the template program to GPU, it will be shuffled per thread.
        // Each thread will need its own mutable program buffer. This buffer
        // will be an input/output buffer for the kernel.
        let programs_buffer_size = (num_hashes * program_size) as u64;
        let programs_buffer = self.device.new_buffer(
            programs_buffer_size,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        // Initialize programs_buffer with the template program (all zeros) replicated for each thread
        unsafe {
            let ptr = programs_buffer.contents() as *mut u8;
            for i in 0..num_hashes {
                ptr::copy_nonoverlapping(
                    template_program_bytes.as_ptr(),
                    ptr.add(i * program_size),
                    program_size,
                );
            }
        }

        let results_buffer = self.device.new_buffer(
            (num_hashes * 64) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Set buffers
        compute_encoder.set_buffer(0, Some(&rom_buffer), 0); // ROM array
        compute_encoder.set_buffer(1, Some(&results_buffer), 0); // Results array
        compute_encoder.set_buffer(2, Some(&salts_buffer), 0); // Salts array
        compute_encoder.set_buffer(3, Some(&rom_digest_buffer), 0); // ROM digest array
        compute_encoder.set_buffer(4, Some(&programs_buffer), 0); // Programs array (mutable, per-thread)
        compute_encoder.set_buffer(5, Some(&initial_prog_seeds_buffer), 0); // Initial prog_seeds for each thread

        // Create parameter buffer
        let params_data = [
            rom_data.len() as u32, // ROM size at offset 0
            nb_loops,              // nb_loops at offset 4
            nb_instrs,             // nb_instrs at offset 8
            program_size as u32,   // program_size at offset 12
        ];
        let params_buffer = self.device.new_buffer_with_data(
            params_data.as_ptr() as *const c_void,
            std::mem::size_of_val(&params_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );

        compute_encoder.set_buffer(6, Some(&params_buffer), 0); // ROM size
        compute_encoder.set_buffer(7, Some(&params_buffer), std::mem::size_of::<u32>() as u64); // nb_loops
        compute_encoder.set_buffer(
            8,
            Some(&params_buffer),
            (2 * std::mem::size_of::<u32>()) as u64,
        ); // nb_instrs
        compute_encoder.set_buffer(
            9,
            Some(&params_buffer),
            (3 * std::mem::size_of::<u32>()) as u64,
        ); // program_size

        // Configure thread groups
        let max_threads_per_group = self.pipeline_state.max_total_threads_per_threadgroup();
        let threads_per_threadgroup = std::cmp::min(256, max_threads_per_group) as u64; // Use up to 256 threads per group

        // Calculate how many thread groups we need
        let num_thread_groups =
            (num_hashes as u64 + threads_per_threadgroup - 1) / threads_per_threadgroup;

        let threadgroup_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: num_thread_groups,
            height: 1,
            depth: 1,
        };

        // Dispatch compute kernel
        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let results_ptr = results_buffer.contents() as *const u8;
        let mut results = Vec::new();

        unsafe {
            for i in 0..num_hashes {
                let mut result = [0u8; 64];
                ptr::copy_nonoverlapping(results_ptr.add(i * 64), result.as_mut_ptr(), 64);
                results.push(result);
            }
        }

        Ok(results)
    }

    pub fn test_blake2b_kernel(
        &self,
        input: &[u8],
    ) -> Result<[u8; 64], Box<dyn std::error::Error>> {
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            input.len() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let output_buffer = self.device.new_buffer(
            64, // Blake2b output is 64 bytes
            metal::MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_buffer(0, Some(&input_buffer), 0);
        compute_encoder.set_buffer(1, Some(&output_buffer), 0);

        let input_len_data = input.len() as u32;
        let input_len_buffer = self.device.new_buffer_with_data(
            &input_len_data as *const u32 as *const c_void,
            std::mem::size_of_val(&input_len_data) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
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
    pub fn test_execute_program_kernel(
        &self,
        rom: &Rom,
        rom_digest: &[u8; 64],
        salt: &[u8],
        nb_instrs: u32,
    ) -> Result<[u64; crate::b2::NB_REGS], Box<dyn std::error::Error>> {
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
}

// Helper function to hash on GPU with fallback to CPU
pub fn hash_gpu_or_cpu(salt: &[u8], rom: &Rom, nb_loops: u32, nb_instrs: u32) -> [u8; 64] {
    match MetalAshmaize::new() {
        Some(metal_ashmaize) => {
            metal_ashmaize
                .hash_parallel(&[salt], rom, nb_loops, nb_instrs)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
            // match metal_ashmaize.hash_parallel(&[salt], rom, nb_loops, nb_instrs) {
            //     Ok(results) => results.into_iter().next().unwrap_or_else(|| {
            //         // Fallback to CPU implementation if GPU fails
            //         crate::b2::hash(salt, rom, nb_loops, nb_instrs)
            //     }),
            //     Err(_) => {
            //         // Fallback to CPU implementation
            //         crate::b2::hash(salt, rom, nb_loops, nb_instrs)
            //     }
            // }
        }
        None => {
            // No GPU available, use CPU implementation
            crate::b2::hash(salt, rom, nb_loops, nb_instrs)
        }
    }
}

pub fn hash_gpu(salt: &[u8], rom: &Rom, nb_loops: u32, nb_instrs: u32) -> [u8; 64] {
    let metal_ashmaize = MetalAshmaize::new().expect("MetalAshmaize::new returned None!");
    metal_ashmaize
        .hash_parallel(&[salt], rom, nb_loops, nb_instrs)
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b2::argon2;
    use crate::{Rom, RomGenerationType};
    use blake2::{Blake2b512, Digest}; // Import for CPU Blake2b

    #[test]
    fn test_gpu_hash_basic() {
        let rom = Rom::new(
            b"test_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salt = b"test_salt";
        let result = hash_gpu_or_cpu(salt, &rom, 8, 256);

        // Should produce a 64-byte result
        assert_eq!(result.len(), 64);

        // Should be different from zero array
        assert!(!result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_gpu_hash_consistency() {
        let rom = Rom::new(
            b"test_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salt = b"consistent_test";
        let result1 = hash_gpu_or_cpu(salt, &rom, 8, 256);
        let result2 = hash_gpu_or_cpu(salt, &rom, 8, 256);

        // Same input should produce same output
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_gpu_hash_vs_cpu() {
        let rom = Rom::new(
            b"comparison_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salt = b"comparison_test";
        let cpu_result = crate::b2::hash(salt, &rom, 8, 256);
        let gpu_result = hash_gpu(salt, &rom, 8, 256);

        assert_eq!(cpu_result, gpu_result);
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
}
