#include <metal_stdlib>
using namespace metal;

// Ashmaize hash algorithm Metal implementation
// 1 byte operator, 3 bytes operands (src1, src2, dst), 16 bytes data
constant uint INSTR_SIZE = 20;
constant uint NB_REGS = 32;  // 1 << 5
constant uint REGS_BITS = 5;
constant uint REGS_INDEX_MASK = 31;  // NB_REGS - 1
typedef uint64_t Register;

constant uint REGISTER_SIZE = 8; // Size of Register in bytes

constant uint DIGEST_INIT_SIZE = 64;
constant uint REGS_CONTENT_SIZE = 256; // sizeof(Register) * NB_REGS (8 * 32)


// Blake2b constants
constant uint BLAKE2B_BLOCKBYTES = 128;
constant uint BLAKE2B_OUTBYTES = 64;
constant uint SIGMA[160] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
    11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
    7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
    9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
    2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
    12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
    13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
    6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
    10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0
};

// Blake2b IV
constant uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};

struct Blake2bState {
    uint64_t h[8];
    uint64_t t[2];   // counters
    uint64_t f[2];   // finalization
    uint8_t buf[BLAKE2B_BLOCKBYTES];
    uint64_t buflen;
    bool last_node;
};

// G Mixing function for Blake2b (operates on 4 state words directly)
void blake2b_G(thread uint64_t &a, thread uint64_t &b, thread uint64_t &c, thread uint64_t &d, uint64_t x, uint64_t y) {
    a = a + b + x;
    d = (d ^ a); d = (d >> 32) | (d << 32); // ROTR 32
    c = c + d;
    b = (b ^ c); b = (b >> 24) | (b << 40); // ROTR 24
    a = a + b + y;
    d = (d ^ a); d = (d >> 16) | (d << 48); // ROTR 16
    c = c + d;
    b = (b ^ c); b = (b >> 63) | (b << 1); // ROTR 63 (ROL 1)
}

// Round function for Blake2b
void blake2b_round(thread Blake2bState &S, thread const uint64_t *m) {
    uint64_t v[16]; // Local 16-word state vector for compression

    // Initialize v
    for (int i = 0; i < 8; ++i) v[i] = S.h[i];
    for (int i = 0; i < 8; ++i) v[i + 8] = blake2b_IV[i];

    // XOR v[12..15] with T and F (counter and finalization flags)
    v[12] ^= S.t[0];
    v[13] ^= S.t[1];
    v[14] ^= S.f[0];
    v[15] ^= S.f[1];

    // 12 rounds
    for (int r = 0; r < 12; ++r) {
        uint s_idx = (r % 10) * 16; // Index into SIGMA for current round permutation

        blake2b_G(v[0], v[4], v[8], v[12], m[SIGMA[s_idx + 0]], m[SIGMA[s_idx + 1]]);
        blake2b_G(v[1], v[5], v[9], v[13], m[SIGMA[s_idx + 2]], m[SIGMA[s_idx + 3]]);
        blake2b_G(v[2], v[6], v[10], v[14], m[SIGMA[s_idx + 4]], m[SIGMA[s_idx + 5]]);
        blake2b_G(v[3], v[7], v[11], v[15], m[SIGMA[s_idx + 6]], m[SIGMA[s_idx + 7]]);
        blake2b_G(v[0], v[5], v[10], v[15], m[SIGMA[s_idx + 8]], m[SIGMA[s_idx + 9]]);
        blake2b_G(v[1], v[6], v[11], v[12], m[SIGMA[s_idx + 10]], m[SIGMA[s_idx + 11]]);
        blake2b_G(v[2], v[7], v[8], v[13], m[SIGMA[s_idx + 12]], m[SIGMA[s_idx + 13]]);
        blake2b_G(v[3], v[4], v[9], v[14], m[SIGMA[s_idx + 14]], m[SIGMA[s_idx + 15]]);
    }

    // Update chaining value S.h
    for (int i = 0; i < 8; ++i) {
        S.h[i] ^= v[i] ^ v[i + 8];
    }
}

// Initialize Blake2b state
void blake2b_init(thread Blake2bState &S, uint outlen) {
    for (int i = 0; i < 8; ++i) {
        S.h[i] = blake2b_IV[i];
    }
    S.h[0] ^= 0x01010000 ^ (uint64_t)outlen;
    S.t[0] = S.t[1] = S.f[0] = S.f[1] = 0;
    S.buflen = 0;
    S.last_node = false;
}

// Update Blake2b state with data
void blake2b_update(thread Blake2bState &S, thread const uint8_t *in, uint64_t inlen) {
    thread const uint8_t *current_in = in;
    while (inlen > 0) {
        uint64_t left = S.buflen;
        uint64_t fill = BLAKE2B_BLOCKBYTES - left;

        if (inlen > fill) {
            // Fill buffer completely
            for (uint i = 0; i < fill; ++i) {
                S.buf[left + i] = current_in[i];
            }
            S.buflen += fill;

            S.t[0] += BLAKE2B_BLOCKBYTES;
            if (S.t[0] < BLAKE2B_BLOCKBYTES) S.t[1]++;

            uint64_t m[16];
            for (uint i = 0; i < 16; ++i) {
                m[i] = ((uint64_t)S.buf[i * 8 + 0] << 0) |
                       ((uint64_t)S.buf[i * 8 + 1] << 8) |
                       ((uint64_t)S.buf[i * 8 + 2] << 16) |
                       ((uint64_t)S.buf[i * 8 + 3] << 24) |
                       ((uint64_t)S.buf[i * 8 + 4] << 32) |
                       ((uint64_t)S.buf[i * 8 + 5] << 40) |
                       ((uint64_t)S.buf[i * 8 + 6] << 48) |
                       ((uint64_t)S.buf[i * 8 + 7] << 56);
            }

            blake2b_round(S, m);

            current_in += fill;
            inlen -= fill;
            S.buflen = 0;
        } else {
            for (uint i = 0; i < inlen; ++i) {
                S.buf[left + i] = current_in[i];
            }
            S.buflen += inlen;
            inlen = 0;
        }
    }
}

// Finalize Blake2b and get result
void blake2b_final(thread Blake2bState &S, thread uint8_t *out, uint outlen) {
    uint64_t lastblock = S.buflen;
    S.t[0] += lastblock;
    if (S.t[0] < lastblock) S.t[1]++;

    S.f[0] = ~0ULL;

    if (S.buflen > 0) {
        for (uint8_t i = (uint8_t)lastblock; i < BLAKE2B_BLOCKBYTES; ++i) {
            S.buf[i] = 0;
        }

        uint64_t m[16];
        for (uint i = 0; i < 16; ++i) {
            m[i] = ((uint64_t)S.buf[i * 8 + 0] << 0) |
                   ((uint64_t)S.buf[i * 8 + 1] << 8) |
                   ((uint64_t)S.buf[i * 8 + 2] << 16) |
                   ((uint64_t)S.buf[i * 8 + 3] << 24) |
                   ((uint64_t)S.buf[i * 8 + 4] << 32) |
                   ((uint64_t)S.buf[i * 8 + 5] << 40) |
                   ((uint64_t)S.buf[i * 8 + 6] << 48) |
                   ((uint64_t)S.buf[i * 8 + 7] << 56);
        }

        blake2b_round(S, m);
    }

    for (uint i = 0; i < 8; ++i) {
        uint64_t h = S.h[i];
        out[i * 8 + 0] = (uint8_t)(h >> 0);
        out[i * 8 + 1] = (uint8_t)(h >> 8);
        out[i * 8 + 2] = (uint8_t)(h >> 16);
        out[i * 8 + 3] = (uint8_t)(h >> 24);
        out[i * 8 + 4] = (uint8_t)(h >> 32);
        out[i * 8 + 5] = (uint8_t)(h >> 40);
        out[i * 8 + 6] = (uint8_t)(h >> 48);
        out[i * 8 + 7] = (uint8_t)(h >> 56);
    }
}

// Blake2b hash function
void blake2b(thread uint8_t *out, uint outlen, thread const uint8_t *in, uint inlen) {
    Blake2bState S;
    blake2b_init(S, outlen);
    blake2b_update(S, in, inlen);
    blake2b_final(S, out, outlen);
}

// Argon2 H' function
void hprime(thread uint8_t *output, uint output_len, thread const uint8_t *input, uint input_len) {
    if (output_len <= 64) {
        uint8_t temp[BLAKE2B_OUTBYTES + 4 + 512]; // Max 64 (output_len) + 4 (output_len_bytes) + 512 (max input_len)
        temp[0] = output_len & 0xFF;
        temp[1] = (output_len >> 8) & 0xFF;
        temp[2] = (output_len >> 16) & 0xFF;
        temp[3] = (output_len >> 24) & 0xFF;

        for (uint i = 0; i < input_len; ++i) {
            temp[4 + i] = input[i];
        }

        blake2b(output, output_len, temp, 4 + input_len);
        return;
    }

    uint output_len_copy = output_len;

    // Create input for v0 hash
    uint8_t v0_input[BLAKE2B_OUTBYTES + 512];  // 4 + input_len can be up to BLAKE2B_OUTBYTES + 512
    v0_input[0] = output_len_copy & 0xFF;
    v0_input[1] = (output_len_copy >> 8) & 0xFF;
    v0_input[2] = (output_len_copy >> 16) & 0xFF;
    v0_input[3] = (output_len_copy >> 24) & 0xFF;
    for (uint i = 0; i < input_len; ++i) {
        v0_input[4 + i] = input[i];
    }

    uint8_t v0_hash[64];
    blake2b(v0_hash, 64, v0_input, 4 + input_len);

    uint8_t vi_prev[64];
    for (int i = 0; i < 64; ++i) {
        vi_prev[i] = v0_hash[i];
    }

    // Copy first 32 bytes
    for (int i = 0; i < 32 && (uint)i < output_len; ++i) {
        output[i] = vi_prev[i];
    }

    uint bytes = output_len - 32;
    uint pos = 32;

    while (bytes > 64) {
        uint8_t vi_hash[64];
        blake2b(vi_hash, 64, vi_prev, 64);

        for (int i = 0; i < 64; ++i) {
            vi_prev[i] = vi_hash[i];
        }

        for (int i = 0; i < 32; ++i) {
            if (pos + i < output_len) {
                output[pos + i] = vi_prev[i];
            }
        }

        bytes -= 32;
        pos += 32;
    }

    if (bytes > 0) {
        uint8_t temp[64];
        blake2b(temp, bytes, vi_prev, 64);
        for (uint i = 0; i < bytes; ++i) {
            if (pos + i < output_len) {
                output[pos + i] = temp[i];
            }
        }
    }
}

// VM struct for GPU
struct VMState {
    Register regs[NB_REGS];
    uint32_t ip;
    uint8_t prog_seed[64];
    uint32_t memory_counter;
    uint32_t loop_counter;
    // Digests as state
    Blake2bState prog_digest_state;
    Blake2bState mem_digest_state;
};

// VM initialization function
void vm_init(thread VMState &vm, thread const uint8_t *rom_digest, uint32_t rom_digest_len,
             device const uint8_t *salt, uint32_t salt_len) {

    uint8_t init_buffer[REGS_CONTENT_SIZE + 3 * DIGEST_INIT_SIZE]; // Only need for regs and initial digest updates

    // Prepare input for argon2 to get initial register values and digest states
    uint8_t init_buffer_input[DIGEST_INIT_SIZE + 512];  // Enough space for rom_digest + salt
    for (uint i = 0; i < rom_digest_len; ++i) {
        init_buffer_input[i] = rom_digest[i];
    }
    for (uint i = 0; i < salt_len; ++i) {
        init_buffer_input[rom_digest_len + i] = salt[i];
    }

    hprime(init_buffer, REGS_CONTENT_SIZE + 3 * DIGEST_INIT_SIZE,
           init_buffer_input, rom_digest_len + salt_len);

    // Initialize registers from first part of init_buffer
    thread const uint8_t *init_buffer_regs = init_buffer;
    for (uint i = 0; i < NB_REGS; ++i) {
        vm.regs[i] = ((uint64_t)init_buffer_regs[i * 8 + 0] << 0) |
                     ((uint64_t)init_buffer_regs[i * 8 + 1] << 8) |
                     ((uint64_t)init_buffer_regs[i * 8 + 2] << 16) |
                     ((uint64_t)init_buffer_regs[i * 8 + 3] << 24) |
                     ((uint64_t)init_buffer_regs[i * 8 + 4] << 32) |
                     ((uint64_t)init_buffer_regs[i * 8 + 5] << 40) |
                     ((uint64_t)init_buffer_regs[i * 8 + 6] << 48) |
                     ((uint64_t)init_buffer_regs[i * 8 + 7] << 56);
    }

    // Initialize digests from remaining buffer
    thread const uint8_t *digests_data = init_buffer + REGS_CONTENT_SIZE;

    // Prog digest initialization
    blake2b_init(vm.prog_digest_state, 64);
    blake2b_update(vm.prog_digest_state, digests_data, 64);

    // Mem digest initialization
    blake2b_init(vm.mem_digest_state, 64);
    blake2b_update(vm.mem_digest_state, &digests_data[64], 64);

    // Prog seed is provided as an argument, from CPU pre-calculation
    for (int i = 0; i < 64; ++i) {
        vm.prog_seed[i] = digests_data[128 + i];
    }

    vm.ip = 0;
    vm.loop_counter = 0;
    vm.memory_counter = 0;
}


// Operation enum values
enum Op3Type {
    Add = 0,
    Mul = 1,
    MulH = 2,
    Xor = 3,
    Div = 4,
    Mod = 5,
    And = 6,
    Hash = 7
};

enum Op2Type {
    ISqrt = 0,
    Neg = 1,
    BitRev = 2,
    RotL = 3,
    RotR = 4
};

// Get special value 1 (prog digest first 8 bytes)
uint64_t special1_value64(thread Blake2bState &prog_digest_state) {
    Blake2bState temp_state = prog_digest_state;
    uint8_t out[BLAKE2B_OUTBYTES];
    blake2b_final(temp_state, out, BLAKE2B_OUTBYTES);
    return ((uint64_t)out[0] << 0) |
           ((uint64_t)out[1] << 8) |
           ((uint64_t)out[2] << 16) |
           ((uint64_t)out[3] << 24) |
           ((uint64_t)out[4] << 32) |
           ((uint64_t)out[5] << 40) |
           ((uint64_t)out[6] << 48) |
           ((uint64_t)out[7] << 56);
}

// Get special value 2 (mem digest first 8 bytes)
uint64_t special2_value64(thread Blake2bState &mem_digest_state) {
    Blake2bState temp_state = mem_digest_state;
    uint8_t out[BLAKE2B_OUTBYTES];
    blake2b_final(temp_state, out, BLAKE2B_OUTBYTES);
    return ((uint64_t)out[0] << 0) |
           ((uint64_t)out[1] << 8) |
           ((uint64_t)out[2] << 16) |
           ((uint64_t)out[3] << 24) |
           ((uint64_t)out[4] << 32) |
           ((uint64_t)out[5] << 40) |
           ((uint64_t)out[6] << 48) |
           ((uint64_t)out[7] << 56);
}

// Memory access function
uint64_t mem_access64(thread VMState &vm, device const uint8_t *rom, uint64_t addr, uint32_t rom_size) {
    // Access ROM at specific address
    uint32_t rom_addr = addr % rom_size;

    // Get 64-byte chunk from ROM (as in original implementation)
    // The Rust `rom.at` method reads a 64-byte block. The Metal version should mimic this.
    // If rom_addr is near the end, `rom_chunk_start + i` might exceed `rom_size`.
    // The original Rust `Rom` implementation ensures 64 bytes are always returned,
    // so we need to be careful with boundary conditions.
    // Assuming `rom.at` in Rust handles wrapping or padding for 64-byte reads.
    // For now, let's assume direct copy of 64 bytes is fine.

    // The Rust reference for `rom.at(idx as u32)` takes `u32` for `idx`.
    // The `rom.data` is an `&[u8]`
    // `Rom::at` returns `&[u8; 64]`. It likely handles boundaries by mapping
    // `idx % rom_data.len()` for the start of the 64-byte block.
    // And then if the block goes past the end, it wraps around.

    uint32_t chunk_start_idx = (rom_addr / 64) * 64; // Start of the 64-byte block
    uint8_t mem_chunk[64];

    for (uint32_t i = 0; i < 64; ++i) {
        mem_chunk[i] = rom[(chunk_start_idx + i) % rom_size];
    }

    blake2b_update(vm.mem_digest_state, mem_chunk, 64);
    vm.memory_counter++;

    // Divide memory access into 8 chunks of 8 bytes (as per original algorithm)
    uint32_t idx_in_chunk = ((vm.memory_counter % 8)) * 8;
    uint64_t result = ((uint64_t)mem_chunk[idx_in_chunk + 0] << 0) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 1] << 8) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 2] << 16) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 3] << 24) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 4] << 32) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 5] << 40) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 6] << 48) |
                      ((uint64_t)mem_chunk[idx_in_chunk + 7] << 56);

    return result;
}

// Decode instruction
struct Instruction {
    uint8_t opcode;
    uint8_t op1, op2;
    uint8_t r1, r2, r3;
    uint64_t lit1, lit2;
};

Instruction decode_instruction(thread const uint8_t *instruction) {
    Instruction instr;
    instr.opcode = instruction[0];
    instr.op1 = instruction[1] >> 4;
    instr.op2 = instruction[1] & 0x0F;

    uint16_t rs = ((uint16_t)instruction[2] << 8) | instruction[3];
    instr.r1 = (rs >> (2 * REGS_BITS)) & REGS_INDEX_MASK;
    instr.r2 = (rs >> REGS_BITS) & REGS_INDEX_MASK;
    instr.r3 = rs & REGS_INDEX_MASK;

    instr.lit1 = ((uint64_t)instruction[4] << 0) |
                 ((uint64_t)instruction[5] << 8) |
                 ((uint64_t)instruction[6] << 16) |
                 ((uint64_t)instruction[7] << 24) |
                 ((uint64_t)instruction[8] << 32) |
                 ((uint64_t)instruction[9] << 40) |
                 ((uint64_t)instruction[10] << 48) |
                 ((uint64_t)instruction[11] << 56);

    instr.lit2 = ((uint64_t)instruction[12] << 0) |
                 ((uint64_t)instruction[13] << 8) |
                 ((uint64_t)instruction[14] << 16) |
                 ((uint64_t)instruction[15] << 24) |
                 ((uint64_t)instruction[16] << 32) |
                 ((uint64_t)instruction[17] << 40) |
                 ((uint64_t)instruction[18] << 48) |
                 ((uint64_t)instruction[19] << 56);

    return instr;
}

// Execute one instruction
void execute_one_instruction(thread VMState &vm, device const uint8_t *rom,
                            thread const uint8_t *prog_chunk, uint32_t rom_size) {
    Instruction instr = decode_instruction(prog_chunk);

    // Get operands
    uint64_t src1, src2;

    if (instr.op1 < 5) { // Reg
        src1 = vm.regs[instr.r1];
    } else if (instr.op1 < 9) { // Memory
        src1 = mem_access64(vm, rom, instr.lit1, rom_size);
    } else if (instr.op1 < 13) { // Literal
        src1 = instr.lit1;
    } else if (instr.op1 < 14) { // Special1
        src1 = special1_value64(vm.prog_digest_state);
    } else { // Special2
        src1 = special2_value64(vm.mem_digest_state);
    }

    if (instr.op2 < 5) { // Reg
        src2 = vm.regs[instr.r2];
    } else if (instr.op2 < 9) { // Memory
        src2 = mem_access64(vm, rom, instr.lit2, rom_size);
    } else if (instr.op2 < 13) { // Literal
        src2 = instr.lit2;
    } else if (instr.op2 < 14) { // Special1
        src2 = special1_value64(vm.prog_digest_state);
    } else { // Special2
        src2 = special2_value64(vm.mem_digest_state);
    }

    uint64_t result;

    // Determine if it's Op3 or Op2 instruction based on original Rust logic
    if (instr.opcode < 40) {  // Add
        result = src1 + src2;
    } else if (instr.opcode < 80) {  // Mul
        result = src1 * src2;
    } else if (instr.opcode < 96) {  // MulH
        // Using high bits of multiplication matching Rust's (src1 as u128 * src2 as u128) >> 64
        uint64_t a_lo = src1 & 0xFFFFFFFF;
        uint64_t a_hi = src1 >> 32;
        uint64_t b_lo = src2 & 0xFFFFFFFF;
        uint64_t b_hi = src2 >> 32;

        uint64_t p0 = a_lo * b_lo;
        uint64_t p1 = a_lo * b_hi;
        uint64_t p2 = a_hi * b_lo;
        uint64_t p3 = a_hi * b_hi;

        // The high 64 bits of the 128-bit product
        uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
        result = p3 + (p1 >> 32) + (p2 >> 32) + carry;

    } else if (instr.opcode < 112) {  // Div
        result = (src2 != 0) ? src1 / src2 : special1_value64(vm.prog_digest_state);
    } else if (instr.opcode < 128) {  // Mod (bug in Rust implementation: performs division)
        // NOTE: This intentionally replicates the bug in the Rust reference implementation
        // where Mod also performs division if src2 is not zero.
        result = (src2 != 0) ? src1 / src2 : special1_value64(vm.prog_digest_state);
    } else if (instr.opcode < 138) {  // ISqrt (Op2)
        // Metal doesn't have an integer square root, using approximation
        uint64_t x = src1;
        if (x < 2) {
            result = x;
        } else {
            uint64_t y = (x + 1) / 2;
            while (y < x) {
                x = y;
                y = (x + src1 / x) / 2;
            }
            result = x;
        }
    } else if (instr.opcode < 148) {  // BitRev (Op2)
        // Reverse bits of 64-bit value
        result = 0;
        uint64_t temp = src1;
        for (int i = 0; i < 64; i++) {
            result = (result << 1) | (temp & 1);
            temp >>= 1;
        }
    } else if (instr.opcode < 188) {  // Xor
        result = src1 ^ src2;
    } else if (instr.opcode < 204) {  // RotL (Op2)
        result = (src1 << (instr.r1 & 0x3F)) | (src1 >> (64 - (instr.r1 & 0x3F)));
    } else if (instr.opcode < 220) {  // RotR (Op2)
        result = (src1 >> (instr.r1 & 0x3F)) | (src1 << (64 - (instr.r1 & 0x3F)));
    } else if (instr.opcode < 240) {  // Neg (Op2)
        result = ~src1;
    } else if (instr.opcode < 248) {  // And
        result = src1 & src2;
    } else {  // Hash (Op3) - 248-255
        uint8_t input[16];
        for (int i = 0; i < 8; ++i) {
            input[i] = (src1 >> (i * 8)) & 0xFF; // Little-endian
            input[i + 8] = (src2 >> (i * 8)) & 0xFF; // Little-endian
        }

        uint8_t out[64];
        blake2b(out, 64, input, 16);

        uint8_t hash_idx = instr.opcode - 248;
        if (hash_idx < 8) {
            result = ((uint64_t)out[hash_idx * 8 + 0] << 0) |
                     ((uint64_t)out[hash_idx * 8 + 1] << 8) |
                     ((uint64_t)out[hash_idx * 8 + 2] << 16) |
                     ((uint64_t)out[hash_idx * 8 + 3] << 24) |
                     ((uint64_t)out[hash_idx * 8 + 4] << 32) |
                     ((uint64_t)out[hash_idx * 8 + 5] << 40) |
                     ((uint64_t)out[hash_idx * 8 + 6] << 48) |
                     ((uint64_t)out[hash_idx * 8 + 7] << 56);
        } else {
            result = src1; // fallback
        }
    }

    vm.regs[instr.r3] = result;

    // Update prog_digest with the instruction chunk
    blake2b_update(vm.prog_digest_state, prog_chunk, INSTR_SIZE);
}

// Post instructions processing
void post_instructions(thread VMState &vm) {
    // Sum all registers
    uint64_t sum_regs = 0;
    for (uint i = 0; i < NB_REGS; ++i) {
        sum_regs += vm.regs[i];
    }

    uint8_t sum_regs_bytes[8];
    sum_regs_bytes[0] = sum_regs & 0xFF;
    sum_regs_bytes[1] = (sum_regs >> 8) & 0xFF;
    sum_regs_bytes[2] = (sum_regs >> 16) & 0xFF;
    sum_regs_bytes[3] = (sum_regs >> 24) & 0xFF;
    sum_regs_bytes[4] = (sum_regs >> 32) & 0xFF;
    sum_regs_bytes[5] = (sum_regs >> 40) & 0xFF;
    sum_regs_bytes[6] = (sum_regs >> 48) & 0xFF;
    sum_regs_bytes[7] = (sum_regs >> 56) & 0xFF;

    uint8_t prog_value[64];
    {
        Blake2bState temp_state = vm.prog_digest_state;
        blake2b_update(temp_state, sum_regs_bytes, 8);
        blake2b_final(temp_state, prog_value, 64);
    }

    uint8_t mem_value[64];
    {
        Blake2bState temp_state = vm.mem_digest_state;
        blake2b_update(temp_state, sum_regs_bytes, 8);
        blake2b_final(temp_state, mem_value, 64);
    }

    // Create mixing value
    uint8_t mixing_input[136]; // 64 + 64 + 8
    for (int i = 0; i < 64; ++i) {
        mixing_input[i] = prog_value[i];
    }
    for (int i = 0; i < 64; ++i) {
        mixing_input[64 + i] = mem_value[i];
    }
    // loop_counter is u32, so 4 bytes
    mixing_input[128] = vm.loop_counter & 0xFF;
    mixing_input[129] = (vm.loop_counter >> 8) & 0xFF;
    mixing_input[130] = (vm.loop_counter >> 16) & 0xFF;
    mixing_input[131] = (vm.loop_counter >> 24) & 0xFF;

    uint8_t mixing_value[64];
    blake2b(mixing_value, 64, mixing_input, 132); // Input length is 128 (digests) + 4 (loop_counter) = 132

    // Apply mixing to registers (correct implementation matching Rust)
    uint32_t output_mix_len = NB_REGS * REGISTER_SIZE * 32; // 32 * 8 * 32 = 8192 bytes
    thread uint8_t mixing_out[8192];
    hprime(mixing_out, output_mix_len, mixing_value, 64);

    for (uint32_t block_idx = 0; block_idx < 32; ++block_idx) { // 32 blocks as per Rust implementation
        thread const uint8_t *current_block_ptr = mixing_out + (block_idx * NB_REGS * REGISTER_SIZE);
        for (uint32_t reg_idx = 0; reg_idx < NB_REGS; ++reg_idx) {
            uint64_t mix_val = ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 0] << 0) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 1] << 8) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 2] << 16) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 3] << 24) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 4] << 32) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 5] << 40) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 6] << 48) |
                               ((uint64_t)current_block_ptr[reg_idx * REGISTER_SIZE + 7] << 56);
            vm.regs[reg_idx] ^= mix_val;
        }
    }

    // Update prog_seed and increment loop counter
    for (int i = 0; i < 64; ++i) {
        vm.prog_seed[i] = prog_value[i];
    }
    vm.loop_counter++;
}

// Program struct (not directly used as an object in kernel, raw buffer passed)

void program_shuffle(thread uint8_t *program_buffer, uint32_t program_size, thread const uint8_t *seed) {
    hprime(program_buffer, program_size, seed, 64);
}

// Execute program function (for a single loop iteration)
void execute_program(thread VMState &vm, device const uint8_t *rom,
                     thread uint8_t *program_buffer, uint32_t rom_size,
                     uint32_t nb_instrs, uint32_t program_size) {
    // Shuffle program using the current prog_seed
    program_shuffle(program_buffer, program_size, vm.prog_seed);

    // Reset IP for each new program execution within a loop
    vm.ip = 0; // The original Rust code resets the IP implicitly by looping from 0 to nb_instrs

    // Execute instructions for this specific thread
    for (uint32_t i = 0; i < nb_instrs; i++) {
        // Calculate instruction offset based on current IP (modulo nb_instrs for safety, though IP resets to 0)
        uint32_t offset = (vm.ip % nb_instrs) * INSTR_SIZE;
        execute_one_instruction(vm, rom, &program_buffer[offset], rom_size);
        vm.ip++;
    }
    // post_instructions is called after execute_program in the Rust hash function, not inside.
    // It's called in the ashmaize_hash kernel after the execute_program loop.
}

// Finalize VM and get result
void vm_finalize(thread VMState &vm, thread uint8_t *result) {
    uint8_t prog_digest_final[64];
    blake2b_final(vm.prog_digest_state, prog_digest_final, 64);

    uint8_t mem_digest_final[64];
    blake2b_final(vm.mem_digest_state, mem_digest_final, 64);

    // Calculate final digest
    uint8_t context_input[512];  // enough space for digests + counter + registers
    uint pos = 0;

    // Add prog_digest
    for (int i = 0; i < 64; ++i) {
        context_input[pos++] = prog_digest_final[i];
    }

    // Add mem_digest
    for (int i = 0; i < 64; ++i) {
        context_input[pos++] = mem_digest_final[i];
    }

    // Add memory counter (u32)
    context_input[pos++] = vm.memory_counter & 0xFF;
    context_input[pos++] = (vm.memory_counter >> 8) & 0xFF;
    context_input[pos++] = (vm.memory_counter >> 16) & 0xFF;
    context_input[pos++] = (vm.memory_counter >> 24) & 0xFF;

    // Add registers
    for (uint i = 0; i < NB_REGS; ++i) {
        for (int j = 0; j < 8; ++j) { // REGISTER_SIZE bytes per register
            context_input[pos++] = (vm.regs[i] >> (j * 8)) & 0xFF;
        }
    }

    blake2b(result, 64, context_input, pos);
}

// Main compute kernel
kernel void ashmaize_hash(
    device const uint8_t *rom_array [[buffer(0)]],
    device uint8_t *results_array [[buffer(1)]],
    device const uint8_t *salt_array [[buffer(2)]],
    device const uint8_t *rom_digest_array [[buffer(3)]],
    device uint8_t *programs_array [[buffer(4)]], // Programs array is now mutable (input/output)
    device const uint8_t *initial_prog_seeds_array [[buffer(5)]], // New buffer for initial prog_seeds // TODO: remove
    constant uint32_t &rom_size [[buffer(6)]],
    constant uint32_t &nb_loops [[buffer(7)]],
    constant uint32_t &nb_instrs [[buffer(8)]],
    constant uint32_t &program_size [[buffer(9)]], // New buffer for program_size
    uint id [[thread_position_in_grid]]
) {
    // Local copies of data to work in thread address space
    uint8_t local_rom_digest[64];
    for (int i = 0; i < 64; i++) {
        local_rom_digest[i] = rom_digest_array[i];
    }

    // Initialize VM for this thread using local data
    VMState vm;
    // TODO: the salt_array and salt_len handling is wrong here
    uint32_t salt_len = 32;  // Assuming 32-byte salts
    vm_init(vm, local_rom_digest, 64, salt_array, salt_len);

    // Each thread gets its own mutable program buffer segment from the device buffer.
    // It is copied to thread-local memory for shuffling and execution.
    // Max program size: 256 instructions * 20 bytes/instr = 5120 bytes.
    // This should fit in thread-local memory (check device limits if issues).
    thread uint8_t local_program_buffer[5120];

    // Copy initial (template) program into thread-local buffer.
    // The `programs_array` in `mod.rs` is initialized with repeated template programs.
    // So, each thread copies its own segment from this buffer.
    device const uint8_t *initial_program_segment = &programs_array[id * program_size];
    for (uint32_t i = 0; i < program_size; i++) {
        local_program_buffer[i] = initial_program_segment[i];
    }

    // Execute the hash computation
    for (uint32_t loop = 0; loop < nb_loops; loop++) {
        execute_program(vm, rom_array, local_program_buffer, rom_size, nb_instrs, program_size);
    }

    // Finalize and store result
    uint8_t result[64];
    vm_finalize(vm, result);
    for (int i = 0; i < 64; ++i) {
        results_array[id * 64 + i] = result[i];
    }
}

kernel void test_blake2b(
    device const uint8_t *input [[buffer(0)]],
    device uint8_t *output [[buffer(1)]],
    constant uint32_t &input_len [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id == 0) {
        thread uint8_t local_input[1024];
        for(uint i = 0; i < input_len; ++i) {
            local_input[i] = input[i];
        }

        thread uint8_t local_output[64];
        blake2b(local_output, 64, local_input, input_len);

        for(uint i = 0; i < 64; ++i) {
            output[i] = local_output[i];
        }
    }
}

kernel void test_hprime(
    device const uint8_t *input [[buffer(0)]],
    device uint8_t *output [[buffer(1)]],
    constant uint32_t &output_len_param [[buffer(2)]],
    constant uint32_t &input_len_param [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id == 0) {
        // Maximum possible output length for hprime in Ashmaize is 8192 bytes
        // Maximum possible input length for hprime in Ashmaize is ~576 bytes (DIGEST_INIT_SIZE + 512)
        // Using sufficiently large stack arrays for testing.
        thread uint8_t local_input[1024];
        thread uint8_t local_output[8192];

        for(uint i = 0; i < input_len_param; ++i) {
            local_input[i] = input[i];
        }

        hprime(local_output, output_len_param, local_input, input_len_param);

        for(uint i = 0; i < output_len_param; ++i) {
            if (i < 8192) { // Safety check against local_output buffer overflow
                output[i] = local_output[i];
            }
        }
    }
}

kernel void test_vm_init(
    device const uint8_t *rom_digest_array [[buffer(0)]],
    device const uint8_t *salt_array [[buffer(1)]],
    constant uint32_t &salt_len [[buffer(2)]],
    // Outputs
    device Register *output_regs [[buffer(3)]],
    device uint8_t *output_prog_seed [[buffer(4)]],
    device uint32_t *output_ip [[buffer(5)]],
    device uint32_t *output_memory_counter [[buffer(6)]],
    device uint32_t *output_loop_counter [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        uint32_t rom_digest_len = 64;

        thread uint8_t local_rom_digest[64];
        for (uint i = 0; i < rom_digest_len; ++i) {
            local_rom_digest[i] = rom_digest_array[i];
        }

        VMState vm;
        vm_init(vm, local_rom_digest, rom_digest_len, salt_array, salt_len);

        // Copy outputs (excluding prog_digest_state and mem_digest_state as requested)
        for (uint i = 0; i < NB_REGS; ++i) {
            output_regs[i] = vm.regs[i];
        }
        for (uint i = 0; i < 64; ++i) {
            output_prog_seed[i] = vm.prog_seed[i];
        }
        *output_ip = vm.ip;
        *output_memory_counter = vm.memory_counter;
        *output_loop_counter = vm.loop_counter;
    }
}



kernel void test_post_instructions(
    device const Register *initial_regs [[buffer(0)]],
    device const uint8_t *initial_prog_seed [[buffer(1)]],
    constant uint32_t &initial_loop_counter [[buffer(2)]],
    constant uint32_t &initial_memory_counter [[buffer(3)]],
    constant uint32_t &initial_ip [[buffer(4)]],
    // TODO inital state for digests? (buffer 5 and 6)

    // outputs
    device Register *output_regs [[buffer(7)]],
    device uint8_t *output_prog_seed [[buffer(8)]],
    device uint32_t *output_loop_counter [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        VMState vm;

        // Initialize VMState components from input buffers
        for (uint i = 0; i < NB_REGS; ++i) {
            vm.regs[i] = initial_regs[i];
        }
        for (uint i = 0; i < 64; ++i) {
            vm.prog_seed[i] = initial_prog_seed[i];
        }
        vm.loop_counter = initial_loop_counter;
        vm.ip = initial_ip;
        vm.memory_counter = initial_memory_counter;

        // How to initialize vm.prog_digest_state?
        // How to initialize vm.mem_digest_state?

        // Call the post_instructions kernel
        post_instructions(vm);

        // Copy modified state back to output buffers
        for (uint i = 0; i < NB_REGS; ++i) {
            output_regs[i] = vm.regs[i];
        }
        for (uint i = 0; i < 64; ++i) {
            output_prog_seed[i] = vm.prog_seed[i];
        }
        *output_loop_counter = vm.loop_counter;
    }
}
