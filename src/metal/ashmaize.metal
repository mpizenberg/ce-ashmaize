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

// Instrumentation control - set to 0 to completely remove instrumentation overhead
#define INSTRUMENTATION_ENABLED 0

#if INSTRUMENTATION_ENABLED
// Instrumentation Metrics
constant uint METRIC_BLAKE2B_ROUND_COUNT = 0;
constant uint METRIC_EXECUTE_INSTR_COUNT = 1;
constant uint METRIC_HPRIME_COUNT = 2;
constant uint METRIC_ROM_ACCESS_COUNT = 3;
constant uint METRIC_SPECIAL_VALUE_COUNT = 4;
constant uint TOTAL_METRICS = 5;

#define INSTRUMENT_INC(buffer, metric) \
    do { if (buffer) atomic_fetch_add_explicit(buffer + metric, 1u, memory_order_relaxed); } while(0)
#else
#define INSTRUMENT_INC(buffer, metric) ((void)0)
#endif

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

inline uint64_t rotate_left_u64(uint64_t v, uint32_t s) {
    s &= 63;
    return (v << s) | (v >> ((64 - s) & 63));
}
inline uint64_t rotate_right_u64(uint64_t v, uint32_t s) {
    s &= 63;
    return (v >> s) | (v << ((64 - s) & 63));
}

struct Blake2bState {
    uint64_t h[8];
    uint64_t t[2];   // counters
    uint64_t f[2];   // finalization
    uint8_t buf[BLAKE2B_BLOCKBYTES];
    uint64_t buflen;
    bool last_node;
};

// G Mixing function for Blake2b
void blake2b_G(thread uint64_t *v, int a, int b, int c, int d, uint64_t x, uint64_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotate_right_u64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = rotate_right_u64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = rotate_right_u64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotate_right_u64(v[b] ^ v[c], 63);
}

// Round function for Blake2b
void blake2b_round(thread Blake2bState &S, thread const uint64_t *m, device atomic_uint *instrumentation_buffer) {
    INSTRUMENT_INC(instrumentation_buffer, METRIC_BLAKE2B_ROUND_COUNT);
    uint64_t v[16]; // Local 16-word state vector for compression

    // Initialize v[0..7] with state, v[8..15] with IV
    for (int i = 0; i < 8; ++i) {
        v[i] = S.h[i];
        v[i + 8] = blake2b_IV[i];
    }

    // XOR v[12..15] with T and F (counter and finalization flags)
    v[12] ^= S.t[0];
    v[13] ^= S.t[1];
    v[14] ^= S.f[0];
    v[15] ^= S.f[1];

    // 12 rounds
    for (int r = 0; r < 12; ++r) {
        uint s_idx = (r % 10) * 16;
        blake2b_G(v, 0, 4, 8,  12, m[SIGMA[s_idx + 0]], m[SIGMA[s_idx + 1]]);
        blake2b_G(v, 1, 5, 9,  13, m[SIGMA[s_idx + 2]], m[SIGMA[s_idx + 3]]);
        blake2b_G(v, 2, 6, 10, 14, m[SIGMA[s_idx + 4]], m[SIGMA[s_idx + 5]]);
        blake2b_G(v, 3, 7, 11, 15, m[SIGMA[s_idx + 6]], m[SIGMA[s_idx + 7]]);
        blake2b_G(v, 0, 5, 10, 15, m[SIGMA[s_idx + 8]], m[SIGMA[s_idx + 9]]);
        blake2b_G(v, 1, 6, 11, 12, m[SIGMA[s_idx + 10]], m[SIGMA[s_idx + 11]]);
        blake2b_G(v, 2, 7, 8,  13, m[SIGMA[s_idx + 12]], m[SIGMA[s_idx + 13]]);
        blake2b_G(v, 3, 4, 9,  14, m[SIGMA[s_idx + 14]], m[SIGMA[s_idx + 15]]);
    }

    // Update chaining value S.h with compressed state
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
void blake2b_update(thread Blake2bState &S, thread const uint8_t *in, uint64_t inlen, device atomic_uint *instrumentation_buffer) {
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

            thread const uint64_t* m = (thread const uint64_t*)S.buf;
            blake2b_round(S, m, instrumentation_buffer);

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
void blake2b_final(thread Blake2bState &S, thread uint8_t *out, uint outlen, device atomic_uint *instrumentation_buffer) {
    uint64_t lastblock = S.buflen;
    S.t[0] += lastblock;
    if (S.t[0] < lastblock) S.t[1]++;

    S.f[0] = ~0ULL;

    if (S.buflen > 0) {
        for (uint8_t i = (uint8_t)lastblock; i < BLAKE2B_BLOCKBYTES; ++i) {
            S.buf[i] = 0;
        }

        thread const uint64_t* m = (thread const uint64_t*)S.buf;
        blake2b_round(S, m, instrumentation_buffer);
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
void blake2b(thread uint8_t *out, uint outlen, thread const uint8_t *in, uint inlen, device atomic_uint *instrumentation_buffer) {
    Blake2bState S;
    blake2b_init(S, outlen);
    blake2b_update(S, in, inlen, instrumentation_buffer);
    blake2b_final(S, out, outlen, instrumentation_buffer);
}

// Argon2 H' function
void hprime(thread uint8_t *output, uint output_len, thread const uint8_t *input, uint input_len, device atomic_uint *instrumentation_buffer) {
    INSTRUMENT_INC(instrumentation_buffer, METRIC_HPRIME_COUNT);
    if (output_len <= 64) {
        uint8_t temp[BLAKE2B_OUTBYTES + 4 + 512]; // Max 64 (output_len) + 4 (output_len_bytes) + 512 (max input_len)
        temp[0] = output_len & 0xFF;
        temp[1] = (output_len >> 8) & 0xFF;
        temp[2] = (output_len >> 16) & 0xFF;
        temp[3] = (output_len >> 24) & 0xFF;

        for (uint i = 0; i < input_len; ++i) {
            temp[4 + i] = input[i];
        }

        blake2b(output, output_len, temp, 4 + input_len, instrumentation_buffer);
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
    blake2b(v0_hash, 64, v0_input, 4 + input_len, instrumentation_buffer);

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
        blake2b(vi_hash, 64, vi_prev, 64, instrumentation_buffer);

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
        blake2b(temp, bytes, vi_prev, 64, instrumentation_buffer);
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
    // Per-instruction caching for special values to avoid duplicate clones/finalizations
    uint64_t cached_special1_value;
    bool special1_valid;  // invalidated after prog_digest update
    uint64_t cached_special2_value;
    bool special2_valid;  // invalidated after mem_digest update
};

// VM initialization function
void vm_init(thread VMState &vm, thread const uint8_t *rom_digest, uint32_t rom_digest_len,
             device const uint8_t *salt, uint32_t salt_len, device atomic_uint *instrumentation_buffer) {

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
           init_buffer_input, rom_digest_len + salt_len, instrumentation_buffer);

    // Initialize registers from first part of init_buffer
    thread const uint64_t *init_buffer_regs = (thread const uint64_t *)init_buffer;
    for (uint i = 0; i < NB_REGS; ++i) {
        vm.regs[i] = init_buffer_regs[i];
    }

    // Initialize digests from remaining buffer
    thread const uint8_t *digests_data = init_buffer + REGS_CONTENT_SIZE;

    // Prog digest initialization
    blake2b_init(vm.prog_digest_state, 64);
    blake2b_update(vm.prog_digest_state, digests_data, 64, instrumentation_buffer);

    // Mem digest initialization
    blake2b_init(vm.mem_digest_state, 64);
    blake2b_update(vm.mem_digest_state, &digests_data[64], 64, instrumentation_buffer);

    // Prog seed is provided as an argument, from CPU pre-calculation
    for (int i = 0; i < 64; ++i) {
        vm.prog_seed[i] = digests_data[128 + i];
    }

    vm.ip = 0;
    vm.loop_counter = 0;
    vm.memory_counter = 0;
    vm.special1_valid = false;
    vm.special2_valid = false;
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

constant uint32_t DATASET_ACCESS_SIZE = 64;
inline device const uint8_t* rom_at(device const uint8_t *rom,
                                    uint32_t rom_size,
                                    uint32_t i,
                                    device atomic_uint *instrumentation_buffer)
{
    INSTRUMENT_INC(instrumentation_buffer, METRIC_ROM_ACCESS_COUNT);
    // avoid division by zero if rom_size < 64: in Rust that would panic on / 0; here we defensively treat blocks=0 -> start=0
    uint32_t blocks = (rom_size / DATASET_ACCESS_SIZE);
    uint32_t start = (blocks == 0) ? 0u : (i % blocks);

    // IMPORTANT: replicates the Rust code that uses `start` directly (byte index),
    // not `start * DATASET_ACCESS_SIZE`. This reproduces the original semantics.
    uint32_t offset = start;

    // We return rom + offset. The Rust implementation would panic if offset+64 > len,
    // but Metal cannot panic similarly — caller must ensure rom_size is large enough.
    return rom + offset;
}

// integer isqrt (returns floor(sqrt(x))) - matches typical integer sqrt semantics.
// Rust used src1.isqrt(); implement deterministic equivalent in Metal.
inline uint64_t int_isqrt(uint64_t x) {
    if (x <= 1) return x;
    // Newton method starting guess
    uint64_t r = 1ULL << ((63 - clz(x)) / 2 + 1); // rough initial
    // iterate a few times (should converge quickly)
    for (int i = 0; i < 6; ++i) {
        uint64_t nr = (r + x / r) >> 1;
        if (nr >= r) break;
        r = nr;
    }
    // fix possible overshoot, but avoid overflow in multiplication
    // Check for potential overflow before multiplication
    while (r < 0xFFFFFFFFFFFFFFFFULL && (r+1) <= x / (r+1) && (r+1) * (r+1) <= x) ++r;
    while (r > 0 && (r > x / r || r * r > x)) --r;
    return r;
}


inline uint64_t special_value64(thread Blake2bState &digest, device atomic_uint *instrumentation_buffer) {
    INSTRUMENT_INC(instrumentation_buffer, METRIC_SPECIAL_VALUE_COUNT);
    // clone the digest state and finalize it, then return first 8 bytes LE
    thread Blake2bState S = digest;
    uint8_t out[64];
    blake2b_final(S, out, 64, instrumentation_buffer);
    return *((thread uint64_t*)out);
}
inline uint64_t special1_value64(thread VMState &vm, device atomic_uint *instrumentation_buffer) {
    if (vm.special1_valid) {
        return vm.cached_special1_value;
    }
    vm.cached_special1_value = special_value64(vm.prog_digest_state, instrumentation_buffer);
    vm.special1_valid = true;
    return vm.cached_special1_value;
}
inline uint64_t special2_value64(thread VMState &vm, device atomic_uint *instrumentation_buffer) {
    if (vm.special2_valid) {
        return vm.cached_special2_value;
    }
    vm.cached_special2_value = special_value64(vm.mem_digest_state, instrumentation_buffer);
    vm.special2_valid = true;
    return vm.cached_special2_value;
}


// The execute function
void execute_one_instruction(thread VMState &vm,
                             device const uint8_t *rom,
                             thread const uint8_t *prog_chunk,
                             uint32_t rom_size,
                             device atomic_uint *instrumentation_buffer) {
    INSTRUMENT_INC(instrumentation_buffer, METRIC_EXECUTE_INSTR_COUNT);
    // --- decode opcode byte into "opcode value" (0..255) ---
    uint8_t opcode_byte = prog_chunk[0];

    // Determine if opcode corresponds to Op3 or Op2 and which operator.
    // The Rust mapping used ranges; we replicate the same boundaries.
    bool is_op3 = false;
    Op3Type op3_kind = (Op3Type)0;
    Op2Type op2_kind = (Op2Type)0;
    uint8_t hash_index = 0; // only meaningful if opcode selects Hash variant

    // replicate Rust match ranges exactly
    if (opcode_byte < 40) {
        is_op3 = true; op3_kind = (Op3Type)Add;
    } else if (opcode_byte < 80) {
        is_op3 = true; op3_kind = (Op3Type)Mul;
    } else if (opcode_byte < 96) {
        is_op3 = true; op3_kind = (Op3Type)MulH;
    } else if (opcode_byte < 112) {
        is_op3 = true; op3_kind = (Op3Type)Div;
    } else if (opcode_byte < 128) {
        is_op3 = true; op3_kind = (Op3Type)Mod;
    } else if (opcode_byte < 138) {
        is_op3 = false; op2_kind = (Op2Type)ISqrt;
    } else if (opcode_byte < 148) {
        is_op3 = false; op2_kind = (Op2Type)BitRev;
    } else if (opcode_byte < 188) {
        is_op3 = true; op3_kind = (Op3Type)Xor;
    } else if (opcode_byte < 204) {
        is_op3 = false; op2_kind = (Op2Type)RotL;
    } else if (opcode_byte < 220) {
        is_op3 = false; op2_kind = (Op2Type)RotR;
    } else if (opcode_byte < 240) {
        is_op3 = false; op2_kind = (Op2Type)Neg;
    } else if (opcode_byte < 248) {
        is_op3 = true; op3_kind = (Op3Type)And;
    } else { // 248..=255 => Hash with parameter (value - 248)
        is_op3 = true;
        op3_kind = (Op3Type)Hash;
        hash_index = (uint8_t)(opcode_byte - 248); // 0..7
    }

    // --- decode operands nibble ---
    uint8_t nibble_hi = (prog_chunk[1] >> 4) & 0x0f;
    uint8_t nibble_lo = (prog_chunk[1] & 0x0f);

    // Operand classification exactly as Rust: 0..4 => Reg, 5..8 => Memory, 9..12 => Literal,
    // 13 => Special1, 14..15 => Special2
    enum OperandKind { OP_REG, OP_MEM, OP_LIT, OP_SP1, OP_SP2 };
    auto decode_operand = [&](uint8_t v) -> OperandKind {
        if (v <= 4) return OP_REG;
        if (v <= 8) return OP_MEM;
        if (v <= 12) return OP_LIT;
        if (v == 13) return OP_SP1;
        return OP_SP2; // 14 or 15
    };

    OperandKind op1 = decode_operand(nibble_hi);
    OperandKind op2 = decode_operand(nibble_lo);

    // --- decode r1, r2, r3 from bytes 2..3 (u16 big-endian in rust) ---
    // rust: let rs = ((instruction[2] as u16) << 8) | (instruction[3] as u16);
    uint16_t rs = ((uint16_t)prog_chunk[2] << 8) | (uint16_t)prog_chunk[3];
    uint8_t r1 = (uint8_t)(((rs >> (2 * REGS_BITS)) & REGS_INDEX_MASK));
    uint8_t r2 = (uint8_t)(((rs >> REGS_BITS) & REGS_INDEX_MASK));
    uint8_t r3 = (uint8_t)((rs) & REGS_INDEX_MASK);

    // --- decode literals lit1 and lit2 (little-endian u64 from prog_chunk[4..12], [12..20]) ---
    // Rust uses from_le_bytes on slices. We'll reconstruct little-endian.
    thread const uint64_t* lit1_ptr = (thread const uint64_t*)(prog_chunk + 4);
    thread const uint64_t* lit2_ptr = (thread const uint64_t*)(prog_chunk + 12);
    uint64_t lit1 = *lit1_ptr;
    uint64_t lit2 = *lit2_ptr;

    // Corresponds to Rust macro mem_access64!(vm, rom, addr)
    auto mem_access64 = [&](thread VMState &vref, device const uint8_t *rom_p, uint64_t addr) -> uint64_t {
        device const uint8_t *mem = rom_at(rom, rom_size, (uint32_t)addr, instrumentation_buffer);
        uint8_t mem_chunk[64];
        // Copy 64 bytes - keep original byte-by-byte copy for correctness
        for (uint32_t i = 0; i < 64; ++i) {
            mem_chunk[i] = mem[i];
        }
        // update mem_digest_state with entire 64-byte chunk
        blake2b_update(vm.mem_digest_state, mem_chunk, 64, instrumentation_buffer);
        // Invalidate special2 cache after mem_digest update
        vm.special2_valid = false;
        // increment memory_counter (wrapping)
        vm.memory_counter = vm.memory_counter + 1; // wrapping in metal C++ will behave but make sure vm.memory_counter is uint64
        // compute index chunk
        uint32_t idx = (uint32_t)((vm.memory_counter % (64u / 8u)) * 8u);
        // read little-endian u64 from mem_chunk[idx..idx+8]
        uint64_t out = 0;
        for (int i = 0; i < 8; ++i) {
            out |= (uint64_t)mem_chunk[idx + i] << (8 * i);
        }
        return out;
    };

    // --- actual operation execution ---
    if (is_op3) {
        // fetch src1
        uint64_t src1;
        switch (op1) {
            case OP_REG: src1 = vm.regs[r1]; break;
            case OP_MEM: src1 = mem_access64(vm, rom, lit1); break;
            case OP_LIT: src1 = lit1; break;
            case OP_SP1: src1 = special1_value64(vm, instrumentation_buffer); break;
            default:      src1 = special2_value64(vm, instrumentation_buffer); break;
        }
        // fetch src2
        uint64_t src2;
        switch (op2) {
            case OP_REG: src2 = vm.regs[r2]; break;
            case OP_MEM: src2 = mem_access64(vm, rom, lit2); break;
            case OP_LIT: src2 = lit2; break;
            case OP_SP1: src2 = special1_value64(vm, instrumentation_buffer); break;
            default:      src2 = special2_value64(vm, instrumentation_buffer); break;
        }

        uint64_t result = 0;
        switch (op3_kind) {
            case Add: {
                // wrapping_add in Rust -> identical in uint64 arithmetic
                result = (uint64_t)(src1 + src2);
                break;
            }
            case Mul: {
                // wrapping_mul
                result = (uint64_t)(src1 * src2);
                break;
            }
            case MulH: {
                // high 64 bits of 128-bit product: ((src1 as u128 * src2 as u128) >> 64) as u64
                // compute via 128-bit emulation if platform supports, else split
                // Metal C++ may not have builtin u128, implement via split multiplication.
                // We'll implement classic 64x64 -> 128 multiplication and extract high 64 bits.
                uint64_t a_lo = (uint32_t)src1;
                uint64_t a_hi = src1 >> 32;
                uint64_t b_lo = (uint32_t)src2;
                uint64_t b_hi = src2 >> 32;

                uint64_t p0 = a_lo * b_lo;
                uint64_t p1 = a_lo * b_hi;
                uint64_t p2 = a_hi * b_lo;
                uint64_t p3 = a_hi * b_hi;

                // combine cross terms
                uint64_t carry = ((p0 >> 32) + (uint32_t)p1 + (uint32_t)p2) >> 32;
                uint64_t mid = (p1 >> 32) + (p2 >> 32) + carry;

                uint64_t high = p3 + mid;
                result = high;
                break;
            }
            case Xor: {
                result = src1 ^ src2;
                break;
            }
            case Div: {
                if (src2 == 0) {
                    result = special1_value64(vm, instrumentation_buffer);
                } else {
                    result = src1 / src2;
                }
                break;
            }
            case Mod: {
                // Note: The provided Rust reference has a bug: Mod returns src1 / src2 (same as Div).
                // We reproduce that bug exactly here.
                if (src2 == 0) {
                    result = special1_value64(vm, instrumentation_buffer);
                } else {
                    // reproduce the reference bug: division instead of modulus
                    result = src1 / src2;
                }
                break;
            }
            case And: {
                result = src1 & src2;
                break;
            }
            case Hash: {
                uint8_t input16[16];
                // write little-endian
                for (int i = 0; i < 8; ++i) input16[i] = (uint8_t)((src1 >> (8 * i)) & 0xFFu);
                for (int i = 0; i < 8; ++i) input16[8 + i] = (uint8_t)((src2 >> (8 * i)) & 0xFFu);

                uint8_t digest64[64];
                blake2b(digest64, 64, input16, 16, instrumentation_buffer);

                // take chunk hash_index (0..7) of 8 bytes and interpret as little-endian u64
                thread uint8_t *chunk = &(digest64[hash_index * 8]);
                uint64_t chunk_u64 = 0;
                for (int i = 0; i < 8; ++i) chunk_u64 |= (uint64_t)chunk[i] << (8 * i);
                result = chunk_u64;
                break;
            }
            default: {
                // Unknown Op3 kind; keep result 0 (shouldn't happen if mapping is correct).
                result = 0;
                break;
            }
        } // end switch op3_kind

        // write to destination register r3
        vm.regs[r3] = result;

    } else { // Op2 operator
        // fetch src1 according to op1
        uint64_t src1;
        switch (op1) {
            case OP_REG: src1 = vm.regs[r1]; break;
            case OP_MEM: src1 = mem_access64(vm, rom, lit1); break;
            case OP_LIT: src1 = lit1; break;
            case OP_SP1: src1 = special1_value64(vm, instrumentation_buffer); break;
            default:      src1 = special2_value64(vm, instrumentation_buffer); break;
        }

        uint64_t result = 0;
        switch (op2_kind) {
            case Neg: {
                result = ~src1;
                break;
            }
            case RotL: {
                result = rotate_left_u64(src1, (uint32_t)r1);
                break;
            }
            case RotR: {
                result = rotate_right_u64(src1, (uint32_t)r1);
                break;
            }
            case ISqrt: {
                result = int_isqrt(src1);
                break;
            }
            case BitRev: {
                // reverse bits, same as Rust u64.reverse_bits()
                // Implement via builtin or manual loop
                uint64_t x = src1;
                // Use typical bit-reversal algorithm (swap halves progressively)
                x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
                x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
                x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
                x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
                x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
                x = (x << 32) | (x >> 32);
                result = x;
                break;
            }
            default: {
                // Should not get here, default safe
                result = src1;
                break;
            }
        } // end switch op2_kind

        vm.regs[r3] = result;
    } // end if is_op3

    // Update program digest with this instruction/chunk
    blake2b_update(vm.prog_digest_state, prog_chunk, INSTR_SIZE, instrumentation_buffer);
    // Invalidate special1 cache for next instruction
    vm.special1_valid = false;
}


// Post instructions processing
void post_instructions(thread VMState &vm, device atomic_uint *instrumentation_buffer) {
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
        blake2b_update(temp_state, sum_regs_bytes, 8, instrumentation_buffer);
        blake2b_final(temp_state, prog_value, 64, instrumentation_buffer);
    }

    uint8_t mem_value[64];
    {
        Blake2bState temp_state = vm.mem_digest_state;
        blake2b_update(temp_state, sum_regs_bytes, 8, instrumentation_buffer);
        blake2b_final(temp_state, mem_value, 64, instrumentation_buffer);
    }

    // Create mixing value
    uint8_t mixing_input[132]; // 64 + 64 + 4
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
    blake2b(mixing_value, 64, mixing_input, 132, instrumentation_buffer); // Input length is 128 (digests) + 4 (loop_counter) = 132

    // Apply mixing to registers (correct implementation matching Rust)
    uint32_t output_mix_len = NB_REGS * REGISTER_SIZE * 32; // 32 * 8 * 32 = 8192 bytes
    thread uint8_t mixing_out[8192];
    hprime(mixing_out, output_mix_len, mixing_value, 64, instrumentation_buffer);

    for (uint32_t block_idx = 0; block_idx < 32; ++block_idx) { // 32 blocks as per Rust implementation
        thread const uint8_t *current_block_ptr = mixing_out + (block_idx * NB_REGS * REGISTER_SIZE);
        for (uint32_t reg_idx = 0; reg_idx < NB_REGS; ++reg_idx) {
            thread const uint64_t* mix_val_ptr = (thread const uint64_t*)(current_block_ptr + reg_idx * REGISTER_SIZE);
            vm.regs[reg_idx] ^= *mix_val_ptr;
        }
    }

    // Update prog_seed and increment loop counter
    for (int i = 0; i < 64; ++i) {
        vm.prog_seed[i] = prog_value[i];
    }
    vm.loop_counter++;
}

// Program struct (not directly used as an object in kernel, raw buffer passed)

void program_shuffle(thread uint8_t *program_buffer, uint32_t program_size, thread const uint8_t *seed, device atomic_uint *instrumentation_buffer) {
    hprime(program_buffer, program_size, seed, 64, instrumentation_buffer);
}

// Execute program function (for a single loop iteration)
void execute_program(thread VMState &vm, device const uint8_t *rom,
                     thread uint8_t *program_buffer, uint32_t rom_size,
                     uint32_t nb_instrs, uint32_t program_size,
                     device atomic_uint *instrumentation_buffer) {
    // Shuffle program using the current prog_seed
    program_shuffle(program_buffer, program_size, vm.prog_seed, instrumentation_buffer);

    // Execute instructions for this specific thread
    for (uint32_t i = 0; i < nb_instrs; i++) {
        // Calculate the actual byte offset into the program_buffer with wrapping
        // This replicates Rust's Program::at(vm.ip) behavior
        uint32_t current_program_instruction_index = vm.ip % nb_instrs; // vm.ip is global, nb_instrs is the program length
        uint32_t byte_offset = current_program_instruction_index * INSTR_SIZE;

        execute_one_instruction(vm, rom, &program_buffer[byte_offset], rom_size, instrumentation_buffer);
        vm.ip++; // Increment global instruction pointer
    }

    // Post instructions
    post_instructions(vm, instrumentation_buffer);
}

// Finalize VM and get result
void vm_finalize(thread VMState &vm, thread uint8_t *result, device atomic_uint *instrumentation_buffer) {
    uint8_t prog_digest_final[64];
    blake2b_final(vm.prog_digest_state, prog_digest_final, 64, instrumentation_buffer);

    uint8_t mem_digest_final[64];
    blake2b_final(vm.mem_digest_state, mem_digest_final, 64, instrumentation_buffer);

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

    blake2b(result, 64, context_input, pos, instrumentation_buffer);
}

// Main compute kernel
kernel void ashmaize_hash(
    device const uint8_t *rom_array [[buffer(0)]],
    device const uint8_t *rom_digest_array [[buffer(1)]],
    device const uint8_t *salt_array [[buffer(2)]],
    constant uint32_t &salt_len [[buffer(3)]],
    constant uint32_t &rom_size [[buffer(4)]],
    constant uint32_t &nb_instrs [[buffer(5)]],
    constant uint32_t &nb_loops [[buffer(6)]],

    // outputs
    device uint8_t *final_hash [[buffer(7)]],
    device atomic_uint *instrumentation_buffer [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    // Select the salt for the current thread
    device const uint8_t *my_salt = salt_array + (id * salt_len);

    // Initialize the VM with given ROM digest and salt
    uint32_t rom_digest_len = 64;
    thread uint8_t local_rom_digest[64];
    for (uint i = 0; i < rom_digest_len; ++i) {
        local_rom_digest[i] = rom_digest_array[i];
    }
    VMState vm;
    vm_init(vm, local_rom_digest, rom_digest_len, my_salt, salt_len, instrumentation_buffer);

    // Max program size: 256 instructions * 20 bytes/instr = 5120 bytes.
    thread uint8_t local_program_buffer[5120];

    // Initialize program with zeros (as is done in VM::new)
    for (uint32_t i = 0; i < nb_instrs * INSTR_SIZE; i++) {
        local_program_buffer[i] = 0;
    }

    // Execute the hash computation
    for (uint32_t loop = 0; loop < nb_loops; loop++) {
        execute_program(vm, rom_array, local_program_buffer, rom_size, nb_instrs, 5120, instrumentation_buffer);
    }

    // Finalize the result and write to output buffer
    thread uint8_t result[64];
    vm_finalize(vm, result, instrumentation_buffer);

    // Select the output location for the current thread
    device uint8_t *my_final_hash = final_hash + (id * 64);
    for (uint i = 0; i < 64; ++i) {
        my_final_hash[i] = result[i];
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
        blake2b(local_output, 64, local_input, input_len, nullptr); // No instrumentation for test kernels

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

        hprime(local_output, output_len_param, local_input, input_len_param, nullptr); // No instrumentation for test kernels

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
        vm_init(vm, local_rom_digest, rom_digest_len, salt_array, salt_len, nullptr); // No instrumentation for test kernels

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
    device const uint8_t *rom_digest_array [[buffer(0)]],
    device const uint8_t *salt_array [[buffer(1)]],
    constant uint32_t &salt_len [[buffer(2)]],

    // outputs
    device Register *output_regs [[buffer(3)]],
    device uint8_t *output_prog_seed [[buffer(4)]],
    device uint32_t *output_loop_counter [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        // Initialize the VM
        uint32_t rom_digest_len = 64;
        thread uint8_t local_rom_digest[64];
        for (uint i = 0; i < rom_digest_len; ++i) {
            local_rom_digest[i] = rom_digest_array[i];
        }
        VMState vm;
        vm_init(vm, local_rom_digest, rom_digest_len, salt_array, salt_len, nullptr); // No instrumentation for test kernels

        // Call the post_instructions kernel
        post_instructions(vm, nullptr); // No instrumentation for test kernels

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


kernel void test_execute_program(
    device const uint8_t *rom_array [[buffer(0)]],
    device const uint8_t *rom_digest_array [[buffer(1)]],
    device const uint8_t *salt_array [[buffer(2)]],
    constant uint32_t &salt_len [[buffer(3)]],
    constant uint32_t &rom_size [[buffer(4)]],
    constant uint32_t &nb_instrs [[buffer(5)]],

    // outputs
    device Register *output_regs [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        // Initialize the VM with given ROM digest and salt
        uint32_t rom_digest_len = 64;
        thread uint8_t local_rom_digest[64];
        for (uint i = 0; i < rom_digest_len; ++i) {
            local_rom_digest[i] = rom_digest_array[i];
        }
        VMState vm;
        vm_init(vm, local_rom_digest, rom_digest_len, salt_array, salt_len, nullptr); // No instrumentation for test kernels

        // Max program size: 256 instructions * 20 bytes/instr = 5120 bytes.
        thread uint8_t local_program_buffer[5120];

        // Initialize program with zeros (as is done in VM::new)
        for (uint32_t i = 0; i < nb_instrs * INSTR_SIZE; i++) {
            local_program_buffer[i] = 0;
        }

        // Now execute the program
        execute_program(vm, rom_array, local_program_buffer, rom_size, nb_instrs, 5120, nullptr); // No instrumentation for test kernels

        // Copy modified registers back to output buffer
        for (uint i = 0; i < NB_REGS; ++i) {
            output_regs[i] = vm.regs[i];
        }
    }
}

// --- Test Kernel for execute_one_instruction ---

// This kernel is designed for unit testing the `execute_one_instruction` function.
// It takes initial VM state components, including seeds for the digests,
// executes one instruction, and writes the final state back for comparison.
kernel void test_execute_one_instruction(
    // Inputs
    device const uint8_t *rom_array [[buffer(0)]],
    device const uint8_t *rom_digest_array [[buffer(1)]],
    device const uint8_t *salt_array [[buffer(2)]],
    constant uint32_t &salt_len [[buffer(3)]],
    constant uint32_t &rom_size [[buffer(4)]],
    constant uint32_t &nb_instrs [[buffer(5)]],
    device const uint8_t* prog_chunk [[buffer(6)]],

    // Outputs
    device uint64_t* final_regs [[buffer(7)]],
    device uint32_t* final_counters [[buffer(8)]], // ip, memory_counter
    device uint8_t* final_prog_digest_hash [[buffer(9)]],
    device uint8_t* final_mem_digest_hash [[buffer(10)]]
) {
    // 1. Reconstruct the initial VMState in thread memory
    uint32_t rom_digest_len = 64;
    thread uint8_t local_rom_digest[64];
    for (uint i = 0; i < rom_digest_len; ++i) {
        local_rom_digest[i] = rom_digest_array[i];
    }
    thread VMState vm;
    vm_init(vm, local_rom_digest, rom_digest_len, salt_array, salt_len, nullptr); // No instrumentation for test kernels

    // 2. Execute the single instruction
    thread uint8_t local_prog_chunk[20];
    for (uint i = 0; i < 20; ++i) {
        local_prog_chunk[i] = prog_chunk[i];
    }
    execute_one_instruction(vm, rom_array, local_prog_chunk, rom_size, nullptr); // No instrumentation for test kernels

    // 3. Write the final state back to output buffers
    for (uint i = 0; i < NB_REGS; ++i) {
        final_regs[i] = vm.regs[i];
    }
    final_counters[0] = vm.ip;
    final_counters[1] = vm.memory_counter;

    // 4. Finalize the digests and write their hashes to the output
    // We clone the state first so the original (now modified) state in the VM is not consumed
    thread Blake2bState temp_prog_digest = vm.prog_digest_state;
    thread uint8_t temp_prog_digest_hash[64];
    blake2b_final(temp_prog_digest, temp_prog_digest_hash, 64, nullptr); // No instrumentation for test kernels
    for (uint i = 0; i < 64; ++i) {
        final_prog_digest_hash[i] = temp_prog_digest_hash[i];
    }

    thread Blake2bState temp_mem_digest = vm.mem_digest_state;
    thread uint8_t temp_mem_digest_hash[64];
    blake2b_final(temp_mem_digest, temp_mem_digest_hash, 64, nullptr); // No instrumentation for test kernels
    for (uint i = 0; i < 64; ++i) {
        final_mem_digest_hash[i] = temp_mem_digest_hash[i];
    }
}

kernel void test_vm_finalize(
    // Inputs to initialize VM
    device const uint8_t *rom_digest_array [[buffer(0)]],
    device const uint8_t *salt_array [[buffer(1)]],
    constant uint32_t &salt_len [[buffer(2)]],
    // Output
    device uint8_t *final_hash [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        // 1. Initialize VM state
        uint32_t rom_digest_len = 64;
        thread uint8_t local_rom_digest[64];
        for (uint i = 0; i < rom_digest_len; ++i) {
            local_rom_digest[i] = rom_digest_array[i];
        }
        thread VMState vm;
        vm_init(vm, local_rom_digest, rom_digest_len, salt_array, salt_len, nullptr); // No instrumentation for test kernels

        // 2. Call vm_finalize
        thread uint8_t result[64];
        vm_finalize(vm, result, nullptr); // No instrumentation for test kernels

        // 3. Write result to output buffer
        for (uint i = 0; i < 64; ++i) {
            final_hash[i] = result[i];
        }
    }
}
