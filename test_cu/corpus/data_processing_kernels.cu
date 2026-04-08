// Real-world-style CUDA kernels: string/text processing & compression.
// Patterns from GPU-accelerated databases, text analytics, and LZ4-style
// compression — byte-level processing with complex control flow.

#include <stdint.h>

// ------ GPU string search (brute-force, one thread per position) ------
// Pattern from GPU-grep / database LIKE operator.

extern "C" __global__ void string_search(
    const char * __restrict__ text,
    const char * __restrict__ pattern,
    int         * __restrict__ matches,  // output: 1 if match starts at pos
    int text_len, int pattern_len
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos > text_len - pattern_len) {
        if (pos < text_len) matches[pos] = 0;
        return;
    }

    int matched = 1;
    for (int i = 0; i < pattern_len; i++) {
        if (text[pos + i] != pattern[i]) {
            matched = 0;
            break;
        }
    }
    matches[pos] = matched;
}

// ------ Run-length encoding (RLE) compress ------
// Each thread handles a chunk of input bytes.

extern "C" __global__ void rle_compress(
    const uint8_t * __restrict__ input,
    uint8_t       * __restrict__ output,  // [count, value] pairs
    int           * __restrict__ out_len, // atomic output length counter
    int n, int chunk_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * chunk_size;
    if (start >= n) return;
    int end = start + chunk_size;
    if (end > n) end = n;

    // Temporary buffer for this thread's RLE output (max 2*chunk_size bytes)
    // In practice this would be in shared/local mem; here simplified.
    int local_len = 0;
    uint8_t local_buf[512]; // worst case

    int i = start;
    while (i < end) {
        uint8_t val = input[i];
        int run = 1;
        while (i + run < end && input[i + run] == val && run < 255) {
            run++;
        }
        if (local_len + 2 <= 512) {
            local_buf[local_len++] = (uint8_t)run;
            local_buf[local_len++] = val;
        }
        i += run;
    }

    // Atomically reserve space in output
    int offset = atomicAdd(out_len, local_len);
    for (int j = 0; j < local_len; j++) {
        output[offset + j] = local_buf[j];
    }
}

// ------ CSV field parser (GPU-accelerated) ------
// Each thread scans one row to find field boundaries.
// Pattern from GPU data frames (cuDF-style).

extern "C" __global__ void csv_find_fields(
    const char * __restrict__ data,
    const int  * __restrict__ row_starts,  // byte offset of each row
    const int  * __restrict__ row_lengths, // length of each row
    int        * __restrict__ field_offsets, // [max_rows][max_fields]
    int        * __restrict__ field_count,   // [max_rows]
    int num_rows, int max_fields, char delimiter
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = row_starts[row];
    int len = row_lengths[row];
    int *offsets = field_offsets + row * max_fields;

    int nf = 0;
    int in_quote = 0;
    offsets[0] = start;
    nf = 1;

    for (int i = 0; i < len && nf < max_fields; i++) {
        char ch = data[start + i];

        if (ch == '"') {
            in_quote = !in_quote;
        } else if (ch == delimiter && !in_quote) {
            if (nf < max_fields) {
                offsets[nf] = start + i + 1;
                nf++;
            }
        }
    }
    field_count[row] = nf;
}

// ------ Radix-2 histogram for radix sort ------
// Each thread computes a 2-bit histogram for its chunk.
// Pattern from GPU radix sort (CUB-style).

extern "C" __global__ void radix_histogram(
    const uint32_t * __restrict__ keys,
    uint32_t       * __restrict__ histograms, // [num_blocks][4]
    int n, int bit_offset
) {
    __shared__ uint32_t local_hist[4];

    if (threadIdx.x < 4) local_hist[threadIdx.x] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        uint32_t digit = (keys[tid] >> bit_offset) & 0x3;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        histograms[blockIdx.x * 4 + threadIdx.x] = local_hist[threadIdx.x];
    }
}

// ------ UTF-8 character counting ------
// Count Unicode code points in a UTF-8 byte stream.
// Each thread processes a chunk. Complex byte-level branching.

extern "C" __global__ void utf8_count_chars(
    const uint8_t * __restrict__ data,
    int           * __restrict__ char_counts, // per-thread counts
    int n, int chunk_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * chunk_size;
    if (start >= n) return;
    int end = start + chunk_size;
    if (end > n) end = n;

    int count = 0;
    int i = start;

    // If we're not at the start of the stream, skip to next code point
    if (start > 0) {
        while (i < end && (data[i] & 0xC0) == 0x80) {
            i++; // skip continuation bytes
        }
    }

    while (i < end) {
        uint8_t b = data[i];
        if (b < 0x80) {
            // ASCII: 1 byte
            i += 1;
        } else if ((b & 0xE0) == 0xC0) {
            // 2-byte sequence
            i += 2;
        } else if ((b & 0xF0) == 0xE0) {
            // 3-byte sequence
            i += 3;
        } else if ((b & 0xF8) == 0xF0) {
            // 4-byte sequence
            i += 4;
        } else {
            // Invalid byte — skip
            i += 1;
        }
        count++;
    }
    char_counts[tid] = count;
}
