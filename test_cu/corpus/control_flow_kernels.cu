// Complex CUDA kernels — control-flow stress tests.
// Each kernel targets a specific structurizer challenge: deeply nested
// if-else trees, multi-exit loops, switch-like dispatch via computed goto
// tables, early returns from inside loops, etc.

#include <stdint.h>

// ------ Multi-level nested if-else tree ------
// 4 levels of branching → 16 leaf paths. Forces the structurizer to
// handle a deep diamond lattice.

extern "C" __global__ void decision_tree(
    const float * __restrict__ features, // 4 features per sample
    int         * __restrict__ classes,   // output class per sample
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float f0 = features[tid*4+0];
    float f1 = features[tid*4+1];
    float f2 = features[tid*4+2];
    float f3 = features[tid*4+3];

    int cls;
    if (f0 < 0.5f) {
        if (f1 < 0.3f) {
            if (f2 < 0.7f) {
                cls = (f3 < 0.4f) ? 0 : 1;
            } else {
                cls = (f3 < 0.6f) ? 2 : 3;
            }
        } else {
            if (f2 < 0.2f) {
                cls = (f3 < 0.5f) ? 4 : 5;
            } else {
                cls = (f3 < 0.8f) ? 6 : 7;
            }
        }
    } else {
        if (f1 < 0.6f) {
            if (f2 < 0.4f) {
                cls = (f3 < 0.3f) ? 8 : 9;
            } else {
                cls = (f3 < 0.7f) ? 10 : 11;
            }
        } else {
            if (f2 < 0.5f) {
                cls = (f3 < 0.2f) ? 12 : 13;
            } else {
                cls = (f3 < 0.9f) ? 14 : 15;
            }
        }
    }
    classes[tid] = cls;
}

// ------ Loop with multiple exit conditions ------
// Iterates with three distinct break paths — tests the structurizer's
// ability to merge multiple loop exits without goto.

extern "C" __global__ void multi_exit_loop(
    const int * __restrict__ data,
    int       * __restrict__ result,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int acc = 0;
    int limit = data[tid];
    for (int i = 0; i < 1000; i++) {
        acc += data[(tid + i) % n];

        // Exit 1: overflow sentinel
        if (acc > 100000) {
            result[tid] = -1;
            return;
        }

        // Exit 2: exact match
        if (acc == limit) {
            result[tid] = i;
            return;
        }

        // Exit 3: negative accumulator
        if (acc < 0) {
            result[tid] = -2;
            return;
        }
    }
    result[tid] = acc;
}

// ------ Dispatch via switch-like pattern ------
// Each thread executes one of 8 operation types. The compiler usually
// generates a jump table or if-chain, stressing the structurizer's
// ability to recover a switch.

extern "C" __global__ void dispatch_ops(
    const int   * __restrict__ opcodes,
    const float * __restrict__ a,
    const float * __restrict__ b,
    float       * __restrict__ c,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float va = a[tid], vb = b[tid];
    float r;
    switch (opcodes[tid]) {
        case 0: r = va + vb; break;
        case 1: r = va - vb; break;
        case 2: r = va * vb; break;
        case 3: r = (vb != 0.0f) ? va / vb : 0.0f; break;
        case 4: r = fminf(va, vb); break;
        case 5: r = fmaxf(va, vb); break;
        case 6: r = sqrtf(va * va + vb * vb); break;
        case 7: r = powf(va, vb); break;
        default: r = 0.0f; break;
    }
    c[tid] = r;
}

// ------ Nested loops with break/continue ------

extern "C" __global__ void nested_loop_break_continue(
    const int * __restrict__ matrix, // row-major [n][m]
    int       * __restrict__ result,
    int n, int m
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int sum = 0;
    for (int i = 0; i < n; i++) {
        int row_sum = 0;
        for (int j = 0; j < m; j++) {
            int val = matrix[i * m + j];
            if (val < 0) continue;    // skip negatives
            if (val > 10000) break;   // sentinel — stop row
            row_sum += val;
        }
        if (row_sum == 0) continue;   // skip zero-sum rows
        sum += row_sum;
        if (sum > 1000000) break;     // global budget
    }
    result[tid] = sum;
}

// ------ Early return inside nested loops ------

extern "C" __global__ void find_pattern(
    const int * __restrict__ haystack, // flat 2D [H][W]
    const int * __restrict__ pattern,  // [PH][PW]
    int       * __restrict__ found_at, // output [2] = {row, col}, or {-1,-1}
    int H, int W, int PH, int PW
) {
    // Single-thread brute-force 2D pattern search.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    for (int r = 0; r <= H - PH; r++) {
        for (int c = 0; c <= W - PW; c++) {
            bool match = true;
            for (int pr = 0; pr < PH && match; pr++) {
                for (int pc = 0; pc < PW; pc++) {
                    if (haystack[(r+pr)*W+(c+pc)] != pattern[pr*PW+pc]) {
                        match = false;
                        break;
                    }
                }
            }
            if (match) {
                found_at[0] = r;
                found_at[1] = c;
                return;
            }
        }
    }
    found_at[0] = -1;
    found_at[1] = -1;
}

// ------ State machine (explicit transitions) ------

extern "C" __global__ void state_machine(
    const char * __restrict__ input,
    int        * __restrict__ counts, // 4 counters: [digits, alpha, spaces, other]
    int len
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    int state = 0; // 0=start, 1=in_word, 2=in_number, 3=whitespace
    int digits = 0, alpha = 0, spaces = 0, other = 0;

    for (int i = 0; i < len; i++) {
        char ch = input[i];
        switch (state) {
            case 0: // start
                if (ch >= '0' && ch <= '9') { state = 2; digits++; }
                else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) { state = 1; alpha++; }
                else if (ch == ' ' || ch == '\t' || ch == '\n') { state = 3; spaces++; }
                else { other++; }
                break;
            case 1: // in_word
                if (ch >= '0' && ch <= '9') { digits++; }
                else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) { alpha++; }
                else if (ch == ' ' || ch == '\t' || ch == '\n') { state = 3; spaces++; }
                else { state = 0; other++; }
                break;
            case 2: // in_number
                if (ch >= '0' && ch <= '9') { digits++; }
                else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) { state = 1; alpha++; }
                else if (ch == ' ' || ch == '\t' || ch == '\n') { state = 3; spaces++; }
                else { state = 0; other++; }
                break;
            case 3: // whitespace
                if (ch >= '0' && ch <= '9') { state = 2; digits++; }
                else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) { state = 1; alpha++; }
                else if (ch == ' ' || ch == '\t' || ch == '\n') { spaces++; }
                else { state = 0; other++; }
                break;
        }
    }
    counts[0] = digits;
    counts[1] = alpha;
    counts[2] = spaces;
    counts[3] = other;
}
