// Complex CUDA kernels — tiled GEMM (shared-memory blocking), bitonic sort,
// Blelloch prefix sum. These produce multi-level nested loops with shared
// memory, sync barriers, and complex index arithmetic.

#include <stdint.h>

// ------ Tiled GEMM (C = A * B) ------
// Tile size compile-time constant so the compiler can unroll.
#define TILE 16

extern "C" __global__ void sgemm_tiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float       * __restrict__ C,
    int M, int N, int K_dim
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < (K_dim + TILE - 1) / TILE; t++) {
        // Collaborative load with bounds check
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K_dim) ? A[row * K_dim + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K_dim && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ------ Bitonic sort (in-place, power-of-2 array) ------

extern "C" __global__ void bitonic_sort(int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            int partner = tid ^ stride;
            if (partner > tid && tid < n && partner < n) {
                bool ascending = ((tid & size) == 0);
                int a = data[tid];
                int b = data[partner];
                if (ascending ? (a > b) : (a < b)) {
                    data[tid] = b;
                    data[partner] = a;
                }
            }
        }
    }
}

// ------ Blelloch exclusive prefix sum (work-efficient, single block) ------

extern "C" __global__ void prefix_sum_blelloch(int *data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;

    // Load input into shared memory
    temp[2*tid]   = (2*tid   < n) ? data[2*tid]   : 0;
    temp[2*tid+1] = (2*tid+1 < n) ? data[2*tid+1] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear the last element
    if (tid == 0) temp[n-1] = 0;
    __syncthreads();

    // Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write output
    if (2*tid   < n) data[2*tid]   = temp[2*tid];
    if (2*tid+1 < n) data[2*tid+1] = temp[2*tid+1];
}

// ------ 2D 5-point stencil with halo ------

#define STENCIL_BLOCK 16
#define STENCIL_HALO 1
#define STENCIL_TILE (STENCIL_BLOCK + 2*STENCIL_HALO)

extern "C" __global__ void stencil2d_5pt(
    const float * __restrict__ in,
    float       * __restrict__ out,
    int W, int H
) {
    __shared__ float tile[STENCIL_TILE][STENCIL_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * STENCIL_BLOCK + tx;
    int gy = blockIdx.y * STENCIL_BLOCK + ty;

    // Load center
    int sx = tx + STENCIL_HALO, sy = ty + STENCIL_HALO;
    tile[sy][sx] = (gx < W && gy < H) ? in[gy * W + gx] : 0.0f;

    // Load halo — top/bottom
    if (ty < STENCIL_HALO) {
        int hy = gy - STENCIL_HALO;
        tile[ty][sx] = (hy >= 0 && gx < W) ? in[hy * W + gx] : 0.0f;
        hy = gy + STENCIL_BLOCK;
        tile[ty + STENCIL_BLOCK + STENCIL_HALO][sx] =
            (hy < H && gx < W) ? in[hy * W + gx] : 0.0f;
    }
    // Load halo — left/right
    if (tx < STENCIL_HALO) {
        int hx = gx - STENCIL_HALO;
        tile[sy][tx] = (hx >= 0 && gy < H) ? in[gy * W + hx] : 0.0f;
        hx = gx + STENCIL_BLOCK;
        tile[sy][tx + STENCIL_BLOCK + STENCIL_HALO] =
            (hx < W && gy < H) ? in[gy * W + hx] : 0.0f;
    }

    __syncthreads();

    if (gx < W && gy < H) {
        float val = 0.25f * tile[sy][sx]
                  + 0.125f * (tile[sy-1][sx] + tile[sy+1][sx]
                            + tile[sy][sx-1] + tile[sy][sx+1]);
        out[gy * W + gx] = val;
    }
}

// ------ Warp-shuffle reduction + cross-warp reduction ------

extern "C" __global__ void warp_reduce_sum(const float *in, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    // Warp-level reduction via shfl_down
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Collect warp results in shared memory
    __shared__ float warp_sums[32]; // max 32 warps per block
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();

    // First warp reduces the warp sums
    if (warp == 0) {
        val = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) out[blockIdx.x] = val;
    }
}
