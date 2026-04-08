// Shared-memory kernels — exercise barrier / convergence handling, shmem
// array styling, and tiled patterns.

extern "C" __global__ void reduce_block(const float* __restrict__ x,
                                         float* __restrict__ out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    s[tid] = gid < n ? x[gid] : 0.f;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

extern "C" __global__ void max_reduce_block(const float* __restrict__ x,
                                              float* __restrict__ out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    s[tid] = gid < n ? x[gid] : -3.402823e38f;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = s[tid];
            float b = s[tid + stride];
            s[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

extern "C" __global__ void transpose_tile(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int width, int height) {
    __shared__ float tile[16][17];  // +1 to avoid bank conflicts
    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    x = blockIdx.y * 16 + threadIdx.x;
    y = blockIdx.x * 16 + threadIdx.y;
    if (x < height && y < width)
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

extern "C" __global__ void stencil1d(const float* __restrict__ in,
                                      float* __restrict__ out, int n) {
    __shared__ float tile[128 + 8];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lid = tid + 4;
    tile[lid] = gid < n ? in[gid] : 0.f;
    if (tid < 4) {
        int left = gid - 4;
        tile[tid] = left >= 0 ? in[left] : 0.f;
        int right = gid + 128;
        tile[lid + 128] = right < n ? in[right] : 0.f;
    }
    __syncthreads();
    if (gid < n) {
        float acc = 0.f;
        for (int k = -4; k <= 4; ++k) acc += tile[lid + k];
        out[gid] = acc / 9.f;
    }
}

extern "C" __global__ void histogram256(const unsigned char* __restrict__ data,
                                          int n, int* __restrict__ hist) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x;
    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < n; i += stride) {
        atomicAdd(&local_hist[data[i]], 1);
    }
    __syncthreads();
    if (tid < 256) atomicAdd(&hist[tid], local_hist[tid]);
}
