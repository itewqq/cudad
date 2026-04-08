// Loop-heavy kernels — exercise do-while / while-do collapse and carry patterns.

extern "C" __global__ void dot_thread(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ partial,
                                       int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int i = tid; i < n; i += stride) {
        acc += a[i] * b[i];
    }
    partial[tid] = acc;
}

extern "C" __global__ void l2_norm_sq(const float* __restrict__ x,
                                       float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float s = 0.f;
    for (int i = tid; i < n; i += stride) {
        float v = x[i];
        s += v * v;
    }
    out[tid] = s;
}

extern "C" __global__ void cumsum_linear(const float* __restrict__ x,
                                          float* __restrict__ y, int n) {
    // Single-thread linear prefix sum — useful for while/do-while collapse.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float acc = 0.f;
    for (int i = 0; i < n; ++i) {
        acc += x[i];
        y[i] = acc;
    }
}

extern "C" __global__ void find_max(const float* __restrict__ x, int n,
                                     float* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float m = -3.402823e38f;
    for (int i = tid; i < n; i += stride) {
        float v = x[i];
        if (v > m) m = v;
    }
    out[tid] = m;
}

extern "C" __global__ void count_above(const float* __restrict__ x, float thr,
                                        int n, int* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int c = 0;
    for (int i = tid; i < n; i += stride) {
        if (x[i] > thr) c += 1;
    }
    out[tid] = c;
}

extern "C" __global__ void power_series(float x, int k, float* __restrict__ y) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float acc = 1.f;
    float p = 1.f;
    for (int i = 1; i <= k; ++i) {
        p *= x / (float)i;
        acc += p;
    }
    *y = acc;
}
