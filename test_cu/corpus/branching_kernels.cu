// Branchy kernels — exercise if-then, if-then-else, nested ifs, and
// one-arm shortcut diamonds.

extern "C" __global__ void abs_clamp(const float* __restrict__ x,
                                      float* __restrict__ y, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v < 0.f) v = -v;
    if (v > cap) v = cap;
    y[i] = v;
}

extern "C" __global__ void select_max(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float y = b[i];
    c[i] = x > y ? x : y;
}

extern "C" __global__ void classify_sign(const float* __restrict__ x,
                                          int* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    int s;
    if (v > 0.f) s = 1;
    else if (v < 0.f) s = -1;
    else s = 0;
    y[i] = s;
}

extern "C" __global__ void nested_ifs(const int* __restrict__ a,
                                       int* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = a[i];
    int y = 0;
    if (x >= 0) {
        if (x < 10) y = x;
        else if (x < 100) y = x * 2;
        else y = x * 3;
    } else {
        if (x > -10) y = -x;
        else y = 0;
    }
    b[i] = y;
}

extern "C" __global__ void early_exit_chain(const int* __restrict__ a,
                                              int n, int* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = a[i];
    if (v == 0) { out[i] = -1; return; }
    if (v < 0)  { out[i] = -v; return; }
    if (v == 1) { out[i] = 42; return; }
    out[i] = v * v;
}

extern "C" __global__ void swap_if_gt(int* __restrict__ a, int* __restrict__ b,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = a[i];
    int y = b[i];
    if (x > y) { a[i] = y; b[i] = x; }
}
