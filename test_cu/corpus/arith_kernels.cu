// Straight-line arithmetic kernels — exercise sequence collapse and SSA lift
// with simple 1-BB bodies, predicated exits, and vector-typed loads/stores.

extern "C" __global__ void saxpy(float a, const float* __restrict__ x,
                                  float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a * x[i] + y[i];
}

extern "C" __global__ void vec_add(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void vec_mul(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

extern "C" __global__ void vec_fma(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    const float* __restrict__ c,
                                    float* __restrict__ d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] * b[i] + c[i];
}

extern "C" __global__ void scale_add(const float* __restrict__ x, float alpha,
                                      float beta, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + beta;
}

extern "C" __global__ void clamp_kernel(float* __restrict__ x, float lo,
                                         float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    x[i] = v;
}

extern "C" __global__ void relu(const float* __restrict__ x,
                                 float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
