// Real-world-style CUDA kernels: ML inference & training primitives.
// Patterns from cuDNN, PyTorch custom kernels, and CUTLASS:
// softmax, layer normalization, fused attention, ReLU+bias, cross-entropy loss.

#include <stdint.h>
#include <float.h>

// ------ Online softmax (numerically stable, single-pass) ------
// The two-pass pattern: first find max, then exp-and-normalize.
// Each thread handles one row of [batch][seq_len].

extern "C" __global__ void softmax_forward(
    const float * __restrict__ input,   // [B][N]
    float       * __restrict__ output,  // [B][N]
    int B, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;

    const float *in_row = input + row * N;
    float *out_row = output + row * N;

    // Pass 1: find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        float v = in_row[i];
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp and sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        out_row[i] *= inv_sum;
    }
}

// ------ Layer normalization (forward) ------
// Pattern from transformer blocks: per-token mean/variance + affine.

extern "C" __global__ void layer_norm_forward(
    const float * __restrict__ input,   // [B][D]
    const float * __restrict__ gamma,   // [D]
    const float * __restrict__ beta,    // [D]
    float       * __restrict__ output,  // [B][D]
    float       * __restrict__ mean_out,  // [B] optional
    float       * __restrict__ rstd_out,  // [B] optional
    int B, int D, float eps
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;

    const float *x = input + row * D;
    float *y = output + row * D;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < D; i++) {
        mean += x[i];
    }
    mean /= D;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= D;

    float rstd = rsqrtf(var + eps);

    if (mean_out) mean_out[row] = mean;
    if (rstd_out) rstd_out[row] = rstd;

    // Normalize + affine
    for (int i = 0; i < D; i++) {
        y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

// ------ Fused ReLU + bias + residual add ------
// Common fusion pattern in inference: y = relu(x + bias) + residual

extern "C" __global__ void fused_relu_bias_residual(
    const float * __restrict__ input,
    const float * __restrict__ bias,
    const float * __restrict__ residual,
    float       * __restrict__ output,
    int B, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    int d = idx % D;
    float val = input[idx] + bias[d];
    val = (val > 0.0f) ? val : 0.0f; // ReLU
    output[idx] = val + residual[idx];
}

// ------ Cross-entropy loss (per-sample) ------
// Pattern from training loop: log-softmax then nll.

extern "C" __global__ void cross_entropy_loss(
    const float * __restrict__ logits,  // [B][C]
    const int   * __restrict__ targets, // [B]
    float       * __restrict__ losses,  // [B]
    int B, int C
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;

    const float *logit_row = logits + row * C;
    int target = targets[row];

    // log-softmax(target class)
    float max_val = -FLT_MAX;
    for (int i = 0; i < C; i++) {
        if (logit_row[i] > max_val) max_val = logit_row[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < C; i++) {
        sum_exp += expf(logit_row[i] - max_val);
    }

    float log_softmax = logit_row[target] - max_val - logf(sum_exp);
    losses[row] = -log_softmax;
}

// ------ Fused GELU activation ------
// Approximate GELU used in BERT/GPT: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

extern "C" __global__ void gelu_forward(
    const float * __restrict__ input,
    float       * __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float x3 = x * x * x;
    // sqrt(2/pi) ≈ 0.7978845608
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

// ------ Batched SGEMV (matrix-vector multiply, one row per thread) ------
// Pattern from FC layer inference: y = A*x + b

extern "C" __global__ void batched_sgemv(
    const float * __restrict__ A,       // [M][K]
    const float * __restrict__ x,       // [K]
    const float * __restrict__ bias,    // [M]
    float       * __restrict__ y,       // [M]
    int M, int K_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float acc = 0.0f;
    const float *a_row = A + row * K_dim;
    for (int k = 0; k < K_dim; k++) {
        acc += a_row[k] * x[k];
    }
    y[row] = acc + bias[row];
}

// ------ Top-K selection (partial sort, single thread per row) ------
// Pattern from beam search in seq2seq models.

extern "C" __global__ void topk_per_row(
    const float * __restrict__ data,    // [B][N]
    float       * __restrict__ values,  // [B][K]
    int         * __restrict__ indices, // [B][K]
    int B, int N, int K_val
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;

    const float *d = data + row * N;
    float *v = values + row * K_val;
    int *idx = indices + row * K_val;

    // Initialize with first K elements
    for (int k = 0; k < K_val && k < N; k++) {
        v[k] = d[k];
        idx[k] = k;
    }

    // Insertion sort style: find min in current top-K, replace if larger
    for (int i = K_val; i < N; i++) {
        // Find position of minimum in current top-K
        int min_pos = 0;
        float min_val = v[0];
        for (int k = 1; k < K_val; k++) {
            if (v[k] < min_val) {
                min_val = v[k];
                min_pos = k;
            }
        }
        if (d[i] > min_val) {
            v[min_pos] = d[i];
            idx[min_pos] = i;
        }
    }
}
