#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

// CUDA kernel with branches, loops, and dynamic loop count
__global__ void complexVectorOp(const float *a, const float *b, float *c, int n, int loop_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = 0.0f;
        float val_a = a[idx];
        float val_b = b[idx];

        // Branch: Choose operation based on input values
        if (val_a + val_b > 1.0f) {
            // Sum branch
            result = val_a + val_b;
        } else {
            // Product branch
            result = val_a * val_b;
        }

        // Loop: Apply transformation loop_count times
        for (int i = 0; i < loop_count; ++i) {
            if (result > 0.5f) {
                result *= 1.1f; // Scale up
            } else {
                result *= 0.9f; // Scale down
            }
        }

        c[idx] = result;
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Vector size
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Number of loop iterations
    const int loop_count = 5; // Configurable from host

    // Host vectors
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX; // Random values in [0,1]
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device vectors
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    complexVectorOp<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, loop_count);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));

    // Verify result (CPU computation for comparison)
    for (int i = 0; i < N; ++i) {
        float result = 0.0f;
        // Replicate branch logic
        if (h_a[i] + h_b[i] > 1.0f) {
            result = h_a[i] + h_b[i];
        } else {
            result = h_a[i] * h_b[i];
        }
        // Replicate loop logic with loop_count
        for (int j = 0; j < loop_count; ++j) {
            if (result > 0.5f) {
                result *= 1.1f;
            } else {
                result *= 0.9f;
            }
        }
        // Compare with GPU result
        if (std::abs(h_c[i] - result) > 1e-5) {
            std::cerr << "Verification failed at index " << i 
                      << ": expected " << result << ", got " << h_c[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Complex vector operation with " << loop_count << " loops successful!" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}