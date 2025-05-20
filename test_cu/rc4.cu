#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

// CUDA kernel for RC4 encryption
__global__ void rc4Encrypt(const unsigned char *key, int key_len, 
                          const unsigned char *input, unsigned char *output, 
                          int data_len, int num_blocks) {
    // Each block processes a separate data chunk
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    // Shared state array S (256 bytes per block)
    __shared__ unsigned char S[256];
    
    // Initialize S array (parallel)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        S[i] = i;
    }
    __syncthreads();

    // KSA: Permute S based on key (single thread per block)
    if (threadIdx.x == 0) {
        int j = 0;
        for (int i = 0; i < 256; ++i) {
            j = (j + S[i] + key[i % key_len]) % 256;
            // Swap S[i] and S[j]
            unsigned char temp = S[i];
            S[i] = S[j];
            S[j] = temp;
        }
    }
    __syncthreads();

    // PRGA: Generate keystream and encrypt (single thread per block)
    if (threadIdx.x == 0) {
        int i = 0, j = 0;
        int data_offset = block_id * data_len;
        for (int pos = 0; pos < data_len; ++pos) {
            i = (i + 1) % 256;
            j = (j + S[i]) % 256;
            // Swap S[i] and S[j]
            unsigned char temp = S[i];
            S[i] = S[j];
            S[j] = temp;
            // Keystream byte
            unsigned char k_byte = S[(S[i] + S[j]) % 256];
            // XOR with input to produce output
            if (data_offset + pos < data_len * num_blocks) {
                output[data_offset + pos] = input[data_offset + pos] ^ k_byte;
            }
        }
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
    // Input parameters
    const std::string key_str = "MySecretKey";
    const std::string plaintext = "Hello, CUDA RC4 Encryption!";
    const int data_len = plaintext.size(); // Bytes per block
    const int num_blocks = 4; // Number of parallel blocks
    const int total_data_len = data_len * num_blocks;

    // Host vectors
    std::vector<unsigned char> h_key(key_str.begin(), key_str.end());
    std::vector<unsigned char> h_input(total_data_len);
    std::vector<unsigned char> h_output(total_data_len);
    std::vector<unsigned char> h_decrypted(total_data_len);

    // Initialize input data (repeat plaintext across blocks)
    for (int i = 0; i < num_blocks; ++i) {
        for (int j = 0; j < data_len; ++j) {
            h_input[i * data_len + j] = plaintext[j];
        }
    }

    // Device pointers
    unsigned char *d_key = nullptr, *d_input = nullptr, *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_key, h_key.size()));
    CUDA_CHECK(cudaMalloc(&d_input, total_data_len));
    CUDA_CHECK(cudaMalloc(&d_output, total_data_len));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), h_key.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), total_data_len, cudaMemcpyHostToDevice));

    // Launch kernel for encryption
    int threadsPerBlock = 256;
    int blocksPerGrid = num_blocks;
    rc4Encrypt<<<blocksPerGrid, threadsPerBlock>>>(d_key, h_key.size(), d_input, d_output, data_len, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, total_data_len, cudaMemcpyDeviceToHost));

    // Verify by decrypting (RC4 is symmetric: encrypt again to decrypt)
    CUDA_CHECK(cudaMemcpy(d_input, h_output.data(), total_data_len, cudaMemcpyHostToDevice));
    rc4Encrypt<<<blocksPerGrid, threadsPerBlock>>>(d_key, h_key.size(), d_input, d_output, data_len, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy decrypted result back to host
    CUDA_CHECK(cudaMemcpy(h_decrypted.data(), d_output, total_data_len, cudaMemcpyDeviceToHost));

    // Verify decryption
    bool success = true;
    for (int i = 0; i < total_data_len; ++i) {
        if (h_decrypted[i] != h_input[i]) {
            std::cerr << "Verification failed at index " << i 
                      << ": expected " << (int)h_input[i] << ", got " << (int)h_decrypted[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "RC4 encryption and decryption successful!" << std::endl;
        // Print first block's plaintext and ciphertext
        std::cout << "Original plaintext: " << plaintext << std::endl;
        std::cout << "Ciphertext (first block, hex): ";
        for (int i = 0; i < data_len; ++i) {
            printf("%02x ", h_output[i]);
        }
        std::cout << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}