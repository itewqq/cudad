// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S1: c[0x0][0x184] -> param_9
// BB2.S3: c[0x0][0x0] -> blockDim.x
// ABI arg aliases (heuristic):
// param_9 -> arg9 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_9 -> arg9 (word32, confidence: low, words: {0})
__global__ void kernel(uint32_t arg9) {
  __shared__ uint8_t shmem_u8[256];
  bool b0; // live-in

  if (b0) return;
    tid_x = threadIdx.x;
  if (!((int32_t)(tid_x) > (int32_t)(255))) {
      shmem_u8[tid_x] = tid_x;
      tid_x = tid_x + blockDim.x;
    do {
      shmem_u8[tid_x] = tid_x;
      tid_x = tid_x + blockDim.x;
    } while(!((int32_t)(tid_x) >= (int32_t)(256)));
  }
}
// --- End Structured Output ---
