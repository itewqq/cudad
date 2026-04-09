// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S2: c[0x0][0x184] -> param_9
// BB1.S1: c[0x0][0x118] -> c[0x0][0x118]
// BB2.S3: c[0x0][0x0] -> blockDim.x
// ABI arg aliases (heuristic):
// param_9 -> arg9 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_9 -> arg9 (word32, confidence: low, words: {0})
__global__ void kernel(uint32_t arg9) {
  __shared__ uint8_t shmem_u8[256];
  bool b0; // live-in
  bool b1;

  if (b0) return;
    tid_x = threadIdx.x;
    b1 = (int32_t)(tid_x) > (int32_t)(255);
  if (!((int32_t)(tid_x) > (int32_t)(255))) {
    do {
      shmem_u8[tid_x] = tid_x;
      tid_x = tid_x + blockDim.x;
      // 2 phi node(s) omitted [BB2]
      // phi merge: v2 <- phi(v2, v2)
      // phi merge: b1 <- phi(b1, b1)
    } while(!(b1));
  }
  // 2 phi node(s) omitted [BB3]
  // phi merge: v2 <- phi(v2, v2)
  // phi merge: b1 <- phi(b1, b1)
}
// --- End Structured Output ---
