// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S2: c[0x0][0x184] -> param_9
// BB1.S1: c[0x0][0x118] -> c[0x0][0x118]
// BB2.S3: c[0x0][0x0] -> blockDimX
// ABI arg aliases (heuristic):
// param_9 -> arg9 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_9 -> arg9 (word32, confidence: low, words: {0})
void kernel(uint32_t arg9) {
  bool P0;
  bool P1;
  uint32_t R0;
  uint32_t R1;
  uint32_t R2;
  uint32_t UR6;
  uint32_t UR7;
  __shared__ uint8_t shmem_u8[256];
  uint32_t v0;
  uint32_t abi_internal_0x28; // live-in
  bool b0; // live-in
  uint32_t u0;
  uint32_t u1;
  bool b1;
  bool b2;

  BB0 {
    v0 = abi_internal_0x28;
    ctaid_x = blockIdx.x;
    if (b0) return;
  }
  BB1 {
    tid_x = threadIdx.x;
    u0 = c[0x0][0x118];
    u1 = ConstMem(0, 284);
    b1 = (int32_t)(tid_x) > (int32_t)(255);
    b2 = tid_x != 0;
  }
  // Condition from BB1
  if (!((int32_t)(tid_x) > (int32_t)(255))) {
    BB2 {
      shmem_u8[tid_x] = tid_x;
      tid_x = tid_x + blockDimX;
    }
    // Loop header BB2
    while (!((int32_t)(tid_x) >= (int32_t)(256))) {
      BB2 {
        shmem_u8[tid_x] = tid_x;
        tid_x = tid_x + blockDimX;
        // 2 phi node(s) omitted
        // phi merge: v2 <- phi(v2, v2)
        // phi merge: b1 <- phi(b1, b1)
      }
    }
  }
  BB3 {
    _ = BSYNC();
    // 2 phi node(s) omitted
    // phi merge: v2 <- phi(v2, v2)
    // phi merge: b1 <- phi(b1, b1)
  }
}
// --- End Structured Output ---
