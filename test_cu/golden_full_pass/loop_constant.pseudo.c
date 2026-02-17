// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S2: c[0x0][0x184] -> param_4[1]
// BB1.S1: c[0x0][0x118] -> c[0x0][0x118]
// BB2.S3: c[0x0][0x0] -> c[0x0][0x0]
// ABI arg aliases (heuristic):
// param_4 -> arg4 (word32, confidence: low, words: {1})
// Typed signature inferred from ABI aliases:
// param_4 -> arg4 (word32, confidence: low, words: {1})
void kernel(uint32_t arg4_word1) {
  bool P0;
  bool P1;
  uint32_t R0;
  uint32_t R1;
  uint32_t R2;
  uint32_t UR6;

  BB0 {
    v0 = abi_internal_0x28;
    ctaid_x = blockIdx.x;
    if (b0) return;
  }
  BB1 {
    tid_x = threadIdx.x;
  }
  // Condition from BB1
  if (!(tid_x > 255)) {
    // Loop header BB2
    while (!(tid_x >= 256)) {
      BB2 {
        shmem_u8[tid_x] = tid_x;
        tid_x = tid_x + c[0x0][0x0];
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
