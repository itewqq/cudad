// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB2.S0: c[0x0][0x168] -> param_1[0]
// ABI arg aliases (heuristic):
// param_1 -> arg1 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_1 -> arg1 (word32, confidence: low, words: {0})
void kernel(uint32_t arg1_word0) {
  uint32_t R1;
  uintptr_t R2;
  uint32_t R26;

  BB0 {
    tid_x = threadIdx.x;
  }
  // Condition from BB0
  if (b0) {
    BB1 {
      v0 = v0 + 1;
    }
  } else {
    BB2 {
      v2 = v3 * v2 + arg1_word0;
    }
  }
  return;
}
// --- End Structured Output ---
