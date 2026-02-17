// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB2.S0: c[0x0][0x168] -> param_2
// ABI arg aliases (heuristic):
// param_2 -> arg2 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_2 -> arg2 (word32, confidence: low, words: {0})
void kernel(uint32_t arg2) {
  uint32_t R1;
  uintptr_t R2;
  uint32_t R26;
  uint32_t v0;
  uint32_t abi_internal_0x28; // live-in
  bool b0; // live-in
  uint32_t v2; // live-in
  uint32_t v3; // live-in

  BB0 {
    v0 = abi_internal_0x28;
    tid_x = threadIdx.x;
  }
  // Condition from BB0
  if (b0) {
    BB1 {
      v0 = v0 + 1;
    }
  } else {
    BB2 {
      v2 = v3 * v2 + arg2;
    }
  }
  return;
}
// --- End Structured Output ---
