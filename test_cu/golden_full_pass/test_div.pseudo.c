// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S1: c[0x0][0x164] -> param_1
// BB0.S2: c[0x0][0x160] -> param_0
// BB0.S6: c[0x0][0x118] -> c[0x0][0x118]
// BB0.S13: c[0x0][0x160] -> param_0
// BB0.S15: c[0x0][0x168] -> param_2
// BB0.S22: c[0x0][0x164] -> param_1
// BB0.S26: c[0x0][0x16c] -> param_3
// BB0.S28: c[0x0][0x164] -> param_1
// ABI arg aliases (heuristic):
// param_0 -> arg0 (word32, confidence: low, words: {0})
// param_1 -> arg1 (word32, confidence: low, words: {0})
// param_2 -> arg2 (word32, confidence: low, words: {0})
// param_3 -> arg3 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_0 -> arg0 (word32, confidence: low, words: {0})
// param_1 -> arg1 (word32, confidence: low, words: {0})
// param_2 -> arg2 (word32, confidence: low, words: {0})
// param_3 -> arg3 (word32, confidence: low, words: {0})
__global__ void kernel(int32_t arg0, int32_t arg1, uint32_t arg2, uint32_t arg3) {
  uint32_t v1;
  uint32_t u0;
  uint32_t u1;
  uint32_t u2;
  uint32_t v2;
  bool b0;
  uint32_t v3;
  uint32_t v4;
  uint32_t v5;
  uint32_t v7;
  uint32_t v8;
  uint32_t v9;
  uint32_t v10;
  uint32_t v11;
  uint32_t v12;
  uint32_t v13;
  bool b1;
  uint32_t v14;
  uint32_t v15;
  bool b2;
  bool b3;
  uint32_t v16;
  uint32_t v17;
  uint32_t v18;
  uint32_t v19;

  v1 = abs(arg1);
  u0 = arg0;
  u1 = arg1;
  u2 = u0 ^ u1;
  v2 = (float)(v1);
  b0 = (int32_t)(0) <= (int32_t)(u2);
  v3 = rcp_approx(v2);
  v4 = v3 + 268435454;
  v5 = (uint32_t)(v4);
  v5 = -v5;
  v7 = v5 * v1;
  v8 = abs(arg0);
  v9 = mul_hi_u32(v5, v7);
  v10 = arg2;
  v11 = mul_hi_u32(v9, v8);
  v12 = -v11;
  v13 = v1 * v12 + v8;
  b1 = v1 > v13;
  v14 = !b1 ? (v13 - v1) : v13;
  v15 = !b1 ? (v11 + 1) : v11;
  b2 = 0 != arg1;
  b3 = v14 >= v1;
  v16 = b3 ? (v15 + 1) : v15;
  v16 = v16;
  v17 = arg3;
  v18 = !b0 ? (-v16) : v16;
  v19 = !b2 ? (~arg1) : v18;
  *addr64(v10, v17) = v19;
  return;
}
// --- End Structured Output ---
