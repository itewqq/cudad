// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x164] -> param_1
// BB0.S1: c[0x0][0x160] -> param_0
// BB0.S10: c[0x0][0x160] -> param_0
// BB0.S12: c[0x0][0x168] -> param_2
// BB0.S19: c[0x0][0x164] -> param_1
// BB0.S22: c[0x0][0x16c] -> param_3
// BB0.S24: c[0x0][0x164] -> param_1
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
  uint32_t v0;
  uint32_t u0;
  uint32_t u1;
  uint32_t u2;
  uint32_t v1;
  bool b0;
  uint32_t v2;
  uint32_t v3;
  uint32_t v4;
  uint32_t v6;
  uint32_t v7;
  uint32_t v8;
  uint32_t v9;
  uint32_t v10;
  uint32_t v11;
  uint32_t v12;
  bool b1;
  uint32_t v13;
  uint32_t v14;
  bool b2;
  bool b3;
  uint32_t v15;
  uint32_t v16;
  uint32_t v17;
  uint32_t v18;

  v0 = abs(arg1);
  u0 = arg0;
  u1 = arg1;
  u2 = u0 ^ u1;
  v1 = (float)(v0);
  b0 = (int32_t)(0) <= (int32_t)(u2);
  v2 = rcp_approx(v1);
  v3 = v2 + 268435454;
  v4 = (uint32_t)(v3);
  v6 = -v4 * v0;
  v7 = abs(arg0);
  v8 = mul_hi_u32(v4, v6);
  v9 = arg2;
  v10 = mul_hi_u32(v8, v7);
  v11 = -v10;
  v12 = v0 * v11 + v7;
  b1 = v0 > v12;
  v13 = !b1 ? (v12 - v0) : v12;
  v14 = !b1 ? (v10 + 1) : v10;
  b2 = 0 != arg1;
  b3 = v13 >= v0;
  v15 = b3 ? (v14 + 1) : v14;
  v16 = arg3;
  v17 = !b0 ? (-v15) : v15;
  v18 = !b2 ? (~arg1) : v17;
  *addr64(v9, v16) = v18;
  return;
}
// --- End Structured Output ---
