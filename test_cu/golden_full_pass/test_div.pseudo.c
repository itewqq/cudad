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
void kernel(int32_t arg0, int32_t arg1, uint32_t arg2, uint32_t arg3) {
  bool P0;
  bool P1;
  bool P2;
  float R0;
  uint32_t R1;
  uint32_t R2;
  uint32_t R3;
  uint32_t R4;
  uint32_t R5;
  uint32_t R7;
  uint32_t UR4;
  uint32_t UR5;
  uint32_t v0;
  uint32_t abi_internal_0x28; // live-in
  uint32_t v1;
  uint32_t u0;
  uint32_t u1;
  uint32_t u2;
  uint32_t v2;
  bool b0;
  uint32_t u3;
  uint32_t u4;
  uint32_t v3;
  uint32_t v4;
  uint32_t v5;
  uint32_t v6;
  uint32_t v7;
  uint32_t v8;
  uint32_t v9;
  uint32_t v10;
  uint32_t v11;
  uint32_t v12;
  uint32_t v13;
  uint32_t v14;
  bool b1;
  uint32_t v15;
  uint32_t v16;
  bool b2;
  bool b3;
  uint32_t v17;
  uint32_t v18;
  uint32_t v19;
  uint32_t v20;
  uint32_t v21;

  BB0 {
    v0 = abi_internal_0x28;
    v1 = abs(arg1);
    u0 = arg0;
    u1 = arg1;
    u2 = u0 ^ u1;
    v2 = i2f_rp(v1);
    b0 = (int32_t)(0) <= (int32_t)(u2);
    u3 = c[0x0][0x118];
    u4 = ConstMem(0, 284);
    v3 = rcp_approx(v2);
    v4 = v3 + 268435454;
    v5 = f2i_trunc_u32_ftz_ntz(v4);
    v6 = 0;
    v7 = -v5;
    v8 = v7 * v1 + 0;
    v9 = abs(arg0);
    v10 = mul_hi_u32(v5, v8) + v6;
    v11 = arg2;
    v12 = mul_hi_u32(v10, v9);
    v13 = -v12 + 0 + 0;
    v14 = v1 * v13 + v9;
    b1 = v1 > v14;
    v15 = !b1 ? (v14 - v1) : v14;
    v16 = !b1 ? (v12 + 1) : v12;
    b2 = 0 != arg1;
    b3 = v15 >= v1;
    v17 = b3 ? (v16 + 1) : v16;
    v18 = v17;
    v19 = arg3;
    v20 = !b0 ? (-v18 + 0 + 0) : v18;
    v21 = !b2 ? (~arg1) : v20;
    *addr64(v11, v19) = v21;
  }
  return;
}
// --- End Structured Output ---
