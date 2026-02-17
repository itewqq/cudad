// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S1: c[0x0][0x164] -> param_0[1]
// BB0.S2: c[0x0][0x160] -> param_0[0]
// BB0.S6: c[0x0][0x118] -> c[0x0][0x118]
// BB0.S13: c[0x0][0x160] -> param_0[0]
// BB0.S15: c[0x0][0x168] -> param_1[0]
// BB0.S22: c[0x0][0x164] -> param_0[1]
// BB0.S26: c[0x0][0x16c] -> param_1[1]
// BB0.S28: c[0x0][0x164] -> param_0[1]
// ABI arg aliases (heuristic):
// param_0 -> arg0 (u64, confidence: medium, words: {0, 1})
// param_1 -> arg1 (u64, confidence: medium, words: {0, 1})
// Typed signature inferred from ABI aliases:
// param_0 -> arg0 (u64, confidence: medium, words: {0, 1})
// param_1 -> arg1 (u64, confidence: medium, words: {0, 1})
void kernel(uint64_t arg0_u64, uint64_t arg1_u64) {
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

  BB0 {
    v0 = abi_internal_0x28;
    v1 = abs(arg0_u64.hi32);
    u0 = arg0_u64.lo32;
    u1 = arg0_u64.hi32;
    u2 = u0 ^ u1;
    v2 = i2f_rp(v1);
    b0 = RZ <= u2;
    u3 = c[0x0][0x118];
    u4 = ConstMem(0, 284);
    v3 = rcp_approx(v2);
    v4 = v3 + 268435454;
    v5 = f2i_trunc_u32_ftz_ntz(v4);
    v6 = RZ;
    v7 = -v5;
    v8 = v7 * v1 + RZ;
    v9 = abs(arg0_u64.lo32);
    v10 = mul_hi_u32(v5, v8) + v6;
    v11 = arg1_u64.lo32;
    v12 = mul_hi_u32(v10, v9);
    v13 = -v12 + RZ + RZ;
    v14 = v1 * v13 + v9;
    b1 = v1 > v14;
    if (!b1) v15 = v14 - v1;
    if (!b1) v16 = v12 + 1;
    b2 = RZ != arg0_u64.hi32;
    b3 = v15 >= v1;
    if (b3) v17 = v16 + 1;
    v18 = v17;
    v19 = MOV(arg1_u64.hi32);
    if (!b0) v20 = -v18 + RZ + RZ;
    if (!b2) v21 = ~arg0_u64.hi32;
    *addr64(v11, v19) = v21;
  }
  return;
}
// --- End Structured Output ---
