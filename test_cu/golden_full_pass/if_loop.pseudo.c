// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S2: c[0x0][0x0] -> blockDim.x
// BB0.S3: c[0x0][0x178] -> param_6
// BB1.S0: c[0x0][0x160] -> param_0
// BB1.S1: c[0x0][0x168] -> param_2
// BB1.S4: c[0x0][0x17c] -> param_7
// BB3.S0: c[0x0][0x17c] -> param_7
// BB15.S9: c[0x0][0x170] -> param_4
// BB15.S10: c[0x0][0x174] -> param_5
// ABI arg aliases (heuristic):
// param_0 -> arg0 (word32, confidence: low, words: {0})
// param_2 -> arg2 (word32, confidence: low, words: {0})
// param_4 -> arg4 (ptr64, confidence: high, words: {0, 1})
// param_6 -> arg6 (word32, confidence: low, words: {0})
// param_7 -> arg7 (word32, confidence: low, words: {0})
// Typed signature inferred from ABI aliases:
// param_0 -> arg0 (word32, confidence: low, words: {0})
// param_2 -> arg2 (word32, confidence: low, words: {0})
// param_4 -> arg4 (ptr64, confidence: high, words: {0, 1})
// param_6 -> arg6 (word32, confidence: low, words: {0})
// param_7 -> arg7 (word32, confidence: low, words: {0})
__global__ void kernel(uint32_t arg0, uint32_t arg2, uintptr_t arg4_ptr, uint32_t arg6, uint32_t arg7) {
  uint32_t v2;
  bool b0; // live-in
  uint32_t v3;
  uint32_t v4;
  uint32_t v5;
  uint32_t v6; // live-in
  uint32_t v7;
  uint32_t v8;
  bool b1;
  uint32_t v10;
  bool b2;
  uint32_t v11;
  bool b4;
  bool b5;
  bool b7;
  uint32_t v13;
  uint32_t v14;
  bool b8;
  uint32_t v15;
  uint32_t v16;
  bool b9;
  uint32_t v17;
  uint32_t v18;
  bool b10;
  uint32_t v19;
  uint32_t v20;
  bool b11;
  uint32_t v21;
  uint32_t v22;
  bool b12;
  uint32_t v23;
  uint32_t v24;
  bool b13;
  uint32_t v25;
  uint32_t v26;
  bool b14;
  uint32_t v27;
  uint32_t v28;
  bool b15;
  uint32_t v29;
  uint32_t v30;
  bool b16;
  uint32_t v31;
  uint32_t v32;
  bool b17;
  uint32_t v33;
  uint32_t v34;
  bool b18;
  uint32_t v35;
  uint32_t v36;
  bool b19;
  uint32_t v37;
  uint32_t v38;
  bool b20;
  uint32_t v39;
  bool b21;
  uint32_t v40;
  uint32_t v41;
  bool b22;
  uint32_t v43;
  uint32_t v44;
  bool b23;
  uint32_t v45;
  uint32_t v46;
  bool b24;
  uint32_t v47;
  uint32_t v48;
  bool b25;
  uint32_t v49;
  uint32_t v50;
  bool b26;
  uint32_t v51;
  uint32_t v52;
  bool b27;
  uint32_t v53;
  bool b28;
  uint32_t v54;
  uint32_t v55;
  bool b29;
  uint32_t v56;
  uint32_t v57;
  bool b30;
  uint32_t v58;
  bool b31;
  uint32_t v59;
  uint32_t v60;
  uint32_t v61;
  uint32_t v62;

  ctaid_x = blockIdx.x;
  tid_x = threadIdx.x;
  v2 = ctaid_x * blockDim.x + tid_x;
  if (b0) return;
    v3 = v2 * 4 + arg0;
    v4 = v2 * 4 + arg2;
    v5 = *addr64(v3, v6);
    v7 = *addr64(v4, 4);
    v8 = arg7;
    b1 = (int32_t)(v8) >= (int32_t)(1);
    v10 = v7 + v5;
    b2 = v10 > 1;
    v11 = !b2 ? (v7 * v5) : v10;
  if ((int32_t)(v8) >= (int32_t)(1)) {
      v3 = v8 - 1;
      v8 = v8 & 3;
    if (v3 >= 3) {
        v7 = -v8 + arg7;
      if ((int32_t)(v7) > (int32_t)(0)) {
          b4 = (int32_t)(v7) > (int32_t)(12);
          b5 = false;
        if ((int32_t)(v7) > (int32_t)(12)) {
          b5 = false;
        }
      }
    }
  }
    b7 = v11 > 0.5;
    v7 = v7 - 16;
    v13 = b7 ? 1066192077 : 0.8999999761581421;
    v14 = v13 * v11;
    b8 = v14 > 0.5;
    v15 = b8 ? 1066192077 : 0.8999999761581421;
    v16 = v14 * v15;
    b9 = v16 > 0.5;
    v17 = b9 ? 1066192077 : 0.8999999761581421;
    v18 = v16 * v17;
    b10 = v18 > 0.5;
    v19 = b10 ? 1066192077 : 0.8999999761581421;
    v20 = v18 * v19;
    b11 = v20 > 0.5;
    v21 = b11 ? 1066192077 : 0.8999999761581421;
    v22 = v20 * v21;
    b12 = v22 > 0.5;
    v23 = b12 ? 1066192077 : 0.8999999761581421;
    v24 = v22 * v23;
    b13 = v24 > 0.5;
    v25 = b13 ? 1066192077 : 0.8999999761581421;
    v26 = v24 * v25;
    b14 = v26 > 0.5;
    v27 = b14 ? 1066192077 : 0.8999999761581421;
    v28 = v26 * v27;
    b15 = v28 > 0.5;
    v29 = b15 ? 1066192077 : 0.8999999761581421;
    v30 = v28 * v29;
    b16 = v30 > 0.5;
    v31 = b16 ? 1066192077 : 0.8999999761581421;
    v32 = v30 * v31;
    b17 = v32 > 0.5;
    v33 = b17 ? 1066192077 : 0.8999999761581421;
    v34 = v32 * v33;
    b18 = v34 > 0.5;
    v35 = b18 ? 1066192077 : 0.8999999761581421;
    v36 = v34 * v35;
    b19 = v36 > 0.5;
    v37 = b19 ? 1066192077 : 0.8999999761581421;
    v38 = v36 * v37;
    b20 = v38 > 0.5;
    v39 = b20 ? 1066192077 : 0.8999999761581421;
    v5 = v38 * v39;
    b21 = v5 > 0.5;
    v40 = b21 ? 1066192077 : 0.8999999761581421;
    v3 = v5 * v40;
    b4 = v3 > 0.5;
    v41 = b4 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v41;
  do {
    b7 = v11 > 0.5;
    v7 = v7 - 16;
    v13 = b7 ? 1066192077 : 0.8999999761581421;
    v14 = v13 * v11;
    b8 = v14 > 0.5;
    v15 = b8 ? 1066192077 : 0.8999999761581421;
    v16 = v14 * v15;
    b9 = v16 > 0.5;
    v17 = b9 ? 1066192077 : 0.8999999761581421;
    v18 = v16 * v17;
    b10 = v18 > 0.5;
    v19 = b10 ? 1066192077 : 0.8999999761581421;
    v20 = v18 * v19;
    b11 = v20 > 0.5;
    v21 = b11 ? 1066192077 : 0.8999999761581421;
    v22 = v20 * v21;
    b12 = v22 > 0.5;
    v23 = b12 ? 1066192077 : 0.8999999761581421;
    v24 = v22 * v23;
    b13 = v24 > 0.5;
    v25 = b13 ? 1066192077 : 0.8999999761581421;
    v26 = v24 * v25;
    b14 = v26 > 0.5;
    v27 = b14 ? 1066192077 : 0.8999999761581421;
    v28 = v26 * v27;
    b15 = v28 > 0.5;
    v29 = b15 ? 1066192077 : 0.8999999761581421;
    v30 = v28 * v29;
    b16 = v30 > 0.5;
    v31 = b16 ? 1066192077 : 0.8999999761581421;
    v32 = v30 * v31;
    b17 = v32 > 0.5;
    v33 = b17 ? 1066192077 : 0.8999999761581421;
    v34 = v32 * v33;
    b18 = v34 > 0.5;
    v35 = b18 ? 1066192077 : 0.8999999761581421;
    v36 = v34 * v35;
    b19 = v36 > 0.5;
    v37 = b19 ? 1066192077 : 0.8999999761581421;
    v38 = v36 * v37;
    b20 = v38 > 0.5;
    v39 = b20 ? 1066192077 : 0.8999999761581421;
    v5 = v38 * v39;
    b21 = v5 > 0.5;
    v40 = b21 ? 1066192077 : 0.8999999761581421;
    v3 = v5 * v40;
    b4 = v3 > 0.5;
    v41 = b4 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v41;
    // phi merge: v12 <- phi(1066192077, 4)
  } while((int32_t)(v7) > (int32_t)(12));
  if ((int32_t)(v7) > (int32_t)(4)) {
    b22 = v11 > 0.5;
    v7 = v7 - 8;
    v43 = b22 ? 1066192077 : 0.8999999761581421;
    v44 = v11 * v43;
    b23 = v44 > 0.5;
    v45 = b23 ? 1066192077 : 0.8999999761581421;
    v46 = v44 * v45;
    b24 = v46 > 0.5;
    v47 = b24 ? 1066192077 : 0.8999999761581421;
    v48 = v46 * v47;
    b25 = v48 > 0.5;
    v49 = b25 ? 1066192077 : 0.8999999761581421;
    v50 = v48 * v49;
    b26 = v50 > 0.5;
    v51 = b26 ? 1066192077 : 0.8999999761581421;
    v52 = v50 * v51;
    b27 = v52 > 0.5;
    v53 = b27 ? 1066192077 : 0.8999999761581421;
    v5 = v52 * v53;
    b28 = v5 > 0.5;
    v54 = b28 ? 1066192077 : 0.8999999761581421;
    b5 = false;
    v3 = v5 * v54;
    b1 = v3 > 0.5;
    v55 = b1 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v55;
  }
  if (v7 != 0 || b5) {
    // phi merge: v42 <- phi(v42, 4)
  }
    b29 = v11 > 0.5;
    v7 = v7 - 4;
    v56 = b29 ? 1066192077 : 0.8999999761581421;
    v57 = v56 * v11;
    b30 = v57 > 0.5;
    v58 = b30 ? 1066192077 : 0.8999999761581421;
    v5 = v57 * v58;
    b31 = v5 > 0.5;
    v59 = b31 ? 1066192077 : 0.8999999761581421;
    v3 = v5 * v59;
    b1 = v3 > 0.5;
    v60 = b1 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v60;
  do {
    b29 = v11 > 0.5;
    v7 = v7 - 4;
    v56 = b29 ? 1066192077 : 0.8999999761581421;
    v57 = v56 * v11;
    b30 = v57 > 0.5;
    v58 = b30 ? 1066192077 : 0.8999999761581421;
    v5 = v57 * v58;
    b31 = v5 > 0.5;
    v59 = b31 ? 1066192077 : 0.8999999761581421;
    v3 = v5 * v59;
    b1 = v3 > 0.5;
    v60 = b1 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v60;
  } while(v7 != 0);
  if (v8 != 0) {
  }
    v8 = v8 - 1;
    b1 = v11 > 0.5;
    b2 = v8 != 0;
    v3 = b1 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v11;
  do {
    v8 = v8 - 1;
    b1 = v11 > 0.5;
    b2 = v8 != 0;
    v3 = b1 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v11;
  } while(v8 != 0);
  v61 = v2 + (arg4_ptr.lo32 << 2);
  v62 = lea_hi_x(v2, arg4_ptr.hi32, 2, b2);
  *addr64(v61, v62) = v11;
  // phi merge: v42 <- phi(v42, v42, 4)
  // phi merge: v5 <- phi(1066192077, v5, v5)
  return;
}
// --- End Structured Output ---
