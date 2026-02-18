// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S3: c[0x0][0x0] -> blockDimX
// BB0.S4: c[0x0][0x178] -> param_6
// BB1.S1: c[0x0][0x118] -> c[0x0][0x118]
// BB1.S2: c[0x0][0x160] -> param_0
// BB1.S3: c[0x0][0x168] -> param_2
// BB1.S6: c[0x0][0x17c] -> param_7
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
void kernel(uint32_t arg0, uint32_t arg2, uintptr_t arg4_ptr, uint32_t arg6, uint32_t arg7) {
  bool P0;
  bool P1;
  bool P2;
  uint32_t R0;
  uint32_t R1;
  uintptr_t R2;
  uintptr_t R3;
  uintptr_t R4;
  uint32_t R5;
  uintptr_t R6;
  uint32_t R7;
  float R9;
  uint32_t UR4;
  uint32_t UR5;
  uint32_t v0;
  uint32_t abi_internal_0x28; // live-in
  uint32_t v3;
  uint32_t v4;
  uint32_t u0;
  uint32_t u1;
  uint32_t v5;
  uint32_t v6;
  uint32_t v7;
  uint32_t v8; // live-in
  uint32_t v9;
  uint32_t v10;
  uint32_t v11;
  bool b1;
  uint32_t v12;
  bool b2;
  uint32_t v13;
  bool b4;
  bool b5;
  bool b7;
  uint32_t v14;
  bool b6;
  uint32_t v15;
  bool b8;
  uint32_t v16;
  uint32_t v17;
  bool b9;
  uint32_t v18;
  uint32_t v19;
  bool b10;
  uint32_t v20;
  uint32_t v21;
  bool b11;
  uint32_t v22;
  uint32_t v23;
  bool b12;
  uint32_t v24;
  uint32_t v25;
  bool b13;
  uint32_t v26;
  uint32_t v27;
  bool b14;
  uint32_t v28;
  uint32_t v29;
  bool b15;
  uint32_t v30;
  uint32_t v31;
  bool b16;
  uint32_t v32;
  uint32_t v33;
  bool b17;
  uint32_t v34;
  uint32_t v35;
  bool b18;
  uint32_t v36;
  uint32_t v37;
  bool b19;
  uint32_t v38;
  uint32_t v39;
  bool b20;
  uint32_t v40;
  bool b21;
  uint32_t v41;
  uint32_t v42;
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
  bool b3; // live-in
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

  BB0 {
    v0 = abi_internal_0x28;
    ctaid_x = blockIdx.x;
    tid_x = threadIdx.x;
    v3 = ctaid_x * blockDimX + tid_x;
  }
  // Condition from BB0
  if (!((int32_t)(v3) >= (int32_t)(arg6))) {
    BB1 {
      v4 = 4;
      u0 = c[0x0][0x118];
      u1 = ConstMem(0, 284);
      v5 = v3 * v4 + arg0;
      v6 = v3 * v4 + arg2;
      v7 = *addr64(v5, v8);
      v9 = *addr64(v6, v4);
      v10 = arg7;
      v11 = ((int32_t)v3) >> 31;
      b1 = (int32_t)(v10) >= (int32_t)(1);
      v12 = v9 + v7;
      b2 = v12 > 1;
      v13 = !b2 ? (v9 * v7) : v12;
    }
    // Condition from BB1
    if ((int32_t)(v10) >= (int32_t)(1)) {
      BB2 {
        v5 = v10 - 1;
        v10 = v10 & 3;
      }
      // Condition from BB2
      if (v5 >= 3) {
        BB3 {
          v9 = -v10 + arg7;
        }
        // Condition from BB3
        if ((int32_t)(v9) > (int32_t)(0)) {
          BB4 {
            b4 = (int32_t)(v9) > (int32_t)(12);
            b5 = PLOP3.LUT(PT, PT, PT, PT, 128, 0);
          }
          // Condition from BB4
          if ((int32_t)(v9) > (int32_t)(12)) {
            BB5 {
              b5 = PLOP3.LUT(PT, PT, PT, PT, 8, 0);
            }
            BB6 {
              v4 = 1066192077;
              b7 = v13 > 0.5;
              v9 = v9 - 16;
              v14 = b7 ? v4 : 0.8999999761581421;
              b6 = (int32_t)(v9) > (int32_t)(12);
              v15 = v14 * v13;
              b8 = v15 > 0.5;
              v16 = b8 ? v4 : 0.8999999761581421;
              v17 = v15 * v16;
              b9 = v17 > 0.5;
              v18 = b9 ? v4 : 0.8999999761581421;
              v19 = v17 * v18;
              b10 = v19 > 0.5;
              v20 = b10 ? v4 : 0.8999999761581421;
              v21 = v19 * v20;
              b11 = v21 > 0.5;
              v22 = b11 ? v4 : 0.8999999761581421;
              v23 = v21 * v22;
              b12 = v23 > 0.5;
              v24 = b12 ? v4 : 0.8999999761581421;
              v25 = v23 * v24;
              b13 = v25 > 0.5;
              v26 = b13 ? v4 : 0.8999999761581421;
              v27 = v25 * v26;
              b14 = v27 > 0.5;
              v28 = b14 ? v4 : 0.8999999761581421;
              v29 = v27 * v28;
              b15 = v29 > 0.5;
              v30 = b15 ? v4 : 0.8999999761581421;
              v31 = v29 * v30;
              b16 = v31 > 0.5;
              v32 = b16 ? v4 : 0.8999999761581421;
              v33 = v31 * v32;
              b17 = v33 > 0.5;
              v34 = b17 ? v4 : 0.8999999761581421;
              v35 = v33 * v34;
              b18 = v35 > 0.5;
              v36 = b18 ? v4 : 0.8999999761581421;
              v37 = v35 * v36;
              b19 = v37 > 0.5;
              v38 = b19 ? v4 : 0.8999999761581421;
              v39 = v37 * v38;
              b20 = v39 > 0.5;
              v40 = b20 ? v4 : 0.8999999761581421;
              v7 = v39 * v40;
              b21 = v7 > 0.5;
              v41 = b21 ? v4 : 0.8999999761581421;
              v5 = v7 * v41;
              b4 = v5 > 0.5;
              v42 = b4 ? v4 : 0.8999999761581421;
              v13 = v5 * v42;
            }
            // Loop header BB6
            while ((int32_t)(v9) > (int32_t)(12)) {
              BB6 {
                v4 = 1066192077;
                b7 = v13 > 0.5;
                v9 = v9 - 16;
                v14 = b7 ? v4 : 0.8999999761581421;
                b6 = (int32_t)(v9) > (int32_t)(12);
                v15 = v14 * v13;
                b8 = v15 > 0.5;
                v16 = b8 ? v4 : 0.8999999761581421;
                v17 = v15 * v16;
                b9 = v17 > 0.5;
                v18 = b9 ? v4 : 0.8999999761581421;
                v19 = v17 * v18;
                b10 = v19 > 0.5;
                v20 = b10 ? v4 : 0.8999999761581421;
                v21 = v19 * v20;
                b11 = v21 > 0.5;
                v22 = b11 ? v4 : 0.8999999761581421;
                v23 = v21 * v22;
                b12 = v23 > 0.5;
                v24 = b12 ? v4 : 0.8999999761581421;
                v25 = v23 * v24;
                b13 = v25 > 0.5;
                v26 = b13 ? v4 : 0.8999999761581421;
                v27 = v25 * v26;
                b14 = v27 > 0.5;
                v28 = b14 ? v4 : 0.8999999761581421;
                v29 = v27 * v28;
                b15 = v29 > 0.5;
                v30 = b15 ? v4 : 0.8999999761581421;
                v31 = v29 * v30;
                b16 = v31 > 0.5;
                v32 = b16 ? v4 : 0.8999999761581421;
                v33 = v31 * v32;
                b17 = v33 > 0.5;
                v34 = b17 ? v4 : 0.8999999761581421;
                v35 = v33 * v34;
                b18 = v35 > 0.5;
                v36 = b18 ? v4 : 0.8999999761581421;
                v37 = v35 * v36;
                b19 = v37 > 0.5;
                v38 = b19 ? v4 : 0.8999999761581421;
                v39 = v37 * v38;
                b20 = v39 > 0.5;
                v40 = b20 ? v4 : 0.8999999761581421;
                v7 = v39 * v40;
                b21 = v7 > 0.5;
                v41 = b21 ? v4 : 0.8999999761581421;
                v5 = v7 * v41;
                b4 = v5 > 0.5;
                v42 = b4 ? v4 : 0.8999999761581421;
                v13 = v5 * v42;
                // 7 phi node(s) omitted
                // phi merge: v13 <- phi(v13, v13)
                // phi merge: v4 <- phi(v4, v4)
                // phi merge: v9 <- phi(v9, v9)
                // phi merge: v7 <- phi(v7, v7)
                // phi merge: v5 <- phi(v5, v5)
                // phi merge: b6 <- phi(b6, b6)
                // phi merge: b4 <- phi(b4, b4)
              }
            }
          }
          // Condition from BB7
          if ((int32_t)(v9) > (int32_t)(4)) {
            BB8 {
              v4 = 1066192077;
              b22 = v13 > 0.5;
              v9 = v9 - 8;
              v43 = b22 ? v4 : 0.8999999761581421;
              v44 = v13 * v43;
              b23 = v44 > 0.5;
              v45 = b23 ? v4 : 0.8999999761581421;
              v46 = v44 * v45;
              b24 = v46 > 0.5;
              v47 = b24 ? v4 : 0.8999999761581421;
              v48 = v46 * v47;
              b25 = v48 > 0.5;
              v49 = b25 ? v4 : 0.8999999761581421;
              v50 = v48 * v49;
              b26 = v50 > 0.5;
              v51 = b26 ? v4 : 0.8999999761581421;
              v52 = v50 * v51;
              b27 = v52 > 0.5;
              v53 = b27 ? v4 : 0.8999999761581421;
              v7 = v52 * v53;
              b28 = v7 > 0.5;
              v54 = b28 ? v4 : 0.8999999761581421;
              b5 = PLOP3.LUT(PT, PT, PT, PT, 8, 0);
              v5 = v7 * v54;
              b1 = v5 > 0.5;
              v55 = b1 ? v4 : 0.8999999761581421;
              v13 = v5 * v55;
            }
          }
          // Condition from BB9
          if (b3) {
            BB10 {
              v4 = 1066192077;
              // 8 phi node(s) omitted
              // phi merge: v13 <- phi(v13, v13)
              // phi merge: v4 <- phi(v4, v4)
              // phi merge: v9 <- phi(v9, v9)
              // phi merge: v7 <- phi(v7, v7)
              // phi merge: v5 <- phi(v5, v5)
              // phi merge: b6 <- phi(b6, b6)
              // phi merge: b1 <- phi(b1, b1)
              // phi merge: b3 <- phi(b3, b3)
            }
            BB11 {
              b29 = v13 > 0.5;
              v9 = v9 - 4;
              v56 = b29 ? v4 : 0.8999999761581421;
              v57 = v56 * v13;
              b30 = v57 > 0.5;
              v58 = b30 ? v4 : 0.8999999761581421;
              v7 = v57 * v58;
              b31 = v7 > 0.5;
              v59 = b31 ? v4 : 0.8999999761581421;
              b3 = v9 != 0;
              v5 = v7 * v59;
              b1 = v5 > 0.5;
              v60 = b1 ? v4 : 0.8999999761581421;
              v13 = v5 * v60;
            }
            // Loop header BB11
            while (v9 != 0) {
              BB11 {
                b29 = v13 > 0.5;
                v9 = v9 - 4;
                v56 = b29 ? v4 : 0.8999999761581421;
                v57 = v56 * v13;
                b30 = v57 > 0.5;
                v58 = b30 ? v4 : 0.8999999761581421;
                v7 = v57 * v58;
                b31 = v7 > 0.5;
                v59 = b31 ? v4 : 0.8999999761581421;
                b3 = v9 != 0;
                v5 = v7 * v59;
                b1 = v5 > 0.5;
                v60 = b1 ? v4 : 0.8999999761581421;
                v13 = v5 * v60;
                // 6 phi node(s) omitted
                // phi merge: v13 <- phi(v13, v13)
                // phi merge: v9 <- phi(v9, v9)
                // phi merge: v7 <- phi(v7, v7)
                // phi merge: v5 <- phi(v5, v5)
                // phi merge: b1 <- phi(b1, b1)
                // phi merge: b3 <- phi(b3, b3)
              }
            }
          }
          // Condition from BB12
          if (v10 != 0) {
            BB13 {
              v7 = 1066192077;
            }
            BB14 {
              v10 = v10 - 1;
              b1 = v13 > 0.5;
              b2 = v10 != 0;
              v5 = b1 ? v7 : 0.8999999761581421;
              v13 = v5 * v13;
            }
            // Loop header BB14
            while (v10 != 0) {
              BB14 {
                v10 = v10 - 1;
                b1 = v13 > 0.5;
                b2 = v10 != 0;
                v5 = b1 ? v7 : 0.8999999761581421;
                v13 = v5 * v13;
                // 5 phi node(s) omitted
                // phi merge: v13 <- phi(v13, v13)
                // phi merge: v5 <- phi(v5, v5)
                // phi merge: v10 <- phi(v10, v10)
                // phi merge: b1 <- phi(b1, b1)
                // phi merge: b2 <- phi(b2, b2)
              }
            }
          }
          BB15 {
            v61 = v3 + (arg4_ptr.lo32 << 2);
            v62 = LEA.HI.X(v3, arg4_ptr.hi32, v11, 2, b2);
            *addr64(v61, v62) = v13;
            return;
            // 9 phi node(s) omitted
            // phi merge: v13 <- phi(v13, v13, v13)
            // phi merge: v4 <- phi(v4, v4, v4)
            // phi merge: v9 <- phi(v9, v9, v9)
            // phi merge: v7 <- phi(v7, v7, v7)
            // phi merge: v5 <- phi(v5, v5, v5)
            // phi merge: v10 <- phi(v10, v10, v10)
            // phi merge: b6 <- phi(b6, b6, b6)
            // phi merge: b1 <- phi(b1, b1, b1)
            // phi merge: b2 <- phi(b2, b2, b2)
          }
        }
      }
    }
  }
}
// --- End Structured Output ---
