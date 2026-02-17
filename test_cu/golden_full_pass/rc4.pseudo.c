// --- Structured Output ---
// ABI const-memory mapping (sample):
// BB0.S0: c[0x0][0x28] -> abi_internal_0x28
// BB0.S2: c[0x0][0x184] -> param_4[1]
// BB1.S1: c[0x0][0x118] -> c[0x0][0x118]
// BB2.S3: c[0x0][0x0] -> c[0x0][0x0]
// BB4.S0: c[0x0][0x168] -> param_1[0]
// BB5.S22: c[0x0][0x168] -> param_1[0]
// BB5.S35: c[0x0][0x168] -> param_1[0]
// BB5.S37: c[0x0][0x168] -> param_1[0]
// BB5.S40: c[0x0][0x160] -> param_0[0]
// BB5.S41: c[0x0][0x160] -> param_0[0]
// BB5.S42: c[0x0][0x164] -> param_0[1]
// BB5.S62: c[0x0][0x160] -> param_0[0]
// BB5.S63: c[0x0][0x160] -> param_0[0]
// BB5.S64: c[0x0][0x164] -> param_0[1]
// BB5.S78: c[0x0][0x160] -> param_0[0]
// BB5.S79: c[0x0][0x160] -> param_0[0]
// ABI arg aliases (heuristic):
// param_0 -> arg0 (ptr64, confidence: high, words: {0, 1})
// param_1 -> arg1 (word32, confidence: low, words: {0})
// param_2 -> arg2 (ptr64, confidence: high, words: {0, 1})
// param_3 -> arg3 (ptr64, confidence: high, words: {0, 1})
// param_4 -> arg4 (word32, confidence: low, words: {0, 1})
// Typed signature inferred from ABI aliases:
// param_0 -> arg0 (ptr64, confidence: high, words: {0, 1})
// param_1 -> arg1 (word32, confidence: low, words: {0})
// param_2 -> arg2 (ptr64, confidence: high, words: {0, 1})
// param_3 -> arg3 (ptr64, confidence: high, words: {0, 1})
// param_4 -> arg4 (word32, confidence: low, words: {0, 1})
void kernel(uint8_t* arg0_ptr, int32_t arg1_word0, uint8_t* arg2_ptr, uint8_t* arg3_ptr, uint32_t arg4_word0, uint32_t arg4_word1) {
  bool P0;
  bool P1;
  bool P2;
  bool P3;
  uint32_t R0;
  uint32_t R1;
  uint32_t R2;
  uint32_t R3;
  uintptr_t R4;
  uintptr_t R5;
  uint32_t R6;
  uint32_t R7;
  uintptr_t R8;
  uintptr_t R9;
  uintptr_t R10;
  uintptr_t R11;
  uintptr_t R12;
  uintptr_t R13;
  uint32_t R14;
  uint8_t R15;
  uint8_t R16;
  uint32_t R17;
  uint32_t R18;
  uint32_t R19;
  uint32_t R20;
  bool UP0;
  bool UP1;
  uint32_t UR4;
  uint32_t UR5;
  uint32_t UR6;
  uint32_t UR8;
  uint32_t UR9;
  uint32_t UR10;
  uint32_t UR11;

  BB0 {
    ctaid_x = blockIdx.x;
  }
  // Condition from BB0
  if (!(ctaid_x >= arg4_word1)) {
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
      __syncthreads();
    }
    // Condition from BB3
    if (!(b2)) {
      BB4 {
        v3 = abs(arg1_word0);
        v4 = RZ;
        v5 = i2f_rp(v3);
        v6 = rcp_approx(v5);
        v7 = v6 + 268435454;
        v8 = f2i_trunc_u32_ftz_ntz(v7);
        v9 = RZ;
        v10 = -v8;
        v11 = v10 * v3 + RZ;
        tid_x = RZ;
        v12 = mul_hi_u32(v8, v11) + v9;
      }
      // Loop header BB5
      while (tid_x != 256) {
        BB5 {
          v22 = abs(tid_x);
          v23 = shmem_u8[tid_x];
          v24 = abs(arg1_word0);
          b5 = tid_x >= RZ;
          v25 = mul_hi_u32(v12, v22);
          v26 = i2f_rp(v24);
          v25 = -v25;
          v27 = v24 * v25 + v22;
          b6 = v3 > v27;
          v28 = rcp_approx(v26);
          if (!b6) v29 = v27 - v24;
          b7 = v3 > v29;
          v30 = v28 + 268435454;
          v31 = f2i_trunc_u32_ftz_ntz(v30);
          if (!b7) v32 = v29 - v24;
          b8 = RZ != arg1_word0;
          v33 = v32;
          v34 = ~arg1_word0;
          if (!b5) v33 = -v33;
          v35 = !b8 ? v34 : v33;
          v36 = v35 + arg0_ptr.lo32;
          b9 = carry_u32_add3(v35, arg0_ptr.lo32, RZ);
          v37 = lea_hi_x_sx32(v35, arg0_ptr.hi32, 1, b9);
          v38 = *((uint8_t*)addr64(v36, v37));
          v39 = -v31;
          v40 = tid_x + 1;
          v41 = RZ;
          v42 = v39 * v24 + RZ;
          v43 = abs(v40);
          b10 = v40 >= RZ;
          v12 = mul_hi_u32(v31, v42) + v41;
          v44 = v43;
          v45 = mul_hi_u32(v12, v44);
          v45 = -v45;
          v46 = v24 * v45 + v44;
          v3 = v24;
          b11 = v3 > v46;
          if (!b11) v47 = v46 - v24;
          b12 = v3 > v47;
          if (!b12) v48 = v47 - v24;
          if (!b10) v48 = -v48;
          v49 = !b8 ? v34 : v48;
          v50 = v49 + arg0_ptr.lo32;
          b13 = carry_u32_add3(v49, arg0_ptr.lo32, RZ);
          v51 = lea_hi_x_sx32(v49, arg0_ptr.hi32, 1, b13);
          v52 = *((uint8_t*)addr64(v50, v51));
          v53 = tid_x + 2;
          v54 = abs(v53);
          b14 = v53 >= RZ;
          v55 = mul_hi_u32(v12, v54);
          v55 = -v55;
          v56 = v24 * v55 + v54;
          b15 = v3 > v56;
          if (!b15) v57 = v56 - v24;
          b16 = v3 > v57;
          if (!b16) v58 = v57 - v24;
          if (!b14) v58 = -v58;
          v59 = !b8 ? v34 : v58;
          v60 = v59 + arg0_ptr.lo32;
          b17 = carry_u32_add3(v59, arg0_ptr.lo32, RZ);
          v61 = lea_hi_x_sx32(v59, arg0_ptr.hi32, 1, b17);
          v62 = *((uint8_t*)addr64(v60, v61));
          v63 = tid_x + 3;
          v64 = abs(v63);
          b18 = v63 >= RZ;
          v65 = mul_hi_u32(v12, v64);
          v65 = -v65;
          v66 = v24 * v65 + v64;
          b19 = v3 > v66;
          if (!b19) v67 = v66 - v24;
          b20 = v3 > v67;
          if (!b20) v68 = v67 - v24;
          if (!b18) v68 = -v68;
          v69 = !b8 ? v34 : v68;
          v70 = v38 + v4 + v23;
          v71 = v69 + arg0_ptr.lo32;
          b21 = carry_u32_add3(v69, arg0_ptr.lo32, RZ);
          v72 = v70 >> 31;
          v73 = lea_hi_x_sx32(v69, arg0_ptr.hi32, 1, b21);
          v74 = hi32(v72 + (v70 << 8));
          v75 = *((uint8_t*)addr64(v71, v73));
          v76 = v74 & 4294967040;
          v77 = v70 - v76;
          v78 = tid_x + 4;
          v79 = shmem_u8[v77];
          v80 = abs(v78);
          b22 = v78 >= RZ;
          v81 = mul_hi_u32(v12, v80);
          v81 = -v81;
          v82 = v24 * v81 + v80;
          b23 = v3 > v82;
          if (!b23) v83 = v82 - v24;
          b24 = v3 > v83;
          shmem_u8[tid_x] = v79;
          shmem_u8[v77] = v23;
          v84 = shmem_u8[tid_x + 1];
          if (!b24) v85 = v83 - v24;
          if (!b22) v85 = -v85;
          v86 = !b8 ? v34 : v85;
          v87 = v86 + arg0_ptr.lo32;
          b25 = carry_u32_add3(v86, arg0_ptr.lo32, RZ);
          v88 = lea_hi_x_sx32(v86, arg0_ptr.hi32, 1, b25);
          v89 = v52 + v77 + v84;
          v90 = v89 >> 31;
          v91 = hi32(v90 + (v89 << 8));
          v92 = *((uint8_t*)addr64(v87, v88));
          v93 = v91 & 4294967040;
          v94 = v89 - v93;
          v95 = tid_x + 5;
          v96 = shmem_u8[v94];
          v97 = abs(v95);
          b26 = v95 >= RZ;
          v98 = mul_hi_u32(v12, v97);
          v98 = -v98;
          v99 = v24 * v98 + v97;
          b27 = v3 > v99;
          if (!b27) v100 = v99 - v24;
          b28 = v3 > v100;
          shmem_u8[tid_x + 1] = v96;
          shmem_u8[v94] = v84;
          v101 = shmem_u8[tid_x + 2];
          if (!b28) v102 = v100 - v24;
          if (!b26) v102 = -v102;
          v103 = !b8 ? v34 : v102;
          v104 = v103 + arg0_ptr.lo32;
          b29 = carry_u32_add3(v103, arg0_ptr.lo32, RZ);
          v105 = lea_hi_x_sx32(v103, arg0_ptr.hi32, 1, b29);
          v106 = v62 + v94 + v101;
          v107 = *((uint8_t*)addr64(v104, v105));
          v108 = v106 >> 31;
          v109 = hi32(v108 + (v106 << 8));
          v110 = v109 & 4294967040;
          v14 = v106 - v110;
          v111 = shmem_u8[v14];
          v112 = tid_x + 6;
          v113 = abs(v112);
          v114 = mul_hi_u32(v12, v113);
          v114 = -v114;
          v115 = v24 * v114 + v113;
          b30 = v3 > v115;
          shmem_u8[tid_x + 2] = v111;
          shmem_u8[v14] = v101;
          v13 = shmem_u8[tid_x + 3];
          if (!b30) v116 = v115 - v24;
          b31 = v112 >= RZ;
          b32 = v3 > v116;
          if (!b32) v117 = v116 - v24;
          if (!b31) v117 = -v117;
          v118 = !b8 ? v34 : v117;
          v119 = v118 + arg0_ptr.lo32;
          b33 = carry_u32_add3(v118, arg0_ptr.lo32, RZ);
          v120 = v75 + v14 + v13;
          v121 = v120 >> 31;
          v122 = lea_hi_x_sx32(v118, arg0_ptr.hi32, 1, b33);
          v123 = hi32(v121 + (v120 << 8));
          v124 = *((uint8_t*)addr64(v119, v122));
          v125 = v123 & 4294967040;
          v15 = v120 - v125;
          v126 = shmem_u8[v15];
          v127 = tid_x + 7;
          v128 = abs(v127);
          v129 = mul_hi_u32(v12, v128);
          v129 = -v129;
          v130 = v24 * v129 + v128;
          b34 = v3 > v130;
          shmem_u8[tid_x + 3] = v126;
          shmem_u8[v15] = v13;
          v131 = shmem_u8[tid_x + 4];
          if (!b34) v132 = v130 - v24;
          b3 = v127 >= RZ;
          b4 = v3 > v132;
          if (!b4) v17 = v132 - v24;
          if (!b3) v17 = -v17;
          v133 = !b8 ? v34 : v17;
          v134 = v133 + arg0_ptr.lo32;
          b35 = carry_u32_add3(v133, arg0_ptr.lo32, RZ);
          v135 = v92 + v15 + v131;
          v136 = v135 >> 31;
          v137 = lea_hi_x_sx32(v133, arg0_ptr.hi32, 1, b35);
          v138 = hi32(v136 + (v135 << 8));
          v139 = *((uint8_t*)addr64(v134, v137));
          v140 = v138 & 4294967040;
          v21 = v135 - v140;
          v141 = shmem_u8[v21];
          shmem_u8[tid_x + 4] = v141;
          shmem_u8[v21] = v131;
          v18 = shmem_u8[tid_x + 5];
          v142 = v107 + v21 + v18;
          v143 = v142 >> 31;
          v144 = hi32(v143 + (v142 << 8));
          v145 = v144 & 4294967040;
          v6 = v142 - v145;
          v146 = shmem_u8[v6];
          shmem_u8[tid_x + 5] = v146;
          shmem_u8[v6] = v18;
          v20 = shmem_u8[tid_x + 6];
          v147 = v124 + v6 + v20;
          v148 = v147 >> 31;
          v149 = hi32(v148 + (v147 << 8));
          v150 = v149 & 4294967040;
          v9 = v147 - v150;
          v11 = shmem_u8[v9];
          shmem_u8[tid_x + 6] = v11;
          shmem_u8[v9] = v20;
          v16 = shmem_u8[tid_x + 7];
          v19 = v139 + v9 + v16;
          v151 = v19 >> 31;
          v152 = hi32(v151 + (v19 << 8));
          v153 = v152 & 4294967040;
          v4 = v19 - v153;
          v8 = shmem_u8[v4];
          shmem_u8[tid_x + 7] = v8;
          shmem_u8[v4] = v16;
          tid_x = tid_x + 8;
          // 20 phi node(s) omitted
          // phi merge: v13 <- phi(v13, v13)
          // phi merge: v14 <- phi(v14, v14)
          // phi merge: v15 <- phi(v15, v15)
          // phi merge: v16 <- phi(v16, v16)
          // phi merge: v17 <- phi(v17, v17)
          // phi merge: v18 <- phi(v18, v18)
          // phi merge: v19 <- phi(v19, v19)
          // phi merge: v20 <- phi(v20, v20)
          // phi merge: v21 <- phi(v21, v21)
          // phi merge: v11 <- phi(v11, v11)
          // phi merge: v6 <- phi(v6, v6)
          // phi merge: v4 <- phi(v4, v4)
          // phi merge: v12 <- phi(v12, v12)
          // phi merge: v8 <- phi(v8, v8)
          // phi merge: v9 <- phi(v9, v9)
          // phi merge: v3 <- phi(v3, v3)
          // phi merge: v2 <- phi(v2, v2)
          // phi merge: b3 <- phi(b3, b3)
          // phi merge: b4 <- phi(b4, b4)
          // phi merge: b1 <- phi(b1, b1)
        }
      }
    }
    BB6 {
      __syncthreads();
    }
    // Condition from BB6
    if (!(b2)) {
      // Condition from BB7
      if (v154 >= 1) {
        // Condition from BB8
        if (v155 >= 3) {
          BB9 {
            v14 = ctaid_x * arg4_word0 + RZ;
            v21 = v156 - arg4_word0;
            v158 = RZ;
            u2 = uldc64(arg2_ptr.lo32);
            v6 = v14 + 3;
            u3 = uldc64(arg3_ptr.lo32);
            v13 = v14 >> 31;
          }
          // Loop header BB10
          while (v21 != RZ) {
            BB10 {
              v162 = v6 - 3;
              b41 = v162 >= v157;
              v163 = v14 + u2;
              b42 = carry_u32_add3(v14, u2, RZ);
              v164 = v13 + u4 + (b42 ? 1 : 0);
              if (!b41) v165 = *((uint8_t*)addr64(v163, v164));
              u7 = u1 + 1;
              u8 = u7 >> 31;
              u9 = hi32(u8 + (u7 << 8));
              u10 = u9 & 4294967040;
              u11 = u7 - u10;
              v166 = shmem_u8[u11];
              v167 = v166 + v159;
              v168 = v167 >> 31;
              v169 = hi32(v168 + (v167 << 8));
              v170 = v169 & 4294967040;
              v171 = v167 - v170;
              v172 = v6 - 2;
              v173 = shmem_u8[v171];
              b43 = v172 >= v157;
              v9 = v14 + u3;
              b3 = carry_u32_add3(v14, u3, RZ);
              shmem_u8[u11] = v173;
              shmem_u8[v171] = v166;
              v174 = shmem_u8[u11];
              v175 = v166 + v174;
              if (!b41) v176 = v175 & 255;
              v8 = v13 + u5 + (b3 ? 1 : 0);
              if (!b41) v177 = shmem_u8[v176];
              u12 = u11 + 1;
              u13 = u12 >> 31;
              u14 = hi32(u13 + (u12 << 8));
              u15 = u14 & 4294967040;
              if (!b41) v178 = v165 ^ v177;
              if (!b41) *((uint8_t*)addr64(v9, v8)) = v178;
              if (!b43) v179 = *((uint8_t*)(addr64(v163, v164) + 1));
              u16 = u12 - u15;
              v180 = shmem_u8[u16];
              v181 = v171 + v180;
              v182 = v181 >> 31;
              v183 = hi32(v182 + (v181 << 8));
              v184 = v183 & 4294967040;
              v160 = v181 - v184;
              v185 = shmem_u8[v160];
              shmem_u8[u16] = v185;
              shmem_u8[v160] = v180;
              v186 = shmem_u8[u16];
              v187 = v180 + v186;
              v188 = v6 - 1;
              if (!b43) v189 = v187 & 255;
              b4 = v188 >= v157;
              if (!b43) v190 = shmem_u8[v189];
              u17 = u16 + 1;
              u18 = u17 >> 31;
              u19 = hi32(u18 + (u17 << 8));
              u20 = u19 & 4294967040;
              u21 = u17 - u20;
              v191 = shmem_u8[u21];
              if (!b43) v192 = v179 ^ v190;
              if (!b43) *((uint8_t*)(addr64(v9, v8) + 1)) = v192;
              if (!b4) v193 = *((uint8_t*)(addr64(v163, v164) + 2));
              v194 = v160 + v191;
              b44 = v6 >= v157;
              v195 = v194 >> 31;
              v196 = hi32(v195 + (v194 << 8));
              v197 = v196 & 4294967040;
              v161 = v194 - v197;
              v198 = shmem_u8[v161];
              shmem_u8[u21] = v198;
              shmem_u8[v161] = v191;
              v199 = shmem_u8[u21];
              v200 = v191 + v199;
              if (!b4) v201 = v200 & 255;
              if (!b4) v202 = shmem_u8[v201];
              u22 = u21 + 1;
              u23 = u22 >> 31;
              u24 = hi32(u23 + (u22 << 8));
              u6 = u24 & 4294967040;
              u1 = u22 - u6;
              v19 = shmem_u8[u1];
              if (!b4) v16 = v193 ^ v202;
              if (!b4) *((uint8_t*)(addr64(v9, v8) + 2)) = v16;
              if (!b44) v15 = *((uint8_t*)(addr64(v163, v164) + 3));
              v203 = v161 + v19;
              v21 = v21 + 4;
              u3 = u3 + 4;
              b40 = u3 + 4;
              v158 = v158 + 4;
              u2 = u2 + 4;
              b39 = u2 + 4;
              v204 = v203 >> 31;
              u5 = URZ + u5 + (b40 ? 1 : 0);
              v6 = v6 + 4;
              u4 = URZ + u4 + (b39 ? 1 : 0);
              v205 = hi32(v204 + (v203 << 8));
              v17 = v205 & 4294967040;
              v159 = v203 - v17;
              v155 = shmem_u8[v159];
              shmem_u8[u1] = v155;
              shmem_u8[v159] = v19;
              v206 = shmem_u8[u1];
              v207 = v19 + v206;
              if (!b44) v208 = v207 & 255;
              if (!b44) v3 = shmem_u8[v208];
              if (!b44) v18 = v15 ^ v3;
              if (!b44) *((uint8_t*)(addr64(v9, v8) + 3)) = v18;
              // 26 phi node(s) omitted
              // phi merge: u4 <- phi(u4, u4)
              // phi merge: u2 <- phi(u2, u2)
              // phi merge: u5 <- phi(u5, u5)
              // phi merge: u3 <- phi(u3, u3)
              // phi merge: u6 <- phi(u6, u6)
              // phi merge: u1 <- phi(u1, u1)
              // phi merge: b39 <- phi(b39, b39)
              // phi merge: b40 <- phi(b40, b40)
              // phi merge: v160 <- phi(v160, v160)
              // phi merge: v161 <- phi(v161, v161)
              // phi merge: v15 <- phi(v15, v15)
              // phi merge: v16 <- phi(v16, v16)
              // phi merge: v17 <- phi(v17, v17)
              // phi merge: v18 <- phi(v18, v18)
              // phi merge: v19 <- phi(v19, v19)
              // phi merge: v159 <- phi(v159, v159)
              // phi merge: v21 <- phi(v21, v21)
              // phi merge: v158 <- phi(v158, v158)
              // phi merge: v6 <- phi(v6, v6)
              // phi merge: v8 <- phi(v8, v8)
              // phi merge: v9 <- phi(v9, v9)
              // phi merge: v3 <- phi(v3, v3)
              // phi merge: v155 <- phi(v155, v155)
              // phi merge: b3 <- phi(b3, b3)
              // phi merge: b4 <- phi(b4, b4)
              // phi merge: b37 <- phi(b37, b37)
            }
          }
        }
        // Condition from BB11
        if (b38) {
          BB12 {
            v209 = ctaid_x * arg4_word0 + v158;
            v210 = v209 + arg3_ptr.lo32;
            b45 = carry_u32_add3(v209, arg3_ptr.lo32, RZ);
            v211 = v209 + arg2_ptr.lo32;
            b46 = carry_u32_add3(v209, arg2_ptr.lo32, RZ);
            v212 = v209 >> 31;
            v213 = v212 + arg3_ptr.hi32 + (b45 ? 1 : 0);
            v214 = v212 + arg2_ptr.hi32 + (b46 ? 1 : 0);
          }
          // Loop header BB13
          while (v156 != RZ) {
            BB13 {
              b47 = v209 >= v157;
              if (!b47) v215 = v211;
              if (!b47) v216 = v214;
              if (!b47) v217 = *((uint8_t*)addr64(v215, v216));
              u25 = u1 + 1;
              v156 = v156 - 1;
              v211 = v211 + 1;
              b4 = carry_u32_add3(v211, 1, RZ);
              u26 = u25 >> 31;
              v209 = v209 + 1;
              u27 = hi32(u26 + (u25 << 8));
              v214 = v214 + (b4 ? 1 : 0);
              u6 = u27 & 4294967040;
              u1 = u25 - u6;
              v9 = shmem_u8[u1];
              v218 = v9 + v159;
              v219 = v218 >> 31;
              v220 = hi32(v219 + (v218 << 8));
              v221 = v220 & 4294967040;
              v159 = v218 - v221;
              v222 = shmem_u8[v159];
              shmem_u8[u1] = v222;
              shmem_u8[v159] = v9;
              v223 = shmem_u8[u1];
              if (!b47) v212 = v210;
              v210 = v210 + 1;
              b46 = carry_u32_add3(v210, 1, RZ);
              v224 = v9 + v223;
              if (!b47) v225 = v224 & 255;
              if (!b47) v6 = shmem_u8[v225];
              if (!b47) v3 = v213;
              v213 = v213 + (b46 ? 1 : 0);
              if (!b47) v8 = v217 ^ v6;
              if (!b47) *((uint8_t*)addr64(v212, v3)) = v8;
              // 17 phi node(s) omitted
              // phi merge: u6 <- phi(u6, u6)
              // phi merge: u1 <- phi(u1, u1)
              // phi merge: v213 <- phi(v213, v213)
              // phi merge: v210 <- phi(v210, v210)
              // phi merge: v159 <- phi(v159, v159)
              // phi merge: v214 <- phi(v214, v214)
              // phi merge: v211 <- phi(v211, v211)
              // phi merge: v6 <- phi(v6, v6)
              // phi merge: v156 <- phi(v156, v156)
              // phi merge: v8 <- phi(v8, v8)
              // phi merge: v9 <- phi(v9, v9)
              // phi merge: v3 <- phi(v3, v3)
              // phi merge: v212 <- phi(v212, v212)
              // phi merge: v209 <- phi(v209, v209)
              // phi merge: b4 <- phi(b4, b4)
              // phi merge: b46 <- phi(b46, b46)
              // phi merge: b45 <- phi(b45, b45)
            }
          }
        }
      }
    }
  }
}
// --- End Structured Output ---
