BB0 {
  v1 = blockIdx.x;
}
// Condition from BB0
if (!(v1 >= ConstMem(0, 388))) {
  BB1 {
    v2 = threadIdx.x;
  }
  // Condition from BB1
  if (!(v2 > 255)) {
    // Loop header BB2
    while (!(v2 >= 256)) {
      BB2 {
        shmem_u8[v2] = v2;
        v2 = v2 + ConstMem(0, 0);
        // 2 phi node(s) omitted
      }
    }
  }
  // Condition from BB3
  if (!(v2 != RZ)) {
    BB4 {
      v3 = IABS(ConstMem(0, 360));
      v4 = RZ;
      v5 = I2F.RP(v3);
      v6 = MUFU.RCP(v5);
      v7 = v6 + 268435454;
      v8 = F2I.FTZ.U32.TRUNC.NTZ(v7);
      v9 = RZ;
      v10 = -v8;
      v11 = IMAD(v10, v3, RZ);
      v2 = RZ;
      v12 = IMAD.HI.U32(v8, v11, v9);
    }
    // Loop header BB5
    while (v2 != 256) {
      BB5 {
        v22 = IABS(v2);
        v23 = shmem_u8[v2];
        v24 = IABS(ConstMem(0, 360));
        b5 = v2 >= RZ;
        v25 = IMAD.HI.U32(v12, v22, RZ);
        v26 = I2F.RP(v24);
        v25 = -v25;
        v27 = IMAD(v24, v25, v22);
        b6 = v3 > v27;
        v28 = MUFU.RCP(v26);
        if (!b6) v29 = v27 - v24;
        b7 = v3 > v29;
        v30 = v28 + 268435454;
        v31 = F2I.FTZ.U32.TRUNC.NTZ(v30);
        if (!b7) v32 = v29 - v24;
        b8 = RZ != ConstMem(0, 360);
        v33 = v32;
        v34 = ~ConstMem(0, 360);
        if (!b5) v33 = -v33;
        v35 = !P1 ? v34 : v33;
        v36 = v35 + ConstMem(0, 352);
        v37 = hi32(v35 + (ConstMem(0, 356) << 1)) + (b5 ? 1 : 0);
        v38 = *v36@64;
        v39 = -v31;
        v40 = v2 + 1;
        v41 = RZ;
        v42 = IMAD(v39, v24, RZ);
        v43 = IABS(v40);
        b9 = v40 >= RZ;
        v12 = IMAD.HI.U32(v31, v42, v41);
        v44 = v43;
        v45 = IMAD.HI.U32(v12, v44, RZ);
        v45 = -v45;
        v46 = IMAD(v24, v45, v44);
        v3 = v24;
        b10 = v3 > v46;
        if (!b10) v47 = v46 - v24;
        b11 = v3 > v47;
        if (!b11) v48 = v47 - v24;
        if (!b9) v48 = -v48;
        v49 = !P1 ? v34 : v48;
        v50 = v49 + ConstMem(0, 352);
        v51 = hi32(v49 + (ConstMem(0, 356) << 1)) + (b11 ? 1 : 0);
        v52 = *v50@64;
        v53 = v2 + 2;
        v54 = IABS(v53);
        b12 = v53 >= RZ;
        v55 = IMAD.HI.U32(v12, v54, RZ);
        v55 = -v55;
        v56 = IMAD(v24, v55, v54);
        b13 = v3 > v56;
        if (!b13) v57 = v56 - v24;
        b14 = v3 > v57;
        if (!b14) v58 = v57 - v24;
        if (!b12) v58 = -v58;
        v59 = !P1 ? v34 : v58;
        v60 = v59 + ConstMem(0, 352);
        v61 = hi32(v59 + (ConstMem(0, 356) << 1)) + (b14 ? 1 : 0);
        v62 = *v60@64;
        v63 = v2 + 3;
        v64 = IABS(v63);
        b15 = v63 >= RZ;
        v65 = IMAD.HI.U32(v12, v64, RZ);
        v65 = -v65;
        v66 = IMAD(v24, v65, v64);
        b16 = v3 > v66;
        if (!b16) v67 = v66 - v24;
        b17 = v3 > v67;
        if (!b17) v68 = v67 - v24;
        if (!b15) v68 = -v68;
        v69 = !P1 ? v34 : v68;
        v70 = IADD3(v38, v4, v23);
        v71 = v69 + ConstMem(0, 352);
        v72 = v70 >> 31;
        v73 = hi32(v69 + (ConstMem(0, 356) << 1)) + (b17 ? 1 : 0);
        v74 = hi32(v72 + (v70 << 8));
        v75 = *v71@64;
        v76 = v74 & 4294967040;
        v77 = v70 - v76;
        v78 = v2 + 4;
        v79 = shmem_u8[v77];
        v80 = IABS(v78);
        b18 = v78 >= RZ;
        v81 = IMAD.HI.U32(v12, v80, RZ);
        v81 = -v81;
        v82 = IMAD(v24, v81, v80);
        b19 = v3 > v82;
        if (!b19) v83 = v82 - v24;
        b20 = v3 > v83;
        shmem_u8[v2] = v79;
        shmem_u8[v77] = v23;
        v84 = shmem_u8[v2 + 1];
        if (!b20) v85 = v83 - v24;
        if (!b18) v85 = -v85;
        v86 = !P1 ? v34 : v85;
        v87 = v86 + ConstMem(0, 352);
        v88 = hi32(v86 + (ConstMem(0, 356) << 1)) + (b20 ? 1 : 0);
        v89 = IADD3(v52, v77, v84);
        v90 = v89 >> 31;
        v91 = hi32(v90 + (v89 << 8));
        v92 = *v87@64;
        v93 = v91 & 4294967040;
        v94 = v89 - v93;
        v95 = v2 + 5;
        v96 = shmem_u8[v94];
        v97 = IABS(v95);
        b21 = v95 >= RZ;
        v98 = IMAD.HI.U32(v12, v97, RZ);
        v98 = -v98;
        v99 = IMAD(v24, v98, v97);
        b22 = v3 > v99;
        if (!b22) v100 = v99 - v24;
        b23 = v3 > v100;
        shmem_u8[v2 + 1] = v96;
        shmem_u8[v94] = v84;
        v101 = shmem_u8[v2 + 2];
        if (!b23) v102 = v100 - v24;
        if (!b21) v102 = -v102;
        v103 = !P1 ? v34 : v102;
        v104 = v103 + ConstMem(0, 352);
        v105 = hi32(v103 + (ConstMem(0, 356) << 1)) + (b23 ? 1 : 0);
        v106 = IADD3(v62, v94, v101);
        v107 = *v104@64;
        v108 = v106 >> 31;
        v109 = hi32(v108 + (v106 << 8));
        v110 = v109 & 4294967040;
        v14 = v106 - v110;
        v111 = shmem_u8[v14];
        v112 = v2 + 6;
        v113 = IABS(v112);
        v114 = IMAD.HI.U32(v12, v113, RZ);
        v114 = -v114;
        v115 = IMAD(v24, v114, v113);
        b24 = v3 > v115;
        shmem_u8[v2 + 2] = v111;
        shmem_u8[v14] = v101;
        v13 = shmem_u8[v2 + 3];
        if (!b24) v116 = v115 - v24;
        b25 = v112 >= RZ;
        b26 = v3 > v116;
        if (!b26) v117 = v116 - v24;
        if (!b25) v117 = -v117;
        v118 = !P1 ? v34 : v117;
        v119 = v118 + ConstMem(0, 352);
        v120 = IADD3(v75, v14, v13);
        v121 = v120 >> 31;
        v122 = hi32(v118 + (ConstMem(0, 356) << 1)) + (b26 ? 1 : 0);
        v123 = hi32(v121 + (v120 << 8));
        v124 = *v119@64;
        v125 = v123 & 4294967040;
        v15 = v120 - v125;
        v126 = shmem_u8[v15];
        v127 = v2 + 7;
        v128 = IABS(v127);
        v129 = IMAD.HI.U32(v12, v128, RZ);
        v129 = -v129;
        v130 = IMAD(v24, v129, v128);
        b27 = v3 > v130;
        shmem_u8[v2 + 3] = v126;
        shmem_u8[v15] = v13;
        v131 = shmem_u8[v2 + 4];
        if (!b27) v132 = v130 - v24;
        b3 = v127 >= RZ;
        b4 = v3 > v132;
        if (!b4) v17 = v132 - v24;
        if (!b3) v17 = -v17;
        v133 = !P1 ? v34 : v17;
        v134 = v133 + ConstMem(0, 352);
        v135 = IADD3(v92, v15, v131);
        v136 = v135 >> 31;
        v137 = hi32(v133 + (ConstMem(0, 356) << 1)) + (b8 ? 1 : 0);
        v138 = hi32(v136 + (v135 << 8));
        v139 = *v134@64;
        v140 = v138 & 4294967040;
        v21 = v135 - v140;
        v141 = shmem_u8[v21];
        shmem_u8[v2 + 4] = v141;
        shmem_u8[v21] = v131;
        v18 = shmem_u8[v2 + 5];
        v142 = IADD3(v107, v21, v18);
        v143 = v142 >> 31;
        v144 = hi32(v143 + (v142 << 8));
        v145 = v144 & 4294967040;
        v6 = v142 - v145;
        v146 = shmem_u8[v6];
        shmem_u8[v2 + 5] = v146;
        shmem_u8[v6] = v18;
        v20 = shmem_u8[v2 + 6];
        v147 = IADD3(v124, v6, v20);
        v148 = v147 >> 31;
        v149 = hi32(v148 + (v147 << 8));
        v150 = v149 & 4294967040;
        v9 = v147 - v150;
        v11 = shmem_u8[v9];
        shmem_u8[v2 + 6] = v11;
        shmem_u8[v9] = v20;
        v16 = shmem_u8[v2 + 7];
        v19 = IADD3(v139, v9, v16);
        v151 = v19 >> 31;
        v152 = hi32(v151 + (v19 << 8));
        v153 = v152 & 4294967040;
        v4 = v19 - v153;
        v8 = shmem_u8[v4];
        shmem_u8[v2 + 7] = v8;
        shmem_u8[v4] = v16;
        v2 = v2 + 8;
        // 20 phi node(s) omitted
      }
    }
  }
  // Condition from BB6
  if (!(v2 != RZ)) {
    // Condition from BB7
    if (v154 >= 1) {
      // Condition from BB8
      if (v155 >= 3) {
        BB9 {
          v14 = IMAD(v1, ConstMem(0, 384), RZ);
          v21 = v156 - ConstMem(0, 384);
          v158 = RZ;
          u2 = ULDC.64(ConstMem(0, 368));
          v6 = v14 + 3;
          u3 = ULDC.64(ConstMem(0, 376));
          v13 = v14 >> 31;
        }
        // Loop header BB10
        while (v21 != RZ) {
          BB10 {
            v162 = v6 - 3;
            b31 = v162 >= v157;
            v163 = v14 + u2;
            v164 = v13 + u4 + (b29 ? 1 : 0);
            if (!b31) v165 = *v163@64;
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
            b32 = v172 >= v157;
            v9 = v14 + u3;
            shmem_u8[u11] = v173;
            shmem_u8[v171] = v166;
            v174 = shmem_u8[u11];
            v175 = v166 + v174;
            if (!b31) v176 = v175 & 255;
            v8 = v13 + u5 + (b3 ? 1 : 0);
            if (!b31) v177 = shmem_u8[v176];
            u12 = u11 + 1;
            u13 = u12 >> 31;
            u14 = hi32(u13 + (u12 << 8));
            u15 = u14 & 4294967040;
            if (!b31) v178 = v165 ^ v177;
            if (!b31) *v9@64 = v178;
            if (!b32) v179 = *(v163 + 1)@64;
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
            if (!b32) v189 = v187 & 255;
            b4 = v188 >= v157;
            if (!b32) v190 = shmem_u8[v189];
            u17 = u16 + 1;
            u18 = u17 >> 31;
            u19 = hi32(u18 + (u17 << 8));
            u20 = u19 & 4294967040;
            u21 = u17 - u20;
            v191 = shmem_u8[u21];
            if (!b32) v192 = v179 ^ v190;
            if (!b32) *(v9 + 1)@64 = v192;
            if (!b4) v193 = *(v163 + 2)@64;
            v194 = v160 + v191;
            b33 = v6 >= v157;
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
            if (!b4) *(v9 + 2)@64 = v16;
            if (!b33) v15 = *(v163 + 3)@64;
            v203 = v161 + v19;
            v21 = v21 + 4;
            u3 = u3 + 4;
            v158 = v158 + 4;
            u2 = u2 + 4;
            v204 = v203 >> 31;
            u5 = URZ + u5 + (b34 ? 1 : 0);
            v6 = v6 + 4;
            u4 = URZ + u4 + (b35 ? 1 : 0);
            v205 = hi32(v204 + (v203 << 8));
            v17 = v205 & 4294967040;
            v159 = v203 - v17;
            v155 = shmem_u8[v159];
            shmem_u8[u1] = v155;
            shmem_u8[v159] = v19;
            v206 = shmem_u8[u1];
            v207 = v19 + v206;
            if (!b33) v208 = v207 & 255;
            if (!b33) v3 = shmem_u8[v208];
            if (!b33) v18 = v15 ^ v3;
            if (!b33) *(v9 + 3)@64 = v18;
            // 23 phi node(s) omitted
          }
        }
      }
      // Condition from BB11
      if (v156 != RZ) {
        BB12 {
          v209 = IMAD(v1, ConstMem(0, 384), v158);
          v210 = v209 + ConstMem(0, 376);
          v211 = v209 + ConstMem(0, 368);
          v212 = v209 >> 31;
          v213 = v212 + ConstMem(0, 380) + (b30 ? 1 : 0);
          v214 = v212 + ConstMem(0, 372) + (b29 ? 1 : 0);
        }
        // Loop header BB13
        while (v156 != RZ) {
          BB13 {
            b36 = v209 >= v157;
            if (!b36) v215 = v211;
            if (!b36) v216 = v214;
            if (!b36) v217 = *v215@64;
            u25 = u1 + 1;
            v156 = v156 - 1;
            v211 = v211 + 1;
            u26 = u25 >> 31;
            v209 = v209 + 1;
            u27 = hi32(u26 + (u25 << 8));
            v214 = IMAD.X(RZ, RZ, v214, b4);
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
            if (!b36) v212 = v210;
            v210 = v210 + 1;
            v224 = v9 + v223;
            if (!b36) v225 = v224 & 255;
            if (!b36) v6 = shmem_u8[v225];
            if (!b36) v3 = v213;
            v213 = IMAD.X(RZ, RZ, v213, b29);
            if (!b36) v8 = v217 ^ v6;
            if (!b36) *v212@64 = v8;
            // 15 phi node(s) omitted
          }
        }
      }
    }
  }
}

