BB0 {
  R0.0 = S2R(SR_CTAID.X());
  R3.0 = S2R(SR_TID.X());
}
// Condition from BB0
if (!(P0.0)) {
  // Condition from BB1
  if (P1.0) {
    // Condition from BB2
    if (P0.2) {
      // Condition from BB3
      if (P0.3) {
        // Condition from BB4
        if (P1.1) {
          BB5 {
            P0.5 = PLOP3.LUT(PT, PT, PT, PT, 8, 0);
          }
          // Loop header BB6
          while (P2.2) {
            BB6 {
              R7.2 = IMAD.MOV.U32(RZ, RZ, 1066192077);
              P1.3 = FSETP.GT.AND(PT, R9.2, 0.5, PT);
              R6.4 = IADD3(R6.3, -16, RZ);
              R4.3 = FSEL(R7.2, 0.8999999761581421, P1.3);
              P2.2 = ISETP.GT.AND(PT, R6.4, 12, PT);
              R4.4 = FMUL(R4.3, R9.2);
              P1.4 = FSETP.GT.AND(PT, R4.4, 0.5, PT);
              R5.3 = FSEL(R7.2, 0.8999999761581421, P1.4);
              R5.4 = FMUL(R4.4, R5.3);
              P1.5 = FSETP.GT.AND(PT, R5.4, 0.5, PT);
              R4.5 = FSEL(R7.2, 0.8999999761581421, P1.5);
              R4.6 = FMUL(R5.4, R4.5);
              P1.6 = FSETP.GT.AND(PT, R4.6, 0.5, PT);
              R5.5 = FSEL(R7.2, 0.8999999761581421, P1.6);
              R5.6 = FMUL(R4.6, R5.5);
              P1.7 = FSETP.GT.AND(PT, R5.6, 0.5, PT);
              R4.7 = FSEL(R7.2, 0.8999999761581421, P1.7);
              R4.8 = FMUL(R5.6, R4.7);
              P1.8 = FSETP.GT.AND(PT, R4.8, 0.5, PT);
              R5.7 = FSEL(R7.2, 0.8999999761581421, P1.8);
              R5.8 = FMUL(R4.8, R5.7);
              P1.9 = FSETP.GT.AND(PT, R5.8, 0.5, PT);
              R4.9 = FSEL(R7.2, 0.8999999761581421, P1.9);
              R4.10 = FMUL(R5.8, R4.9);
              P1.10 = FSETP.GT.AND(PT, R4.10, 0.5, PT);
              R5.9 = FSEL(R7.2, 0.8999999761581421, P1.10);
              R5.10 = FMUL(R4.10, R5.9);
              P1.11 = FSETP.GT.AND(PT, R5.10, 0.5, PT);
              R4.11 = FSEL(R7.2, 0.8999999761581421, P1.11);
              R4.12 = FMUL(R5.10, R4.11);
              P1.12 = FSETP.GT.AND(PT, R4.12, 0.5, PT);
              R5.11 = FSEL(R7.2, 0.8999999761581421, P1.12);
              R5.12 = FMUL(R4.12, R5.11);
              P1.13 = FSETP.GT.AND(PT, R5.12, 0.5, PT);
              R4.13 = FSEL(R7.2, 0.8999999761581421, P1.13);
              R4.14 = FMUL(R5.12, R4.13);
              P1.14 = FSETP.GT.AND(PT, R4.14, 0.5, PT);
              R5.13 = FSEL(R7.2, 0.8999999761581421, P1.14);
              R5.14 = FMUL(R4.14, R5.13);
              P1.15 = FSETP.GT.AND(PT, R5.14, 0.5, PT);
              R4.15 = FSEL(R7.2, 0.8999999761581421, P1.15);
              R4.16 = FMUL(R5.14, R4.15);
              P1.16 = FSETP.GT.AND(PT, R4.16, 0.5, PT);
              R5.15 = FSEL(R7.2, 0.8999999761581421, P1.16);
              R5.16 = FMUL(R4.16, R5.15);
              P1.17 = FSETP.GT.AND(PT, R5.16, 0.5, PT);
              R4.17 = FSEL(R7.2, 0.8999999761581421, P1.17);
              R4.18 = FMUL(R5.16, R4.17);
              P1.18 = FSETP.GT.AND(PT, R4.18, 0.5, PT);
              R9.3 = FSEL(R7.2, 0.8999999761581421, P1.18);
              R9.4 = FMUL(R4.18, R9.3);
              // 7 phi node(s) omitted
            }
          }
        }
        // Condition from BB7
        if (P1.20) {
          BB8 {
            R7.4 = IMAD.MOV.U32(RZ, RZ, 1066192077);
            P0.7 = FSETP.GT.AND(PT, R9.5, 0.5, PT);
            R6.6 = IADD3(R6.5, -8, RZ);
            R4.20 = FSEL(R7.4, 0.8999999761581421, P0.7);
            R4.21 = FMUL(R9.5, R4.20);
            P0.8 = FSETP.GT.AND(PT, R4.21, 0.5, PT);
            R5.18 = FSEL(R7.4, 0.8999999761581421, P0.8);
            R5.19 = FMUL(R4.21, R5.18);
            P0.9 = FSETP.GT.AND(PT, R5.19, 0.5, PT);
            R4.22 = FSEL(R7.4, 0.8999999761581421, P0.9);
            R4.23 = FMUL(R5.19, R4.22);
            P0.10 = FSETP.GT.AND(PT, R4.23, 0.5, PT);
            R5.20 = FSEL(R7.4, 0.8999999761581421, P0.10);
            R5.21 = FMUL(R4.23, R5.20);
            P0.11 = FSETP.GT.AND(PT, R5.21, 0.5, PT);
            R4.24 = FSEL(R7.4, 0.8999999761581421, P0.11);
            R4.25 = FMUL(R5.21, R4.24);
            P0.12 = FSETP.GT.AND(PT, R4.25, 0.5, PT);
            R5.22 = FSEL(R7.4, 0.8999999761581421, P0.12);
            R5.23 = FMUL(R4.25, R5.22);
            P0.13 = FSETP.GT.AND(PT, R5.23, 0.5, PT);
            R4.26 = FSEL(R7.4, 0.8999999761581421, P0.13);
            P0.14 = PLOP3.LUT(PT, PT, PT, PT, 8, 0);
            R4.27 = FMUL(R5.23, R4.26);
            P1.21 = FSETP.GT.AND(PT, R4.27, 0.5, PT);
            R9.6 = FSEL(R7.4, 0.8999999761581421, P1.21);
            R9.7 = FMUL(R4.27, R9.6);
          }
        }
        // Condition from BB9
        if (P0.16) {
          BB10 {
            R7.7 = IMAD.MOV.U32(RZ, RZ, 1066192077);
            // 8 phi node(s) omitted
          }
          // Loop header BB11
          while (P0.22) {
            BB11 {
              P0.19 = FSETP.GT.AND(PT, R9.10, 0.5, PT);
              R6.10 = IADD3(R6.9, -4, RZ);
              R4.31 = FSEL(R7.7, 0.8999999761581421, P0.19);
              R4.32 = FMUL(R4.31, R9.10);
              P0.20 = FSETP.GT.AND(PT, R4.32, 0.5, PT);
              R5.27 = FSEL(R7.7, 0.8999999761581421, P0.20);
              R5.28 = FMUL(R4.32, R5.27);
              P0.21 = FSETP.GT.AND(PT, R5.28, 0.5, PT);
              R4.33 = FSEL(R7.7, 0.8999999761581421, P0.21);
              P0.22 = ISETP.NE.AND(PT, R6.10, RZ, PT);
              R4.34 = FMUL(R5.28, R4.33);
              P1.25 = FSETP.GT.AND(PT, R4.34, 0.5, PT);
              R9.11 = FSEL(R7.7, 0.8999999761581421, P1.25);
              R9.12 = FMUL(R4.34, R9.11);
              // 6 phi node(s) omitted
            }
          }
        }
        // Condition from BB12
        if (P0.24) {
          BB13 {
            R5.30 = IMAD.MOV.U32(RZ, RZ, 1066192077);
          }
          // Loop header BB14
          while (P0.26) {
            BB14 {
              R2.3 = IADD3(R2.2, -1, RZ);
              P1.28 = FSETP.GT.AND(PT, R9.14, 0.5, PT);
              P0.26 = ISETP.NE.AND(PT, R2.3, RZ, PT);
              R4.37 = FSEL(R5.30, 0.8999999761581421, P1.28);
              R9.15 = FMUL(R4.37, R9.14);
              // 5 phi node(s) omitted
            }
          }
        }
        BB15 {
          R2.5 = LEA(P0.27, R0.1, ConstMem(0, 368), 2);
          R3.2 = LEA.HI.X(R0.1, ConstMem(0, 372), R3.1, 2, P0.27);
          _ = STG.E(*addr64(R2.5, R3.2), R9.16);
          return;
          // 9 phi node(s) omitted
        }
      }
    }
  }
}

