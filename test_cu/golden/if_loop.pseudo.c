// Condition from BB0
if (!(P0)) {
  // Condition from BB1
  if (P1) {
    // Condition from BB2
    if (P0) {
      // Condition from BB3
      if (P0) {
        // Condition from BB4
        if (P1) {
          BB5 {
            P0.5 = PLOP3.LUT(PT, PT, PT, PT, 8, 0);
          }
          // Loop header BB6
          while (P2) {
            BB6 {
              R9.2 = phi(R9.4, R9.1);
              R7.1 = phi(R7.2, R7.0);
              R6.3 = phi(R6.4, R6.2);
              R5.1 = phi(R5.15, R5.0);
              R4.2 = phi(R4.18, R4.1);
              P2.1 = phi(P2.2, P2.0);
              P1.2 = phi(P1.18, P1.1);
              R7.2 = IMAD.MOV.U32(RZ, RZ, 1066192077);
              P1.3 = FSETP.GT.AND(PT, R9.2, 0.5, PT);
              R6.4 = IADD3(R6.3, -16, RZ);
              R4.3 = FSEL(R7.2, 0.8999999761581421, P1.3);
              P2.2 = ISETP.GT.AND(PT, R6.4, 12, PT);
              R4.4 = FMUL(R4.3, R9.2);
              P1.4 = FSETP.GT.AND(PT, R4.4, 0.5, PT);
              R5.2 = FSEL(R7.2, 0.8999999761581421, P1.4);
              R5.3 = FMUL(R4.4, R5.2);
              P1.5 = FSETP.GT.AND(PT, R5.3, 0.5, PT);
              R4.5 = FSEL(R7.2, 0.8999999761581421, P1.5);
              R4.6 = FMUL(R5.3, R4.5);
              P1.6 = FSETP.GT.AND(PT, R4.6, 0.5, PT);
              R5.4 = FSEL(R7.2, 0.8999999761581421, P1.6);
              R5.5 = FMUL(R4.6, R5.4);
              P1.7 = FSETP.GT.AND(PT, R5.5, 0.5, PT);
              R4.7 = FSEL(R7.2, 0.8999999761581421, P1.7);
              R4.8 = FMUL(R5.5, R4.7);
              P1.8 = FSETP.GT.AND(PT, R4.8, 0.5, PT);
              R5.6 = FSEL(R7.2, 0.8999999761581421, P1.8);
              R5.7 = FMUL(R4.8, R5.6);
              P1.9 = FSETP.GT.AND(PT, R5.7, 0.5, PT);
              R4.9 = FSEL(R7.2, 0.8999999761581421, P1.9);
              R4.10 = FMUL(R5.7, R4.9);
              P1.10 = FSETP.GT.AND(PT, R4.10, 0.5, PT);
              R5.8 = FSEL(R7.2, 0.8999999761581421, P1.10);
              R5.9 = FMUL(R4.10, R5.8);
              P1.11 = FSETP.GT.AND(PT, R5.9, 0.5, PT);
              R4.11 = FSEL(R7.2, 0.8999999761581421, P1.11);
              R4.12 = FMUL(R5.9, R4.11);
              P1.12 = FSETP.GT.AND(PT, R4.12, 0.5, PT);
              R5.10 = FSEL(R7.2, 0.8999999761581421, P1.12);
              R5.11 = FMUL(R4.12, R5.10);
              P1.13 = FSETP.GT.AND(PT, R5.11, 0.5, PT);
              R4.13 = FSEL(R7.2, 0.8999999761581421, P1.13);
              R4.14 = FMUL(R5.11, R4.13);
              P1.14 = FSETP.GT.AND(PT, R4.14, 0.5, PT);
              R5.12 = FSEL(R7.2, 0.8999999761581421, P1.14);
              R5.13 = FMUL(R4.14, R5.12);
              P1.15 = FSETP.GT.AND(PT, R5.13, 0.5, PT);
              R4.15 = FSEL(R7.2, 0.8999999761581421, P1.15);
              R4.16 = FMUL(R5.13, R4.15);
              P1.16 = FSETP.GT.AND(PT, R4.16, 0.5, PT);
              R5.14 = FSEL(R7.2, 0.8999999761581421, P1.16);
              R5.15 = FMUL(R4.16, R5.14);
              P1.17 = FSETP.GT.AND(PT, R5.15, 0.5, PT);
              R4.17 = FSEL(R7.2, 0.8999999761581421, P1.17);
              R4.18 = FMUL(R5.15, R4.17);
              P1.18 = FSETP.GT.AND(PT, R4.18, 0.5, PT);
              R9.3 = FSEL(R7.2, 0.8999999761581421, P1.18);
              R9.4 = FMUL(R4.18, R9.3);
              if (P2.2) _ = BRA();
            }
            continue;
          }
        } else {
          // Condition from BB7
          if (P1) {
            BB8 {
              R7.4 = IMAD.MOV.U32(RZ, RZ, 1066192077);
              P0.7 = FSETP.GT.AND(PT, R9.5, 0.5, PT);
              R6.6 = IADD3(R6.5, -8, RZ);
              R4.20 = FSEL(R7.4, 0.8999999761581421, P0.7);
              R4.21 = FMUL(R9.5, R4.20);
              P0.8 = FSETP.GT.AND(PT, R4.21, 0.5, PT);
              R5.17 = FSEL(R7.4, 0.8999999761581421, P0.8);
              R5.18 = FMUL(R4.21, R5.17);
              P0.9 = FSETP.GT.AND(PT, R5.18, 0.5, PT);
              R4.22 = FSEL(R7.4, 0.8999999761581421, P0.9);
              R4.23 = FMUL(R5.18, R4.22);
              P0.10 = FSETP.GT.AND(PT, R4.23, 0.5, PT);
              R5.19 = FSEL(R7.4, 0.8999999761581421, P0.10);
              R5.20 = FMUL(R4.23, R5.19);
              P0.11 = FSETP.GT.AND(PT, R5.20, 0.5, PT);
              R4.24 = FSEL(R7.4, 0.8999999761581421, P0.11);
              R4.25 = FMUL(R5.20, R4.24);
              P0.12 = FSETP.GT.AND(PT, R4.25, 0.5, PT);
              R5.21 = FSEL(R7.4, 0.8999999761581421, P0.12);
              R5.22 = FMUL(R4.25, R5.21);
              P0.13 = FSETP.GT.AND(PT, R5.22, 0.5, PT);
              R4.26 = FSEL(R7.4, 0.8999999761581421, P0.13);
              P0.14 = PLOP3.LUT(PT, PT, PT, PT, 8, 0);
              R4.27 = FMUL(R5.22, R4.26);
              P1.21 = FSETP.GT.AND(PT, R4.27, 0.5, PT);
              R9.6 = FSEL(R7.4, 0.8999999761581421, P1.21);
              R9.7 = FMUL(R4.27, R9.6);
            }
            // Condition from BB9
            if (P0) {
              BB10 {
                R9.9 = phi(R9.8, R9.1);
                R7.6 = phi(R7.5, R7.0);
                R6.8 = phi(R6.7, R6.2);
                R5.24 = phi(R5.23, R5.0);
                R4.29 = phi(R4.28, R4.1);
                P2.4 = phi(P2.3, P2.0);
                P1.23 = phi(P1.22, P1.0);
                P0.17 = phi(P0.16, P0.3);
                R7.7 = IMAD.MOV.U32(RZ, RZ, 1066192077);
              }
              // Loop header BB11
              while (P0) {
                BB11 {
                  R9.10 = phi(R9.12, R9.9);
                  R6.9 = phi(R6.10, R6.8);
                  R5.25 = phi(R5.27, R5.24);
                  R4.30 = phi(R4.34, R4.29);
                  P1.24 = phi(P1.25, P1.23);
                  P0.18 = phi(P0.22, P0.17);
                  P0.19 = FSETP.GT.AND(PT, R9.10, 0.5, PT);
                  R6.10 = IADD3(R6.9, -4, RZ);
                  R4.31 = FSEL(R7.7, 0.8999999761581421, P0.19);
                  R4.32 = FMUL(R4.31, R9.10);
                  P0.20 = FSETP.GT.AND(PT, R4.32, 0.5, PT);
                  R5.26 = FSEL(R7.7, 0.8999999761581421, P0.20);
                  R5.27 = FMUL(R4.32, R5.26);
                  P0.21 = FSETP.GT.AND(PT, R5.27, 0.5, PT);
                  R4.33 = FSEL(R7.7, 0.8999999761581421, P0.21);
                  P0.22 = ISETP.NE.AND(PT, R6.10, RZ, PT);
                  R4.34 = FMUL(R5.27, R4.33);
                  P1.25 = FSETP.GT.AND(PT, R4.34, 0.5, PT);
                  R9.11 = FSEL(R7.7, 0.8999999761581421, P1.25);
                  R9.12 = FMUL(R4.34, R9.11);
                  if (P0.22) _ = BRA();
                }
                continue;
              }
            } else {
              // Condition from BB12
              if (P0) {
                BB13 {
                  R5.29 = IMAD.MOV.U32(RZ, RZ, 1066192077);
                }
                // Loop header BB14
                while (P0) {
                  BB14 {
                    R9.14 = phi(R9.15, R9.13);
                    R4.36 = phi(R4.37, R4.35);
                    R2.2 = phi(R2.3, R2.1);
                    P1.27 = phi(P1.28, P1.26);
                    P0.25 = phi(P0.26, P0.24);
                    R2.3 = IADD3(R2.2, -1, RZ);
                    P1.28 = FSETP.GT.AND(PT, R9.14, 0.5, PT);
                    P0.26 = ISETP.NE.AND(PT, R2.3, RZ, PT);
                    R4.37 = FSEL(R5.29, 0.8999999761581421, P1.28);
                    R9.15 = FMUL(R4.37, R9.14);
                    if (P0.26) _ = BRA();
                  }
                  continue;
                }
              } else {
                return;
              }
            }
          }
        }
      }
    }
  }
}

