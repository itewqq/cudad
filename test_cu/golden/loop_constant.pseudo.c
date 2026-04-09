R1.0 = IMAD.MOV.U32(RZ, RZ, ConstMem(0, 40));
R0.0 = S2R(SR_CTAID.X());
if (P0.0) return;
  R2.0 = S2R(SR_TID.X());
  (UR6.0, UR7.0) = ULDC.64(ConstMem(0, 280));
  P1.0 = ISETP.GT.AND(PT, R2.0, 255, PT);
  P0.1 = ISETP.NE.AND(PT, R2.0, RZ, PT);
if (!(P1.0)) {
  do {
    STS.U8(*R2.1, R2.1);
    R2.2 = IADD3(R2.1, ConstMem(0, 0), RZ);
    // 2 phi node(s) omitted [BB2]
  } while(!(P1.2));
}
// 2 phi node(s) omitted [BB3]
