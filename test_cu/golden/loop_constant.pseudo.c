BB0 {
  R1.0 = IMAD.MOV.U32(RZ, RZ, ConstMem(0, 40));
  R0.0 = S2R(SR_CTAID.X());
  P0.0 = ISETP.GE.AND(PT, R0.0, ConstMem(0, 388), PT);
  if (P0.0) return;
}
// Condition from BB1
if (!(P1)) {
  // Loop header BB2
  while (!(P1)) {
    BB2 {
      _ = STS.U8(*R2.1, R2.1);
      R2.2 = IADD3(R2.1, ConstMem(0, 0), RZ);
      P1.2 = ISETP.GE.AND(PT, R2.2, 256, PT);
      // 2 phi node(s) omitted
    }
    continue;
  }
}
BB3 {
  _ = BSYNC();
  // 2 phi node(s) omitted
}

