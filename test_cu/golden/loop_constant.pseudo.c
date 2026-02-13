BB0 {
  R1.0 = IMAD.MOV.U32(RZ, RZ, ConstMem(0, 40));
  R0.0 = S2R(SR_CTAID.X());
  P0.0 = ISETP.GE.AND(PT, R0.0, ConstMem(0, 388), PT);
  if (P0.0) _ = EXIT();
}
// Condition from BB1
if (!(P1)) {
  // Loop header BB2
  while (!(P1)) {
    BB2 {
      R2.1 = phi(R2.2, R2.0);
      P1.1 = phi(P1.2, P1.0);
      _ = STS.U8(*R2.1, R2.1);
      R2.2 = IADD3(R2.1, ConstMem(0, 0), RZ);
      P1.2 = ISETP.GE.AND(PT, R2.2, 256, PT);
      if (P1.2) _ = BRA();
    }
    continue;
  }
}
BB3 {
  R2.3 = phi(R2.2, R2.0);
  P1.3 = phi(P1.2, P1.0);
  _ = BSYNC();
}

