BB0 {
  R1.0 = ConstMem(0, 40);
  R0.0 = S2R(SR_CTAID.X());
  P0.0 = R0.0 >= ConstMem(0, 388);
  if (P0.0) return;
}
// Condition from BB1
if (!(P1)) {
  // Loop header BB2
  while (!(P1)) {
    BB2 {
      _ = STS.U8(*R2.1, R2.1);
      R2.2 = R2.1 + ConstMem(0, 0);
      P1.2 = R2.2 >= 256;
      // 2 phi node(s) omitted
    }
  }
}
BB3 {
  _ = BSYNC();
  // 2 phi node(s) omitted
}
