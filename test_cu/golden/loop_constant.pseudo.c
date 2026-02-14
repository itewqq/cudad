BB0 {
  R1.0 = IMAD.MOV.U32(RZ, RZ, ConstMem(0, 40));
  R0.0 = S2R(SR_CTAID.X());
  if (P0.0) return;
}
BB1 {
  R2.0 = S2R(SR_TID.X());
}
// Condition from BB1
if (!(P1.0)) {
  // Loop header BB2
  while (!(P1.2)) {
    BB2 {
      _ = STS.U8(*R2.1, R2.1);
      R2.2 = IADD3(R2.1, ConstMem(0, 0), RZ);
      // 2 phi node(s) omitted
    }
  }
}
BB3 {
  _ = BSYNC();
  // 2 phi node(s) omitted
}

