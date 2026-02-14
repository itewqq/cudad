BB0 {
  R26.0 = S2R(SR_TID.X());
}
// Condition from BB0
if (P0.0) {
  BB1 {
    R1.1 = IADD3(R1.0, 1, RZ);
  }
} else {
  BB2 {
    R2.1 = IMAD.WIDE(R27.0, R2.0, ConstMem(0, 360));
  }
}
return;

