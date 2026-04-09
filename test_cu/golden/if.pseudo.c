  R1.0 = IMAD.MOV.U32(0, 0, ConstMem(0, 40));
  R26.0 = S2R(SR_TID.X());
if (P0.0) {
  R1.1 = IADD3(R1.0, 1, 0);
} else {
  R2.1 = IMAD.WIDE(R27.0, R2.0, ConstMem(0, 360));
}
return;
