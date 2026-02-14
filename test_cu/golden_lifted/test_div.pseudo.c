BB0 {
  R1.0 = ConstMem(0, 40);
  R5.0 = IABS(ConstMem(0, 356));
  UR4.0 = ULDC.64(ConstMem(0, 352));
  UR4.1 = UR4.0 ^ UR5.0;
  R0.0 = I2F.RP(R5.0);
  P1.0 = RZ <= UR4.1;
  UR4.2 = ULDC.64(ConstMem(0, 280));
  R0.1 = MUFU.RCP(R0.0);
  R2.0 = R0.1 + 268435454;
  R3.0 = F2I.FTZ.U32.TRUNC.NTZ(R2.0);
  R2.1 = RZ;
  R4.0 = -R3.0;
  R7.0 = IMAD(R4.0, R5.0, RZ);
  R4.1 = IABS(ConstMem(0, 352));
  R3.1 = IMAD.HI.U32(R3.0, R7.0, R2.1);
  R2.2 = ConstMem(0, 360);
  R3.2 = IMAD.HI.U32(R3.1, R4.1, RZ);
  R0.2 = IADD3(-R3.2, RZ, RZ);
  R0.3 = IMAD(R5.0, R0.2, R4.1);
  P2.0 = R5.0 > R0.3;
  if (!P2.0) R0.4 = R0.3 - R5.0;
  if (!P2.0) R3.3 = R3.2 + 1;
  P2.1 = RZ != ConstMem(0, 356);
  P0.0 = R0.4 >= R5.0;
  if (P0.0) R3.4 = R3.3 + 1;
  R5.1 = R3.4;
  R3.5 = MOV(ConstMem(0, 364));
  if (!(P1.0)) R5.2 = IADD3(-R5.1, RZ, RZ);
  if (!P2.1) R5.3 = ~ConstMem(0, 356);
  *R2.2@64 = R5.3;
}
return;

