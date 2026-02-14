BB0 {
  v0 = ConstMem(0, 40);
  v1 = IABS(ConstMem(0, 356));
  u0 = ULDC.64(ConstMem(0, 352));
  u1 = u0 ^ u2;
  v2 = I2F.RP(v1);
  b0 = RZ <= u1;
  u3 = ULDC.64(ConstMem(0, 280));
  v3 = MUFU.RCP(v2);
  v4 = v3 + 268435454;
  v5 = F2I.FTZ.U32.TRUNC.NTZ(v4);
  v6 = RZ;
  v7 = -v5;
  v8 = IMAD(v7, v1, RZ);
  v9 = IABS(ConstMem(0, 352));
  v10 = IMAD.HI.U32(v5, v8, v6);
  v11 = ConstMem(0, 360);
  v12 = IMAD.HI.U32(v10, v9, RZ);
  v13 = IADD3(-v12, RZ, RZ);
  v14 = IMAD(v1, v13, v9);
  b1 = v1 > v14;
  if (!b1) v15 = v14 - v1;
  if (!b1) v16 = v12 + 1;
  b2 = RZ != ConstMem(0, 356);
  b3 = v15 >= v1;
  if (b3) v17 = v16 + 1;
  v18 = v17;
  v19 = MOV(ConstMem(0, 364));
  if (!(b0)) v20 = IADD3(-v18, RZ, RZ);
  if (!b2) v21 = ~ConstMem(0, 356);
  *v11@64 = v21;
}
return;

