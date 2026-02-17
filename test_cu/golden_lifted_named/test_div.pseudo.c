BB0 {
  v0 = ConstMem(0, 40);
  v1 = abs(ConstMem(0, 356));
  u0 = ConstMem(0, 352);
  u1 = ConstMem(0, 356);
  u2 = u0 ^ u1;
  v2 = i2f_rp(v1);
  b0 = RZ <= u2;
  u3 = ConstMem(0, 280);
  u4 = ConstMem(0, 284);
  v3 = rcp_approx(v2);
  v4 = v3 + 268435454;
  v5 = f2i_trunc_u32_ftz_ntz(v4);
  v6 = RZ;
  v7 = -v5;
  v8 = v7 * v1 + RZ;
  v9 = abs(ConstMem(0, 352));
  v10 = mul_hi_u32(v5, v8) + v6;
  v11 = ConstMem(0, 360);
  v12 = mul_hi_u32(v10, v9);
  v13 = -v12 + RZ + RZ;
  v14 = v1 * v13 + v9;
  b1 = v1 > v14;
  if (!b1) v15 = v14 - v1;
  if (!b1) v16 = v12 + 1;
  b2 = RZ != ConstMem(0, 356);
  b3 = v15 >= v1;
  if (b3) v17 = v16 + 1;
  v18 = v17;
  v19 = MOV(ConstMem(0, 364));
  if (!b0) v20 = -v18 + RZ + RZ;
  if (!b2) v21 = ~ConstMem(0, 356);
  *addr64(v11, v19) = v21;
}
return;

