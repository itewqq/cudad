BB0 {
  R1.0 = ConstMem(0, 40);
  R0.0 = blockIdx.x;
  if (P0.0) return;
}
BB1 {
  R2.0 = threadIdx.x;
  UR6.0 = ConstMem(0, 280);
  UR7.0 = ConstMem(0, 284);
  P1.0 = R2.0 > 255;
  P0.1 = R2.0 != 0;
}
// Condition from BB1
if (!(R2.0 > 255)) {
  BB2 {
    shmem_u8[R2.1] = R2.1;
    R2.2 = R2.1 + ConstMem(0, 0);
  }
  // Loop header BB2
  while (!(R2.2 >= 256)) {
    BB2 {
      shmem_u8[R2.1] = R2.1;
      R2.2 = R2.1 + ConstMem(0, 0);
      // 2 phi node(s) omitted
    }
  }
}
BB3 {
  _ = BSYNC();
  // 2 phi node(s) omitted
}
