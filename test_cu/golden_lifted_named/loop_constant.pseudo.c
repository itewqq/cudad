BB0 {
  v0 = ConstMem(0, 40);
  v1 = blockIdx.x;
  if (b0) return;
}
BB1 {
  v2 = threadIdx.x;
  u0 = ConstMem(0, 280);
  u1 = ConstMem(0, 284);
  b1 = v2 > 255;
  b2 = v2 != 0;
}
// Condition from BB1
if (!(v2 > 255)) {
  BB2 {
    shmem_u8[v2] = v2;
    v2 = v2 + ConstMem(0, 0);
  }
  // Loop header BB2
  while (!(v2 >= 256)) {
    BB2 {
      shmem_u8[v2] = v2;
      v2 = v2 + ConstMem(0, 0);
      // 2 phi node(s) omitted
    }
  }
}
BB3 {
  _ = BSYNC();
  // 2 phi node(s) omitted
}
