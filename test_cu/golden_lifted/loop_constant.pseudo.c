BB0 {
  R1.0 = ConstMem(0, 40);
  R0.0 = blockIdx.x;
  if (P0.0) return;
}
BB1 {
  R2.0 = threadIdx.x;
}
// Condition from BB1
if (!(R2.0 > 255)) {
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

