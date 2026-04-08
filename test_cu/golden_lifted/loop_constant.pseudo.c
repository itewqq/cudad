BB0 {
  R1.0 = ConstMem(0, 40);
  R0.0 = blockIdx.x;
  if (P0.0) return;
}
BB1 {
  R2.0 = threadIdx.x;
  UR6.0 = ConstMem(0, 280);
  UR7.0 = ConstMem(0, 284);
  P1.0 = (int32_t)(R2.0) > (int32_t)(255);
  P0.1 = R2.0 != 0;
}
// Condition from BB1
if (!((int32_t)(R2.0) > (int32_t)(255))) {
  do {
    BB2 {
      shmem_u8[R2.1] = R2.1;
      R2.2 = R2.1 + ConstMem(0, 0);
      // 2 phi node(s) omitted
    }
  } while(!(P1.2));
}
BB3 {
  // 2 phi node(s) omitted
}
