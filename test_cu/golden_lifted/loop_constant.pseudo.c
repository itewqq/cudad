R1.0 = c[0x0][0x28];
R0.0 = blockIdx.x;
if (P0.0) return;
  R2.0 = threadIdx.x;
  UR6.0 = c[0x0][0x118];
  UR7.0 = c[0x0][0x11c];
  P1.0 = (int32_t)(R2.0) > (int32_t)(255);
  P0.1 = R2.0 != 0;
if (!((int32_t)(R2.0) > (int32_t)(255))) {
  do {
    shmem_u8[R2.1] = R2.1;
    R2.2 = R2.1 + c[0x0][0x0];
    // 2 phi node(s) omitted [BB2]
  } while(!((int32_t)(R2.2) >= (int32_t)(256)));
}
// 2 phi node(s) omitted [BB3]
