v0 = c[0x0][0x28];
v1 = blockIdx.x;
if (b0) return;
  v2 = threadIdx.x;
  u0 = c[0x0][0x118];
  u1 = c[0x0][0x11c];
  b1 = (int32_t)(v2) > (int32_t)(255);
  b2 = v2 != 0;
if (!((int32_t)(v2) > (int32_t)(255))) {
    shmem_u8[v2] = v2;
    v2 = v2 + c[0x0][0x0];
  do {
    shmem_u8[v2] = v2;
    v2 = v2 + c[0x0][0x0];
  } while(!((int32_t)(v2) >= (int32_t)(256)));
}
