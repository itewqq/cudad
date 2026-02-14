BB0 {
  v1 = threadIdx.x;
}
// Condition from BB0
if (b0) {
  BB1 {
    v0 = v0 + 1;
  }
} else {
  BB2 {
    v2 = v3 * v2 + ConstMem(0, 360);
  }
}
return;

