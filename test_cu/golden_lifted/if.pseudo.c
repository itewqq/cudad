// Condition from BB0
if (P0) {
  BB1 {
    R1.1 = R1.0 + 1;
  }
} else {
  BB2 {
    R2.1 = R27.0 * R2.0 + ConstMem(0, 360);
  }
}
return;

