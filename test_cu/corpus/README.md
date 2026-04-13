# CUDA test corpus

This directory contains themed multi-kernel CUDA fixtures used by the invariant-based corpus runner in `src/test.rs`.
Each `*.cu` file is compiled to a multi-function SASS dump, then the test runner uses `split_decoded_functions()` to slice the dump into per-function instruction streams and run each function through the canonical full-pass backend.

The sibling directories `test_cu/corpus_sm100/` and `test_cu/corpus_sm120/` reuse the same idea for newer dump formats and ABI layouts.

## What these files are for

The corpus exists to catch regressions that curated single-kernel goldens can miss.
Instead of checking exact text for every function, the corpus tests assert structural invariants over many real kernels across multiple themes and architectures.

## Contents

| File | Kernels |
| --- | --- |
| `arith_kernels.cu` | `saxpy`, `vec_add`, `vec_mul`, `vec_fma`, `scale_add`, `clamp_kernel`, `relu` |
| `branching_kernels.cu` | `abs_clamp`, `select_max`, `classify_sign`, `nested_ifs`, `early_exit_chain`, `swap_if_gt` |
| `loop_kernels.cu` | `dot_thread`, `l2_norm_sq`, `cumsum_linear`, `find_max`, `count_above`, `power_series` |
| `shared_mem_kernels.cu` | `reduce_block`, `max_reduce_block`, `transpose_tile`, `stencil1d`, `histogram256` |
| `crypto_kernels.cu` | `aes128_encrypt_block`, `sha256_single_block` |
| `compute_kernels.cu` | `sgemm_tiled`, `bitonic_sort`, `prefix_sum_blelloch`, `stencil2d_5pt`, `warp_reduce_sum` |
| `control_flow_kernels.cu` | `decision_tree`, `multi_exit_loop`, `dispatch_ops`, `nested_loop_break_continue`, `find_pattern`, `state_machine` |
| `image_processing_kernels.cu` | `bilinear_resize`, `nms_kernel`, `box_blur_variable_radius`, `histogram_equalize`, `sobel_edge_detect` |
| `ml_kernels.cu` | `topk_per_row`, `batched_sgemv`, `gelu_forward`, `cross_entropy_loss`, `fused_relu_bias_residual`, `layer_norm_forward`, `softmax_forward` |
| `simulation_kernels.cu` | `lj_forces`, `pagerank_iter`, `bfs_expand`, `pic_charge_deposit`, `nbody_forces` |
| `data_processing_kernels.cu` | `utf8_count_chars`, `radix_histogram`, `csv_find_fields`, `rle_compress`, `string_search` |

All kernels use `extern "C" __global__` so the `Function : ...` markers in the dumped SASS remain stable and unmangled.

## Regenerating the SASS dumps

On a machine with the CUDA toolkit installed:

```sh
for f in \
  arith_kernels \
  branching_kernels \
  loop_kernels \
  shared_mem_kernels \
  crypto_kernels \
  compute_kernels \
  control_flow_kernels \
  image_processing_kernels \
  ml_kernels \
  simulation_kernels \
  data_processing_kernels
 do
  nvcc -cubin -arch=sm_89 -O2 "$f.cu" -o "$f.cubin"
  cuobjdump --dump-sass "$f.cubin" > "$f.sass"
 done
```

Or via Docker on macOS:

```sh
docker run --rm --platform linux/amd64 \
  -v $(pwd):/corpus nvidia/cuda:12.6.2-devel-ubuntu22.04 \
  bash -lc 'cd /corpus && for f in \
    arith_kernels \
    branching_kernels \
    loop_kernels \
    shared_mem_kernels \
    crypto_kernels \
    compute_kernels \
    control_flow_kernels \
    image_processing_kernels \
    ml_kernels \
    simulation_kernels \
    data_processing_kernels
  do
    nvcc -cubin -arch=sm_89 -O2 "$f.cu" -o "/tmp/$f.cubin" &&
    cuobjdump --dump-sass "/tmp/$f.cubin" > "$f.sass"
  done'
```

The SM 100 and SM 120 sibling corpora should be regenerated from the same source kernels with the appropriate target architecture / toolchain combination for those dumps.

## Invariants checked by the corpus tests

The exact allow-lists and ceilings live in `src/test.rs`, but the high-level guarantees are:

- every corpus file splits into at least one function
- every function produces non-empty pseudo-C output
- output is deterministic across repeated runs
- raw convergence barrier mnemonics do not leak into rendered output
- SSA suffix tokens such as `R3.0` or `P1.2` do not survive name recovery
- `goto BB...` counts stay within controlled budgets
- SM 100 fixtures keep resolving Blackwell-era builtin slots
- SM 120 fixtures do not leak inline scheduling annotations such as `&req=` or `?WAITn`

## Why this matters

The corpus tests are the closest thing in the repo to a "true backend" validation pass.
They exercise the real canonical decompiler path on many kernels without depending on brittle whole-file textual parity for every case.
