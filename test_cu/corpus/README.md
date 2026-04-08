# CUDA test corpus

Multi-kernel CUDA fixtures used by the invariant-based corpus runner in
`src/test.rs` (`corpus_*` tests). Each `*.cu` holds a themed group of kernels
and is compiled to a single multi-function SASS dump via `cuobjdump`. The
runner uses `parser::split_functions` to slice the dump into per-function
instruction lists and runs each through the full lifted+named pipeline.

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

All kernels use `extern "C" __global__` so the SASS `Function :` markers
emit the raw C name (no C++ mangling).

## Regenerating the SASS dumps

On a box with the CUDA toolkit installed (e.g. `ssh darknavy@172.16.19.115`):

```sh
for f in arith_kernels branching_kernels loop_kernels shared_mem_kernels crypto_kernels compute_kernels control_flow_kernels; do
    nvcc -cubin -arch=sm_89 -O2 "$f.cu" -o "$f.cubin"
    cuobjdump --dump-sass "$f.cubin" > "$f.sass"
done
```

Or via Docker on macOS (no GPU required):

```sh
docker run --rm --platform linux/amd64 \
  -v $(pwd):/corpus nvidia/cuda:12.6.2-devel-ubuntu22.04 \
  bash -c 'cd /corpus && for f in arith_kernels branching_kernels loop_kernels shared_mem_kernels crypto_kernels compute_kernels control_flow_kernels; do
    nvcc -cubin -arch=sm_89 -O2 "$f.cu" -o "/tmp/$f.cubin" && cuobjdump --dump-sass "/tmp/$f.cubin" > "$f.sass"
  done'
```

Then copy the `.sass` files back into this directory. The corpus runner
walks them via `include_str!`, so rebuilding the test binary is enough.

## Invariants asserted by the runner

- Every file splits into at least one function, and the corpus has at
  least 20 functions overall.
- Every function produces non-empty pseudo-C output.
- Output is deterministic across two runs of the same pipeline.
- No raw convergence barrier opcodes (`BSSY`, `BSYNC`, `SSY`, `SYNC`,
  `WARPSYNC`) leak into the rendered C.
- No SSA-suffix tokens (e.g. `R3.0`, `P1.2`) remain after name recovery.
- Goto budget per function defaults to zero. Functions with genuinely
  complex multi-exit/early-return patterns have an explicit per-function
  budget in the allow-list (see `corpus_goto_budget_is_tight` test).
  Current allow-list: `multi_exit_loop` (26), `find_pattern` (8),
  `nested_loop_break_continue` (4). Tighten as the structurizer improves.
