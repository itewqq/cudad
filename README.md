# cudad

`cudad` is an experimental CUDA SASS decompiler.

It takes `.sass` text dumps from NVIDIA tooling (`cuobjdump` / `nvdisasm`) and turns them into conservative, typed pseudo-C that is easier to inspect than raw SASS.

This is a reverse-engineering and compiler-internals project, not a production decompiler. The code is actively refactored, the output is intentionally conservative, and some kernels still degrade to temp-heavy code or explicit `goto BB...` fallbacks. That said, the current pipeline is real, end-to-end, and usable for inspection work.

## What it does today

- decodes real SASS instruction lines into typed instruction/operand records
- builds CFG + SSA and runs optimization passes before lifting
- structurizes recoverable control flow into `if` / loop-shaped pseudocode
- infers CUDA builtins and ABI-backed kernel arguments from constant-memory usage
- emits typed pseudo-C with shared-memory declarations and stable recovered names
- regression-tests the full pipeline on curated fixtures plus multi-kernel corpora for SM 89 / SM 100 / SM 120

## What the output looks like

For `test_cu/if_loop.sass`, the canonical output already looks like real structured code rather than a flat opcode dump. This is an abridged excerpt from the current full-pass snapshot:

```c
__global__ void kernel(uint32_t arg0, uint32_t arg2, uintptr_t arg4_ptr, uint32_t arg6, uint32_t arg7) {
  uint32_t ctaid_x;
  uint32_t tid_x;
  ...

  ctaid_x = blockIdx.x;
  tid_x = threadIdx.x;
  v2 = ctaid_x * blockDim.x + tid_x;
  if ((int32_t)(v2) >= (int32_t)(arg6)) return;
  v3 = v2 * 4 + arg0;
  v4 = v2 * 4 + arg2;
  v5 = *addr64(v3, v6);
  v7 = *addr64(v4, 4);
  v8 = arg7;
  v10 = v7 + v5;
  b2 = v10 > 1;
  v11 = !b2 ? (v7 * v5) : v10;

  if ((int32_t)(v8) >= (int32_t)(1)) {
    v3 = v8 - 1;
    v8 = v8 & 3;
    if (v3 >= 3) {
      v7 = -v8 + arg7;
      if ((int32_t)(v7) > (int32_t)(0)) b4 = (int32_t)(v7) > (int32_t)(12);
    }
  }

  do {
    b7 = v11 > 0.5;
    v7 = v7 - 16;
    ...
    v11 = v3 * v41;
  } while((int32_t)(v7) > (int32_t)(12));

  do {
    v8 = v8 - 1;
    b1 = v11 > 0.5;
    b2 = v8 != 0;
    v3 = b1 ? 1066192077 : 0.8999999761581421;
    v11 = v3 * v11;
  } while(v8 != 0);

  v61 = v2 + (arg4_ptr.lo32 << 2);
  v62 = lea_hi_x(v2, arg4_ptr.hi32, 2, b2);
  *addr64(v61, v62) = v11;
  return;
}
```

That is representative of an older snapshot of the backend:
- signatures are inferred heuristically from ABI usage and CUDA builtins such as `blockIdx.x`, `threadIdx.x`, and `blockDim.x` are recovered
- control flow is no longer raw `BRA` / `EXIT`; it becomes structured `if` / `do ... while` code when the CFG is recoverable
- the canonical backend now recovers typed memory accesses, builtins, and arithmetic/comparison structure through `FunctionAnalysis` plus AST lowering instead of post-render text repair
- locals are typed, but names are still generic (`vN`, `bN`, `uN`) unless stronger recovery exists
- pointer reconstruction is improved but still incomplete, so `addr64(lo, hi)` style artifacts can remain in real kernels

### Real compute kernels covered

The corpus also exercises real compute kernels beyond the small single-file fixtures. Good examples in `test_cu/corpus/compute_kernels.cu` include:

- `bitonic_sort` — nested compare/swap passes with `__syncthreads()` and pairwise exchange logic recovered from real sorting code
- `sgemm_tiled` — 2D block indexing, `__shared__` tile buffers, synchronized load/compute phases, and a final matrix-store path
- `stencil2d_5pt` — shared-memory tile + halo loads, `blockIdx.{x,y}` / `threadIdx.{x,y}` recovery, boundary guards, and a weighted output store

Those kernels are still more temp-heavy than ideal, but they are real corpus workloads that the canonical backend decompiles and regression-tests today.

For a larger shared-memory example, see `test_cu/golden_full_pass/rc4.pseudo.c`.
For concrete snapshots, see `test_cu/golden_full_pass/`.

## Quick start

### 1. Build and run on a bundled fixture

```bash
cargo run -- -i test_cu/if_loop.sass
```

A few more useful examples:

```bash
# Decompile a larger fixture and save the pseudo-C
cargo run -- -i test_cu/rc4.sass -o /tmp/rc4.pseudo.c

# Inspect CFG structure
cargo run -- -i test_cu/if_loop.sass --cfg-dot > /tmp/if_loop.cfg.dot

# Inspect optimized SSA
cargo run -- -i test_cu/if_loop.sass --ssa-dot -o /tmp/if_loop.ssa.dot

# Compare ABI interpretations when a dump is ambiguous
cargo run -- -i old_kernel.sass --abi-profile legacy140
cargo run -- -i old_kernel.sass --abi-profile modern160
```

If you omit `-i`, the binary uses the embedded demo file `test_cu/sample_verify_kernel.sass`.

### 2. Run the test suite

```bash
cargo test --quiet
```

### 3. Regenerate curated golden snapshots after an intentional backend change

```bash
cargo run --example regen_goldens
```

## Getting input from a real CUDA binary

`cudad` expects SASS disassembly text, not a `.cu` file and not a `.cubin` directly.

Typical workflow:

```bash
# Compile CUDA source to cubin
nvcc -arch=sm_89 -cubin my_kernel.cu -o my_kernel.cubin

# Dump SASS with cuobjdump
cuobjdump --dump-sass my_kernel.cubin > my_kernel.sass

# Or with nvdisasm
nvdisasm my_kernel.cubin > my_kernel.sass

# Decompile
cargo run -- -i my_kernel.sass -o my_kernel.pseudo.c
```

### Multi-function dumps

If your dump contains multiple `Function : ...` sections, the CLI now decompiles each function in sequence by default.

Useful patterns:

- decompile every function in a corpus-style dump:
  - `cargo run -- -i test_cu/corpus_sm100/arith_kernels.sass`
- select one named function from a multi-function dump:
  - `cargo run -- -i test_cu/corpus_sm100/arith_kernels.sass --function relu`
- inspect DOT for one function from a multi-function dump:
  - `cargo run -- -i test_cu/corpus_sm100/arith_kernels.sass --function relu --cfg-dot`

`--cfg-dot` and `--ssa-dot` require a single selected function, so use `--function <name>` with those modes on multi-function inputs.

## CLI reference

Current `main` options:

```text
-i, --input <INPUT>              Input SASS file (if not given, use SAMPLE_SASS)
-o, --output <OUTPUT>            Output file for structured output or SSA DOT
    --cfg-dot                    Dump CFG as DOT to stdout
    --ssa-dot                    Dump optimized SSA IR as DOT
    --function <FUNCTION>        Select one function from a multi-function dump by `Function :` name
    --abi-profile <ABI_PROFILE>  Force ABI profile (`auto|legacy140|modern160`)
```

Notes:
- default mode is the full decompiler pipeline
- multi-function dumps are split automatically for structured output
- `--function <name>` selects one function from a multi-function dump
- `--cfg-dot` writes DOT to stdout; redirect it with `>`
- `--ssa-dot` can print to stdout or write via `-o`
- `--cfg-dot` and `--ssa-dot` require a single selected function on multi-function inputs
- on the decompile path, `-o` writes the final pseudo-C file while stdout still prints a small status banner

## Canonical pipeline

The active backend is architecture-first and canonical-only. The main path is:

1. `decode_sass` in `src/parser.rs`
2. `build_cfg` in `src/cfg.rs`
3. `build_ssa` in `src/ir.rs`
4. IR optimization passes in `src/ir_dce.rs`, `src/ir_constprop.rs`, `src/ir_algebra.rs`, `src/ir_cse.rs`, and `src/ir_copyprop.rs`
5. `FunctionAnalysis` in `src/function_analysis.rs`
6. control-flow structurization in `src/structurizer/`
7. structured AST lowering in `src/ast_lowering.rs`
8. symbol/declaration planning in `src/symbol_plan.rs`
9. direct rendering from the canonical AST

The legacy `semantic_lift` backend has been deleted. The canonical pipeline is
the only supported full-pass path.

The default binary already drives that full pass. There is no separate "legacy pretty-printer mode" exposed as the normal CLI path.

## Practical workflows

### Review a kernel quickly

```bash
cargo run -- -i kernel.sass -o /tmp/kernel.pseudo.c
```

Then compare the pseudocode against source, PTX, or your own notes.

### Debug bad control flow

```bash
cargo run -- -i kernel.sass --cfg-dot > /tmp/kernel.cfg.dot
```

If the CFG is wrong, structurization will also be wrong.

### Debug analysis / lowering issues

```bash
cargo run -- -i kernel.sass --ssa-dot -o /tmp/kernel.ssa.dot
```

Use this when the output still contains low-level value plumbing and you need
to see the SSA that feeds `FunctionAnalysis` and canonical AST lowering.

### Refresh the canonical snapshots

```bash
cargo run --example regen_goldens
```

This updates `test_cu/golden_full_pass/` from the current backend.

## Library usage

`cudad` is also usable as a Rust library.

Minimal example:

```rust
use cudad::{build_cfg, build_ssa, decode_sass, split_decoded_functions};

fn main() {
    let sass = std::fs::read_to_string("kernel.sass").unwrap();

    let functions = split_decoded_functions(&sass);
    if functions.is_empty() {
        let cfg = build_cfg(decode_sass(&sass));
        let ssa = build_ssa(&cfg);
        println!("single function: {} SSA blocks", ssa.blocks.len());
    } else {
        for func in functions {
            let cfg = build_cfg(func.instrs.clone());
            println!("{} -> {} CFG nodes", func.name, cfg.node_count());
        }
    }
}
```

Useful public entry points include:
- `decode_instruction_line`, `decode_sass`, `split_decoded_functions`
- `build_cfg`
- `build_ssa`
- `build_decompile_artifacts`, `build_named_decompile_artifacts`
- `analyze_function_ir`, `analyze_function_ir_with_profile`
- ABI helpers in `src/abi.rs`

## Repository layout

- `src/parser.rs` — decoded SASS front-end
- `src/cfg.rs` — basic-block and CFG construction
- `src/ir.rs` — SSA IR builder
- `src/function_analysis.rs` — post-SSA ABI, type, and memory-space facts
- `src/structurizer/` — control-flow recovery
- `src/ast_lowering.rs` — canonical AST lowering from structured SSA + analysis
- `src/symbol_plan.rs` — deterministic declaration and temp planning
- `src/backend_pipeline.rs` — canonical full-pass driver
- `test_cu/` — fixtures, corpora, and golden outputs
- `docs/dev/decompiler_design.MD` — current backend design notes

## Test coverage

The repo keeps two kinds of regression coverage:

- curated full-pass snapshots in `test_cu/golden_full_pass/`
- invariant-based corpus tests across themed multi-kernel dumps in:
  - `test_cu/corpus/`
  - `test_cu/corpus_sm100/`
  - `test_cu/corpus_sm120/`

Those tests currently check things such as:
- deterministic output
- non-empty decompilation for every corpus function
- no leaked raw convergence barrier mnemonics
- no leaked SSA suffix tokens in named output
- bounded `goto BB...` counts on representative corpora
- SM 120 scheduling annotations stripped from final output

## Current strengths

- robust decoded front-end for operands, predicates, terminators, and scheduling-annotated dumps
- CFG/SSA construction is stable enough to support whole-corpus regression tests
- good coverage for arithmetic, compares, loads/stores, atomics, special registers, and common CUDA builtins
- ABI-aware rendering works on the curated SM 89 / SM 100 / SM 120 corpora
- the output is deterministic and snapshot-tested

## Current limitations

- output is pseudocode, not recompilable CUDA C
- pointer reconstruction is still incomplete; `addr64(...)` is not fully eliminated
- some kernels still produce too many temporaries or generic names
- structurization still falls back to `goto BB...` on harder multi-exit / irreducible shapes
- `switch` recovery is not implemented
- CLI does not yet select a function from a multi-function dump
- coverage outside the tested dump formats and architectures is limited

## Non-goals

Right now this project does not try to be:
- a verified decompiler
- a source-recovering pretty-printer
- a complete CUDA architecture database
- a drop-in replacement for mature commercial reversing tools

## Related reading

- [PNF Software: Reversing NVIDIA CUDA SASS code](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/)

## License

See `LICENSE`.
