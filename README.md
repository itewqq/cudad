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

For `test_cu/test_div.sass`, the canonical output looks like this:

```c
__global__ void kernel(int32_t arg0, int32_t arg1, uint32_t arg2, uint32_t arg3) {
  uint32_t v0;
  int32_t u0;
  uint32_t u1;
  int32_t u2;
  uint32_t v1;
  bool b0;
  uint32_t v2;
  uint32_t v3;
  ...

  v0 = abs(arg1);
  u0 = arg0;
  u1 = arg1;
  u2 = u0 ^ u1;
  v1 = (float)(v0);
  b0 = (int32_t)(0) <= (int32_t)(u2);
  ...
  *addr64(v9, v15) = v16;
  return;
}
```

That is representative of the current backend:
- signatures are inferred heuristically from ABI usage
- locals are typed, but names are still generic (`vN`, `bN`, `uN`) unless stronger recovery exists
- semantic lift recovers helpers such as `abs`, `rcp_approx`, `mul_hi_u32`, CUDA builtins, and shared-memory access
- pointer reconstruction is partial, so `addr64(lo, hi)` still appears in many kernels

For concrete snapshots, see `test_cu/golden_full_pass/`.

## Quick start

### 1. Build and run on a bundled fixture

```bash
cargo run -- -i test_cu/test_div.sass
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

### Important current limitation: one function per CLI input

The CLI currently runs the canonical pipeline on one instruction stream. If your dump contains multiple `Function : ...` sections, either:

- extract the single kernel you care about into its own `.sass` file, or
- use the library helper `split_decoded_functions()` and run functions one by one

The test corpus uses `split_decoded_functions()` for multi-kernel dumps, but the public CLI does not yet expose a `--function <name>` selector.

## CLI reference

Current `main` options:

```text
-i, --input <INPUT>              Input SASS file (if not given, use SAMPLE_SASS)
-o, --output <OUTPUT>            Output file for structured output or SSA DOT
    --cfg-dot                    Dump CFG as DOT to stdout
    --ssa-dot                    Dump optimized SSA IR as DOT
    --abi-profile <ABI_PROFILE>  Force ABI profile (`auto|legacy140|modern160`)
```

Notes:
- default mode is the full decompiler pipeline
- `--cfg-dot` writes DOT to stdout; redirect it with `>`
- `--ssa-dot` can print to stdout or write via `-o`
- on the decompile path, `-o` writes the final pseudo-C file while stdout still prints a small status banner

## Canonical pipeline

The active backend is architecture-first and canonical-only. The main path is:

1. `decode_sass` in `src/parser.rs`
2. `build_cfg` in `src/cfg.rs`
3. `build_ssa` in `src/ir.rs`
4. IR optimization passes in `src/ir_dce.rs`, `src/ir_constprop.rs`, `src/ir_algebra.rs`, `src/ir_cse.rs`, and `src/ir_copyprop.rs`
5. control-flow structurization in `src/structurizer.rs`
6. semantic lift in `src/semantic_lift.rs`
7. ABI annotation + argument aliasing in `src/abi.rs`
8. structural name recovery in `src/name_recovery.rs`
9. AST cleanup in `src/ast_passes.rs`
10. typed final rendering in `src/typed_output.rs`

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

### Debug lifting / naming issues

```bash
cargo run -- -i kernel.sass --ssa-dot -o /tmp/kernel.ssa.dot
```

Use this when the output still contains low-level value plumbing and you need to see what SSA the lifter actually received.

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
- `lift_function_ir`
- ABI helpers in `src/abi.rs`

## Repository layout

- `src/parser.rs` — decoded SASS front-end
- `src/cfg.rs` — basic-block and CFG construction
- `src/ir.rs` — SSA IR builder
- `src/semantic_lift.rs` — opcode-to-expression lift rules
- `src/structurizer.rs` — control-flow recovery
- `src/abi.rs` — ABI slot decoding and argument alias inference
- `src/name_recovery.rs` — structural symbol assignment
- `src/typed_output.rs` — typed final render
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
