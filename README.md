# cudad

`cudad` is an experimental CUDA SASS decompiler.

> ⚠️ **Warning**
>
> This project is heavily vibe-coded and experimental. Do **not** rely on it for production reverse engineering, security decisions, or correctness-critical workflows.

It currently emphasizes:

- stable parse/CFG/SSA construction,
- conservative structurization,
- semantic lifting with a rule-registry pattern for SASS opcodes,
- name recovery, ABI-aware typed declarations, and semantic symbolization,
- post-rendering optimization passes (DCE, CSE, constant propagation),
- golden-based regression testing across four fixture directories.

It does **not** yet aim for broad architecture/version coverage or production-grade decompilation quality.

## Why this project exists

Yes, there is already nice prior work on CUDA SASS reversing/decompilation (for example: [Jeb's SASS decompiler](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/)).

This project exists mostly because:

- it is fun,
- it is a hands-on learning exercise for compiler/decompiler internals,
- it is an experiment to probe LLM ability and limits on graph-heavy tasks (CFG, dominators, SSA, structurization).

So the goal here is not to compete with mature tools; it is to learn by building and iterating.

## Input source expectation

The `.sass` input used by this project is expected to be disassembly text produced by NVIDIA CUDA Toolkit tools (for example `cuobjdump` / `nvdisasm`).

Current parsing/testing is primarily aligned with `cuobjdump`-style text dumps seen in the fixtures.

### Why SASS instead of PTX?

PTX is usually easier to read and reverse than SASS.

However, PTX can be stripped from CUDA binaries, while SASS machine code is what must exist for execution.

So this project focuses on SASS as the more robust target when PTX is missing.

### Current machine/version coverage

- Primary fixture coverage: `sm_89` with `code version = [1,7]`-style dumps.
- SM 100 (Blackwell) corpus: 11 kernel files in `test_cu/corpus_sm100/` covering arithmetic, branching, control flow, crypto, data processing, FFT, image processing, linear algebra, ML, sorting, and stencil kernels.
- Coverage across older SM targets and different disassembly formats is still limited.

## Quick showcase

### Example input (`test_cu/rc4.sass`, abridged)

```sass
...
/*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
/*0010*/ S2R R0, SR_CTAID.X ;
/*0020*/ ISETP.GE.AND P0, PT, R0, c[0x0][0x184], PT ;
/*0030*/ @P0 EXIT ;
/*0040*/ S2R R2, SR_TID.X ;
/*0050*/ ULDC.64 UR6, c[0x0][0x118] ;
/*0060*/ BSSY B0, 0xf0 ;
/*0070*/ ISETP.GT.AND P1, PT, R2, 0xff, PT ;
/*0080*/ ISETP.NE.AND P0, PT, R2, RZ, PT ;
/*0090*/ @P1 BRA 0xe0 ;
/*00a0*/ STS.U8 [R2], R2 ;
/*00b0*/ IADD3 R2, R2, c[0x0][0x0], RZ ;
/*00c0*/ ISETP.GE.AND P1, PT, R2, 0x100, PT ;
/*00d0*/ @!P1 BRA 0xa0 ;
...
```

### Example output (abridged)

By default, running `cargo run -- -i test_cu/rc4.sass` produces full decompilation (structured code + semantic lift + name recovery + typed declarations + ABI mapping + semantic symbolization):

```c
__global__ void kernel(uint8_t* arg0_ptr, int32_t arg2, uint8_t* arg4_ptr,
                       uint8_t* arg6_ptr, uint32_t arg8, uint32_t arg9) {
  __shared__ uint8_t shmem_u8[256];
  uint32_t ctaid_x;
  uint32_t tid_x;
  ...
  ctaid_x = blockIdx.x;
  if (b0) return;
    tid_x = threadIdx.x;
    b1 = (int32_t)(tid_x) > (int32_t)(255);
    b2 = tid_x != 0;
  if (!((int32_t)(tid_x) > (int32_t)(255))) {
    do {
      shmem_u8[tid_x] = tid_x;
      tid_x = tid_x + blockDim.x;
    } while(!(b1));
  }
    __syncthreads();
  if (!(b2)) {
    v3 = abs(arg2);
    v5 = (float)(v3);
    v6 = rcp_approx(v5);
    ...
  }
  do {
    v22 = abs(tid_x);
    v23 = shmem_u8[tid_x];
    ...
  } while(!(b26));
  ...
}
```

This is the style of transformation the project targets: low-level SASS blocks/predicates into conservative structured pseudocode with CUDA-style types, ABI-inferred arguments, shared memory notation, and semantic intrinsic names.

---

## Current decompiler pipeline

### Core path

1. **Parse SASS**  
   `parse_sass` in `src/parser.rs`
2. **Build CFG**  
   `build_cfg` in `src/cfg.rs`
3. **Build SSA IR**  
   `build_ssa` in `src/ir.rs`
4. **Structurize control flow**  
   `Structurizer::structure_function` in `src/structurizer.rs`
5. **Render structured pseudocode**  
   `Structurizer::pretty_print` in `src/structurizer.rs`

### Semantic lifting and name recovery

1. **Semantic lifting** (expression-level cleanup)  
   `lift_function_ir` in `src/semantic_lift.rs` — uses a rule-registry pattern (`src/semantic_lift/rules/`) to map SASS opcodes to C-like expressions. Rules cover arithmetic (`IMAD`, `IADD3`, `FFMA`), memory (`LDG`, `LDS`, `STG`, `STS`, `ULDC`, `ATOMS`), comparisons (`ISETP`, `FSETP`), bitwise (`LOP3`, `SHF`, `PLOP3`), and type conversions (`I2F`, `F2I`).
2. **Name recovery + post-render cleanup**  
   `recover_structured_output_names` in `src/name_recovery.rs`
3. **ABI-aware typing/display pass**  
   ABI profile detection, const-memory annotation, arg aliasing and typed signatures in `src/abi.rs`. Builtins render as CUDA C notation: `blockDim.x`, `gridDim.y`, `threadIdx.z`, etc.

### Post-rendering optimization passes

Applied as text-level transformations after structured rendering:

1. **Common Subexpression Elimination (CSE)** — deduplicates repeated `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x`, and `ConstMem(...)` loads.
2. **Constant propagation** — inlines single-definition small literal constants into their use sites and removes the defining assignment.
3. **Dead Code Elimination (DCE)** — removes unused variable assignments in a fixpoint loop.
4. **Duplicate guard elimination** — collapses identical `if (pred) return;` guards.

### Full-pass test/golden pipeline

`src/test.rs` composes CFG + SSA + structurizer + lift + name recovery + ABI rendering + post-rendering optimization, then compares against fixtures in:

- `test_cu/golden/`
- `test_cu/golden_lifted/`
- `test_cu/golden_lifted_named/`
- `test_cu/golden_full_pass/`

---

## Current progress

### What is working well

- End-to-end parse → CFG → SSA → structured output is stable.
- Semantic lifting via rule registry covers most common SASS opcodes, including arithmetic, memory, comparisons, bitwise, atomics (`atomicAdd`, `atomicInc`, etc.), and type casts (`(float)`, `(uint32_t)`).
- Post-rendering passes (CSE, constant propagation, DCE, duplicate guard elimination) significantly reduce variable count and noise.
- Name recovery deterministically rewrites SSA tokens to C-like names.
- ABI profile/alias inference provides `__global__` qualified typed signatures with pointer/scalar arg inference.
- Shared memory accesses render as `shmem_u8[addr]` / `shmem_u32[addr]` notation.
- CUDA builtins use standard dot notation: `blockDim.x`, `gridDim.y`, `threadIdx.z`.
- NOP instructions, dead code after `return`, BB labels, and zero-register writes are suppressed.
- PLOP3.LUT constant folding evaluates boolean LUT when predicate inputs are known.
- Default CLI (`cargo run -- -i file.sass`) runs full decompilation with all passes enabled.

### Regression status

- Golden fixtures are synchronized with current behavior across all four directories.
- Test suite currently passes (`cargo test`): **226 passed, 0 failed**.

---

## Next steps

1. **64-bit pointer reconstruction**  
   Replace `addr64(lo, hi)` patterns with typed pointer expressions (`*(uint32_t*)(ptr + offset)`).

2. **Parameter name resolution**  
   Map `param_0`/`param_1` through constant-memory loads to resolve kernel argument names in expressions.

3. **Phi node lowering**  
   Improve rendering of SSA phi nodes — currently shown as comments or omitted; lower to explicit variable assignments at merge points.

4. **Variable naming improvements**  
   Use semantic seeds more aggressively (e.g., name variables based on their defining operation) and reduce live-in/undefined variable noise.

5. **Switch/multi-way branch support**  
   Add normalization for branch tables / multi-way control flow.

6. **Richer type propagation**  
   Strengthen inferred local/argument typing and pointer/value distinction beyond current heuristics.

7. **Cross-version/cross-arch validation**  
   Expand test corpus across additional SM targets and SASS dump formats beyond sm_89 and sm_100.

---

## Developer workflow

### Common use cases

#### 1) Generate a CFG graph (DOT)

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --cfg-dot > cfg.dot
```

#### 2) Generate an SSA graph (DOT)

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --ssa-dot --output ssa.dot
```

If you have Graphviz installed, render to SVG:

```bash
dot -Tsvg ssa.dot -o ssa.svg
```

#### 3) Generate structured pseudocode

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --struct-code
```

#### 4) Generate lifted + named + typed output (full decompilation)

This is now the **default** when using `-i` with no other flags:

```bash
cargo run -- -i test_cu/if_loop.sass
```

Equivalent to explicitly specifying all passes:

```bash
cargo run -- -i test_cu/if_loop.sass --struct-code --semantic-lift --recover-names --typed-decls --abi-map --semantic-symbolize
```

#### 5) Inspect phi/live-in hints (closest thing to def-use visibility today)

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --struct-code --recover-names --phi-merge-comments
```

Notes:

- We do **not** currently provide a first-class standalone def-use chain dump CLI yet.
- Def-use information exists internally in SSA/lifting/name-recovery passes, and partial hints can be seen via phi/live-in comments.

### Basic usage (CLI)

Use the binary on a SASS file (defaults to full decompilation):

```bash
cargo run -- -i test_cu/if_loop.sass
```

Individual passes can be enabled selectively with:

- `--struct-code`
- `--semantic-lift`
- `--recover-names`
- `--typed-decls`
- `--abi-map`
- `--semantic-symbolize`
- `--abi-profile auto|legacy140|modern160`

### Run tests

```bash
cargo test
```

### Regenerate all goldens

```bash
REGEN_GOLDEN=1 cargo test regen_golden_files -- --ignored
```

### Key docs

- Design notes: `docs/dev/decompiler_design.MD`
- Instruction notes: `docs/dev/insts.MD`

---

## Project status

This project is in active development. It is best viewed as a learning and experimentation codebase that prioritizes conservative behavior over aggressive prettification.
