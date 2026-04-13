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
- structural AST cleanup and typed rendering on the canonical backend,
- golden-based regression testing across the curated fixture directories.

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

1. **Decode SASS**  
   `decode_sass` in `src/parser.rs`
2. **Build CFG**  
   `build_cfg` in `src/cfg.rs`
3. **Build SSA IR**  
   `build_ssa` in `src/ir.rs`
4. **Structurize control flow**  
   `Structurizer::structure_function` in `src/structurizer.rs`
5. **Render structured pseudocode**  
   `Structurizer::pretty_print_with_lift_cleanup_and_names` in `src/structurizer.rs`

### Semantic lifting, naming, and final rendering

1. **Semantic lifting**  
   `lift_function_ir` in `src/semantic_lift.rs` maps SSA ops to typed AST expressions.
2. **Structural AST cleanup**  
   `src/ast_passes.rs` handles structural simplification, predicate cleanup, and `addr64` folding before final formatting.
3. **Structural name recovery**  
   `plan_structured_name_recovery_with_lift` in `src/name_recovery.rs` computes SSA-to-symbol mappings that are applied on the AST path, not by post-render regex surgery.
4. **ABI-aware typing/display pass**  
   `src/abi.rs` and `src/typed_output.rs` produce typed signatures, declarations, and CUDA builtin names such as `blockDim.x`, `gridDim.y`, and `threadIdx.z`.

### Full-pass test/golden pipeline

`src/test.rs` and `examples/regen_goldens.rs` keep curated snapshots for the canonical full-pass backend in:

- `test_cu/golden_full_pass/`

---

## Current progress

### What is working well

- End-to-end parse → CFG → SSA → structured output is stable.
- Semantic lifting via rule registry covers most common SASS opcodes, including arithmetic, memory, comparisons, bitwise, atomics (`atomicAdd`, `atomicInc`, etc.), and type casts (`(float)`, `(uint32_t)`).
- Structural AST cleanup and typed rendering now handle the final output cleanup on the canonical backend.
- Name recovery deterministically maps SSA values onto stable C-like symbols before the final render.
- ABI profile/alias inference provides `__global__` qualified typed signatures with pointer/scalar arg inference.
- Shared memory accesses render as `shmem_u8[addr]` / `shmem_u32[addr]` notation.
- CUDA builtins use standard dot notation: `blockDim.x`, `gridDim.y`, `threadIdx.z`.
- NOP instructions, dead code after `return`, BB labels, and zero-register writes are suppressed.
- PLOP3.LUT constant folding evaluates boolean LUT when predicate inputs are known.
- Default CLI (`cargo run -- -i file.sass`) runs full decompilation with all passes enabled.

### Regression status

- Golden fixtures are synchronized with the curated full-pass snapshot set.
- Test suite currently passes locally with `cargo test`.

---

## Running on real-world CUDA kernels

### Step 1: Extract SASS from a CUDA binary

You need a CUDA binary (`.cubin`, `.fatbin`, or executable with embedded CUDA). Use NVIDIA's `cuobjdump` to extract SASS disassembly:

```bash
# For a .cubin or .fatbin file:
cuobjdump -sass my_kernel.cubin > my_kernel.sass

# For an executable with embedded CUDA:
cuobjdump -sass my_program > all_kernels.sass

# To list available kernels first:
cuobjdump -symbols my_kernel.cubin
```

Alternatively, use `nvdisasm`:

```bash
nvdisasm my_kernel.cubin > my_kernel.sass
```

### Step 2: Isolate a single kernel (if needed)

If the binary contains multiple kernels, the SASS dump will contain all of them. You can either:

- Pass the full file — `cudad` processes all instructions as one function (suitable for single-kernel binaries).
- Manually extract the section for one kernel (look for `Function :` headers in the dump).

### Step 3: Run the decompiler

```bash
# Full decompilation (canonical full-pass backend):
cargo run -- -i my_kernel.sass

# Save output to a file:
cargo run -- -i my_kernel.sass -o my_kernel.pseudo.c

# Debug CFG only:
cargo run -- -i my_kernel.sass --cfg-dot
```

### Step 4: Understand the output

The decompiler produces pseudo-C with:
- **`__global__ void kernel(...)`** — inferred typed signature from ABI const-memory patterns
- **`__shared__` arrays** — detected shared memory usage
- **CUDA builtins** — `threadIdx.x`, `blockDim.x`, `blockIdx.x`, etc.
- **Structured control flow** — `if/else`, `do/while`, `while` loops recovered from SASS branches

### Great usage examples

```bash
# 1. Decompile one kernel with the canonical pipeline
cargo run -- -i test_cu/if_loop.sass

# 2. Save canonical output for side-by-side source comparison
cargo run -- -i test_cu/rc4.sass -o /tmp/rc4.pseudo.c

# 3. Force an older ABI profile when testing pre-Ampere style dumps
cargo run -- -i old_kernel.sass --abi-profile legacy140

# 4. Inspect CFG shape when structurization looks suspicious
cargo run -- -i test_cu/if_loop.sass --cfg-dot > /tmp/if_loop.cfg.dot

# 5. Inspect optimized SSA when debugging lifting/name-recovery issues
cargo run -- -i test_cu/if_loop.sass --ssa-dot > /tmp/if_loop.ssa.dot

# 6. Refresh curated full-pass snapshots after intentional backend changes
cargo run --example regen_goldens
```

A practical review loop for real kernels is usually:

1. extract one kernel with `cuobjdump -sass` or `nvdisasm`
2. run `cargo run -- -i kernel.sass -o kernel.pseudo.c`
3. diff the result against source or expected semantics
4. if the output shape looks wrong, inspect `--cfg-dot` first, then `--ssa-dot`

### Supported architectures

| SM Target | Status |
|-----------|--------|
| sm_89 (Ada Lovelace) | Primary testing, best coverage |
| sm_100 (Blackwell) | Corpus tested (11 kernels), good coverage |
| sm_75–sm_86 | Should work (similar SASS format), limited testing |
| sm_50–sm_70 | May work with `--abi-profile legacy140` |

### Example end-to-end

```bash
# 1. Compile a CUDA source to cubin
nvcc -arch=sm_89 -cubin my_kernel.cu -o my_kernel.cubin

# 2. Extract SASS
cuobjdump -sass my_kernel.cubin > my_kernel.sass

# 3. Decompile
cargo run -- -i my_kernel.sass -o my_kernel.pseudo.c

# 4. View result
cat my_kernel.pseudo.c
```

---

## Next steps

1. **64-bit pointer reconstruction**  
   Replace `addr64(lo, hi)` patterns with typed pointer expressions (`*(uint32_t*)(ptr + offset)`).

2. **Parameter name resolution**  
   Map `param_0`/`param_1` through constant-memory loads to resolve kernel argument names in expressions.

3. **Cross-version/cross-arch validation**  
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

#### 3) Generate canonical full-pass decompilation

This is now the **default** when using `-i` with no other flags:

```bash
cargo run -- -i test_cu/if_loop.sass
```

#### 4) Inspect the canonical structured output

```bash
cargo run --bin main -- --input test_cu/if_loop.sass
```

Notes:

- We do **not** currently provide a first-class standalone def-use chain dump CLI yet.
- Def-use information exists internally in SSA and lifting passes, but the canonical renderer no longer emits legacy phi/live-in comment scaffolding.

### Basic usage (CLI)

Use the binary on a SASS file (defaults to full decompilation):

```bash
cargo run -- -i test_cu/if_loop.sass
```

The CLI now exposes the canonical full-pass decompiler by default.

Debug outputs still exist for lower layers:

- `--cfg-dot`
- `--ssa-dot`
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
