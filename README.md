# cudad

`cudad` is an experimental CUDA SASS decompiler.

> ⚠️ **Warning**
>
> This project is heavily vibe-coded and experimental. Do **not** rely on it for production reverse engineering, security decisions, or correctness-critical workflows.

It currently emphasizes:

- stable parse/CFG/SSA construction,
- conservative structurization,
- optional semantic lifting and name recovery,
- ABI-aware display/declaration inference,
- golden-based regression testing.

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

- Current fixture coverage is primarily one SASS family/style: mostly `sm_89` with `code version = [1,7]`-style dumps.
- Coverage across older/newer SM targets and different disassembly formats is still limited.

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

### Example output (`golden_full_pass/rc4.pseudo.c`, abridged)

```c
...
void kernel(uint8_t* arg0_ptr, int32_t arg2, uint8_t* arg4_ptr, uint8_t* arg6_ptr, uint32_t arg8, uint32_t arg9) {
   BB0 {
      v0 = abi_internal_0x28;
      ctaid_x = blockIdx.x;
   }
   if (!((int32_t)(ctaid_x) >= (int32_t)(arg9))) {
      BB1 {
         tid_x = threadIdx.x;
         b1 = (int32_t)(tid_x) > (int32_t)(255);
         b2 = tid_x != 0;
      }
      if (!((int32_t)(tid_x) > (int32_t)(255))) {
         while (!((int32_t)(tid_x) >= (int32_t)(256))) {
            shmem_u8[tid_x] = tid_x;
            tid_x = tid_x + blockDimX;
         }
      }
   }
}
...
```

This is the style of transformation the project targets: low-level SASS blocks/predicates into conservative structured pseudocode with optional naming and ABI hints.

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

### Optional stages (used in lifted/full-pass outputs)

1. **Semantic lifting** (expression-level cleanup)  
   `lift_function_ir` in `src/semantic_lift.rs`
2. **Name recovery + post-render cleanup**  
   `recover_structured_output_names` in `src/name_recovery.rs`
3. **ABI-aware typing/display pass**  
   ABI profile detection, const-memory annotation, arg aliasing and typed signatures in `src/abi.rs`

### Full-pass test/golden pipeline

`src/test.rs` composes CFG + SSA + structurizer + lift + name recovery + ABI rendering, then compares against fixtures in:

- `test_cu/golden/`
- `test_cu/golden_lifted/`
- `test_cu/golden_lifted_named/`
- `test_cu/golden_full_pass/`

---

## Current progress

### What is working well

- End-to-end parse → CFG → SSA → structured output is stable.
- Semantic lifting reduces raw opcode noise while preserving conservative fallbacks.
- Name recovery deterministically rewrites SSA tokens to C-like names.
- ABI profile/alias inference provides typed signatures and more readable const-memory semantics.
- Predication cleanup has improved, especially for predicated-only temporary handling and fake-merge ternary patterns.

### Regression status

- Golden fixtures are synchronized with current behavior.
- Test suite currently passes (`cargo test`): **145 passed, 0 failed**.

---

## Next steps

1. **Push predication cleanup earlier in pipeline**  
   Move more predication semantics from post-render text rewriting into IR/structurizer-level representation.

2. **Reduce variable reuse ambiguity**  
   Improve naming/SSA presentation so reused temps (e.g. `v3`) are less likely to look semantically conflated.

3. **Control-flow recovery expansion**  
   Improve structurizer handling for harder loop/branch shapes and reduce fallback `goto` usage.

4. **Switch/multi-way branch support**  
   Add normalization for branch tables / multi-way control flow.

5. **Richer type propagation**  
   Strengthen inferred local/argument typing and pointer/value distinction beyond current heuristics.

6. **Golden/test quality gates**  
   Add more fixture coverage for predication corner cases and loop-carried dataflow.

7. **Cross-version/cross-arch validation**  
   Expand test corpus across multiple SM targets and SASS dump formats.

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

#### 4) Generate lifted + named + typed output

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --struct-code --semantic-lift --recover-names --typed-decls --abi-map
```

#### 5) Inspect phi/live-in hints (closest thing to def-use visibility today)

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --struct-code --recover-names --phi-merge-comments
```

Notes:

- We do **not** currently provide a first-class standalone def-use chain dump CLI yet.
- Def-use information exists internally in SSA/lifting/name-recovery passes, and partial hints can be seen via phi/live-in comments.

### Basic usage (CLI)

Use the binary on a SASS file:

```bash
cargo run --bin main -- --input test_cu/if_loop.sass --struct-code
```

Useful options (can be combined):

- `--semantic-lift`
- `--recover-names`
- `--typed-decls`
- `--abi-map`
- `--abi-profile auto|legacy140|modern160`

### Run tests

```bash
cargo test
```

### Regenerate all goldens

```bash
cargo run --example regen_goldens
```

### Key docs

- Design notes: `docs/dev/decompiler_design.MD`
- Instruction notes: `docs/dev/insts.MD`

---

## Project status

This project is in active development. It is best viewed as a learning and experimentation codebase that prioritizes conservative behavior over aggressive prettification.
