# Decompiler Design Notes

This file describes the current canonical backend, not historical experiments.
The public CLI and the regression suite both target the same end-to-end path.

## Canonical pipeline

The active decompiler flow is:

1. `decode_sass` in `src/parser.rs`
2. `build_cfg` in `src/cfg.rs`
3. `build_ssa` in `src/ir.rs`
4. IR cleanup/optimization passes:
   - `ir_dce`
   - `ir_constprop`
   - `ir_algebra`
   - `ir_cse`
   - `ir_copyprop`
5. `analyze_function_ir` / `FunctionAnalysis` in `src/function_analysis.rs`
6. `Structurizer::structure_function` in `src/structurizer/`
7. `lower_structured_function` in `src/ast_lowering.rs`
8. symbol/declaration planning in `src/symbol_plan.rs`
9. canonical AST cleanup in `src/canonical_ast_passes.rs`
10. direct render from `StructuredFunction`

The CLI in `src/bin/main.rs` already drives that full pass by default. The only public debug side exits are `--cfg-dot` and `--ssa-dot`.

## Front-end model

The parser is no longer a raw text splitter. The canonical front-end decodes SASS into typed records:

- `DecodedFunction`
- `DecodedInstruction`
- `DecodedOperand`
- `Predicate`
- `SchedulingInfo`
- `TerminatorKind`

Important front-end properties:

- operand parsing is tokenization-aware; nested brackets and descriptors are handled structurally
- terminators are decoded explicitly instead of reconstructed later from opcode string heuristics
- scheduling annotations in newer dumps are stripped before tokenization but preserved as metadata on the decoded instruction
- multi-function dumps are split with `split_decoded_functions`, which is what the corpus tests use

## CFG and SSA boundaries

`build_cfg` consumes decoded instructions and explicit terminators.

The control-flow layer depends on a few invariants:

1. basic-block leaders come from entry, jump targets, and fallthrough boundaries
2. successor edges are derived from `TerminatorKind`, not ad hoc opcode matching
3. CFG node payloads carry decoded instructions all the way into SSA construction
4. pseudo-register sentinels such as `RZ`, `PT`, `URZ`, and `UPT` are never treated as normal SSA definition targets

`build_ssa` then creates the mid-end IR used by optimization, analysis,
structurization, and canonical lowering.

## Function analysis, lowering, and symbol planning

The canonical backend no longer depends on semantic-lift side tables or
post-render name recovery. Instead it threads explicit analysis facts through
the structured lowering pipeline.

Key pieces:

- `FunctionAnalysis` classifies ABI facts, pointer roots, builtin accesses,
  memory spaces, and scalar/pointee types on optimized SSA
- `lower_structured_function` lowers `StructuredStatement + FunctionAnalysis`
  directly into the canonical AST
- `plan_symbols` assigns params, locals, shared objects, and deterministic temp
  names without reading rendered text back in
- `src/semantic_lift.rs` remains only as a test-only harness for targeted
  legacy lift coverage that has not yet been translated into canonical tests

Symbol planning is intentionally conservative:

- params/builtins get semantic names when analysis proves them
- reused hardware registers can still split into multiple canonical temporaries
- when stronger recovery is unavailable, the output falls back to deterministic
  temp-style names instead of heuristic pretty names

## ABI, memory spaces, and rendering

`src/abi.rs` and `src/function_analysis.rs` resolve constant-memory accesses and
address roots into CUDA-facing meaning where possible.

That layer is responsible for:

- detecting an ABI profile (`auto`, `legacy140`, `modern160`)
- mapping known builtins such as `blockDim.x`, `gridDim.y`, `threadIdx.x`, and `blockIdx.x`
- inferring argument aliases from parameter-window accesses
- classifying `Param`, `Const`, `Global`, `Local`, `Shared`, and `Generic`
  memory operations before lowering
- feeding typed declarations and signature hints into the final render

Rendering now happens directly from the canonical AST plus symbol plan; there
is no `typed_output.rs`, regex declaration synthesis, or post-render pointer
repair stage in the live backend.

## Current regression model

The repo uses two complementary regression strategies:

1. curated snapshots in `test_cu/golden_full_pass/`
2. invariant-based whole-corpus tests in `src/test.rs`

The corpus runner feeds every function from:

- `test_cu/corpus/`
- `test_cu/corpus_sm100/`
- `test_cu/corpus_sm120/`

through the real full-pass backend and checks properties such as:

- non-empty output
- determinism across repeated runs
- no leaked convergence barrier mnemonics
- no leaked SSA suffix tokens in named output
- bounded `goto BB...` counts
- Blackwell builtin resolution on the SM 100 corpus
- scheduling annotation stripping on the SM 120 corpus

This is the best "real backend" test in the repo; it is deliberately stronger than byte-for-byte golden comparisons alone.

## Practical debugging recipes

### Canonical smoke test

```bash
cargo run -- -i test_cu/if_loop.sass
```

### Inspect CFG structure first

```bash
cargo run -- -i test_cu/if_loop.sass --cfg-dot > /tmp/if_loop.cfg.dot
```

Use this when control flow looks wrong. If the CFG is wrong, structurization and lifting will also be wrong.

### Inspect optimized SSA

```bash
cargo run -- -i test_cu/if_loop.sass --ssa-dot -o /tmp/if_loop.ssa.dot
```

Use this when value flow, comparisons, pointer reconstruction, or naming looks suspicious.

### Refresh canonical snapshots

```bash
cargo run --example regen_goldens
```

Do this only after `cargo test --quiet` is green and the output change is intentional.

## Known limitations

These are current backend limits, not aspirational future behavior:

- output is pseudocode, not recompilable CUDA C
- `addr64(...)` recovery is still incomplete
- some kernels still overproduce temporaries
- harder multi-exit or irreducible regions still degrade to explicit `goto BB...`
- `switch` recovery is not implemented
- the CLI currently expects a single function stream, even though the library and corpus runner support multi-function dumps via `split_decoded_functions`

## Non-canonical / experimental work

There are still internal experiments and utility modules in the tree, but they are not the contract surface for the CLI.
When documenting behavior, prefer the pipeline and tests described above over historical notes or abandoned intermediate paths.
