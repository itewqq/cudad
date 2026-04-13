# Decompiler Design Notes (Stabilization Pass)

## Pipeline

The canonical path for this project is:

1. `decode_sass` (`src/parser.rs`)
2. `build_cfg` (`src/cfg.rs`)
3. `build_ssa` (`src/ir.rs`)
4. `Structurizer::structure_function` (`src/structurizer.rs`)
5. `Structurizer::pretty_print_with_lift_cleanup_and_names` (canonical structured render)
6. `render_typed_structured_output` (`src/typed_output.rs`)

Semantic lifting stage:

1. `lift_function_ir` (`src/semantic_lift.rs`) computes conservative
   expression rewrites keyed by `StatementRef`.
2. Lift rules are organized via a **rule registry** (`src/semantic_lift/rules/`):
   - `basic.rs` — arithmetic (`IMAD`, `IADD3`, `FFMA`, `FMUL`, `FADD`), moves (`MOV`, `UMOV`), type casts (`I2F`, `I2FP`, `F2I`), math intrinsics (`MUFU.*`), select/min-max (`SEL`, `FSEL`, `FMNMX`, `VIMNMX`, `VIADDMNMX`), byte permute (`PRMT`), special registers (`S2R`, `S2UR`).
   - `memory.rs` — loads (`LDG`, `LDS`, `LDC`, `ULDC`, `LDCU`), atomic operations (`ATOMS`, `ATOM`, `RED` → CUDA `atomicXxx()` calls).
   - `bitwise.rs` — shifts (`SHF`), logical (`LOP3`, `PLOP3.LUT` with constant folding), `LEA.HI`.
   - `comparison.rs` — integer/float comparisons (`ISETP`, `FSETP`, `UISETP`) including unordered float variants (`GEU`, `GTU`, `LEU`, `LTU`, `EQU`, `NEU`).
3. The active backend always feeds lifted expressions into the canonical AST/render path; there is no separate CLI-exposed partial lift mode anymore.

Name recovery stage:

1. `plan_structured_name_recovery_with_lift` (`src/name_recovery.rs`) computes
   SSA-to-symbol mappings before final rendering.
2. Merging is conservative: only phi/copy-congruent SSA versions are merged;
   reused register families can split into multiple recovered names.
3. Name recovery is part of the default CLI path; the style is currently fixed to stable temp-style names in the binary.

Declaration stage:

1. `AbiArgAliases::render_typed_arg_declarations` (`src/abi.rs`) emits
   heuristic typed argument declarations from ABI alias inference.
2. `infer_local_typed_declarations` (`src/abi.rs`) emits heuristic local
   declarations from SSA destinations and opcode usage hints
   (`float`, `uint8_t`, `uintptr_t`, etc.).
3. Typed declarations are part of the default CLI path; structured output is wrapped as `__global__ void kernel(<typed args>) { ... }` with inferred locals.

Structural cleanup stage (`src/ast_passes.rs`):

Applied on the typed AST before final rendering:

1. `ast_dce` removes dead pure assignments while preserving stores, returns, and side effects.
2. `ast_predicate_cleanup` drops duplicate return guards and other redundant predicate scaffolding.
3. `ast_addr64_fold` structurally recovers pointer expressions from low/high helper chains.
4. `ast_simplify` normalizes the final statement tree before rendering.

Experimental modules are intentionally not part of this critical path.

## ABI Profile Layer

`src/abi.rs` adds an architecture-aware mapping layer for `c[bank][offset]`:

1. `AbiProfile::detect` infers a profile from observed parameter windows (e.g. `0x140` vs `0x160` bases).
   - `AbiProfile::detect_with_sm` can also use parsed SM metadata from `.headerflags` / `.target sm_xx` as fallback.
2. `AbiProfile::resolve_constmem` maps known built-ins (`blockDim.x`, `gridDim.y`, etc.) and parameter words (`param_i[j]`).
3. `AbiDisplay` renders symbolic ABI names in DOT/pseudocode output.
   - CLI control: `--abi-profile auto|legacy140|modern160`.
4. `annotate_function_ir_constmem` creates typed annotations (`ConstMemSemantic`) over SSA statements.
   - This pass is non-invasive and keeps SSA algorithm/output semantics intact.
   - It is intended as the bridge for future structurizer/HL passes that need ABI meaning, not raw offsets.
5. `infer_arg_aliases` derives argument-oriented alias names from typed annotations:
   - Example renderings: `arg0_ptr.lo32`, `arg0_ptr.hi32`, `arg2_u64.lo32`.
   - Emits heuristic confidence (`high|medium|low`) based on usage patterns.

This layer is display-oriented and does not change SSA construction semantics.
Unknown offsets remain explicit raw `ConstMem(...)` values or `abi_internal_0x..` slots.

## IR Invariants Required by Structurizer

1. `IRBlock::id` must match the CFG basic-block `id`.
2. `IRBlock::start_addr` must match CFG block `start` for address->node mapping.
3. `IRBlock::irdst` must represent outgoing control edges with conditions:
   - `IRCond::True` for unconditional edges.
   - `IRCond::Pred` for predicate-conditioned edges.
4. Predicate conditions should be explicit and stable after SSA renaming.
5. Immutable pseudo-registers (`RZ`, `PT`, `URZ`, `UPT`) are not SSA definition targets.

## Structurizer Pattern Order

`structure_region_recursive` processes patterns in this order:

1. Loop recovery (`try_structure_loop`)
2. If / if-else recovery (`try_structure_if`)
3. Sequence extension (single successor)
4. Fallback to `UnstructuredJump` (`goto`) for unsupported branch shapes

This order intentionally prefers high-level constructs first and preserves output totality
by falling back to goto when recognition is uncertain.

## Rendering Cleanup (Structurizer)

The structurizer's canonical render path applies several cleanup rules during rendering:

1. **BB label suppression**: Basic block labels (`BB0 { ... }`) are not emitted; statements render directly at the current indentation level.
2. **NOP suppression**: `NOP` instructions are skipped.
3. **Dead code after return**: After an unconditional `RET`/`EXIT`, remaining statements in the block are not rendered.
4. **RZ/PT-destination suppression**: Writes to the zero register (`RZ`) or always-true predicate (`PT`) that have no memory side effect are skipped. When the dest is `RZ` but the expression has a side effect (e.g., `atomicInc(...)`), it is rendered as an expression statement without an assignment.
5. **Convergence barrier suppression**: `BSYNC`/`BSSY`/`WARPSYNC`/`BAR.SYNC` instructions are suppressed in lifted mode (except `BAR.SYNC` which renders as `__syncthreads()`).
6. **Internal comment suppression**: `// Condition from BB{N}` and `// Loop header BB{N}` comments are not emitted.

## Known Unsupported / Partial Cases

1. Complex nested branch reconstruction inside loop bodies is still partial,
   but simple branch diamonds inside loop bodies are now recovered as `if`
   instead of unconditional goto fallback when merge structure is clear.
2. Multi-way branch normalization to `switch` is not implemented.
3. Some loop exits still degrade to goto fallback depending on CFG shape.

These cases should not panic; they should emit explicit goto-based pseudocode.

## Debug Checkpoints

Suggested instrumentation points when debugging structurization:

1. Start of `structure_region_recursive` loop for active node transitions.
2. Entry/exit of `try_structure_if` with detected merge node.
3. Entry/exit of `try_structure_loop` with computed loop body and exits.
4. `structure_loop_body_recursive` successor classification (`break`, `continue`, fallback goto).

Use these checkpoints to correlate CFG nodes, `irdst`, and final pseudocode shape.

## Practical Recipes

A few high-signal workflows for development and regression work:

1. **Canonical decompile smoke test**
   - `cargo run -- -i test_cu/if_loop.sass`
   - Verifies the real public path: decode -> CFG -> SSA -> optimize -> structurize -> lift -> name -> typed render.
2. **CFG-first debugging**
   - `cargo run -- -i test_cu/if_loop.sass --cfg-dot > /tmp/if_loop.cfg.dot`
   - Use when recovered control flow looks wrong; inspect leader splits and successor edges before touching the structurizer.
3. **SSA/lift debugging**
   - `cargo run -- -i test_cu/if_loop.sass --ssa-dot > /tmp/if_loop.ssa.dot`
   - Use when compare lifting, pointer recovery, or predicate cleanup looks wrong.
4. **Architecture-specific ABI checks**
   - `cargo run -- -i kernel.sass --abi-profile legacy140`
   - `cargo run -- -i kernel.sass --abi-profile modern160`
   - Useful when a dump lacks trustworthy SM metadata and you need to compare inferred signatures.
5. **Snapshot refresh after intentional backend changes**
   - `cargo run --example regen_goldens`
   - Only do this after `cargo test` is green and the canonical output change is intentional.
