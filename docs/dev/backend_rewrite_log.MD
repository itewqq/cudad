# Backend Rewrite Log

## Status

- Branch: `codex/memory-aware-backend-refactor`
- Checkpoint commit: `f464db3`
- Current stage: phase 5 cutover of structured output entrypoints onto the canonical backend

## Landed invariants

- A pre-refactor checkpoint commit exists.
- The refactor is expected to proceed in atomic commits.
- Periodic sub-agent review should run after every 3 implementation commits.
- The canonical backend scaffolding now has dedicated modules for:
  - pipeline orchestration
  - memory model vocabulary
  - function analysis
  - AST lowering
  - symbol planning
- `FunctionAnalysis` now classifies memory accesses by CUDA space and carries
  propagated address-root facts.
- Canonical AST lowering now emits C/CUDA-safe SSA identifiers instead of raw
  `R4.0` / `P0.0` register displays.
- Predicated single-def statements now preserve their false-path value in the
  structured AST instead of dropping the SSA old-def edge.
- Atomic statements now preserve return-value defs when the opcode writes back
  the old memory value.
- Param-window `LDC` loads now lower to parameter symbols instead of falling
  back to raw `ConstMem(...)` text.
- Shared/local backing declarations now use deterministic backing objects when
  the accessed extent is proven; unresolved local-space cases still require
  explicit helper lowering instead of a declaration-backed object.
- Address-root propagation now keeps provenance through common `IADD`/`IADD3`
  pointer arithmetic, including split pointer increments before `LDG`.
- AST lowering now keeps byte-offset expressions for rooted global/shared
  accesses instead of dropping the dynamic address component.
- Non-memory AST lowering now folds trivial `MOV` and add-family ops into
  native AST expressions instead of rendering every scalar op as a call-like
  helper.
- Scalar IR expressions now lower simple unary/binary operators directly into
  AST operators, which reduces `+(a, b)` / `*(a, b)` helper leakage in the
  canonical path.
- Shared/local declaration planning now avoids fake fixed extents for dynamic
  offsets, but wide/subword declaration planning is still intentionally kept
  conservative around the shared/local backing model.
- The renderer now preserves byte-pointer widening for complex rooted offsets
  instead of collapsing them back into element indices too early.
- The canonical backend driver can now render with an explicit function name,
  which will be required when the CLI/test entrypoints switch over.
- Forced ABI-profile selection can now be threaded through the canonical
  backend driver, which keeps the CLI cutover compatible with `--abi-profile`.
- The CLI structured-output path now goes through the canonical backend driver
  and ABI summary facts from `FunctionAnalysis` instead of the old lift/name/
  cleanup pipeline.
- The first curated corpus acceptance check (`histogram256` shared atomics on
  SM89) now runs directly against the canonical driver instead of the old
  structured-output helper path.
- Canonical backend tests now enforce:
  - semantic pointer-param names (`argN_ptr`)
  - no raw SSA token leaks
  - dynamic shared-index rendering
  - named entrypoint rendering
- Canonical test helpers now fail hard when the new backend does not render,
  instead of silently substituting a fake `kernel` stub.
- Dynamic local-memory accesses now stay explicit unresolved operations instead
  of degrading into invalid local declarations or fake pointer variables.
- Canonical lowering now drops structurization-owned control skeleton ops
  (`BRA`, `BSYNC`, `BSSY`, etc.), lowers `EXIT`/`RET` to structured returns,
  and maps `BAR.SYNC` to `__syncthreads()`.
- Shared `ATOMS.POPC.INC` lowering now emits `atomicAdd(..., 1)` instead of a
  raw fallback helper call.
- Simple scalar lowering now only rewrites unmodified opcodes; modifier-bearing
  add-family instructions stay explicit until their semantics are modeled.
- Early AST lowering can now emit space-aware AST for:
  - global param-rooted loads
  - shared/local indexed stores
  - shared atomics
- The canonical backend driver now builds:
  - optimized SSA
  - `FunctionAnalysis`
  - structured control flow
  - an early structured AST
  - rendered output from that AST

## Next safe step

1. Switch the remaining curated full-pass tests and helper entrypoints from the
   old structured-output path onto the canonical driver, starting with the
   corpus checks that only assert output-shape contracts.
2. Delete duplicated CLI/test helper logic that still builds semantic-lift,
   token-map, or post-render cleanup artifacts after the canonical path is
   proven on those fixtures.
3. Replace the remaining placeholder non-memory AST lowering with semantic
   lowering that no longer depends on the old semantic-lift backend.

## Open questions

- How much of the current AST can be kept unchanged versus extended.
- Which current SSA passes need semantic updates once explicit memory ops land.
- How much of the existing ABI logic can be lifted directly into
  `FunctionAnalysis`.
- What the right canonical AST representation is for multi-def memory ops
  (`LDC.64`, vector loads, etc.) before the big-bang cutover.

## Files/modules touched

- `docs/dev/backend_rewrite.MD`
- `docs/dev/backend_rewrite_log.MD`
- `src/backend_pipeline.rs`
- `src/backend_names.rs`
- `src/function_analysis.rs`
- `src/ast_lowering.rs`
- `src/ast.rs`
- `src/memory_model.rs`
- `src/symbol_plan.rs`
- `src/bin/main.rs`
- `src/lib.rs`

## Interrupt resume note

Resume from the next unchecked atomic commit. Do not start deleting the old
backend until the new canonical pipeline stages exist and have stage-local
tests.
