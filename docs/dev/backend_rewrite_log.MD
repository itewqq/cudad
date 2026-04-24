# Backend Rewrite Log

## Status

- Branch: `codex/memory-aware-backend-refactor`
- Checkpoint commit: `f464db3`
- Current stage: phase 3 early AST lowering on top of memory-aware analysis

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

1. Replace the remaining placeholder non-memory AST lowering with semantic
   lowering that no longer depends on the old semantic-lift backend.
2. Switch CLI/tests/examples to the canonical driver once the new path covers
   enough structured statements.
3. Delete duplicated full-pass driver logic and transitional tests after the
   new path is used by default.

## Open questions

- How much of the current AST can be kept unchanged versus extended.
- Which current SSA passes need semantic updates once explicit memory ops land.
- How much of the existing ABI logic can be lifted directly into
  `FunctionAnalysis`.

## Files/modules touched

- `docs/dev/backend_rewrite.MD`
- `docs/dev/backend_rewrite_log.MD`
- `src/backend_pipeline.rs`
- `src/function_analysis.rs`
- `src/ast_lowering.rs`
- `src/memory_model.rs`
- `src/symbol_plan.rs`
- `src/lib.rs`

## Interrupt resume note

Resume from the next unchecked atomic commit. Do not start deleting the old
backend until the new canonical pipeline stages exist and have stage-local
tests.
