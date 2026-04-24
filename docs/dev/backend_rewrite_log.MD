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

## Next safe step

1. Expand AST lowering from memory-only helpers into full structured statement
   lowering.
2. Introduce deterministic symbol/declaration planning driven by AST plus
   analysis facts.
3. Start switching the canonical driver away from semantic lift and text repair.

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
