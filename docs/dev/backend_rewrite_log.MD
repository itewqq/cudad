# Backend Rewrite Log

## Status

- Branch: `codex/memory-aware-backend-refactor`
- Checkpoint commit: `db0e02e`
- Current stage: phase 5 cutover of structured output entrypoints onto the canonical backend, with front-half address-contract fixes still landing as needed

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
- Canonical scalar lowering now materializes:
  - `S2R`/`CS2R`/`S2UR` builtin-register reads
  - `ISETP`/`UISETP` comparison predicates
  - unmodified `FFMA`
  - unmodified `FMNMX`
  - `MUFU` transcendental helpers
  before falling back to raw opcode-shaped helpers; modifier-bearing scalar ops
  stay explicit until their semantics are modeled.
- Pointer-root propagation now ignores predicate/carry defs and other
  non-address registers, which prevents non-convergent root worklists on
  Blackwell carry-heavy loops while keeping real `R`/`UR` address facts.
- Canonical address recovery now propagates rooted `IADD.64` pairs and lowers
  `Addr64` AST nodes structurally, which removes raw `addr64(...)` helper leaks
  from the canonical path and lets more wide-addressed loads/stores render
  directly from analysis facts.
- `FunctionAnalysis` now propagates roots through `UIADD3.64` / `IADD3.64`
  wide-add chains, so canonical lowering can keep more Blackwell pointer loops
  attached to typed param roots instead of falling back to anonymous register
  bases.
- Canonical global-index lowering now recursively expands rooted byte-offset
  expressions before rendering, which collapses nested rooted temps out of
  typed global loads/stores instead of leaking intermediate offset registers
  into the rendered index.
- Canonical scalar lowering now lowers low-lane results of `IADD.64` and
  `UIADD3.64` structurally, which removes another class of raw wide-add helper
  leaks even in functions where structurization still leaves `goto` blocks and
  the canonical dead-temp pass stays conservative.
- Canonical scalar lowering now resolves scalar `ConstMem(bank, off)` operands
  through the ABI/profile fact base, so non-memory ops can render param aliases
  and builtin symbols instead of leaking raw `ConstMem(...)` calls.
- Canonical scalar lowering now handles `IABS` and `MUFU.RCP` semantically,
  which restores `abs(...)` / `rcp_approx(...)` rendering on the canonical path
  without reviving semantic-lift side tables.
- Canonical memory lowering now wraps unresolved register-backed global/generic
  bases in explicit typed pointer casts before indexing, which removes raw
  pseudo-pointer forms like `r2_5[0]` from canonical output while preserving
  unresolved pointer semantics structurally.
- Canonical AST cleanup now prunes structurally empty `if` shells after local
  dead-code cleanup, which removes uninitialized-predicate scaffolding from
  trivial kernels like `if.sass` without reviving the old whole-function repair
  pipeline.
- `FunctionAnalysis` now treats `UIMAD.WIDE` as pointer arithmetic, so
  uniform-rooted address loops keep their param provenance instead of degrading
  into anonymous register-backed pointer casts on the canonical path.
- Shared-memory declaration planning now carries the pointee type recovered
  from SSA/shared round-trips into canonical symbol planning, so canonical
  `shmem` backing objects can render as `float` instead of a hard-coded
  `uint32_t` storage type when the dataflow proves the element type.
- The canonical AST now runs a local dead-pure-definition pass before symbol
  planning, so analysis-inlined addresses can delete now-unused helper temps
  without reviving the old fixed-point cleanup pipeline.
- Early AST lowering can now emit space-aware AST for:
  - global param-rooted loads
  - shared/local indexed stores
  - shared atomics
- Canonical full-pass test coverage now owns:
  - `if.sass`
  - `test_div.sass`
  - `dot_thread`
  - `nbody_forces`
- Canonical targeted regression tests now also pin:
  - `cumsum_linear` arg-rooted uniform tail-loop loads/stores
  - `sgemm_tiled` predicated typed global loads
  - `sgemm_tiled` absence of `lea_hi_x` / split-pointer helper artifacts
- The canonical backend driver now builds:
  - optimized SSA
  - `FunctionAnalysis`
  - structured control flow
  - an early structured AST
  - rendered output from that AST
- The tolerant decode / SSA front half now has an explicit contract to keep
  shared/global address scale modifiers (`.X4`, `.X8`, `.X16`) structured
  instead of erasing them before analysis/lowering sees the memory operand.
- `FunctionAnalysis` now also recovers special-register builtin values (for
  example `threadIdx.x`) through simple SSA copy chains, so canonical AST
  lowering can substitute builtin roots directly instead of keeping raw
  register temps alive in obvious address expressions.
- Canonical shared-memory index lowering can now consume scaled thread-index
  addresses from the new decode/IR contract, which makes fixtures like
  `reduce_block` and `stencil1d` render `shmem[threadIdx.x + k]` forms on the
  canonical path instead of leaking `(rN_0 + byte_offset) / 4`.
- Canonical non-memory lowering now selects the first non-sink result def
  instead of blindly taking operand 0, which keeps real data defs alive for
  multi-def ops like `SHFL.DOWN PT, R0, ...` instead of silently assigning the
  sink predicate and dropping the shuffle result.
- Canonical scalar lowering now maps `SHFL.DOWN`/`SHFL.UP`/`SHFL.XOR` style ops
  to CUDA shuffle intrinsics, so warp reductions can keep `__shfl_*_sync`
  structure on the canonical path without depending on semantic-lift output.
- Canonical scalar lowering now folds common `SHF*.R.*.HI` right-shift patterns
  into native `>>` expressions, which makes warp/shared lane-group arithmetic
  readable and prevents raw funnel-shift helpers from leaking into the
  canonical path when the source is a known builtin like `threadIdx.x` or
  `blockDim.x`.
- Canonical targeted coverage now also pins `warp_reduce_sum` shuffle lowering,
  and the broader float-pointer/signature smoke for `warp_reduce_sum` now runs
  on the canonical helper instead of the legacy structured-output path.
- Remaining legacy helper usage in `src/test.rs`: `18`.

## Next safe step

1. Re-run the canonical full-pass fixtures whose remaining gaps are now mostly
   structural rather than scalar-expression loss (`warp_reduce_sum`, then the
   still-partial `sgemm_tiled` / shared-memory kernels) and migrate only the
   assertions that the canonical backend now proves directly.
2. Keep shrinking the remaining old helper surface in `src/test.rs` only where
   canonical semantics are now genuinely stronger, not just cosmetically
   different.
3. Tackle the next structurally visible canonical holes: shared index/root
   recovery beyond direct `threadIdx` forms, unresolved control predicates, and
   the remaining unmodeled scalar ops (notably `LOP3.LUT` predicate shaping)
   that still force raw helper-shaped output.

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
- `src/test.rs`
- `src/function_analysis.rs`

## Interrupt resume note

Resume from the next unchecked atomic commit. Do not start deleting the old
backend until the new canonical pipeline stages exist and have stage-local
tests.
