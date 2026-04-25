# Backend Rewrite Log

## Status

- Branch: `codex/memory-aware-backend-refactor`
- Checkpoint commit: `3c01801`
- Current stage: phase 6 acceptance/perf cleanup, with the canonical backend already live and the remaining work focused on broader validation plus the last legacy deletion sweep

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
- `FunctionAnalysis` now recovers loop-carried pointer-phi byte offsets from
  companion scalar index phis, so rooted `IMAD.WIDE` address recurrences keep
  their byte stride through optimized SSA instead of collapsing back to raw
  packed pointer arithmetic.
- Canonical AST lowering now recognizes rooted 64-bit pointer pairs and lowers
  them as `WidePtr` expressions instead of anonymous `Addr64` packs, which
  keeps rooted global parameter bases explicit through wide-address arithmetic
  and fixes the main `dot_thread` pointer-param regression.
- Canonical AST cleanup now owns FCHK-guarded reciprocal/division recovery for
  `fmaf`-shaped Newton refinement trees, and it prunes dead helper predicates
  inside rewritten structured blocks before symbol planning.
- The canonical `dot_thread` recurrence contract now accepts the deliberate
  `fmaf(...)` spelling as equivalent to `a * b + c`, so the test suite checks
  for preserved FFMA structure rather than an obsolete render-only spelling.
- Canonical cleanup now intentionally refuses to synthesize `/` from a
  guarded Newton-like polynomial unless the reciprocal seed can still be
  proven structurally; correctness wins over helper-elimination coverage.
- Companion-phi pointer-offset recovery now preserves any proven entry bias
  instead of collapsing every loop-carried pointer phi to `idx * stride`.
- Memory-access classification no longer falls back to the low 32-bit lane of
  a mismatched `Addr64` pair, which avoids misclassifying stale mixed-width
  addresses as rooted param/shared/local accesses.
- The `dot_thread` regression test again requires the loop-carried induction
  state to survive structurization, which protects the canonical path from
  silently deleting accumulator/index updates inside remainder loops.
- Canonical global-index lowering now recursively expands rooted byte-offset
  expressions before rendering, which collapses nested rooted temps out of
  typed global loads/stores instead of leaking intermediate offset registers
- Canonical AST cleanup now runs a bounded local sweep instead of a single
  one-shot pass, which lets newly recovered reciprocal/division expressions
  expose and then drop stale `FCHK` predicate temps without reintroducing the
  old text-repair pipeline.
- Canonical AST cleanup now folds constant predicate branches even when the
  surrounding function still contains `goto`/label barriers, and it restricts
  dead-code cleanup to structured chunks between those barriers instead of
  skipping the whole function.
- The targeted helper-leak regressions are now pinned directly:
  `power_series` must drop loop-carried `FCHK` leftovers after reciprocal
  recovery, and `box_blur_variable_radius` must fold its constant-false
  helper branch before the remaining goto tail.
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
- Remaining legacy helper usage in `src/test.rs`: `16`.
- `LOP3.LUT` instructions with both predicate and data outputs now keep both
  defs on the canonical path, which fixes `warp_reduce_sum` lane-mask recovery
  (`threadIdx.x & 31`) and removes the old undefined `r8_0` artifact from the
  canonical warp-reduction output.
- Canonical scalar lowering now recognizes the common `LOP3.LUT(..., 0xc0, !PT)`
  mask pattern as native bitwise `&`, and canonical statement lowering emits
  the companion predicate def as a structural `!= 0` assignment when the SASS
  instruction writes both a predicate and a data register.
- `FunctionAnalysis` now also carries SSA-versioned scalar type facts, so the
  canonical backend can recover typed scalar semantics without reviving
  semantic-lift side tables or rendered-text heuristics.
- Canonical scalar lowering now covers `FSEL`/`SEL`, `FMUL`, plain
  `IMAD`/`UIMAD`, `IMAD.MOV`, `IMAD.IADD`, and `IMAD.SHL.U32` structurally,
  which removes another band of raw helper-shaped arithmetic from kernels such
  as `gelu_forward`.
- Canonical `LOP3.LUT` lowering now also recognizes the float sign-injection
  patterns used for `copysignf(...)`, and canonical coverage now pins that
  `gelu_forward` keeps those math ops structural instead of leaking raw
  `LOP3.LUT` / `FSEL` / `FMUL` / `IMAD` helper calls.
- The duplicated old-backend `sgemm_tiled` full-pass tests are now deleted; the
  canonical `sgemm_tiled` coverage owns the typed predicated-load and
  LEA-helper-elimination contracts directly.
- Canonical predicate lowering now understands unordered compare spellings such
  as `FSETP.GEU.*`, which removes one more raw control-helper leak from the
  canonical `layer_norm_forward` path and gives the canonical backend ownership
  of that compare contract.
- Canonical scalar lowering now normalizes zero-operand predicate/float
  pseudo-ops (`PT`, `!PT`, `UPT`, `!UPT`, `+INF`, `-INF`, `+QNAN`, `-QNAN`)
  into structural constants, which removes helper-shaped leaks like `PT()` and
  `+QNAN()` from canonical control-flow output.
- Canonical targeted coverage now also pins the control-flow corpus for this
  cleanup: `state_machine` no longer leaks raw predicate pseudo-ops, and
  `dispatch_ops` lowers `+QNAN` to `NAN`.
- Canonical scalar lowering now also lowers `IADD3.X` / `UIADD3.X` carry-chain
  expressions structurally, which removes another raw helper-shaped arithmetic
  leak from canonical control-flow output (`state_machine` now renders
  `ur7_2 + (up0_2 ? 1 : 0)` instead of `UIADD3.X(...)`).
- Canonical scalar lowering now also models the common float modifier surface
  that still leaked through the canonical path: `FRND.FLOOR`/`FRND.TRUNC`
  lower to `floorf(...)` / `truncf(...)`, `FADD.FTZ` / `FMUL.FTZ` now stay
  explicit through canonical FTZ helper calls instead of silently erasing the
  modifier, and modeled `FFMA` variants (`.SAT`, `.RM`, `.RP`, `.RZ`, `.RN`)
  lower to structural CUDA math intrinsics such as `__saturatef(...)` /
  `__fmaf_rd(...)` instead of raw helper-shaped calls.
- Canonical float compare lowering now preserves unordered predicate semantics
  instead of treating `GEU`/`GTU`/`LEU`/`LTU`/`EQU`/`NEU` as ordered compares;
  the canonical path emits explicit `isnan(...)` disjuncts and also lowers
  `NUM` / `NAN` predicate tokens structurally.
- Canonical scalar/address lowering now also covers `IMAD.X` carry-add forms
  and non-`.HI` `LEA` address expressions structurally; multi-def `LEA`
  statements now keep both the data result and the carry predicate on the
  canonical path instead of leaking raw `LEA(...)` / `IMAD.X(...)` helpers.
- Sub-agent review fixes are now folded back into the canonical lowering
  contract: predicated multi-def `LEA`/`LOP3` statements preserve false-path
  SSA old-def values for every written result, unqualified `FFMA` now lowers
  through fused `fmaf(...)` instead of a decomposed multiply-plus-add, and
  `FFMA.SAT` saturates that fused form rather than a two-rounding expression.
- `FunctionAnalysis` now propagates loop-carried global pointer roots even when
  the relative byte offset is not yet recoverable, and canonical global
  lowering falls back to root-relative `uintptr_t` index expressions instead
  of raw register-backed pointer casts. This fixes the `cumsum_linear`
  remainder paths without reintroducing text-driven recovery.
- Canonical scalar lowering now also handles low-lane `IMAD.WIDE` /
  `UIMAD.WIDE` expressions structurally, which removes another raw helper leak
  from the canonical path and keeps widened pointer setup readable in kernels
  like `cumsum_linear`.
- Canonical scalar lowering now also handles the common high-lane address
  helper surface structurally: `LEA.HI` / `LEA.HI.X(.SX32)` lower to explicit
  high-word arithmetic, `SHF.L.U64.HI` lowers to a lane extract of a widened
  shift, and trivial `PLOP3.LUT(PT, PT, PT, PT, imm, 0)` predicates collapse
  to `false`. This clears those helper leaks from `layer_norm_forward` and
  `softmax_forward`, and shrinks the remaining `topk_per_row` cleanup down to
  pointer-root recovery rather than opcode spelling.
- Canonical scalar lowering now also covers the remaining `multi_exit_loop`
  scalar-helper surface (`I2F.RP`, `F2I.FTZ.U32.TRUNC.NTZ`, `IMAD.HI.U32`),
  and `FunctionAnalysis` now uses a low-lane-root fallback for inconsistent
  `Addr64` pairs when the full wide-root proof is missing. That keeps the old
  `topk_per_row` remainder path rooted on `arg0_ptr` instead of falling back
  to anonymous `uint32_t*` casts.
- Canonical targeted coverage now also pins this float-modifier slice:
  `dispatch_ops` no longer leaks raw `FRND.FLOOR(...)` / `FMUL.FTZ(...)`, and
  `softmax_forward` no longer leaks raw `FFMA.SAT(...)` / `FFMA.RM(...)` /
  `FADD.FTZ(...)` on the canonical path.
- Canonical targeted coverage now also pins `topk_per_row` against this new
  helper slice: the canonical path must avoid raw `IMAD.X(...)` and plain
  `LEA(...)` leakage while keeping the typed `arg2_ptr` / `arg4_ptr` contract.
- The first sub-agent review findings from the scalar-helper cleanup are now
  addressed in the live branch: FTZ float ops lower through documented CUDA
  rounding intrinsics instead of invented `*_ftz` helper names, and
  `IMAD.WIDE` / `UIMAD.WIDE` now build an explicitly widened 64-bit value
  before extracting lanes so signedness and overflow semantics are preserved.
- Canonical AST tests now also pin predicated multi-def `IMAD.WIDE` lowering,
  including false-path SSA preservation for both low/high written lanes.
- The last obvious scalar/helper leak in the tracked corpus is now gone:
  `state_machine` lowers indirect `BRX` control to an explicit switch-shaped
  placeholder instead of emitting raw `BRX(...)` text in the canonical output.
- The corpus invariant runner and curated golden regeneration entrypoint are
  now pointed at the canonical backend driver, which removes another old
  full-pass call site from the branch and makes the remaining legacy-test
  cleanup more mechanical.
- Canonical AST DCE now treats synchronization intrinsics as effectful
  statements, so `__syncthreads()` / `__syncwarp()` survive the late pure-call
  pruning pass instead of disappearing from rendered kernels such as
  `histogram256`.
- The remaining old lifted/full-pass helper surface in `src/test.rs` is now
  hard-deleted: the legacy preview/full-pass helpers and their RC4/name-
  recovery/text-repair contract tests are gone instead of being kept as
  compatibility coverage.
- Replacement coverage now stays on-rail with the rewrite plan:
  - tolerant decode + SSA explicitly keep unknown opcodes representable
  - canonical sha256 coverage pins explicit local-memory intrinsics plus
    `arg2_ptr[...]` digest stores
  - cumsum / gelu / layer-norm coverage is canonical-only for the surviving
    contracts instead of routing back through the deleted post-struct path
- `src/test.rs` no longer references `plan_structured_name_recovery_with_lift`,
  `render_typed_structured_output`, `collect_shared_memory_decls`, or the old
  full-pass test driver; the remaining legacy deletion work is now in the
  production modules rather than the test harness.
- The legacy text-repair modules are now off the live library surface:
  `src/lib.rs` no longer exports `name_recovery` or `typed_output`, and the
  corresponding source files are deleted instead of left behind as dormant
  compatibility code.
- The old structurizer-side cleanup/token-map bridge is also deleted:
  `pretty_print_with_lift_cleanup*` and the seeded-wide-addr/token-alias helper
  machinery are gone, and the structurizer tests no longer pin those obsolete
  post-struct rendering contracts.
- The last semantic-lift production coupling is gone from the tree entirely:
  `structurizer` no longer reaches into lift side tables, `src/lib.rs` no
  longer exports the old lift surface, and the dormant `src/semantic_lift.rs`
  sources plus their rule tables are deleted instead of hanging around as
  pseudo-reference code.
- The review follow-up for this slice is also addressed: `README.md` and
  `docs/dev/decompiler_design.MD` now describe the live canonical pipeline as
  SSA optimization -> `FunctionAnalysis` -> structurization -> AST lowering ->
  symbol planning -> render without any remaining semantic-lift side path.
- Canonical structured lowering now owns structural phi materialization instead
  of leaking raw `phi(...)` helpers from SSA blocks: condition-block preludes,
  edge phi connectors, loop-entry seeds, loop backedge updates, and jump-target
  labels are emitted directly in `src/ast_lowering.rs` on the live backend
  path.
- CFG construction now rewrites predicated non-returning `CALL.REL.NOINC`
  helper slowpaths into ordinary conditional edges before block partitioning,
  so structurization can reason about those helpers as control flow instead of
  opaque call opcodes.
- Structured AST lowering now also materializes phi connector chains on
  structured `if` entry edges, which keeps helper-entry SSA values defined
  before the lowered branch body consumes them.
- Structurizer node-splitting now allows duplication of short acyclic helper
  composites (up to 6 constituent blocks) while still refusing loop/goto
  composites, which is enough to recover the shared `box_blur_variable_radius`
  slowpath without regressing `power_series`.
- Structurizer unit coverage now pins that shared terminal helper chains use
  node splitting instead of goto fallback once the helper tail is small,
  acyclic, and return-terminated.
- Review follow-up fixes are now landed for the helper-slowpath slice:
  branch-entry phi preludes are injected inside entry labels so surviving gotos
  cannot skip them, and the CFG helper rewrite now rejects unknown opcode
  classes while requiring explicit lexical nested-return setup before treating
  an internal `CALL.REL.NOINC` chain as inline control flow.
- Canonical local cleanup continues to treat raw `CALL.REL.NOINC` sites as
  pure helper placeholders; the live contract is to eliminate them by control-
  flow recovery and reciprocal/division simplification rather than by pinning
  them as effectful AST calls.
- Canonical AST cleanup now gates reciprocal/division recovery on a direct
  in-scope `FCHK` witness and resolves local name aliases with cycle guards, so
  large non-reciprocal kernels like `sha256_single_block` no longer spend the
  cleanup pass recursively inflating loop-carried expressions that have no
  reciprocal proof to recover.
- The previously timing-out `sha256_single_block` corpus slice now completes on
  the live CLI across SM 89 / SM 100 / SM 120, and the structured-helper-leak
  corpus invariants plus SM 89 determinism all pass again after that cleanup
  guard landed.
- The legacy semantic-lift harness is now deleted rather than merely
  disconnected: the remaining `src/test.rs` checks that only asserted
  lift-specific pretty-print details are gone, and there is no dormant
  `semantic_lift` module left to silently fall out of coverage.
- The broader acceptance refresh is moving again on the branch tip after that
  cleanup: SM 100 and SM 120 corpus determinism both pass on the canonical
  backend, so the remaining corpus refresh work is narrower (`goto` budgets
  plus any final curated-output regeneration) instead of being blocked on the
  old lift harness or the old SHA timeout.
- The obsolete `src/ast_passes/tests.rs` cleanup gauntlet is now cut back to
  only the live `ast_simplify` contracts. The deleted cases were asserting the
  old pre-canonical addr64 / helper-cleanup / FCHK-repair pipeline rather than
  the canonical backend contract owned by `src/canonical_ast_passes.rs`.

## Next safe step

1. Re-run the remaining broader acceptance slices that still matter after the
   `sha256_single_block` perf fix (mainly the corpus `goto` budgets and any
   corpus-wide invariants that have not yet been refreshed on this branch tip).
2. Continue the final phase 6 deletion sweep: remove any remaining legacy-only
   backend helpers/tests that no longer back the canonical pipeline, then
   regenerate the curated acceptance outputs from the canonical driver only.
3. Once another 2-4 atomic commits land, run the next review sub-agent and
   wait for the result before starting the final corpus refresh / PR cleanup
   pass.

## Open questions

- Whether the new acyclic helper-duplication bound (`<= 6` blocks, no
  loops/gotos) is enough for the remaining corpus slowpaths, or whether one
  more structurizer-side heuristic is still needed for a few larger shared
  helper tails.
- Whether the remaining packed-pointer heavy kernels (`sobel_edge_detect`,
  broader affine `layer_norm_forward` slices, and any still-temp-heavy SHA
  idioms beyond the fixed `sha256_single_block` timeout) need another canonical
  lowering pass before their old smoke tests can be replaced one-for-one
  instead of simply deleted.
- How aggressively the public `lib.rs` surface should shrink in the same
  cutover versus a final follow-up cleanup commit after all internal callers
  are gone.

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
- `src/canonical_ast_passes.rs`
- `src/cfg.rs`
- `src/structurizer/mod.rs`
- `src/structurizer/collapse.rs`
- `src/structurizer/tests.rs`
- `README.md`
- `docs/dev/decompiler_design.MD`
- `src/ast_passes/tests.rs`

## Interrupt resume note

Resume from the post-`sha256_single_block` perf fix: canonical cleanup is now
bounded on non-reciprocal kernels again, so the next safe work is the last
legacy semantic-lift deletion sweep plus the remaining corpus acceptance
refresh while keeping the canonical helper-leak regressions green after each
atomic commit.
