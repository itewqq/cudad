# Memory-Aware Backend Rewrite

This document is the contract for the backend refactor that replaces the
string-repair post-struct pipeline with a memory-aware SSA analysis and a
structured AST lowering pipeline.

## Canonical pipeline

The target backend is:

1. tolerant decode
2. CFG construction
3. SSA construction
4. SSA optimization
5. `FunctionAnalysis`
6. structurization
7. structured AST lowering
8. symbol/declaration planning
9. rendering

Only one full-pass driver is allowed once the cutover is complete. The CLI,
tests, and golden regeneration must all use that driver.

The canonical driver must also preserve decoded function identity; multi-
function dumps are not allowed to collapse every rendered function back to
`kernel`.

## Front-half contract

- The parser must stay tolerant and lossless.
- Memory-bearing operand modifiers such as address width/scale (`.64`, `.X4`,
  `.X8`, `.X16`) must survive decode and IR lowering as structured data, not
  only as raw strings.
- CFG stays control-flow-only.
- CFG may rewrite predicated non-returning helper calls into conditional
  branch edges when a conservative forward walk over inline-safe opcodes,
  forward-only branches, and explicit lexical nested-return setup shows the
  helper exits the kernel instead of returning to the caller; that rewrite is
  still control-flow modeling, not semantic lifting.
- SSA stays the canonical mid-level control/dataflow representation.
- Memory-space information must become explicit before AST lowering.

## Memory model contract

The core CUDA memory spaces for this rewrite are:

- `Param`
- `Const`
- `Global`
- `Local`
- `Shared`
- `Generic`

Texture, surface, sampler, and other special engines are out of scope for the
first cutover and should remain explicit unsupported/special forms.

Shared/local declaration planning must not invent fixed extents when analysis
only proves a dynamic rooted offset. In those cases the AST must preserve that
the backing object has unresolved extent instead of pretending it is a small
constant array.

## Quality bar

- No post-render text parsing for semantics.
- No regex-driven semantic reconstruction.
- Unknown SASS stays representable and conservative.
- Multi-def scalar ops must keep their non-sink data results; optional
  predicate/sink defs (for example on `SHFL`) must not cause the canonical path
  to drop the real value-producing def.
- Optional predicate+data scalar outputs (for example `LOP3.LUT P0, R8, ...`)
  must preserve both results structurally; the canonical path must not turn the
  data def into an undefined use or silently drop the predicate fact.
- `FunctionAnalysis` owns post-SSA scalar type facts as well as memory-space
  facts, so canonical lowering can recover typed scalar semantics such as
  `copysignf(...)` / `FSEL` / `IMAD` directly from SSA instead of leaning on
  semantic-lift text helpers.
- Predicate compare lowering must understand the real compare-token surface
  (`EQ`/`EQU`, `NE`/`NEU`, `LT`/`LTU`, `GE`/`GEU`, etc.) so later structured
  stages do not leak raw `FSETP.*` helpers just because the opcode used an
  unordered token spelling.
- Canonical lowering must also normalize zero-operand pseudo-ops such as
  `PT`/`!PT` and `+INF`/`+QNAN` structurally instead of leaking helper-shaped
  call syntax like `PT()` or `+QNAN()` into rendered CUDA.
- Canonical scalar lowering should treat common carry-chain arithmetic forms
  (`IADD3.X` / `UIADD3.X`) as real arithmetic on the surviving data result
  instead of leaving raw helper mnemonics in structured output.
- Canonical scalar lowering should also model the common float modifier surface
  structurally where CUDA has a direct semantic spelling (`FRND.*`,
  `FADD.FTZ`, `FMUL.*`, `FFMA.SAT`, `FFMA.R*`) instead of leaking raw opcode
  helper calls for routine math code.
- Canonical address/arithmetic lowering should also cover carry-aware helper
  forms such as non-`.HI` `LEA` and `IMAD.X` structurally, preserving any
  predicate carry defs without dropping back to raw opcode-shaped helper calls.
- Structurization may duplicate only short acyclic helper tails to recover a
  reducible CFG shape; loop-bearing or goto-bearing composites must not be
  duplicated just to paper over irreducible control flow.
- Canonical compare lowering must preserve unordered floating-point predicate
  semantics: `*U` compare tokens and `NUM`/`NAN` forms need explicit NaN-aware
  logic, not ordered C relational operators that silently change branch
  behavior on NaN inputs.
- Canonical AST cleanup must stay proof-driven and bounded: reciprocal/division
  recovery only runs when a direct `FCHK` witness is still live in the local
  statement region, and local name resolution must stop at self/mutual cycles
  in loop-carried defs instead of recursively inflating the expression tree.
- New backend modules must document their inputs, outputs, invariants, and
  forbidden responsibilities in rustdoc headers.

## Canonical identifier contract

- New backend AST lowering must emit legal C/CUDA identifiers directly.
- SSA register names are normalized structurally from `(class, idx, ssa)`; the
  renderer must not clean up raw `R4.0`-style names after the fact.
- Predicate registers must stay recognizable after renaming so symbol planning
  can type them as `bool`.

## Deletion bar

The old post-struct backend is removed at cutover. That includes:

- semantic-lift-as-render-backend behavior
- rendered-text token rewriting
- regex declaration synthesis
- post-render pointer repair
- fixed-point cleanup loops used as backend repair
- duplicated full-pass drivers

Only logic that remains valid after translation into decode, SSA analysis, AST
lowering, or rendering should survive.
