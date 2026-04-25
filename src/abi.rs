//! ABI profile and constant-memory parameter mapping for SASS decompilation.
//! This module is intentionally conservative: it maps well-known offsets to
//! symbolic names and falls back to raw `ConstMem(bank, off)` semantics.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{DisplayCtx, FunctionIR, IRExpr, IRStatement, RValue, RegId};
use crate::parser::{DecodedInstruction, DecodedOperand};
use crate::type_inference::{
    infer_arg_types_from_opcode, infer_ssa_types, infer_types, InferredType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiGeneration {
    /// Common older layout with parameter window beginning near c[0x0][0x140].
    LegacyParam140,
    /// Common newer layout with parameter window beginning near c[0x0][0x160].
    ModernParam160,
    /// Blackwell (SM 100+) layout: parameter window starts at c[0x0][0x380],
    /// built-in thread/block dimensions live at c[0x0][0x360+], and the
    /// descriptor register source sits at c[0x0][0x358].
    BlackwellParam380,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiProfile {
    pub generation: AbiGeneration,
    pub const_bank: u32,
    pub param_base: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstMemKind {
    Builtin(&'static str),
    ParamWord { param_idx: u32, word_idx: u32 },
    AbiInternal(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedConstMem {
    pub symbol: String,
    pub kind: ConstMemKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstMemSemantic {
    Builtin(&'static str),
    ParamWord { param_idx: u32, word_idx: u32 },
    AbiInternal(u32),
    Unknown { bank: u32, offset: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StatementRef {
    pub block_id: usize,
    pub stmt_idx: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstMemAnnotation {
    pub bank: u32,
    pub offset: u32,
    pub semantic: ConstMemSemantic,
}

impl ConstMemAnnotation {
    pub fn symbol(&self) -> String {
        match &self.semantic {
            ConstMemSemantic::Builtin(name) => (*name).to_string(),
            ConstMemSemantic::ParamWord { param_idx, .. } => {
                format!("param_{}", param_idx)
            }
            ConstMemSemantic::AbiInternal(offset) => format!("abi_internal_0x{:x}", offset),
            ConstMemSemantic::Unknown { bank, offset } => {
                format!("c[0x{:x}][0x{:x}]", bank, offset)
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbiAnnotations {
    pub constmem_by_stmt: BTreeMap<StatementRef, Vec<ConstMemAnnotation>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasConfidence {
    Low,
    Medium,
    High,
}

impl AliasConfidence {
    pub fn as_str(self) -> &'static str {
        match self {
            AliasConfidence::Low => "low",
            AliasConfidence::Medium => "medium",
            AliasConfidence::High => "high",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgAliasKind {
    Word32,
    U64,
    Ptr64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgScalarKind {
    U32,
    I32,
    F32,
}

impl ArgScalarKind {
    fn summary_label(self) -> &'static str {
        match self {
            ArgScalarKind::U32 => "u32",
            ArgScalarKind::I32 => "i32",
            ArgScalarKind::F32 => "f32",
        }
    }

    fn c_type(self) -> &'static str {
        match self {
            ArgScalarKind::U32 => "uint32_t",
            ArgScalarKind::I32 => "int32_t",
            ArgScalarKind::F32 => "float",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgAlias {
    pub param_idx: u32,
    pub kind: ArgAliasKind,
    pub confidence: AliasConfidence,
    pub observed_words: BTreeSet<u32>,
    pub scalar_kind: Option<ArgScalarKind>,
    pub signed_words: BTreeSet<u32>,
    pub pointee_ty: Option<&'static str>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbiArgAliases {
    pub by_param: BTreeMap<u32, ArgAlias>,
}

impl AbiAnnotations {
    pub fn is_empty(&self) -> bool {
        self.constmem_by_stmt.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&StatementRef, &Vec<ConstMemAnnotation>)> + '_ {
        self.constmem_by_stmt.iter()
    }

    pub fn summarize_lines(&self, max_lines: usize) -> Vec<String> {
        self.constmem_by_stmt
            .iter()
            .flat_map(|(stmt_ref, anns)| {
                anns.iter().map(|ann| {
                    format!(
                        "BB{}.S{}: c[0x{:x}][0x{:x}] -> {}",
                        stmt_ref.block_id,
                        stmt_ref.stmt_idx,
                        ann.bank,
                        ann.offset,
                        ann.symbol()
                    )
                })
            })
            .take(max_lines)
            .collect()
    }
}

impl AbiArgAliases {
    pub fn is_empty(&self) -> bool {
        self.by_param.is_empty()
    }

    /// Returns true if `param_idx` is part of a Ptr64 alias — either as the
    /// lo word (direct entry) or as the hi word (base = param_idx - 1).
    pub fn is_ptr_param(&self, param_idx: u32) -> bool {
        if let Some(alias) = self.by_param.get(&param_idx) {
            if alias.kind == ArgAliasKind::Ptr64 {
                return true;
            }
        }
        if param_idx > 0 {
            if let Some(alias) = self.by_param.get(&(param_idx - 1)) {
                if alias.kind == ArgAliasKind::Ptr64 {
                    return true;
                }
            }
        }
        false
    }

    pub fn render_param_word(&self, param_idx: u32, word_idx: u32) -> Option<String> {
        // Direct hit: the param_idx exists in our alias map.
        if let Some(alias) = self.by_param.get(&param_idx) {
            let rendered = match alias.kind {
                ArgAliasKind::Ptr64 => {
                    let lane = if word_idx == 0 { "lo32" } else { "hi32" };
                    format!("arg{}_ptr.{}", param_idx, lane)
                }
                ArgAliasKind::U64 => {
                    let lane = if word_idx == 0 { "lo32" } else { "hi32" };
                    format!("arg{}_u64.{}", param_idx, lane)
                }
                ArgAliasKind::Word32 => {
                    // With per-word param indexing the canonical word is always 0.
                    // Render as plain "argN" instead of "argN_word0".
                    format!("arg{}", param_idx)
                }
            };
            return Some(rendered);
        }
        // The param_idx might be the hi half of a Ptr64 pair that was merged
        // under the even word.  E.g. param_idx=5 merged into param_idx=4.
        if param_idx > 0 {
            let base = param_idx - 1;
            if let Some(alias) = self.by_param.get(&base) {
                if alias.kind == ArgAliasKind::Ptr64 {
                    return Some(format!("arg{}_ptr.hi32", base));
                }
            }
        }
        None
    }

    pub fn summarize_lines(&self, max_lines: usize) -> Vec<String> {
        self.by_param
            .values()
            .map(|alias| {
                let kind = match alias.kind {
                    ArgAliasKind::Ptr64 => "ptr64",
                    ArgAliasKind::U64 => "u64",
                    ArgAliasKind::Word32 => alias
                        .scalar_kind
                        .map(ArgScalarKind::summary_label)
                        .unwrap_or("word32"),
                };
                format!(
                    "param_{} -> arg{} ({}, confidence: {}, words: {:?})",
                    alias.param_idx,
                    alias.param_idx,
                    kind,
                    alias.confidence.as_str(),
                    alias.observed_words
                )
            })
            .take(max_lines)
            .collect()
    }

    pub fn render_typed_arg_declarations(&self) -> Vec<String> {
        let mut out = Vec::new();
        for alias in self.by_param.values() {
            match alias.kind {
                ArgAliasKind::Ptr64 => {
                    if let Some(elem) = alias.pointee_ty {
                        out.push(format!(
                            "{}* arg{}_ptr; // confidence: {}",
                            elem,
                            alias.param_idx,
                            alias.confidence.as_str()
                        ));
                    } else {
                        out.push(format!(
                            "uintptr_t arg{}_ptr; // confidence: {}",
                            alias.param_idx,
                            alias.confidence.as_str()
                        ));
                    }
                }
                ArgAliasKind::U64 => {
                    out.push(format!(
                        "uint64_t arg{}_u64; // confidence: {}",
                        alias.param_idx,
                        alias.confidence.as_str()
                    ));
                }
                ArgAliasKind::Word32 => {
                    out.push(format!(
                        "{} arg{}; // confidence: {}",
                        alias_scalar_c_type(alias),
                        alias.param_idx,
                        alias.confidence.as_str()
                    ));
                }
            }
        }
        out
    }

    pub fn render_typed_param_list(&self) -> Vec<String> {
        let mut out = Vec::new();
        for alias in self.by_param.values() {
            match alias.kind {
                ArgAliasKind::Ptr64 => {
                    if let Some(elem) = alias.pointee_ty {
                        out.push(format!("{}* arg{}_ptr", elem, alias.param_idx));
                    } else {
                        out.push(format!("uintptr_t arg{}_ptr", alias.param_idx));
                    }
                }
                ArgAliasKind::U64 => {
                    out.push(format!("uint64_t arg{}_u64", alias.param_idx));
                }
                ArgAliasKind::Word32 => {
                    out.push(format!(
                        "{} arg{}",
                        alias_scalar_c_type(alias),
                        alias.param_idx
                    ));
                }
            }
        }
        out
    }
}

fn alias_scalar_kind(alias: &ArgAlias) -> ArgScalarKind {
    alias.scalar_kind.unwrap_or_else(|| {
        if alias.signed_words.contains(&0) {
            ArgScalarKind::I32
        } else {
            ArgScalarKind::U32
        }
    })
}

fn alias_scalar_c_type(alias: &ArgAlias) -> &'static str {
    alias_scalar_kind(alias).c_type()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LocalTypeHint {
    PtrStrong,
    PtrWeak,
    F32,
    U32,
    U16,
    U8,
}

pub fn infer_local_typed_declarations(function_ir: &FunctionIR) -> Vec<String> {
    infer_local_typed_declarations_with_abi(function_ir, None, None)
}

pub fn infer_local_typed_declarations_with_abi(
    function_ir: &FunctionIR,
    annotations: Option<&AbiAnnotations>,
    aliases: Option<&AbiArgAliases>,
) -> Vec<String> {
    let mut regs: BTreeSet<(String, i32)> = BTreeSet::new();
    let mut hints: BTreeMap<(String, i32), BTreeSet<LocalTypeHint>> = BTreeMap::new();

    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            for (def_idx, def) in stmt.defs.iter().enumerate() {
                let IRExpr::Reg(r) = def else {
                    continue;
                };
                if is_immutable_reg(r) {
                    continue;
                }
                let key = (r.class.clone(), r.idx);
                regs.insert(key.clone());
                if let Some(h) = hint_from_def_stmt(r, stmt) {
                    hints.entry(key.clone()).or_default().insert(h);
                }
                if def_idx == 0 {
                    if let Some(h) =
                        abi_pointer_hint_from_dest_stmt(stmt_ref, stmt, annotations, aliases)
                    {
                        hints.entry(key.clone()).or_default().insert(h);
                    }
                }
            }
            collect_pointer_hints_from_stmt(stmt, &mut hints);
        }
    }

    // Use dataflow-based type inference as an additional signal.
    let df_types = infer_types(function_ir);

    regs.into_iter()
        .map(|(class, idx)| {
            if class == "P" || class == "UP" {
                format!("bool {}{};", class, idx)
            } else {
                let key = (class.clone(), idx);
                let hint_ty = select_local_decl_type(hints.get(&key));
                // If per-statement hints produced a non-default type, prefer them
                // (they already account for ABI pointer annotations).
                // Otherwise, use the dataflow inference result.
                let ty = if hint_ty != "uint32_t" {
                    hint_ty
                } else if let Some(df) = df_types.get(&key) {
                    df.to_c_type()
                } else {
                    hint_ty
                };
                format!("{} {}{};", ty, class, idx)
            }
        })
        .collect()
}

fn abi_pointer_hint_from_dest_stmt(
    stmt_ref: StatementRef,
    stmt: &crate::ir::IRStatement,
    annotations: Option<&AbiAnnotations>,
    aliases: Option<&AbiArgAliases>,
) -> Option<LocalTypeHint> {
    let anns = annotations?;
    let aliases = aliases?;
    let RValue::Op { opcode, .. } = &stmt.value else {
        return None;
    };
    if !is_pointer_arith_opcode(opcode) {
        return None;
    }
    let stmt_anns = anns.constmem_by_stmt.get(&stmt_ref)?;
    let has_ptr_param = stmt_anns.iter().any(|ann| {
        let ConstMemSemantic::ParamWord { param_idx, .. } = ann.semantic else {
            return false;
        };
        aliases.is_ptr_param(param_idx)
    });
    if has_ptr_param {
        Some(LocalTypeHint::PtrStrong)
    } else {
        None
    }
}

fn hint_from_def_stmt(dest: &RegId, stmt: &IRStatement) -> Option<LocalTypeHint> {
    let RValue::Op { opcode, .. } = &stmt.value else {
        return None;
    };
    if dest.class == "P" || dest.class == "UP" {
        return None;
    }
    hint_from_dest_opcode(opcode)
}

fn hint_from_dest_opcode(opcode: &str) -> Option<LocalTypeHint> {
    if opcode.starts_with("IMAD.WIDE") || opcode.starts_with("IADD.64") || opcode == "LEA" {
        return Some(LocalTypeHint::PtrStrong);
    }
    if opcode.starts_with("I2F")
        || opcode.starts_with("FADD")
        || opcode.starts_with("FMUL")
        || opcode.starts_with("FFMA")
        || opcode.starts_with("FSEL")
        || opcode.starts_with("MUFU")
    {
        return Some(LocalTypeHint::F32);
    }
    hint_from_opcode_data_suffix(opcode)
}

fn is_pointer_arith_opcode(opcode: &str) -> bool {
    opcode.starts_with("IMAD")
        || opcode.starts_with("IADD.64")
        || opcode.starts_with("LEA")
        || opcode.starts_with("IADD3")
        || opcode.starts_with("UIADD3")
}

fn hint_from_opcode_data_suffix(opcode: &str) -> Option<LocalTypeHint> {
    for tok in opcode.split('.') {
        match tok {
            "U8" | "S8" => return Some(LocalTypeHint::U8),
            "U16" | "S16" => return Some(LocalTypeHint::U16),
            "U32" | "S32" => return Some(LocalTypeHint::U32),
            _ => {}
        }
    }
    None
}

fn collect_pointer_hints_from_stmt(
    stmt: &IRStatement,
    hints: &mut BTreeMap<(String, i32), BTreeSet<LocalTypeHint>>,
) {
    for def in &stmt.defs {
        collect_pointer_hints_from_expr(def, hints);
    }
    if let Some(pred) = &stmt.pred {
        collect_pointer_hints_from_expr(pred, hints);
    }
    match &stmt.value {
        RValue::Op { args, .. } => {
            for arg in args {
                collect_pointer_hints_from_expr(arg, hints);
            }
        }
        RValue::Phi(args) => {
            for arg in args {
                collect_pointer_hints_from_expr(arg, hints);
            }
        }
        RValue::ImmI(_) | RValue::ImmF(_) => {}
    }
    if let Some(mem_args) = &stmt.mem_addr_args {
        for arg in mem_args {
            collect_pointer_hints_from_expr(arg, hints);
        }
    }
}

fn collect_pointer_hints_from_expr(
    expr: &IRExpr,
    hints: &mut BTreeMap<(String, i32), BTreeSet<LocalTypeHint>>,
) {
    match expr {
        IRExpr::Addr64 { lo, hi } => {
            collect_pointer_hints_from_expr(lo, hints);
            collect_pointer_hints_from_expr(hi, hints);
        }
        IRExpr::Mem {
            base,
            offset,
            width,
        } => {
            if matches!(width, Some(64)) {
                mark_pointer_base_reg(base, hints);
            }
            collect_pointer_hints_from_expr(base, hints);
            if let Some(off) = offset {
                collect_pointer_hints_from_expr(off, hints);
            }
        }
        IRExpr::Op { args, .. } => {
            for arg in args {
                collect_pointer_hints_from_expr(arg, hints);
            }
        }
        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
    }
}

fn mark_pointer_base_reg(
    base: &IRExpr,
    hints: &mut BTreeMap<(String, i32), BTreeSet<LocalTypeHint>>,
) {
    if let IRExpr::Op { op, args } = base {
        if op == "addr64" {
            if let Some(lo) = args.first() {
                mark_pointer_base_reg(lo, hints);
            }
            return;
        }
    }
    if let IRExpr::Reg(r) = base {
        if is_immutable_reg(r) || matches!(r.class.as_str(), "P" | "UP" | "PT" | "UPT") {
            return;
        }
        hints
            .entry((r.class.clone(), r.idx))
            .or_default()
            .insert(LocalTypeHint::PtrWeak);
    }
}

fn select_local_decl_type(hints: Option<&BTreeSet<LocalTypeHint>>) -> &'static str {
    let Some(hints) = hints else {
        return "uint32_t";
    };
    let has_float = hints.contains(&LocalTypeHint::F32);
    let has_int = hints.contains(&LocalTypeHint::U32)
        || hints.contains(&LocalTypeHint::U16)
        || hints.contains(&LocalTypeHint::U8);
    if hints.contains(&LocalTypeHint::PtrStrong)
        || (hints.contains(&LocalTypeHint::PtrWeak) && !has_float && !has_int)
    {
        return "uintptr_t";
    }
    if has_float && !has_int {
        return "float";
    }
    if hints.contains(&LocalTypeHint::U32) || (has_float && has_int) {
        return "uint32_t";
    }
    if hints.contains(&LocalTypeHint::U16) {
        return "uint16_t";
    }
    if hints.contains(&LocalTypeHint::U8) {
        return "uint8_t";
    }
    "uint32_t"
}

fn is_immutable_reg(r: &RegId) -> bool {
    matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
}

impl From<ConstMemKind> for ConstMemSemantic {
    fn from(value: ConstMemKind) -> Self {
        match value {
            ConstMemKind::Builtin(name) => ConstMemSemantic::Builtin(name),
            ConstMemKind::ParamWord {
                param_idx,
                word_idx,
            } => ConstMemSemantic::ParamWord {
                param_idx,
                word_idx,
            },
            ConstMemKind::AbiInternal(offset) => ConstMemSemantic::AbiInternal(offset),
        }
    }
}

impl AbiProfile {
    pub fn legacy_param_140() -> Self {
        Self {
            generation: AbiGeneration::LegacyParam140,
            const_bank: 0,
            param_base: 0x140,
        }
    }

    pub fn modern_param_160() -> Self {
        Self {
            generation: AbiGeneration::ModernParam160,
            const_bank: 0,
            param_base: 0x160,
        }
    }

    pub fn blackwell_param_380() -> Self {
        Self {
            generation: AbiGeneration::BlackwellParam380,
            const_bank: 0,
            param_base: 0x380,
        }
    }

    /// Detect the best-effort ABI profile from observed constant-memory offsets.
    ///
    /// This is the offset-only entry point and is preserved for callers that
    /// genuinely have no SM metadata.  The offset heuristic is fundamentally
    /// fragile in two directions: a modern kernel that addresses a
    /// high-numbered parameter at `c[0x0][0x360]` looks like a Blackwell
    /// builtin, and a Blackwell trace that touches `c[0x0][0x140/0x160]`
    /// (e.g. from a runtime helper) looks legacy/modern.  Prefer
    /// [`Self::detect_with_sm`] whenever SM metadata is available — it
    /// trusts the architecture name and avoids both failure modes.
    pub fn detect(instructions: &[DecodedInstruction]) -> Self {
        Self::detect_with_sm(instructions, None)
    }

    /// Pick an ABI profile, trusting SM metadata when present.
    ///
    /// SM metadata comes from the SASS dump's `// arch=` header, which is
    /// authoritative: the toolchain stamps it on every dump and it cannot
    /// disagree with the actual ABI generation.  When `sm` is `Some`, this
    /// method uses [`Self::profile_for_sm`] directly and ignores constant-
    /// memory offsets entirely; otherwise it falls back to
    /// [`Self::detect_from_offsets`] (and finally to the modern profile
    /// to keep the historical default stable for empty inputs).
    pub fn detect_with_sm(instructions: &[DecodedInstruction], sm: Option<u32>) -> Self {
        if let Some(sm_val) = sm {
            return Self::profile_for_sm(sm_val);
        }

        if let Some(by_offset) = Self::detect_from_offsets(instructions) {
            return by_offset;
        }

        // Default keeps existing behavior and current fixtures stable.
        Self::modern_param_160()
    }

    /// Map a known SM number to the matching ABI generation.
    /// SM 100+ is Blackwell (param_base 0x380), 80–99 is modern
    /// (param_base 0x160), anything older is legacy (param_base 0x140).
    pub(crate) fn profile_for_sm(sm: u32) -> Self {
        if sm >= 100 {
            return Self::blackwell_param_380();
        }
        if sm >= 80 {
            return Self::modern_param_160();
        }
        Self::legacy_param_140()
    }

    fn detect_from_offsets(instructions: &[DecodedInstruction]) -> Option<Self> {
        let mut near_140_hits = 0usize;
        let mut near_160_hits = 0usize;
        let mut near_380_hits = 0usize;
        // Blackwell-only evidence: the relocated builtin block
        // (`0x360..0x378`) and the new ABI internal slots (`0x358`, `0x37c`).
        // Legacy/modern kernels can still legitimately read offsets in
        // `[0x380, 0x3a0)` when their parameter list is huge, so the bare
        // `near_380_hits` counter is not enough to disambiguate; we need
        // at least one slot the older profiles never touch before forcing
        // Blackwell over a kernel that also looks like SM 80/89.
        let mut blackwell_unique_hits = 0usize;

        for ins in instructions {
            for op in &ins.operands {
                collect_constmem_hits(op, &mut |bank, off| {
                    if bank != 0 {
                        return;
                    }
                    if (0x140..0x160).contains(&off) && off % 4 == 0 {
                        near_140_hits += 1;
                    }
                    if (0x160..0x180).contains(&off) && off % 4 == 0 {
                        near_160_hits += 1;
                    }
                    if (0x380..0x3a0).contains(&off) && off % 4 == 0 {
                        near_380_hits += 1;
                    }
                    if (0x360..0x378).contains(&off) && off % 4 == 0 {
                        blackwell_unique_hits += 1;
                    }
                    if off == 0x358 || off == 0x37c {
                        blackwell_unique_hits += 1;
                    }
                });
            }
        }

        // The "Blackwell-unique" slots are unique only relative to the
        // *built-in* tables of legacy/modern profiles — under those
        // profiles `resolve_constmem` will still happily classify any
        // aligned `offset >= param_base` as a far parameter word, so a
        // very-large-param-list SM 80/89 kernel can land hits at exactly
        // the same `0x358` / `0x360..0x378` / `0x37c` slots.  To stay
        // safe we require *both* a Blackwell-area fingerprint *and* no
        // contradicting hit in either older window before forcing
        // Blackwell.  Real Blackwell kernels never read the legacy
        // parameter windows because the Blackwell `param_base` is
        // 0x380, so this gate is loss-free for the SM 100 corpus.
        let blackwell_evidence = blackwell_unique_hits > 0 || near_380_hits > 0;
        let older_evidence = near_140_hits > 0 || near_160_hits > 0;
        if blackwell_evidence && !older_evidence {
            return Some(Self::blackwell_param_380());
        }
        if near_160_hits > 0 && near_160_hits >= near_140_hits {
            return Some(Self::modern_param_160());
        }
        if near_140_hits > 0 {
            return Some(Self::legacy_param_140());
        }
        None
    }

    /// Resolve `c[bank][offset]` into a symbolic ABI meaning when possible.
    pub fn resolve_constmem(&self, bank: u32, offset: u32) -> Option<ResolvedConstMem> {
        if bank != self.const_bank {
            return None;
        }

        // Built-in thread/block dimensions.  Blackwell relocates this block
        // from c[0x0][0x0..] to c[0x0][0x360..]; older architectures keep it
        // anchored at zero.
        let builtin_base = match self.generation {
            AbiGeneration::BlackwellParam380 => 0x360u32,
            _ => 0x0,
        };
        let builtin = offset.checked_sub(builtin_base).and_then(|rel| match rel {
            0x0 => Some("blockDim.x"),
            0x4 => Some("blockDim.y"),
            0x8 => Some("blockDim.z"),
            0xc => Some("gridDim.x"),
            0x10 => Some("gridDim.y"),
            0x14 => Some("gridDim.z"),
            _ => None,
        });
        if let Some(name) = builtin {
            return Some(ResolvedConstMem {
                symbol: name.to_string(),
                kind: ConstMemKind::Builtin(name),
            });
        }

        if offset >= self.param_base && (offset - self.param_base) % 4 == 0 {
            let word = (offset - self.param_base) / 4;
            // Each 4-byte word is its own param slot.  The merging of
            // consecutive slots into Ptr64 / U64 happens later in
            // infer_arg_aliases(), where we have actual usage evidence.
            let param_idx = word;
            let word_idx: u32 = 0;
            return Some(ResolvedConstMem {
                symbol: format!("param_{}", param_idx),
                kind: ConstMemKind::ParamWord {
                    param_idx,
                    word_idx,
                },
            });
        }

        // ABI-internal scratch slots (frame pointer source, descriptor
        // register source, etc.).  The exact offsets depend on the ABI
        // generation.
        let is_internal = match self.generation {
            // Blackwell: 0x358 = descriptor register source, 0x37c = frame
            // pointer source (R1 is loaded from this at kernel entry).
            AbiGeneration::BlackwellParam380 => matches!(offset, 0x358 | 0x37c),
            // Older generations: scratch slots at 0x28 / 0x44.
            _ => matches!(offset, 0x28 | 0x44),
        };
        if is_internal {
            return Some(ResolvedConstMem {
                symbol: format!("abi_internal_0x{:x}", offset),
                kind: ConstMemKind::AbiInternal(offset),
            });
        }

        None
    }

    pub fn classify_constmem(&self, bank: u32, offset: u32) -> ConstMemSemantic {
        if let Some(resolved) = self.resolve_constmem(bank, offset) {
            return resolved.kind.into();
        }
        ConstMemSemantic::Unknown { bank, offset }
    }
}

pub struct AbiDisplay {
    profile: AbiProfile,
    aliases: Option<AbiArgAliases>,
}

impl AbiDisplay {
    pub fn new(profile: AbiProfile) -> Self {
        Self {
            profile,
            aliases: None,
        }
    }

    pub fn with_aliases(profile: AbiProfile, aliases: AbiArgAliases) -> Self {
        Self {
            profile,
            aliases: Some(aliases),
        }
    }

    pub fn profile(&self) -> AbiProfile {
        self.profile
    }

    fn try_constmem_symbol(&self, op: &str, args: &[IRExpr]) -> Option<String> {
        if op != "ConstMem" || args.len() != 2 {
            return None;
        }
        let bank = imm_as_u32(&args[0])?;
        let offset = imm_as_u32(&args[1])?;
        let resolved = self.profile.resolve_constmem(bank, offset)?;
        if let ConstMemKind::ParamWord {
            param_idx,
            word_idx,
        } = resolved.kind
        {
            if let Some(sym) = self
                .aliases
                .as_ref()
                .and_then(|a| a.render_param_word(param_idx, word_idx))
            {
                return Some(sym);
            }
        }
        Some(resolved.symbol)
    }
}

impl DisplayCtx for AbiDisplay {
    fn reg(&self, r: &RegId) -> String {
        r.display()
    }

    fn expr(&self, e: &IRExpr) -> String {
        match e {
            IRExpr::Reg(r) => self.reg(r),
            IRExpr::ImmI(i) => format!("{}", i),
            IRExpr::ImmF(f) => format!("{}", f),
            IRExpr::Addr64 { lo, hi } => {
                format!("addr64({}, {})", self.expr(lo), self.expr(hi))
            }
            IRExpr::Mem {
                base,
                offset,
                width,
            } => {
                let s = if let Some(off) = offset {
                    format!("*({} + {})", self.expr(base), self.expr(off))
                } else {
                    format!("*{}", self.expr(base))
                };
                let _ = width;
                s
            }
            IRExpr::Op { op, args } => {
                if op == "-" && args.len() == 1 {
                    let inner = self.expr(&args[0]);
                    let simple = match &args[0] {
                        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => true,
                        IRExpr::Op { op, .. } if op == "ConstMem" => true,
                        _ => false,
                    };
                    if simple {
                        return format!("-{}", inner);
                    }
                    return format!("-({})", inner);
                }
                if let Some(sym) = self.try_constmem_symbol(op, args) {
                    return sym;
                }
                let list = args
                    .iter()
                    .map(|a| self.expr(a))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", op, list)
            }
        }
    }
}

fn imm_as_u32(e: &IRExpr) -> Option<u32> {
    match e {
        IRExpr::ImmI(i) if *i >= 0 => u32::try_from(*i).ok(),
        _ => None,
    }
}

fn collect_constmem_hits<F>(op: &DecodedOperand, f: &mut F)
where
    F: FnMut(u32, u32),
{
    match op {
        DecodedOperand::ConstMem { bank, offset } => f(*bank, *offset),
        DecodedOperand::Address { base, .. } => collect_constmem_hits(base.as_ref(), f),
        DecodedOperand::DescriptorMem {
            descriptor, addr, ..
        } => {
            collect_constmem_hits(descriptor.as_ref(), f);
            collect_constmem_hits(addr.as_ref(), f);
        }
        _ => {}
    }
}

pub fn annotate_function_ir_constmem(
    function_ir: &FunctionIR,
    profile: AbiProfile,
) -> AbiAnnotations {
    let mut out = AbiAnnotations::default();
    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let mut raw_pairs = Vec::new();
            for def in &stmt.defs {
                collect_constmem_from_expr(def, &mut raw_pairs);
            }
            if let Some(pred) = &stmt.pred {
                collect_constmem_from_expr(pred, &mut raw_pairs);
            }
            match &stmt.value {
                RValue::Op { args, .. } => {
                    for arg in args {
                        collect_constmem_from_expr(arg, &mut raw_pairs);
                    }
                }
                RValue::Phi(args) => {
                    for arg in args {
                        collect_constmem_from_expr(arg, &mut raw_pairs);
                    }
                }
                RValue::ImmI(_) | RValue::ImmF(_) => {}
            }
            if let Some(mem_args) = &stmt.mem_addr_args {
                for arg in mem_args {
                    collect_constmem_from_expr(arg, &mut raw_pairs);
                }
            }

            if raw_pairs.is_empty() {
                continue;
            }

            let mut unique = BTreeSet::new();
            let mut anns = Vec::new();
            for (bank, offset) in raw_pairs {
                if !unique.insert((bank, offset)) {
                    continue;
                }
                anns.push(ConstMemAnnotation {
                    bank,
                    offset,
                    semantic: profile.classify_constmem(bank, offset),
                });
            }
            out.constmem_by_stmt.insert(
                StatementRef {
                    block_id: block.id,
                    stmt_idx,
                },
                anns,
            );
        }
    }
    out
}

pub fn infer_arg_aliases(function_ir: &FunctionIR, annotations: &AbiAnnotations) -> AbiArgAliases {
    /// Per-word-param usage statistics.
    #[derive(Default)]
    struct WordUsage {
        pointer_like_hits: usize,
        total_hits: usize,
        signed_hits: usize,
    }

    let flow = build_param_word_flow(function_ir, annotations);

    // ---- 1. Build opcode index ----
    let mut opcode_by_stmt: BTreeMap<StatementRef, String> = BTreeMap::new();
    let mut param_word_span_by_stmt: BTreeMap<StatementRef, usize> = BTreeMap::new();
    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            let opcode = match &stmt.value {
                RValue::Op { opcode, .. } => opcode.clone(),
                RValue::Phi(_) => "phi".to_string(),
                RValue::ImmI(_) | RValue::ImmF(_) => "imm".to_string(),
            };
            opcode_by_stmt.insert(stmt_ref, opcode);
            let span = rvalue_op_parts(stmt)
                .map(|(opcode, args)| {
                    constmem_load_word_span(opcode).max(constmem_copy_word_span(opcode, args))
                })
                .unwrap_or(0);
            if span > 0 {
                param_word_span_by_stmt.insert(stmt_ref, span);
            }
        }
    }

    // ---- 2. Gather per-word usage ----
    // With per-word param indexing every param_idx is a single 4-byte word
    // (word_idx is always 0).
    let mut by_word: BTreeMap<u32, WordUsage> = BTreeMap::new();
    for (stmt_ref, anns) in annotations.iter() {
        let opcode = opcode_by_stmt
            .get(stmt_ref)
            .map(String::as_str)
            .unwrap_or("");
        let span = param_word_span_by_stmt.get(stmt_ref).copied().unwrap_or(1);
        let mut touched_words = BTreeSet::new();
        for ann in anns {
            if let ConstMemSemantic::ParamWord { param_idx, .. } = ann.semantic {
                if span > 1 {
                    for lane in 0..span {
                        touched_words.insert(param_idx + lane as u32);
                    }
                } else {
                    touched_words.insert(param_idx);
                }
            }
        }
        for param_idx in touched_words {
            let usage = by_word.entry(param_idx).or_default();
            usage.total_hits += 1;
            if is_pointer_context_opcode(opcode) {
                usage.pointer_like_hits += 1;
            }
            if is_signed_word_context_opcode(opcode) {
                usage.signed_hits += 1;
            }
        }
    }

    let scalar_evidence_by_word = collect_param_scalar_evidence(function_ir, &flow);
    let addr64_pair_bases = collect_param_loaded_addr64_pair_bases(function_ir, annotations);
    let pointer_pointee_tys = collect_param_pointer_pointee_types(function_ir, annotations, &flow);

    // ---- 3. Merge consecutive even/odd pairs into Ptr64 when evidence supports it ----
    let word_indices: Vec<u32> = by_word.keys().copied().collect();
    let mut merged_as_hi: BTreeSet<u32> = BTreeSet::new(); // odd words consumed by a pair

    let mut out = AbiArgAliases::default();

    for &lo in &word_indices {
        if lo % 2 != 0 {
            continue; // only start pairs on even-numbered words
        }
        let hi = lo + 1;
        if !by_word.contains_key(&lo) {
            continue;
        }
        // Only merge into a ptr64 when we have explicit pair evidence from the
        // SSA/address pipeline. Broad "both words appeared near pointer math"
        // heuristics over-merge scalar pairs like (W, H) or (N, eps2) into fake
        // pointers, which then leak `.lo32/.hi32` machine detail into final AST.
        let addr64_pair = addr64_pair_bases.contains(&lo);
        if addr64_pair {
            // Merge into Ptr64 keyed by the even (lo) word index.
            let pointee_ty = pointer_pointee_tys
                .get(&lo)
                .and_then(infer_pointer_pointee_ty_from_evidence);
            out.by_param.insert(
                lo,
                ArgAlias {
                    param_idx: lo,
                    kind: ArgAliasKind::Ptr64,
                    confidence: AliasConfidence::High,
                    observed_words: [0u32, 1].iter().copied().collect(),
                    scalar_kind: None,
                    signed_words: BTreeSet::new(),
                    pointee_ty,
                },
            );
            merged_as_hi.insert(hi);
        }
    }

    // ---- 4. Remaining words become Word32 ----
    for (&word_idx, usage) in &by_word {
        if out.by_param.contains_key(&word_idx) || merged_as_hi.contains(&word_idx) {
            continue; // already handled as part of a Ptr64 pair
        }
        let scalar_evidence = scalar_evidence_by_word
            .get(&word_idx)
            .copied()
            .unwrap_or_default();
        let scalar_kind = select_word_scalar_kind(&scalar_evidence);
        let is_signed = matches!(scalar_kind, ArgScalarKind::I32) || usage.signed_hits > 0;
        out.by_param.insert(
            word_idx,
            ArgAlias {
                param_idx: word_idx,
                kind: ArgAliasKind::Word32,
                confidence: if scalar_evidence.has_signal() {
                    AliasConfidence::Medium
                } else {
                    AliasConfidence::Low
                },
                observed_words: [0u32].iter().copied().collect(),
                scalar_kind: Some(scalar_kind),
                signed_words: if is_signed {
                    [0u32].iter().copied().collect()
                } else {
                    BTreeSet::new()
                },
                pointee_ty: None,
            },
        );
    }
    out
}

#[derive(Debug, Default)]
struct ParamWordFlow {
    loaded_param_by_reg: BTreeMap<RegId, u32>,
    constmem_word_by_pair: BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: BTreeMap<RegId, RegId>,
    phi_sources_by_reg: BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ScalarEvidence {
    float_hits: usize,
    signed_int_hits: usize,
    unsigned_int_hits: usize,
}

impl ScalarEvidence {
    fn has_signal(self) -> bool {
        self.float_hits > 0 || self.signed_int_hits > 0 || self.unsigned_int_hits > 0
    }
}

fn select_word_scalar_kind(evidence: &ScalarEvidence) -> ArgScalarKind {
    if evidence.float_hits > 0 {
        ArgScalarKind::F32
    } else if evidence.signed_int_hits > 0 {
        ArgScalarKind::I32
    } else {
        ArgScalarKind::U32
    }
}

fn build_param_word_flow(function_ir: &FunctionIR, annotations: &AbiAnnotations) -> ParamWordFlow {
    let mut flow = ParamWordFlow::default();

    for (_stmt, anns) in annotations.iter() {
        for ann in anns {
            if let ConstMemSemantic::ParamWord { param_idx, .. } = ann.semantic {
                flow.constmem_word_by_pair
                    .insert((ann.bank, ann.offset), param_idx);
            }
        }
    }

    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            let opcode = match &stmt.value {
                RValue::Op { opcode, .. } => Some(opcode.as_str()),
                _ => None,
            };
            if let Some(opcode) = opcode {
                if let Some(src) = extract_copy_source_expr(opcode, rvalue_args(stmt)) {
                    if stmt.defs.len() == 1 {
                        if let Some(IRExpr::Reg(dst)) = stmt.defs.first() {
                            flow.copy_source_by_reg.insert(dst.clone(), src.clone());
                        }
                    }
                }
            }
            if let Some((opcode, args)) = rvalue_op_parts(stmt) {
                for (def_idx, def) in stmt.defs.iter().enumerate() {
                    if let IRExpr::Reg(reg) = def {
                        flow.def_by_reg
                            .insert(reg.clone(), (opcode.to_string(), args.to_vec(), def_idx));
                    }
                }
            } else if let RValue::Phi(args) = &stmt.value {
                for def in &stmt.defs {
                    if let IRExpr::Reg(reg) = def {
                        flow.phi_sources_by_reg.insert(reg.clone(), args.to_vec());
                    }
                }
            }

            let Some(base_param) = annotations
                .constmem_by_stmt
                .get(&stmt_ref)
                .and_then(|anns| {
                    anns.iter().find_map(|ann| match ann.semantic {
                        ConstMemSemantic::ParamWord { param_idx, .. } => Some(param_idx),
                        _ => None,
                    })
                })
            else {
                continue;
            };
            let Some(opcode) = opcode else {
                continue;
            };
            let word_span = constmem_load_word_span(opcode)
                .max(constmem_copy_word_span(opcode, rvalue_args(stmt)));
            if word_span == 0 {
                continue;
            }
            for lane in 0..word_span.min(stmt.defs.len()) {
                if let Some(IRExpr::Reg(reg)) = stmt.defs.get(lane) {
                    flow.loaded_param_by_reg
                        .insert(reg.clone(), base_param + lane as u32);
                }
            }
        }
    }

    flow
}

fn collect_param_scalar_evidence(
    function_ir: &FunctionIR,
    flow: &ParamWordFlow,
) -> BTreeMap<u32, ScalarEvidence> {
    let mut evidence_by_word = BTreeMap::<u32, ScalarEvidence>::new();

    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            let Some((opcode, args)) = rvalue_op_parts(stmt) else {
                continue;
            };
            let arg_types = infer_arg_types_from_opcode(opcode, args.len());
            for (arg, inferred_ty) in args.iter().zip(arg_types.into_iter()) {
                let Some(word) = resolve_param_word_from_expr(
                    arg,
                    &flow.loaded_param_by_reg,
                    &flow.constmem_word_by_pair,
                    &flow.copy_source_by_reg,
                    &flow.phi_sources_by_reg,
                    &flow.def_by_reg,
                    &mut BTreeSet::new(),
                ) else {
                    continue;
                };
                let Some(kind) = scalar_kind_from_inferred_type(inferred_ty) else {
                    continue;
                };
                let entry = evidence_by_word.entry(word).or_default();
                match kind {
                    ArgScalarKind::F32 => entry.float_hits += 1,
                    ArgScalarKind::I32 => entry.signed_int_hits += 1,
                    ArgScalarKind::U32 => entry.unsigned_int_hits += 1,
                }
            }
        }
    }

    evidence_by_word
}

fn scalar_kind_from_inferred_type(ty: Option<InferredType>) -> Option<ArgScalarKind> {
    match ty? {
        InferredType::I32 => Some(ArgScalarKind::I32),
        InferredType::F16 | InferredType::F32 | InferredType::AnyFloat => Some(ArgScalarKind::F32),
        InferredType::U8 | InferredType::U16 | InferredType::U32 | InferredType::AnyInt => {
            Some(ArgScalarKind::U32)
        }
        InferredType::Bottom | InferredType::U64 | InferredType::Ptr64 | InferredType::Top => None,
    }
}

fn collect_param_pointer_pointee_types(
    function_ir: &FunctionIR,
    _annotations: &AbiAnnotations,
    flow: &ParamWordFlow,
) -> BTreeMap<u32, BTreeSet<&'static str>> {
    let ssa_types = infer_ssa_types(function_ir);
    let shared_reg_fallback_tys = infer_shared_word_reg_fallback_types(function_ir, &ssa_types);
    let copy_phi_predecessors = build_copy_phi_predecessors(function_ir);
    let copy_phi_successors = reverse_reg_edges(&copy_phi_predecessors);
    let mut evidence = BTreeMap::<u32, BTreeSet<&'static str>>::new();

    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            let RValue::Op { opcode, args } = &stmt.value else {
                continue;
            };
            let Some(mem_expr) = stmt.mem_addr_args.as_ref().and_then(|args| args.first()) else {
                continue;
            };
            let Some(base_word) = resolve_param_pair_base_from_mem_expr(mem_expr, flow) else {
                continue;
            };
            let Some(pointee_ty) = memory_pointee_type_from_stmt(
                stmt,
                opcode,
                args,
                &ssa_types,
                Some(&shared_reg_fallback_tys),
                Some(&copy_phi_predecessors),
                Some(&copy_phi_successors),
            ) else {
                continue;
            };
            evidence.entry(base_word).or_default().insert(pointee_ty);
        }
    }

    evidence
}

fn resolve_param_pair_base_from_mem_expr(mem_expr: &IRExpr, flow: &ParamWordFlow) -> Option<u32> {
    let IRExpr::Mem { base, width, .. } = mem_expr else {
        return None;
    };
    if !matches!(width, Some(64)) {
        return None;
    }
    resolve_param_pair_base_from_addr_base_expr(base, flow)
}

fn resolve_param_pair_base_from_addr_base_expr(base: &IRExpr, flow: &ParamWordFlow) -> Option<u32> {
    match base {
        IRExpr::Addr64 { lo, hi } => resolve_param_pair_base_from_lo_hi_exprs(
            lo,
            hi,
            &flow.loaded_param_by_reg,
            &flow.constmem_word_by_pair,
            &flow.copy_source_by_reg,
            &flow.phi_sources_by_reg,
            &flow.def_by_reg,
            &mut BTreeSet::new(),
        ),
        IRExpr::Op { op, args } if op == "addr64" && args.len() >= 2 => {
            resolve_param_pair_base_from_lo_hi_exprs(
                &args[0],
                &args[1],
                &flow.loaded_param_by_reg,
                &flow.constmem_word_by_pair,
                &flow.copy_source_by_reg,
                &flow.phi_sources_by_reg,
                &flow.def_by_reg,
                &mut BTreeSet::new(),
            )
        }
        IRExpr::Reg(reg) => {
            let lo_word = resolve_param_word_from_reg(
                reg,
                &flow.loaded_param_by_reg,
                &flow.constmem_word_by_pair,
                &flow.copy_source_by_reg,
                &flow.phi_sources_by_reg,
                &flow.def_by_reg,
                &mut BTreeSet::new(),
            )?;
            if lo_word % 2 != 0 {
                return None;
            }
            let mut hi = reg.clone();
            hi.idx += 1;
            let hi_word = resolve_param_word_from_reg(
                &hi,
                &flow.loaded_param_by_reg,
                &flow.constmem_word_by_pair,
                &flow.copy_source_by_reg,
                &flow.phi_sources_by_reg,
                &flow.def_by_reg,
                &mut BTreeSet::new(),
            )?;
            (hi_word == lo_word + 1).then_some(lo_word)
        }
        _ => None,
    }
}

fn memory_pointee_type_from_stmt(
    stmt: &IRStatement,
    opcode: &str,
    args: &[IRExpr],
    ssa_types: &BTreeMap<RegId, InferredType>,
    reg_fallback_tys: Option<&BTreeMap<RegId, &'static str>>,
    copy_phi_predecessors: Option<&BTreeMap<RegId, Vec<RegId>>>,
    copy_phi_successors: Option<&BTreeMap<RegId, Vec<RegId>>>,
) -> Option<&'static str> {
    let expected_width = memory_scalar_width_bytes(opcode);
    if let Some(ty) = scalar_pointee_type_from_opcode(opcode) {
        return Some(ty);
    }

    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if opcode.starts_with("LD") && !matches!(mnem, "LDC" | "LDCU" | "ULDC" | "LDS") {
        return stmt
            .defs
            .first()
            .and_then(IRExpr::get_reg)
            .and_then(|reg| {
                reg_pointee_type(
                    reg,
                    ssa_types,
                    reg_fallback_tys,
                    copy_phi_successors,
                    expected_width,
                )
            })
            .or_else(|| default_pointee_ty_for_width(expected_width));
    }

    if opcode.starts_with("ST") {
        return args
            .get(1)
            .and_then(|expr| {
                expr_pointee_type(
                    expr,
                    ssa_types,
                    reg_fallback_tys,
                    copy_phi_predecessors,
                    expected_width,
                )
            })
            .or_else(|| default_pointee_ty_for_width(expected_width));
    }

    if opcode.starts_with("ATOM")
        || opcode.starts_with("ATOMG")
        || opcode.starts_with("RED")
        || opcode.starts_with("REDG")
    {
        return args
            .iter()
            .find(|arg| !matches!(arg, IRExpr::Mem { .. }))
            .and_then(|expr| {
                expr_pointee_type(
                    expr,
                    ssa_types,
                    reg_fallback_tys,
                    copy_phi_predecessors,
                    expected_width,
                )
            })
            .or_else(|| default_pointee_ty_for_width(expected_width));
    }

    None
}

pub(crate) fn infer_shared_word_pointee_type_for_function(
    function_ir: &FunctionIR,
) -> Option<&'static str> {
    let ssa_types = infer_ssa_types(function_ir);
    infer_shared_word_pointee_type(function_ir, &ssa_types)
}

fn infer_shared_word_reg_fallback_types(
    function_ir: &FunctionIR,
    ssa_types: &BTreeMap<RegId, InferredType>,
) -> BTreeMap<RegId, &'static str> {
    let Some(shared_word_ty) = infer_shared_word_pointee_type(function_ir, ssa_types) else {
        return BTreeMap::new();
    };

    let predecessors = build_copy_phi_predecessors(function_ir);
    let successors = reverse_reg_edges(&predecessors);
    let mut map = BTreeMap::new();
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            let RValue::Op { opcode, args } = &stmt.value else {
                continue;
            };
            if memory_scalar_width_bytes(opcode) != 4 {
                continue;
            }
            if opcode.starts_with("STS") {
                for src in args.iter().skip(1) {
                    let Some(reg) = src.get_reg() else {
                        continue;
                    };
                    mark_reg_and_predecessors_with_type(
                        reg,
                        shared_word_ty,
                        &predecessors,
                        &mut map,
                        &mut BTreeSet::new(),
                    );
                }
            } else if opcode.starts_with("LDS") {
                for def in &stmt.defs {
                    let Some(reg) = def.get_reg() else {
                        continue;
                    };
                    mark_reg_and_predecessors_with_type(
                        reg,
                        shared_word_ty,
                        &successors,
                        &mut map,
                        &mut BTreeSet::new(),
                    );
                }
            }
        }
    }
    map
}

fn build_copy_phi_predecessors(function_ir: &FunctionIR) -> BTreeMap<RegId, Vec<RegId>> {
    let mut predecessors = BTreeMap::<RegId, Vec<RegId>>::new();
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            match &stmt.value {
                RValue::Op { opcode, args } => {
                    let Some(src) = extract_copy_source_expr(opcode, args) else {
                        continue;
                    };
                    if stmt.defs.len() != 1 {
                        continue;
                    }
                    let Some(dst) = stmt.defs.first().and_then(IRExpr::get_reg) else {
                        continue;
                    };
                    if is_immutable_reg(dst) || is_immutable_reg(&src) {
                        continue;
                    }
                    predecessors.entry(dst.clone()).or_default().push(src);
                }
                RValue::Phi(args) => {
                    let phi_srcs: Vec<RegId> = args
                        .iter()
                        .filter_map(|arg| arg.get_reg())
                        .filter(|reg| !is_immutable_reg(reg))
                        .cloned()
                        .collect();
                    if phi_srcs.is_empty() {
                        continue;
                    }
                    for def in &stmt.defs {
                        let Some(dst) = def.get_reg() else {
                            continue;
                        };
                        if is_immutable_reg(dst) {
                            continue;
                        }
                        predecessors
                            .entry(dst.clone())
                            .or_default()
                            .extend(phi_srcs.iter().cloned());
                    }
                }
                _ => {}
            }
        }
    }
    predecessors
}

fn reverse_reg_edges(edges: &BTreeMap<RegId, Vec<RegId>>) -> BTreeMap<RegId, Vec<RegId>> {
    let mut reversed = BTreeMap::<RegId, Vec<RegId>>::new();
    for (dst, srcs) in edges {
        for src in srcs {
            reversed.entry(src.clone()).or_default().push(dst.clone());
        }
    }
    reversed
}

fn mark_reg_and_predecessors_with_type(
    reg: &RegId,
    ty: &'static str,
    predecessors: &BTreeMap<RegId, Vec<RegId>>,
    out: &mut BTreeMap<RegId, &'static str>,
    visiting: &mut BTreeSet<RegId>,
) {
    if !visiting.insert(reg.clone()) {
        return;
    }
    out.insert(reg.clone(), ty);
    if let Some(preds) = predecessors.get(reg) {
        for pred in preds {
            mark_reg_and_predecessors_with_type(pred, ty, predecessors, out, visiting);
        }
    }
}

fn infer_shared_word_pointee_type(
    function_ir: &FunctionIR,
    ssa_types: &BTreeMap<RegId, InferredType>,
) -> Option<&'static str> {
    let mut shared_ty = InferredType::Bottom;

    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            let RValue::Op { opcode, args } = &stmt.value else {
                continue;
            };
            if memory_scalar_width_bytes(opcode) != 4 {
                continue;
            }
            if opcode.starts_with("LDS") {
                for def in &stmt.defs {
                    let Some(reg) = def.get_reg() else {
                        continue;
                    };
                    if let Some(ty) = ssa_types.get(reg).copied() {
                        shared_ty = shared_ty.join(ty);
                    }
                }
                continue;
            }
            if opcode.starts_with("STS") {
                for src in args.iter().skip(1) {
                    if let Some(ty) = expr_inferred_type(src, ssa_types) {
                        shared_ty = shared_ty.join(ty);
                    }
                }
            }
        }
    }

    inferred_type_to_pointee_ty_for_width(shared_ty, 4)
}

fn expr_inferred_type(
    expr: &IRExpr,
    ssa_types: &BTreeMap<RegId, InferredType>,
) -> Option<InferredType> {
    match expr {
        IRExpr::ImmF(_) => Some(InferredType::F32),
        IRExpr::ImmI(_) => None,
        _ => expr.get_reg().and_then(|reg| ssa_types.get(reg).copied()),
    }
}

fn expr_pointee_type(
    expr: &IRExpr,
    ssa_types: &BTreeMap<RegId, InferredType>,
    reg_fallback_tys: Option<&BTreeMap<RegId, &'static str>>,
    copy_phi_edges: Option<&BTreeMap<RegId, Vec<RegId>>>,
    expected_width: usize,
) -> Option<&'static str> {
    match expr {
        IRExpr::ImmF(_) => (expected_width == 4).then_some("float"),
        IRExpr::ImmI(_) => None,
        _ => expr.get_reg().and_then(|reg| {
            reg_pointee_type(
                reg,
                ssa_types,
                reg_fallback_tys,
                copy_phi_edges,
                expected_width,
            )
        }),
    }
}

fn reg_pointee_type(
    reg: &RegId,
    ssa_types: &BTreeMap<RegId, InferredType>,
    reg_fallback_tys: Option<&BTreeMap<RegId, &'static str>>,
    copy_phi_edges: Option<&BTreeMap<RegId, Vec<RegId>>>,
    expected_width: usize,
) -> Option<&'static str> {
    ssa_types
        .get(reg)
        .copied()
        .and_then(|ty| inferred_type_to_pointee_ty_for_width(ty, expected_width))
        .or_else(|| reg_fallback_tys.and_then(|map| map.get(reg).copied()))
        .or_else(|| {
            reg_reachable_pointee_type(
                reg,
                ssa_types,
                reg_fallback_tys,
                copy_phi_edges,
                expected_width,
            )
        })
}

fn reg_reachable_pointee_type(
    reg: &RegId,
    ssa_types: &BTreeMap<RegId, InferredType>,
    reg_fallback_tys: Option<&BTreeMap<RegId, &'static str>>,
    copy_phi_edges: Option<&BTreeMap<RegId, Vec<RegId>>>,
    expected_width: usize,
) -> Option<&'static str> {
    let edges = copy_phi_edges?;
    let mut stack = vec![reg.clone()];
    let mut visited = BTreeSet::<RegId>::new();
    let mut evidence = BTreeSet::<&'static str>::new();

    while let Some(current) = stack.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        if let Some(ty) = ssa_types
            .get(&current)
            .copied()
            .and_then(|ty| inferred_type_to_pointee_ty_for_width(ty, expected_width))
        {
            evidence.insert(ty);
        }
        if let Some(ty) = reg_fallback_tys.and_then(|map| map.get(&current).copied()) {
            if pointee_type_width_bytes(ty) == Some(expected_width) {
                evidence.insert(ty);
            }
        }
        if let Some(next_regs) = edges.get(&current) {
            stack.extend(next_regs.iter().cloned());
        }
    }

    infer_pointer_pointee_ty_from_evidence(&evidence)
}

fn inferred_type_to_pointee_ty_for_width(
    ty: InferredType,
    expected_width: usize,
) -> Option<&'static str> {
    match ty {
        InferredType::U8 if expected_width == 1 => Some("uint8_t"),
        InferredType::U16 if expected_width == 2 => Some("uint16_t"),
        InferredType::U32 | InferredType::AnyInt if expected_width == 4 => Some("uint32_t"),
        InferredType::I32 if expected_width == 4 => Some("int32_t"),
        InferredType::F16 if expected_width == 2 => Some("__half"),
        InferredType::F32 | InferredType::AnyFloat if expected_width == 4 => Some("float"),
        InferredType::U64 if expected_width == 8 => Some("uint64_t"),
        InferredType::Bottom | InferredType::Ptr64 | InferredType::Top => None,
        _ => None,
    }
}

fn default_pointee_ty_for_width(width: usize) -> Option<&'static str> {
    match width {
        1 => Some("uint8_t"),
        2 => Some("uint16_t"),
        4 => Some("uint32_t"),
        8 => Some("uint64_t"),
        _ => None,
    }
}

fn memory_scalar_width_bytes(opcode: &str) -> usize {
    for tok in opcode.split('.') {
        match tok {
            "U8" | "S8" => return 1,
            "U16" | "S16" | "F16" => return 2,
            "U32" | "S32" | "F32" => return 4,
            "U64" | "S64" => return 8,
            _ => {}
        }
    }
    if opcode.contains(".64") || opcode.contains(".128") {
        return 4;
    }
    4
}

fn scalar_pointee_type_from_opcode(opcode: &str) -> Option<&'static str> {
    for tok in opcode.split('.') {
        match tok {
            "U8" => return Some("uint8_t"),
            "S8" => return Some("int8_t"),
            "U16" => return Some("uint16_t"),
            "S16" => return Some("int16_t"),
            "U32" => return Some("uint32_t"),
            "S32" => return Some("int32_t"),
            "U64" => return Some("uint64_t"),
            "S64" => return Some("int64_t"),
            "F16" => return Some("__half"),
            "F32" => return Some("float"),
            _ => {}
        }
    }
    None
}

fn infer_pointer_pointee_ty_from_evidence(types: &BTreeSet<&'static str>) -> Option<&'static str> {
    if types.len() == 1 {
        return types.iter().next().copied();
    }
    for preferred in ["float", "__half"] {
        if !types.contains(preferred) {
            continue;
        }
        let Some(width) = pointee_type_width_bytes(preferred) else {
            continue;
        };
        if types.iter().all(|ty| {
            pointee_type_width_bytes(ty) == Some(width)
                && (*ty == preferred || is_integer_pointee_type(ty))
        }) {
            return Some(preferred);
        }
    }
    let width = pointee_type_width_bytes(*types.iter().next()?)?;
    if types
        .iter()
        .all(|ty| pointee_type_width_bytes(ty) == Some(width) && is_integer_pointee_type(ty))
    {
        return preferred_integer_pointee_type(types, width);
    }
    None
}

fn pointee_type_width_bytes(ty: &str) -> Option<usize> {
    match ty {
        "uint8_t" | "int8_t" => Some(1),
        "uint16_t" | "int16_t" | "__half" => Some(2),
        "uint32_t" | "int32_t" | "float" => Some(4),
        "uint64_t" | "int64_t" => Some(8),
        _ => None,
    }
}

fn is_integer_pointee_type(ty: &str) -> bool {
    matches!(
        ty,
        "uint8_t"
            | "int8_t"
            | "uint16_t"
            | "int16_t"
            | "uint32_t"
            | "int32_t"
            | "uint64_t"
            | "int64_t"
    )
}

fn preferred_integer_pointee_type(
    types: &BTreeSet<&'static str>,
    width: usize,
) -> Option<&'static str> {
    match width {
        1 => {
            if types.contains("int8_t") {
                Some("int8_t")
            } else if types.contains("uint8_t") {
                Some("uint8_t")
            } else {
                None
            }
        }
        2 => {
            if types.contains("int16_t") {
                Some("int16_t")
            } else if types.contains("uint16_t") {
                Some("uint16_t")
            } else {
                None
            }
        }
        4 => {
            if types.contains("int32_t") {
                Some("int32_t")
            } else if types.contains("uint32_t") {
                Some("uint32_t")
            } else {
                None
            }
        }
        8 => {
            if types.contains("int64_t") {
                Some("int64_t")
            } else if types.contains("uint64_t") {
                Some("uint64_t")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn collect_param_loaded_addr64_pair_bases(
    function_ir: &FunctionIR,
    annotations: &AbiAnnotations,
) -> BTreeSet<u32> {
    let flow = build_param_word_flow(function_ir, annotations);

    let mut pair_bases = BTreeSet::new();
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            collect_stmt_param_loaded_addr64_pairs(
                stmt,
                &flow.loaded_param_by_reg,
                &flow.constmem_word_by_pair,
                &flow.copy_source_by_reg,
                &flow.phi_sources_by_reg,
                &flow.def_by_reg,
                &mut pair_bases,
            );
            visit_expr_for_param_loaded_addr64_pairs(
                stmt,
                &flow.loaded_param_by_reg,
                &flow.constmem_word_by_pair,
                &flow.copy_source_by_reg,
                &flow.phi_sources_by_reg,
                &flow.def_by_reg,
                &mut pair_bases,
            );
        }
    }
    pair_bases
}

fn collect_stmt_param_loaded_addr64_pairs(
    stmt: &IRStatement,
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
    pair_bases: &mut BTreeSet<u32>,
) {
    let Some((opcode, args)) = rvalue_op_parts(stmt) else {
        return;
    };
    if let Some(lo) = detect_param_loaded_wide_pair(
        opcode,
        args,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
    ) {
        pair_bases.insert(lo);
    }
    if let Some(lo) = detect_param_loaded_addx_pair(
        opcode,
        args,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
    ) {
        pair_bases.insert(lo);
    }
    if let Some(lo) = detect_param_loaded_lea_hi_pair(
        opcode,
        args,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
    ) {
        pair_bases.insert(lo);
    }
}

fn rvalue_op_parts(stmt: &IRStatement) -> Option<(&str, &[IRExpr])> {
    match &stmt.value {
        RValue::Op { opcode, args } => Some((opcode.as_str(), args.as_slice())),
        _ => None,
    }
}

fn rvalue_args(stmt: &IRStatement) -> &[IRExpr] {
    match &stmt.value {
        RValue::Op { args, .. } | RValue::Phi(args) => args.as_slice(),
        RValue::ImmI(_) | RValue::ImmF(_) => &[],
    }
}

fn constmem_load_word_span(opcode: &str) -> usize {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "ULDC" | "LDC" | "LDCU") {
        return 0;
    }
    if opcode.split('.').any(|tok| tok == "128") {
        4
    } else if opcode.split('.').any(|tok| tok == "64") {
        2
    } else {
        1
    }
}

fn constmem_copy_word_span(opcode: &str, args: &[IRExpr]) -> usize {
    if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        return 1;
    }
    if opcode == "MOV" || opcode.starts_with("MOV.") {
        return 1;
    }
    0
}

fn extract_copy_source_expr(opcode: &str, args: &[IRExpr]) -> Option<RegId> {
    if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        if is_zero_ir_expr(&args[0]) && is_zero_ir_expr(&args[1]) {
            return args[2].get_reg().cloned();
        }
    }
    if (opcode == "MOV" || opcode.starts_with("MOV.")) && args.len() == 1 {
        return args[0].get_reg().cloned();
    }
    None
}

fn is_zero_ir_expr(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::ImmI(value) => *value == 0,
        IRExpr::ImmF(value) => *value == 0.0,
        IRExpr::Reg(reg) => matches!(reg.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn visit_expr_for_param_loaded_addr64_pairs(
    stmt: &IRStatement,
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
    pair_bases: &mut BTreeSet<u32>,
) {
    if let Some(pred) = &stmt.pred {
        scan_expr_for_param_loaded_addr64_pairs(
            pred,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            pair_bases,
        );
    }
    for def in &stmt.defs {
        scan_expr_for_param_loaded_addr64_pairs(
            def,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            pair_bases,
        );
    }
    for arg in rvalue_args(stmt) {
        scan_expr_for_param_loaded_addr64_pairs(
            arg,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            pair_bases,
        );
    }
    if let Some(mem_args) = &stmt.mem_addr_args {
        for arg in mem_args {
            scan_expr_for_param_loaded_addr64_pairs(
                arg,
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                pair_bases,
            );
        }
    }
}

fn scan_expr_for_param_loaded_addr64_pairs(
    expr: &IRExpr,
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
    pair_bases: &mut BTreeSet<u32>,
) {
    match expr {
        IRExpr::Addr64 { lo, hi } => {
            let lo_word = resolve_param_word_from_expr(
                lo.as_ref(),
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                &mut BTreeSet::new(),
            );
            let hi_word = resolve_param_word_from_expr(
                hi.as_ref(),
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                &mut BTreeSet::new(),
            );
            if let (Some(lo_word), Some(hi_word)) = (lo_word, hi_word) {
                if lo_word % 2 == 0 && hi_word == lo_word + 1 {
                    pair_bases.insert(lo_word);
                }
            }
            scan_expr_for_param_loaded_addr64_pairs(
                lo.as_ref(),
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                pair_bases,
            );
            scan_expr_for_param_loaded_addr64_pairs(
                hi.as_ref(),
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                pair_bases,
            );
        }
        IRExpr::Op { args, .. } => {
            for arg in args {
                scan_expr_for_param_loaded_addr64_pairs(
                    arg,
                    loaded_param_by_reg,
                    constmem_word_by_pair,
                    copy_source_by_reg,
                    phi_sources_by_reg,
                    def_by_reg,
                    pair_bases,
                );
            }
        }
        IRExpr::Mem {
            base,
            offset,
            width: _,
        } => {
            scan_expr_for_param_loaded_addr64_pairs(
                base,
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                pair_bases,
            );
            if let Some(offset) = offset {
                scan_expr_for_param_loaded_addr64_pairs(
                    offset,
                    loaded_param_by_reg,
                    constmem_word_by_pair,
                    copy_source_by_reg,
                    phi_sources_by_reg,
                    def_by_reg,
                    pair_bases,
                );
            }
        }
        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
    }
}

fn detect_param_loaded_wide_pair(
    opcode: &str,
    args: &[IRExpr],
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
) -> Option<u32> {
    if opcode.starts_with("IADD.64") && args.len() >= 4 {
        let lhs_pair = resolve_param_pair_base_from_lo_hi_exprs(
            &args[0],
            &args[2],
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut BTreeSet::new(),
        );
        let rhs_pair = resolve_param_pair_base_from_lo_hi_exprs(
            &args[1],
            &args[3],
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut BTreeSet::new(),
        );
        return choose_unique_param_pair_base(lhs_pair, rhs_pair);
    }
    None
}

fn detect_param_loaded_addx_pair(
    opcode: &str,
    args: &[IRExpr],
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
) -> Option<u32> {
    if !(opcode.starts_with("IADD3.X") || opcode.starts_with("UIADD3.X")) {
        return None;
    }

    if args.len() == 3 {
        let mut hi_word = None;
        for arg in args {
            if is_zero_ir_expr(arg) {
                continue;
            }
            let Some(word) = resolve_param_word_from_expr(
                arg,
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                &mut BTreeSet::new(),
            ) else {
                continue;
            };
            match hi_word {
                None => hi_word = Some(word),
                Some(existing) if existing == word => {}
                Some(_) => return None,
            }
        }
        let hi_word = hi_word?;
        return (hi_word % 2 == 1).then_some(hi_word - 1);
    }

    if args.len() < 5 {
        return None;
    }
    let carry_expr = &args[args.len() - 2];
    if !is_pred_reg_expr(carry_expr) {
        return None;
    }
    let lo_word = resolve_param_word_from_expr(
        carry_expr,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
        &mut BTreeSet::new(),
    )?;

    let mut hi_word = None;
    for arg in &args[..args.len() - 2] {
        if is_zero_ir_expr(arg) {
            continue;
        }
        let Some(word) = resolve_param_word_from_expr(
            arg,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut BTreeSet::new(),
        ) else {
            continue;
        };
        if word == lo_word {
            continue;
        }
        match hi_word {
            None => hi_word = Some(word),
            Some(existing) if existing == word => {}
            Some(_) => return None,
        }
    }

    let hi_word = hi_word?;
    if lo_word % 2 == 0 && hi_word == lo_word + 1 {
        Some(lo_word)
    } else {
        None
    }
}

fn detect_param_loaded_lea_hi_pair(
    opcode: &str,
    args: &[IRExpr],
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
) -> Option<u32> {
    if !(opcode.starts_with("LEA.HI.X") || opcode.starts_with("ULEA.HI.X")) {
        return None;
    }

    let (hi_base_expr, carry_expr) = if args.len() >= 5
        && matches!(args[3], IRExpr::ImmI(_))
        && is_pred_reg_expr(&args[4])
    {
        (&args[1], &args[4])
    } else if args.len() == 4 && matches!(args[2], IRExpr::ImmI(_)) && is_pred_reg_expr(&args[3]) {
        (&args[1], &args[3])
    } else {
        return None;
    };

    let lo_word = resolve_param_word_from_expr(
        carry_expr,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
        &mut BTreeSet::new(),
    )?;
    let hi_word = resolve_param_word_from_expr(
        hi_base_expr,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
        &mut BTreeSet::new(),
    )?;

    if lo_word % 2 == 0 && hi_word == lo_word + 1 {
        Some(lo_word)
    } else {
        None
    }
}

fn resolve_param_word_from_expr(
    expr: &IRExpr,
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
    visited: &mut BTreeSet<RegId>,
) -> Option<u32> {
    if let IRExpr::Op { op, args } = expr {
        if op == "ConstMem" && args.len() == 2 {
            let bank = imm_as_u32(&args[0])?;
            let offset = imm_as_u32(&args[1])?;
            if let Some(word) = constmem_word_by_pair.get(&(bank, offset)) {
                return Some(*word);
            }
        }
    }
    let reg = expr.get_reg()?;
    resolve_param_word_from_reg(
        reg,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
        visited,
    )
}

fn resolve_param_pair_base_from_lo_hi_exprs(
    lo_expr: &IRExpr,
    hi_expr: &IRExpr,
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
    visited: &mut BTreeSet<RegId>,
) -> Option<u32> {
    let lo_word = resolve_param_word_from_expr(
        lo_expr,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
        visited,
    )?;
    let hi_word = resolve_param_word_from_expr(
        hi_expr,
        loaded_param_by_reg,
        constmem_word_by_pair,
        copy_source_by_reg,
        phi_sources_by_reg,
        def_by_reg,
        visited,
    )?;
    if lo_word % 2 == 0 && hi_word == lo_word + 1 {
        Some(lo_word)
    } else {
        None
    }
}

fn choose_unique_param_pair_base(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) if x == y => Some(x),
        (Some(x), None) | (None, Some(x)) => Some(x),
        _ => None,
    }
}

fn resolve_param_word_from_reg(
    reg: &RegId,
    loaded_param_by_reg: &BTreeMap<RegId, u32>,
    constmem_word_by_pair: &BTreeMap<(u32, u32), u32>,
    copy_source_by_reg: &BTreeMap<RegId, RegId>,
    phi_sources_by_reg: &BTreeMap<RegId, Vec<IRExpr>>,
    def_by_reg: &BTreeMap<RegId, (String, Vec<IRExpr>, usize)>,
    visited: &mut BTreeSet<RegId>,
) -> Option<u32> {
    if let Some(word) = loaded_param_by_reg.get(reg) {
        return Some(*word);
    }
    if !visited.insert(reg.clone()) {
        return None;
    }
    if let Some(src) = copy_source_by_reg.get(reg) {
        if let Some(word) = resolve_param_word_from_reg(
            src,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            visited,
        ) {
            return Some(word);
        }
    }
    if let Some(phi_args) = phi_sources_by_reg.get(reg) {
        let mut resolved = None;
        for arg in phi_args {
            let Some(word) = resolve_param_word_from_expr(
                arg,
                loaded_param_by_reg,
                constmem_word_by_pair,
                copy_source_by_reg,
                phi_sources_by_reg,
                def_by_reg,
                &mut visited.clone(),
            ) else {
                continue;
            };
            match resolved {
                None => resolved = Some(word),
                Some(existing) if existing == word => {}
                Some(_) => return None,
            }
        }
        if let Some(word) = resolved {
            return Some(word);
        }
    }
    let (opcode, args, def_idx) = def_by_reg.get(reg)?;
    if (opcode.starts_with("IMAD.WIDE") || opcode.starts_with("UIMAD.WIDE")) && *def_idx <= 1 {
        let base_word = resolve_param_word_from_expr(
            args.get(2)?,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            visited,
        )?;
        return Some(base_word + *def_idx as u32);
    }
    if opcode.starts_with("IADD.64") && *def_idx <= 1 && args.len() >= 4 {
        let lhs_pair = resolve_param_pair_base_from_lo_hi_exprs(
            &args[0],
            &args[2],
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut visited.clone(),
        );
        let rhs_pair = resolve_param_pair_base_from_lo_hi_exprs(
            &args[1],
            &args[3],
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut visited.clone(),
        );
        let base_word = choose_unique_param_pair_base(lhs_pair, rhs_pair)?;
        return Some(base_word + *def_idx as u32);
    }
    if (opcode.starts_with("IADD3.X") || opcode.starts_with("UIADD3.X")) && *def_idx == 0 {
        if args.len() == 3 {
            let mut hi_word = None;
            for arg in args {
                if is_zero_ir_expr(arg) {
                    continue;
                }
                let Some(word) = resolve_param_word_from_expr(
                    arg,
                    loaded_param_by_reg,
                    constmem_word_by_pair,
                    copy_source_by_reg,
                    phi_sources_by_reg,
                    def_by_reg,
                    &mut visited.clone(),
                ) else {
                    continue;
                };
                match hi_word {
                    None => hi_word = Some(word),
                    Some(existing) if existing == word => {}
                    Some(_) => return None,
                }
            }
            let hi_word = hi_word?;
            if hi_word % 2 == 1 {
                return Some(hi_word);
            }
        } else if args.len() >= 5 {
            let carry_expr = &args[args.len() - 2];
            if is_pred_reg_expr(carry_expr) {
                let lo_word = resolve_param_word_from_expr(
                    carry_expr,
                    loaded_param_by_reg,
                    constmem_word_by_pair,
                    copy_source_by_reg,
                    phi_sources_by_reg,
                    def_by_reg,
                    &mut visited.clone(),
                )?;
                let mut hi_word = None;
                for arg in &args[..args.len() - 2] {
                    if is_zero_ir_expr(arg) {
                        continue;
                    }
                    let Some(word) = resolve_param_word_from_expr(
                        arg,
                        loaded_param_by_reg,
                        constmem_word_by_pair,
                        copy_source_by_reg,
                        phi_sources_by_reg,
                        def_by_reg,
                        &mut visited.clone(),
                    ) else {
                        continue;
                    };
                    if word == lo_word {
                        continue;
                    }
                    match hi_word {
                        None => hi_word = Some(word),
                        Some(existing) if existing == word => {}
                        Some(_) => return None,
                    }
                }
                let hi_word = hi_word?;
                if lo_word % 2 == 0 && hi_word == lo_word + 1 {
                    return Some(hi_word);
                }
            }
        }
    }
    if (opcode.starts_with("LEA.HI.X") || opcode.starts_with("ULEA.HI.X")) && *def_idx == 0 {
        let (hi_base_expr, carry_expr) = if args.len() >= 5
            && matches!(args[3], IRExpr::ImmI(_))
            && is_pred_reg_expr(&args[4])
        {
            (&args[1], &args[4])
        } else if args.len() == 4
            && matches!(args[2], IRExpr::ImmI(_))
            && is_pred_reg_expr(&args[3])
        {
            (&args[1], &args[3])
        } else {
            return None;
        };
        let lo_word = resolve_param_word_from_expr(
            carry_expr,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut visited.clone(),
        )?;
        let hi_word = resolve_param_word_from_expr(
            hi_base_expr,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            &mut visited.clone(),
        )?;
        if lo_word % 2 == 0 && hi_word == lo_word + 1 {
            return Some(hi_word);
        }
    }
    if !opcode_preserves_param_word(opcode) {
        return None;
    }
    let mut resolved = None;
    for arg in args {
        let Some(word) = resolve_param_word_from_expr(
            arg,
            loaded_param_by_reg,
            constmem_word_by_pair,
            copy_source_by_reg,
            phi_sources_by_reg,
            def_by_reg,
            visited,
        ) else {
            continue;
        };
        match resolved {
            None => resolved = Some(word),
            Some(existing) if existing == word => {}
            Some(_) => return None,
        }
    }
    resolved
}

fn is_pred_reg_expr(expr: &IRExpr) -> bool {
    matches!(
        expr.get_reg(),
        Some(reg) if reg.class == "P" || reg.class == "UP"
    )
}

fn opcode_preserves_param_word(opcode: &str) -> bool {
    opcode.starts_with("IADD.64")
        || opcode.starts_with("IADD3")
        || opcode.starts_with("UIADD3")
        || opcode.starts_with("IMAD.X")
        || opcode.starts_with("LEA")
        || opcode == "MOV"
        || opcode.starts_with("MOV.")
}

fn collect_constmem_from_expr(expr: &IRExpr, out: &mut Vec<(u32, u32)>) {
    match expr {
        IRExpr::Addr64 { lo, hi } => {
            collect_constmem_from_expr(lo, out);
            collect_constmem_from_expr(hi, out);
        }
        IRExpr::Op { op, args } => {
            if op == "ConstMem" && args.len() == 2 {
                if let (Some(bank), Some(offset)) = (imm_as_u32(&args[0]), imm_as_u32(&args[1])) {
                    out.push((bank, offset));
                }
            }
            for arg in args {
                collect_constmem_from_expr(arg, out);
            }
        }
        IRExpr::Mem { base, offset, .. } => {
            collect_constmem_from_expr(base, out);
            if let Some(off) = offset {
                collect_constmem_from_expr(off, out);
            }
        }
        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
    }
}

fn is_pointer_context_opcode(opcode: &str) -> bool {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    let is_memory_data = (opcode.starts_with("LD") || opcode.starts_with("ST"))
        && !matches!(mnem, "LDC" | "LDCU" | "ULDC");
    opcode.starts_with("IMAD.WIDE")
        || opcode.starts_with("UIMAD.WIDE")
        || opcode.starts_with("IADD.64")
        || opcode.starts_with("IADD3")
        || opcode.starts_with("UIADD3")
        || is_memory_data
        || opcode.contains("LEA")
}

fn is_signed_word_context_opcode(opcode: &str) -> bool {
    opcode.starts_with("IABS")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, parser::decode_sass};

    #[test]
    fn detects_legacy_profile_from_param_window() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
            /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x148] ;
            /*0020*/ IADD3 R3, R3, c[0x0][0x154], RZ ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::legacy_param_140());
    }

    #[test]
    fn detects_modern_profile_from_param_window() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ IMAD.WIDE R6, R0, R7, c[0x0][0x168] ;
            /*0020*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x17c] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::modern_param_160());
    }

    #[test]
    fn detect_with_sm_uses_sm_when_offsets_absent() {
        let sass = r#"
            /*0000*/ S2R R0, SR_CTAID.X ;
            /*0010*/ S2R R1, SR_TID.X ;
        "#;
        let instrs = decode_sass(sass);
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(89)),
            AbiProfile::modern_param_160()
        );
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(70)),
            AbiProfile::legacy_param_140()
        );
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(100)),
            AbiProfile::blackwell_param_380()
        );
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(120)),
            AbiProfile::blackwell_param_380()
        );
    }

    #[test]
    fn sm_metadata_beats_overlapping_offset_evidence_for_modern() {
        // A modern (SM 89) kernel that addresses a high-numbered parameter
        // at `c[0x0][0x360]` (modern param index 128) trips the offset-only
        // heuristic into thinking the trace is Blackwell, because that
        // slot collides with the relocated `blockDimX` builtin.  When the
        // SM is known, `detect_with_sm` must trust the architecture name
        // and ignore the misleading overlap entirely.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x360] ;
        "#;
        let instrs = decode_sass(sass);
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(89)),
            AbiProfile::modern_param_160()
        );
        // Sanity-check the failure mode: with SM unknown the offset-only
        // path still picks Blackwell on this isolated hit, which is the
        // exact reason `detect_with_sm` is the preferred entry point.
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, None),
            AbiProfile::blackwell_param_380()
        );
    }

    #[test]
    fn sm_metadata_beats_stray_legacy_window_hits_for_blackwell() {
        // A real Blackwell kernel never reads `c[0x0][0x140/0x160]`, but
        // if some pathological trace ever did, the offset-only veto would
        // suppress Blackwell and pick the older profile.  When the SM is
        // pinned at 100+, `detect_with_sm` must trust the architecture
        // name even in the face of contradictory offset hints.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x140] ;
            /*0010*/ LDC R2, c[0x0][0x37c] ;
        "#;
        let instrs = decode_sass(sass);
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(100)),
            AbiProfile::blackwell_param_380()
        );
        // And without SM, the offset-only veto kicks in and we fall
        // through to the legacy profile — exactly the false negative the
        // SM-priority rule guards against.
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, None),
            AbiProfile::legacy_param_140()
        );
    }

    #[test]
    fn detects_blackwell_profile_from_param_window() {
        // These are the offsets actually observed at the start of an SM 100
        // crypto kernel in the corpus.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x37c] ;
            /*0010*/ LDCU UR5, c[0x0][0x360] ;
            /*0020*/ LDC.64 R2, c[0x0][0x380] ;
            /*0030*/ LDCU.64 UR6, c[0x0][0x358] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::blackwell_param_380());
    }

    #[test]
    fn modern_wins_over_overlapping_blackwell_slots() {
        // The "Blackwell-unique" slots (`0x358`, `0x360..0x378`, `0x37c`)
        // are only unique relative to the built-in tables of older
        // profiles.  Under the modern profile `resolve_constmem` will
        // still classify these offsets as far parameter words (modern
        // indices 126, 128..133, 135) so a 128+-param SM 80/89 kernel
        // could legitimately address them.  The presence of any modern
        // window hit must therefore force the modern profile, even when
        // those slots are also touched.
        let sass = r#"
            /*0000*/ LDCU UR5, c[0x0][0x360] ;
            /*0010*/ LDC R1, c[0x0][0x37c] ;
            /*0020*/ LDC R2, c[0x0][0x160] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::modern_param_160());
    }

    #[test]
    fn blackwell_wins_when_only_blackwell_internals_appear() {
        // A minimal Blackwell function may not touch any of the
        // `0x380+` parameter slots (e.g. a void(void) kernel that only
        // reads the frame pointer).  The lone `0x37c` hit must still
        // resolve to Blackwell, otherwise the SM-fallback path would
        // never trigger.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x37c] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::blackwell_param_380());
    }

    #[test]
    fn offset_0x378_is_not_blackwell_unique() {
        // The Blackwell built-in block ends at `0x378` (gridDimZ lives
        // at `0x374`), so an isolated hit at `0x378` should NOT be
        // treated as Blackwell-unique evidence.  Without any other
        // signal, profile detection should fall through to `None`.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x378] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        // Default fallback is the modern profile; the important thing
        // is that we did NOT pick Blackwell on a single 0x378 hit.
        assert_ne!(p, AbiProfile::blackwell_param_380());
    }

    #[test]
    fn modern_window_wins_when_no_blackwell_unique_evidence() {
        // A modern (SM 80/89) kernel with a huge parameter list can put a
        // load at c[0x0][0x380] (param_0x220 from base 0x160).  Without
        // any Blackwell-unique evidence we must keep treating it as a
        // modern kernel; otherwise the relocated builtin map would
        // mis-resolve every constmem hit.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x160] ;
            /*0010*/ LDC R2, c[0x0][0x164] ;
            /*0020*/ LDC R3, c[0x0][0x380] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::modern_param_160());
    }

    #[test]
    fn legacy_window_wins_when_no_blackwell_unique_evidence() {
        // Same idea as above for the legacy window: a 0x380 hit alone is
        // not enough to override a clearly-legacy parameter pattern.
        let sass = r#"
            /*0000*/ LDC R1, c[0x0][0x140] ;
            /*0010*/ LDC R2, c[0x0][0x144] ;
            /*0020*/ LDC R3, c[0x0][0x380] ;
        "#;
        let instrs = decode_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::legacy_param_140());
    }

    #[test]
    fn resolves_blackwell_builtins_and_internals() {
        let p = AbiProfile::blackwell_param_380();
        // Built-in dimensions live at 0x360+.
        assert_eq!(p.resolve_constmem(0, 0x360).unwrap().symbol, "blockDim.x");
        assert_eq!(p.resolve_constmem(0, 0x364).unwrap().symbol, "blockDim.y");
        assert_eq!(p.resolve_constmem(0, 0x368).unwrap().symbol, "blockDim.z");
        assert_eq!(p.resolve_constmem(0, 0x36c).unwrap().symbol, "gridDim.x");
        assert_eq!(p.resolve_constmem(0, 0x370).unwrap().symbol, "gridDim.y");
        assert_eq!(p.resolve_constmem(0, 0x374).unwrap().symbol, "gridDim.z");
        // ABI internals relocated for Blackwell.
        assert_eq!(
            p.resolve_constmem(0, 0x358).unwrap().symbol,
            "abi_internal_0x358"
        );
        assert_eq!(
            p.resolve_constmem(0, 0x37c).unwrap().symbol,
            "abi_internal_0x37c"
        );
        // Old legacy slots must NOT resolve under Blackwell.
        assert!(p.resolve_constmem(0, 0x0).is_none());
        assert!(p.resolve_constmem(0, 0x28).is_none());
        assert!(p.resolve_constmem(0, 0x44).is_none());
        // Params start at 0x380.
        assert_eq!(p.resolve_constmem(0, 0x380).unwrap().symbol, "param_0");
        assert_eq!(p.resolve_constmem(0, 0x384).unwrap().symbol, "param_1");
        assert_eq!(p.resolve_constmem(0, 0x388).unwrap().symbol, "param_2");
    }

    #[test]
    fn legacy_profile_still_resolves_builtins_at_zero() {
        // Double-check the relocation gate doesn't break older generations.
        let p = AbiProfile::modern_param_160();
        assert_eq!(p.resolve_constmem(0, 0x0).unwrap().symbol, "blockDim.x");
        assert_eq!(p.resolve_constmem(0, 0x4).unwrap().symbol, "blockDim.y");
        // Under the modern profile, 0x360 must NOT be misclassified as a
        // Blackwell built-in — it should look like a far-away param slot.
        let sym = p.resolve_constmem(0, 0x360).unwrap().symbol;
        assert!(sym.starts_with("param_"), "got {}", sym);
        assert_ne!(sym, "blockDim.x");
    }

    #[test]
    fn profile_selection_changes_param_indexing() {
        let legacy = AbiProfile::legacy_param_140();
        let modern = AbiProfile::modern_param_160();
        // legacy param_base = 0x140, so offset 0x160 is word (0x160-0x140)/4 = 8
        assert!(matches!(
            legacy.classify_constmem(0, 0x160),
            ConstMemSemantic::ParamWord {
                param_idx: 8,
                word_idx: 0
            }
        ));
        // modern param_base = 0x160, so offset 0x160 is word 0
        assert!(matches!(
            modern.classify_constmem(0, 0x160),
            ConstMemSemantic::ParamWord {
                param_idx: 0,
                word_idx: 0
            }
        ));
    }

    #[test]
    fn resolves_user_asked_offsets_in_legacy_profile() {
        let p = AbiProfile::legacy_param_140();
        assert_eq!(p.resolve_constmem(0, 0x0).unwrap().symbol, "blockDim.x");
        assert_eq!(p.resolve_constmem(0, 0x8).unwrap().symbol, "blockDim.z");
        assert_eq!(
            p.resolve_constmem(0, 0x28).unwrap().symbol,
            "abi_internal_0x28"
        );
        assert_eq!(
            p.resolve_constmem(0, 0x44).unwrap().symbol,
            "abi_internal_0x44"
        );
        // Per-word param indexing: each 4-byte word is its own param.
        assert_eq!(p.resolve_constmem(0, 0x140).unwrap().symbol, "param_0");
        assert_eq!(p.resolve_constmem(0, 0x144).unwrap().symbol, "param_1");
        assert_eq!(p.resolve_constmem(0, 0x148).unwrap().symbol, "param_2");
        assert_eq!(p.resolve_constmem(0, 0x14c).unwrap().symbol, "param_3");
        assert_eq!(p.resolve_constmem(0, 0x150).unwrap().symbol, "param_4");
        assert_eq!(p.resolve_constmem(0, 0x154).unwrap().symbol, "param_5");
    }

    #[test]
    fn abi_display_formats_constmem_symbolically() {
        let display = AbiDisplay::new(AbiProfile::legacy_param_140());
        let expr = IRExpr::Op {
            op: "ConstMem".to_string(),
            args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x148)],
        };
        assert_eq!(display.expr(&expr), "param_2");
    }

    #[test]
    fn infers_pointer_alias_for_u64_param_used_in_wide_ops() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ IADD3.X R5, R5, c[0x0][0x164], RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let alias = aliases.by_param.get(&0).expect("missing alias for param 0");
        assert_eq!(alias.kind, ArgAliasKind::Ptr64);
        assert_eq!(alias.confidence, AliasConfidence::High);
        assert_eq!(aliases.render_param_word(0, 0).unwrap(), "arg0_ptr.lo32");
        assert_eq!(aliases.render_param_word(0, 1).unwrap(), "arg0_ptr.hi32");
    }

    #[test]
    fn infers_pointer_alias_for_iadd3_plus_lea_hi_pair() {
        let sass = r#"
            /*0000*/ IADD3 R10, P2, R6, c[0x0][0x160], RZ ;
            /*0010*/ LEA.HI.X.SX32 R11, R6, c[0x0][0x164], 0x1, P2 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let alias = aliases.by_param.get(&0).expect("missing alias for param 0");
        assert_eq!(alias.kind, ArgAliasKind::Ptr64);
    }

    #[test]
    fn infers_pointer_alias_for_imad_wide_base_used_by_memory_op() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R6, R3, 0x4, c[0x0][0x160] ;
            /*0010*/ LDG.E.U32 R11, [R6.64] ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let pairs = collect_param_loaded_addr64_pair_bases(&fir, &anns);
        assert!(pairs.contains(&0), "expected arg0 pair, got {pairs:?}");
        let aliases = infer_arg_aliases(&fir, &anns);
        let alias = aliases.by_param.get(&0).expect("missing alias for param 0");
        assert_eq!(alias.kind, ArgAliasKind::Ptr64);
        assert_eq!(aliases.render_param_word(0, 0).unwrap(), "arg0_ptr.lo32");
        assert_eq!(aliases.render_param_word(0, 1).unwrap(), "arg0_ptr.hi32");
    }

    #[test]
    fn infers_pointer_alias_for_atomic_with_implicit_global_addr_pair() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R6, R3, 0x4, c[0x0][0x160] ;
            /*0010*/ ATOMG.E.CAS.STRONG.GPU PT, R4, [R6], -0x1, R5 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let pairs = collect_param_loaded_addr64_pair_bases(&fir, &anns);
        assert!(pairs.contains(&0), "expected arg0 pair, got {pairs:?}");
        let aliases = infer_arg_aliases(&fir, &anns);
        let alias = aliases.by_param.get(&0).expect("missing alias for param 0");
        assert_eq!(alias.kind, ArgAliasKind::Ptr64);
    }

    #[test]
    fn infers_pointer_alias_for_ldc64_pointer_pair_used_by_memory_op() {
        let sass = r#"
            /*0000*/ LDC.64 R2, c[0x0][0x160] ;
            /*0010*/ IMAD.WIDE R2, R0, 0x4, R2 ;
            /*0020*/ LDG.E.U32 R8, [R2.64] ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let alias = aliases.by_param.get(&0).expect("missing alias for param 0");
        assert_eq!(alias.kind, ArgAliasKind::Ptr64);
        assert_eq!(alias.confidence, AliasConfidence::High);
        assert_eq!(alias.pointee_ty, Some("uint32_t"));
    }

    #[test]
    fn infers_pointer_alias_for_ldcu64_lea_pair() {
        let sass = r#"
            /*0000*/ LDCU.64 UR8, c[0x0][0x160] ;
            /*0010*/ LEA R18, P0, R4, UR8, 0x2 ;
            /*0020*/ LEA.HI.X R19, R4, UR9, R5, 0x2, P0 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let alias = aliases.by_param.get(&0).expect("missing alias for param 0");
        assert_eq!(alias.kind, ArgAliasKind::Ptr64);
        assert_eq!(alias.confidence, AliasConfidence::High);
    }

    #[test]
    fn topk_per_row_ldcu_pointer_pairs_infer_ptr64_aliases() {
        let funcs = crate::parser::split_decoded_functions(include_str!(
            "../test_cu/corpus_sm100/ml_kernels.sass"
        ));
        let func = funcs
            .into_iter()
            .find(|func| func.name == "topk_per_row")
            .expect("missing topk_per_row fixture");
        let cfg = build_cfg(func.instrs.clone());
        let fir = build_ssa(&cfg);
        let profile = AbiProfile::detect_with_sm(&func.instrs, func.sm);
        let anns = annotate_function_ir_constmem(&fir, profile);
        let pairs = collect_param_loaded_addr64_pair_bases(&fir, &anns);
        assert!(pairs.contains(&0), "expected arg0 pair, got {pairs:?}");
        assert!(pairs.contains(&4), "expected arg4 pair, got {pairs:?}");

        let aliases = infer_arg_aliases(&fir, &anns);
        assert_eq!(
            aliases.by_param.get(&0).map(|a| a.kind),
            Some(ArgAliasKind::Ptr64)
        );
        assert_eq!(
            aliases.by_param.get(&4).map(|a| a.kind),
            Some(ArgAliasKind::Ptr64)
        );
    }

    #[test]
    fn topk_per_row_infers_pointer_pointee_types() {
        let funcs = crate::parser::split_decoded_functions(include_str!(
            "../test_cu/corpus_sm120/ml_kernels.sass"
        ));
        let func = funcs
            .into_iter()
            .find(|func| func.name == "topk_per_row")
            .expect("missing topk_per_row fixture");
        let cfg = build_cfg(func.instrs.clone());
        let fir = build_ssa(&cfg);
        let profile = AbiProfile::detect_with_sm(&func.instrs, func.sm);
        let anns = annotate_function_ir_constmem(&fir, profile);
        let aliases = infer_arg_aliases(&fir, &anns);

        assert_eq!(
            aliases.by_param.get(&0).and_then(|a| a.pointee_ty),
            Some("float")
        );
        assert_eq!(
            aliases.by_param.get(&2).and_then(|a| a.pointee_ty),
            Some("float")
        );
        assert_eq!(
            aliases.by_param.get(&4).and_then(|a| a.pointee_ty),
            Some("int32_t")
        );
    }

    #[test]
    fn mixed_lo_hi_usage_does_not_force_ptr64_alias() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x180] ;
            /*0010*/ ISETP.GE.AND P0, PT, R1, c[0x0][0x184], PT ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        // With per-word indexing: c[0x180] is param 8, c[0x184] is param 9.
        // They are in the same even/odd pair (8/9).  Word 8 has pointer-like
        // context (IMAD.WIDE) but word 9 does not (ISETP).  Since only one
        // of the pair is pointer-like, they must NOT be merged into Ptr64.
        let alias8 = aliases.by_param.get(&8).expect("missing alias for param 8");
        assert_eq!(alias8.kind, ArgAliasKind::Word32);
        let alias9 = aliases.by_param.get(&9).expect("missing alias for param 9");
        assert_eq!(alias9.kind, ArgAliasKind::Word32);
        assert_eq!(aliases.render_param_word(8, 0).unwrap(), "arg8");
        assert_eq!(aliases.render_param_word(9, 0).unwrap(), "arg9");
    }

    #[test]
    fn nbody_scalar_param_pair_does_not_merge_into_ptr64() {
        let funcs = crate::parser::split_decoded_functions(include_str!(
            "../test_cu/corpus_sm100/simulation_kernels.sass"
        ));
        let func = funcs
            .into_iter()
            .find(|func| func.name == "nbody_forces")
            .expect("missing nbody_forces fixture");
        let cfg = build_cfg(func.instrs.clone());
        let fir = build_ssa(&cfg);
        let profile = AbiProfile::detect_with_sm(&func.instrs, func.sm);
        let anns = annotate_function_ir_constmem(&fir, profile);
        let pairs = collect_param_loaded_addr64_pair_bases(&fir, &anns);
        assert!(
            pairs.contains(&0),
            "expected arg0 pointer pair, got {pairs:?}"
        );
        assert!(
            pairs.contains(&2),
            "expected arg2 pointer pair, got {pairs:?}"
        );
        assert!(
            !pairs.contains(&4),
            "scalar N/eps2 words were misclassified as a pointer pair: {pairs:?}"
        );

        let aliases = infer_arg_aliases(&fir, &anns);
        let alias4 = aliases.by_param.get(&4).expect("missing alias for param 4");
        let alias5 = aliases.by_param.get(&5).expect("missing alias for param 5");
        assert_eq!(alias4.kind, ArgAliasKind::Word32);
        assert_eq!(alias5.kind, ArgAliasKind::Word32);
    }

    #[test]
    fn infers_real_power_series_scalar_param_types() {
        let funcs = crate::parser::split_decoded_functions(include_str!(
            "../test_cu/corpus/loop_kernels.sass"
        ));
        let func = funcs
            .into_iter()
            .find(|func| func.name == "power_series")
            .expect("missing power_series fixture");
        let cfg = build_cfg(func.instrs.clone());
        let fir = build_ssa(&cfg);
        let profile = AbiProfile::detect_with_sm(&func.instrs, func.sm);
        let anns = annotate_function_ir_constmem(&fir, profile);
        let aliases = infer_arg_aliases(&fir, &anns);

        assert_eq!(
            aliases.by_param.get(&0).and_then(|alias| alias.scalar_kind),
            Some(ArgScalarKind::F32)
        );
        assert_eq!(
            aliases.by_param.get(&1).and_then(|alias| alias.scalar_kind),
            Some(ArgScalarKind::I32)
        );
        assert!(aliases
            .render_typed_param_list()
            .contains(&"float arg0".to_string()));
        assert!(aliases
            .render_typed_param_list()
            .contains(&"int32_t arg1".to_string()));
    }

    #[test]
    fn infers_real_pic_charge_deposit_float_params() {
        let funcs = crate::parser::split_decoded_functions(include_str!(
            "../test_cu/corpus/simulation_kernels.sass"
        ));
        let func = funcs
            .into_iter()
            .find(|func| func.name == "pic_charge_deposit")
            .expect("missing pic_charge_deposit fixture");
        let cfg = build_cfg(func.instrs.clone());
        let fir = build_ssa(&cfg);
        let profile = AbiProfile::detect_with_sm(&func.instrs, func.sm);
        let anns = annotate_function_ir_constmem(&fir, profile);
        let aliases = infer_arg_aliases(&fir, &anns);

        assert_eq!(
            aliases
                .by_param
                .get(&11)
                .and_then(|alias| alias.scalar_kind),
            Some(ArgScalarKind::F32)
        );
        assert_eq!(
            aliases
                .by_param
                .get(&12)
                .and_then(|alias| alias.scalar_kind),
            Some(ArgScalarKind::F32)
        );
        let params = aliases.render_typed_param_list();
        assert!(params.contains(&"float arg11".to_string()));
        assert!(params.contains(&"float arg12".to_string()));
    }

    #[test]
    fn abi_display_prefers_inferred_arg_aliases_for_param_words() {
        let mut aliases = AbiArgAliases::default();
        // Legacy param_base = 0x140. Offset 0x148 → word 2.
        // Put a Word32 alias at param_idx=2 so the display picks it up.
        aliases.by_param.insert(
            2,
            ArgAlias {
                param_idx: 2,
                kind: ArgAliasKind::Word32,
                confidence: AliasConfidence::Low,
                observed_words: [0].into_iter().collect(),
                scalar_kind: Some(ArgScalarKind::U32),
                signed_words: BTreeSet::new(),
                pointee_ty: None,
            },
        );
        let display = AbiDisplay::with_aliases(AbiProfile::legacy_param_140(), aliases);
        let expr = IRExpr::Op {
            op: "ConstMem".to_string(),
            args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x148)],
        };
        assert_eq!(display.expr(&expr), "arg2");
    }

    #[test]
    fn renders_typed_arg_declarations_from_aliases() {
        let mut aliases = AbiArgAliases::default();
        aliases.by_param.insert(
            0,
            ArgAlias {
                param_idx: 0,
                kind: ArgAliasKind::Ptr64,
                confidence: AliasConfidence::High,
                observed_words: [0, 1].into_iter().collect(),
                scalar_kind: None,
                signed_words: BTreeSet::new(),
                pointee_ty: None,
            },
        );
        aliases.by_param.insert(
            3,
            ArgAlias {
                param_idx: 3,
                kind: ArgAliasKind::Word32,
                confidence: AliasConfidence::Low,
                observed_words: [0].into_iter().collect(),
                scalar_kind: Some(ArgScalarKind::U32),
                signed_words: BTreeSet::new(),
                pointee_ty: None,
            },
        );

        let decls = aliases.render_typed_arg_declarations();
        assert!(decls.iter().any(|d| d.contains("uintptr_t arg0_ptr;")));
        assert!(decls.iter().any(|d| d.contains("uint32_t arg3;")));
    }

    #[test]
    fn renders_typed_param_list_from_aliases() {
        let mut aliases = AbiArgAliases::default();
        aliases.by_param.insert(
            0,
            ArgAlias {
                param_idx: 0,
                kind: ArgAliasKind::Ptr64,
                confidence: AliasConfidence::High,
                observed_words: [0, 1].into_iter().collect(),
                scalar_kind: None,
                signed_words: BTreeSet::new(),
                pointee_ty: None,
            },
        );
        aliases.by_param.insert(
            2,
            ArgAlias {
                param_idx: 2,
                kind: ArgAliasKind::Word32,
                confidence: AliasConfidence::Low,
                observed_words: [0].into_iter().collect(),
                scalar_kind: Some(ArgScalarKind::U32),
                signed_words: BTreeSet::new(),
                pointee_ty: None,
            },
        );
        let params = aliases.render_typed_param_list();
        assert_eq!(params[0], "uintptr_t arg0_ptr");
        assert_eq!(params[1], "uint32_t arg2");
    }

    #[test]
    fn infers_pointer_pointee_type_from_memory_width_usage() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ IADD3.X R5, R5, c[0x0][0x164], RZ ;
            /*0020*/ LDG.E.U8 R8, [R4.64] ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let params = aliases.render_typed_param_list();
        assert!(params.iter().any(|p| p == "uint8_t* arg0_ptr"));
    }

    #[test]
    fn infers_float_pointer_from_shared_roundtrip_consumed_as_float() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R4, RZ, RZ, c[0x0][0x160] ;
            /*0010*/ IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x164] ;
            /*0020*/ LDG.E.CONSTANT R8, [R4.64] ;
            /*0030*/ STS [R0], R8 ;
            /*0040*/ LDS R9, [R0] ;
            /*0050*/ FADD R10, R9, 0f00000000 ;
            /*0060*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let params = aliases.render_typed_param_list();
        assert!(
            params.iter().any(|p| p == "float* arg0_ptr"),
            "expected float pointee recovered from LDS->FADD shared-memory roundtrip, got {:?}",
            params
        );
    }

    #[test]
    fn infers_float_store_pointer_from_shared_roundtrip_through_copy_chain() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R4, RZ, RZ, c[0x0][0x160] ;
            /*0010*/ IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x164] ;
            /*0020*/ IMAD.MOV.U32 R6, RZ, RZ, c[0x0][0x168] ;
            /*0030*/ IMAD.MOV.U32 R7, RZ, RZ, c[0x0][0x16c] ;
            /*0040*/ LDG.E.CONSTANT R8, [R4.64] ;
            /*0050*/ STS [R0], R8 ;
            /*0060*/ LDS R9, [R0] ;
            /*0070*/ MOV R10, R9 ;
            /*0080*/ FADD R11, R9, 0f00000000 ;
            /*0090*/ STG.E [R6.64], R10 ;
            /*00a0*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let params = aliases.render_typed_param_list();
        assert!(
            params.iter().any(|p| p == "float* arg2_ptr"),
            "expected float pointee recovered through LDS->MOV->STG shared-memory roundtrip, got {:?}",
            params
        );
    }

    #[test]
    fn abi_aware_local_decl_inference_marks_pointer_arith_dest() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R8, R0, c[0x0][0x160], R1 ;
            /*0010*/ IADD3.X R9, R9, c[0x0][0x164], RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::modern_param_160());
        let aliases = infer_arg_aliases(&fir, &anns);
        let decls = infer_local_typed_declarations_with_abi(&fir, Some(&anns), Some(&aliases));
        assert!(decls.iter().any(|d| d == "uintptr_t R8;"));
    }

    #[test]
    fn infers_local_typed_declarations_from_ssa_dests() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ IADD3 R1, R1, 0x1, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let decls = infer_local_typed_declarations(&fir);
        assert!(decls.iter().any(|d| d == "bool P0;"));
        assert!(decls.iter().any(|d| d == "uint32_t R1;"));
    }

    #[test]
    fn infers_richer_local_types_from_opcode_hints() {
        let sass = r#"
            /*0000*/ I2F.RP R2, R1 ;
            /*0010*/ FMUL R3, R2, R2 ;
            /*0020*/ LDS.U8 R4, [UR0] ;
            /*0030*/ IMAD.WIDE R8, R0, 0x4, RZ ;
            /*0040*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let decls = infer_local_typed_declarations(&fir);
        assert!(decls.iter().any(|d| d == "float R2;"));
        assert!(decls.iter().any(|d| d == "float R3;"));
        assert!(decls.iter().any(|d| d == "uint8_t R4;"));
        assert!(decls.iter().any(|d| d == "uintptr_t R8;"));
    }

    #[test]
    fn annotates_ir_with_typed_constmem_semantics() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
            /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x0] ;
            /*0020*/ IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x44] ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::legacy_param_140());
        assert!(!anns.is_empty());

        let mut saw_param = false;
        let mut saw_builtin = false;
        let mut saw_internal = false;
        for (_stmt, entries) in anns.iter() {
            for e in entries {
                match e.semantic {
                    ConstMemSemantic::ParamWord {
                        param_idx: 0,
                        word_idx: 0,
                    } => saw_param = true,
                    ConstMemSemantic::Builtin("blockDim.x") => saw_builtin = true,
                    ConstMemSemantic::AbiInternal(0x44) => saw_internal = true,
                    _ => {}
                }
            }
        }
        assert!(saw_param);
        assert!(saw_builtin);
        assert!(saw_internal);
    }
}
