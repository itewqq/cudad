//! ABI profile and constant-memory parameter mapping for SASS decompilation.
//! This module is intentionally conservative: it maps well-known offsets to
//! symbolic names and falls back to raw `ConstMem(bank, off)` semantics.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{DisplayCtx, FunctionIR, IRExpr, RegId, RValue};
use crate::parser::{Instruction, Operand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiGeneration {
    /// Common older layout with parameter window beginning near c[0x0][0x140].
    LegacyParam140,
    /// Common newer layout with parameter window beginning near c[0x0][0x160].
    ModernParam160,
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
            ConstMemSemantic::ParamWord { param_idx, word_idx } => {
                format!("param_{}[{}]", param_idx, word_idx)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgAlias {
    pub param_idx: u32,
    pub kind: ArgAliasKind,
    pub confidence: AliasConfidence,
    pub observed_words: BTreeSet<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbiArgAliases {
    pub by_param: BTreeMap<u32, ArgAlias>,
}

impl AbiAnnotations {
    pub fn is_empty(&self) -> bool {
        self.constmem_by_stmt.is_empty()
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (&StatementRef, &Vec<ConstMemAnnotation>)> + '_ {
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

    pub fn render_param_word(&self, param_idx: u32, word_idx: u32) -> Option<String> {
        let alias = self.by_param.get(&param_idx)?;
        let rendered = match alias.kind {
            ArgAliasKind::Ptr64 => {
                let lane = if word_idx == 0 { "lo32" } else { "hi32" };
                format!("arg{}_ptr.{}", param_idx, lane)
            }
            ArgAliasKind::U64 => {
                let lane = if word_idx == 0 { "lo32" } else { "hi32" };
                format!("arg{}_u64.{}", param_idx, lane)
            }
            ArgAliasKind::Word32 => format!("arg{}_word{}", param_idx, word_idx),
        };
        Some(rendered)
    }

    pub fn summarize_lines(&self, max_lines: usize) -> Vec<String> {
        self.by_param
            .values()
            .map(|alias| {
                let kind = match alias.kind {
                    ArgAliasKind::Ptr64 => "ptr64",
                    ArgAliasKind::U64 => "u64",
                    ArgAliasKind::Word32 => "word32",
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
                    out.push(format!(
                        "uintptr_t arg{}_ptr; // confidence: {}",
                        alias.param_idx,
                        alias.confidence.as_str()
                    ));
                }
                ArgAliasKind::U64 => {
                    out.push(format!(
                        "uint64_t arg{}_u64; // confidence: {}",
                        alias.param_idx,
                        alias.confidence.as_str()
                    ));
                }
                ArgAliasKind::Word32 => {
                    for w in &alias.observed_words {
                        out.push(format!(
                            "uint32_t arg{}_word{}; // confidence: {}",
                            alias.param_idx,
                            w,
                            alias.confidence.as_str()
                        ));
                    }
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
                    out.push(format!("uintptr_t arg{}_ptr", alias.param_idx));
                }
                ArgAliasKind::U64 => {
                    out.push(format!("uint64_t arg{}_u64", alias.param_idx));
                }
                ArgAliasKind::Word32 => {
                    for w in &alias.observed_words {
                        out.push(format!("uint32_t arg{}_word{}", alias.param_idx, w));
                    }
                }
            }
        }
        out
    }
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
    let mut regs: BTreeSet<(String, i32)> = BTreeSet::new();
    let mut hints: BTreeMap<(String, i32), BTreeSet<LocalTypeHint>> = BTreeMap::new();

    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            if let Some(IRExpr::Reg(r)) = &stmt.dest {
                if is_immutable_reg(r) {
                    continue;
                }
                let key = (r.class.clone(), r.idx);
                regs.insert(key.clone());
                if let Some(h) = hint_from_dest_stmt(stmt) {
                    hints.entry(key).or_default().insert(h);
                }
            }
            collect_pointer_hints_from_stmt(stmt, &mut hints);
        }
    }

    regs.into_iter()
        .map(|(class, idx)| {
            if class == "P" || class == "UP" {
                format!("bool {}{};", class, idx)
            } else {
                let ty = select_local_decl_type(hints.get(&(class.clone(), idx)));
                format!("{} {}{};", ty, class, idx)
            }
        })
        .collect()
}

fn hint_from_dest_stmt(stmt: &crate::ir::IRStatement) -> Option<LocalTypeHint> {
    let RValue::Op { opcode, .. } = &stmt.value else {
        return None;
    };
    let Some(IRExpr::Reg(dest)) = &stmt.dest else {
        return None;
    };
    if dest.class == "P" || dest.class == "UP" {
        return None;
    }
    hint_from_dest_opcode(opcode)
}

fn hint_from_dest_opcode(opcode: &str) -> Option<LocalTypeHint> {
    if opcode.starts_with("IMAD.WIDE") || opcode == "LEA" {
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
    stmt: &crate::ir::IRStatement,
    hints: &mut BTreeMap<(String, i32), BTreeSet<LocalTypeHint>>,
) {
    if let Some(dest) = &stmt.dest {
        collect_pointer_hints_from_expr(dest, hints);
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
        IRExpr::Mem { base, offset, width } => {
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
    let IRExpr::Reg(r) = base else {
        return;
    };
    if is_immutable_reg(r) || matches!(r.class.as_str(), "P" | "UP" | "PT" | "UPT") {
        return;
    }
    hints
        .entry((r.class.clone(), r.idx))
        .or_default()
        .insert(LocalTypeHint::PtrWeak);
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
            ConstMemKind::ParamWord { param_idx, word_idx } => {
                ConstMemSemantic::ParamWord { param_idx, word_idx }
            }
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

    /// Detect the best-effort ABI profile from observed constant-memory offsets.
    /// This keeps current behavior for callers without metadata.
    pub fn detect(instructions: &[Instruction]) -> Self {
        Self::detect_with_sm(instructions, None)
    }

    /// Detect profile using offsets first, then optional SM metadata fallback.
    ///
    /// Fallback rule (when no offset evidence exists):
    /// - `sm >= 80` -> modern parameter base (`0x160`)
    /// - otherwise -> legacy parameter base (`0x140`)
    pub fn detect_with_sm(instructions: &[Instruction], sm: Option<u32>) -> Self {
        if let Some(by_offset) = Self::detect_from_offsets(instructions) {
            return by_offset;
        }

        if let Some(sm_val) = sm {
            if sm_val >= 80 {
                return Self::modern_param_160();
            }
            return Self::legacy_param_140();
        }

        // Default keeps existing behavior and current fixtures stable.
        Self::modern_param_160()
    }

    fn detect_from_offsets(instructions: &[Instruction]) -> Option<Self> {
        let mut near_140_hits = 0usize;
        let mut near_160_hits = 0usize;

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
                });
            }
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

        let builtin = match offset {
            0x8 => Some("blockDimX"),
            0xc => Some("blockDimY"),
            0x10 => Some("blockDimZ"),
            0x14 => Some("gridDimX"),
            0x18 => Some("gridDimY"),
            0x1c => Some("gridDimZ"),
            _ => None,
        };
        if let Some(name) = builtin {
            return Some(ResolvedConstMem {
                symbol: name.to_string(),
                kind: ConstMemKind::Builtin(name),
            });
        }

        if offset >= self.param_base && (offset - self.param_base) % 4 == 0 {
            let word = (offset - self.param_base) / 4;
            let param_idx = word / 2;
            let word_idx = word % 2;
            return Some(ResolvedConstMem {
                symbol: format!("param_{}[{}]", param_idx, word_idx),
                kind: ConstMemKind::ParamWord { param_idx, word_idx },
            });
        }

        if matches!(offset, 0x28 | 0x44) {
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
        Self { profile, aliases: None }
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
        if let ConstMemKind::ParamWord { param_idx, word_idx } = resolved.kind {
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
            IRExpr::Mem { base, offset, width } => {
                let mut s = format!("*{}", self.expr(base));
                if let Some(off) = offset {
                    s.push_str(&format!("+{}", self.expr(off)));
                }
                if let Some(w) = width {
                    s.push_str(&format!("@{}", w));
                }
                s
            }
            IRExpr::Op { op, args } => {
                if let Some(sym) = self.try_constmem_symbol(op, args) {
                    return sym;
                }
                let list = args.iter().map(|a| self.expr(a)).collect::<Vec<_>>().join(", ");
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

fn collect_constmem_hits<F>(op: &Operand, f: &mut F)
where
    F: FnMut(u32, u32),
{
    match op {
        Operand::ConstMem { bank, offset } => f(*bank, *offset),
        Operand::MemRef { base, .. } => collect_constmem_hits(base.as_ref(), f),
        _ => {}
    }
}

pub fn annotate_function_ir_constmem(function_ir: &FunctionIR, profile: AbiProfile) -> AbiAnnotations {
    let mut out = AbiAnnotations::default();
    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let mut raw_pairs = Vec::new();
            if let Some(dest) = &stmt.dest {
                collect_constmem_from_expr(dest, &mut raw_pairs);
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
    #[derive(Default)]
    struct ParamUsage {
        words: BTreeSet<u32>,
        pointer_like_hits: usize,
        total_hits: usize,
    }

    let mut opcode_by_stmt: BTreeMap<StatementRef, String> = BTreeMap::new();
    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let opcode = match &stmt.value {
                RValue::Op { opcode, .. } => opcode.clone(),
                RValue::Phi(_) => "phi".to_string(),
                RValue::ImmI(_) | RValue::ImmF(_) => "imm".to_string(),
            };
            opcode_by_stmt.insert(
                StatementRef {
                    block_id: block.id,
                    stmt_idx,
                },
                opcode,
            );
        }
    }

    let mut by_param: BTreeMap<u32, ParamUsage> = BTreeMap::new();
    for (stmt_ref, anns) in annotations.iter() {
        let opcode = opcode_by_stmt
            .get(stmt_ref)
            .map(String::as_str)
            .unwrap_or("");
        for ann in anns {
            if let ConstMemSemantic::ParamWord { param_idx, word_idx } = ann.semantic {
                let usage = by_param.entry(param_idx).or_default();
                usage.words.insert(word_idx);
                usage.total_hits += 1;
                if is_pointer_context_opcode(opcode) {
                    usage.pointer_like_hits += 1;
                }
            }
        }
    }

    let mut out = AbiArgAliases::default();
    for (param_idx, usage) in by_param {
        let has_lo = usage.words.contains(&0);
        let has_hi = usage.words.contains(&1);
        let (kind, confidence) = if has_lo && has_hi {
            if usage.pointer_like_hits > 0 {
                (ArgAliasKind::Ptr64, AliasConfidence::High)
            } else {
                (ArgAliasKind::U64, AliasConfidence::Medium)
            }
        } else {
            (ArgAliasKind::Word32, AliasConfidence::Low)
        };
        out.by_param.insert(
            param_idx,
            ArgAlias {
                param_idx,
                kind,
                confidence,
                observed_words: usage.words,
            },
        );
    }
    out
}

fn collect_constmem_from_expr(expr: &IRExpr, out: &mut Vec<(u32, u32)>) {
    match expr {
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
    opcode.contains("WIDE")
        || opcode.starts_with("LD")
        || opcode.starts_with("ST")
        || opcode.contains("LEA")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, parser::parse_sass};

    #[test]
    fn detects_legacy_profile_from_param_window() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
            /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x148] ;
            /*0020*/ IADD3 R3, R3, c[0x0][0x154], RZ ;
        "#;
        let instrs = parse_sass(sass);
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
        let instrs = parse_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::modern_param_160());
    }

    #[test]
    fn detects_profile_from_sm_fallback_when_offsets_absent() {
        let sass = r#"
            /*0000*/ S2R R0, SR_CTAID.X ;
            /*0010*/ S2R R1, SR_TID.X ;
        "#;
        let instrs = parse_sass(sass);
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(89)),
            AbiProfile::modern_param_160()
        );
        assert_eq!(
            AbiProfile::detect_with_sm(&instrs, Some(70)),
            AbiProfile::legacy_param_140()
        );
    }

    #[test]
    fn profile_selection_changes_param_indexing() {
        let legacy = AbiProfile::legacy_param_140();
        let modern = AbiProfile::modern_param_160();
        assert!(matches!(
            legacy.classify_constmem(0, 0x160),
            ConstMemSemantic::ParamWord { param_idx: 4, word_idx: 0 }
        ));
        assert!(matches!(
            modern.classify_constmem(0, 0x160),
            ConstMemSemantic::ParamWord { param_idx: 0, word_idx: 0 }
        ));
    }

    #[test]
    fn resolves_user_asked_offsets_in_legacy_profile() {
        let p = AbiProfile::legacy_param_140();
        assert_eq!(p.resolve_constmem(0, 0x8).unwrap().symbol, "blockDimX");
        assert_eq!(p.resolve_constmem(0, 0x28).unwrap().symbol, "abi_internal_0x28");
        assert_eq!(p.resolve_constmem(0, 0x44).unwrap().symbol, "abi_internal_0x44");
        assert_eq!(p.resolve_constmem(0, 0x140).unwrap().symbol, "param_0[0]");
        assert_eq!(p.resolve_constmem(0, 0x144).unwrap().symbol, "param_0[1]");
        assert_eq!(p.resolve_constmem(0, 0x148).unwrap().symbol, "param_1[0]");
        assert_eq!(p.resolve_constmem(0, 0x14c).unwrap().symbol, "param_1[1]");
        assert_eq!(p.resolve_constmem(0, 0x150).unwrap().symbol, "param_2[0]");
        assert_eq!(p.resolve_constmem(0, 0x154).unwrap().symbol, "param_2[1]");
    }

    #[test]
    fn abi_display_formats_constmem_symbolically() {
        let display = AbiDisplay::new(AbiProfile::legacy_param_140());
        let expr = IRExpr::Op {
            op: "ConstMem".to_string(),
            args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x148)],
        };
        assert_eq!(display.expr(&expr), "param_1[0]");
    }

    #[test]
    fn infers_pointer_alias_for_u64_param_used_in_wide_ops() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ IADD3.X R5, R5, c[0x0][0x164], RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
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
    fn abi_display_prefers_inferred_arg_aliases_for_param_words() {
        let mut aliases = AbiArgAliases::default();
        aliases.by_param.insert(
            1,
            ArgAlias {
                param_idx: 1,
                kind: ArgAliasKind::U64,
                confidence: AliasConfidence::Medium,
                observed_words: [0, 1].into_iter().collect(),
            },
        );
        let display = AbiDisplay::with_aliases(AbiProfile::legacy_param_140(), aliases);
        let expr = IRExpr::Op {
            op: "ConstMem".to_string(),
            args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x148)],
        };
        assert_eq!(display.expr(&expr), "arg1_u64.lo32");
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
            },
        );
        aliases.by_param.insert(
            3,
            ArgAlias {
                param_idx: 3,
                kind: ArgAliasKind::Word32,
                confidence: AliasConfidence::Low,
                observed_words: [1].into_iter().collect(),
            },
        );

        let decls = aliases.render_typed_arg_declarations();
        assert!(decls.iter().any(|d| d.contains("uintptr_t arg0_ptr;")));
        assert!(decls.iter().any(|d| d.contains("uint32_t arg3_word1;")));
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
            },
        );
        aliases.by_param.insert(
            2,
            ArgAlias {
                param_idx: 2,
                kind: ArgAliasKind::Word32,
                confidence: AliasConfidence::Low,
                observed_words: [1].into_iter().collect(),
            },
        );
        let params = aliases.render_typed_param_list();
        assert_eq!(params[0], "uintptr_t arg0_ptr");
        assert_eq!(params[1], "uint32_t arg2_word1");
    }

    #[test]
    fn infers_local_typed_declarations_from_ssa_dests() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ IADD3 R1, R1, 0x1, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
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
        let cfg = build_cfg(parse_sass(sass));
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
            /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x8] ;
            /*0020*/ IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x44] ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = annotate_function_ir_constmem(&fir, AbiProfile::legacy_param_140());
        assert!(!anns.is_empty());

        let mut saw_param = false;
        let mut saw_builtin = false;
        let mut saw_internal = false;
        for (_stmt, entries) in anns.iter() {
            for e in entries {
                match e.semantic {
                    ConstMemSemantic::ParamWord { param_idx: 0, word_idx: 0 } => saw_param = true,
                    ConstMemSemantic::Builtin("blockDimX") => saw_builtin = true,
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
