//! Test-only semantic lifting harness for focused SSA regression coverage.
//!
//! Purpose:
//! - keep narrow opcode-to-expression tests alive while the canonical backend
//!   finishes moving every surviving semantic contract into
//!   `FunctionAnalysis`, `ast_lowering`, and renderer tests
//!
//! Inputs:
//! - optimized `FunctionIR`
//! - optional ABI annotations and argument aliases
//!
//! Outputs:
//! - optional lifted expressions/statements used by unit tests
//!
//! Invariants:
//! - non-mutating: this module never edits CFG, SSA, or structured control
//! - conservative: unsupported ops stay explicit instead of guessed
//!
//! This module must not:
//! - participate in the production backend pipeline
//! - be treated as part of the public library API

use std::collections::{BTreeMap, BTreeSet};

use crate::abi::{AbiAnnotations, AbiArgAliases, ArgAliasKind, ConstMemSemantic, StatementRef};
use crate::ast::{Expr, IntrinsicOp, LValue, PointerLane};
use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue, RegId};
use crate::type_inference::{infer_ssa_types, InferredType};

mod op_sig;
mod registry;
mod rules;

#[derive(Clone, Debug)]
pub struct SemanticLiftConfig<'a> {
    pub abi_annotations: Option<&'a AbiAnnotations>,
    pub abi_aliases: Option<&'a AbiArgAliases>,
    /// Strict mode only applies high-confidence rewrites.
    pub strict: bool,
}

impl Default for SemanticLiftConfig<'_> {
    fn default() -> Self {
        Self {
            abi_annotations: None,
            abi_aliases: None,
            strict: true,
        }
    }
}

pub type LiftedExpr = Expr;

fn intrinsic_expr(op: IntrinsicOp, args: Vec<LiftedExpr>) -> LiftedExpr {
    LiftedExpr::Intrinsic { op, args }
}

fn call_expr(func: &str, args: Vec<LiftedExpr>) -> LiftedExpr {
    LiftedExpr::CallLike {
        func: func.to_string(),
        args,
    }
}

fn min_expr(lhs: LiftedExpr, rhs: LiftedExpr) -> LiftedExpr {
    intrinsic_expr(IntrinsicOp::Min, vec![lhs, rhs])
}

fn max_expr(lhs: LiftedExpr, rhs: LiftedExpr) -> LiftedExpr {
    intrinsic_expr(IntrinsicOp::Max, vec![lhs, rhs])
}

fn lift_named_symbol_expr(name: &str) -> LiftedExpr {
    if let Some((base, lane)) = PointerLane::parse_named(name) {
        return LiftedExpr::PtrLane { base, lane };
    }
    LiftedExpr::ConstMemSymbol(name.to_string())
}

#[derive(Clone, Debug, PartialEq)]
pub struct LiftedStmt {
    pub dest: LValue,
    pub pred: Option<Expr>,
    pub rhs: Expr,
    /// The previous SSA value of the destination when the predicate is false.
    /// Populated from `IRStatement::pred_old_defs` so the renderer can emit
    /// `dest = pred ? rhs : pred_old_val` instead of `if (pred) dest = rhs`.
    pub pred_old_val: Option<Expr>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticLiftStats {
    pub attempted: usize,
    pub lifted: usize,
    pub fallback: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticLiftResult {
    pub by_def: BTreeMap<DefRef, LiftedStmt>,
    pub stats: SemanticLiftStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DefRef {
    pub block_id: usize,
    pub stmt_idx: usize,
    pub def_idx: usize,
}

pub fn lift_function_ir(
    function_ir: &FunctionIR,
    config: &SemanticLiftConfig<'_>,
) -> SemanticLiftResult {
    let mut out = SemanticLiftResult::default();
    let inferred_types = infer_ssa_types(function_ir);
    let defined_regs = collect_defined_regs(function_ir);
    let pair_hi_map = build_pair_hi_map(function_ir);
    let (wide32_hi_map, known_zero_regs) = build_widened_32_maps(function_ir);

    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            out.stats.attempted += 1;
            if let Some(lifted) = lift_literal_stmt(
                stmt,
                stmt_ref,
                config,
                &inferred_types,
                &defined_regs,
            ) {
                for (def_idx, lifted_stmt) in lifted.into_iter().enumerate() {
                    out.by_def.insert(
                        DefRef {
                            block_id: block.id,
                            stmt_idx,
                            def_idx,
                        },
                        lifted_stmt,
                    );
                }
                out.stats.lifted += 1;
                continue;
            }

            let RValue::Op { opcode, args } = &stmt.value else {
                continue;
            };
            if let Some(stores) =
                lift_store_stmt_lanes(opcode, args, stmt_ref, config, &inferred_types)
            {
                let pred = stmt
                    .pred
                    .as_ref()
                    .map(|p| lift_ir_expr(p, stmt_ref, config));
                // Store statements have no def register, so no pred_old_val.
                for (def_idx, (dest, rhs)) in stores.into_iter().enumerate() {
                    out.by_def.insert(
                        DefRef {
                            block_id: block.id,
                            stmt_idx,
                            def_idx,
                        },
                        LiftedStmt {
                            dest,
                            pred: pred.clone(),
                            rhs,
                            pred_old_val: None,
                        },
                    );
                }
                out.stats.lifted += 1;
                continue;
            }
            let mut any_lifted = false;
            let def_count = stmt.defs.len().max(1);
            for def_idx in 0..def_count {
                let lifted_rhs = lift_opcode_expr_for_def(
                    opcode,
                    args,
                    def_idx,
                    stmt_ref,
                    config,
                    stmt.defs.get(def_idx).and_then(IRExpr::get_reg),
                    &inferred_types,
                    &pair_hi_map,
                    &wide32_hi_map,
                    &known_zero_regs,
                );
                if let Some(rhs) = lifted_rhs {
                    let dest = stmt.defs.get(def_idx).map_or_else(
                        || LValue::Raw("_".to_string()),
                        |d| lvalue_from_ir_def(d, stmt_ref, config),
                    );
                    let pred = stmt
                        .pred
                        .as_ref()
                        .map(|p| lift_ir_expr(p, stmt_ref, config));
                    let pred_old_val = lifted_pred_old_val(
                        stmt,
                        def_idx,
                        stmt_ref,
                        config,
                        &inferred_types,
                        &defined_regs,
                    );
                    out.by_def.insert(
                        DefRef {
                            block_id: block.id,
                            stmt_idx,
                            def_idx,
                        },
                        LiftedStmt {
                            dest,
                            pred,
                            rhs,
                            pred_old_val,
                        },
                    );
                    any_lifted = true;
                }
            }
            if any_lifted {
                out.stats.lifted += 1;
            } else {
                out.stats.fallback += 1;
            }
        }
    }

    normalize_shared_word_indices(&mut out);
    out
}

#[derive(Clone)]
struct SignedExprTerm {
    sign: i8,
    expr: Expr,
}

fn normalize_shared_word_indices(lifted: &mut SemanticLiftResult) {
    let defs = collect_inlineable_lifted_defs(lifted);
    if defs.is_empty() {
        return;
    }
    for stmt in lifted.by_def.values_mut() {
        normalize_shared_word_indices_in_lvalue(&mut stmt.dest, &defs);
        if let Some(pred) = &mut stmt.pred {
            normalize_shared_word_indices_in_expr(pred, &defs);
        }
        normalize_shared_word_indices_in_expr(&mut stmt.rhs, &defs);
        if let Some(old) = &mut stmt.pred_old_val {
            normalize_shared_word_indices_in_expr(old, &defs);
        }
    }
}

fn collect_inlineable_lifted_defs(lifted: &SemanticLiftResult) -> BTreeMap<String, Expr> {
    let mut defs = BTreeMap::new();
    for stmt in lifted.by_def.values() {
        let LValue::Var(name) = &stmt.dest else {
            continue;
        };
        if is_inlineable_shared_index_expr(&stmt.rhs) {
            defs.insert(name.clone(), stmt.rhs.clone());
        }
    }
    defs
}

fn is_inlineable_shared_index_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Raw(_) | Expr::Imm(_) | Expr::Reg(_) | Expr::Builtin(_) => true,
        Expr::Unary { arg, .. } => is_inlineable_shared_index_expr(arg),
        Expr::Binary { lhs, rhs, .. } => {
            is_inlineable_shared_index_expr(lhs) && is_inlineable_shared_index_expr(rhs)
        }
        Expr::Cast { expr, .. } => is_inlineable_shared_index_expr(expr),
        Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::Ternary { .. }
        | Expr::CallLike { .. }
        | Expr::Intrinsic { .. }
        | Expr::Load { .. }
        | Expr::WidePtr { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Addr64 { .. }
        | Expr::Index { .. } => false,
    }
}

fn normalize_shared_word_indices_in_lvalue(
    lvalue: &mut LValue,
    defs: &BTreeMap<String, Expr>,
) {
    match lvalue {
        LValue::Indexed { base, index } => {
            if expr_is_shared_word_array(base) {
                *index = Box::new(normalize_shared_word_index_expr(index, defs));
            } else {
                normalize_shared_word_indices_in_expr(base, defs);
                normalize_shared_word_indices_in_expr(index, defs);
            }
        }
        LValue::Deref { addr, .. } => normalize_shared_word_indices_in_expr(addr, defs),
        LValue::Raw(_) | LValue::Var(_) | LValue::PtrLane { .. } => {}
    }
}

fn normalize_shared_word_indices_in_expr(expr: &mut Expr, defs: &BTreeMap<String, Expr>) {
    match expr {
        Expr::Unary { arg, .. } => normalize_shared_word_indices_in_expr(arg, defs),
        Expr::Binary { lhs, rhs, .. } => {
            normalize_shared_word_indices_in_expr(lhs, defs);
            normalize_shared_word_indices_in_expr(rhs, defs);
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            normalize_shared_word_indices_in_expr(cond, defs);
            normalize_shared_word_indices_in_expr(then_expr, defs);
            normalize_shared_word_indices_in_expr(else_expr, defs);
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                normalize_shared_word_indices_in_expr(arg, defs);
            }
        }
        Expr::Load { addr, .. } | Expr::Cast { expr: addr, .. } => {
            normalize_shared_word_indices_in_expr(addr, defs);
        }
        Expr::WidePtr { base, offset } => {
            normalize_shared_word_indices_in_expr(base, defs);
            normalize_shared_word_indices_in_expr(offset, defs);
        }
        Expr::Addr64 { lo, hi } => {
            normalize_shared_word_indices_in_expr(lo, defs);
            normalize_shared_word_indices_in_expr(hi, defs);
        }
        Expr::Index { base, index } => {
            if expr_is_shared_word_array(base) {
                *index = Box::new(normalize_shared_word_index_expr(index, defs));
            } else {
                normalize_shared_word_indices_in_expr(base, defs);
                normalize_shared_word_indices_in_expr(index, defs);
            }
        }
        Expr::LaneExtract { value, .. } => normalize_shared_word_indices_in_expr(value, defs),
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_) => {}
    }
}

fn expr_is_shared_word_array(expr: &Expr) -> bool {
    matches!(expr, Expr::Builtin(name) if name == "shmem")
}

fn normalize_shared_word_index_expr(index: &Expr, defs: &BTreeMap<String, Expr>) -> Expr {
    let expanded = expand_shared_index_defs(index, defs, 8);
    let mut terms = Vec::new();
    collect_signed_add_terms(&expanded, 1, &mut terms);
    if terms.is_empty() {
        return index.clone();
    }

    let cluster_shift_idx = terms
        .iter()
        .position(|term| term.sign > 0 && expr_is_cga_ctaid_shift(&term.expr));
    let cluster_base_1024_idx = cluster_shift_idx.and_then(|_| {
        terms.iter().position(|term| {
            term.sign > 0 && expr_integer_value(&term.expr).is_some_and(|value| value == 1024)
        })
    });

    let mut changed = false;
    let mut normalized_terms = Vec::new();
    for (idx, term) in terms.into_iter().enumerate() {
        if Some(idx) == cluster_shift_idx || Some(idx) == cluster_base_1024_idx {
            changed = true;
            continue;
        }
        let (expr, term_changed) = normalize_shared_word_term(&term.expr);
        if term_changed {
            changed = true;
        }
        if expr_is_zero(&expr) {
            changed = true;
            continue;
        }
        normalized_terms.push(SignedExprTerm {
            sign: term.sign,
            expr,
        });
    }

    if !changed {
        return index.clone();
    }
    rebuild_signed_sum(normalized_terms)
}

fn expand_shared_index_defs(
    expr: &Expr,
    defs: &BTreeMap<String, Expr>,
    depth: usize,
) -> Expr {
    if depth == 0 {
        return expr.clone();
    }
    match expr {
        Expr::Reg(name) => defs
            .get(name)
            .map(|rhs| expand_shared_index_defs(rhs, defs, depth - 1))
            .unwrap_or_else(|| expr.clone()),
        Expr::Unary { op, arg } => Expr::Unary {
            op: op.clone(),
            arg: Box::new(expand_shared_index_defs(arg, defs, depth)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: op.clone(),
            lhs: Box::new(expand_shared_index_defs(lhs, defs, depth)),
            rhs: Box::new(expand_shared_index_defs(rhs, defs, depth)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty: ty.clone(),
            expr: Box::new(expand_shared_index_defs(expr, defs, depth)),
        },
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::Ternary { .. }
        | Expr::CallLike { .. }
        | Expr::Intrinsic { .. }
        | Expr::Load { .. }
        | Expr::WidePtr { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::Addr64 { .. }
        | Expr::Index { .. } => expr.clone(),
    }
}

fn collect_signed_add_terms(expr: &Expr, sign: i8, out: &mut Vec<SignedExprTerm>) {
    match strip_shared_index_casts(expr) {
        Expr::Binary { op, lhs, rhs } if op == "+" => {
            collect_signed_add_terms(lhs, sign, out);
            collect_signed_add_terms(rhs, sign, out);
        }
        Expr::Binary { op, lhs, rhs } if op == "-" => {
            collect_signed_add_terms(lhs, sign, out);
            collect_signed_add_terms(rhs, -sign, out);
        }
        other => out.push(SignedExprTerm {
            sign,
            expr: other.clone(),
        }),
    }
}

fn strip_shared_index_casts<'a>(expr: &'a Expr) -> &'a Expr {
    let mut current = expr;
    while let Expr::Cast { ty, expr } = current {
        if matches!(
            ty.as_str(),
            "int32_t" | "uint32_t" | "int64_t" | "uint64_t" | "intptr_t" | "uintptr_t"
        ) {
            current = expr;
        } else {
            break;
        }
    }
    current
}

fn expr_integer_value(expr: &Expr) -> Option<i64> {
    match strip_shared_index_casts(expr) {
        Expr::Imm(text) | Expr::Raw(text) => text.parse::<i64>().ok(),
        _ => None,
    }
}

fn expr_is_cga_ctaid(expr: &Expr) -> bool {
    matches!(strip_shared_index_casts(expr), Expr::Raw(text) if text == "cgaCtaId")
}

fn expr_is_cga_ctaid_shift(expr: &Expr) -> bool {
    match strip_shared_index_casts(expr) {
        Expr::Binary { op, lhs, rhs } if op == "<<" => {
            expr_is_cga_ctaid(lhs) && expr_integer_value(rhs) == Some(24)
        }
        _ => false,
    }
}

fn normalize_shared_word_term(expr: &Expr) -> (Expr, bool) {
    if let Some(value) = expr_integer_value(expr) {
        if value % 4 == 0 {
            return (Expr::Imm((value / 4).to_string()), true);
        }
        return (strip_shared_index_casts(expr).clone(), false);
    }

    if let Some(normalized) = normalize_shared_word_scaled_term(expr) {
        return (normalized, true);
    }

    (strip_shared_index_casts(expr).clone(), false)
}

fn normalize_shared_word_scaled_term(expr: &Expr) -> Option<Expr> {
    match strip_shared_index_casts(expr) {
        Expr::Binary { op, lhs, rhs } if op == "*" => {
            if let Some(factor) = expr_integer_value(rhs) {
                return divide_scaled_shared_word_term(lhs, factor);
            }
            if let Some(factor) = expr_integer_value(lhs) {
                return divide_scaled_shared_word_term(rhs, factor);
            }
            None
        }
        Expr::Binary { op, lhs, rhs } if op == "<<" => {
            let shift = expr_integer_value(rhs)?;
            if shift < 2 {
                return None;
            }
            let base = strip_shared_index_casts(lhs).clone();
            Some(if shift == 2 {
                base
            } else {
                Expr::Binary {
                    op: "<<".to_string(),
                    lhs: Box::new(base),
                    rhs: Box::new(Expr::Imm((shift - 2).to_string())),
                }
            })
        }
        _ => None,
    }
}

fn divide_scaled_shared_word_term(expr: &Expr, factor: i64) -> Option<Expr> {
    if factor % 4 != 0 {
        return None;
    }
    let reduced = factor / 4;
    let base = strip_shared_index_casts(expr).clone();
    Some(if reduced == 1 {
        base
    } else {
        Expr::Binary {
            op: "*".to_string(),
            lhs: Box::new(base),
            rhs: Box::new(Expr::Imm(reduced.to_string())),
        }
    })
}

fn rebuild_signed_sum(terms: Vec<SignedExprTerm>) -> Expr {
    let mut iter = terms.into_iter();
    let Some(first) = iter.next() else {
        return Expr::Imm("0".to_string());
    };

    let mut acc = if first.sign < 0 {
        Expr::Unary {
            op: "-".to_string(),
            arg: Box::new(first.expr),
        }
    } else {
        first.expr
    };

    for term in iter {
        if term.sign < 0 {
            acc = Expr::Binary {
                op: "-".to_string(),
                lhs: Box::new(acc),
                rhs: Box::new(term.expr),
            };
        } else {
            acc = add_like_expr(acc, term.expr);
        }
    }
    acc
}

fn expr_is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Imm(text) | Expr::Raw(text) if text == "0")
}

fn lift_literal_stmt(
    stmt: &IRStatement,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    inferred_types: &BTreeMap<RegId, InferredType>,
    defined_regs: &BTreeSet<RegId>,
) -> Option<Vec<LiftedStmt>> {
    let literal = match &stmt.value {
        RValue::ImmI(bits) => {
            let expr = if stmt
                .defs
                .first()
                .and_then(IRExpr::get_reg)
                .and_then(|reg| inferred_reg_type(Some(reg), inferred_types))
                .is_some_and(is_floatish_type)
            {
                lift_float_ir_expr(&IRExpr::ImmI(*bits), stmt_ref, config)
            } else {
                lift_ir_expr(&IRExpr::ImmI(*bits), stmt_ref, config)
            };
            expr.render()
        }
        RValue::ImmF(value) => lift_ir_expr(&IRExpr::ImmF(*value), stmt_ref, config).render(),
        RValue::Op { .. } | RValue::Phi(_) => return None,
    };
    let dest = stmt
        .defs
        .first()
        .map_or_else(|| LValue::Raw("_".to_string()), |d| lvalue_from_ir_def(d, stmt_ref, config));
    let pred = stmt.pred.as_ref().map(|p| lift_ir_expr(p, stmt_ref, config));
    let pred_old_val = lifted_pred_old_val(
        stmt,
        0,
        stmt_ref,
        config,
        inferred_types,
        defined_regs,
    );
    Some(vec![LiftedStmt {
        dest,
        pred,
        rhs: Expr::Raw(literal),
        pred_old_val,
    }])
}

fn collect_defined_regs(function_ir: &FunctionIR) -> BTreeSet<RegId> {
    let mut out = BTreeSet::new();
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            for def in &stmt.defs {
                if let Some(reg) = def.get_reg() {
                    out.insert(reg.clone());
                }
            }
        }
    }
    out
}

fn lifted_pred_old_val(
    stmt: &IRStatement,
    def_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    inferred_types: &BTreeMap<RegId, InferredType>,
    defined_regs: &BTreeSet<RegId>,
) -> Option<Expr> {
    let old = stmt.pred_old_defs.get(def_idx)?;
    if should_zero_undefined_pred_old(stmt, old, defined_regs) {
        return Some(zero_literal_for_def(
            stmt.defs.get(def_idx).and_then(IRExpr::get_reg),
            inferred_types,
        ));
    }
    Some(lift_ir_expr(old, stmt_ref, config))
}

fn should_zero_undefined_pred_old(
    stmt: &IRStatement,
    old: &IRExpr,
    defined_regs: &BTreeSet<RegId>,
) -> bool {
    let Some(old_reg) = old.get_reg() else {
        return false;
    };
    if defined_regs.contains(old_reg) {
        return false;
    }
    matches!(
        &stmt.value,
        RValue::Op { opcode, .. } if opcode.starts_with("LD")
    )
}

fn zero_literal_for_def(
    reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Expr {
    if inferred_reg_type(reg, inferred_types).is_some_and(is_floatish_type) {
        Expr::Imm("0.0".to_string())
    } else {
        Expr::Imm("0".to_string())
    }
}

fn build_pair_hi_map(function_ir: &FunctionIR) -> BTreeMap<RegId, IRExpr> {
    let mut map = BTreeMap::new();
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            let Some(lo_reg) = stmt.defs.first().and_then(IRExpr::get_reg) else {
                continue;
            };
            let Some(hi_expr) = paired_hi_def_expr(stmt, lo_reg) else {
                continue;
            };
            map.insert(lo_reg.clone(), hi_expr);
        }
    }

    loop {
        let mut changed = false;
        for block in &function_ir.blocks {
            for stmt in &block.stmts {
                let Some(dst) = stmt.defs.first().and_then(IRExpr::get_reg) else {
                    continue;
                };
                if stmt.defs.len() != 1 || map.contains_key(dst) {
                    continue;
                }
                match &stmt.value {
                    RValue::Op { opcode, args }
                        if matches!(opcode_mnemonic(opcode), "MOV" | "UMOV") && args.len() == 1 =>
                    {
                        let Some(hi_expr) = pair_hi_expr_from_lo(&args[0], &map) else {
                            continue;
                        };
                        map.insert(dst.clone(), hi_expr);
                        changed = true;
                    }
                    RValue::Phi(low_args) => {
                        let Some(hi_dst) = find_phi_hi_partner(block, dst, low_args, &map) else {
                            continue;
                        };
                        map.insert(dst.clone(), IRExpr::Reg(hi_dst));
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        if !changed {
            break;
        }
    }

    map
}

fn paired_hi_def_expr(stmt: &IRStatement, lo_reg: &RegId) -> Option<IRExpr> {
    if !matches!(lo_reg.class.as_str(), "R" | "UR") {
        return None;
    }
    let RValue::Op { opcode, .. } = &stmt.value else {
        return None;
    };
    if !opcode_has_implicit_pair_hi(opcode) {
        return None;
    }
    stmt.defs.iter().skip(1).find_map(|def| {
        let reg = def.get_reg()?;
        (reg.class == lo_reg.class && reg.idx == lo_reg.idx + 1).then(|| def.clone())
    })
}

fn opcode_has_implicit_pair_hi(opcode: &str) -> bool {
    let mnem = opcode_mnemonic(opcode);
    if matches!(mnem, "MOV" | "IADD") && opcode_has_mod(opcode, "64") {
        return true;
    }
    if matches!(mnem, "IADD3" | "UIADD3") && opcode_has_mod(opcode, "64") {
        return true;
    }
    if matches!(mnem, "IMAD" | "UIMAD") && opcode_has_mod(opcode, "WIDE") {
        return true;
    }
    if matches!(mnem, "ULDC" | "LDC" | "LDCU") && (opcode_has_mod(opcode, "64") || opcode_has_mod(opcode, "128")) {
        return true;
    }
    mnem.starts_with("LD") && (opcode_has_mod(opcode, "64") || opcode_has_mod(opcode, "128"))
}

fn find_phi_hi_partner(
    block: &IRBlock,
    low_dst: &RegId,
    low_args: &[IRExpr],
    pair_hi_map: &BTreeMap<RegId, IRExpr>,
) -> Option<RegId> {
    for stmt in &block.stmts {
        let Some(hi_dst) = stmt.defs.first().and_then(IRExpr::get_reg) else {
            continue;
        };
        if hi_dst == low_dst
            || stmt.defs.len() != 1
            || hi_dst.class != low_dst.class
            || hi_dst.idx != low_dst.idx + 1
        {
            continue;
        }
        let RValue::Phi(hi_args) = &stmt.value else {
            continue;
        };
        if low_args.len() != hi_args.len() {
            continue;
        }
        let mut all_match = true;
        for (lo_arg, hi_arg) in low_args.iter().zip(hi_args.iter()) {
            let Some(expected_hi) = pair_hi_expr_from_lo(lo_arg, pair_hi_map) else {
                all_match = false;
                break;
            };
            if expected_hi != *hi_arg {
                all_match = false;
                break;
            }
        }
        if all_match {
            return Some((*hi_dst).clone());
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wide32ExtKind {
    Signed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Wide32HiInfo {
    low: RegId,
    kind: Wide32ExtKind,
}

fn build_widened_32_maps(
    function_ir: &FunctionIR,
) -> (BTreeMap<RegId, Wide32HiInfo>, BTreeSet<RegId>) {
    let mut hi_map = BTreeMap::<RegId, Wide32HiInfo>::new();
    let mut zero_regs = BTreeSet::<RegId>::new();

    loop {
        let mut changed = false;
        for block in &function_ir.blocks {
            for stmt in &block.stmts {
                let Some(dst) = stmt.defs.first().and_then(IRExpr::get_reg) else {
                    continue;
                };
                if stmt.defs.len() != 1 {
                    continue;
                }
                match &stmt.value {
                    value if rvalue_is_known_zero(value) => {
                        changed |= zero_regs.insert(dst.clone());
                    }
                    RValue::Op { opcode, args } => {
                        if let Some(low) = shf_signext_low_reg(opcode, args) {
                            let info = Wide32HiInfo {
                                low,
                                kind: Wide32ExtKind::Signed,
                            };
                            changed |= hi_map.get(dst) != Some(&info);
                            hi_map.insert(dst.clone(), info);
                            continue;
                        }
                        if matches!(opcode_mnemonic(opcode), "MOV" | "UMOV") && args.len() == 1 {
                            if let Some(src) = args[0].get_reg() {
                                if zero_regs.contains(&src) {
                                    changed |= zero_regs.insert(dst.clone());
                                }
                                if let Some(info) = hi_map.get(&src).cloned() {
                                    changed |= hi_map.get(dst) != Some(&info);
                                    hi_map.insert(dst.clone(), info);
                                }
                            } else if is_zero_expr(&args[0]) {
                                changed |= zero_regs.insert(dst.clone());
                            }
                        }
                    }
                    RValue::Phi(args) => {
                        let zero_phi = args.iter().all(|arg| {
                            arg.get_reg()
                                .map(|reg| zero_regs.contains(&reg))
                                .unwrap_or_else(|| is_zero_expr(arg))
                        });
                        if zero_phi {
                            changed |= zero_regs.insert(dst.clone());
                        }
                        let mut resolved = None::<Wide32HiInfo>;
                        let mut saw_any = false;
                        for arg in args {
                            let Some(reg) = arg.get_reg() else {
                                resolved = None;
                                saw_any = false;
                                break;
                            };
                            let Some(info) = hi_map.get(&reg) else {
                                resolved = None;
                                saw_any = false;
                                break;
                            };
                            saw_any = true;
                            match &resolved {
                                None => resolved = Some(info.clone()),
                                Some(existing) if existing == info => {}
                                Some(_) => {
                                    resolved = None;
                                    saw_any = false;
                                    break;
                                }
                            }
                        }
                        if saw_any {
                            let info = resolved.expect("phi resolution should produce info");
                            changed |= hi_map.get(dst) != Some(&info);
                            hi_map.insert(dst.clone(), info);
                        }
                    }
                    _ => {}
                }
            }
        }
        if !changed {
            break;
        }
    }

    (hi_map, zero_regs)
}

fn rvalue_is_known_zero(value: &RValue) -> bool {
    match value {
        RValue::ImmI(bits) => *bits == 0,
        RValue::ImmF(bits) => *bits == 0.0,
        RValue::Op { opcode, args } if opcode.starts_with("HFMA2") => {
            args.iter().all(is_zero_expr)
        }
        _ => false,
    }
}

fn shf_signext_low_reg(opcode: &str, args: &[IRExpr]) -> Option<RegId> {
    if args.len() != 3 {
        return None;
    }
    if !opcode.starts_with("SHF") {
        return None;
    }
    let parts = opcode.split('.').collect::<Vec<_>>();
    if !parts.contains(&"R") || !parts.contains(&"HI") || !parts.contains(&"S32") {
        return None;
    }
    if !is_zero_expr(&args[0]) || !matches!(args[1], IRExpr::ImmI(31)) {
        return None;
    }
    args[2].get_reg().cloned()
}

fn lift_opcode_expr(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    registry::dispatch_opcode(opcode, args, stmt_ref, config)
}

fn opcode_mnemonic(opcode: &str) -> &str {
    opcode.split('.').next().unwrap_or(opcode)
}

fn opcode_has_mod(opcode: &str, needle: &str) -> bool {
    opcode.split('.').skip(1).any(|m| m == needle)
}

fn lift_opcode_expr_for_def(
    opcode: &str,
    args: &[IRExpr],
    def_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    def_reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
    pair_hi_map: &BTreeMap<RegId, IRExpr>,
    wide32_hi_map: &BTreeMap<RegId, Wide32HiInfo>,
    known_zero_regs: &BTreeSet<RegId>,
) -> Option<LiftedExpr> {
    let mnem = opcode_mnemonic(opcode);
    // LDCU is Blackwell's rename of ULDC — same wide-load semantics.
    // `.64` defines a register pair, `.128` defines a 4-wide tuple, and
    // every implicit hi-half def lifts to the corresponding param word
    // (`+4`, `+8`, `+12`).
    if matches!(mnem, "ULDC" | "LDC" | "LDCU") {
        let extra_defs = if opcode_has_mod(opcode, "128") {
            3
        } else if opcode_has_mod(opcode, "64") {
            1
        } else {
            0
        };
        if extra_defs > 0 {
            if def_idx == 0 {
                return lift_opcode_expr(opcode, args, stmt_ref, config);
            }
            if def_idx <= extra_defs {
                if let Some(expr) =
                    lift_uldc_wide_def_from_lo(args.first()?, def_idx, stmt_ref, config)
                {
                    return Some(expr);
                }
                let hi_arg = args
                    .first()
                    .and_then(|e| constmem_plus_word_offset_n(e, def_idx))?;
                return Some(lift_ir_expr(&hi_arg, stmt_ref, config));
            }
            return None;
        }
    }
    if matches!(mnem, "IMAD" | "UIMAD") && opcode_has_mod(opcode, "WIDE") {
        if def_idx == 0 {
            return lift_opcode_expr(opcode, args, stmt_ref, config);
        }
        if def_idx == 1 {
            return lift_imad_wide_hi(args, stmt_ref, config, pair_hi_map);
        }
        return None;
    }
    if mnem == "IADD" && opcode_has_mod(opcode, "64") {
        return lift_iadd64_def(
            args,
            def_idx,
            stmt_ref,
            config,
            wide32_hi_map,
            known_zero_regs,
        );
    }
    if matches!(mnem, "IADD3" | "UIADD3") && opcode_has_mod(opcode, "64") {
        return lift_iadd3_64_def(
            args,
            def_idx,
            stmt_ref,
            config,
            def_reg,
            wide32_hi_map,
            known_zero_regs,
        );
    }
    if mnem == "MOV" && opcode_has_mod(opcode, "64") {
        return lift_mov64_def(args, def_idx, stmt_ref, config);
    }
    if mnem.starts_with("LD") && !matches!(mnem, "ULDC" | "LDC" | "LDCU") {
        let extra_defs = memory_lane_count(opcode).saturating_sub(1);
        if def_idx > extra_defs {
            return None;
        }
        return match mnem {
            "LDS" => lift_lds_lane_expr(opcode, args, def_idx, stmt_ref, config),
            "LDL" => lift_ldl_lane_expr(
                opcode,
                args,
                def_idx,
                stmt_ref,
                config,
                def_reg,
                inferred_types,
            ),
            _ => lift_ldg_lane_expr(
                opcode,
                args,
                def_idx,
                stmt_ref,
                config,
                def_reg,
                inferred_types,
            ),
        };
    }
    if def_idx > 0 && matches!(mnem, "LEA" | "ULEA") && !opcode_has_mod(opcode, "HI") {
        return lift_lea_carry(args, stmt_ref, config);
    }
    if def_idx > 0 && matches!(mnem, "IADD3" | "UIADD3") && !opcode_has_mod(opcode, "X") {
        if def_reg.is_some_and(|reg| matches!(reg.class.as_str(), "P" | "UP")) {
            return lift_iadd3_carry(args, stmt_ref, config);
        }
        return None;
    }
    if let Some(expr) =
        lift_lop3_float_sign_inject(opcode, args, stmt_ref, config, def_reg, inferred_types)
    {
        return Some(expr);
    }
    if let Some(expr) = lift_hfma2_constant_materialization(opcode, args) {
        return Some(expr);
    }
    if let Some(expr) = lift_typed_move_immediate(opcode, args, def_reg, inferred_types) {
        return Some(expr);
    }
    lift_opcode_expr(opcode, args, stmt_ref, config)
}

fn lift_hfma2_constant_materialization(opcode: &str, args: &[IRExpr]) -> Option<LiftedExpr> {
    if !opcode.starts_with("HFMA2") {
        return None;
    }
    if args.len() < 3 {
        return None;
    }
    if !is_zero_like_half2_arg(&args[0]) || !is_zero_like_half2_arg(&args[1]) {
        return None;
    }
    let hi = half_immediate_bits(&args[2])? as u32;
    let lo = match args.get(3) {
        Some(arg) => half_immediate_bits(arg)? as u32,
        None => 0,
    };
    Some(render_packed_half_constant((hi << 16) | lo))
}

fn inferred_reg_type(
    reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<InferredType> {
    let reg = reg?;
    inferred_types.get(reg).copied()
}

fn is_floatish_type(ty: InferredType) -> bool {
    matches!(
        ty,
        InferredType::F16 | InferredType::F32 | InferredType::AnyFloat
    )
}

fn expr_looks_floatish(expr: &IRExpr, inferred_types: &BTreeMap<RegId, InferredType>) -> bool {
    match expr {
        IRExpr::ImmF(_) => true,
        _ => expr
            .get_reg()
            .and_then(|reg| inferred_reg_type(Some(reg), inferred_types))
            .is_some_and(is_floatish_type),
    }
}

fn is_sign_mask_imm(expr: &IRExpr) -> bool {
    imm_as_u32(expr) == Some(0x8000_0000)
}

fn lift_lop3_float_sign_inject(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    def_reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<LiftedExpr> {
    let mnem = opcode_mnemonic(opcode);
    if !matches!(mnem, "LOP3" | "ULOP3") || args.len() != 5 || !is_pred_control_expr(&args[4]) {
        return None;
    }
    let imm = u8::try_from(imm_as_u32(&args[3])? & 0xff).ok()?;
    let has_float_context = inferred_reg_type(def_reg, inferred_types).is_some_and(is_floatish_type);

    match imm {
        // a | (SIGN_MASK & c)
        0xF8 if is_sign_mask_imm(&args[1])
            && (has_float_context
                || (expr_looks_floatish(&args[0], inferred_types)
                    && expr_looks_floatish(&args[2], inferred_types))) =>
        {
            let a = lift_ir_expr(&args[0], stmt_ref, config);
            let c = lift_ir_expr(&args[2], stmt_ref, config);
            Some(LiftedExpr::Raw(format!(
                "copysignf({}, {})",
                a.render(),
                c.render()
            )))
        }
        // a | (b & SIGN_MASK)
        0xF8 if is_sign_mask_imm(&args[2])
            && (has_float_context
                || (expr_looks_floatish(&args[0], inferred_types)
                    && expr_looks_floatish(&args[1], inferred_types))) =>
        {
            let a = lift_ir_expr(&args[0], stmt_ref, config);
            let b = lift_ir_expr(&args[1], stmt_ref, config);
            Some(LiftedExpr::Raw(format!(
                "copysignf({}, {})",
                a.render(),
                b.render()
            )))
        }
        // (SIGN_MASK & b) | c
        0xEA if is_sign_mask_imm(&args[0])
            && (has_float_context
                || (expr_looks_floatish(&args[1], inferred_types)
                    && expr_looks_floatish(&args[2], inferred_types))) =>
        {
            let b = lift_ir_expr(&args[1], stmt_ref, config);
            let c = lift_ir_expr(&args[2], stmt_ref, config);
            Some(LiftedExpr::Raw(format!(
                "copysignf({}, {})",
                c.render(),
                b.render()
            )))
        }
        // (a & SIGN_MASK) | c
        0xEA if is_sign_mask_imm(&args[1])
            && (has_float_context
                || (expr_looks_floatish(&args[0], inferred_types)
                    && expr_looks_floatish(&args[2], inferred_types))) =>
        {
            let a = lift_ir_expr(&args[0], stmt_ref, config);
            let c = lift_ir_expr(&args[2], stmt_ref, config);
            Some(LiftedExpr::Raw(format!(
                "copysignf({}, {})",
                c.render(),
                a.render()
            )))
        }
        _ => None,
    }
}

fn is_zero_like_half2_arg(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::ImmI(0) => true,
        IRExpr::ImmF(value) if *value == 0.0 => true,
        IRExpr::Reg(reg) if matches!(reg.class.as_str(), "RZ" | "URZ") => true,
        IRExpr::Op { op, args } if op == "-" && args.len() == 1 => {
            matches!(&args[0], IRExpr::Reg(reg) if matches!(reg.class.as_str(), "RZ" | "URZ"))
        }
        _ => false,
    }
}

fn half_immediate_bits(expr: &IRExpr) -> Option<u16> {
    match expr {
        IRExpr::ImmI(value) => {
            if *value >= i16::MIN as i64 && *value <= u16::MAX as i64 {
                Some((*value as i16) as u16)
            } else {
                None
            }
        }
        IRExpr::ImmF(value) => Some(f32_to_f16_bits(*value as f32)),
        _ => None,
    }
}

fn render_packed_half_constant(bits: u32) -> LiftedExpr {
    // HFMA2/HADD2/HMUL2 zero-source materialization yields a raw 32-bit packed
    // half payload in the destination register. Reinterpreting those bits as a
    // scalar f32 is semantically wrong and poisons integer kernels such as
    // SHA/crypto code with tiny float literals. Keep the exact bit pattern.
    LiftedExpr::Imm(bits.to_string())
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = 14 - half_exp;
        let mut half = (mantissa >> shift) as u16;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half = half.wrapping_add(1);
        }
        return sign | half;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x0000_1000) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

fn lift_typed_move_immediate(
    opcode: &str,
    args: &[IRExpr],
    def_reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<LiftedExpr> {
    let def_reg = def_reg?;
    let def_ty = inferred_types.get(def_reg).copied()?;
    if !matches!(def_ty, InferredType::F32 | InferredType::AnyFloat) {
        return None;
    }
    let src = if opcode.starts_with("IMAD.MOV") {
        args.get(2)?
    } else if opcode == "MOV" || opcode.starts_with("MOV.") {
        args.first()?
    } else {
        return None;
    };
    let IRExpr::ImmI(bits_i64) = src else {
        return None;
    };
    if !looks_like_f32_bitpattern(*bits_i64) {
        return None;
    }
    let bits = *bits_i64 as u32;
    Some(LiftedExpr::Imm(format_f32_bitpattern(bits)))
}

fn looks_like_f32_bitpattern(bits_i64: i64) -> bool {
    let bits = bits_i64 as u32;
    if bits_i64.unsigned_abs() < 0x1_0000 {
        return false;
    }
    let value = f32::from_bits(bits);
    value.is_finite()
}

fn format_f32_bitpattern(bits: u32) -> String {
    let value = f32::from_bits(bits);
    value.to_string()
}

fn constmem_plus_word_offset_n(expr: &IRExpr, words: usize) -> Option<IRExpr> {
    let IRExpr::Op { op, args } = expr else {
        return None;
    };
    if op != "ConstMem" || args.len() != 2 {
        return None;
    }
    let IRExpr::ImmI(bank) = args[0] else {
        return None;
    };
    let IRExpr::ImmI(offset) = args[1] else {
        return None;
    };
    let delta = i64::try_from(words).ok()?.checked_mul(4)?;
    Some(IRExpr::Op {
        op: "ConstMem".to_string(),
        args: vec![IRExpr::ImmI(bank), IRExpr::ImmI(offset + delta)],
    })
}

fn pair_hi_expr_from_lo(expr: &IRExpr, pair_hi_map: &BTreeMap<RegId, IRExpr>) -> Option<IRExpr> {
    match expr {
        IRExpr::Reg(reg) if matches!(reg.class.as_str(), "R" | "UR") => {
            pair_hi_map.get(reg).cloned()
        }
        _ => constmem_plus_word_offset_n(expr, 1),
    }
}

fn paired_hi_symbol_from_lo_expr(expr: &LiftedExpr) -> Option<LiftedExpr> {
    match expr {
        LiftedExpr::PtrLane {
            base,
            lane: PointerLane::Lo32,
        } => Some(LiftedExpr::PtrLane {
            base: base.clone(),
            lane: PointerLane::Hi32,
        }),
        _ => {
            let text = expr.render();
            let (base, lane) = PointerLane::parse_named(&text)?;
            (lane == PointerLane::Lo32).then_some(LiftedExpr::PtrLane {
                base,
                lane: PointerLane::Hi32,
            })
        }
    }
}

fn lift_paired_hi_from_lo(
    lo: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    pair_hi_map: &BTreeMap<RegId, IRExpr>,
) -> LiftedExpr {
    if is_zero_expr(lo) {
        return LiftedExpr::Imm("0".to_string());
    }
    let lo_expr = lift_ir_expr(lo, stmt_ref, config);
    if let Some(hi_expr) = pair_hi_expr_from_lo(lo, pair_hi_map) {
        return lift_ir_expr(&hi_expr, stmt_ref, config);
    }
    if let Some(symbol) = paired_hi_symbol_from_lo_expr(&lo_expr) {
        return symbol;
    }
    intrinsic_expr(IntrinsicOp::PairHi, vec![lo_expr])
}

/// Resolve the `def_idx`-th high-half def of a wide ULDC/LDC/LDCU load
/// (`.64` has one hi def, `.128` has three) by looking up the relocated
/// param word in the ABI annotations.  Falls through to `None` so the
/// caller can use the raw `+4*def_idx` offset as a backup.
fn lift_uldc_wide_def_from_lo(
    lo_expr: &IRExpr,
    def_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let IRExpr::Op { op, args } = lo_expr else {
        return None;
    };
    if op != "ConstMem" || args.len() != 2 {
        return None;
    }
    let bank = imm_as_u32(&args[0])?;
    let offset = imm_as_u32(&args[1])?;
    let anns = config.abi_annotations?;
    let matches_stmt = anns.constmem_by_stmt.get(&stmt_ref)?;
    let ann = matches_stmt
        .iter()
        .find(|ann| ann.bank == bank && ann.offset == offset)?;
    if let ConstMemSemantic::ParamWord { param_idx, .. } = ann.semantic {
        // With per-word param indexing each successive 32-bit lane of the
        // wide load lives at `param_idx + def_idx`, not at a different
        // word_idx within the same param.
        let hi_param = u32::try_from(def_idx).ok()?;
        let hi_param = param_idx.checked_add(hi_param)?;
        if let Some(aliases) = config.abi_aliases {
            // Try rendering via the alias map — this handles Ptr64 pairs that
            // were merged under the even param_idx.
            if let Some(alias) = aliases.render_param_word(hi_param, 0) {
                return Some(LiftedExpr::ConstMemSymbol(alias));
            }
        }
        return Some(LiftedExpr::ConstMemSymbol(format!("param_{}", hi_param)));
    }
    None
}

fn lift_s2r(args: &[IRExpr]) -> Option<LiftedExpr> {
    if args.len() != 1 {
        return None;
    }
    let IRExpr::Op { op, args } = &args[0] else {
        return None;
    };
    if !args.is_empty() {
        return None;
    }
    let sym = match op.as_str() {
        "SR_CTAID.X" => "blockIdx.x",
        "SR_CTAID.Y" => "blockIdx.y",
        "SR_CTAID.Z" => "blockIdx.z",
        "SR_TID.X" => "threadIdx.x",
        "SR_TID.Y" => "threadIdx.y",
        "SR_TID.Z" => "threadIdx.z",
        "SR_NTID.X" => "blockDim.x",
        "SR_NTID.Y" => "blockDim.y",
        "SR_NTID.Z" => "blockDim.z",
        "SR_LANEID" => "laneId",
        "SR_CgaCtaId" => "cgaCtaId",
        "SRZ" => "0",
        _ => return None,
    };
    Some(LiftedExpr::Raw(sym.to_string()))
}

fn memory_lane_count(opcode: &str) -> usize {
    if opcode_has_mod(opcode, "128") {
        4
    } else if opcode_has_mod(opcode, "64") {
        2
    } else {
        1
    }
}

fn scalar_byte_width_from_opcode(opcode: &str) -> Option<i64> {
    for tok in opcode.split('.') {
        match tok {
            "U8" | "S8" => return Some(1),
            "U16" | "S16" => return Some(2),
            "U32" | "S32" => return Some(4),
            "U64" | "S64" => return Some(8),
            _ => {}
        }
    }
    None
}

fn memory_lane_byte_stride(opcode: &str) -> i64 {
    if memory_lane_count(opcode) > 1 {
        4
    } else {
        scalar_byte_width_from_opcode(opcode).unwrap_or(4)
    }
}

fn memory_lane_type(opcode: &str) -> Option<String> {
    if memory_lane_count(opcode) > 1 {
        Some("uint32_t".to_string())
    } else {
        scalar_type_from_opcode(opcode).map(str::to_string)
    }
}

fn nth_lane_ir_expr(expr: &IRExpr, lane_idx: usize) -> Option<IRExpr> {
    match expr {
        IRExpr::Reg(reg) => {
            let mut lane = reg.clone();
            lane.idx += lane_idx as i32;
            Some(IRExpr::Reg(lane))
        }
        IRExpr::Op { op, args } if op == "ConstMem" && args.len() == 2 => {
            constmem_plus_word_offset_n(expr, lane_idx)
        }
        _ if lane_idx == 0 => Some(expr.clone()),
        _ => None,
    }
}

fn store_lane_ir_expr(args: &[IRExpr], lane_idx: usize, lane_count: usize) -> Option<IRExpr> {
    if lane_count > 1 {
        args.get(1 + lane_idx).cloned()
    } else {
        nth_lane_ir_expr(&args[1], lane_idx)
    }
}

fn store_lane_type(
    opcode: &str,
    mem_expr: &IRExpr,
    source_expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<String> {
    let expected_width = memory_lane_byte_stride(opcode) as usize;
    let pointee_ty =
        memory_lane_type_from_symbolic_pointer(mem_expr, stmt_ref, config, expected_width);
    let inferred_ty = expr_store_lane_type(source_expr, inferred_types, expected_width);
    memory_lane_type(opcode)
        .map(|ty| maybe_prefer_pointer_lane_type(opcode, ty, pointee_ty.as_deref(), expected_width))
        .or_else(|| {
            inferred_ty.clone().map(|ty| {
                maybe_prefer_pointer_lane_type(opcode, ty, pointee_ty.as_deref(), expected_width)
            })
        })
        .or(pointee_ty)
        .or_else(|| default_memory_lane_type(expected_width).map(str::to_string))
}

fn expr_store_lane_type(
    source_expr: &IRExpr,
    inferred_types: &BTreeMap<RegId, InferredType>,
    expected_width: usize,
) -> Option<String> {
    match source_expr {
        IRExpr::ImmF(_) if expected_width == 4 => Some("float".to_string()),
        IRExpr::ImmI(_) => None,
        _ => source_expr
            .get_reg()
            .and_then(|reg| inferred_reg_type(Some(reg), inferred_types))
            .and_then(|ty| inferred_memory_lane_type(ty, expected_width))
            .map(str::to_string),
    }
}

fn inferred_memory_lane_type(ty: InferredType, expected_width: usize) -> Option<&'static str> {
    match ty {
        InferredType::U8 if expected_width == 1 => Some("uint8_t"),
        InferredType::U16 if expected_width == 2 => Some("uint16_t"),
        InferredType::U32 | InferredType::AnyInt if expected_width == 4 => Some("uint32_t"),
        InferredType::I32 if expected_width == 4 => Some("int32_t"),
        InferredType::F16 if expected_width == 2 => Some("__half"),
        InferredType::F32 | InferredType::AnyFloat if expected_width == 4 => Some("float"),
        InferredType::U64 if expected_width == 8 => Some("uint64_t"),
        _ => None,
    }
}

fn default_memory_lane_type(expected_width: usize) -> Option<&'static str> {
    match expected_width {
        1 => Some("uint8_t"),
        2 => Some("uint16_t"),
        4 => Some("uint32_t"),
        8 => Some("uint64_t"),
        _ => None,
    }
}

fn memory_lane_type_from_symbolic_pointer(
    mem_expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    expected_width: usize,
) -> Option<String> {
    let aliases = config.abi_aliases?;
    let addr = lift_addr_expr(mem_expr, stmt_ref, config)?;
    expr_symbolic_pointer_pointee_type(&addr, aliases, expected_width)
}

fn expr_symbolic_pointer_pointee_type(
    expr: &Expr,
    aliases: &AbiArgAliases,
    expected_width: usize,
) -> Option<String> {
    match expr {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            symbolic_pointer_name_pointee_type(text, aliases, expected_width)
        }
        Expr::PtrLane { base, .. } => symbolic_pointer_name_pointee_type(base, aliases, expected_width),
        Expr::LaneExtract { value, .. }
        | Expr::Unary { arg: value, .. }
        | Expr::Cast { expr: value, .. }
        | Expr::Load { addr: value, .. } => {
            expr_symbolic_pointer_pointee_type(value, aliases, expected_width)
        }
        Expr::Binary { lhs, rhs, .. } => {
            expr_symbolic_pointer_pointee_type(lhs, aliases, expected_width)
                .or_else(|| expr_symbolic_pointer_pointee_type(rhs, aliases, expected_width))
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => expr_symbolic_pointer_pointee_type(cond, aliases, expected_width)
            .or_else(|| expr_symbolic_pointer_pointee_type(then_expr, aliases, expected_width))
            .or_else(|| expr_symbolic_pointer_pointee_type(else_expr, aliases, expected_width)),
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => args
            .iter()
            .find_map(|arg| expr_symbolic_pointer_pointee_type(arg, aliases, expected_width)),
        Expr::WidePtr { base, offset } => expr_symbolic_pointer_pointee_type(base, aliases, expected_width)
            .or_else(|| expr_symbolic_pointer_pointee_type(offset, aliases, expected_width)),
        Expr::Addr64 { lo, hi } => {
            expr_symbolic_pointer_pointee_type(lo, aliases, expected_width)
                .or_else(|| expr_symbolic_pointer_pointee_type(hi, aliases, expected_width))
        }
        Expr::Index { base, index } => expr_symbolic_pointer_pointee_type(base, aliases, expected_width)
            .or_else(|| expr_symbolic_pointer_pointee_type(index, aliases, expected_width)),
        Expr::Imm(_) => None,
    }
}

fn symbolic_pointer_name_pointee_type(
    text: &str,
    aliases: &AbiArgAliases,
    expected_width: usize,
) -> Option<String> {
    let text = text
        .strip_suffix(".lo32")
        .or_else(|| text.strip_suffix(".hi32"))
        .unwrap_or(text);
    let idx = text.strip_prefix("arg")?.strip_suffix("_ptr")?.parse::<u32>().ok()?;
    let alias = aliases.by_param.get(&idx)?;
    if alias.kind != ArgAliasKind::Ptr64 {
        return None;
    }
    let pointee = alias.pointee_ty?;
    (scalar_type_width_bytes(pointee) == Some(expected_width)).then(|| pointee.to_string())
}

fn preferred_pointer_lane_type(
    current: &str,
    pointee: Option<&str>,
    expected_width: usize,
) -> Option<String> {
    let pointee = pointee?;
    if current == pointee {
        return Some(current.to_string());
    }
    if scalar_type_width_bytes(current) != Some(expected_width)
        || scalar_type_width_bytes(pointee) != Some(expected_width)
    {
        return Some(current.to_string());
    }
    if is_integer_lane_type(current) && !is_integer_lane_type(pointee) {
        return Some(pointee.to_string());
    }
    Some(current.to_string())
}

fn maybe_prefer_pointer_lane_type(
    opcode: &str,
    current: String,
    pointee: Option<&str>,
    expected_width: usize,
) -> String {
    if memory_lane_count(opcode) == 1 && scalar_type_from_opcode(opcode).is_some() {
        return current;
    }
    preferred_pointer_lane_type(&current, pointee, expected_width).unwrap_or(current)
}

fn is_integer_lane_type(ty: &str) -> bool {
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

fn scalar_type_width_bytes(ty: &str) -> Option<usize> {
    match ty {
        "uint8_t" | "int8_t" => Some(1),
        "uint16_t" | "int16_t" | "__half" => Some(2),
        "uint32_t" | "int32_t" | "float" => Some(4),
        "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t" => Some(8),
        _ => None,
    }
}

fn add_byte_offset_expr(base: Expr, byte_offset: i64) -> Expr {
    if byte_offset == 0 {
        base
    } else {
        add_like_expr(base, Expr::Imm(byte_offset.to_string()))
    }
}

fn lift_addr_expr_with_byte_offset(
    mem_expr: &IRExpr,
    byte_offset: i64,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<Expr> {
    let addr = lift_addr_expr(mem_expr, stmt_ref, config)?;
    Some(add_byte_offset_expr(addr, byte_offset))
}

fn shared_lvalue_with_byte_offset(
    mem_expr: &IRExpr,
    byte_mode: bool,
    byte_offset: i64,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LValue> {
    let index = shared_index_expr(mem_expr, stmt_ref, config)?;
    let index = add_byte_offset_expr(index, byte_offset);
    let base = if byte_mode { "shmem_u8" } else { "shmem" };
    Some(LValue::Indexed {
        base: Box::new(Expr::Builtin(base.to_string())),
        index: Box::new(index),
    })
}

fn mem_load_expr_with_byte_offset(
    mem_expr: &IRExpr,
    lane_ty: Option<String>,
    byte_offset: i64,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<Expr> {
    let addr = lift_addr_expr_with_byte_offset(mem_expr, byte_offset, stmt_ref, config)?;
    Some(Expr::Load {
        ty: lane_ty,
        addr: Box::new(addr),
    })
}

fn mem_store_lvalue_with_byte_offset(
    mem_expr: &IRExpr,
    lane_ty: Option<String>,
    byte_offset: i64,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LValue> {
    let addr = lift_addr_expr_with_byte_offset(mem_expr, byte_offset, stmt_ref, config)?;
    Some(LValue::Deref {
        ty: lane_ty,
        addr: Box::new(addr),
    })
}

fn lift_store_stmt_lanes(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<Vec<(LValue, LiftedExpr)>> {
    if args.len() < 2 || !is_mem_expr(&args[0]) {
        return None;
    }
    let lane_count = memory_lane_count(opcode);
    let lane_stride = memory_lane_byte_stride(opcode);
    let mut stores = Vec::with_capacity(lane_count);

    if opcode.starts_with("STS") {
        let byte_mode = opcode.contains(".U8");
        for lane_idx in 0..lane_count {
            let dest = shared_lvalue_with_byte_offset(
                &args[0],
                byte_mode,
                lane_idx as i64 * lane_stride,
                stmt_ref,
                config,
            )?;
            let rhs = lift_ir_expr(
                &store_lane_ir_expr(args, lane_idx, lane_count)?,
                stmt_ref,
                config,
            );
            stores.push((dest, rhs));
        }
        return Some(stores);
    }

    if opcode.starts_with("STG") || opcode.starts_with("STL") {
        for lane_idx in 0..lane_count {
            let source_expr = store_lane_ir_expr(args, lane_idx, lane_count)?;
            let dest = mem_store_lvalue_with_byte_offset(
                &args[0],
                store_lane_type(opcode, &args[0], &source_expr, stmt_ref, config, inferred_types),
                lane_idx as i64 * lane_stride,
                stmt_ref,
                config,
            )
            .unwrap_or_else(|| LValue::Raw(render_expr_raw(&args[0], stmt_ref, config)));
            let rhs = lift_ir_expr(&source_expr, stmt_ref, config);
            stores.push((dest, rhs));
        }
        return Some(stores);
    }

    None
}

fn lvalue_from_ir_def(
    expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> LValue {
    match expr {
        IRExpr::Reg(r) => {
            if matches!(r.class.as_str(), "RZ" | "URZ") {
                return LValue::Raw("0".to_string());
            }
            if matches!(r.class.as_str(), "PT" | "UPT") {
                return LValue::Raw("true".to_string());
            }
            LValue::Var(r.display())
        }
        IRExpr::ImmI(i) => LValue::Raw(i.to_string()),
        IRExpr::ImmF(f) => LValue::Raw(f.to_string()),
        IRExpr::Addr64 { .. } | IRExpr::Mem { .. } | IRExpr::Op { .. } => {
            LValue::Raw(render_expr_raw(expr, stmt_ref, config))
        }
    }
}

pub(crate) fn lift_shared_ref_expr(
    mem_expr: &IRExpr,
    byte_mode: bool,
    byte_offset: i64,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<Expr> {
    let shared = shared_lvalue_with_byte_offset(mem_expr, byte_mode, byte_offset, stmt_ref, config)?;
    Some(match shared {
        LValue::Indexed { base, index } => Expr::Index { base, index },
        other => Expr::Raw(other.render()),
    })
}

fn shared_index_expr(
    mem_expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<Expr> {
    let IRExpr::Mem { base, offset, .. } = mem_expr else {
        return None;
    };
    let index = lift_ir_expr(base, stmt_ref, config);
    Some(match offset {
        Some(off) => add_like_expr(index, lift_ir_expr(off, stmt_ref, config)),
        None => index,
    })
}

pub(crate) fn lift_addr_expr(
    mem_expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<Expr> {
    let IRExpr::Mem {
        base,
        offset,
        width,
    } = mem_expr
    else {
        return None;
    };
    let addr = if matches!(width, Some(64)) {
        match base.as_ref() {
            IRExpr::Addr64 { lo, hi } => Expr::Addr64 {
                lo: Box::new(lift_ir_expr(lo.as_ref(), stmt_ref, config)),
                hi: Box::new(lift_ir_expr(hi.as_ref(), stmt_ref, config)),
            },
            _ => lift_ir_expr(base, stmt_ref, config),
        }
    } else {
        lift_ir_expr(base, stmt_ref, config)
    };
    Some(match (matches!(width, Some(64)), offset) {
        (true, Some(off)) => Expr::WidePtr {
            base: Box::new(addr),
            offset: Box::new(lift_ir_expr(off, stmt_ref, config)),
        },
        (_, Some(off)) => add_like_expr(addr, lift_ir_expr(off, stmt_ref, config)),
        (_, None) => addr,
    })
}

fn lift_imad_mov(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
        return Some(lift_ir_expr(&args[2], stmt_ref, config));
    }
    None
}

fn lift_iadd3(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    let a0z = is_zero_expr(a0);
    let a1z = is_zero_expr(a1);
    let a2z = is_zero_expr(a2);

    let expr = if a0z && !a1z && !a2z {
        add_like_expr(
            lift_ir_expr(a1, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        )
    } else if a1z && !a0z && !a2z {
        add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        )
    } else if a2z && !a0z && !a1z {
        add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a1, stmt_ref, config),
        )
    } else {
        let left = add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a1, stmt_ref, config),
        );
        add_like_expr(left, lift_ir_expr(a2, stmt_ref, config))
    };

    Some(expr)
}

fn lift_iadd3_64_def(
    args: &[IRExpr],
    def_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    def_reg: Option<&RegId>,
    wide32_hi_map: &BTreeMap<RegId, Wide32HiInfo>,
    known_zero_regs: &BTreeSet<RegId>,
) -> Option<LiftedExpr> {
    if args.len() != 6 {
        return None;
    }
    if let Some(value) = lift_iadd3_64_scalar_value(
        args,
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    ) {
        return match def_idx {
            0 => Some(LiftedExpr::LaneExtract {
                value: Box::new(value),
                lane: PointerLane::Lo32,
            }),
            idx if def_reg.is_some_and(|reg| matches!(reg.class.as_str(), "P" | "UP")) => {
                debug_assert!(idx > 0);
                lift_iadd3_carry(&args[..3], stmt_ref, config)
            }
            _ => Some(LiftedExpr::LaneExtract {
                value: Box::new(value),
                lane: PointerLane::Hi32,
            }),
        };
    }
    if def_idx == 0 {
        return lift_iadd3(&args[..3], stmt_ref, config);
    }
    if def_reg.is_some_and(|reg| matches!(reg.class.as_str(), "P" | "UP")) {
        return lift_iadd3_carry(&args[..3], stmt_ref, config);
    }

    let hi0 = lift_ir_expr(&args[3], stmt_ref, config);
    let hi1 = lift_ir_expr(&args[4], stmt_ref, config);
    let hi2 = lift_ir_expr(&args[5], stmt_ref, config);
    let carry = LiftedExpr::Ternary {
        cond: Box::new(intrinsic_expr(
            IntrinsicOp::CarryU32Add3,
            vec![
                lift_ir_expr(&args[0], stmt_ref, config),
                lift_ir_expr(&args[1], stmt_ref, config),
                lift_ir_expr(&args[2], stmt_ref, config),
            ],
        )),
        then_expr: Box::new(LiftedExpr::Imm("1".to_string())),
        else_expr: Box::new(LiftedExpr::Imm("0".to_string())),
    };
    let hi_sum = add_like_expr(add_like_expr(hi0, hi1), hi2);
    Some(add_like_expr(hi_sum, carry))
}

fn lift_iadd3_64_scalar_value(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    wide32_hi_map: &BTreeMap<RegId, Wide32HiInfo>,
    known_zero_regs: &BTreeSet<RegId>,
) -> Option<LiftedExpr> {
    let lhs = lift_wide_add_input(
        &args[0],
        &args[3],
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    )?;
    let rhs = lift_wide_add_input(
        &args[1],
        &args[4],
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    )?;
    let extra = lift_wide_add_input(
        &args[2],
        &args[5],
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    )?;
    Some(add_like_expr(add_like_expr(lhs, rhs), extra))
}

fn lift_iadd3_carry(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    Some(intrinsic_expr(
        IntrinsicOp::CarryU32Add3,
        vec![
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a1, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        ],
    ))
}

fn lift_iadd3_x(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2, carry_pred) = extract_addx_operands(args)?;
    if config.strict && !is_zero_expr(a2) {
        return None;
    }

    let mut sum = add_like_expr(
        lift_ir_expr(a0, stmt_ref, config),
        lift_ir_expr(a1, stmt_ref, config),
    );
    if !is_zero_expr(a2) {
        sum = add_like_expr(sum, lift_ir_expr(a2, stmt_ref, config));
    }
    let carry = carry_inc_expr(carry_pred, stmt_ref, config)?;
    Some(add_like_expr(sum, carry))
}

fn lift_imad_iadd(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    if is_imm_i(a1, 1) {
        return Some(add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        ));
    }
    if is_imm_i(a0, 1) {
        return Some(add_like_expr(
            lift_ir_expr(a1, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        ));
    }
    let mul = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lift_ir_expr(a0, stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(a1, stmt_ref, config)),
    };
    Some(add_like_expr(mul, lift_ir_expr(a2, stmt_ref, config)))
}

fn lift_imad_wide(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    let mul = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    };
    Some(add_like_expr(mul, lift_ir_expr(&args[2], stmt_ref, config)))
}

fn lift_imad_wide_hi(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    pair_hi_map: &BTreeMap<RegId, IRExpr>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    let lhs = lift_ir_expr(a0, stmt_ref, config);
    let rhs = lift_ir_expr(a1, stmt_ref, config);
    let base_lo = lift_ir_expr(a2, stmt_ref, config);
    let base_hi = lift_paired_hi_from_lo(a2, stmt_ref, config, pair_hi_map);
    let mul_lo = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    };
    let mul_hi = call_expr("mul_hi_u32", vec![lhs.clone(), rhs.clone()]);
    let carry = LiftedExpr::Ternary {
        cond: Box::new(intrinsic_expr(
            IntrinsicOp::CarryU32Add3,
            vec![
                mul_lo.clone(),
                base_lo.clone(),
                LiftedExpr::Imm("0".to_string()),
            ],
        )),
        then_expr: Box::new(LiftedExpr::Imm("1".to_string())),
        else_expr: Box::new(LiftedExpr::Imm("0".to_string())),
    };
    Some(add_like_expr(add_like_expr(mul_hi, base_hi), carry))
}

fn lift_imad(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    let mul = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lift_ir_expr(a0, stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(a1, stmt_ref, config)),
    };
    Some(add_like_expr(mul, lift_ir_expr(a2, stmt_ref, config)))
}

fn lift_imad_hi_u32(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    let hi = call_expr(
        "mul_hi_u32",
        vec![
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a1, stmt_ref, config),
        ],
    );
    if is_zero_expr(a2) {
        return Some(hi);
    }
    Some(add_like_expr(hi, lift_ir_expr(a2, stmt_ref, config)))
}

fn lift_imad_x(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 4 {
        return None;
    }
    let a0z = is_zero_expr(&args[0]);
    let a1z = is_zero_expr(&args[1]);
    if !(a0z && a1z) {
        return None;
    }
    let base = lift_ir_expr(&args[2], stmt_ref, config);
    let carry = carry_inc_expr(&args[3], stmt_ref, config)?;
    Some(add_like_expr(base, carry))
}

fn lift_iabs(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    lift_unary_intrinsic("abs", args, stmt_ref, config)
}

fn lift_mufu_rcp(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    lift_unary_intrinsic("rcp_approx", args, stmt_ref, config)
}

fn lift_ffma(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    let mul = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lift_float_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_float_ir_expr(&args[1], stmt_ref, config)),
    };
    Some(add_like_expr(
        mul,
        lift_float_ir_expr(&args[2], stmt_ref, config),
    ))
}

fn lift_fmnmx(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    let pred = &args[2];

    // FMNMX semantics: when pred is true → min, when false → max.
    // Check for constant predicates first to avoid unnecessary allocations.
    if is_true_pred_expr(pred) {
        let a = lift_ir_expr(&args[0], stmt_ref, config);
        let b = lift_ir_expr(&args[1], stmt_ref, config);
        return Some(LiftedExpr::Raw(format!(
            "fminf({}, {})",
            a.render(),
            b.render()
        )));
    }
    if is_false_pred_expr(pred) {
        let a = lift_ir_expr(&args[0], stmt_ref, config);
        let b = lift_ir_expr(&args[1], stmt_ref, config);
        return Some(LiftedExpr::Raw(format!(
            "fmaxf({}, {})",
            a.render(),
            b.render()
        )));
    }

    // General case: pred ? fminf(a,b) : fmaxf(a,b)
    let a = lift_ir_expr(&args[0], stmt_ref, config);
    let b = lift_ir_expr(&args[1], stmt_ref, config);
    let fmin = LiftedExpr::Raw(format!("fminf({}, {})", a.render(), b.render()));
    let fmax = LiftedExpr::Raw(format!("fmaxf({}, {})", a.render(), b.render()));
    Some(LiftedExpr::Ternary {
        cond: Box::new(lift_ir_expr(pred, stmt_ref, config)),
        then_expr: Box::new(fmin),
        else_expr: Box::new(fmax),
    })
}

fn lift_uldc64(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 1 {
        return None;
    }
    // ULDC/LDC/LDCU loads a constant memory operand into a register.
    // For the modeled low-half (or only) def, render the source symbol
    // directly instead of a synthetic helper call.
    Some(lift_ir_expr(&args[0], stmt_ref, config))
}

fn lift_unary_intrinsic(
    name: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 1 {
        return None;
    }
    let arg = lift_ir_expr(&args[0], stmt_ref, config);
    if let Some(ty) = name
        .strip_prefix('(')
        .and_then(|text| text.strip_suffix(')'))
    {
        return Some(LiftedExpr::Cast {
            ty: ty.to_string(),
            expr: Box::new(arg),
        });
    }
    Some(LiftedExpr::CallLike {
        func: name.to_string(),
        args: vec![arg],
    })
}

#[allow(dead_code)]
fn lift_binary_infix(
    op: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 2 {
        return None;
    }
    Some(LiftedExpr::Binary {
        op: op.to_string(),
        lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
}

pub(crate) fn lift_float_binary_infix(
    op: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 2 {
        return None;
    }
    Some(LiftedExpr::Binary {
        op: op.to_string(),
        lhs: Box::new(lift_float_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_float_ir_expr(&args[1], stmt_ref, config)),
    })
}

fn lift_binary_add_like(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 2 {
        return None;
    }
    Some(add_like_expr(
        lift_ir_expr(&args[0], stmt_ref, config),
        lift_ir_expr(&args[1], stmt_ref, config),
    ))
}

fn lift_mov64_def(
    args: &[IRExpr],
    def_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if def_idx >= args.len() {
        return None;
    }
    Some(lift_ir_expr(&args[def_idx], stmt_ref, config))
}

fn lift_iadd64_def(
    args: &[IRExpr],
    def_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    wide32_hi_map: &BTreeMap<RegId, Wide32HiInfo>,
    known_zero_regs: &BTreeSet<RegId>,
) -> Option<LiftedExpr> {
    if args.len() != 4 {
        return None;
    }
    if let Some(value) = lift_iadd64_scalar_value(
        args,
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    ) {
        return Some(LiftedExpr::LaneExtract {
            value: Box::new(value),
            lane: if def_idx == 0 {
                PointerLane::Lo32
            } else if def_idx == 1 {
                PointerLane::Hi32
            } else {
                return None;
            },
        });
    }
    let lo_lhs = lift_ir_expr(&args[0], stmt_ref, config);
    let lo_rhs = lift_ir_expr(&args[1], stmt_ref, config);
    if def_idx == 0 {
        return Some(add_like_expr(lo_lhs, lo_rhs));
    }
    if def_idx != 1 {
        return None;
    }

    let hi_lhs = lift_ir_expr(&args[2], stmt_ref, config);
    let hi_rhs = lift_ir_expr(&args[3], stmt_ref, config);
    let carry = LiftedExpr::Ternary {
        cond: Box::new(intrinsic_expr(
            IntrinsicOp::CarryU32Add3,
            vec![
                lift_ir_expr(&args[0], stmt_ref, config),
                lift_ir_expr(&args[1], stmt_ref, config),
                LiftedExpr::Imm("0".to_string()),
            ],
        )),
        then_expr: Box::new(LiftedExpr::Imm("1".to_string())),
        else_expr: Box::new(LiftedExpr::Imm("0".to_string())),
    };
    Some(add_like_expr(add_like_expr(hi_lhs, hi_rhs), carry))
}

fn lift_iadd64_scalar_value(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    wide32_hi_map: &BTreeMap<RegId, Wide32HiInfo>,
    known_zero_regs: &BTreeSet<RegId>,
) -> Option<LiftedExpr> {
    let lhs = lift_wide_add_input(
        &args[0],
        &args[2],
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    )?;
    let rhs = lift_wide_add_input(
        &args[1],
        &args[3],
        stmt_ref,
        config,
        wide32_hi_map,
        known_zero_regs,
    )?;
    Some(add_like_expr(lhs, rhs))
}

fn lift_wide_add_input(
    low: &IRExpr,
    hi: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    wide32_hi_map: &BTreeMap<RegId, Wide32HiInfo>,
    known_zero_regs: &BTreeSet<RegId>,
) -> Option<LiftedExpr> {
    if is_zero_expr(hi) || hi.get_reg().is_some_and(|reg| known_zero_regs.contains(&reg)) {
        return Some(widen_u32_expr(lift_ir_expr(low, stmt_ref, config)));
    }
    if let (Some(low_reg), Some(hi_reg)) = (low.get_reg(), hi.get_reg()) {
        if let Some(info) = wide32_hi_map.get(&hi_reg) {
            if info.low == *low_reg && info.kind == Wide32ExtKind::Signed {
                return Some(widen_i32_expr(lift_ir_expr(low, stmt_ref, config)));
            }
        }
    }

    // Blackwell often mixes a real 64-bit pointer pair with a widened
    // 32-bit scalar in IADD.64/UIADD3.64 chains. If we only recognize the
    // widened scalar case, we discard the dominant pointer pair and later
    // addr64 recovery can invert base/offset roles. Preserve any remaining
    // explicit wide input as a packed lo/hi pair so downstream AST folding
    // can still reconstruct the correct base-relative pointer.
    Some(LiftedExpr::Addr64 {
        lo: Box::new(lift_ir_expr(low, stmt_ref, config)),
        hi: Box::new(lift_ir_expr(hi, stmt_ref, config)),
    })
}

fn widen_u32_expr(expr: LiftedExpr) -> LiftedExpr {
    LiftedExpr::Cast {
        ty: "uint64_t".to_string(),
        expr: Box::new(LiftedExpr::Cast {
            ty: "uint32_t".to_string(),
            expr: Box::new(expr),
        }),
    }
}

fn widen_i32_expr(expr: LiftedExpr) -> LiftedExpr {
    LiftedExpr::Cast {
        ty: "int64_t".to_string(),
        expr: Box::new(LiftedExpr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(expr),
        }),
    }
}

pub(crate) fn lift_float_binary_add_like(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 2 {
        return None;
    }
    Some(add_like_expr(
        lift_float_ir_expr(&args[0], stmt_ref, config),
        lift_float_ir_expr(&args[1], stmt_ref, config),
    ))
}

pub(crate) fn lift_lds_expr(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    lift_lds_lane_expr(opcode, args, 0, stmt_ref, config)
}

pub(crate) fn lift_lds_lane_expr(
    opcode: &str,
    args: &[IRExpr],
    lane_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 1 || !is_mem_expr(&args[0]) {
        return None;
    }
    let byte_mode = opcode.contains(".U8");
    let byte_offset = lane_idx as i64 * memory_lane_byte_stride(opcode);
    lift_shared_ref_expr(&args[0], byte_mode, byte_offset, stmt_ref, config)
}

pub(crate) fn lift_ldg_expr(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    lift_ldg_lane_expr(opcode, args, 0, stmt_ref, config, None, &BTreeMap::new())
}

pub(crate) fn lift_ldg_lane_expr(
    opcode: &str,
    args: &[IRExpr],
    lane_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    def_reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<LiftedExpr> {
    if args.len() != 1 || !is_mem_expr(&args[0]) {
        return None;
    }
    let expected_width = memory_lane_byte_stride(opcode) as usize;
    let byte_offset = lane_idx as i64 * expected_width as i64;
    let pointee_ty =
        memory_lane_type_from_symbolic_pointer(&args[0], stmt_ref, config, expected_width);
    let inferred_ty =
        inferred_reg_type(def_reg, inferred_types).map(|ty| ty.to_c_type().to_string());
    let lane_ty = memory_lane_type(opcode)
        .map(|ty| maybe_prefer_pointer_lane_type(opcode, ty, pointee_ty.as_deref(), expected_width))
        .or_else(|| {
            inferred_ty.clone().map(|ty| {
                maybe_prefer_pointer_lane_type(opcode, ty, pointee_ty.as_deref(), expected_width)
            })
        })
        .or(pointee_ty)
        .or_else(|| default_memory_lane_type(expected_width).map(str::to_string));
    if let Some(rendered) =
        mem_load_expr_with_byte_offset(&args[0], lane_ty, byte_offset, stmt_ref, config)
    {
        return Some(rendered);
    }
    Some(LiftedExpr::Raw(render_expr_raw(&args[0], stmt_ref, config)))
}

pub(crate) fn lift_ldl_expr(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    lift_ldl_lane_expr(opcode, args, 0, stmt_ref, config, None, &BTreeMap::new())
}

pub(crate) fn lift_ldl_lane_expr(
    opcode: &str,
    args: &[IRExpr],
    lane_idx: usize,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
    def_reg: Option<&RegId>,
    inferred_types: &BTreeMap<RegId, InferredType>,
) -> Option<LiftedExpr> {
    if args.len() != 1 || !is_mem_expr(&args[0]) {
        return None;
    }
    let expected_width = memory_lane_byte_stride(opcode) as usize;
    let byte_offset = lane_idx as i64 * expected_width as i64;
    let pointee_ty =
        memory_lane_type_from_symbolic_pointer(&args[0], stmt_ref, config, expected_width);
    let inferred_ty =
        inferred_reg_type(def_reg, inferred_types).map(|ty| ty.to_c_type().to_string());
    let lane_ty = memory_lane_type(opcode)
        .map(|ty| maybe_prefer_pointer_lane_type(opcode, ty, pointee_ty.as_deref(), expected_width))
        .or_else(|| {
            inferred_ty.clone().map(|ty| {
                maybe_prefer_pointer_lane_type(opcode, ty, pointee_ty.as_deref(), expected_width)
            })
        })
        .or(pointee_ty)
        .or_else(|| default_memory_lane_type(expected_width).map(str::to_string));
    if let Some(rendered) =
        mem_load_expr_with_byte_offset(&args[0], lane_ty, byte_offset, stmt_ref, config)
    {
        return Some(rendered);
    }
    Some(LiftedExpr::Raw(render_expr_raw(&args[0], stmt_ref, config)))
}

pub(crate) fn scalar_type_from_opcode(opcode: &str) -> Option<&'static str> {
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

fn lift_fsel(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    Some(LiftedExpr::Ternary {
        cond: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
        then_expr: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        else_expr: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
}

fn lift_setp_compare(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (lhs_arg, rhs_arg, combine_arg) = match args {
        [lhs, rhs, combine] => (lhs, rhs, combine),
        [_, lhs, rhs, combine, ..] => (lhs, rhs, combine),
        _ => return None,
    };

    // Determine predicate combine mode from opcode: AND, OR, XOR.
    let combine_mode = opcode
        .split('.')
        .find(|p| matches!(*p, "AND" | "OR" | "XOR"));

    let cmp = cmp_token_to_op(opcode)?;

    let mut lhs = lift_ir_expr(lhs_arg, stmt_ref, config);
    let mut rhs = lift_ir_expr(rhs_arg, stmt_ref, config);

    // For integer comparisons (ISETP): if the opcode does NOT contain `.U32`,
    // the comparison is signed. Wrap operands in `(int32_t)` casts for
    // ordered comparisons (< <= > >=) so the C output has correct semantics
    // even when the recovered locals are declared `uint32_t`.
    if is_signed_int_compare(opcode) && is_ordered_cmp(cmp) {
        lhs = signed_cast(lhs);
        rhs = signed_cast(rhs);
    }

    let compare_expr = LiftedExpr::Binary {
        op: cmp.to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    };

    if is_true_pred_expr(combine_arg) {
        return Some(compare_expr);
    }

    let pred_expr = lift_ir_expr(combine_arg, stmt_ref, config);
    match combine_mode {
        Some("OR") => Some(LiftedExpr::Binary {
            op: "||".to_string(),
            lhs: Box::new(compare_expr),
            rhs: Box::new(pred_expr),
        }),
        Some("AND") => Some(LiftedExpr::Binary {
            op: "&&".to_string(),
            lhs: Box::new(compare_expr),
            rhs: Box::new(pred_expr),
        }),
        Some("XOR") => Some(LiftedExpr::Binary {
            op: "^".to_string(),
            lhs: Box::new(compare_expr),
            rhs: Box::new(pred_expr),
        }),
        _ if config.strict => None,
        _ => Some(compare_expr),
    }
}

fn lift_lop3_lut(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    // Conservative subset:
    // LOP3.LUT dst, a, b, c, imm, PT/!PT
    // currently only when one or two inputs are zero (RZ/URZ),
    // reducing to unary/binary bitwise forms.
    if args.len() != 5 {
        return None;
    }
    if !is_pred_control_expr(&args[4]) {
        return None;
    }

    // The & 0xff mask guarantees the value fits in u8; try_from is belt-and-suspenders.
    let imm = u8::try_from(imm_as_u32(&args[3])? & 0xff).ok()?;
    let a0z = is_zero_expr(&args[0]);
    let a1z = is_zero_expr(&args[1]);
    let a2z = is_zero_expr(&args[2]);

    // Two fixed-zero inputs -> unary form on the remaining input.
    if a0z && a1z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 0, 0, 1) as u8) << 1);
        return lop3_unary_expr(lift_ir_expr(&args[2], stmt_ref, config), bits);
    }
    if a0z && a2z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 0, 1, 0) as u8) << 1);
        return lop3_unary_expr(lift_ir_expr(&args[1], stmt_ref, config), bits);
    }
    if a1z && a2z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 1, 0, 0) as u8) << 1);
        return lop3_unary_expr(lift_ir_expr(&args[0], stmt_ref, config), bits);
    }

    // One fixed-zero input -> binary form on remaining inputs.
    if a2z {
        // f(a, b, 0): truth-table rows (a,b)=00,01,10,11 -> bits 0,2,4,6
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 1, 0) as u8) << 1)
            | ((lop3_bit(imm, 1, 0, 0) as u8) << 2)
            | ((lop3_bit(imm, 1, 1, 0) as u8) << 3);
        return lop3_binary_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[1], stmt_ref, config),
            nibble,
        );
    }
    if a0z {
        // f(0, b, c): rows (b,c)=00,01,10,11 -> bits 0,1,2,3
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 0, 1) as u8) << 1)
            | ((lop3_bit(imm, 0, 1, 0) as u8) << 2)
            | ((lop3_bit(imm, 0, 1, 1) as u8) << 3);
        return lop3_binary_expr(
            lift_ir_expr(&args[1], stmt_ref, config),
            lift_ir_expr(&args[2], stmt_ref, config),
            nibble,
        );
    }
    if a1z {
        // f(a, 0, c): rows (a,c)=00,01,10,11 -> bits 0,1,4,5
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 0, 1) as u8) << 1)
            | ((lop3_bit(imm, 1, 0, 0) as u8) << 2)
            | ((lop3_bit(imm, 1, 0, 1) as u8) << 3);
        return lop3_binary_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[2], stmt_ref, config),
            nibble,
        );
    }

    // General 3-input case: all inputs are non-zero.
    // Check for common recognizable patterns first, then fall back to a
    // readable helper form `lop3_lut_0xNN(a, b, c)`.
    let a_expr = lift_ir_expr(&args[0], stmt_ref, config);
    let b_expr = lift_ir_expr(&args[1], stmt_ref, config);
    let c_expr = lift_ir_expr(&args[2], stmt_ref, config);

    lop3_ternary_expr(a_expr, b_expr, c_expr, imm)
}

fn lop3_bit(imm: u8, a: u8, b: u8, c: u8) -> bool {
    let idx = (a << 2) | (b << 1) | c;
    ((imm >> idx) & 1) != 0
}

fn lop3_unary_expr(x: LiftedExpr, bits: u8) -> Option<LiftedExpr> {
    match bits & 0x3 {
        0x0 => Some(LiftedExpr::Imm("0".to_string())),
        0x1 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x),
        }),
        0x2 => Some(x),
        0x3 => Some(LiftedExpr::Imm("0xffffffff".to_string())),
        _ => None,
    }
}

fn lop3_binary_expr(x: LiftedExpr, y: LiftedExpr, nibble: u8) -> Option<LiftedExpr> {
    let x_not = || LiftedExpr::Unary {
        op: "~".to_string(),
        arg: Box::new(x.clone()),
    };
    let y_not = || LiftedExpr::Unary {
        op: "~".to_string(),
        arg: Box::new(y.clone()),
    };
    let x_and_y = || LiftedExpr::Binary {
        op: "&".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    let x_or_y = || LiftedExpr::Binary {
        op: "|".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    let x_xor_y = || LiftedExpr::Binary {
        op: "^".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };

    match nibble & 0xf {
        0x0 => Some(LiftedExpr::Imm("0".to_string())),
        0x1 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_or_y()),
        }),
        0x2 => Some(LiftedExpr::Binary {
            op: "&".to_string(),
            lhs: Box::new(x_not()),
            rhs: Box::new(y.clone()),
        }),
        0x3 => Some(x_not()),
        0x4 => Some(LiftedExpr::Binary {
            op: "&".to_string(),
            lhs: Box::new(x.clone()),
            rhs: Box::new(y_not()),
        }),
        0x5 => Some(y_not()),
        0x6 => Some(x_xor_y()),
        0x7 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_and_y()),
        }),
        0x8 => Some(x_and_y()),
        0x9 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_xor_y()),
        }),
        0xa => Some(y.clone()),
        0xb => Some(LiftedExpr::Binary {
            op: "|".to_string(),
            lhs: Box::new(x_not()),
            rhs: Box::new(y.clone()),
        }),
        0xc => Some(x.clone()),
        0xd => Some(LiftedExpr::Binary {
            op: "|".to_string(),
            lhs: Box::new(x.clone()),
            rhs: Box::new(y_not()),
        }),
        0xe => Some(x_or_y()),
        0xf => Some(LiftedExpr::Imm("0xffffffff".to_string())),
        _ => None,
    }
}

fn lop3_ternary_expr(a: LiftedExpr, b: LiftedExpr, c: LiftedExpr, imm: u8) -> Option<LiftedExpr> {
    // Common recognizable 3-input LOP3 patterns.
    // Bit index = (a<<2)|(b<<1)|c, so a is the MSB of the 3-bit index.
    match imm {
        // Bitwise MUX: (a & b) | (~a & c)  (very common in crypto)
        // Note: this is a per-bit select, NOT C's scalar ternary operator.
        0xCA => {
            return Some(LiftedExpr::Raw(format!(
                "bitmux({}, {}, {})",
                a.render(),
                b.render(),
                c.render()
            )));
        }
        // Reversed bitwise MUX: (a & c) | (~a & b)
        0xAC => {
            return Some(LiftedExpr::Raw(format!(
                "bitmux({}, {}, {})",
                a.render(),
                c.render(),
                b.render()
            )));
        }
        // Bitwise MUX with b as selector: (b & a) | (~b & c)
        0xE2 => {
            return Some(LiftedExpr::Raw(format!(
                "bitmux({}, {}, {})",
                b.render(),
                a.render(),
                c.render()
            )));
        }
        // 3-way AND: a & b & c
        0x80 => {
            return Some(LiftedExpr::Binary {
                op: "&".to_string(),
                lhs: Box::new(LiftedExpr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(a),
                    rhs: Box::new(b),
                }),
                rhs: Box::new(c),
            });
        }
        // 3-way OR: a | b | c
        0xFE => {
            return Some(LiftedExpr::Binary {
                op: "|".to_string(),
                lhs: Box::new(LiftedExpr::Binary {
                    op: "|".to_string(),
                    lhs: Box::new(a),
                    rhs: Box::new(b),
                }),
                rhs: Box::new(c),
            });
        }
        // 3-way XOR: a ^ b ^ c
        0x96 => {
            return Some(LiftedExpr::Binary {
                op: "^".to_string(),
                lhs: Box::new(LiftedExpr::Binary {
                    op: "^".to_string(),
                    lhs: Box::new(a),
                    rhs: Box::new(b),
                }),
                rhs: Box::new(c),
            });
        }
        // Majority: (a & b) | (a & c) | (b & c)
        0xE8 => {
            return Some(LiftedExpr::Raw(format!(
                "majority({}, {}, {})",
                a.render(),
                b.render(),
                c.render()
            )));
        }
        // a | (b & c)
        0xF8 => {
            return Some(LiftedExpr::Binary {
                op: "|".to_string(),
                lhs: Box::new(a),
                rhs: Box::new(LiftedExpr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(b),
                    rhs: Box::new(c),
                }),
            });
        }
        // c & (a | b)
        0xA8 => {
            return Some(LiftedExpr::Binary {
                op: "&".to_string(),
                lhs: Box::new(c),
                rhs: Box::new(LiftedExpr::Binary {
                    op: "|".to_string(),
                    lhs: Box::new(a),
                    rhs: Box::new(b),
                }),
            });
        }
        // a & (b ^ c)  — common in crypto (conditional XOR)
        0x60 => {
            return Some(LiftedExpr::Binary {
                op: "&".to_string(),
                lhs: Box::new(a),
                rhs: Box::new(LiftedExpr::Binary {
                    op: "^".to_string(),
                    lhs: Box::new(b),
                    rhs: Box::new(c),
                }),
            });
        }
        // a & (b | c)  — symmetric to 0xA8
        0xE0 => {
            return Some(LiftedExpr::Binary {
                op: "&".to_string(),
                lhs: Box::new(a),
                rhs: Box::new(LiftedExpr::Binary {
                    op: "|".to_string(),
                    lhs: Box::new(b),
                    rhs: Box::new(c),
                }),
            });
        }
        // (a & b) | c  — symmetric to 0xF8
        0xEA => {
            return Some(LiftedExpr::Binary {
                op: "|".to_string(),
                lhs: Box::new(LiftedExpr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(a),
                    rhs: Box::new(b),
                }),
                rhs: Box::new(c),
            });
        }
        // a ^ (b & c)  — common in crypto (SHA)
        0x78 => {
            return Some(LiftedExpr::Binary {
                op: "^".to_string(),
                lhs: Box::new(a),
                rhs: Box::new(LiftedExpr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(b),
                    rhs: Box::new(c),
                }),
            });
        }
        _ => {}
    }

    // Fallback: emit a readable helper call instead of raw LOP3.LUT(...)
    Some(LiftedExpr::Raw(format!(
        "lop3_lut_0x{:02x}({}, {}, {})",
        imm,
        a.render(),
        b.render(),
        c.render()
    )))
}

fn prmt_selector_imm(expr: &IRExpr) -> Option<u16> {
    match expr {
        IRExpr::ImmI(value) if *value >= 0 && *value <= u16::MAX as i64 => Some(*value as u16),
        _ => None,
    }
}

fn lift_prmt_single_source(selector: u16, src: LiftedExpr) -> Option<LiftedExpr> {
    let nibbles = [
        (selector & 0xF) as u8,
        ((selector >> 4) & 0xF) as u8,
        ((selector >> 8) & 0xF) as u8,
        ((selector >> 12) & 0xF) as u8,
    ];

    if let Some(byte_idx) = match_zero_extend_byte_selector(nibbles) {
        return Some(mask_bits(shift_right(src, byte_idx * 8), 0xff));
    }
    if let Some(byte_idx) = match_sign_extend_byte_selector(nibbles) {
        return Some(sign_extend_byte(src, byte_idx * 8));
    }
    if let Some(byte_idx) = match_zero_extend_half_selector(nibbles) {
        return Some(mask_bits(shift_right(src, byte_idx * 8), 0xffff));
    }
    if let Some(byte_idx) = match_sign_extend_half_selector(nibbles) {
        return Some(sign_extend_half(src, byte_idx * 8));
    }
    if let Some(rotate_bytes) = match_byte_rotate_selector(nibbles) {
        let shift = rotate_bytes * 8;
        if shift == 0 {
            return Some(src);
        }
        return Some(LiftedExpr::Binary {
            op: "|".to_string(),
            lhs: Box::new(shift_right(src.clone(), shift)),
            rhs: Box::new(shift_left(src, 32 - shift)),
        });
    }
    None
}

fn match_zero_extend_byte_selector(nibbles: [u8; 4]) -> Option<u32> {
    if nibbles[0] < 4 && nibbles[1..].iter().all(|&n| n == 7) {
        return Some(nibbles[0] as u32);
    }
    None
}

fn match_sign_extend_byte_selector(nibbles: [u8; 4]) -> Option<u32> {
    let byte = nibbles[0];
    if byte < 4 && nibbles[1..].iter().all(|&n| n == (8 + byte)) {
        return Some(byte as u32);
    }
    None
}

fn match_zero_extend_half_selector(nibbles: [u8; 4]) -> Option<u32> {
    let lo = nibbles[0];
    let hi = nibbles[1];
    if lo < 4 && hi == lo + 1 && nibbles[2] == 7 && nibbles[3] == 7 {
        return Some(lo as u32);
    }
    None
}

fn match_sign_extend_half_selector(nibbles: [u8; 4]) -> Option<u32> {
    let lo = nibbles[0];
    let hi = nibbles[1];
    if lo < 4 && hi == lo + 1 && nibbles[2] == 8 + hi && nibbles[3] == 8 + hi {
        return Some(lo as u32);
    }
    None
}

fn lift_prmt_same_source(selector: u16, src: LiftedExpr) -> Option<LiftedExpr> {
    let normalized = [
        (selector & 0xF) as u8 % 4,
        ((selector >> 4) & 0xF) as u8 % 4,
        ((selector >> 8) & 0xF) as u8 % 4,
        ((selector >> 12) & 0xF) as u8 % 4,
    ];
    if let Some(expr) = lift_prmt_single_source(
        (normalized[3] as u16) << 12
            | (normalized[2] as u16) << 8
            | (normalized[1] as u16) << 4
            | normalized[0] as u16,
        src.clone(),
    ) {
        return Some(expr);
    }
    if normalized[0] + 1 == normalized[1]
        && normalized[0] == normalized[2]
        && normalized[1] == normalized[3]
        && normalized[0] % 2 == 0
    {
        let half = mask_bits(shift_right(src, normalized[0] as u32 * 8), 0xffff);
        return Some(LiftedExpr::Binary {
            op: "|".to_string(),
            lhs: Box::new(half.clone()),
            rhs: Box::new(shift_left(half, 16)),
        });
    }
    None
}

fn lift_prmt_zero_src0(selector: u16, src1: LiftedExpr) -> Option<LiftedExpr> {
    let nibbles = [
        (selector & 0xF) as u8,
        ((selector >> 4) & 0xF) as u8,
        ((selector >> 8) & 0xF) as u8,
        ((selector >> 12) & 0xF) as u8,
    ];
    if nibbles == [0, 1, 6, 7] {
        return Some(mask_bits(src1, 0xffff0000));
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrmtSource {
    Src0,
    Src1,
}

fn decode_prmt_nibbles(selector: u16) -> [u8; 4] {
    [
        (selector & 0xF) as u8,
        ((selector >> 4) & 0xF) as u8,
        ((selector >> 8) & 0xF) as u8,
        ((selector >> 12) & 0xF) as u8,
    ]
}

fn decode_prmt_lane(nibble: u8) -> Option<(PrmtSource, u32)> {
    match nibble {
        0..=3 => Some((PrmtSource::Src0, nibble as u32)),
        4..=7 => Some((PrmtSource::Src1, (nibble - 4) as u32)),
        _ => None,
    }
}

fn lift_prmt_byte_blend(selector: u16, src0: LiftedExpr, src1: LiftedExpr) -> Option<LiftedExpr> {
    let nibbles = decode_prmt_nibbles(selector);
    let mut pieces = Vec::new();
    let mut lane = 0usize;
    while lane < nibbles.len() {
        let (source, src_byte) = decode_prmt_lane(nibbles[lane])?;
        let mut run_end = lane + 1;
        while run_end < nibbles.len() {
            let Some((next_source, next_src_byte)) = decode_prmt_lane(nibbles[run_end]) else {
                return None;
            };
            if next_source != source || next_src_byte != src_byte + (run_end - lane) as u32 {
                break;
            }
            run_end += 1;
        }
        pieces.push(build_prmt_blend_chunk(
            source,
            src_byte,
            lane as u32,
            (run_end - lane) as u32,
            &src0,
            &src1,
        )?);
        lane = run_end;
    }

    if pieces.len() > 2 {
        return None;
    }

    let mut combined = None;
    for piece in pieces {
        if is_zero_lifted_expr(&piece) {
            continue;
        }
        combined = Some(match combined {
            None => piece,
            Some(prev) => bit_or_expr(prev, piece),
        });
    }
    Some(combined.unwrap_or_else(zero_expr))
}

fn build_prmt_blend_chunk(
    source: PrmtSource,
    src_byte: u32,
    dst_byte: u32,
    len: u32,
    src0: &LiftedExpr,
    src1: &LiftedExpr,
) -> Option<LiftedExpr> {
    let source_expr = match source {
        PrmtSource::Src0 => src0.clone(),
        PrmtSource::Src1 => src1.clone(),
    };
    if is_zero_lifted_expr(&source_expr) {
        return Some(zero_expr());
    }

    let source_mask = prmt_byte_mask(src_byte, len)?;
    let masked = mask_or_const_expr(source_expr, source_mask);
    let shift = dst_byte as i32 - src_byte as i32;
    Some(shift_or_const_expr(masked, shift * 8))
}

fn prmt_byte_mask(start_byte: u32, len: u32) -> Option<u32> {
    if len == 0 || start_byte >= 4 || start_byte + len > 4 {
        return None;
    }
    let mut mask = 0u32;
    for byte in start_byte..start_byte + len {
        mask |= 0xffu32 << (byte * 8);
    }
    Some(mask)
}

fn zero_expr() -> LiftedExpr {
    LiftedExpr::Imm("0".to_string())
}

fn is_zero_lifted_expr(expr: &LiftedExpr) -> bool {
    matches!(expr_u32_value(expr), Some(0))
}

fn expr_u32_value(expr: &LiftedExpr) -> Option<u32> {
    match expr {
        LiftedExpr::Imm(text) => parse_u32_literal(text),
        LiftedExpr::Cast { expr, .. } => expr_u32_value(expr),
        _ => None,
    }
}

fn parse_u32_literal(text: &str) -> Option<u32> {
    let trimmed = text.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        return u32::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = trimmed
        .strip_prefix("-0x")
        .or_else(|| trimmed.strip_prefix("-0X"))
    {
        let magnitude = u32::from_str_radix(hex, 16).ok()?;
        return Some((0u32).wrapping_sub(magnitude));
    }
    if let Ok(value) = trimmed.parse::<u64>() {
        return (value <= u32::MAX as u64).then_some(value as u32);
    }
    let value = trimmed.parse::<i64>().ok()?;
    Some(value as u32)
}

fn mask_or_const_expr(expr: LiftedExpr, mask: u32) -> LiftedExpr {
    if mask == u32::MAX {
        return expr;
    }
    if let Some(value) = expr_u32_value(&expr) {
        return LiftedExpr::Imm(format!("{}", value & mask));
    }
    mask_bits(expr, mask)
}

fn shift_or_const_expr(expr: LiftedExpr, bits: i32) -> LiftedExpr {
    if bits == 0 {
        return expr;
    }
    if let Some(value) = expr_u32_value(&expr) {
        let shifted = if bits > 0 {
            value.wrapping_shl(bits as u32)
        } else {
            value.wrapping_shr((-bits) as u32)
        };
        return LiftedExpr::Imm(format!("{}", shifted));
    }
    if bits > 0 {
        shift_left(expr, bits as u32)
    } else {
        shift_right(expr, (-bits) as u32)
    }
}

fn bit_or_expr(lhs: LiftedExpr, rhs: LiftedExpr) -> LiftedExpr {
    if is_zero_lifted_expr(&lhs) {
        return rhs;
    }
    if is_zero_lifted_expr(&rhs) {
        return lhs;
    }
    if let (Some(lhs), Some(rhs)) = (expr_u32_value(&lhs), expr_u32_value(&rhs)) {
        return LiftedExpr::Imm(format!("{}", lhs | rhs));
    }
    LiftedExpr::Binary {
        op: "|".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn match_byte_rotate_selector(nibbles: [u8; 4]) -> Option<u32> {
    for rotate in 0..4u8 {
        let want = [rotate, (rotate + 1) % 4, (rotate + 2) % 4, (rotate + 3) % 4];
        if nibbles == want {
            return Some(rotate as u32);
        }
    }
    None
}

fn shift_right(expr: LiftedExpr, bits: u32) -> LiftedExpr {
    if bits == 0 {
        return expr;
    }
    LiftedExpr::Binary {
        op: ">>".to_string(),
        lhs: Box::new(expr),
        rhs: Box::new(LiftedExpr::Imm(bits.to_string())),
    }
}

fn shift_left(expr: LiftedExpr, bits: u32) -> LiftedExpr {
    if bits == 0 {
        return expr;
    }
    LiftedExpr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(expr),
        rhs: Box::new(LiftedExpr::Imm(bits.to_string())),
    }
}

fn mask_bits(expr: LiftedExpr, mask: u32) -> LiftedExpr {
    LiftedExpr::Binary {
        op: "&".to_string(),
        lhs: Box::new(expr),
        rhs: Box::new(LiftedExpr::Imm(mask.to_string())),
    }
}

fn sign_extend_byte(expr: LiftedExpr, shift_bits: u32) -> LiftedExpr {
    LiftedExpr::Cast {
        ty: "int32_t".to_string(),
        expr: Box::new(LiftedExpr::Cast {
            ty: "int8_t".to_string(),
            expr: Box::new(mask_bits(shift_right(expr, shift_bits), 0xff)),
        }),
    }
}

fn sign_extend_half(expr: LiftedExpr, shift_bits: u32) -> LiftedExpr {
    LiftedExpr::Cast {
        ty: "int32_t".to_string(),
        expr: Box::new(LiftedExpr::Cast {
            ty: "int16_t".to_string(),
            expr: Box::new(mask_bits(shift_right(expr, shift_bits), 0xffff)),
        }),
    }
}

fn lift_sel(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    // SEL and FSEL share identical semantics: pred ? src0 : src1
    lift_fsel(args, stmt_ref, config)
}

/// PRMT: byte permute instruction.
/// PRMT dst, src0, selector, src1 → prmt(src0, selector, src1)
/// Renders as a readable intrinsic call rather than a raw opcode.
fn lift_prmt(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() < 2 {
        return None;
    }
    let src0 = lift_ir_expr(&args[0], stmt_ref, config);
    let selector = prmt_selector_imm(&args[1]);
    let src1_zero = args.get(2).is_none_or(is_zero_expr);
    if let (Some(selector), true) = (selector, src1_zero) {
        if let Some(expr) = lift_prmt_single_source(selector, src0.clone()) {
            return Some(expr);
        }
    }
    if let (Some(selector), Some(src1_ir)) = (selector, args.get(2)) {
        let src1 = lift_ir_expr(src1_ir, stmt_ref, config);
        if &args[0] == src1_ir {
            if let Some(expr) = lift_prmt_same_source(selector, src0.clone()) {
                return Some(expr);
            }
        }
        if is_zero_expr(&args[0]) {
            if let Some(expr) = lift_prmt_zero_src0(selector, src1.clone()) {
                return Some(expr);
            }
        }
        if let Some(expr) = lift_prmt_byte_blend(selector, src0.clone(), src1) {
            return Some(expr);
        }
    }
    let rendered: Vec<String> = args
        .iter()
        .map(|arg| lift_ir_expr(arg, stmt_ref, config).render())
        .collect();
    Some(LiftedExpr::Raw(format!("prmt({})", rendered.join(", "))))
}

fn integer_minmax_expr(
    sig: &op_sig::OpSig,
    lhs: LiftedExpr,
    rhs: LiftedExpr,
    pred: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> LiftedExpr {
    let base = if is_true_pred_expr(pred) {
        min_expr(lhs, rhs)
    } else if is_negated_true_pred(pred) {
        max_expr(lhs, rhs)
    } else {
        let cond = lift_ir_expr(pred, stmt_ref, config);
        LiftedExpr::Ternary {
            cond: Box::new(cond),
            then_expr: Box::new(min_expr(lhs.clone(), rhs.clone())),
            else_expr: Box::new(max_expr(lhs, rhs)),
        }
    };
    if sig.has_mod("RELU") {
        max_expr(base, LiftedExpr::Imm("0".to_string()))
    } else {
        base
    }
}

fn vimnmx_operands(args: &[IRExpr]) -> Option<(&IRExpr, &IRExpr, &IRExpr)> {
    if args.len() < 3 {
        return None;
    }
    let pred = args.last()?;
    let rhs = args.get(args.len() - 2)?;
    let lhs = args.get(args.len() - 3)?;
    Some((lhs, rhs, pred))
}

/// VIMNMX / IMNMX: 32-bit integer min/max family.
fn lift_vimnmx(
    sig: &op_sig::OpSig,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (lhs, rhs, pred) = vimnmx_operands(args)?;
    Some(integer_minmax_expr(
        sig,
        lift_ir_expr(lhs, stmt_ref, config),
        lift_ir_expr(rhs, stmt_ref, config),
        pred,
        stmt_ref,
        config,
    ))
}

/// VIADDMNMX: integer add-then-min/max (video instruction set).
/// Computes: Psel ? min(src0, src1 + src2) : max(src0, src1 + src2)
fn lift_viaddmnmx(
    sig: &op_sig::OpSig,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() < 4 {
        return None;
    }
    let src0 = lift_ir_expr(&args[0], stmt_ref, config);
    let sum = add_like_expr(
        lift_ir_expr(&args[1], stmt_ref, config),
        lift_ir_expr(&args[2], stmt_ref, config),
    );
    Some(integer_minmax_expr(
        sig, src0, sum, &args[3], stmt_ref, config,
    ))
}

fn lift_shf(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    // Conservative: only fold common right-shift forms used for sign/high extraction:
    // SHF*.R.*.HI dst, RZ/URZ, imm, src  ->  src >> imm
    let is_right = opcode.split('.').any(|t| t == "R");
    let is_hi = opcode.split('.').any(|t| t == "HI");
    if !is_right || !is_hi {
        return None;
    }
    if !is_zero_expr(&args[0]) {
        return None;
    }
    // Accept both immediate and register shift amounts.
    let lhs_expr = lift_ir_expr(&args[2], stmt_ref, config);
    let rhs_expr = lift_ir_expr(&args[1], stmt_ref, config);
    let signed = opcode.split('.').any(|t| t == "S32");
    if signed {
        // Signed shift: cast to int32_t first, then shift.
        // The cast is a unary prefix, composed properly with the >> binary.
        return Some(LiftedExpr::Binary {
            op: ">>".to_string(),
            lhs: Box::new(LiftedExpr::Raw(format!("(int32_t){}", lhs_expr.render()))),
            rhs: Box::new(rhs_expr),
        });
    }
    Some(LiftedExpr::Binary {
        op: ">>".to_string(),
        lhs: Box::new(lhs_expr),
        rhs: Box::new(rhs_expr),
    })
}

fn lift_shfl(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    let src = lift_ir_expr(&args[0], stmt_ref, config);
    let lane = lift_ir_expr(&args[1], stmt_ref, config);
    let clamp = lift_ir_expr(&args[2], stmt_ref, config);

    if opcode.starts_with("SHFL.DOWN") {
        return Some(call_expr(
            "__shfl_down_sync",
            vec![LiftedExpr::Imm("0xffffffff".to_string()), src, lane],
        ));
    }
    if opcode.starts_with("SHFL.UP") {
        return Some(call_expr(
            "__shfl_up_sync",
            vec![LiftedExpr::Imm("0xffffffff".to_string()), src, lane],
        ));
    }
    if opcode.starts_with("SHFL.BFLY") || opcode.starts_with("SHFL.XOR") {
        return Some(call_expr(
            "__shfl_xor_sync",
            vec![LiftedExpr::Imm("0xffffffff".to_string()), src, lane],
        ));
    }

    Some(call_expr(
        "__shfl_sync",
        vec![LiftedExpr::Imm("0xffffffff".to_string()), src, lane, clamp],
    ))
}

fn lift_lea_scaled_offset(
    offset: &IRExpr,
    shift: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if is_imm_i(shift, 0) {
        return Some(lift_ir_expr(offset, stmt_ref, config));
    }
    if let Some(bits) = imm_as_u32(shift) {
        if (1..=4).contains(&bits) {
            return Some(LiftedExpr::Binary {
                op: "*".to_string(),
                lhs: Box::new(lift_ir_expr(offset, stmt_ref, config)),
                rhs: Box::new(LiftedExpr::Imm((1u64 << bits).to_string())),
            });
        }
    }
    Some(LiftedExpr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(lift_ir_expr(offset, stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(shift, stmt_ref, config)),
    })
}

fn lift_lea_carry(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (offset, base, shift) = extract_triplet_operands(args)?;
    if !matches!(shift, IRExpr::ImmI(_)) {
        return None;
    }
    let scaled = lift_lea_scaled_offset(offset, shift, stmt_ref, config)?;
    Some(intrinsic_expr(
        IntrinsicOp::CarryU32Add3,
        vec![
            scaled,
            lift_ir_expr(base, stmt_ref, config),
            LiftedExpr::Imm("0".to_string()),
        ],
    ))
}

fn lift_lea(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if opcode.starts_with("LEA.HI") || opcode.starts_with("ULEA.HI") {
        return lift_lea_hi(opcode, args, stmt_ref, config);
    }
    if opcode != "LEA" && opcode != "ULEA" {
        return None;
    }
    // LEA / ULEA consume (scaled-index, base, shift) after IR strips any
    // carry-out predicate def. Keep the scaled offset on the lhs so addr64
    // recovery can match `offset + ptr_lo` structurally.
    let (offset, base, shift) = extract_triplet_operands(args)?;
    if !matches!(shift, IRExpr::ImmI(_)) {
        return None;
    }
    Some(add_like_expr(
        lift_lea_scaled_offset(offset, shift, stmt_ref, config)?,
        lift_ir_expr(base, stmt_ref, config),
    ))
}

fn lift_lea_hi(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    // 4-arg common form: LEA.HI dst, base, off, sh
    if args.len() == 4 {
        // Conservative render: hi32(base + (off << sh))
        if matches!(args[3], IRExpr::ImmI(_)) {
            if config.strict && !is_zero_expr(&args[2]) {
                return None;
            }
            let shifted = LiftedExpr::Binary {
                op: "<<".to_string(),
                lhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
                rhs: Box::new(lift_ir_expr(&args[3], stmt_ref, config)),
            };
            let mut inner = add_like_expr(lift_ir_expr(&args[0], stmt_ref, config), shifted);
            if !is_zero_expr(&args[2]) {
                inner = add_like_expr(inner, lift_ir_expr(&args[2], stmt_ref, config));
            }
            return Some(hi32_expr(inner));
        }

        // Carry form: LEA.HI.X.* dst, base, off, sh, Pcarry
        // IR keeps 4 args (base, off, sh, Pcarry).
        if matches!(args[2], IRExpr::ImmI(_)) && is_pred_reg_expr(&args[3]) {
            return Some(intrinsic_expr(
                if opcode.contains(".SX32") {
                    IntrinsicOp::LeaHiXSx32
                } else {
                    IntrinsicOp::LeaHiX
                },
                vec![
                    lift_ir_expr(&args[0], stmt_ref, config),
                    lift_ir_expr(&args[1], stmt_ref, config),
                    lift_ir_expr(&args[2], stmt_ref, config),
                    lift_ir_expr(&args[3], stmt_ref, config),
                ],
            ));
        }
        return None;
    }

    // 5-arg form: LEA.HI.X dst, offset, hi_base, accum, shift, carry
    // IR gets 5 args: (offset, hi_base, accum, shift_imm, carry_pred).
    // Semantics: hi32(offset << shift) + hi_base + carry
    // (accum is the prior hi-word value in multi-precision chains)
    if args.len() == 5 && matches!(args[3], IRExpr::ImmI(_)) && is_pred_reg_expr(&args[4]) {
        return Some(intrinsic_expr(
            if opcode.contains(".SX32") {
                IntrinsicOp::LeaHiXSx32
            } else {
                IntrinsicOp::LeaHiX
            },
            vec![
                lift_ir_expr(&args[0], stmt_ref, config),
                lift_ir_expr(&args[1], stmt_ref, config),
                lift_ir_expr(&args[3], stmt_ref, config),
                lift_ir_expr(&args[4], stmt_ref, config),
            ],
        ));
    }

    None
}

fn cmp_token_to_op(opcode: &str) -> Option<&'static str> {
    for tok in opcode.split('.') {
        let op = match tok {
            "EQ" | "EQU" => "==",
            "NE" | "NEU" => "!=",
            "LT" | "LTU" => "<",
            "LE" | "LEU" => "<=",
            "GT" | "GTU" => ">",
            "GE" | "GEU" => ">=",
            _ => continue,
        };
        return Some(op);
    }
    None
}

/// Returns `true` when the ISETP opcode uses signed comparison.
/// ISETP without `.U32` is signed; ISETP with `.U32` is unsigned.
/// FSETP is always floating-point (signedness N/A), so returns false.
fn is_signed_int_compare(opcode: &str) -> bool {
    let parts: Vec<&str> = opcode.split('.').collect();
    if parts.first().map_or(true, |m| *m != "ISETP") {
        return false;
    }
    !parts.iter().any(|p| *p == "U32")
}

/// Returns `true` for relational operators where signedness matters.
fn is_ordered_cmp(op: &str) -> bool {
    matches!(op, "<" | "<=" | ">" | ">=")
}

/// Wrap a lifted expression in an `(int32_t)` cast.
fn signed_cast(expr: LiftedExpr) -> LiftedExpr {
    LiftedExpr::Raw(format!("(int32_t)({})", expr.render()))
}

fn add_like_expr(lhs: LiftedExpr, rhs: LiftedExpr) -> LiftedExpr {
    // Fold `x + 0` → `x` and `0 + x` → `x`.
    if matches!(&rhs, LiftedExpr::Imm(s) if s == "0") {
        return lhs;
    }
    if matches!(&lhs, LiftedExpr::Imm(s) if s == "0") {
        return rhs;
    }
    if let Some((is_neg, mag)) = rhs_signed_imm(&rhs) {
        if is_neg {
            return LiftedExpr::Binary {
                op: "-".to_string(),
                lhs: Box::new(lhs),
                rhs: Box::new(LiftedExpr::Imm(mag)),
            };
        }
    }
    if let Some(mag) = rhs_negated_nonimm(&rhs) {
        return LiftedExpr::Binary {
            op: "-".to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(LiftedExpr::Raw(mag)),
        };
    }
    LiftedExpr::Binary {
        op: "+".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn rhs_signed_imm(rhs: &LiftedExpr) -> Option<(bool, String)> {
    let LiftedExpr::Imm(text) = rhs else {
        return None;
    };
    if let Some(rest) = text.strip_prefix('-') {
        return Some((true, rest.to_string()));
    }
    // Detect large unsigned values that represent negative 32-bit integers.
    // If the value fits in i64 and its low 32 bits are negative when interpreted
    // as i32, display as subtraction (e.g., 4294966784 → -512).
    if let Ok(v) = text.parse::<i64>() {
        let as_u32 = v as u32;
        let as_i32 = as_u32 as i32;
        if v == as_u32 as i64 && as_i32 < 0 && as_i32 > i32::MIN / 2 {
            // The value is the unsigned representation of a small negative number.
            let mag = (-(as_i32 as i64)).to_string();
            return Some((true, mag));
        }
    }
    Some((false, text.clone()))
}

fn rhs_negated_nonimm(rhs: &LiftedExpr) -> Option<String> {
    match rhs {
        LiftedExpr::Reg(s) | LiftedExpr::Raw(s) => {
            let rest = s.strip_prefix('-')?;
            Some(rest.to_string())
        }
        LiftedExpr::Unary { op, arg } if op == "-" => Some(arg.render()),
        _ => None,
    }
}

fn hi32_expr(inner: LiftedExpr) -> LiftedExpr {
    LiftedExpr::Raw(format!("hi32({})", inner.render()))
}

fn is_true_pred_expr(e: &IRExpr) -> bool {
    let IRExpr::Reg(r) = e else {
        return false;
    };
    matches!(r.class.as_str(), "PT" | "UPT")
}

fn is_imm_i(e: &IRExpr, v: i64) -> bool {
    matches!(e, IRExpr::ImmI(i) if *i == v)
}

fn is_false_pred_expr(e: &IRExpr) -> bool {
    matches!(e, IRExpr::Op { op, args } if args.is_empty() && matches!(op.as_str(), "!PT" | "!UPT"))
}

/// Check if expression is a negated true predicate (!PT / !UPT).
fn is_negated_true_pred(e: &IRExpr) -> bool {
    is_false_pred_expr(e)
        || matches!(e, IRExpr::Op { op, args } if op == "!" && args.len() == 1 && is_true_pred_expr(&args[0]))
}

fn carry_inc_expr(
    e: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if is_true_pred_expr(e) {
        return Some(LiftedExpr::Imm("1".to_string()));
    }
    if is_false_pred_expr(e) {
        return Some(LiftedExpr::Imm("0".to_string()));
    }
    if !is_pred_reg_expr(e) {
        return None;
    }
    let cond = lift_ir_expr(e, stmt_ref, config);
    Some(LiftedExpr::Ternary {
        cond: Box::new(cond),
        then_expr: Box::new(LiftedExpr::Imm("1".to_string())),
        else_expr: Box::new(LiftedExpr::Imm("0".to_string())),
    })
}

fn extract_triplet_operands(args: &[IRExpr]) -> Option<(&IRExpr, &IRExpr, &IRExpr)> {
    if args.len() < 3 {
        return None;
    }
    let mut start = 0usize;
    let mut end = args.len();

    // Strip leading predicate output operands (e.g., IADD3(..., P0, a, b, c)).
    while end.saturating_sub(start) > 3 && is_pred_reg_expr(&args[start]) {
        start += 1;
    }
    // Strip trailing predicate control operands (e.g., ..., !PT()).
    while end.saturating_sub(start) > 3 && is_pred_control_expr(&args[end - 1]) {
        end -= 1;
    }
    if end.saturating_sub(start) == 3 {
        Some((&args[start], &args[start + 1], &args[start + 2]))
    } else {
        None
    }
}

fn extract_addx_operands(args: &[IRExpr]) -> Option<(&IRExpr, &IRExpr, &IRExpr, &IRExpr)> {
    if args.len() < 4 {
        return None;
    }
    let mut start = 0usize;
    let mut end = args.len();

    while end.saturating_sub(start) > 5 && is_pred_reg_expr(&args[start]) {
        start += 1;
    }
    while end.saturating_sub(start) > 4 && is_pred_control_expr(&args[end - 1]) {
        end -= 1;
    }
    if end.saturating_sub(start) == 4 {
        Some((
            &args[start],
            &args[start + 1],
            &args[start + 2],
            &args[start + 3],
        ))
    } else {
        None
    }
}

fn is_zero_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 0,
        IRExpr::ImmF(f) => *f == 0.0,
        IRExpr::Reg(r) => matches!(r.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn is_pred_reg_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::Reg(r) => matches!(r.class.as_str(), "P" | "UP" | "PT" | "UPT"),
        IRExpr::Op { op, args } if args.is_empty() => {
            let core = op.strip_prefix('!').unwrap_or(op.as_str());
            if core == "PT" || core == "UPT" {
                return true;
            }
            let core_no_ssa = core.split('.').next().unwrap_or(core);
            if let Some(num) = core_no_ssa.strip_prefix('P') {
                return num.parse::<u32>().is_ok();
            }
            if let Some(num) = core_no_ssa.strip_prefix("UP") {
                return num.parse::<u32>().is_ok();
            }
            false
        }
        _ => false,
    }
}

fn is_pred_control_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::Reg(r) => matches!(r.class.as_str(), "PT" | "UPT"),
        IRExpr::Op { op, args } if args.is_empty() => {
            matches!(op.as_str(), "PT" | "!PT" | "UPT" | "!UPT")
        }
        _ => false,
    }
}

fn is_mem_expr(e: &IRExpr) -> bool {
    matches!(e, IRExpr::Mem { .. })
}

fn lift_float_ir_expr(
    expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> LiftedExpr {
    if let IRExpr::ImmI(bits_i64) = expr {
        if looks_like_f32_bitpattern(*bits_i64) {
            return LiftedExpr::Imm(format_f32_bitpattern(*bits_i64 as u32));
        }
    }
    lift_ir_expr(expr, stmt_ref, config)
}

fn lift_ir_expr(
    expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> LiftedExpr {
    match expr {
        IRExpr::Reg(r) => {
            // Render RZ/URZ as literal 0 instead of a register name.
            // These are hardware-zero registers and showing them as "RZ"
            // confuses downstream name-recovery into inventing live-in vars.
            if matches!(r.class.as_str(), "RZ" | "URZ") {
                return LiftedExpr::Imm("0".to_string());
            }
            // PT/UPT are always-true predicates.
            if matches!(r.class.as_str(), "PT" | "UPT") {
                return LiftedExpr::Imm("true".to_string());
            }
            LiftedExpr::Reg(r.display())
        }
        IRExpr::ImmI(i) => LiftedExpr::Imm(i.to_string()),
        IRExpr::ImmF(f) => LiftedExpr::Imm(f.to_string()),
        IRExpr::Mem { .. } => LiftedExpr::Raw(render_expr_raw(expr, stmt_ref, config)),
        IRExpr::Addr64 { lo, hi } => LiftedExpr::Addr64 {
            lo: Box::new(lift_ir_expr(lo.as_ref(), stmt_ref, config)),
            hi: Box::new(lift_ir_expr(hi.as_ref(), stmt_ref, config)),
        },
        IRExpr::Op { op, args } => {
            if op == "!" && args.len() == 1 {
                let child = lift_ir_expr(&args[0], stmt_ref, config);
                return simplify_not(child);
            }
            if args.len() == 2 {
                match op.as_str() {
                    "+" => {
                        return add_like_expr(
                            lift_ir_expr(&args[0], stmt_ref, config),
                            lift_ir_expr(&args[1], stmt_ref, config),
                        );
                    }
                    "*" | "<<" | "-" => {
                        return LiftedExpr::Binary {
                            op: op.clone(),
                            lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
                            rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
                        };
                    }
                    _ => {}
                }
            }
            if args.is_empty() {
                if let Some(pred_expr) = parse_inline_predicate_expr(op) {
                    return pred_expr;
                }
            }
            if op == "ConstMem" && args.len() == 2 {
                if let (Some(bank), Some(offset)) = (imm_as_u32(&args[0]), imm_as_u32(&args[1])) {
                    if let Some(sym) = resolve_constmem_symbol(stmt_ref, bank, offset, config) {
                        return lift_named_symbol_expr(&sym);
                    }
                    // Render unresolved constant memory in hex (SASS convention).
                    return LiftedExpr::Raw(format!("c[0x{:x}][0x{:x}]", bank, offset));
                }
            }
            LiftedExpr::Raw(render_expr_raw(expr, stmt_ref, config))
        }
    }
}

fn parse_inline_predicate_expr(op: &str) -> Option<LiftedExpr> {
    let (negated, core) = if let Some(rest) = op.strip_prefix('!') {
        (true, rest)
    } else {
        (false, op)
    };

    if core == "PT" || core == "UPT" {
        return Some(LiftedExpr::Imm(
            if negated { "false" } else { "true" }.to_string(),
        ));
    }

    let pred_name = if let Some(num) = core.strip_prefix('P') {
        if num.parse::<u32>().is_ok() {
            Some(core.to_string())
        } else {
            None
        }
    } else if let Some(num) = core.strip_prefix("UP") {
        if num.parse::<u32>().is_ok() {
            Some(core.to_string())
        } else {
            None
        }
    } else {
        None
    }?;

    let base = LiftedExpr::Reg(pred_name);
    if negated {
        Some(simplify_not(base))
    } else {
        Some(base)
    }
}

fn simplify_not(expr: LiftedExpr) -> LiftedExpr {
    if let LiftedExpr::Unary { op, arg } = expr {
        if op == "!" {
            return *arg;
        }
        return LiftedExpr::Unary { op, arg };
    }
    LiftedExpr::Unary {
        op: "!".to_string(),
        arg: Box::new(expr),
    }
}

fn render_expr_raw(
    expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> String {
    match expr {
        IRExpr::Reg(r) => {
            if matches!(r.class.as_str(), "RZ" | "URZ") {
                return "0".to_string();
            }
            if matches!(r.class.as_str(), "PT" | "UPT") {
                return "true".to_string();
            }
            r.display()
        }
        IRExpr::ImmI(i) => i.to_string(),
        IRExpr::ImmF(f) => f.to_string(),
        IRExpr::Addr64 { lo, hi } => {
            format!(
                "addr64({}, {})",
                render_expr_raw(lo, stmt_ref, config),
                render_expr_raw(hi, stmt_ref, config)
            )
        }
        IRExpr::Mem {
            base,
            offset,
            width,
        } => {
            let s = if let Some(off) = offset {
                format!(
                    "*({} + {})",
                    render_expr_raw(base, stmt_ref, config),
                    render_expr_raw(off, stmt_ref, config)
                )
            } else {
                format!("*{}", render_expr_raw(base, stmt_ref, config))
            };
            let _ = width;
            s
        }
        IRExpr::Op { op, args } => {
            if op == "-" && args.len() == 1 {
                let inner = render_expr_raw(&args[0], stmt_ref, config);
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
            if args.len() == 2 && matches!(op.as_str(), "+" | "-" | "*" | "<<") {
                return format!(
                    "{} {} {}",
                    render_expr_raw(&args[0], stmt_ref, config),
                    op,
                    render_expr_raw(&args[1], stmt_ref, config)
                );
            }
            if args.is_empty() {
                match op.as_str() {
                    "PT" | "UPT" => return "true".to_string(),
                    "!PT" | "!UPT" => return "false".to_string(),
                    _ => {}
                }
            }
            if op == "ConstMem" && args.len() == 2 {
                if let (Some(bank), Some(offset)) = (imm_as_u32(&args[0]), imm_as_u32(&args[1])) {
                    if let Some(sym) = resolve_constmem_symbol(stmt_ref, bank, offset, config) {
                        return sym;
                    }
                }
            }
            let list = args
                .iter()
                .map(|a| render_expr_raw(a, stmt_ref, config))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", op, list)
        }
    }
}

fn imm_as_u32(e: &IRExpr) -> Option<u32> {
    match e {
        IRExpr::ImmI(i) if *i >= 0 => u32::try_from(*i).ok(),
        _ => None,
    }
}

fn resolve_constmem_symbol(
    stmt_ref: StatementRef,
    bank: u32,
    offset: u32,
    config: &SemanticLiftConfig<'_>,
) -> Option<String> {
    let anns = config.abi_annotations?;

    let exact = anns
        .constmem_by_stmt
        .get(&stmt_ref)
        .and_then(|entries| {
            entries
                .iter()
                .find(|ann| ann.bank == bank && ann.offset == offset)
        })
        .or_else(|| {
            anns.constmem_by_stmt.values().find_map(|entries| {
                entries
                    .iter()
                    .find(|ann| ann.bank == bank && ann.offset == offset)
            })
        });

    if let Some(ann) = exact {
        if let ConstMemSemantic::ParamWord {
            param_idx,
            word_idx,
        } = ann.semantic
        {
            if let Some(aliases) = config.abi_aliases {
                if let Some(alias) = aliases.render_param_word(param_idx, word_idx) {
                    return Some(alias);
                }
            }
        }
        return Some(ann.symbol());
    }

    let inferred_param = anns.constmem_by_stmt.values().find_map(|entries| {
        let mut inferred = None;
        for ann in entries {
            let ConstMemSemantic::ParamWord { param_idx, .. } = ann.semantic else {
                continue;
            };
            if ann.bank != bank {
                continue;
            }
            let delta = offset as i64 - ann.offset as i64;
            if delta % 4 != 0 {
                continue;
            }
            let candidate = param_idx as i64 + delta / 4;
            if candidate < 0 {
                continue;
            }
            let candidate = candidate as u32;
            match inferred {
                None => inferred = Some(candidate),
                Some(existing) if existing == candidate => {}
                Some(_) => return None,
            }
        }
        inferred
    });

    if let Some(param_idx) = inferred_param {
        if let Some(aliases) = config.abi_aliases {
            if let Some(alias) = aliases.render_param_word(param_idx, 0) {
                return Some(alias);
            }
        }
        return Some(format!("param_{}", param_idx));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_constprop::ir_constprop;
    use crate::{build_cfg, build_ssa, decode_sass};

    fn run_lift(sass: &str) -> SemanticLiftResult {
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        lift_function_ir(&fir, &SemanticLiftConfig::default())
    }

    /// Like `run_lift`, but builds ABI annotations under the requested
    /// profile and threads them through `SemanticLiftConfig`.  Used by
    /// the wide-load tests that need to exercise the
    /// `lift_uldc_wide_def_from_lo` annotation path (which exits early
    /// when `config.abi_annotations` is `None`).
    fn run_lift_with_abi(sass: &str, profile: crate::abi::AbiProfile) -> SemanticLiftResult {
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let anns = crate::abi::annotate_function_ir_constmem(&fir, profile);
        let config = SemanticLiftConfig {
            abi_annotations: Some(&anns),
            abi_aliases: None,
            strict: true,
        };
        lift_function_ir(&fir, &config)
    }

    #[test]
    fn lifts_imad_mov_to_direct_rhs() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x2a ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD.MOV");
        assert_eq!(lifted.rhs.render(), "42");
    }

    #[test]
    fn lifts_constprop_float_bitpattern_literals_for_float_defs() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x3f800000 ;
            /*0010*/ FADD R2, R1, 0f00000000 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = ir_constprop(&build_ssa(&cfg));
        let out = lift_function_ir(&fir, &SemanticLiftConfig::default());
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted constprop literal");
        assert_eq!(lifted.rhs.render(), "1");
    }

    #[test]
    fn lifts_iadd3_with_zero_elision() {
        let sass = r#"
            /*0000*/ IADD3 R1, R1, -0x1, RZ ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IADD3");
        let rendered = lifted.rhs.render();
        assert!(rendered.contains("- 1"));
    }

    #[test]
    fn lifts_uiadd3_with_pred_output_form() {
        let sass = r#"
            /*0000*/ UIADD3 UR8, UP0, UR8, 0x4, URZ ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted UIADD3");
        assert_eq!(lifted.rhs.render(), "UR8.0 + 4");
    }

    #[test]
    fn paired_hi_of_zero_base_is_zero() {
        let expr = lift_paired_hi_from_lo(
            &IRExpr::ImmI(0),
            StatementRef {
                block_id: 0,
                stmt_idx: 0,
            },
            &SemanticLiftConfig::default(),
            &BTreeMap::new(),
        );
        assert_eq!(expr.render(), "0");
    }

    #[test]
    fn pair_hi_map_ignores_predicate_defs_from_nonwide_iadd3() {
        let sass = r#"
            /*0000*/ IADD3 R4, P0, P1, R2, 0x1, RZ ;
            /*0010*/ MOV R8, R4 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let pair_hi = build_pair_hi_map(&fir);
        assert!(
            !pair_hi.contains_key(&RegId::new("R", 4, 1)),
            "non-wide IADD3 predicate defs must not seed pair_hi_map: {:?}",
            pair_hi
        );
    }

    #[test]
    fn pair_hi_map_tracks_uiadd3_64_implicit_high_lane() {
        let sass = r#"
            /*0000*/ UIADD3.64 UR4, UPT, UPT, UR8, 0x4, URZ ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let pair_hi = build_pair_hi_map(&fir);
        let hi = pair_hi
            .iter()
            .find_map(|(lo, hi)| (lo.class == "UR" && lo.idx == 4).then_some(hi))
            .and_then(IRExpr::get_reg)
            .expect("UIADD3.64 low def should map to the implicit hi lane");
        assert_eq!(hi.class, "UR");
        assert_eq!(hi.idx, 5);
    }

    #[test]
    fn lifts_imad_iadd_with_mul_by_one() {
        let sass = r#"
            /*0000*/ IMAD.IADD R11, R12, 0x1, R11 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD.IADD");
        assert_eq!(lifted.rhs.render(), "R12.0 + R11.0");
    }

    #[test]
    fn renders_pt_constants_as_bools() {
        let sass = r#"
            /*0000*/ IMAD.IADD R11, R12, 0x1, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD.IADD");
        assert_eq!(lifted.rhs.render(), "R12.0 + false");
    }

    #[test]
    fn lifts_setp_compare_to_infix() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted ISETP");
        // ISETP.GE (no .U32) is signed → operands wrapped in (int32_t).
        assert_eq!(lifted.rhs.render(), "(int32_t)(R0.0) >= (int32_t)(1)");
    }

    #[test]
    fn lifts_s2r_ctaid_x_to_blockidx_x() {
        let sass = r#"
            /*0000*/ S2R R0, SR_CTAID.X ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted S2R");
        assert_eq!(lifted.rhs.render(), "blockIdx.x");
    }

    #[test]
    fn lifts_fsel_to_ternary() {
        let sass = r#"
            /*0000*/ FSEL R5, R7, 0.89999997615814208984, P1 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FSEL");
        let rendered = lifted.rhs.render();
        assert!(rendered.starts_with("P1.0 ? R7.0 : "));
    }

    #[test]
    fn lifts_sel_to_ternary() {
        let sass = r#"
            /*0000*/ SEL R6, R5, R6, !P1 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted SEL");
        assert_eq!(lifted.rhs.render(), "!P1.0 ? R5.0 : R6.0");
    }

    #[test]
    fn lifts_shf_hi_right_pattern_to_shift() {
        let sass = r#"
            /*0000*/ SHF.R.S32.HI R3, RZ, 0x1f, R0 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted SHF");
        assert_eq!(lifted.rhs.render(), "(int32_t)R0.0 >> 31");
    }

    #[test]
    fn lifts_shfl_down_to_cuda_shuffle_intrinsic() {
        let sass = r#"
            /*0000*/ SHFL.DOWN PT, R0, R3, 0x10, 0x1f ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted SHFL.DOWN");
        assert_eq!(lifted.rhs.render(), "__shfl_down_sync(0xffffffff, R3.0, 16)");
    }

    #[test]
    fn predicated_load_with_undefined_old_value_defaults_to_zero() {
        let sass = r#"
            /*0000*/ @!P0 LDG.E.CONSTANT R4, [R2.64] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted predicated load");
        assert_eq!(
            lifted
                .pred_old_val
                .as_ref()
                .expect("predicated load should keep a false-path value")
                .render(),
            "0"
        );
    }

    #[test]
    fn lifts_imad_wide_to_mul_add() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R2, R27, R2, c[0x0][0x168] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD.WIDE");
        assert_eq!(lifted.rhs.render(), "R27.0 * R2.0 + c[0x0][0x168]");
    }

    #[test]
    fn lifts_simple_lea_form() {
        let sass = r#"
            /*0000*/ LEA R2, P0, R0, c[0x0][0x170], 0x2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LEA");
        assert_eq!(lifted.rhs.render(), "R0.0 * 4 + c[0x0][0x170]");
    }

    #[test]
    fn lifts_lea_carry_def_from_scaled_low_sum() {
        let sass = r#"
            /*0000*/ LEA R2, P0, R0, c[0x0][0x170], 0x2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let carry = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 1,
            })
            .expect("expected lifted LEA carry def");
        assert_eq!(
            carry.rhs.render(),
            "carry_u32_add3(R0.0 * 4, c[0x0][0x170], 0)"
        );
    }

    #[test]
    fn lifts_iadd3_x_to_add_with_carry_term() {
        let sass = r#"
            /*0000*/ IADD3.X R3, R1, R2, RZ, P1, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IADD3.X");
        assert_eq!(lifted.rhs.render(), "R1.0 + R2.0 + (P1.0 ? 1 : 0)");
    }

    #[test]
    fn lifts_lea_hi_to_hi32_form() {
        let sass = r#"
            /*0000*/ LEA.HI R4, R5, R6, RZ, 0x8 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LEA.HI");
        assert_eq!(lifted.rhs.render(), "hi32(R5.0 + (R6.0 << 8))");
    }

    #[test]
    fn lifts_lea_hi_x_to_hi32_plus_carry_term() {
        let sass = r#"
            /*0000*/ LEA.HI.X.SX32 R11, R6, c[0x0][0x164], 0x1, P2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LEA.HI.X");
        assert_eq!(
            lifted.rhs.render(),
            "lea_hi_x_sx32(R6.0, c[0x0][0x164], 1, P2.0)"
        );
    }

    #[test]
    fn lifts_lop3_lut_and_with_zero_third_operand() {
        let sass = r#"
            /*0000*/ LOP3.LUT R17, R22, R17, RZ, 0xc0, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "R22.0 & R17.0");
    }

    #[test]
    fn lifts_lop3_lut_or_with_zero_third_operand() {
        let sass = r#"
            /*0000*/ LOP3.LUT R8, R8, R9, RZ, 0xfc, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "R8.0 | R9.0");
    }

    #[test]
    fn lifts_lop3_lut_not_with_two_zero_inputs() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, RZ, R6, RZ, 0x33, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "~R6.0");
    }

    #[test]
    fn lifts_lop3_lut_binary_when_first_input_zero() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, RZ, R6, R7, 0x66, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "R6.0 ^ R7.0");
    }

    #[test]
    fn lifts_lds_to_shared_indexed_read() {
        let sass = r#"
            /*0000*/ LDS.U8 R12, [UR4] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LDS");
        assert_eq!(lifted.rhs.render(), "shmem_u8[UR4.0]");
    }

    #[test]
    fn lifts_sts_to_shared_indexed_write() {
        let sass = r#"
            /*0000*/ STS.U8 [UR4], R11 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted STS");
        assert_eq!(lifted.dest.render(), "shmem_u8[UR4.0]");
        assert_eq!(lifted.rhs.render(), "R11.0");
    }

    #[test]
    fn lifts_imad_generic_to_mul_add() {
        let sass = r#"
            /*0000*/ IMAD R5, R1, R2, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD");
        assert_eq!(lifted.rhs.render(), "R1.0 * R2.0 + R3.0");
    }

    #[test]
    fn lifts_imad_hi_u32_to_helper_intrinsic() {
        let sass = r#"
            /*0000*/ IMAD.HI.U32 R12, R8, R11, R9 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD.HI.U32");
        assert_eq!(lifted.rhs.render(), "mul_hi_u32(R8.0, R11.0) + R9.0");
    }

    #[test]
    fn lifts_imad_x_zero_mul_form_to_add_carry() {
        let sass = r#"
            /*0000*/ IMAD.X R5, RZ, RZ, R3, P0 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IMAD.X");
        assert_eq!(lifted.rhs.render(), "R3.0 + (P0.0 ? 1 : 0)");
    }

    #[test]
    fn lifts_iabs_to_abs_intrinsic() {
        let sass = r#"
            /*0000*/ IABS R5, R2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IABS");
        assert_eq!(lifted.rhs.render(), "abs(R2.0)");
    }

    #[test]
    fn lifts_iabs_to_typed_calllike_expr() {
        let sass = r#"
            /*0000*/ IABS R5, R2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IABS");
        assert!(matches!(
            &lifted.rhs,
            LiftedExpr::CallLike { func, args }
                if func == "abs"
                    && matches!(args.as_slice(), [Expr::Reg(name)] if name == "R2.0")
        ));
    }

    #[test]
    fn lifts_i2f_rp_to_helper_intrinsic() {
        let sass = r#"
            /*0000*/ I2F.RP R7, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted I2F.RP");
        assert_eq!(lifted.rhs.render(), "(float)(R3.0)");
    }

    #[test]
    fn lifts_i2f_rp_to_typed_cast_expr() {
        let sass = r#"
            /*0000*/ I2F.RP R7, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted I2F.RP");
        assert!(matches!(
            &lifted.rhs,
            LiftedExpr::Cast { ty, expr }
                if ty == "float"
                    && matches!(expr.as_ref(), Expr::Reg(name) if name == "R3.0")
        ));
    }

    #[test]
    fn lifts_f2i_ftz_trunc_ntz_to_helper_intrinsic() {
        let sass = r#"
            /*0000*/ F2I.FTZ.U32.TRUNC.NTZ R8, R7 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted F2I.FTZ.U32.TRUNC.NTZ");
        assert_eq!(lifted.rhs.render(), "(uint32_t)(R7.0)");
    }

    #[test]
    fn lifts_mufu_rcp_to_helper_intrinsic() {
        let sass = r#"
            /*0000*/ MUFU.RCP R6, R5 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted MUFU.RCP");
        assert_eq!(lifted.rhs.render(), "rcp_approx(R5.0)");
    }

    #[test]
    fn lifts_uldc_64_to_intrinsic_form() {
        let sass = r#"
            /*0000*/ ULDC.64 UR8, c[0x0][0x118] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted_lo = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted ULDC.64");
        assert_eq!(lifted_lo.rhs.render(), "c[0x0][0x118]");
        let lifted_hi = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 1,
            })
            .expect("expected lifted ULDC.64 high half");
        assert_eq!(lifted_hi.rhs.render(), "c[0x0][0x11c]");
    }

    #[test]
    fn lifts_ldcu_64_like_uldc_64() {
        // LDCU is the SM 100+ rename of ULDC. Both halves of the register pair
        // must lift to the underlying ConstMem words, matching ULDC.64 output.
        let sass = r#"
            /*0000*/ LDCU.64 UR6, c[0x0][0x358] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted_lo = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LDCU.64 low half");
        assert_eq!(lifted_lo.rhs.render(), "c[0x0][0x358]");
        let lifted_hi = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 1,
            })
            .expect("expected lifted LDCU.64 high half");
        assert_eq!(lifted_hi.rhs.render(), "c[0x0][0x35c]");
    }

    #[test]
    fn lifts_ldcu_128_into_four_constmem_words() {
        // `LDCU.128 UR8, c[0x0][0x380]` (taken from
        // `test_cu/corpus_sm100/loop_kernels.sass`) defines UR8..UR11.
        // Each lane must lift to the corresponding 32-bit ConstMem word
        // so downstream uses see the right symbolic source.
        let sass = r#"
            /*0000*/ LDCU.128 UR8, c[0x0][0x380] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let expected = [
            (0, "c[0x0][0x380]"),
            (1, "c[0x0][0x384]"),
            (2, "c[0x0][0x388]"),
            (3, "c[0x0][0x38c]"),
        ];
        for (def_idx, want) in expected {
            let lifted = out
                .by_def
                .get(&DefRef {
                    block_id: 0,
                    stmt_idx: 0,
                    def_idx,
                })
                .unwrap_or_else(|| panic!("expected lifted LDCU.128 def {}", def_idx));
            assert_eq!(
                lifted.rhs.render(),
                want,
                "LDCU.128 def {} did not lift correctly",
                def_idx
            );
        }
    }

    #[test]
    fn lifts_ldcu_128_high_lanes_through_abi_annotations() {
        // The previous test only exercises the literal-offset fallback
        // (`constmem_plus_word_offset_n`) because it runs with no ABI
        // config, so `lift_uldc_wide_def_from_lo` exits at the
        // `config.abi_annotations?` early return.  This test threads a
        // real `BlackwellParam380` annotation set through and pins the
        // `param_idx + def_idx` walk for `def_idx 1..=3`, which is the
        // path the production pipeline actually takes for SM 100/120
        // dumps.  Without it a regression that broke the
        // `param_idx + def_idx` arithmetic could leave the literal
        // fallback test passing while every annotated kernel emitted
        // wrong source labels for hi lanes.
        let sass = r#"
            /*0000*/ LDCU.128 UR8, c[0x0][0x380] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift_with_abi(sass, crate::abi::AbiProfile::blackwell_param_380());
        // Lo lane goes through the normal `lift_uldc_wide` rule and
        // resolves the bank/offset annotation directly to `param_0`.
        let expected = [
            (0, "param_0"),
            (1, "param_1"),
            (2, "param_2"),
            (3, "param_3"),
        ];
        for (def_idx, want) in expected {
            let lifted = out
                .by_def
                .get(&DefRef {
                    block_id: 0,
                    stmt_idx: 0,
                    def_idx,
                })
                .unwrap_or_else(|| {
                    panic!(
                        "expected lifted LDCU.128 def {} under ABI annotations",
                        def_idx
                    )
                });
            assert_eq!(
                lifted.rhs.render(),
                want,
                "LDCU.128 def {} did not walk param_idx + def_idx via ABI annotations",
                def_idx
            );
        }
    }

    #[test]
    fn lifts_scalar_uldc_and_ldcu_to_symbolic_constmem() {
        // Scalar ULDC/LDC/LDCU (no .64/.128 suffix) loads a single 32-bit
        // register from constant memory.  All three should lift through the
        // same path as the lo lane of a wide load.
        let modern = crate::abi::AbiProfile::modern_param_160();
        let blackwell = crate::abi::AbiProfile::blackwell_param_380();
        let cases: &[(&str, &str, crate::abi::AbiProfile)] = &[
            ("ULDC UR5, c[0x0][0x0]", "blockDim.x", modern), // SM 89 builtin (uniform)
            ("LDC R5, c[0x0][0x0]", "blockDim.x", modern),   // SM 89 builtin (non-uniform)
            ("LDCU UR5, c[0x0][0x360]", "blockDim.x", blackwell), // SM 100 builtin
            ("LDC R1, c[0x0][0x360]", "blockDim.x", blackwell), // SM 100 builtin (non-uniform)
            ("LDCU UR4, c[0x0][0x390]", "param_4", blackwell), // SM 100 param
            ("LDC R4, c[0x0][0x390]", "param_4", blackwell), // SM 100 param (non-uniform)
        ];
        for &(instr, want, profile) in cases {
            let sass = format!("/*0000*/ {} ;\n/*0010*/ EXIT ;\n", instr);
            let out = run_lift_with_abi(&sass, profile);
            let lifted = out
                .by_def
                .get(&DefRef {
                    block_id: 0,
                    stmt_idx: 0,
                    def_idx: 0,
                })
                .unwrap_or_else(|| panic!("scalar {} should be lifted, not raw", instr));
            assert_eq!(
                lifted.rhs.render(),
                want,
                "scalar {}: expected {} but got {}",
                instr,
                want,
                lifted.rhs.render()
            );
        }
    }

    #[test]
    fn strict_iadd3_allows_exact_three_term_addition() {
        let sass = r#"
            /*0000*/ IADD3 R1, R2, R3, R4 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted IADD3 in strict mode");
        assert_eq!(lifted.rhs.render(), "R2.0 + R3.0 + R4.0");
    }

    #[test]
    fn unsupported_opcode_is_counted_as_fallback() {
        // CSET (condition-code set) has no lift rule, but writes to a register,
        // so the lift loop will attempt it and count it as a fallback.
        let sass = r#"
            /*0000*/ CSET R5, CC.CF ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        assert!(out.by_def.is_empty(), "CSET should not be lifted");
        assert!(
            out.stats.attempted > 0,
            "CSET should be attempted (has defs)"
        );
        assert!(
            out.stats.fallback > 0,
            "CSET should fall through as unsupported"
        );
        assert_eq!(
            out.stats.lifted, 0,
            "nothing should be lifted for a single unsupported opcode"
        );
    }

    #[test]
    fn lifts_ffma_to_mul_add() {
        let sass = r#"
            /*0000*/ FFMA R5, R1, R2, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FFMA");
        assert_eq!(lifted.rhs.render(), "R1.0 * R2.0 + R3.0");
    }

    #[test]
    fn lifts_ffma_ftz_to_mul_add() {
        let sass = r#"
            /*0000*/ FFMA.FTZ R5, R1, R2, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FFMA.FTZ");
        assert_eq!(lifted.rhs.render(), "R1.0 * R2.0 + R3.0");
    }

    #[test]
    fn lifts_fmnmx_with_pt_to_fminf() {
        let sass = r#"
            /*0000*/ FMNMX R5, R1, R2, PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FMNMX");
        assert_eq!(lifted.rhs.render(), "fminf(R1.0, R2.0)");
    }

    #[test]
    fn lifts_fmnmx_with_not_pt_to_fmaxf() {
        let sass = r#"
            /*0000*/ FMNMX R5, R1, R2, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FMNMX with !PT");
        assert_eq!(lifted.rhs.render(), "fmaxf(R1.0, R2.0)");
    }

    #[test]
    fn lifts_fmnmx_with_pred_to_ternary() {
        let sass = r#"
            /*0000*/ FMNMX R5, R1, R2, P0 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FMNMX with P0");
        let rendered = lifted.rhs.render();
        assert!(
            rendered.contains("fminf") && rendered.contains("fmaxf"),
            "expected ternary with fminf/fmaxf, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_mufu_rsq_to_rsqrtf() {
        let sass = r#"
            /*0000*/ MUFU.RSQ R5, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted MUFU.RSQ");
        assert_eq!(lifted.rhs.render(), "rsqrtf(R3.0)");
    }

    #[test]
    fn lifts_mufu_ex2_to_exp2f() {
        let sass = r#"
            /*0000*/ MUFU.EX2 R5, R3 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted MUFU.EX2");
        assert_eq!(lifted.rhs.render(), "exp2f(R3.0)");
    }

    #[test]
    fn hfma2_zero_source_materialization_keeps_raw_packed_bits() {
        let args = vec![
            IRExpr::Reg(RegId {
                class: "RZ".to_string(),
                idx: 0,
                sign: -1,
                ssa: None,
            }),
            IRExpr::Reg(RegId {
                class: "RZ".to_string(),
                idx: 0,
                sign: 1,
                ssa: None,
            }),
            IRExpr::ImmF(-0.92529296875),
            IRExpr::ImmF(-0.10186767578125),
        ];
        let lifted = lift_hfma2_constant_materialization("HFMA2", &args)
            .expect("expected zero-source HFMA2 materialization");
        let hi = half_immediate_bits(&args[2]).expect("hi half bits") as u32;
        let lo = half_immediate_bits(&args[3]).expect("lo half bits") as u32;
        assert_eq!(lifted.render(), ((hi << 16) | lo).to_string());
    }

    #[test]
    fn lifts_abs_operand_modifier_without_raw_bars() {
        let sass = r#"
            /*0000*/ FMUL R5, |R7|.reuse, 2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted FMUL");
        let rendered = lifted.rhs.render();
        assert!(rendered.contains("abs(R7.0)"), "got: {}", rendered);
        assert!(!rendered.contains("|R7|"), "got: {}", rendered);
        assert!(!rendered.contains("reuse"), "got: {}", rendered);
    }

    #[test]
    fn lifts_lop3_lut_ternary_select_0xca() {
        // LOP3.LUT with imm=0xCA: bitwise MUX (a & b) | (~a & c)
        // Bit index = (a<<2)|(b<<1)|c, so a (=R1) is the selector.
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0xca, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0xCA");
        let rendered = lifted.rhs.render();
        // bitmux(sel, if1, if0) = (sel & if1) | (~sel & if0)
        assert_eq!(
            rendered, "bitmux(R1.0, R2.0, R3.0)",
            "expected bitmux(a, b, c) for 0xCA, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_lop3_lut_reversed_bitmux_0xac() {
        // LOP3.LUT with imm=0xAC: bitwise MUX (a & c) | (~a & b)
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0xac, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0xAC");
        let rendered = lifted.rhs.render();
        // 0xAC = (a & c) | (~a & b) → bitmux(a, c, b) — note reversed if1/if0
        assert_eq!(
            rendered, "bitmux(R1.0, R3.0, R2.0)",
            "expected bitmux(a, c, b) for 0xAC, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_lop3_lut_choose_with_second_operand_selector_0xe2() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0xe2, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0xE2");
        let rendered = lifted.rhs.render();
        assert_eq!(
            rendered, "bitmux(R2.0, R1.0, R3.0)",
            "expected bitmux(b, a, c) for 0xE2, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_lop3_lut_three_way_and_0x80() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0x80, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0x80");
        let rendered = lifted.rhs.render();
        // 0x80 = a & b & c → "R1.0 & R2.0 & R3.0"
        // (renderer elides parens for same-precedence associative chains)
        assert_eq!(
            rendered, "R1.0 & R2.0 & R3.0",
            "expected a & b & c, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_lop3_lut_three_way_xor_0x96() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0x96, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0x96");
        let rendered = lifted.rhs.render();
        // 0x96 = a ^ b ^ c → "R1.0 ^ R2.0 ^ R3.0"
        assert_eq!(
            rendered, "R1.0 ^ R2.0 ^ R3.0",
            "expected a ^ b ^ c, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_lop3_lut_unknown_to_helper() {
        // Use an uncommon LUT value that doesn't match any named pattern
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0x17, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0x17");
        let rendered = lifted.rhs.render();
        // Unknown patterns use lop3_lut_0xNN(a, b, c) with correct operand order
        assert_eq!(
            rendered, "lop3_lut_0x17(R1.0, R2.0, R3.0)",
            "expected lop3_lut_0x17 helper with correct operands, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_lop3_lut_or_bc_0xf8() {
        // 0xF8 = a | (b & c)
        let sass = r#"
            /*0000*/ LOP3.LUT R5, R1, R2, R3, 0xf8, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted LOP3.LUT 0xF8");
        let rendered = lifted.rhs.render();
        // 0xF8 = a | (b & c) — with correct C precedence (& > |), no parens needed
        assert_eq!(
            rendered, "R1.0 | R2.0 & R3.0",
            "expected a | (b & c), got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_prmt_two_source_halfword_blend() {
        let sass = r#"
            /*0000*/ PRMT R5, R1, 0x7610, R2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted PRMT 0x7610");
        let rendered = lifted.rhs.render();
        assert_eq!(
            rendered, "R1.0 & 65535 | R2.0 & 4294901760",
            "expected low-half/high-half blend, got: {}",
            rendered
        );
    }

    #[test]
    fn lifts_prmt_zero_src0_to_high_half_mask() {
        let sass = r#"
            /*0000*/ PRMT R5, RZ, 0x7610, R2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_def
            .get(&DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            })
            .expect("expected lifted PRMT zero-src0 0x7610");
        let rendered = lifted.rhs.render();
        assert_eq!(
            rendered, "R2.0 & 4294901760",
            "expected high-half mask from src1, got: {}",
            rendered
        );
    }
}
