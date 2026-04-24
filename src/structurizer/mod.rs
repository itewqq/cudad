//! CFG structurization pass used by the decompiler pipeline.
//! This pass is intentionally conservative: when a region cannot be safely
//! converted into structured control flow, it falls back to explicit goto.

use crate::ast::{
    Expr as AstExpr, IntrinsicOp, LValue as AstLValue, LoopKind as AstLoopKind, PointerLane,
    Stmt as AstStmt,
};
#[cfg(test)]
use crate::ast_passes::ast_simplify;
use crate::ast_passes::{
    ast_apply_token_map, ast_cleanup_with_seeded_wide_addrs, SeededWideAddrInfo,
    SeededWideAddrMaps,
};
use crate::cfg::ControlFlowGraph;
use crate::ir::{DisplayCtx, FunctionIR, IRBlock, IRCond, IRExpr, IRStatement, RValue, RegId};
use crate::name_recovery::apply_token_map_to_rendered;
use crate::semantic_lift::{DefRef, SemanticLiftResult};

use petgraph::algo::dominators::{simple_fast, Dominators};
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// --- Structured AST Definition ---
#[derive(Debug, Clone, PartialEq)]
pub enum LoopType {
    While,
    DoWhile, // Detection not fully implemented
    Endless,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructuredStatement {
    BasicBlock {
        block_id: usize,
        stmts: Vec<IRStatement>,
    },
    Sequence(Vec<StructuredStatement>),
    If {
        condition_block_id: usize,
        condition_expr: IRExpr,
        then_branch: Box<StructuredStatement>,
        else_branch: Option<Box<StructuredStatement>>,
    },
    Loop {
        loop_type: LoopType,
        header_block_id: Option<usize>,
        condition_expr: Option<IRExpr>,
        body: Box<StructuredStatement>,
    },
    Break(Option<usize>),
    Continue(Option<usize>),
    Return(Option<IRExpr>),
    UnstructuredJump {
        from_block_id: usize,
        to_block_id: usize,
        condition: Option<IRExpr>,
    },
    /// Multi-way branch (switch/case), recognized from nodes with >2
    /// outgoing edges (e.g. branch table / cascaded if-else chains).
    Switch {
        header_block_id: usize,
        /// The expression being switched on (if recoverable).
        discriminant: Option<IRExpr>,
        /// Each case arm: (case label/index, body).
        cases: Vec<(usize, StructuredStatement)>,
        /// Optional default arm.
        default: Option<Box<StructuredStatement>>,
    },
    Empty,
}

impl fmt::Display for StructuredStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                write!(f, "BB{}[{} stmts]", block_id, stmts.len())
            }
            StructuredStatement::Sequence(s) => {
                write!(f, "Sequence(")?;
                for (i, stmt) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    stmt.fmt(f)?;
                }
                write!(f, ")")
            }
            StructuredStatement::If {
                condition_block_id,
                condition_expr,
                then_branch,
                else_branch,
                ..
            } => {
                write!(
                    f,
                    "If(BB{}, Cond: {:?}, Then: {}",
                    condition_block_id, condition_expr, then_branch
                )?;
                if let Some(eb) = else_branch {
                    write!(f, ", Else: {}", eb)?;
                }
                write!(f, ")")
            }
            StructuredStatement::Loop {
                header_block_id,
                loop_type,
                condition_expr,
                body,
                ..
            } => {
                write!(
                    f,
                    "{:?}-Loop(Hdr: BB{:?}, Cond: {:?}, Body: {})",
                    loop_type,
                    header_block_id.unwrap_or(usize::MAX),
                    condition_expr,
                    body
                )
            }
            StructuredStatement::Break(_) => write!(f, "Break"),
            StructuredStatement::Continue(_) => write!(f, "Continue"),
            StructuredStatement::Return(_) => write!(f, "Return"),
            StructuredStatement::UnstructuredJump {
                from_block_id,
                to_block_id,
                ..
            } => {
                write!(f, "Goto(BB{}->BB{})", from_block_id, to_block_id)
            }
            StructuredStatement::Switch {
                header_block_id,
                cases,
                default,
                ..
            } => {
                write!(f, "Switch(BB{}, {} cases", header_block_id, cases.len())?;
                if default.is_some() {
                    write!(f, " +default")?;
                }
                write!(f, ")")
            }
            StructuredStatement::Empty => write!(f, "Empty"),
        }
    }
}

// --- Structurizer Implementation ---
fn seeded_wide_ptr_info(base: &str) -> SeededWideAddrInfo {
    SeededWideAddrInfo {
        base: AstExpr::Raw(base.to_string()),
        seed_lo: AstExpr::PtrLane {
            base: base.to_string(),
            lane: PointerLane::Lo32,
        },
    }
}

fn parse_constmem_word_symbol(text: &str) -> Option<(u32, u32)> {
    let body = text.strip_prefix("c[0x")?;
    let (bank_hex, rest) = body.split_once("][0x")?;
    let offset_hex = rest.strip_suffix(']')?;
    let bank = u32::from_str_radix(bank_hex, 16).ok()?;
    let offset = u32::from_str_radix(offset_hex, 16).ok()?;
    Some((bank, offset))
}

fn adjacent_constmem_word_symbol(text: &str) -> Option<String> {
    let (bank, offset) = parse_constmem_word_symbol(text)?;
    Some(format!("c[0x{:x}][0x{:x}]", bank, offset + 4))
}

fn seeded_constmem_pair_info(lo_name: &str, lo_sym: &str, hi_sym: &str) -> SeededWideAddrInfo {
    let render_sym = |text: &str| {
        if text.starts_with("c[0x") {
            AstExpr::Raw(text.to_string())
        } else {
            AstExpr::ConstMemSymbol(text.to_string())
        }
    };
    SeededWideAddrInfo {
        base: AstExpr::Addr64 {
            lo: Box::new(render_sym(lo_sym)),
            hi: Box::new(render_sym(hi_sym)),
        },
        seed_lo: AstExpr::Reg(lo_name.to_string()),
    }
}

fn ast_lvalue_var_name(lvalue: &AstLValue) -> Option<String> {
    match lvalue {
        AstLValue::Var(name) => Some(name.clone()),
        AstLValue::PtrLane { base, lane } => Some(format!("{}.{}", base, lane.render_suffix())),
        _ => None,
    }
}

fn expr_ptr_lane_base(expr: &AstExpr, lane: PointerLane) -> Option<String> {
    match expr {
        AstExpr::PtrLane {
            base,
            lane: actual_lane,
        } if *actual_lane == lane => Some(base.clone()),
        AstExpr::ConstMemSymbol(name) | AstExpr::Reg(name) => PointerLane::parse_named(name)
            .and_then(|(base, actual_lane)| (actual_lane == lane).then_some(base)),
        _ => None,
    }
}

fn infer_lo_seed_info(
    expr: &AstExpr,
    lo_by_name: &HashMap<String, SeededWideAddrInfo>,
) -> Option<SeededWideAddrInfo> {
    if let Some(base) = expr_ptr_lane_base(expr, PointerLane::Lo32) {
        return Some(seeded_wide_ptr_info(&base));
    }
    match expr {
        AstExpr::Reg(name) => lo_by_name.get(name).cloned(),
        AstExpr::Binary { op, lhs, rhs } if op == "+" => {
            infer_lo_seed_info(lhs, lo_by_name).or_else(|| infer_lo_seed_info(rhs, lo_by_name))
        }
        AstExpr::Ternary {
            then_expr,
            else_expr,
            ..
        } => {
            let then_info = infer_lo_seed_info(then_expr, lo_by_name)?;
            let else_info = infer_lo_seed_info(else_expr, lo_by_name)?;
            (then_info == else_info).then_some(then_info)
        }
        _ => None,
    }
}

fn infer_hi_seed_info(
    expr: &AstExpr,
    lo_by_name: &HashMap<String, SeededWideAddrInfo>,
    hi_by_name: &HashMap<String, SeededWideAddrInfo>,
) -> Option<SeededWideAddrInfo> {
    if let Some(base) = expr_ptr_lane_base(expr, PointerLane::Hi32) {
        return Some(seeded_wide_ptr_info(&base));
    }
    match expr {
        AstExpr::Reg(name) => hi_by_name.get(name).cloned(),
        AstExpr::Intrinsic { op, args }
            if matches!(op, IntrinsicOp::LeaHiX | IntrinsicOp::LeaHiXSx32) && args.len() >= 2 =>
        {
            infer_hi_seed_info(&args[1], lo_by_name, hi_by_name)
        }
        AstExpr::CallLike { func, args }
            if matches!(func.as_str(), "lea_hi_x" | "lea_hi_x_sx32") && args.len() >= 2 =>
        {
            infer_hi_seed_info(&args[1], lo_by_name, hi_by_name)
        }
        AstExpr::Binary { op, lhs, rhs } if op == "+" => {
            infer_hi_seed_info(lhs, lo_by_name, hi_by_name)
                .or_else(|| infer_hi_seed_info(rhs, lo_by_name, hi_by_name))
                .or_else(|| infer_lo_seed_info(lhs, lo_by_name))
                .or_else(|| infer_lo_seed_info(rhs, lo_by_name))
        }
        AstExpr::Ternary {
            then_expr,
            else_expr,
            ..
        } => {
            let then_info = infer_hi_seed_info(then_expr, lo_by_name, hi_by_name)?;
            let else_info = infer_hi_seed_info(else_expr, lo_by_name, hi_by_name)?;
            (then_info == else_info).then_some(then_info)
        }
        _ => None,
    }
}

fn unify_phi_seed_info(
    args: &[IRExpr],
    known: &HashMap<String, SeededWideAddrInfo>,
) -> Option<SeededWideAddrInfo> {
    let mut resolved = None;
    let mut saw_known = false;
    for arg in args {
        let Some(reg) = arg.get_reg() else {
            continue;
        };
        let Some(info) = known.get(&reg.display()) else {
            continue;
        };
        saw_known = true;
        match &resolved {
            None => resolved = Some(info.clone()),
            Some(existing) if existing == info => {}
            Some(_) => return None,
        }
    }
    if saw_known {
        resolved
    } else {
        None
    }
}

fn pointer_seed_preserving_opcode(opcode: &str) -> bool {
    opcode.starts_with("IADD")
        || opcode.starts_with("UIADD")
        || opcode.starts_with("LEA")
        || opcode == "MOV"
        || opcode.starts_with("MOV.")
        || opcode.starts_with("IMAD.MOV")
        || opcode.starts_with("IMAD.X")
}

fn infer_seed_info_from_ir_args(
    args: &[IRExpr],
    known: &HashMap<String, SeededWideAddrInfo>,
) -> Option<SeededWideAddrInfo> {
    let mut resolved = None;
    for arg in args {
        let Some(reg) = arg.get_reg() else {
            continue;
        };
        let Some(info) = known.get(&reg.display()) else {
            continue;
        };
        match &resolved {
            None => resolved = Some(info.clone()),
            Some(existing) if existing == info => {}
            Some(_) => return None,
        }
    }
    resolved
}

pub fn build_seeded_wide_addr_maps(
    function_ir: &FunctionIR,
    lifted: Option<&SemanticLiftResult>,
) -> SeededWideAddrMaps {
    let Some(lifted) = lifted else {
        return SeededWideAddrMaps::default();
    };

    let mut rhs_by_name = HashMap::<String, AstExpr>::new();
    for lifted_stmt in lifted.by_def.values() {
        let Some(name) = ast_lvalue_var_name(&lifted_stmt.dest) else {
            continue;
        };
        rhs_by_name
            .entry(name)
            .or_insert_with(|| lifted_stmt.rhs.clone());
    }

    let mut seeded = SeededWideAddrMaps::default();
    let constmem_word_defs = rhs_by_name
        .iter()
        .filter_map(|(name, rhs)| match rhs {
            AstExpr::ConstMemSymbol(sym) | AstExpr::Raw(sym)
                if parse_constmem_word_symbol(sym).is_some() =>
            {
                Some((sym.clone(), name.clone()))
            }
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    for (name, rhs) in &rhs_by_name {
        let lo_sym = match rhs {
            AstExpr::ConstMemSymbol(sym) | AstExpr::Raw(sym)
                if parse_constmem_word_symbol(sym).is_some() =>
            {
                sym
            }
            _ => continue,
        };
        let Some(hi_sym) = adjacent_constmem_word_symbol(lo_sym) else {
            continue;
        };
        let Some(hi_name) = constmem_word_defs.get(&hi_sym) else {
            continue;
        };
        let info = seeded_constmem_pair_info(name, lo_sym, &hi_sym);
        seeded.lo_by_name.entry(name.clone()).or_insert_with(|| info.clone());
        seeded.hi_by_name
            .entry(hi_name.clone())
            .or_insert_with(|| info.clone());
    }
    for _ in 0..64 {
        let mut changed = false;
        for (name, rhs) in &rhs_by_name {
            if !seeded.lo_by_name.contains_key(name) {
                if let Some(info) = infer_lo_seed_info(rhs, &seeded.lo_by_name) {
                    seeded.lo_by_name.insert(name.clone(), info);
                    changed = true;
                }
            }
            if !seeded.hi_by_name.contains_key(name) {
                if let Some(info) = infer_hi_seed_info(rhs, &seeded.lo_by_name, &seeded.hi_by_name)
                {
                    seeded.hi_by_name.insert(name.clone(), info);
                    changed = true;
                }
            }
        }
        for block in &function_ir.blocks {
            for stmt in &block.stmts {
                if let RValue::Op { opcode, args } = &stmt.value {
                    if stmt.defs.len() == 1 && pointer_seed_preserving_opcode(opcode) {
                        if let Some(reg) = stmt.defs[0].get_reg() {
                            let name = reg.display();
                            if !seeded.lo_by_name.contains_key(&name) {
                                if let Some(info) =
                                    infer_seed_info_from_ir_args(args, &seeded.lo_by_name)
                                {
                                    seeded.lo_by_name.insert(name.clone(), info);
                                    changed = true;
                                }
                            }
                            if !seeded.hi_by_name.contains_key(&name) {
                                if let Some(info) =
                                    infer_seed_info_from_ir_args(args, &seeded.hi_by_name)
                                {
                                    seeded.hi_by_name.insert(name.clone(), info);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                let RValue::Phi(args) = &stmt.value else {
                    continue;
                };
                for def in &stmt.defs {
                    let Some(reg) = def.get_reg() else {
                        continue;
                    };
                    let name = reg.display();
                    if !seeded.lo_by_name.contains_key(&name) {
                        if let Some(info) = unify_phi_seed_info(args, &seeded.lo_by_name) {
                            seeded.lo_by_name.insert(name.clone(), info);
                            changed = true;
                        }
                    }
                    if !seeded.hi_by_name.contains_key(&name) {
                        if let Some(info) = unify_phi_seed_info(args, &seeded.hi_by_name) {
                            seeded.hi_by_name.insert(name.clone(), info);
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    seeded
}

fn collect_ast_stmt_uses(stmt: &AstStmt, out: &mut HashSet<String>) {
    match stmt {
        AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
            for stmt in stmts {
                collect_ast_stmt_uses(stmt, out);
            }
        }
        AstStmt::Label { body, .. } => collect_ast_stmt_uses(body, out),
        AstStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_ast_expr_uses(condition, out);
            collect_ast_stmt_uses(then_branch, out);
            if let Some(else_branch) = else_branch.as_deref() {
                collect_ast_stmt_uses(else_branch, out);
            }
        }
        AstStmt::Loop {
            condition, body, ..
        } => {
            if let Some(condition) = condition {
                collect_ast_expr_uses(condition, out);
            }
            collect_ast_stmt_uses(body, out);
        }
        AstStmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            if let Some(discriminant) = discriminant {
                collect_ast_expr_uses(discriminant, out);
            }
            for (_, body) in cases {
                collect_ast_stmt_uses(body, out);
            }
            if let Some(default) = default.as_deref() {
                collect_ast_stmt_uses(default, out);
            }
        }
        AstStmt::Return(expr) => {
            if let Some(expr) = expr {
                collect_ast_expr_uses(expr, out);
            }
        }
        AstStmt::Assign { dst, src } => {
            collect_ast_lvalue_uses(dst, out);
            collect_ast_expr_uses(src, out);
        }
        AstStmt::ExprStmt(expr) => collect_ast_expr_uses(expr, out),
        AstStmt::Break | AstStmt::Continue | AstStmt::Goto(_) | AstStmt::Empty => {}
    }
}

fn collect_ast_lvalue_uses(lvalue: &AstLValue, out: &mut HashSet<String>) {
    match lvalue {
        AstLValue::Var(_) | AstLValue::Raw(_) => {}
        AstLValue::PtrLane { base, lane } => {
            out.insert(format!("{}.{}", base, lane.render_suffix()));
        }
        AstLValue::Deref { addr, .. } => collect_ast_expr_uses(addr, out),
        AstLValue::Indexed { base, index } => {
            collect_ast_expr_uses(base, out);
            collect_ast_expr_uses(index, out);
        }
    }
}

fn collect_ast_expr_uses(expr: &AstExpr, out: &mut HashSet<String>) {
    match expr {
        AstExpr::Raw(_) | AstExpr::Imm(_) | AstExpr::Builtin(_) => {}
        AstExpr::Reg(name) | AstExpr::ConstMemSymbol(name) => {
            out.insert(name.clone());
        }
        AstExpr::PtrLane { base, lane } => {
            out.insert(format!("{}.{}", base, lane.render_suffix()));
        }
        AstExpr::LaneExtract { value, .. } => collect_ast_expr_uses(value, out),
        AstExpr::Unary { arg, .. } => collect_ast_expr_uses(arg, out),
        AstExpr::Binary { lhs, rhs, .. } => {
            collect_ast_expr_uses(lhs, out);
            collect_ast_expr_uses(rhs, out);
        }
        AstExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_ast_expr_uses(cond, out);
            collect_ast_expr_uses(then_expr, out);
            collect_ast_expr_uses(else_expr, out);
        }
        AstExpr::CallLike { args, .. } | AstExpr::Intrinsic { args, .. } => {
            for arg in args {
                collect_ast_expr_uses(arg, out);
            }
        }
        AstExpr::Load { addr, .. } => collect_ast_expr_uses(addr, out),
        AstExpr::WidePtr { base, offset } => {
            collect_ast_expr_uses(base, out);
            collect_ast_expr_uses(offset, out);
        }
        AstExpr::Addr64 { lo, hi } => {
            collect_ast_expr_uses(lo, out);
            collect_ast_expr_uses(hi, out);
        }
        AstExpr::Cast { expr, .. } => collect_ast_expr_uses(expr, out),
        AstExpr::Index { base, index } => {
            collect_ast_expr_uses(base, out);
            collect_ast_expr_uses(index, out);
        }
    }
}

fn fresh_split_name(base: &str, used: &mut HashSet<String>) -> String {
    let stem = format!("{}_next", base);
    if used.insert(stem.clone()) {
        return stem;
    }
    let mut idx = 1usize;
    loop {
        let candidate = format!("{}_{}", stem, idx);
        if used.insert(candidate.clone()) {
            return candidate;
        }
        idx += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum AliasTypeClass {
    Bool,
    Int,
    Float,
    Half,
    Pointer,
}

fn split_type_conflicting_aliases(
    stmt: &AstStmt,
    token_map: &mut HashMap<String, String>,
    used_output_names: &mut HashSet<String>,
) {
    let mut classes = HashMap::<String, HashSet<AliasTypeClass>>::new();
    collect_token_type_classes(stmt, &mut classes);

    let mut grouped = HashMap::<String, Vec<(String, AliasTypeClass)>>::new();
    for (token, token_classes) in classes {
        let Some(mapped) = token_map.get(&token).cloned() else {
            continue;
        };
        let mut sorted = token_classes.into_iter().collect::<Vec<_>>();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != 1 {
            continue;
        }
        grouped
            .entry(mapped)
            .or_default()
            .push((token, sorted[0]));
    }

    for (shared_name, originals) in grouped {
        let mut by_class = HashMap::<AliasTypeClass, Vec<String>>::new();
        for (token, class) in originals {
            by_class.entry(class).or_default().push(token);
        }
        if by_class.len() < 2 {
            continue;
        }

        let keep_class = by_class
            .iter()
            .max_by_key(|(_, tokens)| tokens.len())
            .map(|(class, _)| *class)
            .unwrap();
        for (class, mut tokens) in by_class {
            if class == keep_class {
                continue;
            }
            tokens.sort();
            tokens.dedup();
            for token in tokens {
                if token_map.get(&token) != Some(&shared_name) {
                    continue;
                }
                let fresh = fresh_split_name(&shared_name, used_output_names);
                token_map.insert(token, fresh);
            }
        }
        used_output_names.insert(shared_name);
    }
}

fn collect_token_type_classes(
    stmt: &AstStmt,
    out: &mut HashMap<String, HashSet<AliasTypeClass>>,
) {
    match stmt {
        AstStmt::Label { body, .. } => collect_token_type_classes(body, out),
        AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
            for stmt in stmts {
                collect_token_type_classes(stmt, out);
            }
        }
        AstStmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_token_type_classes(then_branch, out);
            if let Some(else_branch) = else_branch.as_deref() {
                collect_token_type_classes(else_branch, out);
            }
        }
        AstStmt::Loop { body, .. } => collect_token_type_classes(body, out),
        AstStmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_token_type_classes(body, out);
            }
            if let Some(default) = default.as_deref() {
                collect_token_type_classes(default, out);
            }
        }
        AstStmt::Assign { dst, src } => {
            let Some(name) = ast_lvalue_var_name(dst) else {
                return;
            };
            let Some(class) = ast_expr_type_class(src) else {
                return;
            };
            out.entry(name.to_string()).or_default().insert(class);
        }
        AstStmt::Break
        | AstStmt::Continue
        | AstStmt::Return(_)
        | AstStmt::ExprStmt(_)
        | AstStmt::Goto(_)
        | AstStmt::Empty => {}
    }
}

fn ast_expr_type_class(expr: &AstExpr) -> Option<AliasTypeClass> {
    match expr {
        AstExpr::Imm(text) | AstExpr::Raw(text) => raw_expr_type_class(text),
        AstExpr::Reg(_) | AstExpr::ConstMemSymbol(_) | AstExpr::Builtin(_) => None,
        AstExpr::PtrLane { .. } | AstExpr::LaneExtract { .. } => Some(AliasTypeClass::Int),
        AstExpr::Unary { arg, .. } => ast_expr_type_class(arg),
        AstExpr::Binary { op, lhs, rhs } => {
            let lhs_class = ast_expr_type_class(lhs);
            let rhs_class = ast_expr_type_class(rhs);
            if matches!(op.as_str(), "+" | "-") {
                if lhs_class == Some(AliasTypeClass::Pointer) && rhs_class == Some(AliasTypeClass::Int)
                {
                    return Some(AliasTypeClass::Pointer);
                }
                if rhs_class == Some(AliasTypeClass::Pointer) && lhs_class == Some(AliasTypeClass::Int)
                {
                    return Some(AliasTypeClass::Pointer);
                }
            }
            match (lhs_class, rhs_class) {
                (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
                (Some(AliasTypeClass::Float), _) | (_, Some(AliasTypeClass::Float)) => {
                    Some(AliasTypeClass::Float)
                }
                (Some(AliasTypeClass::Half), _) | (_, Some(AliasTypeClass::Half)) => {
                    Some(AliasTypeClass::Half)
                }
                (Some(AliasTypeClass::Int), Some(AliasTypeClass::Int)) => {
                    Some(AliasTypeClass::Int)
                }
                _ => None,
            }
        }
        AstExpr::Ternary {
            then_expr,
            else_expr,
            ..
        } => {
            let then_class = ast_expr_type_class(then_expr);
            let else_class = ast_expr_type_class(else_expr);
            match (then_class, else_class) {
                (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
                (Some(lhs), None) => Some(lhs),
                (None, Some(rhs)) => Some(rhs),
                _ => None,
            }
        }
        AstExpr::CallLike { func, args } => {
            if func.ends_with('f') {
                return Some(AliasTypeClass::Float);
            }
            if matches!(func.as_str(), "bitmux" | "majority" | "mul_hi_u32") {
                return Some(AliasTypeClass::Int);
            }
            args.iter().find_map(ast_expr_type_class)
        }
        AstExpr::Intrinsic { op, args } => match op {
            IntrinsicOp::CarryU32Add3
            | IntrinsicOp::LeaHiX
            | IntrinsicOp::LeaHiXSx32
            | IntrinsicOp::PairHi => Some(AliasTypeClass::Int),
            IntrinsicOp::Min | IntrinsicOp::Max | IntrinsicOp::Clamp => {
                args.iter().find_map(ast_expr_type_class)
            }
        },
        AstExpr::Load { ty, addr } => ty
            .as_deref()
            .and_then(type_name_class)
            .or_else(|| load_addr_type_class(addr)),
        AstExpr::WidePtr { .. } | AstExpr::Addr64 { .. } => Some(AliasTypeClass::Pointer),
        AstExpr::Cast { ty, expr } => type_name_class(ty).or_else(|| ast_expr_type_class(expr)),
        AstExpr::Index { base, .. } => ast_expr_type_class(base),
    }
}

fn raw_expr_type_class(text: &str) -> Option<AliasTypeClass> {
    let text = text.trim();
    if matches!(text, "true" | "false") {
        return Some(AliasTypeClass::Bool);
    }
    if text.contains('.') || text.contains("e-") || text.contains("e+") || text.contains("E-") || text.contains("E+") {
        if text.chars().any(|ch| ch.is_ascii_digit()) {
            return Some(AliasTypeClass::Float);
        }
    }
    text.parse::<i128>().ok().map(|_| AliasTypeClass::Int)
}

fn type_name_class(ty: &str) -> Option<AliasTypeClass> {
    let ty = ty.trim();
    if ty.ends_with('*') || matches!(ty, "uintptr_t" | "intptr_t") {
        return Some(AliasTypeClass::Pointer);
    }
    match ty {
        "float" => Some(AliasTypeClass::Float),
        "__half" => Some(AliasTypeClass::Half),
        "bool" => Some(AliasTypeClass::Bool),
        "uint8_t" | "int8_t" | "uint16_t" | "int16_t" | "uint32_t" | "int32_t" | "uint64_t"
        | "int64_t" => Some(AliasTypeClass::Int),
        _ => None,
    }
}

fn load_addr_type_class(addr: &AstExpr) -> Option<AliasTypeClass> {
    match addr {
        AstExpr::Cast { ty, .. } => ty
            .trim()
            .strip_suffix('*')
            .and_then(type_name_class)
            .or_else(|| type_name_class(ty)),
        AstExpr::Binary { lhs, rhs, .. } => load_addr_type_class(lhs).or_else(|| load_addr_type_class(rhs)),
        AstExpr::WidePtr { base, .. } => load_addr_type_class(base),
        _ => None,
    }
}

fn split_unsafe_token_aliases(stmt: &AstStmt, token_map: &mut HashMap<String, String>) {
    let mut used_output_names = token_map.values().cloned().collect::<HashSet<_>>();
    split_type_conflicting_aliases(stmt, token_map, &mut used_output_names);
    split_unsafe_token_aliases_stmt(stmt, token_map, &mut used_output_names);
}

fn split_stmt_local_alias_collisions(
    stmt: &AstStmt,
    token_map: &mut HashMap<String, String>,
    used_output_names: &mut HashSet<String>,
) {
    let mut symbols = HashSet::new();
    collect_ast_stmt_uses(stmt, &mut symbols);
    if let AstStmt::Assign { dst, .. } = stmt {
        if let Some(name) = ast_lvalue_var_name(dst) {
            symbols.insert(name.to_string());
        }
    }

    let mut by_output = HashMap::<String, Vec<String>>::new();
    for symbol in symbols {
        if let Some(mapped) = token_map.get(&symbol).cloned() {
            by_output.entry(mapped).or_default().push(symbol);
        }
    }

    for (shared_name, mut originals) in by_output {
        originals.sort();
        originals.dedup();
        if originals.len() < 2 {
            continue;
        }
        let keep = originals.remove(0);
        for original in originals {
            if token_map.get(&original) != Some(&shared_name) {
                continue;
            }
            let fresh = fresh_split_name(&shared_name, used_output_names);
            token_map.insert(original, fresh);
        }
        used_output_names.insert(shared_name);
        let _ = keep;
    }
}

fn split_unsafe_token_aliases_stmt(
    stmt: &AstStmt,
    token_map: &mut HashMap<String, String>,
    used_output_names: &mut HashSet<String>,
) {
    split_stmt_local_alias_collisions(stmt, token_map, used_output_names);
    match stmt {
        AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
            let mut future_uses = vec![HashSet::<String>::new(); stmts.len() + 1];
            for idx in (0..stmts.len()).rev() {
                future_uses[idx] = future_uses[idx + 1].clone();
                collect_ast_stmt_uses(&stmts[idx], &mut future_uses[idx]);
            }
            for (idx, child) in stmts.iter().enumerate() {
                if let AstStmt::Assign { dst, src } = child {
                    if let Some(dst_name) = ast_lvalue_var_name(dst) {
                        if let Some(shared_name) = token_map.get(&dst_name).cloned() {
                            let mut rhs_uses = HashSet::new();
                            collect_ast_lvalue_uses(dst, &mut rhs_uses);
                            collect_ast_expr_uses(src, &mut rhs_uses);
                            let future = &future_uses[idx + 1];
                            let overlaps = rhs_uses.into_iter().any(|used| {
                                used != dst_name
                                    && future.contains(&used)
                                    && token_map.get(&used) == Some(&shared_name)
                            });
                            if overlaps {
                                let fresh = fresh_split_name(&shared_name, used_output_names);
                                token_map.insert(dst_name, fresh);
                            }
                        }
                    }
                }
                split_unsafe_token_aliases_stmt(child, token_map, used_output_names);
            }
        }
        AstStmt::Label { body, .. } => {
            split_unsafe_token_aliases_stmt(body, token_map, used_output_names);
        }
        AstStmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            split_unsafe_token_aliases_stmt(then_branch, token_map, used_output_names);
            if let Some(else_branch) = else_branch.as_deref() {
                split_unsafe_token_aliases_stmt(else_branch, token_map, used_output_names);
            }
        }
        AstStmt::Loop { body, .. } => {
            split_unsafe_token_aliases_stmt(body, token_map, used_output_names);
        }
        AstStmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                split_unsafe_token_aliases_stmt(body, token_map, used_output_names);
            }
            if let Some(default) = default.as_deref() {
                split_unsafe_token_aliases_stmt(default, token_map, used_output_names);
            }
        }
        AstStmt::Break
        | AstStmt::Continue
        | AstStmt::Return(_)
        | AstStmt::Assign { .. }
        | AstStmt::ExprStmt(_)
        | AstStmt::Goto(_)
        | AstStmt::Empty => {}
    }
}

pub struct Structurizer<'a> {
    cfg: &'a ControlFlowGraph, // DiGraph<CfgBasicBlock, EdgeKind>
    function_ir: &'a FunctionIR,
    dom: Dominators<NodeIndex>,
    addr_to_node_index: HashMap<u32, NodeIndex>,
    /// Maps IRBlock.id → index into function_ir.blocks for O(1) lookup.
    block_id_to_idx: HashMap<usize, usize>,
}

impl<'a> Structurizer<'a> {
    fn choose_entry_node(cfg: &ControlFlowGraph) -> Option<NodeIndex> {
        let mut candidates = cfg
            .node_indices()
            .filter(|&n| cfg.neighbors_directed(n, Direction::Incoming).count() == 0)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return cfg.node_indices().next();
        }
        candidates.sort_by_key(|&n| cfg[n].id);
        candidates.first().copied()
    }

    fn lifted_def_emit_order(stmt: &IRStatement) -> Vec<usize> {
        let RValue::Op { opcode, .. } = &stmt.value else {
            return (0..stmt.defs.len()).collect();
        };
        if stmt.defs.is_empty() {
            if opcode.starts_with("ST") {
                if opcode.split('.').skip(1).any(|t| t == "128") {
                    return vec![0, 1, 2, 3];
                }
                if opcode.split('.').skip(1).any(|t| t == "64") {
                    return vec![0, 1];
                }
            }
            return vec![0];
        }
        let filter_sink_defs = |indices: &mut Vec<usize>| {
            if stmt
                .defs
                .iter()
                .any(|def| !Self::is_zero_or_true_reg(def))
            {
                indices.retain(|idx| {
                    stmt.defs
                        .get(*idx)
                        .is_some_and(|def| !Self::is_zero_or_true_reg(def))
                });
            }
        };
        let mnem = opcode.split('.').next().unwrap_or(opcode);
        let is_iadd3_non_x =
            matches!(mnem, "IADD3" | "UIADD3") && !opcode.split('.').any(|t| t == "X");
        if is_iadd3_non_x && stmt.defs.len() > 1 {
            // Predicate carry defs conceptually happen before the low-result write,
            // but implicit .64 hi-lane defs are still data results and should stay
            // after the low half so mutable-name views keep the tuple update ordered.
            let mut preds = Vec::new();
            let mut data_after_first = Vec::new();
            for idx in 1..stmt.defs.len() {
                match stmt.defs.get(idx).and_then(IRExpr::get_reg) {
                    Some(reg) if matches!(reg.class.as_str(), "P" | "UP") => preds.push(idx),
                    _ => data_after_first.push(idx),
                }
            }
            let mut out = preds;
            out.push(0);
            out.extend(data_after_first);
            filter_sink_defs(&mut out);
            return out;
        }
        let mut out = (0..stmt.defs.len()).collect::<Vec<_>>();
        filter_sink_defs(&mut out);
        out
    }

    fn is_branch_only_opcode(opcode: &str) -> bool {
        matches!(opcode, "BRA" | "JMP" | "JMPP")
    }

    fn is_return_opcode(opcode: &str) -> bool {
        opcode == "RET" || opcode == "EXIT" || opcode.starts_with("RET")
    }

    /// Checks if a structured statement ends with an unconditional return/exit,
    /// meaning any subsequent siblings in a Sequence are dead code.
    fn ends_with_unconditional_return(stmt: &StructuredStatement) -> bool {
        match stmt {
            StructuredStatement::Return(_) => true,
            StructuredStatement::BasicBlock { stmts, .. } => {
                // Look for an unconditional RET/EXIT in the block's stmts.
                stmts.iter().any(|s| {
                    if let RValue::Op { opcode, .. } = &s.value {
                        Self::is_return_opcode(opcode) && s.pred.is_none()
                    } else {
                        false
                    }
                })
            }
            StructuredStatement::Sequence(children) => {
                // A sequence ends with return if its last non-empty child does.
                children
                    .iter()
                    .rev()
                    .find(|c| !matches!(c, StructuredStatement::Empty))
                    .map_or(false, Self::ends_with_unconditional_return)
            }
            _ => false,
        }
    }

    fn is_setp_opcode(opcode: &str) -> bool {
        opcode.starts_with("ISETP") || opcode.starts_with("FSETP")
    }

    fn is_barrier_opcode(opcode: &str) -> bool {
        opcode.starts_with("BAR.SYNC")
    }

    /// Returns true if the register is one of the "sink" registers (RZ, PT, URZ, UPT)
    /// whose writes have no observable effect.
    fn is_zero_or_true_reg(r: &IRExpr) -> bool {
        match r {
            IRExpr::Reg(reg) => matches!(reg.class.as_str(), "RZ" | "PT" | "URZ" | "UPT"),
            _ => false,
        }
    }

    /// Render an IR expression for decompiled output, replacing hardware-zero
    /// and always-true predicate registers with literals ("0" and "true").
    fn decompile_expr(ctx: &dyn DisplayCtx, e: &IRExpr) -> String {
        match e {
            IRExpr::Reg(r) => match r.class.as_str() {
                "RZ" | "URZ" => "0".to_string(),
                "PT" | "UPT" => "true".to_string(),
                _ => ctx.expr(e),
            },
            IRExpr::Op { op, args } => {
                // Recursively decompile arguments so nested RZ/PT are replaced.
                if op == "-" && args.len() == 1 {
                    let inner = Self::decompile_expr(ctx, &args[0]);
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
                if op == "!" && args.len() == 1 {
                    let inner = Self::decompile_expr(ctx, &args[0]);
                    // !true → false, !(expr) → !(expr)
                    if inner == "true" {
                        return "false".to_string();
                    }
                    return format!("!({})", inner);
                }
                let list = args
                    .iter()
                    .map(|a| Self::decompile_expr(ctx, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", op, list)
            }
            _ => ctx.expr(e),
        }
    }

    /// Returns true if the instruction has memory side effects that must be preserved
    /// even when all destination registers are RZ/PT.
    fn is_memory_side_effect_opcode(value: &RValue) -> bool {
        let opcode = match value {
            RValue::Op { opcode, .. } => opcode.as_str(),
            _ => return false,
        };
        // Store instructions, atomics, and shared-memory operations
        opcode.starts_with("STG")
            || opcode.starts_with("STS")
            || opcode.starts_with("STL")
            || opcode.starts_with("ST.")
            || opcode.starts_with("ATOM")
            || opcode.starts_with("RED")
            || opcode.starts_with("CCTL")
            || opcode.starts_with("MEMBAR")
            || opcode.starts_with("BAR")
    }

    fn should_omit_control_predicate_def(
        stmt: &IRStatement,
        stmt_idx: usize,
        block: &IRBlock,
        control_pred_regs: &[RegId],
    ) -> bool {
        let opcode = match &stmt.value {
            RValue::Op { opcode, .. } => opcode.as_str(),
            _ => return false,
        };
        if !Self::is_setp_opcode(opcode) {
            return false;
        }
        if stmt.defs.is_empty() {
            return false;
        }
        let pred_defs = stmt
            .defs
            .iter()
            .filter_map(|d| match d {
                IRExpr::Reg(r) if matches!(r.class.as_str(), "P" | "UP") => Some(r),
                _ => None,
            })
            .collect::<Vec<_>>();
        if pred_defs.is_empty() || pred_defs.len() != stmt.defs.len() {
            return false;
        }
        let all_control_preds = pred_defs.iter().all(|dest_reg| {
            control_pred_regs.iter().any(|r| {
                r.class == dest_reg.class && r.idx == dest_reg.idx && r.ssa == dest_reg.ssa
            })
        });
        if !all_control_preds {
            return false;
        }
        block
            .stmts
            .iter()
            .skip(stmt_idx + 1)
            .all(|s| match &s.value {
                RValue::Op { opcode, .. } => {
                    Self::is_branch_only_opcode(opcode) || Self::is_return_opcode(opcode)
                }
                _ => false,
            })
    }

    fn without_redundant_loop_tail_continue(stmt: &StructuredStatement) -> StructuredStatement {
        match stmt {
            StructuredStatement::Continue(_) => StructuredStatement::Empty,
            StructuredStatement::Sequence(stmts) => {
                if matches!(stmts.last(), Some(StructuredStatement::Continue(_))) {
                    let trimmed = &stmts[..stmts.len() - 1];
                    if trimmed.is_empty() {
                        StructuredStatement::Empty
                    } else if trimmed.len() == 1 {
                        trimmed[0].clone()
                    } else {
                        StructuredStatement::Sequence(trimmed.to_vec())
                    }
                } else {
                    stmt.clone()
                }
            }
            _ => stmt.clone(),
        }
    }

    fn node_is_dominated_by(
        &self,
        dom_results: &Dominators<NodeIndex>,
        node_to_check: NodeIndex,
        potential_dominator: NodeIndex,
    ) -> bool {
        if node_to_check == potential_dominator {
            return true;
        }
        let mut current = node_to_check;
        while let Some(idom) = dom_results.immediate_dominator(current) {
            if idom == potential_dominator {
                return true;
            }
            if idom == current {
                break;
            }
            current = idom;
            if current == NodeIndex::end() {
                break;
            }
        }
        false
    }

    pub fn new(cfg: &'a ControlFlowGraph, function_ir: &'a FunctionIR) -> Self {
        let entry_node = Self::choose_entry_node(cfg).unwrap_or_else(|| NodeIndex::new(0));
        let dom = simple_fast(cfg, entry_node);

        let mut addr_to_node_index = HashMap::new();
        for node_idx in cfg.node_indices() {
            let summary = cfg
                .node_weight(node_idx)
                .expect("CFG node should have a weight");
            addr_to_node_index.insert(summary.start, node_idx);
        }

        // Pre-compute block.id → index for O(1) lookup.
        let block_id_to_idx: HashMap<usize, usize> = function_ir
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();

        Structurizer {
            cfg,
            function_ir,
            dom,
            addr_to_node_index,
            block_id_to_idx,
        }
    }

    fn is_block_return(ir_block: &IRBlock) -> bool {
        if let Some(last_stmt) = ir_block.stmts.last() {
            if let RValue::Op { opcode, .. } = &last_stmt.value {
                // Only treat unconditional terminators as function returns.
                // Predicated EXIT/RET are conditional control-flow instructions.
                return last_stmt.pred.is_none()
                    && (opcode == "RET" || opcode == "EXIT" || opcode.starts_with("RET"));
            }
        }
        false
    }

    pub fn structure_function(&mut self) -> Option<StructuredStatement> {
        // New collapse-based structurizer. Iteratively rewrites a region graph
        // built from the CFG by pattern-matching collapse rules until one
        // region remains (or the goto fallback fires).
        let mut graph = collapse::RegionGraph::build_from_cfg(self);
        collapse::run_collapse(&mut graph, self);
        Some(collapse::emit_root(&graph, self))
    }

    fn get_ir_block_by_cfg_node(&self, cfg_node: NodeIndex) -> Option<&'a IRBlock> {
        let summary = self.cfg.node_weight(cfg_node)?;
        let &idx = self.block_id_to_idx.get(&summary.id)?;
        Some(&self.function_ir.blocks[idx])
    }

    /// O(1) IR block lookup by block id.
    fn get_ir_block(&self, block_id: usize) -> Option<&'a IRBlock> {
        let &idx = self.block_id_to_idx.get(&block_id)?;
        Some(&self.function_ir.blocks[idx])
    }

    fn extract_if_targets_and_condition(
        &self,
        ir_block: &'a IRBlock,
    ) -> Option<(IRExpr, NodeIndex, Option<NodeIndex>)> {
        if ir_block.irdst.is_empty() || ir_block.irdst.len() > 2 {
            return None;
        }

        if ir_block.irdst.len() == 2 {
            let (cond1_opt, addr1) = &ir_block.irdst[0];
            let (cond2_opt, addr2) = &ir_block.irdst[1];
            let node1 = self.addr_to_node_index.get(addr1).copied()?;
            let node2 = self.addr_to_node_index.get(addr2).copied()?;

            match (cond1_opt, cond2_opt) {
                (
                    Some(IRCond::Pred { reg: r1, sense: s1 }),
                    Some(IRCond::Pred { reg: r2, sense: s2 }),
                ) => {
                    if r1.class == r2.class && r1.idx == r2.idx && *s1 != *s2 {
                        let cond_expr = IRExpr::Reg(r1.clone());
                        return if *s1 {
                            Some((cond_expr, node1, Some(node2)))
                        } else {
                            Some((cond_expr, node2, Some(node1)))
                        };
                    }
                }
                (Some(IRCond::Pred { reg, sense }), Some(IRCond::True)) => {
                    let cond_expr_val = IRExpr::Reg(reg.clone());
                    return if *sense {
                        Some((cond_expr_val, node1, Some(node2)))
                    } else {
                        Some((negate_condition(cond_expr_val), node1, Some(node2)))
                    };
                }
                (Some(IRCond::True), Some(IRCond::Pred { reg, sense })) => {
                    let cond_expr_val = IRExpr::Reg(reg.clone());
                    return if *sense {
                        Some((cond_expr_val, node2, Some(node1)))
                    } else {
                        Some((negate_condition(cond_expr_val), node2, Some(node1)))
                    };
                }
                _ => {}
            }
        } else if ir_block.irdst.len() == 1 {
            let (cond_opt, addr) = &ir_block.irdst[0];
            if let Some(IRCond::Pred { reg, sense }) = cond_opt {
                let target_node = self.addr_to_node_index.get(addr).copied()?;
                let base_reg_expr = IRExpr::Reg(reg.clone());
                let cond_expr = if *sense {
                    base_reg_expr
                } else {
                    negate_condition(base_reg_expr)
                };
                return Some((cond_expr, target_node, None));
            }
        }
        None
    }

    fn is_convergence_barrier_opcode(opcode: &str) -> bool {
        // SM 100+ (Blackwell) annotates convergence barriers with reliability
        // hints: `BSSY.RECONVERGENT`, `BSYNC.RELIABLE`, etc.  Compare against
        // the base mnemonic so both the plain and annotated forms match.
        let base = opcode.split('.').next().unwrap_or(opcode);
        matches!(base, "BSSY" | "BSYNC" | "SSY" | "SYNC" | "WARPSYNC")
    }

    fn ast_sequence(stmts: Vec<AstStmt>) -> AstStmt {
        let mut flat = Vec::new();
        for stmt in stmts {
            match stmt {
                AstStmt::Empty => {}
                AstStmt::Sequence(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => AstStmt::Empty,
            1 => flat.into_iter().next().unwrap(),
            _ => AstStmt::Sequence(flat),
        }
    }

    fn ast_is_empty(stmt: &AstStmt) -> bool {
        match stmt {
            AstStmt::Empty => true,
            AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
                stmts.iter().all(Self::ast_is_empty)
            }
            _ => false,
        }
    }

    fn ast_expr_from_display(ctx: &dyn DisplayCtx, expr: &IRExpr) -> AstExpr {
        match expr {
            IRExpr::Reg(r) => {
                if matches!(r.class.as_str(), "RZ" | "URZ") {
                    AstExpr::Imm("0".to_string())
                } else if matches!(r.class.as_str(), "PT" | "UPT") {
                    AstExpr::Imm("true".to_string())
                } else {
                    AstExpr::Reg(ctx.expr(expr))
                }
            }
            IRExpr::ImmI(i) => AstExpr::Imm(i.to_string()),
            IRExpr::ImmF(f) => AstExpr::Imm(f.to_string()),
            IRExpr::Addr64 { lo, hi } => AstExpr::Addr64 {
                lo: Box::new(Self::ast_expr_from_display(ctx, lo)),
                hi: Box::new(Self::ast_expr_from_display(ctx, hi)),
            },
            IRExpr::Mem { base, offset, .. } => {
                if let Some(offset) = offset {
                    AstExpr::WidePtr {
                        base: Box::new(Self::ast_expr_from_display(ctx, base)),
                        offset: Box::new(Self::ast_expr_from_display(ctx, offset)),
                    }
                } else {
                    Self::ast_expr_from_display(ctx, base)
                }
            }
            IRExpr::Op { op, args } => {
                let args = args
                    .iter()
                    .map(|arg| Self::ast_expr_from_display(ctx, arg))
                    .collect::<Vec<_>>();
                match (op.as_str(), args.as_slice()) {
                    ("!", [arg]) => AstExpr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(arg.clone()),
                    },
                    ("-", [arg]) => AstExpr::Unary {
                        op: "-".to_string(),
                        arg: Box::new(arg.clone()),
                    },
                    (
                        "+"
                        | "-"
                        | "*"
                        | "/"
                        | "%"
                        | "&"
                        | "|"
                        | "^"
                        | "<<"
                        | ">>"
                        | "&&"
                        | "||"
                        | "=="
                        | "!="
                        | "<"
                        | "<="
                        | ">"
                        | ">=",
                        [lhs, rhs],
                    ) => AstExpr::Binary {
                        op: op.clone(),
                        lhs: Box::new(lhs.clone()),
                        rhs: Box::new(rhs.clone()),
                    },
                    ("ConstMem", _) => AstExpr::ConstMemSymbol(ctx.expr(expr)),
                    ("pair_hi", [value]) => AstExpr::Intrinsic {
                        op: IntrinsicOp::PairHi,
                        args: vec![value.clone()],
                    },
                    _ => AstExpr::CallLike {
                        func: op.clone(),
                        args,
                    },
                }
            }
        }
    }

    fn ast_lvalue_from_display(ctx: &dyn DisplayCtx, expr: &IRExpr) -> AstLValue {
        match expr {
            IRExpr::Reg(r) => {
                if matches!(r.class.as_str(), "RZ" | "URZ") {
                    AstLValue::Raw("0".to_string())
                } else if matches!(r.class.as_str(), "PT" | "UPT") {
                    AstLValue::Raw("true".to_string())
                } else {
                    AstLValue::Var(ctx.expr(expr))
                }
            }
            _ => AstLValue::Raw(Self::decompile_expr(ctx, expr)),
        }
    }

    fn ast_dest_from_defs(ctx: &dyn DisplayCtx, defs: &[IRExpr]) -> Option<AstLValue> {
        if defs.is_empty() {
            return None;
        }
        if defs.len() == 1 {
            return Some(Self::ast_lvalue_from_display(ctx, &defs[0]));
        }
        let rendered = defs
            .iter()
            .filter(|d| !Self::is_zero_or_true_reg(d))
            .map(|d| ctx.expr(d))
            .collect::<Vec<_>>();
        if rendered.is_empty() {
            None
        } else {
            Some(AstLValue::Raw(format!("({})", rendered.join(", "))))
        }
    }

    fn ast_value_from_rvalue(ctx: &dyn DisplayCtx, value: &RValue) -> AstExpr {
        match value {
            RValue::Op { opcode, args } => {
                let args_s = args
                    .iter()
                    .map(|a| Self::decompile_expr(ctx, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                AstExpr::Raw(format!("{}({})", opcode, args_s))
            }
            RValue::Phi(args) => {
                let args_s = args
                    .iter()
                    .map(|a| Self::decompile_expr(ctx, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                AstExpr::Raw(format!("phi({})", args_s))
            }
            RValue::ImmI(i) => AstExpr::Imm(i.to_string()),
            RValue::ImmF(f) => AstExpr::Imm(f.to_string()),
        }
    }

    fn ast_is_trivial_self_assign(stmt: &AstStmt) -> bool {
        match stmt {
            AstStmt::Assign {
                dst: AstLValue::Var(dst),
                src: AstExpr::Reg(src),
            }
            | AstStmt::Assign {
                dst: AstLValue::Var(dst),
                src: AstExpr::Raw(src),
            }
            | AstStmt::Assign {
                dst: AstLValue::Raw(dst),
                src: AstExpr::Reg(src),
            }
            | AstStmt::Assign {
                dst: AstLValue::Raw(dst),
                src: AstExpr::Raw(src),
            } => dst == src,
            _ => false,
        }
    }

    fn ast_guarded(condition: Option<AstExpr>, stmt: AstStmt) -> AstStmt {
        if matches!(stmt, AstStmt::Empty) || Self::ast_is_trivial_self_assign(&stmt) {
            return AstStmt::Empty;
        }
        match condition {
            Some(condition) => AstStmt::If {
                condition,
                then_branch: Box::new(stmt),
                else_branch: None,
            },
            None => stmt,
        }
    }

    fn lower_lifted_stmt_ast(lifted_stmt: &crate::semantic_lift::LiftedStmt) -> AstStmt {
        if lifted_stmt.dest.is_sink_literal() {
            return Self::ast_guarded(
                lifted_stmt.pred.clone(),
                AstStmt::ExprStmt(lifted_stmt.rhs.clone()),
            );
        }
        match (&lifted_stmt.pred, &lifted_stmt.pred_old_val) {
            (Some(pred), Some(old)) => AstStmt::Assign {
                dst: lifted_stmt.dest.clone(),
                src: AstExpr::Ternary {
                    cond: Box::new(pred.clone()),
                    then_expr: Box::new(lifted_stmt.rhs.clone()),
                    else_expr: Box::new(old.clone()),
                },
            },
            (Some(pred), None) => Self::ast_guarded(
                Some(pred.clone()),
                AstStmt::Assign {
                    dst: lifted_stmt.dest.clone(),
                    src: lifted_stmt.rhs.clone(),
                },
            ),
            (None, _) => AstStmt::Assign {
                dst: lifted_stmt.dest.clone(),
                src: lifted_stmt.rhs.clone(),
            },
        }
    }

    fn resolve_stmt_lookup_idx(
        fir_block: Option<&IRBlock>,
        fir_search_from: &mut usize,
        fallback_idx: usize,
        stmt: &IRStatement,
    ) -> usize {
        let Some(orig_block) = fir_block else {
            return fallback_idx;
        };
        if let Some((found_idx, _)) = orig_block
            .stmts
            .iter()
            .enumerate()
            .skip(*fir_search_from)
            .find(|(_, s)| s.defs == stmt.defs && s.value == stmt.value && s.pred == stmt.pred)
        {
            *fir_search_from = found_idx + 1;
            found_idx
        } else {
            fallback_idx
        }
    }

    fn lower_non_control_stmt_ast(
        &self,
        block_id: usize,
        stmt_idx: usize,
        ir_s: &IRStatement,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> Vec<AstStmt> {
        if let Some(res) = lifted {
            let mut lowered = Vec::new();
            for def_idx in Self::lifted_def_emit_order(ir_s) {
                let def_ref = DefRef {
                    block_id,
                    stmt_idx,
                    def_idx,
                };
                let Some(lifted_stmt) = res.by_def.get(&def_ref) else {
                    continue;
                };
                lowered.push(Self::lower_lifted_stmt_ast(lifted_stmt));
            }
            if !lowered.is_empty() {
                return lowered;
            }
        }

        let side_effect_only =
            ir_s.defs.is_empty() || ir_s.defs.iter().all(Self::is_zero_or_true_reg);
        let value_expr = Self::ast_value_from_rvalue(ctx, &ir_s.value);
        let pred_expr = ir_s.pred.as_ref().map(|pred| {
            self.resolve_stmt_predicate_expr_ast(block_id, stmt_idx, pred, ctx, lifted)
        });

        if side_effect_only {
            return vec![Self::ast_guarded(pred_expr, AstStmt::ExprStmt(value_expr))];
        }

        let dest = Self::ast_dest_from_defs(ctx, &ir_s.defs)
            .unwrap_or_else(|| AstLValue::Raw("_".to_string()));

        if ir_s.pred.is_some() && !ir_s.pred_old_defs.is_empty() && ir_s.defs.len() == 1 {
            let pred = pred_expr.unwrap();
            let old = Self::ast_expr_from_display(ctx, &ir_s.pred_old_defs[0]);
            return vec![AstStmt::Assign {
                dst: dest,
                src: AstExpr::Ternary {
                    cond: Box::new(pred),
                    then_expr: Box::new(value_expr),
                    else_expr: Box::new(old),
                },
            }];
        }

        vec![Self::ast_guarded(
            pred_expr,
            AstStmt::Assign {
                dst: dest,
                src: value_expr,
            },
        )]
    }

    fn lower_stmt_list_ast(
        &self,
        block_id: usize,
        stmts: &[IRStatement],
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
        emit_returns: bool,
    ) -> AstStmt {
        let fir_block = self.get_ir_block(block_id);
        let control_pred_regs: Vec<RegId> = fir_block
            .map(|b| {
                b.irdst
                    .iter()
                    .filter_map(|(cond, _)| match cond {
                        Some(IRCond::Pred { reg, .. }) => Some(reg.clone()),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut fir_search_from = 0usize;
        let mut out = Vec::new();

        for (stmt_idx, ir_s) in stmts.iter().enumerate() {
            let lookup_stmt_idx =
                Self::resolve_stmt_lookup_idx(fir_block, &mut fir_search_from, stmt_idx, ir_s);

            if matches!(ir_s.value, RValue::Phi(_)) {
                continue;
            }

            if let Some(orig_block) = fir_block {
                if Self::should_omit_control_predicate_def(
                    ir_s,
                    lookup_stmt_idx,
                    orig_block,
                    &control_pred_regs,
                ) {
                    continue;
                }
            }

            if let RValue::Op { opcode, .. } = &ir_s.value {
                if Self::is_branch_only_opcode(opcode)
                    || Self::is_convergence_barrier_opcode(opcode)
                    || opcode == "NOP"
                {
                    continue;
                }
                if Self::is_barrier_opcode(opcode) {
                    out.push(Self::ast_guarded(
                        ir_s.pred.as_ref().map(|pred| {
                            self.resolve_stmt_predicate_expr_ast(
                                block_id,
                                lookup_stmt_idx,
                                pred,
                                ctx,
                                lifted,
                            )
                        }),
                        AstStmt::ExprStmt(AstExpr::CallLike {
                            func: "__syncthreads".to_string(),
                            args: vec![],
                        }),
                    ));
                    continue;
                }
                if Self::is_return_opcode(opcode) {
                    if !emit_returns {
                        continue;
                    }
                    let guarded = Self::ast_guarded(
                        ir_s.pred.as_ref().map(|pred| {
                            self.resolve_stmt_predicate_expr_ast(
                                block_id,
                                lookup_stmt_idx,
                                pred,
                                ctx,
                                lifted,
                            )
                        }),
                        AstStmt::Return(None),
                    );
                    let terminate = ir_s.pred.is_none();
                    out.push(guarded);
                    if terminate {
                        break;
                    }
                    continue;
                }
            }

            if !ir_s.defs.is_empty()
                && ir_s.defs.iter().all(Self::is_zero_or_true_reg)
                && !Self::is_memory_side_effect_opcode(&ir_s.value)
            {
                continue;
            }

            out.extend(self.lower_non_control_stmt_ast(
                block_id,
                lookup_stmt_idx,
                ir_s,
                ctx,
                lifted,
            ));
        }

        Self::ast_sequence(out)
    }

    fn cfg_node_for_block_id(&self, block_id: usize) -> Option<NodeIndex> {
        self.cfg
            .node_indices()
            .find(|&node| self.cfg[node].id == block_id)
    }

    fn collect_structured_block_ids(stmt: &StructuredStatement, blocks: &mut HashSet<usize>) {
        match stmt {
            StructuredStatement::BasicBlock { block_id, .. } => {
                blocks.insert(*block_id);
            }
            StructuredStatement::Sequence(stmts) => {
                for stmt in stmts {
                    Self::collect_structured_block_ids(stmt, blocks);
                }
            }
            StructuredStatement::If {
                condition_block_id,
                then_branch,
                else_branch,
                ..
            } => {
                blocks.insert(*condition_block_id);
                Self::collect_structured_block_ids(then_branch, blocks);
                if let Some(else_branch) = else_branch {
                    Self::collect_structured_block_ids(else_branch, blocks);
                }
            }
            StructuredStatement::Loop {
                header_block_id,
                body,
                ..
            } => {
                if let Some(header_block_id) = header_block_id {
                    blocks.insert(*header_block_id);
                }
                Self::collect_structured_block_ids(body, blocks);
            }
            StructuredStatement::Switch {
                header_block_id,
                cases,
                default,
                ..
            } => {
                blocks.insert(*header_block_id);
                for (_, body) in cases {
                    Self::collect_structured_block_ids(body, blocks);
                }
                if let Some(default) = default {
                    Self::collect_structured_block_ids(default, blocks);
                }
            }
            StructuredStatement::UnstructuredJump { from_block_id, .. } => {
                blocks.insert(*from_block_id);
            }
            StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::Empty => {}
        }
    }

    fn collect_jump_targets(stmt: &StructuredStatement, targets: &mut HashSet<usize>) {
        match stmt {
            StructuredStatement::Sequence(stmts) => {
                for stmt in stmts {
                    Self::collect_jump_targets(stmt, targets);
                }
            }
            StructuredStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                Self::collect_jump_targets(then_branch, targets);
                if let Some(else_branch) = else_branch {
                    Self::collect_jump_targets(else_branch, targets);
                }
            }
            StructuredStatement::Loop { body, .. } => {
                Self::collect_jump_targets(body, targets);
            }
            StructuredStatement::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    Self::collect_jump_targets(body, targets);
                }
                if let Some(default) = default {
                    Self::collect_jump_targets(default, targets);
                }
            }
            StructuredStatement::UnstructuredJump { to_block_id, .. } => {
                targets.insert(*to_block_id);
            }
            StructuredStatement::BasicBlock { .. }
            | StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::Empty => {}
        }
    }

    fn lower_loop_phi_prelude_ast(
        &self,
        header_block_id: usize,
        body: &StructuredStatement,
        ctx: &dyn DisplayCtx,
    ) -> AstStmt {
        self.lower_phi_assignments_for_loop_preds_ast(
            header_block_id,
            body,
            ctx,
            |preds, loop_blocks| {
                let external = preds
                    .iter()
                    .copied()
                    .filter(|pred| !loop_blocks.contains(&self.cfg[*pred].id))
                    .collect::<Vec<_>>();
                if external.len() == 1 {
                    Some(external)
                } else {
                    None
                }
            },
            |_| true,
        )
    }

    fn lower_loop_phi_backedge_updates_ast(
        &self,
        header_block_id: usize,
        body: &StructuredStatement,
        ctx: &dyn DisplayCtx,
    ) -> AstStmt {
        let relevant_defs = self.collect_loop_backedge_phi_relevant_defs(body);
        self.lower_phi_assignments_for_loop_preds_ast(
            header_block_id,
            body,
            ctx,
            |preds, loop_blocks| {
                let internal = preds
                    .iter()
                    .copied()
                    .filter(|pred| loop_blocks.contains(&self.cfg[*pred].id))
                    .collect::<Vec<_>>();
                if internal.is_empty() {
                    None
                } else {
                    Some(internal)
                }
            },
            |def| relevant_defs.is_empty() || Self::phi_def_is_relevant(def, &relevant_defs),
        )
    }

    fn lower_phi_assignments_for_loop_preds_ast<F, G>(
        &self,
        header_block_id: usize,
        body: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        pick_preds: F,
        keep_def: G,
    ) -> AstStmt
    where
        F: Fn(&[NodeIndex], &HashSet<usize>) -> Option<Vec<NodeIndex>>,
        G: Fn(&IRExpr) -> bool,
    {
        let Some(header_node) = self.cfg_node_for_block_id(header_block_id) else {
            return AstStmt::Empty;
        };
        let Some(header_block) = self.get_ir_block(header_block_id) else {
            return AstStmt::Empty;
        };

        let mut loop_blocks = HashSet::new();
        loop_blocks.insert(header_block_id);
        Self::collect_structured_block_ids(body, &mut loop_blocks);

        let preds: Vec<_> = self
            .cfg
            .neighbors_directed(header_node, Direction::Incoming)
            .collect();
        let Some(selected_preds) = pick_preds(&preds, &loop_blocks) else {
            return AstStmt::Empty;
        };

        let mut out = Vec::new();
        for stmt in header_block
            .stmts
            .iter()
            .filter(|stmt| matches!(stmt.value, RValue::Phi(_)))
        {
            let Some(def) = stmt.defs.first() else {
                continue;
            };
            if !keep_def(def) {
                continue;
            }
            let Some(dst) = Self::ast_dest_from_defs(ctx, &stmt.defs) else {
                continue;
            };
            if dst.is_sink_literal() {
                continue;
            }
            let RValue::Phi(args) = &stmt.value else {
                continue;
            };
            let Some(src_expr) = self.resolve_phi_source_for_preds(args, &preds, &selected_preds)
            else {
                continue;
            };
            let assign = AstStmt::Assign {
                dst,
                src: Self::ast_expr_from_display(ctx, &src_expr),
            };
            if std::env::var("DEBUG_LOOP_PHI").is_ok() {
                eprintln!(
                    "loop_phi header={} def={:?} src={:?}",
                    header_block_id, stmt.defs, src_expr
                );
            }
            if !Self::ast_is_trivial_self_assign(&assign) {
                out.push(assign);
            }
        }

        Self::ast_sequence(out)
    }

    fn resolve_phi_source_for_preds(
        &self,
        args: &[IRExpr],
        preds: &[NodeIndex],
        selected_preds: &[NodeIndex],
    ) -> Option<IRExpr> {
        let mut chosen: Option<IRExpr> = None;
        for pred in selected_preds {
            let pred_idx = preds.iter().position(|candidate| candidate == pred)?;
            let arg = args.get(pred_idx)?.clone();
            match &chosen {
                None => chosen = Some(arg),
                Some(existing) if existing == &arg => {}
                Some(_) => return None,
            }
        }
        chosen
    }

    fn lower_block_phi_assignments_from_pred_ast(
        &self,
        target_block_id: usize,
        from_block_id: usize,
        ctx: &dyn DisplayCtx,
    ) -> AstStmt {
        let Some(target_node) = self.cfg_node_for_block_id(target_block_id) else {
            return AstStmt::Empty;
        };
        let Some(target_block) = self.get_ir_block(target_block_id) else {
            return AstStmt::Empty;
        };
        let preds: Vec<_> = self
            .cfg
            .neighbors_directed(target_node, Direction::Incoming)
            .collect();
        let Some(selected_pred) = preds
            .iter()
            .copied()
            .find(|pred| self.cfg[*pred].id == from_block_id)
        else {
            return AstStmt::Empty;
        };

        let mut out = Vec::new();
        for stmt in target_block
            .stmts
            .iter()
            .filter(|stmt| matches!(stmt.value, RValue::Phi(_)))
        {
            let Some(dst) = Self::ast_dest_from_defs(ctx, &stmt.defs) else {
                continue;
            };
            if dst.is_sink_literal() {
                continue;
            }
            let RValue::Phi(args) = &stmt.value else {
                continue;
            };
            let Some(src_expr) = self.resolve_phi_source_for_preds(args, &preds, &[selected_pred])
            else {
                continue;
            };
            let assign = AstStmt::Assign {
                dst,
                src: Self::ast_expr_from_display(ctx, &src_expr),
            };
            if !Self::ast_is_trivial_self_assign(&assign) {
                out.push(assign);
            }
        }
        Self::ast_sequence(out)
    }

    fn lower_phi_connector_chain_from_pred_ast(
        &self,
        target_block_id: usize,
        from_block_id: usize,
        ctx: &dyn DisplayCtx,
    ) -> AstStmt {
        let mut out = Vec::new();
        let mut current_from = from_block_id;
        let mut seen = HashSet::new();

        for _ in 0..16 {
            if current_from == target_block_id {
                break;
            }

            if self.is_direct_cfg_successor(current_from, target_block_id) {
                let direct =
                    self.lower_block_phi_assignments_from_pred_ast(target_block_id, current_from, ctx);
                if !Self::ast_is_empty(&direct) {
                    out.push(direct);
                }
                break;
            }

            let connectors = self
                .cfg_successor_block_ids(current_from)
                .into_iter()
                .filter(|succ| self.block_is_phi_only(*succ))
                .filter(|succ| self.cfg_path_exists(*succ, target_block_id))
                .collect::<Vec<_>>();
            if connectors.len() != 1 {
                break;
            }

            let connector = connectors[0];
            if !seen.insert(connector) {
                break;
            }

            let phi_assigns =
                self.lower_block_phi_assignments_from_pred_ast(connector, current_from, ctx);
            if !Self::ast_is_empty(&phi_assigns) {
                out.push(phi_assigns);
            }
            current_from = connector;
        }

        Self::ast_sequence(out)
    }

    fn cfg_successor_block_ids(&self, block_id: usize) -> Vec<usize> {
        let Some(node) = self.cfg_node_for_block_id(block_id) else {
            return Vec::new();
        };
        self.cfg
            .neighbors_directed(node, Direction::Outgoing)
            .map(|succ| self.cfg[succ].id)
            .collect()
    }

    fn is_direct_cfg_successor(&self, from_block_id: usize, to_block_id: usize) -> bool {
        self.cfg_successor_block_ids(from_block_id)
            .into_iter()
            .any(|succ| succ == to_block_id)
    }

    fn block_is_phi_only(&self, block_id: usize) -> bool {
        let Some(block) = self.get_ir_block(block_id) else {
            return false;
        };
        !block.stmts.is_empty() && block.stmts.iter().all(|stmt| matches!(stmt.value, RValue::Phi(_)))
    }

    fn cfg_path_exists(&self, start_block_id: usize, goal_block_id: usize) -> bool {
        let Some(start) = self.cfg_node_for_block_id(start_block_id) else {
            return false;
        };
        let Some(goal) = self.cfg_node_for_block_id(goal_block_id) else {
            return false;
        };
        if start == goal {
            return true;
        }
        let mut seen = HashSet::new();
        let mut work = VecDeque::new();
        seen.insert(start);
        work.push_back(start);
        while let Some(node) = work.pop_front() {
            for succ in self.cfg.neighbors_directed(node, Direction::Outgoing) {
                if succ == goal {
                    return true;
                }
                if seen.insert(succ) {
                    work.push_back(succ);
                }
            }
        }
        false
    }

    fn phi_def_is_relevant(def: &IRExpr, relevant_defs: &HashSet<(String, i32)>) -> bool {
        let Some(reg) = def.get_reg() else {
            return true;
        };
        relevant_defs.contains(&(reg.class.clone(), reg.idx))
    }

    fn collect_loop_backedge_phi_relevant_defs(
        &self,
        body: &StructuredStatement,
    ) -> HashSet<(String, i32)> {
        let mut block_ids = HashSet::new();
        Self::collect_structured_block_ids(body, &mut block_ids);
        let mut relevant = HashSet::new();
        for block_id in block_ids {
            let Some(block) = self.get_ir_block(block_id) else {
                continue;
            };
            for stmt in &block.stmts {
                if let Some(mem_args) = &stmt.mem_addr_args {
                    for expr in mem_args {
                        Self::collect_expr_reg_bases(expr, &mut relevant);
                    }
                }
                if let RValue::Op { args, .. } = &stmt.value {
                    for def_reg in stmt.defs.iter().filter_map(IRExpr::get_reg) {
                        if args
                            .iter()
                            .any(|arg| Self::expr_mentions_reg_base(arg, def_reg))
                        {
                            relevant.insert((def_reg.class.clone(), def_reg.idx));
                        }
                    }
                }
            }
            for (cond, _) in &block.irdst {
                if let Some(IRCond::Pred { reg, .. }) = cond {
                    relevant.insert((reg.class.clone(), reg.idx));
                }
            }
        }
        relevant
    }

    fn expr_mentions_reg_base(expr: &IRExpr, target: &RegId) -> bool {
        match expr {
            IRExpr::Reg(reg) => reg.class == target.class && reg.idx == target.idx,
            IRExpr::Addr64 { lo, hi } => {
                Self::expr_mentions_reg_base(lo, target) || Self::expr_mentions_reg_base(hi, target)
            }
            IRExpr::Mem { base, offset, .. } => {
                Self::expr_mentions_reg_base(base, target)
                    || offset
                        .as_ref()
                        .is_some_and(|offset| Self::expr_mentions_reg_base(offset, target))
            }
            IRExpr::Op { args, .. } => args
                .iter()
                .any(|arg| Self::expr_mentions_reg_base(arg, target)),
            IRExpr::ImmI(_) | IRExpr::ImmF(_) => false,
        }
    }

    fn collect_expr_reg_bases(expr: &IRExpr, out: &mut HashSet<(String, i32)>) {
        match expr {
            IRExpr::Reg(reg) => {
                out.insert((reg.class.clone(), reg.idx));
            }
            IRExpr::Addr64 { lo, hi } => {
                Self::collect_expr_reg_bases(lo, out);
                Self::collect_expr_reg_bases(hi, out);
            }
            IRExpr::Mem { base, offset, .. } => {
                Self::collect_expr_reg_bases(base, out);
                if let Some(offset) = offset {
                    Self::collect_expr_reg_bases(offset, out);
                }
            }
            IRExpr::Op { args, .. } => {
                for arg in args {
                    Self::collect_expr_reg_bases(arg, out);
                }
            }
            IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
        }
    }

    fn mark_loop_phi_updates(stmt: AstStmt) -> AstStmt {
        match stmt {
            AstStmt::Assign { dst, src } => AstStmt::Assign {
                dst,
                src: AstExpr::CallLike {
                    func: "__loop_phi".to_string(),
                    args: vec![src],
                },
            },
            AstStmt::Sequence(stmts) => {
                Self::ast_sequence(stmts.into_iter().map(Self::mark_loop_phi_updates).collect())
            }
            AstStmt::Block(stmts) => {
                AstStmt::Block(stmts.into_iter().map(Self::mark_loop_phi_updates).collect())
            }
            other => other,
        }
    }

    fn inject_loop_phi_backedge_updates(body: AstStmt, updates: &AstStmt) -> AstStmt {
        // Large real kernels can legitimately need more than a dozen carried
        // phi updates to keep pointer and induction state coherent. Dropping the
        // entire set freezes loop-carried addresses at their entry seed.
        if Self::ast_is_empty(updates)
            || Self::ast_contains_continue(&body)
            || Self::ast_stmt_count(updates) > 64
        {
            return body;
        }
        let updates = Self::mark_loop_phi_updates(updates.clone());
        let body = Self::inject_loop_phi_updates_before_continue(body, &updates);
        if Self::ast_may_fallthrough(&body) {
            Self::ast_sequence(vec![body, updates])
        } else {
            body
        }
    }

    fn ast_contains_continue(stmt: &AstStmt) -> bool {
        match stmt {
            AstStmt::Continue => true,
            AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
                stmts.iter().any(Self::ast_contains_continue)
            }
            AstStmt::Label { body, .. } => Self::ast_contains_continue(body),
            AstStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                Self::ast_contains_continue(then_branch)
                    || else_branch
                        .as_deref()
                        .map(Self::ast_contains_continue)
                        .unwrap_or(false)
            }
            AstStmt::Switch { cases, default, .. } => {
                cases
                    .iter()
                    .any(|(_, body)| Self::ast_contains_continue(body))
                    || default
                        .as_deref()
                        .map(Self::ast_contains_continue)
                        .unwrap_or(false)
            }
            AstStmt::Loop { .. }
            | AstStmt::Break
            | AstStmt::Return(_)
            | AstStmt::Assign { .. }
            | AstStmt::ExprStmt(_)
            | AstStmt::Goto(_)
            | AstStmt::Empty => false,
        }
    }

    fn ast_stmt_count(stmt: &AstStmt) -> usize {
        match stmt {
            AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
                stmts.iter().map(Self::ast_stmt_count).sum()
            }
            AstStmt::Label { body, .. } => Self::ast_stmt_count(body),
            AstStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                1 + Self::ast_stmt_count(then_branch)
                    + else_branch
                        .as_deref()
                        .map(Self::ast_stmt_count)
                        .unwrap_or(0)
            }
            AstStmt::Switch { cases, default, .. } => {
                1 + cases
                    .iter()
                    .map(|(_, body)| Self::ast_stmt_count(body))
                    .sum::<usize>()
                    + default.as_deref().map(Self::ast_stmt_count).unwrap_or(0)
            }
            AstStmt::Loop { body, .. } => 1 + Self::ast_stmt_count(body),
            AstStmt::Empty => 0,
            _ => 1,
        }
    }

    fn inject_loop_phi_updates_before_continue(stmt: AstStmt, updates: &AstStmt) -> AstStmt {
        match stmt {
            AstStmt::Continue => Self::ast_sequence(vec![updates.clone(), AstStmt::Continue]),
            AstStmt::Block(stmts) => AstStmt::Block(
                stmts
                    .into_iter()
                    .map(|stmt| Self::inject_loop_phi_updates_before_continue(stmt, updates))
                    .collect(),
            ),
            AstStmt::Sequence(stmts) => Self::ast_sequence(
                stmts
                    .into_iter()
                    .map(|stmt| Self::inject_loop_phi_updates_before_continue(stmt, updates))
                    .collect(),
            ),
            AstStmt::Label { name, body } => AstStmt::Label {
                name,
                body: Box::new(Self::inject_loop_phi_updates_before_continue(*body, updates)),
            },
            AstStmt::If {
                condition,
                then_branch,
                else_branch,
            } => AstStmt::If {
                condition,
                then_branch: Box::new(Self::inject_loop_phi_updates_before_continue(
                    *then_branch,
                    updates,
                )),
                else_branch: else_branch.map(|branch| {
                    Box::new(Self::inject_loop_phi_updates_before_continue(
                        *branch, updates,
                    ))
                }),
            },
            AstStmt::Switch {
                discriminant,
                cases,
                default,
            } => AstStmt::Switch {
                discriminant,
                cases: cases
                    .into_iter()
                    .map(|(label, body)| {
                        (
                            label,
                            Self::inject_loop_phi_updates_before_continue(body, updates),
                        )
                    })
                    .collect(),
                default: default.map(|body| {
                    Box::new(Self::inject_loop_phi_updates_before_continue(
                        *body, updates,
                    ))
                }),
            },
            AstStmt::Loop { .. } => stmt,
            other => other,
        }
    }

    fn ast_may_fallthrough(stmt: &AstStmt) -> bool {
        match stmt {
            AstStmt::Empty
            | AstStmt::Assign { .. }
            | AstStmt::ExprStmt(_)
            | AstStmt::Loop { .. } => true,
            AstStmt::Break | AstStmt::Continue | AstStmt::Return(_) | AstStmt::Goto(_) => false,
            AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
                let mut reachable = true;
                for stmt in stmts {
                    if !reachable {
                        return false;
                    }
                    reachable = Self::ast_may_fallthrough(stmt);
                }
                reachable
            }
            AstStmt::Label { body, .. } => Self::ast_may_fallthrough(body),
            AstStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                let then_fallthrough = Self::ast_may_fallthrough(then_branch);
                match else_branch {
                    Some(branch) => then_fallthrough || Self::ast_may_fallthrough(branch),
                    None => true,
                }
            }
            AstStmt::Switch { cases, default, .. } => {
                if default.is_none() {
                    return true;
                }
                cases
                    .iter()
                    .any(|(_, body)| Self::ast_may_fallthrough(body))
                    || default
                        .as_deref()
                        .map(Self::ast_may_fallthrough)
                        .unwrap_or(false)
            }
        }
    }

    fn lower_condition_prelude_ast(
        &self,
        block_id: usize,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstStmt {
        let Some(block) = self.get_ir_block(block_id) else {
            return AstStmt::Empty;
        };
        match self.lower_stmt_list_ast(block_id, &block.stmts, ctx, lifted, false) {
            AstStmt::Empty => AstStmt::Empty,
            AstStmt::Sequence(stmts) => AstStmt::Block(stmts),
            stmt => AstStmt::Block(vec![stmt]),
        }
    }

    fn resolve_predicate_condition_expr_ast(
        &self,
        block_id: usize,
        max_stmt_idx: Option<usize>,
        pred: &RegId,
        lifted: Option<&SemanticLiftResult>,
    ) -> Option<AstExpr> {
        let lifted = lifted?;
        let block = self.get_ir_block(block_id)?;
        let end = max_stmt_idx
            .map(|idx| idx.saturating_add(1).min(block.stmts.len()))
            .unwrap_or(block.stmts.len());
        for (stmt_idx, stmt) in block.stmts[..end].iter().enumerate().rev() {
            for (def_idx, def) in stmt.defs.iter().enumerate() {
                let IRExpr::Reg(dst) = def else {
                    continue;
                };
                if dst.class != pred.class || dst.idx != pred.idx {
                    continue;
                }
                if let Some(want_ssa) = pred.ssa {
                    if dst.ssa != Some(want_ssa) {
                        continue;
                    }
                }
                let def_ref = DefRef {
                    block_id: block.id,
                    stmt_idx,
                    def_idx,
                };
                if let Some(ls) = lifted.by_def.get(&def_ref) {
                    return Some(ls.rhs.clone());
                }
            }
        }

        let want_ssa = pred.ssa?;
        let mut resolved = None;
        for other_block in &self.function_ir.blocks {
            for (stmt_idx, stmt) in other_block.stmts.iter().enumerate() {
                for (def_idx, def) in stmt.defs.iter().enumerate() {
                    let IRExpr::Reg(dst) = def else {
                        continue;
                    };
                    if dst.class != pred.class || dst.idx != pred.idx || dst.ssa != Some(want_ssa) {
                        continue;
                    }
                    let def_ref = DefRef {
                        block_id: other_block.id,
                        stmt_idx,
                        def_idx,
                    };
                    let Some(ls) = lifted.by_def.get(&def_ref) else {
                        continue;
                    };
                    if let Some(existing) = &resolved {
                        if existing != &ls.rhs {
                            return None;
                        }
                    } else {
                        resolved = Some(ls.rhs.clone());
                    }
                }
            }
        }
        resolved
    }

    fn resolve_stmt_predicate_expr_ast(
        &self,
        block_id: usize,
        stmt_idx: usize,
        pred_expr: &IRExpr,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstExpr {
        match pred_expr {
            IRExpr::Op { op, args } if op == "!" && args.len() == 1 => AstExpr::Unary {
                op: "!".to_string(),
                arg: Box::new(
                    self.resolve_stmt_predicate_expr_ast(block_id, stmt_idx, &args[0], ctx, lifted),
                ),
            },
            IRExpr::Reg(pred) => self
                .resolve_predicate_condition_expr_ast(block_id, Some(stmt_idx), pred, lifted)
                .unwrap_or_else(|| Self::ast_expr_from_display(ctx, pred_expr)),
            _ => Self::ast_expr_from_display(ctx, pred_expr),
        }
    }

    fn lower_condition_expr_ast(
        &self,
        condition_block_id: usize,
        condition_expr: &IRExpr,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstExpr {
        match condition_expr {
            IRExpr::Op { op, args } if op == "!" && args.len() == 1 => AstExpr::Unary {
                op: "!".to_string(),
                arg: Box::new(self.lower_condition_expr_ast(
                    condition_block_id,
                    &args[0],
                    ctx,
                    lifted,
                )),
            },
            IRExpr::Reg(pred) => self
                .resolve_predicate_condition_expr_ast(condition_block_id, None, pred, lifted)
                .unwrap_or_else(|| Self::ast_expr_from_display(ctx, condition_expr)),
            _ => Self::ast_expr_from_display(ctx, condition_expr),
        }
    }

    fn lower_structured_stmt_ast(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstStmt {
        let mut jump_targets = HashSet::new();
        Self::collect_jump_targets(stmt, &mut jump_targets);
        self.lower_structured_stmt_ast_with_targets(stmt, ctx, lifted, &jump_targets, None)
    }

    fn entry_block_id(stmt: &StructuredStatement) -> Option<usize> {
        match stmt {
            StructuredStatement::BasicBlock { block_id, .. } => Some(*block_id),
            StructuredStatement::Sequence(stmts) => {
                stmts.iter().find_map(Self::entry_block_id)
            }
            StructuredStatement::If {
                condition_block_id, ..
            } => Some(*condition_block_id),
            StructuredStatement::Loop {
                loop_type,
                header_block_id,
                body,
                ..
            } => match loop_type {
                // Do-while regions execute the body before the trailing
                // condition/header block, so predecessor phi materialization
                // must target the first body block.
                LoopType::DoWhile => Self::entry_block_id(body).or(*header_block_id),
                LoopType::While | LoopType::Endless => {
                    header_block_id.or_else(|| Self::entry_block_id(body))
                }
            },
            StructuredStatement::Switch {
                header_block_id, ..
            } => Some(*header_block_id),
            StructuredStatement::Empty
            | StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::UnstructuredJump { .. } => None,
        }
    }

    fn lower_structured_stmt_ast_with_targets(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
        jump_targets: &HashSet<usize>,
        fallthrough_target_block: Option<usize>,
    ) -> AstStmt {
        match stmt {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                let body = self.lower_stmt_list_ast(*block_id, stmts, ctx, lifted, true);
                let body = if let Some(target_block_id) = fallthrough_target_block {
                    Self::ast_sequence(vec![
                        body,
                        self.lower_phi_connector_chain_from_pred_ast(
                            target_block_id,
                            *block_id,
                            ctx,
                        ),
                    ])
                } else {
                    body
                };
                if jump_targets.contains(block_id) {
                    AstStmt::Label {
                        name: format!("BB{}", block_id),
                        body: Box::new(body),
                    }
                } else {
                    body
                }
            }
            StructuredStatement::Sequence(stmts) => {
                let mut out = Vec::new();
                for (idx, child) in stmts.iter().enumerate() {
                    if matches!(child, StructuredStatement::Empty) {
                        continue;
                    }
                    let next_target = stmts[idx + 1..]
                        .iter()
                        .find_map(Self::entry_block_id)
                        .or(fallthrough_target_block);
                    out.push(self.lower_structured_stmt_ast_with_targets(
                        child,
                        ctx,
                        lifted,
                        jump_targets,
                        next_target,
                    ));
                    if Self::ends_with_unconditional_return(child) {
                        break;
                    }
                }
                Self::ast_sequence(out)
            }
            StructuredStatement::If {
                condition_block_id,
                condition_expr,
                then_branch,
                else_branch,
            } => {
                let prelude = self.lower_condition_prelude_ast(*condition_block_id, ctx, lifted);
                let else_ref = else_branch.as_deref();
                let then_ast = self.lower_structured_stmt_ast_with_targets(
                    then_branch,
                    ctx,
                    lifted,
                    jump_targets,
                    fallthrough_target_block,
                );
                let else_ast = else_ref
                    .map(|stmt| {
                        self.lower_structured_stmt_ast_with_targets(
                            stmt,
                            ctx,
                            lifted,
                            jump_targets,
                            fallthrough_target_block,
                        )
                    })
                    .or_else(|| {
                        fallthrough_target_block.map(|target_block_id| {
                            self.lower_phi_connector_chain_from_pred_ast(
                                target_block_id,
                                *condition_block_id,
                                ctx,
                            )
                        })
                    })
                    .filter(|stmt| !Self::ast_is_empty(stmt));
                let then_empty = Self::ast_is_empty(&then_ast);
                let else_empty = else_ast
                    .as_ref()
                    .map(Self::ast_is_empty)
                    .unwrap_or(true);
                let core = if then_empty && else_empty {
                    AstStmt::Empty
                } else if then_empty && !else_empty {
                    AstStmt::If {
                        condition: self.lower_condition_expr_ast(
                            *condition_block_id,
                            &negate_condition(condition_expr.clone()),
                            ctx,
                            lifted,
                        ),
                        then_branch: Box::new(else_ast.unwrap()),
                        else_branch: None,
                    }
                } else {
                    AstStmt::If {
                        condition: self.lower_condition_expr_ast(
                            *condition_block_id,
                            condition_expr,
                            ctx,
                            lifted,
                        ),
                        then_branch: Box::new(then_ast),
                        else_branch: else_ast.map(Box::new),
                    }
                };
                let lowered = Self::ast_sequence(vec![prelude, core]);
                if jump_targets.contains(condition_block_id) {
                    AstStmt::Label {
                        name: format!("BB{}", condition_block_id),
                        body: Box::new(lowered),
                    }
                } else {
                    lowered
                }
            }
            StructuredStatement::Loop {
                loop_type,
                header_block_id,
                condition_expr,
                body,
            } => {
                let printable_body = Self::without_redundant_loop_tail_continue(body);
                let phi_entry_block_id = match (loop_type, header_block_id) {
                    (LoopType::DoWhile, Some(header_block_id)) => {
                        let body_entry_block_id = Self::entry_block_id(&printable_body);
                        match body_entry_block_id {
                            Some(body_entry_block_id) if body_entry_block_id != *header_block_id => {
                                Some(body_entry_block_id)
                            }
                            _ => Some(*header_block_id),
                        }
                    }
                    (_, Some(header_block_id)) => Some(*header_block_id),
                    _ => None,
                };
                let phi_prelude = phi_entry_block_id
                    .map(|bid| self.lower_loop_phi_prelude_ast(bid, &printable_body, ctx))
                    .unwrap_or(AstStmt::Empty);
                let phi_backedge_updates = header_block_id
                    .map(|hid| self.lower_loop_phi_backedge_updates_ast(hid, &printable_body, ctx))
                    .unwrap_or(AstStmt::Empty);
                let body_entry_backedge_updates = match (loop_type, header_block_id) {
                    (LoopType::DoWhile, Some(header_block_id)) => {
                        let body_entry_block_id = Self::entry_block_id(&printable_body);
                        match body_entry_block_id {
                            Some(body_entry_block_id) if body_entry_block_id != *header_block_id => {
                                self.lower_phi_connector_chain_from_pred_ast(
                                    body_entry_block_id,
                                    *header_block_id,
                                    ctx,
                                )
                            }
                            _ => AstStmt::Empty,
                        }
                    }
                    _ => AstStmt::Empty,
                };
                let loop_backedge_updates =
                    Self::ast_sequence(vec![phi_backedge_updates, body_entry_backedge_updates]);
                let condition_prelude = if *loop_type != LoopType::DoWhile {
                    header_block_id
                        .map(|hid| self.lower_condition_prelude_ast(hid, ctx, lifted))
                        .unwrap_or(AstStmt::Empty)
                } else {
                    AstStmt::Empty
                };
                let loop_stmt = AstStmt::Loop {
                    kind: match loop_type {
                        LoopType::While => AstLoopKind::While,
                        LoopType::DoWhile => AstLoopKind::DoWhile,
                        LoopType::Endless => AstLoopKind::Endless,
                    },
                    condition: condition_expr.as_ref().map(|expr| {
                        if let Some(hid) = header_block_id {
                            self.lower_condition_expr_ast(*hid, expr, ctx, lifted)
                        } else {
                            Self::ast_expr_from_display(ctx, expr)
                        }
                    }),
                    body: Box::new(Self::inject_loop_phi_backedge_updates(
                        self.lower_structured_stmt_ast_with_targets(
                            &printable_body,
                            ctx,
                            lifted,
                            jump_targets,
                            None,
                        ),
                        &loop_backedge_updates,
                    )),
                };
                let loop_exit_phi = match (fallthrough_target_block, header_block_id, loop_type) {
                    (Some(target_block_id), Some(header_block_id), LoopType::While | LoopType::DoWhile) => {
                        self.lower_phi_connector_chain_from_pred_ast(
                            target_block_id,
                            *header_block_id,
                            ctx,
                        )
                    }
                    _ => AstStmt::Empty,
                };
                let lowered =
                    Self::ast_sequence(vec![phi_prelude, condition_prelude, loop_stmt, loop_exit_phi]);
                if let Some(header_block_id) = header_block_id {
                    if jump_targets.contains(header_block_id) {
                        AstStmt::Label {
                            name: format!("BB{}", header_block_id),
                            body: Box::new(lowered),
                        }
                    } else {
                        lowered
                    }
                } else {
                    lowered
                }
            }
            StructuredStatement::Break(_) => AstStmt::Break,
            StructuredStatement::Continue(_) => AstStmt::Continue,
            StructuredStatement::Return(expr_opt) => AstStmt::Return(
                expr_opt
                    .as_ref()
                    .map(|expr| Self::ast_expr_from_display(ctx, expr)),
            ),
            StructuredStatement::UnstructuredJump {
                from_block_id,
                to_block_id,
                condition,
            } => {
                let phi_prelude = self.lower_phi_connector_chain_from_pred_ast(
                    *to_block_id,
                    *from_block_id,
                    ctx,
                );
                let jump = Self::ast_sequence(vec![
                    phi_prelude,
                    AstStmt::Goto(format!("BB{}", to_block_id)),
                ]);
                Self::ast_guarded(
                    condition
                        .as_ref()
                        .map(|expr| Self::ast_expr_from_display(ctx, expr)),
                    jump,
                )
            }
            StructuredStatement::Switch {
                header_block_id,
                discriminant,
                cases,
                default,
            } => {
                let prelude = self.lower_condition_prelude_ast(*header_block_id, ctx, lifted);
                let switch_stmt = AstStmt::Switch {
                    discriminant: discriminant
                        .as_ref()
                        .map(|expr| Self::ast_expr_from_display(ctx, expr)),
                    cases: cases
                        .iter()
                        .map(|(case_idx, body)| {
                            (
                                *case_idx,
                                self.lower_structured_stmt_ast_with_targets(
                                    body,
                                    ctx,
                                    lifted,
                                    jump_targets,
                                    fallthrough_target_block,
                                ),
                            )
                        })
                        .collect(),
                    default: default.as_ref().map(|body| {
                        Box::new(self.lower_structured_stmt_ast_with_targets(
                            body,
                            ctx,
                            lifted,
                            jump_targets,
                            fallthrough_target_block,
                        ))
                    }),
                };
                let lowered = Self::ast_sequence(vec![prelude, switch_stmt]);
                if jump_targets.contains(header_block_id) {
                    AstStmt::Label {
                        name: format!("BB{}", header_block_id),
                        body: Box::new(lowered),
                    }
                } else {
                    lowered
                }
            }
            StructuredStatement::Empty => AstStmt::Empty,
        }
    }

    #[cfg(test)]
    pub(crate) fn pretty_print(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
    ) -> String {
        self.pretty_print_with_lift(stmt, ctx, indent_level, None)
    }

    #[cfg(test)]
    pub(crate) fn pretty_print_with_lift(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        ast_simplify(self.lower_structured_stmt_ast(stmt, ctx, lifted))
            .render_with_indent(indent_level)
    }

    pub fn pretty_print_with_lift_cleanup(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        let seeded_wide_addrs = build_seeded_wide_addr_maps(self.function_ir, lifted);
        ast_cleanup_with_seeded_wide_addrs(
            self.lower_structured_stmt_ast(stmt, ctx, lifted),
            &seeded_wide_addrs,
        )
        .render_with_indent(indent_level)
    }

    pub fn pretty_print_with_lift_cleanup_and_names(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
        token_map: &HashMap<String, String>,
        _enable_post_name_addr64_fold: bool,
    ) -> String {
        // Run the structural cleanup while the AST is still in SSA form.
        // Re-running addr64/DCE cleanup after token coalescing is unsafe:
        // name recovery intentionally maps many SSA defs onto the same source
        // variable, which destroys the single-def invariant those passes rely on
        // and can trigger recursive alias blowups on real corpus kernels.
        let seeded_wide_addrs = build_seeded_wide_addr_maps(self.function_ir, lifted);
        let cleaned = ast_cleanup_with_seeded_wide_addrs(
            self.lower_structured_stmt_ast(stmt, ctx, lifted),
            &seeded_wide_addrs,
        );
        let mut token_map = token_map.clone();
        split_unsafe_token_aliases(&cleaned, &mut token_map);
        // Name recovery intentionally coalesces multiple SSA values onto the same
        // recovered source variable. Re-running pointer-pair/addr64 folding after
        // that coalescing can silently change pointer offsets, so keep the final
        // named stage as a pure token remap.
        let named = ast_apply_token_map(cleaned, &token_map);
        // Keep the final named pass structural-only. Post-name DCE-style cleanup
        // can erase defs once multiple SSA values intentionally coalesce onto
        // one recovered name, but the final render still needs token remapping
        // for raw helper fragments that survive as text.
        let rendered = named.render_with_indent(indent_level);
        apply_token_map_to_rendered(&rendered, &token_map)
    }
}

fn negate_condition(expr: IRExpr) -> IRExpr {
    match expr {
        IRExpr::Op { op, mut args } if op == "!" && args.len() == 1 => args.remove(0),
        _ => IRExpr::Op {
            op: "!".to_string(),
            args: vec![expr],
        },
    }
}

// ---------------------------------------------------------------------------
// Ghidra-style collapse-based structurizer
// ---------------------------------------------------------------------------
//
// The region graph is built once from the CFG and then iteratively rewritten
// by pattern-matching collapse rules. Unlike petgraph, region indices never
// shift: regions are stored in `Vec<Option<...>>`-like storage and marked
// `Tombstone` when consumed. This keeps `RegionId` stable for the entire run
// and avoids the stale-NodeIndex bug that killed the old `src/region.rs`.
//
// Phase A covers: region construction, sequence collapse, a minimal goto
// fallback, and the root emitter.

mod collapse;

#[cfg(test)]
mod tests;
