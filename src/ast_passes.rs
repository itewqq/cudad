//! Structural cleanup passes over the typed AST.
//! These operate before final rendering so backend cleanup stops depending
//! exclusively on post-render text rewriting.

use crate::ast::{Expr, LValue, Stmt};
use std::collections::{BTreeSet, HashMap};

pub fn ast_cleanup(stmt: Stmt) -> Stmt {
    let mut current = stmt;
    loop {
        let next = ast_simplify(ast_predicate_cleanup(ast_dce(ast_addr64_fold(
            current.clone(),
        ))));
        if next == current {
            return next;
        }
        current = next;
    }
}

pub fn ast_simplify(stmt: Stmt) -> Stmt {
    simplify_stmt(stmt)
}

pub fn ast_dce(stmt: Stmt) -> Stmt {
    dce_stmt(stmt, BTreeSet::new()).0
}

pub fn ast_addr64_fold(stmt: Stmt) -> Stmt {
    addr64_fold_stmt(stmt, &Addr64Defs::default())
}

pub fn ast_apply_token_map(stmt: Stmt, token_map: &HashMap<String, String>) -> Stmt {
    rename_stmt_tokens(stmt, token_map)
}

fn rename_stmt_tokens(stmt: Stmt, token_map: &HashMap<String, String>) -> Stmt {
    match stmt {
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .into_iter()
                .map(|stmt| rename_stmt_tokens(stmt, token_map))
                .collect(),
        ),
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts
                .into_iter()
                .map(|stmt| rename_stmt_tokens(stmt, token_map))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: rename_expr_tokens(condition, token_map),
            then_branch: Box::new(rename_stmt_tokens(*then_branch, token_map)),
            else_branch: else_branch.map(|stmt| Box::new(rename_stmt_tokens(*stmt, token_map))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(|expr| rename_expr_tokens(expr, token_map)),
            body: Box::new(rename_stmt_tokens(*body, token_map)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(|expr| rename_expr_tokens(expr, token_map)),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, rename_stmt_tokens(body, token_map)))
                .collect(),
            default: default.map(|body| Box::new(rename_stmt_tokens(*body, token_map))),
        },
        Stmt::Return(expr) => Stmt::Return(expr.map(|expr| rename_expr_tokens(expr, token_map))),
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: rename_lvalue_tokens(dst, token_map),
            src: rename_expr_tokens(src, token_map),
        },
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(rename_expr_tokens(expr, token_map)),
        other => other,
    }
}

fn rename_lvalue_tokens(lvalue: LValue, token_map: &HashMap<String, String>) -> LValue {
    match lvalue {
        LValue::Raw(text) => LValue::Raw(rewrite_raw_tokens(&text, token_map)),
        LValue::Var(name) => LValue::Var(rename_token(&name, token_map)),
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(rename_expr_tokens(*addr, token_map)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(rename_expr_tokens(*base, token_map)),
            index: Box::new(rename_expr_tokens(*index, token_map)),
        },
    }
}

fn rename_expr_tokens(expr: Expr, token_map: &HashMap<String, String>) -> Expr {
    match expr {
        Expr::Raw(text) => Expr::Raw(rewrite_raw_tokens(&text, token_map)),
        Expr::Reg(name) => Expr::Reg(rename_token(&name, token_map)),
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(rename_expr_tokens(*arg, token_map)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(rename_expr_tokens(*lhs, token_map)),
            rhs: Box::new(rename_expr_tokens(*rhs, token_map)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(rename_expr_tokens(*cond, token_map)),
            then_expr: Box::new(rename_expr_tokens(*then_expr, token_map)),
            else_expr: Box::new(rename_expr_tokens(*else_expr, token_map)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| rename_expr_tokens(expr, token_map))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(rename_expr_tokens(*addr, token_map)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(rename_expr_tokens(*lo, token_map)),
            hi: Box::new(rename_expr_tokens(*hi, token_map)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(rename_expr_tokens(*expr, token_map)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(rename_expr_tokens(*base, token_map)),
            index: Box::new(rename_expr_tokens(*index, token_map)),
        },
        other => other,
    }
}

fn rename_token(token: &str, token_map: &HashMap<String, String>) -> String {
    token_map
        .get(token)
        .cloned()
        .unwrap_or_else(|| token.to_string())
}

fn rewrite_raw_tokens(text: &str, token_map: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut token = String::new();

    let flush = |out: &mut String, token: &mut String| {
        if token.is_empty() {
            return;
        }
        out.push_str(token_map.get(token).map(String::as_str).unwrap_or(token));
        token.clear();
    };

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.') {
            token.push(ch);
        } else {
            flush(&mut out, &mut token);
            out.push(ch);
        }
    }
    flush(&mut out, &mut token);
    out
}

#[derive(Clone, Default)]
struct Addr64Defs {
    lo_defs: HashMap<String, LoAddDef>,
    carry_defs: HashMap<String, CarryDef>,
    lea_hi_defs: HashMap<String, LeaHiDef>,
    hi_add_defs: HashMap<String, HiAddCarryDef>,
    select_defs: HashMap<String, SelectDef>,
    copy_defs: HashMap<String, String>,
}

#[derive(Clone)]
struct LoAddDef {
    offset: Expr,
    ptr_lo: String,
}

#[derive(Clone)]
struct CarryDef {
    offset: Expr,
    ptr_lo: String,
}

#[derive(Clone)]
struct LeaHiDef {
    offset: Expr,
    ptr_hi: String,
    carry_var: String,
}

#[derive(Clone)]
struct HiAddCarryDef {
    hi_offset_var: Option<String>,
    ptr_hi: String,
    carry_var: String,
}

#[derive(Clone)]
struct SelectDef {
    condition: Expr,
    then_expr: Expr,
    else_expr: Expr,
}

impl Addr64Defs {
    fn merged_with(&self, local: Addr64Defs) -> Addr64Defs {
        let mut merged = self.clone();
        merged.lo_defs.extend(local.lo_defs);
        merged.carry_defs.extend(local.carry_defs);
        merged.lea_hi_defs.extend(local.lea_hi_defs);
        merged.hi_add_defs.extend(local.hi_add_defs);
        merged.select_defs.extend(local.select_defs);
        merged.copy_defs.extend(local.copy_defs);
        merged
    }
}

fn addr64_fold_stmt(stmt: Stmt, inherited: &Addr64Defs) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => addr64_fold_sequence(stmts, false, inherited),
        Stmt::Block(stmts) => addr64_fold_sequence(stmts, true, inherited),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let mut candidates = BTreeSet::new();
            Stmt::If {
                condition: rewrite_addr64_expr(condition, inherited, &mut candidates, &[]),
                then_branch: Box::new(addr64_fold_stmt(*then_branch, inherited)),
                else_branch: else_branch.map(|stmt| Box::new(addr64_fold_stmt(*stmt, inherited))),
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let mut candidates = BTreeSet::new();
            Stmt::Loop {
                kind,
                condition: condition
                    .map(|expr| rewrite_addr64_expr(expr, inherited, &mut candidates, &[])),
                body: Box::new(addr64_fold_stmt(*body, inherited)),
            }
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            let mut candidates = BTreeSet::new();
            Stmt::Switch {
                discriminant: discriminant
                    .map(|expr| rewrite_addr64_expr(expr, inherited, &mut candidates, &[])),
                cases: cases
                    .into_iter()
                    .map(|(label, body)| (label, addr64_fold_stmt(body, inherited)))
                    .collect(),
                default: default.map(|body| Box::new(addr64_fold_stmt(*body, inherited))),
            }
        }
        Stmt::Assign { dst, src } => {
            let mut candidates = BTreeSet::new();
            Stmt::Assign {
                dst: rewrite_addr64_lvalue(dst, inherited, &mut candidates, &[]),
                src: rewrite_addr64_expr(src, inherited, &mut candidates, &[]),
            }
        }
        Stmt::ExprStmt(expr) => {
            let mut candidates = BTreeSet::new();
            Stmt::ExprStmt(rewrite_addr64_expr(expr, inherited, &mut candidates, &[]))
        }
        Stmt::Return(expr) => {
            let mut candidates = BTreeSet::new();
            Stmt::Return(
                expr.map(|expr| rewrite_addr64_expr(expr, inherited, &mut candidates, &[])),
            )
        }
        other => other,
    }
}

fn addr64_fold_sequence(stmts: Vec<Stmt>, as_block: bool, inherited: &Addr64Defs) -> Stmt {
    let folded = stmts
        .into_iter()
        .map(|stmt| addr64_fold_stmt(stmt, inherited))
        .collect::<Vec<_>>();
    let defs = inherited.merged_with(collect_addr64_defs(&folded));
    let mut candidates = BTreeSet::new();
    let rewritten = folded
        .into_iter()
        .map(|stmt| rewrite_addr64_stmt(stmt, &defs, &mut candidates, &[]))
        .collect::<Vec<_>>();
    let rewritten = eliminate_addr64_candidate_defs(rewritten, &candidates);
    if as_block {
        Stmt::Block(rewritten)
    } else {
        Stmt::Sequence(rewritten)
    }
}

fn collect_addr64_defs(stmts: &[Stmt]) -> Addr64Defs {
    let mut defs = Addr64Defs::default();
    for stmt in stmts {
        let Some((name, rhs)) = direct_var_assign(stmt) else {
            continue;
        };
        if let Some(def) = match_lo_add_expr(rhs) {
            defs.lo_defs.insert(name.to_string(), def);
        }
        if let Some(def) = match_carry_expr(rhs) {
            defs.carry_defs.insert(name.to_string(), def);
        }
        if let Some(def) = match_lea_hi_expr(rhs) {
            defs.lea_hi_defs.insert(name.to_string(), def);
        }
        if let Some(def) = match_hi_add_carry_expr(rhs) {
            defs.hi_add_defs.insert(name.to_string(), def);
        }
        if let Some(def) = match_select_expr(rhs) {
            defs.select_defs.insert(name.to_string(), def);
        }
        if let Some(def) = match_copy_expr(rhs) {
            defs.copy_defs.insert(name.to_string(), def);
        }
    }
    defs
}

fn direct_var_assign(stmt: &Stmt) -> Option<(&str, &Expr)> {
    let Stmt::Assign {
        dst: LValue::Var(name),
        src,
    } = stmt
    else {
        return None;
    };
    Some((name.as_str(), src))
}

fn match_lo_add_expr(rhs: &Expr) -> Option<LoAddDef> {
    let Expr::Binary { op, lhs, rhs } = rhs else {
        return None;
    };
    if op != "+" {
        return None;
    }
    Some(LoAddDef {
        offset: (**lhs).clone(),
        ptr_lo: ptr_lane_name(rhs, ".lo32", "_lo32")?,
    })
}

fn match_carry_expr(rhs: &Expr) -> Option<CarryDef> {
    let Expr::CallLike { func, args } = rhs else {
        return None;
    };
    if func != "carry_u32_add3" || args.len() != 3 || !expr_is_zero(&args[2]) {
        return None;
    }
    Some(CarryDef {
        offset: args[0].clone(),
        ptr_lo: ptr_lane_name(&args[1], ".lo32", "_lo32")?,
    })
}

fn match_lea_hi_expr(rhs: &Expr) -> Option<LeaHiDef> {
    let Expr::CallLike { func, args } = rhs else {
        return None;
    };
    if !matches!(func.as_str(), "lea_hi_x" | "lea_hi_x_sx32") || args.len() != 4 {
        return None;
    }
    Some(LeaHiDef {
        offset: args[0].clone(),
        ptr_hi: ptr_lane_name(&args[1], ".hi32", "_hi32")?,
        carry_var: match_var_name(&args[3])?,
    })
}

fn match_hi_add_carry_expr(rhs: &Expr) -> Option<HiAddCarryDef> {
    let Expr::Binary { op, lhs, rhs } = rhs else {
        return None;
    };
    if op != "+" {
        return None;
    }
    let Expr::Binary {
        op: sum_op,
        lhs: hi_offset,
        rhs: ptr_hi,
    } = lhs.as_ref()
    else {
        return None;
    };
    if sum_op != "+" {
        return None;
    }
    Some(HiAddCarryDef {
        hi_offset_var: match_var_name(hi_offset),
        ptr_hi: ptr_lane_name(ptr_hi, ".hi32", "_hi32")?,
        carry_var: carry_increment_var(rhs)?,
    })
}

fn match_copy_expr(rhs: &Expr) -> Option<String> {
    match_var_name(rhs)
}

fn match_select_expr(rhs: &Expr) -> Option<SelectDef> {
    let Expr::Ternary {
        cond,
        then_expr,
        else_expr,
    } = rhs
    else {
        return None;
    };
    Some(SelectDef {
        condition: (**cond).clone(),
        then_expr: (**then_expr).clone(),
        else_expr: (**else_expr).clone(),
    })
}

fn carry_increment_var(expr: &Expr) -> Option<String> {
    let Expr::Ternary {
        cond,
        then_expr,
        else_expr,
    } = expr
    else {
        return None;
    };
    if !expr_is_one(then_expr) || !expr_is_zero(else_expr) {
        return None;
    }
    match_var_name(cond)
}

fn rewrite_addr64_stmt(
    stmt: Stmt,
    defs: &Addr64Defs,
    candidates: &mut BTreeSet<String>,
    active_guards: &[Expr],
) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts
                .into_iter()
                .map(|stmt| rewrite_addr64_stmt(stmt, defs, candidates, active_guards))
                .collect(),
        ),
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .into_iter()
                .map(|stmt| rewrite_addr64_stmt(stmt, defs, candidates, active_guards))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = rewrite_addr64_expr(condition, defs, candidates, active_guards);
            let then_guards = push_active_guard(active_guards, condition.clone());
            let else_guards = push_active_guard(active_guards, negate_expr(condition.clone()));
            Stmt::If {
                condition,
                then_branch: Box::new(rewrite_addr64_stmt(
                    *then_branch,
                    defs,
                    candidates,
                    &then_guards,
                )),
                else_branch: else_branch.map(|stmt| {
                    Box::new(rewrite_addr64_stmt(*stmt, defs, candidates, &else_guards))
                }),
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition
                .map(|expr| rewrite_addr64_expr(expr, defs, candidates, active_guards)),
            body: Box::new(rewrite_addr64_stmt(*body, defs, candidates, active_guards)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant
                .map(|expr| rewrite_addr64_expr(expr, defs, candidates, active_guards)),
            cases: cases
                .into_iter()
                .map(|(label, body)| {
                    (
                        label,
                        rewrite_addr64_stmt(body, defs, candidates, active_guards),
                    )
                })
                .collect(),
            default: default
                .map(|body| Box::new(rewrite_addr64_stmt(*body, defs, candidates, active_guards))),
        },
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: rewrite_addr64_lvalue(dst, defs, candidates, active_guards),
            src: rewrite_addr64_expr(src, defs, candidates, active_guards),
        },
        Stmt::ExprStmt(expr) => {
            Stmt::ExprStmt(rewrite_addr64_expr(expr, defs, candidates, active_guards))
        }
        Stmt::Return(expr) => Stmt::Return(
            expr.map(|expr| rewrite_addr64_expr(expr, defs, candidates, active_guards)),
        ),
        other => other,
    }
}

fn rewrite_addr64_lvalue(
    lvalue: LValue,
    defs: &Addr64Defs,
    candidates: &mut BTreeSet<String>,
    active_guards: &[Expr],
) -> LValue {
    match lvalue {
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(rewrite_addr64_expr(*addr, defs, candidates, active_guards)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(rewrite_addr64_expr(*base, defs, candidates, active_guards)),
            index: Box::new(rewrite_addr64_expr(*index, defs, candidates, active_guards)),
        },
        other => other,
    }
}

fn rewrite_addr64_expr(
    expr: Expr,
    defs: &Addr64Defs,
    candidates: &mut BTreeSet<String>,
    active_guards: &[Expr],
) -> Expr {
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(rewrite_addr64_expr(*arg, defs, candidates, active_guards)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(rewrite_addr64_expr(*lhs, defs, candidates, active_guards)),
            rhs: Box::new(rewrite_addr64_expr(*rhs, defs, candidates, active_guards)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond = rewrite_addr64_expr(*cond, defs, candidates, active_guards);
            let then_guards = push_active_guard(active_guards, cond.clone());
            let else_guards = push_active_guard(active_guards, negate_expr(cond.clone()));
            Expr::Ternary {
                cond: Box::new(cond),
                then_expr: Box::new(rewrite_addr64_expr(
                    *then_expr,
                    defs,
                    candidates,
                    &then_guards,
                )),
                else_expr: Box::new(rewrite_addr64_expr(
                    *else_expr,
                    defs,
                    candidates,
                    &else_guards,
                )),
            }
        }
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| rewrite_addr64_expr(expr, defs, candidates, active_guards))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(rewrite_addr64_expr(*addr, defs, candidates, active_guards)),
        },
        Expr::Addr64 { lo, hi } => {
            let lo = resolve_guard_selected_expr(
                rewrite_addr64_expr(*lo, defs, candidates, active_guards),
                defs,
                active_guards,
            );
            let hi = resolve_guard_selected_expr(
                rewrite_addr64_expr(*hi, defs, candidates, active_guards),
                defs,
                active_guards,
            );
            fold_addr64_use(&lo, &hi, defs, candidates).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(rewrite_addr64_expr(*expr, defs, candidates, active_guards)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(rewrite_addr64_expr(*base, defs, candidates, active_guards)),
            index: Box::new(rewrite_addr64_expr(*index, defs, candidates, active_guards)),
        },
        other => other,
    }
}

fn push_active_guard(active_guards: &[Expr], guard: Expr) -> Vec<Expr> {
    let mut guards = active_guards.to_vec();
    guards.push(guard);
    guards
}

fn resolve_guard_selected_expr(expr: Expr, defs: &Addr64Defs, active_guards: &[Expr]) -> Expr {
    let mut current = expr;
    let mut seen = BTreeSet::new();
    loop {
        let Some(name) = match_var_name(&current) else {
            return current;
        };
        if !seen.insert(name.clone()) {
            return current;
        }
        let Some(select) = defs.select_defs.get(&name) else {
            return current;
        };
        match resolve_guard_branch(active_guards, &select.condition) {
            Some(true) => current = select.then_expr.clone(),
            Some(false) => current = select.else_expr.clone(),
            None => return current,
        }
    }
}

fn resolve_guard_branch(active_guards: &[Expr], condition: &Expr) -> Option<bool> {
    let negated = negate_expr(condition.clone());
    for guard in active_guards.iter().rev() {
        if guard == condition {
            return Some(true);
        }
        if *guard == negated {
            return Some(false);
        }
    }
    None
}

fn fold_addr64_use(
    lo: &Expr,
    hi: &Expr,
    defs: &Addr64Defs,
    candidates: &mut BTreeSet<String>,
) -> Option<Expr> {
    let (lo_name, lo_aliases) = resolve_copy_alias(match_var_name(lo)?, defs);
    let (hi_name, hi_aliases) = resolve_copy_alias(match_var_name(hi)?, defs);
    let lo_info = defs.lo_defs.get(&lo_name)?;
    let base_lo = strip_ptr_suffix(&lo_info.ptr_lo, ".lo32", "_lo32")?;

    let (ptr_hi, carry_var, hi_offset_var) = if let Some(hi_info) = defs.lea_hi_defs.get(&hi_name) {
        if hi_info.offset != lo_info.offset {
            return None;
        }
        (hi_info.ptr_hi.clone(), hi_info.carry_var.clone(), None)
    } else if let Some(hi_info) = defs.hi_add_defs.get(&hi_name) {
        let carry_info = defs.carry_defs.get(&hi_info.carry_var)?;
        if carry_info.offset != lo_info.offset || carry_info.ptr_lo != lo_info.ptr_lo {
            return None;
        }
        (
            hi_info.ptr_hi.clone(),
            hi_info.carry_var.clone(),
            hi_info.hi_offset_var.clone(),
        )
    } else {
        return None;
    };

    let base_hi = strip_ptr_suffix(&ptr_hi, ".hi32", "_hi32")?;
    if base_lo != base_hi {
        return None;
    }

    let (carry_var, carry_aliases) = resolve_copy_alias(carry_var, defs);
    if let Some(carry_info) = defs.carry_defs.get(&carry_var) {
        if carry_info.offset != lo_info.offset || carry_info.ptr_lo != lo_info.ptr_lo {
            return None;
        }
    }

    candidates.extend(lo_aliases);
    candidates.insert(lo_name);
    candidates.extend(hi_aliases);
    candidates.insert(hi_name);
    candidates.extend(carry_aliases);
    candidates.insert(carry_var);
    if let Some(hi_offset_var) = hi_offset_var {
        let (resolved_hi_offset, hi_offset_aliases) = resolve_copy_alias(hi_offset_var, defs);
        candidates.extend(hi_offset_aliases);
        candidates.insert(resolved_hi_offset);
    }

    Some(Expr::Raw(format!(
        "({} + (int64_t){})",
        base_lo,
        lo_info.offset.render()
    )))
}

fn resolve_copy_alias(name: String, defs: &Addr64Defs) -> (String, Vec<String>) {
    let mut current = name;
    let mut aliases = Vec::new();
    let mut seen = BTreeSet::new();
    while let Some(next) = defs.copy_defs.get(&current) {
        if !seen.insert(current.clone()) {
            break;
        }
        aliases.push(current.clone());
        current = next.clone();
    }
    (current, aliases)
}

fn expr_atom_text(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Raw(text)
        | Expr::Imm(text)
        | Expr::Reg(text)
        | Expr::ConstMemSymbol(text)
        | Expr::Builtin(text) => Some(text),
        _ => None,
    }
}

fn match_var_name(expr: &Expr) -> Option<String> {
    let text = expr_atom_text(expr)?;
    if is_atomic_name(text) {
        Some(text.to_string())
    } else {
        None
    }
}

fn ptr_lane_name(expr: &Expr, dot_suffix: &str, underscore_suffix: &str) -> Option<String> {
    let text = expr_atom_text(expr)?;
    strip_ptr_suffix(text, dot_suffix, underscore_suffix).map(|_| text.to_string())
}

fn strip_ptr_suffix(name: &str, dot_suffix: &str, underscore_suffix: &str) -> Option<String> {
    if let Some(base) = name.strip_suffix(dot_suffix) {
        return Some(base.to_string());
    }
    name.strip_suffix(underscore_suffix).map(str::to_string)
}

fn expr_is_zero(expr: &Expr) -> bool {
    matches!(expr_atom_text(expr), Some("0"))
}

fn expr_is_one(expr: &Expr) -> bool {
    matches!(expr_atom_text(expr), Some("1"))
}

fn eliminate_addr64_candidate_defs(
    mut stmts: Vec<Stmt>,
    candidates: &BTreeSet<String>,
) -> Vec<Stmt> {
    if candidates.is_empty() {
        return stmts;
    }
    loop {
        let next = eliminate_addr64_candidate_defs_once(&stmts, candidates);
        if next == stmts {
            return next;
        }
        stmts = next;
    }
}

fn eliminate_addr64_candidate_defs_once(
    stmts: &[Stmt],
    candidates: &BTreeSet<String>,
) -> Vec<Stmt> {
    let mut used = BTreeSet::new();
    for stmt in stmts {
        collect_stmt_used_vars(stmt, &mut used);
    }
    stmts
        .iter()
        .filter(|stmt| {
            direct_assign_name(stmt)
                .map(|name| !candidates.contains(name) || used.contains(name))
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

fn direct_assign_name(stmt: &Stmt) -> Option<&str> {
    let Stmt::Assign {
        dst: LValue::Var(name),
        ..
    } = stmt
    else {
        return None;
    };
    Some(name.as_str())
}

fn collect_stmt_used_vars(stmt: &Stmt, used: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_stmt_used_vars(stmt, used);
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_used_expr_vars(condition, used);
            collect_stmt_used_vars(then_branch, used);
            if let Some(else_branch) = else_branch {
                collect_stmt_used_vars(else_branch, used);
            }
        }
        Stmt::Loop {
            condition, body, ..
        } => {
            if let Some(condition) = condition {
                collect_used_expr_vars(condition, used);
            }
            collect_stmt_used_vars(body, used);
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            if let Some(discriminant) = discriminant {
                collect_used_expr_vars(discriminant, used);
            }
            for (_, body) in cases {
                collect_stmt_used_vars(body, used);
            }
            if let Some(default) = default {
                collect_stmt_used_vars(default, used);
            }
        }
        Stmt::Assign { dst, src } => {
            collect_used_expr_vars(src, used);
            collect_used_lvalue_vars(dst, used);
        }
        Stmt::ExprStmt(expr) => collect_used_expr_vars(expr, used),
        Stmt::Return(expr) => {
            if let Some(expr) = expr {
                collect_used_expr_vars(expr, used);
            }
        }
        Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Empty => {}
    }
}

fn is_atomic_name(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.'))
}

pub fn ast_predicate_cleanup(stmt: Stmt) -> Stmt {
    cleanup_predicate_stmt(stmt)
}

fn cleanup_predicate_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => cleanup_predicate_sequence(stmts, false),
        Stmt::Block(stmts) => cleanup_predicate_sequence(stmts, true),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(cleanup_predicate_stmt(*then_branch)),
            else_branch: else_branch.map(|stmt| Box::new(cleanup_predicate_stmt(*stmt))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(cleanup_predicate_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, cleanup_predicate_stmt(body)))
                .collect(),
            default: default.map(|body| Box::new(cleanup_predicate_stmt(*body))),
        },
        other => other,
    }
}

fn cleanup_predicate_sequence(stmts: Vec<Stmt>, as_block: bool) -> Stmt {
    let mut out = Vec::new();
    let mut active_guards = Vec::<(Expr, BTreeSet<String>)>::new();

    for stmt in stmts {
        let stmt = cleanup_predicate_stmt(stmt);
        if matches!(stmt, Stmt::Empty) {
            continue;
        }

        if let Some((condition, used_vars)) = duplicate_return_guard_key(&stmt) {
            if active_guards
                .iter()
                .any(|(active_condition, _)| *active_condition == condition)
            {
                continue;
            }
            active_guards.push((condition, used_vars));
        }

        let assigned = stmt_assigned_vars(&stmt);
        if !assigned.is_empty() {
            active_guards.retain(|(_, used_vars)| assigned.is_disjoint(used_vars));
        }

        out.push(stmt);
    }

    if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    }
}

fn duplicate_return_guard_key(stmt: &Stmt) -> Option<(Expr, BTreeSet<String>)> {
    let Stmt::If {
        condition,
        then_branch,
        else_branch: None,
    } = stmt
    else {
        return None;
    };
    if !is_plain_return(then_branch) {
        return None;
    }
    let mut used_vars = BTreeSet::new();
    collect_used_expr_vars(condition, &mut used_vars);
    Some((condition.clone(), used_vars))
}

fn is_plain_return(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(None) => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            let mut non_empty = stmts.iter().filter(|stmt| !matches!(stmt, Stmt::Empty));
            matches!(non_empty.next(), Some(stmt) if is_plain_return(stmt))
                && non_empty.next().is_none()
        }
        _ => false,
    }
}

fn stmt_assigned_vars(stmt: &Stmt) -> BTreeSet<String> {
    let mut assigned = BTreeSet::new();
    collect_assigned_vars(stmt, &mut assigned);
    assigned
}

fn collect_assigned_vars(stmt: &Stmt, assigned: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_assigned_vars(stmt, assigned);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_assigned_vars(then_branch, assigned);
            if let Some(else_branch) = else_branch {
                collect_assigned_vars(else_branch, assigned);
            }
        }
        Stmt::Loop { body, .. } => collect_assigned_vars(body, assigned),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_assigned_vars(body, assigned);
            }
            if let Some(default) = default {
                collect_assigned_vars(default, assigned);
            }
        }
        Stmt::Assign {
            dst: LValue::Var(name),
            ..
        } => {
            assigned.insert(name.clone());
        }
        Stmt::Assign { .. }
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::ExprStmt(_)
        | Stmt::Goto(_)
        | Stmt::Empty => {}
    }
}

fn simplify_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => simplify_sequence(stmts),
        Stmt::Block(stmts) => simplify_block(stmts),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => simplify_if(condition, *then_branch, else_branch.map(|stmt| *stmt)),
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(simplify_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            let cases = cases
                .into_iter()
                .map(|(label, body)| (label, simplify_stmt(body)))
                .collect();
            let default = default
                .map(|body| simplify_stmt(*body))
                .filter(|stmt| !is_effectively_empty(stmt))
                .map(Box::new);
            Stmt::Switch {
                discriminant,
                cases,
                default,
            }
        }
        Stmt::Assign { dst, src } => simplify_assign(dst, src),
        other => other,
    }
}

fn simplify_assign(dst: LValue, src: Expr) -> Stmt {
    if dst.render() == src.render() {
        return Stmt::Empty;
    }

    if let Expr::Ternary {
        cond,
        then_expr,
        else_expr,
    } = src
    {
        let dst_text = dst.render();
        if else_expr.render() == dst_text {
            return simplify_if(
                *cond,
                Stmt::Assign {
                    dst,
                    src: *then_expr,
                },
                None,
            );
        }
        if then_expr.render() == dst_text {
            return simplify_if(
                negate_expr(*cond),
                Stmt::Assign {
                    dst,
                    src: *else_expr,
                },
                None,
            );
        }
        return Stmt::Assign {
            dst,
            src: Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            },
        };
    }

    Stmt::Assign { dst, src }
}

fn simplify_sequence(stmts: Vec<Stmt>) -> Stmt {
    let mut out = Vec::new();
    for stmt in stmts {
        match simplify_stmt(stmt) {
            Stmt::Empty => {}
            Stmt::Sequence(inner) => out.extend(inner),
            other => out.push(other),
        }
    }
    match out.len() {
        0 => Stmt::Empty,
        1 => out.into_iter().next().unwrap_or(Stmt::Empty),
        _ => Stmt::Sequence(out),
    }
}

fn simplify_block(stmts: Vec<Stmt>) -> Stmt {
    let mut out = Vec::new();
    for stmt in stmts {
        match simplify_stmt(stmt) {
            Stmt::Empty => {}
            Stmt::Block(inner) => out.extend(inner),
            other => out.push(other),
        }
    }
    match out.len() {
        0 => Stmt::Empty,
        _ => Stmt::Block(out),
    }
}

fn simplify_if(condition: Expr, then_branch: Stmt, else_branch: Option<Stmt>) -> Stmt {
    let then_branch = simplify_stmt(then_branch);
    let else_branch = else_branch
        .map(simplify_stmt)
        .filter(|stmt| !is_effectively_empty(stmt));

    match (is_effectively_empty(&then_branch), else_branch) {
        (true, None) => Stmt::Empty,
        (true, Some(else_branch)) => Stmt::If {
            condition: negate_expr(condition),
            then_branch: Box::new(else_branch),
            else_branch: None,
        },
        (false, None) => Stmt::If {
            condition,
            then_branch: Box::new(then_branch),
            else_branch: None,
        },
        (false, Some(else_branch)) => Stmt::If {
            condition,
            then_branch: Box::new(then_branch),
            else_branch: Some(Box::new(else_branch)),
        },
    }
}

fn negate_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Unary { op, arg } if op == "!" => *arg,
        other => Expr::Unary {
            op: "!".to_string(),
            arg: Box::new(other),
        },
    }
}

fn is_effectively_empty(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Empty => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => stmts.iter().all(is_effectively_empty),
        _ => false,
    }
}

fn dce_stmt(stmt: Stmt, live_out: BTreeSet<String>) -> (Stmt, BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) => dce_sequence_like(stmts, false, live_out),
        Stmt::Block(stmts) => dce_sequence_like(stmts, true, live_out),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let (then_branch, then_live) = dce_stmt(*then_branch, live_out.clone());
            let (else_branch, else_live) = if let Some(else_branch) = else_branch {
                let (stmt, live) = dce_stmt(*else_branch, live_out.clone());
                (Some(Box::new(stmt)), live)
            } else {
                (None, live_out.clone())
            };
            let mut live_before = then_live;
            live_before.extend(else_live);
            collect_used_expr_vars(&condition, &mut live_before);
            (
                Stmt::If {
                    condition,
                    then_branch: Box::new(then_branch),
                    else_branch,
                },
                live_before,
            )
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let mut body_live_out = live_out.clone();
            if let Some(condition) = &condition {
                collect_used_expr_vars(condition, &mut body_live_out);
            }
            let (body, body_live_before) = dce_stmt(*body, body_live_out);
            let mut live_before = live_out;
            live_before.extend(body_live_before);
            if let Some(condition) = &condition {
                collect_used_expr_vars(condition, &mut live_before);
            }
            (
                Stmt::Loop {
                    kind,
                    condition,
                    body: Box::new(body),
                },
                live_before,
            )
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            let mut live_before = live_out.clone();
            let cases = cases
                .into_iter()
                .map(|(label, body)| {
                    let (body, case_live) = dce_stmt(body, live_out.clone());
                    live_before.extend(case_live);
                    (label, body)
                })
                .collect();
            let default = default.map(|body| {
                let (body, default_live) = dce_stmt(*body, live_out.clone());
                live_before.extend(default_live);
                Box::new(body)
            });
            if let Some(discriminant) = &discriminant {
                collect_used_expr_vars(discriminant, &mut live_before);
            }
            (
                Stmt::Switch {
                    discriminant,
                    cases,
                    default,
                },
                live_before,
            )
        }
        Stmt::Assign { dst, src } => dce_assign(dst, src, live_out),
        Stmt::ExprStmt(expr) => {
            let mut live_before = live_out;
            collect_used_expr_vars(&expr, &mut live_before);
            (Stmt::ExprStmt(expr), live_before)
        }
        Stmt::Return(expr) => {
            let mut live_before = BTreeSet::new();
            if let Some(expr) = &expr {
                collect_used_expr_vars(expr, &mut live_before);
            }
            (Stmt::Return(expr), live_before)
        }
        Stmt::Break => (Stmt::Break, live_out),
        Stmt::Continue => (Stmt::Continue, live_out),
        Stmt::Goto(label) => (Stmt::Goto(label), live_out),
        Stmt::Empty => (Stmt::Empty, live_out),
    }
}

fn dce_sequence_like(
    stmts: Vec<Stmt>,
    as_block: bool,
    live_out: BTreeSet<String>,
) -> (Stmt, BTreeSet<String>) {
    let mut live = live_out;
    let mut kept = Vec::new();

    for stmt in stmts.into_iter().rev() {
        let (stmt, live_before) = dce_stmt(stmt, live);
        live = live_before;
        if !matches!(stmt, Stmt::Empty) {
            kept.push(stmt);
        }
    }

    kept.reverse();
    let stmt = if as_block {
        Stmt::Block(kept)
    } else {
        Stmt::Sequence(kept)
    };
    (stmt, live)
}

fn dce_assign(dst: LValue, src: Expr, mut live_out: BTreeSet<String>) -> (Stmt, BTreeSet<String>) {
    if let LValue::Var(name) = &dst {
        if !live_out.contains(name) && expr_is_pure_for_dce(&src) {
            return (Stmt::Empty, live_out);
        }
        live_out.remove(name);
        collect_used_expr_vars(&src, &mut live_out);
        return (Stmt::Assign { dst, src }, live_out);
    }

    collect_used_expr_vars(&src, &mut live_out);
    collect_used_lvalue_vars(&dst, &mut live_out);
    (Stmt::Assign { dst, src }, live_out)
}

fn expr_is_pure_for_dce(expr: &Expr) -> bool {
    match expr {
        Expr::Raw(_) => false,
        Expr::Imm(_) | Expr::Reg(_) | Expr::ConstMemSymbol(_) => true,
        Expr::Unary { arg, .. } => expr_is_pure_for_dce(arg),
        Expr::Binary { lhs, rhs, .. } => expr_is_pure_for_dce(lhs) && expr_is_pure_for_dce(rhs),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_is_pure_for_dce(cond)
                && expr_is_pure_for_dce(then_expr)
                && expr_is_pure_for_dce(else_expr)
        }
        Expr::CallLike { func, args } => {
            is_pure_calllike(func) && args.iter().all(expr_is_pure_for_dce)
        }
        Expr::Load { .. } | Expr::Builtin(_) | Expr::Addr64 { .. } => false,
        Expr::Cast { expr, .. } => expr_is_pure_for_dce(expr),
        Expr::Index { base, index } => expr_is_pure_for_dce(base) && expr_is_pure_for_dce(index),
    }
}

fn is_pure_calllike(func: &str) -> bool {
    matches!(
        func,
        "abs"
            | "carry_u32_add3"
            | "lea_hi_x"
            | "lea_hi_x_sx32"
            | "rcp_approx"
            | "rsqrtf"
            | "exp2f"
            | "log2f"
            | "sinf"
            | "cosf"
            | "sqrtf"
    )
}

fn collect_used_lvalue_vars(lvalue: &LValue, live: &mut BTreeSet<String>) {
    match lvalue {
        LValue::Raw(text) => collect_raw_identifiers(text, live),
        LValue::Var(_) => {}
        LValue::Deref { addr, .. } => collect_used_expr_vars(addr, live),
        LValue::Indexed { base, index } => {
            collect_used_expr_vars(base, live);
            collect_used_expr_vars(index, live);
        }
    }
}

fn collect_used_expr_vars(expr: &Expr, live: &mut BTreeSet<String>) {
    match expr {
        Expr::Reg(name) => {
            live.insert(name.clone());
        }
        Expr::Raw(text) => collect_raw_identifiers(text, live),
        Expr::Imm(_) | Expr::ConstMemSymbol(_) | Expr::Builtin(_) => {}
        Expr::Unary { arg, .. } => collect_used_expr_vars(arg, live),
        Expr::Binary { lhs, rhs, .. } => {
            collect_used_expr_vars(lhs, live);
            collect_used_expr_vars(rhs, live);
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_used_expr_vars(cond, live);
            collect_used_expr_vars(then_expr, live);
            collect_used_expr_vars(else_expr, live);
        }
        Expr::CallLike { args, .. } => {
            for arg in args {
                collect_used_expr_vars(arg, live);
            }
        }
        Expr::Load { addr, .. } => collect_used_expr_vars(addr, live),
        Expr::Addr64 { lo, hi } => {
            collect_used_expr_vars(lo, live);
            collect_used_expr_vars(hi, live);
        }
        Expr::Cast { expr, .. } => collect_used_expr_vars(expr, live),
        Expr::Index { base, index } => {
            collect_used_expr_vars(base, live);
            collect_used_expr_vars(index, live);
        }
    }
}

fn collect_raw_identifiers(text: &str, live: &mut BTreeSet<String>) {
    let chars = text.as_bytes();
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx] as char;
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = idx;
            idx += 1;
            while idx < chars.len() {
                let ch = chars[idx] as char;
                if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.') {
                    idx += 1;
                } else {
                    break;
                }
            }
            if let Some(token) = text.get(start..idx) {
                live.insert(token.to_string());
            }
        } else {
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::LoopKind;

    #[test]
    fn simplify_drops_self_assignment() {
        let stmt = Stmt::Assign {
            dst: LValue::Var("v0".to_string()),
            src: Expr::Reg("v0".to_string()),
        };
        assert_eq!(ast_simplify(stmt), Stmt::Empty);
    }

    #[test]
    fn simplify_inverts_empty_then_if() {
        let stmt = Stmt::If {
            condition: Expr::Reg("b0".to_string()),
            then_branch: Box::new(Stmt::Empty),
            else_branch: Some(Box::new(Stmt::Return(None))),
        };
        assert_eq!(
            ast_simplify(stmt),
            Stmt::If {
                condition: Expr::Unary {
                    op: "!".to_string(),
                    arg: Box::new(Expr::Reg("b0".to_string())),
                },
                then_branch: Box::new(Stmt::Return(None)),
                else_branch: None,
            }
        );
    }

    #[test]
    fn simplify_flattens_nested_sequences() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Empty,
            Stmt::Sequence(vec![Stmt::ExprStmt(Expr::Reg("a".to_string()))]),
            Stmt::ExprStmt(Expr::Reg("b".to_string())),
        ]);
        assert_eq!(
            ast_simplify(stmt),
            Stmt::Sequence(vec![
                Stmt::ExprStmt(Expr::Reg("a".to_string())),
                Stmt::ExprStmt(Expr::Reg("b".to_string())),
            ])
        );
    }

    #[test]
    fn simplify_preserves_loop_structure() {
        let stmt = Stmt::Loop {
            kind: LoopKind::While,
            condition: Some(Expr::Reg("b0".to_string())),
            body: Box::new(Stmt::Sequence(vec![Stmt::Empty, Stmt::Continue])),
        };
        assert_eq!(
            ast_simplify(stmt),
            Stmt::Loop {
                kind: LoopKind::While,
                condition: Some(Expr::Reg("b0".to_string())),
                body: Box::new(Stmt::Continue),
            }
        );
    }

    #[test]
    fn dce_removes_dead_pure_assignment() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v0".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v1".to_string())),
                    rhs: Box::new(Expr::Imm("1".to_string())),
                },
            },
            Stmt::Return(None),
        ]);
        assert_eq!(ast_cleanup(stmt), Stmt::Return(None));
    }

    #[test]
    fn dce_removes_dead_pure_helper_call_assignment() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v0".to_string()),
                src: Expr::CallLike {
                    func: "abs".to_string(),
                    args: vec![Expr::Reg("v1".to_string())],
                },
            },
            Stmt::Return(None),
        ]);
        assert_eq!(ast_cleanup(stmt), Stmt::Return(None));
    }

    #[test]
    fn dce_removes_dead_carry_helper_assignment() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("b0".to_string()),
                src: Expr::CallLike {
                    func: "carry_u32_add3".to_string(),
                    args: vec![
                        Expr::Reg("v1".to_string()),
                        Expr::Reg("v2".to_string()),
                        Expr::Imm("0".to_string()),
                    ],
                },
            },
            Stmt::Return(None),
        ]);
        assert_eq!(ast_cleanup(stmt), Stmt::Return(None));
    }

    #[test]
    fn dce_preserves_live_assignment_flow() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v0".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v1".to_string())),
                    rhs: Box::new(Expr::Imm("1".to_string())),
                },
            },
            Stmt::Return(Some(Expr::Reg("v0".to_string()))),
        ]);
        assert_eq!(ast_cleanup(stmt.clone()), stmt);
    }

    #[test]
    fn dce_keeps_branch_defs_live_out_of_if() {
        let stmt = Stmt::Sequence(vec![
            Stmt::If {
                condition: Expr::Reg("b0".to_string()),
                then_branch: Box::new(Stmt::Assign {
                    dst: LValue::Var("v0".to_string()),
                    src: Expr::Reg("v1".to_string()),
                }),
                else_branch: None,
            },
            Stmt::Return(Some(Expr::Reg("v0".to_string()))),
        ]);
        assert_eq!(ast_cleanup(stmt.clone()), stmt);
    }

    #[test]
    fn dce_preserves_memory_store_even_when_result_unused() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Deref {
                    ty: Some("uint32_t".to_string()),
                    addr: Box::new(Expr::Reg("ptr".to_string())),
                },
                src: Expr::Reg("v0".to_string()),
            },
            Stmt::Return(None),
        ]);
        assert_eq!(ast_cleanup(stmt.clone()), stmt);
    }

    #[test]
    fn dce_keeps_unknown_calllike_assignment() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v0".to_string()),
                src: Expr::CallLike {
                    func: "mystery".to_string(),
                    args: vec![Expr::Reg("v1".to_string())],
                },
            },
            Stmt::Return(None),
        ]);
        assert_eq!(ast_cleanup(stmt.clone()), stmt);
    }

    #[test]
    fn predicate_cleanup_removes_duplicate_return_guard() {
        let guard = Stmt::If {
            condition: Expr::Reg("b0".to_string()),
            then_branch: Box::new(Stmt::Return(None)),
            else_branch: None,
        };
        let stmt = Stmt::Sequence(vec![
            guard.clone(),
            Stmt::ExprStmt(Expr::CallLike {
                func: "touch".to_string(),
                args: vec![Expr::Reg("v0".to_string())],
            }),
            guard,
        ]);
        assert_eq!(
            ast_cleanup(stmt),
            Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("b0".to_string()),
                    then_branch: Box::new(Stmt::Return(None)),
                    else_branch: None,
                },
                Stmt::ExprStmt(Expr::CallLike {
                    func: "touch".to_string(),
                    args: vec![Expr::Reg("v0".to_string())],
                }),
            ])
        );
    }

    #[test]
    fn predicate_cleanup_keeps_guard_after_nested_reassignment() {
        let guard = Stmt::If {
            condition: Expr::Reg("b0".to_string()),
            then_branch: Box::new(Stmt::Return(None)),
            else_branch: None,
        };
        let stmt = Stmt::Sequence(vec![
            guard.clone(),
            Stmt::If {
                condition: Expr::Reg("v1".to_string()),
                then_branch: Box::new(Stmt::Assign {
                    dst: LValue::Var("b0".to_string()),
                    src: Expr::Reg("v2".to_string()),
                }),
                else_branch: None,
            },
            guard.clone(),
        ]);
        assert_eq!(ast_cleanup(stmt.clone()), stmt);
    }

    #[test]
    fn addr64_fold_rewrites_helper_chain_structurally() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v5".to_string()),
                src: Expr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(Expr::Reg("v4".to_string())),
                    rhs: Box::new(Expr::Imm("255".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("b0".to_string()),
                src: Expr::CallLike {
                    func: "carry_u32_add3".to_string(),
                    args: vec![
                        Expr::Reg("v5".to_string()),
                        Expr::ConstMemSymbol("arg0_ptr.lo32".to_string()),
                        Expr::Imm("0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("v6".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v5".to_string())),
                    rhs: Box::new(Expr::ConstMemSymbol("arg0_ptr.lo32".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("v7".to_string()),
                src: Expr::CallLike {
                    func: "lea_hi_x_sx32".to_string(),
                    args: vec![
                        Expr::Reg("v5".to_string()),
                        Expr::ConstMemSymbol("arg0_ptr.hi32".to_string()),
                        Expr::Imm("1".to_string()),
                        Expr::Reg("b0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("v8".to_string()),
                src: Expr::Load {
                    ty: Some("uint8_t".to_string()),
                    addr: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("v6".to_string())),
                        hi: Box::new(Expr::Reg("v7".to_string())),
                    }),
                },
            },
        ]);

        let rendered = ast_cleanup(stmt).render_with_indent(0);
        assert!(rendered.contains("v5 = v4 & 255;"));
        assert!(rendered.contains("v8 = *((uint8_t*)(arg0_ptr + (int64_t)v5));"));
        assert!(!rendered.contains("addr64("));
        assert!(!rendered.contains("carry_u32_add3("));
        assert!(!rendered.contains("lea_hi_x_sx32("));
        assert!(!rendered.contains("arg0_ptr.lo32"));
        assert!(!rendered.contains("arg0_ptr.hi32"));
    }

    #[test]
    fn addr64_fold_resolves_guard_selected_pointer_pair() {
        let guard = Expr::Unary {
            op: "!".to_string(),
            arg: Box::new(Expr::Reg("p0".to_string())),
        };
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v5".to_string()),
                src: Expr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(Expr::Reg("v4".to_string())),
                    rhs: Box::new(Expr::Imm("255".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("b0".to_string()),
                src: Expr::CallLike {
                    func: "carry_u32_add3".to_string(),
                    args: vec![
                        Expr::Reg("v5".to_string()),
                        Expr::ConstMemSymbol("arg4_ptr.lo32".to_string()),
                        Expr::Imm("0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("lo_live".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v5".to_string())),
                    rhs: Box::new(Expr::ConstMemSymbol("arg4_ptr.lo32".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("hi_live".to_string()),
                src: Expr::CallLike {
                    func: "lea_hi_x_sx32".to_string(),
                    args: vec![
                        Expr::Reg("v5".to_string()),
                        Expr::ConstMemSymbol("arg4_ptr.hi32".to_string()),
                        Expr::Imm("1".to_string()),
                        Expr::Reg("b0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("lo_selected".to_string()),
                src: Expr::Ternary {
                    cond: Box::new(guard.clone()),
                    then_expr: Box::new(Expr::Reg("lo_live".to_string())),
                    else_expr: Box::new(Expr::Reg("old_lo".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("hi_selected".to_string()),
                src: Expr::Ternary {
                    cond: Box::new(guard.clone()),
                    then_expr: Box::new(Expr::Reg("hi_live".to_string())),
                    else_expr: Box::new(Expr::Reg("old_hi".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("out".to_string()),
                src: Expr::Ternary {
                    cond: Box::new(guard),
                    then_expr: Box::new(Expr::Load {
                        ty: Some("uint8_t".to_string()),
                        addr: Box::new(Expr::Addr64 {
                            lo: Box::new(Expr::Reg("lo_selected".to_string())),
                            hi: Box::new(Expr::Reg("hi_selected".to_string())),
                        }),
                    }),
                    else_expr: Box::new(Expr::Reg("out_old".to_string())),
                },
            },
        ]);

        let rendered = ast_cleanup(stmt).render_with_indent(0);
        assert!(rendered.contains("out = !p0 ? (*((uint8_t*)(arg4_ptr + (int64_t)v5))) : out_old;"));
        assert!(!rendered.contains("addr64("));
        assert!(!rendered.contains("carry_u32_add3("));
        assert!(!rendered.contains("lea_hi_x_sx32("));
        assert!(!rendered.contains("lo_selected"));
        assert!(!rendered.contains("hi_selected"));
        assert!(!rendered.contains("arg4_ptr.lo32"));
        assert!(!rendered.contains("arg4_ptr.hi32"));
    }

    #[test]
    fn addr64_fold_preserves_outer_pointer_offset_shape() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("v5".to_string()),
                src: Expr::Binary {
                    op: "&".to_string(),
                    lhs: Box::new(Expr::Reg("v4".to_string())),
                    rhs: Box::new(Expr::Imm("255".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("b0".to_string()),
                src: Expr::CallLike {
                    func: "carry_u32_add3".to_string(),
                    args: vec![
                        Expr::Reg("v5".to_string()),
                        Expr::ConstMemSymbol("arg4_ptr.lo32".to_string()),
                        Expr::Imm("0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("v6".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v5".to_string())),
                    rhs: Box::new(Expr::ConstMemSymbol("arg4_ptr.lo32".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("v8".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Reg("v5".to_string())),
                        rhs: Box::new(Expr::ConstMemSymbol("arg4_ptr.hi32".to_string())),
                    }),
                    rhs: Box::new(Expr::Ternary {
                        cond: Box::new(Expr::Reg("b0".to_string())),
                        then_expr: Box::new(Expr::Imm("1".to_string())),
                        else_expr: Box::new(Expr::Imm("0".to_string())),
                    }),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("v9".to_string()),
                src: Expr::Load {
                    ty: Some("uint8_t".to_string()),
                    addr: Box::new(Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Addr64 {
                            lo: Box::new(Expr::Reg("v6".to_string())),
                            hi: Box::new(Expr::Reg("v8".to_string())),
                        }),
                        rhs: Box::new(Expr::Imm("1".to_string())),
                    }),
                },
            },
        ]);

        let rendered = ast_cleanup(stmt).render_with_indent(0);
        assert!(rendered.contains("v9 = *((uint8_t*)((arg4_ptr + (int64_t)v5) + 1));"));
        assert!(!rendered.contains("addr64("));
        assert!(!rendered.contains("carry_u32_add3("));
        assert!(!rendered.contains("arg4_ptr.lo32"));
        assert!(!rendered.contains("arg4_ptr.hi32"));
    }
}
