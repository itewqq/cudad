//! Local AST cleanup passes for the canonical backend.
//!
//! Purpose:
//! - run narrow, AST-native simplifications on canonical lowering output
//! - prune dead pure helper temps that survive because memory-aware lowering
//!   inlines addresses directly from analysis facts
//!
//! Inputs:
//! - canonical `StructuredFunction`
//! - structured AST statements and expressions produced by `ast_lowering`
//!
//! Outputs:
//! - equivalent canonical AST with dead pure assignments removed
//!
//! Invariants:
//! - this module works only on the AST; it does not inspect rendered text
//! - cleanup stays local and deterministic
//! - unstructured control flow is left untouched rather than guessed through
//!   with unsafe liveness assumptions
//!
//! Algorithm:
//! - backward liveness over structured statements
//! - loop-body fixpoint for structured loops
//! - pure-expression pruning for dead assignments and dead expression
//!   statements
//!
//! This module must not:
//! - perform regex/text cleanup
//! - rediscover memory facts from rendered output
//! - run the old fixed-point backend repair pipeline

use std::collections::{BTreeSet, HashMap};

use crate::ast::{Expr, LValue, Stmt, StructuredFunction};

const MATCH_RESOLVE_DEPTH: usize = 12;

pub fn canonicalize_function(function: StructuredFunction) -> StructuredFunction {
    if contains_unstructured_control_flow(&function.body) {
        return prune_dead_pure_defs(function);
    }
    let mut defs = HashMap::new();
    let body = recover_rcp_division_stmt_tree(function.body, &mut defs, &BTreeSet::new());
    prune_dead_pure_defs(StructuredFunction {
        params: function.params,
        locals: function.locals,
        body,
    })
}

pub fn prune_dead_pure_defs(function: StructuredFunction) -> StructuredFunction {
    if contains_unstructured_control_flow(&function.body) {
        return function;
    }
    let body = simplify_trivial_stmt(prune_dead_pure_stmt(function.body));
    StructuredFunction {
        params: function.params,
        locals: function.locals,
        body,
    }
}

pub fn prune_dead_pure_stmt(stmt: Stmt) -> Stmt {
    dce_stmt(stmt, BTreeSet::new()).0
}

fn dce_stmt(stmt: Stmt, live_out: BTreeSet<String>) -> (Stmt, BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) => dce_sequence_like(stmts, false, live_out),
        Stmt::Block(stmts) => dce_sequence_like(stmts, true, live_out),
        Stmt::Label { name, body } => {
            let (body, live_before) = dce_stmt(*body, live_out);
            (
                Stmt::Label {
                    name,
                    body: Box::new(body),
                },
                live_before,
            )
        }
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
            let cond_live = condition
                .as_ref()
                .map(loop_condition_live_vars)
                .unwrap_or_default();
            let original_body = *body;
            let mut body_live_out = live_out.clone();
            body_live_out.extend(cond_live.clone());

            loop {
                let (_, body_live_before) = dce_stmt(original_body.clone(), body_live_out.clone());
                let mut next_live_out = live_out.clone();
                next_live_out.extend(cond_live.clone());
                next_live_out.extend(body_live_before);
                if next_live_out == body_live_out {
                    break;
                }
                body_live_out = next_live_out;
            }

            let (body, body_live_before) = dce_stmt(original_body, body_live_out);
            let mut live_before = live_out;
            live_before.extend(cond_live);
            live_before.extend(body_live_before);
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
            if expr_is_pure_for_dce(&expr) {
                (Stmt::Empty, live_out)
            } else {
                let mut live_before = live_out;
                collect_used_expr_vars(&expr, &mut live_before);
                (Stmt::ExprStmt(expr), live_before)
            }
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

fn simplify_trivial_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            let kept = stmts
                .into_iter()
                .map(simplify_trivial_stmt)
                .filter(|stmt| !stmt_is_trivial_empty(stmt))
                .collect::<Vec<_>>();
            match kept.len() {
                0 => Stmt::Empty,
                1 => kept.into_iter().next().expect("single stmt"),
                _ => Stmt::Sequence(kept),
            }
        }
        Stmt::Block(stmts) => {
            let kept = stmts
                .into_iter()
                .map(simplify_trivial_stmt)
                .filter(|stmt| !stmt_is_trivial_empty(stmt))
                .collect::<Vec<_>>();
            match kept.len() {
                0 => Stmt::Empty,
                1 => kept.into_iter().next().expect("single stmt"),
                _ => Stmt::Block(kept),
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let then_branch = simplify_trivial_stmt(*then_branch);
            let else_branch = else_branch.map(|branch| simplify_trivial_stmt(*branch));
            if stmt_is_trivial_empty(&then_branch)
                && else_branch.as_ref().is_none_or(stmt_is_trivial_empty)
            {
                Stmt::Empty
            } else {
                Stmt::If {
                    condition,
                    then_branch: Box::new(then_branch),
                    else_branch: else_branch
                        .filter(|branch| !stmt_is_trivial_empty(branch))
                        .map(Box::new),
                }
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(simplify_trivial_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, simplify_trivial_stmt(body)))
                .collect(),
            default: default.map(|body| Box::new(simplify_trivial_stmt(*body))),
        },
        Stmt::Label { name, body } => {
            let body = simplify_trivial_stmt(*body);
            if stmt_is_trivial_empty(&body) {
                Stmt::Empty
            } else {
                Stmt::Label {
                    name,
                    body: Box::new(body),
                }
            }
        }
        Stmt::Goto(_)
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::Assign { .. }
        | Stmt::ExprStmt(_)
        | Stmt::Empty => stmt,
    }
}

fn stmt_is_trivial_empty(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Empty => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => stmts.iter().all(stmt_is_trivial_empty),
        _ => false,
    }
}

fn dce_assign(dst: LValue, src: Expr, mut live_out: BTreeSet<String>) -> (Stmt, BTreeSet<String>) {
    if let Some(name) = lvalue_symbol_name(&dst) {
        if !live_out.contains(&name) && expr_is_pure_for_dce(&src) {
            return (Stmt::Empty, live_out);
        }
        live_out.remove(&name);
        collect_used_expr_vars(&src, &mut live_out);
        return (Stmt::Assign { dst, src }, live_out);
    }

    collect_used_expr_vars(&src, &mut live_out);
    collect_used_lvalue_vars(&dst, &mut live_out);
    (Stmt::Assign { dst, src }, live_out)
}

fn loop_condition_live_vars(condition: &Expr) -> BTreeSet<String> {
    let mut live = BTreeSet::new();
    collect_used_expr_vars(condition, &mut live);
    live
}

fn lvalue_symbol_name(lvalue: &LValue) -> Option<String> {
    match lvalue {
        LValue::Var(name) => Some(name.clone()),
        _ => None,
    }
}

fn expr_is_pure_for_dce(expr: &Expr) -> bool {
    match expr {
        Expr::Raw(_) => false,
        Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::Addr64 { .. } => true,
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
            !calllike_may_have_side_effects(func) && args.iter().all(expr_is_pure_for_dce)
        }
        Expr::Intrinsic { args, .. } => args.iter().all(expr_is_pure_for_dce),
        Expr::Load { .. } | Expr::Index { .. } => false,
        Expr::WidePtr { base, offset } => {
            expr_is_pure_for_dce(base) && expr_is_pure_for_dce(offset)
        }
        Expr::Cast { expr, .. } => expr_is_pure_for_dce(expr),
    }
}

fn calllike_may_have_side_effects(func: &str) -> bool {
    matches!(
        func,
        "atomicAdd"
            | "atomicMin"
            | "atomicMax"
            | "atomicCAS"
            | "atomicExch"
            | "atomicAnd"
            | "atomicOr"
            | "atomicXor"
            | "atomicOp"
            | "__syncthreads"
            | "__syncwarp"
    ) || func.starts_with("local_store_")
}

fn collect_used_lvalue_vars(lvalue: &LValue, live: &mut BTreeSet<String>) {
    match lvalue {
        LValue::Raw(text) | LValue::Var(text) => collect_raw_identifiers(text, live),
        LValue::PtrLane { base, .. } => {
            live.insert(base.clone());
        }
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
        Expr::PtrLane { base, .. } => {
            live.insert(base.clone());
        }
        Expr::LaneExtract { value, .. } => collect_used_expr_vars(value, live),
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
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_used_expr_vars(arg, live);
            }
        }
        Expr::Load { addr, .. } => collect_used_expr_vars(addr, live),
        Expr::WidePtr { base, offset } => {
            collect_used_expr_vars(base, live);
            collect_used_expr_vars(offset, live);
        }
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

fn contains_unstructured_control_flow(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Label { .. } | Stmt::Goto(_) => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            stmts.iter().any(contains_unstructured_control_flow)
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            contains_unstructured_control_flow(then_branch)
                || else_branch
                    .as_deref()
                    .map(contains_unstructured_control_flow)
                    .unwrap_or(false)
        }
        Stmt::Loop { body, .. } => contains_unstructured_control_flow(body),
        Stmt::Switch { cases, default, .. } => {
            cases
                .iter()
                .any(|(_, body)| contains_unstructured_control_flow(body))
                || default
                    .as_deref()
                    .map(contains_unstructured_control_flow)
                    .unwrap_or(false)
        }
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::Assign { .. }
        | Stmt::ExprStmt(_)
        | Stmt::Empty => false,
    }
}

fn recover_rcp_division_stmt_tree(
    stmt: Stmt,
    defs: &mut HashMap<String, Expr>,
    future_used: &BTreeSet<String>,
) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            let entry_defs = defs.clone();
            let mut local_defs = defs.clone();
            let mut local_future = vec![BTreeSet::new(); stmts.len()];
            let mut live_tail = future_used.clone();
            for (idx, stmt) in stmts.iter().enumerate().rev() {
                local_future[idx] = live_tail.clone();
                collect_stmt_used_vars(stmt, &mut live_tail);
            }
            let rewritten = simplify_trivial_stmt(
                dce_stmt(
                    Stmt::Sequence(
                        stmts
                            .into_iter()
                            .enumerate()
                            .map(|(idx, stmt)| {
                                recover_rcp_division_stmt_tree(
                                    stmt,
                                    &mut local_defs,
                                    &local_future[idx],
                                )
                            })
                            .collect(),
                    ),
                    future_used.clone(),
                )
                .0,
            );
            let mut final_defs = entry_defs;
            update_linear_defs(&rewritten, &mut final_defs);
            *defs = final_defs;
            rewritten
        }
        Stmt::Block(stmts) => {
            let entry_defs = defs.clone();
            let mut local_defs = defs.clone();
            let mut local_future = vec![BTreeSet::new(); stmts.len()];
            let mut live_tail = future_used.clone();
            for (idx, stmt) in stmts.iter().enumerate().rev() {
                local_future[idx] = live_tail.clone();
                collect_stmt_used_vars(stmt, &mut live_tail);
            }
            let rewritten = simplify_trivial_stmt(
                dce_stmt(
                    Stmt::Block(
                        stmts
                            .into_iter()
                            .enumerate()
                            .map(|(idx, stmt)| {
                                recover_rcp_division_stmt_tree(
                                    stmt,
                                    &mut local_defs,
                                    &local_future[idx],
                                )
                            })
                            .collect(),
                    ),
                    future_used.clone(),
                )
                .0,
            );
            let mut final_defs = entry_defs;
            update_linear_defs(&rewritten, &mut final_defs);
            *defs = final_defs;
            rewritten
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let candidate = Stmt::If {
                condition: condition.clone(),
                then_branch: then_branch.clone(),
                else_branch: else_branch.clone(),
            };
            if let Some(rewritten) = recover_fchk_division_stmt(&candidate, defs, future_used) {
                update_linear_defs(&rewritten, defs);
                return rewritten;
            }

            let mut then_defs = defs.clone();
            let then_branch = Box::new(recover_rcp_division_stmt_tree(
                *then_branch,
                &mut then_defs,
                future_used,
            ));
            let else_branch = else_branch.map(|branch| {
                let mut else_defs = defs.clone();
                Box::new(recover_rcp_division_stmt_tree(
                    *branch,
                    &mut else_defs,
                    future_used,
                ))
            });
            let rewritten = Stmt::If {
                condition,
                then_branch,
                else_branch,
            };
            update_linear_defs(&rewritten, defs);
            rewritten
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let mut body_defs = defs.clone();
            let rewritten = Stmt::Loop {
                kind,
                condition,
                body: Box::new(recover_rcp_division_stmt_tree(
                    *body,
                    &mut body_defs,
                    future_used,
                )),
            };
            update_linear_defs(&rewritten, defs);
            rewritten
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            let cases = cases
                .into_iter()
                .map(|(label, body)| {
                    let mut case_defs = defs.clone();
                    (
                        label,
                        recover_rcp_division_stmt_tree(body, &mut case_defs, future_used),
                    )
                })
                .collect();
            let default = default.map(|body| {
                let mut default_defs = defs.clone();
                Box::new(recover_rcp_division_stmt_tree(
                    *body,
                    &mut default_defs,
                    future_used,
                ))
            });
            let rewritten = Stmt::Switch {
                discriminant,
                cases,
                default,
            };
            update_linear_defs(&rewritten, defs);
            rewritten
        }
        other => {
            if let Some(rewritten) = recover_fchk_division_assign(&other, defs) {
                update_linear_defs(&rewritten, defs);
                return rewritten;
            }
            update_linear_defs(&other, defs);
            other
        }
    }
}

fn recover_fchk_division_stmt(
    stmt: &Stmt,
    defs: &HashMap<String, Expr>,
    future_used: &BTreeSet<String>,
) -> Option<Stmt> {
    let Stmt::If {
        condition,
        then_branch,
        else_branch: Some(else_branch),
    } = stmt
    else {
        return None;
    };
    let (pred_name, fast_branch, slow_branch) = match condition {
        Expr::Reg(name) => (name.clone(), else_branch.as_ref(), then_branch.as_ref()),
        Expr::Unary { op, arg } if op == "!" => (
            match_var_name(arg)?,
            then_branch.as_ref(),
            else_branch.as_ref(),
        ),
        _ => return None,
    };
    let fast_assigns = match_assign_list(fast_branch)?;
    let slow_assigns = match_assign_list(slow_branch)?;
    let Some(pred_expr) = defs.get(&pred_name) else {
        return recover_division_merge_without_pred_def(&fast_assigns, defs);
    };
    let (fchk_num, fchk_den) = match_fchk_expr(pred_expr)?;
    let fchk_num = normalize_match_expr(&fchk_num, defs, MATCH_RESOLVE_DEPTH);
    let fchk_den = normalize_match_expr(&fchk_den, defs, MATCH_RESOLVE_DEPTH);
    let mut slow_by_dst = HashMap::new();
    for (dst, src) in &slow_assigns {
        slow_by_dst.insert(dst.render(), src.clone());
    }

    let cond_pred = Expr::Reg(pred_name);
    let mut recovered_key = None::<String>;
    let mut recovered_stmt = None::<Stmt>;
    let mut out = Vec::new();

    for (dst_fast, src_fast) in &fast_assigns {
        let dst_key = dst_fast.render();
        let Some(src_slow) = slow_by_dst.get(&dst_key) else {
            if let Some(name) = lvalue_symbol_name(dst_fast) {
                if future_used.contains(&name) {
                    return None;
                }
            }
            continue;
        };

        if let Some((num, den)) = match_division_expr(src_fast, defs) {
            let num_matches = same_match_expr(&fchk_num, &num);
            let den_matches = same_match_expr(&fchk_den, &den);
            if !num_matches && !den_matches {
                return None;
            }
            recovered_key = Some(dst_key);
            recovered_stmt = Some(Stmt::Assign {
                dst: dst_fast.clone(),
                src: Expr::Binary {
                    op: "/".to_string(),
                    lhs: Box::new(fchk_num.clone()),
                    rhs: Box::new(prefer_matched_division_denominator(&fchk_den, &den)),
                },
            });
            continue;
        }
        if match_fchk_guided_division_expr(src_fast, defs, &fchk_num, &fchk_den) {
            recovered_key = Some(dst_key);
            recovered_stmt = Some(Stmt::Assign {
                dst: dst_fast.clone(),
                src: Expr::Binary {
                    op: "/".to_string(),
                    lhs: Box::new(fchk_num.clone()),
                    rhs: Box::new(fchk_den.clone()),
                },
            });
            continue;
        }

        if let Some(name) = lvalue_symbol_name(dst_fast) {
            if !future_used.contains(&name) {
                continue;
            }
        }

        out.push(Stmt::Assign {
            dst: dst_fast.clone(),
            src: Expr::Ternary {
                cond: Box::new(cond_pred.clone()),
                then_expr: Box::new(src_slow.clone()),
                else_expr: Box::new(src_fast.clone()),
            },
        });
    }

    let recovered_key = recovered_key?;
    let recovered_stmt = recovered_stmt?;
    for (dst_slow, _) in &slow_assigns {
        let dst_key = dst_slow.render();
        if dst_key == recovered_key || fast_assigns.iter().any(|(dst, _)| dst.render() == dst_key) {
            continue;
        }
        if let Some(name) = lvalue_symbol_name(dst_slow) {
            if future_used.contains(&name) {
                return None;
            }
        }
    }
    out.push(recovered_stmt);
    Some(match out.len() {
        0 => Stmt::Empty,
        1 => out.into_iter().next().unwrap_or(Stmt::Empty),
        _ => Stmt::Sequence(out),
    })
}

fn recover_fchk_division_assign(stmt: &Stmt, defs: &HashMap<String, Expr>) -> Option<Stmt> {
    let Stmt::Assign { dst, src } = stmt else {
        return None;
    };
    let (num, den) = match_rcp_division_expr(src, defs)?;
    if !defs.values().any(|expr| {
        let Some((fchk_num, fchk_den)) = match_fchk_expr(expr) else {
            return false;
        };
        let fchk_num = normalize_match_expr(&fchk_num, defs, MATCH_RESOLVE_DEPTH);
        let fchk_den = normalize_match_expr(&fchk_den, defs, MATCH_RESOLVE_DEPTH);
        same_match_expr(&fchk_num, &num) || same_match_expr(&fchk_den, &den)
    }) {
        return None;
    }

    Some(Stmt::Assign {
        dst: dst.clone(),
        src: Expr::Binary {
            op: "/".to_string(),
            lhs: Box::new(num),
            rhs: Box::new(den),
        },
    })
}

fn recover_division_merge_without_pred_def(
    fast_assigns: &[(LValue, Expr)],
    defs: &HashMap<String, Expr>,
) -> Option<Stmt> {
    for (dst_fast, src_fast) in fast_assigns {
        let Some((num, den)) = match_division_expr(src_fast, defs) else {
            continue;
        };
        return Some(Stmt::Assign {
            dst: dst_fast.clone(),
            src: Expr::Binary {
                op: "/".to_string(),
                lhs: Box::new(num),
                rhs: Box::new(den),
            },
        });
    }
    None
}

fn prefer_matched_division_denominator(fchk_den: &Expr, matched_den: &Expr) -> Expr {
    if expr_looks_unresolved_constmem(fchk_den) && !expr_looks_unresolved_constmem(matched_den) {
        matched_den.clone()
    } else {
        fchk_den.clone()
    }
}

fn expr_looks_unresolved_constmem(expr: &Expr) -> bool {
    matches!(expr, Expr::Raw(text) | Expr::Imm(text) | Expr::Reg(text) if {
        let text = text.trim();
        text.starts_with("ConstMem(") || text.starts_with("c[0x")
    })
}

fn match_assign_list(stmt: &Stmt) -> Option<Vec<(LValue, Expr)>> {
    match stmt {
        Stmt::Assign { dst, src } => Some(vec![(dst.clone(), src.clone())]),
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            let mut assigns = Vec::new();
            for stmt in stmts {
                match stmt {
                    Stmt::Empty => {}
                    Stmt::ExprStmt(expr) if is_runtime_slowpath_expr(expr) => {}
                    Stmt::Assign { dst, src } => assigns.push((dst.clone(), src.clone())),
                    _ => return None,
                }
            }
            Some(assigns)
        }
        _ => None,
    }
}

fn is_runtime_slowpath_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::CallLike { func, args } if func == "CALL.REL.NOINC" && args.is_empty()
    )
}

fn match_fchk_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    match expr {
        Expr::CallLike { func, args } if func == "FCHK" && args.len() == 2 => {
            Some((args[0].clone(), args[1].clone()))
        }
        _ => None,
    }
}

fn match_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    match_direct_division_expr(expr, defs).or_else(|| match_rcp_division_expr(expr, defs))
}

fn match_fchk_guided_division_expr(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    expected_num: &Expr,
    expected_den: &Expr,
) -> bool {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let Some((lhs, rhs)) = match_binary_expr(&expr, "+", defs) else {
        return false;
    };
    match_fchk_guided_division_parts(&lhs, &rhs, defs, expected_num, expected_den)
        || match_fchk_guided_division_parts(&rhs, &lhs, defs, expected_num, expected_den)
}

fn match_fchk_guided_division_parts(
    scaled_expr: &Expr,
    refined_term: &Expr,
    defs: &HashMap<String, Expr>,
    expected_num: &Expr,
    expected_den: &Expr,
) -> bool {
    let scaled_resolved = normalize_match_expr(scaled_expr, defs, MATCH_RESOLVE_DEPTH);
    let expected_num = normalize_match_expr(expected_num, defs, MATCH_RESOLVE_DEPTH);
    let expected_den = normalize_match_expr(expected_den, defs, MATCH_RESOLVE_DEPTH);
    let Some(scaled_pairs) = match_mul_factor_pairs(&scaled_resolved, defs) else {
        return false;
    };
    let Some(refined_pairs) = match_mul_factor_pairs(refined_term, defs) else {
        return false;
    };
    for (approx_a, num) in &scaled_pairs {
        if !same_match_expr(num, &expected_num) {
            continue;
        }
        for (approx_b, corr) in &refined_pairs {
            if !same_match_expr(approx_a, approx_b) {
                continue;
            }
            if match_newton_correction_with_den(
                corr,
                &scaled_resolved,
                &expected_num,
                &expected_den,
                defs,
            ) {
                return true;
            }
        }
    }
    false
}

fn match_newton_correction_with_den(
    expr: &Expr,
    expected_scaled: &Expr,
    expected_num: &Expr,
    expected_den: &Expr,
    defs: &HashMap<String, Expr>,
) -> bool {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let Some((lhs, rhs)) = match_binary_expr(&expr, "+", defs) else {
        return false;
    };
    for (neg_mul_term, num_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        if !same_match_expr(&num_term, expected_num) {
            continue;
        }
        let Some((corr_den, corr_scaled)) = match_neg_mul_expr(&neg_mul_term, defs) else {
            continue;
        };
        if same_match_expr(&corr_den, expected_den)
            && same_match_expr(&corr_scaled, expected_scaled)
        {
            return true;
        }
    }
    false
}

fn match_direct_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "/" => Some((
            normalize_match_expr(&lhs, defs, MATCH_RESOLVE_DEPTH),
            normalize_match_expr(&rhs, defs, MATCH_RESOLVE_DEPTH),
        )),
        _ => None,
    }
}

fn match_rcp_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    match_rcp_division_parts(&lhs, &rhs, defs)
        .or_else(|| match_rcp_division_parts(&rhs, &lhs, defs))
}

fn match_rcp_division_parts(
    scaled_expr: &Expr,
    refined_term: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr)> {
    let scaled_resolved = normalize_match_expr(scaled_expr, defs, MATCH_RESOLVE_DEPTH);
    let scaled_pairs = match_mul_factor_pairs(&scaled_resolved, defs)?;
    let refined_pairs = match_mul_factor_pairs(refined_term, defs)?;
    for (recip_a, num) in &scaled_pairs {
        for (recip_b, corr) in &refined_pairs {
            if !same_match_expr(recip_a, recip_b) {
                continue;
            }
            let den = match_rcp_refine_expr(recip_a, defs)?;
            let (corr_den, corr_scaled, corr_num) = match_newton_correction_expr(corr, defs)?;
            if same_match_expr(&corr_scaled, &scaled_resolved) && same_match_expr(&corr_num, num) {
                let chosen_den = if same_match_expr(&corr_den, &den) {
                    den
                } else {
                    corr_den
                };
                return Some((num.clone(), chosen_den));
            }
        }
    }
    None
}

fn match_rcp_refine_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    for (mul_term, approx_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        let factor_pairs = match_mul_factor_pairs(&mul_term, defs)?;
        for (approx_a, err) in factor_pairs {
            let (err_den, err_approx) = match_neg_mul_plus_one_expr(&err, defs)?;
            if same_recip_seed_expr(&err_approx, &approx_a)
                || (same_recip_seed_expr(&approx_term, &approx_a)
                    && same_recip_seed_expr(&err_approx, &approx_a))
            {
                return Some(err_den);
            }
        }
    }
    None
}

fn match_newton_correction_expr(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    for (neg_mul_term, num_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        let (den, scaled) = match_neg_mul_expr(&neg_mul_term, defs)?;
        return Some((
            den,
            scaled,
            normalize_match_expr(&num_term, defs, MATCH_RESOLVE_DEPTH),
        ));
    }
    None
}

fn match_neg_mul_plus_one_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    for (neg_mul_term, one_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        if !expr_is_one(&normalize_match_expr(&one_term, defs, MATCH_RESOLVE_DEPTH)) {
            continue;
        }
        let (den, approx) = match_neg_mul_expr(&neg_mul_term, defs)?;
        return Some((den, approx));
    }
    None
}

fn match_neg_mul_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    let (lhs, rhs) = match_mul_expr(&expr, defs)?;
    if let Some(den) = match_negated_expr(&lhs, defs) {
        return Some((den, rhs));
    }
    if let Some(den) = match_negated_expr(&rhs, defs) {
        return Some((den, lhs));
    }
    None
}

fn match_negated_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    match expr {
        Expr::Unary { op, arg } if op == "-" => {
            Some(normalize_match_expr(&arg, defs, MATCH_RESOLVE_DEPTH))
        }
        Expr::Imm(text) => text
            .strip_prefix('-')
            .filter(|inner| !inner.is_empty())
            .map(|inner| Expr::Imm(inner.to_string())),
        _ => None,
    }
}

fn match_mul_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "*" => Some((
            normalize_match_expr(&lhs, defs, MATCH_RESOLVE_DEPTH),
            normalize_match_expr(&rhs, defs, MATCH_RESOLVE_DEPTH),
        )),
        _ => None,
    }
}

fn match_mul_factor_pairs(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Vec<(Expr, Expr)>> {
    let (lhs, rhs) = match_mul_expr(expr, defs)?;
    if same_match_expr(&lhs, &rhs) {
        Some(vec![(lhs, rhs)])
    } else {
        Some(vec![(lhs.clone(), rhs.clone()), (rhs, lhs)])
    }
}

fn match_binary_expr(expr: &Expr, op: &str, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, MATCH_RESOLVE_DEPTH);
    match expr {
        Expr::Binary {
            op: actual,
            lhs,
            rhs,
        } if actual == op => Some((
            normalize_match_expr(&lhs, defs, MATCH_RESOLVE_DEPTH),
            normalize_match_expr(&rhs, defs, MATCH_RESOLVE_DEPTH),
        )),
        _ => None,
    }
}

fn normalize_match_expr(expr: &Expr, defs: &HashMap<String, Expr>, depth: usize) -> Expr {
    rewrite_match_expr(resolve_named_expr(expr, defs, depth))
}

fn resolve_named_expr(expr: &Expr, defs: &HashMap<String, Expr>, depth: usize) -> Expr {
    if depth == 0 {
        return expr.clone();
    }
    if let Some(name) = match_var_name(expr) {
        if let Some(mapped) = defs.get(&name) {
            return resolve_named_expr(mapped, defs, depth - 1);
        }
    }
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op: op.clone(),
            arg: Box::new(resolve_named_expr(arg, defs, depth)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: op.clone(),
            lhs: Box::new(resolve_named_expr(lhs, defs, depth)),
            rhs: Box::new(resolve_named_expr(rhs, defs, depth)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(resolve_named_expr(cond, defs, depth)),
            then_expr: Box::new(resolve_named_expr(then_expr, defs, depth)),
            else_expr: Box::new(resolve_named_expr(else_expr, defs, depth)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func: func.clone(),
            args: args
                .iter()
                .map(|arg| resolve_named_expr(arg, defs, depth))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op: op.clone(),
            args: args
                .iter()
                .map(|arg| resolve_named_expr(arg, defs, depth))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty: ty.clone(),
            addr: Box::new(resolve_named_expr(addr, defs, depth)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(resolve_named_expr(base, defs, depth)),
            offset: Box::new(resolve_named_expr(offset, defs, depth)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(resolve_named_expr(lo, defs, depth)),
            hi: Box::new(resolve_named_expr(hi, defs, depth)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty: ty.clone(),
            expr: Box::new(resolve_named_expr(expr, defs, depth)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(resolve_named_expr(base, defs, depth)),
            index: Box::new(resolve_named_expr(index, defs, depth)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(resolve_named_expr(value, defs, depth)),
            lane: *lane,
        },
        other => other.clone(),
    }
}

fn rewrite_match_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(rewrite_match_expr(*arg)),
        },
        Expr::Binary { op, lhs, rhs } => {
            let lhs = rewrite_match_expr(*lhs);
            let rhs = rewrite_match_expr(*rhs);
            match op.as_str() {
                "+" if expr_is_zero(&lhs) => rhs,
                "+" if expr_is_zero(&rhs) => lhs,
                _ => Expr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            }
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(rewrite_match_expr(*cond)),
            then_expr: Box::new(rewrite_match_expr(*then_expr)),
            else_expr: Box::new(rewrite_match_expr(*else_expr)),
        },
        Expr::CallLike { func, args } if is_fma_like(&func) && args.len() == 3 => {
            let mut args = args.into_iter().map(rewrite_match_expr);
            let lhs = args.next().unwrap_or_else(|| Expr::Imm("0".to_string()));
            let rhs = args.next().unwrap_or_else(|| Expr::Imm("0".to_string()));
            let acc = args.next().unwrap_or_else(|| Expr::Imm("0".to_string()));
            let mul = Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
            if expr_is_zero(&acc) {
                mul
            } else {
                Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(mul),
                    rhs: Box::new(acc),
                }
            }
        }
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args.into_iter().map(rewrite_match_expr).collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args.into_iter().map(rewrite_match_expr).collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(rewrite_match_expr(*addr)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(rewrite_match_expr(*base)),
            offset: Box::new(rewrite_match_expr(*offset)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(rewrite_match_expr(*lo)),
            hi: Box::new(rewrite_match_expr(*hi)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(rewrite_match_expr(*expr)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(rewrite_match_expr(*base)),
            index: Box::new(rewrite_match_expr(*index)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(rewrite_match_expr(*value)),
            lane,
        },
        other => other,
    }
}

fn is_fma_like(func: &str) -> bool {
    matches!(
        func,
        "fmaf" | "__fmaf_rn" | "__fmaf_rd" | "__fmaf_ru" | "__fmaf_rz"
    )
}

fn expr_is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Imm(text) if text == "0" || text == "0.0")
}

fn match_var_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Reg(name) => Some(name.clone()),
        _ => None,
    }
}

fn same_match_expr(lhs: &Expr, rhs: &Expr) -> bool {
    canonical_match_expr(lhs) == canonical_match_expr(rhs)
}

fn same_recip_seed_expr(lhs: &Expr, rhs: &Expr) -> bool {
    same_match_expr(lhs, rhs)
}

fn canonical_match_expr(expr: &Expr) -> String {
    match expr {
        Expr::Binary { op, .. } if matches!(op.as_str(), "+" | "*") => {
            let mut terms = Vec::new();
            collect_assoc_match_terms(expr, op, &mut terms);
            let mut keys = terms
                .into_iter()
                .map(canonical_match_expr)
                .collect::<Vec<_>>();
            keys.sort();
            format!("{}({})", op, keys.join(","))
        }
        Expr::Unary { op, arg } => format!("{}{}", op, canonical_match_expr(arg)),
        Expr::CallLike { func, args } => format!(
            "{}({})",
            func,
            args.iter()
                .map(canonical_match_expr)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Expr::Intrinsic { op, args } => format!(
            "{}({})",
            op.render_name(),
            args.iter()
                .map(canonical_match_expr)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Expr::Load { ty, addr } => format!(
            "load:{}:{}",
            ty.as_deref().unwrap_or("_"),
            canonical_match_expr(addr)
        ),
        Expr::WidePtr { base, offset } => {
            format!(
                "wide({}, {})",
                canonical_match_expr(base),
                canonical_match_expr(offset)
            )
        }
        Expr::Addr64 { lo, hi } => {
            format!(
                "addr64({}, {})",
                canonical_match_expr(lo),
                canonical_match_expr(hi)
            )
        }
        Expr::Cast { ty, expr } => format!("cast:{}:{}", ty, canonical_match_expr(expr)),
        Expr::Index { base, index } => {
            format!(
                "idx({}, {})",
                canonical_match_expr(base),
                canonical_match_expr(index)
            )
        }
        Expr::LaneExtract { value, lane } => {
            format!("lane:{:?}:{}", lane, canonical_match_expr(value))
        }
        _ => expr.render(),
    }
}

fn collect_assoc_match_terms<'a>(expr: &'a Expr, op: &str, out: &mut Vec<&'a Expr>) {
    if let Expr::Binary {
        op: actual,
        lhs,
        rhs,
    } = expr
    {
        if actual == op {
            collect_assoc_match_terms(lhs, op, out);
            collect_assoc_match_terms(rhs, op, out);
            return;
        }
    }
    out.push(expr);
}

fn expr_is_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Imm(text) if text == "1")
}

fn update_linear_defs(stmt: &Stmt, defs: &mut HashMap<String, Expr>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            let mut local = defs.clone();
            for stmt in stmts {
                update_linear_defs(stmt, &mut local);
            }
            *defs = local;
        }
        Stmt::Assign { dst, src } => {
            if let Some(name) = lvalue_symbol_name(dst) {
                defs.insert(name, src.clone());
            } else {
                defs.clear();
            }
        }
        Stmt::ExprStmt(_) | Stmt::Empty => {}
        _ => defs.clear(),
    }
}

fn collect_stmt_used_vars(stmt: &Stmt, live: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_stmt_used_vars(stmt, live);
            }
        }
        Stmt::Label { body, .. } => collect_stmt_used_vars(body, live),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_used_expr_vars(condition, live);
            collect_stmt_used_vars(then_branch, live);
            if let Some(else_branch) = else_branch {
                collect_stmt_used_vars(else_branch, live);
            }
        }
        Stmt::Loop {
            condition, body, ..
        } => {
            if let Some(condition) = condition {
                collect_used_expr_vars(condition, live);
            }
            collect_stmt_used_vars(body, live);
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            if let Some(discriminant) = discriminant {
                collect_used_expr_vars(discriminant, live);
            }
            for (_, body) in cases {
                collect_stmt_used_vars(body, live);
            }
            if let Some(default) = default {
                collect_stmt_used_vars(default, live);
            }
        }
        Stmt::Assign { dst, src } => {
            collect_used_lvalue_vars(dst, live);
            collect_used_expr_vars(src, live);
        }
        Stmt::ExprStmt(expr) => collect_used_expr_vars(expr, live),
        Stmt::Return(expr) => {
            if let Some(expr) = expr {
                collect_used_expr_vars(expr, live);
            }
        }
        Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Empty => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::LoopKind;

    #[test]
    fn removes_dead_pure_helper_assignments() {
        let stmt = Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var("tmp0".to_string()),
                src: Expr::CallLike {
                    func: "IADD.64".to_string(),
                    args: vec![Expr::Reg("a".to_string()), Expr::Reg("b".to_string())],
                },
            },
            Stmt::Return(None),
        ]);
        assert_eq!(
            prune_dead_pure_stmt(stmt),
            Stmt::Sequence(vec![Stmt::Return(None)])
        );
    }

    #[test]
    fn removes_dead_pure_expr_statements() {
        let stmt = Stmt::Sequence(vec![
            Stmt::ExprStmt(Expr::CallLike {
                func: "CALL.REL.NOINC".to_string(),
                args: Vec::new(),
            }),
            Stmt::Return(None),
        ]);
        assert_eq!(
            prune_dead_pure_stmt(stmt),
            Stmt::Sequence(vec![Stmt::Return(None)])
        );
    }

    #[test]
    fn keeps_barrier_expr_statements_live() {
        let stmt = Stmt::Sequence(vec![
            Stmt::ExprStmt(Expr::CallLike {
                func: "__syncthreads".to_string(),
                args: Vec::new(),
            }),
            Stmt::Return(None),
        ]);
        assert_eq!(prune_dead_pure_stmt(stmt.clone()), stmt,);
    }

    #[test]
    fn keeps_loops_live_until_fixpoint() {
        let stmt = Stmt::Loop {
            kind: LoopKind::DoWhile,
            condition: Some(Expr::Reg("x".to_string())),
            body: Box::new(Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("tmp0".to_string()),
                    src: Expr::CallLike {
                        func: "IADD.64".to_string(),
                        args: vec![Expr::Reg("x".to_string()), Expr::Reg("y".to_string())],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("x".to_string()),
                    src: Expr::Reg("tmp0".to_string()),
                },
            ])),
        };
        assert_eq!(prune_dead_pure_stmt(stmt.clone()), stmt);
    }

    #[test]
    fn prunes_empty_ifs_after_dead_code_cleanup() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Empty),
                    else_branch: Some(Box::new(Stmt::Block(Vec::new()))),
                },
                Stmt::Return(None),
            ]),
        };
        assert_eq!(prune_dead_pure_defs(function).body, Stmt::Return(None),);
    }

    #[test]
    fn canonicalize_recovers_fchk_guarded_fmaf_division() {
        let seq = vec![
            Stmt::Assign {
                dst: LValue::Var("num".to_string()),
                src: Expr::Reg("arg0".to_string()),
            },
            Stmt::Assign {
                dst: LValue::Var("den".to_string()),
                src: Expr::Reg("arg1".to_string()),
            },
            Stmt::Assign {
                dst: LValue::Var("rcp0".to_string()),
                src: Expr::CallLike {
                    func: "rcp_approx".to_string(),
                    args: vec![Expr::Reg("den".to_string())],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("pred".to_string()),
                src: Expr::CallLike {
                    func: "FCHK".to_string(),
                    args: vec![Expr::Reg("num".to_string()), Expr::Reg("den".to_string())],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("err".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Unary {
                            op: "-".to_string(),
                            arg: Box::new(Expr::Reg("den".to_string())),
                        },
                        Expr::Reg("rcp0".to_string()),
                        Expr::Imm("1".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("rcp1".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("rcp0".to_string()),
                        Expr::Reg("err".to_string()),
                        Expr::Reg("rcp0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("scaled".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("rcp1".to_string()),
                        Expr::Reg("arg0".to_string()),
                        Expr::Imm("0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("corr".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Unary {
                            op: "-".to_string(),
                            arg: Box::new(Expr::Reg("den".to_string())),
                        },
                        Expr::Reg("scaled".to_string()),
                        Expr::Reg("arg0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("refined".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("rcp1".to_string()),
                        Expr::Reg("corr".to_string()),
                        Expr::Reg("scaled".to_string()),
                    ],
                },
            },
            Stmt::If {
                condition: Expr::Reg("pred".to_string()),
                then_branch: Box::new(Stmt::Block(vec![
                    Stmt::ExprStmt(Expr::CallLike {
                        func: "CALL.REL.NOINC".to_string(),
                        args: Vec::new(),
                    }),
                    Stmt::Assign {
                        dst: LValue::Var("out".to_string()),
                        src: Expr::Reg("rcp1".to_string()),
                    },
                ])),
                else_branch: Some(Box::new(Stmt::Block(vec![Stmt::Assign {
                    dst: LValue::Var("out".to_string()),
                    src: Expr::Reg("refined".to_string()),
                }]))),
            },
            Stmt::Return(Some(Expr::Reg("out".to_string()))),
        ];

        let mut defs = HashMap::new();
        for stmt in seq.iter().take(seq.len() - 2) {
            update_linear_defs(stmt, &mut defs);
        }
        assert!(match_rcp_division_expr(&Expr::Reg("refined".to_string()), &defs).is_some());
        assert!(recover_fchk_division_stmt(&seq[seq.len() - 2], &defs, &BTreeSet::new()).is_some());

        let rendered = canonicalize_function(StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(seq),
        })
        .body
        .render_with_indent(0);
        assert!(
            rendered.contains("out = arg0 / arg1;"),
            "got:\n{}",
            rendered
        );
        assert!(!rendered.contains("FCHK("), "got:\n{}", rendered);
        assert!(!rendered.contains("CALL.REL.NOINC"), "got:\n{}", rendered);
    }

    #[test]
    fn prune_dead_pure_stmt_removes_dead_fchk_inside_loop() {
        let stmt = Stmt::Loop {
            kind: LoopKind::DoWhile,
            condition: Some(Expr::Reg("keep".to_string())),
            body: Box::new(Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("p0".to_string()),
                    src: Expr::CallLike {
                        func: "FCHK".to_string(),
                        args: vec![Expr::Reg("arg0".to_string()), Expr::Reg("arg1".to_string())],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("out".to_string()),
                    src: Expr::Binary {
                        op: "/".to_string(),
                        lhs: Box::new(Expr::Reg("arg0".to_string())),
                        rhs: Box::new(Expr::Reg("arg1".to_string())),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("keep".to_string()),
                    src: Expr::Imm("0".to_string()),
                },
            ])),
        };
        let rendered = prune_dead_pure_stmt(stmt).render_with_indent(0);
        assert!(!rendered.contains("FCHK("), "got:\n{}", rendered);
    }

    #[test]
    fn canonicalize_recovers_fchk_guarded_division_from_constant_newton_seed() {
        let seq = vec![
            Stmt::Assign {
                dst: LValue::Var("num".to_string()),
                src: Expr::Reg("arg0".to_string()),
            },
            Stmt::Assign {
                dst: LValue::Var("pred".to_string()),
                src: Expr::CallLike {
                    func: "FCHK".to_string(),
                    args: vec![Expr::Reg("num".to_string()), Expr::Imm("9".to_string())],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("err".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Imm("0.1111111119389534".to_string()),
                        Expr::Imm("-9".to_string()),
                        Expr::Imm("1".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("approx".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("err".to_string()),
                        Expr::Imm("0.1111111119389534".to_string()),
                        Expr::Imm("0.1111111119389534".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("scaled".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("num".to_string()),
                        Expr::Reg("approx".to_string()),
                        Expr::Imm("0".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("corr".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("scaled".to_string()),
                        Expr::Imm("-9".to_string()),
                        Expr::Reg("num".to_string()),
                    ],
                },
            },
            Stmt::Assign {
                dst: LValue::Var("refined".to_string()),
                src: Expr::CallLike {
                    func: "fmaf".to_string(),
                    args: vec![
                        Expr::Reg("approx".to_string()),
                        Expr::Reg("corr".to_string()),
                        Expr::Reg("scaled".to_string()),
                    ],
                },
            },
            Stmt::If {
                condition: Expr::Reg("pred".to_string()),
                then_branch: Box::new(Stmt::Assign {
                    dst: LValue::Var("out".to_string()),
                    src: Expr::Reg("num".to_string()),
                }),
                else_branch: Some(Box::new(Stmt::Assign {
                    dst: LValue::Var("out".to_string()),
                    src: Expr::Reg("refined".to_string()),
                })),
            },
            Stmt::Return(Some(Expr::Reg("out".to_string()))),
        ];

        let rendered = canonicalize_function(StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(seq),
        })
        .body
        .render_with_indent(0);
        assert!(rendered.contains("out = arg0 / 9;"), "got:\n{}", rendered);
        assert!(!rendered.contains("FCHK("), "got:\n{}", rendered);
    }
}
