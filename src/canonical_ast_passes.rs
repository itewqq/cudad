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

use std::collections::BTreeSet;

use crate::ast::{Expr, LValue, Stmt, StructuredFunction};

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
                && else_branch
                    .as_ref()
                    .is_none_or(stmt_is_trivial_empty)
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
        assert_eq!(prune_dead_pure_stmt(stmt), Stmt::Sequence(vec![Stmt::Return(None)]));
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
        assert_eq!(prune_dead_pure_stmt(stmt), Stmt::Sequence(vec![Stmt::Return(None)]));
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
        assert_eq!(
            prune_dead_pure_defs(function).body,
            Stmt::Return(None),
        );
    }
}
