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
//! - bounded local recovery / condition-fold / dead-code sweeps
//! - backward liveness over structured statements
//! - loop-body fixpoint for structured loops
//! - pure-expression pruning for dead assignments and dead expression
//!   statements
//!
//! This module must not:
//! - perform regex/text cleanup
//! - rediscover memory facts from rendered output
//! - run the old fixed-point backend repair pipeline
//! - guess liveness across unstructured jump boundaries

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::ast::{Expr, LValue, Stmt, StructuredFunction};

const MATCH_RESOLVE_DEPTH: usize = 12;
const MAX_CANONICAL_SWEEPS: usize = 3;

pub fn canonicalize_function(function: StructuredFunction) -> StructuredFunction {
    let mut current = function;
    for _ in 0..MAX_CANONICAL_SWEEPS {
        let next = canonicalize_function_once(current.clone());
        if next == current {
            return next;
        }
        current = next;
    }
    current
}

fn canonicalize_function_once(function: StructuredFunction) -> StructuredFunction {
    let mut defs = HashMap::new();
    let body = recover_rcp_division_stmt_tree(function.body, &mut defs, &BTreeSet::new());
    prune_dead_pure_defs(StructuredFunction {
        params: function.params,
        locals: function.locals,
        body,
    })
}

pub fn prune_dead_pure_defs(function: StructuredFunction) -> StructuredFunction {
    let mut defs = HashMap::new();
    let body = fold_known_conditions_stmt(function.body, &mut defs);
    let mut goto_targets = BTreeSet::new();
    collect_goto_targets(&body, &mut goto_targets);
    let body = simplify_local_cleanup_regions(body, &goto_targets, &BTreeSet::new());
    let body = cleanup_local_control_flow(body);
    StructuredFunction {
        params: function.params,
        locals: function.locals,
        body,
    }
}

pub fn canonicalize_post_bind_control_flow(function: StructuredFunction) -> StructuredFunction {
    StructuredFunction {
        params: function.params,
        locals: function.locals,
        body: cleanup_local_control_flow(function.body),
    }
}

pub fn prune_dead_pure_stmt(stmt: Stmt) -> Stmt {
    prune_dead_pure_stmt_with_live_out(stmt, BTreeSet::new())
}

fn prune_dead_pure_stmt_with_live_out(stmt: Stmt, live_out: BTreeSet<String>) -> Stmt {
    dce_stmt(stmt, live_out).0
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

fn fold_known_conditions_stmt(stmt: Stmt, defs: &mut HashMap<String, Expr>) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            let mut local_defs = defs.clone();
            let rewritten = Stmt::Sequence(
                stmts
                    .into_iter()
                    .map(|stmt| fold_known_conditions_stmt(stmt, &mut local_defs))
                    .collect(),
            );
            *defs = local_defs;
            rewritten
        }
        Stmt::Block(stmts) => {
            let mut local_defs = defs.clone();
            let rewritten = Stmt::Block(
                stmts
                    .into_iter()
                    .map(|stmt| fold_known_conditions_stmt(stmt, &mut local_defs))
                    .collect(),
            );
            *defs = local_defs;
            rewritten
        }
        Stmt::Label { name, body } => {
            let mut label_defs = HashMap::new();
            let body = fold_known_conditions_stmt(*body, &mut label_defs);
            defs.clear();
            Stmt::Label {
                name,
                body: Box::new(body),
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = normalize_match_expr(&condition, defs, MATCH_RESOLVE_DEPTH);
            if let Some(value) = expr_const_bool(&condition) {
                return if value {
                    fold_known_conditions_stmt(*then_branch, defs)
                } else if let Some(else_branch) = else_branch {
                    fold_known_conditions_stmt(*else_branch, defs)
                } else {
                    Stmt::Empty
                };
            }

            let mut then_defs = defs.clone();
            let then_branch = Box::new(fold_known_conditions_stmt(*then_branch, &mut then_defs));
            let else_branch = else_branch.map(|branch| {
                let mut else_defs = defs.clone();
                Box::new(fold_known_conditions_stmt(*branch, &mut else_defs))
            });
            defs.clear();
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let condition = condition
                .map(|condition| normalize_match_expr(&condition, defs, MATCH_RESOLVE_DEPTH));
            let mut body_defs = HashMap::new();
            let body = Box::new(fold_known_conditions_stmt(*body, &mut body_defs));
            defs.clear();
            Stmt::Loop {
                kind,
                condition,
                body,
            }
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            let discriminant =
                discriminant.map(|expr| normalize_match_expr(&expr, defs, MATCH_RESOLVE_DEPTH));
            let cases = cases
                .into_iter()
                .map(|(label, body)| {
                    let mut case_defs = HashMap::new();
                    (label, fold_known_conditions_stmt(body, &mut case_defs))
                })
                .collect();
            let default = default.map(|body| {
                let mut default_defs = HashMap::new();
                Box::new(fold_known_conditions_stmt(*body, &mut default_defs))
            });
            defs.clear();
            Stmt::Switch {
                discriminant,
                cases,
                default,
            }
        }
        Stmt::Goto(_) | Stmt::Break | Stmt::Continue | Stmt::Return(_) => {
            defs.clear();
            stmt
        }
        other => {
            update_linear_defs(&other, defs);
            other
        }
    }
}

fn simplify_local_cleanup_regions(
    stmt: Stmt,
    goto_targets: &BTreeSet<String>,
    live_out: &BTreeSet<String>,
) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            simplify_local_cleanup_sequence(stmts, false, goto_targets, live_out)
        }
        Stmt::Block(stmts) => simplify_local_cleanup_sequence(stmts, true, goto_targets, live_out),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(simplify_local_cleanup_regions(*body, goto_targets, live_out)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => simplify_trivial_stmt(
            Stmt::If {
                condition,
                then_branch: Box::new(simplify_local_cleanup_regions(
                    *then_branch,
                    goto_targets,
                    live_out,
                )),
                else_branch: else_branch.map(|branch| {
                    Box::new(simplify_local_cleanup_regions(*branch, goto_targets, live_out))
                }),
            },
            goto_targets,
        ),
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let mut body_live_out = live_out.clone();
            if let Some(condition) = &condition {
                collect_used_expr_vars(condition, &mut body_live_out);
            }
            simplify_trivial_stmt(
                Stmt::Loop {
                    kind,
                    condition,
                    body: Box::new(simplify_local_cleanup_regions(
                        *body,
                        goto_targets,
                        &body_live_out,
                    )),
                },
                goto_targets,
            )
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => simplify_trivial_stmt(
            Stmt::Switch {
                discriminant,
                cases: cases
                    .into_iter()
                    .map(|(label, body)| {
                        (label, simplify_local_cleanup_regions(body, goto_targets, live_out))
                    })
                    .collect(),
                default: default
                    .map(|body| Box::new(simplify_local_cleanup_regions(*body, goto_targets, live_out))),
            },
            goto_targets,
        ),
        other => other,
    }
}

fn simplify_local_cleanup_sequence(
    stmts: Vec<Stmt>,
    as_block: bool,
    goto_targets: &BTreeSet<String>,
    live_out: &BTreeSet<String>,
) -> Stmt {
    let mut future_used = vec![BTreeSet::new(); stmts.len()];
    let mut live_tail = live_out.clone();
    for (idx, stmt) in stmts.iter().enumerate().rev() {
        future_used[idx] = live_tail.clone();
        collect_stmt_used_vars(stmt, &mut live_tail);
    }

    let mut out = Vec::new();
    let mut chunk = Vec::new();
    let mut chunk_live_out = live_out.clone();

    let flush_chunk =
        |out: &mut Vec<Stmt>, chunk: &mut Vec<Stmt>, chunk_live_out: &mut BTreeSet<String>| {
        if chunk.is_empty() {
            return;
        }
        let chunk_stmt = if as_block {
            Stmt::Block(std::mem::take(chunk))
        } else {
            Stmt::Sequence(std::mem::take(chunk))
        };
        let simplified = simplify_trivial_stmt(
            prune_dead_pure_stmt_with_live_out(chunk_stmt, chunk_live_out.clone()),
            goto_targets,
        );
        match simplified {
            Stmt::Empty => {}
            Stmt::Sequence(inner) | Stmt::Block(inner) => out.extend(inner),
            other => out.push(other),
        }
        *chunk_live_out = live_out.clone();
    };

    for (idx, stmt) in stmts.into_iter().enumerate() {
        let stmt = simplify_local_cleanup_regions(stmt, goto_targets, &future_used[idx]);
        if let Stmt::Label { name, body } = stmt {
            flush_chunk(&mut out, &mut chunk, &mut chunk_live_out);
            let labeled = Stmt::Label {
                name,
                body: Box::new(*body),
            };
            if !stmt_is_trivial_empty(&labeled, goto_targets) {
                out.push(labeled);
            }
            continue;
        }
        if contains_unstructured_control_flow(&stmt) {
            flush_chunk(&mut out, &mut chunk, &mut chunk_live_out);
            if !stmt_is_trivial_empty(&stmt, goto_targets) {
                out.push(stmt);
            }
        } else {
            if chunk.is_empty() {
                chunk_live_out = future_used[idx].clone();
            }
            chunk.push(stmt);
        }
    }
    flush_chunk(&mut out, &mut chunk, &mut chunk_live_out);

    simplify_trivial_stmt(
        if as_block {
            Stmt::Block(out)
        } else {
            Stmt::Sequence(out)
        },
        goto_targets,
    )
}

fn simplify_trivial_stmt(stmt: Stmt, goto_targets: &BTreeSet<String>) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            let kept = stmts
                .into_iter()
                .map(|stmt| simplify_trivial_stmt(stmt, goto_targets))
                .filter(|stmt| !stmt_is_trivial_empty(stmt, goto_targets))
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
                .map(|stmt| simplify_trivial_stmt(stmt, goto_targets))
                .filter(|stmt| !stmt_is_trivial_empty(stmt, goto_targets))
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
            if let Some(value) = expr_const_bool(&condition) {
                return if value {
                    simplify_trivial_stmt(*then_branch, goto_targets)
                } else {
                    else_branch
                        .map(|branch| simplify_trivial_stmt(*branch, goto_targets))
                        .unwrap_or(Stmt::Empty)
                };
            }
            let then_branch = simplify_trivial_stmt(*then_branch, goto_targets);
            let else_branch =
                else_branch.map(|branch| simplify_trivial_stmt(*branch, goto_targets));
            if stmt_is_trivial_empty(&then_branch, goto_targets)
                && else_branch
                    .as_ref()
                    .is_none_or(|branch| stmt_is_trivial_empty(branch, goto_targets))
            {
                Stmt::Empty
            } else {
                Stmt::If {
                    condition,
                    then_branch: Box::new(then_branch),
                    else_branch: else_branch
                        .filter(|branch| !stmt_is_trivial_empty(branch, goto_targets))
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
            body: Box::new(simplify_trivial_stmt(*body, goto_targets)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, simplify_trivial_stmt(body, goto_targets)))
                .collect(),
            default: default.map(|body| Box::new(simplify_trivial_stmt(*body, goto_targets))),
        },
        Stmt::Label { name, body } => {
            let body = simplify_trivial_stmt(*body, goto_targets);
            if stmt_is_trivial_empty(&body, goto_targets) && !goto_targets.contains(&name) {
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

fn stmt_is_trivial_empty(stmt: &Stmt, goto_targets: &BTreeSet<String>) -> bool {
    match stmt {
        Stmt::Empty => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => stmts
            .iter()
            .all(|stmt| stmt_is_trivial_empty(stmt, goto_targets)),
        Stmt::Label { name, body } => {
            !goto_targets.contains(name) && stmt_is_trivial_empty(body, goto_targets)
        }
        _ => false,
    }
}

fn collect_goto_targets(stmt: &Stmt, out: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_goto_targets(stmt, out);
            }
        }
        Stmt::Label { body, .. } => collect_goto_targets(body, out),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_goto_targets(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_goto_targets(else_branch, out);
            }
        }
        Stmt::Loop { body, .. } => collect_goto_targets(body, out),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_goto_targets(body, out);
            }
            if let Some(default) = default {
                collect_goto_targets(default, out);
            }
        }
        Stmt::Goto(label) => {
            out.insert(label.clone());
        }
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::Assign { .. }
        | Stmt::ExprStmt(_)
        | Stmt::Empty => {}
    }
}

fn redirect_trivial_gotos(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => redirect_trivial_goto_sequence(stmts, false),
        Stmt::Block(stmts) => redirect_trivial_goto_sequence(stmts, true),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(redirect_trivial_gotos(*body)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(redirect_trivial_gotos(*then_branch)),
            else_branch: else_branch.map(|branch| Box::new(redirect_trivial_gotos(*branch))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(redirect_trivial_gotos(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, redirect_trivial_gotos(body)))
                .collect(),
            default: default.map(|body| Box::new(redirect_trivial_gotos(*body))),
        },
        other => other,
    }
}

fn redirect_trivial_goto_sequence(stmts: Vec<Stmt>, as_block: bool) -> Stmt {
    let mut stmts = stmts
        .into_iter()
        .map(redirect_trivial_gotos)
        .collect::<Vec<_>>();
    let alias_map = collect_trivial_label_aliases(&stmts);
    if !alias_map.is_empty() {
        stmts = stmts
            .into_iter()
            .map(|stmt| rewrite_stmt_gotos(stmt, &alias_map))
            .collect();
    }

    let mut out = Vec::new();
    for idx in 0..stmts.len() {
        let stmt = stmts[idx].clone();
        if let Stmt::Goto(label) = &stmt {
            if next_stmt_is_label(&stmts[idx + 1..], label) {
                continue;
            }
        }
        if redundant_fallthrough_guarded_goto(&stmt, &stmts[idx + 1..]) {
            continue;
        }
        out.push(stmt);
    }

    if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    }
}

/// `if (cond) goto L; if (!cond) { ... } L:` is equivalent to
/// `if (!cond) { ... } L:` because the guarded block is already skipped when
/// `cond` is true. Dropping the redundant goto keeps this cleanup purely
/// local and avoids relying on rendered-text post-processing.
fn redundant_fallthrough_guarded_goto(stmt: &Stmt, trailing: &[Stmt]) -> bool {
    let Some((condition, target)) = stmt_as_guarded_goto(stmt) else {
        return false;
    };
    if !expr_is_pure_for_control_cleanup(condition) {
        return false;
    }
    if next_stmt_is_label(trailing, target) {
        return true;
    }

    let Some((next_stmt, after_next)) = split_first_nonempty_stmt(trailing) else {
        return false;
    };
    if guarded_goto_duplicates_fallthrough_terminator(target, trailing) {
        return true;
    }
    let Stmt::If {
        condition: next_condition,
        else_branch,
        ..
    } = next_stmt
    else {
        return false;
    };
    if !else_branch
        .as_deref()
        .map(stmt_is_linear_empty)
        .unwrap_or(true)
    {
        return false;
    }
    if !expr_is_pure_for_control_cleanup(next_condition) {
        return false;
    }
    next_stmt_is_label(after_next, target)
        && exprs_are_boolean_complements(condition, next_condition)
}

fn stmt_as_guarded_goto(stmt: &Stmt) -> Option<(&Expr, &str)> {
    let Stmt::If {
        condition,
        then_branch,
        else_branch,
    } = stmt
    else {
        return None;
    };
    if !else_branch
        .as_deref()
        .map(stmt_is_linear_empty)
        .unwrap_or(true)
    {
        return None;
    }
    let target = single_goto_target(then_branch)?;
    Some((condition, target))
}

fn single_goto_target(stmt: &Stmt) -> Option<&str> {
    match stmt {
        Stmt::Goto(label) => Some(label.as_str()),
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            let mut nonempty = stmts.iter().filter(|stmt| !stmt_is_linear_empty(stmt));
            let only = nonempty.next()?;
            if nonempty.next().is_some() {
                return None;
            }
            single_goto_target(only)
        }
        _ => None,
    }
}

fn split_first_nonempty_stmt(stmts: &[Stmt]) -> Option<(&Stmt, &[Stmt])> {
    for (idx, stmt) in stmts.iter().enumerate() {
        if !stmt_is_linear_empty(stmt) {
            return Some((stmt, &stmts[idx + 1..]));
        }
    }
    None
}

fn first_nonempty_stmt_index(stmts: &[Stmt]) -> Option<usize> {
    stmts.iter().position(|stmt| !stmt_is_linear_empty(stmt))
}

fn guarded_goto_duplicates_fallthrough_terminator(target: &str, trailing: &[Stmt]) -> bool {
    let Some((fallthrough, after_fallthrough)) = split_first_nonempty_stmt(trailing) else {
        return false;
    };
    if !stmt_definitely_terminates(fallthrough) {
        return false;
    }
    let Some((next_stmt, _)) = split_first_nonempty_stmt(after_fallthrough) else {
        return false;
    };
    let Stmt::Label { name, body } = next_stmt else {
        return false;
    };
    name == target && body.as_ref() == fallthrough
}

/// Control-flow cleanup may erase an entire `if` statement, so keep the
/// predicate vocabulary narrower than general DCE purity. In particular we do
/// not fold away raw helpers, loads, or call-like nodes even if some of them
/// are otherwise classified as pure for expression pruning.
fn expr_is_pure_for_control_cleanup(expr: &Expr) -> bool {
    match expr {
        Expr::Raw(_) | Expr::CallLike { .. } | Expr::Load { .. } | Expr::Index { .. } => false,
        Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::Addr64 { .. } => true,
        Expr::Unary { arg, .. } => expr_is_pure_for_control_cleanup(arg),
        Expr::Binary { lhs, rhs, .. } => {
            expr_is_pure_for_control_cleanup(lhs) && expr_is_pure_for_control_cleanup(rhs)
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_is_pure_for_control_cleanup(cond)
                && expr_is_pure_for_control_cleanup(then_expr)
                && expr_is_pure_for_control_cleanup(else_expr)
        }
        Expr::Intrinsic { args, .. } => args.iter().all(expr_is_pure_for_control_cleanup),
        Expr::WidePtr { base, offset } => {
            expr_is_pure_for_control_cleanup(base) && expr_is_pure_for_control_cleanup(offset)
        }
        Expr::Cast { expr, .. } => expr_is_pure_for_control_cleanup(expr),
    }
}

fn exprs_are_boolean_complements(lhs: &Expr, rhs: &Expr) -> bool {
    strip_boolean_not(lhs) == Some(rhs) || strip_boolean_not(rhs) == Some(lhs)
}

fn strip_boolean_not(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Unary { op, arg } if op == "!" => Some(arg.as_ref()),
        _ => None,
    }
}

fn negate_boolean_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Unary { op, arg } if op == "!" => *arg,
        Expr::Imm(text) if text == "true" => Expr::Imm("false".to_string()),
        Expr::Imm(text) if text == "false" => Expr::Imm("true".to_string()),
        other => Expr::Unary {
            op: "!".to_string(),
            arg: Box::new(other),
        },
    }
}

fn normalize_local_if_control(stmt: Stmt) -> Stmt {
    simplify_empty_if_arms(strip_fallthrough_branch_gotos(
        merge_complementary_linear_ifs(stmt),
    ))
}

fn merge_complementary_linear_ifs(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => merge_complementary_if_sequence(stmts, false),
        Stmt::Block(stmts) => merge_complementary_if_sequence(stmts, true),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(merge_complementary_linear_ifs(*body)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(merge_complementary_linear_ifs(*then_branch)),
            else_branch: else_branch
                .map(|branch| Box::new(merge_complementary_linear_ifs(*branch))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(merge_complementary_linear_ifs(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, merge_complementary_linear_ifs(body)))
                .collect(),
            default: default
                .map(|body| Box::new(merge_complementary_linear_ifs(*body))),
        },
        other => other,
    }
}

fn merge_complementary_if_sequence(stmts: Vec<Stmt>, as_block: bool) -> Stmt {
    let stmts = stmts
        .into_iter()
        .map(merge_complementary_linear_ifs)
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < stmts.len() {
        if let Some((merged, consumed)) = try_merge_complementary_if_pair(&stmts[idx..]) {
            out.push(merged);
            idx += consumed;
            continue;
        }
        out.push(stmts[idx].clone());
        idx += 1;
    }

    if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    }
}

fn try_merge_complementary_if_pair(stmts: &[Stmt]) -> Option<(Stmt, usize)> {
    let first = stmts.first()?;
    let (first_condition, first_branch) = stmt_as_single_arm_if(first)?;
    if !expr_is_pure_for_control_cleanup(first_condition) || stmt_contains_any_label(first_branch) {
        return None;
    }

    let second_idx = first_nonempty_stmt_index(&stmts[1..])?;
    let second = &stmts[second_idx + 1];
    let (second_condition, second_branch) = stmt_as_single_arm_if(second)?;
    if !expr_is_pure_for_control_cleanup(second_condition)
        || stmt_contains_any_label(second_branch)
        || !exprs_are_boolean_complements(first_condition, second_condition)
    {
        return None;
    }
    let mut condition_vars = BTreeSet::new();
    collect_used_expr_vars(first_condition, &mut condition_vars);
    collect_used_expr_vars(second_condition, &mut condition_vars);
    if stmt_writes_any_named_var(first_branch, &condition_vars) {
        return None;
    }

    Some((
        Stmt::If {
            condition: first_condition.clone(),
            then_branch: Box::new(first_branch.clone()),
            else_branch: Some(Box::new(second_branch.clone())),
        },
        second_idx + 2,
    ))
}

fn stmt_as_single_arm_if(stmt: &Stmt) -> Option<(&Expr, &Stmt)> {
    let Stmt::If {
        condition,
        then_branch,
        else_branch,
    } = stmt
    else {
        return None;
    };
    if !else_branch
        .as_deref()
        .map(stmt_is_linear_empty)
        .unwrap_or(true)
        || stmt_is_linear_empty(then_branch)
    {
        return None;
    }
    Some((condition, then_branch))
}

fn strip_fallthrough_branch_gotos(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => strip_fallthrough_branch_goto_sequence(stmts, false),
        Stmt::Block(stmts) => strip_fallthrough_branch_goto_sequence(stmts, true),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(strip_fallthrough_branch_gotos(*body)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(strip_fallthrough_branch_gotos(*then_branch)),
            else_branch: else_branch
                .map(|branch| Box::new(strip_fallthrough_branch_gotos(*branch))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(strip_fallthrough_branch_gotos(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, strip_fallthrough_branch_gotos(body)))
                .collect(),
            default: default
                .map(|body| Box::new(strip_fallthrough_branch_gotos(*body))),
        },
        other => other,
    }
}

fn simplify_empty_if_arms(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            Stmt::Sequence(stmts.into_iter().map(simplify_empty_if_arms).collect())
        }
        Stmt::Block(stmts) => Stmt::Block(stmts.into_iter().map(simplify_empty_if_arms).collect()),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(simplify_empty_if_arms(*body)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let then_branch = Box::new(simplify_empty_if_arms(*then_branch));
            let else_branch = else_branch.map(|branch| Box::new(simplify_empty_if_arms(*branch)));
            let then_empty = stmt_is_linear_empty(&then_branch);
            let else_empty = else_branch
                .as_deref()
                .map(stmt_is_linear_empty)
                .unwrap_or(true);
            if then_empty && else_empty {
                return if expr_is_pure_for_control_cleanup(&condition) {
                    Stmt::Empty
                } else {
                    Stmt::ExprStmt(condition)
                };
            }
            if then_empty {
                return Stmt::If {
                    condition: negate_boolean_expr(condition),
                    then_branch: else_branch.unwrap_or_else(|| Box::new(Stmt::Empty)),
                    else_branch: None,
                };
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch: (!else_empty).then_some(else_branch).flatten(),
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(simplify_empty_if_arms(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, simplify_empty_if_arms(body)))
                .collect(),
            default: default.map(|body| Box::new(simplify_empty_if_arms(*body))),
        },
        other => other,
    }
}

fn strip_fallthrough_branch_goto_sequence(stmts: Vec<Stmt>, as_block: bool) -> Stmt {
    let stmts = stmts
        .into_iter()
        .map(strip_fallthrough_branch_gotos)
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(stmts.len());
    for idx in 0..stmts.len() {
        if let Some(rewritten) = strip_fallthrough_goto_from_if(&stmts[idx], &stmts[idx + 1..]) {
            out.push(rewritten);
        } else {
            out.push(stmts[idx].clone());
        }
    }

    if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    }
}

fn strip_fallthrough_goto_from_if(stmt: &Stmt, trailing: &[Stmt]) -> Option<Stmt> {
    let target_label = next_immediate_label_name(trailing)?;
    let Stmt::If {
        condition,
        then_branch,
        else_branch,
    } = stmt
    else {
        return None;
    };
    if !expr_is_pure_for_control_cleanup(condition) {
        return None;
    }

    let new_then = drop_trailing_goto_to_label(then_branch, target_label);
    let new_else = else_branch
        .as_deref()
        .and_then(|branch| drop_trailing_goto_to_label(branch, target_label));
    if new_then.is_none() && new_else.is_none() {
        return None;
    }

    Some(Stmt::If {
        condition: condition.clone(),
        then_branch: Box::new(new_then.unwrap_or_else(|| then_branch.as_ref().clone())),
        else_branch: else_branch.as_ref().map(|branch| {
            Box::new(new_else.unwrap_or_else(|| branch.as_ref().clone()))
        }),
    })
}

fn next_immediate_label_name(stmts: &[Stmt]) -> Option<&str> {
    let (stmt, _) = split_first_nonempty_stmt(stmts)?;
    let Stmt::Label { name, .. } = stmt else {
        return None;
    };
    Some(name.as_str())
}

fn stmt_writes_any_named_var(stmt: &Stmt, vars: &BTreeSet<String>) -> bool {
    match stmt {
        Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
            stmts.iter().any(|stmt| stmt_writes_any_named_var(stmt, vars))
        }
        Stmt::Label { body, .. } => stmt_writes_any_named_var(body, vars),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_writes_any_named_var(then_branch, vars)
                || else_branch
                    .as_deref()
                    .is_some_and(|branch| stmt_writes_any_named_var(branch, vars))
        }
        Stmt::Loop { body, .. } => stmt_writes_any_named_var(body, vars),
        Stmt::Switch { cases, default, .. } => {
            cases
                .iter()
                .any(|(_, body)| stmt_writes_any_named_var(body, vars))
                || default
                    .as_deref()
                    .is_some_and(|body| stmt_writes_any_named_var(body, vars))
        }
        Stmt::Assign { dst, .. } => lvalue_writes_any_named_var(dst, vars),
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::ExprStmt(_)
        | Stmt::Goto(_)
        | Stmt::Empty => false,
    }
}

fn lvalue_writes_any_named_var(dst: &LValue, vars: &BTreeSet<String>) -> bool {
    match dst {
        LValue::Raw(name) | LValue::Var(name) => vars.contains(name),
        LValue::PtrLane { base, .. } => vars.contains(base),
        LValue::Deref { .. } | LValue::Indexed { .. } => false,
    }
}

fn drop_trailing_goto_to_label(stmt: &Stmt, target_label: &str) -> Option<Stmt> {
    if stmt_contains_any_label(stmt) {
        return None;
    }
    match stmt {
        Stmt::Goto(label) if label == target_label => Some(Stmt::Empty),
        Stmt::Sequence(stmts) => drop_trailing_goto_to_label_sequence(stmts, false, target_label),
        Stmt::Block(stmts) => drop_trailing_goto_to_label_sequence(stmts, true, target_label),
        _ => None,
    }
}

fn drop_trailing_goto_to_label_sequence(
    stmts: &[Stmt],
    as_block: bool,
    target_label: &str,
) -> Option<Stmt> {
    let idx = stmts.iter().rposition(|stmt| !stmt_is_linear_empty(stmt))?;
    let Stmt::Goto(label) = &stmts[idx] else {
        return None;
    };
    if label != target_label {
        return None;
    }

    let mut out = Vec::with_capacity(stmts.len().saturating_sub(1));
    out.extend(stmts[..idx].iter().cloned());
    out.extend(stmts[idx + 1..].iter().cloned());
    Some(if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    })
}

fn trim_unreachable_linear_suffixes(stmt: Stmt, goto_targets: &BTreeSet<String>) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => trim_unreachable_sequence(stmts, false, goto_targets),
        Stmt::Block(stmts) => trim_unreachable_sequence(stmts, true, goto_targets),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(trim_unreachable_linear_suffixes(*body, goto_targets)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(trim_unreachable_linear_suffixes(*then_branch, goto_targets)),
            else_branch: else_branch
                .map(|branch| Box::new(trim_unreachable_linear_suffixes(*branch, goto_targets))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(trim_unreachable_linear_suffixes(*body, goto_targets)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, trim_unreachable_linear_suffixes(body, goto_targets)))
                .collect(),
            default: default
                .map(|body| Box::new(trim_unreachable_linear_suffixes(*body, goto_targets))),
        },
        other => other,
    }
}

fn cleanup_local_control_flow(body: Stmt) -> Stmt {
    let body = normalize_local_if_control(body);
    let mut goto_targets = BTreeSet::new();
    collect_goto_targets(&body, &mut goto_targets);
    let body = redirect_trivial_gotos(trim_unreachable_linear_suffixes(body, &goto_targets));
    let mut goto_target_counts = HashMap::new();
    collect_goto_target_counts(&body, &mut goto_target_counts);
    let body = normalize_local_if_control(fold_jump_over_branch_labels(
        body,
        &goto_target_counts,
    ));
    redirect_trivial_gotos(body)
}

fn trim_unreachable_sequence(
    stmts: Vec<Stmt>,
    as_block: bool,
    goto_targets: &BTreeSet<String>,
) -> Stmt {
    let mut out = Vec::new();
    let mut terminated = false;

    for stmt in stmts
        .into_iter()
        .map(|stmt| trim_unreachable_linear_suffixes(stmt, goto_targets))
        .filter(|stmt| !stmt_is_linear_empty(stmt))
    {
        if terminated && !stmt_contains_targeted_label(&stmt, goto_targets) {
            continue;
        }
        terminated = stmt_definitely_terminates(&stmt);
        out.push(stmt);
    }

    if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    }
}

fn stmt_definitely_terminates(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue => true,
        Stmt::Label { body, .. } => stmt_definitely_terminates(body),
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => stmts
            .iter()
            .rev()
            .find(|stmt| !stmt_is_linear_empty(stmt))
            .is_some_and(stmt_definitely_terminates),
        Stmt::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => stmt_definitely_terminates(then_branch) && stmt_definitely_terminates(else_branch),
        Stmt::Switch { cases, default, .. } => {
            !cases.is_empty()
                && cases
                    .iter()
                    .all(|(_, body)| stmt_definitely_terminates(body))
                && default
                    .as_deref()
                    .is_some_and(stmt_definitely_terminates)
        }
        _ => false,
    }
}

fn collect_goto_target_counts(stmt: &Stmt, counts: &mut HashMap<String, usize>) {
    match stmt {
        Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
            for stmt in stmts {
                collect_goto_target_counts(stmt, counts);
            }
        }
        Stmt::Label { body, .. } => collect_goto_target_counts(body, counts),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_goto_target_counts(then_branch, counts);
            if let Some(else_branch) = else_branch {
                collect_goto_target_counts(else_branch, counts);
            }
        }
        Stmt::Loop { body, .. } => collect_goto_target_counts(body, counts),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_goto_target_counts(body, counts);
            }
            if let Some(default) = default {
                collect_goto_target_counts(default, counts);
            }
        }
        Stmt::Goto(label) => {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::Assign { .. }
        | Stmt::ExprStmt(_)
        | Stmt::Empty => {}
    }
}

/// Fold local jump-over-branch shapes that the structurizer still emits as a
/// guarded goto plus a linear branch/label pair. This keeps the cleanup
/// AST-native and local: it only removes a target label when that label is
/// uniquely targeted and the moved branch bodies contain no labels.
fn fold_jump_over_branch_labels(
    stmt: Stmt,
    goto_target_counts: &HashMap<String, usize>,
) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => {
            fold_jump_over_branch_label_sequence(stmts, false, goto_target_counts)
        }
        Stmt::Block(stmts) => fold_jump_over_branch_label_sequence(stmts, true, goto_target_counts),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(fold_jump_over_branch_labels(*body, goto_target_counts)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(fold_jump_over_branch_labels(
                *then_branch,
                goto_target_counts,
            )),
            else_branch: else_branch.map(|branch| {
                Box::new(fold_jump_over_branch_labels(*branch, goto_target_counts))
            }),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(fold_jump_over_branch_labels(*body, goto_target_counts)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| {
                    (label, fold_jump_over_branch_labels(body, goto_target_counts))
                })
                .collect(),
            default: default.map(|body| {
                Box::new(fold_jump_over_branch_labels(*body, goto_target_counts))
            }),
        },
        other => other,
    }
}

fn fold_jump_over_branch_label_sequence(
    stmts: Vec<Stmt>,
    as_block: bool,
    goto_target_counts: &HashMap<String, usize>,
) -> Stmt {
    let stmts = stmts
        .into_iter()
        .map(|stmt| fold_jump_over_branch_labels(stmt, goto_target_counts))
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < stmts.len() {
        if let Some((folded, consumed)) =
            try_fold_jump_over_branch_label(&stmts[idx..], goto_target_counts)
        {
            out.push(folded);
            idx += consumed;
            continue;
        }
        out.push(stmts[idx].clone());
        idx += 1;
    }

    if as_block {
        Stmt::Block(out)
    } else {
        Stmt::Sequence(out)
    }
}

fn try_fold_jump_over_branch_label(
    stmts: &[Stmt],
    goto_target_counts: &HashMap<String, usize>,
) -> Option<(Stmt, usize)> {
    let first = stmts.first()?;
    let (jump_condition, target_label) = stmt_as_guarded_goto(first)?;
    if goto_target_counts.get(target_label).copied().unwrap_or(0) != 1 {
        return None;
    }

    let branch_idx = first_nonempty_stmt_index(&stmts[1..])?;
    let branch_stmt = &stmts[branch_idx + 1];
    let label_slice = &stmts[branch_idx + 2..];
    // Only linear-empty placeholders may sit between the terminating branch and
    // the target label. Any real statement must block the fold so we never
    // consume or reorder side effects while collapsing the goto ladder.
    let (label_idx, body) = take_immediate_target_label(label_slice, target_label)?;
    if stmt_contains_any_label(body) {
        return None;
    }

    if !stmt_contains_any_label(branch_stmt) && stmt_definitely_terminates(branch_stmt) {
        return Some((
            Stmt::If {
                condition: negate_boolean_expr(jump_condition.clone()),
                then_branch: Box::new(branch_stmt.clone()),
                else_branch: Some(Box::new(body.clone())),
            },
            branch_idx + 2 + label_idx + 1,
        ));
    }

    let Stmt::If {
        condition: branch_condition,
        then_branch,
        else_branch,
    } = branch_stmt
    else {
        return None;
    };
    if else_branch.is_some()
        || !stmt_definitely_terminates(then_branch)
        || stmt_contains_any_label(then_branch)
    {
        return None;
    }

    Some((
        Stmt::If {
            condition: Expr::Binary {
                op: "||".to_string(),
                lhs: Box::new(jump_condition.clone()),
                rhs: Box::new(negate_boolean_expr(branch_condition.clone())),
            },
            then_branch: Box::new(body.clone()),
            else_branch: Some(Box::new((**then_branch).clone())),
        },
        branch_idx + 2 + label_idx + 1,
    ))
}

fn take_immediate_target_label<'a>(stmts: &'a [Stmt], target_label: &str) -> Option<(usize, &'a Stmt)> {
    let label_idx = first_nonempty_stmt_index(stmts)?;
    let Stmt::Label { name, body } = &stmts[label_idx] else {
        return None;
    };
    (name == target_label).then_some((label_idx, body.as_ref()))
}

fn stmt_is_linear_empty(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Empty => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => stmts.iter().all(stmt_is_linear_empty),
        _ => false,
    }
}

fn stmt_contains_targeted_label(stmt: &Stmt, goto_targets: &BTreeSet<String>) -> bool {
    match stmt {
        Stmt::Label { name, body } => {
            goto_targets.contains(name) || stmt_contains_targeted_label(body, goto_targets)
        }
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => stmts
            .iter()
            .any(|stmt| stmt_contains_targeted_label(stmt, goto_targets)),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_contains_targeted_label(then_branch, goto_targets)
                || else_branch
                    .as_deref()
                    .is_some_and(|branch| stmt_contains_targeted_label(branch, goto_targets))
        }
        Stmt::Loop { body, .. } => stmt_contains_targeted_label(body, goto_targets),
        Stmt::Switch { cases, default, .. } => {
            cases
                .iter()
                .any(|(_, body)| stmt_contains_targeted_label(body, goto_targets))
                || default
                    .as_deref()
                    .is_some_and(|body| stmt_contains_targeted_label(body, goto_targets))
        }
        _ => false,
    }
}

fn stmt_contains_any_label(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Label { .. } => true,
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            stmts.iter().any(stmt_contains_any_label)
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_contains_any_label(then_branch)
                || else_branch
                    .as_deref()
                    .is_some_and(stmt_contains_any_label)
        }
        Stmt::Loop { body, .. } => stmt_contains_any_label(body),
        Stmt::Switch { cases, default, .. } => {
            cases.iter().any(|(_, body)| stmt_contains_any_label(body))
                || default
                    .as_deref()
                    .is_some_and(stmt_contains_any_label)
        }
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::Assign { .. }
        | Stmt::ExprStmt(_)
        | Stmt::Goto(_)
        | Stmt::Empty => false,
    }
}

fn collect_trivial_label_aliases(stmts: &[Stmt]) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for (idx, stmt) in stmts.iter().enumerate() {
        let Stmt::Label { name, body } = stmt else {
            continue;
        };
        let next_target = match body.as_ref() {
            Stmt::Empty => next_stmt_label_or_goto(&stmts[idx + 1..]),
            Stmt::Goto(target) => Some(target.clone()),
            _ => None,
        };
        let Some(target) = next_target else {
            continue;
        };
        if target != *name {
            aliases.insert(name.clone(), target);
        }
    }

    let keys = aliases.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        if let Some(resolved) = resolve_label_alias(&key, &aliases) {
            aliases.insert(key, resolved);
        }
    }
    aliases
}

fn resolve_label_alias(label: &str, aliases: &HashMap<String, String>) -> Option<String> {
    let mut current = aliases.get(label)?.clone();
    let mut seen = HashSet::new();
    seen.insert(label.to_string());
    while let Some(next) = aliases.get(&current) {
        if !seen.insert(current.clone()) {
            return None;
        }
        current = next.clone();
    }
    Some(current)
}

fn rewrite_stmt_gotos(stmt: Stmt, aliases: &HashMap<String, String>) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts.into_iter()
                .map(|stmt| rewrite_stmt_gotos(stmt, aliases))
                .collect(),
        ),
        Stmt::Block(stmts) => Stmt::Block(
            stmts.into_iter()
                .map(|stmt| rewrite_stmt_gotos(stmt, aliases))
                .collect(),
        ),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(rewrite_stmt_gotos(*body, aliases)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(rewrite_stmt_gotos(*then_branch, aliases)),
            else_branch: else_branch.map(|branch| Box::new(rewrite_stmt_gotos(*branch, aliases))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(rewrite_stmt_gotos(*body, aliases)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, rewrite_stmt_gotos(body, aliases)))
                .collect(),
            default: default.map(|body| Box::new(rewrite_stmt_gotos(*body, aliases))),
        },
        Stmt::Goto(label) => Stmt::Goto(
            resolve_label_alias(&label, aliases).unwrap_or(label),
        ),
        other => other,
    }
}

fn next_stmt_is_label(stmts: &[Stmt], label: &str) -> bool {
    stmts.iter().find(|stmt| !matches!(stmt, Stmt::Empty)).is_some_and(|stmt| {
        matches!(stmt, Stmt::Label { name, .. } if name == label)
    })
}

fn next_stmt_label_or_goto(stmts: &[Stmt]) -> Option<String> {
    stmts
        .iter()
        .find(|stmt| !matches!(stmt, Stmt::Empty))
        .and_then(|stmt| match stmt {
            Stmt::Label { name, .. } => Some(name.clone()),
            Stmt::Goto(label) => Some(label.clone()),
            _ => None,
        })
}

fn expr_const_bool(expr: &Expr) -> Option<bool> {
    match expr {
        Expr::Imm(text) | Expr::Raw(text) => match text.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => expr_const_i64(expr).map(|value| value != 0),
        },
        Expr::Unary { op, arg } if op == "!" => Some(!expr_const_bool(arg)?),
        Expr::Binary { op, lhs, rhs } => match op.as_str() {
            "&&" => Some(expr_const_bool(lhs)? && expr_const_bool(rhs)?),
            "||" => Some(expr_const_bool(lhs)? || expr_const_bool(rhs)?),
            "==" => Some(expr_const_i64(lhs)? == expr_const_i64(rhs)?),
            "!=" => Some(expr_const_i64(lhs)? != expr_const_i64(rhs)?),
            "<" => Some(expr_const_i64(lhs)? < expr_const_i64(rhs)?),
            "<=" => Some(expr_const_i64(lhs)? <= expr_const_i64(rhs)?),
            ">" => Some(expr_const_i64(lhs)? > expr_const_i64(rhs)?),
            ">=" => Some(expr_const_i64(lhs)? >= expr_const_i64(rhs)?),
            _ => None,
        },
        Expr::Cast { expr, .. } => {
            expr_const_bool(expr).or_else(|| expr_const_i64(expr).map(|v| v != 0))
        }
        _ => expr_const_i64(expr).map(|value| value != 0),
    }
}

fn expr_const_i64(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Imm(text) | Expr::Raw(text) => text.trim().parse::<i64>().ok(),
        Expr::Unary { op, arg } if op == "-" => expr_const_i64(arg)?.checked_neg(),
        Expr::Cast { expr, .. } => expr_const_i64(expr),
        _ => None,
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
            let rewritten = Stmt::Sequence(
                stmts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, stmt)| {
                        recover_rcp_division_stmt_tree(stmt, &mut local_defs, &local_future[idx])
                    })
                    .collect(),
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
            let rewritten = Stmt::Block(
                stmts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, stmt)| {
                        recover_rcp_division_stmt_tree(stmt, &mut local_defs, &local_future[idx])
                    })
                    .collect(),
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
    // Most kernels do not carry any reciprocal slow-path proof at all; skip the
    // expensive matcher unless a direct `FCHK` seed is still live in scope.
    if !defs_contain_direct_fchk(defs) {
        return None;
    }
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

fn defs_contain_direct_fchk(defs: &HashMap<String, Expr>) -> bool {
    defs.values().any(|expr| match_fchk_expr(expr).is_some())
}

fn match_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    match_direct_division_expr(expr, defs).or_else(|| match_rcp_division_expr(expr, defs))
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
    let mut active = HashSet::new();
    rewrite_match_expr(resolve_named_expr(expr, defs, depth, &mut active))
}

fn resolve_named_expr(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    depth: usize,
    active: &mut HashSet<String>,
) -> Expr {
    if depth == 0 {
        return expr.clone();
    }
    if let Some(name) = match_var_name(expr) {
        // Structured phi lowering can leave loop-carried defs that point back to
        // the current variable. Keep that variable as a leaf instead of
        // repeatedly inlining it into an exponentially larger tree.
        if active.contains(&name) {
            return expr.clone();
        }
        if let Some(mapped) = defs.get(&name) {
            active.insert(name.clone());
            let resolved = resolve_named_expr(mapped, defs, depth - 1, active);
            active.remove(&name);
            return resolved;
        }
    }
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op: op.clone(),
            arg: Box::new(resolve_named_expr(arg, defs, depth, active)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: op.clone(),
            lhs: Box::new(resolve_named_expr(lhs, defs, depth, active)),
            rhs: Box::new(resolve_named_expr(rhs, defs, depth, active)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(resolve_named_expr(cond, defs, depth, active)),
            then_expr: Box::new(resolve_named_expr(then_expr, defs, depth, active)),
            else_expr: Box::new(resolve_named_expr(else_expr, defs, depth, active)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func: func.clone(),
            args: args
                .iter()
                .map(|arg| resolve_named_expr(arg, defs, depth, active))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op: op.clone(),
            args: args
                .iter()
                .map(|arg| resolve_named_expr(arg, defs, depth, active))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty: ty.clone(),
            addr: Box::new(resolve_named_expr(addr, defs, depth, active)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(resolve_named_expr(base, defs, depth, active)),
            offset: Box::new(resolve_named_expr(offset, defs, depth, active)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(resolve_named_expr(lo, defs, depth, active)),
            hi: Box::new(resolve_named_expr(hi, defs, depth, active)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty: ty.clone(),
            expr: Box::new(resolve_named_expr(expr, defs, depth, active)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(resolve_named_expr(base, defs, depth, active)),
            index: Box::new(resolve_named_expr(index, defs, depth, active)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(resolve_named_expr(value, defs, depth, active)),
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
    use std::collections::HashMap;

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
    fn canonicalize_does_not_invent_division_without_reciprocal_proof() {
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
        assert!(!rendered.contains("arg0 / 9"), "got:\n{}", rendered);
        assert!(rendered.contains("FCHK("), "got:\n{}", rendered);
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
    fn canonicalize_cleans_structured_chunks_even_with_outer_goto_barrier() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("pred".to_string()),
                    src: Expr::Binary {
                        op: ">=".to_string(),
                        lhs: Box::new(Expr::Cast {
                            ty: "int32_t".to_string(),
                            expr: Box::new(Expr::Imm("0".to_string())),
                        }),
                        rhs: Box::new(Expr::Cast {
                            ty: "int32_t".to_string(),
                            expr: Box::new(Expr::Imm("1".to_string())),
                        }),
                    },
                },
                Stmt::If {
                    condition: Expr::Reg("pred".to_string()),
                    then_branch: Box::new(Stmt::Block(vec![
                        Stmt::Assign {
                            dst: LValue::Var("chk".to_string()),
                            src: Expr::CallLike {
                                func: "FCHK".to_string(),
                                args: vec![Expr::Imm("0".to_string()), Expr::Imm("1".to_string())],
                            },
                        },
                        Stmt::ExprStmt(Expr::CallLike {
                            func: "CALL.REL.NOINC".to_string(),
                            args: Vec::new(),
                        }),
                    ])),
                    else_branch: Some(Box::new(Stmt::Assign {
                        dst: LValue::Var("out".to_string()),
                        src: Expr::Imm("7".to_string()),
                    })),
                },
                Stmt::Goto("BB3".to_string()),
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);
        assert!(rendered.contains("goto BB3;"), "got:\n{rendered}");
        assert!(!rendered.contains("FCHK("), "got:\n{rendered}");
        assert!(!rendered.contains("CALL.REL.NOINC"), "got:\n{rendered}");
    }

    #[test]
    fn normalize_match_expr_stops_self_referential_defs_at_the_cycle() {
        let mut defs = HashMap::new();
        defs.insert(
            "idx".to_string(),
            Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("idx".to_string())),
                rhs: Box::new(Expr::Imm("4".to_string())),
            },
        );

        let resolved = normalize_match_expr(&Expr::Reg("idx".to_string()), &defs, 12);

        assert_eq!(canonical_match_expr(&resolved), "+(4,idx)");
    }

    #[test]
    fn normalize_match_expr_stops_mutual_cycles_without_blowing_up() {
        let mut defs = HashMap::new();
        defs.insert(
            "x".to_string(),
            Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("y".to_string())),
                rhs: Box::new(Expr::Imm("4".to_string())),
            },
        );
        defs.insert(
            "y".to_string(),
            Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("x".to_string())),
                rhs: Box::new(Expr::Imm("8".to_string())),
            },
        );

        let resolved = normalize_match_expr(&Expr::Reg("x".to_string()), &defs, 12);

        assert_eq!(canonical_match_expr(&resolved), "+(4,8,x)");
    }

    #[test]
    fn canonicalize_drops_goto_that_falls_through_to_next_label() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::Goto("BB1".to_string()),
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("BB1:"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_drops_guarded_goto_around_complementary_fallthrough_branch() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("p0".to_string())),
                    },
                    then_branch: Box::new(Stmt::Block(vec![Stmt::ExprStmt(Expr::Raw(
                        "side_effect()".to_string(),
                    ))])),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("if (!p0)"), "got:\n{rendered}");
        assert!(rendered.contains("BB1:"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_guarded_goto_when_following_branch_is_not_complementary() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Reg("p1".to_string()),
                    then_branch: Box::new(Stmt::Block(vec![Stmt::ExprStmt(Expr::Raw(
                        "side_effect()".to_string(),
                    ))])),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (p0) goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("if (p1)"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_drops_guarded_goto_to_duplicate_return_tail() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::Return(None),
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert_eq!(rendered.trim(), "return;", "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_impure_guarded_goto_to_duplicate_return_tail() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Raw("atomicAdd(ptr, 1)".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::Return(None),
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("atomicAdd(ptr, 1)"), "got:\n{rendered}");
        assert_ne!(rendered.trim(), "return;", "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_impure_complementary_guarded_branch_pair() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Raw("atomicAdd(ptr, 1)".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Raw("atomicAdd(ptr, 1)".to_string())),
                    },
                    then_branch: Box::new(Stmt::Block(vec![Stmt::ExprStmt(Expr::Raw(
                        "side_effect()".to_string(),
                    ))])),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (atomicAdd(ptr, 1)) goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("if (!atomicAdd(ptr, 1))"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_folds_jump_over_terminating_branch_into_if_else() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::Return(Some(Expr::Imm("7".to_string()))),
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::ExprStmt(Expr::Raw("side_effect()".to_string()))),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("if (!p0)"), "got:\n{rendered}");
        assert!(rendered.contains("return 7;"), "got:\n{rendered}");
        assert!(rendered.contains("side_effect();"), "got:\n{rendered}");
        assert!(!rendered.contains("BB1:"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_folds_guarded_jump_over_guarded_terminating_branch() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("p1".to_string())),
                    },
                    then_branch: Box::new(Stmt::Return(Some(Expr::Imm("9".to_string())))),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::ExprStmt(Expr::Raw("side_effect()".to_string()))),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("if (p0 || p1)"), "got:\n{rendered}");
        assert!(rendered.contains("return 9;"), "got:\n{rendered}");
        assert!(rendered.contains("side_effect();"), "got:\n{rendered}");
        assert!(!rendered.contains("BB1:"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_jump_over_branch_when_statement_intervenes_before_label() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Reg("p1".to_string()),
                    then_branch: Box::new(Stmt::Return(Some(Expr::Imm("7".to_string())))),
                    else_branch: None,
                },
                Stmt::ExprStmt(Expr::Raw("side_effect()".to_string())),
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::ExprStmt(Expr::Raw("tail_effect()".to_string()))),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (p0) goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("if (p1)"), "got:\n{rendered}");
        assert!(rendered.contains("return 7;"), "got:\n{rendered}");
        assert!(rendered.contains("side_effect();"), "got:\n{rendered}");
        assert!(rendered.contains("BB1:"), "got:\n{rendered}");
        assert!(rendered.contains("tail_effect();"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_merges_complementary_single_arm_ifs() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::ExprStmt(Expr::Raw("hot_path()".to_string()))),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("p0".to_string())),
                    },
                    then_branch: Box::new(Stmt::ExprStmt(Expr::Raw("cold_path()".to_string()))),
                    else_branch: None,
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (p0)"), "got:\n{rendered}");
        assert!(rendered.contains("} else {"), "got:\n{rendered}");
        assert!(rendered.contains("hot_path();"), "got:\n{rendered}");
        assert!(rendered.contains("cold_path();"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_strips_fallthrough_goto_from_if_branch() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Block(vec![
                        Stmt::ExprStmt(Expr::Raw("prepare()".to_string())),
                        Stmt::Goto("BB1".to_string()),
                    ])),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::ExprStmt(Expr::Raw("consume()".to_string()))),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (p0)"), "got:\n{rendered}");
        assert!(rendered.contains("prepare();"), "got:\n{rendered}");
        assert!(rendered.contains("consume();"), "got:\n{rendered}");
        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_impure_fallthrough_guarded_goto() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Raw("atomicAdd(ptr, 1)".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::ExprStmt(Expr::Raw("tail_effect()".to_string()))),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("atomicAdd(ptr, 1)"), "got:\n{rendered}");
        assert!(rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("tail_effect();"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_complementary_ifs_when_first_arm_mutates_condition_var() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Assign {
                        dst: LValue::Var("p0".to_string()),
                        src: Expr::Imm("false".to_string()),
                    }),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("p0".to_string())),
                    },
                    then_branch: Box::new(Stmt::ExprStmt(Expr::Raw("slow_path()".to_string()))),
                    else_branch: None,
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (p0)"), "got:\n{rendered}");
        assert!(rendered.contains("p0 = false;"), "got:\n{rendered}");
        assert!(rendered.contains("if (!p0)"), "got:\n{rendered}");
        assert!(rendered.contains("slow_path();"), "got:\n{rendered}");
        assert!(!rendered.contains("} else {"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_drops_duplicate_complementary_terminal_pair_suffix() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("p0".to_string())),
                    },
                    then_branch: Box::new(Stmt::Return(Some(Expr::Imm("7".to_string())))),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Goto("BB1".to_string())),
                    else_branch: None,
                },
                Stmt::If {
                    condition: Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("p0".to_string())),
                    },
                    then_branch: Box::new(Stmt::Return(Some(Expr::Imm("9".to_string())))),
                    else_branch: None,
                },
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::ExprStmt(Expr::Raw("tail_effect()".to_string()))),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("if (!p0)"), "got:\n{rendered}");
        assert!(rendered.contains("return 7;"), "got:\n{rendered}");
        assert!(!rendered.contains("return 9;"), "got:\n{rendered}");
        assert!(rendered.contains("tail_effect();"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_redirects_gotos_across_empty_label_aliases() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::Goto("BB1".to_string()),
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Empty),
                },
                Stmt::Goto("BB2".to_string()),
                Stmt::Label {
                    name: "BB2".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(!rendered.contains("BB1:"), "got:\n{rendered}");
        assert!(rendered.contains("BB2:"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_drops_unreachable_suffix_after_terminal_before_bind_cleanup() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::Label {
                    name: "BB1".to_string(),
                    body: Box::new(Stmt::Sequence(vec![
                        Stmt::Return(None),
                        Stmt::If {
                            condition: Expr::Reg("p0".to_string()),
                            then_branch: Box::new(Stmt::Goto("BB2".to_string())),
                            else_branch: None,
                        },
                    ])),
                },
                Stmt::Label {
                    name: "BB2".to_string(),
                    body: Box::new(Stmt::Return(None)),
                },
            ]),
        };

        let rendered = canonicalize_post_bind_control_flow(function).body.render_with_indent(0);

        assert!(rendered.contains("BB1:"), "got:\n{rendered}");
        assert!(rendered.contains("BB2:"), "got:\n{rendered}");
        assert!(!rendered.contains("if (p0) goto BB2;"), "got:\n{rendered}");
    }

    #[test]
    fn post_bind_cleanup_keeps_target_labels_inside_structured_suffixes() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::Goto("BB1".to_string()),
                Stmt::Return(None),
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Label {
                        name: "BB1".to_string(),
                        body: Box::new(Stmt::Assign {
                            dst: LValue::Var("r0".to_string()),
                            src: Expr::Imm("7".to_string()),
                        }),
                    }),
                    else_branch: None,
                },
            ]),
        };

        let rendered = canonicalize_post_bind_control_flow(function)
            .body
            .render_with_indent(0);

        assert!(rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("BB1:"), "got:\n{rendered}");
        assert!(rendered.contains("r0 = 7;"), "got:\n{rendered}");
    }

    #[test]
    fn canonicalize_keeps_loop_values_live_when_used_after_region() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::If {
                    condition: Expr::Reg("p0".to_string()),
                    then_branch: Box::new(Stmt::Loop {
                        kind: LoopKind::DoWhile,
                        condition: Some(Expr::Reg("p1".to_string())),
                        body: Box::new(Stmt::Sequence(vec![Stmt::Assign {
                            dst: LValue::Var("acc".to_string()),
                            src: Expr::CallLike {
                                func: "fmaf".to_string(),
                                args: vec![
                                    Expr::Reg("lhs".to_string()),
                                    Expr::Reg("rhs".to_string()),
                                    Expr::Reg("acc".to_string()),
                                ],
                            },
                        }])),
                    }),
                    else_branch: Some(Box::new(Stmt::Assign {
                        dst: LValue::Var("acc".to_string()),
                        src: Expr::Reg("fallback".to_string()),
                    })),
                },
                Stmt::Return(Some(Expr::Reg("acc".to_string()))),
            ]),
        };

        let rendered = canonicalize_function(function).body.render_with_indent(0);

        assert!(rendered.contains("acc = fmaf(lhs, rhs, acc);"), "got:\n{rendered}");
        assert!(rendered.contains("return acc;"), "got:\n{rendered}");
    }
}
