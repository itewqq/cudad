//! Structural cleanup passes over the typed AST.
//! These operate before final rendering so backend cleanup stops depending
//! exclusively on post-render text rewriting.
//!
//! This module is internal to the crate. The live canonical path only relies on
//! the local structural simplifier; the broader legacy cleanup entry points are
//! being deleted instead of kept as public backend API.

use crate::ast::{Expr, IntrinsicOp, LValue, PointerLane, Stmt};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct SeededWideAddrInfo {
    pub base: Expr,
    pub seed_lo: Expr,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeededWideAddrMaps {
    pub lo_by_name: HashMap<String, SeededWideAddrInfo>,
    pub hi_by_name: HashMap<String, SeededWideAddrInfo>,
}

pub fn ast_cleanup(stmt: Stmt) -> Stmt {
    ast_cleanup_with_seeded_wide_addrs(stmt, &SeededWideAddrMaps::default())
}

pub fn ast_cleanup_with_seeded_wide_addrs(
    stmt: Stmt,
    seeded_wide_addrs: &SeededWideAddrMaps,
) -> Stmt {
    let debug = std::env::var("DEBUG_AST_CLEANUP").is_ok();
    let mut current = stmt;
    loop {
        let has_unstructured = contains_unstructured_control_flow(&current);
        let stage_addr64 = ast_addr64_fold(current.clone());
        let stage_guard = if has_unstructured {
            stage_addr64.clone()
        } else {
            ast_guard_select_specialize(stage_addr64.clone())
        };
        let stage_split = ast_split_wide_lane_fold(stage_guard.clone());
        let stage_inline = ast_inline_wide_lane_defs(stage_split.clone());
        let stage_seeded = ast_seeded_wide_addr_fold(stage_inline.clone(), seeded_wide_addrs);
        let stage_typed = ast_typed_wide_addr_fold(stage_seeded.clone());
        let stage_shift = ast_reconstruct_split_wide_shifts(stage_typed.clone());
        let stage_norm = ast_raw_addr64_normalize(stage_shift.clone());
        // Keep cleanup conservative here: a full AST liveness walk is still too
        // aggressive around some loop-carried phi materialization, but purely
        // syntactic dead-temp pruning is safe and removes the helper chains that
        // otherwise leak into rendered output.
        let stage_dce = ast_conservative_dce(stage_norm.clone());
        let stage_pred = ast_predicate_cleanup(stage_dce.clone());
        let next = ast_simplify(stage_pred.clone());
        if debug {
            eprintln!(
                "=== addr64 ===
{}",
                stage_addr64.render_with_indent(0)
            );
            eprintln!(
                "=== guard ===
{}",
                stage_guard.render_with_indent(0)
            );
            eprintln!(
                "=== split ===
{}",
                stage_split.render_with_indent(0)
            );
            eprintln!(
                "=== inline ===
{}",
                stage_inline.render_with_indent(0)
            );
            eprintln!(
                "=== seeded ===
{}",
                stage_seeded.render_with_indent(0)
            );
            eprintln!(
                "=== typed ===
{}",
                stage_typed.render_with_indent(0)
            );
            eprintln!(
                "=== shift ===
{}",
                stage_shift.render_with_indent(0)
            );
            eprintln!(
                "=== norm ===
{}",
                stage_norm.render_with_indent(0)
            );
            eprintln!(
                "=== dce ===
{}",
                stage_dce.render_with_indent(0)
            );
            eprintln!(
                "=== pred ===
{}",
                stage_pred.render_with_indent(0)
            );
            eprintln!(
                "=== simplify ===
{}",
                next.render_with_indent(0)
            );
        }
        if next == current {
            // Once we erase the explicit `__loop_phi(...)` markers we also erase
            // the liveness breadcrumb that tells DCE which loop-header seeds and
            // loop-exit phis are semantically required. Running another generic
            // DCE pass after marker stripping can therefore delete the very
            // initializers/exit values that make the lowered loop correct
            // (`dot_thread`, `loop_constant`, real corpus kernels). Preserve the
            // finalized AST structure here and let earlier iterations handle the
            // safe cleanup while the markers are still present.
            return strip_loop_phi_markers_stmt(next);
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

pub fn ast_named_addr64_fold(stmt: Stmt) -> Stmt {
    addr64_fold_stmt(stmt, &Addr64Defs::named_mode())
}

pub fn ast_apply_token_map(stmt: Stmt, token_map: &HashMap<String, String>) -> Stmt {
    rename_stmt_tokens(stmt, token_map)
}

// Safe post-name cleanup: collapse only explicit named pointer-lane pairs like
// `arg0_ptr_lo32`/`arg0_ptr_hi32` back into `arg0_ptr + delta`. Unlike the
// pre-name cleanup, this pass does not rely on SSA-style single-def structure.
pub fn ast_named_ptr_pair_fold(stmt: Stmt) -> Stmt {
    named_ptr_pair_fold_stmt(stmt)
}

pub fn ast_post_name_cleanup(stmt: Stmt) -> Stmt {
    // After name recovery, multiple SSA defs intentionally collapse onto a
    // smaller set of source-level variables. Running the full AST simplifier
    // here is therefore unsound: its DCE/self-assign logic assumes single-def
    // names and can erase necessary initializations/join assignments once the
    // names have been coalesced (`dot_thread`, `loop_constant`, SM100 builtin
    // recovery). Keep post-name cleanup strictly local and syntax-oriented.
    ast_conservative_dce(stmt)
}

fn contains_unstructured_control_flow(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Label { body, .. } => contains_unstructured_control_flow(body) || true,
        Stmt::Goto(_) => true,
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

pub fn ast_raw_addr64_normalize(stmt: Stmt) -> Stmt {
    normalize_raw_addr64_stmt(stmt)
}

#[derive(Clone)]
struct WideShiftLoDef {
    input_lo: Expr,
    shift: Expr,
}

#[derive(Clone)]
struct WideShiftHiDef {
    input_lo: Expr,
    input_hi: Expr,
    shift: Expr,
}

#[derive(Clone)]
struct WideShiftPairDef {
    input_lo: Expr,
    input_hi: Expr,
    shift: Expr,
}

fn ast_reconstruct_split_wide_shifts(stmt: Stmt) -> Stmt {
    let defs = collect_split_wide_shift_defs(std::slice::from_ref(&stmt));
    reconstruct_split_wide_shift_stmt(stmt, &defs)
}

fn collect_split_wide_shift_defs(stmts: &[Stmt]) -> HashMap<String, WideShiftPairDef> {
    let mut lo_defs = HashMap::<String, WideShiftLoDef>::new();
    let mut hi_defs = Vec::<WideShiftHiDef>::new();
    for stmt in stmts {
        collect_split_wide_shift_defs_from_stmt(stmt, &mut lo_defs, &mut hi_defs);
    }

    let mut paired = HashMap::new();
    for (name, lo_def) in lo_defs {
        let lo_key = split_wide_shift_key(&lo_def.input_lo);
        let shift_key = split_wide_shift_key(&lo_def.shift);
        let mut matches = hi_defs.iter().filter(|hi_def| {
            split_wide_shift_key(&hi_def.input_lo) == lo_key
                && split_wide_shift_key(&hi_def.shift) == shift_key
        });
        let first = matches.next().cloned();
        if first.is_none() || matches.next().is_some() {
            continue;
        }
        let hi_def = first.unwrap();
        paired.insert(
            name,
            WideShiftPairDef {
                input_lo: lo_def.input_lo,
                input_hi: hi_def.input_hi,
                shift: lo_def.shift,
            },
        );
    }
    paired
}

fn collect_split_wide_shift_defs_from_stmt(
    stmt: &Stmt,
    lo_defs: &mut HashMap<String, WideShiftLoDef>,
    hi_defs: &mut Vec<WideShiftHiDef>,
) {
    match stmt {
        Stmt::Label { body, .. } => {
            collect_split_wide_shift_defs_from_stmt(body, lo_defs, hi_defs);
        }
        Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
            for stmt in stmts {
                collect_split_wide_shift_defs_from_stmt(stmt, lo_defs, hi_defs);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_split_wide_shift_defs_from_stmt(then_branch, lo_defs, hi_defs);
            if let Some(else_branch) = else_branch {
                collect_split_wide_shift_defs_from_stmt(else_branch, lo_defs, hi_defs);
            }
        }
        Stmt::Loop { body, .. } => {
            collect_split_wide_shift_defs_from_stmt(body, lo_defs, hi_defs);
        }
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_split_wide_shift_defs_from_stmt(body, lo_defs, hi_defs);
            }
            if let Some(default) = default {
                collect_split_wide_shift_defs_from_stmt(default, lo_defs, hi_defs);
            }
        }
        _ => {
            let Some((name, rhs)) = direct_var_assign(stmt) else {
                return;
            };
            if let Some(def) = match_split_wide_shift_lo_expr(rhs) {
                lo_defs.insert(name.clone(), def);
            }
            if let Some(def) = match_split_wide_shift_hi_expr(rhs) {
                hi_defs.push(def);
            }
        }
    }
}

fn match_split_wide_shift_lo_expr(rhs: &Expr) -> Option<WideShiftLoDef> {
    let rhs = strip_loop_phi_expr(rhs);
    let Expr::Binary { op, lhs, rhs } = rhs else {
        return None;
    };
    if op != "<<" {
        return None;
    }
    Some(WideShiftLoDef {
        input_lo: strip_loop_phi_expr(lhs).clone(),
        shift: strip_loop_phi_expr(rhs).clone(),
    })
}

fn match_split_wide_shift_hi_expr(rhs: &Expr) -> Option<WideShiftHiDef> {
    let rhs = strip_loop_phi_expr(rhs);
    if let Expr::CallLike { func, args } = rhs {
        if func == "SHF.L.U64.HI" && args.len() == 3 {
            return Some(WideShiftHiDef {
                input_lo: strip_loop_phi_expr(&args[0]).clone(),
                shift: strip_loop_phi_expr(&args[1]).clone(),
                input_hi: strip_loop_phi_expr(&args[2]).clone(),
            });
        }
    }
    let Expr::Raw(text) = rhs else {
        return None;
    };
    let args = parse_raw_helper_call_args(text, "SHF.L.U64.HI")?;
    if args.len() != 3 {
        return None;
    }
    Some(WideShiftHiDef {
        input_lo: raw_helper_arg_expr(args[0]),
        shift: raw_helper_arg_expr(args[1]),
        input_hi: raw_helper_arg_expr(args[2]),
    })
}

fn parse_raw_helper_call_args<'a>(text: &'a str, func: &str) -> Option<Vec<&'a str>> {
    let prefix = format!("{}(", func);
    let body = text.strip_prefix(&prefix)?.strip_suffix(')')?;
    let mut args = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, ch) in body.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                args.push(body[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    args.push(body[start..].trim());
    Some(args)
}

fn raw_helper_arg_expr(text: &str) -> Expr {
    if text.parse::<i64>().is_ok() || text.parse::<u64>().is_ok() {
        Expr::Imm(text.to_string())
    } else {
        Expr::Raw(text.to_string())
    }
}

fn split_wide_shift_key(expr: &Expr) -> String {
    strip_loop_phi_expr(expr).render()
}

fn reconstruct_split_wide_shift_stmt(stmt: Stmt, defs: &HashMap<String, WideShiftPairDef>) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(reconstruct_split_wide_shift_stmt(*body, defs)),
        },
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .into_iter()
                .map(|stmt| reconstruct_split_wide_shift_stmt(stmt, defs))
                .collect(),
        ),
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts
                .into_iter()
                .map(|stmt| reconstruct_split_wide_shift_stmt(stmt, defs))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: reconstruct_split_wide_shift_expr(condition, defs),
            then_branch: Box::new(reconstruct_split_wide_shift_stmt(*then_branch, defs)),
            else_branch: else_branch
                .map(|stmt| Box::new(reconstruct_split_wide_shift_stmt(*stmt, defs))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(|expr| reconstruct_split_wide_shift_expr(expr, defs)),
            body: Box::new(reconstruct_split_wide_shift_stmt(*body, defs)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(|expr| reconstruct_split_wide_shift_expr(expr, defs)),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, reconstruct_split_wide_shift_stmt(body, defs)))
                .collect(),
            default: default.map(|body| Box::new(reconstruct_split_wide_shift_stmt(*body, defs))),
        },
        Stmt::Return(expr) => {
            Stmt::Return(expr.map(|expr| reconstruct_split_wide_shift_expr(expr, defs)))
        }
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: reconstruct_split_wide_shift_lvalue(dst, defs),
            src: reconstruct_split_wide_shift_expr(src, defs),
        },
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(reconstruct_split_wide_shift_expr(expr, defs)),
        other => other,
    }
}

fn reconstruct_split_wide_shift_lvalue(
    lvalue: LValue,
    defs: &HashMap<String, WideShiftPairDef>,
) -> LValue {
    match lvalue {
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(reconstruct_split_wide_shift_expr(*addr, defs)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(reconstruct_split_wide_shift_expr(*base, defs)),
            index: Box::new(reconstruct_split_wide_shift_expr(*index, defs)),
        },
        other => other,
    }
}

fn reconstruct_split_wide_shift_expr(expr: Expr, defs: &HashMap<String, WideShiftPairDef>) -> Expr {
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(reconstruct_split_wide_shift_expr(*arg, defs)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(reconstruct_split_wide_shift_expr(*lhs, defs)),
            rhs: Box::new(reconstruct_split_wide_shift_expr(*rhs, defs)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(reconstruct_split_wide_shift_expr(*cond, defs)),
            then_expr: Box::new(reconstruct_split_wide_shift_expr(*then_expr, defs)),
            else_expr: Box::new(reconstruct_split_wide_shift_expr(*else_expr, defs)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| reconstruct_split_wide_shift_expr(expr, defs))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| reconstruct_split_wide_shift_expr(expr, defs))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(reconstruct_split_wide_shift_expr(*addr, defs)),
        },
        Expr::WidePtr { base, offset } => {
            let base = reconstruct_split_wide_shift_expr(*base, defs);
            let offset = reconstruct_split_wide_shift_expr(*offset, defs);
            let offset = match_var_name(&offset)
                .and_then(|name| defs.get(&name))
                .map(|def| build_split_wide_shift_expr(def))
                .unwrap_or(offset);
            Expr::WidePtr {
                base: Box::new(base),
                offset: Box::new(offset),
            }
        }
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(reconstruct_split_wide_shift_expr(*lo, defs)),
            hi: Box::new(reconstruct_split_wide_shift_expr(*hi, defs)),
        },
        Expr::Cast { ty, expr } => {
            let expr = reconstruct_split_wide_shift_expr(*expr, defs);
            if is_wide_integer_cast(&ty) {
                if let Some(name) = match_var_name(&expr) {
                    if let Some(def) = defs.get(&name) {
                        return Expr::Cast {
                            ty,
                            expr: Box::new(build_split_wide_shift_expr(def)),
                        };
                    }
                }
            }
            Expr::Cast {
                ty,
                expr: Box::new(expr),
            }
        }
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(reconstruct_split_wide_shift_expr(*base, defs)),
            index: Box::new(reconstruct_split_wide_shift_expr(*index, defs)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(reconstruct_split_wide_shift_expr(*value, defs)),
            lane,
        },
        other => other,
    }
}

fn is_wide_integer_cast(ty: &str) -> bool {
    matches!(
        ty,
        "int64_t" | "uint64_t" | "intptr_t" | "uintptr_t" | "size_t"
    )
}

fn build_split_wide_shift_expr(def: &WideShiftPairDef) -> Expr {
    scale_addr_offset_expr(
        &build_u64_pack_expr(def.input_lo.clone(), def.input_hi.clone()),
        &def.shift,
    )
    .unwrap_or_else(|| Expr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(build_u64_pack_expr(
            def.input_lo.clone(),
            def.input_hi.clone(),
        )),
        rhs: Box::new(def.shift.clone()),
    })
}

fn build_u64_pack_expr(lo: Expr, hi: Expr) -> Expr {
    Expr::Binary {
        op: "|".to_string(),
        lhs: Box::new(Expr::Cast {
            ty: "uint64_t".to_string(),
            expr: Box::new(Expr::Cast {
                ty: "uint32_t".to_string(),
                expr: Box::new(lo),
            }),
        }),
        rhs: Box::new(Expr::Binary {
            op: "<<".to_string(),
            lhs: Box::new(Expr::Cast {
                ty: "uint64_t".to_string(),
                expr: Box::new(Expr::Cast {
                    ty: "uint32_t".to_string(),
                    expr: Box::new(hi),
                }),
            }),
            rhs: Box::new(Expr::Imm("32".to_string())),
        }),
    }
}

fn ast_guard_select_specialize(stmt: Stmt) -> Stmt {
    let defs = collect_addr64_defs(std::slice::from_ref(&stmt), &Addr64Defs::default());
    guard_select_specialize_stmt(stmt, &defs, &[])
}

fn guard_select_specialize_stmt(stmt: Stmt, defs: &Addr64Defs, active_guards: &[Expr]) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => {
            let label_defs = defs.merged_with(collect_addr64_defs(
                std::slice::from_ref(body.as_ref()),
                defs,
            ));
            Stmt::Label {
                name,
                body: Box::new(guard_select_specialize_stmt(
                    *body,
                    &label_defs,
                    active_guards,
                )),
            }
        }
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .into_iter()
                .map(|stmt| guard_select_specialize_stmt(stmt, defs, active_guards))
                .collect(),
        ),
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts
                .into_iter()
                .map(|stmt| guard_select_specialize_stmt(stmt, defs, active_guards))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = guard_select_specialize_expr(condition, defs, active_guards);
            let then_guards = push_active_guard(active_guards, condition.clone());
            let else_guards = push_active_guard(active_guards, negate_expr(condition.clone()));
            Stmt::If {
                condition,
                then_branch: Box::new(guard_select_specialize_stmt(
                    *then_branch,
                    defs,
                    &then_guards,
                )),
                else_branch: else_branch
                    .map(|stmt| Box::new(guard_select_specialize_stmt(*stmt, defs, &else_guards))),
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let mut loop_defs = defs.clone();
            let loop_names = collect_loop_carried_names(body.as_ref());
            loop_defs.loop_carried_names.extend(loop_names.clone());
            let mut loop_scope = defs.clone();
            loop_scope.loop_carried_names.extend(loop_names);
            loop_defs = loop_defs.merged_with(collect_addr64_defs(
                std::slice::from_ref(body.as_ref()),
                &loop_scope,
            ));
            Stmt::Loop {
                kind,
                condition: condition
                    .map(|expr| guard_select_specialize_expr(expr, &loop_defs, active_guards)),
                body: Box::new(guard_select_specialize_stmt(
                    *body,
                    &loop_defs,
                    active_guards,
                )),
            }
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant
                .map(|expr| guard_select_specialize_expr(expr, defs, active_guards)),
            cases: cases
                .into_iter()
                .map(|(label, body)| {
                    (
                        label,
                        guard_select_specialize_stmt(body, defs, active_guards),
                    )
                })
                .collect(),
            default: default
                .map(|body| Box::new(guard_select_specialize_stmt(*body, defs, active_guards))),
        },
        Stmt::Return(expr) => {
            Stmt::Return(expr.map(|expr| guard_select_specialize_expr(expr, defs, active_guards)))
        }
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: guard_select_specialize_lvalue(dst, defs, active_guards),
            src: guard_select_specialize_expr(src, defs, active_guards),
        },
        Stmt::ExprStmt(expr) => {
            Stmt::ExprStmt(guard_select_specialize_expr(expr, defs, active_guards))
        }
        other => other,
    }
}

fn guard_select_specialize_lvalue(
    lvalue: LValue,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> LValue {
    match lvalue {
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(guard_select_specialize_expr(*addr, defs, active_guards)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(guard_select_specialize_expr(*base, defs, active_guards)),
            index: Box::new(guard_select_specialize_expr(*index, defs, active_guards)),
        },
        other => other,
    }
}

fn guard_select_specialize_expr(expr: Expr, defs: &Addr64Defs, active_guards: &[Expr]) -> Expr {
    let expr = match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(guard_select_specialize_expr(*arg, defs, active_guards)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(guard_select_specialize_expr(*lhs, defs, active_guards)),
            rhs: Box::new(guard_select_specialize_expr(*rhs, defs, active_guards)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond = guard_select_specialize_expr(*cond, defs, active_guards);
            let then_guards = push_active_guard(active_guards, cond.clone());
            let else_guards = push_active_guard(active_guards, negate_expr(cond.clone()));
            Expr::Ternary {
                cond: Box::new(cond),
                then_expr: Box::new(guard_select_specialize_expr(*then_expr, defs, &then_guards)),
                else_expr: Box::new(guard_select_specialize_expr(*else_expr, defs, &else_guards)),
            }
        }
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| guard_select_specialize_expr(expr, defs, active_guards))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| guard_select_specialize_expr(expr, defs, active_guards))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(guard_select_specialize_expr(*addr, defs, active_guards)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(guard_select_specialize_expr(*base, defs, active_guards)),
            offset: Box::new(guard_select_specialize_expr(*offset, defs, active_guards)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(guard_select_specialize_expr(*lo, defs, active_guards)),
            hi: Box::new(guard_select_specialize_expr(*hi, defs, active_guards)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(guard_select_specialize_expr(*expr, defs, active_guards)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(guard_select_specialize_expr(*base, defs, active_guards)),
            index: Box::new(guard_select_specialize_expr(*index, defs, active_guards)),
        },
        other => other,
    };
    resolve_guard_selected_expr(expr, defs, active_guards)
}

fn ast_split_wide_lane_fold(stmt: Stmt) -> Stmt {
    let defs = collect_addr64_defs(std::slice::from_ref(&stmt), &Addr64Defs::default());
    split_wide_lane_fold_stmt(stmt, &defs, &[])
}

fn split_wide_lane_fold_stmt(stmt: Stmt, defs: &Addr64Defs, active_guards: &[Expr]) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => {
            let label_defs = defs.merged_with(collect_addr64_defs(
                std::slice::from_ref(body.as_ref()),
                defs,
            ));
            Stmt::Label {
                name,
                body: Box::new(split_wide_lane_fold_stmt(*body, &label_defs, active_guards)),
            }
        }
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .into_iter()
                .map(|stmt| split_wide_lane_fold_stmt(stmt, defs, active_guards))
                .collect(),
        ),
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts
                .into_iter()
                .map(|stmt| split_wide_lane_fold_stmt(stmt, defs, active_guards))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = split_wide_lane_fold_expr(condition, defs, active_guards);
            let then_guards = push_active_guard(active_guards, condition.clone());
            let else_guards = push_active_guard(active_guards, negate_expr(condition.clone()));
            Stmt::If {
                condition,
                then_branch: Box::new(split_wide_lane_fold_stmt(*then_branch, defs, &then_guards)),
                else_branch: else_branch
                    .map(|stmt| Box::new(split_wide_lane_fold_stmt(*stmt, defs, &else_guards))),
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(|expr| split_wide_lane_fold_expr(expr, defs, active_guards)),
            body: Box::new(split_wide_lane_fold_stmt(*body, defs, active_guards)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant
                .map(|expr| split_wide_lane_fold_expr(expr, defs, active_guards)),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, split_wide_lane_fold_stmt(body, defs, active_guards)))
                .collect(),
            default: default
                .map(|body| Box::new(split_wide_lane_fold_stmt(*body, defs, active_guards))),
        },
        Stmt::Return(expr) => {
            Stmt::Return(expr.map(|expr| split_wide_lane_fold_expr(expr, defs, active_guards)))
        }
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: split_wide_lane_fold_lvalue(dst, defs, active_guards),
            src: split_wide_lane_fold_expr(src, defs, active_guards),
        },
        Stmt::ExprStmt(expr) => {
            Stmt::ExprStmt(split_wide_lane_fold_expr(expr, defs, active_guards))
        }
        other => other,
    }
}

fn split_wide_lane_fold_lvalue(
    lvalue: LValue,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> LValue {
    match lvalue {
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(split_wide_lane_fold_addr_expr(*addr, defs, active_guards)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(split_wide_lane_fold_expr(*base, defs, active_guards)),
            index: Box::new(split_wide_lane_fold_expr(*index, defs, active_guards)),
        },
        other => other,
    }
}

fn split_wide_lane_fold_expr(expr: Expr, defs: &Addr64Defs, active_guards: &[Expr]) -> Expr {
    let expr = match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(split_wide_lane_fold_expr(*arg, defs, active_guards)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(split_wide_lane_fold_expr(*lhs, defs, active_guards)),
            rhs: Box::new(split_wide_lane_fold_expr(*rhs, defs, active_guards)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond = split_wide_lane_fold_expr(*cond, defs, active_guards);
            let then_guards = push_active_guard(active_guards, cond.clone());
            let else_guards = push_active_guard(active_guards, negate_expr(cond.clone()));
            Expr::Ternary {
                cond: Box::new(cond),
                then_expr: Box::new(split_wide_lane_fold_expr(*then_expr, defs, &then_guards)),
                else_expr: Box::new(split_wide_lane_fold_expr(*else_expr, defs, &else_guards)),
            }
        }
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| split_wide_lane_fold_expr(expr, defs, active_guards))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| split_wide_lane_fold_expr(expr, defs, active_guards))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(split_wide_lane_fold_addr_expr(*addr, defs, active_guards)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(split_wide_lane_fold_expr(*base, defs, active_guards)),
            offset: Box::new(split_wide_lane_fold_expr(*offset, defs, active_guards)),
        },
        Expr::Addr64 { lo, hi } => {
            let lo = split_wide_lane_fold_expr(*lo, defs, active_guards);
            let hi = split_wide_lane_fold_expr(*hi, defs, active_guards);
            collapse_split_wide_addr64(&lo, &hi).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(split_wide_lane_fold_expr(*value, defs, active_guards)),
            lane,
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(split_wide_lane_fold_expr(*expr, defs, active_guards)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(split_wide_lane_fold_expr(*base, defs, active_guards)),
            index: Box::new(split_wide_lane_fold_expr(*index, defs, active_guards)),
        },
        other => other,
    };
    fold_split_wide_lane_expr(&expr, defs, active_guards).unwrap_or(expr)
}

fn split_wide_lane_fold_addr_expr(expr: Expr, defs: &Addr64Defs, active_guards: &[Expr]) -> Expr {
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(split_wide_lane_fold_addr_expr(*arg, defs, active_guards)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(split_wide_lane_fold_addr_expr(*lhs, defs, active_guards)),
            rhs: Box::new(split_wide_lane_fold_addr_expr(*rhs, defs, active_guards)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(split_wide_lane_fold_expr(*cond, defs, active_guards)),
            then_expr: Box::new(split_wide_lane_fold_addr_expr(
                *then_expr,
                defs,
                active_guards,
            )),
            else_expr: Box::new(split_wide_lane_fold_addr_expr(
                *else_expr,
                defs,
                active_guards,
            )),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| split_wide_lane_fold_addr_expr(expr, defs, active_guards))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| split_wide_lane_fold_addr_expr(expr, defs, active_guards))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(split_wide_lane_fold_addr_expr(*addr, defs, active_guards)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(split_wide_lane_fold_addr_expr(*base, defs, active_guards)),
            offset: Box::new(split_wide_lane_fold_addr_expr(*offset, defs, active_guards)),
        },
        Expr::Addr64 { lo, hi } => {
            let lo = split_wide_lane_fold_expr(*lo, defs, active_guards);
            let hi = split_wide_lane_fold_expr(*hi, defs, active_guards);
            collapse_split_wide_addr64(&lo, &hi).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(split_wide_lane_fold_expr(*value, defs, active_guards)),
            lane,
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(split_wide_lane_fold_addr_expr(*expr, defs, active_guards)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(split_wide_lane_fold_addr_expr(*base, defs, active_guards)),
            index: Box::new(split_wide_lane_fold_expr(*index, defs, active_guards)),
        },
        other => other,
    }
}

fn collapse_split_wide_addr64(lo: &Expr, hi: &Expr) -> Option<Expr> {
    match (lo, hi) {
        (
            Expr::LaneExtract {
                value: lo_value,
                lane: PointerLane::Lo32,
            },
            Expr::LaneExtract {
                value: hi_value,
                lane: PointerLane::Hi32,
            },
        ) if lo_value == hi_value => Some((**lo_value).clone()),
        _ => None,
    }
}

fn fold_split_wide_lane_expr(
    expr: &Expr,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<Expr> {
    if let Expr::LaneExtract { .. } = expr {
        return Some(expr.clone());
    }
    if let Expr::PtrLane { base, lane } = expr {
        return Some(Expr::LaneExtract {
            value: Box::new(Expr::Raw(base.clone())),
            lane: *lane,
        });
    }
    if let Some((value, lane)) = explicit_wide_lane_extract(strip_loop_phi_expr(expr)) {
        return Some(Expr::LaneExtract {
            value: Box::new(value),
            lane,
        });
    }
    if let Some(args) = intrinsic_args(expr, IntrinsicOp::PairHi) {
        if args.len() == 1 {
            return Some(Expr::LaneExtract {
                value: Box::new(args[0].clone()),
                lane: PointerLane::Hi32,
            });
        }
    }
    if let Some(expr) = fold_direct_lo_wide_lane(expr, defs, active_guards) {
        return Some(expr);
    }
    if let Some(expr) = fold_direct_hi_wide_lane(expr, defs, active_guards) {
        return Some(expr);
    }
    let name = match_var_name(expr)?;
    if defs.loop_carried_names.contains(&name) {
        return None;
    }
    let mut seen = BTreeSet::new();
    resolve_named_wide_lane_expr(&name, defs, active_guards, &mut seen)
}

fn resolve_named_wide_lane_expr(
    name: &str,
    defs: &Addr64Defs,
    active_guards: &[Expr],
    seen: &mut BTreeSet<String>,
) -> Option<Expr> {
    if defs.loop_carried_names.contains(name) {
        return None;
    }
    if !seen.insert(name.to_string()) {
        return None;
    }
    if let Some(base) = strip_ptr_suffix(name, ".lo32", "_lo32") {
        return Some(Expr::LaneExtract {
            value: Box::new(Expr::Raw(base)),
            lane: PointerLane::Lo32,
        });
    }
    if let Some(base) = strip_ptr_suffix(name, ".hi32", "_hi32") {
        return Some(Expr::LaneExtract {
            value: Box::new(Expr::Raw(base)),
            lane: PointerLane::Hi32,
        });
    }
    if let Some(copy) = defs.copy_defs.get(name) {
        if let Some(expr) = resolve_named_wide_lane_expr(copy, defs, active_guards, seen) {
            return Some(expr);
        }
    }
    if let Some(select) = defs.select_defs.get(name) {
        if let Some(branch) = match resolve_guard_branch(active_guards, &select.condition) {
            Some(true) => Some(&select.then_expr),
            Some(false) => Some(&select.else_expr),
            None => None,
        } {
            if let Some(name) = match_var_name(branch) {
                if let Some(expr) = resolve_named_wide_lane_expr(&name, defs, active_guards, seen) {
                    return Some(expr);
                }
            }
            if let Some(expr) = fold_split_wide_lane_expr(branch, defs, active_guards) {
                return Some(expr);
            }
        }
    }
    fold_named_lo_wide_lane(name, defs, active_guards)
        .or_else(|| fold_named_hi_wide_lane(name, defs, active_guards))
}

fn clone_lane_value(expr: &Expr, lane: PointerLane) -> Option<Expr> {
    match expr {
        Expr::LaneExtract {
            value,
            lane: actual_lane,
        } if *actual_lane == lane => Some((**value).clone()),
        _ => explicit_wide_lane_extract(strip_loop_phi_expr(expr))
            .and_then(|(value, actual_lane)| (actual_lane == lane).then_some(value)),
    }
}

fn extend_wide_value(base: Expr, offset: Expr) -> Expr {
    if expr_is_zero(&offset) {
        return base;
    }
    match base {
        Expr::WidePtr {
            base: inner_base,
            offset: inner_offset,
        } => render_folded_addr64_base(
            *inner_base,
            &Expr::Binary {
                op: "+".to_string(),
                lhs: inner_offset,
                rhs: Box::new(offset),
            },
        ),
        other => render_folded_addr64_base(other, &offset),
    }
}

fn match_lane_lo_add_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    let Expr::Binary { op, lhs, rhs } = expr else {
        return None;
    };
    if op != "+" {
        return None;
    }
    if let Some(value) = clone_lane_value(lhs, PointerLane::Lo32) {
        return Some((value, (**rhs).clone()));
    }
    Some((clone_lane_value(rhs, PointerLane::Lo32)?, (**lhs).clone()))
}

fn match_lane_carry_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    let args = intrinsic_args(expr, IntrinsicOp::CarryU32Add3)?;
    if args.len() != 3 || !expr_is_zero(&args[2]) {
        return None;
    }
    if let Some(value) = clone_lane_value(&args[0], PointerLane::Lo32) {
        return Some((value, args[1].clone()));
    }
    Some((
        clone_lane_value(&args[1], PointerLane::Lo32)?,
        args[0].clone(),
    ))
}

fn match_lane_hi_add_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    let Expr::Binary { op, lhs, rhs } = expr else {
        return None;
    };
    if op != "+" {
        return None;
    }
    if let Some(value) = clone_lane_value(lhs, PointerLane::Hi32) {
        let Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } = rhs.as_ref()
        else {
            return None;
        };
        if !expr_is_one(then_expr) || !expr_is_zero(else_expr) {
            return None;
        }
        let (carry_value, offset) = match_lane_carry_expr(cond)?;
        if value == carry_value {
            return Some((value, offset));
        }
    }
    if let Some(value) = clone_lane_value(rhs, PointerLane::Hi32) {
        let Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } = lhs.as_ref()
        else {
            return None;
        };
        if !expr_is_one(then_expr) || !expr_is_zero(else_expr) {
            return None;
        }
        let (carry_value, offset) = match_lane_carry_expr(cond)?;
        if value == carry_value {
            return Some((value, offset));
        }
    }
    None
}

fn match_lane_lea_hi_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    let args = intrinsic_args(expr, IntrinsicOp::LeaHiX)
        .or_else(|| intrinsic_args(expr, IntrinsicOp::LeaHiXSx32))?;
    if args.len() != 4 {
        return None;
    }
    let value = clone_lane_value(&args[1], PointerLane::Hi32)?;
    let (carry_value, _) = match_lane_carry_expr(&args[3])?;
    if value != carry_value {
        return None;
    }
    Some((value, scale_addr_offset_expr(&args[0], &args[2])?))
}

fn fold_named_lo_wide_lane(name: &str, defs: &Addr64Defs, active_guards: &[Expr]) -> Option<Expr> {
    let def = defs.lo_defs.get(name)?;
    let (ptr_lo, _) =
        resolve_guarded_alias_preserving_loops(def.ptr_lo.clone(), defs, active_guards);
    let mut seen = BTreeSet::new();
    let (base_expr, _) =
        resolve_addr64_base_expr_from_lo_name(&ptr_lo, defs, &mut seen, active_guards)?;
    Some(Expr::LaneExtract {
        value: Box::new(render_folded_addr64_base(base_expr, &def.offset)),
        lane: PointerLane::Lo32,
    })
}

fn fold_named_hi_wide_lane(name: &str, defs: &Addr64Defs, active_guards: &[Expr]) -> Option<Expr> {
    if let Some(hi_info) = defs.lea_hi_defs.get(name) {
        let (ptr_hi, _) =
            resolve_guarded_alias_preserving_loops(hi_info.ptr_hi.clone(), defs, active_guards);
        let (carry_var, _) =
            resolve_guarded_alias_preserving_loops(hi_info.carry_var.clone(), defs, active_guards);
        let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
        let (ptr_lo, _) =
            resolve_guarded_alias_preserving_loops(carry_info.ptr_lo.clone(), defs, active_guards);
        if !lea_hi_matches_lo_offset(hi_info, &carry_info.offset) {
            return None;
        }
        let mut seen = BTreeSet::new();
        let (base_expr, _) =
            resolve_addr64_pair_base_expr(&ptr_lo, &ptr_hi, defs, &mut seen, active_guards)?;
        return Some(Expr::LaneExtract {
            value: Box::new(render_folded_addr64_base(base_expr, &carry_info.offset)),
            lane: PointerLane::Hi32,
        });
    }

    let hi_info = defs.hi_add_defs.get(name)?;
    let (carry_offset, ptr_lo) = match &hi_info.carry {
        CarryIncrement::Var(carry_var) => {
            let (carry_var, _) =
                resolve_guarded_alias_preserving_loops(carry_var.clone(), defs, active_guards);
            let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
            let (ptr_lo, _) = resolve_guarded_alias_preserving_loops(
                carry_info.ptr_lo.clone(),
                defs,
                active_guards,
            );
            (carry_info.offset, ptr_lo)
        }
        CarryIncrement::Inline(carry_info) => {
            let (ptr_lo, _) = resolve_guarded_alias_preserving_loops(
                carry_info.ptr_lo.clone(),
                defs,
                active_guards,
            );
            (carry_info.offset.clone(), ptr_lo)
        }
    };
    let (base_expr, _) = resolve_hi_add_base_expr(hi_info, &ptr_lo, defs, active_guards)?;
    Some(Expr::LaneExtract {
        value: Box::new(render_folded_addr64_base(base_expr, &carry_offset)),
        lane: PointerLane::Hi32,
    })
}

fn fold_direct_lo_wide_lane(
    expr: &Expr,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<Expr> {
    if let Some((value, offset)) = match_lane_lo_add_expr(expr) {
        return Some(Expr::LaneExtract {
            value: Box::new(extend_wide_value(value, offset)),
            lane: PointerLane::Lo32,
        });
    }
    let def = match_lo_add_expr(expr)?;
    let (ptr_lo, _) =
        resolve_guarded_alias_preserving_loops(def.ptr_lo.clone(), defs, active_guards);
    let mut seen = BTreeSet::new();
    let (base_expr, _) =
        resolve_addr64_base_expr_from_lo_name(&ptr_lo, defs, &mut seen, active_guards)?;
    Some(Expr::LaneExtract {
        value: Box::new(render_folded_addr64_base(base_expr, &def.offset)),
        lane: PointerLane::Lo32,
    })
}

fn fold_direct_hi_wide_lane(
    expr: &Expr,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<Expr> {
    if let Some((value, offset)) =
        match_lane_lea_hi_expr(expr).or_else(|| match_lane_hi_add_expr(expr))
    {
        return Some(Expr::LaneExtract {
            value: Box::new(extend_wide_value(value, offset)),
            lane: PointerLane::Hi32,
        });
    }
    if let Some(hi_info) = match_lea_hi_expr(expr) {
        let (ptr_hi, _) =
            resolve_guarded_alias_preserving_loops(hi_info.ptr_hi.clone(), defs, active_guards);
        let (carry_var, _) =
            resolve_guarded_alias_preserving_loops(hi_info.carry_var.clone(), defs, active_guards);
        let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
        let (ptr_lo, _) =
            resolve_guarded_alias_preserving_loops(carry_info.ptr_lo.clone(), defs, active_guards);
        if !lea_hi_matches_lo_offset(&hi_info, &carry_info.offset) {
            return None;
        }
        let mut seen = BTreeSet::new();
        let (base_expr, _) =
            resolve_addr64_pair_base_expr(&ptr_lo, &ptr_hi, defs, &mut seen, active_guards)?;
        return Some(Expr::LaneExtract {
            value: Box::new(render_folded_addr64_base(base_expr, &carry_info.offset)),
            lane: PointerLane::Hi32,
        });
    }

    let hi_info = match_hi_add_carry_expr(expr)?;
    let (carry_offset, ptr_lo) = match &hi_info.carry {
        CarryIncrement::Var(carry_var) => {
            let (carry_var, _) =
                resolve_guarded_alias_preserving_loops(carry_var.clone(), defs, active_guards);
            let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
            let (ptr_lo, _) = resolve_guarded_alias_preserving_loops(
                carry_info.ptr_lo.clone(),
                defs,
                active_guards,
            );
            (carry_info.offset, ptr_lo)
        }
        CarryIncrement::Inline(carry_info) => {
            let (ptr_lo, _) = resolve_guarded_alias_preserving_loops(
                carry_info.ptr_lo.clone(),
                defs,
                active_guards,
            );
            (carry_info.offset.clone(), ptr_lo)
        }
    };
    let (base_expr, _) = resolve_hi_add_base_expr(&hi_info, &ptr_lo, defs, active_guards)?;
    Some(Expr::LaneExtract {
        value: Box::new(render_folded_addr64_base(base_expr, &carry_offset)),
        lane: PointerLane::Hi32,
    })
}

fn ast_inline_wide_lane_defs(stmt: Stmt) -> Stmt {
    let blocked = collect_multi_assigned_names(std::slice::from_ref(&stmt));
    let mut defs = HashMap::<String, Expr>::new();
    inline_wide_lane_defs_stmt(stmt, &mut defs, &blocked)
}

fn inline_wide_lane_defs_stmt(
    stmt: Stmt,
    defs: &mut HashMap<String, Expr>,
    blocked: &HashSet<String>,
) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(inline_wide_lane_defs_stmt(*body, defs, blocked)),
        },
        Stmt::Block(stmts) => {
            let mut local_defs = defs.clone();
            Stmt::Block(
                stmts
                    .into_iter()
                    .map(|stmt| inline_wide_lane_defs_stmt(stmt, &mut local_defs, blocked))
                    .collect(),
            )
        }
        Stmt::Sequence(stmts) => {
            let mut local_defs = defs.clone();
            Stmt::Sequence(
                stmts
                    .into_iter()
                    .map(|stmt| inline_wide_lane_defs_stmt(stmt, &mut local_defs, blocked))
                    .collect(),
            )
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: inline_wide_lane_defs_expr(condition, defs),
            then_branch: Box::new(inline_wide_lane_defs_stmt(
                *then_branch,
                &mut defs.clone(),
                blocked,
            )),
            else_branch: else_branch.map(|stmt| {
                Box::new(inline_wide_lane_defs_stmt(
                    *stmt,
                    &mut defs.clone(),
                    blocked,
                ))
            }),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(|expr| inline_wide_lane_defs_expr(expr, defs)),
            body: Box::new(inline_wide_lane_defs_stmt(
                *body,
                &mut defs.clone(),
                blocked,
            )),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(|expr| inline_wide_lane_defs_expr(expr, defs)),
            cases: cases
                .into_iter()
                .map(|(label, body)| {
                    (
                        label,
                        inline_wide_lane_defs_stmt(body, &mut defs.clone(), blocked),
                    )
                })
                .collect(),
            default: default.map(|body| {
                Box::new(inline_wide_lane_defs_stmt(
                    *body,
                    &mut defs.clone(),
                    blocked,
                ))
            }),
        },
        Stmt::Return(expr) => Stmt::Return(expr.map(|expr| inline_wide_lane_defs_expr(expr, defs))),
        Stmt::Assign { dst, src } => {
            let src = inline_wide_lane_defs_expr(src, defs);
            match &dst {
                LValue::Var(name) => {
                    if blocked.contains(name) {
                        defs.remove(name);
                    } else if let Some(lane_expr) = extract_lane_def_expr(&src) {
                        let mut used = BTreeSet::new();
                        collect_used_expr_vars(&lane_expr, &mut used);
                        if used.contains(name) {
                            defs.remove(name);
                        } else {
                            defs.insert(name.clone(), lane_expr);
                        }
                    } else {
                        defs.remove(name);
                    }
                }
                _ => {}
            }
            Stmt::Assign { dst, src }
        }
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(inline_wide_lane_defs_expr(expr, defs)),
        other => other,
    }
}

fn inline_wide_lane_defs_expr(expr: Expr, defs: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Reg(name) => defs.get(&name).cloned().unwrap_or(Expr::Reg(name)),
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(inline_wide_lane_defs_expr(*arg, defs)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(inline_wide_lane_defs_expr(*lhs, defs)),
            rhs: Box::new(inline_wide_lane_defs_expr(*rhs, defs)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(inline_wide_lane_defs_expr(*cond, defs)),
            then_expr: Box::new(inline_wide_lane_defs_expr(*then_expr, defs)),
            else_expr: Box::new(inline_wide_lane_defs_expr(*else_expr, defs)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| inline_wide_lane_defs_expr(expr, defs))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| inline_wide_lane_defs_expr(expr, defs))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(inline_wide_lane_defs_expr(*addr, defs)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(inline_wide_lane_defs_expr(*base, defs)),
            offset: Box::new(inline_wide_lane_defs_expr(*offset, defs)),
        },
        Expr::Addr64 { lo, hi } => {
            let lo = inline_wide_lane_defs_expr(*lo, defs);
            let hi = inline_wide_lane_defs_expr(*hi, defs);
            collapse_split_wide_addr64(&lo, &hi).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(inline_wide_lane_defs_expr(*value, defs)),
            lane,
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(inline_wide_lane_defs_expr(*expr, defs)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(inline_wide_lane_defs_expr(*base, defs)),
            index: Box::new(inline_wide_lane_defs_expr(*index, defs)),
        },
        other => other,
    }
}

fn ast_seeded_wide_addr_fold(stmt: Stmt, seeded_wide_addrs: &SeededWideAddrMaps) -> Stmt {
    if seeded_wide_addrs.lo_by_name.is_empty() && seeded_wide_addrs.hi_by_name.is_empty() {
        return stmt;
    }
    seeded_wide_addr_fold_stmt(stmt, seeded_wide_addrs)
}

fn seeded_wide_addr_fold_stmt(stmt: Stmt, seeded_wide_addrs: &SeededWideAddrMaps) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(seeded_wide_addr_fold_stmt(*body, seeded_wide_addrs)),
        },
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .into_iter()
                .map(|stmt| seeded_wide_addr_fold_stmt(stmt, seeded_wide_addrs))
                .collect(),
        ),
        Stmt::Sequence(stmts) => Stmt::Sequence(
            stmts
                .into_iter()
                .map(|stmt| seeded_wide_addr_fold_stmt(stmt, seeded_wide_addrs))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: seeded_wide_addr_fold_expr(condition, seeded_wide_addrs),
            then_branch: Box::new(seeded_wide_addr_fold_stmt(*then_branch, seeded_wide_addrs)),
            else_branch: else_branch
                .map(|stmt| Box::new(seeded_wide_addr_fold_stmt(*stmt, seeded_wide_addrs))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(|expr| seeded_wide_addr_fold_expr(expr, seeded_wide_addrs)),
            body: Box::new(seeded_wide_addr_fold_stmt(*body, seeded_wide_addrs)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant
                .map(|expr| seeded_wide_addr_fold_expr(expr, seeded_wide_addrs)),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, seeded_wide_addr_fold_stmt(body, seeded_wide_addrs)))
                .collect(),
            default: default
                .map(|body| Box::new(seeded_wide_addr_fold_stmt(*body, seeded_wide_addrs))),
        },
        Stmt::Return(expr) => {
            Stmt::Return(expr.map(|expr| seeded_wide_addr_fold_expr(expr, seeded_wide_addrs)))
        }
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: seeded_wide_addr_fold_lvalue(dst, seeded_wide_addrs),
            src: seeded_wide_addr_fold_expr(src, seeded_wide_addrs),
        },
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(seeded_wide_addr_fold_expr(expr, seeded_wide_addrs)),
        other => other,
    }
}

fn seeded_wide_addr_fold_lvalue(lvalue: LValue, seeded_wide_addrs: &SeededWideAddrMaps) -> LValue {
    match lvalue {
        LValue::Raw(text) => LValue::Raw(rewrite_raw_seeded_addr64_calls(&text, seeded_wide_addrs)),
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(seeded_wide_addr_fold_expr(*addr, seeded_wide_addrs)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(seeded_wide_addr_fold_expr(*base, seeded_wide_addrs)),
            index: Box::new(seeded_wide_addr_fold_expr(*index, seeded_wide_addrs)),
        },
        other => other,
    }
}

fn seeded_wide_addr_fold_expr(expr: Expr, seeded_wide_addrs: &SeededWideAddrMaps) -> Expr {
    match expr {
        Expr::Raw(text) => Expr::Raw(rewrite_raw_seeded_addr64_calls(&text, seeded_wide_addrs)),
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(seeded_wide_addr_fold_expr(*arg, seeded_wide_addrs)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(seeded_wide_addr_fold_expr(*lhs, seeded_wide_addrs)),
            rhs: Box::new(seeded_wide_addr_fold_expr(*rhs, seeded_wide_addrs)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(seeded_wide_addr_fold_expr(*cond, seeded_wide_addrs)),
            then_expr: Box::new(seeded_wide_addr_fold_expr(*then_expr, seeded_wide_addrs)),
            else_expr: Box::new(seeded_wide_addr_fold_expr(*else_expr, seeded_wide_addrs)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| seeded_wide_addr_fold_expr(expr, seeded_wide_addrs))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| seeded_wide_addr_fold_expr(expr, seeded_wide_addrs))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(seeded_wide_addr_fold_expr(*addr, seeded_wide_addrs)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(seeded_wide_addr_fold_expr(*base, seeded_wide_addrs)),
            offset: Box::new(seeded_wide_addr_fold_expr(*offset, seeded_wide_addrs)),
        },
        Expr::Addr64 { lo, hi } => {
            let lo = seeded_wide_addr_fold_expr(*lo, seeded_wide_addrs);
            let hi = seeded_wide_addr_fold_expr(*hi, seeded_wide_addrs);
            fold_seeded_wide_addr_use(&lo, &hi, seeded_wide_addrs).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(seeded_wide_addr_fold_expr(*value, seeded_wide_addrs)),
            lane,
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(seeded_wide_addr_fold_expr(*expr, seeded_wide_addrs)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(seeded_wide_addr_fold_expr(*base, seeded_wide_addrs)),
            index: Box::new(seeded_wide_addr_fold_expr(*index, seeded_wide_addrs)),
        },
        other => other,
    }
}

fn fold_seeded_wide_addr_use(
    lo: &Expr,
    hi: &Expr,
    seeded_wide_addrs: &SeededWideAddrMaps,
) -> Option<Expr> {
    let lo_name = match_var_name(lo)?;
    let hi_name = match_var_name(hi)?;
    let lo_info = seeded_wide_addrs.lo_by_name.get(&lo_name)?;
    let hi_info = seeded_wide_addrs.hi_by_name.get(&hi_name)?;
    if lo_info.base != hi_info.base || lo_info.seed_lo != hi_info.seed_lo {
        return None;
    }
    let offset = if seeded_base_hi_expr(&lo_info.base)
        .is_some_and(|base_hi| same_match_expr(hi, &base_hi))
    {
        if *lo == lo_info.seed_lo {
            Expr::Imm("0".to_string())
        } else {
            Expr::Binary {
                op: "-".to_string(),
                lhs: Box::new(lo.clone()),
                rhs: Box::new(lo_info.seed_lo.clone()),
            }
        }
    } else {
        typed_full_width_delta_expr(lo.clone(), hi.clone(), lo_info.base.clone())
    };
    Some(render_folded_addr64_base(lo_info.base.clone(), &offset))
}

fn seeded_base_hi_expr(base: &Expr) -> Option<Expr> {
    match base {
        Expr::Raw(_) | Expr::Reg(_) | Expr::ConstMemSymbol(_) | Expr::Builtin(_) => {
            Some(Expr::LaneExtract {
                value: Box::new(base.clone()),
                lane: PointerLane::Hi32,
            })
        }
        Expr::Addr64 { hi, .. } => Some((**hi).clone()),
        Expr::WidePtr { base, .. } => seeded_base_hi_expr(base),
        _ => None,
    }
}

fn ast_typed_wide_addr_fold(stmt: Stmt) -> Stmt {
    let blocked = collect_multi_assigned_names(std::slice::from_ref(&stmt));
    typed_wide_addr_fold_stmt(stmt, &mut HashMap::new(), &blocked)
}

#[derive(Clone)]
struct TypedLaneSource {
    value: Expr,
    lane: PointerLane,
}

fn typed_wide_addr_fold_stmt(
    stmt: Stmt,
    defs: &mut HashMap<String, Expr>,
    blocked: &HashSet<String>,
) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(typed_wide_addr_fold_stmt(*body, defs, blocked)),
        },
        Stmt::Block(stmts) => {
            let mut local_defs = defs.clone();
            Stmt::Block(
                stmts
                    .into_iter()
                    .map(|stmt| typed_wide_addr_fold_stmt(stmt, &mut local_defs, blocked))
                    .collect(),
            )
        }
        Stmt::Sequence(stmts) => {
            let mut local_defs = defs.clone();
            Stmt::Sequence(
                stmts
                    .into_iter()
                    .map(|stmt| typed_wide_addr_fold_stmt(stmt, &mut local_defs, blocked))
                    .collect(),
            )
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = typed_wide_addr_fold_expr(condition, defs);
            let mut then_defs = defs.clone();
            let then_branch = typed_wide_addr_fold_stmt(*then_branch, &mut then_defs, blocked);
            let else_branch = if let Some(stmt) = else_branch {
                Some(Box::new(typed_wide_addr_fold_stmt(
                    *stmt,
                    &mut defs.clone(),
                    blocked,
                )))
            } else {
                defs.extend(then_defs);
                None
            };
            Stmt::If {
                condition,
                then_branch: Box::new(then_branch),
                else_branch,
            }
        }
        Stmt::Loop {
            kind,
            condition,
            body,
        } => {
            let condition = condition.map(|expr| typed_wide_addr_fold_expr(expr, defs));
            let loop_names = collect_loop_carried_names(body.as_ref());
            let loop_seed_defs = defs.clone();
            let mut body_defs = defs.clone();
            for name in &loop_names {
                if loop_seed_defs
                    .get(name)
                    .and_then(extract_lane_def_expr)
                    .is_none()
                {
                    body_defs.remove(name);
                }
            }
            seed_loop_carried_typed_lane_defs(body.as_ref(), &loop_seed_defs, &mut body_defs);
            let body = typed_wide_addr_fold_stmt(*body, &mut body_defs, blocked);
            defs.extend(body_defs);
            Stmt::Loop {
                kind,
                condition,
                body: Box::new(body),
            }
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(|expr| typed_wide_addr_fold_expr(expr, defs)),
            cases: cases
                .into_iter()
                .map(|(label, body)| {
                    (
                        label,
                        typed_wide_addr_fold_stmt(body, &mut defs.clone(), blocked),
                    )
                })
                .collect(),
            default: default
                .map(|body| Box::new(typed_wide_addr_fold_stmt(*body, &mut defs.clone(), blocked))),
        },
        Stmt::Return(expr) => Stmt::Return(expr.map(|expr| typed_wide_addr_fold_expr(expr, defs))),
        Stmt::Assign { dst, src } => {
            let dst = typed_wide_addr_fold_lvalue(dst, defs);
            let src = typed_wide_addr_fold_expr(src, defs);
            if let LValue::Var(name) = &dst {
                if let Some(lane_expr) = extract_lane_def_expr_with_defs(&src, defs) {
                    let mut used = BTreeSet::new();
                    collect_used_expr_vars(&lane_expr, &mut used);
                    if blocked.contains(name) && used.contains(name) {
                        defs.remove(name);
                    } else if used.contains(name) {
                        defs.remove(name);
                    } else {
                        defs.insert(name.clone(), lane_expr);
                    }
                } else {
                    defs.remove(name);
                }
            }
            Stmt::Assign { dst, src }
        }
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(typed_wide_addr_fold_expr(expr, defs)),
        other => other,
    }
}

fn typed_wide_addr_fold_lvalue(lvalue: LValue, defs: &HashMap<String, Expr>) -> LValue {
    match lvalue {
        LValue::Deref { ty, addr } => {
            let original_addr = (*addr).clone();
            if ty.is_none() {
                if let Some(addr) =
                    recover_plain_symbolic_pointer_index_from_wide_addr(&original_addr)
                {
                    return LValue::Deref {
                        ty,
                        addr: Box::new(addr),
                    };
                }
            }
            let addr = typed_wide_addr_fold_expr(*addr, defs);
            let rewritten = ty.as_deref().and_then(|ty_name| {
                recover_explicit_symbolic_pointer_index(&original_addr, ty_name, defs).or_else(
                    || {
                        should_apply_symbolic_index_recovery(&original_addr, ty_name)
                            .then(|| recover_symbolic_pointer_index_from_wide_addr(&addr, ty_name))
                            .flatten()
                    },
                )
            });
            let untyped_index = ty
                .is_none()
                .then(|| recover_plain_symbolic_pointer_index_from_wide_addr(&addr))
                .flatten();
            LValue::Deref {
                ty,
                addr: Box::new(rewritten.or(untyped_index).unwrap_or(addr)),
            }
        }
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(typed_wide_addr_fold_expr(*base, defs)),
            index: Box::new(typed_wide_addr_fold_expr(*index, defs)),
        },
        other => other,
    }
}

fn typed_wide_addr_fold_expr(expr: Expr, defs: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(typed_wide_addr_fold_expr(*arg, defs)),
        },
        Expr::Binary { op, lhs, rhs } => {
            let lhs = typed_wide_addr_fold_expr(*lhs, defs);
            let rhs = typed_wide_addr_fold_expr(*rhs, defs);
            let expr = fold_resolved_typed_lane_expr(
                Expr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                defs,
            );
            if symbolic_pointer_addr_is_already_indexed(&expr) {
                expr
            } else {
                fold_base_relative_wide_expr(&expr, defs).unwrap_or_else(|| {
                    match_base_relative_wide_ptr_expr(&expr)
                        .map(|(base, offset)| simplify_typed_wide_ptr_expr(base, offset, defs))
                        .unwrap_or(expr)
                })
            }
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(typed_wide_addr_fold_expr(*cond, defs)),
            then_expr: Box::new(typed_wide_addr_fold_expr(*then_expr, defs)),
            else_expr: Box::new(typed_wide_addr_fold_expr(*else_expr, defs)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|expr| typed_wide_addr_fold_expr(expr, defs))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|expr| typed_wide_addr_fold_expr(expr, defs))
                .collect(),
        },
        Expr::Load { ty, addr } => {
            let original_addr = (*addr).clone();
            if ty.is_none() {
                if let Some(addr) =
                    recover_plain_symbolic_pointer_index_from_wide_addr(&original_addr)
                {
                    return Expr::Load {
                        ty,
                        addr: Box::new(addr),
                    };
                }
            }
            let addr = {
                let addr = typed_wide_addr_fold_expr(*addr, defs);
                if symbolic_pointer_addr_is_already_indexed(&addr) {
                    addr
                } else {
                    match_base_relative_wide_ptr_expr(&addr)
                        .map(|(base, offset)| simplify_typed_wide_ptr_expr(base, offset, defs))
                        .unwrap_or(addr)
                }
            };
            let rewritten = ty.as_deref().and_then(|ty_name| {
                recover_explicit_symbolic_pointer_index(&original_addr, ty_name, defs).or_else(
                    || {
                        should_apply_symbolic_index_recovery(&original_addr, ty_name)
                            .then(|| recover_symbolic_pointer_index_from_wide_addr(&addr, ty_name))
                            .flatten()
                    },
                )
            });
            let untyped_index = ty
                .is_none()
                .then(|| recover_plain_symbolic_pointer_index_from_wide_addr(&addr))
                .flatten();
            Expr::Load {
                ty,
                addr: Box::new(rewritten.or(untyped_index).unwrap_or(addr)),
            }
        }
        Expr::WidePtr { base, offset } => simplify_typed_wide_ptr_expr(
            typed_wide_addr_fold_expr(*base, defs),
            typed_wide_addr_fold_expr(*offset, defs),
            defs,
        ),
        Expr::Addr64 { lo, hi } => {
            let lo = typed_wide_addr_fold_expr(*lo, defs);
            let hi = typed_wide_addr_fold_expr(*hi, defs);
            collapse_typed_addr64_pair(&lo, &hi, defs).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::Cast { ty, expr } => {
            let expr = typed_wide_addr_fold_expr(*expr, defs);
            let cast = fold_typed_explicit_lane_extract(
                Expr::Cast {
                    ty: ty.clone(),
                    expr: Box::new(expr),
                },
                defs,
            );
            match cast {
                Expr::Cast { ty, expr }
                    if ty.trim_end().ends_with('*')
                        || matches!(ty.as_str(), "uintptr_t" | "intptr_t") =>
                {
                    if let Some((base, offset)) =
                        match_typed_packed_wide_ptr_expr(expr.as_ref(), defs)
                            .or_else(|| match_base_relative_wide_ptr_expr(expr.as_ref()))
                    {
                        let collapsed = simplify_typed_wide_ptr_expr(base, offset, defs);
                        if ty.trim_end().ends_with('*') {
                            collapsed
                        } else {
                            Expr::Cast {
                                ty,
                                expr: Box::new(collapsed),
                            }
                        }
                    } else {
                        Expr::Cast { ty, expr }
                    }
                }
                other => other,
            }
        }
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(typed_wide_addr_fold_expr(*base, defs)),
            index: Box::new(typed_wide_addr_fold_expr(*index, defs)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(typed_wide_addr_fold_expr(*value, defs)),
            lane,
        },
        other => other,
    }
}

fn recover_explicit_symbolic_pointer_index(
    addr: &Expr,
    ty: &str,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let Expr::Addr64 { lo, hi } = addr else {
        return None;
    };
    let byte_addr = resolve_direct_explicit_lo_lane_value(lo, defs)?;
    let (base, byte_offset) = match_base_relative_wide_ptr_expr(&byte_addr)?;
    if !expr_is_symbolic_name_pointer_base(&base) {
        return None;
    }

    let hi_value = resolve_direct_explicit_hi_lane_value(hi, defs)?;
    let (hi_base, _) = match_base_relative_wide_ptr_expr(&hi_value)?;
    if hi_base != base {
        return None;
    }

    let index = divide_pointer_index_expr(&byte_offset, scalar_type_size_bytes_ast(ty)?)?;
    if expr_is_zero(&index) {
        return Some(base);
    }
    Some(Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(base),
        rhs: Box::new(index),
    })
}

fn resolve_direct_explicit_lo_lane_value(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let expr = strip_loop_phi_expr(expr);
    if let Some(value) = syntactic_explicit_lo_lane_value(expr) {
        return Some(value);
    }
    match expr {
        Expr::Reg(name) => defs
            .get(name)
            .and_then(|value| syntactic_explicit_lo_lane_value(value)),
        Expr::Raw(text) if is_symbolic_name(text) => defs
            .get(text)
            .and_then(|value| syntactic_explicit_lo_lane_value(value)),
        _ => None,
    }
}

fn resolve_direct_explicit_hi_lane_value(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let expr = strip_loop_phi_expr(expr);
    if let Some(value) = syntactic_explicit_hi_lane_value(expr) {
        return Some(value);
    }
    match expr {
        Expr::Reg(name) => defs
            .get(name)
            .and_then(|value| syntactic_explicit_hi_lane_value(value)),
        Expr::Raw(text) if is_symbolic_name(text) => defs
            .get(text)
            .and_then(|value| syntactic_explicit_hi_lane_value(value)),
        _ => None,
    }
}

fn recover_symbolic_pointer_index_from_wide_addr(addr: &Expr, ty: &str) -> Option<Expr> {
    let elem_size = scalar_type_size_bytes_ast(ty)?;
    let normalized = normalize_pointer_recovery_addr_expr(addr);
    let (base, byte_offset) = match_base_relative_wide_ptr_expr(&normalized)?;
    if !expr_is_symbolic_name_pointer_base(&base) {
        return None;
    }
    let index = divide_pointer_index_expr(&byte_offset, elem_size)?;
    if expr_mentions_symbolic_pointer_base_expr(&index, &base) {
        return None;
    }
    if expr_is_zero(&index) {
        return Some(base);
    }
    Some(Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(base),
        rhs: Box::new(index),
    })
}

fn recover_plain_symbolic_pointer_index_from_wide_addr(addr: &Expr) -> Option<Expr> {
    let normalized = normalize_pointer_recovery_addr_expr(addr);
    let (base, offset) = match_base_relative_wide_ptr_expr(&normalized)?;
    if !expr_is_symbolic_name_pointer_base(&base) {
        return None;
    }
    if expr_mentions_symbolic_pointer_base_expr(&offset, &base) {
        return None;
    }
    let offset = plain_symbolic_pointer_offset_expr(&offset)?;
    if expr_is_zero(&offset) {
        return Some(base);
    }
    Some(Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(base),
        rhs: Box::new(offset),
    })
}

fn normalize_pointer_recovery_addr_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Cast { ty, expr } if is_transparent_pointer_recovery_cast(ty) => {
            normalize_pointer_recovery_addr_expr(expr)
        }
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => Expr::Binary {
            op: op.clone(),
            lhs: Box::new(normalize_pointer_recovery_addr_expr(lhs)),
            rhs: Box::new(normalize_pointer_recovery_addr_expr(rhs)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(normalize_pointer_recovery_addr_expr(base)),
            offset: Box::new(normalize_pointer_recovery_addr_expr(offset)),
        },
        other => other.clone(),
    }
}

fn is_transparent_pointer_recovery_cast(ty: &str) -> bool {
    matches!(
        ty.trim(),
        "uint32_t" | "int32_t" | "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t"
    ) || ty.trim_end().ends_with('*')
}

fn plain_symbolic_pointer_offset_expr(expr: &Expr) -> Option<Expr> {
    let normalized = normalize_pointer_recovery_addr_expr(expr);
    match strip_offset_like_casts(&normalized) {
        Expr::Imm(_)
        | Expr::Raw(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. } => Some(strip_offset_like_casts(&normalized)),
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => Some(Expr::Binary {
            op,
            lhs: Box::new(plain_symbolic_pointer_offset_expr(&lhs)?),
            rhs: Box::new(plain_symbolic_pointer_offset_expr(&rhs)?),
        }),
        other if expr_is_zero(&other) => Some(Expr::Imm("0".to_string())),
        _ => None,
    }
}

fn expr_is_symbolic_name_pointer_base(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text)
            if text.ends_with("_ptr")
    )
}

fn scalar_type_size_bytes_ast(ty: &str) -> Option<i64> {
    match ty.trim() {
        "uint8_t" | "int8_t" | "bool" => Some(1),
        "uint16_t" | "int16_t" | "__half" => Some(2),
        "uint32_t" | "int32_t" | "float" => Some(4),
        "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t" | "double" => Some(8),
        _ => None,
    }
}

fn divide_pointer_index_expr(expr: &Expr, divisor: i64) -> Option<Expr> {
    if divisor == 1 {
        return Some(strip_offset_like_casts(expr));
    }
    match strip_offset_like_casts(expr) {
        Expr::Imm(text) | Expr::Raw(text) => {
            let value = text.parse::<i64>().ok()?;
            (value % divisor == 0).then(|| Expr::Imm((value / divisor).to_string()))
        }
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => {
            let lhs = divide_pointer_index_expr(&lhs, divisor)?;
            let rhs = divide_pointer_index_expr(&rhs, divisor)?;
            Some(Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        Expr::Binary { op, lhs, rhs } if op == "*" => {
            if let Some(factor) = expr_i64(&rhs) {
                return divide_pointer_index_scaled_product(&lhs, factor, divisor);
            }
            if let Some(factor) = expr_i64(&lhs) {
                return divide_pointer_index_scaled_product(&rhs, factor, divisor);
            }
            None
        }
        other if expr_is_zero(&other) => Some(Expr::Imm("0".to_string())),
        _ => None,
    }
}

fn divide_pointer_index_scaled_product(expr: &Expr, factor: i64, divisor: i64) -> Option<Expr> {
    (factor % divisor == 0).then(|| {
        let reduced = factor / divisor;
        let base = strip_offset_like_casts(expr);
        if reduced == 1 {
            base
        } else {
            Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(base),
                rhs: Box::new(Expr::Imm(reduced.to_string())),
            }
        }
    })
}

fn expr_mentions_symbolic_pointer_base_expr(expr: &Expr, base: &Expr) -> bool {
    let Some(base_name) = (match base {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            Some(text.as_str())
        }
        _ => None,
    }) else {
        return false;
    };

    match expr {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            text == base_name
        }
        Expr::PtrLane { base, .. } => base == base_name,
        Expr::LaneExtract { value, .. } | Expr::Cast { expr: value, .. } => {
            expr_mentions_symbolic_pointer_base_expr(value, base)
        }
        Expr::Unary { arg, .. } => expr_mentions_symbolic_pointer_base_expr(arg, base),
        Expr::Binary { lhs, rhs, .. } => {
            expr_mentions_symbolic_pointer_base_expr(lhs, base)
                || expr_mentions_symbolic_pointer_base_expr(rhs, base)
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_mentions_symbolic_pointer_base_expr(cond, base)
                || expr_mentions_symbolic_pointer_base_expr(then_expr, base)
                || expr_mentions_symbolic_pointer_base_expr(else_expr, base)
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => args
            .iter()
            .any(|arg| expr_mentions_symbolic_pointer_base_expr(arg, base)),
        Expr::Load { addr, .. } => expr_mentions_symbolic_pointer_base_expr(addr, base),
        Expr::WidePtr {
            base: inner_base,
            offset,
        } => {
            expr_mentions_symbolic_pointer_base_expr(inner_base, base)
                || expr_mentions_symbolic_pointer_base_expr(offset, base)
        }
        Expr::Addr64 { lo, hi } => {
            expr_mentions_symbolic_pointer_base_expr(lo, base)
                || expr_mentions_symbolic_pointer_base_expr(hi, base)
        }
        Expr::Index {
            base: inner_base,
            index,
        } => {
            expr_mentions_symbolic_pointer_base_expr(inner_base, base)
                || expr_mentions_symbolic_pointer_base_expr(index, base)
        }
        Expr::Imm(_) => false,
    }
}

fn expr_contains_addr64(expr: &Expr) -> bool {
    match expr {
        Expr::Addr64 { .. } => true,
        Expr::Unary { arg, .. } | Expr::Cast { expr: arg, .. } => expr_contains_addr64(arg),
        Expr::Binary { lhs, rhs, .. } => expr_contains_addr64(lhs) || expr_contains_addr64(rhs),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_contains_addr64(cond)
                || expr_contains_addr64(then_expr)
                || expr_contains_addr64(else_expr)
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            args.iter().any(expr_contains_addr64)
        }
        Expr::Load { addr, .. } => expr_contains_addr64(addr),
        Expr::WidePtr { base, offset } => {
            expr_contains_addr64(base) || expr_contains_addr64(offset)
        }
        Expr::Index { base, index } => expr_contains_addr64(base) || expr_contains_addr64(index),
        Expr::LaneExtract { value, .. } => expr_contains_addr64(value),
        Expr::Imm(_)
        | Expr::Raw(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::PtrLane { .. } => false,
    }
}

fn expr_has_pointer_cast(expr: &Expr) -> bool {
    match expr {
        Expr::Cast { ty, expr } => ty.trim_end().ends_with('*') || expr_has_pointer_cast(expr),
        Expr::Unary { arg, .. } => expr_has_pointer_cast(arg),
        Expr::Binary { lhs, rhs, .. } => expr_has_pointer_cast(lhs) || expr_has_pointer_cast(rhs),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_has_pointer_cast(cond)
                || expr_has_pointer_cast(then_expr)
                || expr_has_pointer_cast(else_expr)
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            args.iter().any(expr_has_pointer_cast)
        }
        Expr::Load { addr, .. } => expr_has_pointer_cast(addr),
        Expr::WidePtr { base, offset } => {
            expr_has_pointer_cast(base) || expr_has_pointer_cast(offset)
        }
        Expr::Addr64 { lo, hi } => expr_has_pointer_cast(lo) || expr_has_pointer_cast(hi),
        Expr::Index { base, index } => expr_has_pointer_cast(base) || expr_has_pointer_cast(index),
        Expr::LaneExtract { value, .. } => expr_has_pointer_cast(value),
        Expr::Imm(_)
        | Expr::Raw(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::PtrLane { .. } => false,
    }
}

fn should_apply_symbolic_index_recovery(original_addr: &Expr, ty: &str) -> bool {
    if expr_contains_addr64(original_addr) || expr_has_pointer_cast(original_addr) {
        return false;
    }
    let elem_size = scalar_type_size_bytes_ast(ty).unwrap_or(0);
    if elem_size == 1 {
        !matches!(original_addr, Expr::WidePtr { .. })
    } else {
        !symbolic_pointer_addr_is_already_indexed(original_addr)
    }
}

fn symbolic_pointer_addr_is_already_indexed(addr: &Expr) -> bool {
    let normalized = normalize_pointer_recovery_addr_expr(addr);
    if matches!(normalized, Expr::WidePtr { .. }) {
        return false;
    }
    let Some((base, offset)) = match_base_relative_wide_ptr_expr(&normalized) else {
        return false;
    };
    expr_is_symbolic_name_pointer_base(&base)
        && plain_symbolic_pointer_offset_expr(&offset).is_some()
}

fn fold_resolved_typed_lane_expr(expr: Expr, defs: &HashMap<String, Expr>) -> Expr {
    if let Some(collapsed) = fold_resolved_typed_lea_hi_expr(&expr, defs) {
        return collapsed;
    }
    if let Some(collapsed) = fold_resolved_typed_scaled_hi_expr(&expr, defs) {
        return collapsed;
    }
    let Expr::Binary { op, lhs, rhs } = &expr else {
        return expr;
    };
    if op != "+" {
        return expr;
    }
    fold_resolved_typed_lo_lane_add(lhs, rhs, defs)
        .or_else(|| fold_resolved_typed_lo_lane_add(rhs, lhs, defs))
        .or_else(|| fold_resolved_typed_hi_lane_add(lhs, rhs, defs))
        .or_else(|| fold_resolved_typed_hi_lane_add(rhs, lhs, defs))
        .unwrap_or(expr)
}

fn fold_resolved_typed_lea_hi_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let args = intrinsic_args(expr, IntrinsicOp::LeaHiX)
        .or_else(|| intrinsic_args(expr, IntrinsicOp::LeaHiXSx32))?;
    if args.len() != 4 {
        return None;
    }
    let hi_source = resolve_typed_lane_source(&args[1], defs)?;
    if hi_source.lane != PointerLane::Hi32 {
        return None;
    }
    let (carry_lo_source, carry_step) = match_typed_carry_increment(&args[3], defs)?;
    if carry_lo_source.lane != PointerLane::Lo32 {
        return None;
    }
    let base = typed_wide_value_base(&hi_source.value);
    if typed_wide_value_base(&carry_lo_source.value) != base {
        return None;
    }
    let hi_offset = typed_wide_value_offset(&hi_source.value, &base)
        .unwrap_or_else(|| Expr::Imm("0".to_string()));
    let lo_offset = typed_wide_value_offset(&carry_lo_source.value, &base)
        .unwrap_or_else(|| Expr::Imm("0".to_string()));
    if normalize_offset_expr(hi_offset.clone()) != normalize_offset_expr(lo_offset) {
        return None;
    }
    let step = scale_addr_offset_expr(&args[0], &args[2])?;
    if !same_match_expr(&step, &carry_step) {
        return None;
    }
    Some(Expr::LaneExtract {
        value: Box::new(simplify_typed_wide_ptr_expr(
            base,
            combine_offset_expr(hi_offset, step, "+"),
            defs,
        )),
        lane: PointerLane::Hi32,
    })
}

fn fold_resolved_typed_scaled_hi_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let Expr::Binary { .. } = expr else {
        return None;
    };
    let mut terms = Vec::new();
    collect_assoc_match_terms(expr, "+", &mut terms);
    if terms.len() < 3 {
        return None;
    }

    for carry_idx in 0..terms.len() {
        let Some((carry_lo_source, step)) = match_typed_carry_increment(terms[carry_idx], defs)
        else {
            continue;
        };
        if carry_lo_source.lane != PointerLane::Lo32 {
            continue;
        }
        let base = typed_wide_value_base(&carry_lo_source.value);
        let lo_offset = typed_wide_value_offset(&carry_lo_source.value, &base)
            .unwrap_or_else(|| Expr::Imm("0".to_string()));

        for hi_idx in 0..terms.len() {
            if hi_idx == carry_idx {
                continue;
            }
            let Some(hi_source) = resolve_typed_lane_source(terms[hi_idx], defs) else {
                continue;
            };
            if hi_source.lane != PointerLane::Hi32 {
                continue;
            }
            let hi_base = typed_wide_value_base(&hi_source.value);
            if hi_base != base {
                continue;
            }
            let hi_offset = typed_wide_value_offset(&hi_source.value, &hi_base)
                .unwrap_or_else(|| Expr::Imm("0".to_string()));
            if normalize_offset_expr(hi_offset.clone()) != normalize_offset_expr(lo_offset.clone())
            {
                continue;
            }

            let extras = terms
                .iter()
                .enumerate()
                .filter_map(|(idx, term)| (idx != carry_idx && idx != hi_idx).then_some(*term))
                .collect::<Vec<_>>();
            if extras.len() != 1 || !expr_matches_step_hi_term(extras[0], &step) {
                continue;
            }

            return Some(Expr::LaneExtract {
                value: Box::new(simplify_typed_wide_ptr_expr(
                    base,
                    combine_offset_expr(hi_offset, step, "+"),
                    defs,
                )),
                lane: PointerLane::Hi32,
            });
        }
    }

    None
}

fn expr_matches_step_hi_term(expr: &Expr, step: &Expr) -> bool {
    let Some((term, factor)) = match_scaled_step_factor(step) else {
        return false;
    };
    let Expr::CallLike { func, args } = expr else {
        return false;
    };
    if func != "mul_hi_u32" || args.len() != 2 {
        return false;
    }
    (same_match_expr(&args[0], &term) && expr_imm_as_u32(&args[1]) == Some(factor))
        || (same_match_expr(&args[1], &term) && expr_imm_as_u32(&args[0]) == Some(factor))
}

fn match_scaled_step_factor(expr: &Expr) -> Option<(Expr, u32)> {
    let expr = strip_offset_like_casts(expr);
    let Expr::Binary { op, lhs, rhs } = &expr else {
        return None;
    };
    if op != "*" {
        return None;
    }
    if let Some(factor) = expr_imm_as_u32(rhs) {
        return Some(((**lhs).clone(), factor));
    }
    expr_imm_as_u32(lhs).map(|factor| ((**rhs).clone(), factor))
}

fn fold_typed_explicit_lane_extract(expr: Expr, defs: &HashMap<String, Expr>) -> Expr {
    match_typed_explicit_lane_extract(&expr, defs)
        .map(|(value, lane)| Expr::LaneExtract {
            value: Box::new(value),
            lane,
        })
        .unwrap_or(expr)
}

fn match_typed_explicit_lane_extract(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, PointerLane)> {
    if let Some(value_expr) = syntactic_explicit_lo_lane_value(expr) {
        let (base, offset) = match_typed_wide_value_parts(&value_expr, defs)
            .or_else(|| match_lo_backed_wide_value_parts(&value_expr, defs))?;
        return Some((
            simplify_typed_wide_ptr_expr(base, offset, defs),
            PointerLane::Lo32,
        ));
    }
    let value_expr = syntactic_explicit_hi_lane_value(expr)?;
    let (base, offset) = match_typed_wide_value_parts(&value_expr, defs)
        .or_else(|| match_lo_backed_wide_value_parts(&value_expr, defs))?;
    Some((
        simplify_typed_wide_ptr_expr(base, offset, defs),
        PointerLane::Hi32,
    ))
}

fn match_lo_backed_wide_value_parts(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr)> {
    if let Some((lo, _hi)) = match_u64_pack_expr(expr) {
        return match_lo_backed_wide_value_parts(
            &Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(Expr::Imm("0".to_string())),
            },
            defs,
        );
    }

    let expr = strip_wide_casts(expr);
    match expr {
        Expr::Addr64 { lo, .. } => {
            let lo_source = resolve_typed_lane_source(lo, defs)?;
            if lo_source.lane != PointerLane::Lo32 {
                return None;
            }
            let base = typed_wide_value_base(&lo_source.value);
            let offset = typed_wide_value_offset(&lo_source.value, &base)
                .unwrap_or_else(|| Expr::Imm("0".to_string()));
            Some((base, offset))
        }
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => {
            if let Some((base, offset)) = match_lo_backed_wide_value_parts(lhs, defs) {
                return Some((
                    base,
                    combine_offset_expr(offset, strip_offset_like_casts(rhs), op),
                ));
            }
            if op == "+" {
                if let Some((base, offset)) = match_lo_backed_wide_value_parts(rhs, defs) {
                    return Some((
                        base,
                        combine_offset_expr(offset, strip_offset_like_casts(lhs), "+"),
                    ));
                }
            }
            None
        }
        _ => None,
    }
}

fn syntactic_explicit_lo_lane_value(expr: &Expr) -> Option<Expr> {
    let Expr::Cast { ty, expr } = expr else {
        return None;
    };
    if ty != "uint32_t" {
        return None;
    }
    Some(strip_wide_casts(expr).clone())
}

fn syntactic_explicit_hi_lane_value(expr: &Expr) -> Option<Expr> {
    let Expr::Cast { ty, expr } = expr else {
        return None;
    };
    if ty != "uint32_t" {
        return None;
    }
    let Expr::Binary { op, lhs, rhs } = expr.as_ref() else {
        return None;
    };
    if op != ">>" || expr_imm_as_u32(rhs) != Some(32) {
        return None;
    }
    Some(strip_wide_casts(lhs).clone())
}

fn fold_resolved_typed_lo_lane_add(
    lane_side: &Expr,
    offset_side: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let source = resolve_typed_lane_source(lane_side, defs)?;
    if source.lane != PointerLane::Lo32 {
        return None;
    }
    let base = typed_wide_value_base(&source.value);
    let prev_offset =
        typed_wide_value_offset(&source.value, &base).unwrap_or_else(|| Expr::Imm("0".to_string()));
    let next_offset = combine_offset_expr(prev_offset, offset_side.clone(), "+");
    Some(Expr::LaneExtract {
        value: Box::new(simplify_typed_wide_ptr_expr(base, next_offset, defs)),
        lane: PointerLane::Lo32,
    })
}

fn fold_resolved_typed_hi_lane_add(
    hi_side: &Expr,
    carry_side: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let hi_source = resolve_typed_lane_source(hi_side, defs)?;
    if hi_source.lane != PointerLane::Hi32 {
        return None;
    }
    let base = typed_wide_value_base(&hi_source.value);
    let hi_offset = typed_wide_value_offset(&hi_source.value, &base)
        .unwrap_or_else(|| Expr::Imm("0".to_string()));
    let (carry_lo_source, step) = match_typed_carry_increment(carry_side, defs)?;
    if carry_lo_source.lane != PointerLane::Lo32 {
        return None;
    }
    let lo_base = typed_wide_value_base(&carry_lo_source.value);
    if lo_base != base {
        return None;
    }
    let lo_offset = typed_wide_value_offset(&carry_lo_source.value, &lo_base)
        .unwrap_or_else(|| Expr::Imm("0".to_string()));
    if normalize_offset_expr(lo_offset) != normalize_offset_expr(hi_offset.clone()) {
        return None;
    }
    let next_offset = combine_offset_expr(hi_offset, step, "+");
    Some(Expr::LaneExtract {
        value: Box::new(simplify_typed_wide_ptr_expr(base, next_offset, defs)),
        lane: PointerLane::Hi32,
    })
}

fn match_typed_carry_increment(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(TypedLaneSource, Expr)> {
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
    let args = intrinsic_args(cond, IntrinsicOp::CarryU32Add3)?;
    if args.len() != 3 || !expr_is_zero(&args[2]) {
        return None;
    }

    for (lane_expr, step_expr) in [(&args[0], &args[1]), (&args[1], &args[0])] {
        let Some(lane_source) = resolve_typed_lane_source(lane_expr, defs) else {
            continue;
        };
        if lane_source.lane != PointerLane::Lo32 {
            continue;
        }
        return Some((lane_source, step_expr.clone()));
    }
    None
}

fn fold_base_relative_wide_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let Expr::Binary { .. } = expr else {
        return None;
    };
    let mut terms = Vec::<SignedOffsetTerm>::new();
    collect_signed_offset_terms(expr, true, &mut terms);
    for (base_idx, base_term) in terms.iter().enumerate() {
        if !base_term.positive {
            continue;
        }
        let (base, mut offset_terms) = if let Some((base, base_offset)) =
            match_typed_wide_value_parts(&base_term.expr, defs)
        {
            let mut offset_terms = Vec::new();
            collect_signed_offset_terms(&base_offset, true, &mut offset_terms);
            (base, offset_terms)
        } else {
            (base_term.expr.clone(), Vec::new())
        };
        offset_terms.extend(
            terms
                .iter()
                .enumerate()
                .filter_map(|(idx, term)| (idx != base_idx).then_some(term.clone())),
        );
        let offset_expr = rebuild_signed_offset_terms(offset_terms);
        let folded_offset = fold_base_relative_wide_offset(&base, &offset_expr, defs)?;
        return Some(simplify_typed_wide_ptr_expr(base, folded_offset, defs));
    }
    None
}

fn match_typed_wide_value_parts(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    match expr {
        Expr::WidePtr { base, offset } => Some(((**base).clone(), (**offset).clone())),
        Expr::Addr64 { lo, hi } => {
            if let Some(collapsed) = collapse_typed_addr64_pair(lo, hi, defs) {
                return match_typed_wide_value_parts(&collapsed, defs);
            }
            let hi_source = resolve_typed_lane_source(hi, defs)?;
            if hi_source.lane != PointerLane::Hi32 {
                return None;
            }
            let base = typed_wide_value_base(&hi_source.value);
            let offset = typed_wide_value_offset(&hi_source.value, &base)
                .unwrap_or_else(|| Expr::Imm("0".to_string()));
            Some((base, offset))
        }
        _ => match_typed_packed_wide_ptr_expr(expr, defs)
            .or_else(|| match_base_relative_wide_ptr_expr(expr)),
    }
}

fn match_typed_packed_wide_ptr_expr(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr)> {
    if let Some((lo, hi)) = match_u64_pack_expr(expr) {
        let collapsed = collapse_typed_addr64_pair(&lo, &hi, defs)?;
        return match_typed_wide_value_parts(&collapsed, defs);
    }

    let expr = strip_wide_casts(expr);
    if let Expr::Addr64 { lo, hi } = expr {
        let collapsed = collapse_typed_addr64_pair(lo, hi, defs)?;
        return match_typed_wide_value_parts(&collapsed, defs);
    }
    let Expr::Binary { op, lhs, rhs } = expr else {
        return None;
    };
    if op != "+" && op != "-" {
        return None;
    }
    if let Some((base, offset)) = match_typed_packed_wide_ptr_expr(lhs, defs) {
        return Some((
            base,
            combine_offset_expr(offset, strip_offset_like_casts(rhs), op),
        ));
    }
    if op == "+" {
        if let Some((base, offset)) = match_typed_packed_wide_ptr_expr(rhs, defs) {
            return Some((
                base,
                combine_offset_expr(offset, strip_offset_like_casts(lhs), "+"),
            ));
        }
    }
    None
}

fn match_u64_pack_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    let expr = strip_wide_casts(expr);
    let Expr::Binary { op, lhs, rhs } = expr else {
        return None;
    };
    if op != "|" {
        return None;
    }

    match (
        match_u64_pack_lo_part(lhs),
        match_u64_pack_hi_part(rhs),
        match_u64_pack_lo_part(rhs),
        match_u64_pack_hi_part(lhs),
    ) {
        (Some(lo), Some(hi), _, _) => Some((lo, hi)),
        (_, _, Some(lo), Some(hi)) => Some((lo, hi)),
        _ => None,
    }
}

fn match_u64_pack_lo_part(expr: &Expr) -> Option<Expr> {
    match strip_wide_casts(expr) {
        Expr::Cast { ty, expr } if ty == "uint32_t" => Some((**expr).clone()),
        other => Some(strip_lane_low_casts(other)),
    }
}

fn match_u64_pack_hi_part(expr: &Expr) -> Option<Expr> {
    let Expr::Binary { op, lhs, rhs } = strip_wide_casts(expr) else {
        return None;
    };
    if op != "<<" || expr_imm_as_u32(rhs) != Some(32) {
        return None;
    }
    match strip_wide_casts(lhs) {
        Expr::Cast { ty, expr } if matches!(ty.as_str(), "uint32_t" | "int32_t") => {
            Some((**expr).clone())
        }
        other => Some(strip_lane_low_casts(other)),
    }
}

fn collapse_typed_addr64_pair(lo: &Expr, hi: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    if let Some(expr) = collapse_split_wide_addr64(lo, hi) {
        return Some(expr);
    }
    if let Some(expr) = collapse_lo_backed_typed_addr64_pair(lo, hi, defs) {
        return Some(expr);
    }
    if let Some(expr) = collapse_scalar_wide_addr64_pair(lo, hi, defs) {
        return Some(expr);
    }
    let lo_source = resolve_typed_lane_source(lo, defs)?;
    let hi_source = resolve_typed_lane_source(hi, defs)?;
    if lo_source.lane != PointerLane::Lo32 || hi_source.lane != PointerLane::Hi32 {
        return None;
    }
    let base = common_typed_wide_base(&lo_source.value, &hi_source.value)?;
    Some(simplify_wide_ptr_expr(
        base.clone(),
        symbolic_typed_wide_offset(lo, hi, &lo_source.value, &base),
    ))
}

fn collapse_lo_backed_typed_addr64_pair(
    lo: &Expr,
    hi: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let (base, offset) = match_lo_backed_wide_value_parts(
        &Expr::Addr64 {
            lo: Box::new(lo.clone()),
            hi: Box::new(hi.clone()),
        },
        defs,
    )?;
    if match_base_relative_wide_ptr_expr(&base).is_none() {
        return None;
    }
    if !hi_expr_matches_lo_backed_base(hi, &base, defs) {
        return None;
    }
    if offset_is_symbolic_lo_lane_delta(&offset, lo, &base) {
        return None;
    }
    Some(simplify_typed_wide_ptr_expr(base, offset, defs))
}

fn collapse_scalar_wide_addr64_pair(
    lo: &Expr,
    hi: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let hi_value = resolve_scalar_hi_lane_value(hi, defs)?;
    let canonical = canonicalize_scalar_wide_value(&hi_value, defs).unwrap_or(hi_value);

    if let Some(lo_value) = resolve_scalar_lo_lane_value(lo, defs) {
        let canonical_lo = canonicalize_scalar_wide_value(&lo_value, defs).unwrap_or(lo_value);
        if same_match_expr(&canonical_lo, &canonical) {
            return Some(canonical);
        }
    }

    let expected_lo = low_32_expr_for_wide_value(&canonical, defs)?;
    let actual_lo = resolve_scalar_lane_expr(lo, defs);
    if same_match_expr(&actual_lo, &expected_lo) {
        Some(canonical)
    } else {
        None
    }
}

fn hi_expr_matches_lo_backed_base(hi: &Expr, base: &Expr, defs: &HashMap<String, Expr>) -> bool {
    resolve_typed_lane_source(hi, defs)
        .and_then(|source| (source.lane == PointerLane::Hi32).then_some(source.value))
        .map(|value| typed_wide_value_base(&value) == *base)
        .or_else(|| {
            syntactic_explicit_hi_lane_value(hi)
                .and_then(|value| match_base_relative_wide_ptr_expr(&value).map(|(base, _)| base))
                .map(|hi_base| hi_base == *base)
        })
        .unwrap_or(false)
}

fn resolve_scalar_hi_lane_value(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    resolve_scalar_hi_lane_value_inner(expr, defs, &mut BTreeSet::new())
}

fn resolve_scalar_hi_lane_value_inner(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    seen: &mut BTreeSet<String>,
) -> Option<Expr> {
    let expr = strip_loop_phi_expr(expr);
    if let Expr::LaneExtract {
        value,
        lane: PointerLane::Hi32,
    } = expr
    {
        return Some((**value).clone());
    }
    if let Some(value) = syntactic_explicit_hi_lane_value(expr) {
        return Some(value);
    }
    match expr {
        Expr::Reg(name) => {
            if !seen.insert(name.clone()) {
                return None;
            }
            defs.get(name)
                .and_then(|value| resolve_scalar_hi_lane_value_inner(value, defs, seen))
        }
        Expr::Raw(text) if is_symbolic_name(text) => {
            if !seen.insert(text.clone()) {
                return None;
            }
            defs.get(text)
                .and_then(|value| resolve_scalar_hi_lane_value_inner(value, defs, seen))
        }
        _ => None,
    }
}

fn resolve_scalar_lo_lane_value(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    resolve_scalar_lo_lane_value_inner(expr, defs, &mut BTreeSet::new())
}

fn resolve_scalar_lo_lane_value_inner(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    seen: &mut BTreeSet<String>,
) -> Option<Expr> {
    let expr = strip_loop_phi_expr(expr);
    if let Expr::LaneExtract {
        value,
        lane: PointerLane::Lo32,
    } = expr
    {
        return Some((**value).clone());
    }
    if let Some(value) = syntactic_explicit_lo_lane_value(expr) {
        return Some(value);
    }
    match expr {
        Expr::Reg(name) => {
            if !seen.insert(name.clone()) {
                return None;
            }
            defs.get(name)
                .and_then(|value| resolve_scalar_lo_lane_value_inner(value, defs, seen))
        }
        Expr::Raw(text) if is_symbolic_name(text) => {
            if !seen.insert(text.clone()) {
                return None;
            }
            defs.get(text)
                .and_then(|value| resolve_scalar_lo_lane_value_inner(value, defs, seen))
        }
        _ => None,
    }
}

fn resolve_scalar_lane_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Expr {
    resolve_scalar_lane_expr_inner(expr, defs, &mut BTreeSet::new()).unwrap_or_else(|| expr.clone())
}

fn resolve_scalar_lane_expr_inner(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    seen: &mut BTreeSet<String>,
) -> Option<Expr> {
    let expr = strip_loop_phi_expr(expr);
    match expr {
        Expr::Reg(name) => {
            if !seen.insert(name.clone()) {
                return None;
            }
            defs.get(name)
                .and_then(|value| resolve_scalar_lane_expr_inner(value, defs, seen))
                .or_else(|| Some(expr.clone()))
        }
        Expr::Raw(text) if is_symbolic_name(text) => {
            if !seen.insert(text.clone()) {
                return None;
            }
            defs.get(text)
                .and_then(|value| resolve_scalar_lane_expr_inner(value, defs, seen))
                .or_else(|| Some(expr.clone()))
        }
        _ => Some(expr.clone()),
    }
}

fn canonicalize_scalar_wide_value(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    match_widened_scalar_sum_expr(expr, defs)
}

fn match_widened_scalar_sum_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let expr = strip_wide_casts(expr);
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => {
            if op == "+" {
                if let Some(expr) = match_pack_plus_scalar_widen_expr(lhs, rhs, defs) {
                    return Some(expr);
                }
                if let Some(expr) = match_pack_plus_scalar_widen_expr(rhs, lhs, defs) {
                    return Some(expr);
                }
            }
            let lhs = canonical_scalar_wide_term(lhs, defs)?;
            let rhs = canonical_scalar_wide_term(rhs, defs)?;
            Some(Expr::Binary {
                op: op.clone(),
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        _ => canonical_scalar_wide_term(expr, defs),
    }
}

fn canonical_scalar_wide_term(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    if let Some(inner) = match_signed_widen_term(expr) {
        return Some(widen_i32_expr_ast(inner));
    }
    if let Some(inner) = match_unsigned_widen_term(expr) {
        return Some(widen_u32_expr_ast(inner));
    }
    let resolved = resolve_scalar_lane_expr(expr, defs);
    if resolved != *expr {
        if let Some(inner) = match_signed_widen_term(&resolved) {
            return Some(widen_i32_expr_ast(inner));
        }
        if let Some(inner) = match_unsigned_widen_term(&resolved) {
            return Some(widen_u32_expr_ast(inner));
        }
    }
    let (lo, hi) = match_u64_pack_expr(expr)?;
    if expr_is_zero_or_resolves(&hi, defs) {
        return Some(widen_u32_expr_ast(lo));
    }
    None
}

fn match_pack_plus_scalar_widen_expr(
    pack_side: &Expr,
    scalar_side: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let (pack_lo, pack_hi) = match_u64_pack_expr(pack_side)?;
    let scalar_lo = strip_lane_low_casts(&resolve_scalar_lane_expr(scalar_side, defs));
    if expr_matches_signext_hi_of(&pack_hi, &scalar_lo, defs) {
        return Some(Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(widen_i32_expr_ast(scalar_lo)),
            rhs: Box::new(widen_u32_expr_ast(pack_lo)),
        });
    }
    if expr_is_zero_or_resolves(&pack_hi, defs) {
        return Some(Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(widen_u32_expr_ast(strip_lane_low_casts(&pack_lo))),
            rhs: Box::new(widen_u32_expr_ast(scalar_lo)),
        });
    }
    None
}

fn low_32_expr_for_wide_value(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let canonical = canonicalize_scalar_wide_value(expr, defs).unwrap_or_else(|| expr.clone());
    low_32_expr_for_wide_value_inner(&canonical)
}

fn low_32_expr_for_wide_value_inner(expr: &Expr) -> Option<Expr> {
    let expr = strip_wide_casts(expr);
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => Some(Expr::Binary {
            op: op.clone(),
            lhs: Box::new(low_32_expr_for_wide_value_inner(lhs)?),
            rhs: Box::new(low_32_expr_for_wide_value_inner(rhs)?),
        }),
        _ => low_32_expr_for_wide_term(expr),
    }
}

fn low_32_expr_for_wide_term(expr: &Expr) -> Option<Expr> {
    if let Some(inner) = match_signed_widen_term(expr) {
        return Some(inner);
    }
    if let Some(inner) = match_unsigned_widen_term(expr) {
        return Some(inner);
    }
    match_u64_pack_expr(expr).map(|(lo, _)| strip_lane_low_casts(&lo))
}

fn match_signed_widen_term(expr: &Expr) -> Option<Expr> {
    let Expr::Cast { ty, expr } = strip_wide_casts(expr) else {
        return None;
    };
    if ty != "int32_t" {
        return None;
    }
    Some((**expr).clone())
}

fn match_unsigned_widen_term(expr: &Expr) -> Option<Expr> {
    let Expr::Cast { ty, expr } = strip_wide_casts(expr) else {
        return None;
    };
    if ty != "uint32_t" {
        return None;
    }
    Some((**expr).clone())
}

fn widen_i32_expr_ast(expr: Expr) -> Expr {
    Expr::Cast {
        ty: "int64_t".to_string(),
        expr: Box::new(Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(strip_lane_low_casts(&expr)),
        }),
    }
}

fn widen_u32_expr_ast(expr: Expr) -> Expr {
    Expr::Cast {
        ty: "uint64_t".to_string(),
        expr: Box::new(Expr::Cast {
            ty: "uint32_t".to_string(),
            expr: Box::new(strip_lane_low_casts(&expr)),
        }),
    }
}

fn strip_lane_low_casts(expr: &Expr) -> Expr {
    let mut current = expr;
    while let Expr::Cast { ty, expr } = current {
        if matches!(
            ty.as_str(),
            "uint32_t" | "int32_t" | "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t"
        ) {
            current = expr;
        } else {
            break;
        }
    }
    current.clone()
}

fn expr_matches_signext_hi_of(expr: &Expr, low: &Expr, defs: &HashMap<String, Expr>) -> bool {
    let resolved = resolve_scalar_lane_expr(expr, defs);
    let expected = Expr::Binary {
        op: ">>".to_string(),
        lhs: Box::new(Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(strip_lane_low_casts(low)),
        }),
        rhs: Box::new(Expr::Imm("31".to_string())),
    };
    same_match_expr(&resolved, &expected)
}

fn expr_is_zero_or_resolves(expr: &Expr, defs: &HashMap<String, Expr>) -> bool {
    let resolved = resolve_scalar_lane_expr(expr, defs);
    expr_is_zero(expr) || expr_is_zero(&resolved)
}

fn resolve_typed_lane_source(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<TypedLaneSource> {
    resolve_typed_lane_source_inner(expr, defs, &mut BTreeSet::new())
}

fn resolve_typed_lane_source_inner(
    expr: &Expr,
    defs: &HashMap<String, Expr>,
    seen: &mut BTreeSet<String>,
) -> Option<TypedLaneSource> {
    let expr = strip_loop_phi_expr(expr);
    if let Some((value, lane)) = explicit_wide_lane_extract(expr) {
        return Some(TypedLaneSource { value, lane });
    }
    match expr {
        Expr::LaneExtract { value, lane } => Some(TypedLaneSource {
            value: (**value).clone(),
            lane: *lane,
        }),
        Expr::PtrLane { base, lane } => Some(TypedLaneSource {
            value: Expr::Raw(base.clone()),
            lane: *lane,
        }),
        Expr::Reg(name) => {
            if !seen.insert(name.clone()) {
                return None;
            }
            defs.get(name)
                .and_then(|expr| resolve_typed_lane_source_inner(expr, defs, seen))
        }
        Expr::Raw(text) if is_symbolic_name(text) => {
            if let Some((base, lane)) = PointerLane::parse_named(text) {
                return Some(TypedLaneSource {
                    value: Expr::Raw(base),
                    lane,
                });
            }
            if !seen.insert(text.clone()) {
                return None;
            }
            defs.get(text)
                .and_then(|expr| resolve_typed_lane_source_inner(expr, defs, seen))
        }
        Expr::ConstMemSymbol(text) | Expr::Builtin(text) | Expr::Raw(text) => {
            PointerLane::parse_named(text).map(|(base, lane)| TypedLaneSource {
                value: Expr::Raw(base),
                lane,
            })
        }
        _ => None,
    }
}

fn common_typed_wide_base(lo_value: &Expr, hi_value: &Expr) -> Option<Expr> {
    let lo_base = typed_wide_value_base(lo_value);
    let hi_base = typed_wide_value_base(hi_value);
    (lo_base == hi_base).then_some(lo_base)
}

fn typed_wide_value_base(value: &Expr) -> Expr {
    match value {
        Expr::WidePtr { base, .. } => (**base).clone(),
        other => other.clone(),
    }
}

fn typed_wide_value_offset(value: &Expr, base: &Expr) -> Option<Expr> {
    match value {
        Expr::WidePtr {
            base: inner_base,
            offset,
        } if inner_base.as_ref() == base => Some((**offset).clone()),
        other if other == base => Some(Expr::Imm("0".to_string())),
        _ => None,
    }
}

fn seed_loop_carried_typed_lane_defs(
    body: &Stmt,
    seed_context: &HashMap<String, Expr>,
    defs: &mut HashMap<String, Expr>,
) {
    let mut assigns = HashMap::<String, Expr>::new();
    collect_direct_var_assign_exprs(body, &mut assigns);
    let mut phi_next_exprs = HashMap::<String, Expr>::new();
    collect_loop_phi_assign_exprs(body, &mut phi_next_exprs);
    let mut seed_defs = seed_context.clone();
    for _ in 0..32 {
        let mut changed = false;
        for (name, rhs) in &assigns {
            if seed_defs.contains_key(name) {
                continue;
            }
            let lane_expr = extract_lane_def_expr_with_defs(rhs, &seed_defs);
            let Some(lane_expr) = lane_expr else {
                continue;
            };
            let mut used = BTreeSet::new();
            collect_used_expr_vars(&lane_expr, &mut used);
            if used.contains(name) {
                continue;
            }
            seed_defs.insert(name.clone(), lane_expr);
            changed = true;
        }
        if !changed {
            break;
        }
    }
    for (current_lo, next_lo_phi_expr) in &phi_next_exprs {
        let next_lo_rhs = resolve_loop_phi_next_rhs(next_lo_phi_expr, &assigns);
        let Some((base, current_offset, step)) =
            match_loop_carried_next_lo_value(current_lo, &next_lo_rhs, &seed_defs)
        else {
            continue;
        };
        let next_offset = normalize_offset_expr(combine_offset_expr(
            current_offset.clone(),
            step.clone(),
            "+",
        ));
        let current_value = Expr::WidePtr {
            base: Box::new(base.clone()),
            offset: Box::new(current_offset.clone()),
        };
        let next_value = Expr::WidePtr {
            base: Box::new(base.clone()),
            offset: Box::new(next_offset.clone()),
        };
        defs.insert(
            current_lo.clone(),
            Expr::LaneExtract {
                value: Box::new(current_value.clone()),
                lane: PointerLane::Lo32,
            },
        );
        if let Some(next_lo) = match_var_name(next_lo_phi_expr) {
            defs.insert(
                next_lo,
                Expr::LaneExtract {
                    value: Box::new(next_value.clone()),
                    lane: PointerLane::Lo32,
                },
            );
        }

        for (current_hi, next_hi_phi_expr) in &phi_next_exprs {
            if current_hi == current_lo {
                continue;
            }
            let next_hi_rhs = resolve_loop_phi_next_rhs(next_hi_phi_expr, &assigns);
            let next_hi_matches = resolve_typed_lane_source(&next_hi_rhs, &seed_defs)
                .and_then(|source| (source.lane == PointerLane::Hi32).then_some(source.value))
                .and_then(|next_hi_value| {
                    let hi_base = typed_wide_value_base(&next_hi_value);
                    if hi_base != base {
                        return None;
                    }
                    let next_hi_offset = typed_wide_value_offset(&next_hi_value, &hi_base)?;
                    (normalize_offset_expr(next_hi_offset) == next_offset).then_some(())
                })
                .is_some()
                || match_loop_carried_next_hi_value(
                    current_hi,
                    current_lo,
                    &next_hi_rhs,
                    &assigns,
                    &step,
                );
            if !next_hi_matches {
                continue;
            }

            defs.insert(
                current_hi.clone(),
                Expr::LaneExtract {
                    value: Box::new(current_value.clone()),
                    lane: PointerLane::Hi32,
                },
            );
            if let Some(next_hi) = match_var_name(next_hi_phi_expr) {
                defs.insert(
                    next_hi,
                    Expr::LaneExtract {
                        value: Box::new(next_value.clone()),
                        lane: PointerLane::Hi32,
                    },
                );
            }
            break;
        }
    }
}

fn collect_direct_var_assign_exprs(stmt: &Stmt, out: &mut HashMap<String, Expr>) {
    match stmt {
        Stmt::Label { body, .. } => collect_direct_var_assign_exprs(body, out),
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_direct_var_assign_exprs(stmt, out);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_direct_var_assign_exprs(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_direct_var_assign_exprs(else_branch, out);
            }
        }
        Stmt::Loop { body, .. } => collect_direct_var_assign_exprs(body, out),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_direct_var_assign_exprs(body, out);
            }
            if let Some(default) = default {
                collect_direct_var_assign_exprs(default, out);
            }
        }
        _ => {
            if let Some((name, rhs)) = direct_var_assign(stmt) {
                out.insert(name, strip_loop_phi_expr(rhs).clone());
            }
        }
    }
}

fn collect_loop_phi_assign_exprs(stmt: &Stmt, out: &mut HashMap<String, Expr>) {
    match stmt {
        Stmt::Label { body, .. } => collect_loop_phi_assign_exprs(body, out),
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_loop_phi_assign_exprs(stmt, out);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_loop_phi_assign_exprs(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_loop_phi_assign_exprs(else_branch, out);
            }
        }
        Stmt::Loop { body, .. } => collect_loop_phi_assign_exprs(body, out),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_loop_phi_assign_exprs(body, out);
            }
            if let Some(default) = default {
                collect_loop_phi_assign_exprs(default, out);
            }
        }
        _ => {
            let Some((name, rhs)) = direct_var_assign(stmt) else {
                return;
            };
            let Some(next_expr) = loop_phi_arg(rhs) else {
                return;
            };
            out.insert(name, strip_loop_phi_expr(next_expr).clone());
        }
    }
}

fn match_loop_carried_next_lo_step(expr: &Expr, current_lo: &str) -> Option<Expr> {
    if let Expr::LaneExtract {
        value,
        lane: PointerLane::Lo32,
    } = expr
    {
        return match_loop_carried_next_lo_step(value, current_lo);
    }
    let Expr::Binary { op, lhs, rhs } = expr else {
        return match_loop_carried_packed_lo_step(expr, current_lo);
    };
    if op != "+" {
        return match_loop_carried_packed_lo_step(expr, current_lo);
    }
    if match_var_name(lhs).as_deref() == Some(current_lo) {
        return Some((**rhs).clone());
    }
    if match_var_name(rhs).as_deref() == Some(current_lo) {
        return Some((**lhs).clone());
    }
    if match_loop_carried_lo_pack_base(lhs, current_lo) {
        return Some(strip_offset_like_casts(rhs));
    }
    if match_loop_carried_lo_pack_base(rhs, current_lo) {
        return Some(strip_offset_like_casts(lhs));
    }
    let mut lhs_used = BTreeSet::new();
    collect_used_expr_vars(lhs, &mut lhs_used);
    let mut rhs_used = BTreeSet::new();
    collect_used_expr_vars(rhs, &mut rhs_used);
    if lhs_used.contains(current_lo) && !rhs_used.contains(current_lo) {
        return Some(strip_offset_like_casts(rhs));
    }
    if rhs_used.contains(current_lo) && !lhs_used.contains(current_lo) {
        return Some(strip_offset_like_casts(lhs));
    }
    match_loop_carried_packed_lo_step(expr, current_lo)
}

fn match_loop_carried_packed_lo_step(expr: &Expr, current_lo: &str) -> Option<Expr> {
    let Expr::Cast { ty, expr } = expr else {
        return None;
    };
    if ty != "uint32_t" {
        return None;
    }
    let Expr::Binary { op, lhs, rhs } = strip_wide_casts(expr) else {
        return None;
    };
    if op != "+" {
        return None;
    }
    if match_loop_carried_lo_pack_base(lhs, current_lo) {
        return Some(strip_offset_like_casts(rhs));
    }
    if match_loop_carried_lo_pack_base(rhs, current_lo) {
        return Some(strip_offset_like_casts(lhs));
    }
    let mut lhs_used = BTreeSet::new();
    collect_used_expr_vars(lhs, &mut lhs_used);
    let mut rhs_used = BTreeSet::new();
    collect_used_expr_vars(rhs, &mut rhs_used);
    if lhs_used.contains(current_lo) && !rhs_used.contains(current_lo) {
        return Some(strip_offset_like_casts(rhs));
    }
    if rhs_used.contains(current_lo) && !lhs_used.contains(current_lo) {
        return Some(strip_offset_like_casts(lhs));
    }
    None
}

fn match_loop_carried_lo_pack_base(expr: &Expr, current_lo: &str) -> bool {
    match strip_wide_casts(expr) {
        Expr::Addr64 { lo, .. } => match_var_name(lo).as_deref() == Some(current_lo),
        other => match_u64_pack_expr(other)
            .is_some_and(|(lo, _hi)| match_var_name(&lo).as_deref() == Some(current_lo)),
    }
}
fn resolve_loop_phi_next_rhs(expr: &Expr, assigns: &HashMap<String, Expr>) -> Expr {
    let mut current = strip_loop_phi_expr(expr);
    let mut seen = BTreeSet::new();
    loop {
        let Some(name) = match_var_name(current) else {
            return current.clone();
        };
        if !seen.insert(name.clone()) {
            return current.clone();
        }
        let Some(next) = assigns.get(&name) else {
            return current.clone();
        };
        current = strip_loop_phi_expr(next);
    }
}

fn match_loop_carried_next_lo_value(
    current_lo: &str,
    next_lo_rhs: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr, Expr)> {
    if let Some(step) = match_loop_carried_next_lo_step(next_lo_rhs, current_lo) {
        let current_lo_source =
            resolve_typed_lane_source(&Expr::Reg(current_lo.to_string()), defs)?;
        if current_lo_source.lane != PointerLane::Lo32 {
            return None;
        }
        let base = typed_wide_value_base(&current_lo_source.value);
        return Some((
            base.clone(),
            typed_lane_delta_expr(Expr::Reg(current_lo.to_string()), base),
            step,
        ));
    }

    if let Some(current_lo_source) =
        resolve_typed_lane_source(&Expr::Reg(current_lo.to_string()), defs)
    {
        if current_lo_source.lane == PointerLane::Lo32 {
            let base = typed_wide_value_base(&current_lo_source.value);
            if let Some(next_value) = resolve_typed_lane_source(next_lo_rhs, defs)
                .and_then(|source| (source.lane == PointerLane::Lo32).then_some(source.value))
            {
                let next_base = typed_wide_value_base(&next_value);
                if next_base == base {
                    let current_offset = typed_wide_value_offset(&current_lo_source.value, &base)?;
                    let next_offset = typed_wide_value_offset(&next_value, &next_base)?;
                    let step = normalize_offset_expr(combine_offset_expr(
                        next_offset,
                        current_offset.clone(),
                        "-",
                    ));
                    return Some((
                        base.clone(),
                        typed_lane_delta_expr(Expr::Reg(current_lo.to_string()), base),
                        step,
                    ));
                }
            }
        }
    }

    let TypedLaneSource {
        value: next_lo_value,
        lane: PointerLane::Lo32,
    } = resolve_typed_lane_source(next_lo_rhs, defs)?
    else {
        return None;
    };
    let base = typed_wide_value_base(&next_lo_value);
    let next_offset = typed_wide_value_offset(&next_lo_value, &base)?;
    let (current_offset, step) =
        split_loop_carried_lo_offset(&next_offset, current_lo, &base, defs)?;
    Some((base, current_offset, step))
}

fn split_loop_carried_lo_offset(
    next_offset: &Expr,
    current_lo: &str,
    base: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr)> {
    let mut terms = Vec::<SignedOffsetTerm>::new();
    collect_signed_offset_terms(next_offset, true, &mut terms);
    let mut current_terms = Vec::new();
    let mut step_terms = Vec::new();
    let mut saw_current_lo = false;
    let mut saw_base_lo = false;

    for term in terms {
        if term.positive && match_var_name(&term.expr).as_deref() == Some(current_lo) {
            saw_current_lo = true;
            current_terms.push(term);
            continue;
        }
        if !term.positive && expr_matches_base_lane(&term.expr, base, PointerLane::Lo32, defs) {
            saw_base_lo = true;
            current_terms.push(term);
            continue;
        }
        step_terms.push(term);
    }

    (saw_current_lo && saw_base_lo).then(|| {
        (
            rebuild_signed_offset_terms(current_terms),
            rebuild_signed_offset_terms(step_terms),
        )
    })
}

fn match_loop_carried_next_hi_value(
    current_hi: &str,
    current_lo: &str,
    next_hi_rhs: &Expr,
    assigns: &HashMap<String, Expr>,
    expected_step: &Expr,
) -> bool {
    if match_loop_carried_packed_hi_expr(next_hi_rhs, current_hi, current_lo, expected_step) {
        return true;
    }
    let Expr::Binary { op, lhs, rhs } = next_hi_rhs else {
        return false;
    };
    if op != "+" {
        return false;
    }
    for (hi_expr, carry_expr) in [(lhs.as_ref(), rhs.as_ref()), (rhs.as_ref(), lhs.as_ref())] {
        if match_var_name(hi_expr).as_deref() != Some(current_hi) {
            continue;
        }
        let Some(carry) = resolve_loop_carried_carry_increment(carry_expr, assigns) else {
            continue;
        };
        if carry.ptr_lo != current_lo {
            continue;
        }
        if normalize_offset_expr(carry.offset) == normalize_offset_expr(expected_step.clone()) {
            return true;
        }
    }
    false
}

fn match_loop_carried_packed_hi_expr(
    expr: &Expr,
    current_hi: &str,
    current_lo: &str,
    expected_step: &Expr,
) -> bool {
    let Some(value_expr) = syntactic_explicit_hi_lane_value(expr) else {
        return false;
    };
    match_loop_carried_packed_wide_expr(&value_expr, current_hi, current_lo, expected_step)
}

fn match_loop_carried_packed_wide_expr(
    expr: &Expr,
    current_hi: &str,
    current_lo: &str,
    expected_step: &Expr,
) -> bool {
    match strip_wide_casts(expr) {
        Expr::WidePtr { base, offset } => {
            match_loop_carried_addr64_base(base, current_hi, current_lo)
                && normalize_offset_expr((**offset).clone())
                    == normalize_offset_expr(expected_step.clone())
        }
        Expr::Binary { op, lhs, rhs } if op == "+" => {
            if match_loop_carried_addr64_base(lhs, current_hi, current_lo) {
                return normalize_offset_expr(strip_offset_like_casts(rhs))
                    == normalize_offset_expr(expected_step.clone());
            }
            if match_loop_carried_addr64_base(rhs, current_hi, current_lo) {
                return normalize_offset_expr(strip_offset_like_casts(lhs))
                    == normalize_offset_expr(expected_step.clone());
            }
            false
        }
        other => {
            match_loop_carried_addr64_base(other, current_hi, current_lo)
                && normalize_offset_expr(Expr::Imm("0".to_string()))
                    == normalize_offset_expr(expected_step.clone())
        }
    }
}

fn match_loop_carried_addr64_base(expr: &Expr, current_hi: &str, current_lo: &str) -> bool {
    match strip_wide_casts(expr) {
        Expr::Addr64 { lo, hi } => {
            match_var_name(lo).as_deref() == Some(current_lo)
                && match_var_name(hi).as_deref() == Some(current_hi)
        }
        other => match_u64_pack_expr(other).is_some_and(|(lo, hi)| {
            match_var_name(&lo).as_deref() == Some(current_lo)
                && match_var_name(&hi).as_deref() == Some(current_hi)
        }),
    }
}

fn resolve_loop_carried_carry_increment(
    expr: &Expr,
    assigns: &HashMap<String, Expr>,
) -> Option<CarryDef> {
    match match_carry_increment(expr)? {
        CarryIncrement::Inline(carry) => Some(carry),
        CarryIncrement::Var(name) => assigns.get(&name).and_then(match_inline_carry_expr),
    }
}

fn symbolic_typed_wide_offset(
    lo_expr: &Expr,
    hi_expr: &Expr,
    lo_value: &Expr,
    base: &Expr,
) -> Expr {
    let lo_expr = strip_loop_phi_expr(lo_expr);
    let hi_expr = strip_loop_phi_expr(hi_expr);
    if let Some(offset) = typed_wide_value_offset(lo_value, base) {
        // If a loop-carried pointer lane was modeled only as
        // `lo_lane - base.lo32`, keep the high lane in the reconstructed
        // address. The low-lane delta is compact, but it is only correct when
        // the pointer never crosses a 32-bit carry boundary.
        if offset_is_symbolic_lo_lane_delta(&offset, lo_expr, base) {
            return typed_full_width_delta_expr(lo_expr.clone(), hi_expr.clone(), base.clone());
        }
        return offset;
    }
    match lo_expr {
        Expr::LaneExtract { .. } | Expr::PtrLane { .. } => {
            typed_lane_delta_expr(lo_expr.clone(), base.clone())
        }
        _ => typed_full_width_delta_expr(lo_expr.clone(), hi_expr.clone(), base.clone()),
    }
}

fn offset_is_symbolic_lo_lane_delta(offset: &Expr, lo_expr: &Expr, base: &Expr) -> bool {
    same_match_expr(
        &normalize_offset_expr(offset.clone()),
        &normalize_offset_expr(typed_lane_delta_expr(lo_expr.clone(), base.clone())),
    )
}

fn typed_lane_delta_expr(lo_expr: Expr, base: Expr) -> Expr {
    Expr::Binary {
        op: "-".to_string(),
        lhs: Box::new(lo_expr),
        rhs: Box::new(Expr::LaneExtract {
            value: Box::new(base),
            lane: PointerLane::Lo32,
        }),
    }
}

fn typed_full_width_delta_expr(lo_expr: Expr, hi_expr: Expr, base: Expr) -> Expr {
    Expr::Binary {
        op: "-".to_string(),
        lhs: Box::new(build_u64_pack_expr(lo_expr, hi_expr)),
        rhs: Box::new(Expr::Cast {
            ty: "uintptr_t".to_string(),
            expr: Box::new(base),
        }),
    }
}

#[derive(Clone)]
struct SignedOffsetTerm {
    positive: bool,
    expr: Expr,
}

fn simplify_typed_wide_ptr_expr(base: Expr, offset: Expr, defs: &HashMap<String, Expr>) -> Expr {
    let (base, mut offset) =
        if let Some((inner_base, inner_offset)) = match_typed_wide_value_parts(&base, defs) {
            (inner_base, combine_offset_expr(inner_offset, offset, "+"))
        } else {
            (base, offset)
        };
    while let Some(next) = fold_base_relative_wide_offset(&base, &offset, defs) {
        if next == offset {
            break;
        }
        offset = next;
    }
    simplify_wide_ptr_expr(base, offset)
}

fn fold_base_relative_wide_offset(
    base: &Expr,
    offset: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<Expr> {
    let mut terms = Vec::<SignedOffsetTerm>::new();
    collect_signed_offset_terms(offset, true, &mut terms);
    let mut changed = false;

    loop {
        if let Some(next_terms) = rewrite_same_base_lo_lane_difference(base, &terms, defs) {
            terms = next_terms;
            changed = true;
            continue;
        }

        let Some(base_lo_idx) = terms.iter().position(|term| {
            !term.positive && expr_matches_base_lane(&term.expr, base, PointerLane::Lo32, defs)
        }) else {
            break;
        };

        let Some((lane_idx, wide_offset)) = terms.iter().enumerate().find_map(|(idx, term)| {
            if !term.positive {
                return None;
            }
            let source = resolve_typed_lane_source(&term.expr, defs)?;
            if source.lane != PointerLane::Lo32 {
                return None;
            }
            if let Some(term_name) = match_var_name(&term.expr) {
                let mut used = BTreeSet::new();
                collect_used_expr_vars(&source.value, &mut used);
                if used.contains(&term_name) {
                    return None;
                }
            }
            let value_base = typed_wide_value_base(&source.value);
            if value_base != *base {
                return None;
            }
            Some((
                idx,
                typed_wide_value_offset(&source.value, &value_base)
                    .unwrap_or_else(|| Expr::Imm("0".to_string())),
            ))
        }) else {
            break;
        };

        let mut next_terms = Vec::with_capacity(terms.len());
        for (idx, term) in terms.into_iter().enumerate() {
            if idx == base_lo_idx || idx == lane_idx {
                continue;
            }
            next_terms.push(term);
        }
        collect_signed_offset_terms(&wide_offset, true, &mut next_terms);
        terms = next_terms;
        changed = true;
    }

    changed.then(|| rebuild_signed_offset_terms(terms))
}

fn rewrite_same_base_lo_lane_difference(
    base: &Expr,
    terms: &[SignedOffsetTerm],
    defs: &HashMap<String, Expr>,
) -> Option<Vec<SignedOffsetTerm>> {
    let original = rebuild_signed_offset_terms(terms.to_vec());
    for (pos_idx, pos_term) in terms.iter().enumerate() {
        if !pos_term.positive {
            continue;
        }
        let Some(pos_source) = resolve_typed_lane_source(&pos_term.expr, defs) else {
            continue;
        };
        if pos_source.lane != PointerLane::Lo32 {
            continue;
        }
        let pos_base = typed_wide_value_base(&pos_source.value);
        if pos_base != *base {
            continue;
        }
        let pos_offset = typed_wide_value_offset(&pos_source.value, &pos_base)
            .unwrap_or_else(|| Expr::Imm("0".to_string()));

        for (neg_idx, neg_term) in terms.iter().enumerate() {
            if neg_term.positive {
                continue;
            }
            let Some(neg_source) = resolve_typed_lane_source(&neg_term.expr, defs) else {
                continue;
            };
            if neg_source.lane != PointerLane::Lo32 {
                continue;
            }
            let neg_base = typed_wide_value_base(&neg_source.value);
            if neg_base != *base {
                continue;
            }
            let neg_offset = typed_wide_value_offset(&neg_source.value, &neg_base)
                .unwrap_or_else(|| Expr::Imm("0".to_string()));

            let mut next_terms = Vec::with_capacity(terms.len());
            for (idx, term) in terms.iter().cloned().enumerate() {
                if idx == pos_idx || idx == neg_idx {
                    continue;
                }
                next_terms.push(term);
            }
            collect_signed_offset_terms(&pos_offset, true, &mut next_terms);
            collect_signed_offset_terms(&neg_offset, false, &mut next_terms);
            if rebuild_signed_offset_terms(next_terms.clone()) != original {
                return Some(next_terms);
            }
        }
    }
    None
}

fn collect_signed_offset_terms(expr: &Expr, positive: bool, out: &mut Vec<SignedOffsetTerm>) {
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "+" => {
            collect_signed_offset_terms(lhs, positive, out);
            collect_signed_offset_terms(rhs, positive, out);
        }
        Expr::Binary { op, lhs, rhs } if op == "-" => {
            collect_signed_offset_terms(lhs, positive, out);
            collect_signed_offset_terms(rhs, !positive, out);
        }
        Expr::Cast { ty, expr }
            if matches!(
                ty.as_str(),
                "int64_t" | "uint64_t" | "intptr_t" | "uintptr_t"
            ) && !expr_is_symbolic_name_pointer_base(expr) =>
        {
            collect_signed_offset_terms(expr, positive, out);
        }
        _ => out.push(SignedOffsetTerm {
            positive,
            expr: expr.clone(),
        }),
    }
}

fn rebuild_signed_offset_terms(terms: Vec<SignedOffsetTerm>) -> Expr {
    let mut iter = terms.into_iter().filter(|term| !expr_is_zero(&term.expr));
    let Some(first) = iter.next() else {
        return Expr::Imm("0".to_string());
    };

    let mut expr = if first.positive {
        first.expr
    } else {
        Expr::Binary {
            op: "-".to_string(),
            lhs: Box::new(Expr::Imm("0".to_string())),
            rhs: Box::new(first.expr),
        }
    };

    for term in iter {
        expr = combine_offset_expr(expr, term.expr, if term.positive { "+" } else { "-" });
    }
    expr
}

fn normalize_offset_expr(expr: Expr) -> Expr {
    let mut terms = Vec::new();
    collect_signed_offset_terms(&expr, true, &mut terms);
    let mut normalized = Vec::<SignedOffsetTerm>::new();
    let mut const_sum = 0i64;

    for term in terms {
        if let Some(value) = expr_i64(&term.expr) {
            const_sum += if term.positive { value } else { -value };
            continue;
        }
        if let Some(idx) = normalized
            .iter()
            .position(|existing| existing.positive != term.positive && existing.expr == term.expr)
        {
            normalized.remove(idx);
            continue;
        }
        normalized.push(term);
    }

    if const_sum != 0 {
        normalized.push(SignedOffsetTerm {
            positive: const_sum > 0,
            expr: Expr::Imm(const_sum.abs().to_string()),
        });
    }

    rebuild_signed_offset_terms(normalized)
}

fn expr_matches_base_lane(
    expr: &Expr,
    base: &Expr,
    lane: PointerLane,
    defs: &HashMap<String, Expr>,
) -> bool {
    resolve_typed_lane_source(expr, defs)
        .is_some_and(|source| source.lane == lane && source.value == *base)
}

fn simplify_wide_ptr_expr(base: Expr, offset: Expr) -> Expr {
    let offset = normalize_offset_expr(offset);
    if expr_is_zero(&offset) {
        return base;
    }
    match base {
        Expr::WidePtr {
            base: inner_base,
            offset: inner_offset,
        } => Expr::WidePtr {
            base: inner_base,
            offset: Box::new(combine_offset_expr(*inner_offset, offset, "+")),
        },
        other => Expr::WidePtr {
            base: Box::new(other),
            offset: Box::new(offset),
        },
    }
}

fn combine_offset_expr(lhs: Expr, rhs: Expr, op: &str) -> Expr {
    if expr_is_zero(&lhs) && op == "+" {
        return rhs;
    }
    if expr_is_zero(&rhs) {
        return lhs;
    }
    if let (Some(lhs_imm), Some(rhs_imm)) = (expr_i64(&lhs), expr_i64(&rhs)) {
        let value = if op == "+" {
            lhs_imm + rhs_imm
        } else {
            lhs_imm - rhs_imm
        };
        return Expr::Imm(value.to_string());
    }
    Expr::Binary {
        op: op.to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn expr_i64(expr: &Expr) -> Option<i64> {
    expr_atom_text(expr)?.parse::<i64>().ok()
}

fn remove_pure_noop_exprs_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Block(stmts) => {
            Stmt::Block(stmts.into_iter().map(remove_pure_noop_exprs_stmt).collect())
        }
        Stmt::Sequence(stmts) => {
            Stmt::Sequence(stmts.into_iter().map(remove_pure_noop_exprs_stmt).collect())
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(remove_pure_noop_exprs_stmt(*then_branch)),
            else_branch: else_branch.map(|stmt| Box::new(remove_pure_noop_exprs_stmt(*stmt))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(remove_pure_noop_exprs_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, remove_pure_noop_exprs_stmt(body)))
                .collect(),
            default: default.map(|body| Box::new(remove_pure_noop_exprs_stmt(*body))),
        },
        Stmt::ExprStmt(expr) if expr_is_pure_for_dce(&expr) => Stmt::Empty,
        other => other,
    }
}

fn normalize_raw_addr64_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(normalize_raw_addr64_stmt(*body)),
        },
        Stmt::Block(stmts) => {
            Stmt::Block(stmts.into_iter().map(normalize_raw_addr64_stmt).collect())
        }
        Stmt::Sequence(stmts) => {
            Stmt::Sequence(stmts.into_iter().map(normalize_raw_addr64_stmt).collect())
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: normalize_raw_addr64_expr(condition),
            then_branch: Box::new(normalize_raw_addr64_stmt(*then_branch)),
            else_branch: else_branch.map(|stmt| Box::new(normalize_raw_addr64_stmt(*stmt))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(normalize_raw_addr64_expr),
            body: Box::new(normalize_raw_addr64_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(normalize_raw_addr64_expr),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, normalize_raw_addr64_stmt(body)))
                .collect(),
            default: default.map(|body| Box::new(normalize_raw_addr64_stmt(*body))),
        },
        Stmt::Return(expr) => Stmt::Return(expr.map(normalize_raw_addr64_expr)),
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: normalize_raw_addr64_lvalue(dst),
            src: normalize_raw_addr64_expr(src),
        },
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(normalize_raw_addr64_expr(expr)),
        other => other,
    }
}

fn normalize_raw_addr64_lvalue(lvalue: LValue) -> LValue {
    match lvalue {
        LValue::Raw(text) => LValue::Raw(rewrite_raw_addr64_calls(&text)),
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(normalize_raw_addr64_expr(*addr)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(normalize_raw_addr64_expr(*base)),
            index: Box::new(normalize_raw_addr64_expr(*index)),
        },
        other => other,
    }
}

fn normalize_raw_addr64_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Raw(text) => Expr::Raw(rewrite_raw_addr64_calls(&text)),
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(normalize_raw_addr64_expr(*arg)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(normalize_raw_addr64_expr(*lhs)),
            rhs: Box::new(normalize_raw_addr64_expr(*rhs)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(normalize_raw_addr64_expr(*cond)),
            then_expr: Box::new(normalize_raw_addr64_expr(*then_expr)),
            else_expr: Box::new(normalize_raw_addr64_expr(*else_expr)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args.into_iter().map(normalize_raw_addr64_expr).collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args.into_iter().map(normalize_raw_addr64_expr).collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(normalize_raw_addr64_expr(*addr)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(normalize_raw_addr64_expr(*lo)),
            hi: Box::new(normalize_raw_addr64_expr(*hi)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(normalize_raw_addr64_expr(*expr)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(normalize_raw_addr64_expr(*base)),
            index: Box::new(normalize_raw_addr64_expr(*index)),
        },
        other => other,
    }
}

fn named_ptr_pair_fold_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(named_ptr_pair_fold_stmt(*body)),
        },
        Stmt::Block(stmts) => {
            Stmt::Block(stmts.into_iter().map(named_ptr_pair_fold_stmt).collect())
        }
        Stmt::Sequence(stmts) => {
            Stmt::Sequence(stmts.into_iter().map(named_ptr_pair_fold_stmt).collect())
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: named_ptr_pair_fold_expr(condition),
            then_branch: Box::new(named_ptr_pair_fold_stmt(*then_branch)),
            else_branch: else_branch.map(|stmt| Box::new(named_ptr_pair_fold_stmt(*stmt))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(named_ptr_pair_fold_expr),
            body: Box::new(named_ptr_pair_fold_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(named_ptr_pair_fold_expr),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, named_ptr_pair_fold_stmt(body)))
                .collect(),
            default: default.map(|body| Box::new(named_ptr_pair_fold_stmt(*body))),
        },
        Stmt::Return(expr) => Stmt::Return(expr.map(named_ptr_pair_fold_expr)),
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst: named_ptr_pair_fold_lvalue(dst),
            src: named_ptr_pair_fold_expr(src),
        },
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(named_ptr_pair_fold_expr(expr)),
        other => other,
    }
}

fn named_ptr_pair_fold_lvalue(lvalue: LValue) -> LValue {
    match lvalue {
        LValue::Deref { ty, addr } => LValue::Deref {
            ty,
            addr: Box::new(named_ptr_pair_fold_expr(*addr)),
        },
        LValue::Indexed { base, index } => LValue::Indexed {
            base: Box::new(named_ptr_pair_fold_expr(*base)),
            index: Box::new(named_ptr_pair_fold_expr(*index)),
        },
        other => other,
    }
}

fn named_ptr_pair_fold_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(named_ptr_pair_fold_expr(*arg)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(named_ptr_pair_fold_expr(*lhs)),
            rhs: Box::new(named_ptr_pair_fold_expr(*rhs)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(named_ptr_pair_fold_expr(*cond)),
            then_expr: Box::new(named_ptr_pair_fold_expr(*then_expr)),
            else_expr: Box::new(named_ptr_pair_fold_expr(*else_expr)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args.into_iter().map(named_ptr_pair_fold_expr).collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args.into_iter().map(named_ptr_pair_fold_expr).collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(named_ptr_pair_fold_expr(*addr)),
        },
        Expr::Addr64 { lo, hi } => {
            let lo = named_ptr_pair_fold_expr(*lo);
            let hi = named_ptr_pair_fold_expr(*hi);
            collapse_named_pointer_pair(&lo, &hi).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(named_ptr_pair_fold_expr(*expr)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(named_ptr_pair_fold_expr(*base)),
            index: Box::new(named_ptr_pair_fold_expr(*index)),
        },
        other => other,
    }
}

fn collapse_named_pointer_pair(lo: &Expr, hi: &Expr) -> Option<Expr> {
    if let (
        Expr::PtrLane {
            base: lo_base,
            lane: PointerLane::Lo32,
        },
        Expr::PtrLane {
            base: hi_base,
            lane: PointerLane::Hi32,
        },
    ) = (lo, hi)
    {
        if lo_base == hi_base {
            return Some(Expr::Raw(lo_base.clone()));
        }
    }

    let lo_name = match_var_name(lo)?;
    let hi_name = match_var_name(hi)?;
    let base = paired_pointer_base(&lo_name, &hi_name)?;
    let base_expr = Expr::Raw(base.clone());
    let canonical_lo = pointer_lane_name(&base, PointerLane::Lo32);
    if lo_name == canonical_lo {
        Some(base_expr)
    } else {
        Some(render_folded_addr64_base(
            base_expr,
            &Expr::Binary {
                op: "-".to_string(),
                lhs: Box::new(Expr::Raw(lo_name)),
                rhs: Box::new(Expr::Raw(canonical_lo)),
            },
        ))
    }
}

fn rewrite_raw_addr64_calls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with("addr64(") {
            if let Some((end, lo_arg, hi_arg)) = parse_raw_addr64_call(text, i) {
                out.push_str(&format!(
                    "((uintptr_t)(((uint64_t)({}) << 32) | (uint32_t)({})))",
                    rewrite_raw_addr64_calls(hi_arg),
                    rewrite_raw_addr64_calls(lo_arg)
                ));
                i = end;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn parse_raw_addr64_call(text: &str, start: usize) -> Option<(usize, &str, &str)> {
    let bytes = text.as_bytes();
    let open = start.checked_add("addr64".len())?;
    if *bytes.get(open)? != b'(' {
        return None;
    }
    let mut depth = 1usize;
    let mut comma = None;
    let mut i = open + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let comma = comma?;
                    let lo = text[open + 1..comma].trim();
                    let hi = text[comma + 1..i].trim();
                    return Some((i + 1, lo, hi));
                }
            }
            b',' if depth == 1 && comma.is_none() => comma = Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

fn rewrite_raw_seeded_addr64_calls(text: &str, seeded_wide_addrs: &SeededWideAddrMaps) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with("addr64(") {
            if let Some((end, lo_arg, hi_arg)) = parse_raw_addr64_call(text, i) {
                let lo = rewrite_raw_seeded_addr64_calls(lo_arg, seeded_wide_addrs);
                let hi = rewrite_raw_seeded_addr64_calls(hi_arg, seeded_wide_addrs);
                if let Some(expr) = fold_seeded_raw_addr64_use(&lo, &hi, seeded_wide_addrs) {
                    out.push_str(&expr.render());
                } else {
                    out.push_str(&format!("addr64({}, {})", lo, hi));
                }
                i = end;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn fold_seeded_raw_addr64_use(
    lo_text: &str,
    hi_text: &str,
    seeded_wide_addrs: &SeededWideAddrMaps,
) -> Option<Expr> {
    if !is_symbolic_name(lo_text) || !is_symbolic_name(hi_text) {
        return None;
    }
    let lo_expr = named_expr_symbol(lo_text.to_string());
    let hi_expr = named_expr_symbol(hi_text.to_string());
    fold_seeded_wide_addr_use(&lo_expr, &hi_expr, seeded_wide_addrs)
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
        LValue::Var(name) => named_lvalue_symbol(rename_token(&name, token_map)),
        LValue::PtrLane { base, lane } => LValue::PtrLane { base, lane },
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
        Expr::Reg(name) => named_expr_symbol(rename_token(&name, token_map)),
        Expr::PtrLane { base, lane } => Expr::PtrLane { base, lane },
        Expr::ConstMemSymbol(name) => named_constmem_symbol(rename_token(&name, token_map)),
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
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
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

fn pointer_lane_name(base: &str, lane: PointerLane) -> String {
    format!("{}.{}", base, lane.render_suffix())
}

fn named_expr_symbol(name: String) -> Expr {
    if let Some((base, lane)) = PointerLane::parse_named(&name) {
        Expr::PtrLane { base, lane }
    } else {
        Expr::Reg(name)
    }
}

fn named_constmem_symbol(name: String) -> Expr {
    if let Some((base, lane)) = PointerLane::parse_named(&name) {
        Expr::PtrLane { base, lane }
    } else {
        Expr::ConstMemSymbol(name)
    }
}

fn named_lvalue_symbol(name: String) -> LValue {
    if let Some((base, lane)) = PointerLane::parse_named(&name) {
        LValue::PtrLane { base, lane }
    } else {
        LValue::Var(name)
    }
}

fn lvalue_symbol_name(lvalue: &LValue) -> Option<String> {
    match lvalue {
        LValue::Raw(name) if is_symbolic_name(name) => Some(name.clone()),
        LValue::Var(name) => Some(name.clone()),
        LValue::PtrLane { base, lane } => Some(pointer_lane_name(base, *lane)),
        _ => None,
    }
}

fn intrinsic_args<'a>(expr: &'a Expr, op: IntrinsicOp) -> Option<&'a [Expr]> {
    match expr {
        Expr::Intrinsic { op: actual, args } if *actual == op => Some(args.as_slice()),
        Expr::CallLike { func, args } if func == op.render_name() => Some(args.as_slice()),
        _ => None,
    }
}

#[derive(Clone)]
struct Addr64Defs {
    lo_defs: HashMap<String, LoAddDef>,
    carry_defs: HashMap<String, CarryDef>,
    guarded_carry_defs: HashMap<String, Vec<GuardedCarryDef>>,
    lea_hi_defs: HashMap<String, LeaHiDef>,
    hi_add_defs: HashMap<String, HiAddCarryDef>,
    select_defs: HashMap<String, SelectDef>,
    copy_defs: HashMap<String, String>,
    loop_carried_names: HashSet<String>,
    allow_alias_resolution: bool,
}

#[derive(Clone)]
struct LoAddDef {
    offset: Expr,
    ptr_lo: String,
    loop_entry_ptr_lo: Option<String>,
}

#[derive(Clone)]
struct CarryDef {
    offset: Expr,
    ptr_lo: String,
}

#[derive(Clone)]
struct LeaHiDef {
    raw_offset: Expr,
    scaled_offset: Expr,
    ptr_hi: String,
    carry_var: String,
    loop_entry_ptr_hi: Option<String>,
}

#[derive(Clone)]
struct GuardedCarryDef {
    condition: Expr,
    carry: CarryDef,
}

#[derive(Clone)]
enum CarryIncrement {
    Var(String),
    Inline(CarryDef),
}

#[derive(Clone)]
struct HiAddCarryDef {
    hi_offset_var: Option<String>,
    ptr_hi: Option<String>,
    pair_hi_of_lo: Option<String>,
    carry: CarryIncrement,
    loop_entry_ptr_hi: Option<String>,
    loop_entry_pair_hi_of_lo: Option<String>,
}

#[derive(Clone)]
struct SelectDef {
    condition: Expr,
    then_expr: Expr,
    else_expr: Expr,
}

impl Default for Addr64Defs {
    fn default() -> Self {
        Self {
            lo_defs: HashMap::new(),
            carry_defs: HashMap::new(),
            guarded_carry_defs: HashMap::new(),
            lea_hi_defs: HashMap::new(),
            hi_add_defs: HashMap::new(),
            select_defs: HashMap::new(),
            copy_defs: HashMap::new(),
            loop_carried_names: HashSet::new(),
            allow_alias_resolution: true,
        }
    }
}

impl Addr64Defs {
    fn named_mode() -> Self {
        Self {
            allow_alias_resolution: false,
            ..Self::default()
        }
    }

    fn merged_with(&self, local: Addr64Defs) -> Addr64Defs {
        let mut merged = self.clone();
        merged.allow_alias_resolution = self.allow_alias_resolution && local.allow_alias_resolution;
        merged.lo_defs.extend(local.lo_defs);
        merged.carry_defs.extend(local.carry_defs);
        for (name, defs) in local.guarded_carry_defs {
            merged
                .guarded_carry_defs
                .entry(name)
                .or_default()
                .extend(defs);
        }
        merged.lea_hi_defs.extend(local.lea_hi_defs);
        merged.hi_add_defs.extend(local.hi_add_defs);
        merged.select_defs.extend(local.select_defs);
        merged.copy_defs.extend(local.copy_defs);
        merged.loop_carried_names.extend(local.loop_carried_names);
        merged
    }
}

fn addr64_fold_stmt(stmt: Stmt, inherited: &Addr64Defs) -> Stmt {
    match stmt {
        Stmt::Label { name, body } => {
            let label_defs = inherited.merged_with(collect_addr64_defs(
                std::slice::from_ref(body.as_ref()),
                inherited,
            ));
            Stmt::Label {
                name,
                body: Box::new(addr64_fold_stmt(*body, &label_defs)),
            }
        }
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
            let mut loop_defs = inherited.clone();
            loop_defs
                .loop_carried_names
                .extend(collect_loop_carried_names(body.as_ref()));
            Stmt::Loop {
                kind,
                condition: condition
                    .map(|expr| rewrite_addr64_expr(expr, &loop_defs, &mut candidates, &[])),
                body: Box::new(addr64_fold_stmt(*body, &loop_defs)),
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
    // Fold left-to-right so nested regions can see pointer-lane aliases from
    // earlier siblings in the same sequence before raw addr64 text leaks.
    let mut defs = inherited.clone();
    let mut folded = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        let stmt = addr64_fold_stmt(stmt, &defs);
        defs = defs.merged_with(collect_addr64_defs(std::slice::from_ref(&stmt), &defs));
        folded.push(stmt);
    }

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
fn collect_addr64_defs(stmts: &[Stmt], inherited: &Addr64Defs) -> Addr64Defs {
    let mut defs = Addr64Defs::default();
    for stmt in stmts {
        let scope = inherited.merged_with(defs.clone());
        collect_addr64_defs_from_stmt(stmt, &mut defs, &scope);
    }
    defs
}

fn collect_addr64_defs_from_stmt(stmt: &Stmt, defs: &mut Addr64Defs, inherited: &Addr64Defs) {
    match stmt {
        Stmt::Label { body, .. } => {
            *defs = defs.merged_with(collect_addr64_defs(
                std::slice::from_ref(body.as_ref()),
                inherited,
            ));
            return;
        }
        // Plain blocks/sequences are just grouping; their defs stay in scope for
        // later siblings in the parent sequence and must be collected.
        Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
            *defs = defs.merged_with(collect_addr64_defs(stmts, inherited));
            return;
        }
        // Single-arm initialization branches often feed phi-lowered copies after the
        // branch. Collect their defs so later siblings can still reconstruct the pair.
        Stmt::If {
            then_branch,
            else_branch: None,
            ..
        } => {
            *defs = defs.merged_with(export_boundary_addr64_defs(collect_addr64_defs(
                std::slice::from_ref(then_branch.as_ref()),
                inherited,
            )));
            return;
        }
        // Loop-carried pointer updates often feed post-loop cleanup/tail blocks.
        // Collecting body defs here lets later siblings reuse the final carried
        // low/high lane relationships instead of falling back to packed addr64.
        Stmt::Loop { body, .. } => {
            let loop_names = collect_loop_carried_names(body.as_ref());
            defs.loop_carried_names.extend(loop_names.clone());
            let mut loop_scope = inherited.clone();
            loop_scope.loop_carried_names.extend(loop_names);
            *defs = defs.merged_with(export_boundary_addr64_defs(collect_addr64_defs(
                std::slice::from_ref(body.as_ref()),
                &loop_scope,
            )));
            return;
        }
        _ => {}
    }

    if let Some((condition, name, rhs)) = guarded_then_var_assign(stmt) {
        if let Some(def) = match_carry_expr(rhs) {
            defs.guarded_carry_defs
                .entry(name.clone())
                .or_default()
                .push(GuardedCarryDef {
                    condition: condition.clone(),
                    carry: def,
                });
        }
    }

    let Some((name, rhs)) = direct_var_assign(stmt) else {
        return;
    };
    let rhs = strip_loop_phi_expr(rhs);
    if let Some(def) = match_lo_add_expr(rhs) {
        defs.lo_defs.insert(name.clone(), def);
    }
    if let Some(def) = match_carry_expr(rhs) {
        defs.carry_defs.insert(name.clone(), def);
    }
    if let Some(def) = match_lea_hi_expr(rhs) {
        defs.lea_hi_defs.insert(name.clone(), def);
    }
    if let Some(def) = match_hi_add_carry_expr(rhs) {
        defs.hi_add_defs.insert(name.clone(), def);
    }
    if let Some(def) = match_select_expr(rhs) {
        defs.select_defs.insert(name.clone(), def);
    }
    if let Some(def) = match_copy_expr(rhs) {
        defs.copy_defs.insert(name.clone(), def);
    }

    if let Some(def) = defs.lo_defs.get_mut(&name) {
        if def.ptr_lo == name && inherited.loop_carried_names.contains(&name) {
            def.loop_entry_ptr_lo = resolve_loop_entry_lo_anchor_name(&name, inherited);
        }
    }
    if let Some(def) = defs.lea_hi_defs.get_mut(&name) {
        if def.ptr_hi == name && inherited.loop_carried_names.contains(&name) {
            def.loop_entry_ptr_hi = resolve_loop_entry_hi_anchor_name(&name, inherited);
        }
    }
    if let Some(def) = defs.hi_add_defs.get_mut(&name) {
        if inherited.loop_carried_names.contains(&name) {
            if def.ptr_hi.as_deref() == Some(name.as_str()) {
                def.loop_entry_ptr_hi = resolve_loop_entry_hi_anchor_name(&name, inherited);
            }
            if def.pair_hi_of_lo.as_deref() == Some(name.as_str()) {
                def.loop_entry_pair_hi_of_lo = resolve_loop_entry_lo_anchor_name(&name, inherited);
            }
        }
    }
}

fn export_boundary_addr64_defs(mut defs: Addr64Defs) -> Addr64Defs {
    // These boundary exports are only meant to preserve pointer-pair structure
    // across region boundaries. Generic copy/select aliases from inside a branch
    // or loop do not dominate later siblings and can corrupt unrelated scalar or
    // predicate cleanup if we treat them as unconditional. Keep only copy chains
    // that still resolve to explicit pointer-lane seeds or structured low/high
    // addr64 defs; later cleanup depends on those dominating aliases.
    let copy_defs = defs.copy_defs.clone();
    let lo_defs = defs.lo_defs.clone();
    let lea_hi_defs = defs.lea_hi_defs.clone();
    let hi_add_defs = defs.hi_add_defs.clone();
    defs.copy_defs.retain(|_, src| {
        should_export_boundary_copy(
            src,
            &lo_defs,
            &lea_hi_defs,
            &hi_add_defs,
            &copy_defs,
            &mut BTreeSet::new(),
        )
    });
    defs.select_defs.clear();
    defs
}

fn should_export_boundary_copy(
    src: &str,
    lo_defs: &HashMap<String, LoAddDef>,
    lea_hi_defs: &HashMap<String, LeaHiDef>,
    hi_add_defs: &HashMap<String, HiAddCarryDef>,
    copy_defs: &HashMap<String, String>,
    seen: &mut BTreeSet<String>,
) -> bool {
    if !seen.insert(src.to_string()) {
        return false;
    }
    if strip_ptr_suffix(src, ".lo32", "_lo32").is_some()
        || strip_ptr_suffix(src, ".hi32", "_hi32").is_some()
        || lo_defs.contains_key(src)
        || lea_hi_defs.contains_key(src)
        || hi_add_defs.contains_key(src)
    {
        return true;
    }
    copy_defs.get(src).is_some_and(|next| {
        should_export_boundary_copy(next, lo_defs, lea_hi_defs, hi_add_defs, copy_defs, seen)
    })
}

fn collect_loop_carried_names(stmt: &Stmt) -> HashSet<String> {
    let mut names = HashSet::new();
    collect_loop_carried_names_into(stmt, &mut names);
    names
}

fn collect_loop_carried_names_into(stmt: &Stmt, out: &mut HashSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_loop_carried_names_into(stmt, out);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_loop_carried_names_into(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_loop_carried_names_into(else_branch, out);
            }
        }
        Stmt::Loop { body, .. } => collect_loop_carried_names_into(body, out),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_loop_carried_names_into(body, out);
            }
            if let Some(default) = default {
                collect_loop_carried_names_into(default, out);
            }
        }
        Stmt::Assign {
            dst: LValue::Var(name),
            src,
        } if loop_phi_arg(src).is_some() => {
            out.insert(name.clone());
        }
        _ => {}
    }
}

fn direct_var_assign(stmt: &Stmt) -> Option<(String, &Expr)> {
    let Stmt::Assign {
        dst: LValue::Var(name),
        src,
    } = stmt
    else {
        return None;
    };
    Some((name.clone(), src))
}

fn single_direct_var_assign(stmt: &Stmt) -> Option<(String, &Expr)> {
    match stmt {
        Stmt::Assign {
            dst: LValue::Var(name),
            src,
        } => Some((name.clone(), src)),
        Stmt::Sequence(stmts) | Stmt::Block(stmts) if stmts.len() == 1 => {
            single_direct_var_assign(&stmts[0])
        }
        _ => None,
    }
}

fn guarded_then_var_assign(stmt: &Stmt) -> Option<(&Expr, String, &Expr)> {
    let Stmt::If {
        condition,
        then_branch,
        else_branch: None,
    } = stmt
    else {
        return None;
    };
    let (name, rhs) = single_direct_var_assign(then_branch.as_ref())?;
    Some((condition, name, rhs))
}

fn match_lo_add_expr(rhs: &Expr) -> Option<LoAddDef> {
    let Expr::Binary { op, lhs, rhs } = rhs else {
        return None;
    };
    if op != "+" {
        return None;
    }
    if let Some(ptr_lo) = match_var_name(rhs) {
        return Some(LoAddDef {
            offset: (**lhs).clone(),
            ptr_lo,
            loop_entry_ptr_lo: None,
        });
    }
    Some(LoAddDef {
        offset: (**rhs).clone(),
        ptr_lo: match_var_name(lhs)?,
        loop_entry_ptr_lo: None,
    })
}

fn match_carry_expr(rhs: &Expr) -> Option<CarryDef> {
    match_carry_args(intrinsic_args(rhs, IntrinsicOp::CarryU32Add3)?)
}

fn match_lea_hi_expr(rhs: &Expr) -> Option<LeaHiDef> {
    let args = intrinsic_args(rhs, IntrinsicOp::LeaHiX)
        .or_else(|| intrinsic_args(rhs, IntrinsicOp::LeaHiXSx32))?;
    if args.len() != 4 {
        return None;
    }
    Some(LeaHiDef {
        raw_offset: args[0].clone(),
        scaled_offset: scale_addr_offset_expr(&args[0], &args[2])?,
        ptr_hi: match_var_name(&args[1])?,
        carry_var: match_var_name(&args[3])?,
        loop_entry_ptr_hi: None,
    })
}

fn match_pair_hi_lo_name(expr: &Expr) -> Option<String> {
    let args = intrinsic_args(expr, IntrinsicOp::PairHi)?;
    if args.len() != 1 {
        return None;
    }
    match_var_name(&args[0])
}

fn match_hi_add_carry_expr(rhs: &Expr) -> Option<HiAddCarryDef> {
    let Expr::Binary { op, lhs, rhs } = rhs else {
        return None;
    };
    if op != "+" {
        return None;
    }
    let (hi_offset_var, ptr_hi, pair_hi_of_lo) = match lhs.as_ref() {
        Expr::Binary {
            op: sum_op,
            lhs: hi_offset,
            rhs: ptr_hi,
        } if sum_op == "+" => (
            match_var_name(hi_offset),
            match_var_name(ptr_hi),
            match_pair_hi_lo_name(ptr_hi),
        ),
        other => (None, match_var_name(other), match_pair_hi_lo_name(other)),
    };
    if ptr_hi.is_none() && pair_hi_of_lo.is_none() {
        return None;
    }
    Some(HiAddCarryDef {
        hi_offset_var,
        ptr_hi,
        pair_hi_of_lo,
        carry: match_carry_increment(rhs)?,
        loop_entry_ptr_hi: None,
        loop_entry_pair_hi_of_lo: None,
    })
}

fn match_inline_carry_expr(expr: &Expr) -> Option<CarryDef> {
    match_carry_args(intrinsic_args(expr, IntrinsicOp::CarryU32Add3)?)
}

fn match_carry_args(args: &[Expr]) -> Option<CarryDef> {
    if args.len() != 3 || !expr_is_zero(&args[2]) {
        return None;
    }
    if let Some(ptr_lo) = match_var_name(&args[1]) {
        return Some(CarryDef {
            offset: args[0].clone(),
            ptr_lo,
        });
    }
    Some(CarryDef {
        offset: args[1].clone(),
        ptr_lo: match_var_name(&args[0])?,
    })
}

fn lea_hi_matches_lo_offset(hi: &LeaHiDef, lo_offset: &Expr) -> bool {
    hi.raw_offset == *lo_offset || hi.scaled_offset == *lo_offset
}

fn match_carry_increment(expr: &Expr) -> Option<CarryIncrement> {
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
    if let Some(var) = match_var_name(cond) {
        return Some(CarryIncrement::Var(var));
    }
    match_inline_carry_expr(cond).map(CarryIncrement::Inline)
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

fn resolve_loop_entry_lo_anchor_name(name: &str, defs: &Addr64Defs) -> Option<String> {
    let mut seen = BTreeSet::new();
    resolve_loop_entry_lo_anchor_name_inner(name, defs, &mut seen)
}

fn resolve_loop_entry_lo_anchor_name_inner(
    name: &str,
    defs: &Addr64Defs,
    seen: &mut BTreeSet<String>,
) -> Option<String> {
    let (resolved, _) = resolve_guarded_alias(name.to_string(), defs, &[]);
    if !seen.insert(resolved.clone()) {
        return None;
    }
    if let Some(def) = defs.lo_defs.get(&resolved) {
        if def.ptr_lo != resolved {
            return Some(def.ptr_lo.clone());
        }
        if let Some(seed) = def.loop_entry_ptr_lo.as_ref() {
            return Some(seed.clone());
        }
    }
    Some(resolved)
}

fn resolve_loop_entry_hi_anchor_name(name: &str, defs: &Addr64Defs) -> Option<String> {
    let mut seen = BTreeSet::new();
    resolve_loop_entry_hi_anchor_name_inner(name, defs, &mut seen)
}

fn resolve_loop_entry_hi_anchor_name_inner(
    name: &str,
    defs: &Addr64Defs,
    seen: &mut BTreeSet<String>,
) -> Option<String> {
    let (resolved, _) = resolve_guarded_alias(name.to_string(), defs, &[]);
    if !seen.insert(resolved.clone()) {
        return None;
    }
    if let Some(def) = defs.lea_hi_defs.get(&resolved) {
        if def.ptr_hi != resolved {
            return Some(def.ptr_hi.clone());
        }
        if let Some(seed) = def.loop_entry_ptr_hi.as_ref() {
            return Some(seed.clone());
        }
    }
    if let Some(def) = defs.hi_add_defs.get(&resolved) {
        if let Some(ptr_hi) = def.ptr_hi.as_ref() {
            if ptr_hi != &resolved {
                return Some(ptr_hi.clone());
            }
        }
        if let Some(seed) = def.loop_entry_ptr_hi.as_ref() {
            return Some(seed.clone());
        }
    }
    Some(resolved)
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
        Expr::Binary { op, lhs, rhs } => {
            let lhs = rewrite_addr64_expr(*lhs, defs, candidates, active_guards);
            let rhs = rewrite_addr64_expr(*rhs, defs, candidates, active_guards);
            if op == "+" {
                if let Some(expr) =
                    fold_split_addr64_sum(&lhs, &rhs, defs, candidates, active_guards)
                {
                    expr
                } else if let Some(expr) =
                    fold_split_addr64_sum(&rhs, &lhs, defs, candidates, active_guards)
                {
                    expr
                } else {
                    Expr::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }
                }
            } else {
                Expr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
        }
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
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
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
            fold_addr64_use(&lo, &hi, defs, candidates, active_guards).unwrap_or(Expr::Addr64 {
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

fn fold_split_addr64_sum(
    addr_side: &Expr,
    other: &Expr,
    defs: &Addr64Defs,
    candidates: &mut BTreeSet<String>,
    active_guards: &[Expr],
) -> Option<Expr> {
    let Expr::Addr64 { lo, hi } = addr_side else {
        return None;
    };
    let (ptr_lo, ptr_lo_aliases) =
        resolve_guarded_alias(match_var_name(other)?, defs, active_guards);
    let (ptr_hi, ptr_hi_aliases) = resolve_guarded_alias(match_var_name(hi)?, defs, active_guards);
    let mut seen = BTreeSet::new();
    let (base_expr, nested_candidates) =
        resolve_addr64_pair_base_expr(&ptr_lo, &ptr_hi, defs, &mut seen, active_guards)?;
    candidates.extend(nested_candidates);
    candidates.extend(ptr_lo_aliases);
    candidates.insert(ptr_lo);
    candidates.extend(ptr_hi_aliases);
    candidates.insert(ptr_hi);
    Some(render_folded_addr64_base(base_expr, lo))
}

fn collect_multi_assigned_names(stmts: &[Stmt]) -> HashSet<String> {
    let mut counts = HashMap::<String, usize>::new();
    for stmt in stmts {
        collect_assignment_counts(stmt, &mut counts);
    }
    counts
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect()
}

fn collect_assignment_counts(stmt: &Stmt, counts: &mut HashMap<String, usize>) {
    match stmt {
        Stmt::Assign { dst, .. } => {
            if let Some(name) = lvalue_symbol_name(dst) {
                *counts.entry(name).or_default() += 1;
            }
        }
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_assignment_counts(stmt, counts);
            }
        }
        Stmt::Label { body, .. } => collect_assignment_counts(body, counts),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_assignment_counts(then_branch, counts);
            if let Some(else_branch) = else_branch {
                collect_assignment_counts(else_branch, counts);
            }
        }
        Stmt::Loop { body, .. } => collect_assignment_counts(body, counts),
        Stmt::Switch { cases, default, .. } => {
            for (_, body) in cases {
                collect_assignment_counts(body, counts);
            }
            if let Some(default) = default {
                collect_assignment_counts(default, counts);
            }
        }
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::ExprStmt(_)
        | Stmt::Goto(_)
        | Stmt::Empty => {}
    }
}

fn push_active_guard(active_guards: &[Expr], guard: Expr) -> Vec<Expr> {
    let mut guards = active_guards.to_vec();
    guards.push(guard);
    guards
}

fn resolve_guard_selected_expr(expr: Expr, defs: &Addr64Defs, active_guards: &[Expr]) -> Expr {
    if !defs.allow_alias_resolution {
        return expr;
    }
    let mut current = expr;
    let mut seen = BTreeSet::new();
    loop {
        let Some(name) = match_var_name(&current) else {
            return current;
        };
        if defs.loop_carried_names.contains(&name) {
            return current;
        }
        if !seen.insert(name.clone()) {
            return current;
        }
        if let Some(copy) = defs.copy_defs.get(&name) {
            current = named_expr_symbol(copy.clone());
            continue;
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

fn resolve_guarded_carry_def(
    name: &str,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<CarryDef> {
    let mut current = name.to_string();
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(current.clone()) {
            return None;
        }
        if let Some(def) = defs.carry_defs.get(&current) {
            return Some(def.clone());
        }
        if let Some(guarded_defs) = defs.guarded_carry_defs.get(&current) {
            for guarded in guarded_defs {
                if resolve_guard_branch(active_guards, &guarded.condition) == Some(true) {
                    return Some(guarded.carry.clone());
                }
            }
        }
        let Some(select) = defs.select_defs.get(&current) else {
            return None;
        };
        let branch = match resolve_guard_branch(active_guards, &select.condition) {
            Some(true) => &select.then_expr,
            Some(false) => &select.else_expr,
            None => return None,
        };
        if let Some(def) = match_inline_carry_expr(branch) {
            return Some(def);
        }
        current = match_var_name(branch)?;
    }
}

fn find_loop_carried_hi_candidate(
    lo_name: &str,
    hi_name: &str,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<String> {
    if defs.loop_carried_names.contains(hi_name) {
        return None;
    }
    let lo_info = defs.lo_defs.get(lo_name)?;
    let (current_ptr_lo, _) = resolve_guarded_alias(lo_info.ptr_lo.clone(), defs, active_guards);

    let hi_add_matches = |name: &String, hi_info: &HiAddCarryDef| {
        if !defs.loop_carried_names.contains(name) {
            return false;
        }
        let derived_from_seed = hi_info.loop_entry_ptr_hi.as_deref() == Some(hi_name)
            || hi_info.ptr_hi.as_deref() == Some(hi_name);
        if !derived_from_seed {
            return false;
        }
        match &hi_info.carry {
            CarryIncrement::Var(carry_var) => {
                resolve_guarded_carry_def(carry_var, defs, active_guards)
                    .and_then(|carry| {
                        let (carry_ptr_lo, _) =
                            resolve_guarded_alias(carry.ptr_lo.clone(), defs, active_guards);
                        Some(carry.offset == lo_info.offset && carry_ptr_lo == current_ptr_lo)
                    })
                    .unwrap_or(false)
            }
            CarryIncrement::Inline(carry) => {
                let (carry_ptr_lo, _) =
                    resolve_guarded_alias(carry.ptr_lo.clone(), defs, active_guards);
                carry.offset == lo_info.offset && carry_ptr_lo == current_ptr_lo
            }
        }
    };

    defs.hi_add_defs
        .iter()
        .find_map(|(name, hi_info)| hi_add_matches(name, hi_info).then(|| name.clone()))
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
    active_guards: &[Expr],
) -> Option<Expr> {
    if let (Some(orig_lo_name), Some(orig_hi_name)) = (match_var_name(lo), match_var_name(hi)) {
        let loop_pair = defs.loop_carried_names.contains(&orig_lo_name)
            && defs.loop_carried_names.contains(&orig_hi_name);
        if loop_pair {
            if let Some(expr) =
                fold_loop_carried_addr64_use(&orig_lo_name, &orig_hi_name, defs, active_guards)
            {
                return Some(expr);
            }
            if let Some(expr) = fold_loop_carried_seed_alias_addr64_use(
                &orig_lo_name,
                &orig_hi_name,
                defs,
                active_guards,
            ) {
                return Some(expr);
            }
        }
        let (lo_name, lo_aliases) =
            resolve_guarded_alias(orig_lo_name.clone(), defs, active_guards);
        let (hi_name, hi_aliases) =
            resolve_guarded_alias(orig_hi_name.clone(), defs, active_guards);
        if defs.loop_carried_names.contains(&lo_name) {
            if let Some(derived_hi) =
                find_loop_carried_hi_candidate(&lo_name, &hi_name, defs, active_guards)
            {
                if let Some(expr) =
                    fold_loop_carried_addr64_use(&lo_name, &derived_hi, defs, active_guards)
                {
                    return Some(expr);
                }
            }
        }
        let alias_chain_crosses_loop = loop_pair
            || lo_aliases
                .iter()
                .chain(hi_aliases.iter())
                .any(|name| defs.loop_carried_names.contains(name));
        if alias_chain_crosses_loop && (orig_lo_name != lo_name || orig_hi_name != hi_name) {
            let mut seen = BTreeSet::new();
            if let Some((base_expr, mut local_candidates)) =
                resolve_addr64_pair_base_expr(&lo_name, &hi_name, defs, &mut seen, active_guards)
            {
                let expr = if orig_lo_name == lo_name {
                    base_expr
                } else {
                    render_folded_addr64_base(
                        base_expr,
                        &Expr::Binary {
                            op: "-".to_string(),
                            lhs: Box::new(Expr::Raw(orig_lo_name.clone())),
                            rhs: Box::new(Expr::Raw(lo_name.clone())),
                        },
                    )
                };
                local_candidates.extend(lo_aliases);
                local_candidates.insert(lo_name);
                local_candidates.extend(hi_aliases);
                local_candidates.insert(hi_name);
                local_candidates.insert(orig_lo_name);
                local_candidates.insert(orig_hi_name);
                candidates.extend(local_candidates);
                return Some(expr);
            }
        }
        let mut seen = BTreeSet::new();
        let (expr, mut local_candidates) =
            fold_addr64_pair_names(&lo_name, &hi_name, defs, &mut seen, active_guards)?;
        local_candidates.extend(lo_aliases);
        local_candidates.insert(lo_name);
        local_candidates.extend(hi_aliases);
        local_candidates.insert(hi_name);
        candidates.extend(local_candidates);
        return Some(expr);
    }

    fold_inline_addr64_use(lo, hi, defs, candidates, active_guards)
}

fn fold_loop_carried_addr64_use(
    lo_name: &str,
    hi_name: &str,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<Expr> {
    let (resolved_lo, _) = resolve_guarded_alias(lo_name.to_string(), defs, active_guards);
    let (resolved_hi, _) = resolve_guarded_alias(hi_name.to_string(), defs, active_guards);
    let lo_info = defs.lo_defs.get(&resolved_lo)?;
    let (current_ptr_lo, _) = resolve_guarded_alias(lo_info.ptr_lo.clone(), defs, active_guards);
    let anchor_lo_raw =
        if defs.loop_carried_names.contains(&resolved_lo) && lo_info.ptr_lo == resolved_lo {
            lo_info
                .loop_entry_ptr_lo
                .clone()
                .unwrap_or_else(|| lo_info.ptr_lo.clone())
        } else {
            lo_info.ptr_lo.clone()
        };
    let (anchor_ptr_lo, _) = resolve_guarded_alias(anchor_lo_raw, defs, active_guards);

    let base_expr = if let Some(hi_info) = defs.lea_hi_defs.get(&resolved_hi) {
        if !lea_hi_matches_lo_offset(hi_info, &lo_info.offset) {
            return None;
        }
        let anchor_hi_raw =
            if defs.loop_carried_names.contains(&resolved_hi) && hi_info.ptr_hi == resolved_hi {
                hi_info
                    .loop_entry_ptr_hi
                    .clone()
                    .unwrap_or_else(|| hi_info.ptr_hi.clone())
            } else {
                hi_info.ptr_hi.clone()
            };
        let (anchor_ptr_hi, _) = resolve_guarded_alias(anchor_hi_raw, defs, active_guards);
        let mut seen = BTreeSet::new();
        let (base_expr, _) = resolve_addr64_pair_base_expr(
            &anchor_ptr_lo,
            &anchor_ptr_hi,
            defs,
            &mut seen,
            active_guards,
        )?;

        let (carry_var, _) = resolve_guarded_alias(hi_info.carry_var.clone(), defs, active_guards);
        let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
        let (carry_ptr_lo, _) =
            resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
        if carry_info.offset != lo_info.offset || carry_ptr_lo != current_ptr_lo {
            return None;
        }
        base_expr
    } else if let Some(hi_info) = defs.hi_add_defs.get(&resolved_hi) {
        match &hi_info.carry {
            CarryIncrement::Var(carry_var) => {
                let carry_info = resolve_guarded_carry_def(carry_var, defs, active_guards)?;
                let (carry_ptr_lo, _) =
                    resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
                if carry_info.offset != lo_info.offset || carry_ptr_lo != current_ptr_lo {
                    return None;
                }
            }
            CarryIncrement::Inline(carry_info) => {
                let (carry_ptr_lo, _) =
                    resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
                if carry_info.offset != lo_info.offset || carry_ptr_lo != current_ptr_lo {
                    return None;
                }
            }
        }

        if let Some(ptr_hi_raw) = &hi_info.ptr_hi {
            let anchor_hi_raw =
                if defs.loop_carried_names.contains(&resolved_hi) && ptr_hi_raw == &resolved_hi {
                    hi_info
                        .loop_entry_ptr_hi
                        .clone()
                        .unwrap_or_else(|| ptr_hi_raw.clone())
                } else {
                    ptr_hi_raw.clone()
                };
            let (anchor_ptr_hi, _) = resolve_guarded_alias(anchor_hi_raw, defs, active_guards);
            let mut seen = BTreeSet::new();
            resolve_addr64_pair_base_expr(
                &anchor_ptr_lo,
                &anchor_ptr_hi,
                defs,
                &mut seen,
                active_guards,
            )
            .or_else(|| {
                Some((
                    Expr::Addr64 {
                        lo: Box::new(Expr::Raw(anchor_ptr_lo.clone())),
                        hi: Box::new(Expr::Raw(anchor_ptr_hi)),
                    },
                    BTreeSet::new(),
                ))
            })?
            .0
        } else {
            let pair_lo_raw = hi_info.pair_hi_of_lo.as_ref()?;
            let anchor_pair_lo_raw =
                if defs.loop_carried_names.contains(pair_lo_raw) && pair_lo_raw == &resolved_lo {
                    hi_info
                        .loop_entry_pair_hi_of_lo
                        .clone()
                        .or_else(|| lo_info.loop_entry_ptr_lo.clone())
                        .unwrap_or_else(|| pair_lo_raw.clone())
                } else {
                    pair_lo_raw.clone()
                };
            let (pair_lo, _) = resolve_guarded_alias(anchor_pair_lo_raw, defs, active_guards);
            let mut seen = BTreeSet::new();
            resolve_addr64_base_expr_from_lo_name(&pair_lo, defs, &mut seen, active_guards)?.0
        }
    } else {
        return None;
    };
    // Preserve the loop-carried low lane in the reconstructed offset instead
    // of collapsing back to the loop-entry seed. Otherwise prelude copies like
    // `phi_lo = init_lo` can freeze a carried pointer at its initial offset
    // (`+16`) even though the loop body updates `phi_lo` on every iteration.
    let offset_name = if defs.loop_carried_names.contains(lo_name) {
        lo_name
    } else {
        resolved_lo.as_str()
    };
    let offset = if offset_name == anchor_ptr_lo {
        lo_info.offset.clone()
    } else {
        Expr::Binary {
            op: "-".to_string(),
            lhs: Box::new(Expr::Raw(offset_name.to_string())),
            rhs: Box::new(Expr::Raw(anchor_ptr_lo)),
        }
    };
    Some(render_folded_addr64_base(base_expr, &offset))
}

fn fold_loop_carried_seed_alias_addr64_use(
    orig_lo: &str,
    orig_hi: &str,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<Expr> {
    let (resolved_lo, _) = resolve_guarded_alias(orig_lo.to_string(), defs, active_guards);
    let (resolved_hi, _) = resolve_guarded_alias(orig_hi.to_string(), defs, active_guards);
    if resolved_lo == orig_lo && resolved_hi == orig_hi {
        return None;
    }

    let lo_info = defs.lo_defs.get(&resolved_lo)?;
    let (seed_ptr_lo, _) = resolve_guarded_alias(lo_info.ptr_lo.clone(), defs, active_guards);

    let base_expr = if let Some(hi_info) = defs.lea_hi_defs.get(&resolved_hi) {
        if !lea_hi_matches_lo_offset(hi_info, &lo_info.offset) {
            return None;
        }
        let (seed_ptr_hi, _) = resolve_guarded_alias(hi_info.ptr_hi.clone(), defs, active_guards);
        let mut seen = BTreeSet::new();
        resolve_addr64_pair_base_expr(&seed_ptr_lo, &seed_ptr_hi, defs, &mut seen, active_guards)?.0
    } else if let Some(hi_info) = defs.hi_add_defs.get(&resolved_hi) {
        match &hi_info.carry {
            CarryIncrement::Var(carry_var) => {
                let (carry_var, _) = resolve_guarded_alias(carry_var.clone(), defs, active_guards);
                let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
                let (carry_ptr_lo, _) =
                    resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
                if carry_info.offset != lo_info.offset || carry_ptr_lo != seed_ptr_lo {
                    return None;
                }
            }
            CarryIncrement::Inline(carry_info) => {
                let (carry_ptr_lo, _) =
                    resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
                if carry_info.offset != lo_info.offset || carry_ptr_lo != seed_ptr_lo {
                    return None;
                }
            }
        }
        resolve_hi_add_base_expr(hi_info, &seed_ptr_lo, defs, active_guards)?.0
    } else {
        return None;
    };

    Some(render_folded_addr64_base(
        base_expr.clone(),
        &Expr::Binary {
            op: "-".to_string(),
            lhs: Box::new(Expr::Raw(orig_lo.to_string())),
            rhs: Box::new(Expr::LaneExtract {
                value: Box::new(base_expr),
                lane: PointerLane::Lo32,
            }),
        },
    ))
}

fn fold_inline_addr64_use(
    lo: &Expr,
    hi: &Expr,
    defs: &Addr64Defs,
    candidates: &mut BTreeSet<String>,
    active_guards: &[Expr],
) -> Option<Expr> {
    let lo_info = match_lo_add_expr(lo)?;
    let (ptr_lo, ptr_lo_aliases) =
        resolve_guarded_alias_preserving_loops(lo_info.ptr_lo.clone(), defs, active_guards);

    if let Some(hi_info) = match_lea_hi_expr(hi) {
        let (ptr_hi, ptr_hi_aliases) =
            resolve_guarded_alias(hi_info.ptr_hi.clone(), defs, active_guards);
        let mut seen = BTreeSet::new();
        let (base_expr, nested_candidates) =
            resolve_addr64_pair_base_expr(&ptr_lo, &ptr_hi, defs, &mut seen, active_guards)?;
        if !lea_hi_matches_lo_offset(&hi_info, &lo_info.offset) {
            return None;
        }

        let (carry_var, carry_aliases) =
            resolve_guarded_alias(hi_info.carry_var.clone(), defs, active_guards);
        let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
        let (carry_ptr_lo, carry_ptr_lo_aliases) =
            resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
        if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
            return None;
        }

        candidates.extend(nested_candidates);
        candidates.extend(carry_aliases);
        candidates.insert(carry_var);
        candidates.extend(carry_ptr_lo_aliases);
        candidates.extend(ptr_lo_aliases);
        candidates.insert(ptr_lo);
        candidates.extend(ptr_hi_aliases);
        candidates.insert(ptr_hi);
        return Some(render_folded_addr64_base(base_expr, &lo_info.offset));
    }

    let hi_info = match_hi_add_carry_expr(hi)?;
    let (base_expr, nested_candidates) =
        resolve_hi_add_base_expr(&hi_info, &ptr_lo, defs, active_guards)?;

    match &hi_info.carry {
        CarryIncrement::Var(carry_var) => {
            let (carry_var, carry_aliases) =
                resolve_guarded_alias(carry_var.clone(), defs, active_guards);
            let carry_info = resolve_guarded_carry_def(&carry_var, defs, active_guards)?;
            let (carry_ptr_lo, carry_ptr_lo_aliases) =
                resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
            if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
                return None;
            }
            candidates.extend(carry_aliases);
            candidates.insert(carry_var);
            candidates.extend(carry_ptr_lo_aliases);
        }
        CarryIncrement::Inline(carry_info) => {
            let (carry_ptr_lo, carry_ptr_lo_aliases) =
                resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
            if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
                return None;
            }
            candidates.extend(carry_ptr_lo_aliases);
        }
    }

    candidates.extend(nested_candidates);
    candidates.extend(ptr_lo_aliases);
    candidates.insert(ptr_lo);
    if let Some(hi_offset_var) = hi_info.hi_offset_var {
        let (resolved_hi_offset, hi_offset_aliases) =
            resolve_guarded_alias(hi_offset_var, defs, active_guards);
        candidates.extend(hi_offset_aliases);
        candidates.insert(resolved_hi_offset);
    }

    Some(render_folded_addr64_base(base_expr, &lo_info.offset))
}

fn resolve_addr64_pair_base_expr(
    ptr_lo: &str,
    ptr_hi: &str,
    defs: &Addr64Defs,
    seen: &mut BTreeSet<(String, String)>,
    active_guards: &[Expr],
) -> Option<(Expr, BTreeSet<String>)> {
    if let Some(base) = paired_pointer_base(ptr_lo, ptr_hi) {
        return Some((Expr::Raw(base), BTreeSet::new()));
    }
    fold_addr64_pair_names(ptr_lo, ptr_hi, defs, seen, active_guards)
}

fn resolve_addr64_base_expr_from_lo_name(
    lo_name: &str,
    defs: &Addr64Defs,
    seen: &mut BTreeSet<String>,
    active_guards: &[Expr],
) -> Option<(Expr, BTreeSet<String>)> {
    let (lo_name, lo_aliases) =
        resolve_guarded_alias_preserving_loops(lo_name.to_string(), defs, active_guards);
    if !seen.insert(lo_name.clone()) {
        return None;
    }
    if let Some(base) = strip_ptr_suffix(&lo_name, ".lo32", "_lo32") {
        let mut candidates = BTreeSet::new();
        candidates.extend(lo_aliases);
        candidates.insert(lo_name);
        return Some((Expr::Raw(base), candidates));
    }
    let lo_info = defs.lo_defs.get(&lo_name)?;
    let (ptr_lo, ptr_lo_aliases) =
        resolve_guarded_alias_preserving_loops(lo_info.ptr_lo.clone(), defs, active_guards);
    let (base_expr, mut nested_candidates) =
        resolve_addr64_base_expr_from_lo_name(&ptr_lo, defs, seen, active_guards)?;
    nested_candidates.extend(lo_aliases);
    nested_candidates.insert(lo_name.clone());
    nested_candidates.extend(ptr_lo_aliases);
    nested_candidates.insert(ptr_lo.clone());
    let expr = if defs.loop_carried_names.contains(&lo_name) && ptr_lo != lo_name {
        render_folded_addr64_base(
            base_expr,
            &Expr::Binary {
                op: "-".to_string(),
                lhs: Box::new(Expr::Raw(lo_name.clone())),
                rhs: Box::new(Expr::Raw(ptr_lo.clone())),
            },
        )
    } else if expr_is_zero(&lo_info.offset) {
        base_expr
    } else {
        render_folded_addr64_base(base_expr, &lo_info.offset)
    };
    Some((expr, nested_candidates))
}

fn resolve_hi_add_base_expr(
    hi_info: &HiAddCarryDef,
    ptr_lo: &str,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> Option<(Expr, BTreeSet<String>)> {
    if let Some(ptr_hi_raw) = &hi_info.ptr_hi {
        let (ptr_hi, ptr_hi_aliases) =
            resolve_guarded_alias_preserving_loops(ptr_hi_raw.clone(), defs, active_guards);
        let mut seen = BTreeSet::new();
        let (base_expr, mut nested_candidates) =
            resolve_addr64_pair_base_expr(ptr_lo, &ptr_hi, defs, &mut seen, active_guards)
                .or_else(|| {
                    Some((
                        Expr::Addr64 {
                            lo: Box::new(Expr::Raw(ptr_lo.to_string())),
                            hi: Box::new(Expr::Raw(ptr_hi.clone())),
                        },
                        BTreeSet::new(),
                    ))
                })?;
        nested_candidates.extend(ptr_hi_aliases);
        nested_candidates.insert(ptr_hi);
        return Some((base_expr, nested_candidates));
    }

    let pair_lo_raw = hi_info.pair_hi_of_lo.as_ref()?;
    let (pair_lo, pair_lo_aliases) =
        resolve_guarded_alias_preserving_loops(pair_lo_raw.clone(), defs, active_guards);
    if pair_lo != ptr_lo {
        return None;
    }
    let mut seen = BTreeSet::new();
    let (base_expr, mut nested_candidates) =
        resolve_addr64_base_expr_from_lo_name(&pair_lo, defs, &mut seen, active_guards)?;
    nested_candidates.extend(pair_lo_aliases);
    nested_candidates.insert(pair_lo);
    Some((base_expr, nested_candidates))
}

fn fold_addr64_pair_names(
    lo_name: &str,
    hi_name: &str,
    defs: &Addr64Defs,
    seen: &mut BTreeSet<(String, String)>,
    active_guards: &[Expr],
) -> Option<(Expr, BTreeSet<String>)> {
    if !seen.insert((lo_name.to_string(), hi_name.to_string())) {
        return None;
    }

    let mut candidates = BTreeSet::new();
    if let Some(base) = paired_pointer_base(lo_name, hi_name) {
        candidates.insert(lo_name.to_string());
        candidates.insert(hi_name.to_string());
        return Some((Expr::Raw(base), candidates));
    }

    let lo_info = defs.lo_defs.get(lo_name)?;
    let (ptr_lo, ptr_lo_aliases) =
        resolve_guarded_alias_preserving_loops(lo_info.ptr_lo.clone(), defs, active_guards);

    let (base_expr, nested_candidates, carry_var, inline_carry, hi_offset_var) =
        if let Some(hi_info) = defs.lea_hi_defs.get(hi_name) {
            if !lea_hi_matches_lo_offset(hi_info, &lo_info.offset) {
                return None;
            }
            let (ptr_hi, ptr_hi_aliases) =
                resolve_guarded_alias(hi_info.ptr_hi.clone(), defs, active_guards);
            let (base_expr, mut nested_candidates) = if let Some(pair) =
                resolve_addr64_pair_base_expr(&ptr_lo, &ptr_hi, defs, seen, active_guards)
            {
                pair
            } else {
                (
                    Expr::Addr64 {
                        lo: Box::new(Expr::Raw(ptr_lo.clone())),
                        hi: Box::new(Expr::Raw(ptr_hi.clone())),
                    },
                    BTreeSet::new(),
                )
            };
            nested_candidates.extend(ptr_hi_aliases);
            nested_candidates.insert(ptr_hi);
            (
                base_expr,
                nested_candidates,
                Some(hi_info.carry_var.clone()),
                None,
                None,
            )
        } else if let Some(hi_info) = defs.hi_add_defs.get(hi_name) {
            match &hi_info.carry {
                CarryIncrement::Var(carry_var) => {
                    let carry_info = resolve_guarded_carry_def(carry_var, defs, active_guards)?;
                    let (carry_ptr_lo, _) =
                        resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
                    if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
                        return None;
                    }
                    let (base_expr, nested_candidates) =
                        resolve_hi_add_base_expr(hi_info, &ptr_lo, defs, active_guards)?;
                    (
                        base_expr,
                        nested_candidates,
                        Some(carry_var.clone()),
                        None,
                        hi_info.hi_offset_var.clone(),
                    )
                }
                CarryIncrement::Inline(carry_info) => {
                    let (carry_ptr_lo, _) =
                        resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
                    if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
                        return None;
                    }
                    let (base_expr, nested_candidates) =
                        resolve_hi_add_base_expr(hi_info, &ptr_lo, defs, active_guards)?;
                    (
                        base_expr,
                        nested_candidates,
                        None,
                        Some(carry_info.clone()),
                        hi_info.hi_offset_var.clone(),
                    )
                }
            }
        } else {
            return None;
        };

    if let Some(carry_var) = carry_var {
        let (carry_var, carry_aliases) = resolve_guarded_alias(carry_var, defs, active_guards);
        if let Some(carry_info) = resolve_guarded_carry_def(&carry_var, defs, active_guards) {
            let (carry_ptr_lo, carry_ptr_lo_aliases) =
                resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
            if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
                return None;
            }
            candidates.extend(carry_ptr_lo_aliases);
        }
        candidates.extend(carry_aliases);
        candidates.insert(carry_var);
    } else if let Some(carry_info) = inline_carry {
        let (carry_ptr_lo, carry_ptr_lo_aliases) =
            resolve_guarded_alias(carry_info.ptr_lo.clone(), defs, active_guards);
        if carry_info.offset != lo_info.offset || carry_ptr_lo != ptr_lo {
            return None;
        }
        candidates.extend(carry_ptr_lo_aliases);
    }

    candidates.extend(nested_candidates);
    candidates.insert(lo_name.to_string());
    candidates.extend(ptr_lo_aliases);
    candidates.insert(ptr_lo);
    candidates.insert(hi_name.to_string());
    if let Some(hi_offset_var) = hi_offset_var {
        let (resolved_hi_offset, hi_offset_aliases) =
            resolve_guarded_alias(hi_offset_var, defs, active_guards);
        candidates.extend(hi_offset_aliases);
        candidates.insert(resolved_hi_offset);
    }

    let expr = if expr_is_zero(&lo_info.offset) {
        base_expr
    } else {
        render_folded_addr64_base(base_expr, &lo_info.offset)
    };
    Some((expr, candidates))
}

fn resolve_guarded_alias(
    name: String,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> (String, Vec<String>) {
    if !defs.allow_alias_resolution {
        return (name, Vec::new());
    }
    let mut current = name;
    let mut aliases = Vec::new();
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(current.clone()) {
            break;
        }
        if let Some(next) = defs.copy_defs.get(&current) {
            aliases.push(current.clone());
            current = next.clone();
            continue;
        }
        if let Some(select) = defs.select_defs.get(&current) {
            let branch = match resolve_guard_branch(active_guards, &select.condition) {
                Some(true) => Some(&select.then_expr),
                Some(false) => Some(&select.else_expr),
                None => None,
            };
            if let Some(next) = branch.and_then(match_var_name) {
                aliases.push(current.clone());
                current = next;
                continue;
            }
        }
        break;
    }
    (current, aliases)
}

fn resolve_guarded_alias_preserving_loops(
    name: String,
    defs: &Addr64Defs,
    active_guards: &[Expr],
) -> (String, Vec<String>) {
    if defs.loop_carried_names.contains(&name) {
        return (name, Vec::new());
    }
    if !defs.allow_alias_resolution {
        return (name, Vec::new());
    }
    let mut current = name;
    let mut aliases = Vec::new();
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(current.clone()) {
            break;
        }
        if defs.loop_carried_names.contains(&current) {
            break;
        }
        if let Some(next) = defs.copy_defs.get(&current) {
            aliases.push(current.clone());
            current = next.clone();
            if defs.loop_carried_names.contains(&current) {
                break;
            }
            continue;
        }
        if let Some(select) = defs.select_defs.get(&current) {
            let branch = match resolve_guard_branch(active_guards, &select.condition) {
                Some(true) => Some(&select.then_expr),
                Some(false) => Some(&select.else_expr),
                None => None,
            };
            if let Some(next) = branch.and_then(match_var_name) {
                aliases.push(current.clone());
                current = next;
                if defs.loop_carried_names.contains(&current) {
                    break;
                }
                continue;
            }
        }
        break;
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
        Expr::PtrLane { .. } | Expr::Intrinsic { .. } => None,
        _ => None,
    }
}

fn loop_phi_arg(expr: &Expr) -> Option<&Expr> {
    let Expr::CallLike { func, args } = expr else {
        return None;
    };
    if func == "__loop_phi" && args.len() == 1 {
        Some(&args[0])
    } else {
        None
    }
}

fn strip_loop_phi_expr<'a>(expr: &'a Expr) -> &'a Expr {
    let mut current = expr;
    while let Some(inner) = loop_phi_arg(current) {
        current = inner;
    }
    current
}

fn extract_lane_def_expr(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::LaneExtract { .. } => Some(expr.clone()),
        Expr::PtrLane { base, lane } => Some(Expr::LaneExtract {
            value: Box::new(Expr::Raw(base.clone())),
            lane: *lane,
        }),
        Expr::ConstMemSymbol(text) | Expr::Raw(text) | Expr::Reg(text) => {
            PointerLane::parse_named(text).map(|(base, lane)| Expr::LaneExtract {
                value: Box::new(Expr::Raw(base)),
                lane,
            })
        }
        _ => {
            let stripped = strip_loop_phi_expr(expr);
            match stripped {
                Expr::LaneExtract { .. } => Some(stripped.clone()),
                Expr::PtrLane { base, lane } => Some(Expr::LaneExtract {
                    value: Box::new(Expr::Raw(base.clone())),
                    lane: *lane,
                }),
                Expr::ConstMemSymbol(text) | Expr::Raw(text) | Expr::Reg(text) => {
                    PointerLane::parse_named(text).map(|(base, lane)| Expr::LaneExtract {
                        value: Box::new(Expr::Raw(base)),
                        lane,
                    })
                }
                _ => explicit_wide_lane_extract(stripped).map(|(value, lane)| Expr::LaneExtract {
                    value: Box::new(value),
                    lane,
                }),
            }
        }
    }
}

fn extract_lane_def_expr_with_defs(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    extract_lane_def_expr(expr)
        .or_else(|| {
            match_typed_explicit_lane_extract(expr, defs).map(|(value, lane)| Expr::LaneExtract {
                value: Box::new(value),
                lane,
            })
        })
        .or_else(|| {
            resolve_typed_lane_source(expr, defs).map(|source| Expr::LaneExtract {
                value: Box::new(source.value),
                lane: source.lane,
            })
        })
}

fn explicit_wide_lane_extract(expr: &Expr) -> Option<(Expr, PointerLane)> {
    if let Some(value) = explicit_lo_lane_value(expr) {
        return Some((value, PointerLane::Lo32));
    }
    explicit_hi_lane_value(expr).map(|value| (value, PointerLane::Hi32))
}

fn explicit_lo_lane_value(expr: &Expr) -> Option<Expr> {
    let Expr::Cast { ty, expr } = expr else {
        return None;
    };
    if ty != "uint32_t" {
        return None;
    }
    explicit_wide_value_candidate(expr)
}

fn explicit_hi_lane_value(expr: &Expr) -> Option<Expr> {
    let Expr::Cast { ty, expr } = expr else {
        return None;
    };
    if ty != "uint32_t" {
        return None;
    }
    let Expr::Binary { op, lhs, rhs } = expr.as_ref() else {
        return None;
    };
    if op != ">>" || expr_imm_as_u32(rhs) != Some(32) {
        return None;
    }
    explicit_wide_value_candidate(lhs)
}

fn strip_wide_casts<'a>(expr: &'a Expr) -> &'a Expr {
    let mut current = expr;
    while let Expr::Cast { ty, expr } = current {
        if !matches!(
            ty.as_str(),
            "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t"
        ) && !ty.trim_end().ends_with('*')
        {
            break;
        }
        current = expr;
    }
    current
}

fn pointer_base_candidate(expr: &Expr) -> Option<Expr> {
    if let Expr::Cast { ty, expr } = expr {
        // Semantic lift often materializes byte-address arithmetic as
        // `((uint8_t*)arg_ptr) + off`; for wide-address recovery the pointee
        // cast is not semantically relevant, so treat it as transparent here.
        if ty.trim_end().ends_with('*') {
            return pointer_base_candidate(expr);
        }
    }
    match strip_wide_casts(expr) {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text)
            if strip_ptr_suffix(text, ".lo32", "_lo32").is_some()
                || strip_ptr_suffix(text, ".hi32", "_hi32").is_some()
                || text.ends_with("_ptr") =>
        {
            Some(strip_wide_casts(expr).clone())
        }
        _ => None,
    }
}

fn strip_offset_like_casts(expr: &Expr) -> Expr {
    let mut current = expr;
    while let Expr::Cast { ty, expr } = current {
        if expr_is_symbolic_name_pointer_base(expr) {
            break;
        }
        if !matches!(
            ty.as_str(),
            "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t"
        ) {
            break;
        }
        current = expr;
    }
    current.clone()
}

fn match_base_relative_wide_ptr_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    let expr = strip_wide_casts(expr);
    match expr {
        Expr::WidePtr { base, offset } => {
            return Some(((**base).clone(), (**offset).clone()));
        }
        _ => {
            if let Some(base) = pointer_base_candidate(expr) {
                return Some((base, Expr::Imm("0".to_string())));
            }
        }
    }

    let Expr::Binary { op, lhs, rhs } = expr else {
        return None;
    };
    if op != "+" && op != "-" {
        return None;
    }

    if let Some((base, offset)) = match_base_relative_wide_ptr_expr(lhs) {
        return Some((
            base,
            combine_offset_expr(offset, strip_offset_like_casts(rhs), op),
        ));
    }
    if op == "+" {
        if let Some((base, offset)) = match_base_relative_wide_ptr_expr(rhs) {
            return Some((
                base,
                combine_offset_expr(offset, strip_offset_like_casts(lhs), "+"),
            ));
        }
    }
    if let Some(base) = pointer_base_candidate(lhs) {
        return Some((base, strip_offset_like_casts(rhs)));
    }
    if op == "+" {
        if let Some(base) = pointer_base_candidate(rhs) {
            return Some((base, strip_offset_like_casts(lhs)));
        }
    }
    None
}

fn explicit_wide_value_candidate(expr: &Expr) -> Option<Expr> {
    let (base, offset) = match_base_relative_wide_ptr_expr(expr)?;
    Some(simplify_wide_ptr_expr(base, offset))
}

fn match_var_name(expr: &Expr) -> Option<String> {
    if let Expr::PtrLane { base, lane } = expr {
        return Some(pointer_lane_name(base, *lane));
    }
    let text = expr_atom_text(expr)?;
    if is_symbolic_name(text) {
        Some(text.to_string())
    } else {
        None
    }
}

fn strip_ptr_suffix(name: &str, dot_suffix: &str, underscore_suffix: &str) -> Option<String> {
    if let Some(base) = name.strip_suffix(dot_suffix) {
        return Some(base.to_string());
    }
    if let Some(base) = name.strip_suffix(underscore_suffix) {
        return Some(base.to_string());
    }
    if let Some((base, tail)) = name.split_once(dot_suffix) {
        if tail.starts_with('_') && tail[1..].chars().all(|c| c.is_ascii_digit()) {
            return Some(base.to_string());
        }
    }
    if let Some((base, tail)) = name.split_once(underscore_suffix) {
        if tail.starts_with('_') && tail[1..].chars().all(|c| c.is_ascii_digit()) {
            return Some(base.to_string());
        }
    }
    None
}

fn parse_abi_word_name(name: &str) -> Option<(&str, u32)> {
    if let Some(rest) = name.strip_prefix("arg") {
        return rest.parse::<u32>().ok().map(|idx| ("arg", idx));
    }
    if let Some(rest) = name.strip_prefix("param_") {
        return rest.parse::<u32>().ok().map(|idx| ("param_", idx));
    }
    None
}

fn paired_pointer_base(lo_name: &str, hi_name: &str) -> Option<String> {
    if let (Some(base_lo), Some(base_hi)) = (
        strip_ptr_suffix(lo_name, ".lo32", "_lo32"),
        strip_ptr_suffix(hi_name, ".hi32", "_hi32"),
    ) {
        if base_lo == base_hi {
            return Some(base_lo);
        }
    }

    let (lo_prefix, lo_idx) = parse_abi_word_name(lo_name)?;
    let (hi_prefix, hi_idx) = parse_abi_word_name(hi_name)?;
    if lo_prefix == hi_prefix && lo_idx % 2 == 0 && hi_idx == lo_idx + 1 {
        return Some(match lo_prefix {
            "arg" => format!("arg{}_ptr", lo_idx),
            "param_" => format!("param_{}_ptr", lo_idx),
            _ => return None,
        });
    }
    None
}

fn expr_imm_as_u32(expr: &Expr) -> Option<u32> {
    expr_atom_text(expr)?.parse::<u32>().ok()
}

fn scale_addr_offset_expr(offset: &Expr, shift: &Expr) -> Option<Expr> {
    if expr_is_zero(shift) {
        return Some(offset.clone());
    }
    if let Some(bits) = expr_imm_as_u32(shift) {
        if (1..=4).contains(&bits) {
            return Some(Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(offset.clone()),
                rhs: Box::new(Expr::Imm((1u64 << bits).to_string())),
            });
        }
    }
    Some(Expr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(offset.clone()),
        rhs: Box::new(shift.clone()),
    })
}

fn render_folded_addr64_base(base: Expr, offset: &Expr) -> Expr {
    Expr::WidePtr {
        base: Box::new(base),
        offset: Box::new(offset.clone()),
    }
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
            candidate_assign_name(stmt)
                .map(|name| !candidates.contains(&name) || used.contains(&name))
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

fn candidate_assign_name(stmt: &Stmt) -> Option<String> {
    direct_assign_name(stmt).or_else(|| guarded_assign_name(stmt))
}

fn direct_assign_name(stmt: &Stmt) -> Option<String> {
    let Stmt::Assign {
        dst: LValue::Var(name),
        ..
    } = stmt
    else {
        return None;
    };
    Some(name.clone())
}

fn guarded_assign_name(stmt: &Stmt) -> Option<String> {
    let Stmt::If {
        then_branch,
        else_branch: None,
        ..
    } = stmt
    else {
        return None;
    };
    single_direct_var_assign(then_branch.as_ref()).map(|(name, _)| name)
}

fn collect_stmt_used_vars(stmt: &Stmt, used: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_stmt_used_vars(stmt, used);
            }
        }
        Stmt::Label { body, .. } => collect_stmt_used_vars(body, used),
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

fn is_symbolic_name(text: &str) -> bool {
    is_atomic_name(text)
        && matches!(text.chars().next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
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
        Stmt::Label { body, .. } => collect_assigned_vars(body, assigned),
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
        Stmt::Assign { dst, .. } => {
            if let Some(name) = lvalue_symbol_name(dst) {
                assigned.insert(name);
            }
        }
        Stmt::Break
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
        } => simplify_if(
            simplify_expr(condition),
            simplify_stmt(*then_branch),
            else_branch.map(|stmt| simplify_stmt(*stmt)),
        ),
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(simplify_expr),
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
                discriminant: discriminant.map(simplify_expr),
                cases,
                default,
            }
        }
        Stmt::Assign { dst, src } => simplify_assign(dst, simplify_expr(src)),
        Stmt::ExprStmt(expr) => {
            let expr = simplify_expr(expr);
            if is_runtime_slowpath_expr(&expr) {
                Stmt::Empty
            } else {
                Stmt::ExprStmt(expr)
            }
        }
        Stmt::Return(expr) => Stmt::Return(expr.map(simplify_expr)),
        other => other,
    }
}

fn is_runtime_slowpath_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::CallLike { func, args } if func == "CALL.REL.NOINC" && args.is_empty()
    ) || matches!(expr, Expr::Raw(text) if is_runtime_slowpath_raw(text))
}

fn is_runtime_slowpath_raw(text: &str) -> bool {
    text.trim() == "CALL.REL.NOINC()"
}

fn is_pure_raw_expr(text: &str) -> bool {
    raw_expr_is_pure(text)
}

fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Raw(text) => fold_constant_raw_expr(&text).unwrap_or(Expr::Raw(text)),
        Expr::Unary { op, arg } => {
            let arg = simplify_expr(*arg);
            fold_constant_unary(&op, &arg).unwrap_or(Expr::Unary {
                op,
                arg: Box::new(arg),
            })
        }
        Expr::Binary { op, lhs, rhs } => {
            let lhs = simplify_expr(*lhs);
            let rhs = simplify_expr(*rhs);
            fold_constant_binary(&op, &lhs, &rhs).unwrap_or(Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(simplify_expr(*cond)),
            then_expr: Box::new(simplify_expr(*then_expr)),
            else_expr: Box::new(simplify_expr(*else_expr)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args.into_iter().map(simplify_expr).collect(),
        },
        Expr::Intrinsic { op, args } => {
            let args = args.into_iter().map(simplify_expr).collect::<Vec<_>>();
            simplify_intrinsic_expr(op, args)
        }
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(simplify_expr(*addr)),
        },
        Expr::WidePtr { base, offset } => {
            simplify_wide_ptr_expr(simplify_expr(*base), simplify_expr(*offset))
        }
        Expr::Addr64 { lo, hi } => {
            let lo = simplify_expr(*lo);
            let hi = simplify_expr(*hi);
            collapse_split_wide_addr64(&lo, &hi).unwrap_or(Expr::Addr64 {
                lo: Box::new(lo),
                hi: Box::new(hi),
            })
        }
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(simplify_expr(*value)),
            lane,
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(simplify_expr(*expr)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(simplify_expr(*base)),
            index: Box::new(simplify_expr(*index)),
        },
        other => other,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstScalar {
    Bool(bool),
    Int(i64),
}

fn expr_const_bool(expr: &Expr) -> Option<bool> {
    match const_scalar(expr)? {
        ConstScalar::Bool(value) => Some(value),
        ConstScalar::Int(value) => Some(value != 0),
    }
}

fn fold_constant_unary(op: &str, arg: &Expr) -> Option<Expr> {
    match (op, const_scalar(arg)?) {
        ("!", ConstScalar::Bool(value)) => Some(bool_expr(!value)),
        ("!", ConstScalar::Int(value)) => Some(bool_expr(value == 0)),
        ("-", ConstScalar::Int(value)) => Some(Expr::Imm((-value).to_string())),
        _ => None,
    }
}

fn fold_constant_binary(op: &str, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
    let lhs_const = const_scalar(lhs)?;
    let rhs_const = const_scalar(rhs)?;
    match (op, lhs_const, rhs_const) {
        ("&&", ConstScalar::Bool(lhs), ConstScalar::Bool(rhs)) => Some(bool_expr(lhs && rhs)),
        ("&&", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => {
            Some(bool_expr(lhs != 0 && rhs != 0))
        }
        ("||", ConstScalar::Bool(lhs), ConstScalar::Bool(rhs)) => Some(bool_expr(lhs || rhs)),
        ("||", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => {
            Some(bool_expr(lhs != 0 || rhs != 0))
        }
        ("==", lhs, rhs) => Some(bool_expr(lhs == rhs)),
        ("!=", lhs, rhs) => Some(bool_expr(lhs != rhs)),
        (">", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(bool_expr(lhs > rhs)),
        (">=", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(bool_expr(lhs >= rhs)),
        ("<", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(bool_expr(lhs < rhs)),
        ("<=", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(bool_expr(lhs <= rhs)),
        _ => None,
    }
}

fn bool_expr(value: bool) -> Expr {
    Expr::Raw(if value { "true" } else { "false" }.to_string())
}

fn const_scalar(expr: &Expr) -> Option<ConstScalar> {
    match strip_const_casts(expr) {
        Expr::Raw(text) | Expr::Builtin(text) | Expr::Imm(text) => raw_const_scalar(text),
        Expr::Unary { op, arg } if op == "-" => match const_scalar(arg)? {
            ConstScalar::Int(value) => Some(ConstScalar::Int(-value)),
            ConstScalar::Bool(_) => None,
        },
        _ => None,
    }
}

fn strip_const_casts<'a>(expr: &'a Expr) -> &'a Expr {
    let mut current = expr;
    while let Expr::Cast { ty, expr } = current {
        if matches!(
            ty.as_str(),
            "bool" | "int32_t" | "uint32_t" | "int64_t" | "uint64_t" | "intptr_t" | "uintptr_t"
        ) {
            current = expr;
        } else {
            break;
        }
    }
    current
}

fn fold_constant_raw_expr(text: &str) -> Option<Expr> {
    match raw_const_scalar(text)? {
        ConstScalar::Bool(value) => Some(bool_expr(value)),
        ConstScalar::Int(value) => Some(Expr::Imm(value.to_string())),
    }
}

fn raw_expr_is_pure(text: &str) -> bool {
    let trimmed = strip_raw_expr_wrappers(text.trim());
    if trimmed.is_empty() {
        return false;
    }
    if raw_const_scalar(trimmed).is_some() || is_symbolic_name(trimmed) {
        return true;
    }
    if let Some(inner) = raw_unary_operand(trimmed, "!") {
        return raw_expr_is_pure(inner);
    }
    if let Some(inner) = raw_unary_operand(trimmed, "-") {
        return raw_expr_is_pure(inner);
    }
    if let Some((func, args)) = parse_raw_call(trimmed) {
        return is_pure_calllike(func) && args.iter().all(|arg| raw_expr_is_pure(arg));
    }
    raw_binary_parts(trimmed)
        .map(|(lhs, _, rhs)| raw_expr_is_pure(lhs) && raw_expr_is_pure(rhs))
        .unwrap_or(false)
}

fn raw_const_scalar(text: &str) -> Option<ConstScalar> {
    let trimmed = strip_raw_expr_wrappers(text.trim());
    if trimmed.is_empty() {
        return None;
    }
    match trimmed {
        "true" => return Some(ConstScalar::Bool(true)),
        "false" => return Some(ConstScalar::Bool(false)),
        _ => {}
    }
    if let Ok(value) = trimmed.parse::<i64>() {
        return Some(ConstScalar::Int(value));
    }
    if let Some(inner) = raw_unary_operand(trimmed, "!") {
        return match raw_const_scalar(inner)? {
            ConstScalar::Bool(value) => Some(ConstScalar::Bool(!value)),
            ConstScalar::Int(value) => Some(ConstScalar::Bool(value == 0)),
        };
    }
    if let Some(inner) = raw_unary_operand(trimmed, "-") {
        return match raw_const_scalar(inner)? {
            ConstScalar::Int(value) => Some(ConstScalar::Int(-value)),
            ConstScalar::Bool(_) => None,
        };
    }
    let (lhs, op, rhs) = raw_binary_parts(trimmed)?;
    let lhs = raw_const_scalar(lhs)?;
    let rhs = raw_const_scalar(rhs)?;
    match (op, lhs, rhs) {
        ("&&", ConstScalar::Bool(lhs), ConstScalar::Bool(rhs)) => {
            Some(ConstScalar::Bool(lhs && rhs))
        }
        ("&&", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => {
            Some(ConstScalar::Bool(lhs != 0 && rhs != 0))
        }
        ("||", ConstScalar::Bool(lhs), ConstScalar::Bool(rhs)) => {
            Some(ConstScalar::Bool(lhs || rhs))
        }
        ("||", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => {
            Some(ConstScalar::Bool(lhs != 0 || rhs != 0))
        }
        ("==", lhs, rhs) => Some(ConstScalar::Bool(lhs == rhs)),
        ("!=", lhs, rhs) => Some(ConstScalar::Bool(lhs != rhs)),
        (">", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(ConstScalar::Bool(lhs > rhs)),
        (">=", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(ConstScalar::Bool(lhs >= rhs)),
        ("<", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(ConstScalar::Bool(lhs < rhs)),
        ("<=", ConstScalar::Int(lhs), ConstScalar::Int(rhs)) => Some(ConstScalar::Bool(lhs <= rhs)),
        _ => None,
    }
}

fn strip_raw_expr_wrappers<'a>(text: &'a str) -> &'a str {
    let mut current = text.trim();
    loop {
        let paren_stripped = strip_balanced_outer_parens(current);
        if paren_stripped != current {
            current = paren_stripped.trim();
            continue;
        }
        let cast_stripped = strip_raw_const_cast(current);
        if cast_stripped != current {
            current = cast_stripped.trim();
            continue;
        }
        return current;
    }
}

fn strip_balanced_outer_parens(text: &str) -> &str {
    let bytes = text.as_bytes();
    if bytes.first() != Some(&b'(') || bytes.last() != Some(&b')') {
        return text;
    }
    let mut depth = 0usize;
    for (idx, byte) in bytes.iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 && idx + 1 != bytes.len() {
                    return text;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        text[1..text.len() - 1].trim()
    } else {
        text
    }
}

fn strip_raw_const_cast<'a>(text: &'a str) -> &'a str {
    const CAST_TYPES: &[&str] = &[
        "bool",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "intptr_t",
        "uintptr_t",
    ];
    for ty in CAST_TYPES {
        let prefix = format!("({})", ty);
        let Some(rest) = text.strip_prefix(&prefix) else {
            continue;
        };
        let rest = rest.trim_start();
        if rest.starts_with('(') && rest.ends_with(')') {
            let inner = strip_balanced_outer_parens(rest);
            if inner != rest {
                return inner;
            }
        }
    }
    text
}

fn raw_unary_operand<'a>(text: &'a str, op: &str) -> Option<&'a str> {
    let remainder = text.strip_prefix(op)?.trim_start();
    if remainder.is_empty() {
        return None;
    }
    if raw_binary_parts(remainder).is_some() {
        return None;
    }
    Some(remainder)
}

fn raw_binary_parts<'a>(text: &'a str) -> Option<(&'a str, &'static str, &'a str)> {
    const OPS: &[&str] = &["||", "&&", "==", "!=", ">=", "<=", ">", "<"];
    for op in OPS {
        if let Some((lhs, rhs)) = split_raw_top_level_binary(text, op) {
            return Some((lhs, *op, rhs));
        }
    }
    None
}

fn split_raw_top_level_binary<'a>(text: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let bytes = text.as_bytes();
    let needle = op.as_bytes();
    let mut depth = 0usize;
    let mut idx = 0usize;
    let mut found = None;
    while idx + needle.len() <= bytes.len() {
        match bytes[idx] {
            b'(' => {
                depth += 1;
                idx += 1;
                continue;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                idx += 1;
                continue;
            }
            _ => {}
        }
        if depth == 0 && &bytes[idx..idx + needle.len()] == needle {
            if matches!(op, "+" | "-") && raw_binary_op_is_unary_position(text, idx) {
                idx += 1;
                continue;
            }
            let lhs = text[..idx].trim();
            let rhs = text[idx + needle.len()..].trim();
            if !lhs.is_empty() && !rhs.is_empty() {
                found = Some((lhs, rhs));
            }
        }
        idx += 1;
    }
    found
}

fn raw_binary_op_is_unary_position(text: &str, idx: usize) -> bool {
    let prefix = text[..idx].trim_end();
    if prefix.is_empty() {
        return true;
    }
    matches!(
        prefix.chars().next_back(),
        Some(
            '(' | '['
                | '{'
                | ','
                | '?'
                | ':'
                | '+'
                | '-'
                | '*'
                | '/'
                | '%'
                | '&'
                | '|'
                | '^'
                | '!'
                | '='
                | '<'
                | '>'
        )
    )
}

fn parse_raw_call<'a>(text: &'a str) -> Option<(&'a str, Vec<&'a str>)> {
    let open = text.find('(')?;
    let func = text[..open].trim();
    if !is_atomic_name(func) || !text.ends_with(')') {
        return None;
    }
    let body = &text[open + 1..text.len() - 1];
    let mut args = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, ch) in body.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                args.push(body[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    let trailing = body[start..].trim();
    if !trailing.is_empty() {
        args.push(trailing);
    }
    Some((func, args))
}

fn normalize_match_expr(expr: &Expr, defs: &HashMap<String, Expr>, depth: usize) -> Expr {
    normalize_match_expr_inner(expr.clone(), defs, depth)
}

fn normalize_match_expr_inner(expr: Expr, defs: &HashMap<String, Expr>, depth: usize) -> Expr {
    if depth == 0 {
        return expr;
    }
    if let Some(name) = match_var_name(&expr) {
        if let Some(mapped) = defs.get(&name) {
            return normalize_match_expr_inner(mapped.clone(), defs, depth - 1);
        }
    }
    match expr {
        Expr::Reg(text) if !is_symbolic_name(&text) => {
            if let Some(parsed) = parse_raw_match_expr(text.trim()) {
                return normalize_match_expr_inner(parsed, defs, depth - 1);
            }
            Expr::Reg(text)
        }
        Expr::Raw(text) => {
            if let Some(parsed) = parse_raw_match_expr(text.trim()) {
                if parsed != Expr::Raw(text.clone()) {
                    return normalize_match_expr_inner(parsed, defs, depth - 1);
                }
            }
            Expr::Raw(text)
        }
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(normalize_match_expr_inner(*arg, defs, depth - 1)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(normalize_match_expr_inner(*lhs, defs, depth - 1)),
            rhs: Box::new(normalize_match_expr_inner(*rhs, defs, depth - 1)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(normalize_match_expr_inner(*cond, defs, depth - 1)),
            then_expr: Box::new(normalize_match_expr_inner(*then_expr, defs, depth - 1)),
            else_expr: Box::new(normalize_match_expr_inner(*else_expr, defs, depth - 1)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args
                .into_iter()
                .map(|arg| normalize_match_expr_inner(arg, defs, depth - 1))
                .collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args
                .into_iter()
                .map(|arg| normalize_match_expr_inner(arg, defs, depth - 1))
                .collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(normalize_match_expr_inner(*addr, defs, depth - 1)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(normalize_match_expr_inner(*base, defs, depth - 1)),
            offset: Box::new(normalize_match_expr_inner(*offset, defs, depth - 1)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(normalize_match_expr_inner(*lo, defs, depth - 1)),
            hi: Box::new(normalize_match_expr_inner(*hi, defs, depth - 1)),
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(normalize_match_expr_inner(*expr, defs, depth - 1)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(normalize_match_expr_inner(*base, defs, depth - 1)),
            index: Box::new(normalize_match_expr_inner(*index, defs, depth - 1)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(normalize_match_expr_inner(*value, defs, depth - 1)),
            lane,
        },
        other => other,
    }
}

fn parse_raw_match_expr(text: &str) -> Option<Expr> {
    let text = strip_balanced_outer_parens(text.trim()).trim();
    if text.is_empty() {
        return None;
    }
    for ops in [
        &["||"][..],
        &["&&"][..],
        &["==", "!=", ">=", "<=", ">", "<"][..],
        &["+", "-"][..],
        &["*", "/"][..],
    ] {
        if let Some((lhs, op, rhs)) = split_raw_top_level_binary_ops(text, ops) {
            return Some(Expr::Binary {
                op: op.to_string(),
                lhs: Box::new(
                    parse_raw_match_expr(lhs).unwrap_or_else(|| Expr::Raw(lhs.to_string())),
                ),
                rhs: Box::new(
                    parse_raw_match_expr(rhs).unwrap_or_else(|| Expr::Raw(rhs.to_string())),
                ),
            });
        }
    }
    if let Some(inner) = raw_unary_operand(text, "!") {
        return Some(Expr::Unary {
            op: "!".to_string(),
            arg: Box::new(
                parse_raw_match_expr(inner).unwrap_or_else(|| Expr::Raw(inner.to_string())),
            ),
        });
    }
    if let Some(inner) = raw_unary_operand(text, "-") {
        return Some(Expr::Unary {
            op: "-".to_string(),
            arg: Box::new(
                parse_raw_match_expr(inner).unwrap_or_else(|| Expr::Raw(inner.to_string())),
            ),
        });
    }
    if let Some((ty, rest)) = parse_raw_leading_cast(text) {
        let inner = parse_raw_match_expr(rest).unwrap_or_else(|| Expr::Raw(rest.to_string()));
        return Some(Expr::Cast {
            ty: ty.to_string(),
            expr: Box::new(inner),
        });
    }
    if let Some((func, args)) = parse_raw_call(text) {
        return Some(Expr::CallLike {
            func: func.to_string(),
            args: args
                .into_iter()
                .map(|arg| parse_raw_match_expr(arg).unwrap_or_else(|| Expr::Raw(arg.to_string())))
                .collect(),
        });
    }
    if matches!(text, "true" | "false") {
        return Some(Expr::Raw(text.to_string()));
    }
    if text.parse::<i64>().is_ok() {
        return Some(Expr::Imm(text.to_string()));
    }
    if text.parse::<f64>().is_ok() {
        return Some(Expr::Raw(text.to_string()));
    }
    if is_symbolic_name(text) {
        return Some(Expr::Reg(text.to_string()));
    }
    None
}

fn split_raw_top_level_binary_ops<'a>(
    text: &'a str,
    ops: &[&'static str],
) -> Option<(&'a str, &'static str, &'a str)> {
    for op in ops {
        if let Some((lhs, rhs)) = split_raw_top_level_binary(text, op) {
            return Some((lhs, *op, rhs));
        }
    }
    None
}

fn parse_raw_leading_cast<'a>(text: &'a str) -> Option<(&'a str, &'a str)> {
    let trimmed = text.trim();
    let rest = trimmed.strip_prefix('(')?;
    let close = rest.find(')')?;
    let ty = rest[..close].trim();
    if !is_cast_type_name(ty) {
        return None;
    }
    let rest = rest[close + 1..].trim_start();
    if rest.is_empty() {
        return None;
    }
    Some((ty, rest))
}

fn is_cast_type_name(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | ' ' | '*' | ':'))
}

fn simplify_intrinsic_expr(op: IntrinsicOp, args: Vec<Expr>) -> Expr {
    if let Some(expr) = fold_clamp_intrinsic(&op, &args) {
        return expr;
    }
    Expr::Intrinsic { op, args }
}

fn fold_clamp_intrinsic(op: &IntrinsicOp, args: &[Expr]) -> Option<Expr> {
    if args.len() != 2 {
        return None;
    }
    match op {
        IntrinsicOp::Max => fold_max_min_to_clamp(&args[0], &args[1])
            .or_else(|| fold_max_min_to_clamp(&args[1], &args[0])),
        IntrinsicOp::Min => fold_min_max_to_clamp(&args[0], &args[1])
            .or_else(|| fold_min_max_to_clamp(&args[1], &args[0])),
        _ => None,
    }
}

fn fold_max_min_to_clamp(min_side: &Expr, lo: &Expr) -> Option<Expr> {
    let Expr::Intrinsic {
        op: IntrinsicOp::Min,
        args,
    } = min_side
    else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    Some(Expr::Intrinsic {
        op: IntrinsicOp::Clamp,
        args: vec![args[0].clone(), lo.clone(), args[1].clone()],
    })
}

fn fold_min_max_to_clamp(max_side: &Expr, hi: &Expr) -> Option<Expr> {
    let Expr::Intrinsic {
        op: IntrinsicOp::Max,
        args,
    } = max_side
    else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    Some(Expr::Intrinsic {
        op: IntrinsicOp::Clamp,
        args: vec![args[0].clone(), args[1].clone(), hi.clone()],
    })
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
    let out = recover_rcp_division_slowpaths(out);
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
    let out = recover_rcp_division_slowpaths(out);
    match out.len() {
        0 => Stmt::Empty,
        _ => Stmt::Block(out),
    }
}

fn recover_rcp_division_slowpaths(stmts: Vec<Stmt>) -> Vec<Stmt> {
    let mut defs = HashMap::<String, Expr>::new();
    let mut out = Vec::with_capacity(stmts.len());
    let mut future_used: Vec<BTreeSet<String>> = vec![BTreeSet::new(); stmts.len()];
    let mut live_tail = BTreeSet::new();
    for (idx, stmt) in stmts.iter().enumerate().rev() {
        future_used[idx] = live_tail.clone();
        collect_stmt_used_vars(stmt, &mut live_tail);
    }
    for (idx, stmt) in stmts.into_iter().enumerate() {
        out.push(recover_rcp_division_stmt_tree(
            stmt,
            &mut defs,
            &future_used[idx],
        ));
    }
    let cleaned = prune_dead_runtime_helper_temps_stmt(Stmt::Sequence(out), BTreeSet::new()).0;
    let cleaned = dedup_consecutive_pure_assigns(cleaned);
    match cleaned {
        Stmt::Sequence(stmts) => stmts,
        Stmt::Empty => Vec::new(),
        other => vec![other],
    }
}

fn dedup_consecutive_pure_assigns(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => rebuild_stmt_list(
            stmts
                .into_iter()
                .map(dedup_consecutive_pure_assigns)
                .collect(),
            true,
        ),
        Stmt::Block(stmts) => rebuild_stmt_list(
            stmts
                .into_iter()
                .map(dedup_consecutive_pure_assigns)
                .collect(),
            false,
        ),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(dedup_consecutive_pure_assigns(*body)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: Box::new(dedup_consecutive_pure_assigns(*then_branch)),
            else_branch: else_branch
                .map(|branch| Box::new(dedup_consecutive_pure_assigns(*branch))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(dedup_consecutive_pure_assigns(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, dedup_consecutive_pure_assigns(body)))
                .collect(),
            default: default.map(|body| Box::new(dedup_consecutive_pure_assigns(*body))),
        },
        other => other,
    }
}

fn rebuild_stmt_list(stmts: Vec<Stmt>, is_sequence: bool) -> Stmt {
    let mut out = Vec::with_capacity(stmts.len());
    let mut available_assigns: HashMap<String, (Expr, BTreeSet<String>)> = HashMap::new();
    for stmt in stmts {
        let stmt = match stmt {
            Stmt::Empty => continue,
            other => other,
        };
        match &stmt {
            Stmt::Assign {
                dst: LValue::Var(name),
                src,
            } if expr_is_pure_for_dce(src) => {
                let redundant = available_assigns
                    .get(name)
                    .is_some_and(|(prev_src, _)| prev_src == src);
                invalidate_linear_assign_cache(&mut available_assigns, name);
                if redundant {
                    continue;
                }
                let mut deps = BTreeSet::new();
                collect_used_expr_vars(src, &mut deps);
                available_assigns.insert(name.clone(), (src.clone(), deps));
            }
            Stmt::Assign {
                dst: LValue::Var(name),
                ..
            } => {
                invalidate_linear_assign_cache(&mut available_assigns, name);
                available_assigns.clear();
            }
            _ => {
                available_assigns.clear();
            }
        }
        out.push(stmt);
    }

    match (is_sequence, out.len()) {
        (_, 0) => Stmt::Empty,
        (true, 1) => out.into_iter().next().unwrap_or(Stmt::Empty),
        (true, _) => Stmt::Sequence(out),
        (false, _) => Stmt::Block(out),
    }
}

fn invalidate_linear_assign_cache(
    cache: &mut HashMap<String, (Expr, BTreeSet<String>)>,
    assigned_name: &str,
) {
    cache.retain(|name, (_, deps)| name != assigned_name && !deps.contains(assigned_name));
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
            let mut out = Vec::with_capacity(stmts.len());
            let mut local_future: Vec<BTreeSet<String>> = vec![BTreeSet::new(); stmts.len()];
            let mut live_tail = future_used.clone();
            for (idx, stmt) in stmts.iter().enumerate().rev() {
                local_future[idx] = live_tail.clone();
                collect_stmt_used_vars(stmt, &mut live_tail);
            }
            for (idx, stmt) in stmts.into_iter().enumerate() {
                out.push(recover_rcp_division_stmt_tree(
                    stmt,
                    &mut local_defs,
                    &local_future[idx],
                ));
            }
            let rewritten =
                prune_dead_runtime_helper_temps_stmt(Stmt::Sequence(out), future_used.clone()).0;
            let mut final_defs = entry_defs;
            update_linear_defs(&rewritten, &mut final_defs);
            *defs = final_defs;
            rewritten
        }
        Stmt::Block(stmts) => {
            let entry_defs = defs.clone();
            let mut local_defs = defs.clone();
            let mut out = Vec::with_capacity(stmts.len());
            let mut local_future: Vec<BTreeSet<String>> = vec![BTreeSet::new(); stmts.len()];
            let mut live_tail = future_used.clone();
            for (idx, stmt) in stmts.iter().enumerate().rev() {
                local_future[idx] = live_tail.clone();
                collect_stmt_used_vars(stmt, &mut live_tail);
            }
            for (idx, stmt) in stmts.into_iter().enumerate() {
                out.push(recover_rcp_division_stmt_tree(
                    stmt,
                    &mut local_defs,
                    &local_future[idx],
                ));
            }
            let rewritten =
                prune_dead_runtime_helper_temps_stmt(Stmt::Block(out), future_used.clone()).0;
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

fn recover_fchk_division_assign(stmt: &Stmt, defs: &HashMap<String, Expr>) -> Option<Stmt> {
    let Stmt::Assign { dst, src } = stmt else {
        return None;
    };
    let (num, den) = match_rcp_division_expr(src, defs)?;
    if !defs.values().any(|expr| {
        let Some((fchk_num, fchk_den)) = match_fchk_expr(expr) else {
            return false;
        };
        let fchk_num = normalize_match_expr(&fchk_num, defs, 12);
        let fchk_den = normalize_match_expr(&fchk_den, defs, 12);
        let num_matches = same_match_expr(&fchk_num, &num);
        let den_matches = same_match_expr(&fchk_den, &den);
        num_matches || den_matches
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

fn recover_fchk_division_stmt(
    stmt: &Stmt,
    defs: &HashMap<String, Expr>,
    future_used: &BTreeSet<String>,
) -> Option<Stmt> {
    let debug = std::env::var("DEBUG_RCP_RECOVER").is_ok();
    let Stmt::If {
        condition,
        then_branch,
        else_branch: Some(else_branch),
    } = stmt
    else {
        return None;
    };
    if debug {
        eprintln!("recover candidate condition={}", condition.render());
    }
    let (pred_name, fast_branch, slow_branch) = match condition {
        Expr::Reg(name) => (name.clone(), else_branch.as_ref(), then_branch.as_ref()),
        Expr::Unary { op, arg } if op == "!" => (
            match_var_name(arg)?,
            then_branch.as_ref(),
            else_branch.as_ref(),
        ),
        _ => return None,
    };
    let Some(fast_assigns) = match_assign_list(fast_branch) else {
        if debug {
            eprintln!("recover rejected non-assign fast branch");
        }
        return None;
    };
    let Some(slow_assigns) = match_assign_list(slow_branch) else {
        if debug {
            eprintln!("recover rejected non-assign slow branch");
        }
        return None;
    };
    let Some(pred_expr) = defs.get(&pred_name) else {
        if let Some(recovered) = recover_division_merge_without_pred_def(&fast_assigns, defs) {
            if debug {
                eprintln!("recover accepted direct-division merge without predicate def");
            }
            return Some(recovered);
        }
        if debug {
            eprintln!("recover missing predicate def for {}", pred_name);
        }
        return None;
    };
    if debug {
        eprintln!("recover predicate {} = {}", pred_name, pred_expr.render());
        if let Some(den_name) = match_var_name(&match_fchk_expr(pred_expr)?.1) {
            eprintln!(
                "recover defs has {} = {}",
                den_name,
                defs.get(&den_name)
                    .map(Expr::render)
                    .unwrap_or_else(|| "<missing>".to_string())
            );
        }
    }
    let (fchk_num, fchk_den) = match_fchk_expr(pred_expr)?;
    let fchk_num = normalize_match_expr(&fchk_num, defs, 12);
    let fchk_den = normalize_match_expr(&fchk_den, defs, 12);
    let mut slow_by_dst = HashMap::new();
    for (dst, src) in &slow_assigns {
        slow_by_dst.insert(dst.render(), src.clone());
    }

    let cond_pred = Expr::Reg(pred_name.clone());
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
            if debug {
                eprintln!(
                    "recover compare fchk_num={} fchk_den={} num={} den={}",
                    fchk_num.render(),
                    fchk_den.render(),
                    num.render(),
                    den.render()
                );
            }
            let num_matches = same_match_expr(&fchk_num, &num);
            let den_matches = same_match_expr(&fchk_den, &den);
            if !num_matches && !den_matches {
                if debug {
                    eprintln!("recover rejected mismatched numerator and denominator");
                }
                return None;
            }
            if debug {
                if !num_matches {
                    eprintln!("recover accepted unresolved numerator alias mismatch");
                }
                if !den_matches {
                    eprintln!("recover accepted unresolved denominator alias mismatch");
                }
                eprintln!(
                    "recover rewrote {} = {}/{}",
                    dst_fast.render(),
                    fchk_num.render(),
                    fchk_den.render()
                );
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
    match expr {
        Expr::Raw(text) | Expr::Imm(text) | Expr::Reg(text) => {
            let text = text.trim();
            text.starts_with("ConstMem(") || text.starts_with("c[0x")
        }
        _ => false,
    }
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

fn prune_dead_runtime_helper_temps_stmt(
    stmt: Stmt,
    live_out: BTreeSet<String>,
) -> (Stmt, BTreeSet<String>) {
    let mut current = stmt;
    loop {
        let helper_vars = collect_runtime_helper_vars(&current);
        if helper_vars.is_empty() {
            return (current, live_out);
        }
        let mut condition_vars = BTreeSet::new();
        collect_condition_vars(&current, &mut condition_vars);
        let (next, live_before) = runtime_helper_dce_stmt(
            current.clone(),
            &helper_vars,
            &condition_vars,
            live_out.clone(),
        );
        if next == current {
            return (next, live_before);
        }
        current = next;
    }
}

fn collect_condition_vars(stmt: &Stmt, out: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_condition_vars(stmt, out);
            }
        }
        Stmt::Label { body, .. } => collect_condition_vars(body, out),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_used_expr_vars(condition, out);
            collect_condition_vars(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_condition_vars(else_branch, out);
            }
        }
        Stmt::Loop {
            condition, body, ..
        } => {
            if let Some(condition) = condition {
                collect_used_expr_vars(condition, out);
            }
            collect_condition_vars(body, out);
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            if let Some(discriminant) = discriminant {
                collect_used_expr_vars(discriminant, out);
            }
            for (_, body) in cases {
                collect_condition_vars(body, out);
            }
            if let Some(default) = default {
                collect_condition_vars(default, out);
            }
        }
        Stmt::Assign { .. }
        | Stmt::ExprStmt(_)
        | Stmt::Return(_)
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Goto(_)
        | Stmt::Empty => {}
    }
}

fn collect_runtime_helper_vars(stmt: &Stmt) -> BTreeSet<String> {
    let mut helper_vars = BTreeSet::new();
    loop {
        let snapshot = helper_vars.clone();
        collect_runtime_helper_vars_into(stmt, &mut helper_vars, false);
        if helper_vars == snapshot {
            return helper_vars;
        }
    }
}

fn collect_runtime_helper_vars_into(
    stmt: &Stmt,
    helper_vars: &mut BTreeSet<String>,
    forced_helper_context: bool,
) {
    match stmt {
        Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
            for stmt in stmts {
                collect_runtime_helper_vars_into(stmt, helper_vars, forced_helper_context);
            }
        }
        Stmt::Label { body, .. } => {
            collect_runtime_helper_vars_into(body, helper_vars, forced_helper_context)
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let branch_forced = forced_helper_context
                || expr_uses_any_helper_var(condition, helper_vars)
                || branches_look_like_parallel_assigns(then_branch, else_branch.as_deref());
            collect_runtime_helper_vars_into(then_branch, helper_vars, branch_forced);
            if let Some(else_branch) = else_branch {
                collect_runtime_helper_vars_into(else_branch, helper_vars, branch_forced);
            }
        }
        Stmt::Loop {
            condition, body, ..
        } => {
            let body_forced = forced_helper_context
                || condition
                    .as_ref()
                    .map(|condition| expr_uses_any_helper_var(condition, helper_vars))
                    .unwrap_or(false);
            collect_runtime_helper_vars_into(body, helper_vars, body_forced);
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            let branch_forced = forced_helper_context
                || discriminant
                    .as_ref()
                    .map(|condition| expr_uses_any_helper_var(condition, helper_vars))
                    .unwrap_or(false);
            for (_, body) in cases {
                collect_runtime_helper_vars_into(body, helper_vars, branch_forced);
            }
            if let Some(default) = default {
                collect_runtime_helper_vars_into(default, helper_vars, branch_forced);
            }
        }
        Stmt::Assign { dst, src } => {
            let Some(name) = lvalue_symbol_name(dst) else {
                return;
            };
            if forced_helper_context
                || expr_is_runtime_helper_seed(src)
                || expr_uses_any_helper_var(src, helper_vars)
            {
                helper_vars.insert(name);
            }
        }
        Stmt::Break
        | Stmt::Continue
        | Stmt::Return(_)
        | Stmt::ExprStmt(_)
        | Stmt::Goto(_)
        | Stmt::Empty => {}
    }
}

fn branches_look_like_parallel_assigns(then_branch: &Stmt, else_branch: Option<&Stmt>) -> bool {
    let Some(else_branch) = else_branch else {
        return false;
    };
    let Some(then_assigns) = match_assign_list(then_branch) else {
        return false;
    };
    let Some(else_assigns) = match_assign_list(else_branch) else {
        return false;
    };
    if then_assigns.is_empty() || then_assigns.len() != else_assigns.len() {
        return false;
    }
    then_assigns
        .iter()
        .zip(else_assigns.iter())
        .all(|((then_dst, _), (else_dst, _))| then_dst.render() == else_dst.render())
}

fn expr_is_runtime_helper_seed(expr: &Expr) -> bool {
    match expr {
        Expr::Intrinsic {
            op:
                IntrinsicOp::CarryU32Add3
                | IntrinsicOp::LeaHiX
                | IntrinsicOp::LeaHiXSx32
                | IntrinsicOp::PairHi,
            ..
        } => true,
        Expr::Intrinsic { args, .. } => args.iter().any(expr_is_runtime_helper_seed),
        Expr::CallLike { func, args } => {
            matches!(
                func.as_str(),
                "FCHK"
                    | "CALL.REL.NOINC"
                    | "carry_u32_add3"
                    | "lea_hi_x"
                    | "lea_hi_x_sx32"
                    | "pair_hi"
            ) || args.iter().any(expr_is_runtime_helper_seed)
        }
        Expr::Raw(text) => {
            is_runtime_slowpath_raw(text)
                || parse_raw_call(text)
                    .map(|(func, _)| {
                        matches!(
                            func,
                            "FCHK"
                                | "CALL.REL.NOINC"
                                | "carry_u32_add3"
                                | "lea_hi_x"
                                | "lea_hi_x_sx32"
                                | "pair_hi"
                        )
                    })
                    .unwrap_or(false)
        }
        Expr::Unary { arg, .. } => expr_is_runtime_helper_seed(arg),
        Expr::Binary { lhs, rhs, .. } => {
            expr_is_runtime_helper_seed(lhs) || expr_is_runtime_helper_seed(rhs)
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_is_runtime_helper_seed(cond)
                || expr_is_runtime_helper_seed(then_expr)
                || expr_is_runtime_helper_seed(else_expr)
        }
        Expr::Load { addr, .. } | Expr::Cast { expr: addr, .. } => {
            expr_is_runtime_helper_seed(addr)
        }
        Expr::WidePtr { base, offset } => {
            expr_is_runtime_helper_seed(base) || expr_is_runtime_helper_seed(offset)
        }
        Expr::Addr64 { lo, hi } => {
            expr_is_runtime_helper_seed(lo) || expr_is_runtime_helper_seed(hi)
        }
        Expr::Index { base, index } => {
            expr_is_runtime_helper_seed(base) || expr_is_runtime_helper_seed(index)
        }
        Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_) => false,
    }
}

fn expr_uses_any_helper_var(expr: &Expr, helper_vars: &BTreeSet<String>) -> bool {
    let mut used = BTreeSet::new();
    collect_used_expr_vars(expr, &mut used);
    !helper_vars.is_disjoint(&used)
}

fn runtime_helper_dce_stmt(
    stmt: Stmt,
    helper_vars: &BTreeSet<String>,
    condition_vars: &BTreeSet<String>,
    live_out: BTreeSet<String>,
) -> (Stmt, BTreeSet<String>) {
    match stmt {
        Stmt::Sequence(stmts) => {
            runtime_helper_dce_sequence(stmts, false, helper_vars, condition_vars, live_out)
        }
        Stmt::Block(stmts) => {
            runtime_helper_dce_sequence(stmts, true, helper_vars, condition_vars, live_out)
        }
        Stmt::Label { name, body } => {
            let (body, live_before) =
                runtime_helper_dce_stmt(*body, helper_vars, condition_vars, live_out);
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
            let (then_branch, then_live) = runtime_helper_dce_stmt(
                *then_branch,
                helper_vars,
                condition_vars,
                live_out.clone(),
            );
            let (else_branch, else_live) = if let Some(else_branch) = else_branch {
                let (stmt, live) =
                    runtime_helper_dce_stmt(*else_branch, helper_vars, condition_vars, live_out);
                (Some(Box::new(stmt)), live)
            } else {
                (None, live_out)
            };
            let mut live_before = then_live;
            live_before.extend(else_live);
            collect_used_expr_vars(&condition, &mut live_before);
            let rewritten = simplify_if(condition, then_branch, else_branch.map(|branch| *branch));
            (rewritten, live_before)
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
                let (_, body_live_before) = runtime_helper_dce_stmt(
                    original_body.clone(),
                    helper_vars,
                    condition_vars,
                    body_live_out.clone(),
                );
                let mut next_live_out = live_out.clone();
                next_live_out.extend(cond_live.clone());
                next_live_out.extend(body_live_before);
                if next_live_out == body_live_out {
                    break;
                }
                body_live_out = next_live_out;
            }

            let (body, body_live_before) =
                runtime_helper_dce_stmt(original_body, helper_vars, condition_vars, body_live_out);
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
                    let (body, case_live) = runtime_helper_dce_stmt(
                        body,
                        helper_vars,
                        condition_vars,
                        live_out.clone(),
                    );
                    live_before.extend(case_live);
                    (label, body)
                })
                .collect();
            let default = default.map(|body| {
                let (body, default_live) =
                    runtime_helper_dce_stmt(*body, helper_vars, condition_vars, live_out.clone());
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
        Stmt::Assign { dst, src } => {
            runtime_helper_dce_assign(dst, src, helper_vars, condition_vars, live_out)
        }
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

fn runtime_helper_dce_sequence(
    stmts: Vec<Stmt>,
    as_block: bool,
    helper_vars: &BTreeSet<String>,
    condition_vars: &BTreeSet<String>,
    live_out: BTreeSet<String>,
) -> (Stmt, BTreeSet<String>) {
    let mut live = live_out;
    let mut kept = Vec::new();

    for stmt in stmts.into_iter().rev() {
        let (stmt, live_before) = runtime_helper_dce_stmt(stmt, helper_vars, condition_vars, live);
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

fn runtime_helper_dce_assign(
    dst: LValue,
    src: Expr,
    helper_vars: &BTreeSet<String>,
    condition_vars: &BTreeSet<String>,
    mut live_out: BTreeSet<String>,
) -> (Stmt, BTreeSet<String>) {
    let debug = std::env::var("DEBUG_HELPER_DCE").is_ok();
    if let Some(name) = lvalue_symbol_name(&dst) {
        let src_is_safe_helper = expr_is_pure_for_dce(&src) || expr_is_runtime_helper_seed(&src);
        if helper_vars.contains(&name)
            && !condition_vars.contains(&name)
            && !live_out.contains(&name)
            && src_is_safe_helper
        {
            if debug {
                eprintln!("helper-dce drop {}", name);
            }
            return (Stmt::Empty, live_out);
        }
        if debug && helper_vars.contains(&name) {
            eprintln!(
                "helper-dce keep {} live={} cond={} pure={}",
                name,
                live_out.contains(&name),
                condition_vars.contains(&name),
                src_is_safe_helper
            );
        }
        live_out.remove(&name);
        collect_used_expr_vars(&src, &mut live_out);
        return (Stmt::Assign { dst, src }, live_out);
    }

    collect_used_expr_vars(&src, &mut live_out);
    collect_used_lvalue_vars(&dst, &mut live_out);
    (Stmt::Assign { dst, src }, live_out)
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

fn match_fchk_expr(expr: &Expr) -> Option<(Expr, Expr)> {
    match expr {
        Expr::CallLike { func, args } if func == "FCHK" && args.len() == 2 => {
            Some((args[0].clone(), args[1].clone()))
        }
        Expr::Raw(text) => {
            let (func, args) = parse_raw_call(text)?;
            if func != "FCHK" || args.len() != 2 {
                return None;
            }
            Some((
                Expr::Raw(args[0].to_string()),
                Expr::Raw(args[1].to_string()),
            ))
        }
        _ => None,
    }
}

fn match_rcp_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, 12);
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    match_rcp_division_parts(&lhs, &rhs, defs)
        .or_else(|| match_rcp_division_parts(&rhs, &lhs, defs))
}

fn match_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    match_direct_division_expr(expr, defs).or_else(|| match_rcp_division_expr(expr, defs))
}

fn match_direct_division_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, 12);
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "/" => Some((
            normalize_match_expr(&lhs, defs, 12),
            normalize_match_expr(&rhs, defs, 12),
        )),
        Expr::Raw(text) => {
            let text = strip_raw_expr_wrappers(text.trim());
            let (lhs, rhs) = split_raw_top_level_binary(text, "/")?;
            Some((
                normalize_match_expr(&Expr::Raw(lhs.to_string()), defs, 12),
                normalize_match_expr(&Expr::Raw(rhs.to_string()), defs, 12),
            ))
        }
        _ => None,
    }
}

fn match_rcp_division_parts(
    scaled_expr: &Expr,
    refined_term: &Expr,
    defs: &HashMap<String, Expr>,
) -> Option<(Expr, Expr)> {
    let debug = std::env::var("DEBUG_RCP_RECOVER").is_ok();
    let scaled_resolved = normalize_match_expr(scaled_expr, defs, 12);
    let scaled_pairs = match_mul_factor_pairs(&scaled_resolved, defs)?;
    let refined_pairs = match_mul_factor_pairs(refined_term, defs)?;
    if debug {
        eprintln!(
            "recover parts scaled_pairs={:?} refined_pairs={:?}",
            scaled_pairs, refined_pairs
        );
    }
    for (recip_a, num) in &scaled_pairs {
        for (recip_b, corr) in &refined_pairs {
            if !same_rendered_expr(recip_a, recip_b) {
                continue;
            }
            let Some(den) = match_rcp_refine_expr(recip_a, defs) else {
                if debug {
                    eprintln!("recover parts failed refine recip={}", recip_a.render());
                }
                continue;
            };
            let Some((corr_den, corr_scaled, corr_num)) = match_newton_correction_expr(corr, defs)
            else {
                if debug {
                    eprintln!("recover parts failed correction corr={}", corr.render());
                }
                continue;
            };
            if same_match_expr(&corr_scaled, &scaled_resolved) && same_match_expr(&corr_num, num) {
                let chosen_den = if same_rendered_expr(&corr_den, &den) {
                    den
                } else {
                    if debug {
                        eprintln!(
                            "recover parts preferring correction denominator {} over {}",
                            corr_den.render(),
                            den.render()
                        );
                    }
                    corr_den
                };
                return Some((num.clone(), chosen_den));
            }
            if debug {
                eprintln!(
                    "recover parts compare mismatch den={} corr_den={} scaled={} corr_scaled={} num={} corr_num={}",
                    den.render(),
                    corr_den.render(),
                    scaled_resolved.render(),
                    corr_scaled.render(),
                    num.render(),
                    corr_num.render()
                );
            }
        }
    }
    if debug {
        eprintln!(
            "recover parts recip mismatch scaled={} refined={}",
            scaled_resolved.render(),
            normalize_match_expr(refined_term, defs, 12).render()
        );
    }
    None
}

fn match_rcp_refine_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Expr> {
    let debug = std::env::var("DEBUG_RCP_RECOVER").is_ok();
    let expr = normalize_match_expr(expr, defs, 12);
    if debug {
        eprintln!("recover refine normalized {:?}", expr);
    }
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    for (mul_term, approx_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        let Some(factor_pairs) = match_mul_factor_pairs(&mul_term, defs) else {
            continue;
        };
        for (approx_a, err) in factor_pairs {
            let Some((err_den, err_approx)) = match_neg_mul_plus_one_expr(&err, defs) else {
                continue;
            };
            if same_recip_seed_expr(&err_approx, &approx_a) {
                return Some(err_den);
            }
            if same_recip_seed_expr(&approx_term, &approx_a)
                && same_recip_seed_expr(&err_approx, &approx_a)
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
    let debug = std::env::var("DEBUG_RCP_RECOVER").is_ok();
    let expr = normalize_match_expr(expr, defs, 12);
    if debug {
        eprintln!("recover correction normalized {:?}", expr);
    }
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    for (neg_mul_term, num_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        if debug {
            eprintln!(
                "recover correction try neg_mul={} num={}",
                neg_mul_term.render(),
                num_term.render()
            );
        }
        let Some((den, scaled)) = match_neg_mul_expr(&neg_mul_term, defs) else {
            if debug {
                eprintln!("recover correction no neg_mul in {}", neg_mul_term.render());
            }
            continue;
        };
        return Some((den, scaled, normalize_match_expr(&num_term, defs, 12)));
    }
    None
}

fn match_neg_mul_plus_one_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, 12);
    let (lhs, rhs) = match_binary_expr(&expr, "+", defs)?;
    for (neg_mul_term, one_term) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
        if !expr_is_one(&resolve_named_expr(&one_term, defs, 12)) {
            continue;
        }
        let Some((den, approx)) = match_neg_mul_expr(&neg_mul_term, defs) else {
            continue;
        };
        return Some((den, approx));
    }
    None
}

fn match_neg_mul_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, 12);
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
    let expr = normalize_match_expr(expr, defs, 12);
    match expr {
        Expr::Unary { op, arg } if op == "-" => Some(normalize_match_expr(&arg, defs, 12)),
        Expr::Raw(text) | Expr::Imm(text) | Expr::Reg(text) => {
            let text = strip_raw_expr_wrappers(text.trim());
            let inner = raw_unary_operand(text, "-")?;
            let inner_expr = if inner.parse::<i64>().is_ok() {
                Expr::Imm(inner.to_string())
            } else {
                Expr::Raw(inner.to_string())
            };
            Some(normalize_match_expr(&inner_expr, defs, 12))
        }
        _ => None,
    }
}

fn match_mul_expr(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, 12);
    match expr {
        Expr::Binary { op, lhs, rhs } if op == "*" => Some((
            normalize_match_expr(&lhs, defs, 12),
            normalize_match_expr(&rhs, defs, 12),
        )),
        Expr::Raw(text) => {
            let text = strip_raw_expr_wrappers(text.trim());
            let (lhs, rhs) = split_raw_top_level_binary(text, "*")?;
            Some((
                normalize_match_expr(&Expr::Raw(lhs.to_string()), defs, 12),
                normalize_match_expr(&Expr::Raw(rhs.to_string()), defs, 12),
            ))
        }
        _ => None,
    }
}

fn match_mul_factor_pairs(expr: &Expr, defs: &HashMap<String, Expr>) -> Option<Vec<(Expr, Expr)>> {
    let (lhs, rhs) = match_mul_expr(expr, defs)?;
    if same_rendered_expr(&lhs, &rhs) {
        Some(vec![(lhs, rhs)])
    } else {
        Some(vec![(lhs.clone(), rhs.clone()), (rhs, lhs)])
    }
}

fn match_binary_expr(expr: &Expr, op: &str, defs: &HashMap<String, Expr>) -> Option<(Expr, Expr)> {
    let expr = normalize_match_expr(expr, defs, 12);
    match expr {
        Expr::Binary {
            op: actual,
            lhs,
            rhs,
        } if actual == op => Some((
            normalize_match_expr(&lhs, defs, 12),
            normalize_match_expr(&rhs, defs, 12),
        )),
        Expr::Raw(text) => {
            let text = strip_raw_expr_wrappers(text.trim());
            let (lhs, rhs) = split_raw_top_level_binary(text, op)?;
            Some((
                normalize_match_expr(&Expr::Raw(lhs.to_string()), defs, 12),
                normalize_match_expr(&Expr::Raw(rhs.to_string()), defs, 12),
            ))
        }
        _ => None,
    }
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
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_) => expr.clone(),
    }
}

fn same_rendered_expr(lhs: &Expr, rhs: &Expr) -> bool {
    normalize_match_render(&lhs.render()) == normalize_match_render(&rhs.render())
}

fn same_match_expr(lhs: &Expr, rhs: &Expr) -> bool {
    same_rendered_expr(lhs, rhs) || canonical_match_expr(lhs) == canonical_match_expr(rhs)
}

fn canonical_match_expr(expr: &Expr) -> String {
    match expr {
        Expr::Binary { op, .. } if matches!(op.as_str(), "+" | "*") => {
            let mut terms = Vec::new();
            collect_assoc_match_terms(expr, op, &mut terms);
            let mut keys = terms
                .into_iter()
                .map(|term| canonical_match_expr(term))
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
            format!(
                "lane:{}:{}",
                lane.render_suffix(),
                canonical_match_expr(value)
            )
        }
        _ => normalize_match_render(&expr.render()),
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

fn same_recip_seed_expr(lhs: &Expr, rhs: &Expr) -> bool {
    if same_rendered_expr(lhs, rhs) {
        return true;
    }
    match (parse_expr_f64(lhs), parse_expr_f64(rhs)) {
        (Some(lhs), Some(rhs)) => (lhs - rhs).abs() <= 1.0e-6,
        _ => false,
    }
}

fn parse_expr_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Imm(text) | Expr::Raw(text) | Expr::Builtin(text) | Expr::Reg(text) => {
            normalize_match_render(text).parse::<f64>().ok()
        }
        _ => None,
    }
}

fn normalize_match_render(text: &str) -> String {
    strip_raw_expr_wrappers(text.trim()).to_string()
}

fn strip_loop_phi_markers_stmt(stmt: Stmt) -> Stmt {
    match stmt {
        Stmt::Block(stmts) => {
            Stmt::Block(stmts.into_iter().map(strip_loop_phi_markers_stmt).collect())
        }
        Stmt::Sequence(stmts) => {
            Stmt::Sequence(stmts.into_iter().map(strip_loop_phi_markers_stmt).collect())
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition: strip_loop_phi_markers_expr(condition),
            then_branch: Box::new(strip_loop_phi_markers_stmt(*then_branch)),
            else_branch: else_branch.map(|branch| Box::new(strip_loop_phi_markers_stmt(*branch))),
        },
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition: condition.map(strip_loop_phi_markers_expr),
            body: Box::new(strip_loop_phi_markers_stmt(*body)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant: discriminant.map(strip_loop_phi_markers_expr),
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, strip_loop_phi_markers_stmt(body)))
                .collect(),
            default: default.map(|body| Box::new(strip_loop_phi_markers_stmt(*body))),
        },
        Stmt::Return(expr) => Stmt::Return(expr.map(strip_loop_phi_markers_expr)),
        Stmt::Assign { dst, src } => Stmt::Assign {
            dst,
            src: strip_loop_phi_markers_expr(src),
        },
        Stmt::ExprStmt(expr) => Stmt::ExprStmt(strip_loop_phi_markers_expr(expr)),
        other => other,
    }
}

fn strip_loop_phi_markers_expr(expr: Expr) -> Expr {
    match expr {
        Expr::CallLike { func, mut args } if func == "__loop_phi" && args.len() == 1 => {
            strip_loop_phi_markers_expr(args.remove(0))
        }
        Expr::Unary { op, arg } => Expr::Unary {
            op,
            arg: Box::new(strip_loop_phi_markers_expr(*arg)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(strip_loop_phi_markers_expr(*lhs)),
            rhs: Box::new(strip_loop_phi_markers_expr(*rhs)),
        },
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ternary {
            cond: Box::new(strip_loop_phi_markers_expr(*cond)),
            then_expr: Box::new(strip_loop_phi_markers_expr(*then_expr)),
            else_expr: Box::new(strip_loop_phi_markers_expr(*else_expr)),
        },
        Expr::CallLike { func, args } => Expr::CallLike {
            func,
            args: args.into_iter().map(strip_loop_phi_markers_expr).collect(),
        },
        Expr::Intrinsic { op, args } => Expr::Intrinsic {
            op,
            args: args.into_iter().map(strip_loop_phi_markers_expr).collect(),
        },
        Expr::Load { ty, addr } => Expr::Load {
            ty,
            addr: Box::new(strip_loop_phi_markers_expr(*addr)),
        },
        Expr::WidePtr { base, offset } => Expr::WidePtr {
            base: Box::new(strip_loop_phi_markers_expr(*base)),
            offset: Box::new(strip_loop_phi_markers_expr(*offset)),
        },
        Expr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(strip_loop_phi_markers_expr(*lo)),
            hi: Box::new(strip_loop_phi_markers_expr(*hi)),
        },
        Expr::LaneExtract { value, lane } => Expr::LaneExtract {
            value: Box::new(strip_loop_phi_markers_expr(*value)),
            lane,
        },
        Expr::Cast { ty, expr } => Expr::Cast {
            ty,
            expr: Box::new(strip_loop_phi_markers_expr(*expr)),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(strip_loop_phi_markers_expr(*base)),
            index: Box::new(strip_loop_phi_markers_expr(*index)),
        },
        other => other,
    }
}

fn simplify_if(condition: Expr, then_branch: Stmt, else_branch: Option<Stmt>) -> Stmt {
    let then_branch = simplify_stmt(then_branch);
    let else_branch = else_branch
        .map(simplify_stmt)
        .filter(|stmt| !is_effectively_empty(stmt));

    if let Some(condition_value) = expr_const_bool(&condition) {
        return if condition_value {
            then_branch
        } else {
            else_branch.unwrap_or(Stmt::Empty)
        };
    }

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
        Stmt::Label { body, .. } => is_effectively_empty(body),
        _ => false,
    }
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

fn loop_condition_live_vars(condition: &Expr) -> BTreeSet<String> {
    let mut live = BTreeSet::new();
    collect_used_expr_vars(condition, &mut live);
    live
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

fn expr_is_pure_for_dce(expr: &Expr) -> bool {
    if matches!(expr, Expr::CallLike { func, .. } if func == "__loop_phi") {
        return false;
    }
    match expr {
        Expr::Raw(text) => is_pure_raw_expr(text),
        Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::ConstMemSymbol(_) => true,
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
        Expr::Intrinsic { args, .. } => args.iter().all(expr_is_pure_for_dce),
        Expr::WidePtr { base, offset } => {
            expr_is_pure_for_dce(base) && expr_is_pure_for_dce(offset)
        }
        Expr::Load { .. } | Expr::Builtin(_) | Expr::Addr64 { .. } => false,
        Expr::Cast { expr, .. } => expr_is_pure_for_dce(expr),
        Expr::Index { base, index } => expr_is_pure_for_dce(base) && expr_is_pure_for_dce(index),
    }
}

fn prune_globally_unused_pure_assigns(stmt: Stmt) -> Stmt {
    let mut current = stmt;
    loop {
        let mut used = BTreeSet::new();
        collect_stmt_used_vars(&current, &mut used);
        let next = prune_globally_unused_pure_assigns_once(current.clone(), &used);
        if next == current {
            return next;
        }
        current = next;
    }
}

fn prune_globally_unused_pure_assigns_once(stmt: Stmt, used: &BTreeSet<String>) -> Stmt {
    match stmt {
        Stmt::Sequence(stmts) => rebuild_stmt_preserving_shape(
            stmts
                .into_iter()
                .map(|stmt| prune_globally_unused_pure_assigns_once(stmt, used))
                .collect(),
            true,
        ),
        Stmt::Block(stmts) => rebuild_stmt_preserving_shape(
            stmts
                .into_iter()
                .map(|stmt| prune_globally_unused_pure_assigns_once(stmt, used))
                .collect(),
            false,
        ),
        Stmt::Label { name, body } => Stmt::Label {
            name,
            body: Box::new(prune_globally_unused_pure_assigns_once(*body, used)),
        },
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => simplify_if(
            condition,
            prune_globally_unused_pure_assigns_once(*then_branch, used),
            else_branch.map(|branch| prune_globally_unused_pure_assigns_once(*branch, used)),
        ),
        Stmt::Loop {
            kind,
            condition,
            body,
        } => Stmt::Loop {
            kind,
            condition,
            body: Box::new(prune_globally_unused_pure_assigns_once(*body, used)),
        },
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => Stmt::Switch {
            discriminant,
            cases: cases
                .into_iter()
                .map(|(label, body)| (label, prune_globally_unused_pure_assigns_once(body, used)))
                .collect(),
            default: default
                .map(|body| Box::new(prune_globally_unused_pure_assigns_once(*body, used))),
        },
        Stmt::Assign { dst, src } => {
            let drop_assign = lvalue_symbol_name(&dst).is_some_and(|name| {
                can_prune_globally_unused_name(&name)
                    && !used.contains(&name)
                    && expr_is_pure_for_dce(&src)
            });
            if drop_assign {
                Stmt::Empty
            } else {
                Stmt::Assign { dst, src }
            }
        }
        other => other,
    }
}

fn can_prune_globally_unused_name(name: &str) -> bool {
    is_symbolic_name(name) && PointerLane::parse_named(name).is_none()
}

fn ast_conservative_dce(stmt: Stmt) -> Stmt {
    let cleaned = prune_globally_unused_pure_assigns(remove_pure_noop_exprs_stmt(stmt));
    let cleaned = prune_dead_runtime_helper_temps_stmt(cleaned, BTreeSet::new()).0;
    prune_globally_unused_pure_assigns(cleaned)
}

fn rebuild_stmt_preserving_shape(stmts: Vec<Stmt>, is_sequence: bool) -> Stmt {
    let out = stmts
        .into_iter()
        .filter(|stmt| !matches!(stmt, Stmt::Empty))
        .collect::<Vec<_>>();
    match (is_sequence, out.len()) {
        (_, 0) => Stmt::Empty,
        (true, 1) => out.into_iter().next().unwrap_or(Stmt::Empty),
        (true, _) => Stmt::Sequence(out),
        (false, _) => Stmt::Block(out),
    }
}

fn is_pure_calllike(func: &str) -> bool {
    matches!(
        func,
        "abs"
            | "carry_u32_add3"
            | "FCHK"
            | "lea_hi_x"
            | "lea_hi_x_sx32"
            | "mul_hi_u32"
            | "rcp_approx"
            | "rsqrtf"
            | "exp2f"
            | "log2f"
            | "sinf"
            | "cosf"
            | "sqrtf"
            // Raw nameless CALL.REL.NOINC sites in corpus kernels are
            // compiler-emitted libdevice slow paths, not source-level calls.
            | "CALL.REL.NOINC"
    )
}

fn collect_used_lvalue_vars(lvalue: &LValue, live: &mut BTreeSet<String>) {
    match lvalue {
        LValue::Raw(text) => collect_raw_identifiers(text, live),
        LValue::Var(_) => {}
        LValue::PtrLane { base, lane } => {
            live.insert(pointer_lane_name(base, *lane));
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
        Expr::PtrLane { base, lane } => {
            live.insert(pointer_lane_name(base, *lane));
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

#[cfg(test)]
#[path = "ast_passes/tests.rs"]
mod tests;
