//! Symbol and declaration planning scaffolding for the rewritten backend.
//!
//! Purpose:
//! - own deterministic parameter/local/shared declaration planning from the
//!   structured AST and analysis facts
//!
//! Inputs:
//! - `StructuredFunction`
//! - `FunctionAnalysis`
//!
//! Outputs:
//! - `SymbolPlan`, containing deterministic declaration and naming decisions
//!
//! Invariants:
//! - declarations are produced structurally, not inferred from rendered text
//! - naming is deterministic under identical AST traversal order
//!
//! This module must not:
//! - scan rendered output
//! - use regex fallback declaration synthesis

use std::collections::BTreeSet;

use crate::abi::{ArgAlias, ArgAliasKind, ArgScalarKind};
use crate::ast::{Decl, Expr, LValue, StorageClass, StructuredFunction, Stmt};
use crate::backend_names::is_predicate_ident;
use crate::function_analysis::FunctionAnalysis;
use crate::memory_model::CudaMemorySpace;

const SPACE_BACKING_BYTES: usize = 4;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolPlan {
    pub params: Vec<Decl>,
    pub locals: Vec<Decl>,
}

pub fn plan_symbols(function: &StructuredFunction, analysis: &FunctionAnalysis) -> SymbolPlan {
    let mut params = planned_params(analysis);
    let mut seen = params
        .iter()
        .map(|decl| decl.name.clone())
        .collect::<BTreeSet<_>>();
    let mut locals = planned_space_decls(analysis);
    for decl in &locals {
        seen.insert(decl.name.clone());
    }
    collect_declared_locals(&function.body, &mut locals, &mut seen);
    params.append(&mut Vec::new());
    SymbolPlan { params, locals }
}

fn planned_params(analysis: &FunctionAnalysis) -> Vec<Decl> {
    let mut out = Vec::new();
    for (param_idx, alias) in &analysis.abi_aliases.by_param {
        let (name, ty) = alias_decl(*param_idx, alias);
        out.push(Decl {
            name,
            ty,
            array_len: None,
            dynamic_extent: false,
            storage: StorageClass::Param,
            live_in: false,
        });
    }
    out
}

fn alias_decl(param_idx: u32, alias: &ArgAlias) -> (String, String) {
    match alias.kind {
        ArgAliasKind::Ptr64 => {
            let pointee = alias.pointee_ty.unwrap_or("uint32_t");
            (format!("arg{}_ptr", param_idx), format!("{}*", pointee))
        }
        ArgAliasKind::U64 => (format!("arg{}_u64", param_idx), "uint64_t".to_string()),
        ArgAliasKind::Word32 => (
            format!("arg{}", param_idx),
            match alias.scalar_kind.unwrap_or(ArgScalarKind::U32) {
                ArgScalarKind::U32 => "uint32_t",
                ArgScalarKind::I32 => "int32_t",
                ArgScalarKind::F32 => "float",
            }
            .to_string(),
        ),
    }
}

fn planned_space_decls(analysis: &FunctionAnalysis) -> Vec<Decl> {
    let mut out = Vec::new();
    if analysis
        .mem_accesses
        .iter()
        .any(|access| access.space == CudaMemorySpace::Shared)
    {
        let (array_len, dynamic_extent) = planned_space_decl_shape(analysis, CudaMemorySpace::Shared);
        out.push(Decl {
            name: "shmem".to_string(),
            ty: "uint32_t".to_string(),
            array_len,
            dynamic_extent,
            storage: StorageClass::Shared,
            live_in: false,
        });
    }
    if analysis
        .mem_accesses
        .iter()
        .any(|access| access.space == CudaMemorySpace::Local)
    {
        let (array_len, dynamic_extent) = planned_space_decl_shape(analysis, CudaMemorySpace::Local);
        if !dynamic_extent {
            out.push(Decl {
                name: "local_mem".to_string(),
                ty: "uint32_t".to_string(),
                array_len,
                dynamic_extent,
                storage: StorageClass::Local,
                live_in: false,
            });
        }
    }
    out
}

fn planned_space_decl_shape(
    analysis: &FunctionAnalysis,
    space: CudaMemorySpace,
) -> (Option<usize>, bool) {
    let relevant = analysis
        .mem_accesses
        .iter()
        .filter(|access| access.space == space)
        .collect::<Vec<_>>();
    let needs_dynamic_extent = relevant.iter().any(|access| {
        access.has_dynamic_offset
            || access.constant_byte_offset.is_none()
            || access.constant_byte_offset.is_some_and(|offset| offset < 0)
    });
    if needs_dynamic_extent {
        return (None, true);
    }
    let Some(array_len) = relevant
        .into_iter()
        .try_fold(1usize, |max_len, access| {
            let base_offset = usize::try_from(access.constant_byte_offset?).ok()?;
            let access_bytes = access
                .bit_width
                .and_then(|bits| usize::try_from(bits / 8).ok())
                .filter(|bytes| *bytes > 0)
                .unwrap_or(SPACE_BACKING_BYTES);
            let lanes = usize::from(access.vector_width.unwrap_or(1)).max(1);
            let end_offset = base_offset.checked_add(access_bytes.checked_mul(lanes)?)?;
            let storage_len = end_offset.div_ceil(SPACE_BACKING_BYTES).max(1);
            Some(max_len.max(storage_len))
        })
    else {
        return (None, true);
    };
    (Some(array_len), false)
}

fn collect_declared_locals(stmt: &Stmt, locals: &mut Vec<Decl>, seen: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
            for stmt in stmts {
                collect_declared_locals(stmt, locals, seen);
            }
        }
        Stmt::Label { body, .. } => collect_declared_locals(body, locals, seen),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_expr_names(condition, locals, seen);
            collect_declared_locals(then_branch, locals, seen);
            if let Some(else_branch) = else_branch {
                collect_declared_locals(else_branch, locals, seen);
            }
        }
        Stmt::Loop { condition, body, .. } => {
            if let Some(condition) = condition {
                collect_expr_names(condition, locals, seen);
            }
            collect_declared_locals(body, locals, seen);
        }
        Stmt::Switch {
            discriminant,
            cases,
            default,
        } => {
            if let Some(discriminant) = discriminant {
                collect_expr_names(discriminant, locals, seen);
            }
            for (_, body) in cases {
                collect_declared_locals(body, locals, seen);
            }
            if let Some(default) = default {
                collect_declared_locals(default, locals, seen);
            }
        }
        Stmt::Assign { dst, src } => {
            collect_lvalue_name(dst, locals, seen);
            collect_expr_names(src, locals, seen);
        }
        Stmt::ExprStmt(expr) | Stmt::Return(Some(expr)) => collect_expr_names(expr, locals, seen),
        Stmt::Return(None) | Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Empty => {}
    }
}

fn collect_lvalue_name(lvalue: &LValue, locals: &mut Vec<Decl>, seen: &mut BTreeSet<String>) {
    match lvalue {
        LValue::Var(name) => maybe_add_local(name, locals, seen),
        LValue::Raw(_)
        | LValue::PtrLane { .. }
        | LValue::Deref { .. }
        | LValue::Indexed { .. } => {}
    }
}

fn collect_expr_names(expr: &Expr, locals: &mut Vec<Decl>, seen: &mut BTreeSet<String>) {
    match expr {
        Expr::Reg(name) => maybe_add_local(name, locals, seen),
        Expr::PtrLane { base, .. } => maybe_add_local(base, locals, seen),
        Expr::Unary { arg, .. }
        | Expr::Cast { expr: arg, .. }
        | Expr::LaneExtract { value: arg, .. } => collect_expr_names(arg, locals, seen),
        Expr::Binary { lhs, rhs, .. } => {
            collect_expr_names(lhs, locals, seen);
            collect_expr_names(rhs, locals, seen);
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_expr_names(cond, locals, seen);
            collect_expr_names(then_expr, locals, seen);
            collect_expr_names(else_expr, locals, seen);
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_expr_names(arg, locals, seen);
            }
        }
        Expr::Load { addr, .. } => collect_expr_names(addr, locals, seen),
        Expr::WidePtr { base, offset } => {
            collect_expr_names(base, locals, seen);
            collect_expr_names(offset, locals, seen);
        }
        Expr::Addr64 { lo, hi } => {
            collect_expr_names(lo, locals, seen);
            collect_expr_names(hi, locals, seen);
        }
        Expr::Index { base, index } => {
            collect_expr_names(base, locals, seen);
            collect_expr_names(index, locals, seen);
        }
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_) => {}
    }
}

fn maybe_add_local(name: &str, locals: &mut Vec<Decl>, seen: &mut BTreeSet<String>) {
    if seen.contains(name)
        || matches!(name, "shmem" | "local_mem")
        || name.starts_with("arg")
        || !name.chars().next().is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
    {
        return;
    }
    seen.insert(name.to_string());
    locals.push(Decl {
        name: name.to_string(),
        ty: if is_predicate_ident(name) {
            "bool".to_string()
        } else {
            "uint32_t".to_string()
        },
        array_len: None,
        dynamic_extent: false,
        storage: StorageClass::Local,
        live_in: false,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{AbiArgAliases, ArgAlias, AliasConfidence};
    use crate::function_analysis::{AddressRoot, MemAccessInfo};
    use crate::memory_model::{CudaMemorySpace, MemAccessKind};

    #[test]
    fn plans_params_shared_and_structural_locals_deterministically() {
        let mut analysis = FunctionAnalysis::default();
        analysis.abi_aliases = AbiArgAliases {
            by_param: [
                (
                    0,
                    ArgAlias {
                        param_idx: 0,
                        kind: ArgAliasKind::Ptr64,
                        confidence: AliasConfidence::High,
                        observed_words: [0, 1].into_iter().collect(),
                        scalar_kind: None,
                        signed_words: BTreeSet::new(),
                        pointee_ty: Some("float"),
                    },
                ),
                (
                    4,
                    ArgAlias {
                        param_idx: 4,
                        kind: ArgAliasKind::Word32,
                        confidence: AliasConfidence::Medium,
                        observed_words: [0].into_iter().collect(),
                        scalar_kind: Some(ArgScalarKind::I32),
                        signed_words: [0].into_iter().collect(),
                        pointee_ty: None,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(8),
            has_dynamic_offset: false,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let function = StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body: Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("v0".to_string()),
                    src: Expr::Imm("1".to_string()),
                },
                Stmt::Assign {
                    dst: LValue::Var("p1".to_string()),
                    src: Expr::Reg("v0".to_string()),
                },
            ]),
        };

        let plan = plan_symbols(&function, &analysis);
        assert_eq!(plan.params[0].name, "arg0_ptr");
        assert_eq!(plan.params[0].ty, "float*");
        assert_eq!(plan.params[1].name, "arg4");
        assert_eq!(plan.locals[0].name, "shmem");
        assert_eq!(plan.locals[0].array_len, Some(3));
        assert!(!plan.locals[0].dynamic_extent);
        assert_eq!(plan.locals[0].storage, StorageClass::Shared);
        assert_eq!(plan.locals[1].name, "v0");
        assert_eq!(plan.locals[2].name, "p1");
        assert_eq!(plan.locals[2].ty, "bool");
    }

    #[test]
    fn plans_dynamic_shared_objects_without_fake_fixed_extent() {
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Store,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(8),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });

        let plan = plan_symbols(
            &StructuredFunction {
                params: Vec::new(),
                locals: Vec::new(),
                body: Stmt::Empty,
            },
            &analysis,
        );
        assert_eq!(plan.locals[0].name, "shmem");
        assert_eq!(plan.locals[0].array_len, None);
        assert!(plan.locals[0].dynamic_extent);
    }

    #[test]
    fn omits_dynamic_local_backing_decls() {
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Store,
            space: CudaMemorySpace::Local,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(8),
            has_dynamic_offset: true,
            root: AddressRoot::LocalObject("local_mem".to_string()),
        });

        let plan = plan_symbols(
            &StructuredFunction {
                params: Vec::new(),
                locals: Vec::new(),
                body: Stmt::Empty,
            },
            &analysis,
        );
        assert!(plan.locals.is_empty());
    }

    #[test]
    fn sizes_shared_backing_objects_in_storage_words() {
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Store,
            space: CudaMemorySpace::Shared,
            bit_width: Some(64),
            vector_width: None,
            constant_byte_offset: Some(4),
            has_dynamic_offset: false,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });

        let plan = plan_symbols(
            &StructuredFunction {
                params: Vec::new(),
                locals: Vec::new(),
                body: Stmt::Empty,
            },
            &analysis,
        );
        assert_eq!(plan.locals[0].name, "shmem");
        assert_eq!(plan.locals[0].array_len, Some(3));
    }
}
