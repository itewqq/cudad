//! Structured AST lowering helpers for the rewritten backend.
//!
//! Purpose:
//! - lower SSA-backed memory operations into the canonical AST using
//!   `FunctionAnalysis` facts rather than semantic-lift side tables or
//!   rendered-text repair
//!
//! Inputs:
//! - `IRStatement`
//! - statement location
//! - `FunctionAnalysis`
//!
//! Outputs:
//! - AST expressions/lvalues/statements for memory operations
//!
//! Invariants:
//! - lowering consumes analysis facts directly
//! - lowering is deterministic
//! - no rendered-text parsing is allowed
//!
//! This module must not:
//! - inspect previously rendered code
//! - repair names or declarations after rendering

use std::collections::BTreeSet;

use crate::abi::ConstMemSemantic;
use crate::ast::{Expr, LValue, PointerLane, Stmt};
use crate::backend_names::canonical_reg_ident;
use crate::canonical_ast_passes::prune_dead_pure_defs;
use crate::function_analysis::{AddressRoot, FunctionAnalysis, MemAccessInfo};
use crate::ir::{IRExpr, IRStatement, RValue};
use crate::memory_model::{CudaMemorySpace, MemAccessKind};
use crate::structurizer::{LoopType, StructuredStatement};
use crate::symbol_plan::plan_symbols;

#[derive(Clone, Debug)]
struct LoweredStmt {
    stmt: Stmt,
    predicate_def_idx: Option<usize>,
}

pub fn lower_memory_stmt(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Option<Stmt> {
    lower_memory_stmt_detail(block_id, stmt_idx, stmt, analysis).map(|lowered| lowered.stmt)
}

fn lower_memory_stmt_detail(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Option<LoweredStmt> {
    let access = lookup_mem_access(analysis, block_id, stmt_idx)?;
    if access.space == CudaMemorySpace::Local && local_space_requires_explicit_ops(analysis) {
        return lower_explicit_local_stmt(stmt, access);
    }
    match access.kind {
        MemAccessKind::Load => {
            let dst = stmt.defs.first()?.get_reg()?;
            let src = lower_memory_load_expr(stmt, access, analysis)?;
            Some(LoweredStmt {
                stmt: Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(dst)),
                    src,
                },
                predicate_def_idx: Some(0),
            })
        }
        MemAccessKind::Store => {
            let dst = lower_memory_store_lvalue(stmt, access, analysis)?;
            let src = store_value_expr(stmt, Some(analysis))?;
            Some(LoweredStmt {
                stmt: Stmt::Assign { dst, src },
                predicate_def_idx: None,
            })
        }
        MemAccessKind::Atomic | MemAccessKind::Reduction => {
            let opcode = stmt_opcode(stmt);
            let func = atomic_func_name(opcode);
            let Some((base, index)) = lower_base_and_index(stmt, access, analysis) else {
                return None;
            };
            let mut args = vec![Expr::Unary {
                op: "&".to_string(),
                arg: Box::new(Expr::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                }),
            }];
            args.extend(
                stmt_args(stmt)
                    .iter()
                    .filter(|expr| !matches!(expr, IRExpr::Mem { .. }))
                    .map(|expr| lower_scalar_expr_with_analysis(expr, Some(analysis))),
            );
            if opcode.contains(".POPC.INC") && args.len() == 1 {
                args.push(Expr::Imm("1".to_string()));
            }
            let call = Expr::CallLike {
                func: func.to_string(),
                args,
            };
            if let Some((def_idx, def)) = select_memory_result_def(stmt) {
                Some(LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src: call,
                    },
                    predicate_def_idx: Some(def_idx),
                })
            } else {
                Some(LoweredStmt {
                    stmt: Stmt::ExprStmt(call),
                    predicate_def_idx: None,
                })
            }
        }
    }
}

pub fn lower_structured_function(
    structured: &StructuredStatement,
    analysis: &FunctionAnalysis,
) -> crate::ast::StructuredFunction {
    let body = lower_structured_stmt(structured, analysis);
    let seed = prune_dead_pure_defs(crate::ast::StructuredFunction {
        params: Vec::new(),
        locals: Vec::new(),
        body,
    });
    let plan = plan_symbols(&seed, analysis);
    crate::ast::StructuredFunction {
        params: plan.params,
        locals: plan.locals,
        body: seed.body,
    }
}

pub fn lower_structured_stmt(
    structured: &StructuredStatement,
    analysis: &FunctionAnalysis,
) -> Stmt {
    match structured {
        StructuredStatement::BasicBlock { block_id, stmts } => Stmt::Sequence(
            stmts
                .iter()
                .enumerate()
                .map(|(stmt_idx, stmt)| lower_basic_stmt(*block_id, stmt_idx, stmt, analysis))
                .collect(),
        ),
        StructuredStatement::Sequence(parts) => Stmt::Sequence(
            parts
                .iter()
                .map(|part| lower_structured_stmt(part, analysis))
                .collect(),
        ),
        StructuredStatement::If {
            condition_expr,
            then_branch,
            else_branch,
            ..
        } => Stmt::If {
            condition: lower_scalar_expr_with_analysis(condition_expr, Some(analysis)),
            then_branch: Box::new(lower_structured_stmt(then_branch, analysis)),
            else_branch: else_branch
                .as_ref()
                .map(|branch| Box::new(lower_structured_stmt(branch, analysis))),
        },
        StructuredStatement::Loop {
            loop_type,
            condition_expr,
            body,
            ..
        } => Stmt::Loop {
            kind: match loop_type {
                LoopType::While => crate::ast::LoopKind::While,
                LoopType::DoWhile => crate::ast::LoopKind::DoWhile,
                LoopType::Endless => crate::ast::LoopKind::Endless,
            },
            condition: condition_expr.as_ref().map(lower_scalar_expr),
            body: Box::new(lower_structured_stmt(body, analysis)),
        },
        StructuredStatement::Break(_) => Stmt::Break,
        StructuredStatement::Continue(_) => Stmt::Continue,
        StructuredStatement::Return(expr) => Stmt::Return(expr.as_ref().map(lower_scalar_expr)),
        StructuredStatement::UnstructuredJump { to_block_id, .. } => {
            Stmt::Goto(format!("BB{}", to_block_id))
        }
        StructuredStatement::Switch {
            discriminant,
            cases,
            default,
            ..
        } => Stmt::Switch {
            discriminant: discriminant.as_ref().map(lower_scalar_expr),
            cases: cases
                .iter()
                .map(|(label, body)| (*label, lower_structured_stmt(body, analysis)))
                .collect(),
            default: default
                .as_ref()
                .map(|body| Box::new(lower_structured_stmt(body, analysis))),
        },
        StructuredStatement::Empty => Stmt::Empty,
    }
}

pub fn lower_memory_load_expr(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<Expr> {
    if let Some(special) = lower_constmem_load_expr(access, analysis) {
        return Some(special);
    }
    let (base, index) = lower_base_and_index(stmt, access, analysis)?;
    Some(Expr::Index {
        base: Box::new(base),
        index: Box::new(index),
    })
}

pub fn lower_memory_store_lvalue(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<LValue> {
    let (base, index) = lower_base_and_index(stmt, access, analysis)?;
    Some(LValue::Indexed {
        base: Box::new(base),
        index: Box::new(index),
    })
}

fn lookup_mem_access<'a>(
    analysis: &'a FunctionAnalysis,
    block_id: usize,
    stmt_idx: usize,
) -> Option<&'a MemAccessInfo> {
    analysis
        .mem_accesses
        .iter()
        .find(|access| access.block_id == block_id && access.stmt_idx == stmt_idx)
}

fn lower_constmem_load_expr(access: &MemAccessInfo, analysis: &FunctionAnalysis) -> Option<Expr> {
    let stmt_ref = crate::abi::StatementRef {
        block_id: access.block_id,
        stmt_idx: access.stmt_idx,
    };
    let annotation = analysis
        .abi_annotations
        .constmem_by_stmt
        .get(&stmt_ref)
        .and_then(|annotations| annotations.first())?;
    Some(constmem_semantic_expr(&annotation.semantic, analysis))
}

fn lower_base_and_index(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<(Expr, Expr)> {
    let mem_expr = stmt.mem_addr_args.as_ref()?.first()?;
    let IRExpr::Mem {
        base: addr_base,
        offset,
        width,
    } = mem_expr
    else {
        return None;
    };
    let base = match access.space {
        CudaMemorySpace::Shared => Expr::Builtin(match &access.root {
            AddressRoot::SharedObject(name) => name.clone(),
            _ => "shmem".to_string(),
        }),
        CudaMemorySpace::Local => Expr::Reg(match &access.root {
            AddressRoot::LocalObject(name) => name.clone(),
            _ => "local_mem".to_string(),
        }),
        CudaMemorySpace::Global => match &access.root {
            AddressRoot::ParamWord(param_idx) => {
                Expr::Reg(global_param_base_name(*param_idx, analysis))
            }
            AddressRoot::RegisterBase(reg) => lower_reg_expr(reg),
            _ => lower_scalar_expr_with_analysis(mem_expr, Some(analysis)),
        },
        CudaMemorySpace::Const => match &access.root {
            AddressRoot::ConstSymbol(name) => Expr::ConstMemSymbol(name.clone()),
            _ => lower_scalar_expr_with_analysis(mem_expr, Some(analysis)),
        },
        CudaMemorySpace::Param | CudaMemorySpace::Generic => {
            lower_scalar_expr_with_analysis(mem_expr, Some(analysis))
        }
    };
    let index = match access.space {
        CudaMemorySpace::Shared | CudaMemorySpace::Local => lower_rooted_element_index_expr(
            addr_base.as_ref(),
            offset.as_deref(),
            access,
            analysis,
            Some(space_backing_bit_width(access.space)),
        ),
        CudaMemorySpace::Global => lower_global_index_expr(
            addr_base.as_ref(),
            offset.as_deref(),
            access,
            analysis,
            access.bit_width.or(*width),
        ),
        _ => lower_index_expr(offset.as_deref(), access.bit_width.or(*width), Some(analysis)),
    };
    Some((normalize_index_base(base, access, analysis), index))
}

fn lower_explicit_local_stmt(stmt: &IRStatement, access: &MemAccessInfo) -> Option<LoweredStmt> {
    let byte_expr = lower_local_byte_offset_expr(stmt)?;
    match access.kind {
        MemAccessKind::Load => {
            let dst = stmt.defs.first()?.get_reg()?;
            let src = Expr::CallLike {
                func: local_space_helper_name("load", access),
                args: vec![byte_expr],
            };
            Some(LoweredStmt {
                stmt: Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(dst)),
                    src,
                },
                predicate_def_idx: Some(0),
            })
        }
        MemAccessKind::Store => {
            let src = store_value_expr(stmt, None)?;
            Some(LoweredStmt {
                stmt: Stmt::ExprStmt(Expr::CallLike {
                    func: local_space_helper_name("store", access),
                    args: vec![byte_expr, src],
                }),
                predicate_def_idx: None,
            })
        }
        MemAccessKind::Atomic | MemAccessKind::Reduction => None,
    }
}

fn lower_local_byte_offset_expr(stmt: &IRStatement) -> Option<Expr> {
    let mem_expr = stmt.mem_addr_args.as_ref()?.first()?;
    let IRExpr::Mem { base, offset, .. } = mem_expr else {
        return None;
    };
    Some(combine_byte_offset_expr(base.as_ref(), offset.as_deref(), None))
}

fn local_space_helper_name(prefix: &str, access: &MemAccessInfo) -> String {
    let width = access.bit_width.unwrap_or(32);
    let lanes = access.vector_width.unwrap_or(1);
    if lanes > 1 {
        format!("local_{}_bits{}_x{}", prefix, width, lanes)
    } else {
        format!("local_{}_bits{}", prefix, width)
    }
}

fn local_space_requires_explicit_ops(analysis: &FunctionAnalysis) -> bool {
    analysis.mem_accesses.iter().any(|access| {
        access.space == CudaMemorySpace::Local
            && (access.has_dynamic_offset
                || access.constant_byte_offset.is_none()
                || access.constant_byte_offset.is_some_and(|offset| offset < 0))
    })
}

fn space_backing_bit_width(space: CudaMemorySpace) -> u32 {
    match space {
        CudaMemorySpace::Shared | CudaMemorySpace::Local => 32,
        _ => 32,
    }
}

fn lower_index_expr(
    offset: Option<&IRExpr>,
    bit_width: Option<u32>,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    match offset {
        None => Expr::Imm("0".to_string()),
        Some(IRExpr::ImmI(value)) => {
            let bytes = bit_width
                .and_then(|bits| bits.checked_div(8))
                .filter(|bytes| *bytes > 0);
            if let Some(bytes) = bytes {
                if value % i64::from(bytes) == 0 {
                    return Expr::Imm((value / i64::from(bytes)).to_string());
                }
            }
            Expr::Imm(value.to_string())
        }
        Some(expr) => lower_scalar_expr_with_analysis(expr, analysis),
    }
}

fn lower_element_index_expr(
    base: &IRExpr,
    offset: Option<&IRExpr>,
    bit_width: Option<u32>,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    let byte_expr = combine_byte_offset_expr(base, offset, analysis);
    scale_index_expr(byte_expr, bit_width)
}

fn lower_rooted_element_index_expr(
    base: &IRExpr,
    offset: Option<&IRExpr>,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
    bit_width: Option<u32>,
) -> Expr {
    if let Some(rooted_offset) = rooted_space_byte_offset(base, access, analysis) {
        let byte_expr = combine_byte_offset_expr(&rooted_offset, offset, Some(analysis));
        return scale_index_expr(byte_expr, bit_width);
    }
    lower_element_index_expr(base, offset, bit_width, Some(analysis))
}

fn combine_byte_offset_expr(
    base: &IRExpr,
    offset: Option<&IRExpr>,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    let mut terms = Vec::new();
    if !ir_expr_is_zero(base) {
        terms.push(lower_scalar_expr_with_analysis(base, analysis));
    }
    if let Some(offset) = offset.filter(|expr| !ir_expr_is_zero(expr)) {
        terms.push(lower_scalar_expr_with_analysis(offset, analysis));
    }
    match terms.len() {
        0 => Expr::Imm("0".to_string()),
        1 => terms.remove(0),
        _ => Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(terms.remove(0)),
            rhs: Box::new(terms.remove(0)),
        },
    }
}

fn scale_index_expr(expr: Expr, bit_width: Option<u32>) -> Expr {
    let bytes = bit_width
        .and_then(|bits| bits.checked_div(8))
        .filter(|bytes| *bytes > 1);
    let Some(bytes) = bytes else {
        return expr;
    };
    if let Expr::Imm(value) = &expr {
        if let Ok(value) = value.parse::<i64>() {
            if value % i64::from(bytes) == 0 {
                return Expr::Imm((value / i64::from(bytes)).to_string());
            }
        }
    }
    if let Some(simplified) = divide_expr_by_const(&expr, i64::from(bytes)) {
        return simplified;
    }
    Expr::Binary {
        op: "/".to_string(),
        lhs: Box::new(expr),
        rhs: Box::new(Expr::Imm(bytes.to_string())),
    }
}

fn divide_expr_by_const(expr: &Expr, bytes: i64) -> Option<Expr> {
    match expr {
        Expr::Imm(text) => text
            .parse::<i64>()
            .ok()
            .filter(|value| value % bytes == 0)
            .map(|value| Expr::Imm((value / bytes).to_string())),
        Expr::Binary { op, lhs, rhs } if op == "*" => {
            if expr_is_imm_i64(lhs, bytes) {
                return Some((**rhs).clone());
            }
            if expr_is_imm_i64(rhs, bytes) {
                return Some((**lhs).clone());
            }
            None
        }
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => Some(Expr::Binary {
            op: op.clone(),
            lhs: Box::new(divide_expr_by_const(lhs, bytes)?),
            rhs: Box::new(divide_expr_by_const(rhs, bytes)?),
        }),
        _ => None,
    }
}

fn expr_is_imm_i64(expr: &Expr, expected: i64) -> bool {
    matches!(expr, Expr::Imm(text) if text.parse::<i64>().ok() == Some(expected))
}

fn lower_global_index_expr(
    addr_base: &IRExpr,
    offset: Option<&IRExpr>,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
    bit_width: Option<u32>,
) -> Expr {
    let Some(rooted_offset) = rooted_space_byte_offset(addr_base, access, analysis) else {
        return lower_index_expr(offset, bit_width, Some(analysis));
    };
    let byte_expr = combine_byte_offset_expr(&rooted_offset, offset, Some(analysis));
    scale_index_expr(byte_expr, bit_width)
}

fn global_param_base_name(param_idx: u32, analysis: &FunctionAnalysis) -> String {
    if let Some(alias) = analysis.abi_aliases.by_param.get(&param_idx) {
        if alias.kind == crate::abi::ArgAliasKind::Ptr64 {
            return format!("arg{}_ptr", param_idx);
        }
    }
    format!("param_{}", param_idx)
}

fn normalize_index_base(base: Expr, access: &MemAccessInfo, analysis: &FunctionAnalysis) -> Expr {
    if !index_base_requires_pointer_cast(access, &base) {
        return base;
    }
    Expr::Cast {
        ty: format!("{}*", unresolved_access_scalar_ty(access, analysis)),
        expr: Box::new(base),
    }
}

fn index_base_requires_pointer_cast(access: &MemAccessInfo, base: &Expr) -> bool {
    matches!(
        access.space,
        CudaMemorySpace::Global
            | CudaMemorySpace::Const
            | CudaMemorySpace::Param
            | CudaMemorySpace::Generic
    ) && !expr_is_indexable_base(base, access)
}

fn expr_is_indexable_base(expr: &Expr, access: &MemAccessInfo) -> bool {
    match expr {
        Expr::Reg(text) | Expr::Raw(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            text.ends_with("_ptr")
                || text == "shmem"
                || text == "shmem_u8"
                || (matches!(access.root, AddressRoot::ParamWord(_)) && text.starts_with("param_"))
        }
        Expr::Cast { ty, .. } => ty.ends_with('*'),
        _ => false,
    }
}

fn unresolved_access_scalar_ty(access: &MemAccessInfo, analysis: &FunctionAnalysis) -> &'static str {
    match access.space {
        CudaMemorySpace::Shared => analysis.shared_pointee_ty.unwrap_or("uint32_t"),
        _ => scalar_ty_for_bit_width(access.bit_width),
    }
}

fn scalar_ty_for_bit_width(bit_width: Option<u32>) -> &'static str {
    match bit_width.unwrap_or(32) {
        8 => "uint8_t",
        16 => "uint16_t",
        32 => "uint32_t",
        64 => "uint64_t",
        128 => "uint4",
        _ => "uint32_t",
    }
}

fn store_value_expr(stmt: &IRStatement, analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    stmt_args(stmt)
        .iter()
        .find(|expr| !matches!(expr, IRExpr::Mem { .. }))
        .map(|expr| lower_scalar_expr_with_analysis(expr, analysis))
}

fn stmt_args(stmt: &IRStatement) -> &[IRExpr] {
    match &stmt.value {
        RValue::Op { args, .. } => args,
        RValue::Phi(_) | RValue::ImmI(_) | RValue::ImmF(_) => &[],
    }
}

fn stmt_opcode(stmt: &IRStatement) -> &str {
    match &stmt.value {
        RValue::Op { opcode, .. } => opcode,
        RValue::Phi(_) => "phi",
        RValue::ImmI(_) => "immi",
        RValue::ImmF(_) => "immf",
    }
}

fn atomic_func_name(opcode: &str) -> &'static str {
    if opcode.contains(".POPC.INC") || opcode.contains(".ADD") {
        "atomicAdd"
    } else if opcode.contains(".MIN") {
        "atomicMin"
    } else if opcode.contains(".MAX") {
        "atomicMax"
    } else if opcode.contains(".CAS") {
        "atomicCAS"
    } else if opcode.contains(".EXCH") {
        "atomicExch"
    } else if opcode.contains(".AND") {
        "atomicAnd"
    } else if opcode.contains(".OR") {
        "atomicOr"
    } else if opcode.contains(".XOR") {
        "atomicXor"
    } else {
        "atomicOp"
    }
}

fn lower_scalar_expr(expr: &IRExpr) -> Expr {
    lower_scalar_expr_with_analysis(expr, None)
}

fn lower_scalar_expr_with_analysis(expr: &IRExpr, analysis: Option<&FunctionAnalysis>) -> Expr {
    match expr {
        IRExpr::Reg(reg) => analysis
            .and_then(|facts| facts.builtin_by_reg.get(reg))
            .map(|name| Expr::Builtin(name.clone()))
            .unwrap_or_else(|| lower_reg_expr(reg)),
        IRExpr::ImmI(value) => Expr::Imm(value.to_string()),
        IRExpr::ImmF(value) => Expr::Imm(value.to_string()),
        IRExpr::Addr64 { lo, hi } => Expr::Addr64 {
            lo: Box::new(lower_scalar_expr_with_analysis(lo, analysis)),
            hi: Box::new(lower_scalar_expr_with_analysis(hi, analysis)),
        },
        IRExpr::Mem { .. } => Expr::Raw("/*mem*/".to_string()),
        IRExpr::Op { op, args } => lower_ir_op_expr_with_analysis(op, args, analysis),
    }
}

fn lower_basic_stmt(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Stmt {
    let lowered = lower_memory_stmt_detail(block_id, stmt_idx, stmt, analysis)
        .unwrap_or_else(|| lower_non_memory_stmt_with_analysis(stmt, Some(analysis)));
    apply_stmt_predicate(stmt, lowered)
}

#[cfg(test)]
fn lower_non_memory_stmt(stmt: &IRStatement) -> LoweredStmt {
    lower_non_memory_stmt_with_analysis(stmt, None)
}

fn lower_non_memory_stmt_with_analysis(
    stmt: &IRStatement,
    analysis: Option<&FunctionAnalysis>,
) -> LoweredStmt {
    if let Some(lowered) = lower_structural_control_stmt(stmt) {
        return lowered;
    }
    match &stmt.value {
        RValue::Op { opcode, args } => {
            let rhs = lower_op_expr_with_analysis(opcode, args, analysis);
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src: rhs,
                    },
                    predicate_def_idx: Some(0),
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(rhs),
                    predicate_def_idx: None,
                }
            }
        }
        RValue::Phi(args) => {
            let rhs = Expr::CallLike {
                func: "phi".to_string(),
                args: args
                    .iter()
                    .map(|arg| lower_scalar_expr_with_analysis(arg, analysis))
                    .collect(),
            };
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src: rhs,
                    },
                    predicate_def_idx: Some(0),
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(rhs),
                    predicate_def_idx: None,
                }
            }
        }
        RValue::ImmI(value) => {
            let src = Expr::Imm(value.to_string());
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src,
                    },
                    predicate_def_idx: Some(0),
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(src),
                    predicate_def_idx: None,
                }
            }
        }
        RValue::ImmF(value) => {
            let src = Expr::Imm(value.to_string());
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src,
                    },
                    predicate_def_idx: Some(0),
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(src),
                    predicate_def_idx: None,
                }
            }
        }
    }
}

fn lower_structural_control_stmt(stmt: &IRStatement) -> Option<LoweredStmt> {
    let RValue::Op { opcode, .. } = &stmt.value else {
        return None;
    };
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    let lowered_stmt = match mnem {
        "EXIT" | "RET" => Stmt::Return(None),
        "BAR" if opcode.contains("SYNC") => Stmt::ExprStmt(Expr::CallLike {
            func: "__syncthreads".to_string(),
            args: Vec::new(),
        }),
        "WARPSYNC" => Stmt::ExprStmt(Expr::CallLike {
            func: "__syncwarp".to_string(),
            args: Vec::new(),
        }),
        "BRA" | "BSSY" | "BSYNC" | "SSY" | "SYNC" | "NOP" | "DEPBAR" | "MEMBAR" | "FENCE" => {
            Stmt::Empty
        }
        _ => return None,
    };
    Some(LoweredStmt {
        stmt: lowered_stmt,
        predicate_def_idx: None,
    })
}

#[cfg(test)]
fn lower_op_expr(opcode: &str, args: &[IRExpr]) -> Expr {
    lower_op_expr_with_analysis(opcode, args, None)
}

fn lower_op_expr_with_analysis(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    if let Some(expr) = lower_semantic_op_expr(opcode, args, analysis) {
        return expr;
    }
    if let Some(expr) = lower_simple_op_expr(opcode, args, analysis) {
        return expr;
    }
    Expr::CallLike {
        func: opcode.to_string(),
        args: args
            .iter()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis))
            .collect(),
    }
}

fn lower_semantic_op_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    lower_setp_expr(opcode, args, analysis)
        .or_else(|| lower_wide_add_lo_expr(opcode, args, analysis))
        .or_else(|| lower_iabs_expr(opcode, args, analysis))
        .or_else(|| lower_ffma_expr(opcode, args, analysis))
        .or_else(|| lower_fmnmx_expr(opcode, args, analysis))
        .or_else(|| lower_mufu_expr(opcode, args, analysis))
}

fn lower_simple_op_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    match mnem {
        "S2R" | "CS2R" | "S2UR" => args
            .first()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis)),
        "MOV" | "UMOV" | "FMOV" => args
            .first()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis)),
        "IADD" | "IADD3" | "UIADD" | "UIADD3" | "FADD" if !opcode.contains('.') => {
            lower_add_expr(args, analysis)
        }
        _ => None,
    }
}

fn lower_setp_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "ISETP" | "UISETP" | "FSETP" | "DSETP" | "HSETP2") || args.len() < 2 {
        return None;
    }
    let parts = opcode.split('.').collect::<Vec<_>>();
    let cmp = parts
        .iter()
        .skip(1)
        .find_map(|part| setp_comparison_op(part))?;
    let mut lhs = lower_scalar_expr_with_analysis(&args[0], analysis);
    let mut rhs = lower_scalar_expr_with_analysis(&args[1], analysis);
    if mnem == "ISETP" && !parts.iter().any(|part| *part == "U32") && !matches!(cmp, "==" | "!=") {
        lhs = Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(lhs),
        };
        rhs = Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(rhs),
        };
    }
    let cmp_expr = Expr::Binary {
        op: cmp.to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    };
    let Some(combine) = args.get(2) else {
        return Some(cmp_expr);
    };
    if ir_expr_is_true_pred(combine) {
        return Some(cmp_expr);
    }
    if ir_expr_is_false_pred(combine) {
        return Some(Expr::Imm("false".to_string()));
    }
    let combine_op = if parts.iter().any(|part| *part == "OR") {
        "||"
    } else {
        "&&"
    };
    Some(Expr::Binary {
        op: combine_op.to_string(),
        lhs: Box::new(cmp_expr),
        rhs: Box::new(lower_scalar_expr_with_analysis(combine, analysis)),
    })
}

fn setp_comparison_op(part: &str) -> Option<&'static str> {
    match part {
        "LT" => Some("<"),
        "LE" => Some("<="),
        "GT" => Some(">"),
        "GE" => Some(">="),
        "EQ" => Some("=="),
        "NE" => Some("!="),
        _ => None,
    }
}

fn lower_add_expr(args: &[IRExpr], analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    let mut terms = args
        .iter()
        .filter(|expr| !ir_expr_is_zero(expr))
        .map(|expr| lower_scalar_expr_with_analysis(expr, analysis));
    let first = terms.next()?;
    Some(terms.fold(first, |lhs, rhs| Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }))
}

fn lower_iabs_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if !opcode.starts_with("IABS") || args.len() != 1 {
        return None;
    }
    Some(Expr::CallLike {
        func: "abs".to_string(),
        args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
    })
}

fn lower_ffma_expr(opcode: &str, args: &[IRExpr], analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    if opcode != "FFMA" || args.len() != 3 {
        return None;
    }
    Some(Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(Expr::Binary {
            op: "*".to_string(),
            lhs: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
            rhs: Box::new(lower_scalar_expr_with_analysis(&args[1], analysis)),
        }),
        rhs: Box::new(lower_scalar_expr_with_analysis(&args[2], analysis)),
    })
}

fn lower_fmnmx_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if opcode != "FMNMX" || args.len() != 3 {
        return None;
    }
    let fmin = Expr::CallLike {
        func: "fminf".to_string(),
        args: vec![
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[1], analysis),
        ],
    };
    let fmax = Expr::CallLike {
        func: "fmaxf".to_string(),
        args: vec![
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[1], analysis),
        ],
    };
    if ir_expr_is_true_pred(&args[2]) {
        return Some(fmin);
    }
    if ir_expr_is_false_pred(&args[2]) {
        return Some(fmax);
    }
    Some(Expr::Ternary {
        cond: Box::new(lower_scalar_expr_with_analysis(&args[2], analysis)),
        then_expr: Box::new(fmin),
        else_expr: Box::new(fmax),
    })
}

fn lower_mufu_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if args.len() != 1 {
        return None;
    }
    let func = if opcode.starts_with("MUFU.RCP") {
        "rcp_approx"
    } else if opcode.starts_with("MUFU.RSQ") {
        "rsqrtf"
    } else if opcode.starts_with("MUFU.EX2") {
        "exp2f"
    } else if opcode.starts_with("MUFU.LG2") {
        "log2f"
    } else if opcode.starts_with("MUFU.SIN") {
        "sinf"
    } else if opcode.starts_with("MUFU.COS") {
        "cosf"
    } else if opcode.starts_with("MUFU.SQRT") {
        "sqrtf"
    } else {
        return None;
    };
    Some(Expr::CallLike {
        func: func.to_string(),
        args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
    })
}

fn lower_wide_add_lo_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let wide_value = if opcode == "IADD.64" && args.len() == 4 {
        Some(add_expr(
            lower_wide_input_expr(&args[0], &args[2], analysis),
            lower_wide_input_expr(&args[1], &args[3], analysis),
        ))
    } else if matches!(opcode, "IADD3.64" | "UIADD3.64") && args.len() == 6 {
        Some(add_expr(
            add_expr(
                lower_wide_input_expr(&args[0], &args[3], analysis),
                lower_wide_input_expr(&args[1], &args[4], analysis),
            ),
            lower_wide_input_expr(&args[2], &args[5], analysis),
        ))
    } else {
        None
    }?;
    Some(Expr::LaneExtract {
        value: Box::new(wide_value),
        lane: PointerLane::Lo32,
    })
}

fn lower_wide_input_expr(lo: &IRExpr, hi: &IRExpr, analysis: Option<&FunctionAnalysis>) -> Expr {
    if ir_expr_is_zero(hi) {
        return Expr::Cast {
            ty: "uint64_t".to_string(),
            expr: Box::new(Expr::Cast {
                ty: "uint32_t".to_string(),
                expr: Box::new(lower_scalar_expr_with_analysis(lo, analysis)),
            }),
        };
    }
    Expr::Addr64 {
        lo: Box::new(lower_scalar_expr_with_analysis(lo, analysis)),
        hi: Box::new(lower_scalar_expr_with_analysis(hi, analysis)),
    }
}

fn add_expr(lhs: Expr, rhs: Expr) -> Expr {
    if matches!(lhs, Expr::Imm(ref text) if text == "0") {
        return rhs;
    }
    if matches!(rhs, Expr::Imm(ref text) if text == "0") {
        return lhs;
    }
    Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn lower_ir_op_expr_with_analysis(
    op: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    if let Some(expr) = lower_constmem_ir_op_expr(op, args, analysis) {
        return expr;
    }
    if let Some(expr) = lower_builtin_expr(op, args) {
        return expr;
    }
    if let Some(expr) = lower_simple_ir_op_expr(op, args, analysis) {
        return expr;
    }
    Expr::CallLike {
        func: op.to_string(),
        args: args
            .iter()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis))
            .collect(),
    }
}

fn lower_constmem_ir_op_expr(
    op: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if op != "ConstMem" || args.len() != 2 {
        return None;
    }
    let bank = expr_i64(&args[0]).and_then(|value| u32::try_from(value).ok())?;
    let offset = expr_i64(&args[1]).and_then(|value| u32::try_from(value).ok())?;
    let semantic = analysis
        .and_then(|facts| facts.abi_profile)
        .map(|profile| profile.classify_constmem(bank, offset))
        .unwrap_or(ConstMemSemantic::Unknown { bank, offset });
    Some(match analysis {
        Some(facts) => constmem_semantic_expr(&semantic, facts),
        None => Expr::ConstMemSymbol(format!("c[0x{:x}][0x{:x}]", bank, offset)),
    })
}

fn lower_builtin_expr(op: &str, args: &[IRExpr]) -> Option<Expr> {
    if !args.is_empty() {
        return None;
    }
    let name = match op {
        "SR_CTAID.X" => "blockIdx.x",
        "SR_CTAID.Y" => "blockIdx.y",
        "SR_CTAID.Z" => "blockIdx.z",
        "SR_TID.X" => "threadIdx.x",
        "SR_TID.Y" => "threadIdx.y",
        "SR_TID.Z" => "threadIdx.z",
        "SR_NTID.X" => "blockDim.x",
        "SR_NTID.Y" => "blockDim.y",
        "SR_NTID.Z" => "blockDim.z",
        "SR_GRIDID" => "gridId",
        "SR_LANEID" => "laneId",
        "SRZ" => "0",
        _ => return None,
    };
    Some(Expr::Builtin(name.to_string()))
}

fn lower_simple_ir_op_expr(
    op: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    match (op, args) {
        ("!", [arg]) | ("-", [arg]) => Some(Expr::Unary {
            op: op.to_string(),
            arg: Box::new(lower_scalar_expr_with_analysis(arg, analysis)),
        }),
        (binary_op, [lhs, rhs]) if is_simple_binary_op(binary_op) => Some(Expr::Binary {
            op: binary_op.to_string(),
            lhs: Box::new(lower_scalar_expr_with_analysis(lhs, analysis)),
            rhs: Box::new(lower_scalar_expr_with_analysis(rhs, analysis)),
        }),
        _ => None,
    }
}

fn is_simple_binary_op(op: &str) -> bool {
    matches!(
        op,
        "+" | "-"
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
            | ">="
    )
}

fn lower_reg_expr(reg: &crate::ir::RegId) -> Expr {
    match reg.class.as_str() {
        "RZ" | "URZ" => Expr::Imm("0".to_string()),
        "PT" | "UPT" => Expr::Imm("true".to_string()),
        _ if reg.sign < 0 => Expr::Unary {
            op: "-".to_string(),
            arg: Box::new(Expr::Reg(lower_reg_name(reg))),
        },
        _ => Expr::Reg(lower_reg_name(reg)),
    }
}

fn lower_reg_name(reg: &crate::ir::RegId) -> String {
    canonical_reg_ident(reg)
}

fn ir_expr_is_zero(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::ImmI(0) => true,
        IRExpr::Reg(reg) => matches!(reg.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn ir_expr_is_true_pred(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::Reg(reg) if matches!(reg.class.as_str(), "PT" | "UPT"))
}

fn ir_expr_is_false_pred(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::Op { op, args } if args.is_empty() && matches!(op.as_str(), "!PT" | "!UPT"))
        || matches!(expr, IRExpr::Op { op, args } if op == "!" && args.len() == 1 && ir_expr_is_true_pred(&args[0]))
}

fn named_param_word_expr(name: &str) -> Expr {
    if let Some((base, lane)) = PointerLane::parse_named(name) {
        Expr::PtrLane { base, lane }
    } else {
        Expr::Reg(name.to_string())
    }
}

fn rooted_space_byte_offset(
    addr_base: &IRExpr,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<IRExpr> {
    let expected_root = match &access.root {
        AddressRoot::ParamWord(param_idx) => AddressRoot::ParamWord(*param_idx),
        AddressRoot::RegisterBase(reg) => analysis
            .root_by_reg
            .get(reg)
            .cloned()
            .unwrap_or_else(|| AddressRoot::RegisterBase(reg.clone())),
        AddressRoot::SharedObject(name) => AddressRoot::SharedObject(name.clone()),
        AddressRoot::LocalObject(name) => AddressRoot::LocalObject(name.clone()),
        AddressRoot::ConstSymbol(name) => AddressRoot::ConstSymbol(name.clone()),
        AddressRoot::Generic => AddressRoot::Generic,
    };
    let byte_offset = match addr_base {
        IRExpr::Reg(reg) => analysis.byte_offset_by_reg.get(reg).cloned(),
        IRExpr::Addr64 { lo, .. } => lo
            .get_reg()
            .and_then(|reg| analysis.byte_offset_by_reg.get(reg))
            .cloned(),
        _ => None,
    }?;
    Some(expand_rooted_offset_expr(
        &byte_offset,
        &expected_root,
        analysis,
        &mut BTreeSet::new(),
    ))
}

fn expand_rooted_offset_expr(
    expr: &IRExpr,
    expected_root: &AddressRoot,
    analysis: &FunctionAnalysis,
    seen: &mut BTreeSet<crate::ir::RegId>,
) -> IRExpr {
    match expr {
        IRExpr::Reg(reg) if can_expand_rooted_reg(reg, expected_root, analysis) => {
            if !seen.insert(reg.clone()) {
                return IRExpr::Reg(reg.clone());
            }
            let expanded = analysis
                .byte_offset_by_reg
                .get(reg)
                .map(|inner| expand_rooted_offset_expr(inner, expected_root, analysis, seen))
                .unwrap_or_else(|| IRExpr::Reg(reg.clone()));
            seen.remove(reg);
            expanded
        }
        IRExpr::Addr64 { lo, hi } => IRExpr::Addr64 {
            lo: Box::new(expand_rooted_offset_expr(lo, expected_root, analysis, seen)),
            hi: Box::new(expand_rooted_offset_expr(hi, expected_root, analysis, seen)),
        },
        IRExpr::Mem {
            base,
            offset,
            width,
        } => IRExpr::Mem {
            base: Box::new(expand_rooted_offset_expr(
                base,
                expected_root,
                analysis,
                seen,
            )),
            offset: offset.as_ref().map(|expr| {
                Box::new(expand_rooted_offset_expr(
                    expr,
                    expected_root,
                    analysis,
                    seen,
                ))
            }),
            width: *width,
        },
        IRExpr::Op { op, args } => IRExpr::Op {
            op: op.clone(),
            args: args
                .iter()
                .map(|arg| expand_rooted_offset_expr(arg, expected_root, analysis, seen))
                .collect(),
        },
        IRExpr::ImmI(_) | IRExpr::ImmF(_) | IRExpr::Reg(_) => expr.clone(),
    }
}

fn can_expand_rooted_reg(
    reg: &crate::ir::RegId,
    expected_root: &AddressRoot,
    analysis: &FunctionAnalysis,
) -> bool {
    matches!(reg.class.as_str(), "R" | "UR")
        && analysis
            .root_by_reg
            .get(reg)
            .is_some_and(|root| root == expected_root)
}

fn select_memory_result_def(stmt: &IRStatement) -> Option<(usize, &crate::ir::RegId)> {
    stmt.defs
        .iter()
        .enumerate()
        .filter_map(|(idx, def)| {
            let reg = def.get_reg()?;
            (!is_sink_or_predicate_reg(reg)).then_some((idx, reg))
        })
        .next()
}

fn is_sink_or_predicate_reg(reg: &crate::ir::RegId) -> bool {
    matches!(reg.class.as_str(), "PT" | "UPT" | "P" | "UP" | "RZ" | "URZ")
}

fn apply_stmt_predicate(stmt: &IRStatement, lowered: LoweredStmt) -> Stmt {
    let LoweredStmt {
        stmt: lowered_stmt,
        predicate_def_idx,
    } = lowered;
    if matches!(lowered_stmt, Stmt::Empty) {
        return Stmt::Empty;
    }
    let Some(pred) = &stmt.pred else {
        return lowered_stmt;
    };
    if let Some(def_idx) = predicate_def_idx {
        if let Some(old_def) = stmt.pred_old_defs.get(def_idx) {
            if let Stmt::Assign { dst, src } = lowered_stmt {
                return Stmt::Assign {
                    dst,
                    src: Expr::Ternary {
                        cond: Box::new(lower_scalar_expr(pred)),
                        then_expr: Box::new(src),
                        else_expr: Box::new(lower_scalar_expr(old_def)),
                    },
                };
            }
        }
    }
    Stmt::If {
        condition: lower_scalar_expr(pred),
        then_branch: Box::new(lowered_stmt),
        else_branch: None,
    }
}

fn constmem_semantic_expr(semantic: &ConstMemSemantic, analysis: &FunctionAnalysis) -> Expr {
    match semantic {
        ConstMemSemantic::ParamWord {
            param_idx,
            word_idx,
        } => {
            let rendered = analysis
                .abi_aliases
                .render_param_word(*param_idx, *word_idx)
                .unwrap_or_else(|| format!("param_{}", param_idx.saturating_add(*word_idx)));
            named_param_word_expr(&rendered)
        }
        ConstMemSemantic::Builtin(name) => Expr::Builtin((*name).to_string()),
        ConstMemSemantic::AbiInternal(offset) => {
            Expr::ConstMemSymbol(format!("abi_internal_0x{:x}", offset))
        }
        ConstMemSemantic::Unknown { bank, offset } => {
            Expr::ConstMemSymbol(format!("c[0x{:x}][0x{:x}]", bank, offset))
        }
    }
}

fn expr_i64(expr: &IRExpr) -> Option<i64> {
    match expr {
        IRExpr::ImmI(value) => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{analyze_function_ir, build_cfg, build_ssa, decode_sass};

    fn analyze_stmt(
        sass: &str,
        pred: impl Fn(&IRStatement) -> bool,
    ) -> (FunctionAnalysis, usize, usize, IRStatement) {
        let instrs = decode_sass(sass);
        let cfg = build_cfg(instrs.clone());
        let fir = build_ssa(&cfg);
        let analysis = analyze_function_ir(&fir, &instrs, None);
        for block in &fir.blocks {
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                if pred(stmt) {
                    return (analysis, block.id, stmt_idx, stmt.clone());
                }
            }
        }
        panic!("statement not found");
    }

    fn analyze_sass(sass: &str) -> FunctionAnalysis {
        let instrs = decode_sass(sass);
        let cfg = build_cfg(instrs.clone());
        let fir = build_ssa(&cfg);
        analyze_function_ir(&fir, &instrs, None)
    }

    #[test]
    fn lowers_global_param_rooted_loads_to_index_exprs() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E R6, [R4.64+0x4] ;
            /*0030*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("LDG"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(src.render(), "arg0_ptr[1]");
    }

    #[test]
    fn lowers_pointer_arith_global_loads_from_rooted_base_offsets() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ IADD3 R4, R4, 0x4, RZ ;
            /*0030*/ LDG.E R6, [R4.64] ;
            /*0040*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("LDG"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(src.render(), "arg0_ptr[1]");
    }

    #[test]
    fn expands_nested_rooted_offsets_in_global_index_lowering() {
        let rooted = crate::ir::RegId::new("R", 10, 1).with_ssa(0);
        let nested = crate::ir::RegId::new("R", 8, 1).with_ssa(0);
        let mem_expr = IRExpr::Mem {
            base: Box::new(IRExpr::Reg(nested.clone())),
            offset: Some(Box::new(IRExpr::ImmI(4))),
            width: Some(32),
        };
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "LDG.E".to_string(),
                args: vec![mem_expr.clone()],
            },
            pred: None,
            mem_addr_args: Some(vec![mem_expr]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis
            .root_by_reg
            .insert(rooted.clone(), AddressRoot::ParamWord(0));
        analysis.byte_offset_by_reg.insert(
            rooted.clone(),
            IRExpr::Op {
                op: "+".to_string(),
                args: vec![IRExpr::ImmI(32), IRExpr::ImmI(2)],
            },
        );
        analysis
            .root_by_reg
            .insert(nested.clone(), AddressRoot::ParamWord(0));
        analysis.byte_offset_by_reg.insert(
            nested,
            IRExpr::Op {
                op: "+".to_string(),
                args: vec![IRExpr::Reg(rooted), IRExpr::ImmI(64)],
            },
        );
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Global,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(4),
            has_dynamic_offset: true,
            root: AddressRoot::ParamWord(0),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        let rendered = src.render();
        assert!(
            rendered.contains("arg0_ptr[") || rendered.contains("param_0["),
            "expected a rooted param base, got: {rendered}"
        );
        assert!(
            !rendered.contains("r8_0"),
            "nested rooted temp leaked: {rendered}"
        );
        assert!(
            !rendered.contains("r10_0"),
            "rooted temp leaked: {rendered}"
        );
    }

    #[test]
    fn expands_rooted_shared_offsets_before_scaling_indices() {
        let rooted = crate::ir::RegId::new("R", 19, 1).with_ssa(0);
        let mem_expr = IRExpr::Mem {
            base: Box::new(IRExpr::Reg(rooted.clone())),
            offset: Some(Box::new(IRExpr::ImmI(528))),
            width: Some(32),
        };
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "LDS".to_string(),
                args: vec![mem_expr.clone()],
            },
            pred: None,
            mem_addr_args: Some(vec![mem_expr]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis
            .root_by_reg
            .insert(rooted.clone(), AddressRoot::SharedObject("shmem".to_string()));
        analysis.byte_offset_by_reg.insert(
            rooted,
            IRExpr::Op {
                op: "*".to_string(),
                args: vec![
                    IRExpr::Op {
                        op: "SR_TID.X".to_string(),
                        args: Vec::new(),
                    },
                    IRExpr::ImmI(4),
                ],
            },
        );
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(528),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "shmem[threadIdx.x + 132]");
    }

    #[test]
    fn lowers_shared_store_to_indexed_lvalue() {
        let sass = r#"
            /*0000*/ STS [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(dst.render(), "shmem[(r2_0 + 8) / 4]");
    }

    #[test]
    fn lowers_scaled_shared_thread_indices_to_element_offsets() {
        let sass = r#"
            /*0000*/ S2R R7, SR_TID.X ;
            /*0010*/ STS [R7.X4+0x10], R4 ;
            /*0020*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(dst.render(), "shmem[threadIdx.x + 4]");
    }

    #[test]
    fn lowers_dynamic_local_stores_to_explicit_helpers() {
        let sass = r#"
            /*0000*/ STL [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STL"));
        let lowered = lower_basic_stmt(block_id, stmt_idx, &stmt, &analysis);
        let Stmt::ExprStmt(expr) = lowered else {
            panic!("expected explicit helper call");
        };
        assert_eq!(expr.render(), "local_store_bits32(r2_0 + 8, r4_0)");
    }

    #[test]
    fn lowers_shared_atomics_to_explicit_calls() {
        let sass = r#"
            /*0000*/ ATOMS.ADD R0, [R2], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("ATOMS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, src } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(dst.render(), "r0_0");
        assert!(src.render().contains("atomicAdd"));
        assert!(src.render().contains("&shmem[r2_0 / 4]"));
    }

    #[test]
    fn lowers_popc_inc_shared_atomics_to_atomic_add_one() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "ATOMS.POPC.INC".to_string(),
                args: vec![IRExpr::Mem {
                    base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                    offset: None,
                    width: Some(32),
                }],
            },
            pred: None,
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                offset: None,
                width: Some(32),
            }]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Atomic,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "atomicAdd(&shmem[r2 / 4], 1)");
    }

    #[test]
    fn lowers_param_window_loads_to_kernel_param_symbols() {
        let sass = r#"
            /*0000*/ LDC R4, c[0x0][0x160] ;
            /*0010*/ LDC R5, c[0x0][0x164] ;
            /*0020*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) = analyze_stmt(sass, |stmt| {
            stmt_opcode(stmt).starts_with("LDC")
                && stmt
                    .defs
                    .first()
                    .and_then(IRExpr::get_reg)
                    .is_some_and(|reg| reg.idx == 4)
        });
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert!(src.render().starts_with("arg0"));
        assert!(!src.render().contains("ConstMem"));
    }

    #[test]
    fn lowers_predicated_defs_to_ternary_assignments() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(2))],
            value: RValue::Op {
                opcode: "MOV".to_string(),
                args: vec![IRExpr::ImmI(1)],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: None,
            pred_old_defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(1))],
        };
        let lowered = lower_basic_stmt(0, 0, &stmt, &FunctionAnalysis::default());
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "p0 ? 1 : r4_1");
    }

    #[test]
    fn lowers_predicated_atomic_results_with_false_path_value() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(1))],
            value: RValue::Op {
                opcode: "ATOMS.ADD".to_string(),
                args: vec![
                    IRExpr::Mem {
                        base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                        offset: None,
                        width: Some(32),
                    },
                    IRExpr::Reg(crate::ir::RegId::new("R", 4, 1)),
                ],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                offset: None,
                width: Some(32),
            }]),
            pred_old_defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Atomic,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_basic_stmt(0, 0, &stmt, &analysis);
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "p0 ? (atomicAdd(&shmem[r2 / 4], r4)) : r0_0");
    }

    #[test]
    fn lowers_multi_def_atomic_results_to_data_reg_and_preserves_false_path() {
        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(1)),
            ],
            value: RValue::Op {
                opcode: "ATOMG.E.ADD.STRONG.GPU".to_string(),
                args: vec![
                    IRExpr::Mem {
                        base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1))),
                        offset: None,
                        width: Some(64),
                    },
                    IRExpr::Reg(crate::ir::RegId::new("R", 13, 1)),
                ],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 2, 1))),
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1))),
                offset: None,
                width: Some(64),
            }]),
            pred_old_defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
            ],
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Atomic,
            space: CudaMemorySpace::Global,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::RegisterBase(crate::ir::RegId::new("R", 4, 1)),
        });
        let lowered = lower_basic_stmt(0, 0, &stmt, &analysis);
        let Stmt::Assign { dst, src } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(dst.render(), "r7_1");
        assert_eq!(src.render(), "p2 ? (atomicAdd(&((uint32_t*)(r4))[0], r13)) : r7_0");
    }

    #[test]
    fn lowers_exit_and_barrier_ops_without_raw_helpers() {
        let exit_stmt = IRStatement {
            defs: Vec::new(),
            value: RValue::Op {
                opcode: "EXIT".to_string(),
                args: Vec::new(),
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let exit_lowered = lower_basic_stmt(0, 0, &exit_stmt, &FunctionAnalysis::default());
        assert_eq!(exit_lowered.render_with_indent(0), "if (p0) return;\n");

        let barrier_stmt = IRStatement {
            defs: Vec::new(),
            value: RValue::Op {
                opcode: "BAR.SYNC".to_string(),
                args: Vec::new(),
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let barrier_lowered = lower_basic_stmt(0, 0, &barrier_stmt, &FunctionAnalysis::default());
        assert_eq!(barrier_lowered.render_with_indent(0), "__syncthreads();\n");
    }

    #[test]
    fn lowers_structured_if_and_memory_load_body() {
        let analysis = FunctionAnalysis::default();
        let structured = StructuredStatement::If {
            condition_block_id: 0,
            condition_expr: IRExpr::Reg(crate::ir::RegId::new("P", 0, 1)),
            then_branch: Box::new(StructuredStatement::BasicBlock {
                block_id: 1,
                stmts: vec![IRStatement {
                    defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1))],
                    value: RValue::Op {
                        opcode: "MOV".to_string(),
                        args: vec![IRExpr::ImmI(1)],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                }],
            }),
            else_branch: None,
        };
        let lowered = lower_structured_stmt(&structured, &analysis);
        let rendered = lowered.render_with_indent(0);
        assert!(rendered.contains("if (p0)"));
        assert!(rendered.contains("r4 = 1;"));
    }

    #[test]
    fn lowers_iadd3_to_addition_chain() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "IADD3".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                    IRExpr::ImmI(4),
                    IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let Stmt::Assign { src, .. } = lowered.stmt else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "r4_0 + 4");
    }

    #[test]
    fn lowers_wide_add_helpers_without_calllike_leaks() {
        let iadd64 = lower_op_expr(
            "IADD.64",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
            ],
        );
        assert!(
            !iadd64.render().contains("IADD.64("),
            "got: {}",
            iadd64.render()
        );

        let uiadd3_64 = lower_op_expr(
            "UIADD3.64",
            &[
                IRExpr::Reg(crate::ir::RegId::new("UR", 18, 1).with_ssa(0)),
                IRExpr::ImmI(16),
                IRExpr::Reg(crate::ir::RegId::new("URZ", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("UR", 19, 1).with_ssa(0)),
                IRExpr::ImmI(0),
                IRExpr::Reg(crate::ir::RegId::new("URZ", 0, 1)),
            ],
        );
        assert!(
            !uiadd3_64.render().contains("UIADD3.64("),
            "got: {}",
            uiadd3_64.render()
        );
    }

    #[test]
    fn keeps_modifier_bearing_add_ops_explicit() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "IADD3.X".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                    IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                    IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let Stmt::Assign { src, .. } = lowered.stmt else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "IADD3.X(r2_0, r3_0, 0)");
    }

    #[test]
    fn lowers_special_register_and_setp_ops_semantically() {
        let s2r = lower_op_expr(
            "S2R",
            &[IRExpr::Op {
                op: "SR_TID.X".to_string(),
                args: Vec::new(),
            }],
        );
        assert_eq!(s2r.render(), "threadIdx.x");

        let setp = lower_op_expr(
            "ISETP.GE.AND",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::ImmI(1),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(setp.render(), "(int32_t)(r2_0) >= (int32_t)(1)");
    }

    #[test]
    fn lowers_constmem_scalars_to_param_aliases_and_builtin_symbols() {
        let analysis = analyze_sass(
            "/*0000*/ IABS R5, c[0x0][0x164] ;\n\
             /*0010*/ EXIT ;\n",
        );
        let param = lower_ir_op_expr_with_analysis(
            "ConstMem",
            &[IRExpr::ImmI(0), IRExpr::ImmI(0x164)],
            Some(&analysis),
        );
        assert_eq!(param.render(), "arg1");

        let legacy_analysis = FunctionAnalysis {
            abi_profile: Some(crate::abi::AbiProfile::legacy_param_140()),
            ..FunctionAnalysis::default()
        };
        let builtin = lower_ir_op_expr_with_analysis(
            "ConstMem",
            &[IRExpr::ImmI(0), IRExpr::ImmI(0x0)],
            Some(&legacy_analysis),
        );
        assert_eq!(builtin.render(), "blockDim.x");
    }

    #[test]
    fn wraps_unresolved_register_bases_before_indexing() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "LDG.E".to_string(),
                args: vec![IRExpr::Mem {
                    base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))),
                    offset: None,
                    width: Some(32),
                }],
            },
            pred: None,
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))),
                offset: None,
                width: Some(32),
            }]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Global,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::RegisterBase(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "((uint32_t*)(r4_0))[0]");
    }

    #[test]
    fn lowers_ffma_fmnmx_and_mufu_ops_semantically() {
        let ffma = lower_op_expr(
            "FFMA",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
            ],
        );
        assert_eq!(ffma.render(), "r1_0 * r2_0 + r3_0");

        let fmnmx = lower_op_expr(
            "FMNMX",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Op {
                    op: "!PT".to_string(),
                    args: Vec::new(),
                },
            ],
        );
        assert_eq!(fmnmx.render(), "fmaxf(r1_0, r2_0)");

        let mufu = lower_op_expr(
            "MUFU.EX2",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))],
        );
        assert_eq!(mufu.render(), "exp2f(r4_0)");
    }

    #[test]
    fn lowers_iabs_and_mufu_rcp_semantically() {
        let analysis = analyze_sass(
            "/*0000*/ IABS R5, c[0x0][0x164] ;\n\
             /*0010*/ MUFU.RCP R6, R5 ;\n\
             /*0020*/ EXIT ;\n",
        );
        let iabs = lower_op_expr_with_analysis(
            "IABS",
            &[IRExpr::Op {
                op: "ConstMem".to_string(),
                args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x164)],
            }],
            Some(&analysis),
        );
        assert_eq!(iabs.render(), "abs(arg1)");

        let rcp = lower_op_expr_with_analysis(
            "MUFU.RCP",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0))],
            Some(&analysis),
        );
        assert_eq!(rcp.render(), "rcp_approx(r5_0)");
    }

    #[test]
    fn keeps_modifier_bearing_ffma_and_fmnmx_explicit_until_modeled() {
        let ffma = lower_op_expr(
            "FFMA.RM",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
            ],
        );
        assert_eq!(ffma.render(), "FFMA.RM(r1_0, r2_0, r3_0)");

        let fmnmx = lower_op_expr(
            "FMNMX.NAN",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(fmnmx.render(), "FMNMX.NAN(r1_0, r2_0, true)");
    }

    #[test]
    fn lowers_ir_binary_ops_to_ast_binary_exprs() {
        let expr = IRExpr::Op {
            op: "*".to_string(),
            args: vec![
                IRExpr::Op {
                    op: "+".to_string(),
                    args: vec![
                        IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                        IRExpr::ImmI(1),
                    ],
                },
                IRExpr::ImmI(4),
            ],
        };
        assert_eq!(lower_scalar_expr(&expr).render(), "(r2_0 + 1) * 4");
    }
}
