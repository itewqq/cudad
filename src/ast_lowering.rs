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

use crate::abi::ConstMemSemantic;
use crate::ast::{Expr, LValue, PointerLane, Stmt};
use crate::backend_names::canonical_reg_ident;
use crate::function_analysis::{AddressRoot, FunctionAnalysis, MemAccessInfo};
use crate::ir::{IRExpr, IRStatement, RValue};
use crate::memory_model::{CudaMemorySpace, MemAccessKind};
use crate::structurizer::{LoopType, StructuredStatement};
use crate::symbol_plan::plan_symbols;

pub fn lower_memory_stmt(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Option<Stmt> {
    let access = lookup_mem_access(analysis, block_id, stmt_idx)?;
    match access.kind {
        MemAccessKind::Load => {
            let dst = stmt.defs.first()?.get_reg()?;
            let src = lower_memory_load_expr(stmt, access, analysis)?;
            Some(Stmt::Assign {
                dst: LValue::Var(lower_reg_name(dst)),
                src,
            })
        }
        MemAccessKind::Store => {
            let dst = lower_memory_store_lvalue(stmt, access, analysis)?;
            let src = store_value_expr(stmt)?;
            Some(Stmt::Assign { dst, src })
        }
        MemAccessKind::Atomic | MemAccessKind::Reduction => {
            let func = atomic_func_name(stmt_opcode(stmt));
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
                    .map(lower_scalar_expr),
            );
            let call = Expr::CallLike {
                func: func.to_string(),
                args,
            };
            if let Some(def) = stmt.defs.first().and_then(IRExpr::get_reg) {
                Some(Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(def)),
                    src: call,
                })
            } else {
                Some(Stmt::ExprStmt(call))
            }
        }
    }
}

pub fn lower_structured_function(
    structured: &StructuredStatement,
    analysis: &FunctionAnalysis,
) -> crate::ast::StructuredFunction {
    let body = lower_structured_stmt(structured, analysis);
    let seed = crate::ast::StructuredFunction {
        params: Vec::new(),
        locals: Vec::new(),
        body,
    };
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
            stmts.iter()
                .enumerate()
                .map(|(stmt_idx, stmt)| lower_basic_stmt(*block_id, stmt_idx, stmt, analysis))
                .collect(),
        ),
        StructuredStatement::Sequence(parts) => Stmt::Sequence(
            parts.iter()
                .map(|part| lower_structured_stmt(part, analysis))
                .collect(),
        ),
        StructuredStatement::If {
            condition_expr,
            then_branch,
            else_branch,
            ..
        } => Stmt::If {
            condition: lower_scalar_expr(condition_expr),
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

fn lower_constmem_load_expr(
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<Expr> {
    let stmt_ref = crate::abi::StatementRef {
        block_id: access.block_id,
        stmt_idx: access.stmt_idx,
    };
    let annotation = analysis
        .abi_annotations
        .constmem_by_stmt
        .get(&stmt_ref)
        .and_then(|annotations| annotations.first())?;
    match &annotation.semantic {
        ConstMemSemantic::ParamWord { param_idx, word_idx } => {
            let rendered = analysis
                .abi_aliases
                .render_param_word(*param_idx, *word_idx)
                .unwrap_or_else(|| format!("param_{}", param_idx.saturating_add(*word_idx)));
            Some(named_param_word_expr(&rendered))
        }
        ConstMemSemantic::Builtin(name) => Some(Expr::Builtin((*name).to_string())),
        ConstMemSemantic::AbiInternal(offset) => {
            Some(Expr::ConstMemSymbol(format!("abi_internal_0x{:x}", offset)))
        }
        ConstMemSemantic::Unknown { bank, offset } => {
            Some(Expr::ConstMemSymbol(format!("c[0x{:x}][0x{:x}]", bank, offset)))
        }
    }
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
            AddressRoot::ParamWord(param_idx) => Expr::Reg(global_param_base_name(*param_idx, analysis)),
            AddressRoot::RegisterBase(reg) => lower_reg_expr(reg),
            _ => lower_scalar_expr(mem_expr),
        },
        CudaMemorySpace::Const => match &access.root {
            AddressRoot::ConstSymbol(name) => Expr::ConstMemSymbol(name.clone()),
            _ => lower_scalar_expr(mem_expr),
        },
        CudaMemorySpace::Param | CudaMemorySpace::Generic => lower_scalar_expr(mem_expr),
    };
    let index = match access.space {
        CudaMemorySpace::Shared | CudaMemorySpace::Local => {
            lower_element_index_expr(addr_base.as_ref(), offset.as_deref(), access.bit_width.or(*width))
        }
        CudaMemorySpace::Global => lower_global_index_expr(
            addr_base.as_ref(),
            offset.as_deref(),
            access,
            analysis,
            access.bit_width.or(*width),
        ),
        _ => lower_index_expr(offset.as_deref(), access.bit_width.or(*width)),
    };
    Some((base, index))
}

fn lower_index_expr(offset: Option<&IRExpr>, bit_width: Option<u32>) -> Expr {
    match offset {
        None => Expr::Imm("0".to_string()),
        Some(IRExpr::ImmI(value)) => {
            let bytes = bit_width.and_then(|bits| bits.checked_div(8)).filter(|bytes| *bytes > 0);
            if let Some(bytes) = bytes {
                if value % i64::from(bytes) == 0 {
                    return Expr::Imm((value / i64::from(bytes)).to_string());
                }
            }
            Expr::Imm(value.to_string())
        }
        Some(expr) => lower_scalar_expr(expr),
    }
}

fn lower_element_index_expr(base: &IRExpr, offset: Option<&IRExpr>, bit_width: Option<u32>) -> Expr {
    let byte_expr = combine_byte_offset_expr(base, offset);
    scale_index_expr(byte_expr, bit_width)
}

fn combine_byte_offset_expr(base: &IRExpr, offset: Option<&IRExpr>) -> Expr {
    let mut terms = Vec::new();
    if !ir_expr_is_zero(base) {
        terms.push(lower_scalar_expr(base));
    }
    if let Some(offset) = offset.filter(|expr| !ir_expr_is_zero(expr)) {
        terms.push(lower_scalar_expr(offset));
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
    let bytes = bit_width.and_then(|bits| bits.checked_div(8)).filter(|bytes| *bytes > 1);
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
    Expr::Binary {
        op: "/".to_string(),
        lhs: Box::new(expr),
        rhs: Box::new(Expr::Imm(bytes.to_string())),
    }
}

fn lower_global_index_expr(
    addr_base: &IRExpr,
    offset: Option<&IRExpr>,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
    bit_width: Option<u32>,
) -> Expr {
    let Some(rooted_offset) = rooted_global_byte_offset(addr_base, access, analysis) else {
        return lower_index_expr(offset, bit_width);
    };
    let byte_expr = combine_byte_offset_expr(&rooted_offset, offset);
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

fn store_value_expr(stmt: &IRStatement) -> Option<Expr> {
    stmt_args(stmt)
        .iter()
        .find(|expr| !matches!(expr, IRExpr::Mem { .. }))
        .map(lower_scalar_expr)
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
    if opcode.contains(".ADD") {
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
    match expr {
        IRExpr::Reg(reg) => lower_reg_expr(reg),
        IRExpr::ImmI(value) => Expr::Imm(value.to_string()),
        IRExpr::ImmF(value) => Expr::Imm(value.to_string()),
        IRExpr::Addr64 { lo, hi } => Expr::CallLike {
            func: "addr64".to_string(),
            args: vec![lower_scalar_expr(lo), lower_scalar_expr(hi)],
        },
        IRExpr::Mem { .. } => Expr::Raw("/*mem*/".to_string()),
        IRExpr::Op { op, args } => Expr::CallLike {
            func: op.clone(),
            args: args.iter().map(lower_scalar_expr).collect(),
        },
    }
}

fn lower_basic_stmt(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Stmt {
    let lowered = lower_memory_stmt(block_id, stmt_idx, stmt, analysis)
        .unwrap_or_else(|| lower_non_memory_stmt(stmt));
    apply_stmt_predicate(stmt, lowered)
}

fn lower_non_memory_stmt(stmt: &IRStatement) -> Stmt {
    match &stmt.value {
        RValue::Op { opcode, args } => {
            let rhs = lower_op_expr(opcode, args);
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(def)),
                    src: rhs,
                }
            } else {
                Stmt::ExprStmt(rhs)
            }
        }
        RValue::Phi(args) => {
            let rhs = Expr::CallLike {
                func: "phi".to_string(),
                args: args.iter().map(lower_scalar_expr).collect(),
            };
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(def)),
                    src: rhs,
                }
            } else {
                Stmt::ExprStmt(rhs)
            }
        }
        RValue::ImmI(value) => {
            let src = Expr::Imm(value.to_string());
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(def)),
                    src,
                }
            } else {
                Stmt::ExprStmt(src)
            }
        }
        RValue::ImmF(value) => {
            let src = Expr::Imm(value.to_string());
            if let Some(def) = stmt.defs.first().and_then(|expr| expr.get_reg()) {
                Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(def)),
                    src,
                }
            } else {
                Stmt::ExprStmt(src)
            }
        }
    }
}

fn lower_op_expr(opcode: &str, args: &[IRExpr]) -> Expr {
    if let Some(expr) = lower_simple_op_expr(opcode, args) {
        return expr;
    }
    Expr::CallLike {
        func: opcode.to_string(),
        args: args.iter().map(lower_scalar_expr).collect(),
    }
}

fn lower_simple_op_expr(opcode: &str, args: &[IRExpr]) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    match mnem {
        "MOV" | "UMOV" | "FMOV" => args.first().map(lower_scalar_expr),
        "IADD" | "IADD3" | "UIADD" | "UIADD3" | "FADD" => lower_add_expr(args),
        _ => None,
    }
}

fn lower_add_expr(args: &[IRExpr]) -> Option<Expr> {
    let mut terms = args
        .iter()
        .filter(|expr| !ir_expr_is_zero(expr))
        .map(lower_scalar_expr);
    let first = terms.next()?;
    Some(terms.fold(first, |lhs, rhs| Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }))
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

fn named_param_word_expr(name: &str) -> Expr {
    if let Some((base, lane)) = PointerLane::parse_named(name) {
        Expr::PtrLane { base, lane }
    } else {
        Expr::Reg(name.to_string())
    }
}

fn rooted_global_byte_offset(
    addr_base: &IRExpr,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<IRExpr> {
    match access.root {
        AddressRoot::ParamWord(_) | AddressRoot::RegisterBase(_) => match addr_base {
            IRExpr::Reg(reg) => analysis.byte_offset_by_reg.get(reg).cloned(),
            IRExpr::Addr64 { lo, .. } => lo
                .get_reg()
                .and_then(|reg| analysis.byte_offset_by_reg.get(reg))
                .cloned(),
            _ => None,
        },
        _ => None,
    }
}

fn apply_stmt_predicate(stmt: &IRStatement, lowered: Stmt) -> Stmt {
    let Some(pred) = &stmt.pred else {
        return lowered;
    };
    if stmt.defs.len() == 1 && !stmt.pred_old_defs.is_empty() {
        if let Stmt::Assign { dst, src } = lowered {
            return Stmt::Assign {
                dst,
                src: Expr::Ternary {
                    cond: Box::new(lower_scalar_expr(pred)),
                    then_expr: Box::new(src),
                    else_expr: Box::new(lower_scalar_expr(&stmt.pred_old_defs[0])),
                },
            };
        }
    }
    Stmt::If {
        condition: lower_scalar_expr(pred),
        then_branch: Box::new(lowered),
        else_branch: None,
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

    #[test]
    fn lowers_global_param_rooted_loads_to_index_exprs() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E R6, [R4.64+0x4] ;
            /*0030*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) = analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("LDG"));
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
    fn lowers_shared_store_to_indexed_lvalue() {
        let sass = r#"
            /*0000*/ STS [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) = analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(dst.render(), "shmem[(r2_0 + 8) / 4]");
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
    fn lowers_param_window_loads_to_kernel_param_symbols() {
        let sass = r#"
            /*0000*/ LDC R4, c[0x0][0x160] ;
            /*0010*/ LDC R5, c[0x0][0x164] ;
            /*0020*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("LDC") && stmt.defs.first().and_then(IRExpr::get_reg).is_some_and(|reg| reg.idx == 4));
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
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_basic_stmt(0, 0, &stmt, &analysis);
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "p0 ? (atomicAdd(&shmem[r2 / 4], r4)) : r0_0");
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
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "r4_0 + 4");
    }
}
