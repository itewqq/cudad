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

use crate::ast::{Expr, LValue, Stmt};
use crate::function_analysis::{AddressRoot, FunctionAnalysis, MemAccessInfo};
use crate::ir::{IRExpr, IRStatement, RValue};
use crate::memory_model::{CudaMemorySpace, MemAccessKind};

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
                dst: LValue::Var(dst.display()),
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
            Some(Stmt::ExprStmt(Expr::CallLike {
                func: func.to_string(),
                args,
            }))
        }
    }
}

pub fn lower_memory_load_expr(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<Expr> {
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

fn lower_base_and_index(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<(Expr, Expr)> {
    let mem_expr = stmt.mem_addr_args.as_ref()?.first()?;
    let IRExpr::Mem { offset, width, .. } = mem_expr else {
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
            AddressRoot::RegisterBase(reg) => Expr::Reg(reg.display()),
            _ => lower_scalar_expr(mem_expr),
        },
        CudaMemorySpace::Const => match &access.root {
            AddressRoot::ConstSymbol(name) => Expr::ConstMemSymbol(name.clone()),
            _ => lower_scalar_expr(mem_expr),
        },
        CudaMemorySpace::Param | CudaMemorySpace::Generic => lower_scalar_expr(mem_expr),
    };
    let index = lower_index_expr(offset.as_deref(), access.bit_width.or(*width));
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
        IRExpr::Reg(reg) => Expr::Reg(reg.display()),
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
        assert_eq!(dst.render(), "shmem[2]");
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
        let Stmt::ExprStmt(expr) = lowered else {
            panic!("expected expr stmt");
        };
        assert!(expr.render().contains("atomicAdd"));
        assert!(expr.render().contains("&shmem[0]"));
    }
}
