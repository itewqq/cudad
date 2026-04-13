use std::collections::HashMap;

use crate::abi::AbiArgAliases;
use crate::ast::{Expr, LValue};
use crate::ir::FunctionIR;
use crate::name_recovery::RecoveredSymbol;
use crate::semantic_lift::SemanticLiftResult;
use crate::type_inference::infer_types;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedMemoryDecls {
    pub shmem_u8: bool,
    pub shmem_words: bool,
}

pub fn collect_shared_memory_decls(lifted: Option<&SemanticLiftResult>) -> SharedMemoryDecls {
    let Some(lifted) = lifted else {
        return SharedMemoryDecls::default();
    };

    let mut usage = SharedMemoryDecls::default();
    for stmt in lifted.by_def.values() {
        visit_lvalue(&stmt.dest, &mut usage);
        if let Some(pred) = &stmt.pred {
            visit_expr(pred, &mut usage);
        }
        visit_expr(&stmt.rhs, &mut usage);
        if let Some(old) = &stmt.pred_old_val {
            visit_expr(old, &mut usage);
        }
    }
    usage
}

pub fn infer_recovered_name_types(
    function_ir: &FunctionIR,
    recovered_symbols: &[RecoveredSymbol],
) -> HashMap<String, &'static str> {
    let inferred = infer_types(function_ir);
    recovered_symbols
        .iter()
        .filter_map(|symbol| {
            inferred
                .get(&symbol.reg_base)
                .map(|ty| (symbol.name.clone(), ty.to_c_type()))
        })
        .collect()
}

pub fn render_typed_structured_output(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
    local_decls: &[String],
    recovered_symbols: Option<&[RecoveredSymbol]>,
    name_type_map: &HashMap<String, &'static str>,
    shared_memory: SharedMemoryDecls,
) -> String {
    let params = aliases
        .map(|a| a.render_typed_param_list())
        .unwrap_or_default();
    let sig = if params.is_empty() {
        "__global__ void kernel(void)".to_string()
    } else {
        format!("__global__ void kernel({})", params.join(", "))
    };

    let mut decls = Vec::new();
    if shared_memory.shmem_u8 {
        decls.push("__shared__ uint8_t shmem_u8[256];".to_string());
    }
    if shared_memory.shmem_words {
        decls.push("__shared__ uint32_t shmem[];".to_string());
    }

    match recovered_symbols {
        None => decls.extend(local_decls.iter().cloned()),
        Some(symbols) => {
            for symbol in symbols {
                let ty = preferred_symbol_type(symbol, name_type_map);
                decls.push(render_decl(ty, &symbol.name, symbol.live_in));
            }
        }
    }

    let mut out = String::new();
    out.push_str(&sig);
    out.push_str(" {\n");
    for decl in &decls {
        out.push_str("  ");
        out.push_str(decl);
        out.push('\n');
    }
    if !decls.is_empty() {
        out.push('\n');
    }
    for line in code_output.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

fn render_decl(ty: &str, name: &str, live_in: bool) -> String {
    if live_in {
        format!("{} {}; // live-in", ty, name)
    } else {
        format!("{} {};", ty, name)
    }
}

fn preferred_symbol_type(
    symbol: &RecoveredSymbol,
    name_type_map: &HashMap<String, &'static str>,
) -> &'static str {
    match symbol.reg_base.0.as_str() {
        "P" | "UP" => return "bool",
        _ => {}
    }
    if is_semantic_u32_name(&symbol.name) || is_param_lane_name(&symbol.name) {
        return "uint32_t";
    }
    name_type_map
        .get(&symbol.name)
        .copied()
        .unwrap_or("uint32_t")
}

fn is_semantic_u32_name(name: &str) -> bool {
    const SEEDS: &[&str] = &[
        "tid_x",
        "tid_y",
        "tid_z",
        "ctaid_x",
        "ctaid_y",
        "ctaid_z",
        "block_dim_x",
        "block_dim_y",
        "block_dim_z",
        "grid_dim_x",
        "grid_dim_y",
        "grid_dim_z",
        "lane_id",
        "cga_cta_id",
    ];
    SEEDS.iter().any(|seed| {
        name == *seed
            || name.strip_prefix(seed).is_some_and(|suffix| {
                suffix.starts_with('_') && suffix[1..].chars().all(|c| c.is_ascii_digit())
            })
    })
}

fn is_param_lane_name(name: &str) -> bool {
    (name.starts_with("arg") && name.ends_with("_ptr_lo32"))
        || (name.starts_with("arg") && name.ends_with("_ptr_hi32"))
        || (name.starts_with("arg") && name.ends_with("_u64_lo32"))
        || (name.starts_with("arg") && name.ends_with("_u64_hi32"))
}

fn visit_lvalue(lvalue: &LValue, usage: &mut SharedMemoryDecls) {
    match lvalue {
        LValue::Raw(_) | LValue::Var(_) => {}
        LValue::Deref { addr, .. } => visit_expr(addr, usage),
        LValue::Indexed { base, index } => {
            visit_expr(base, usage);
            visit_expr(index, usage);
        }
    }
}

fn visit_expr(expr: &Expr, usage: &mut SharedMemoryDecls) {
    match expr {
        Expr::Builtin(name) if name == "shmem_u8" => usage.shmem_u8 = true,
        Expr::Builtin(name) if name == "shmem" => usage.shmem_words = true,
        Expr::Unary { arg, .. } => visit_expr(arg, usage),
        Expr::Binary { lhs, rhs, .. } => {
            visit_expr(lhs, usage);
            visit_expr(rhs, usage);
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            visit_expr(cond, usage);
            visit_expr(then_expr, usage);
            visit_expr(else_expr, usage);
        }
        Expr::CallLike { args, .. } => {
            for arg in args {
                visit_expr(arg, usage);
            }
        }
        Expr::Load { addr, .. } => visit_expr(addr, usage),
        Expr::Addr64 { lo, hi } => {
            visit_expr(lo, usage);
            visit_expr(hi, usage);
        }
        Expr::Cast { expr, .. } => visit_expr(expr, usage),
        Expr::Index { base, index } => {
            visit_expr(base, usage);
            visit_expr(index, usage);
        }
        Expr::Raw(_) | Expr::Imm(_) | Expr::Reg(_) | Expr::ConstMemSymbol(_) | Expr::Builtin(_) => {
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, LValue};
    use crate::semantic_lift::{DefRef, LiftedStmt, SemanticLiftResult};

    #[test]
    fn render_uses_metadata_driven_decls() {
        let output = render_typed_structured_output(
            "tid_x = threadIdx.x;\nif (b0) return;",
            None,
            &[],
            Some(&[
                RecoveredSymbol {
                    name: "tid_x".to_string(),
                    reg_base: ("R".to_string(), 0),
                    live_in: false,
                    order: 0,
                },
                RecoveredSymbol {
                    name: "b0".to_string(),
                    reg_base: ("P".to_string(), 0),
                    live_in: true,
                    order: 1,
                },
            ]),
            &HashMap::from([(String::from("tid_x"), "uint32_t")]),
            SharedMemoryDecls::default(),
        );

        assert!(output.contains("uint32_t tid_x;"));
        assert!(output.contains("bool b0; // live-in"));
    }

    #[test]
    fn collect_shared_memory_decls_detects_lifted_shared_refs() {
        let mut lifted = SemanticLiftResult::default();
        lifted.by_def.insert(
            DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            },
            LiftedStmt {
                dest: LValue::Indexed {
                    base: Box::new(Expr::Builtin("shmem_u8".to_string())),
                    index: Box::new(Expr::Imm("0".to_string())),
                },
                pred: None,
                rhs: Expr::Index {
                    base: Box::new(Expr::Builtin("shmem".to_string())),
                    index: Box::new(Expr::Imm("1".to_string())),
                },
                pred_old_val: None,
            },
        );

        let usage = collect_shared_memory_decls(Some(&lifted));
        assert!(usage.shmem_u8);
        assert!(usage.shmem_words);
    }
}
