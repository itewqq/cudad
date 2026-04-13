//! IR-level sparse constant propagation for SSA-form `FunctionIR`.
//!
//! This pass runs after SSA construction and before structurization.
//! It identifies definitions whose value is a known constant (immediate
//! or a copy of an immediate through identity-like opcodes) and replaces
//! all uses of that SSA register with the literal value.
//!
//! Patterns recognized:
//!   - `RValue::ImmI(v)` / `RValue::ImmF(v)` — direct immediates
//!   - `IMAD.MOV.U32  Rdst, RZ, RZ, <imm>` — identity move (0*0 + imm)
//!   - `MOV Rdst, <imm>` — register move of immediate
//!
//! After propagation, an IR-DCE pass should be re-run to clean up the
//! now-dead constant definitions.

use std::collections::HashMap;

use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue, RegId};

/// Run sparse constant propagation on `fir`, returning a new `FunctionIR`
/// with constant uses inlined.
pub fn ir_constprop(fir: &FunctionIR) -> FunctionIR {
    // ---- Step 1: Identify constant definitions ----
    let mut const_map: HashMap<RegId, IRExpr> = HashMap::new();

    for block in &fir.blocks {
        for stmt in &block.stmts {
            // Only single-def statements can be propagated.
            if stmt.defs.len() != 1 {
                continue;
            }
            // Skip predicated instructions — the def value depends on the predicate.
            if stmt.pred.is_some() {
                continue;
            }
            let def_reg = match stmt.defs[0].get_reg() {
                Some(r) if !is_immutable_reg(r) => r.clone(),
                _ => continue,
            };

            if let Some(const_expr) = extract_constant_value(&stmt.value) {
                const_map.insert(def_reg, const_expr);
            }
        }
    }

    if const_map.is_empty() {
        return fir.clone();
    }

    // ---- Step 2: Replace uses of constant registers with literals ----
    let mut new_blocks = Vec::with_capacity(fir.blocks.len());
    for block in &fir.blocks {
        let mut new_stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            new_stmts.push(propagate_in_stmt(stmt, &const_map));
        }
        new_blocks.push(IRBlock {
            id: block.id,
            start_addr: block.start_addr,
            irdst: block.irdst.clone(),
            stmts: new_stmts,
        });
    }

    FunctionIR { blocks: new_blocks }
}

/// If `value` defines a known constant, return the constant as an `IRExpr`.
fn extract_constant_value(value: &RValue) -> Option<IRExpr> {
    match value {
        RValue::ImmI(v) => Some(IRExpr::ImmI(*v)),
        RValue::ImmF(v) => Some(IRExpr::ImmF(*v)),
        RValue::Op { opcode, args } => {
            // IMAD.MOV.U32 Rdst, RZ, RZ, <imm>  →  imm (0 * 0 + imm)
            if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
                if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
                    if let Some(c) = as_immediate(&args[2]) {
                        return Some(c);
                    }
                }
            }
            // MOV Rdst, <imm>
            if (opcode == "MOV" || opcode.starts_with("MOV.")) && args.len() == 1 {
                if let Some(c) = as_immediate(&args[0]) {
                    return Some(c);
                }
            }
            None
        }
        RValue::Phi(_) => None,
    }
}

fn propagate_in_stmt(stmt: &IRStatement, const_map: &HashMap<RegId, IRExpr>) -> IRStatement {
    IRStatement {
        defs: stmt.defs.clone(),
        value: propagate_in_rvalue(&stmt.value, const_map),
        pred: stmt.pred.as_ref().map(|p| propagate_in_expr(p, const_map)),
        mem_addr_args: stmt.mem_addr_args.as_ref().map(|args| {
            args.iter()
                .map(|a| propagate_in_expr(a, const_map))
                .collect()
        }),
        pred_old_defs: stmt
            .pred_old_defs
            .iter()
            .map(|d| propagate_in_expr(d, const_map))
            .collect(),
    }
}

fn propagate_in_rvalue(value: &RValue, const_map: &HashMap<RegId, IRExpr>) -> RValue {
    match value {
        RValue::Op { opcode, args } => RValue::Op {
            opcode: opcode.clone(),
            args: args
                .iter()
                .map(|a| propagate_in_expr(a, const_map))
                .collect(),
        },
        RValue::Phi(args) => RValue::Phi(
            args.iter()
                .map(|a| propagate_in_expr(a, const_map))
                .collect(),
        ),
        RValue::ImmI(_) | RValue::ImmF(_) => value.clone(),
    }
}

fn propagate_in_expr(expr: &IRExpr, const_map: &HashMap<RegId, IRExpr>) -> IRExpr {
    match expr {
        IRExpr::Reg(r) => {
            // Look up the SSA register (normalize sign away for lookup).
            let lookup_key = RegId {
                class: r.class.clone(),
                idx: r.idx,
                sign: r.sign.abs(),
                ssa: r.ssa,
            };
            if let Some(const_val) = const_map.get(&lookup_key) {
                // If the register is used with negative sign and the constant is
                // an integer, negate it.
                if r.sign < 0 {
                    match const_val {
                        IRExpr::ImmI(v) => return IRExpr::ImmI(-v),
                        IRExpr::ImmF(v) => return IRExpr::ImmF(-v),
                        _ => {}
                    }
                }
                return const_val.clone();
            }
            expr.clone()
        }
        IRExpr::Mem {
            base,
            offset,
            width,
        } => IRExpr::Mem {
            base: Box::new(propagate_in_expr(base, const_map)),
            offset: offset
                .as_ref()
                .map(|o| Box::new(propagate_in_expr(o, const_map))),
            width: *width,
        },
        IRExpr::Op { op, args } => IRExpr::Op {
            op: op.clone(),
            args: args
                .iter()
                .map(|a| propagate_in_expr(a, const_map))
                .collect(),
        },
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => expr.clone(),
    }
}

fn is_zero_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 0,
        IRExpr::ImmF(f) => *f == 0.0,
        IRExpr::Reg(r) => matches!(r.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn as_immediate(e: &IRExpr) -> Option<IRExpr> {
    match e {
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => Some(e.clone()),
        _ => None,
    }
}

fn is_immutable_reg(r: &RegId) -> bool {
    matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, decode_sass};

    #[test]
    fn constprop_inlines_imad_mov_immediate() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x5 ;
            /*0010*/ IADD3 R2, R1, R0, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let propagated = ir_constprop(&fir);

        // After constprop, the IADD3 should use ImmI(5) instead of R1
        let iadd3_stmt = propagated
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| matches!(&s.value, RValue::Op { opcode, .. } if opcode == "IADD3"))
            .expect("expected IADD3 statement");
        if let RValue::Op { args, .. } = &iadd3_stmt.value {
            // First arg should now be ImmI(5) instead of Reg(R1)
            assert!(
                matches!(args[0], IRExpr::ImmI(5)),
                "expected ImmI(5), got {:?}",
                args[0]
            );
        }
    }

    #[test]
    fn constprop_does_not_propagate_predicated_defs() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ @P0 IMAD.MOV.U32 R1, RZ, RZ, 0x5 ;
            /*0020*/ IADD3 R2, R1, R0, RZ ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let propagated = ir_constprop(&fir);

        // Predicated def should NOT be propagated
        let iadd3_stmt = propagated
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| matches!(&s.value, RValue::Op { opcode, .. } if opcode == "IADD3"))
            .expect("expected IADD3 statement");
        if let RValue::Op { args, .. } = &iadd3_stmt.value {
            // First arg should still be a register, not inlined constant
            assert!(
                matches!(args[0], IRExpr::Reg(_)),
                "predicated def should not be propagated, got {:?}",
                args[0]
            );
        }
    }
}
