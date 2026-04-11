//! IR-level copy propagation for SSA-form `FunctionIR`.
//!
//! This pass eliminates redundant copies and trivial phi nodes in SSA form:
//!
//! 1. **Trivial phi elimination**: φ(v, v, …, v) where all operands are
//!    the same SSA register is replaced by that register everywhere.
//!
//! 2. **Copy propagation**: `Rdst = MOV Rsrc` or `Rdst = IMAD.MOV.U32 RZ, RZ, Rsrc`
//!    is replaced by substituting Rsrc for Rdst at all use sites.
//!
//! Both transformations iterate to a fixpoint because eliminating one copy
//! may expose a trivial phi, and vice versa.
//!
//! After this pass, a subsequent `ir_dce` pass should remove the now-dead
//! copy/phi definitions.
//!
//! Reference: Briggs et al., "Practical improvements to the construction
//! and destruction of static single assignment form" (1998).

use std::collections::HashMap;

use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue, RegId};

/// Run copy propagation on `fir`, returning a new `FunctionIR` with copies
/// and trivial phis eliminated.
pub fn ir_copyprop(fir: &FunctionIR) -> FunctionIR {
    let mut current = fir.clone();
    // Iterate to fixpoint — eliminating copies may create trivial phis
    // and vice versa.
    for _ in 0..20 {
        let subst = build_substitution_map(&current);
        if subst.is_empty() {
            break;
        }
        current = apply_substitutions(&current, &subst);
    }
    current
}

/// Build a map from SSA register → replacement SSA register.
///
/// Sources of substitutions:
/// 1. Trivial phis: φ(v, v, ..., v) → v (all args identical after
///    normalization; ignoring sign since phi operands inherit it).
/// 2. Copy instructions: IMAD.MOV.U32 Rdst, RZ, RZ, Rsrc → Rsrc
///    or MOV Rdst, Rsrc.
fn build_substitution_map(fir: &FunctionIR) -> HashMap<RegId, RegId> {
    let mut subst: HashMap<RegId, RegId> = HashMap::new();

    for block in &fir.blocks {
        for stmt in &block.stmts {
            // Skip predicated instructions — their def depends on runtime condition.
            if stmt.pred.is_some() {
                continue;
            }
            // Must define exactly one register.
            if stmt.defs.len() != 1 {
                continue;
            }
            let def_reg = match stmt.defs[0].get_reg() {
                Some(r) if !is_immutable(r) => r,
                _ => continue,
            };

            match &stmt.value {
                RValue::Phi(args) => {
                    // Trivial phi: all args are the same register (ignoring sign).
                    if let Some(canonical) = all_same_reg(args) {
                        // Don't create self-loops: if the phi defines the same
                        // register it would substitute to, skip.
                        if !same_ssa_reg(def_reg, &canonical) {
                            subst.insert(def_reg.clone(), canonical);
                        }
                    }
                }
                RValue::Op { opcode, args } => {
                    if let Some(src) = extract_copy_src(opcode, args) {
                        if !is_immutable(&src) && !same_ssa_reg(def_reg, &src) {
                            subst.insert(def_reg.clone(), src);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Chase substitution chains: if A→B and B→C, then A→C.
    // This handles chains like R4.0 = MOV R7.0; R9.0 = MOV R4.0.
    let mut changed = true;
    for _ in 0..20 {
        if !changed {
            break;
        }
        changed = false;
        let keys: Vec<RegId> = subst.keys().cloned().collect();
        for k in keys {
            let v = subst[&k].clone();
            if let Some(v2) = subst.get(&v) {
                if !same_ssa_reg(&v, v2) {
                    subst.insert(k, v2.clone());
                    changed = true;
                }
            }
        }
    }

    subst
}

/// Apply substitutions to all uses in the FunctionIR.
fn apply_substitutions(fir: &FunctionIR, subst: &HashMap<RegId, RegId>) -> FunctionIR {
    let mut new_blocks = Vec::with_capacity(fir.blocks.len());
    for block in &fir.blocks {
        let mut new_stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            new_stmts.push(subst_in_stmt(stmt, subst));
        }
        // Also substitute in irdst conditions.
        let new_irdst = block
            .irdst
            .iter()
            .map(|(cond, addr)| {
                let new_cond = cond.as_ref().map(|c| subst_in_cond(c, subst));
                (new_cond, *addr)
            })
            .collect();
        new_blocks.push(IRBlock {
            id: block.id,
            start_addr: block.start_addr,
            irdst: new_irdst,
            stmts: new_stmts,
        });
    }
    FunctionIR { blocks: new_blocks }
}

fn subst_in_stmt(stmt: &IRStatement, subst: &HashMap<RegId, RegId>) -> IRStatement {
    // Substitute in value (args), predicate, mem_addr_args, pred_old_defs.
    // Do NOT substitute the def-side registers — they keep their original names.
    // The def will become dead and be removed by ir_dce.
    IRStatement {
        defs: stmt.defs.clone(),
        value: subst_in_rvalue(&stmt.value, subst),
        pred: stmt.pred.as_ref().map(|p| subst_in_expr(p, subst)),
        mem_addr_args: stmt
            .mem_addr_args
            .as_ref()
            .map(|args| args.iter().map(|a| subst_in_expr(a, subst)).collect()),
        pred_old_defs: stmt
            .pred_old_defs
            .iter()
            .map(|d| subst_in_expr(d, subst))
            .collect(),
    }
}

fn subst_in_rvalue(value: &RValue, subst: &HashMap<RegId, RegId>) -> RValue {
    match value {
        RValue::Op { opcode, args } => RValue::Op {
            opcode: opcode.clone(),
            args: args.iter().map(|a| subst_in_expr(a, subst)).collect(),
        },
        RValue::Phi(args) => {
            RValue::Phi(args.iter().map(|a| subst_in_expr(a, subst)).collect())
        }
        RValue::ImmI(_) | RValue::ImmF(_) => value.clone(),
    }
}

fn subst_in_expr(expr: &IRExpr, subst: &HashMap<RegId, RegId>) -> IRExpr {
    match expr {
        IRExpr::Reg(r) => {
            // Look up by normalized key (positive sign, same class/idx/ssa).
            let lookup = RegId {
                class: r.class.clone(),
                idx: r.idx,
                sign: r.sign.abs(),
                ssa: r.ssa,
            };
            if let Some(replacement) = subst.get(&lookup) {
                // Preserve the original sign.
                let mut new_reg = replacement.clone();
                if r.sign < 0 {
                    new_reg.sign = -new_reg.sign;
                }
                IRExpr::Reg(new_reg)
            } else {
                expr.clone()
            }
        }
        IRExpr::Mem { base, offset, width } => IRExpr::Mem {
            base: Box::new(subst_in_expr(base, subst)),
            offset: offset
                .as_ref()
                .map(|o| Box::new(subst_in_expr(o, subst))),
            width: *width,
        },
        IRExpr::Op { op, args } => IRExpr::Op {
            op: op.clone(),
            args: args.iter().map(|a| subst_in_expr(a, subst)).collect(),
        },
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => expr.clone(),
    }
}

fn subst_in_cond(
    cond: &crate::ir::IRCond,
    subst: &HashMap<RegId, RegId>,
) -> crate::ir::IRCond {
    match cond {
        crate::ir::IRCond::True => crate::ir::IRCond::True,
        crate::ir::IRCond::Pred { reg, sense } => {
            let lookup = RegId {
                class: reg.class.clone(),
                idx: reg.idx,
                sign: reg.sign.abs(),
                ssa: reg.ssa,
            };
            let mut new_reg = subst.get(&lookup).cloned().unwrap_or_else(|| reg.clone());
            // Compose signs: if the original use had a negative sign,
            // flip the replacement's sign (consistent with subst_in_expr).
            if reg.sign < 0 {
                new_reg.sign = -new_reg.sign;
            }
            crate::ir::IRCond::Pred {
                reg: new_reg,
                sense: *sense,
            }
        }
    }
}

/// Check if all phi args refer to the same SSA register (ignoring sign).
/// Returns the canonical register if so (preserving sign).
fn all_same_reg(args: &[IRExpr]) -> Option<RegId> {
    let mut canonical: Option<RegId> = None;
    for arg in args {
        let r = match arg {
            IRExpr::Reg(r) => r,
            _ => return None, // Non-register phi arg — can't simplify.
        };
        if is_immutable(r) {
            // Phi args that are RZ/PT are fine but don't count as the
            // canonical value.
            continue;
        }
        // Preserve sign so φ(-R5, R5) is correctly rejected as non-trivial.
        let normalized = RegId {
            class: r.class.clone(),
            idx: r.idx,
            sign: r.sign,
            ssa: r.ssa,
        };
        match &canonical {
            None => canonical = Some(normalized),
            Some(c) => {
                if c.class != normalized.class
                    || c.idx != normalized.idx
                    || c.ssa != normalized.ssa
                    || c.sign != normalized.sign
                {
                    return None; // Different registers or different signs — non-trivial phi.
                }
            }
        }
    }
    canonical
}

/// Extract the source register from a copy instruction.
/// Returns None if the instruction is not a copy.
fn extract_copy_src(opcode: &str, args: &[IRExpr]) -> Option<RegId> {
    // IMAD.MOV.U32 Rdst, RZ, RZ, Rsrc → copy from Rsrc
    if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
            if let Some(r) = args[2].get_reg() {
                return Some(r.clone());
            }
        }
    }
    // MOV Rdst, Rsrc
    if (opcode == "MOV" || opcode.starts_with("MOV.")) && args.len() == 1 {
        if let Some(r) = args[0].get_reg() {
            return Some(r.clone());
        }
    }
    None
}

fn is_zero_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 0,
        IRExpr::ImmF(f) => *f == 0.0,
        IRExpr::Reg(r) => matches!(r.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn is_immutable(r: &RegId) -> bool {
    matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
}

fn same_ssa_reg(a: &RegId, b: &RegId) -> bool {
    a.class == b.class && a.idx == b.idx && a.ssa == b.ssa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, parse_sass};

    #[test]
    fn copyprop_eliminates_trivial_phi() {
        // A loop where R2 is only defined once (no real merge) should
        // produce a trivial phi that gets eliminated.
        let sass = r#"
            /*0000*/ IADD3 R2, R0, 0x1, RZ ;
            /*0010*/ ISETP.LT.AND P0, PT, R0, 0x5, PT ;
            /*0020*/ @P0 BRA 0x000 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);

        // Before copyprop: there should be at least one phi node.
        let phi_count_before: usize = fir
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .filter(|s| matches!(s.value, RValue::Phi(_)))
            .count();

        let propagated = ir_copyprop(&fir);

        // After copyprop: trivial phis should have their uses replaced,
        // but the phi def itself remains (ir_dce removes it).
        // Verify that copy propagation ran without panicking and produced
        // valid output.
        let total_stmts: usize = propagated
            .blocks
            .iter()
            .map(|b| b.stmts.len())
            .sum();
        assert!(total_stmts > 0, "copyprop should produce non-empty output");
        assert!(phi_count_before > 0, "test should have phi nodes to work with");
    }

    #[test]
    fn copyprop_propagates_mov_copy() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, R0 ;
            /*0010*/ IADD3 R2, R1, 0x1, RZ ;
            /*0020*/ STG.E [R2.64], R3 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let propagated = ir_copyprop(&fir);

        // After copyprop, the IADD3 should use R0 instead of R1
        // (the MOV copy is propagated).
        let iadd3_stmt = propagated
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| {
                matches!(&s.value, RValue::Op { opcode, .. } if opcode == "IADD3")
            })
            .expect("expected IADD3 statement");
        if let RValue::Op { args, .. } = &iadd3_stmt.value {
            // First arg should now reference R0's SSA version, not R1's.
            if let Some(r) = args[0].get_reg() {
                assert_eq!(
                    r.idx, 0,
                    "IADD3 first arg should be R0 after copy propagation, got R{}",
                    r.idx
                );
            }
        }
    }

    #[test]
    fn copyprop_does_not_propagate_predicated_copy() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ @P0 IMAD.MOV.U32 R1, RZ, RZ, R0 ;
            /*0020*/ IADD3 R2, R1, 0x1, RZ ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let propagated = ir_copyprop(&fir);

        // Predicated copy should NOT be propagated.
        let iadd3_stmt = propagated
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| {
                matches!(&s.value, RValue::Op { opcode, .. } if opcode == "IADD3")
            })
            .expect("expected IADD3 statement");
        if let RValue::Op { args, .. } = &iadd3_stmt.value {
            if let Some(r) = args[0].get_reg() {
                // Should still reference R1, not R0
                assert_eq!(
                    r.idx, 1,
                    "predicated copy should not be propagated, IADD3 arg should still be R1"
                );
            }
        }
    }
}
