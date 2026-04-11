//! IR-level algebraic simplification for SSA-form `FunctionIR`.
//!
//! Peephole rewrite pass that applies algebraic identities to `RValue::Op`
//! nodes in the SSA IR.  This runs after constant propagation (which has
//! already inlined known immediates), making more zero/one/identity patterns
//! visible.
//!
//! Identities recognized:
//!
//! | Pattern | Result |
//! |---|---|
//! | `IADD3(x, 0, 0)` | copy x |
//! | `IADD3(0, x, 0)` | copy x |
//! | `IADD3(0, 0, x)` | copy x |
//! | `IADD3(x, y, 0)` | `IADD3(x, y, 0)` (unchanged — 2 non-zero) |
//! | `IMAD(0, _, x)` | copy x |
//! | `IMAD(_, 0, x)` | copy x |
//! | `IMAD(x, 1, 0)` | copy x |
//! | `IMAD(1, x, 0)` | copy x |
//! | `SHF.*(x, 0, _)` | copy x (shift by 0) |
//! | `LOP3.LUT(x, ?, ?, 0xFC)` | copy x (identity: output = A) |
//! | `LOP3.LUT(?, x, ?, 0xF0)` | copy x (identity: output = B) |
//! | `LOP3.LUT(?, ?, x, 0xCC)` | copy x (identity: output = C, but not standard) |
//!
//! After this pass, a subsequent `ir_copyprop` + `ir_dce` pass will remove
//! the now-trivial copies.

use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue};

/// Run algebraic simplification on `fir`, returning a new `FunctionIR`
/// with identity operations replaced by copies.
pub fn ir_algebra(fir: &FunctionIR) -> FunctionIR {
    let new_blocks = fir
        .blocks
        .iter()
        .map(|block| IRBlock {
            id: block.id,
            start_addr: block.start_addr,
            irdst: block.irdst.clone(),
            stmts: block
                .stmts
                .iter()
                .map(|stmt| simplify_stmt(stmt))
                .collect(),
        })
        .collect();
    FunctionIR { blocks: new_blocks }
}

fn simplify_stmt(stmt: &IRStatement) -> IRStatement {
    IRStatement {
        defs: stmt.defs.clone(),
        value: simplify_rvalue(&stmt.value),
        pred: stmt.pred.clone(),
        mem_addr_args: stmt.mem_addr_args.clone(),
        pred_old_defs: stmt.pred_old_defs.clone(),
    }
}

fn simplify_rvalue(value: &RValue) -> RValue {
    match value {
        RValue::Op { opcode, args } => {
            if let Some(simplified) = try_simplify(opcode, args) {
                simplified
            } else {
                value.clone()
            }
        }
        _ => value.clone(),
    }
}

/// Try to simplify an opcode+args into a simpler RValue.
/// Returns None if no simplification applies.
fn try_simplify(opcode: &str, args: &[IRExpr]) -> Option<RValue> {
    // ------- IADD3: three-input addition -------
    // IADD3(x, 0, 0) → copy x  (and permutations)
    if opcode == "IADD3" && args.len() >= 3 {
        let z0 = is_zero_expr(&args[0]);
        let z1 = is_zero_expr(&args[1]);
        let z2 = is_zero_expr(&args[2]);
        // Exactly two zeros → result is the non-zero operand
        if z0 && z1 && !z2 {
            return Some(make_copy(&args[2]));
        }
        if z0 && !z1 && z2 {
            return Some(make_copy(&args[1]));
        }
        if !z0 && z1 && z2 {
            return Some(make_copy(&args[0]));
        }
        // All three zero → immediate 0
        if z0 && z1 && z2 {
            return Some(RValue::ImmI(0));
        }
        // Exactly one zero → reduce to two-operand add (keep as IADD3 but
        // this doesn't simplify further; leave it alone).
    }

    // ------- IMAD / IMAD.* : multiply-add -------
    // IMAD(a, b, c) = a*b + c
    // If a==0 or b==0 → result = c (copy)
    // If a==1 and c==0 → result = b (copy)
    // If b==1 and c==0 → result = a (copy)
    if opcode.starts_with("IMAD") && !opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        let a = &args[0];
        let b = &args[1];
        let c = &args[2];
        // a*b + c where a==0 or b==0 → c
        if is_zero_expr(a) || is_zero_expr(b) {
            return Some(make_copy(c));
        }
        // a*b + c where c==0 and (a==1 or b==1)
        if is_zero_expr(c) {
            if is_one_expr(a) {
                return Some(make_copy(b));
            }
            if is_one_expr(b) {
                return Some(make_copy(a));
            }
        }
        // a*1 + c → IADD-like(a, c) — we keep as IMAD since it's still valid
        // (would need a different opcode otherwise)
    }

    // ------- SHF: funnel shift -------
    // SHF.*(x, 0, _) → copy x when shift amount is 0
    if opcode.starts_with("SHF") && args.len() >= 2 {
        // SHF.R has args: (data_lo, shift_amount, data_hi) or similar.
        // For SHF.R.S32/SHF.L: if shift amount (arg[1]) is 0, result is arg[0].
        if is_zero_expr(&args[1]) {
            return Some(make_copy(&args[0]));
        }
    }

    // ------- LOP3.LUT: 3-input logic with truth table -------
    // LOP3.LUT(a, b, c, lut) where lut encodes the truth table.
    // Identity patterns:
    //   lut = 0xFC (252) → output = A (first input)
    //   lut = 0xF0 (240) → output = B (second input)  (actually 0xF0 = B in NVIDIA's ABC ordering)
    //   lut = 0xCC (204) → output = C (third input)
    //   lut = 0xAA (170) → output = A (alternate encoding)
    //   lut = 0x00 (0)   → output = 0
    //   lut = 0xFF (255) → output = ~0 = -1
    if opcode.starts_with("LOP3") && args.len() >= 4 {
        if let IRExpr::ImmI(lut) = &args[3] {
            match *lut {
                0xFC | 0xAA => return Some(make_copy(&args[0])), // output = A
                0xF0 => return Some(make_copy(&args[1])),        // output = B
                0xCC => return Some(make_copy(&args[2])),        // output = C
                0x00 => return Some(RValue::ImmI(0)),            // output = 0
                0xFF => return Some(RValue::ImmI(-1)),           // output = all-ones
                _ => {}
            }
        }
    }

    None
}

/// Create a copy RValue from an expression.
/// If the expression is a register, produce a MOV-like copy.
/// If it's an immediate, produce the immediate directly.
fn make_copy(expr: &IRExpr) -> RValue {
    match expr {
        IRExpr::ImmI(v) => RValue::ImmI(*v),
        IRExpr::ImmF(v) => RValue::ImmF(*v),
        // For register or complex expressions, create a single-arg MOV-like op
        // that ir_copyprop will later handle.
        _ => RValue::Op {
            opcode: "MOV".to_string(),
            args: vec![expr.clone()],
        },
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

fn is_one_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 1,
        IRExpr::ImmF(f) => *f == 1.0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, ir_constprop, ir_copyprop, ir_dce, parse_sass};

    #[test]
    fn iadd3_x_0_0_simplifies_to_copy() {
        let sass = r#"
            /*0000*/ IADD3 R2, R0, RZ, RZ ;
            /*0010*/ IADD3 R3, R2, 0x1, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let cp = ir_constprop(&fir);
        let simplified = ir_algebra(&cp);

        // After simplification, R2 = IADD3(R0, RZ, RZ) → R2 = MOV(R0)
        let r2_stmt = simplified
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| {
                s.defs
                    .first()
                    .and_then(|d| d.get_reg())
                    .map(|r| r.idx == 2 && r.class == "R")
                    .unwrap_or(false)
            })
            .expect("expected R2 definition");
        match &r2_stmt.value {
            RValue::Op { opcode, args } => {
                assert_eq!(opcode, "MOV", "IADD3(x, 0, 0) should simplify to MOV");
                assert_eq!(args.len(), 1);
                // The arg should reference R0
                if let Some(r) = args[0].get_reg() {
                    assert_eq!(r.idx, 0, "MOV arg should be R0");
                }
            }
            _ => panic!("expected Op(MOV), got {:?}", r2_stmt.value),
        }
    }

    #[test]
    fn iadd3_all_zero_simplifies_to_zero() {
        let sass = r#"
            /*0000*/ IADD3 R2, RZ, RZ, RZ ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let simplified = ir_algebra(&fir);

        let r2_stmt = simplified
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| {
                s.defs
                    .first()
                    .and_then(|d| d.get_reg())
                    .map(|r| r.idx == 2 && r.class == "R")
                    .unwrap_or(false)
            })
            .expect("expected R2 definition");
        assert!(
            matches!(r2_stmt.value, RValue::ImmI(0)),
            "IADD3(0, 0, 0) should simplify to ImmI(0), got {:?}",
            r2_stmt.value
        );
    }

    #[test]
    fn imad_zero_mult_simplifies_to_addend() {
        // IMAD(RZ, R1, R2) = 0*R1 + R2 → copy R2
        let sass = r#"
            /*0000*/ IMAD R3, RZ, R1, R2 ;
            /*0010*/ IADD3 R4, R3, 0x1, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let simplified = ir_algebra(&fir);

        let r3_stmt = simplified
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| {
                s.defs
                    .first()
                    .and_then(|d| d.get_reg())
                    .map(|r| r.idx == 3 && r.class == "R")
                    .unwrap_or(false)
            })
            .expect("expected R3 definition");
        match &r3_stmt.value {
            RValue::Op { opcode, args } => {
                assert_eq!(opcode, "MOV", "IMAD(0, _, x) should simplify to MOV(x)");
                assert_eq!(args.len(), 1);
                // The arg should reference R2
                if let Some(r) = args[0].get_reg() {
                    assert_eq!(r.idx, 2, "MOV arg should be R2");
                }
            }
            _ => panic!("expected Op(MOV), got {:?}", r3_stmt.value),
        }
    }

    #[test]
    fn algebra_then_copyprop_eliminates_identity() {
        // Full pipeline: IADD3(R0, RZ, RZ) → MOV(R0) → copyprop substitutes R0 for R2
        let sass = r#"
            /*0000*/ IADD3 R2, R0, RZ, RZ ;
            /*0010*/ IADD3 R3, R2, 0x1, RZ ;
            /*0020*/ STG.E [R4.64], R3 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let cp = ir_constprop(&fir);
        let alg = ir_algebra(&cp);
        let copyprop = ir_copyprop(&alg);
        let dce = ir_dce(&copyprop);

        // After full pipeline, R3 = IADD3(R2, 1, 0) should become R3 = IADD3(R0, 1, 0)
        // because R2 = MOV(R0) was copy-propagated away.
        let r3_stmt = dce
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find(|s| {
                matches!(&s.value, RValue::Op { opcode, .. } if opcode == "IADD3")
            })
            .expect("expected IADD3 statement");
        if let RValue::Op { args, .. } = &r3_stmt.value {
            if let Some(r) = args[0].get_reg() {
                assert_eq!(
                    r.idx, 0,
                    "after algebra+copyprop, IADD3 first arg should be R0 (got R{})",
                    r.idx
                );
            }
        }
    }
}
