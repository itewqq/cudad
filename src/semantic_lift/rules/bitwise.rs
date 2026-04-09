use crate::semantic_lift::registry::RuleRegistry;

/// Try to constant-fold a PLOP3.LUT when all predicate inputs are known.
///
/// Args (rendered): [src0, src1, src2, combine_pred, lut_a, lut_b]
/// The LUT byte selects one of 256 possible 3-input boolean functions.
/// Bit index = (src0 << 2) | (src1 << 1) | src2, and the result is
/// `(lut >> bit_index) & 1`.
fn try_fold_plop3(rendered: &[String]) -> Option<String> {
    fn parse_bool(s: &str) -> Option<bool> {
        match s {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }
    fn parse_u8(s: &str) -> Option<u8> {
        s.parse::<u8>().ok()
    }

    let src0 = parse_bool(&rendered[0])?;
    let src1 = parse_bool(&rendered[1])?;
    let src2 = parse_bool(&rendered[2])?;
    let lut = parse_u8(&rendered[rendered.len() - 1])?;

    let bit_index = ((src0 as u8) << 2) | ((src1 as u8) << 1) | (src2 as u8);
    let result = (lut >> bit_index) & 1;
    Some(if result != 0 { "true" } else { "false" }.to_string())
}

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("LOP3", "lop3_lut", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("LOP3.LUT") {
            return crate::semantic_lift::lift_lop3_lut(args, stmt_ref, config);
        }
        None
    });
    registry.register("ULOP3", "ulop3_lut", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("ULOP3.LUT") {
            return crate::semantic_lift::lift_lop3_lut(args, stmt_ref, config);
        }
        None
    });
    // PLOP3: predicate logic operation (single-bit version of LOP3).
    // Try to constant-fold when all predicate inputs are known.
    // Otherwise render as plop3_lut(...) intrinsic for readability.
    registry.register("PLOP3", "plop3_lut", |sig, args, stmt_ref, config| {
        if !sig.raw_opcode.starts_with("PLOP3.LUT") {
            return None;
        }
        let rendered: Vec<String> = args
            .iter()
            .map(|a| crate::semantic_lift::lift_ir_expr(a, stmt_ref, config).render())
            .collect();

        // Attempt constant-folding: PLOP3.LUT has args
        // [src0, src1, src2, combine_pred, lut_a, lut_b]
        // If all predicate inputs are constant true/false, evaluate the LUT.
        if rendered.len() >= 6 {
            if let Some(result) = try_fold_plop3(&rendered) {
                return Some(crate::semantic_lift::LiftedExpr::Imm(result));
            }
        }

        Some(crate::semantic_lift::LiftedExpr::Raw(format!(
            "plop3_lut({})",
            rendered.join(", ")
        )))
    });
    registry.register("SHF", "shf", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_shf(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("USHF", "ushf", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_shf(&sig.raw_opcode, args, stmt_ref, config)
    });
}
