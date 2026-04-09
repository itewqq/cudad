use crate::semantic_lift::registry::RuleRegistry;

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
    // Render as plop3_lut(...) intrinsic for readability.
    registry.register("PLOP3", "plop3_lut", |sig, args, stmt_ref, config| {
        if !sig.raw_opcode.starts_with("PLOP3.LUT") {
            return None;
        }
        let rendered: Vec<String> = args
            .iter()
            .map(|a| crate::semantic_lift::lift_ir_expr(a, stmt_ref, config).render())
            .collect();
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
