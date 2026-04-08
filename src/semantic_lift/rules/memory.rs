use crate::semantic_lift::registry::RuleRegistry;

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("LDS", "lds_u8", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("LDS") && sig.raw_opcode.contains(".U8") {
            return crate::semantic_lift::lift_lds_expr(args, stmt_ref, config);
        }
        None
    });
    registry.register("LDG", "ldg", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_ldg_expr(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("ULDC", "uldc64", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("ULDC.64") {
            return crate::semantic_lift::lift_uldc64(args, stmt_ref, config);
        }
        None
    });
    // LDCU is the SM 100+ (Blackwell) rename of ULDC.  Reuse the same lift
    // helper so the rendered output is identical across generations.
    registry.register("LDCU", "ldcu64", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("LDCU.64") {
            return crate::semantic_lift::lift_uldc64(args, stmt_ref, config);
        }
        None
    });
}
