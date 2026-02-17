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
    registry.register("SHF", "shf", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_shf(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("USHF", "ushf", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_shf(&sig.raw_opcode, args, stmt_ref, config)
    });
}
