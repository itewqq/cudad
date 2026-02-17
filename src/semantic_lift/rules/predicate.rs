use crate::semantic_lift::registry::RuleRegistry;

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("ISETP", "setp_compare", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_setp_compare(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("FSETP", "fsetp_compare", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_setp_compare(&sig.raw_opcode, args, stmt_ref, config)
    });
}
