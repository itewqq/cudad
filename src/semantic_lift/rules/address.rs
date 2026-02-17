use crate::semantic_lift::registry::RuleRegistry;

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("LEA", "lea", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_lea(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("ULEA", "ulea", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_lea(&sig.raw_opcode, args, stmt_ref, config)
    });
}
