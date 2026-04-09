use crate::semantic_lift::registry::RuleRegistry;

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("LDS", "lds", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_lds_expr(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("LDG", "ldg", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_ldg_expr(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("LDC", "ldc", |_sig, args, stmt_ref, config| {
        // LDC loads a (non-uniform) register from constant memory.
        // Scalar, `.64`, and `.128` all lift the low-half (or only) def
        // from the operand directly; hi defs for wide loads are handled
        // in `lift_opcode_expr_for_def`.
        crate::semantic_lift::lift_uldc64(args, stmt_ref, config)
    });
    registry.register("ULDC", "uldc", |_sig, args, stmt_ref, config| {
        // Scalar, `.64`, and `.128` all lift the low-half (or only) def
        // from the operand directly.  Implicit hi defs for wide loads
        // (`.64`/`.128`) are synthesised in `lift_opcode_expr_for_def`.
        crate::semantic_lift::lift_uldc64(args, stmt_ref, config)
    });
    // LDCU is the SM 100+ (Blackwell) rename of ULDC.  Reuse the same lift
    // helper so the rendered output is identical across generations.
    registry.register("LDCU", "ldcu", |_sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_uldc64(args, stmt_ref, config)
    });
}
