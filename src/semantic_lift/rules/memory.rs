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
    // ATOMS: shared-memory atomic operations.
    registry.register("ATOMS", "atoms", |sig, args, stmt_ref, config| {
        lift_atoms(&sig.raw_opcode, args, stmt_ref, config)
    });
    // ATOM: global-memory atomic operations.
    registry.register("ATOM", "atom", |sig, args, stmt_ref, config| {
        lift_atoms(&sig.raw_opcode, args, stmt_ref, config)
    });
    // RED: global-memory reduction (atomic without return value).
    registry.register("RED", "red", |sig, args, stmt_ref, config| {
        lift_atoms(&sig.raw_opcode, args, stmt_ref, config)
    });
}

/// Lift ATOMS/ATOM/RED to CUDA atomicXxx() calls.
fn lift_atoms(
    opcode: &str,
    args: &[crate::ir::IRExpr],
    stmt_ref: crate::StatementRef,
    config: &crate::semantic_lift::SemanticLiftConfig<'_>,
) -> Option<crate::semantic_lift::LiftedExpr> {
    let rendered: Vec<String> = args
        .iter()
        .map(|a| crate::semantic_lift::lift_ir_expr(a, stmt_ref, config).render())
        .collect();

    // Map the atomic operation modifier to a CUDA API function name.
    let cuda_fn = if opcode.contains(".ADD") {
        "atomicAdd"
    } else if opcode.contains(".MIN") {
        "atomicMin"
    } else if opcode.contains(".MAX") {
        "atomicMax"
    } else if opcode.contains(".INC") {
        "atomicInc"
    } else if opcode.contains(".DEC") {
        "atomicDec"
    } else if opcode.contains(".EXCH") {
        "atomicExch"
    } else if opcode.contains(".CAS") {
        "atomicCAS"
    } else if opcode.contains(".AND") {
        "atomicAnd"
    } else if opcode.contains(".OR") {
        "atomicOr"
    } else if opcode.contains(".XOR") {
        "atomicXor"
    } else {
        // Unknown atomic variant — render as raw opcode(...).
        return Some(crate::semantic_lift::LiftedExpr::Raw(format!(
            "{}({})",
            opcode,
            rendered.join(", ")
        )));
    };

    // For shared-memory atomics, try to render the address as shmem[...].
    let addr_str = if opcode.starts_with("ATOMS") && !args.is_empty() {
        crate::semantic_lift::render_shared_ref(&args[0], stmt_ref, config)
            .map(|s| format!("&{}", s))
            .unwrap_or_else(|| rendered[0].clone())
    } else if !args.is_empty() {
        rendered[0].clone()
    } else {
        return None;
    };

    let rest_args = &rendered[1..];
    if rest_args.is_empty() {
        Some(crate::semantic_lift::LiftedExpr::Raw(format!(
            "{}({})",
            cuda_fn, addr_str
        )))
    } else {
        Some(crate::semantic_lift::LiftedExpr::Raw(format!(
            "{}({}, {})",
            cuda_fn,
            addr_str,
            rest_args.join(", ")
        )))
    }
}
