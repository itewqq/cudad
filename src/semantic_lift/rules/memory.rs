use crate::semantic_lift::registry::RuleRegistry;

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("LDS", "lds", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_lds_expr(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("LDG", "ldg", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_ldg_expr(&sig.raw_opcode, args, stmt_ref, config)
    });
    registry.register("LDL", "ldl", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_ldl_expr(&sig.raw_opcode, args, stmt_ref, config)
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
    // ATOMG: explicit global-memory atomic operations (common in real kernels).
    registry.register("ATOMG", "atomg", |sig, args, stmt_ref, config| {
        lift_atoms(&sig.raw_opcode, args, stmt_ref, config)
    });
    // RED: global-memory reduction (atomic without return value).
    registry.register("RED", "red", |sig, args, stmt_ref, config| {
        lift_atoms(&sig.raw_opcode, args, stmt_ref, config)
    });
    // REDG: explicit global-memory reduction variant used with descriptors.
    registry.register("REDG", "redg", |sig, args, stmt_ref, config| {
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
    let lifted_args: Vec<crate::semantic_lift::LiftedExpr> = args
        .iter()
        .map(|a| crate::semantic_lift::lift_ir_expr(a, stmt_ref, config))
        .collect();
    let mem_idx = args
        .iter()
        .position(|arg| matches!(arg, crate::ir::IRExpr::Mem { .. }))?;

    // Map the atomic operation modifier to a CUDA API function name.
    let popc_inc = opcode.contains(".POPC.INC.");
    let cuda_fn = if popc_inc || opcode.contains(".ADD") {
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
        opcode
    };

    // For shared-memory atomics, try to render the address as shmem[...].
    let addr_expr = if opcode.starts_with("ATOMS") {
        crate::semantic_lift::lift_shared_ref_expr(&args[mem_idx], false, 0, stmt_ref, config)
            .map(|expr| crate::semantic_lift::LiftedExpr::Unary {
                op: "&".to_string(),
                arg: Box::new(expr),
            })
            .unwrap_or_else(|| lifted_args[mem_idx].clone())
    } else {
        let addr = crate::semantic_lift::lift_addr_expr(&args[mem_idx], stmt_ref, config)
            .unwrap_or_else(|| lifted_args[mem_idx].clone());
        crate::semantic_lift::scalar_type_from_opcode(opcode)
            .map(|ty| crate::semantic_lift::LiftedExpr::Cast {
                ty: format!("{}*", ty),
                expr: Box::new(addr.clone()),
            })
            .unwrap_or(addr)
    };

    let mut call_args = Vec::with_capacity(args.len());
    call_args.push(addr_expr);
    call_args.extend(
        lifted_args
        .iter()
        .enumerate()
        .filter_map(|(idx, arg)| (idx != mem_idx).then_some(arg.clone())),
    );
    if popc_inc && call_args.len() == 1 {
        call_args.push(crate::semantic_lift::LiftedExpr::Imm("1".to_string()));
    }
    Some(crate::semantic_lift::LiftedExpr::CallLike {
        func: cuda_fn.to_string(),
        args: call_args,
    })
}
