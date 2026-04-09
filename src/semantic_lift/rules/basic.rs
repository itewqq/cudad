use crate::abi::StatementRef;
use crate::ir::IRExpr;

use crate::semantic_lift::op_sig::OpSig;
use crate::semantic_lift::registry::RuleRegistry;
use crate::semantic_lift::{
    lift_ffma, lift_fmnmx, lift_fsel, lift_i2f_rp, lift_iabs, lift_iadd3, lift_iadd3_x, lift_imad,
    lift_imad_hi_u32, lift_imad_iadd, lift_imad_mov, lift_imad_wide, lift_imad_x, lift_ir_expr,
    lift_mufu_rcp, lift_unary_intrinsic, LiftedExpr, SemanticLiftConfig,
};

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("S2R", "s2r", |_, args, _, _| crate::semantic_lift::lift_s2r(args));
    registry.register("IMAD", "imad_mov", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("IMAD.MOV") {
            return lift_imad_mov(args, stmt_ref, config);
        }
        None
    });
    registry.register("IMAD", "imad_iadd", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("IMAD.IADD") {
            return lift_imad_iadd(args, stmt_ref, config);
        }
        None
    });
    registry.register("IMAD", "imad_wide", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("IMAD.WIDE") {
            return lift_imad_wide(args, stmt_ref, config);
        }
        None
    });
    registry.register("IMAD", "imad_hi_u32", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("IMAD.HI.U32") {
            return lift_imad_hi_u32(args, stmt_ref, config);
        }
        None
    });
    registry.register("IMAD", "imad_x", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("IMAD.X") {
            return lift_imad_x(args, stmt_ref, config);
        }
        None
    });
    registry.register("IMAD", "imad_generic", |sig, args, stmt_ref, config| {
        if sig.raw_opcode == "IMAD" {
            return lift_imad(args, stmt_ref, config);
        }
        None
    });
    registry.register("IADD3", "iadd3", |_, args, stmt_ref, config| {
        lift_iadd3(args, stmt_ref, config)
    });
    registry.register("UIADD3", "uiadd3", |_, args, stmt_ref, config| {
        lift_iadd3(args, stmt_ref, config)
    });
    registry.register("IADD3", "iadd3_x", |sig, args, stmt_ref, config| {
        if sig.raw_opcode == "IADD3.X" {
            return lift_iadd3_x(args, stmt_ref, config);
        }
        None
    });
    registry.register("UIADD3", "uiadd3_x", |sig, args, stmt_ref, config| {
        if sig.raw_opcode == "UIADD3.X" {
            return lift_iadd3_x(args, stmt_ref, config);
        }
        None
    });
    registry.register("FMUL", "fmul", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_binary_infix("*", args, stmt_ref, config)
    });
    registry.register("FADD", "fadd", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_binary_add_like(args, stmt_ref, config)
    });
    registry.register("IABS", "iabs", |_, args, stmt_ref, config| {
        lift_iabs(args, stmt_ref, config)
    });
    registry.register("I2F", "i2f_rp", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("I2F.RP") {
            return lift_i2f_rp(args, stmt_ref, config);
        }
        None
    });
    registry.register("F2I", "f2i_trunc_u32_ftz_ntz", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("F2I.FTZ.U32.TRUNC.NTZ") {
            return lift_unary_intrinsic("f2i_trunc_u32_ftz_ntz", args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_rcp", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.RCP") {
            return lift_mufu_rcp(args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_rsq", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.RSQ") {
            return lift_unary_intrinsic("rsqrtf", args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_ex2", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.EX2") {
            return lift_unary_intrinsic("exp2f", args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_lg2", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.LG2") {
            return lift_unary_intrinsic("log2f", args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_sin", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.SIN") {
            return lift_unary_intrinsic("sinf", args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_cos", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.COS") {
            return lift_unary_intrinsic("cosf", args, stmt_ref, config);
        }
        None
    });
    registry.register("MUFU", "mufu_sqrt", |sig, args, stmt_ref, config| {
        if sig.raw_opcode.starts_with("MUFU.SQRT") {
            return lift_unary_intrinsic("sqrtf", args, stmt_ref, config);
        }
        None
    });
    registry.register("FFMA", "ffma", |_, args, stmt_ref, config| {
        lift_ffma(args, stmt_ref, config)
    });
    registry.register("FMNMX", "fmnmx", |_, args, stmt_ref, config| {
        lift_fmnmx(args, stmt_ref, config)
    });
    registry.register("FSEL", "fsel", |_, args, stmt_ref, config| {
        lift_fsel(args, stmt_ref, config)
    });
    registry.register("SEL", "sel", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_sel(args, stmt_ref, config)
    });
    // UMOV is a uniform register move: UMOV URd, URs → URd = URs.
    // Just pass through the single source operand.
    registry.register("UMOV", "umov", |_, args, stmt_ref, config| {
        if args.len() == 1 {
            Some(crate::semantic_lift::lift_ir_expr(&args[0], stmt_ref, config))
        } else {
            None
        }
    });
    // MOV is a regular register move: MOV Rd, Rs → Rd = Rs.
    registry.register("MOV", "mov", |_, args, stmt_ref, config| {
        if args.len() == 1 {
            Some(crate::semantic_lift::lift_ir_expr(&args[0], stmt_ref, config))
        } else {
            None
        }
    });
    // VIADD: 2-operand integer add (video instruction set).
    // VIADD Rd, Ra, imm → Rd = Ra + imm
    registry.register("VIADD", "viadd", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_binary_add_like(args, stmt_ref, config)
    });
    // IMAD.SHL.U32: shift-left via multiply-add with zero addend.
    // IMAD.SHL.U32 Rd, Ra, 2^n, RZ → Rd = Ra << n
    registry.register("IMAD", "imad_shl_u32", |sig, args, stmt_ref, config| {
        if !sig.raw_opcode.starts_with("IMAD.SHL") {
            return None;
        }
        if args.len() < 3 {
            return None;
        }
        // The second arg is the shift amount as a power of 2
        let shift_amount = match &args[1] {
            IRExpr::ImmI(n) if *n > 0 && (*n as u64).is_power_of_two() => {
                (*n as u64).trailing_zeros() as i64
            }
            _ => return None,
        };
        Some(LiftedExpr::Binary {
            op: "<<".to_string(),
            lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
            rhs: Box::new(LiftedExpr::Imm(shift_amount.to_string())),
        })
    });
    // S2UR: copy special register to uniform register.
    // Already handled by S2R lift, but S2UR is the uniform variant.
    registry.register("S2UR", "s2ur", |_, args, _, _| crate::semantic_lift::lift_s2r(args));
    // HFMA2: half-precision FMA. When all args are zero-like, simplify.
    registry.register("HFMA2", "hfma2", |_, args, stmt_ref, config| {
        // Check if this is a zero-initialization pattern: HFMA2(-RZ, RZ, 0, 0)
        let all_zero = args.iter().all(|a| match a {
            IRExpr::ImmI(0) => true,
            IRExpr::ImmF(f) if *f == 0.0 => true,
            IRExpr::Reg(r) if r.class == "RZ" => true,
            IRExpr::Op { op, args: inner } if op == "NEG" && inner.len() == 1 => {
                matches!(&inner[0], IRExpr::Reg(r) if r.class == "RZ")
            }
            _ => false,
        });
        if all_zero {
            return Some(LiftedExpr::Imm("0".to_string()));
        }
        // General case: render as hfma2(a, b, c)
        if args.len() >= 3 {
            let rendered_args: Vec<String> = args.iter().take(3)
                .map(|a| lift_ir_expr(a, stmt_ref, config).render())
                .collect();
            Some(LiftedExpr::Raw(format!("hfma2({})", rendered_args.join(", "))))
        } else {
            None
        }
    });
}

#[allow(dead_code)]
fn _sig_only(
    _sig: &OpSig,
    _args: &[IRExpr],
    _stmt_ref: StatementRef,
    _config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    None
}
