use crate::abi::StatementRef;
use crate::ir::IRExpr;

use crate::semantic_lift::op_sig::OpSig;
use crate::semantic_lift::registry::RuleRegistry;
use crate::semantic_lift::{
    lift_ffma, lift_float_binary_add_like, lift_float_binary_infix, lift_fmnmx, lift_fsel,
    lift_iabs, lift_iadd3, lift_iadd3_x, lift_imad, lift_imad_hi_u32, lift_imad_iadd,
    lift_imad_mov, lift_imad_wide, lift_imad_x, lift_ir_expr, lift_mufu_rcp, lift_prmt,
    lift_unary_intrinsic, lift_viaddmnmx, lift_vimnmx, LiftedExpr, SemanticLiftConfig,
};

pub(super) fn register(registry: &mut RuleRegistry) {
    registry.register("S2R", "s2r", |_, args, _, _| {
        crate::semantic_lift::lift_s2r(args)
    });
    registry.register("CS2R", "cs2r", |_, args, _, _| {
        crate::semantic_lift::lift_s2r(args)
    });
    registry.register("SHFL", "shfl", |sig, args, stmt_ref, config| {
        crate::semantic_lift::lift_shfl(&sig.raw_opcode, args, stmt_ref, config)
    });
    for mnemonic in ["IMAD", "UIMAD"] {
        registry.register(mnemonic, "imad_mov", |sig, args, stmt_ref, config| {
            if sig.raw_opcode.ends_with(".MOV") || sig.raw_opcode.contains(".MOV.") {
                return lift_imad_mov(args, stmt_ref, config);
            }
            None
        });
        registry.register(mnemonic, "imad_iadd", |sig, args, stmt_ref, config| {
            if sig.raw_opcode.ends_with(".IADD") || sig.raw_opcode.contains(".IADD.") {
                return lift_imad_iadd(args, stmt_ref, config);
            }
            None
        });
        registry.register(mnemonic, "imad_wide", |sig, args, stmt_ref, config| {
            if sig.raw_opcode.ends_with(".WIDE") || sig.raw_opcode.contains(".WIDE.") {
                return lift_imad_wide(args, stmt_ref, config);
            }
            None
        });
        registry.register(mnemonic, "imad_hi_u32", |sig, args, stmt_ref, config| {
            if sig.raw_opcode.ends_with(".HI.U32") || sig.raw_opcode.contains(".HI.U32.") {
                return lift_imad_hi_u32(args, stmt_ref, config);
            }
            None
        });
        registry.register(mnemonic, "imad_x", |sig, args, stmt_ref, config| {
            if sig.raw_opcode.ends_with(".X") || sig.raw_opcode.contains(".X.") {
                return lift_imad_x(args, stmt_ref, config);
            }
            None
        });
        registry.register(mnemonic, "imad_generic", |sig, args, stmt_ref, config| {
            if sig.raw_opcode == sig.mnemonic || sig.raw_opcode == format!("{}.U32", sig.mnemonic) {
                return lift_imad(args, stmt_ref, config);
            }
            None
        });
        // IMAD.SHL.U32 and UIMAD.SHL.U32 both lower to a left shift.
        registry.register(mnemonic, "imad_shl_u32", |sig, args, stmt_ref, config| {
            if !sig.raw_opcode.starts_with("IMAD.SHL") && !sig.raw_opcode.starts_with("UIMAD.SHL") {
                return None;
            }
            if args.len() < 3 {
                return None;
            }
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
    }
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
    registry.register("IADD", "iadd", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_binary_add_like(args, stmt_ref, config)
    });
    registry.register("FMUL", "fmul", |_, args, stmt_ref, config| {
        lift_float_binary_infix("*", args, stmt_ref, config)
    });
    registry.register("FADD", "fadd", |_, args, stmt_ref, config| {
        lift_float_binary_add_like(args, stmt_ref, config)
    });
    registry.register("IABS", "iabs", |_, args, stmt_ref, config| {
        lift_iabs(args, stmt_ref, config)
    });
    registry.register("I2F", "i2f", |_sig, args, stmt_ref, config| {
        // I2F converts integer to float.  Any variant gets a (float) cast.
        lift_unary_intrinsic("(float)", args, stmt_ref, config)
    });
    registry.register("I2FP", "i2fp", |_sig, args, stmt_ref, config| {
        // I2FP is the SM100+ variant of I2F.
        lift_unary_intrinsic("(float)", args, stmt_ref, config)
    });
    registry.register("F2I", "f2i", |sig, args, stmt_ref, config| {
        // F2I converts float to integer.  Render as (uint32_t) or (int32_t) cast.
        if sig.raw_opcode.contains(".U32") {
            lift_unary_intrinsic("(uint32_t)", args, stmt_ref, config)
        } else {
            lift_unary_intrinsic("(int32_t)", args, stmt_ref, config)
        }
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
            Some(crate::semantic_lift::lift_ir_expr(
                &args[0], stmt_ref, config,
            ))
        } else {
            None
        }
    });
    // MOV is a regular register move: MOV Rd, Rs → Rd = Rs.
    registry.register("MOV", "mov", |_, args, stmt_ref, config| {
        if args.len() == 1 {
            Some(crate::semantic_lift::lift_ir_expr(
                &args[0], stmt_ref, config,
            ))
        } else {
            None
        }
    });
    // VIADD: 2-operand integer add (video instruction set).
    // VIADD Rd, Ra, imm → Rd = Ra + imm
    registry.register("VIADD", "viadd", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_binary_add_like(args, stmt_ref, config)
    });
    // S2UR: copy special register to uniform register.
    // Already handled by S2R lift, but S2UR is the uniform variant.
    registry.register("S2UR", "s2ur", |_, args, _, _| {
        crate::semantic_lift::lift_s2r(args)
    });
    // Half2 math keeps all source operands unless the semantic lifter can
    // decode a constant-materialization pattern earlier in the pipeline.
    registry.register("HFMA2", "hfma2", |_, args, stmt_ref, config| {
        let zero_like = |arg: &IRExpr| match arg {
            IRExpr::ImmI(0) => true,
            IRExpr::ImmF(value) if *value == 0.0 => true,
            IRExpr::Reg(r) if matches!(r.class.as_str(), "RZ" | "URZ") => true,
            IRExpr::Op { op, args: inner } if op == "-" && inner.len() == 1 => {
                matches!(&inner[0], IRExpr::Reg(r) if matches!(r.class.as_str(), "RZ" | "URZ"))
            }
            _ => false,
        };
        if args.iter().all(zero_like) {
            return Some(LiftedExpr::Imm("0".to_string()));
        }
        let rendered_args: Vec<String> = args
            .iter()
            .map(|arg| lift_ir_expr(arg, stmt_ref, config).render())
            .collect();
        Some(LiftedExpr::Raw(format!(
            "hfma2({})",
            rendered_args.join(", ")
        )))
    });
    registry.register("HADD2", "hadd2", |_, args, stmt_ref, config| {
        let rendered_args: Vec<String> = args
            .iter()
            .map(|arg| lift_ir_expr(arg, stmt_ref, config).render())
            .collect();
        Some(LiftedExpr::Raw(format!(
            "hadd2({})",
            rendered_args.join(", ")
        )))
    });
    registry.register("HMUL2", "hmul2", |_, args, stmt_ref, config| {
        let rendered_args: Vec<String> = args
            .iter()
            .map(|arg| lift_ir_expr(arg, stmt_ref, config).render())
            .collect();
        Some(LiftedExpr::Raw(format!(
            "hmul2({})",
            rendered_args.join(", ")
        )))
    });
    // PRMT: byte permute instruction → prmt(src0, selector, src1)
    registry.register("PRMT", "prmt", |_, args, stmt_ref, config| {
        lift_prmt(args, stmt_ref, config)
    });
    // VIMNMX: integer min/max (video instruction set)
    registry.register("VIMNMX", "vimnmx", |sig, args, stmt_ref, config| {
        lift_vimnmx(sig, args, stmt_ref, config)
    });
    registry.register("IMNMX", "imnmx", |sig, args, stmt_ref, config| {
        lift_vimnmx(sig, args, stmt_ref, config)
    });
    // VIADDMNMX: integer add-then-min/max (video instruction set)
    registry.register("VIADDMNMX", "viaddmnmx", |sig, args, stmt_ref, config| {
        lift_viaddmnmx(sig, args, stmt_ref, config)
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
