use crate::abi::StatementRef;
use crate::ir::IRExpr;

use crate::semantic_lift::op_sig::OpSig;
use crate::semantic_lift::registry::RuleRegistry;
use crate::semantic_lift::{
    lift_fsel, lift_i2f_rp, lift_iabs, lift_iadd3, lift_iadd3_x, lift_imad,
    lift_imad_hi_u32, lift_imad_iadd, lift_imad_mov, lift_imad_wide, lift_imad_x, lift_mufu_rcp,
    lift_unary_intrinsic, LiftedExpr, SemanticLiftConfig,
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
    registry.register("FSEL", "fsel", |_, args, stmt_ref, config| {
        lift_fsel(args, stmt_ref, config)
    });
    registry.register("SEL", "sel", |_, args, stmt_ref, config| {
        crate::semantic_lift::lift_sel(args, stmt_ref, config)
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
