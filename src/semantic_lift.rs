//! Conservative semantic lifting for SSA IR expression rendering.
//! This pass is non-mutating: it does not alter CFG/SSA/structure, only
//! computes optional expression rewrites for display.

use std::collections::BTreeMap;

use crate::abi::{AbiAnnotations, AbiArgAliases, ConstMemSemantic, StatementRef};
use crate::ir::{FunctionIR, IRExpr, RValue};

#[derive(Clone, Debug)]
pub struct SemanticLiftConfig<'a> {
    pub abi_annotations: Option<&'a AbiAnnotations>,
    pub abi_aliases: Option<&'a AbiArgAliases>,
    /// Strict mode only applies high-confidence rewrites.
    pub strict: bool,
}

impl Default for SemanticLiftConfig<'_> {
    fn default() -> Self {
        Self {
            abi_annotations: None,
            abi_aliases: None,
            strict: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LiftedExpr {
    Raw(String),
    Imm(String),
    Reg(String),
    Unary {
        op: String,
        arg: Box<LiftedExpr>,
    },
    Binary {
        op: String,
        lhs: Box<LiftedExpr>,
        rhs: Box<LiftedExpr>,
    },
    Ternary {
        cond: Box<LiftedExpr>,
        then_expr: Box<LiftedExpr>,
        else_expr: Box<LiftedExpr>,
    },
}

impl LiftedExpr {
    pub fn render(&self) -> String {
        self.render_with_prec(0)
    }

    fn render_with_prec(&self, parent_prec: u8) -> String {
        match self {
            LiftedExpr::Raw(s) | LiftedExpr::Imm(s) | LiftedExpr::Reg(s) => s.clone(),
            LiftedExpr::Unary { op, arg } => {
                let prec = 7;
                let inner = format!("{}{}", op, arg.render_with_prec(prec));
                if prec < parent_prec {
                    format!("({})", inner)
                } else {
                    inner
                }
            }
            LiftedExpr::Binary { op, lhs, rhs } => {
                let prec = binary_prec(op);
                let inner = format!(
                    "{} {} {}",
                    lhs.render_with_prec(prec),
                    op,
                    rhs.render_with_prec(prec + 1)
                );
                if prec < parent_prec {
                    format!("({})", inner)
                } else {
                    inner
                }
            }
            LiftedExpr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let prec = 1;
                let inner = format!(
                    "{} ? {} : {}",
                    cond.render_with_prec(prec),
                    then_expr.render_with_prec(prec),
                    else_expr.render_with_prec(prec)
                );
                if prec < parent_prec {
                    format!("({})", inner)
                } else {
                    inner
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LiftedStmt {
    pub dest: String,
    pub pred: Option<LiftedExpr>,
    pub rhs: LiftedExpr,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticLiftStats {
    pub attempted: usize,
    pub lifted: usize,
    pub fallback: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticLiftResult {
    pub by_stmt: BTreeMap<StatementRef, LiftedStmt>,
    pub stats: SemanticLiftStats,
}

pub fn lift_function_ir(function_ir: &FunctionIR, config: &SemanticLiftConfig<'_>) -> SemanticLiftResult {
    let mut out = SemanticLiftResult::default();

    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            let RValue::Op { opcode, args } = &stmt.value else {
                continue;
            };

            out.stats.attempted += 1;
            if let Some((dest, rhs)) = lift_store_stmt(opcode, args, stmt_ref, config) {
                let pred = stmt
                    .pred
                    .as_ref()
                    .map(|p| lift_ir_expr(p, stmt_ref, config));
                out.by_stmt.insert(
                    stmt_ref,
                    LiftedStmt {
                        dest,
                        pred,
                        rhs,
                    },
                );
                out.stats.lifted += 1;
                continue;
            }
            let lifted_rhs = lift_opcode_expr(opcode, args, stmt_ref, config);
            if let Some(rhs) = lifted_rhs {
                let dest = stmt
                    .dest
                    .as_ref()
                    .map_or_else(|| "_".to_string(), |d| render_expr_raw(d, stmt_ref, config));
                let pred = stmt
                    .pred
                    .as_ref()
                    .map(|p| lift_ir_expr(p, stmt_ref, config));
                out.by_stmt.insert(
                    stmt_ref,
                    LiftedStmt {
                        dest,
                        pred,
                        rhs,
                    },
                );
                out.stats.lifted += 1;
            } else {
                out.stats.fallback += 1;
            }
        }
    }

    out
}

fn lift_opcode_expr(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if opcode.starts_with("IMAD.MOV") {
        return lift_imad_mov(args, stmt_ref, config);
    }
    if opcode.starts_with("IMAD.WIDE") {
        return lift_imad_wide(args, stmt_ref, config);
    }
    if opcode.starts_with("IMAD.IADD") {
        return lift_imad_iadd(args, stmt_ref, config);
    }
    if opcode == "IADD3.X" || opcode == "UIADD3.X" {
        return lift_iadd3_x(args, stmt_ref, config);
    }
    if opcode == "IADD3" || opcode == "UIADD3" {
        return lift_iadd3(args, stmt_ref, config);
    }
    if opcode.starts_with("FMUL") {
        return lift_binary_infix("*", args, stmt_ref, config);
    }
    if opcode.starts_with("FADD") {
        return lift_binary_add_like(args, stmt_ref, config);
    }
    if opcode.starts_with("LDS") && opcode.contains(".U8") {
        return lift_lds_expr(args, stmt_ref, config);
    }
    if opcode.starts_with("LDG") {
        return lift_ldg_expr(args, stmt_ref, config);
    }
    if opcode.starts_with("FSEL") {
        return lift_fsel(args, stmt_ref, config);
    }
    if opcode.starts_with("ISETP") || opcode.starts_with("FSETP") {
        return lift_setp_compare(opcode, args, stmt_ref, config);
    }
    if opcode.starts_with("LOP3.LUT") || opcode.starts_with("ULOP3.LUT") {
        return lift_lop3_lut(args, stmt_ref, config);
    }
    if opcode.starts_with("SEL") {
        return lift_sel(args, stmt_ref, config);
    }
    if opcode.starts_with("SHF") || opcode.starts_with("USHF") {
        return lift_shf(opcode, args, stmt_ref, config);
    }
    if opcode.starts_with("LEA") || opcode.starts_with("ULEA") {
        return lift_lea(opcode, args, stmt_ref, config);
    }
    None
}

fn lift_store_stmt(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<(String, LiftedExpr)> {
    if args.len() < 2 || !is_mem_expr(&args[0]) {
        return None;
    }
    if opcode.starts_with("STS") {
        if !opcode.contains(".U8") {
            return None;
        }
        let dest = render_shared_u8_ref(&args[0], stmt_ref, config)?;
        let rhs = lift_ir_expr(&args[1], stmt_ref, config);
        return Some((dest, rhs));
    }
    if opcode.starts_with("STG") {
        let dest = render_expr_raw(&args[0], stmt_ref, config);
        let rhs = lift_ir_expr(&args[1], stmt_ref, config);
        return Some((dest, rhs));
    }
    None
}

fn lift_imad_mov(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
        return Some(lift_ir_expr(&args[2], stmt_ref, config));
    }
    None
}

fn lift_iadd3(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    let a0z = is_zero_expr(a0);
    let a1z = is_zero_expr(a1);
    let a2z = is_zero_expr(a2);

    if config.strict && !(a0z || a1z || a2z) {
        return None;
    }

    let expr = if a0z && !a1z && !a2z {
        add_like_expr(
            lift_ir_expr(a1, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        )
    } else if a1z && !a0z && !a2z {
        add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        )
    } else if a2z && !a0z && !a1z {
        add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a1, stmt_ref, config),
        )
    } else if config.strict {
        return None;
    } else {
        let left = add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a1, stmt_ref, config),
        );
        add_like_expr(left, lift_ir_expr(a2, stmt_ref, config))
    };

    Some(expr)
}

fn lift_iadd3_x(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2, carry_pred) = extract_addx_operands(args)?;
    if config.strict && !is_zero_expr(a2) {
        return None;
    }

    let mut sum = add_like_expr(
        lift_ir_expr(a0, stmt_ref, config),
        lift_ir_expr(a1, stmt_ref, config),
    );
    if !is_zero_expr(a2) {
        sum = add_like_expr(sum, lift_ir_expr(a2, stmt_ref, config));
    }
    let carry = carry_inc_expr(carry_pred, stmt_ref, config)?;
    Some(add_like_expr(sum, carry))
}

fn lift_imad_iadd(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    let (a0, a1, a2) = extract_triplet_operands(args)?;
    if is_imm_i(a1, 1) {
        return Some(add_like_expr(
            lift_ir_expr(a0, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        ));
    }
    if is_imm_i(a0, 1) {
        return Some(add_like_expr(
            lift_ir_expr(a1, stmt_ref, config),
            lift_ir_expr(a2, stmt_ref, config),
        ));
    }
    let mul = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lift_ir_expr(a0, stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(a1, stmt_ref, config)),
    };
    Some(add_like_expr(mul, lift_ir_expr(a2, stmt_ref, config)))
}

fn lift_imad_wide(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    let mul = LiftedExpr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    };
    Some(add_like_expr(mul, lift_ir_expr(&args[2], stmt_ref, config)))
}

fn lift_binary_infix(
    op: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 2 {
        return None;
    }
    Some(LiftedExpr::Binary {
        op: op.to_string(),
        lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
}

fn lift_binary_add_like(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 2 {
        return None;
    }
    Some(add_like_expr(
        lift_ir_expr(&args[0], stmt_ref, config),
        lift_ir_expr(&args[1], stmt_ref, config),
    ))
}

fn lift_lds_expr(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 1 {
        return None;
    }
    let shared = render_shared_u8_ref(&args[0], stmt_ref, config)?;
    Some(LiftedExpr::Raw(shared))
}

fn lift_ldg_expr(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 1 || !is_mem_expr(&args[0]) {
        return None;
    }
    Some(LiftedExpr::Raw(render_expr_raw(&args[0], stmt_ref, config)))
}

fn lift_fsel(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    Some(LiftedExpr::Ternary {
        cond: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
        then_expr: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        else_expr: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
}

fn lift_setp_compare(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() < 4 {
        return None;
    }
    if config.strict && (!is_true_pred_expr(&args[0]) || !is_true_pred_expr(&args[3])) {
        return None;
    }
    if config.strict && !opcode.split('.').any(|p| p == "AND") {
        return None;
    }
    let cmp = cmp_token_to_op(opcode)?;
    Some(LiftedExpr::Binary {
        op: cmp.to_string(),
        lhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
    })
}

fn lift_lop3_lut(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    // Conservative subset:
    // LOP3.LUT dst, a, b, c, imm, PT/!PT
    // currently only when one or two inputs are zero (RZ/URZ),
    // reducing to unary/binary bitwise forms.
    if args.len() != 5 {
        return None;
    }
    if !is_pred_control_expr(&args[4]) {
        return None;
    }

    let imm = u8::try_from(imm_as_u32(&args[3])? & 0xff).ok()?;
    let a0z = is_zero_expr(&args[0]);
    let a1z = is_zero_expr(&args[1]);
    let a2z = is_zero_expr(&args[2]);

    // Two fixed-zero inputs -> unary form on the remaining input.
    if a0z && a1z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 0, 0, 1) as u8) << 1);
        return lop3_unary_expr(lift_ir_expr(&args[2], stmt_ref, config), bits);
    }
    if a0z && a2z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 0, 1, 0) as u8) << 1);
        return lop3_unary_expr(lift_ir_expr(&args[1], stmt_ref, config), bits);
    }
    if a1z && a2z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 1, 0, 0) as u8) << 1);
        return lop3_unary_expr(lift_ir_expr(&args[0], stmt_ref, config), bits);
    }

    // One fixed-zero input -> binary form on remaining inputs.
    if a2z {
        // f(a, b, 0): truth-table rows (a,b)=00,01,10,11 -> bits 0,2,4,6
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 1, 0) as u8) << 1)
            | ((lop3_bit(imm, 1, 0, 0) as u8) << 2)
            | ((lop3_bit(imm, 1, 1, 0) as u8) << 3);
        return lop3_binary_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[1], stmt_ref, config),
            nibble,
        );
    }
    if a0z {
        // f(0, b, c): rows (b,c)=00,01,10,11 -> bits 0,1,2,3
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 0, 1) as u8) << 1)
            | ((lop3_bit(imm, 0, 1, 0) as u8) << 2)
            | ((lop3_bit(imm, 0, 1, 1) as u8) << 3);
        return lop3_binary_expr(
            lift_ir_expr(&args[1], stmt_ref, config),
            lift_ir_expr(&args[2], stmt_ref, config),
            nibble,
        );
    }
    if a1z {
        // f(a, 0, c): rows (a,c)=00,01,10,11 -> bits 0,1,4,5
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 0, 1) as u8) << 1)
            | ((lop3_bit(imm, 1, 0, 0) as u8) << 2)
            | ((lop3_bit(imm, 1, 0, 1) as u8) << 3);
        return lop3_binary_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[2], stmt_ref, config),
            nibble,
        );
    }

    None
}

fn lop3_bit(imm: u8, a: u8, b: u8, c: u8) -> bool {
    let idx = (a << 2) | (b << 1) | c;
    ((imm >> idx) & 1) != 0
}

fn lop3_unary_expr(x: LiftedExpr, bits: u8) -> Option<LiftedExpr> {
    match bits & 0x3 {
        0x0 => Some(LiftedExpr::Imm("0".to_string())),
        0x1 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x),
        }),
        0x2 => Some(x),
        0x3 => Some(LiftedExpr::Imm("0xffffffff".to_string())),
        _ => None,
    }
}

fn lop3_binary_expr(x: LiftedExpr, y: LiftedExpr, nibble: u8) -> Option<LiftedExpr> {
    let x_not = || LiftedExpr::Unary {
        op: "~".to_string(),
        arg: Box::new(x.clone()),
    };
    let y_not = || LiftedExpr::Unary {
        op: "~".to_string(),
        arg: Box::new(y.clone()),
    };
    let x_and_y = || LiftedExpr::Binary {
        op: "&".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    let x_or_y = || LiftedExpr::Binary {
        op: "|".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    let x_xor_y = || LiftedExpr::Binary {
        op: "^".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };

    match nibble & 0xf {
        0x0 => Some(LiftedExpr::Imm("0".to_string())),
        0x1 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_or_y()),
        }),
        0x2 => Some(LiftedExpr::Binary {
            op: "&".to_string(),
            lhs: Box::new(x_not()),
            rhs: Box::new(y.clone()),
        }),
        0x3 => Some(x_not()),
        0x4 => Some(LiftedExpr::Binary {
            op: "&".to_string(),
            lhs: Box::new(x.clone()),
            rhs: Box::new(y_not()),
        }),
        0x5 => Some(y_not()),
        0x6 => Some(x_xor_y()),
        0x7 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_and_y()),
        }),
        0x8 => Some(x_and_y()),
        0x9 => Some(LiftedExpr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_xor_y()),
        }),
        0xa => Some(y.clone()),
        0xb => Some(LiftedExpr::Binary {
            op: "|".to_string(),
            lhs: Box::new(x_not()),
            rhs: Box::new(y.clone()),
        }),
        0xc => Some(x.clone()),
        0xd => Some(LiftedExpr::Binary {
            op: "|".to_string(),
            lhs: Box::new(x.clone()),
            rhs: Box::new(y_not()),
        }),
        0xe => Some(x_or_y()),
        0xf => Some(LiftedExpr::Imm("0xffffffff".to_string())),
        _ => None,
    }
}

fn lift_sel(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    Some(LiftedExpr::Ternary {
        cond: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
        then_expr: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        else_expr: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
}

fn lift_shf(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 3 {
        return None;
    }
    // Conservative: only fold common right-shift forms used for sign/high extraction:
    // SHF*.R.*.HI dst, RZ/URZ, imm, src  ->  src >> imm
    let is_right = opcode.split('.').any(|t| t == "R");
    let is_hi = opcode.split('.').any(|t| t == "HI");
    if !is_right || !is_hi {
        return None;
    }
    if !is_zero_expr(&args[0]) {
        return None;
    }
    if !matches!(args[1], IRExpr::ImmI(_)) {
        return None;
    }
    Some(LiftedExpr::Binary {
        op: ">>".to_string(),
        lhs: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
}

fn lift_lea(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if opcode.starts_with("LEA.HI") || opcode.starts_with("ULEA.HI") {
        return lift_lea_hi(args, stmt_ref, config);
    }
    if opcode != "LEA" {
        return None;
    }
    // Conservative LEA low-part form:
    // LEA dst, P?, base, off, sh  -> base + (off << sh)
    if args.len() != 4 {
        return None;
    }
    if !is_pred_reg_expr(&args[0]) {
        return None;
    }
    if !matches!(args[3], IRExpr::ImmI(_)) {
        return None;
    }
    let shifted = LiftedExpr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[3], stmt_ref, config)),
    };
    Some(add_like_expr(
        lift_ir_expr(&args[1], stmt_ref, config),
        shifted,
    ))
}

fn lift_lea_hi(
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if args.len() != 4 {
        return None;
    }
    // Common form:
    //   LEA.HI dst, base, off, sh
    // Conservative render: hi32(base + (off << sh))
    if matches!(args[3], IRExpr::ImmI(_)) {
        if config.strict && !is_zero_expr(&args[2]) {
            return None;
        }
        let shifted = LiftedExpr::Binary {
            op: "<<".to_string(),
            lhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
            rhs: Box::new(lift_ir_expr(&args[3], stmt_ref, config)),
        };
        let mut inner = add_like_expr(lift_ir_expr(&args[0], stmt_ref, config), shifted);
        if !is_zero_expr(&args[2]) {
            inner = add_like_expr(inner, lift_ir_expr(&args[2], stmt_ref, config));
        }
        return Some(hi32_expr(inner));
    }

    // Carry form commonly emitted as:
    //   LEA.HI.X.* dst, base, off, sh, Pcarry
    // IR keeps 4 args (base, off, sh, Pcarry).
    if matches!(args[2], IRExpr::ImmI(_)) && is_pred_reg_expr(&args[3]) {
        let shifted = LiftedExpr::Binary {
            op: "<<".to_string(),
            lhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
            rhs: Box::new(lift_ir_expr(&args[2], stmt_ref, config)),
        };
        let inner = add_like_expr(lift_ir_expr(&args[0], stmt_ref, config), shifted);
        let hi = hi32_expr(inner);
        let carry = carry_inc_expr(&args[3], stmt_ref, config)?;
        return Some(add_like_expr(hi, carry));
    }
    None
}

fn cmp_token_to_op(opcode: &str) -> Option<&'static str> {
    for tok in opcode.split('.') {
        let op = match tok {
            "EQ" => "==",
            "NE" => "!=",
            "LT" => "<",
            "LE" => "<=",
            "GT" => ">",
            "GE" => ">=",
            _ => continue,
        };
        return Some(op);
    }
    None
}

fn add_like_expr(lhs: LiftedExpr, rhs: LiftedExpr) -> LiftedExpr {
    if let Some((is_neg, mag)) = rhs_signed_imm(&rhs) {
        if is_neg {
            return LiftedExpr::Binary {
                op: "-".to_string(),
                lhs: Box::new(lhs),
                rhs: Box::new(LiftedExpr::Imm(mag)),
            };
        }
    }
    if let Some(mag) = rhs_negated_nonimm(&rhs) {
        return LiftedExpr::Binary {
            op: "-".to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(LiftedExpr::Raw(mag)),
        };
    }
    LiftedExpr::Binary {
        op: "+".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn rhs_signed_imm(rhs: &LiftedExpr) -> Option<(bool, String)> {
    let LiftedExpr::Imm(text) = rhs else {
        return None;
    };
    if let Some(rest) = text.strip_prefix('-') {
        return Some((true, rest.to_string()));
    }
    Some((false, text.clone()))
}

fn rhs_negated_nonimm(rhs: &LiftedExpr) -> Option<String> {
    match rhs {
        LiftedExpr::Reg(s) | LiftedExpr::Raw(s) => {
            let rest = s.strip_prefix('-')?;
            Some(rest.to_string())
        }
        LiftedExpr::Unary { op, arg } if op == "-" => Some(arg.render()),
        _ => None,
    }
}

fn hi32_expr(inner: LiftedExpr) -> LiftedExpr {
    LiftedExpr::Raw(format!("hi32({})", inner.render()))
}

fn is_true_pred_expr(e: &IRExpr) -> bool {
    let IRExpr::Reg(r) = e else {
        return false;
    };
    matches!(r.class.as_str(), "PT" | "UPT")
}

fn is_imm_i(e: &IRExpr, v: i64) -> bool {
    matches!(e, IRExpr::ImmI(i) if *i == v)
}

fn is_false_pred_expr(e: &IRExpr) -> bool {
    matches!(e, IRExpr::Op { op, args } if args.is_empty() && matches!(op.as_str(), "!PT" | "!UPT"))
}

fn carry_inc_expr(
    e: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    if is_true_pred_expr(e) {
        return Some(LiftedExpr::Imm("1".to_string()));
    }
    if is_false_pred_expr(e) {
        return Some(LiftedExpr::Imm("0".to_string()));
    }
    if !is_pred_reg_expr(e) {
        return None;
    }
    let cond = lift_ir_expr(e, stmt_ref, config);
    Some(LiftedExpr::Ternary {
        cond: Box::new(cond),
        then_expr: Box::new(LiftedExpr::Imm("1".to_string())),
        else_expr: Box::new(LiftedExpr::Imm("0".to_string())),
    })
}

fn extract_triplet_operands(args: &[IRExpr]) -> Option<(&IRExpr, &IRExpr, &IRExpr)> {
    if args.len() < 3 {
        return None;
    }
    let mut start = 0usize;
    let mut end = args.len();

    // Strip leading predicate output operands (e.g., IADD3(..., P0, a, b, c)).
    while end.saturating_sub(start) > 3 && is_pred_reg_expr(&args[start]) {
        start += 1;
    }
    // Strip trailing predicate control operands (e.g., ..., !PT()).
    while end.saturating_sub(start) > 3 && is_pred_control_expr(&args[end - 1]) {
        end -= 1;
    }
    if end.saturating_sub(start) == 3 {
        Some((&args[start], &args[start + 1], &args[start + 2]))
    } else {
        None
    }
}

fn extract_addx_operands(args: &[IRExpr]) -> Option<(&IRExpr, &IRExpr, &IRExpr, &IRExpr)> {
    if args.len() < 4 {
        return None;
    }
    let mut start = 0usize;
    let mut end = args.len();

    while end.saturating_sub(start) > 5 && is_pred_reg_expr(&args[start]) {
        start += 1;
    }
    while end.saturating_sub(start) > 4 && is_pred_control_expr(&args[end - 1]) {
        end -= 1;
    }
    if end.saturating_sub(start) == 4 {
        Some((
            &args[start],
            &args[start + 1],
            &args[start + 2],
            &args[start + 3],
        ))
    } else {
        None
    }
}

fn is_zero_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 0,
        IRExpr::ImmF(f) => *f == 0.0,
        IRExpr::Reg(r) => matches!(r.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn is_pred_reg_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::Reg(r) => matches!(r.class.as_str(), "P" | "UP" | "PT" | "UPT"),
        IRExpr::Op { op, args } if args.is_empty() => {
            let core = op.strip_prefix('!').unwrap_or(op.as_str());
            if core == "PT" || core == "UPT" {
                return true;
            }
            let core_no_ssa = core.split('.').next().unwrap_or(core);
            if let Some(num) = core_no_ssa.strip_prefix('P') {
                return num.parse::<u32>().is_ok();
            }
            if let Some(num) = core_no_ssa.strip_prefix("UP") {
                return num.parse::<u32>().is_ok();
            }
            false
        }
        _ => false,
    }
}

fn is_pred_control_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::Reg(r) => matches!(r.class.as_str(), "PT" | "UPT"),
        IRExpr::Op { op, args } if args.is_empty() => {
            matches!(op.as_str(), "PT" | "!PT" | "UPT" | "!UPT")
        }
        _ => false,
    }
}

fn is_mem_expr(e: &IRExpr) -> bool {
    matches!(e, IRExpr::Mem { .. })
}

fn render_shared_u8_ref(
    mem_expr: &IRExpr,
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<String> {
    let IRExpr::Mem { base, offset, .. } = mem_expr else {
        return None;
    };
    let mut idx = render_expr_raw(base, stmt_ref, config);
    if let Some(off) = offset {
        idx.push_str(" + ");
        idx.push_str(&render_expr_raw(off, stmt_ref, config));
    }
    Some(format!("shmem_u8[{}]", idx))
}

fn lift_ir_expr(expr: &IRExpr, stmt_ref: StatementRef, config: &SemanticLiftConfig<'_>) -> LiftedExpr {
    match expr {
        IRExpr::Reg(r) => LiftedExpr::Reg(r.display()),
        IRExpr::ImmI(i) => LiftedExpr::Imm(i.to_string()),
        IRExpr::ImmF(f) => LiftedExpr::Imm(f.to_string()),
        IRExpr::Mem { .. } => LiftedExpr::Raw(render_expr_raw(expr, stmt_ref, config)),
        IRExpr::Op { op, args } => {
            if op == "!" && args.len() == 1 {
                let child = lift_ir_expr(&args[0], stmt_ref, config);
                return simplify_not(child);
            }
            if args.is_empty() {
                if let Some(pred_expr) = parse_inline_predicate_expr(op) {
                    return pred_expr;
                }
            }
            if op == "ConstMem" && args.len() == 2 {
                if let (Some(bank), Some(offset)) = (imm_as_u32(&args[0]), imm_as_u32(&args[1])) {
                    if let Some(sym) = resolve_constmem_symbol(stmt_ref, bank, offset, config) {
                        return LiftedExpr::Raw(sym);
                    }
                }
            }
            LiftedExpr::Raw(render_expr_raw(expr, stmt_ref, config))
        }
    }
}

fn parse_inline_predicate_expr(op: &str) -> Option<LiftedExpr> {
    let (negated, core) = if let Some(rest) = op.strip_prefix('!') {
        (true, rest)
    } else {
        (false, op)
    };

    if core == "PT" || core == "UPT" {
        return Some(LiftedExpr::Imm(if negated { "false" } else { "true" }.to_string()));
    }

    let pred_name = if let Some(num) = core.strip_prefix('P') {
        if num.parse::<u32>().is_ok() {
            Some(core.to_string())
        } else {
            None
        }
    } else if let Some(num) = core.strip_prefix("UP") {
        if num.parse::<u32>().is_ok() {
            Some(core.to_string())
        } else {
            None
        }
    } else {
        None
    }?;

    let base = LiftedExpr::Reg(pred_name);
    if negated {
        Some(simplify_not(base))
    } else {
        Some(base)
    }
}

fn binary_prec(op: &str) -> u8 {
    match op {
        "*" | "/" | "%" => 6,
        "+" | "-" => 5,
        "<<" | ">>" => 4,
        "==" | "!=" | "<" | "<=" | ">" | ">=" => 3,
        "&&" => 2,
        "||" => 1,
        _ => 3,
    }
}

fn simplify_not(expr: LiftedExpr) -> LiftedExpr {
    if let LiftedExpr::Unary { op, arg } = expr {
        if op == "!" {
            return *arg;
        }
        return LiftedExpr::Unary { op, arg };
    }
    LiftedExpr::Unary {
        op: "!".to_string(),
        arg: Box::new(expr),
    }
}

fn render_expr_raw(expr: &IRExpr, stmt_ref: StatementRef, config: &SemanticLiftConfig<'_>) -> String {
    match expr {
        IRExpr::Reg(r) => r.display(),
        IRExpr::ImmI(i) => i.to_string(),
        IRExpr::ImmF(f) => f.to_string(),
        IRExpr::Mem {
            base,
            offset,
            width,
        } => {
            let mut s = format!("*{}", render_expr_raw(base, stmt_ref, config));
            if let Some(off) = offset {
                s.push_str(&format!("+{}", render_expr_raw(off, stmt_ref, config)));
            }
            if let Some(w) = width {
                s.push_str(&format!("@{}", w));
            }
            s
        }
        IRExpr::Op { op, args } => {
            if args.is_empty() {
                match op.as_str() {
                    "PT" | "UPT" => return "true".to_string(),
                    "!PT" | "!UPT" => return "false".to_string(),
                    _ => {}
                }
            }
            if op == "ConstMem" && args.len() == 2 {
                if let (Some(bank), Some(offset)) = (imm_as_u32(&args[0]), imm_as_u32(&args[1])) {
                    if let Some(sym) = resolve_constmem_symbol(stmt_ref, bank, offset, config) {
                        return sym;
                    }
                }
            }
            let list = args
                .iter()
                .map(|a| render_expr_raw(a, stmt_ref, config))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", op, list)
        }
    }
}

fn imm_as_u32(e: &IRExpr) -> Option<u32> {
    match e {
        IRExpr::ImmI(i) if *i >= 0 => u32::try_from(*i).ok(),
        _ => None,
    }
}

fn resolve_constmem_symbol(
    stmt_ref: StatementRef,
    bank: u32,
    offset: u32,
    config: &SemanticLiftConfig<'_>,
) -> Option<String> {
    let anns = config.abi_annotations?;
    let matches_stmt = anns.constmem_by_stmt.get(&stmt_ref)?;
    let ann = matches_stmt
        .iter()
        .find(|ann| ann.bank == bank && ann.offset == offset)?;

    if let ConstMemSemantic::ParamWord { param_idx, word_idx } = ann.semantic {
        if let Some(aliases) = config.abi_aliases {
            if let Some(alias) = aliases.render_param_word(param_idx, word_idx) {
                return Some(alias);
            }
        }
    }

    Some(ann.symbol())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, parse_sass};

    fn run_lift(sass: &str) -> SemanticLiftResult {
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        lift_function_ir(&fir, &SemanticLiftConfig::default())
    }

    #[test]
    fn lifts_imad_mov_to_direct_rhs() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x2a ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted IMAD.MOV");
        assert_eq!(lifted.rhs.render(), "42");
    }

    #[test]
    fn lifts_iadd3_with_zero_elision() {
        let sass = r#"
            /*0000*/ IADD3 R1, R1, -0x1, RZ ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted IADD3");
        let rendered = lifted.rhs.render();
        assert!(rendered.contains("- 1"));
    }

    #[test]
    fn lifts_uiadd3_with_pred_output_form() {
        let sass = r#"
            /*0000*/ UIADD3 UR8, UP0, UR8, 0x4, URZ ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted UIADD3");
        assert_eq!(lifted.rhs.render(), "UR8.0 + 4");
    }

    #[test]
    fn lifts_imad_iadd_with_mul_by_one() {
        let sass = r#"
            /*0000*/ IMAD.IADD R11, R12, 0x1, R11 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted IMAD.IADD");
        assert_eq!(lifted.rhs.render(), "R12.0 + R11.0");
    }

    #[test]
    fn renders_pt_constants_as_bools() {
        let sass = r#"
            /*0000*/ IMAD.IADD R11, R12, 0x1, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted IMAD.IADD");
        assert_eq!(lifted.rhs.render(), "R12.0 + false");
    }

    #[test]
    fn lifts_setp_compare_to_infix() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted ISETP");
        assert_eq!(lifted.rhs.render(), "R0.0 >= 1");
    }

    #[test]
    fn lifts_fsel_to_ternary() {
        let sass = r#"
            /*0000*/ FSEL R5, R7, 0.89999997615814208984, P1 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted FSEL");
        let rendered = lifted.rhs.render();
        assert!(rendered.starts_with("P1.0 ? R7.0 : "));
    }

    #[test]
    fn lifts_sel_to_ternary() {
        let sass = r#"
            /*0000*/ SEL R6, R5, R6, !P1 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted SEL");
        assert_eq!(lifted.rhs.render(), "!P1 ? R5.0 : R6.0");
    }

    #[test]
    fn lifts_shf_hi_right_pattern_to_shift() {
        let sass = r#"
            /*0000*/ SHF.R.S32.HI R3, RZ, 0x1f, R0 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted SHF");
        assert_eq!(lifted.rhs.render(), "R0.0 >> 31");
    }

    #[test]
    fn lifts_imad_wide_to_mul_add() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R2, R27, R2, c[0x0][0x168] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted IMAD.WIDE");
        assert_eq!(lifted.rhs.render(), "R27.0 * R2.0 + ConstMem(0, 360)");
    }

    #[test]
    fn lifts_simple_lea_form() {
        let sass = r#"
            /*0000*/ LEA R2, P0, R0, c[0x0][0x170], 0x2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LEA");
        assert_eq!(lifted.rhs.render(), "R0.0 + (ConstMem(0, 368) << 2)");
    }

    #[test]
    fn lifts_iadd3_x_to_add_with_carry_term() {
        let sass = r#"
            /*0000*/ IADD3.X R3, R1, R2, RZ, P1, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted IADD3.X");
        assert_eq!(lifted.rhs.render(), "R1.0 + R2.0 + (P1.0 ? 1 : 0)");
    }

    #[test]
    fn lifts_lea_hi_to_hi32_form() {
        let sass = r#"
            /*0000*/ LEA.HI R4, R5, R6, RZ, 0x8 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LEA.HI");
        assert_eq!(lifted.rhs.render(), "hi32(R5.0 + (R6.0 << 8))");
    }

    #[test]
    fn lifts_lea_hi_x_to_hi32_plus_carry_term() {
        let sass = r#"
            /*0000*/ LEA.HI.X.SX32 R11, R6, c[0x0][0x164], 0x1, P2 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LEA.HI.X");
        assert_eq!(
            lifted.rhs.render(),
            "hi32(R6.0 + (ConstMem(0, 356) << 1)) + (P2.0 ? 1 : 0)"
        );
    }

    #[test]
    fn lifts_lop3_lut_and_with_zero_third_operand() {
        let sass = r#"
            /*0000*/ LOP3.LUT R17, R22, R17, RZ, 0xc0, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "R22.0 & R17.0");
    }

    #[test]
    fn lifts_lop3_lut_or_with_zero_third_operand() {
        let sass = r#"
            /*0000*/ LOP3.LUT R8, R8, R9, RZ, 0xfc, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "R8.0 | R9.0");
    }

    #[test]
    fn lifts_lop3_lut_not_with_two_zero_inputs() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, RZ, R6, RZ, 0x33, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "~R6.0");
    }

    #[test]
    fn lifts_lop3_lut_binary_when_first_input_zero() {
        let sass = r#"
            /*0000*/ LOP3.LUT R5, RZ, R6, R7, 0x66, !PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LOP3.LUT");
        assert_eq!(lifted.rhs.render(), "R6.0 ^ R7.0");
    }

    #[test]
    fn lifts_lds_to_shared_indexed_read() {
        let sass = r#"
            /*0000*/ LDS.U8 R12, [UR4] ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted LDS");
        assert_eq!(lifted.rhs.render(), "shmem_u8[UR4.0]");
    }

    #[test]
    fn lifts_sts_to_shared_indexed_write() {
        let sass = r#"
            /*0000*/ STS.U8 [UR4], R11 ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        let lifted = out
            .by_stmt
            .get(&StatementRef {
                block_id: 0,
                stmt_idx: 0,
            })
            .expect("expected lifted STS");
        assert_eq!(lifted.dest, "shmem_u8[UR4.0]");
        assert_eq!(lifted.rhs.render(), "R11.0");
    }

    #[test]
    fn unsupported_opcode_is_counted_as_fallback() {
        let sass = r#"
            /*0000*/ LOP3.LUT R0, R1, R2, R3, 0xf8, PT ;
            /*0010*/ EXIT ;
        "#;
        let out = run_lift(sass);
        assert!(out.by_stmt.is_empty());
        assert_eq!(out.stats.attempted, 2);
        assert_eq!(out.stats.lifted, 0);
        assert_eq!(out.stats.fallback, 2);
    }
}
