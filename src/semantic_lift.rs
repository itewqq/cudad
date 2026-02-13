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
    if opcode == "IADD3" {
        return lift_iadd3(args, stmt_ref, config);
    }
    if opcode.starts_with("FMUL") {
        return lift_binary_infix("*", args, stmt_ref, config);
    }
    if opcode.starts_with("FADD") {
        return lift_binary_add_like(args, stmt_ref, config);
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
    if opcode == "LEA" {
        return lift_lea(args, stmt_ref, config);
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
    if args.len() != 3 {
        return None;
    }
    let a0z = is_zero_expr(&args[0]);
    let a1z = is_zero_expr(&args[1]);
    let a2z = is_zero_expr(&args[2]);

    if config.strict && !(a0z || a1z || a2z) {
        return None;
    }

    let expr = if a0z && !a1z && !a2z {
        add_like_expr(
            lift_ir_expr(&args[1], stmt_ref, config),
            lift_ir_expr(&args[2], stmt_ref, config),
        )
    } else if a1z && !a0z && !a2z {
        add_like_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[2], stmt_ref, config),
        )
    } else if a2z && !a0z && !a1z {
        add_like_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[1], stmt_ref, config),
        )
    } else if config.strict {
        return None;
    } else {
        let left = add_like_expr(
            lift_ir_expr(&args[0], stmt_ref, config),
            lift_ir_expr(&args[1], stmt_ref, config),
        );
        add_like_expr(left, lift_ir_expr(&args[2], stmt_ref, config))
    };

    Some(expr)
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
    // LOP3.LUT dst, a, b, RZ, imm, PT/!PT
    // where imm encodes a binary op independent of c:
    //   0xc0 -> a & b
    //   0xfc -> a | b
    //   0x3c -> a ^ b
    if args.len() != 5 {
        return None;
    }
    if !is_zero_expr(&args[2]) {
        return None;
    }
    if !is_pred_control_expr(&args[4]) {
        return None;
    }

    let imm = imm_as_u32(&args[3])?;
    let op = match imm & 0xff {
        0xc0 => "&",
        0xfc => "|",
        0x3c => "^",
        _ => return None,
    };
    Some(LiftedExpr::Binary {
        op: op.to_string(),
        lhs: Box::new(lift_ir_expr(&args[0], stmt_ref, config)),
        rhs: Box::new(lift_ir_expr(&args[1], stmt_ref, config)),
    })
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
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
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

fn is_true_pred_expr(e: &IRExpr) -> bool {
    let IRExpr::Reg(r) = e else {
        return false;
    };
    matches!(r.class.as_str(), "PT" | "UPT")
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

    let pred_name = if core == "PT" || core == "UPT" {
        Some(core.to_string())
    } else if let Some(num) = core.strip_prefix('P') {
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
