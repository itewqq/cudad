//! SSA construction and decoded-operand lowering for the canonical backend.
//!
//! Purpose:
//! - lower tolerant decoded operands into a stable SSA-friendly IR
//! - build SSA with Cytron-style phi placement and renaming
//! - preserve address structure needed by `FunctionAnalysis`
//!
//! Inputs:
//! - `ControlFlowGraph`
//! - decoded instructions / operands from the tolerant parser
//!
//! Outputs:
//! - `FunctionIR` and `IRExpr` values suitable for optimization and analysis
//!
//! Invariants:
//! - address operands must preserve wide-pair and scaled-index structure
//! - SSA statements keep explicit memory-address operands in `mem_addr_args`
//! - this module owns IR construction, not pseudo-C rendering
//!
//! This module must not:
//! - recover semantics by inspecting rendered output
//! - silently discard address modifiers that later stages need

use crate::cfg::{ControlFlowGraph, EdgeKind};
use crate::op_semantics::{derive_op_semantics, OpSemantics};
use crate::op_semantics::UseRole;
use crate::parser::DecodedOperand;
use petgraph::visit::EdgeRef;
use petgraph::{graph::NodeIndex, Direction};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/* =======================================================================
   Section 0 – Printing context
======================================================================= */
/// An external formatter that decides how to show registers / expressions
pub trait DisplayCtx {
    fn reg(&self, r: &RegId) -> String;
    fn expr(&self, e: &IRExpr) -> String {
        match e {
            IRExpr::Reg(r) => self.reg(r),
            IRExpr::ImmI(i) => format!("{}", i),
            IRExpr::ImmF(f) => format!("{}", f),
            IRExpr::Addr64 { lo, hi } => {
                format!("addr64({}, {})", self.expr(lo), self.expr(hi))
            }
            IRExpr::Mem {
                base,
                offset,
                width,
            } => {
                let s = if let Some(off) = offset {
                    format!("*({} + {})", self.expr(base), self.expr(off))
                } else {
                    format!("*{}", self.expr(base))
                };
                let _ = width;
                s
            }
            IRExpr::Op { op, args } => {
                if op == "-" && args.len() == 1 {
                    let inner = self.expr(&args[0]);
                    let simple = match &args[0] {
                        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => true,
                        IRExpr::Op { op, .. } if op == "ConstMem" => true,
                        _ => false,
                    };
                    if simple {
                        return format!("-{}", inner);
                    }
                    return format!("-({})", inner);
                }
                let list = args
                    .iter()
                    .map(|a| self.expr(a))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", op, list)
            }
        }
    }
}
/// Default formatter ⟶ 原先 display() 效果
pub struct DefaultDisplay;
impl DisplayCtx for DefaultDisplay {
    fn reg(&self, r: &RegId) -> String {
        r.display()
    }
}

/* =======================================================================
   Section 1 – Core IR data structures  (unchanged except `DisplayCtx`)
======================================================================= */
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegType {
    BitWidth(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct RegId {
    pub class: String,
    pub idx: i32,
    pub sign: i32,
    pub ssa: Option<usize>,
}
impl RegId {
    pub fn new(class: &str, idx: i32, sign: i32) -> Self {
        Self {
            class: class.into(),
            idx,
            sign,
            ssa: None,
        }
    }
    pub fn with_ssa(&self, v: usize) -> Self {
        let mut r = self.clone();
        r.ssa = Some(v);
        r
    }
    pub fn display(&self) -> String {
        let base = match self.class.as_str() {
            // Immutable pseudo-registers print without numeric suffix.
            "RZ" | "PT" | "URZ" | "UPT" => self.class.clone(),
            _ => format!("{}{}", self.class, self.idx),
        };
        let ssa = self.ssa.map(|v| format!(".{}", v)).unwrap_or_default();
        let sign = if self.sign < 0 { "-" } else { "" };
        format!("{}{}{}", sign, base, ssa)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IRCond {
    True,
    Pred { reg: RegId, sense: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub enum IRExpr {
    Reg(RegId),
    ImmI(i64),
    ImmF(f64),
    Addr64 {
        lo: Box<IRExpr>,
        hi: Box<IRExpr>,
    },
    Mem {
        base: Box<IRExpr>,
        offset: Option<Box<IRExpr>>,
        width: Option<u32>,
    },
    Op {
        op: String,
        args: Vec<IRExpr>,
    },
}
impl IRExpr {
    pub fn get_reg(&self) -> Option<&RegId> {
        if let IRExpr::Reg(r) = self {
            Some(r)
        } else {
            None
        }
    }
    pub fn get_reg_mut(&mut self) -> Option<&mut RegId> {
        if let IRExpr::Reg(r) = self {
            Some(r)
        } else {
            None
        }
    }

    /// Collect all register references reachable from this expression.
    pub fn collect_reg_uses(&self, out: &mut Vec<RegId>) {
        match self {
            IRExpr::Reg(r) => out.push(r.clone()),
            IRExpr::Addr64 { lo, hi } => {
                lo.collect_reg_uses(out);
                hi.collect_reg_uses(out);
            }
            IRExpr::Mem { base, offset, .. } => {
                base.collect_reg_uses(out);
                if let Some(off) = offset {
                    off.collect_reg_uses(out);
                }
            }
            IRExpr::Op { args, .. } => {
                for a in args {
                    a.collect_reg_uses(out);
                }
            }
            IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RValue {
    Op { opcode: String, args: Vec<IRExpr> },
    Phi(Vec<IRExpr>),
    ImmI(i64),
    ImmF(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRStatement {
    pub defs: Vec<IRExpr>,
    pub value: RValue,
    pub pred: Option<IRExpr>,
    pub mem_addr_args: Option<Vec<IRExpr>>,
    /// For predicated non-branch instructions: the *previous* SSA version of
    /// each def register (i.e. the value the register keeps when the predicate
    /// is false).  Populated during SSA renaming so the renderer can emit
    /// `dest = pred ? rhs : old;` instead of `if (pred) dest = rhs;`.
    pub pred_old_defs: Vec<IRExpr>,
}

impl IRStatement {
    /// Collect all register references *used* by this statement (arguments,
    /// predicates, memory address args, pred_old_defs — but NOT the def LHS).
    pub fn collect_all_uses(&self) -> Vec<RegId> {
        let mut out = Vec::new();
        if let Some(p) = &self.pred {
            p.collect_reg_uses(&mut out);
        }
        match &self.value {
            RValue::Op { args, .. } => {
                for a in args {
                    a.collect_reg_uses(&mut out);
                }
            }
            RValue::Phi(args) => {
                for a in args {
                    a.collect_reg_uses(&mut out);
                }
            }
            RValue::ImmI(_) | RValue::ImmF(_) => {}
        }
        if let Some(mem) = &self.mem_addr_args {
            for a in mem {
                a.collect_reg_uses(&mut out);
            }
        }
        for old in &self.pred_old_defs {
            old.collect_reg_uses(&mut out);
        }
        out
    }

    /// Returns true if this statement has observable side effects beyond
    /// defining registers (memory stores, barriers, atomics, etc.).
    pub fn is_side_effectful(&self) -> bool {
        match &self.value {
            RValue::Op { opcode, .. } => {
                let mnem = opcode.split('.').next().unwrap_or(opcode);
                // Memory stores
                if opcode.starts_with("ST")   // STG, STS, STL, ST
                    || opcode.starts_with("RED")   // Reduction
                    || mnem == "MEMBAR"
                    || mnem == "DEPBAR"
                    || mnem == "FENCE"
                    || mnem == "BAR"
                    || mnem == "WARPSYNC"
                {
                    return true;
                }
                // Atomics (ATOM, ATOMS, ATOMG, etc.)
                if mnem.starts_with("ATOM") {
                    return true;
                }
                // EXIT / RET / BRA / BRX — control flow
                if matches!(
                    mnem,
                    "EXIT"
                        | "RET"
                        | "BRA"
                        | "BRX"
                        | "BREAK"
                        | "CONT"
                        | "BSSY"
                        | "BSYNC"
                        | "SSY"
                        | "SYNC"
                        | "CALL"
                        | "VOTE"
                ) {
                    return true;
                }
                // If has memory address args for stores, it's side-effectful
                if self.mem_addr_args.is_some() && opcode.starts_with("ST") {
                    return true;
                }
                false
            }
            RValue::Phi(_) | RValue::ImmI(_) | RValue::ImmF(_) => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRBlock {
    pub id: usize,
    pub start_addr: u32,
    pub irdst: Vec<(Option<IRCond>, u32)>,
    pub stmts: Vec<IRStatement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionIR {
    pub blocks: Vec<IRBlock>,
}

/* =======================================================================
   Section 2 – Helpers: parser→IR lowering
======================================================================= */
fn phys_reg_of(op: &DecodedOperand) -> Option<RegId> {
    match op {
        DecodedOperand::Register {
            class, idx, sign, ..
        }
        | DecodedOperand::UniformRegister {
            class, idx, sign, ..
        } => match class.as_str() {
            "RZ" | "PT" | "URZ" | "UPT" => Some(RegId::new(class, 0, *sign)),
            "UP" | "P" => Some(RegId::new(class, *idx, *sign)),
            _ => Some(RegId::new(class, *idx, *sign)),
        },
        DecodedOperand::PredicateRegister { class, idx, .. } => Some(RegId::new(class, *idx, 1)),
        _ => None,
    }
}

fn parse_raw_inline_expr(text: &str) -> Option<IRExpr> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let parts: Vec<&str> = text.split('+').map(str::trim).collect();
    if parts.len() > 1 {
        let mut iter = parts.into_iter();
        let first = parse_raw_inline_expr(iter.next()?)?;
        return iter.try_fold(first, |lhs, part| {
            Some(IRExpr::Op {
                op: "+".to_string(),
                args: vec![lhs, parse_raw_inline_expr(part)?],
            })
        });
    }

    let (negated, core) = text
        .strip_prefix('-')
        .map(|rest| (true, rest.trim()))
        .unwrap_or((false, text));
    let expr = parse_raw_scaled_reg(core)
        .or_else(|| core.parse::<i64>().ok().map(IRExpr::ImmI))?;
    if negated {
        Some(IRExpr::Op {
            op: "-".to_string(),
            args: vec![expr],
        })
    } else {
        Some(expr)
    }
}

fn parse_raw_scaled_reg(text: &str) -> Option<IRExpr> {
    let (base, scale) = if let Some((base, scale)) = text.split_once(".X") {
        (base, scale.parse::<i64>().ok()?)
    } else {
        (text, 1)
    };
    let reg = parse_raw_reg_token(base)?;
    let expr = IRExpr::Reg(reg);
    if scale == 1 {
        Some(expr)
    } else {
        Some(IRExpr::Op {
            op: "*".to_string(),
            args: vec![expr, IRExpr::ImmI(scale)],
        })
    }
}

fn parse_raw_reg_token(text: &str) -> Option<RegId> {
    let text = text.trim();
    if matches!(text, "RZ" | "URZ" | "PT" | "UPT") {
        return Some(RegId::new(text, 0, 1));
    }
    let (class, digits) = if let Some(rest) = text.strip_prefix("UR") {
        ("UR", rest)
    } else if let Some(rest) = text.strip_prefix("UP") {
        ("UP", rest)
    } else if let Some(rest) = text.strip_prefix('R') {
        ("R", rest)
    } else if let Some(rest) = text.strip_prefix('P') {
        ("P", rest)
    } else {
        return None;
    };
    let idx = digits.parse::<i32>().ok()?;
    Some(RegId::new(class, idx, 1))
}

fn lower_operand(op: &DecodedOperand) -> IRExpr {
    match op {
        DecodedOperand::Register { sign, abs, .. }
        | DecodedOperand::UniformRegister { sign, abs, .. } => {
            if !*abs {
                if let Some(r) = phys_reg_of(op) {
                    IRExpr::Reg(r)
                } else {
                    IRExpr::ImmI(0)
                }
            } else {
                let mut base = op.clone();
                match &mut base {
                    DecodedOperand::Register { sign, abs, .. }
                    | DecodedOperand::UniformRegister { sign, abs, .. } => {
                        *sign = 1;
                        *abs = false;
                    }
                    _ => {}
                }
                let reg = phys_reg_of(&base)
                    .map(IRExpr::Reg)
                    .unwrap_or(IRExpr::ImmI(0));
                let mut expr = IRExpr::Op {
                    op: "abs".into(),
                    args: vec![reg],
                };
                if *sign < 0 {
                    expr = IRExpr::Op {
                        op: "-".into(),
                        args: vec![expr],
                    };
                }
                expr
            }
        }
        DecodedOperand::PredicateRegister {
            class,
            idx: _,
            sense,
        } => {
            if *sense {
                phys_reg_of(op).map(IRExpr::Reg).unwrap_or(IRExpr::ImmI(0))
            } else if matches!(class.as_str(), "PT" | "UPT") {
                IRExpr::Op {
                    op: format!("!{}", class),
                    args: vec![],
                }
            } else {
                let base = phys_reg_of(op).map(IRExpr::Reg).unwrap_or(IRExpr::ImmI(0));
                IRExpr::Op {
                    op: "!".into(),
                    args: vec![base],
                }
            }
        }
        DecodedOperand::ImmediateI(i) => IRExpr::ImmI(*i),
        DecodedOperand::ImmediateF(f) => IRExpr::ImmF(*f),
        DecodedOperand::ConstMem { bank, offset } => IRExpr::Op {
            op: "ConstMem".into(),
            args: vec![IRExpr::ImmI(*bank as i64), IRExpr::ImmI(*offset as i64)],
        },
        DecodedOperand::Address {
            base,
            offset,
            width,
            scale,
            ..
        } => {
            let mut base_expr = lower_operand(base.as_ref());
            if matches!(width, Some(64)) {
                if let Some(hi_expr) = infer_64bit_pair_hi_expr(base.as_ref()) {
                    base_expr = IRExpr::Addr64 {
                        lo: Box::new(base_expr),
                        hi: Box::new(hi_expr),
                    };
                }
            }
            if let Some(scale) = scale.filter(|scale| *scale > 1) {
                base_expr = IRExpr::Op {
                    op: "*".into(),
                    args: vec![base_expr, IRExpr::ImmI(i64::from(scale))],
                };
            }
            let off_expr = offset.as_ref().map(|v| Box::new(IRExpr::ImmI(*v)));
            IRExpr::Mem {
                base: Box::new(base_expr),
                offset: off_expr,
                width: *width,
            }
        }
        DecodedOperand::DescriptorMem { addr, .. } => lower_operand(addr.as_ref()),
        DecodedOperand::Raw(s) => {
            if let Some(pred) = parse_raw_predicate_token(s) {
                return pred;
            }
            if let Some((neg, bank, off)) = parse_raw_constmem_token(s) {
                let cm = IRExpr::Op {
                    op: "ConstMem".into(),
                    args: vec![IRExpr::ImmI(bank as i64), IRExpr::ImmI(off as i64)],
                };
                if neg {
                    return IRExpr::Op {
                        op: "-".into(),
                        args: vec![cm],
                    };
                }
                return cm;
            }
            if let Some(expr) = parse_raw_inline_expr(s) {
                return expr;
            }
            if let Ok(i) = s.parse::<i64>() {
                IRExpr::ImmI(i)
            } else if let Ok(f) = s.parse::<f64>() {
                IRExpr::ImmF(f)
            } else {
                IRExpr::Op {
                    op: s.clone(),
                    args: vec![],
                }
            }
        }
    }
}

fn opcode_has_implicit_global_addr64(opcode: &str) -> bool {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    matches!(mnem, "ATOM" | "ATOMG")
}

fn lower_addr_operand_with_implicit_width(op: &DecodedOperand) -> Option<IRExpr> {
    match op {
        DecodedOperand::Address {
            base,
            offset,
            width: None,
            scale,
            ..
        } => {
            let mut lo = lower_operand(base.as_ref());
            let hi = infer_64bit_pair_hi_expr(base.as_ref())?;
            if let Some(scale) = scale.filter(|scale| *scale > 1) {
                lo = IRExpr::Op {
                    op: "*".into(),
                    args: vec![lo, IRExpr::ImmI(i64::from(scale))],
                };
            }
            Some(IRExpr::Mem {
                base: Box::new(IRExpr::Addr64 {
                    lo: Box::new(lo),
                    hi: Box::new(hi),
                }),
                offset: offset.as_ref().map(|v| Box::new(IRExpr::ImmI(*v))),
                width: Some(64),
            })
        }
        DecodedOperand::DescriptorMem { addr, .. } => {
            lower_addr_operand_with_implicit_width(addr.as_ref())
        }
        _ => None,
    }
}

fn lower_operand_for_role(opcode: &str, role: Option<&UseRole>, op: &DecodedOperand) -> IRExpr {
    if role == Some(&UseRole::Addr) && opcode_has_implicit_global_addr64(opcode) {
        if let Some(promoted) = lower_addr_operand_with_implicit_width(op) {
            return promoted;
        }
    }
    lower_operand(op)
}

fn implicit_extra_data_defs(opcode: &str) -> usize {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    let has_mod = |needle: &str| opcode.split('.').skip(1).any(|t| t == needle);
    if matches!(mnem, "ULDC" | "LDC" | "LDCU") {
        if has_mod("128") {
            return 3;
        }
        if has_mod("64") {
            return 1;
        }
        return 0;
    }
    // MOV.64 is a true register-pair copy. Model the hi lane explicitly so
    // SSA, copyprop, and addr64 recovery can track both halves of the tuple.
    if mnem == "MOV" && has_mod("64") {
        return 1;
    }
    // IADD.64 writes a register-pair result. Model the hi lane explicitly so
    // later `Rn.64` users bind to the updated high half instead of a stale
    // live-in value of `R(n+1)`.
    if mnem == "IADD" && has_mod("64") {
        return 1;
    }
    // IADD3.64 / UIADD3.64 also update the implicit hi lane of the destination
    // pair. If we fail to model that lane explicitly, later addr64 recovery
    // sees a stale live-in hi word and reconstructs packed pointers.
    if matches!(mnem, "IADD3" | "UIADD3") && has_mod("64") {
        return 1;
    }
    if matches!(mnem, "IMAD" | "UIMAD") && has_mod("WIDE") {
        return 1;
    }
    // Model implicit register tuples for wide memory loads so SSA renaming,
    // lifting, and name recovery do not treat the upper lanes as live-ins.
    if mnem.starts_with("LD") {
        if has_mod("128") {
            return 3;
        }
        if has_mod("64") {
            return 1;
        }
    }
    0
}

fn sign_extend_i32_hi_word(value: i64) -> i64 {
    if ((value as u32) & 0x8000_0000) != 0 {
        -1
    } else {
        0
    }
}

fn parse_i64_literal(text: &str) -> Option<i64> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("-0x") {
        return i64::from_str_radix(rest, 16).ok().map(|v| -v);
    }
    if let Some(rest) = trimmed.strip_prefix("0x") {
        return i64::from_str_radix(rest, 16).ok();
    }
    trimmed.parse::<i64>().ok()
}

fn operand_lane(op: &DecodedOperand, lane: usize) -> Option<DecodedOperand> {
    match op {
        DecodedOperand::Register {
            class,
            idx,
            sign,
            abs,
            reuse,
            ty,
        } => Some(DecodedOperand::Register {
            class: class.clone(),
            idx: idx + lane as i32,
            sign: *sign,
            abs: *abs,
            reuse: *reuse,
            ty: ty.clone(),
        }),
        DecodedOperand::UniformRegister {
            class,
            idx,
            sign,
            abs,
            reuse,
            ty,
        } => Some(DecodedOperand::UniformRegister {
            class: class.clone(),
            idx: idx + lane as i32,
            sign: *sign,
            abs: *abs,
            reuse: *reuse,
            ty: ty.clone(),
        }),
        DecodedOperand::ConstMem { bank, offset } => {
            let delta = u32::try_from(lane).ok()?.checked_mul(4)?;
            Some(DecodedOperand::ConstMem {
                bank: *bank,
                offset: offset.checked_add(delta)?,
            })
        }
        DecodedOperand::ImmediateI(i) => Some(DecodedOperand::ImmediateI(if lane == 0 {
            *i
        } else {
            sign_extend_i32_hi_word(*i)
        })),
        DecodedOperand::Raw(text) if lane == 0 => Some(DecodedOperand::Raw(text.clone())),
        DecodedOperand::Raw(text) => {
            if let Some((negated, bank, offset)) = parse_raw_constmem_token(text) {
                if negated {
                    return None;
                }
                let delta = u32::try_from(lane).ok()?.checked_mul(4)?;
                return Some(DecodedOperand::ConstMem {
                    bank,
                    offset: offset.checked_add(delta)?,
                });
            }
            if let Some(value) = parse_i64_literal(text) {
                return Some(DecodedOperand::ImmediateI(if lane == 0 {
                    value
                } else {
                    sign_extend_i32_hi_word(value)
                }));
            }
            None
        }
        _ => None,
    }
}

fn lower_operand_lane(op: &DecodedOperand, lane: usize) -> Option<IRExpr> {
    let lane_op = operand_lane(op, lane)?;
    Some(lower_operand(&lane_op))
}

fn implicit_store_source_lane_use_exprs(opcode: &str, operands: &[DecodedOperand]) -> Vec<IRExpr> {
    if !opcode.starts_with("ST") {
        return Vec::new();
    }
    let extra_defs = if opcode.split('.').skip(1).any(|t| t == "128") {
        3
    } else if opcode.split('.').skip(1).any(|t| t == "64") {
        1
    } else {
        0
    };
    if extra_defs == 0 {
        return Vec::new();
    }

    let Some(src) = operands.get(1) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(extra_defs);
    for lane in 1..=extra_defs {
        let Some(expr) = lower_operand_lane(src, lane) else {
            break;
        };
        out.push(expr);
    }
    out
}

fn implicit_extra_use_exprs(
    opcode: &str,
    operands: &[DecodedOperand],
    sem: &OpSemantics,
) -> Vec<IRExpr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    let has_mod = |needle: &str| opcode.split('.').skip(1).any(|t| t == needle);
    if mnem == "MOV" && has_mod("64") {
        let Some(src) = operands.get(1) else {
            return Vec::new();
        };
        let Some(src_hi) = lower_operand_lane(src, 1) else {
            return Vec::new();
        };
        return vec![src_hi];
    }
    if mnem == "IADD" && has_mod("64") {
        let (Some(lhs), Some(rhs)) = (operands.get(1), operands.get(2)) else {
            return Vec::new();
        };
        let (Some(lhs_hi), Some(rhs_hi)) = (lower_operand_lane(lhs, 1), lower_operand_lane(rhs, 1))
        else {
            return Vec::new();
        };
        return vec![lhs_hi, rhs_hi];
    }
    if matches!(mnem, "IADD3" | "UIADD3") && has_mod("64") {
        let mut out = Vec::new();
        for operand_idx in sem.use_operand_indices.iter().take(3) {
            let Some(op) = operands.get(*operand_idx) else {
                return Vec::new();
            };
            let Some(expr) = lower_operand_lane(op, 1) else {
                return Vec::new();
            };
            out.push(expr);
        }
        return out;
    }
    Vec::new()
}

fn infer_64bit_pair_hi_expr(base: &DecodedOperand) -> Option<IRExpr> {
    match base {
        DecodedOperand::Register { class, idx, .. }
        | DecodedOperand::UniformRegister { class, idx, .. }
            if class == "R" || class == "UR" =>
        {
            Some(IRExpr::Reg(RegId::new(class, idx + 1, 1)))
        }
        _ => None,
    }
}

fn parse_raw_constmem_token(s: &str) -> Option<(bool, u32, u32)> {
    let t = s.trim();
    let (neg, core) = if let Some(rest) = t.strip_prefix('-') {
        (true, rest)
    } else {
        (false, t)
    };
    if !(core.starts_with("c[") && core.ends_with(']')) {
        return None;
    }
    let mut parts = core.split('[');
    let head = parts.next()?;
    if head != "c" {
        return None;
    }
    let bank_part = parts.next()?.strip_suffix(']')?;
    let off_part = parts.next()?.strip_suffix(']')?;
    if parts.next().is_some() {
        return None;
    }
    if !bank_part.starts_with("0x") || !off_part.starts_with("0x") {
        return None;
    }
    let bank = u32::from_str_radix(&bank_part[2..], 16).ok()?;
    let off = u32::from_str_radix(&off_part[2..], 16).ok()?;
    Some((neg, bank, off))
}

fn parse_raw_predicate_token(s: &str) -> Option<IRExpr> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let (neg, core) = if let Some(rest) = t.strip_prefix('!') {
        (true, rest)
    } else {
        (false, t)
    };

    let base = if let Some(num) = core.strip_prefix('P') {
        let idx = num.parse::<i32>().ok()?;
        IRExpr::Reg(RegId::new("P", idx, 1))
    } else if let Some(num) = core.strip_prefix("UP") {
        let idx = num.parse::<i32>().ok()?;
        IRExpr::Reg(RegId::new("UP", idx, 1))
    } else {
        return None;
    };

    if neg {
        Some(IRExpr::Op {
            op: "!".into(),
            args: vec![base],
        })
    } else {
        Some(base)
    }
}
/* mem load/store heuristics */
fn is_mem_load(op: &str) -> bool {
    op.starts_with("LD")
}
fn is_mem_store(op: &str) -> bool {
    op.starts_with("ST")
}

/* =======================================================================
   Section 3 – Build SSA (Cytron algorithm)
======================================================================= */

/// 构建最小 Φ + 重命名后的 SSA IR
pub fn build_ssa(cfg: &ControlFlowGraph) -> FunctionIR {
    use petgraph::algo::dominators::simple_fast;
    use petgraph::graph::NodeIndex;
    use petgraph::Direction;
    use std::collections::{HashMap, HashSet};

    if cfg.node_count() == 0 {
        return FunctionIR { blocks: Vec::new() };
    }

    /* ---------- helpers ---------- */
    fn base_reg(r: &RegId) -> RegId {
        // SSA identity is register family + index only.
        // Unary sign is an expression-level property and must not fork SSA chains.
        let mut out = r.clone();
        out.ssa = None;
        out.sign = 1;
        out
    }
    fn is_immutable_reg(r: &RegId) -> bool {
        matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
    }
    fn new_ssa(r: &RegId, cnt: &mut HashMap<RegId, usize>) -> RegId {
        let key = base_reg(r);
        let v = cnt.entry(key.clone()).or_insert(0);
        let out = key.with_ssa(*v);
        *v += 1;
        out
    }
    fn top_or_new<'a>(
        key: &RegId,
        stack: &'a mut HashMap<RegId, Vec<RegId>>,
        cnt: &mut HashMap<RegId, usize>,
    ) -> &'a RegId {
        let slot = stack.entry(key.clone()).or_default();
        if slot.is_empty() {
            let tmp = key.with_ssa(*cnt.entry(key.clone()).or_insert(0));
            *cnt.get_mut(key).unwrap() += 1;
            slot.push(tmp);
        }
        slot.last().unwrap()
    }
    fn find_entry_node(cfg: &ControlFlowGraph) -> Option<NodeIndex> {
        let mut candidates = cfg
            .node_indices()
            .filter(|&n| cfg.neighbors_directed(n, Direction::Incoming).count() == 0)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return cfg.node_indices().next();
        }
        candidates.sort_by_key(|&n| cfg[n].id);
        candidates.first().copied()
    }
    fn rename_expr(
        e: &mut IRExpr,
        stack: &mut HashMap<RegId, Vec<RegId>>,
        cnt: &mut HashMap<RegId, usize>,
    ) {
        match e {
            IRExpr::Reg(r) => {
                if is_immutable_reg(r) {
                    return;
                }
                let use_sign = r.sign;
                let mut top = top_or_new(&base_reg(r), stack, cnt).clone();
                top.sign = use_sign;
                *r = top;
            }
            IRExpr::Addr64 { lo, hi } => {
                rename_expr(lo, stack, cnt);
                rename_expr(hi, stack, cnt);
            }
            IRExpr::Mem { base, offset, .. } => {
                rename_expr(base, stack, cnt);
                if let Some(off) = offset {
                    rename_expr(off, stack, cnt);
                }
            }
            IRExpr::Op { args, .. } => {
                for a in args {
                    rename_expr(a, stack, cnt);
                }
            }
            _ => {}
        }
    }
    fn parse_pred_reg_name(name: &str) -> Option<RegId> {
        if let Some(num) = name.strip_prefix("UP") {
            let idx = num.parse::<i32>().ok()?;
            return Some(RegId::new("UP", idx, 1));
        }
        if let Some(num) = name.strip_prefix('P') {
            let idx = num.parse::<i32>().ok()?;
            return Some(RegId::new("P", idx, 1));
        }
        None
    }

    /* ---------- step-0: build IR blocks & defsites ---------- */
    let mut ir_blocks = HashMap::<usize, IRBlock>::new();
    let mut defsites = BTreeMap::<RegId, BTreeSet<NodeIndex>>::new();

    for n in cfg.node_indices() {
        let bb = &cfg[n];
        let mut stmts = Vec::<IRStatement>::new();

        for ins in &bb.instrs {
            let mut defs = Vec::<IRExpr>::new();
            let mut args = Vec::<IRExpr>::new();
            let mut mem_addr_args = Vec::<IRExpr>::new();
            let sem = derive_op_semantics(
                &ins.opcode,
                &ins.operands,
                is_mem_load(&ins.opcode),
                is_mem_store(&ins.opcode),
            );

            for idx in &sem.def_operand_indices {
                if let Some(r) = ins.operands.get(*idx).and_then(phys_reg_of) {
                    defs.push(IRExpr::Reg(r.clone()));
                    if !is_immutable_reg(&r) {
                        defsites.entry(base_reg(&r)).or_default().insert(n);
                    }
                }
            }
            // Model implicit multi-register defs explicitly so SSA renaming
            // does not bind `Rn.64` high halves to an unrelated live value
            // of `R(n+1)`.
            let extra_defs = implicit_extra_data_defs(&ins.opcode);
            if extra_defs > 0 {
                if let Some(base_def) = ins.operands.first().and_then(phys_reg_of) {
                    for k in 1..=extra_defs {
                        let mut hi = base_def.clone();
                        hi.idx += k as i32;
                        hi.sign = 1;
                        defs.push(IRExpr::Reg(hi.clone()));
                        if !is_immutable_reg(&hi) {
                            defsites.entry(base_reg(&hi)).or_default().insert(n);
                        }
                    }
                }
            }
            for (use_pos, idx) in sem.use_operand_indices.iter().enumerate() {
                if let Some(op) = ins.operands.get(*idx) {
                    let lowered = lower_operand_for_role(&ins.opcode, sem.use_roles.get(use_pos), op);
                    if sem.use_roles.get(use_pos) == Some(&UseRole::Addr) {
                        mem_addr_args.push(lowered.clone());
                    }
                    args.push(lowered);
                }
            }
            args.extend(implicit_extra_use_exprs(&ins.opcode, &ins.operands, &sem));
            if is_mem_store(&ins.opcode) {
                args.extend(implicit_store_source_lane_use_exprs(
                    &ins.opcode,
                    &ins.operands,
                ));
            }

            let pred_expr = ins.pred.as_ref().and_then(|p| {
                let base = IRExpr::Reg(parse_pred_reg_name(&p.reg)?);
                if p.sense {
                    Some(base)
                } else {
                    Some(IRExpr::Op {
                        op: "!".into(),
                        args: vec![base],
                    })
                }
            });

            stmts.push(IRStatement {
                defs,
                value: RValue::Op {
                    opcode: ins.opcode.clone(),
                    args,
                },
                pred: pred_expr,
                mem_addr_args: (!mem_addr_args.is_empty()).then_some(mem_addr_args),
                pred_old_defs: vec![],
            });
        }

        /* IRDst */
        let mut irdst = Vec::<(Option<IRCond>, u32)>::new();
        let last_pred = bb.instrs.last().and_then(|i| i.pred.as_ref());
        for e in cfg.edges(n) {
            let tgt_addr = cfg[e.target()].start;
            match *e.weight() {
                EdgeKind::CondBranch => {
                    if let Some(p) = last_pred {
                        if let Some(pred_reg) = parse_pred_reg_name(&p.reg) {
                            irdst.push((
                                Some(IRCond::Pred {
                                    reg: pred_reg,
                                    sense: p.sense,
                                }),
                                tgt_addr,
                            ));
                        }
                    }
                }
                EdgeKind::FallThrough => {
                    if let Some(p) = last_pred {
                        if let Some(pred_reg) = parse_pred_reg_name(&p.reg) {
                            irdst.push((
                                Some(IRCond::Pred {
                                    reg: pred_reg,
                                    sense: !p.sense,
                                }),
                                tgt_addr,
                            ));
                        }
                    } else {
                        irdst.push((Some(IRCond::True), tgt_addr));
                    }
                }
                EdgeKind::UncondBranch => irdst.push((Some(IRCond::True), tgt_addr)),
            }
        }

        ir_blocks.insert(
            bb.id,
            IRBlock {
                id: bb.id,
                start_addr: bb.start,
                irdst,
                stmts,
            },
        );
    }

    /* ---------- step-1: DomTree + DF ---------- */
    let entry = find_entry_node(cfg).unwrap_or_else(|| NodeIndex::new(0));
    let doms = simple_fast(cfg, entry);
    let mut idom = BTreeMap::<NodeIndex, NodeIndex>::new();
    for n in cfg.node_indices() {
        if let Some(i) = doms.immediate_dominator(n) {
            idom.insert(n, i);
        }
    }
    let df = compute_df(cfg, &idom, entry);

    /* ---------- step-2: Φ placement ---------- */
    let mut phi_needed = BTreeSet::<(NodeIndex, RegId)>::new();
    for (reg, defs) in &defsites {
        let mut work: Vec<_> = defs.iter().copied().collect();
        let mut seen = HashSet::<NodeIndex>::new();
        while let Some(x) = work.pop() {
            for &y in df.get(&x).unwrap_or(&BTreeSet::new()) {
                if seen.insert(y) {
                    phi_needed.insert((y, reg.clone()));
                    if !defs.contains(&y) {
                        work.push(y);
                    }
                }
            }
        }
    }

    /* 在块头插入 Φ，占位向量长度 = succ.in_edges() 长度 */
    for (blk_node, reg) in &phi_needed {
        let block_id = cfg[*blk_node].id;
        let preds: Vec<_> = cfg
            .neighbors_directed(*blk_node, Direction::Incoming)
            .collect();
        let placeholder = vec![IRExpr::ImmI(0); preds.len()];
        ir_blocks.get_mut(&block_id).unwrap().stmts.insert(
            0,
            IRStatement {
                defs: vec![IRExpr::Reg(reg.clone())],
                value: RValue::Phi(placeholder),
                pred: None,
                mem_addr_args: None,
                pred_old_defs: vec![],
            },
        );
    }

    /* ---------- step-3: Rename (Cytron WHICH-PRED) ---------- */
    // dom children
    let mut children = BTreeMap::<NodeIndex, Vec<NodeIndex>>::new();
    for (&b, &p) in &idom {
        children.entry(p).or_default().push(b);
    }

    let mut stack = HashMap::<RegId, Vec<RegId>>::new();
    let mut counter = HashMap::<RegId, usize>::new();

    fn rename(
        n: NodeIndex,
        cfg: &ControlFlowGraph,
        children: &BTreeMap<NodeIndex, Vec<NodeIndex>>,
        ir_blocks: &mut HashMap<usize, IRBlock>,
        stack: &mut HashMap<RegId, Vec<RegId>>,
        counter: &mut HashMap<RegId, usize>,
    ) {
        let bid = cfg[n].id;

        /* 1. 处理当前块 */
        {
            let blk = ir_blocks.get_mut(&bid).unwrap();

            /* φ 左值 */
            for stmt in blk
                .stmts
                .iter_mut()
                .filter(|s| matches!(s.value, RValue::Phi(_)))
            {
                let key = base_reg(stmt.defs.first().unwrap().get_reg().unwrap());
                let new = IRExpr::Reg(new_ssa(&key, counter));
                stack
                    .entry(key.clone())
                    .or_default()
                    .push(new.get_reg().unwrap().clone());
                stmt.defs = vec![new];
            }

            /* 普通语句 */
            for stmt in &mut blk.stmts {
                match &mut stmt.value {
                    RValue::Op { args, .. } => {
                        for a in args {
                            rename_expr(a, stack, counter);
                        }
                    }
                    RValue::Phi(_) | RValue::ImmI(_) | RValue::ImmF(_) => {}
                }
                if let Some(ma) = &mut stmt.mem_addr_args {
                    for a in ma {
                        rename_expr(a, stack, counter);
                    }
                }
                if let Some(p) = &mut stmt.pred {
                    rename_expr(p, stack, counter);
                }

                if !matches!(stmt.value, RValue::Phi(_)) {
                    let is_predicated = stmt.pred.is_some();
                    for def in &mut stmt.defs {
                        let cur = def.get_reg().unwrap().clone();
                        if is_immutable_reg(&cur) {
                            continue;
                        }
                        let k = base_reg(&cur);
                        // For predicated instructions, record the *previous*
                        // SSA version so the renderer can emit a conditional
                        // select instead of an unconditional assignment.
                        if is_predicated {
                            let old = top_or_new(&k, stack, counter).clone();
                            stmt.pred_old_defs.push(IRExpr::Reg(old));
                        }
                        let new = IRExpr::Reg(new_ssa(&k, counter));
                        stack
                            .entry(k.clone())
                            .or_default()
                            .push(new.get_reg().unwrap().clone());
                        *def = new;
                    }
                }
            }

            /* IRDst 条件 */
            for (cond, _) in &mut blk.irdst {
                if let Some(IRCond::Pred { reg, .. }) = cond {
                    *reg = top_or_new(&base_reg(reg), stack, counter).clone();
                }
            }
        } // blk borrow drop

        /* 2. 为后继块按 WhichPred 填充 Φ 参数 */
        for succ in cfg.neighbors_directed(n, Direction::Outgoing) {
            let succ_id = cfg[succ].id;

            // 获取 succ 的前驱列表（固定顺序）
            let preds: Vec<_> = cfg.neighbors_directed(succ, Direction::Incoming).collect();
            let idx_in_succ = preds
                .iter()
                .position(|&p| p == n)
                .expect("predecessor not found");

            let blk_succ = ir_blocks.get_mut(&succ_id).unwrap();
            for stmt in blk_succ
                .stmts
                .iter_mut()
                .filter(|s| matches!(s.value, RValue::Phi(_)))
            {
                let key = base_reg(stmt.defs.first().unwrap().get_reg().unwrap());
                let src = top_or_new(&key, stack, counter).clone();
                if let RValue::Phi(ref mut vec) = stmt.value {
                    vec[idx_in_succ] = IRExpr::Reg(src);
                }
            }
        }

        /* 3. 递归 */
        if let Some(chs) = children.get(&n) {
            for &c in chs {
                rename(c, cfg, children, ir_blocks, stack, counter);
            }
        }

        /* 4. pop */
        {
            let blk = ir_blocks.get(&bid).unwrap();
            for stmt in &blk.stmts {
                for def in &stmt.defs {
                    let d = def.get_reg().unwrap();
                    if is_immutable_reg(d) {
                        continue;
                    }
                    let k = base_reg(d);
                    if let Some(v) = stack.get_mut(&k) {
                        v.pop();
                    }
                }
            }
        }
    }

    rename(
        entry,
        cfg,
        &children,
        &mut ir_blocks,
        &mut stack,
        &mut counter,
    );

    /* ---------- collect ---------- */
    let mut blocks: Vec<_> = ir_blocks.into_iter().map(|(_, b)| b).collect();
    blocks.sort_by_key(|b| b.id);
    FunctionIR { blocks }
}

/* ==================== DF helper ==================== */
fn compute_df(
    cfg: &ControlFlowGraph,
    idom: &BTreeMap<NodeIndex, NodeIndex>,
    root: NodeIndex,
) -> HashMap<NodeIndex, BTreeSet<NodeIndex>> {
    let mut local = HashMap::<NodeIndex, BTreeSet<NodeIndex>>::new();
    for n in cfg.node_indices() {
        for succ in cfg.neighbors_directed(n, Direction::Outgoing) {
            if idom.get(&succ).copied() != Some(n) {
                local.entry(n).or_default().insert(succ);
            }
        }
    }
    let mut children: BTreeMap<NodeIndex, Vec<NodeIndex>> = BTreeMap::new();
    for (&b, &p) in idom {
        children.entry(p).or_default().push(b);
    }

    fn up(
        n: NodeIndex,
        child: &BTreeMap<NodeIndex, Vec<NodeIndex>>,
        df: &mut HashMap<NodeIndex, BTreeSet<NodeIndex>>,
        idom: &BTreeMap<NodeIndex, NodeIndex>,
    ) {
        if let Some(ch) = child.get(&n) {
            for &c in ch {
                up(c, child, df, idom);
            }
        }
        for &c in child.get(&n).unwrap_or(&Vec::new()) {
            let propagate: Vec<NodeIndex> = df
                .get(&c)
                .map(|s| {
                    s.iter()
                        .copied()
                        .filter(|w| idom.get(w).copied() != Some(n))
                        .collect()
                })
                .unwrap_or_default();
            for w in propagate {
                df.entry(n).or_default().insert(w);
            }
        }
    }
    let mut df = local;
    up(root, &children, &mut df, idom);
    df
}

/* =======================================================================
   Section 4 – DOT Debugging
======================================================================= */
impl FunctionIR {
    pub fn to_dot(&self, cfg: &ControlFlowGraph, ctx: &dyn DisplayCtx) -> String {
        use std::fmt::Write;

        /// 将 IRCond 显示成 “(P0.3)” / “(!P1.7)” / “(uncond)”
        fn cond_str(c: &Option<IRCond>, ctx: &dyn DisplayCtx) -> String {
            match c {
                Some(IRCond::True) | None => "(uncond)".into(),
                Some(IRCond::Pred { reg, sense }) => {
                    let s = ctx.reg(reg);
                    if *sense {
                        format!("({})", s)
                    } else {
                        format!("(!{})", s)
                    }
                }
            }
        }

        let mut dot = String::from("digraph SSA {\n  node[shape=box];\n");

        /* ----------- 节点 ----------- */
        for b in &self.blocks {
            let mut label = format!("BB{} | Start: 0x{:x}\\l", b.id, b.start_addr);

            /* 语句 */
            for stmt in &b.stmts {
                let line = match &stmt.value {
                    RValue::Op { opcode, args } => {
                        let dst = if stmt.defs.is_empty() {
                            "_".to_string()
                        } else if stmt.defs.len() == 1 {
                            ctx.expr(&stmt.defs[0])
                        } else {
                            let list = stmt
                                .defs
                                .iter()
                                .map(|d| ctx.expr(d))
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("({})", list)
                        };
                        let a = args
                            .iter()
                            .map(|e| ctx.expr(e))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{} = {}({})", dst, opcode, a)
                    }
                    RValue::Phi(vars) => {
                        let dst = ctx.expr(stmt.defs.first().unwrap());
                        let list = vars
                            .iter()
                            .map(|v| ctx.expr(v))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{} = phi({})", dst, list)
                    }
                    RValue::ImmI(i) => {
                        format!("{} = {}", ctx.expr(stmt.defs.first().unwrap()), i)
                    }
                    RValue::ImmF(f) => {
                        format!("{} = {}", ctx.expr(stmt.defs.first().unwrap()), f)
                    }
                };
                label.push_str(&line);
                label.push_str("\\l");
            }

            /* IRDst 列表 */
            label.push_str("\\l");
            for (cond, addr) in &b.irdst {
                let cstr = cond_str(cond, ctx);
                label.push_str(&format!("IRDst: {} -> 0x{:x}\\l", cstr, addr));
            }

            // 写入节点
            writeln!(
                dot,
                "  {} [label=\"{}\"];",
                b.id,
                label.replace('\"', "\\\"")
            )
            .unwrap();
        }

        /* ----------- 边 ----------- */
        for e in cfg.edge_references() {
            let (sid, did) = (cfg[e.source()].id, cfg[e.target()].id);
            writeln!(dot, "  {} -> {};", sid, did).unwrap();
        }
        dot.push_str("}");
        dot
    }
}
