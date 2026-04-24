//! Typed AST used by the new backend boundary.
//! The current pipeline still structurizes through `StructuredStatement`,
//! but semantic lifting and rendering can target these canonical AST nodes.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopKind {
    While,
    DoWhile,
    Endless,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageClass {
    Param,
    Local,
    Shared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerLane {
    Lo32,
    Hi32,
}

impl PointerLane {
    pub(crate) fn render_suffix(self) -> &'static str {
        match self {
            PointerLane::Lo32 => "lo32",
            PointerLane::Hi32 => "hi32",
        }
    }

    pub(crate) fn parse_named(text: &str) -> Option<(String, Self)> {
        parse_pointer_lane_name(text)
    }
}

fn parse_pointer_lane_name(text: &str) -> Option<(String, PointerLane)> {
    pointer_lane_base(text, ".lo32", "_lo32")
        .map(|base| (base, PointerLane::Lo32))
        .or_else(|| pointer_lane_base(text, ".hi32", "_hi32").map(|base| (base, PointerLane::Hi32)))
}

fn pointer_lane_base(text: &str, dot_suffix: &str, underscore_suffix: &str) -> Option<String> {
    if let Some(base) = text.strip_suffix(dot_suffix) {
        return Some(base.to_string());
    }
    if let Some(base) = text.strip_suffix(underscore_suffix) {
        return Some(base.to_string());
    }
    None
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicOp {
    CarryU32Add3,
    LeaHiX,
    LeaHiXSx32,
    PairHi,
    Min,
    Max,
    Clamp,
}

impl IntrinsicOp {
    pub(crate) fn render_name(&self) -> &'static str {
        match self {
            IntrinsicOp::CarryU32Add3 => "carry_u32_add3",
            IntrinsicOp::LeaHiX => "lea_hi_x",
            IntrinsicOp::LeaHiXSx32 => "lea_hi_x_sx32",
            IntrinsicOp::PairHi => "pair_hi",
            IntrinsicOp::Min => "min",
            IntrinsicOp::Max => "max",
            IntrinsicOp::Clamp => "clamp",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decl {
    pub name: String,
    pub ty: String,
    pub array_len: Option<usize>,
    pub dynamic_extent: bool,
    pub storage: StorageClass,
    pub live_in: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Raw(String),
    Imm(String),
    Reg(String),
    PtrLane {
        base: String,
        lane: PointerLane,
    },
    LaneExtract {
        value: Box<Expr>,
        lane: PointerLane,
    },
    Unary {
        op: String,
        arg: Box<Expr>,
    },
    Binary {
        op: String,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    CallLike {
        func: String,
        args: Vec<Expr>,
    },
    Intrinsic {
        op: IntrinsicOp,
        args: Vec<Expr>,
    },
    Load {
        ty: Option<String>,
        addr: Box<Expr>,
    },
    WidePtr {
        base: Box<Expr>,
        offset: Box<Expr>,
    },
    ConstMemSymbol(String),
    Builtin(String),
    Addr64 {
        lo: Box<Expr>,
        hi: Box<Expr>,
    },
    Cast {
        ty: String,
        expr: Box<Expr>,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
}

impl Expr {
    pub fn render(&self) -> String {
        self.render_with_prec(0)
    }

    fn render_with_prec(&self, parent_prec: u8) -> String {
        match self {
            Expr::Raw(s)
            | Expr::Imm(s)
            | Expr::Reg(s)
            | Expr::ConstMemSymbol(s)
            | Expr::Builtin(s) => s.clone(),
            Expr::PtrLane { base, lane } => {
                render_lane_extract_expr(&Expr::Raw(base.clone()), *lane)
            }
            Expr::LaneExtract { value, lane } => render_lane_extract_expr(value, *lane),
            Expr::Unary { op, arg } => {
                let prec = 14;
                let inner = format!("{}{}", op, arg.render_with_prec(prec));
                maybe_wrap(inner, prec, parent_prec)
            }
            Expr::Binary { op, lhs, rhs } => {
                let prec = binary_prec(op);
                let (lhs_render, rhs_render) = if is_comparison_op(op) {
                    (
                        render_comparison_operand(lhs, rhs, prec),
                        render_comparison_operand(rhs, lhs, prec + 1),
                    )
                } else {
                    (lhs.render_with_prec(prec), rhs.render_with_prec(prec + 1))
                };
                let inner = format!(
                    "{} {} {}",
                    lhs_render,
                    op,
                    rhs_render
                );
                maybe_wrap(inner, prec, parent_prec)
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let prec = 1;
                let inner = format!(
                    "{} ? {} : {}",
                    cond.render_with_prec(prec + 1),
                    render_ternary_arm(then_expr),
                    else_expr.render_with_prec(prec)
                );
                maybe_wrap(inner, prec, parent_prec)
            }
            Expr::CallLike { func, args } => {
                let rendered = args.iter().map(Expr::render).collect::<Vec<_>>().join(", ");
                format!("{}({})", func, rendered)
            }
            Expr::Intrinsic { op, args } => render_intrinsic_expr(op, args),
            Expr::Load { ty, addr } => {
                match ty {
                    Some(ty) => render_typed_deref_expr(ty, addr),
                    None => format!("*{}", render_addr_operand(addr)),
                }
            }
            Expr::WidePtr { base, offset } => render_wide_ptr_expr(base, offset),
            Expr::Addr64 { lo, hi } => render_addr64_expr(lo, hi),
            Expr::Cast { ty, expr } => {
                format!("({})({})", ty, expr.render())
            }
            Expr::Index { base, index } => {
                format!("{}[{}]", render_index_base_expr(base), index.render())
            }
        }
    }
}

fn render_comparison_operand(expr: &Expr, other: &Expr, parent_prec: u8) -> String {
    if let Some(cast_ty) = explicit_cast_type(other) {
        let text = match expr {
            Expr::Imm(text) | Expr::Raw(text) if text.parse::<i64>().is_ok() => Some(text.as_str()),
            _ => None,
        };
        if let Some(text) = text.filter(|_| is_integer_cast_type(cast_ty)) {
            return format!("({})({})", cast_ty, text);
        }
    }
    expr.render_with_prec(parent_prec)
}

fn explicit_cast_type(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Cast { ty, .. } => Some(ty.as_str()),
        Expr::Raw(text) | Expr::Reg(text) | Expr::Imm(text) => raw_cast_type(text),
        _ => None,
    }
}

fn is_integer_cast_type(ty: &str) -> bool {
    matches!(
        ty,
        "int8_t"
            | "uint8_t"
            | "int16_t"
            | "uint16_t"
            | "int32_t"
            | "uint32_t"
            | "int64_t"
            | "uint64_t"
            | "uintptr_t"
    )
}

fn is_comparison_op(op: &str) -> bool {
    matches!(op, "<" | "<=" | ">" | ">=" | "==" | "!=")
}

fn raw_cast_type(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if !trimmed.starts_with('(') {
        return None;
    }
    let bytes = trimmed.as_bytes();
    let mut depth = 0usize;
    for (idx, byte) in bytes.iter().enumerate() {
        match *byte as char {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let ty = &trimmed[1..idx];
                    let rest = trimmed[idx + 1..].trim_start();
                    if !ty.is_empty() && rest.starts_with('(') {
                        return Some(ty.trim());
                    }
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LValue {
    Raw(String),
    Var(String),
    PtrLane { base: String, lane: PointerLane },
    Deref { ty: Option<String>, addr: Box<Expr> },
    Indexed { base: Box<Expr>, index: Box<Expr> },
}

impl LValue {
    pub fn render(&self) -> String {
        match self {
            LValue::Raw(s) | LValue::Var(s) => s.clone(),
            LValue::PtrLane { base, lane } => format!("{}.{}", base, lane.render_suffix()),
            LValue::Deref { ty, addr } => {
                match ty {
                    Some(ty) => render_typed_deref_expr(ty, addr),
                    None => format!("*{}", render_addr_operand(addr)),
                }
            }
            LValue::Indexed { base, index } => {
                format!("{}[{}]", render_index_base_expr(base), index.render())
            }
        }
    }

    pub fn is_sink_literal(&self) -> bool {
        matches!(self.render().as_str(), "0" | "true" | "_")
    }
}

impl fmt::Display for LValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Block(Vec<Stmt>),
    Sequence(Vec<Stmt>),
    Label {
        name: String,
        body: Box<Stmt>,
    },
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    Loop {
        kind: LoopKind,
        condition: Option<Expr>,
        body: Box<Stmt>,
    },
    Switch {
        discriminant: Option<Expr>,
        cases: Vec<(usize, Stmt)>,
        default: Option<Box<Stmt>>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
    Assign {
        dst: LValue,
        src: Expr,
    },
    ExprStmt(Expr),
    Goto(String),
    Empty,
}

impl Stmt {
    pub fn render_with_indent(&self, indent: usize) -> String {
        let mut rendered = String::new();
        self.render_into(&mut rendered, indent);
        rendered
    }

    fn render_into(&self, out: &mut String, indent: usize) {
        let pad = "  ".repeat(indent);
        match self {
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    stmt.render_into(out, indent + 1);
                }
            }
            Stmt::Sequence(stmts) => {
                for stmt in stmts {
                    stmt.render_into(out, indent);
                }
            }
            Stmt::Label { name, body } => match body.as_ref() {
                Stmt::Empty => out.push_str(&format!("{}{}: ;\n", pad, name)),
                _ => {
                    out.push_str(&format!("{}{}:\n", pad, name));
                    body.render_into(out, indent + 1);
                }
            },
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                if else_branch.is_none() {
                    if let Some(inline) = then_branch.render_inline() {
                        out.push_str(&format!("{}if ({}) {};\n", pad, condition.render(), inline));
                        return;
                    }
                }
                out.push_str(&format!("{}if ({}) {{\n", pad, condition.render()));
                then_branch.render_into(out, indent + 1);
                out.push_str(&format!("{}}}", pad));
                if let Some(else_branch) = else_branch {
                    out.push_str(" else {\n");
                    else_branch.render_into(out, indent + 1);
                    out.push_str(&format!("{}}}", pad));
                }
                out.push('\n');
            }
            Stmt::Loop {
                kind,
                condition,
                body,
            } => match kind {
                LoopKind::While => {
                    let cond = condition
                        .as_ref()
                        .map(Expr::render)
                        .unwrap_or_else(|| "true".to_string());
                    out.push_str(&format!("{}while ({}) {{\n", pad, cond));
                    body.render_into(out, indent + 1);
                    out.push_str(&format!("{}}}\n", pad));
                }
                LoopKind::DoWhile => {
                    out.push_str(&format!("{}do {{\n", pad));
                    body.render_into(out, indent + 1);
                    let cond = condition
                        .as_ref()
                        .map(Expr::render)
                        .unwrap_or_else(|| "true".to_string());
                    out.push_str(&format!("{}}} while({});\n", pad, cond));
                }
                LoopKind::Endless => {
                    out.push_str(&format!("{}while (true) {{\n", pad));
                    body.render_into(out, indent + 1);
                    out.push_str(&format!("{}}}\n", pad));
                }
            },
            Stmt::Switch {
                discriminant,
                cases,
                default,
            } => {
                let discr = discriminant
                    .as_ref()
                    .map(Expr::render)
                    .unwrap_or_else(|| "/* unknown */".to_string());
                out.push_str(&format!("{}switch ({}) {{\n", pad, discr));
                for (label, body) in cases {
                    out.push_str(&format!("{}  case {}:\n", pad, label));
                    body.render_into(out, indent + 2);
                    out.push_str(&format!("{}    break;\n", pad));
                }
                if let Some(default) = default {
                    out.push_str(&format!("{}  default:\n", pad));
                    default.render_into(out, indent + 2);
                    out.push_str(&format!("{}    break;\n", pad));
                }
                out.push_str(&format!("{}}}\n", pad));
            }
            Stmt::Break => out.push_str(&format!("{}break;\n", pad)),
            Stmt::Continue => out.push_str(&format!("{}continue;\n", pad)),
            Stmt::Return(expr) => match expr {
                Some(expr) => out.push_str(&format!("{}return {};\n", pad, expr.render())),
                None => out.push_str(&format!("{}return;\n", pad)),
            },
            Stmt::Assign { dst, src } => {
                out.push_str(&format!("{}{} = {};\n", pad, dst.render(), src.render()));
            }
            Stmt::ExprStmt(expr) => out.push_str(&format!("{}{};\n", pad, expr.render())),
            Stmt::Goto(label) => out.push_str(&format!("{}goto {};\n", pad, label)),
            Stmt::Empty => {}
        }
    }

    fn render_inline(&self) -> Option<String> {
        match self {
            Stmt::Break => Some("break".to_string()),
            Stmt::Continue => Some("continue".to_string()),
            Stmt::Return(expr) => Some(match expr {
                Some(expr) => format!("return {}", expr.render()),
                None => "return".to_string(),
            }),
            Stmt::Assign { dst, src } => Some(format!("{} = {}", dst.render(), src.render())),
            Stmt::ExprStmt(expr) => Some(expr.render()),
            Stmt::Goto(label) => Some(format!("goto {}", label)),
            Stmt::Label { .. } => None,
            Stmt::Block(stmts) | Stmt::Sequence(stmts) if stmts.len() == 1 => {
                stmts.first().and_then(Stmt::render_inline)
            }
            _ => None,
        }
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render_with_indent(0))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuredFunction {
    pub params: Vec<Decl>,
    pub locals: Vec<Decl>,
    pub body: Stmt,
}

impl StructuredFunction {
    pub fn render(&self, name: &str) -> String {
        let params = if self.params.is_empty() {
            "void".to_string()
        } else {
            self.params
                .iter()
                .map(render_decl_signature)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut out = format!("void {}({}) {{\n", name, params);
        for decl in &self.locals {
            out.push_str("  ");
            out.push_str(&render_decl_line(decl));
            out.push('\n');
        }
        if !self.locals.is_empty() {
            out.push('\n');
        }
        self.body.render_into(&mut out, 1);
        out.push_str("}\n");
        out
    }
}

fn maybe_wrap(inner: String, my_prec: u8, parent_prec: u8) -> String {
    if my_prec < parent_prec {
        format!("({})", inner)
    } else {
        inner
    }
}

fn render_addr64_expr(lo: &Expr, hi: &Expr) -> String {
    format!(
        "((uintptr_t)(((uint64_t)({}) << 32) | (uint32_t)({})))",
        hi.render(),
        lo.render()
    )
}

fn render_lane_extract_expr(value: &Expr, lane: PointerLane) -> String {
    if let Some(symbolic) = render_symbolic_pointer_lane(value, lane) {
        return symbolic;
    }
    let value = render_lane_extract_operand(value);
    match lane {
        PointerLane::Lo32 => format!("((uint32_t)({}))", value),
        PointerLane::Hi32 => format!("((uint32_t)(((uint64_t)({})) >> 32))", value),
    }
}

fn render_symbolic_pointer_lane(value: &Expr, lane: PointerLane) -> Option<String> {
    match value {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text)
            if text.ends_with("_ptr") =>
        {
            Some(match lane {
                PointerLane::Lo32 => format!("((uint32_t)(uintptr_t){})", text),
                PointerLane::Hi32 => {
                    format!("((uint32_t)(((uint64_t)(uintptr_t){}) >> 32))", text)
                }
            })
        }
        _ => None,
    }
}

fn render_lane_extract_operand(value: &Expr) -> String {
    match value {
        Expr::Raw(text)
        | Expr::Imm(text)
        | Expr::Reg(text)
        | Expr::ConstMemSymbol(text)
        | Expr::Builtin(text) => {
            if text.ends_with("_ptr") {
                format!("(uintptr_t){}", text)
            } else {
                text.clone()
            }
        }
        Expr::WidePtr { base, offset } if symbolic_pointer_base_name(base).is_some() => {
            format!(
                "(((uint8_t*){}) + (int64_t){})",
                render_pointer_base_expr(base),
                render_pointer_index_expr(offset)
            )
        }
        Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::Unary { .. }
        | Expr::Binary { .. }
        | Expr::Ternary { .. }
        | Expr::CallLike { .. }
        | Expr::Intrinsic { .. }
        | Expr::Load { .. }
        | Expr::WidePtr { .. }
        | Expr::Addr64 { .. }
        | Expr::Cast { .. }
        | Expr::Index { .. } => {
            let rendered = value.render();
            if rendered.starts_with('(') && rendered.ends_with(')') {
                rendered
            } else {
                format!("({})", rendered)
            }
        }
    }
}

fn render_wide_ptr_expr(base: &Expr, offset: &Expr) -> String {
    if matches!(offset, Expr::Imm(text) if text == "0") {
        return base.render();
    }
    if let Some(base_name) = symbolic_pointer_base_name(base)
        .filter(|name| !expr_mentions_pointer_base(offset, name))
    {
        let _ = base_name;
        return format!("{} + (int64_t){}", base.render(), render_byte_offset_expr(offset));
    }
    let base = match base {
        Expr::Binary { .. } | Expr::Ternary { .. } => format!("({})", base.render()),
        _ => base.render(),
    };
    format!("((uint8_t*){}) + (int64_t){}", base, render_byte_offset_expr(offset))
}

fn render_typed_deref_expr(ty: &str, addr: &Expr) -> String {
    if let Some(ptr_expr) = render_typed_pointer_expr(ty, addr) {
        return format!("*({})", ptr_expr);
    }
    let addr = render_addr_operand(addr);
    format!("*(({}*){})", ty, addr)
}

fn render_typed_pointer_expr(ty: &str, addr: &Expr) -> Option<String> {
    let elem_size = scalar_type_size_bytes(ty)?;
    let (base, offset) = match addr {
        Expr::WidePtr { base, offset } => (base.as_ref(), offset.as_ref()),
        other => (other, &Expr::Imm("0".to_string())),
    };
    let symbolic_base = expr_is_symbolic_pointer_base(base);
    if symbolic_base {
        let base_render = render_pointer_base_expr(base);
        if let Some(index) =
            divide_offset_expr(offset, elem_size).filter(looks_like_element_index_expr)
        {
            let index_render = render_pointer_index_expr(&index);
            return Some(if expr_is_zero(&index) {
                base_render
            } else {
                format!("{} + {}", base_render, index_render)
            });
        }
        if looks_like_element_index_expr(offset) {
            let index_render = render_pointer_index_expr(offset);
            return Some(if expr_is_zero(offset) {
                base_render
            } else {
                format!("{} + {}", base_render, index_render)
            });
        }

        let byte_offset = render_byte_offset_expr(offset);
        return Some(format!(
            "({}*)(((uint8_t*){}) + (int64_t){})",
            ty, base_render, byte_offset
        ));
    }
    let index = divide_offset_expr(offset, elem_size)?;
    let base_render = render_pointer_base_expr(base);
    let index_render = render_pointer_index_expr(&index);
    if expr_is_zero(&index) {
        Some(format!("({}*){}", ty, base_render))
    } else {
        Some(format!("(({}*){}) + {}", ty, base_render, index_render))
    }
}

fn expr_is_symbolic_pointer_base(expr: &Expr) -> bool {
    symbolic_pointer_base_name(expr).is_some()
}

fn symbolic_pointer_base_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text)
            if text.ends_with("_ptr") =>
        {
            Some(text.as_str())
        }
        _ => None,
    }
}

fn expr_mentions_pointer_base(expr: &Expr, base_name: &str) -> bool {
    match expr {
        Expr::Raw(text) | Expr::Reg(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            text == base_name
        }
        Expr::PtrLane { base, .. } => base == base_name,
        Expr::LaneExtract { value, .. } | Expr::Cast { expr: value, .. } => {
            expr_mentions_pointer_base(value, base_name)
        }
        Expr::Unary { arg, .. } => expr_mentions_pointer_base(arg, base_name),
        Expr::Binary { lhs, rhs, .. } => {
            expr_mentions_pointer_base(lhs, base_name)
                || expr_mentions_pointer_base(rhs, base_name)
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_mentions_pointer_base(cond, base_name)
                || expr_mentions_pointer_base(then_expr, base_name)
                || expr_mentions_pointer_base(else_expr, base_name)
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            args.iter().any(|arg| expr_mentions_pointer_base(arg, base_name))
        }
        Expr::Load { addr, .. } => expr_mentions_pointer_base(addr, base_name),
        Expr::WidePtr { base, offset } => {
            expr_mentions_pointer_base(base, base_name)
                || expr_mentions_pointer_base(offset, base_name)
        }
        Expr::Addr64 { lo, hi } => {
            expr_mentions_pointer_base(lo, base_name)
                || expr_mentions_pointer_base(hi, base_name)
        }
        Expr::Index { base, index } => {
            expr_mentions_pointer_base(base, base_name)
                || expr_mentions_pointer_base(index, base_name)
        }
        Expr::Imm(_) => false,
    }
}

fn render_pointer_base_expr(base: &Expr) -> String {
    match base {
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::PtrLane { .. } => base.render(),
        _ => format!("({})", base.render()),
    }
}

fn render_pointer_index_expr(index: &Expr) -> String {
    match index {
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::PtrLane { .. } => index.render(),
        _ => format!("({})", index.render()),
    }
}

fn render_byte_offset_expr(offset: &Expr) -> String {
    match strip_index_widen_casts(offset) {
        Expr::Binary { op, lhs, rhs } if op == "*" => {
            if expr_integer_value(rhs).is_some() {
                return format!(
                    "{} * {}",
                    render_pointer_index_expr(lhs),
                    render_pointer_index_expr(rhs)
                );
            }
            if expr_integer_value(lhs).is_some() {
                return format!(
                    "{} * {}",
                    render_pointer_index_expr(rhs),
                    render_pointer_index_expr(lhs)
                );
            }
            render_pointer_index_expr(offset)
        }
        _ => render_pointer_index_expr(offset),
    }
}

fn scalar_type_size_bytes(ty: &str) -> Option<i64> {
    match ty.trim() {
        "uint8_t" | "int8_t" | "bool" => Some(1),
        "uint16_t" | "int16_t" | "__half" => Some(2),
        "uint32_t" | "int32_t" | "float" => Some(4),
        "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t" | "double" => Some(8),
        _ => None,
    }
}

fn divide_offset_expr(expr: &Expr, divisor: i64) -> Option<Expr> {
    if divisor == 1 {
        return Some(strip_index_widen_casts(expr).clone());
    }
    match strip_index_widen_casts(expr) {
        Expr::Imm(text) | Expr::Raw(text) => {
            let value = text.parse::<i64>().ok()?;
            (value % divisor == 0).then(|| Expr::Imm((value / divisor).to_string()))
        }
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => {
            let lhs = divide_offset_expr(lhs, divisor)?;
            let rhs = divide_offset_expr(rhs, divisor)?;
            Some(Expr::Binary {
                op: op.clone(),
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        Expr::Binary { op, lhs, rhs } if op == "*" => {
            if let Some(factor) = expr_integer_value(rhs) {
                return divide_scaled_product(lhs, factor, divisor);
            }
            if let Some(factor) = expr_integer_value(lhs) {
                return divide_scaled_product(rhs, factor, divisor);
            }
            None
        }
        Expr::Cast { expr, .. } => divide_offset_expr(expr, divisor),
        other if expr_is_zero(&other) => Some(Expr::Imm("0".to_string())),
        _ => None,
    }
}

fn divide_scaled_product(expr: &Expr, factor: i64, divisor: i64) -> Option<Expr> {
    (factor % divisor == 0).then(|| {
        let reduced = factor / divisor;
        let base = strip_index_widen_casts(expr).clone();
        if reduced == 1 {
            base
        } else {
            Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(base),
                rhs: Box::new(Expr::Imm(reduced.to_string())),
            }
        }
    })
}

fn looks_like_element_index_expr(expr: &Expr) -> bool {
    match strip_index_widen_casts(expr) {
        Expr::Imm(_)
        | Expr::Raw(_)
        | Expr::Reg(_)
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::PtrLane { .. } => true,
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" || op == "*" => {
            looks_like_element_index_expr(lhs) && looks_like_element_index_expr(rhs)
        }
        Expr::Unary { arg, .. } => looks_like_element_index_expr(arg),
        Expr::Cast { expr, .. } => looks_like_element_index_expr(expr),
        _ => false,
    }
}

fn expr_integer_value(expr: &Expr) -> Option<i64> {
    match strip_index_widen_casts(expr) {
        Expr::Imm(text) | Expr::Raw(text) => text.parse::<i64>().ok(),
        _ => None,
    }
}

fn strip_index_widen_casts<'a>(expr: &'a Expr) -> &'a Expr {
    let mut current = expr;
    while let Expr::Cast { ty, expr } = current {
        if matches!(
            ty.as_str(),
            "int64_t" | "uint64_t" | "intptr_t" | "uintptr_t" | "int32_t" | "uint32_t"
        ) {
            current = expr;
        } else {
            break;
        }
    }
    current
}

fn expr_is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Imm(text) | Expr::Raw(text) if text.trim() == "0")
}

fn render_addr_operand(addr: &Expr) -> String {
    match addr {
        Expr::Binary { .. }
        | Expr::Ternary { .. }
        | Expr::LaneExtract { .. }
        | Expr::WidePtr { .. }
        | Expr::Addr64 { .. } => format!("({})", addr.render()),
        _ => addr.render(),
    }
}

fn render_index_base_expr(base: &Expr) -> String {
    match base {
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::CallLike { .. }
        | Expr::Intrinsic { .. }
        | Expr::Load { .. }
        | Expr::WidePtr { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_)
        | Expr::Addr64 { .. }
        | Expr::Index { .. } => base.render(),
        Expr::Unary { .. }
        | Expr::Binary { .. }
        | Expr::Ternary { .. }
        | Expr::Cast { .. } => format!("({})", base.render()),
    }
}

fn render_ternary_arm(expr: &Expr) -> String {
    match expr {
        Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::LaneExtract { .. }
        | Expr::WidePtr { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_) => expr.render(),
        _ => format!("({})", expr.render()),
    }
}

fn render_intrinsic_expr(op: &IntrinsicOp, args: &[Expr]) -> String {
    match op {
        IntrinsicOp::PairHi if args.len() == 1 => {
            format!("((uint32_t)(((uint64_t)({})) >> 32))", args[0].render())
        }
        _ => {
            let rendered = args.iter().map(Expr::render).collect::<Vec<_>>().join(", ");
            format!("{}({})", op.render_name(), rendered)
        }
    }
}

fn binary_prec(op: &str) -> u8 {
    match op {
        "*" | "/" | "%" => 13,
        "+" | "-" => 12,
        "<<" | ">>" => 11,
        "<" | "<=" | ">" | ">=" => 10,
        "==" | "!=" => 9,
        "&" => 8,
        "^" => 7,
        "|" => 6,
        "&&" => 5,
        "||" => 4,
        _ => 9,
    }
}

fn render_decl_signature(decl: &Decl) -> String {
    format!("{} {}", render_decl_ty(decl), render_decl_name(decl))
}

fn render_decl_line(decl: &Decl) -> String {
    if decl.storage == StorageClass::Local && decl.dynamic_extent {
        return format!("/* dynamic local memory: {} */", decl.name);
    }
    let storage = match decl.storage {
        StorageClass::Param | StorageClass::Local => "",
        StorageClass::Shared if decl.dynamic_extent => "extern __shared__ ",
        StorageClass::Shared => "__shared__ ",
    };
    let mut comments = Vec::new();
    if decl.dynamic_extent {
        comments.push("dynamic extent");
    }
    if decl.live_in {
        comments.push("live-in");
    }
    let comment_suffix = if comments.is_empty() {
        String::new()
    } else {
        format!(" // {}", comments.join(", "))
    };
    format!(
        "{}{} {};{}",
        storage,
        render_decl_ty(decl),
        render_decl_name(decl),
        comment_suffix
    )
        .replace(&format!(";{}", comment_suffix), ";")
        + &comment_suffix
}

fn render_decl_name(decl: &Decl) -> String {
    if decl.dynamic_extent && decl.storage == StorageClass::Shared {
        return format!("{}[]", decl.name);
    }
    match decl.array_len {
        Some(len) => format!("{}[{}]", decl.name, len),
        None => decl.name.clone(),
    }
}

fn render_decl_ty(decl: &Decl) -> String {
    decl.ty.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_render_respects_precedence() {
        let expr = Expr::Binary {
            op: "*".into(),
            lhs: Box::new(Expr::Binary {
                op: "+".into(),
                lhs: Box::new(Expr::Reg("R1.0".into())),
                rhs: Box::new(Expr::Imm("1".into())),
            }),
            rhs: Box::new(Expr::Reg("R2.0".into())),
        };
        assert_eq!(expr.render(), "(R1.0 + 1) * R2.0");
    }

    #[test]
    fn ternary_else_branch_is_parenthesized_when_needed() {
        let expr = Expr::Binary {
            op: "+".into(),
            lhs: Box::new(Expr::Reg("R1.0".into())),
            rhs: Box::new(Expr::Ternary {
                cond: Box::new(Expr::Reg("P0.0".into())),
                then_expr: Box::new(Expr::Imm("1".into())),
                else_expr: Box::new(Expr::Imm("0".into())),
            }),
        };
        assert_eq!(expr.render(), "R1.0 + (P0.0 ? 1 : 0)");
    }

    #[test]
    fn typed_lvalue_render_matches_legacy_pointer_style() {
        let lvalue = LValue::Deref {
            ty: Some("uint8_t".into()),
            addr: Box::new(Expr::Binary {
                op: "+".into(),
                lhs: Box::new(Expr::Addr64 {
                    lo: Box::new(Expr::Reg("R4.0".into())),
                    hi: Box::new(Expr::Reg("R5.0".into())),
                }),
                rhs: Box::new(Expr::Imm("1".into())),
            }),
        };
        assert_eq!(
            lvalue.render(),
            "*((uint8_t*)(((uintptr_t)(((uint64_t)(R5.0) << 32) | (uint32_t)(R4.0))) + 1))"
        );
    }

    #[test]
    fn sink_literals_include_placeholder_assign_target() {
        assert!(LValue::Raw("_".into()).is_sink_literal());
    }

    #[test]
    fn dynamic_local_decls_render_as_comments() {
        let function = StructuredFunction {
            params: Vec::new(),
            locals: vec![Decl {
                name: "local_mem".into(),
                ty: "uint32_t".into(),
                array_len: None,
                dynamic_extent: true,
                storage: StorageClass::Local,
                live_in: false,
            }],
            body: Stmt::Empty,
        };
        assert!(function.render("kernel").contains("/* dynamic local memory: local_mem */"));
        assert!(!function.render("kernel").contains("local_mem[]"));
    }

    #[test]
    fn typed_pointer_index_is_parenthesized_when_needed() {
        let expr = Expr::Load {
            ty: Some("float".into()),
            addr: Box::new(Expr::WidePtr {
                base: Box::new(Expr::Raw("arg0_ptr".into())),
                offset: Box::new(Expr::Binary {
                    op: "*".into(),
                    lhs: Box::new(Expr::Binary {
                        op: "|".into(),
                        lhs: Box::new(Expr::Reg("lo".into())),
                        rhs: Box::new(Expr::Binary {
                            op: "<<".into(),
                            lhs: Box::new(Expr::Reg("hi".into())),
                            rhs: Box::new(Expr::Imm("32".into())),
                        }),
                    }),
                    rhs: Box::new(Expr::Imm("4".into())),
                }),
            }),
        };
        assert_eq!(
            expr.render(),
            "*((float*)(((uint8_t*)arg0_ptr) + (int64_t)(lo | hi << 32) * 4))"
        );
    }

    #[test]
    fn typed_pointer_index_divides_byte_offsets_for_symbolic_bases() {
        let expr = Expr::Load {
            ty: Some("float".into()),
            addr: Box::new(Expr::WidePtr {
                base: Box::new(Expr::Raw("arg2_ptr".into())),
                offset: Box::new(Expr::Binary {
                    op: "+".into(),
                    lhs: Box::new(Expr::Binary {
                        op: "*".into(),
                        lhs: Box::new(Expr::Reg("v4".into())),
                        rhs: Box::new(Expr::Imm("12".into())),
                    }),
                    rhs: Box::new(Expr::Imm("4".into())),
                }),
            }),
        };
        assert_eq!(expr.render(), "*(arg2_ptr + (v4 * 3 + 1))");
    }
}
