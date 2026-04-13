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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decl {
    pub name: String,
    pub ty: String,
    pub storage: StorageClass,
    pub live_in: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Raw(String),
    Imm(String),
    Reg(String),
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
    Load {
        ty: Option<String>,
        addr: Box<Expr>,
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
            Expr::Unary { op, arg } => {
                let prec = 14;
                let inner = format!("{}{}", op, arg.render_with_prec(prec));
                maybe_wrap(inner, prec, parent_prec)
            }
            Expr::Binary { op, lhs, rhs } => {
                let prec = binary_prec(op);
                let inner = format!(
                    "{} {} {}",
                    lhs.render_with_prec(prec),
                    op,
                    rhs.render_with_prec(prec + 1)
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
            Expr::Load { ty, addr } => {
                let addr = render_addr_operand(addr);
                match ty {
                    Some(ty) => format!("*(({}*){})", ty, addr),
                    None => format!("*{}", addr),
                }
            }
            Expr::Addr64 { lo, hi } => {
                format!("addr64({}, {})", lo.render(), hi.render())
            }
            Expr::Cast { ty, expr } => {
                format!("({})({})", ty, expr.render())
            }
            Expr::Index { base, index } => {
                format!("{}[{}]", base.render(), index.render())
            }
        }
    }
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
    Deref { ty: Option<String>, addr: Box<Expr> },
    Indexed { base: Box<Expr>, index: Box<Expr> },
}

impl LValue {
    pub fn render(&self) -> String {
        match self {
            LValue::Raw(s) | LValue::Var(s) => s.clone(),
            LValue::Deref { ty, addr } => {
                let addr = render_addr_operand(addr);
                match ty {
                    Some(ty) => format!("*(({}*){})", ty, addr),
                    None => format!("*{}", addr),
                }
            }
            LValue::Indexed { base, index } => {
                format!("{}[{}]", base.render(), index.render())
            }
        }
    }

    pub fn is_sink_literal(&self) -> bool {
        matches!(self.render().as_str(), "0" | "true")
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

fn render_addr_operand(addr: &Expr) -> String {
    match addr {
        Expr::Binary { .. } | Expr::Ternary { .. } => format!("({})", addr.render()),
        _ => addr.render(),
    }
}

fn render_ternary_arm(expr: &Expr) -> String {
    match expr {
        Expr::Imm(_) | Expr::Reg(_) | Expr::ConstMemSymbol(_) | Expr::Builtin(_) => expr.render(),
        _ => format!("({})", expr.render()),
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
    format!("{} {}", decl.ty, decl.name)
}

fn render_decl_line(decl: &Decl) -> String {
    let storage = match decl.storage {
        StorageClass::Param | StorageClass::Local => "",
        StorageClass::Shared => "__shared__ ",
    };
    let live_in = if decl.live_in { " // live-in" } else { "" };
    format!("{}{} {};{}", storage, decl.ty, decl.name, live_in)
        .replace(&format!(";{}", live_in), ";")
        + live_in
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
        assert_eq!(lvalue.render(), "*((uint8_t*)(addr64(R4.0, R5.0) + 1))");
    }
}
