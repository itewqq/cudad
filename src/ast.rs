//! Abstract Syntax Tree (AST) for structured pseudocode
//! -----------------------------------------------------
//! This module defines a minimal AST used by the `structurizer` pass
//! and provides a pretty‑printer that emits C‑like code. The initial
//! scope covers `if/else`, three loop flavours, and basic control
//! statements. `switch` support can be added later.

use std::fmt;

/// A lightweight wrapper around a boolean condition. In a full
/// implementation this would reference an IR expression; for now
/// we keep a plain string that already contains printable code.
#[derive(Clone, Debug)]
pub struct Condition(pub String);

impl Condition {
    pub fn new(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Flavours of loop recognised by the structuriser.
#[derive(Clone, Copy, Debug)]
pub enum LoopKind {
    /// `while (cond)` – pre‑test loop.
    While,
    /// `do { .. } while (cond);` – post‑test loop.
    DoWhile,
    /// `while (true)` – infinite loop; exits use `break`/`return`.
    Endless,
}

impl fmt::Display for LoopKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoopKind::While   => write!(f, "while"),
            LoopKind::DoWhile => write!(f, "do"),
            LoopKind::Endless => write!(f, "while"),
        }
    }
}

/// Core AST node. It is deliberately small – transformations that
/// need more detail (e.g. short‑circuit conditions) can attach extra
/// fields later.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum AstNode {
    /// A leaf basic‑block (prints a label or the raw text for now).
    Basic(String),
    /// Sequential composition.
    Seq(Vec<AstNode>),
    /// `if` or `if‑else`.
    IfElse {
        cond: Condition,
        then_br: Box<AstNode>,
        else_br: Option<Box<AstNode>>, // None ⇒ simple `if`
    },
    /// Loop container.
    Loop {
        kind: LoopKind,
        cond: Option<Condition>,       // None ⇒ endless loop
        body: Box<AstNode>,
    },
    /// Control statements.
    Break,
    Continue,
    ReturnStmt,
}

impl AstNode {
    /// Pretty‑print recursively with 4‑space indentation.
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent_lvl: usize) -> fmt::Result {
        let ind = |lvl| "    ".repeat(lvl);

        match self {
            AstNode::Basic(text) => writeln!(f, "{}{}", ind(indent_lvl), text),

            AstNode::Seq(nodes) => {
                for n in nodes {
                    n.fmt_with_indent(f, indent_lvl)?;
                }
                Ok(())
            }

            AstNode::IfElse { cond, then_br, else_br } => {
                writeln!(f, "{}if ({}) {{", ind(indent_lvl), cond)?;
                then_br.fmt_with_indent(f, indent_lvl + 1)?;

                if let Some(else_ast) = else_br {
                    writeln!(f, "{}}} else {{", ind(indent_lvl))?;
                    else_ast.fmt_with_indent(f, indent_lvl + 1)?;
                }

                writeln!(f, "{}}}", ind(indent_lvl))
            }

            AstNode::Loop { kind, cond, body } => {
                match kind {
                    LoopKind::While => {
                        let cond_str = cond.as_ref().map_or_else(|| "true".to_owned(), |c| c.to_string());
                        writeln!(f, "{}while ({}) {{", ind(indent_lvl), cond_str)?;
                        body.fmt_with_indent(f, indent_lvl + 1)?;
                        writeln!(f, "{}}}", ind(indent_lvl))
                    }
                    LoopKind::DoWhile => {
                        writeln!(f, "{}do {{", ind(indent_lvl))?;
                        body.fmt_with_indent(f, indent_lvl + 1)?;
                        let cond_str = cond.as_ref().map_or_else(|| "true".to_owned(), |c| c.to_string());
                        writeln!(f, "{}}} while ({});", ind(indent_lvl), cond_str)
                    }
                    LoopKind::Endless => {
                        writeln!(f, "{}while (true) {{", ind(indent_lvl))?;
                        body.fmt_with_indent(f, indent_lvl + 1)?;
                        writeln!(f, "{}}}", ind(indent_lvl))
                    }
                }
            }

            AstNode::Break       => writeln!(f, "{}break;",    ind(indent_lvl)),
            AstNode::Continue    => writeln!(f, "{}continue;", ind(indent_lvl)),
            AstNode::ReturnStmt  => writeln!(f, "{}return;",   ind(indent_lvl)),
        }
    }
}

impl fmt::Display for AstNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_indent(f, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_simple_if() {
        let ast = AstNode::IfElse {
            cond: Condition("x < 0".into()),
            then_br: Box::new(AstNode::Basic("neg();".into())),
            else_br: Some(Box::new(AstNode::Basic("pos();".into()))),
        };

        let expected = concat!(
            "if (x < 0) {\n",
            "    neg();\n",
            "} else {\n",
            "    pos();\n",
            "}\n"
        );

        assert_eq!(ast.to_string(), expected);
    }
}
