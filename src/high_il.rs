use std::fmt::Write;

#[derive(Clone, Debug)]
pub enum HExpr {
    Reg(String),
    ImmI(i64),
    ImmF(f64),
    Call(String, Vec<HExpr>),
}

#[derive(Clone, Debug)]
pub enum HNode {
    Seq(Vec<HNode>),
    Assign { dst: String, rhs: HExpr },
    If { cond: HExpr, then_br: Box<HNode>, else_br: Option<Box<HNode>> },
    While { cond: HExpr, body: Box<HNode> },
    DoWhile { cond: HExpr, body: Box<HNode> },
    Goto(usize),
    Label(usize),
}

impl HNode {
    pub fn pretty(&self, mut indent: usize) -> String {
        let mut out = String::new();
        let tab = |i| "    ".repeat(i);

        match self {
            HNode::Seq(nodes) => {
                for n in nodes {
                    out.push_str(&n.pretty(indent));
                }
            }
            HNode::Assign { dst, rhs } => {
                writeln!(out, "{}{} = {};", tab(indent), dst, fmt_expr(rhs)).unwrap();
            }
            HNode::If { cond, then_br, else_br } => {
                writeln!(out, "{}if ({}) {{", tab(indent), fmt_expr(cond)).unwrap();
                indent += 1;
                out.push_str(&then_br.pretty(indent));
                indent -= 1;
                if let Some(e) = else_br {
                    writeln!(out, "{}}} else {{", tab(indent)).unwrap();
                    indent += 1;
                    out.push_str(&e.pretty(indent));
                    indent -= 1;
                }
                writeln!(out, "{}}}", tab(indent)).unwrap();
            }
            HNode::While { cond, body } => {
                writeln!(out, "{}while ({}) {{", tab(indent), fmt_expr(cond)).unwrap();
                indent += 1;
                out.push_str(&body.pretty(indent));
                indent -= 1;
                writeln!(out, "{}}}", tab(indent)).unwrap();
            }
            HNode::DoWhile { cond, body } => {
                writeln!(out, "{}do {{", tab(indent)).unwrap();
                indent += 1;
                out.push_str(&body.pretty(indent));
                indent -= 1;
                writeln!(out, "{}}} while ({});", tab(indent), fmt_expr(cond)).unwrap();
            }
            HNode::Goto(l) => {
                writeln!(out, "{}goto L{};", tab(indent), l).unwrap();
            }
            HNode::Label(l) => {
                writeln!(out, "L{}:", l).unwrap();
            }
        }
        out
    }
}

fn fmt_expr(e: &HExpr) -> String {
    match e {
        HExpr::Reg(s) => s.clone(),
        HExpr::ImmI(i) => i.to_string(),
        HExpr::ImmF(f) => f.to_string(),
        HExpr::Call(name, args) => {
            let a = args.iter().map(fmt_expr).collect::<Vec<_>>().join(", ");
            format!("{}({})", name, a)
        }
    }
}
