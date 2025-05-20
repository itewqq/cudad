//!  当前进度：极简 SSA (RPO 遍历 + 简单 Φ 插入，可处理循环)

use crate::cfg::ControlFlowGraph;
use crate::parser::Operand;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{visit::DfsPostOrder, Direction};
use std::collections::{HashMap, HashSet};
use crate::cfg::EdgeKind;
use std::env;
use petgraph::algo::dominators::simple_fast;
use std::cell::RefCell;

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").map(|v| v == "1").unwrap_or(false) {
            eprintln!($($arg)*);
        }
    };
}

/* ---------- 数据结构 ---------- */
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Var {
    pub name: String, // 原始寄存器名，如R1/P0
    pub id: usize,    // SSA版本号
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IRCond {
    True,
    Pred { name: String, id: usize, sense: bool }, // 例如P0.3==1
}

#[derive(Clone, Debug, PartialEq)]
pub enum IRArg {
    SSA(Var),
    Operand(Operand),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RValue {
    Op { opcode: String, args: Vec<IRArg> },
    Phi(Vec<Var>),
    ImmI(i64),
    ImmF(f64),
    Undef,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRStatement {
    pub dest: Option<Var>,
    pub value: RValue,
    /// SSA变量，表示该语句的谓词条件（如@P0）
    pub pred: Option<Var>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRBlock {
    pub id: usize,
    pub start_addr: u32, // 当前块起始地址
    pub irdst: Vec<(Option<IRCond>, u32)>, // (条件, 目标地址)列表
    pub stmts: Vec<IRStatement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionIR {
    pub blocks: Vec<IRBlock>,
}

/* ---------- PhysReg helper ---------- */
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PhysReg {
    class: String,
    idx: i32,
}

fn phys_reg_of(op: &Operand) -> Option<PhysReg> {
    match op {
        Operand::Register { class, idx } => {
            if class == "PT" || class == "RZ" {
                Some(PhysReg { class: class.clone(), idx: 0 })
            } else if class.starts_with("P") {
                // P0-P6
                if (0..=6).contains(idx) {
                    Some(PhysReg { class: "P".to_string(), idx: *idx })
                } else {
                    None
                }
            } else {
                Some(PhysReg { class: class.clone(), idx: *idx })
            }
        }
        Operand::Uniform { idx } => Some(PhysReg {
            class: "UR".into(),
            idx: *idx,
        }),
        _ => None,
    }
}

fn is_mem_store(opcode: &str) -> bool {
    opcode.starts_with("STS") || opcode.starts_with("STG") || opcode.starts_with("ST")
}

fn is_mem_load(opcode: &str) -> bool {
    opcode.starts_with("LDG") || opcode.starts_with("LD")
}

/* ---------- SSA 主流程 ---------- */
pub fn build_ssa(cfg: &ControlFlowGraph) -> FunctionIR {
    use petgraph::graph::NodeIndex;
    use petgraph::Direction;
    use petgraph::algo::dominators::simple_fast;
    use std::collections::{HashMap, HashSet};

    let entry = NodeIndex::new(0);
    let doms = simple_fast(cfg, entry);
    
    // 1. 收集所有变量的定义点（A(V)）
    let mut defsites: HashMap<PhysReg, HashSet<NodeIndex>> = HashMap::new();
    let mut ir_blocks = HashMap::<usize, IRBlock>::new();
    for n in cfg.node_indices() {
        let bb = &cfg[n];
        let mut stmts = Vec::<IRStatement>::new();
        for ins in &bb.instrs {
            let mut args = Vec::<IRArg>::new();
            for (i, op) in ins.operands.iter().enumerate() {
                let is_source = if is_mem_store(&ins.opcode) {
                    true
                } else if is_mem_load(&ins.opcode) {
                    i != 0
                } else {
                    i != 0
                };
                if is_source {
                    if let Some(reg) = phys_reg_of(op) {
                        args.push(IRArg::SSA(Var { name: reg.class.clone() + &reg.idx.to_string(), id: 0 }));
                        continue;
                    }
                }
                if i != 0 {
                    args.push(IRArg::Operand(op.clone()));
                }
            }
            let pred_var = if let Some(pred) = &ins.pred {
                if pred.reg.starts_with("P") {
                    if let Ok(idx) = pred.reg[1..].parse::<i32>() {
                        let preg = PhysReg { class: "P".to_string(), idx };
                        Some(Var { name: format!("P{}", idx), id: 0 })
                    } else { None }
                } else { None }
            } else { None };
            let mut dest = None;
            if is_mem_load(&ins.opcode) {
                if let Some(reg) = ins.operands.first().and_then(phys_reg_of) {
                    dest = Some(Var { name: reg.class.clone() + &reg.idx.to_string(), id: 0 });
                    defsites.entry(reg).or_default().insert(n);
                }
            } else if !is_mem_store(&ins.opcode) {
                if let Some(reg) = ins.operands.first().and_then(phys_reg_of) {
                    dest = Some(Var { name: reg.class.clone() + &reg.idx.to_string(), id: 0 });
                    defsites.entry(reg).or_default().insert(n);
                }
            }
            let rv = match ins.operands.last() {
                Some(Operand::ImmediateI(i)) => RValue::ImmI(*i),
                Some(Operand::ImmediateF(f)) => RValue::ImmF(*f),
                _ => RValue::Op {
                    opcode: ins.opcode.clone(),
                    args,
                },
            };
            stmts.push(IRStatement { dest, value: rv, pred: pred_var });
        }
        // 计算所有出边的条件跳转表
        let mut irdst = Vec::new();
        let this_idx = n;
        let mut cond_branch_info = None;
        if let Some(last) = bb.instrs.last() {
            if let Some(pred) = &last.pred {
                if pred.reg.starts_with("P") {
                    if let Ok(idx) = pred.reg[1..].parse::<i32>() {
                        cond_branch_info = Some((format!("P{}", idx), 0, pred.sense));
                    }
                }
            }
        }
        for edge in cfg.edges(this_idx) {
            let target_idx = edge.target();
            let target_addr = cfg[target_idx].start;
            match *edge.weight() {
                EdgeKind::CondBranch => {
                    if let Some((ref name, id, sense)) = cond_branch_info {
                        irdst.push((Some(IRCond::Pred { name: name.clone(), id, sense }), target_addr));
                    } else {
                        irdst.push((Some(IRCond::True), target_addr));
                    }
                }
                EdgeKind::FallThrough => {
                    if let Some((ref name, id, sense)) = cond_branch_info {
                        irdst.push((Some(IRCond::Pred { name: name.clone(), id, sense: !sense }), target_addr));
                    } else {
                        irdst.push((Some(IRCond::True), target_addr));
                    }
                }
                EdgeKind::UncondBranch => {
                    irdst.push((Some(IRCond::True), target_addr));
                }
            }
        }
        ir_blocks.insert(bb.id, IRBlock {
            id: bb.id,
            start_addr: bb.start,
            irdst,
            stmts,
        });
    }

    // 2. 计算支配前沿 DF(X)
    let mut df = HashMap::<NodeIndex, HashSet<NodeIndex>>::new();
    for y in cfg.node_indices() {
        let preds: Vec<_> = cfg.neighbors_directed(y, Direction::Incoming).collect();
        if preds.len() < 2 { continue; }
        let idom_y = doms.immediate_dominator(y);
        for &p in &preds {
            let mut runner = p;
            while Some(runner) != idom_y {
                df.entry(runner).or_default().insert(y);
                if let Some(idom_r) = doms.immediate_dominator(runner) {
                    runner = idom_r;
                } else {
                    break;
                }
            }
        }
    }

    // 3. Cytron算法插入phi节点
    let mut phi_blocks: HashMap<PhysReg, HashSet<NodeIndex>> = HashMap::new();
    let mut iter_count = 0usize;
    let all_nodes: Vec<_> = cfg.node_indices().collect();
    for (reg, defset) in &defsites {
        // 只对 P0-P6、R0-R255、UR* 做SSA
        let is_ssa_reg = (reg.class == "P" && (0..=6).contains(&reg.idx)) ||
                         (reg.class == "R" && (0..=255).contains(&reg.idx)) ||
                         (reg.class == "UR");
        if !is_ssa_reg { continue; }
        iter_count += 1;
        let mut has_already: HashMap<NodeIndex, usize> = HashMap::new();
        let mut work: HashMap<NodeIndex, usize> = HashMap::new();
        for &x in &all_nodes {
            has_already.insert(x, 0);
            work.insert(x, 0);
        }
        let mut W: Vec<NodeIndex> = Vec::new();
        for &x in defset {
            work.insert(x, iter_count);
            if !W.contains(&x) { W.push(x); }
        }
        while !W.is_empty() {
            let x = W.pop().unwrap();
            if let Some(dfset) = df.get(&x) {
                for &y in dfset {
                    if has_already.get(&y).copied().unwrap_or(0) < iter_count {
                        phi_blocks.entry(reg.clone()).or_default().insert(y);
                        has_already.insert(y, iter_count);
                        if work.get(&y).copied().unwrap_or(0) < iter_count {
                            work.insert(y, iter_count);
                            if !W.contains(&y) { W.push(y); }
                        }
                    }
                }
            }
        }
    }
    // 插入phi节点
    let mut phi_map: HashMap<(usize, usize), PhysReg> = HashMap::new();
    for (reg, blocks) in &phi_blocks {
        for &n in blocks {
            let preds: Vec<_> = cfg.neighbors_directed(n, Direction::Incoming).collect();
            if preds.len() > 1 {
                let block = ir_blocks.get_mut(&cfg[n].id).unwrap();
                let phi = IRStatement {
                    dest: Some(Var { name: reg.class.clone() + &reg.idx.to_string(), id: 0 }),
                    value: RValue::Phi(vec![]),
                    pred: None,
                };
                block.stmts.insert(0, phi);
                phi_map.insert((block.id, 0), reg.clone());
                debug_log!("insert phi for {:?} at block {}", reg, block.id);
            }
        }
    }

    // 4. Cytron算法变量重命名
    let mut domtree_children: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for n in cfg.node_indices() {
        if let Some(idom) = doms.immediate_dominator(n) {
            domtree_children.entry(idom).or_default().push(n);
        }
    }
    let mut stacks: HashMap<PhysReg, Vec<Var>> = HashMap::new();
    let mut reg_version: HashMap<PhysReg, usize> = HashMap::new();
    fn new_ssa_var(reg: &PhysReg, reg_version: &mut HashMap<PhysReg, usize>) -> Var {
        if reg.class == "PT" || reg.class == "RZ" {
            Var { name: reg.class.clone(), id: 0 }
        } else {
            let v = reg_version.entry(reg.clone()).or_insert(0);
            let ssa_var = Var { name: format!("{}{}", reg.class, reg.idx), id: *v };
            *v += 1;
            ssa_var
        }
    }
    fn parse_var_physreg(v: &Var) -> PhysReg {
        if v.name == "PT" || v.name == "RZ" {
            PhysReg { class: v.name.clone(), idx: 0 }
        } else if v.name.starts_with("P") {
            PhysReg { class: "P".to_string(), idx: v.name[1..].parse().unwrap_or(0) }
        } else if v.name.starts_with("R") {
            PhysReg { class: "R".to_string(), idx: v.name[1..].parse().unwrap_or(0) }
        } else if v.name.starts_with("UR") {
            PhysReg { class: "UR".to_string(), idx: v.name[2..].parse().unwrap_or(0) }
        } else {
            PhysReg { class: v.name.clone(), idx: 0 }
        }
    }
    struct RenameRecord {
        block_id: usize,
        phi_dest: Vec<(usize, Var)>,
        phi_args: Vec<(usize, usize, Var)>,
        normal_renames: Vec<(usize, Var)>,
        op_arg_renames: Vec<(usize, usize, Var)>,
    }
    fn collect_rename(
        n: NodeIndex,
        cfg: &ControlFlowGraph,
        domtree_children: &HashMap<NodeIndex, Vec<NodeIndex>>,
        stacks: &mut HashMap<PhysReg, Vec<Var>>,
        reg_version: &mut HashMap<PhysReg, usize>,
        phi_map: &HashMap<(usize, usize), PhysReg>,
        ir_blocks: &HashMap<usize, IRBlock>,
        out: &mut Vec<RenameRecord>,
    ) {
        let block = ir_blocks.get(&cfg[n].id).unwrap();
        let mut phi_dest = Vec::new();
        let mut normal_renames = Vec::new();
        let mut op_arg_renames = Vec::new();
        // 1. 处理Phi节点，分配新SSA版本，push栈，并记录新Var
        for (i, stmt) in block.stmts.iter().enumerate() {
            if let Some(dest) = &stmt.dest {
                if let RValue::Phi(_) = stmt.value {
                    if let Some(reg) = phi_map.get(&(block.id, i)) {
                        let v = new_ssa_var(reg, reg_version);
                        stacks.entry(reg.clone()).or_default().push(v.clone());
                        let v_debug = v.clone();
                        debug_log!("phi rename: block {} phi {} -> {:?}", block.id, i, v_debug);
                        phi_dest.push((i, v));
                    }
                }
            }
        }
        // 2. 重命名普通语句的参数
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            match &stmt.value {
                RValue::Op { args, .. } => {
                    for (arg_idx, arg) in args.iter().enumerate() {
                        if let IRArg::SSA(v) = arg {
                            let reg = parse_var_physreg(v);
                            if let Some(stack) = stacks.get(&reg) {
                                if let Some(top) = stack.last() {
                                    op_arg_renames.push((stmt_idx, arg_idx, top.clone()));
                                    debug_log!("op arg rename: block {} stmt {} arg {} -> {:?}", block.id, stmt_idx, arg_idx, top);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        // 3. 普通赋值分配新SSA版本，push栈
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            if let Some(dest) = &stmt.dest {
                if !matches!(stmt.value, RValue::Phi(_)) {
                    let reg = parse_var_physreg(dest);
                    let v = new_ssa_var(&reg, reg_version);
                    stacks.entry(reg.clone()).or_default().push(v.clone());
                    let v_debug = v.clone();
                    debug_log!("assign rename: block {} stmt {} -> {:?}", block.id, stmt_idx, v_debug);
                    normal_renames.push((stmt_idx, v));
                }
            }
        }
        // 4. 收集phi填充信息
        let mut phi_args = Vec::new();
        let succs: Vec<_> = cfg.neighbors(n).collect();
        for &succ in &succs {
            let succ_block_id = cfg[succ].id;
            let phi_indices: Vec<usize> = phi_map.iter()
                .filter(|(&(bid, _), _)| bid == succ_block_id)
                .map(|(&(bid, idx), _)| idx)
                .collect();
            for &i in &phi_indices {
                if let Some(reg) = phi_map.get(&(succ_block_id, i)) {
                    if let Some(stack) = stacks.get(reg) {
                        if let Some(top) = stack.last() {
                            phi_args.push((succ_block_id, i, top.clone()));
                            debug_log!("phi param: succ block {} phi {} <- {:?}", succ_block_id, i, top);
                        }
                    }
                }
            }
        }
        out.push(RenameRecord {
            block_id: block.id,
            phi_dest,
            phi_args,
            normal_renames,
            op_arg_renames,
        });
        // 5. 递归遍历支配树
        if let Some(children) = domtree_children.get(&n) {
            for &child in children {
                collect_rename(child, cfg, domtree_children, stacks, reg_version, phi_map, ir_blocks, out);
            }
        }
        // 6. pop本块分配的SSA版本
        for stmt in &block.stmts {
            if let Some(dest) = &stmt.dest {
                let reg = parse_var_physreg(dest);
                if let Some(stack) = stacks.get_mut(&reg) {
                    stack.pop();
                }
            }
        }
    }
    // 收集所有重命名和phi参数填充信息
    let mut rename_records = Vec::new();
    collect_rename(entry, cfg, &domtree_children, &mut stacks, &mut reg_version, &phi_map, &ir_blocks, &mut rename_records);
    // 统一批量可变借用ir_blocks，应用所有重命名和phi参数填充
    for rec in &rename_records {
        if let Some(block) = ir_blocks.get_mut(&rec.block_id) {
            // phi dest重命名
            for (i, v) in &rec.phi_dest {
                if let Some(stmt) = block.stmts.get_mut(*i) {
                    stmt.dest = Some(v.clone());
                }
            }
            // 普通赋值重命名
            for (stmt_idx, v) in &rec.normal_renames {
                if let Some(stmt) = block.stmts.get_mut(*stmt_idx) {
                    stmt.dest = Some(v.clone());
                }
            }
            // op参数重命名
            for (stmt_idx, arg_idx, v) in &rec.op_arg_renames {
                if let Some(stmt) = block.stmts.get_mut(*stmt_idx) {
                    if let RValue::Op { args, .. } = &mut stmt.value {
                        if let Some(IRArg::SSA(var)) = args.get_mut(*arg_idx) {
                            *var = v.clone();
                        }
                    }
                }
            }
        }
        // phi参数填充
        for (succ_block_id, i, val) in &rec.phi_args {
            if let Some(succ_block) = ir_blocks.get_mut(succ_block_id) {
                if let Some(IRStatement { value: RValue::Phi(ref mut args), .. }) = succ_block.stmts.get_mut(*i) {
                    args.push(val.clone());
                }
            }
        }
    }
    // 清理空phi节点
    for block in ir_blocks.values_mut() {
        block.stmts.retain(|stmt| {
            if let RValue::Phi(ref args) = stmt.value {
                !args.is_empty()
            } else {
                true
            }
        });
    }
    let mut blocks: Vec<_> = ir_blocks.into_iter().map(|(_, b)| b).collect();
    blocks.sort_by_key(|b| b.id);
    FunctionIR { blocks }
}

/// Output the SSA IR as a Graphviz DOT graph for debugging
pub fn ssa_to_dot(cfg: &ControlFlowGraph, fir: &FunctionIR) -> String {
    use std::fmt::Write;
    let mut s = String::from("digraph SSA_IR {\n");
    // Use rectangle shape for nodes
    s.push_str("  node [shape=box];\n");
    // Map block id to node index
    let mut id2idx = std::collections::HashMap::new();
    for idx in cfg.node_indices() {
        let bb = &cfg[idx];
        id2idx.insert(bb.id, idx);
    }
    // Emit nodes with IR
    for block in &fir.blocks {
        let mut label = format!("BB{} | Start: 0x{:x}\\n\\n",
            block.id,
            block.start_addr,
        );

        for stmt in &block.stmts {
            // 跳过分支类指令（BRA/JMP/RET/EXIT等）
            let is_branch = match &stmt.value {
                RValue::Op { opcode, .. } => {
                    let op = opcode.to_ascii_uppercase();
                    op == "BRA" || op == "JMP" || op == "JMPP" 
                },
                _ => false,
            };
            if is_branch { continue; }

            let dest = stmt.dest.as_ref().map(|v| {
                if v.name == "PT" || v.name == "RZ" { format!("{} = ", v.name) } else { format!("{}.{} = ", v.name, v.id) }
            }).unwrap_or_default();
            let val = match &stmt.value {
                RValue::Op { opcode, args } => {
                    let args_str = args.iter().map(|a| match a {
                        IRArg::SSA(v) => {
                            if v.name == "PT" || v.name == "RZ" { v.name.clone() } else { format!("{}.{}", v.name, v.id) }
                        },
                        IRArg::Operand(op) => format!("{:?}", op),
                    }).collect::<Vec<_>>().join(", ");
                    format!("{}({})", opcode, args_str)
                },
                RValue::Phi(vars) => {
                    let args_str = vars.iter().map(|v| format!("{}.{}", v.name, v.id)).collect::<Vec<_>>().join(", ");
                    format!("phi({})", args_str)
                },
                RValue::ImmI(i) => format!("imm {}", i),
                RValue::ImmF(f) => format!("imm {}", f),
                RValue::Undef => "undef".to_string(),
            };
            let pred = stmt.pred.as_ref().map(|v| {
                if v.name == "PT" || v.name == "RZ" { format!(" [@{}]", v.name) } else { format!(" [@{}.{}]", v.name, v.id) }
            }).unwrap_or_default();
            label.push_str(&format!("{}{}{}\\n", dest, val, pred));
        }
        label.push_str("\\n");
        // 显示所有IRDst分支
        for (cond, addr) in &block.irdst {
            let cond_str = match cond {
                Some(IRCond::True) => "(uncond)".to_string(),
                Some(IRCond::Pred { name, id, sense }) => {
                    if name == "PT" || name == "RZ" {
                        format!("({}{})", if *sense { "" } else { "!" }, name)
                    } else {
                        format!("({}{}.{})", if *sense { "" } else { "!" }, name, id)
                    }
                },
                None => "(unknown)".to_string(),
            };
            label.push_str(&format!("IRDst: {} -> 0x{:x}\\n", cond_str, addr));
        }
        writeln!(s, "  {} [label=\"{}\"];", block.id, label.replace("\"", "\\\"")).unwrap();
    }
    // Emit edges
    for e in cfg.edge_indices() {
        let (sidx, didx) = cfg.edge_endpoints(e).unwrap();
        let sid = &cfg[sidx].id;
        let did = &cfg[didx].id;
        writeln!(s, "  {} -> {};", sid, did).unwrap();
    }
    s.push('}');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Instruction, Operand, PredicateUse};
    use crate::cfg::{BasicBlock, ControlFlowGraph, EdgeKind};
    use petgraph::graph::Graph;

    #[test]
    fn test_irdst_conditional() {
        // 构造一个简单的条件跳转CFG: BB0 --(P0=1)--> BB1, BB0 --(P0=0)--> BB2
        // BB0: @P0 BRA 0x20; (fallthrough 0x10)
        let bb0 = BasicBlock {
            id: 0,
            start: 0x0,
            instrs: vec![Instruction {
                addr: 0x0,
                pred: Some(PredicateUse { reg: "P0".to_string(), sense: true }),
                opcode: "BRA".to_string(),
                operands: vec![Operand::ImmediateI(0x20)],
                raw: "@P0 BRA 0x20;".to_string(),
            }],
        };
        let bb1 = BasicBlock { id: 1, start: 0x10, instrs: vec![] };
        let bb2 = BasicBlock { id: 2, start: 0x20, instrs: vec![] };
        let mut g: ControlFlowGraph = Graph::new();
        let n0 = g.add_node(bb0);
        let n1 = g.add_node(bb1);
        let n2 = g.add_node(bb2);
        g.add_edge(n0, n1, EdgeKind::FallThrough); // fallthrough
        g.add_edge(n0, n2, EdgeKind::CondBranch); // branch
        let fir = build_ssa(&g);
        // 检查irdst
        let block0 = &fir.blocks[0];
        assert_eq!(block0.irdst.len(), 2);
        // 必须有一个是条件跳转到0x20, 一个是无条件到0x10
        let mut found_cond = false;
        let mut found_uncond = false;
        for (cond, addr) in &block0.irdst {
            if let Some(IRCond::Pred { name, sense, .. }) = cond {
                assert_eq!(name, "P0");
                if *sense {
                    assert_eq!(*addr, 0x20);
                    found_cond = true;
                } else {
                    assert_eq!(*addr, 0x10);
                    found_uncond = true;
                }
            }
        }
        assert!(found_cond && found_uncond);
    }
}

