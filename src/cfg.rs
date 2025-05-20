//! 基本块划分与控制流图 (CFG) 构建
//! * 仅面向单函数、同一段线性指令。
//! * 使用 petgraph::Graph 表示 CFG。

//! 基本块划分 & CFG 构建

use crate::parser::{Instruction, Operand};
use petgraph::{
    graph::{Graph, NodeIndex},
};

#[derive(Debug)]
pub struct BasicBlock {
    pub id: usize,
    pub start: u32,
    pub instrs: Vec<Instruction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    FallThrough,
    CondBranch,
    UncondBranch,
}

pub type ControlFlowGraph = Graph<BasicBlock, EdgeKind>;

fn is_branch(op: &str) -> bool { matches!(op, "BRA" | "JMP" | "JMPP" | "RET" | "EXIT") }

pub fn build_cfg(mut instrs: Vec<Instruction>) -> ControlFlowGraph {
    instrs.sort_by_key(|i| i.addr);
    use std::collections::{HashSet, BTreeMap};

    let mut leaders = HashSet::new();
    if let Some(first) = instrs.first() { leaders.insert(first.addr); }

    for win in instrs.windows(2) {
        let cur = &win[0];
        let next = &win[1];
        if is_branch(&cur.opcode) {
            // 目标
            if let Some(tgt) = branch_target_addr(cur) { leaders.insert(tgt); }
            // fall‑through 情况：
            let unconditional_term = matches!(cur.opcode.as_str(), "RET" | "EXIT") && cur.pred.is_none();
            let unconditional_jump = matches!(cur.opcode.as_str(), "BRA" | "JMP" | "JMPP") && cur.pred.is_none();
            if !(unconditional_term || unconditional_jump) {
                leaders.insert(next.addr);
            }
        }
    }

    // -- basic block 划分 --
    let mut map: BTreeMap<u32, Vec<Instruction>> = BTreeMap::new();
    for ins in instrs { map.entry(ins.addr).or_default().push(ins); }

    let mut blocks = Vec::<BasicBlock>::new();
    let mut cur: Option<BasicBlock> = None;
    for (addr, mut bucket) in map {
        if leaders.contains(&addr) { if let Some(b) = cur.take() { blocks.push(b); } cur = Some(BasicBlock{ id: blocks.len(), start: addr, instrs: Vec::new() }); }
        if let Some(b) = &mut cur { b.instrs.append(&mut bucket); }
    }
    if let Some(b) = cur { blocks.push(b); }

    // -- build graph nodes --
    let mut g: ControlFlowGraph = Graph::new();
    let mut addr2node = std::collections::HashMap::<u32, NodeIndex>::new();
    for bb in blocks { let idx = g.add_node(bb); addr2node.insert(g[idx].start, idx); }

    // -- edges --
    for idx in g.node_indices() {
        let bb_start = g[idx].start;
        let last = g[idx].instrs.last().unwrap();
        // 跳转目标边
        if is_branch(&last.opcode) {
            if let Some(tgt) = branch_target_addr(last) {
                if let Some(&tidx) = addr2node.get(&tgt) {
                    // 区分有无谓词
                    let ek = if last.pred.is_some() {
                        EdgeKind::CondBranch
                    } else if last.opcode == "BRA" {
                        EdgeKind::UncondBranch
                    } else {
                        EdgeKind::CondBranch // 其他分支默认条件分支
                    };
                    g.update_edge(idx, tidx, ek);
                }
            }
        }
        let last = g[idx].instrs.last().unwrap();
        // fall‑through边判定
        let unconditional_term = matches!(last.opcode.as_str(), "RET" | "EXIT") && last.pred.is_none();
        let unconditional_jump = matches!(last.opcode.as_str(), "BRA" | "JMP" | "JMPP") && last.pred.is_none();
        if !(unconditional_term || unconditional_jump) {
            if let Some((&_next_addr, &nidx)) = addr2node.iter().filter(|(&a, _)| a > bb_start).min_by_key(|(&a, _)| a) {
                if g.find_edge(idx, nidx).is_none() {
                    g.update_edge(idx, nidx, EdgeKind::FallThrough);
                }
            }
        }
    }
    g
}

/// 辅助：尝试从分支指令提取立即地址
fn branch_target_addr(ins: &Instruction) -> Option<u32> {
    ins.operands.first().and_then(|op| match op {
        Operand::ImmediateI(v) => Some(*v as u32),
        Operand::Raw(s) => u32::from_str_radix(s.trim_start_matches("0x"), 16).ok(),
        _ => None,
    })
}

pub fn graph_to_dot(cfg: &ControlFlowGraph) -> String {
    use std::fmt::Write;
    let mut s = String::from("digraph CFG {\n");
    for idx in cfg.node_indices() {
        let bb = &cfg[idx];
        let _ = writeln!(s, "  {} [label=\"BB{}\\n0x{:04x}\"];", bb.id, bb.id, bb.start);
    }
    for e in cfg.edge_indices() {
        let (sidx, didx) = cfg.edge_endpoints(e).unwrap();
        let _ = writeln!(s, "  {} -> {};", cfg[sidx].id, cfg[didx].id);
    }
    s.push('}');
    s
}