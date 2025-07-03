//! CFG分析辅助模块：支配树、后支配树、回边、循环体、出口分析
use crate::cfg::ControlFlowGraph;
use crate::debug_log;
use petgraph::graph::{NodeIndex, EdgeReference};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

/// 计算支配树（每个节点的直接支配者）
pub fn compute_dominators(cfg: &ControlFlowGraph, entry: NodeIndex) -> HashMap<NodeIndex, NodeIndex> {
    let doms = petgraph::algo::dominators::simple_fast(cfg, entry);
    let mut result = HashMap::new();
    for n in cfg.node_indices() {
        if let Some(idom) = doms.immediate_dominator(n) {
            result.insert(n, idom);
        }
    }
    result
}

/// 计算后支配树（每个节点的直接后支配者）
pub fn compute_post_dominators(cfg: &ControlFlowGraph, exit: NodeIndex) -> HashMap<NodeIndex, NodeIndex> {
    let mut rev = cfg.clone();
    rev.reverse();
    let doms = petgraph::algo::dominators::simple_fast(&rev, exit);
    let mut result = HashMap::new();
    for n in rev.node_indices() {
        if let Some(idom) = doms.immediate_dominator(n) {
            if idom != n {
                result.insert(n, idom);
            }
        }
    }
    result
}

/// 检测所有回边（back-edges），返回 (from, to) 块id对
pub fn find_back_edges(cfg: &ControlFlowGraph, doms: &HashMap<NodeIndex, NodeIndex>) -> Vec<(usize, usize)> {
    let mut res = Vec::new();
    for e in cfg.edge_references() {
        let from = e.source();
        let to = e.target();
        // 回边定义：目标支配源
        let mut cur = from;
        // 先特判自环
        if from == to {
            res.push((cfg[from].id, cfg[to].id));
            continue;
        }
        // 从from开始沿着支配链向上找，直到找到to
        while let Some(idom) = doms.get(&cur) {
            if *idom == to {
                res.push((cfg[from].id, cfg[to].id));
                break;
            }
            cur = *idom;
        }
    }
    res
}

/// 给定循环头，返回自然循环体（所有能从回边源到头的块）
pub fn collect_natural_loop(cfg: &ControlFlowGraph, head: usize, tail: usize) -> HashSet<usize> {
    // 先特判一下自环
    if head == tail {
        return vec![head].into_iter().collect();
    }
    // 初始化：body只包括head和tail
    let mut body = HashSet::new();
    body.insert(tail);
    body.insert(head);
    let mut stack = vec![tail];
    while let Some(cur) = stack.pop() {
        // 对于cur的每个前驱 pred，如果pred不在body中，则将pred加入到body和stack中
        for pred in cfg.neighbors_directed(
            cfg.node_indices().find(|&i| cfg[i].id == cur).unwrap(),
            Direction::Incoming) {
            let pid = cfg[pred].id;
            if pid != head {
                body.insert(pid);
                stack.push(pid);
            }
        }
    }
    body
}

/// 给定循环体，返回所有出口边 (from, to)（体内指向体外的边）
pub fn find_loop_exits(cfg: &ControlFlowGraph, loop_body: &HashSet<usize>) -> Vec<(usize, usize)> {
    let mut exits = Vec::new();
    for &bid in loop_body {
        let idx = cfg.node_indices().find(|&i| cfg[i].id == bid).unwrap();
        for succ in cfg.neighbors(idx) {
            let sid = cfg[succ].id;
            if !loop_body.contains(&sid) {
                exits.push((bid, sid));
            }
        }
    }
    exits
} 