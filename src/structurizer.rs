//! 结构化控制流（CFG → 结构化块树）
//! 目标：将SSA IR的CFG结构还原为结构化的顺序/条件/循环/基本块树，便于生成高级伪码。

use crate::cfg::ControlFlowGraph;
use crate::ir::{FunctionIR, IRCond};
use std::collections::{HashSet, HashMap};
use crate::debug_log;
use crate::cfg_analysis;
use petgraph::graph::NodeIndex;

#[derive(Debug, Clone)]
pub enum StructuredBlock {
    Seq(Vec<StructuredBlock>),
    If {
        cond: IRCond,
        then_branch: Box<StructuredBlock>,
        else_branch: Option<Box<StructuredBlock>>,
    },
    While {
        cond: IRCond,
        body: Box<StructuredBlock>,
    },
    Goto(usize), // 不可结构化跳转目标块id
    Basic(usize), // IRBlock id
    // 预留：DoWhile, Switch, Label, ...
}

/// 结构化主入口：后序遍历+结构识别，严格模式匹配顺序：循环→条件→顺序→goto
pub fn structurize(cfg: &ControlFlowGraph, fir: &FunctionIR) -> StructuredBlock {
    let entry = cfg.node_indices().min_by_key(|&i| cfg[i].id).unwrap();
    let doms = cfg_analysis::compute_dominators(cfg, entry);
    let back_edges = cfg_analysis::find_back_edges(cfg, &doms);
    // 构建回边映射：to -> Vec<from>
    let mut back_edge_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (from, to) in &back_edges {
        back_edge_map.entry(*to).or_default().push(*from);
    }
    let mut visited = HashSet::new();
    debug_log!("[structurizer] entry: {} doms: {:?} back_edges: {:?}", cfg[entry].id, doms, back_edges);
    structurize_postorder(cfg, fir, entry, &back_edge_map, &mut visited)
}

/// 后序递归结构化：先递归结构化所有后继，再在回溯时识别结构
fn structurize_postorder(
    cfg: &ControlFlowGraph,
    fir: &FunctionIR,
    node: NodeIndex,
    back_edge_map: &HashMap<usize, Vec<usize>>,
    visited: &mut HashSet<usize>,
) -> StructuredBlock {
    let block_id = cfg[node].id;
    if !visited.insert(block_id) {
        debug_log!("[structurizer] Block {} already visited, fallback to Goto", block_id);
        return StructuredBlock::Goto(block_id);
    }
    debug_log!("[structurizer] Structurizing block {} (postorder)", block_id);
    let mut succs: Vec<_> = cfg.neighbors(node).collect();
    // 递归结构化所有后继，收集结构化子块
    let mut sub_blocks = Vec::new();
    for &succ in &succs {
        sub_blocks.push((cfg[succ].id, structurize_postorder(cfg, fir, succ, back_edge_map, visited)));
    }
    // 1. 匹配循环（自然循环头）
    if let Some(back_froms) = back_edge_map.get(&block_id) {
        // block_id是循环头
        let mut loop_body = HashSet::new();
        for &tail in back_froms {
            let body = cfg_analysis::collect_natural_loop(cfg, block_id, tail);
            loop_body.extend(body);
        }
        debug_log!("[structurizer] Block {} is loop head, body: {:?}", block_id, loop_body);
        // 递归结构化循环体（只结构化loop_body内的块，避免重复结构化）
        let mut body_blocks = Vec::new();
        let mut loop_body_sorted: Vec<_> = loop_body.iter().cloned().collect();
        loop_body_sorted.sort();
        let mut loop_visited = HashSet::new();
        for bid in loop_body_sorted {
            if !loop_visited.insert(bid) { continue; }
            let nidx = cfg.node_indices().find(|&i| cfg[i].id == bid).unwrap();
            body_blocks.push(structurize_postorder(cfg, fir, nidx, back_edge_map, &mut loop_visited));
        }
        return StructuredBlock::While {
            cond: fir.blocks.iter().find(|b| b.id == block_id)
                .and_then(|b| b.irdst.get(0)).map(|(c,_)| c.clone().unwrap_or(IRCond::True)).unwrap_or(IRCond::True),
            body: Box::new(StructuredBlock::Seq(body_blocks))
        };
    }
    // 2. 匹配条件菱形（if/else）
    if let Some((then_id, else_id, merge_id)) = find_if_else_merge(cfg, block_id) {
        debug_log!("[structurizer] Block {} is if/else, then: {}, else: {}, merge: {:?}", block_id, then_id, else_id, merge_id);
        let cond = fir.blocks.iter().find(|b| b.id == block_id)
            .and_then(|b| b.irdst.get(0)).map(|(c,_)| c.clone().unwrap_or(IRCond::True)).unwrap_or(IRCond::True);
        let then_nidx = cfg.node_indices().find(|&i| cfg[i].id == then_id).unwrap();
        let else_nidx = cfg.node_indices().find(|&i| cfg[i].id == else_id).unwrap();
        let then_branch = Box::new(structurize_postorder(cfg, fir, then_nidx, back_edge_map, visited));
        let else_branch = Some(Box::new(structurize_postorder(cfg, fir, else_nidx, back_edge_map, visited)));
        let mut seq = vec![StructuredBlock::Basic(block_id)];
        seq.push(StructuredBlock::If { cond, then_branch, else_branch });
        if let Some(merge) = merge_id {
            let merge_nidx = cfg.node_indices().find(|&i| cfg[i].id == merge).unwrap();
            let merge_branch = structurize_postorder(cfg, fir, merge_nidx, back_edge_map, visited);
            seq.push(merge_branch);
        }
        return StructuredBlock::Seq(seq);
    }
    // 3. switch结构（预留，暂不实现）
    // TODO: switch识别
    // 4. 顺序块合并
    if succs.len() == 1 {
        let next = succs[0];
        let next_id = cfg[next].id;
        let preds: Vec<_> = cfg.neighbors_directed(next, petgraph::Direction::Incoming).collect();
        if preds.len() == 1 {
            debug_log!("[structurizer] Block {} and {} are sequential", block_id, next_id);
            return StructuredBlock::Seq(vec![
                StructuredBlock::Basic(block_id),
                sub_blocks[0].1.clone()
            ]);
        } else {
            debug_log!("[structurizer] Block {} sequential to {} but {} has multiple preds, insert Goto", block_id, next_id, next_id);
            return StructuredBlock::Seq(vec![
                StructuredBlock::Basic(block_id),
                StructuredBlock::Goto(next_id)
            ]);
        }
    }
    // 5. 多分支或不可结构化，降级为Goto
    if succs.len() > 1 {
        debug_log!("[structurizer] Block {} has multiple successors {:?}, insert Goto for each", block_id, succs.iter().map(|n| cfg[*n].id).collect::<Vec<_>>());
        let mut seq = vec![StructuredBlock::Basic(block_id)];
        for &target in &succs {
            seq.push(StructuredBlock::Goto(cfg[target].id));
        }
        return StructuredBlock::Seq(seq);
    }
    // 6. 默认：单独基本块
    debug_log!("[structurizer] Block {} is a single basic block", block_id);
    StructuredBlock::Basic(block_id)
}

fn get_successors(cfg: &ControlFlowGraph, block_id: usize) -> Vec<usize> {
    let idx = cfg.node_indices().find(|&i| cfg[i].id == block_id).unwrap();
    cfg.neighbors(idx).map(|n| cfg[n].id).collect()
}

fn get_predecessors(cfg: &ControlFlowGraph, block_id: usize) -> Vec<usize> {
    let idx = cfg.node_indices().find(|&i| cfg[i].id == block_id).unwrap();
    cfg.neighbors_directed(idx, petgraph::Direction::Incoming).map(|n| cfg[n].id).collect()
}

/// 改进版：支持自环和一般回边，允许循环体只有一个块
fn find_natural_loop(cfg: &ControlFlowGraph, block_id: usize) -> Option<std::collections::HashSet<usize>> {
    let succs = get_successors(cfg, block_id);
    if succs.contains(&block_id) {
        // 自环
        let mut body = std::collections::HashSet::new();
        body.insert(block_id);
        return Some(body);
    }
    // 一般回边
    let preds = get_predecessors(cfg, block_id);
    for &pred in &preds {
        if has_path(cfg, pred, block_id) {
            let mut body = std::collections::HashSet::new();
            collect_loop_body_rec(cfg, block_id, pred, &mut body);
            body.insert(block_id);
            return Some(body);
        }
    }
    None
}

fn has_path(cfg: &ControlFlowGraph, from: usize, to: usize) -> bool {
    let mut stack = vec![from];
    let mut visited = std::collections::HashSet::new();
    while let Some(cur) = stack.pop() {
        if cur == to { return true; }
        if !visited.insert(cur) { continue; }
        for s in get_successors(cfg, cur) { stack.push(s); }
    }
    false
}

fn collect_loop_body_rec(cfg: &ControlFlowGraph, head: usize, cur: usize, body: &mut std::collections::HashSet<usize>) {
    if !body.insert(cur) { return; }
    for pred in get_predecessors(cfg, cur) {
        if pred != head {
            collect_loop_body_rec(cfg, head, pred, body);
        }
    }
}

/// 改进版：查找if/else结构，允许merge为None（如一分支为exit）
fn find_if_else_merge(cfg: &ControlFlowGraph, block_id: usize) -> Option<(usize, usize, Option<usize>)> {
    let succs = get_successors(cfg, block_id);
    if succs.len() != 2 { return None; }
    let (a, b) = (succs[0], succs[1]);
    let mut exits_a = find_branch_exits(cfg, a, block_id);
    let mut exits_b = find_branch_exits(cfg, b, block_id);
    // 交集为merge
    let merge: Vec<_> = exits_a.intersection(&exits_b).cloned().collect();
    if merge.len() == 1 {
        Some((a, b, Some(merge[0])))
    } else if merge.is_empty() {
        // 允许没有merge（如一分支为exit）
        Some((a, b, None))
    } else {
        None
    }
}

/// 递归查找分支的所有出口（遇到回到if头则不算）
fn find_branch_exits(cfg: &ControlFlowGraph, start: usize, if_head: usize) -> std::collections::HashSet<usize> {
    let mut exits = std::collections::HashSet::new();
    let mut stack = vec![start];
    let mut visited = std::collections::HashSet::new();
    while let Some(cur) = stack.pop() {
        if !visited.insert(cur) { continue; }
        let succs = get_successors(cfg, cur);
        if succs.is_empty() || succs.contains(&if_head) {
            exits.insert(cur);
        } else {
            for s in succs { stack.push(s); }
        }
    }
    exits
} 

pub fn structured_block_to_text(sb: &StructuredBlock, indent: usize) -> String {
    let pad = |n| "  ".repeat(n);
    match sb {
        StructuredBlock::Seq(v) => {
            let mut s = String::new();
            s.push_str(&format!("{}Seq\n", pad(indent)));
            for b in v { s.push_str(&structured_block_to_text(b, indent+1)); }
            s
        }
        StructuredBlock::If{cond, then_branch, else_branch} => {
            let mut s = format!("{}If({:?})\n", pad(indent), cond);
            s.push_str(&structured_block_to_text(then_branch, indent+1));
            if let Some(e) = else_branch {
                s.push_str(&format!("{}Else\n", pad(indent+1)));
                s.push_str(&structured_block_to_text(e, indent+2));
            }
            s
        }
        StructuredBlock::While{cond, body} => {
            let mut s = format!("{}While({:?})\n", pad(indent), cond);
            s.push_str(&structured_block_to_text(body, indent+1));
            s
        }
        StructuredBlock::Goto(tgt) => {
            format!("{}Goto({})\n", pad(indent), tgt)
        }
        StructuredBlock::Basic(id) => {
            format!("{}Basic({})\n", pad(indent), id)
        }
    }
}