//! cfg_analysis.rs  –– 静态 CFG 分析工具集
//! 1. 直接/后支配树
//! 2. 回边 + 自然循环 + 出口

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::ControlFlowGraph;
use petgraph::{algo::dominators::simple_fast, graph::NodeIndex, Direction};

/// 单个自然循环
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    pub head: NodeIndex,              // 回边的 head(入口)
    pub tail: NodeIndex,              // 回边的 tail
    pub body: BTreeSet<NodeIndex>,    // 包含 head+tail 的所有节点
    pub exits: BTreeSet<NodeIndex>,   // body→外部 的出口节点(目标)
}

/// 综合分析结果
#[derive(Debug, Clone)]
pub struct CFGAnalysis {
    pub idom: BTreeMap<NodeIndex, NodeIndex>,
    pub ipdom: BTreeMap<NodeIndex, NodeIndex>,
    pub loops: Vec<NaturalLoop>,
}

impl CFGAnalysis {
    pub fn new(cfg: &ControlFlowGraph) -> Self {
        let idom = Self::compute_idom(cfg);
        let ipdom = Self::compute_postdom(cfg);
        let loops = Self::compute_loops(cfg, &idom);

        Self { idom, ipdom, loops }
    }

    /* ---------- 基本算法 ---------- */

    fn compute_idom(cfg: &ControlFlowGraph) -> BTreeMap<NodeIndex, NodeIndex> {
        let entry = NodeIndex::new(0);
        let doms = simple_fast(cfg, entry);
        let mut out = BTreeMap::new();
        for n in cfg.node_indices() {
            if let Some(i) = doms.immediate_dominator(n) {
                out.insert(n, i);
            }
        }
        out
    }

    /// 后支配树：对 **反向图** 计算支配即可
    fn compute_postdom(cfg: &ControlFlowGraph) -> BTreeMap<NodeIndex, NodeIndex> {
        let exit = cfg
            .node_indices()
            .find(|&n| cfg.neighbors_directed(n, Direction::Outgoing).next().is_none())
            .unwrap_or(NodeIndex::new(0));
        let rev = petgraph::visit::Reversed(cfg);
        let doms = simple_fast(&rev, exit);
        let mut out = BTreeMap::new();
        for n in cfg.node_indices() {
            if let Some(i) = doms.immediate_dominator(n) {
                out.insert(n, i);
            }
        }
        out
    }

    /// 查 dom 关系
    fn dom(idom: &BTreeMap<NodeIndex, NodeIndex>, mut x: NodeIndex, y: NodeIndex) -> bool {
        // y dom x ?
        while let Some(&p) = idom.get(&x) {
            if p == y {
                return true;
            }
            if p == x {
                break;
            }
            x = p;
        }
        false
    }

    /* ---------- 回边 & 循环 ---------- */

    fn compute_loops(
        cfg: &ControlFlowGraph,
        idom: &BTreeMap<NodeIndex, NodeIndex>,
    ) -> Vec<NaturalLoop> {
        let mut loops = Vec::<NaturalLoop>::new();

        for tail in cfg.node_indices() {
            for head in cfg.neighbors_directed(tail, Direction::Outgoing) {
                if Self::dom(idom, tail, head) {
                    // (tail, head) is a back-edge
                    let mut body: BTreeSet<NodeIndex> = BTreeSet::new();
                    body.insert(head);
                    body.insert(tail);

                    // 经典算法：从 tail 逆 CFG 收集直到 head 支配
                    let mut work = vec![tail];
                    while let Some(n) = work.pop() {
                        for pred in cfg.neighbors_directed(n, Direction::Incoming) {
                            if !body.contains(&pred) {
                                body.insert(pred);
                                if !Self::dom(idom, pred, head) {
                                    // pred 可能在循环支配之外，但仍算体内
                                }
                                work.push(pred);
                            }
                        }
                    }

                    // 出口：体内 → 体外
                    let mut exits = BTreeSet::new();
                    for &n in &body {
                        for succ in cfg.neighbors_directed(n, Direction::Outgoing) {
                            if !body.contains(&succ) {
                                exits.insert(succ);
                            }
                        }
                    }
                    loops.push(NaturalLoop {
                        head,
                        tail,
                        body,
                        exits,
                    });
                }
            }
        }
        loops
    }

    /* ---------- 查询辅助 ---------- */

    pub fn loop_of(&self, n: NodeIndex) -> Option<&NaturalLoop> {
        self.loops.iter().find(|lp| lp.body.contains(&n))
    }
}
