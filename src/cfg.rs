//! Basic block partitioning and CFG construction over decoded terminators.

use crate::parser::{DecodedInstruction, TerminatorKind};
use petgraph::graph::{Graph, NodeIndex};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub start: u32,
    pub instrs: Vec<DecodedInstruction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    FallThrough,
    CondBranch,
    UncondBranch,
}

pub type ControlFlowGraph = Graph<BasicBlock, EdgeKind>;

fn next_block_node(addr2node: &BTreeMap<u32, NodeIndex>, start: u32) -> Option<NodeIndex> {
    use std::ops::Bound::{Excluded, Unbounded};

    addr2node
        .range((Excluded(start), Unbounded))
        .next()
        .map(|(_, node)| *node)
}

fn maybe_add_edge(
    graph: &mut ControlFlowGraph,
    from: NodeIndex,
    to: Option<NodeIndex>,
    kind: EdgeKind,
) {
    if let Some(to) = to {
        if graph.find_edge(from, to).is_none() {
            graph.update_edge(from, to, kind);
        }
    }
}

pub fn build_cfg(mut instrs: Vec<DecodedInstruction>) -> ControlFlowGraph {
    instrs.sort_by_key(|instr| instr.addr);
    let mut graph: ControlFlowGraph = Graph::new();
    if instrs.is_empty() {
        return graph;
    }

    let mut leaders = BTreeSet::new();
    leaders.insert(instrs[0].addr);

    for (idx, instr) in instrs.iter().enumerate() {
        let next_addr = instrs.get(idx + 1).map(|next| next.addr);
        match &instr.terminator {
            TerminatorKind::None => {}
            TerminatorKind::FallthroughOnly => {
                if let Some(next_addr) = next_addr {
                    leaders.insert(next_addr);
                }
            }
            TerminatorKind::CondBranch { taken, fallthrough } => {
                if let Some(target) = taken {
                    leaders.insert(*target);
                }
                if let Some(target) = fallthrough {
                    leaders.insert(*target);
                }
                if let Some(next_addr) = next_addr {
                    leaders.insert(next_addr);
                }
            }
            TerminatorKind::Jump { target } => {
                leaders.insert(*target);
                if let Some(next_addr) = next_addr {
                    leaders.insert(next_addr);
                }
            }
            TerminatorKind::Return | TerminatorKind::IndirectOrUnknown => {
                if let Some(next_addr) = next_addr {
                    leaders.insert(next_addr);
                }
            }
        }
    }

    let mut blocks = Vec::<BasicBlock>::new();
    let mut current: Option<BasicBlock> = None;
    for instr in instrs {
        if leaders.contains(&instr.addr) {
            if let Some(block) = current.take() {
                blocks.push(block);
            }
            current = Some(BasicBlock {
                id: blocks.len(),
                start: instr.addr,
                instrs: Vec::new(),
            });
        }
        if let Some(block) = &mut current {
            block.instrs.push(instr);
        }
    }
    if let Some(block) = current {
        blocks.push(block);
    }

    let mut addr2node = BTreeMap::<u32, NodeIndex>::new();
    for block in blocks {
        let node = graph.add_node(block);
        addr2node.insert(graph[node].start, node);
    }

    for node in graph.node_indices() {
        let start = graph[node].start;
        let last = graph[node]
            .instrs
            .last()
            .expect("basic block should contain at least one instruction")
            .clone();
        match last.terminator {
            TerminatorKind::None | TerminatorKind::FallthroughOnly => {
                maybe_add_edge(
                    &mut graph,
                    node,
                    next_block_node(&addr2node, start),
                    EdgeKind::FallThrough,
                );
            }
            TerminatorKind::CondBranch { taken, fallthrough } => {
                maybe_add_edge(
                    &mut graph,
                    node,
                    taken.and_then(|addr| addr2node.get(&addr).copied()),
                    EdgeKind::CondBranch,
                );
                let fallthrough_node = fallthrough
                    .and_then(|addr| addr2node.get(&addr).copied())
                    .or_else(|| next_block_node(&addr2node, start));
                maybe_add_edge(&mut graph, node, fallthrough_node, EdgeKind::FallThrough);
            }
            TerminatorKind::Jump { target } => {
                maybe_add_edge(
                    &mut graph,
                    node,
                    addr2node.get(&target).copied(),
                    EdgeKind::UncondBranch,
                );
            }
            TerminatorKind::Return | TerminatorKind::IndirectOrUnknown => {}
        }
    }

    graph
}

pub fn graph_to_dot(cfg: &ControlFlowGraph) -> String {
    use std::fmt::Write;

    let mut out = String::from(
        "digraph CFG {
",
    );
    for idx in cfg.node_indices() {
        let block = &cfg[idx];
        let _ = writeln!(
            out,
            r#"  {} [label="BB{}\n0x{:04x}"];"#,
            block.id, block.id, block.start
        );
    }
    for edge in cfg.edge_indices() {
        let (src, dst) = cfg.edge_endpoints(edge).unwrap();
        let _ = writeln!(out, "  {} -> {};", cfg[src].id, cfg[dst].id);
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::decode_sass;
    use petgraph::visit::EdgeRef;

    fn node_for_start(cfg: &ControlFlowGraph, start: u32) -> NodeIndex {
        cfg.node_indices()
            .find(|idx| cfg[*idx].start == start)
            .expect("missing block")
    }

    fn outgoing(cfg: &ControlFlowGraph, start: u32) -> Vec<(u32, EdgeKind)> {
        let node = node_for_start(cfg, start);
        let mut edges = cfg
            .edges(node)
            .map(|edge| (cfg[edge.target()].start, *edge.weight()))
            .collect::<Vec<_>>();
        edges.sort_by_key(|(target, kind)| (*target, *kind as u8));
        edges
    }

    #[test]
    fn cfg_uses_conditional_terminators() {
        let sass = r#"
            /*0000*/ @P0 BRA 0x0020 ;
            /*0010*/ IADD3 R1, R1, 0x1, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        assert_eq!(cfg.node_count(), 3);
        assert_eq!(
            outgoing(&cfg, 0x0),
            vec![(0x10, EdgeKind::FallThrough), (0x20, EdgeKind::CondBranch)]
        );
    }

    #[test]
    fn cfg_predicated_exit_only_falls_through() {
        let sass = r#"
            /*0040*/ @P0 EXIT ;
            /*0050*/ IADD3 R7, RZ, 0x4, RZ ;
            /*0060*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        assert_eq!(cfg.node_count(), 2);
        assert_eq!(outgoing(&cfg, 0x40), vec![(0x50, EdgeKind::FallThrough)]);
    }

    #[test]
    fn cfg_indirect_branch_has_no_speculative_edges() {
        let sass = r#"
            /*0000*/ BRX R0 ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        assert_eq!(cfg.node_count(), 2);
        assert!(outgoing(&cfg, 0x0).is_empty());
    }
}
