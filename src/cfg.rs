//! Basic block partitioning and CFG construction over decoded terminators.

use crate::parser::{DecodedInstruction, TerminatorKind};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

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

fn decoded_target_addr(instr: &DecodedInstruction) -> Option<u32> {
    instr.operands.first().and_then(|op| match op {
        crate::parser::DecodedOperand::ImmediateI(value) if *value >= 0 => Some(*value as u32),
        crate::parser::DecodedOperand::Raw(text) => {
            let trimmed = text.trim();
            if let Some(rest) = trimmed.strip_prefix("0x") {
                u32::from_str_radix(rest, 16).ok()
            } else {
                trimmed.parse::<u32>().ok()
            }
        }
        _ => None,
    })
}

fn call_has_explicit_lexical_return(instrs: &[DecodedInstruction], call_idx: usize) -> bool {
    let Some(next_addr) = instrs.get(call_idx + 1).map(|instr| instr.addr) else {
        return false;
    };
    let Some(prev) = call_idx.checked_sub(1).and_then(|idx| instrs.get(idx)) else {
        return false;
    };
    if prev.pred.is_some() {
        return false;
    }
    let is_return_setup =
        opcode_mnemonic(&prev.opcode) == "MOV" || prev.opcode.starts_with("IMAD.MOV");
    is_return_setup
        && prev
            .operands
            .iter()
            .any(|operand| decoded_operand_addr(operand) == Some(next_addr))
}

fn decoded_operand_addr(operand: &crate::parser::DecodedOperand) -> Option<u32> {
    match operand {
        crate::parser::DecodedOperand::ImmediateI(value) if *value >= 0 => Some(*value as u32),
        crate::parser::DecodedOperand::Raw(text) => {
            let trimmed = text.trim();
            if let Some(rest) = trimmed.strip_prefix("0x") {
                u32::from_str_radix(rest, 16).ok()
            } else {
                trimmed.parse::<u32>().ok()
            }
        }
        _ => None,
    }
}

fn helper_inline_opcode_allowed(opcode: &str) -> bool {
    matches!(
        opcode_mnemonic(opcode),
        "BSYNC"
            | "BSSY"
            | "CS2R"
            | "F2F"
            | "F2I"
            | "FADD"
            | "FCHK"
            | "FFMA"
            | "FMUL"
            | "FSEL"
            | "I2F"
            | "I2FP"
            | "IADD3"
            | "IMAD"
            | "IMNMX"
            | "ISETP"
            | "LDC"
            | "LDG"
            | "LDL"
            | "LDS"
            | "LEA"
            | "LOP3"
            | "MOV"
            | "MUFU"
            | "NOP"
            | "PLOP3"
            | "PRMT"
            | "S2R"
            | "S2UR"
            | "SEL"
            | "SHF"
            | "STG"
            | "STL"
            | "STS"
            | "UIADD3"
            | "UISETP"
            | "ULDC"
    )
}

fn helper_subcall_returns(
    start_addr: u32,
    instrs: &[DecodedInstruction],
    addr2idx: &BTreeMap<u32, usize>,
    cache: &mut HashMap<u32, bool>,
) -> bool {
    if let Some(cached) = cache.get(&start_addr) {
        return *cached;
    }
    let Some(&start_idx) = addr2idx.get(&start_addr) else {
        cache.insert(start_addr, false);
        return false;
    };

    let mut seen = BTreeSet::new();
    let mut work = VecDeque::from([start_idx]);
    while let Some(idx) = work.pop_front() {
        if !seen.insert(idx) {
            continue;
        }
        let Some(instr) = instrs.get(idx) else {
            cache.insert(start_addr, false);
            return false;
        };
        let mnem = opcode_mnemonic(&instr.opcode);
        match mnem {
            "RET" => {
                if instr.pred.is_some() {
                    let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len())
                    else {
                        cache.insert(start_addr, false);
                        return false;
                    };
                    work.push_back(next_idx);
                }
            }
            "EXIT" | "BRX" | "PRET" => {
                cache.insert(start_addr, false);
                return false;
            }
            "BRA" | "JMP" | "JMPP" => {
                let Some(target_addr) = decoded_target_addr(instr) else {
                    cache.insert(start_addr, false);
                    return false;
                };
                if target_addr < start_addr {
                    cache.insert(start_addr, false);
                    return false;
                }
                let Some(&target_idx) = addr2idx.get(&target_addr) else {
                    cache.insert(start_addr, false);
                    return false;
                };
                work.push_back(target_idx);
                if instr.pred.is_some() {
                    let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len())
                    else {
                        cache.insert(start_addr, false);
                        return false;
                    };
                    work.push_back(next_idx);
                }
            }
            "CALL" | "CAL" | "JCAL" => {
                let Some(target_addr) = decoded_target_addr(instr) else {
                    cache.insert(start_addr, false);
                    return false;
                };
                if target_addr <= instr.addr
                    || !addr2idx.contains_key(&target_addr)
                    || !call_has_explicit_lexical_return(instrs, idx)
                    || !helper_subcall_returns(target_addr, instrs, addr2idx, cache)
                {
                    cache.insert(start_addr, false);
                    return false;
                }
                let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len()) else {
                    cache.insert(start_addr, false);
                    return false;
                };
                work.push_back(next_idx);
            }
            _ if helper_inline_opcode_allowed(&instr.opcode) => {
                let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len()) else {
                    cache.insert(start_addr, false);
                    return false;
                };
                work.push_back(next_idx);
            }
            _ => {
                cache.insert(start_addr, false);
                return false;
            }
        }
    }

    cache.insert(start_addr, true);
    true
}

/// Conservatively recognize compiler-emitted slowpath helpers that stay
/// in-line with the caller: they only branch forward within the helper
/// slice, any nested internal calls have an explicit lexical return target,
/// and the helper ultimately exits the kernel instead of returning.
fn helper_region_exits_without_return(
    start_addr: u32,
    instrs: &[DecodedInstruction],
    addr2idx: &BTreeMap<u32, usize>,
    helper_cache: &mut HashMap<u32, bool>,
    subcall_cache: &mut HashMap<u32, bool>,
) -> bool {
    if let Some(cached) = helper_cache.get(&start_addr) {
        return *cached;
    }
    let Some(&start_idx) = addr2idx.get(&start_addr) else {
        helper_cache.insert(start_addr, false);
        return false;
    };

    let mut seen = BTreeSet::new();
    let mut work = VecDeque::from([start_idx]);
    while let Some(idx) = work.pop_front() {
        if !seen.insert(idx) {
            continue;
        }
        let Some(instr) = instrs.get(idx) else {
            helper_cache.insert(start_addr, false);
            return false;
        };
        let mnem = opcode_mnemonic(&instr.opcode);
        match mnem {
            "RET" | "BRX" | "PRET" => {
                helper_cache.insert(start_addr, false);
                return false;
            }
            "EXIT" => {
                if instr.pred.is_some() {
                    let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len())
                    else {
                        helper_cache.insert(start_addr, false);
                        return false;
                    };
                    work.push_back(next_idx);
                }
            }
            "BRA" | "JMP" | "JMPP" => {
                let Some(target_addr) = decoded_target_addr(instr) else {
                    helper_cache.insert(start_addr, false);
                    return false;
                };
                if target_addr < start_addr {
                    helper_cache.insert(start_addr, false);
                    return false;
                }
                let Some(&target_idx) = addr2idx.get(&target_addr) else {
                    helper_cache.insert(start_addr, false);
                    return false;
                };
                work.push_back(target_idx);
                if instr.pred.is_some() {
                    let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len())
                    else {
                        helper_cache.insert(start_addr, false);
                        return false;
                    };
                    work.push_back(next_idx);
                }
            }
            "CALL" | "CAL" | "JCAL" => {
                let Some(target_addr) = decoded_target_addr(instr) else {
                    helper_cache.insert(start_addr, false);
                    return false;
                };
                if target_addr <= instr.addr
                    || !addr2idx.contains_key(&target_addr)
                    || !call_has_explicit_lexical_return(instrs, idx)
                    || !helper_subcall_returns(target_addr, instrs, addr2idx, subcall_cache)
                {
                    helper_cache.insert(start_addr, false);
                    return false;
                }
                let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len()) else {
                    helper_cache.insert(start_addr, false);
                    return false;
                };
                work.push_back(next_idx);
            }
            _ if helper_inline_opcode_allowed(&instr.opcode) => {
                let Some(next_idx) = idx.checked_add(1).filter(|next| *next < instrs.len()) else {
                    helper_cache.insert(start_addr, false);
                    return false;
                };
                work.push_back(next_idx);
            }
            _ => {
                helper_cache.insert(start_addr, false);
                return false;
            }
        }
    }

    helper_cache.insert(start_addr, true);
    true
}

fn rewrite_predicated_nonreturning_calls(instrs: &mut [DecodedInstruction]) {
    let addr2idx = instrs
        .iter()
        .enumerate()
        .map(|(idx, instr)| (instr.addr, idx))
        .collect::<BTreeMap<_, _>>();
    let mut helper_cache = HashMap::new();
    let mut subcall_cache = HashMap::new();

    for idx in 0..instrs.len() {
        let next_addr = instrs.get(idx + 1).map(|next| next.addr);
        let instr = &instrs[idx];
        if instr.pred.is_none() || opcode_mnemonic(&instr.opcode) != "CALL" {
            continue;
        }
        let Some(target) = decoded_target_addr(instr) else {
            continue;
        };
        if target <= instr.addr {
            continue;
        }
        // Only rewrite helpers whose inlined target stays inside this function,
        // uses known inline-safe opcodes, and falls back to the caller's next
        // instruction instead of returning through the normal call path.
        if !helper_region_exits_without_return(
            target,
            instrs,
            &addr2idx,
            &mut helper_cache,
            &mut subcall_cache,
        ) {
            continue;
        }
        instrs[idx].terminator = TerminatorKind::CondBranch {
            taken: Some(target),
            fallthrough: next_addr,
        };
    }
}

pub fn build_cfg(mut instrs: Vec<DecodedInstruction>) -> ControlFlowGraph {
    instrs.sort_by_key(|instr| instr.addr);
    rewrite_predicated_nonreturning_calls(&mut instrs);
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

fn opcode_mnemonic(opcode: &str) -> &str {
    opcode.split('.').next().unwrap_or(opcode)
}

fn has_conditional_exit(block: &BasicBlock) -> bool {
    let Some(last) = block.instrs.last() else {
        return false;
    };
    last.pred.is_some() && matches!(opcode_mnemonic(&last.opcode), "EXIT" | "RET")
}

fn reachable_nodes(cfg: &ControlFlowGraph) -> BTreeSet<NodeIndex> {
    let mut reachable = BTreeSet::new();
    let Some(entry) = cfg.node_indices().next() else {
        return reachable;
    };

    let mut queue = VecDeque::from([entry]);
    while let Some(node) = queue.pop_front() {
        if !reachable.insert(node) {
            continue;
        }
        for succ in cfg.neighbors(node) {
            if !reachable.contains(&succ) {
                queue.push_back(succ);
            }
        }
    }

    reachable
}

fn edge_label(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::FallThrough => "fallthrough",
        EdgeKind::CondBranch => "cond",
        EdgeKind::UncondBranch => "jump",
    }
}

pub fn graph_to_dot(cfg: &ControlFlowGraph) -> String {
    use std::fmt::Write;

    let reachable = reachable_nodes(cfg);
    let mut nodes = reachable.iter().copied().collect::<Vec<_>>();
    nodes.sort_by_key(|idx| cfg[*idx].id);

    let needs_exit = nodes.iter().any(|idx| {
        let block = &cfg[*idx];
        matches!(
            block.instrs.last().map(|instr| &instr.terminator),
            Some(TerminatorKind::Return)
        ) || has_conditional_exit(block)
    });

    let mut out = String::from("digraph CFG {\n");
    for idx in &nodes {
        let block = &cfg[*idx];
        let _ = writeln!(
            out,
            r#"  {} [label="BB{}\n0x{:04x}"];"#,
            block.id, block.id, block.start
        );
    }
    if needs_exit {
        let _ = writeln!(out, r#"  exit [shape=doublecircle,label="EXIT"];"#);
    }

    let mut edges = Vec::new();
    for idx in &nodes {
        for edge in cfg.edges(*idx) {
            if reachable.contains(&edge.target()) {
                edges.push((
                    cfg[*idx].id,
                    cfg[edge.target()].id,
                    edge_label(*edge.weight()).to_string(),
                ));
            }
        }

        let block = &cfg[*idx];
        if let Some(last) = block.instrs.last() {
            match last.terminator {
                TerminatorKind::Return => {
                    edges.push((cfg[*idx].id, usize::MAX, "return".to_string()));
                }
                TerminatorKind::CondBranch { .. } if has_conditional_exit(block) => {
                    edges.push((cfg[*idx].id, usize::MAX, "cond-exit".to_string()));
                }
                _ => {}
            }
        }
    }
    edges.sort();

    for (src, dst, label) in edges {
        if dst == usize::MAX {
            let _ = writeln!(out, r#"  {} -> exit [label="{}"];"#, src, label);
        } else {
            let _ = writeln!(out, r#"  {} -> {} [label="{}"];"#, src, dst, label);
        }
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

    fn helper_exits_without_return(sass: &str, start_addr: u32) -> bool {
        let instrs = decode_sass(sass);
        let addr2idx = instrs
            .iter()
            .enumerate()
            .map(|(idx, instr)| (instr.addr, idx))
            .collect::<BTreeMap<_, _>>();
        let mut helper_cache = HashMap::new();
        let mut subcall_cache = HashMap::new();
        helper_region_exits_without_return(
            start_addr,
            &instrs,
            &addr2idx,
            &mut helper_cache,
            &mut subcall_cache,
        )
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

    #[test]
    fn cfg_rewrites_predicated_nonreturning_calls_as_conditional_edges() {
        let sass = r#"
            /*0000*/ @P0 CALL.REL.NOINC 0x0020 ;
            /*0010*/ BRA 0x0010 ;
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
    fn helper_scan_rejects_unknown_opcodes() {
        let sass = r#"
            /*0000*/ @P0 CALL.REL.NOINC 0x0020 ;
            /*0010*/ BRA 0x0010 ;
            /*0020*/ FOO R0, R0 ;
            /*0030*/ EXIT ;
        "#;
        assert!(!helper_exits_without_return(sass, 0x20));
    }

    #[test]
    fn helper_scan_requires_explicit_lexical_return_for_nested_calls() {
        let sass = r#"
            /*0000*/ @P0 CALL.REL.NOINC 0x0020 ;
            /*0010*/ BRA 0x0010 ;
            /*0020*/ CALL.REL.NOINC 0x0040 ;
            /*0030*/ EXIT ;
            /*0040*/ RET ;
        "#;
        assert!(!helper_exits_without_return(sass, 0x20));
    }

    #[test]
    fn helper_scan_accepts_nested_calls_with_explicit_lexical_return() {
        let sass = r#"
            /*0000*/ @P0 CALL.REL.NOINC 0x0020 ;
            /*0010*/ BRA 0x0010 ;
            /*0020*/ MOV R4, 0x0040 ;
            /*0030*/ CALL.REL.NOINC 0x0050 ;
            /*0040*/ EXIT ;
            /*0050*/ RET ;
        "#;
        assert!(helper_exits_without_return(sass, 0x20));
        let cfg = build_cfg(decode_sass(sass));
        assert_eq!(
            outgoing(&cfg, 0x0),
            vec![(0x10, EdgeKind::FallThrough), (0x20, EdgeKind::CondBranch)]
        );
    }

    #[test]
    fn dot_adds_exit_edges_and_prunes_unreachable_tail_blocks() {
        let sass = r#"
            /*0000*/ @P0 EXIT ;
            /*0010*/ EXIT ;
            /*0020*/ BRA 0x0020 ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let dot = graph_to_dot(&cfg);
        assert!(dot.contains("exit [shape=doublecircle,label=\"EXIT\"]"));
        assert!(dot.contains("0 -> 1 [label=\"fallthrough\"]"));
        assert!(dot.contains("0 -> exit [label=\"cond-exit\"]"));
        assert!(dot.contains("1 -> exit [label=\"return\"]"));
        assert!(!dot.contains("BB2\n0x0020"));
    }
}
