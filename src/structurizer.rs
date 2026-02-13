//! CFG structurization pass used by the decompiler pipeline.
//! This pass is intentionally conservative: when a region cannot be safely
//! converted into structured control flow, it falls back to explicit goto.

use crate::ir::{FunctionIR, IRBlock, IRExpr, IRCond, RegId, DisplayCtx, IRStatement, RValue};
use crate::cfg::{ControlFlowGraph, BasicBlock as CfgBasicBlock, EdgeKind}; 

use petgraph::graph::{NodeIndex, DiGraph};
use petgraph::visit::EdgeRef;
use petgraph::algo::dominators::{Dominators, simple_fast};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use petgraph::Direction;
// --- Structured AST Definition ---
#[derive(Debug, Clone, PartialEq)]
pub enum LoopType {
    While,
    DoWhile,    // Detection not fully implemented
    Endless,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructuredStatement {
    BasicBlock {
        block_id: usize,
        stmts: Vec<IRStatement>,
    },
    Sequence(Vec<StructuredStatement>),
    If {
        condition_block_id: usize,
        condition_expr: IRExpr,
        then_branch: Box<StructuredStatement>,
        else_branch: Option<Box<StructuredStatement>>,
    },
    Loop {
        loop_type: LoopType,
        header_block_id: Option<usize>,
        condition_expr: Option<IRExpr>,
        body: Box<StructuredStatement>,
    },
    Break(Option<usize>),
    Continue(Option<usize>),
    Return(Option<IRExpr>),
    UnstructuredJump {
        from_block_id: usize,
        to_block_id: usize,
        condition: Option<IRExpr>,
    },
    Empty,
}

impl fmt::Display for StructuredStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructuredStatement::BasicBlock { block_id, stmts } => write!(f, "BB{}[{} stmts]", block_id, stmts.len()),
            StructuredStatement::Sequence(s) => {
                write!(f, "Sequence(")?;
                for (i, stmt) in s.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    stmt.fmt(f)?;
                }
                write!(f, ")")
            },
            StructuredStatement::If { condition_block_id, condition_expr, then_branch, else_branch, .. } => {
                write!(f, "If(BB{}, Cond: {:?}, Then: {}", condition_block_id, condition_expr, then_branch)?;
                if let Some(eb) = else_branch {
                    write!(f, ", Else: {}", eb)?;
                }
                write!(f, ")")
            },
            StructuredStatement::Loop { header_block_id, loop_type, condition_expr, body, .. } => {
                write!(f, "{:?}-Loop(Hdr: BB{:?}, Cond: {:?}, Body: {})", loop_type, header_block_id.unwrap_or(usize::MAX), condition_expr, body)
            }
            StructuredStatement::Break(_) => write!(f, "Break"),
            StructuredStatement::Continue(_) => write!(f, "Continue"),
            StructuredStatement::Return(_) => write!(f, "Return"),
            StructuredStatement::UnstructuredJump { from_block_id, to_block_id, .. } => {
                write!(f, "Goto(BB{}->BB{})", from_block_id, to_block_id)
            }
            StructuredStatement::Empty => write!(f, "Empty"),
        }
    }
}


// --- Structurizer Implementation ---
pub struct Structurizer<'a> {
    cfg: &'a ControlFlowGraph, // DiGraph<CfgBasicBlock, EdgeKind>
    function_ir: &'a FunctionIR,
    dom: Dominators<NodeIndex>,
    pdom_algo_result: Dominators<NodeIndex>,
    immediate_pdom_map: HashMap<NodeIndex, NodeIndex>,
    block_id_to_node_index: HashMap<usize, NodeIndex>,
    addr_to_node_index: HashMap<u32, NodeIndex>,
    processed_cfg_nodes: HashSet<NodeIndex>,
}

impl<'a> Structurizer<'a> {
    fn node_is_dominated_by(&self, dom_results: &Dominators<NodeIndex>, node_to_check: NodeIndex, potential_dominator: NodeIndex) -> bool {
        if node_to_check == potential_dominator {
            return true;
        }
        let mut current = node_to_check;
        while let Some(idom) = dom_results.immediate_dominator(current) {
            if idom == potential_dominator {
                return true;
            }
            if idom == current { break; } 
            current = idom;
            if current == NodeIndex::end() { break; } 
        }
        false
    }

    pub fn new(cfg: &'a ControlFlowGraph, function_ir: &'a FunctionIR) -> Self {
        let entry_node = NodeIndex::new(0); // Assuming block 0 is the entry
        let dom = simple_fast(cfg, entry_node);

        let mut block_id_to_node_index = HashMap::new();
        let mut addr_to_node_index = HashMap::new();
        for node_idx in cfg.node_indices() {
            let summary = cfg.node_weight(node_idx).expect("CFG node should have a weight");
            block_id_to_node_index.insert(summary.id, node_idx);
            addr_to_node_index.insert(summary.start, node_idx);
        }

        let (pdom_algo_result, immediate_pdom_map) =
            Self::calculate_post_dominators(cfg, function_ir);

        Structurizer {
            cfg,
            function_ir,
            dom,
            pdom_algo_result,
            immediate_pdom_map,
            block_id_to_node_index,
            addr_to_node_index,
            processed_cfg_nodes: HashSet::new(),
        }
    }

    fn is_block_return(ir_block: &IRBlock) -> bool {
        if let Some(last_stmt) = ir_block.stmts.last() {
            if let RValue::Op { opcode, .. } = &last_stmt.value {
                // Only treat unconditional terminators as function returns.
                // Predicated EXIT/RET are conditional control-flow instructions.
                return last_stmt.pred.is_none()
                    && (opcode == "RET" || opcode == "EXIT" || opcode.starts_with("RET"));
            }
        }
        false
    }

    fn calculate_post_dominators(
        original_cfg: &ControlFlowGraph, 
        function_ir: &FunctionIR,
    ) -> (Dominators<NodeIndex>, HashMap<NodeIndex, NodeIndex>) {
        let mut reversed_cfg = DiGraph::<CfgBasicBlock, EdgeKind>::new();
        let mut node_map_orig_to_rev = HashMap::new();

        for node_idx in original_cfg.node_indices() {
            let weight = original_cfg.node_weight(node_idx).expect("Original CFG node missing weight").clone();
            let new_node = reversed_cfg.add_node(weight);
            node_map_orig_to_rev.insert(node_idx, new_node);
        }

        for edge_ref in original_cfg.edge_references() {
            if let (Some(source_rev), Some(target_rev)) = (
                node_map_orig_to_rev.get(&edge_ref.target()),
                node_map_orig_to_rev.get(&edge_ref.source())
            ) {
                 reversed_cfg.add_edge(*source_rev, *target_rev, *edge_ref.weight());
            }
        }

        let mut original_exit_nodes_in_rev_cfg = Vec::new();
        for node_idx_orig in original_cfg.node_indices() {
            let is_natural_exit = original_cfg.neighbors_directed(node_idx_orig, Direction::Outgoing).count() == 0;
            let block_id = original_cfg.node_weight(node_idx_orig).unwrap().id;
            let ir_block = function_ir.blocks.iter().find(|b| b.id == block_id);
            let is_return_exit = ir_block.map_or(false, Self::is_block_return);

            if is_natural_exit || is_return_exit {
                if let Some(rev_node) = node_map_orig_to_rev.get(&node_idx_orig) {
                    original_exit_nodes_in_rev_cfg.push(*rev_node);
                }
            }
        }

        let pdom_entry_node: NodeIndex;
        if original_exit_nodes_in_rev_cfg.is_empty() {
            if reversed_cfg.node_count() > 0 {
                pdom_entry_node = node_map_orig_to_rev.get(&NodeIndex::new(0)).copied().unwrap_or_else(|| reversed_cfg.node_indices().next().unwrap());
            } else {
                return (simple_fast(&reversed_cfg, NodeIndex::new(0)), HashMap::new());
            }
        } else if original_exit_nodes_in_rev_cfg.len() == 1 {
            pdom_entry_node = original_exit_nodes_in_rev_cfg[0];
        } else {
            // Ensure CfgBasicBlock has a constructor or all fields are public for this direct init
            let synthetic_block_data = CfgBasicBlock { 
                id: usize::MAX, start: u32::MAX, instrs: vec![]
                // Adjust if your CfgBasicBlock has different fields (e.g. 'instrs')
                // If it has 'instrs: Vec<YourInstructionType>', use vec![]
            };
            let synthetic_super_exit_rev = reversed_cfg.add_node(synthetic_block_data);
            for orig_exit_rev_idx in original_exit_nodes_in_rev_cfg {
                reversed_cfg.add_edge(orig_exit_rev_idx, synthetic_super_exit_rev, EdgeKind::FallThrough);
            }
            pdom_entry_node = synthetic_super_exit_rev;
        }

        let pdom_algo_result = simple_fast(&reversed_cfg, pdom_entry_node);
        let mut immediate_pdom_map_orig_indices = HashMap::new();
        let rev_to_orig_map: HashMap<NodeIndex, NodeIndex> = node_map_orig_to_rev.into_iter().map(|(k, v)| (v, k)).collect();

        for node_idx_rev in reversed_cfg.node_indices() {
            if reversed_cfg.node_weight(node_idx_rev).map_or(false, |s| s.id == usize::MAX) { continue; }
            if let Some(ipdom_rev) = pdom_algo_result.immediate_dominator(node_idx_rev) {
                if reversed_cfg.node_weight(ipdom_rev).map_or(false, |s| s.id == usize::MAX) { continue; }
                if let (Some(orig_node), Some(orig_ipdom_node)) = (rev_to_orig_map.get(&node_idx_rev), rev_to_orig_map.get(&ipdom_rev)) {
                    immediate_pdom_map_orig_indices.insert(*orig_node, *orig_ipdom_node);
                }
            }
        }
        (pdom_algo_result, immediate_pdom_map_orig_indices)
    }

    pub fn structure_function(&mut self) -> Option<StructuredStatement> {
        self.processed_cfg_nodes.clear(); // Ensure clean state for new function
        let entry_cfg_node = self.block_id_to_node_index.get(&0).copied().unwrap_or_else(|| NodeIndex::new(0));
        self.structure_region_recursive(entry_cfg_node, None)
    }

    fn structure_region_recursive(
        &mut self,
        current_cfg_node: NodeIndex,
        desired_exit_node: Option<NodeIndex>,
    ) -> Option<StructuredStatement> {
        if self.processed_cfg_nodes.contains(&current_cfg_node) {
            return Some(StructuredStatement::Empty);
        }
        if Some(current_cfg_node) == desired_exit_node {
            return Some(StructuredStatement::Empty);
        }

        let mut region_statements = Vec::new();
        let mut active_cfg_node = current_cfg_node;

        // Debug checkpoint: this loop is the top-level region sequencer.
        'structure_sequence: loop {
            if self.processed_cfg_nodes.contains(&active_cfg_node) { break 'structure_sequence; }
            if Some(active_cfg_node) == desired_exit_node { break 'structure_sequence; }

            let ir_block = match self.get_ir_block_by_cfg_node(active_cfg_node) {
                Some(b) => b,
                None => { break 'structure_sequence; }
            };

            if Self::is_block_return(ir_block) && desired_exit_node.is_none() {
                self.processed_cfg_nodes.insert(active_cfg_node);
                let ret_val = ir_block.stmts.last().and_then(|s| {
                    if let RValue::Op { opcode, args } = &s.value {
                        if (opcode.starts_with("RET") || opcode == "EXIT") && !args.is_empty() {
                            return Some(args[0].clone());
                        }
                    }
                    None
                });
                region_statements.push(StructuredStatement::Return(ret_val));
                break 'structure_sequence;
            }

            // Step 1: loop recovery (highest priority).
            let mut next_node_after_pattern: Option<NodeIndex> = None;
            let mut matched_pattern = false;

            if let Some(loop_stmt) = self.try_structure_loop(active_cfg_node, desired_exit_node) {
                next_node_after_pattern = self.get_loop_successor(&loop_stmt);
                region_statements.push(loop_stmt);
                matched_pattern = true;
            // Step 2: if/if-else recovery.
            } else if let Some(if_stmt) = self.try_structure_if(active_cfg_node, desired_exit_node) {
                next_node_after_pattern = self.get_if_merge_or_successor(&if_stmt);
                region_statements.push(if_stmt);
                matched_pattern = true;
            }

            if matched_pattern {
                if let Some(next_node) = next_node_after_pattern {
                    if next_node == active_cfg_node && !matches!(region_statements.last(), Some(StructuredStatement::Loop{loop_type: LoopType::Endless,..})) { 
                        break 'structure_sequence;
                    }
                    active_cfg_node = next_node;
                } else { 
                    break 'structure_sequence;
                }
            } else {
                // Step 3: sequence recovery. If branching is still unstructured,
                // emit an explicit goto fallback to keep output total.
                self.processed_cfg_nodes.insert(active_cfg_node);
                region_statements.push(StructuredStatement::BasicBlock {
                    block_id: ir_block.id,
                    stmts: ir_block.stmts.clone(),
                });

                let successors: Vec<NodeIndex> = self.cfg.neighbors_directed(active_cfg_node, Direction::Outgoing).collect();
                if successors.len() == 1 {
                    active_cfg_node = successors[0];
                } else if successors.is_empty() {
                    break 'structure_sequence; 
                } else { 
                    // Unstructured multi-branch fallback.
                    region_statements.push(StructuredStatement::UnstructuredJump {
                        from_block_id: ir_block.id,
                        to_block_id: self.cfg[successors[0]].id,
                        condition: None,
                    });
                    break 'structure_sequence;
                }
            }
        }

        match region_statements.len() {
            0 => Some(StructuredStatement::Empty),
            1 => Some(region_statements.remove(0)),
            _ => Some(StructuredStatement::Sequence(region_statements)),
        }
    }

    fn get_ir_block_by_cfg_node(&self, cfg_node: NodeIndex) -> Option<&'a IRBlock> {
        self.cfg.node_weight(cfg_node).and_then(|summary| {
            self.function_ir.blocks.iter().find(|b| b.id == summary.id)
        })
    }

    fn extract_if_targets_and_condition(
        &self,
        ir_block: &'a IRBlock,
    ) -> Option<(IRExpr, NodeIndex, Option<NodeIndex>)> {
        if ir_block.irdst.is_empty() || ir_block.irdst.len() > 2 { return None; }

        if ir_block.irdst.len() == 2 {
            let (cond1_opt, addr1) = &ir_block.irdst[0];
            let (cond2_opt, addr2) = &ir_block.irdst[1];
            let node1 = self.addr_to_node_index.get(addr1).copied()?;
            let node2 = self.addr_to_node_index.get(addr2).copied()?;

            match (cond1_opt, cond2_opt) {
                (Some(IRCond::Pred { reg: r1, sense: s1 }), Some(IRCond::Pred { reg: r2, sense: s2 })) => {
                    if r1.class == r2.class && r1.idx == r2.idx && *s1 != *s2 {
                        let cond_expr = IRExpr::Reg(RegId::new(&r1.class, r1.idx, r1.sign));
                        return if *s1 { Some((cond_expr, node1, Some(node2))) } else { Some((cond_expr, node2, Some(node1))) };
                    }
                }
                 (Some(IRCond::Pred { reg, sense }), Some(IRCond::True)) => {
                    let cond_expr_val = IRExpr::Reg(RegId::new(&reg.class, reg.idx, reg.sign));
                    return if *sense { Some((cond_expr_val, node1, Some(node2))) } else { Some((negate_condition(cond_expr_val), node1, Some(node2))) };
                }
                (Some(IRCond::True), Some(IRCond::Pred { reg, sense })) => {
                    let cond_expr_val = IRExpr::Reg(RegId::new(&reg.class, reg.idx, reg.sign));
                    return if *sense { Some((cond_expr_val, node2, Some(node1))) } else { Some((negate_condition(cond_expr_val), node2, Some(node1))) };
                }
                _ => {}
            }
        } else if ir_block.irdst.len() == 1 { 
            let (cond_opt, addr) = &ir_block.irdst[0];
            if let Some(IRCond::Pred { reg, sense }) = cond_opt {
                let target_node = self.addr_to_node_index.get(addr).copied()?;
                let base_reg_expr = IRExpr::Reg(RegId::new(&reg.class, reg.idx, reg.sign));
                let cond_expr = if *sense { base_reg_expr } else { negate_condition(base_reg_expr) };
                return Some((cond_expr, target_node, None)); 
            }
        }
        None
    }
    
    fn try_structure_if(
        &mut self,
        condition_cfg_node: NodeIndex,
        overall_desired_exit: Option<NodeIndex>,
    ) -> Option<StructuredStatement> {
        if self.processed_cfg_nodes.contains(&condition_cfg_node) { return None; }

        let ir_cond_block = self.get_ir_block_by_cfg_node(condition_cfg_node)?;
        let (if_condition_expr, true_target_cfg_node, false_target_cfg_node_opt) =
            self.extract_if_targets_and_condition(ir_cond_block)?;

        let if_merge_cfg_node = self.immediate_pdom_map.get(&condition_cfg_node).copied();

        let actual_branch_exit_target = match (if_merge_cfg_node, overall_desired_exit) {
            (Some(imn), Some(ode)) => if self.node_is_dominated_by(&self.pdom_algo_result, imn, ode) || imn == ode { Some(ode) } else { Some(imn) },
            (Some(imn), None) => Some(imn),
            (None, Some(ode)) => Some(ode),
            (None, None) => None,
        };
        
        let is_true_target_exit = Some(true_target_cfg_node) == actual_branch_exit_target;
        let is_false_target_exit = match false_target_cfg_node_opt {
            Some(ftn) => Some(ftn) == actual_branch_exit_target,
            None => actual_branch_exit_target.is_none() || Some(condition_cfg_node) == actual_branch_exit_target, // For if-then, false path is implicitly the merge
        };

        self.processed_cfg_nodes.insert(condition_cfg_node);

        let then_statement = if is_true_target_exit {
            StructuredStatement::Empty
        } else {
            self.structure_region_recursive(true_target_cfg_node, actual_branch_exit_target)
                .unwrap_or(StructuredStatement::Empty)
        };

        let else_statement_opt = match false_target_cfg_node_opt {
            Some(false_target_node) => {
                if is_false_target_exit || Some(false_target_node) == actual_branch_exit_target { None }
                else {
                    Some(self.structure_region_recursive(false_target_node, actual_branch_exit_target)
                             .unwrap_or(StructuredStatement::Empty))
                }
            },
            None => None, 
        };
        
        if matches!(then_statement, StructuredStatement::Empty) && else_statement_opt.as_ref().map_or(true, |s| matches!(s, StructuredStatement::Empty)) {
             // If both branches are empty, this IF effectively does nothing or leads directly to merge.
             // Avoid creating an empty if { } else { }.
             // The condition_cfg_node will be processed as a BasicBlock by the outer loop if it's not truly terminal.
             self.processed_cfg_nodes.remove(&condition_cfg_node); // Allow it to be picked up as a basic block
            return None; 
        }

        Some(StructuredStatement::If {
            condition_block_id: ir_cond_block.id,
            condition_expr: if_condition_expr, 
            then_branch: Box::new(then_statement),
            else_branch: else_statement_opt.map(Box::new),
        })
    }

    fn get_if_merge_or_successor(&self, if_stmt: &StructuredStatement) -> Option<NodeIndex> {
        if let StructuredStatement::If { condition_block_id, .. } = if_stmt {
            let cond_cfg_node = self.block_id_to_node_index.get(condition_block_id)?;
            return self.immediate_pdom_map.get(cond_cfg_node).copied();
        }
        None
    }

    fn try_structure_loop(
        &mut self,
        potential_header_cfg_node: NodeIndex,
        _overall_desired_exit: Option<NodeIndex>, 
    ) -> Option<StructuredStatement> {
        if self.processed_cfg_nodes.contains(&potential_header_cfg_node) { return None; }

        let mut back_edges_to_header = Vec::new();
        for edge_ref in self.cfg.edges_directed(potential_header_cfg_node, Direction::Incoming) {
            let pred_node_idx = edge_ref.source();
            if self.node_is_dominated_by(&self.dom, pred_node_idx, potential_header_cfg_node) {
                // Keep self-latches: many SASS loops are represented as
                // a single header block with a back edge to itself.
                back_edges_to_header.push(pred_node_idx);
            }
        }
        if back_edges_to_header.is_empty() { return None; }

        let mut loop_body_cfg_nodes = HashSet::new();
        loop_body_cfg_nodes.insert(potential_header_cfg_node);

        let mut worklist_for_body: VecDeque<NodeIndex> = back_edges_to_header.iter().copied().collect();
        let mut visited_for_body_bfs = HashSet::new(); 
        visited_for_body_bfs.insert(potential_header_cfg_node);

        while let Some(node_to_explore_from) = worklist_for_body.pop_front() { 
            if node_to_explore_from == potential_header_cfg_node { continue; }
            if visited_for_body_bfs.contains(&node_to_explore_from) ||
               !self.node_is_dominated_by(&self.dom, node_to_explore_from, potential_header_cfg_node) {
                continue;
            }
            visited_for_body_bfs.insert(node_to_explore_from);
            loop_body_cfg_nodes.insert(node_to_explore_from);
            for pred in self.cfg.neighbors_directed(node_to_explore_from, Direction::Incoming) {
                if !visited_for_body_bfs.contains(&pred) && 
                   (pred == potential_header_cfg_node || self.node_is_dominated_by(&self.dom, pred, potential_header_cfg_node)) {
                     worklist_for_body.push_back(pred); 
                }
            }
        }
        
        if loop_body_cfg_nodes.len() <= 1 && !back_edges_to_header.iter().any(|&latch| latch == potential_header_cfg_node) { 
             // A single node loop must have the header as a latch.
             // If body is just header, but latches are external, something is off with body calc or it's not a simple natural loop.
             if loop_body_cfg_nodes.len() == 1 && loop_body_cfg_nodes.contains(&potential_header_cfg_node) {} else {return None;}
        }
        
        let mut loop_exit_edges = Vec::new();
        for &node_in_loop in &loop_body_cfg_nodes {
            for succ_node in self.cfg.neighbors_directed(node_in_loop, Direction::Outgoing) {
                if !loop_body_cfg_nodes.contains(&succ_node) {
                    loop_exit_edges.push((node_in_loop, succ_node));
                }
            }
        }
        
        let mut loop_natural_successor_node = self.immediate_pdom_map.get(&potential_header_cfg_node).copied();
        let unique_exit_targets: HashSet<NodeIndex> = loop_exit_edges.iter().map(|(_, to)| *to).collect();

        if loop_natural_successor_node.is_none() && unique_exit_targets.len() == 1 {
            loop_natural_successor_node = unique_exit_targets.iter().next().copied();
        } else if unique_exit_targets.len() > 1 {
            if let Some(lnsn) = loop_natural_successor_node {
                if !unique_exit_targets.iter().all(|target| *target == lnsn || self.node_is_dominated_by(&self.pdom_algo_result, lnsn, *target)) {
                     return None; 
                }
            } else { return None; }
        } else if unique_exit_targets.is_empty() {
            // No CFG exits - could be endless loop or exits via returns within body
            // loop_natural_successor_node will remain None
        }


        let header_ir_block = self.get_ir_block_by_cfg_node(potential_header_cfg_node)?;
        let mut loop_type = LoopType::Endless;
        let mut loop_condition_expr_val: Option<IRExpr> = None;

        let header_successors: Vec<NodeIndex> = self.cfg.neighbors_directed(potential_header_cfg_node, Direction::Outgoing).collect();
        if header_successors.len() == 2 {
            let s1 = header_successors[0];
            let s2 = header_successors[1];
            let s1_is_exit = Some(s1) == loop_natural_successor_node || !loop_body_cfg_nodes.contains(&s1);
            let s2_is_exit = Some(s2) == loop_natural_successor_node || !loop_body_cfg_nodes.contains(&s2);
            let s1_is_body_entry = loop_body_cfg_nodes.contains(&s1);
            let s2_is_body_entry = loop_body_cfg_nodes.contains(&s2);

            if (s1_is_exit && s2_is_body_entry) || (s2_is_exit && s1_is_body_entry) {
                if let Some((cond, true_target, _)) = self.extract_if_targets_and_condition(header_ir_block) {
                    let true_target_leads_to_body = 
                        (true_target == s1 && s1_is_body_entry) || (true_target == s2 && s2_is_body_entry);
                    loop_condition_expr_val = Some(if true_target_leads_to_body { cond } else { negate_condition(cond) });
                    loop_type = LoopType::While;
                }
            }
        }
        
        let mut backup_processed = self.processed_cfg_nodes.clone();
        for node in &loop_body_cfg_nodes { if *node != potential_header_cfg_node { self.processed_cfg_nodes.remove(node); } }
        // Header itself is part of Loop construct, not to be re-processed as a loose block by body structuring.
        self.processed_cfg_nodes.insert(potential_header_cfg_node); 


        let body_entry_for_structuring = if loop_type == LoopType::While {
            let entry_node = header_successors.iter().find(|&&succ| loop_body_cfg_nodes.contains(&succ) && succ != potential_header_cfg_node).copied();
            if entry_node.is_none() && loop_body_cfg_nodes.len() == 1 && loop_body_cfg_nodes.contains(&potential_header_cfg_node) {
                 potential_header_cfg_node // Single block loop where header branches to itself (implicitly handled by body being empty then)
            } else {
                entry_node.unwrap_or(potential_header_cfg_node) // Fallback, but should find a body entry for While
            }
        } else { 
            potential_header_cfg_node
        };
        
        let structured_body = if body_entry_for_structuring == potential_header_cfg_node && loop_type == LoopType::While {
            // While loop where condition leads to itself - body might be empty if no other path.
            // Or, the "body entry" is the header, but then it's more like an Endless loop with conditional break at top.
            // This case needs careful handling. For now, assume body call is okay.
             self.structure_loop_body_recursive(
                body_entry_for_structuring, 
                &loop_body_cfg_nodes, 
                potential_header_cfg_node, 
                loop_natural_successor_node
            ).unwrap_or(StructuredStatement::Empty)
        } else {
             self.structure_loop_body_recursive(
                body_entry_for_structuring, 
                &loop_body_cfg_nodes, 
                potential_header_cfg_node, 
                loop_natural_successor_node
            ).unwrap_or(StructuredStatement::Empty)
        };


        self.processed_cfg_nodes = backup_processed;
        for node in &loop_body_cfg_nodes { self.processed_cfg_nodes.insert(*node); }
        self.processed_cfg_nodes.insert(potential_header_cfg_node);

        Some(StructuredStatement::Loop {
            loop_type,
            header_block_id: Some(header_ir_block.id),
            condition_expr: loop_condition_expr_val, 
            body: Box::new(structured_body),
        })
    }
    
    fn structure_loop_body_recursive(
        &mut self,
        current_body_cfg_node: NodeIndex,
        nodes_in_this_loop: &HashSet<NodeIndex>,
        loop_header_for_continue: NodeIndex,
        loop_exit_for_break: Option<NodeIndex>,
    ) -> Option<StructuredStatement> {
        // Single-header loop fallback: keep the header body visible and model
        // the latch as an explicit continue.
        if current_body_cfg_node == loop_header_for_continue && nodes_in_this_loop.len() == 1 {
            if let Some(ir_block) = self.get_ir_block_by_cfg_node(current_body_cfg_node) {
                return Some(StructuredStatement::Sequence(vec![
                    StructuredStatement::BasicBlock {
                        block_id: ir_block.id,
                        stmts: ir_block.stmts.clone(),
                    },
                    StructuredStatement::Continue(None),
                ]));
            }
            return Some(StructuredStatement::Empty);
        }
        if current_body_cfg_node == loop_header_for_continue { return Some(StructuredStatement::Empty); } // Body starts *after* header for While/Do, or includes it for Endless
        if !nodes_in_this_loop.contains(&current_body_cfg_node) || self.processed_cfg_nodes.contains(&current_body_cfg_node) {
            return Some(StructuredStatement::Empty);
        }

        // Use the main region structurer for the body, but with special exit conditions
        // This is a placeholder for a more nuanced call that passes loop context
        // For now, we use structure_region_recursive, but this might not correctly form breaks/continues
        // if it doesn't know it's inside a loop targeting these specific header/exit nodes.
        // This is the primary simplification being made.
        // A true solution would have structure_region_recursive take loop_header and loop_exit as params.

        // Mark as processed for THIS recursive call for the body
        // self.processed_cfg_nodes.insert(current_body_cfg_node);
        
        // Try to structure this part of the body as a region aiming for either a break or continue.
        // This is still simplified.
        let mut body_statements = Vec::new();
        let mut active_node_in_body = current_body_cfg_node;
        
        'body_sequence: loop {
            if !nodes_in_this_loop.contains(&active_node_in_body) ||
               self.processed_cfg_nodes.contains(&active_node_in_body) ||
               active_node_in_body == loop_header_for_continue { // Don't re-enter header from body sequence
                break 'body_sequence;
            }
            if Some(active_node_in_body) == loop_exit_for_break { // Reached explicit break target
                body_statements.push(StructuredStatement::Break(None));
                self.processed_cfg_nodes.insert(active_node_in_body); // Mark break target as "handled" by break
                break 'body_sequence;
            }

            let ir_block = match self.get_ir_block_by_cfg_node(active_node_in_body) {
                Some(b) => b,
                None => break 'body_sequence,
            };
            
            // If this block is a return, it's like a break from the loop
            if Self::is_block_return(ir_block) {
                 self.processed_cfg_nodes.insert(active_node_in_body);
                 body_statements.push(StructuredStatement::BasicBlock{block_id: ir_block.id, stmts: ir_block.stmts.clone()}); // Include stmts of return block
                 body_statements.push(StructuredStatement::Return(None)); // Simplified
                 break 'body_sequence;
            }


            // For now, treat blocks in loop body sequentially if not direct break/continue
            // TODO: Integrate `try_structure_if` here for branches *within* the loop body.
            self.processed_cfg_nodes.insert(active_node_in_body);
            body_statements.push(StructuredStatement::BasicBlock {
                block_id: ir_block.id,
                stmts: ir_block.stmts.clone(),
            });

            let successors: Vec<NodeIndex> = self.cfg.neighbors_directed(active_node_in_body, Direction::Outgoing).collect();
            if successors.len() == 1 {
                let succ_node = successors[0];
                if succ_node == loop_header_for_continue {
                    body_statements.push(StructuredStatement::Continue(None));
                    break 'body_sequence; // Sequence ends with a continue
                }
                if Some(succ_node) == loop_exit_for_break {
                    body_statements.push(StructuredStatement::Break(None));
                    break 'body_sequence; // Sequence ends with a break
                }
                if nodes_in_this_loop.contains(&succ_node) {
                    active_node_in_body = succ_node; // Continue sequence within loop
                } else {
                    // Jump outside loop to non-designated exit: unhandled for now
                    body_statements.push(StructuredStatement::UnstructuredJump{ from_block_id: ir_block.id, to_block_id: self.cfg[succ_node].id, condition: None });
                    break 'body_sequence;
                }
            } else if successors.is_empty() { // End of path within loop?
                break 'body_sequence;
            } else { // Branch within loop body
                 body_statements.push(StructuredStatement::UnstructuredJump{ from_block_id: ir_block.id, to_block_id: self.cfg[successors[0]].id, condition: None });
                break 'body_sequence; // TODO: Handle with try_structure_if
            }
        }


        if body_statements.is_empty() { Some(StructuredStatement::Empty) }
        else if body_statements.len() == 1 { Some(body_statements.remove(0)) }
        else { Some(StructuredStatement::Sequence(body_statements)) }
    }

    fn get_loop_successor(&self, loop_stmt: &StructuredStatement) -> Option<NodeIndex> {
        if let StructuredStatement::Loop { header_block_id: Some(h_id), .. } = loop_stmt {
            let header_cfg_node = self.block_id_to_node_index.get(h_id)?;
            // Prefer actual exit targets if they are unique and known,
            // otherwise, fall back to immediate post-dominator of the header.
            // This information isn't directly stored in Loop stmt yet.
            // So current implementation is a heuristic.
            return self.immediate_pdom_map.get(header_cfg_node).copied();
        }
        None
    }

    pub fn pretty_print(&self, stmt: &StructuredStatement, ctx: &dyn DisplayCtx, indent_level: usize) -> String {
        let indent = "  ".repeat(indent_level);
        let mut s_out = String::new();

        match stmt {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                s_out.push_str(&format!("{}BB{} {{\n", indent, block_id));
                for ir_s in stmts {
                    let dest_str = ir_s.dest.as_ref().map_or_else(|| "_".to_string(), |d| ctx.expr(d));
                    let value_str = match &ir_s.value {
                        RValue::Op { opcode, args } => {
                            let args_s = args.iter().map(|a| ctx.expr(a)).collect::<Vec<_>>().join(", ");
                            format!("{}({})", opcode, args_s)
                        }
                        RValue::Phi(args) => {
                            let args_s = args.iter().map(|a| ctx.expr(a)).collect::<Vec<_>>().join(", ");
                            format!("phi({})", args_s)
                        }
                        RValue::ImmI(i) => format!("{}", i),
                        RValue::ImmF(f) => format!("{}", f),
                    };
                    let pred_str = ir_s.pred.as_ref().map_or("".to_string(), |p| format!("if ({}) ", ctx.expr(p)));
                    s_out.push_str(&format!("{}{}{} = {};\n", "  ".repeat(indent_level + 1), pred_str, dest_str, value_str));
                }
                s_out.push_str(&format!("{}}}\n", indent));
            }
            StructuredStatement::Sequence(stmts) => {
                for s_child in stmts {
                    if !matches!(s_child, StructuredStatement::Empty) { // Don't print empty
                        s_out.push_str(&self.pretty_print(s_child, ctx, indent_level));
                    }
                }
            }
            StructuredStatement::If { condition_block_id, condition_expr, then_branch, else_branch } => {
                s_out.push_str(&format!("{}// Condition from BB{}\n", indent, condition_block_id));
                s_out.push_str(&format!("{}if ({}) {{\n", indent, ctx.expr(condition_expr)));
                s_out.push_str(&self.pretty_print(then_branch, ctx, indent_level + 1));
                if let Some(eb) = else_branch {
                    s_out.push_str(&format!("{}}} else {{\n", indent));
                    s_out.push_str(&self.pretty_print(eb, ctx, indent_level + 1));
                }
                s_out.push_str(&format!("{}}}\n", indent));
            }
            StructuredStatement::Loop { loop_type, header_block_id, condition_expr, body } => {
                s_out.push_str(&format!("{}// Loop header BB{:?}\n", indent, header_block_id.unwrap_or(usize::MAX)));
                match loop_type {
                    LoopType::While => {
                        s_out.push_str(&format!("{}while ({}) {{\n", indent, condition_expr.as_ref().map_or("true".to_string(), |e| ctx.expr(e))));
                    }
                    LoopType::Endless => {
                         s_out.push_str(&format!("{}while (true) {{\n", indent));
                    }
                    LoopType::DoWhile => { 
                        s_out.push_str(&format!("{}do {{\n", indent));
                    }
                }
                s_out.push_str(&self.pretty_print(body, ctx, indent_level + 1));
                if *loop_type == LoopType::DoWhile {
                     s_out.push_str(&format!("{}}} while({});\n", indent, condition_expr.as_ref().map_or("true".to_string(), |e| ctx.expr(e))));
                } else {
                     s_out.push_str(&format!("{}}}\n", indent));
                }
            }
            StructuredStatement::Break(_) => s_out.push_str(&format!("{}break;\n", indent)),
            StructuredStatement::Continue(_) => s_out.push_str(&format!("{}continue;\n", indent)),
            StructuredStatement::Return(expr_opt) => {
                if let Some(e) = expr_opt {
                    s_out.push_str(&format!("{}return {};\n", indent, ctx.expr(e)));
                } else {
                    s_out.push_str(&format!("{}return;\n", indent));
                }
            }
            StructuredStatement::UnstructuredJump { from_block_id, to_block_id, condition } => {
                 if let Some(cond_expr_val) = condition { 
                    s_out.push_str(&format!("{}if ({}) goto BB{}; // from BB{}\n", indent, ctx.expr(cond_expr_val), to_block_id, from_block_id));
                 } else {
                    s_out.push_str(&format!("{}goto BB{}; // from BB{}\n", indent, to_block_id, from_block_id));
                 }
            }
            StructuredStatement::Empty => {}
        }
        s_out
    }
}

fn negate_condition(expr: IRExpr) -> IRExpr {
    match expr {
        IRExpr::Op { op, mut args } if op == "!" && args.len() == 1 => args.remove(0),
        _ => IRExpr::Op { op: "!".to_string(), args: vec![expr] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock as CfgBasicBlock, ControlFlowGraph, EdgeKind};
    use petgraph::graph::DiGraph;
    use std::collections::{HashMap, HashSet};

    fn stmt(opcode: &str) -> IRStatement {
        IRStatement {
            dest: None,
            value: RValue::Op { opcode: opcode.to_string(), args: vec![] },
            pred: None,
            mem_addr_args: None,
        }
    }

    fn build_case(
        specs: &[(usize, u32, Vec<(Option<IRCond>, u32)>, Vec<IRStatement>)],
        edges: &[(usize, usize, EdgeKind)],
    ) -> (ControlFlowGraph, FunctionIR, HashMap<usize, NodeIndex>) {
        let mut cfg = DiGraph::<CfgBasicBlock, EdgeKind>::new();
        let mut id_to_idx = HashMap::new();
        for (id, start, _, _) in specs {
            let idx = cfg.add_node(CfgBasicBlock {
                id: *id,
                start: *start,
                instrs: vec![],
            });
            id_to_idx.insert(*id, idx);
        }
        for (from, to, kind) in edges {
            cfg.add_edge(id_to_idx[from], id_to_idx[to], *kind);
        }

        let fir = FunctionIR {
            blocks: specs
                .iter()
                .map(|(id, start, irdst, stmts)| IRBlock {
                    id: *id,
                    start_addr: *start,
                    irdst: irdst.clone(),
                    stmts: stmts.clone(),
                })
                .collect(),
        };
        (cfg, fir, id_to_idx)
    }

    fn contains_if(s: &StructuredStatement) -> bool {
        match s {
            StructuredStatement::If { .. } => true,
            StructuredStatement::Sequence(v) => v.iter().any(contains_if),
            StructuredStatement::Loop { body, .. } => contains_if(body),
            _ => false,
        }
    }

    fn contains_loop(s: &StructuredStatement) -> bool {
        match s {
            StructuredStatement::Loop { .. } => true,
            StructuredStatement::Sequence(v) => v.iter().any(contains_loop),
            StructuredStatement::If { then_branch, else_branch, .. } => {
                contains_loop(then_branch)
                    || else_branch.as_deref().map(contains_loop).unwrap_or(false)
            }
            _ => false,
        }
    }

    fn contains_break(s: &StructuredStatement) -> bool {
        match s {
            StructuredStatement::Break(_) => true,
            StructuredStatement::Sequence(v) => v.iter().any(contains_break),
            _ => false,
        }
    }

    fn contains_continue(s: &StructuredStatement) -> bool {
        match s {
            StructuredStatement::Continue(_) => true,
            StructuredStatement::Sequence(v) => v.iter().any(contains_continue),
            _ => false,
        }
    }

    fn contains_goto(s: &StructuredStatement) -> bool {
        match s {
            StructuredStatement::UnstructuredJump { .. } => true,
            StructuredStatement::Sequence(v) => v.iter().any(contains_goto),
            StructuredStatement::Loop { body, .. } => contains_goto(body),
            StructuredStatement::If { then_branch, else_branch, .. } => {
                contains_goto(then_branch)
                    || else_branch.as_deref().map(contains_goto).unwrap_or(false)
            }
            _ => false,
        }
    }

    #[test]
    fn recovers_sequence() {
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x10)], vec![stmt("OP0")]),
            (1, 0x10, vec![(Some(IRCond::True), 0x20)], vec![stmt("OP1")]),
            (2, 0x20, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![
            (0, 1, EdgeKind::FallThrough),
            (1, 2, EdgeKind::FallThrough),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(matches!(out, StructuredStatement::Sequence(_) | StructuredStatement::BasicBlock { .. } | StructuredStatement::Return(_)));
    }

    #[test]
    fn recovers_if_then() {
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::Pred { reg: p0, sense: true }), 0x10)], vec![stmt("CMP")]),
            (1, 0x10, vec![(Some(IRCond::True), 0x20)], vec![stmt("THEN")]),
            (2, 0x20, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![
            (0, 1, EdgeKind::CondBranch),
            (0, 2, EdgeKind::FallThrough),
            (1, 2, EdgeKind::UncondBranch),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(contains_if(&out));
    }

    #[test]
    fn recovers_if_then_else() {
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (
                0,
                0x00,
                vec![
                    (Some(IRCond::Pred { reg: p0.clone(), sense: true }), 0x10),
                    (Some(IRCond::Pred { reg: p0, sense: false }), 0x20),
                ],
                vec![stmt("CMP")],
            ),
            (1, 0x10, vec![(Some(IRCond::True), 0x30)], vec![stmt("THEN")]),
            (2, 0x20, vec![(Some(IRCond::True), 0x30)], vec![stmt("ELSE")]),
            (3, 0x30, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![
            (0, 1, EdgeKind::CondBranch),
            (0, 2, EdgeKind::FallThrough),
            (1, 3, EdgeKind::UncondBranch),
            (2, 3, EdgeKind::UncondBranch),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(contains_if(&out));
    }

    #[test]
    fn recovers_simple_while_loop() {
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (
                0,
                0x00,
                vec![
                    (Some(IRCond::Pred { reg: p0.clone(), sense: true }), 0x10),
                    (Some(IRCond::Pred { reg: p0, sense: false }), 0x20),
                ],
                vec![stmt("CMP")],
            ),
            (1, 0x10, vec![(Some(IRCond::True), 0x00)], vec![stmt("BODY")]),
            (2, 0x20, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![
            (0, 1, EdgeKind::CondBranch),
            (0, 2, EdgeKind::FallThrough),
            (1, 0, EdgeKind::UncondBranch),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(contains_loop(&out));
    }

    #[test]
    fn loop_body_places_break() {
        let specs = vec![
            (0, 0x00, vec![], vec![stmt("HDR")]),
            (1, 0x10, vec![], vec![stmt("BODY")]),
            (3, 0x30, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![(1, 3, EdgeKind::UncondBranch)];
        let (cfg, fir, id_to_idx) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let loop_nodes: HashSet<NodeIndex> = [id_to_idx[&0], id_to_idx[&1]].into_iter().collect();

        let out = structurizer
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], Some(id_to_idx[&3]))
            .unwrap();
        assert!(contains_break(&out));
    }

    #[test]
    fn loop_body_places_continue() {
        let specs = vec![
            (0, 0x00, vec![], vec![stmt("HDR")]),
            (1, 0x10, vec![], vec![stmt("BODY")]),
        ];
        let edges = vec![(1, 0, EdgeKind::UncondBranch)];
        let (cfg, fir, id_to_idx) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let loop_nodes: HashSet<NodeIndex> = [id_to_idx[&0], id_to_idx[&1]].into_iter().collect();

        let out = structurizer
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], None)
            .unwrap();
        assert!(contains_continue(&out));
    }

    #[test]
    fn loop_body_falls_back_to_goto_on_unstructured_branch() {
        let specs = vec![
            (0, 0x00, vec![], vec![stmt("HDR")]),
            (1, 0x10, vec![], vec![stmt("BODY")]),
            (2, 0x20, vec![], vec![stmt("A")]),
            (3, 0x30, vec![], vec![stmt("B")]),
        ];
        let edges = vec![
            (1, 2, EdgeKind::CondBranch),
            (1, 3, EdgeKind::FallThrough),
        ];
        let (cfg, fir, id_to_idx) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let loop_nodes: HashSet<NodeIndex> = [id_to_idx[&0], id_to_idx[&1]].into_iter().collect();

        let out = structurizer
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], Some(id_to_idx[&3]))
            .unwrap();
        assert!(contains_goto(&out));
    }

    #[test]
    fn predicated_exit_is_not_unconditional_return() {
        let block = IRBlock {
            id: 0,
            start_addr: 0,
            irdst: vec![],
            stmts: vec![IRStatement {
                dest: None,
                value: RValue::Op {
                    opcode: "EXIT".to_string(),
                    args: vec![],
                },
                pred: Some(IRExpr::Reg(RegId::new("P", 0, 1))),
                mem_addr_args: None,
            }],
        };
        assert!(!Structurizer::is_block_return(&block));
    }
}
