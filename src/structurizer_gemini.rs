// structurizer.rs (Consolidated from previous, with fixes)

use crate::ir::{FunctionIR, IRBlock, IRExpr, IRCond, RegId, DisplayCtx, IRStatement, RValue};
// Assuming CfgBasicBlock is the type of your CFG node weights.
// If it's `crate::cfg::BasicBlock`, ensure that's used.
use crate::cfg::{ControlFlowGraph, BasicBlock as CfgBasicBlock, EdgeKind}; 

use petgraph::graph::{NodeIndex, DiGraph};
use petgraph::visit::{DfsPostOrder, Reversed, IntoNodeIdentifiers, EdgeRef};
use petgraph::algo::dominators::{Dominators, simple_fast};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use petgraph::Direction;
use crate::debug_log;
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
                return opcode == "RET" || opcode == "EXIT" || opcode.starts_with("RET");
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

            let mut next_node_after_pattern: Option<NodeIndex> = None;
            let mut matched_pattern = false;

            if let Some(loop_stmt) = self.try_structure_loop(active_cfg_node, desired_exit_node) {
                next_node_after_pattern = self.get_loop_successor(&loop_stmt);
                region_statements.push(loop_stmt);
                matched_pattern = true;
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
                if pred_node_idx != potential_header_cfg_node { 
                    back_edges_to_header.push(pred_node_idx);
                }
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
            let s1_is_body_entry = loop_body_cfg_nodes.contains(&s1) && s1 != potential_header_cfg_node;
            let s2_is_body_entry = loop_body_cfg_nodes.contains(&s2) && s2 != potential_header_cfg_node;

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
                 body_statements.push(StructuredStatement::UnstructuredJump{ from_block_id: ir_block.id, to_block_id: usize::MAX, condition: None }); // Placeholder
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
                         s_out.push_str(&format!("{}loop {{\n", indent));
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
    use super::*; // Import everything from the parent module (structurizer)
    use crate::ir::{IRBlock, IRStatement, RValue, IRExpr, RegId, IRCond, DefaultDisplay};
    use crate::cfg::{ControlFlowGraph, BasicBlock as CfgBasicBlock, EdgeKind}; // Assuming this helper exists or we make one
    use crate::parser::{Instruction as ParserInstruction, Operand as ParserOperand}; // For creating mock IR
    use petgraph::graph::DiGraph;

    // Helper to create a simple CfgBasicBlock (adjust fields as per your actual CfgBasicBlock)
    fn new_cfg_block(id: usize, start_addr: u32) -> CfgBasicBlock {
        CfgBasicBlock { id, start: start_addr, instrs: vec![] } // Dummy end and instrs
    }
    
    // Helper to create a basic FunctionIR from IRBlocks
    fn ir_to_fir(blocks: Vec<IRBlock>) -> FunctionIR {
        FunctionIR { blocks }
    }

    // Helper to build CFG directly from IRBlocks for testing (simplified)
    // This assumes block IDs are contiguous from 0 for NodeIndex mapping
    // And that irdst contains target block IDs (not addresses) for simplicity in test setup.
    // In reality, you'd use your actual cfg::build_cfg or a more robust test helper.
    fn build_test_cfg_from_ir(ir_blocks: &[IRBlock]) -> ControlFlowGraph {
        let mut cfg = DiGraph::<CfgBasicBlock, EdgeKind>::new();
        let mut node_map = HashMap::new(); // Map block_id to NodeIndex

        for ir_block in ir_blocks {
            let node = cfg.add_node(new_cfg_block(ir_block.id, ir_block.start_addr));
            node_map.insert(ir_block.id, node);
        }

        for ir_block in ir_blocks {
            let source_node = node_map[&ir_block.id];
            for (cond_opt, target_block_id_addr) in &ir_block.irdst { // Assuming target_block_id_addr is block ID for test
                // In real CFG, target_block_id_addr would be an address.
                // For tests, let's assume it's a block ID for simplicity if it's not an address.
                // This part needs careful adaptation to how your irdst and cfg builder work.
                // For now, assume target_block_id_addr IS the target block ID.
                if let Some(target_node) = node_map.get(&(target_block_id_addr.clone() as usize)) { // HACK: treat u32 as usize ID
                    let edge_kind = match cond_opt {
                        Some(IRCond::True) => EdgeKind::UncondBranch,
                        Some(IRCond::Pred {..}) => EdgeKind::CondBranch,
                        None => EdgeKind::FallThrough, // Or UncondBranch depending on context
                    };
                    cfg.add_edge(source_node, *target_node, edge_kind);
                }
            }
        }
        cfg
    }


    #[test]
    fn test_node_is_dominated_by_simple() {
        let mut cfg = DiGraph::<CfgBasicBlock, EdgeKind>::new();
        let n0 = cfg.add_node(new_cfg_block(0, 0));
        let n1 = cfg.add_node(new_cfg_block(1, 10));
        let n2 = cfg.add_node(new_cfg_block(2, 20));
        cfg.add_edge(n0, n1, EdgeKind::FallThrough);
        cfg.add_edge(n1, n2, EdgeKind::FallThrough);

        let doms = simple_fast(&cfg, n0);
        let fir = ir_to_fir(vec![]); // Dummy FIR
        let structurizer = Structurizer::new(&cfg, &fir); // Need this for the method

        assert!(structurizer.node_is_dominated_by(&doms, n0, n0)); // Node dominates itself
        assert!(structurizer.node_is_dominated_by(&doms, n1, n0)); // n0 dominates n1
        assert!(structurizer.node_is_dominated_by(&doms, n2, n0)); // n0 dominates n2
        assert!(structurizer.node_is_dominated_by(&doms, n2, n1)); // n1 dominates n2
        assert!(!structurizer.node_is_dominated_by(&doms, n0, n1)); // n1 does not dominate n0
    }

    #[test]
    fn test_post_dominators_simple_diamond() {
        // 0 -> 1 -> 3
        //   -> 2 ->
        let ir_blocks = vec![
            IRBlock { id: 0, start_addr: 0, irdst: vec![(Some(IRCond::Pred{reg: RegId::new("P",0,1), sense: true}), 1), (Some(IRCond::Pred{reg: RegId::new("P",0,1), sense: false}), 2)], stmts: vec![] },
            IRBlock { id: 1, start_addr: 10, irdst: vec![(Some(IRCond::True), 3)], stmts: vec![] },
            IRBlock { id: 2, start_addr: 20, irdst: vec![(Some(IRCond::True), 3)], stmts: vec![] },
            IRBlock { id: 3, start_addr: 30, irdst: vec![], stmts: vec![ // Return block
                IRStatement { dest: None, value: RValue::Op { opcode: "RET".to_string(), args: vec![] }, pred: None, mem_addr_args: None }
            ]},
        ];
        let cfg = build_test_cfg_from_ir(&ir_blocks);
        let fir = ir_to_fir(ir_blocks);
        let (_pdom_results, pdom_map) = Structurizer::calculate_post_dominators(&cfg, &fir);

        let n0 = NodeIndex::new(0); // block id 0
        let n1 = NodeIndex::new(1); // block id 1
        let n2 = NodeIndex::new(2); // block id 2
        let n3 = NodeIndex::new(3); // block id 3

        // Expected: 3 pdoms everything. 0 is pdom'd by 3. 1 by 3. 2 by 3.
        // Immediate pdoms: ipdom(0)=3 (or by path through 1/2 then 3), ipdom(1)=3, ipdom(2)=3.
        // The map should store original indices.
        assert_eq!(pdom_map.get(&n1), Some(&n3));
        assert_eq!(pdom_map.get(&n2), Some(&n3));
        // ipdom of n0 is more complex: it's where paths from n0 *must* reconverge before exit.
        // In a diamond, n0's ipdom is n3 (the merge point before exit).
        assert_eq!(pdom_map.get(&n0), Some(&n3));
        assert!(pdom_map.get(&n3).is_none()); // Exit has no pdom in map
    }
    
    #[test]
    fn test_extract_if_condition_simple() {
        let p0_true = RegId::new("P", 0, 1);
        let block0 = IRBlock {
            id: 0, start_addr: 0x100,
            // if P0 then 0x200 else 0x300
            irdst: vec![
                (Some(IRCond::Pred { reg: p0_true.clone(), sense: true }), 0x200),
                (Some(IRCond::Pred { reg: p0_true.clone(), sense: false }), 0x300),
            ],
            stmts: vec![],
        };
        let block1_true_target = IRBlock { id: 1, start_addr: 0x200, irdst: vec![], stmts: vec![] };
        let block2_false_target = IRBlock { id: 2, start_addr: 0x300, irdst: vec![], stmts: vec![] };

        let ir_blocks = vec![block0.clone(), block1_true_target.clone(), block2_false_target.clone()];
        let cfg = build_test_cfg_from_ir(&ir_blocks);
        let fir = ir_to_fir(ir_blocks);
        let structurizer = Structurizer::new(&cfg, &fir);

        let (cond_expr, true_node_idx, false_node_idx_opt) = 
            structurizer.extract_if_targets_and_condition(&block0).unwrap();

        assert_eq!(cond_expr, IRExpr::Reg(RegId::new("P",0,1))); // Base reg P0
        assert_eq!(true_node_idx, structurizer.addr_to_node_index[&0x200]);
        assert_eq!(false_node_idx_opt, Some(structurizer.addr_to_node_index[&0x300]));
    }
    
    #[test]
    fn test_extract_if_condition_if_then() { // if P0 then target, else merge (implicit)
        let p0_true = RegId::new("P", 0, 1);
        let block0_cond = IRBlock {
            id: 0, start_addr: 0x100,
            irdst: vec![ (Some(IRCond::Pred { reg: p0_true.clone(), sense: true }), 0x200) ], // True target
            // False target is implicit fallthrough/jump to merge point.
            stmts: vec![],
        };
        let block1_then_body = IRBlock { id: 1, start_addr: 0x200, irdst: vec![(Some(IRCond::True),0x300)], stmts: vec![] };
        let block2_merge = IRBlock { id: 2, start_addr: 0x300, irdst: vec![], stmts: vec![] }; // Merge

        let ir_blocks = vec![block0_cond.clone(), block1_then_body.clone(), block2_merge.clone()];
        let cfg = build_test_cfg_from_ir(&ir_blocks); // build_test_cfg needs to handle this
        let fir = ir_to_fir(ir_blocks);
        let structurizer = Structurizer::new(&cfg, &fir);

        let (cond_expr, true_node_idx, false_node_idx_opt) = 
            structurizer.extract_if_targets_and_condition(&block0_cond).unwrap();
        
        assert_eq!(cond_expr, IRExpr::Reg(RegId::new("P",0,1)));
        assert_eq!(true_node_idx, structurizer.addr_to_node_index[&0x200]);
        assert!(false_node_idx_opt.is_none()); // For if-then, no explicit false target from extract
    }


    #[test]
    fn test_structure_simple_if_else() {
        // BB0: if P0 goto BB1 else BB2
        // BB1: ... goto BB3
        // BB2: ... goto BB3
        // BB3: (merge, return)
        let p0 = RegId::new("P", 0, 1);
        let ir_blocks = vec![
            IRBlock { id: 0, start_addr: 0, irdst: vec![(Some(IRCond::Pred{reg: p0.clone(), sense: true}), 10), (Some(IRCond::Pred{reg: p0.clone(), sense: false}), 20)], stmts: vec![/* cond set P0 */] },
            IRBlock { id: 1, start_addr: 10, irdst: vec![(Some(IRCond::True), 30)], stmts: vec![IRStatement::new_dummy(Some(RegId::new("R",1,1)), "ADD")] },
            IRBlock { id: 2, start_addr: 20, irdst: vec![(Some(IRCond::True), 30)], stmts: vec![IRStatement::new_dummy(Some(RegId::new("R",2,1)), "SUB")] },
            IRBlock { id: 3, start_addr: 30, irdst: vec![], stmts: vec![IRStatement::new_ret()]},
        ];
        let cfg = build_test_cfg_from_ir_addr(&ir_blocks); // Use addr based CFG builder for this
        let fir = ir_to_fir(ir_blocks);
        let mut structurizer = Structurizer::new(&cfg, &fir);

        let structured = structurizer.structure_function().unwrap();
        
        if let StructuredStatement::If { condition_expr, then_branch, else_branch, .. } = structured {
            assert_eq!(condition_expr, IRExpr::Reg(RegId::new("P",0,1)));
            assert!(matches!(*then_branch, StructuredStatement::BasicBlock { block_id: 1, .. }));
            assert!(matches!(else_branch.unwrap_as_ref(), StructuredStatement::BasicBlock { block_id: 2, .. }));
        } else {
            panic!("Expected If statement, got {:?}", structured);
        }
    }
    
    #[test]
    fn test_structure_simple_sequence() {
        // BB0 -> BB1 -> BB2 (ret)
        let ir_blocks = vec![
            IRBlock { id: 0, start_addr: 0, irdst: vec![(Some(IRCond::True), 10)], stmts: vec![/* op1 */] },
            IRBlock { id: 1, start_addr: 10, irdst: vec![(Some(IRCond::True), 20)], stmts: vec![/* op2 */] },
            IRBlock { id: 2, start_addr: 20, irdst: vec![], stmts: vec![IRStatement::new_ret()]},
        ];
        let cfg = build_test_cfg_from_ir_addr(&ir_blocks);
        let fir = ir_to_fir(ir_blocks);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let structured = structurizer.structure_function().unwrap();

        if let StructuredStatement::Sequence(seq) = structured {
            for s in seq {
                debug_log!("Structured: {:?}", s);
            }
            // assert_eq!(seq.len(), 3); // BB0, BB1, Return(from BB2)
            // assert!(matches!(seq[0], StructuredStatement::BasicBlock{ block_id: 0, ..}));
            // assert!(matches!(seq[1], StructuredStatement::BasicBlock{ block_id: 1, ..}));
            // assert!(matches!(seq[2], StructuredStatement::Return(_)));
        } else {
            panic!("Expected Sequence, got {:?}", structured);
        }
    }

    #[test]
    fn test_structure_simple_while_loop() {
        // BB0 (header): if P0 goto BB1 (body) else BB2 (exit)
        // BB1 (body): ... goto BB0 (latch)
        // BB2 (exit): return
        let p0 = RegId::new("P", 0, 1);
        let ir_blocks = vec![
            IRBlock { id: 0, start_addr: 0, // Header
                      irdst: vec![(Some(IRCond::Pred{reg: p0.clone(), sense: true}), 10), // to body
                                  (Some(IRCond::Pred{reg: p0.clone(), sense: false}), 20)], // to exit
                      stmts: vec![/* set P0 based on loop var */] },
            IRBlock { id: 1, start_addr: 10, // Body
                      irdst: vec![(Some(IRCond::True), 0)], // Latch to header
                      stmts: vec![IRStatement::new_dummy(None, "LOOP_OP")] },
            IRBlock { id: 2, start_addr: 20, // Exit
                      irdst: vec![], stmts: vec![IRStatement::new_ret()]},
        ];
        let cfg = build_test_cfg_from_ir_addr(&ir_blocks);
        let fir = ir_to_fir(ir_blocks);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let structured = structurizer.structure_function().unwrap();

        debug_log!("Structured: {:?}", structured);

        if let StructuredStatement::Loop { loop_type, condition_expr, body, header_block_id } = structured {
            assert_eq!(loop_type, LoopType::While);
            assert_eq!(header_block_id, Some(0));
            assert_eq!(condition_expr, Some(IRExpr::Reg(RegId::new("P",0,1))));
            assert!(matches!(*body, StructuredStatement::BasicBlock { block_id: 1, .. }), "Loop body was: {:?}", body);
        } else {
            panic!("Expected Loop statement, got {:?}", structured);
        }
    }

    // --- Test Helper Extensions for IRStatement (if not already present) ---
    // Add these to your ir.rs or directly in tests if only used here.
    impl IRStatement {
        fn new_dummy(dest: Option<RegId>, opcode_str: &str) -> Self {
            IRStatement {
                dest: dest.map(IRExpr::Reg),
                value: RValue::Op { opcode: opcode_str.to_string(), args: vec![] },
                pred: None, mem_addr_args: None,
            }
        }
        fn new_ret() -> Self {
            IRStatement {
                dest: None,
                value: RValue::Op{ opcode: "RET".to_string(), args: vec![] },
                pred: None, mem_addr_args: None,
            }
        }
    }
    // Helper to build CFG using addresses in irdst
    fn build_test_cfg_from_ir_addr(ir_blocks: &[IRBlock]) -> ControlFlowGraph {
        let mut cfg = DiGraph::<CfgBasicBlock, EdgeKind>::new();
        let mut node_map_id_to_idx = HashMap::new();
        let mut node_map_addr_to_idx = HashMap::new();

        for ir_block in ir_blocks {
            let node = cfg.add_node(new_cfg_block(ir_block.id, ir_block.start_addr));
            node_map_id_to_idx.insert(ir_block.id, node);
            node_map_addr_to_idx.insert(ir_block.start_addr, node);
        }

        for ir_block in ir_blocks {
            let source_node = node_map_id_to_idx[&ir_block.id];
            for (cond_opt, target_addr) in &ir_block.irdst {
                if let Some(target_node) = node_map_addr_to_idx.get(target_addr) {
                    let edge_kind = match cond_opt {
                        Some(IRCond::True) => EdgeKind::UncondBranch,
                        Some(IRCond::Pred {..}) => EdgeKind::CondBranch,
                        None => EdgeKind::FallThrough,
                    };
                    cfg.add_edge(source_node, *target_node, edge_kind);
                } else {
                    // panic!("IRDst target address 0x{:x} not found for block {}", target_addr, ir_block.id);
                }
            }
        }
        cfg
    }
     // Helper for Option<Box<Struct>> unwrap
    trait UnwrapBoxAsRef<T> {
        fn unwrap_as_ref(&self) -> &T;
    }
    impl<T> UnwrapBoxAsRef<T> for Option<Box<T>> {
        fn unwrap_as_ref(&self) -> &T {
            self.as_ref().expect("Option was None").as_ref()
        }
    }


} // end mod tests