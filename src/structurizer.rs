//! CFG structurization pass used by the decompiler pipeline.
//! This pass is intentionally conservative: when a region cannot be safely
//! converted into structured control flow, it falls back to explicit goto.

use crate::ir::{FunctionIR, IRBlock, IRExpr, IRCond, RegId, DisplayCtx, IRStatement, RValue};
use crate::cfg::{ControlFlowGraph, BasicBlock as CfgBasicBlock, EdgeKind}; 
use crate::semantic_lift::{DefRef, SemanticLiftResult};

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
    /// Registers defined by phi nodes — used to detect loop-carried live-in
    /// old values in predicated ternaries.
    phi_defined_regs: HashSet<RegId>,
}

impl<'a> Structurizer<'a> {
    fn choose_entry_node(cfg: &ControlFlowGraph) -> Option<NodeIndex> {
        let mut candidates = cfg
            .node_indices()
            .filter(|&n| cfg.neighbors_directed(n, Direction::Incoming).count() == 0)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return cfg.node_indices().next();
        }
        candidates.sort_by_key(|&n| cfg[n].id);
        candidates.first().copied()
    }

    /// Check if a predicated instruction's old-value register (at `def_idx`)
    /// was defined by a phi node, indicating it is a loop-carried live-in.
    fn is_pred_old_phi(&self, stmt: &IRStatement, def_idx: usize) -> bool {
        stmt.pred_old_defs
            .get(def_idx)
            .and_then(|e| e.get_reg())
            .map_or(false, |r| self.phi_defined_regs.contains(r))
    }

    fn lifted_def_emit_order(stmt: &IRStatement) -> Vec<usize> {
        if stmt.defs.is_empty() {
            return vec![0];
        }
        let RValue::Op { opcode, .. } = &stmt.value else {
            return (0..stmt.defs.len()).collect();
        };
        let mnem = opcode.split('.').next().unwrap_or(opcode);
        let is_iadd3_non_x =
            matches!(mnem, "IADD3" | "UIADD3") && !opcode.split('.').any(|t| t == "X");
        if is_iadd3_non_x && stmt.defs.len() > 1 {
            // Carry predicate defs conceptually depend on pre-update low operands.
            // Emit them before low-result defs to preserve semantics in mutable-name views.
            let mut out = (1..stmt.defs.len()).collect::<Vec<_>>();
            out.push(0);
            return out;
        }
        (0..stmt.defs.len()).collect()
    }

    fn is_branch_only_opcode(opcode: &str) -> bool {
        matches!(opcode, "BRA" | "JMP" | "JMPP")
    }

    fn is_return_opcode(opcode: &str) -> bool {
        opcode == "RET" || opcode == "EXIT" || opcode.starts_with("RET")
    }

    fn is_setp_opcode(opcode: &str) -> bool {
        opcode.starts_with("ISETP") || opcode.starts_with("FSETP")
    }

    fn is_barrier_opcode(opcode: &str) -> bool {
        opcode.starts_with("BAR.SYNC")
    }

    fn should_omit_control_predicate_def(
        stmt: &IRStatement,
        stmt_idx: usize,
        block: &IRBlock,
        control_pred_regs: &[RegId],
    ) -> bool {
        let opcode = match &stmt.value {
            RValue::Op { opcode, .. } => opcode.as_str(),
            _ => return false,
        };
        if !Self::is_setp_opcode(opcode) {
            return false;
        }
        if stmt.defs.is_empty() {
            return false;
        }
        let pred_defs = stmt
            .defs
            .iter()
            .filter_map(|d| match d {
                IRExpr::Reg(r) if matches!(r.class.as_str(), "P" | "UP") => Some(r),
                _ => None,
            })
            .collect::<Vec<_>>();
        if pred_defs.is_empty() || pred_defs.len() != stmt.defs.len() {
            return false;
        }
        let all_control_preds = pred_defs.iter().all(|dest_reg| {
            control_pred_regs.iter().any(|r| {
                r.class == dest_reg.class && r.idx == dest_reg.idx && r.ssa == dest_reg.ssa
            })
        });
        if !all_control_preds {
            return false;
        }
        block
            .stmts
            .iter()
            .skip(stmt_idx + 1)
            .all(|s| match &s.value {
                RValue::Op { opcode, .. } => {
                    Self::is_branch_only_opcode(opcode) || Self::is_return_opcode(opcode)
                }
                _ => false,
            })
    }

    fn is_effectively_empty(stmt: &StructuredStatement) -> bool {
        match stmt {
            StructuredStatement::Empty => true,
            StructuredStatement::Sequence(stmts) => stmts.iter().all(Self::is_effectively_empty),
            _ => false,
        }
    }

    fn without_redundant_loop_tail_continue(stmt: &StructuredStatement) -> StructuredStatement {
        match stmt {
            StructuredStatement::Continue(_) => StructuredStatement::Empty,
            StructuredStatement::Sequence(stmts) => {
                if matches!(stmts.last(), Some(StructuredStatement::Continue(_))) {
                    let trimmed = &stmts[..stmts.len() - 1];
                    if trimmed.is_empty() {
                        StructuredStatement::Empty
                    } else if trimmed.len() == 1 {
                        trimmed[0].clone()
                    } else {
                        StructuredStatement::Sequence(trimmed.to_vec())
                    }
                } else {
                    stmt.clone()
                }
            }
            _ => stmt.clone(),
        }
    }

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

    fn has_cfg_path(&self, start: NodeIndex, goal: NodeIndex) -> bool {
        if start == goal {
            return true;
        }
        let mut seen = HashSet::new();
        let mut work = VecDeque::new();
        seen.insert(start);
        work.push_back(start);
        while let Some(n) = work.pop_front() {
            for succ in self.cfg.neighbors_directed(n, Direction::Outgoing) {
                if succ == goal {
                    return true;
                }
                if seen.insert(succ) {
                    work.push_back(succ);
                }
            }
        }
        false
    }

    fn select_if_merge_node(
        &self,
        condition_cfg_node: NodeIndex,
        true_target_cfg_node: NodeIndex,
        false_target_cfg_node_opt: Option<NodeIndex>,
    ) -> Option<NodeIndex> {
        let mut if_merge_cfg_node = self.immediate_pdom_map.get(&condition_cfg_node).copied();
        if let Some(false_target_cfg_node) = false_target_cfg_node_opt {
            let true_reaches_false = self.has_cfg_path(true_target_cfg_node, false_target_cfg_node);
            let false_reaches_true = self.has_cfg_path(false_target_cfg_node, true_target_cfg_node);
            if true_reaches_false ^ false_reaches_true {
                // One branch target reaches the other: treat the reached target
                // as the local join to avoid pulling post-merge regions into a branch.
                if_merge_cfg_node = Some(if false_reaches_true {
                    true_target_cfg_node
                } else {
                    false_target_cfg_node
                });
            }
        }
        if_merge_cfg_node
    }

    pub fn new(cfg: &'a ControlFlowGraph, function_ir: &'a FunctionIR) -> Self {
        let entry_node = Self::choose_entry_node(cfg).unwrap_or_else(|| NodeIndex::new(0));
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

        // Pre-compute the set of registers defined by phi nodes.
        let mut phi_defined_regs = HashSet::new();
        for block in &function_ir.blocks {
            for stmt in &block.stmts {
                if matches!(stmt.value, RValue::Phi(_)) {
                    for def in &stmt.defs {
                        if let IRExpr::Reg(r) = def {
                            phi_defined_regs.insert(r.clone());
                        }
                    }
                }
            }
        }

        Structurizer {
            cfg,
            function_ir,
            dom,
            pdom_algo_result,
            immediate_pdom_map,
            block_id_to_node_index,
            addr_to_node_index,
            processed_cfg_nodes: HashSet::new(),
            phi_defined_regs,
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
                pdom_entry_node = Self::choose_entry_node(original_cfg)
                    .and_then(|orig_entry| node_map_orig_to_rev.get(&orig_entry).copied())
                    .unwrap_or_else(|| reversed_cfg.node_indices().next().unwrap());
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
        let entry_cfg_node = Self::choose_entry_node(self.cfg)
            .or_else(|| self.block_id_to_node_index.get(&0).copied())
            .unwrap_or_else(|| NodeIndex::new(0));
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
                let mut pre_return_stmts = ir_block.stmts.clone();
                if let Some(IRStatement { value: RValue::Op { opcode, .. }, pred, .. }) =
                    pre_return_stmts.last()
                {
                    if pred.is_none() && Self::is_return_opcode(opcode) {
                        pre_return_stmts.pop();
                    }
                }
                let has_renderable_pre_return_stmt = pre_return_stmts.iter().any(|s| {
                    if matches!(s.value, RValue::Phi(_)) {
                        return false;
                    }
                    match &s.value {
                        RValue::Op { opcode, .. } => {
                            !Self::is_branch_only_opcode(opcode) && !Self::is_return_opcode(opcode)
                        }
                        _ => true,
                    }
                });
                if has_renderable_pre_return_stmt {
                    region_statements.push(StructuredStatement::BasicBlock {
                        block_id: ir_block.id,
                        stmts: pre_return_stmts,
                    });
                }
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
                        let cond_expr = IRExpr::Reg(r1.clone());
                        return if *s1 { Some((cond_expr, node1, Some(node2))) } else { Some((cond_expr, node2, Some(node1))) };
                    }
                }
                 (Some(IRCond::Pred { reg, sense }), Some(IRCond::True)) => {
                    let cond_expr_val = IRExpr::Reg(reg.clone());
                    return if *sense { Some((cond_expr_val, node1, Some(node2))) } else { Some((negate_condition(cond_expr_val), node1, Some(node2))) };
                }
                (Some(IRCond::True), Some(IRCond::Pred { reg, sense })) => {
                    let cond_expr_val = IRExpr::Reg(reg.clone());
                    return if *sense { Some((cond_expr_val, node2, Some(node1))) } else { Some((negate_condition(cond_expr_val), node2, Some(node1))) };
                }
                _ => {}
            }
        } else if ir_block.irdst.len() == 1 { 
            let (cond_opt, addr) = &ir_block.irdst[0];
            if let Some(IRCond::Pred { reg, sense }) = cond_opt {
                let target_node = self.addr_to_node_index.get(addr).copied()?;
                let base_reg_expr = IRExpr::Reg(reg.clone());
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

        let if_merge_cfg_node = self.select_if_merge_node(
            condition_cfg_node,
            true_target_cfg_node,
            false_target_cfg_node_opt,
        );

        // Prefer the local immediate post-dominator as the IF merge target.
        // Extending to an outer desired-exit can incorrectly pull post-merge
        // code into one branch and distort control flow.
        let actual_branch_exit_target = if if_merge_cfg_node.is_some() {
            if_merge_cfg_node
        } else {
            overall_desired_exit
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
            let ir_cond_block = self.get_ir_block_by_cfg_node(*cond_cfg_node)?;
            let (_, true_target_cfg_node, false_target_cfg_node_opt) =
                self.extract_if_targets_and_condition(ir_cond_block)?;
            return self.select_if_merge_node(
                *cond_cfg_node,
                true_target_cfg_node,
                false_target_cfg_node_opt,
            );
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
        
        let backup_processed = self.processed_cfg_nodes.clone();
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
                loop_natural_successor_node,
                None,
            ).unwrap_or(StructuredStatement::Empty)
        } else {
             self.structure_loop_body_recursive(
                body_entry_for_structuring, 
                &loop_body_cfg_nodes, 
                potential_header_cfg_node, 
                loop_natural_successor_node,
                None,
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
        stop_at: Option<NodeIndex>,
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
            if Some(active_node_in_body) == stop_at {
                break 'body_sequence;
            }
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


            let successors: Vec<NodeIndex> = self.cfg.neighbors_directed(active_node_in_body, Direction::Outgoing).collect();
            if successors.len() > 1 {
                if let Some((cond_expr, true_target, false_target_opt)) =
                    self.extract_if_targets_and_condition(ir_block)
                {
                    let false_target = false_target_opt.or_else(|| {
                        successors
                            .iter()
                            .copied()
                            .find(|succ| *succ != true_target)
                    });

                    if let Some(false_target) = false_target {
                        let merge_node_opt = self.immediate_pdom_map.get(&active_node_in_body).copied();
                        let then_stmt_opt = self.structure_loop_branch_target(
                            true_target,
                            nodes_in_this_loop,
                            loop_header_for_continue,
                            loop_exit_for_break,
                            merge_node_opt,
                        );
                        let else_stmt_opt = self.structure_loop_branch_target(
                            false_target,
                            nodes_in_this_loop,
                            loop_header_for_continue,
                            loop_exit_for_break,
                            merge_node_opt,
                        );

                        if let (Some(then_stmt), Some(else_stmt)) = (then_stmt_opt, else_stmt_opt) {
                            self.processed_cfg_nodes.insert(active_node_in_body);
                            body_statements.push(StructuredStatement::If {
                                condition_block_id: ir_block.id,
                                condition_expr: cond_expr,
                                then_branch: Box::new(then_stmt),
                                else_branch: if matches!(else_stmt, StructuredStatement::Empty) {
                                    None
                                } else {
                                    Some(Box::new(else_stmt))
                                },
                            });

                            if let Some(merge_node) = merge_node_opt {
                                if Some(merge_node) == stop_at {
                                    break 'body_sequence;
                                }
                                if merge_node == loop_header_for_continue {
                                    body_statements.push(StructuredStatement::Continue(None));
                                    break 'body_sequence;
                                }
                                if Some(merge_node) == loop_exit_for_break {
                                    body_statements.push(StructuredStatement::Break(None));
                                    break 'body_sequence;
                                }
                                if nodes_in_this_loop.contains(&merge_node) {
                                    active_node_in_body = merge_node;
                                    continue 'body_sequence;
                                }
                            }
                            break 'body_sequence;
                        }
                    }
                }
            }

            // Default sequential/fallback handling for non-if shapes.
            self.processed_cfg_nodes.insert(active_node_in_body);
            body_statements.push(StructuredStatement::BasicBlock {
                block_id: ir_block.id,
                stmts: ir_block.stmts.clone(),
            });

            if successors.len() == 1 {
                let succ_node = successors[0];
                if Some(succ_node) == stop_at {
                    break 'body_sequence;
                }
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
                break 'body_sequence;
            }
        }


        if body_statements.is_empty() { Some(StructuredStatement::Empty) }
        else if body_statements.len() == 1 { Some(body_statements.remove(0)) }
        else { Some(StructuredStatement::Sequence(body_statements)) }
    }

    fn structure_loop_branch_target(
        &mut self,
        target: NodeIndex,
        nodes_in_this_loop: &HashSet<NodeIndex>,
        loop_header_for_continue: NodeIndex,
        loop_exit_for_break: Option<NodeIndex>,
        stop_at: Option<NodeIndex>,
    ) -> Option<StructuredStatement> {
        if Some(target) == stop_at {
            return Some(StructuredStatement::Empty);
        }
        if target == loop_header_for_continue {
            return Some(StructuredStatement::Continue(None));
        }
        if Some(target) == loop_exit_for_break {
            return Some(StructuredStatement::Break(None));
        }
        if nodes_in_this_loop.contains(&target) {
            return self.structure_loop_body_recursive(
                target,
                nodes_in_this_loop,
                loop_header_for_continue,
                loop_exit_for_break,
                stop_at,
            );
        }
        None
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

    /// Returns true if the opcode is a convergence barrier with no data effect.
    fn is_convergence_barrier_opcode(opcode: &str) -> bool {
        matches!(opcode, "BSSY" | "BSYNC" | "SSY" | "SYNC")
    }

    fn render_condition_prelude_for_block(
        &self,
        block_id: usize,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        let Some(block) = self.function_ir.blocks.iter().find(|b| b.id == block_id) else {
            return String::new();
        };

        // Collect the predicates used by this block's own branch/exit so we can
        // suppress the ISETP that feeds the branch (it is expressed as the
        // structured if/while condition instead).
        let control_pred_regs: Vec<RegId> = block
            .irdst
            .iter()
            .filter_map(|(cond, _)| match cond {
                Some(IRCond::Pred { reg, .. }) => Some(reg.clone()),
                _ => None,
            })
            .collect();

        let mut lines = String::new();
        for (stmt_idx, ir_s) in block.stmts.iter().enumerate() {
            // Skip phi nodes — they are handled separately.
            if matches!(ir_s.value, RValue::Phi(_)) {
                continue;
            }

            let opcode = match &ir_s.value {
                RValue::Op { opcode, .. } => opcode.as_str(),
                _ => "",
            };

            // Skip pure control-flow opcodes (branches, exits, convergence barriers).
            if Self::is_branch_only_opcode(opcode)
                || Self::is_return_opcode(opcode)
                || Self::is_convergence_barrier_opcode(opcode)
            {
                continue;
            }

            // Skip the SETP that defines *only* the block's own branch predicate
            // (its semantics are captured by the structured if/while condition).
            if Self::should_omit_control_predicate_def(ir_s, stmt_idx, block, &control_pred_regs) {
                continue;
            }

            // Render barriers as __syncthreads().
            if Self::is_barrier_opcode(opcode) {
                let pred_str = ir_s
                    .pred
                    .as_ref()
                    .map_or_else(String::new, |p| format!("if ({}) ", ctx.expr(p)));
                lines.push_str(&format!(
                    "{}{}__syncthreads();\n",
                    "  ".repeat(indent_level + 1),
                    pred_str
                ));
                continue;
            }

            // Try lifted (semantic) rendering first.
            if let Some(res) = lifted {
                let mut emitted_any = false;
                for def_idx in Self::lifted_def_emit_order(ir_s) {
                    let def_ref = DefRef {
                        block_id,
                        stmt_idx,
                        def_idx,
                    };
                    let Some(lifted_stmt) = res.by_def.get(&def_ref) else {
                        continue;
                    };
                    match (&lifted_stmt.pred, &lifted_stmt.pred_old_val) {
                        (Some(p), Some(old)) => {
                            // Predicated instruction with old value:
                            // emit `dest = pred ? rhs : [/*phi*/]old;`
                            let phi_tag = if self.is_pred_old_phi(ir_s, def_idx) { "/*phi*/" } else { "" };
                            lines.push_str(&format!(
                                "{}{} = {} ? ({}) : {}{};
",
                                "  ".repeat(indent_level + 1),
                                lifted_stmt.dest,
                                p.render(),
                                lifted_stmt.rhs.render(),
                                phi_tag,
                                old
                            ));
                        }
                        (Some(p), None) => {
                            // Predicated but no old value (e.g. store) –
                            // keep the `if (pred)` guard.
                            lines.push_str(&format!(
                                "{}if ({}) {} = {};\n",
                                "  ".repeat(indent_level + 1),
                                p.render(),
                                lifted_stmt.dest,
                                lifted_stmt.rhs.render()
                            ));
                        }
                        (None, _) => {
                            lines.push_str(&format!(
                                "{}{} = {};\n",
                                "  ".repeat(indent_level + 1),
                                lifted_stmt.dest,
                                lifted_stmt.rhs.render()
                            ));
                        }
                    };
                    emitted_any = true;
                }
                if emitted_any {
                    continue;
                }
            }

            // Fallback: raw SSA rendering.
            let dest_str = if ir_s.defs.is_empty() {
                "_".to_string()
            } else if ir_s.defs.len() == 1 {
                ctx.expr(&ir_s.defs[0])
            } else {
                let defs = ir_s
                    .defs
                    .iter()
                    .map(|d| ctx.expr(d))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", defs)
            };
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
            let pred_str = ir_s
                .pred
                .as_ref()
                .map_or("".to_string(), |p| format!("if ({}) ", ctx.expr(p)));
            if ir_s.pred.is_some() && !ir_s.pred_old_defs.is_empty() && ir_s.defs.len() == 1 {
                let old_str = ctx.expr(&ir_s.pred_old_defs[0]);
                let pred_expr = ir_s.pred.as_ref().unwrap();
                // Predicated instruction with old value: emit ternary select.
                let phi_tag = if self.is_pred_old_phi(ir_s, 0) { "/*phi*/" } else { "" };
                lines.push_str(&format!(
                    "{}{} = {} ? ({}) : {}{};\n",
                    "  ".repeat(indent_level + 1),
                    dest_str,
                    ctx.expr(pred_expr),
                    value_str,
                    phi_tag,
                    old_str
                ));
            } else {
                lines.push_str(&format!(
                    "{}{}{} = {};\n",
                    "  ".repeat(indent_level + 1),
                    pred_str,
                    dest_str,
                    value_str
                ));
            }
        }

        if lines.is_empty() {
            return String::new();
        }

        let indent = "  ".repeat(indent_level);
        let mut out = String::new();
        out.push_str(&format!("{}BB{} {{\n", indent, block_id));
        out.push_str(&lines);
        out.push_str(&format!("{}}}\n", indent));
        out
    }

    fn render_condition_expr(
        &self,
        condition_block_id: usize,
        condition_expr: &IRExpr,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        match condition_expr {
            IRExpr::Op { op, args } if op == "!" && args.len() == 1 => {
                let inner = self.render_condition_expr(
                    condition_block_id,
                    &args[0],
                    ctx,
                    lifted,
                );
                format!("!({})", inner)
            }
            IRExpr::Reg(pred) => {
                if let Some(expr) = self.resolve_predicate_condition_rhs(
                    condition_block_id,
                    pred,
                    lifted,
                ) {
                    expr
                } else {
                    ctx.expr(condition_expr)
                }
            }
            _ => ctx.expr(condition_expr),
        }
    }

    fn resolve_predicate_condition_rhs(
        &self,
        block_id: usize,
        pred: &RegId,
        lifted: Option<&SemanticLiftResult>,
    ) -> Option<String> {
        let lifted = lifted?;

        // Only inline predicate conditions when the defining compare is in the same
        // condition block. Cross-block expansion can be stale once SSA names are
        // recovered into mutable temporaries.
        let block = self.function_ir.blocks.iter().find(|b| b.id == block_id)?;
        for (stmt_idx, stmt) in block.stmts.iter().enumerate().rev() {
            let is_setp = matches!(
                &stmt.value,
                RValue::Op { opcode, .. } if opcode.starts_with("ISETP") || opcode.starts_with("FSETP")
            );
            if !is_setp {
                continue;
            }
            for (def_idx, def) in stmt.defs.iter().enumerate() {
                let IRExpr::Reg(dst) = def else {
                    continue;
                };
                if dst.class != pred.class || dst.idx != pred.idx {
                    continue;
                }
                if let Some(want_ssa) = pred.ssa {
                    if dst.ssa != Some(want_ssa) {
                        continue;
                    }
                }
                let def_ref = DefRef {
                    block_id: block.id,
                    stmt_idx,
                    def_idx,
                };
                if let Some(ls) = lifted.by_def.get(&def_ref) {
                    return Some(ls.rhs.render());
                }
            }
        }

        None
    }

    pub fn pretty_print(&self, stmt: &StructuredStatement, ctx: &dyn DisplayCtx, indent_level: usize) -> String {
        self.pretty_print_with_lift(stmt, ctx, indent_level, None)
    }

    pub fn pretty_print_with_lift(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        let indent = "  ".repeat(indent_level);
        let mut s_out = String::new();

        match stmt {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                s_out.push_str(&format!("{}BB{} {{\n", indent, block_id));
                let mut omitted_phi_count = 0usize;
                let fir_block = self.function_ir.blocks.iter().find(|b| b.id == *block_id);
                let mut fir_search_from = 0usize;
                let control_pred_regs: Vec<RegId> = fir_block
                    .map(|b| {
                        b.irdst
                            .iter()
                            .filter_map(|(cond, _)| match cond {
                                Some(IRCond::Pred { reg, .. }) => Some(reg.clone()),
                                _ => None,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                for (stmt_idx, ir_s) in stmts.iter().enumerate() {
                    let mut lookup_stmt_idx = stmt_idx;
                    if let Some(orig_block) = fir_block {
                        if let Some((found_idx, _)) = orig_block
                            .stmts
                            .iter()
                            .enumerate()
                            .skip(fir_search_from)
                            .find(|(_, s)| {
                                s.defs == ir_s.defs && s.value == ir_s.value && s.pred == ir_s.pred
                            })
                        {
                            lookup_stmt_idx = found_idx;
                            fir_search_from = found_idx + 1;
                        }
                    }
                    if matches!(ir_s.value, RValue::Phi(_)) {
                        omitted_phi_count += 1;
                        continue;
                    }
                    if let Some(orig_block) = fir_block {
                        if Self::should_omit_control_predicate_def(
                            ir_s,
                            lookup_stmt_idx,
                            orig_block,
                            &control_pred_regs,
                        ) {
                            continue;
                        }
                    }
                    if let RValue::Op { opcode, .. } = &ir_s.value {
                        if Self::is_branch_only_opcode(opcode) {
                            // Branch instructions are represented by structured control flow
                            // (if/loop/goto), so omit them from block bodies.
                            continue;
                        }
                        if Self::is_barrier_opcode(opcode) {
                            let pred_str = ir_s
                                .pred
                                .as_ref()
                                .map_or_else(String::new, |p| format!("if ({}) ", ctx.expr(p)));
                            s_out.push_str(&format!(
                                "{}{}__syncthreads();\n",
                                "  ".repeat(indent_level + 1),
                                pred_str
                            ));
                            continue;
                        }
                        if Self::is_return_opcode(opcode) {
                            let pred_prefix = ir_s
                                .pred
                                .as_ref()
                                .map_or(String::new(), |p| format!("if ({}) ", ctx.expr(p)));
                            s_out.push_str(&format!(
                                "{}{}return;\n",
                                "  ".repeat(indent_level + 1),
                                pred_prefix
                            ));
                            continue;
                        }
                    }

                    if let Some(res) = lifted {
                        let mut emitted_any = false;
                        for def_idx in Self::lifted_def_emit_order(ir_s) {
                            let def_ref = DefRef {
                                block_id: *block_id,
                                stmt_idx: lookup_stmt_idx,
                                def_idx,
                            };
                            let Some(lifted_stmt) = res.by_def.get(&def_ref) else {
                                continue;
                            };
                            match (&lifted_stmt.pred, &lifted_stmt.pred_old_val) {
                                (Some(p), Some(old)) => {
                                    let phi_tag = if self.is_pred_old_phi(ir_s, def_idx) { "/*phi*/" } else { "" };
                                    s_out.push_str(&format!(
                                        "{}{} = {} ? ({}) : {}{};\n",
                                        "  ".repeat(indent_level + 1),
                                        lifted_stmt.dest,
                                        p.render(),
                                        lifted_stmt.rhs.render(),
                                        phi_tag,
                                        old
                                    ));
                                }
                                (Some(p), None) => {
                                    s_out.push_str(&format!(
                                        "{}if ({}) {} = {};\n",
                                        "  ".repeat(indent_level + 1),
                                        p.render(),
                                        lifted_stmt.dest,
                                        lifted_stmt.rhs.render()
                                    ));
                                }
                                (None, _) => {
                                    s_out.push_str(&format!(
                                        "{}{} = {};\n",
                                        "  ".repeat(indent_level + 1),
                                        lifted_stmt.dest,
                                        lifted_stmt.rhs.render()
                                    ));
                                }
                            };
                            emitted_any = true;
                        }
                        if emitted_any {
                            continue;
                        }
                    }

                    let dest_str = if ir_s.defs.is_empty() {
                        "_".to_string()
                    } else if ir_s.defs.len() == 1 {
                        ctx.expr(&ir_s.defs[0])
                    } else {
                        let defs = ir_s
                            .defs
                            .iter()
                            .map(|d| ctx.expr(d))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("({})", defs)
                    };
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
                    if ir_s.pred.is_some() && !ir_s.pred_old_defs.is_empty() && ir_s.defs.len() == 1 {
                        let old_str = ctx.expr(&ir_s.pred_old_defs[0]);
                        let pred_expr = ir_s.pred.as_ref().unwrap();
                        // Predicated instruction with old value: emit ternary select.
                        let phi_tag = if self.is_pred_old_phi(ir_s, 0) { "/*phi*/" } else { "" };
                        s_out.push_str(&format!(
                            "{}{} = {} ? ({}) : {}{};\n",
                            "  ".repeat(indent_level + 1),
                            dest_str,
                            ctx.expr(pred_expr),
                            value_str,
                            phi_tag,
                            old_str
                        ));
                    } else {
                        s_out.push_str(&format!("{}{}{} = {};\n", "  ".repeat(indent_level + 1), pred_str, dest_str, value_str));
                    }
                }
                if omitted_phi_count > 0 {
                    s_out.push_str(&format!(
                        "{}// {} phi node(s) omitted\n",
                        "  ".repeat(indent_level + 1),
                        omitted_phi_count
                    ));
                }
                s_out.push_str(&format!("{}}}\n", indent));
            }
            StructuredStatement::Sequence(stmts) => {
                for s_child in stmts {
                    if !matches!(s_child, StructuredStatement::Empty) { // Don't print empty
                        s_out.push_str(&self.pretty_print_with_lift(s_child, ctx, indent_level, lifted));
                    }
                }
            }
            StructuredStatement::If { condition_block_id, condition_expr, then_branch, else_branch } => {
                let then_empty = Self::is_effectively_empty(then_branch);
                let else_ref = else_branch.as_deref();
                let else_empty = else_ref.map(Self::is_effectively_empty).unwrap_or(true);

                if then_empty && else_empty {
                    // No-op condition: suppress noisy empty if blocks in output.
                } else if then_empty && !else_empty {
                    // Canonicalize "if (cond) {} else { X }" -> "if (!cond) { X }".
                    s_out.push_str(&self.render_condition_prelude_for_block(
                        *condition_block_id,
                        ctx,
                        indent_level,
                        lifted,
                    ));
                    s_out.push_str(&format!("{}// Condition from BB{}\n", indent, condition_block_id));
                    let cond_str = self.render_condition_expr(
                        *condition_block_id,
                        &negate_condition(condition_expr.clone()),
                        ctx,
                        lifted,
                    );
                    s_out.push_str(&format!("{}if ({}) {{\n", indent, cond_str));
                    s_out.push_str(&self.pretty_print_with_lift(else_ref.unwrap(), ctx, indent_level + 1, lifted));
                    s_out.push_str(&format!("{}}}\n", indent));
                } else {
                    s_out.push_str(&self.render_condition_prelude_for_block(
                        *condition_block_id,
                        ctx,
                        indent_level,
                        lifted,
                    ));
                    s_out.push_str(&format!("{}// Condition from BB{}\n", indent, condition_block_id));
                    let cond_str = self.render_condition_expr(
                        *condition_block_id,
                        condition_expr,
                        ctx,
                        lifted,
                    );
                    s_out.push_str(&format!("{}if ({}) {{\n", indent, cond_str));
                    s_out.push_str(&self.pretty_print_with_lift(then_branch, ctx, indent_level + 1, lifted));
                    if let Some(eb) = else_ref {
                        if !Self::is_effectively_empty(eb) {
                            s_out.push_str(&format!("{}}} else {{\n", indent));
                            s_out.push_str(&self.pretty_print_with_lift(eb, ctx, indent_level + 1, lifted));
                        }
                    }
                    s_out.push_str(&format!("{}}}\n", indent));
                }
            }
            StructuredStatement::Loop { loop_type, header_block_id, condition_expr, body } => {
                if let Some(hid) = header_block_id {
                    s_out.push_str(&self.render_condition_prelude_for_block(
                        *hid,
                        ctx,
                        indent_level,
                        lifted,
                    ));
                }
                s_out.push_str(&format!("{}// Loop header BB{:?}\n", indent, header_block_id.unwrap_or(usize::MAX)));
                match loop_type {
                    LoopType::While => {
                        let cond_str = condition_expr
                            .as_ref()
                            .map_or("true".to_string(), |e| {
                                if let Some(hid) = header_block_id {
                                    self.render_condition_expr(*hid, e, ctx, lifted)
                                } else {
                                    ctx.expr(e)
                                }
                            });
                        s_out.push_str(&format!("{}while ({}) {{\n", indent, cond_str));
                    }
                    LoopType::Endless => {
                         s_out.push_str(&format!("{}while (true) {{\n", indent));
                    }
                    LoopType::DoWhile => { 
                        s_out.push_str(&format!("{}do {{\n", indent));
                    }
                }
                let printable_body = Self::without_redundant_loop_tail_continue(body);
                s_out.push_str(&self.pretty_print_with_lift(&printable_body, ctx, indent_level + 1, lifted));
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
    use crate::ir::DefaultDisplay;
    use petgraph::graph::DiGraph;
    use std::collections::{HashMap, HashSet};

    fn stmt(opcode: &str) -> IRStatement {
        IRStatement {
            defs: vec![],
            value: RValue::Op { opcode: opcode.to_string(), args: vec![] },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: vec![],
        }
    }

    fn predicated_stmt(opcode: &str, pred: IRExpr) -> IRStatement {
        IRStatement {
            defs: vec![],
            value: RValue::Op { opcode: opcode.to_string(), args: vec![] },
            pred: Some(pred),
            mem_addr_args: None,
            pred_old_defs: vec![],
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
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], Some(id_to_idx[&3]), None)
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
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], None, None)
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
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], Some(id_to_idx[&3]), None)
            .unwrap();
        assert!(contains_goto(&out));
    }

    #[test]
    fn loop_body_recovers_if_inside_loop() {
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (0, 0x00, vec![], vec![stmt("HDR")]),
            (
                1,
                0x10,
                vec![
                    (Some(IRCond::Pred { reg: p0.clone(), sense: true }), 0x20),
                    (Some(IRCond::Pred { reg: p0, sense: false }), 0x30),
                ],
                vec![stmt("CMP")],
            ),
            (2, 0x20, vec![(Some(IRCond::True), 0x40)], vec![stmt("THEN")]),
            (3, 0x30, vec![(Some(IRCond::True), 0x40)], vec![stmt("ELSE")]),
            (4, 0x40, vec![(Some(IRCond::True), 0x00)], vec![stmt("LATCH")]),
        ];
        let edges = vec![
            (1, 2, EdgeKind::CondBranch),
            (1, 3, EdgeKind::FallThrough),
            (2, 4, EdgeKind::UncondBranch),
            (3, 4, EdgeKind::UncondBranch),
            (4, 0, EdgeKind::UncondBranch),
        ];
        let (cfg, fir, id_to_idx) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let loop_nodes: HashSet<NodeIndex> = [
            id_to_idx[&0],
            id_to_idx[&1],
            id_to_idx[&2],
            id_to_idx[&3],
            id_to_idx[&4],
        ]
        .into_iter()
        .collect();

        let out = structurizer
            .structure_loop_body_recursive(id_to_idx[&1], &loop_nodes, id_to_idx[&0], None, None)
            .unwrap();
        assert!(contains_if(&out));
        assert!(!contains_goto(&out));
    }

    #[test]
    fn predicated_exit_is_not_unconditional_return() {
        let block = IRBlock {
            id: 0,
            start_addr: 0,
            irdst: vec![],
            stmts: vec![IRStatement {
                defs: vec![],
                value: RValue::Op {
                    opcode: "EXIT".to_string(),
                    args: vec![],
                },
                pred: Some(IRExpr::Reg(RegId::new("P", 0, 1))),
                mem_addr_args: None,
                pred_old_defs: vec![],
            }],
        };
        assert!(!Structurizer::is_block_return(&block));
    }

    #[test]
    fn pretty_print_omits_raw_branch_ops() {
        let specs = vec![(0, 0x00, vec![], vec![stmt("IADD3"), stmt("BRA")])];
        let edges = vec![];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let structurizer = Structurizer::new(&cfg, &fir);
        let rendered = structurizer.pretty_print(
            &StructuredStatement::BasicBlock {
                block_id: 0,
                stmts: fir.blocks[0].stmts.clone(),
            },
            &DefaultDisplay,
            0,
        );
        assert!(rendered.contains("IADD3("));
        assert!(!rendered.contains("BRA("));
    }

    #[test]
    fn pretty_print_predicated_exit_as_return() {
        let specs = vec![
            (
                0,
                0x00,
                vec![],
                vec![predicated_stmt("EXIT", IRExpr::Reg(RegId::new("P", 0, 1)))],
            ),
        ];
        let edges = vec![];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let structurizer = Structurizer::new(&cfg, &fir);
        let rendered = structurizer.pretty_print(
            &StructuredStatement::BasicBlock {
                block_id: 0,
                stmts: fir.blocks[0].stmts.clone(),
            },
            &DefaultDisplay,
            0,
        );
        assert!(rendered.contains("if (P0) return;"));
        assert!(!rendered.contains("EXIT("));
    }

    #[test]
    fn pretty_print_omits_phi_statements_with_summary_comment() {
        let specs = vec![(
            0,
            0x00,
            vec![],
            vec![
                IRStatement {
                    defs: vec![IRExpr::Reg(RegId::new("R", 1, 1))],
                    value: RValue::Phi(vec![
                        IRExpr::Reg(RegId::new("R", 2, 1)),
                        IRExpr::Reg(RegId::new("R", 3, 1)),
                    ]),
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: vec![],
                },
                stmt("IADD3"),
            ],
        )];
        let edges = vec![];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let structurizer = Structurizer::new(&cfg, &fir);
        let rendered = structurizer.pretty_print(
            &StructuredStatement::BasicBlock {
                block_id: 0,
                stmts: fir.blocks[0].stmts.clone(),
            },
            &DefaultDisplay,
            0,
        );
        assert!(!rendered.contains("phi("));
        assert!(rendered.contains("phi node(s) omitted"));
        assert!(rendered.contains("IADD3("));
    }

    #[test]
    fn pretty_print_omits_redundant_loop_tail_continue() {
        let specs = vec![(0, 0x00, vec![], vec![stmt("BODY")])];
        let edges = vec![];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let structurizer = Structurizer::new(&cfg, &fir);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::While,
            header_block_id: Some(0),
            condition_expr: Some(IRExpr::Reg(RegId::new("P", 0, 1))),
            body: Box::new(StructuredStatement::Sequence(vec![
                StructuredStatement::BasicBlock {
                    block_id: 0,
                    stmts: fir.blocks[0].stmts.clone(),
                },
                StructuredStatement::Continue(None),
            ])),
        };
        let rendered = structurizer.pretty_print(&loop_stmt, &DefaultDisplay, 0);
        assert!(rendered.contains("while (P0)"));
        assert!(rendered.contains("BODY("));
        assert!(!rendered.contains("continue;"));
    }
}
