//! Structured AST lowering helpers for the rewritten backend.
//!
//! Purpose:
//! - lower SSA-backed memory operations into the canonical AST using
//!   `FunctionAnalysis` facts rather than semantic-lift side tables or
//!   rendered-text repair
//!
//! Inputs:
//! - `IRStatement`
//! - statement location
//! - `FunctionAnalysis`
//!
//! Outputs:
//! - AST expressions/lvalues/statements for memory operations
//!
//! Invariants:
//! - lowering consumes analysis facts directly
//! - lowering is deterministic
//! - no rendered-text parsing is allowed
//!
//! This module must not:
//! - inspect previously rendered code
//! - repair names or declarations after rendering

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::abi::ConstMemSemantic;
use crate::ast::{Expr, LValue, PointerLane, Stmt};
use crate::backend_names::canonical_reg_ident;
use crate::canonical_ast_passes::canonicalize_function;
use crate::cfg::ControlFlowGraph;
use crate::function_analysis::{AddressRoot, FunctionAnalysis, MemAccessInfo};
use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue};
use crate::memory_model::{CudaMemorySpace, MemAccessKind};
use crate::structurizer::{LoopType, StructuredStatement};
use crate::symbol_plan::plan_symbols;
use crate::type_inference::InferredType;
use petgraph::graph::NodeIndex;
use petgraph::Direction;

#[derive(Clone, Debug)]
struct LoweredStmt {
    stmt: Stmt,
    predicate_def_indices: Vec<usize>,
}

struct StructuredLowering<'a> {
    cfg: &'a ControlFlowGraph,
    function_ir: &'a FunctionIR,
    analysis: &'a FunctionAnalysis,
    block_id_to_idx: HashMap<usize, usize>,
    block_id_to_node: HashMap<usize, NodeIndex>,
}

pub fn lower_memory_stmt(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Option<Stmt> {
    lower_memory_stmt_detail(block_id, stmt_idx, stmt, analysis).map(|lowered| lowered.stmt)
}

fn lower_memory_stmt_detail(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Option<LoweredStmt> {
    let access = lookup_mem_access(analysis, block_id, stmt_idx)?;
    if access.space == CudaMemorySpace::Local && local_space_requires_explicit_ops(analysis) {
        return lower_explicit_local_stmt(stmt, access);
    }
    match access.kind {
        MemAccessKind::Load => {
            let dst = stmt.defs.first()?.get_reg()?;
            let src = lower_memory_load_expr(stmt, access, analysis)?;
            Some(LoweredStmt {
                stmt: Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(dst)),
                    src,
                },
                predicate_def_indices: vec![0],
            })
        }
        MemAccessKind::Store => {
            let dst = lower_memory_store_lvalue(stmt, access, analysis)?;
            let src = store_value_expr(stmt, Some(analysis))?;
            Some(LoweredStmt {
                stmt: Stmt::Assign { dst, src },
                predicate_def_indices: Vec::new(),
            })
        }
        MemAccessKind::Atomic | MemAccessKind::Reduction => {
            let opcode = stmt_opcode(stmt);
            let func = atomic_func_name(opcode);
            let Some((base, index)) = lower_base_and_index(stmt, access, analysis) else {
                return None;
            };
            let mut args = vec![Expr::Unary {
                op: "&".to_string(),
                arg: Box::new(Expr::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                }),
            }];
            args.extend(
                stmt_args(stmt)
                    .iter()
                    .filter(|expr| !matches!(expr, IRExpr::Mem { .. }))
                    .map(|expr| lower_scalar_expr_with_analysis(expr, Some(analysis))),
            );
            if opcode.contains(".POPC.INC") && args.len() == 1 {
                args.push(Expr::Imm("1".to_string()));
            }
            let call = Expr::CallLike {
                func: func.to_string(),
                args,
            };
            if let Some((def_idx, def)) = select_memory_result_def(stmt) {
                Some(LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src: call,
                    },
                    predicate_def_indices: vec![def_idx],
                })
            } else {
                Some(LoweredStmt {
                    stmt: Stmt::ExprStmt(call),
                    predicate_def_indices: Vec::new(),
                })
            }
        }
    }
}

impl<'a> StructuredLowering<'a> {
    fn new(
        cfg: &'a ControlFlowGraph,
        function_ir: &'a FunctionIR,
        analysis: &'a FunctionAnalysis,
    ) -> Self {
        let block_id_to_idx = function_ir
            .blocks
            .iter()
            .enumerate()
            .map(|(idx, block)| (block.id, idx))
            .collect();
        let block_id_to_node = cfg
            .node_indices()
            .map(|node| (cfg[node].id, node))
            .collect();
        Self {
            cfg,
            function_ir,
            analysis,
            block_id_to_idx,
            block_id_to_node,
        }
    }

    fn lower_function(&self, structured: &StructuredStatement) -> crate::ast::StructuredFunction {
        let (jump_targets, materialized_blocks) = self.plan_render_jump_targets(structured);
        let body =
            self.lower_structured_stmt_with_targets(structured, &jump_targets, &materialized_blocks, None);
        let seed = canonicalize_function(crate::ast::StructuredFunction {
            params: Vec::new(),
            locals: Vec::new(),
            body,
        });
        let seed = crate::ast::StructuredFunction {
            params: seed.params,
            locals: seed.locals,
            body: self.bind_missing_goto_labels(seed.body),
        };
        let plan = plan_symbols(&seed, self.analysis);
        crate::ast::StructuredFunction {
            params: plan.params,
            locals: plan.locals,
            body: seed.body,
        }
    }

    fn lower_structured_stmt(&self, structured: &StructuredStatement) -> Stmt {
        let (jump_targets, materialized_blocks) = self.plan_render_jump_targets(structured);
        self.lower_structured_stmt_with_targets(
            structured,
            &jump_targets,
            &materialized_blocks,
            None,
        )
    }

    fn plan_render_jump_targets(
        &self,
        structured: &StructuredStatement,
    ) -> (HashSet<usize>, HashSet<usize>) {
        let mut materialized_blocks = HashSet::new();
        Self::collect_structured_block_ids(structured, &mut materialized_blocks);

        let mut jump_targets = HashSet::new();
        self.collect_render_jump_targets(structured, &materialized_blocks, &mut jump_targets);

        (jump_targets, materialized_blocks)
    }

    fn lower_structured_stmt_with_targets(
        &self,
        structured: &StructuredStatement,
        jump_targets: &HashSet<usize>,
        materialized_blocks: &HashSet<usize>,
        fallthrough_target_block: Option<usize>,
    ) -> Stmt {
        match structured {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                let body = self.lower_stmt_list(*block_id, stmts, true, false);
                let body = if let Some(target_block_id) = fallthrough_target_block {
                    Self::stmt_sequence(vec![
                        body,
                        self.lower_phi_connector_chain_from_pred(target_block_id, *block_id),
                    ])
                } else {
                    body
                };
                if jump_targets.contains(block_id) {
                    Stmt::Label {
                        name: format!("BB{}", block_id),
                        body: Box::new(body),
                    }
                } else {
                    body
                }
            }
            StructuredStatement::Sequence(parts) => {
                let mut lowered = Vec::new();
                for (idx, part) in parts.iter().enumerate() {
                    if matches!(part, StructuredStatement::Empty) {
                        continue;
                    }
                    let next_target = parts[idx + 1..]
                        .iter()
                        .find_map(Self::entry_block_id)
                        .or(fallthrough_target_block);
                    lowered.push(self.lower_structured_stmt_with_targets(
                        part,
                        jump_targets,
                        materialized_blocks,
                        next_target,
                    ));
                    if Self::ends_with_unconditional_return(part) {
                        break;
                    }
                }
                Self::stmt_sequence(lowered)
            }
            StructuredStatement::If {
                condition_block_id,
                condition_expr,
                then_branch,
                else_branch,
            } => {
                let prelude = self.lower_condition_prelude(*condition_block_id);
                let then_stmt = self.lower_structured_stmt_with_targets(
                    then_branch,
                    jump_targets,
                    materialized_blocks,
                    fallthrough_target_block,
                );
                let then_phi = Self::entry_block_id(then_branch)
                    .map(|entry_block_id| {
                        self.lower_phi_connector_chain_from_pred(
                            entry_block_id,
                            *condition_block_id,
                        )
                    })
                    .filter(|stmt| !Self::stmt_is_empty(stmt));
                let then_stmt = if let Some(then_phi) = then_phi {
                    Self::prepend_entry_prelude(then_stmt, then_phi)
                } else {
                    then_stmt
                };
                let else_stmt = else_branch
                    .as_ref()
                    .map(|branch| {
                        let lowered = self.lower_structured_stmt_with_targets(
                            branch,
                            jump_targets,
                            materialized_blocks,
                            fallthrough_target_block,
                        );
                        let else_phi = Self::entry_block_id(branch)
                            .map(|entry_block_id| {
                                self.lower_phi_connector_chain_from_pred(
                                    entry_block_id,
                                    *condition_block_id,
                                )
                            })
                            .filter(|stmt| !Self::stmt_is_empty(stmt));
                        if let Some(else_phi) = else_phi {
                            Self::prepend_entry_prelude(lowered, else_phi)
                        } else {
                            lowered
                        }
                    })
                    .or_else(|| {
                        fallthrough_target_block.map(|target_block_id| {
                            self.lower_phi_connector_chain_from_pred(
                                target_block_id,
                                *condition_block_id,
                            )
                        })
                    })
                    .filter(|stmt| !Self::stmt_is_empty(stmt));
                let then_empty = Self::stmt_is_empty(&then_stmt);
                let else_empty = else_stmt.as_ref().map(Self::stmt_is_empty).unwrap_or(true);
                let condition =
                    lower_scalar_expr_with_analysis(condition_expr, Some(self.analysis));
                let core = if then_empty && else_empty {
                    Stmt::Empty
                } else if then_empty && !else_empty {
                    Stmt::If {
                        condition: negate_expr(condition),
                        then_branch: Box::new(else_stmt.expect("else branch")),
                        else_branch: None,
                    }
                } else {
                    Stmt::If {
                        condition,
                        then_branch: Box::new(then_stmt),
                        else_branch: else_stmt.map(Box::new),
                    }
                };
                let lowered = Self::stmt_sequence(vec![prelude, core]);
                if jump_targets.contains(condition_block_id) {
                    Stmt::Label {
                        name: format!("BB{}", condition_block_id),
                        body: Box::new(lowered),
                    }
                } else {
                    lowered
                }
            }
            StructuredStatement::Loop {
                loop_type,
                header_block_id,
                condition_expr,
                body,
            } => {
                let printable_body = Self::without_redundant_loop_tail_continue(body);
                let phi_entry_block_id = match (loop_type, header_block_id) {
                    (LoopType::DoWhile, Some(header_block_id)) => {
                        let body_entry = Self::entry_block_id(&printable_body);
                        match body_entry {
                            Some(body_entry) if body_entry != *header_block_id => Some(body_entry),
                            _ => Some(*header_block_id),
                        }
                    }
                    (_, Some(header_block_id)) => Some(*header_block_id),
                    _ => None,
                };
                let phi_prelude = phi_entry_block_id
                    .map(|block_id| self.lower_loop_phi_prelude(block_id, &printable_body))
                    .unwrap_or(Stmt::Empty);
                let phi_backedge = header_block_id
                    .map(|block_id| self.lower_loop_phi_backedge_updates(block_id, &printable_body))
                    .unwrap_or(Stmt::Empty);
                let body_entry_backedge = match (loop_type, header_block_id) {
                    (LoopType::DoWhile, Some(header_block_id)) => {
                        let body_entry = Self::entry_block_id(&printable_body);
                        match body_entry {
                            Some(body_entry) if body_entry != *header_block_id => self
                                .lower_phi_connector_chain_from_pred(body_entry, *header_block_id),
                            _ => Stmt::Empty,
                        }
                    }
                    _ => Stmt::Empty,
                };
                let loop_backedge = Self::stmt_sequence(vec![phi_backedge, body_entry_backedge]);
                let condition_prelude = if *loop_type != LoopType::DoWhile {
                    header_block_id
                        .map(|block_id| self.lower_condition_prelude(block_id))
                        .unwrap_or(Stmt::Empty)
                } else {
                    Stmt::Empty
                };
                let condition = condition_expr
                    .as_ref()
                    .map(|expr| lower_scalar_expr_with_analysis(expr, Some(self.analysis)));
                let loop_stmt = Stmt::Loop {
                    kind: match loop_type {
                        LoopType::While => crate::ast::LoopKind::While,
                        LoopType::DoWhile => crate::ast::LoopKind::DoWhile,
                        LoopType::Endless => crate::ast::LoopKind::Endless,
                    },
                    condition,
                    body: Box::new(Self::inject_loop_phi_backedge_updates(
                        self.lower_structured_stmt_with_targets(
                            &printable_body,
                            jump_targets,
                            materialized_blocks,
                            None,
                        ),
                        &loop_backedge,
                    )),
                };
                let loop_exit_phi = match (fallthrough_target_block, header_block_id, loop_type) {
                    (
                        Some(target_block_id),
                        Some(header_block_id),
                        LoopType::While | LoopType::DoWhile,
                    ) => {
                        self.lower_phi_connector_chain_from_pred(target_block_id, *header_block_id)
                    }
                    _ => Stmt::Empty,
                };
                let lowered = Self::stmt_sequence(vec![
                    phi_prelude,
                    condition_prelude,
                    loop_stmt,
                    loop_exit_phi,
                ]);
                if let Some(header_block_id) = header_block_id {
                    if jump_targets.contains(header_block_id) {
                        Stmt::Label {
                            name: format!("BB{}", header_block_id),
                            body: Box::new(lowered),
                        }
                    } else {
                        lowered
                    }
                } else {
                    lowered
                }
            }
            StructuredStatement::Break(_) => Stmt::Break,
            StructuredStatement::Continue(_) => Stmt::Continue,
            StructuredStatement::Return(expr) => Stmt::Return(
                expr.as_ref()
                    .map(|expr| lower_scalar_expr_with_analysis(expr, Some(self.analysis))),
            ),
            StructuredStatement::UnstructuredJump {
                from_block_id,
                to_block_id,
                condition,
            } => {
                let phi_prelude =
                    self.lower_phi_connector_chain_from_pred(
                        self.resolve_render_jump_target(*to_block_id, materialized_blocks)
                            .unwrap_or(*to_block_id),
                        *from_block_id,
                    );
                let render_target = self
                    .resolve_render_jump_target(*to_block_id, materialized_blocks)
                    .unwrap_or(*to_block_id);
                let jump = Self::stmt_sequence(vec![
                    phi_prelude,
                    Stmt::Goto(format!("BB{}", render_target)),
                ]);
                match condition {
                    Some(condition) => Stmt::If {
                        condition: lower_scalar_expr_with_analysis(condition, Some(self.analysis)),
                        then_branch: Box::new(jump),
                        else_branch: None,
                    },
                    None => jump,
                }
            }
            StructuredStatement::Switch {
                header_block_id,
                discriminant,
                cases,
                default,
            } => {
                let prelude = self.lower_switch_prelude(*header_block_id);
                let switch_stmt = Stmt::Switch {
                    discriminant: discriminant
                        .as_ref()
                        .map(|expr| lower_scalar_expr_with_analysis(expr, Some(self.analysis))),
                    cases: cases
                        .iter()
                        .map(|(label, body)| {
                            (
                                *label,
                                self.lower_structured_stmt_with_targets(
                                    body,
                                    jump_targets,
                                    materialized_blocks,
                                    fallthrough_target_block,
                                ),
                            )
                        })
                        .collect(),
                    default: default.as_ref().map(|body| {
                        Box::new(self.lower_structured_stmt_with_targets(
                            body,
                            jump_targets,
                            materialized_blocks,
                            fallthrough_target_block,
                        ))
                    }),
                };
                let lowered = Self::stmt_sequence(vec![prelude, switch_stmt]);
                if jump_targets.contains(header_block_id) {
                    Stmt::Label {
                        name: format!("BB{}", header_block_id),
                        body: Box::new(lowered),
                    }
                } else {
                    lowered
                }
            }
            StructuredStatement::Empty => Stmt::Empty,
        }
    }

    fn lower_stmt_list(
        &self,
        block_id: usize,
        stmts: &[IRStatement],
        emit_returns: bool,
        skip_brx: bool,
    ) -> Stmt {
        let fir_block = self.get_ir_block(block_id);
        let mut fir_search_from = 0usize;
        let mut lowered = Vec::new();

        for (stmt_idx, stmt) in stmts.iter().enumerate() {
            let lookup_stmt_idx =
                Self::resolve_stmt_lookup_idx(fir_block, &mut fir_search_from, stmt_idx, stmt);
            if matches!(stmt.value, RValue::Phi(_)) {
                continue;
            }
            if let RValue::Op { opcode, .. } = &stmt.value {
                if Self::is_control_jump_opcode(opcode)
                    || (skip_brx && opcode.split('.').next().unwrap_or(opcode) == "BRX")
                    || self.is_cfg_modeled_call_opcode(block_id, stmt_idx, stmts, opcode)
                    || Self::is_convergence_barrier_opcode(opcode)
                    || opcode == "NOP"
                {
                    continue;
                }
                if Self::is_return_opcode(opcode) && !emit_returns {
                    continue;
                }
            }
            lowered.push(lower_basic_stmt(
                block_id,
                lookup_stmt_idx,
                stmt,
                self.analysis,
            ));
            if matches!(
                &stmt.value,
                RValue::Op { opcode, .. } if Self::is_return_opcode(opcode) && stmt.pred.is_none()
            ) {
                break;
            }
        }

        Self::stmt_sequence(lowered)
    }

    fn lower_condition_prelude(&self, block_id: usize) -> Stmt {
        let Some(block) = self.get_ir_block(block_id) else {
            return Stmt::Empty;
        };
        match self.lower_stmt_list(block_id, &block.stmts, false, false) {
            Stmt::Empty => Stmt::Empty,
            Stmt::Sequence(stmts) => Stmt::Block(stmts),
            stmt => Stmt::Block(vec![stmt]),
        }
    }

    fn lower_switch_prelude(&self, block_id: usize) -> Stmt {
        let Some(block) = self.get_ir_block(block_id) else {
            return Stmt::Empty;
        };
        match self.lower_stmt_list(block_id, &block.stmts, false, true) {
            Stmt::Empty => Stmt::Empty,
            Stmt::Sequence(stmts) => Stmt::Block(stmts),
            stmt => Stmt::Block(vec![stmt]),
        }
    }

    fn lower_loop_phi_prelude(&self, header_block_id: usize, body: &StructuredStatement) -> Stmt {
        self.lower_phi_assignments_for_loop_preds(header_block_id, body, |preds, loop_blocks| {
            let external = preds
                .iter()
                .copied()
                .filter(|pred| !loop_blocks.contains(&self.cfg[*pred].id))
                .collect::<Vec<_>>();
            (external.len() == 1).then_some(external)
        })
    }

    fn lower_loop_phi_backedge_updates(
        &self,
        header_block_id: usize,
        body: &StructuredStatement,
    ) -> Stmt {
        self.lower_phi_assignments_for_loop_preds(header_block_id, body, |preds, loop_blocks| {
            let internal = preds
                .iter()
                .copied()
                .filter(|pred| loop_blocks.contains(&self.cfg[*pred].id))
                .collect::<Vec<_>>();
            (!internal.is_empty()).then_some(internal)
        })
    }

    fn lower_phi_assignments_for_loop_preds<F>(
        &self,
        header_block_id: usize,
        body: &StructuredStatement,
        pick_preds: F,
    ) -> Stmt
    where
        F: Fn(&[NodeIndex], &HashSet<usize>) -> Option<Vec<NodeIndex>>,
    {
        let Some(header_node) = self.cfg_node_for_block_id(header_block_id) else {
            return Stmt::Empty;
        };
        let Some(header_block) = self.get_ir_block(header_block_id) else {
            return Stmt::Empty;
        };

        let mut loop_blocks = HashSet::new();
        loop_blocks.insert(header_block_id);
        Self::collect_structured_block_ids(body, &mut loop_blocks);

        let preds: Vec<_> = self
            .cfg
            .neighbors_directed(header_node, Direction::Incoming)
            .collect();
        let Some(selected_preds) = pick_preds(&preds, &loop_blocks) else {
            return Stmt::Empty;
        };

        let mut lowered = Vec::new();
        for stmt in header_block
            .stmts
            .iter()
            .filter(|stmt| matches!(stmt.value, RValue::Phi(_)))
        {
            let Some((_, def)) = select_non_memory_result_def(stmt) else {
                continue;
            };
            let RValue::Phi(args) = &stmt.value else {
                continue;
            };
            let Some(src_expr) = self.resolve_phi_source_for_preds(args, &preds, &selected_preds)
            else {
                continue;
            };
            let assign = Stmt::Assign {
                dst: LValue::Var(lower_reg_name(def)),
                src: lower_scalar_expr_with_analysis(&src_expr, Some(self.analysis)),
            };
            if !Self::stmt_is_trivial_self_assign(&assign) {
                lowered.push(assign);
            }
        }
        Self::stmt_sequence(lowered)
    }

    fn resolve_phi_source_for_preds(
        &self,
        args: &[IRExpr],
        preds: &[NodeIndex],
        selected_preds: &[NodeIndex],
    ) -> Option<IRExpr> {
        let mut chosen = None;
        for pred in selected_preds {
            let pred_idx = preds.iter().position(|candidate| candidate == pred)?;
            let arg = args.get(pred_idx)?.clone();
            match &chosen {
                None => chosen = Some(arg),
                Some(existing) if existing == &arg => {}
                Some(_) => return None,
            }
        }
        chosen
    }

    fn lower_block_phi_assignments_from_pred(
        &self,
        target_block_id: usize,
        from_block_id: usize,
    ) -> Stmt {
        let Some(target_node) = self.cfg_node_for_block_id(target_block_id) else {
            return Stmt::Empty;
        };
        let Some(target_block) = self.get_ir_block(target_block_id) else {
            return Stmt::Empty;
        };
        let preds: Vec<_> = self
            .cfg
            .neighbors_directed(target_node, Direction::Incoming)
            .collect();
        let Some(selected_pred) = preds
            .iter()
            .copied()
            .find(|pred| self.cfg[*pred].id == from_block_id)
        else {
            return Stmt::Empty;
        };

        let mut lowered = Vec::new();
        for stmt in target_block
            .stmts
            .iter()
            .filter(|stmt| matches!(stmt.value, RValue::Phi(_)))
        {
            let Some((_, def)) = select_non_memory_result_def(stmt) else {
                continue;
            };
            let RValue::Phi(args) = &stmt.value else {
                continue;
            };
            let Some(src_expr) = self.resolve_phi_source_for_preds(args, &preds, &[selected_pred])
            else {
                continue;
            };
            let assign = Stmt::Assign {
                dst: LValue::Var(lower_reg_name(def)),
                src: lower_scalar_expr_with_analysis(&src_expr, Some(self.analysis)),
            };
            if !Self::stmt_is_trivial_self_assign(&assign) {
                lowered.push(assign);
            }
        }
        Self::stmt_sequence(lowered)
    }

    fn lower_phi_connector_chain_from_pred(
        &self,
        target_block_id: usize,
        from_block_id: usize,
    ) -> Stmt {
        let mut lowered = Vec::new();
        let mut current_from = from_block_id;
        let mut seen = HashSet::new();

        for _ in 0..16 {
            if current_from == target_block_id {
                break;
            }

            if self.is_direct_cfg_successor(current_from, target_block_id) {
                let direct =
                    self.lower_block_phi_assignments_from_pred(target_block_id, current_from);
                if !Self::stmt_is_empty(&direct) {
                    lowered.push(direct);
                }
                break;
            }

            let connectors = self
                .cfg_successor_block_ids(current_from)
                .into_iter()
                .filter(|succ| self.block_is_phi_connector(*succ))
                .filter(|succ| self.cfg_path_exists(*succ, target_block_id))
                .collect::<Vec<_>>();
            if connectors.len() != 1 {
                break;
            }

            let connector = connectors[0];
            if !seen.insert(connector) {
                break;
            }

            let assigns = self.lower_block_phi_assignments_from_pred(connector, current_from);
            if !Self::stmt_is_empty(&assigns) {
                lowered.push(assigns);
            }
            current_from = connector;
        }

        Self::stmt_sequence(lowered)
    }

    fn get_ir_block(&self, block_id: usize) -> Option<&'a IRBlock> {
        let idx = *self.block_id_to_idx.get(&block_id)?;
        Some(&self.function_ir.blocks[idx])
    }

    fn cfg_node_for_block_id(&self, block_id: usize) -> Option<NodeIndex> {
        self.block_id_to_node.get(&block_id).copied()
    }

    fn cfg_successor_block_ids(&self, block_id: usize) -> Vec<usize> {
        let Some(node) = self.cfg_node_for_block_id(block_id) else {
            return Vec::new();
        };
        self.cfg
            .neighbors_directed(node, Direction::Outgoing)
            .map(|succ| self.cfg[succ].id)
            .collect()
    }

    fn is_direct_cfg_successor(&self, from_block_id: usize, to_block_id: usize) -> bool {
        self.cfg_successor_block_ids(from_block_id)
            .into_iter()
            .any(|succ| succ == to_block_id)
    }

    fn block_is_phi_connector(&self, block_id: usize) -> bool {
        let Some(block) = self.get_ir_block(block_id) else {
            return false;
        };
        block.stmts.is_empty()
            || block
                .stmts
                .iter()
                .all(|stmt| matches!(stmt.value, RValue::Phi(_)))
    }

    fn cfg_path_exists(&self, start_block_id: usize, goal_block_id: usize) -> bool {
        let Some(start) = self.cfg_node_for_block_id(start_block_id) else {
            return false;
        };
        let Some(goal) = self.cfg_node_for_block_id(goal_block_id) else {
            return false;
        };
        if start == goal {
            return true;
        }
        let mut seen = HashSet::new();
        let mut work = VecDeque::new();
        seen.insert(start);
        work.push_back(start);
        while let Some(node) = work.pop_front() {
            for succ in self.cfg.neighbors_directed(node, Direction::Outgoing) {
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

    fn resolve_stmt_lookup_idx(
        fir_block: Option<&IRBlock>,
        fir_search_from: &mut usize,
        fallback_idx: usize,
        stmt: &IRStatement,
    ) -> usize {
        let Some(orig_block) = fir_block else {
            return fallback_idx;
        };
        if let Some((found_idx, _)) = orig_block
            .stmts
            .iter()
            .enumerate()
            .skip(*fir_search_from)
            .find(|(_, candidate)| {
                candidate.defs == stmt.defs
                    && candidate.value == stmt.value
                    && candidate.pred == stmt.pred
            })
        {
            *fir_search_from = found_idx + 1;
            found_idx
        } else {
            fallback_idx
        }
    }

    fn is_control_jump_opcode(opcode: &str) -> bool {
        matches!(opcode, "BRA" | "JMP" | "JMPP")
    }

    fn is_cfg_modeled_call_opcode(
        &self,
        block_id: usize,
        stmt_idx: usize,
        stmts: &[IRStatement],
        opcode: &str,
    ) -> bool {
        if opcode.split('.').next().unwrap_or(opcode) != "CALL" || stmt_idx + 1 != stmts.len() {
            return false;
        }
        let Some(node) = self.cfg_node_for_block_id(block_id) else {
            return false;
        };
        self.cfg
            .edges_directed(node, Direction::Outgoing)
            .any(|edge| !matches!(edge.weight(), crate::cfg::EdgeKind::FallThrough))
    }

    fn is_convergence_barrier_opcode(opcode: &str) -> bool {
        matches!(
            opcode.split('.').next().unwrap_or(opcode),
            "BSSY" | "BSYNC" | "SSY" | "SYNC" | "WARPSYNC"
        )
    }

    fn is_return_opcode(opcode: &str) -> bool {
        opcode == "RET" || opcode == "EXIT" || opcode.starts_with("RET")
    }

    fn stmt_sequence(stmts: Vec<Stmt>) -> Stmt {
        let mut flat = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::Empty => {}
                Stmt::Sequence(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => Stmt::Empty,
            1 => flat.into_iter().next().expect("single stmt"),
            _ => Stmt::Sequence(flat),
        }
    }

    fn stmt_is_empty(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Empty => true,
            Stmt::Block(stmts) | Stmt::Sequence(stmts) => stmts.iter().all(Self::stmt_is_empty),
            _ => false,
        }
    }

    // Branch-entry phi materialization must happen inside the entry label when
    // the structured region is still a jump target; otherwise a surviving goto
    // would jump past the phi prelude and read stale SSA values.
    fn prepend_entry_prelude(stmt: Stmt, prelude: Stmt) -> Stmt {
        if Self::stmt_is_empty(&prelude) {
            return stmt;
        }
        match stmt {
            Stmt::Label { name, body } => Stmt::Label {
                name,
                body: Box::new(Self::prepend_entry_prelude(*body, prelude)),
            },
            Stmt::Sequence(mut stmts) => {
                if let Some(idx) = stmts.iter().position(|stmt| !Self::stmt_is_empty(stmt)) {
                    let head = std::mem::replace(&mut stmts[idx], Stmt::Empty);
                    stmts[idx] = Self::prepend_entry_prelude(head, prelude);
                    Self::stmt_sequence(stmts)
                } else {
                    prelude
                }
            }
            Stmt::Block(mut stmts) => {
                if let Some(idx) = stmts.iter().position(|stmt| !Self::stmt_is_empty(stmt)) {
                    let head = std::mem::replace(&mut stmts[idx], Stmt::Empty);
                    stmts[idx] = Self::prepend_entry_prelude(head, prelude);
                    Stmt::Block(stmts)
                } else {
                    prelude
                }
            }
            other => Self::stmt_sequence(vec![prelude, other]),
        }
    }

    fn stmt_is_trivial_self_assign(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Assign {
                dst: LValue::Var(dst),
                src: Expr::Reg(src),
            }
            | Stmt::Assign {
                dst: LValue::Var(dst),
                src: Expr::Raw(src),
            }
            | Stmt::Assign {
                dst: LValue::Raw(dst),
                src: Expr::Reg(src),
            }
            | Stmt::Assign {
                dst: LValue::Raw(dst),
                src: Expr::Raw(src),
            } => dst == src,
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

    fn stmt_contains_continue(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Continue => true,
            Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
                stmts.iter().any(Self::stmt_contains_continue)
            }
            Stmt::Label { body, .. } => Self::stmt_contains_continue(body),
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                Self::stmt_contains_continue(then_branch)
                    || else_branch
                        .as_deref()
                        .map(Self::stmt_contains_continue)
                        .unwrap_or(false)
            }
            Stmt::Switch { cases, default, .. } => {
                cases
                    .iter()
                    .any(|(_, body)| Self::stmt_contains_continue(body))
                    || default
                        .as_deref()
                        .map(Self::stmt_contains_continue)
                        .unwrap_or(false)
            }
            Stmt::Loop { .. }
            | Stmt::Break
            | Stmt::Return(_)
            | Stmt::Assign { .. }
            | Stmt::ExprStmt(_)
            | Stmt::Goto(_)
            | Stmt::Empty => false,
        }
    }

    fn stmt_may_fallthrough(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Empty | Stmt::Assign { .. } | Stmt::ExprStmt(_) | Stmt::Loop { .. } => true,
            Stmt::Break | Stmt::Continue | Stmt::Return(_) | Stmt::Goto(_) => false,
            Stmt::Block(stmts) | Stmt::Sequence(stmts) => {
                let mut reachable = true;
                for stmt in stmts {
                    if !reachable {
                        return false;
                    }
                    reachable = Self::stmt_may_fallthrough(stmt);
                }
                reachable
            }
            Stmt::Label { body, .. } => Self::stmt_may_fallthrough(body),
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                let then_fallthrough = Self::stmt_may_fallthrough(then_branch);
                match else_branch {
                    Some(branch) => then_fallthrough || Self::stmt_may_fallthrough(branch),
                    None => true,
                }
            }
            Stmt::Switch { cases, default, .. } => {
                if default.is_none() {
                    return true;
                }
                cases
                    .iter()
                    .any(|(_, body)| Self::stmt_may_fallthrough(body))
                    || default
                        .as_deref()
                        .map(Self::stmt_may_fallthrough)
                        .unwrap_or(false)
            }
        }
    }

    fn inject_loop_phi_updates_before_continue(stmt: Stmt, updates: &Stmt) -> Stmt {
        match stmt {
            Stmt::Continue => Self::stmt_sequence(vec![updates.clone(), Stmt::Continue]),
            Stmt::Block(stmts) => Stmt::Block(
                stmts
                    .into_iter()
                    .map(|stmt| Self::inject_loop_phi_updates_before_continue(stmt, updates))
                    .collect(),
            ),
            Stmt::Sequence(stmts) => Self::stmt_sequence(
                stmts
                    .into_iter()
                    .map(|stmt| Self::inject_loop_phi_updates_before_continue(stmt, updates))
                    .collect(),
            ),
            Stmt::Label { name, body } => Stmt::Label {
                name,
                body: Box::new(Self::inject_loop_phi_updates_before_continue(
                    *body, updates,
                )),
            },
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => Stmt::If {
                condition,
                then_branch: Box::new(Self::inject_loop_phi_updates_before_continue(
                    *then_branch,
                    updates,
                )),
                else_branch: else_branch.map(|branch| {
                    Box::new(Self::inject_loop_phi_updates_before_continue(
                        *branch, updates,
                    ))
                }),
            },
            Stmt::Switch {
                discriminant,
                cases,
                default,
            } => Stmt::Switch {
                discriminant,
                cases: cases
                    .into_iter()
                    .map(|(label, body)| {
                        (
                            label,
                            Self::inject_loop_phi_updates_before_continue(body, updates),
                        )
                    })
                    .collect(),
                default: default.map(|body| {
                    Box::new(Self::inject_loop_phi_updates_before_continue(
                        *body, updates,
                    ))
                }),
            },
            Stmt::Loop {
                kind,
                condition,
                body,
            } => Stmt::Loop {
                kind,
                condition,
                body: Box::new(Self::inject_loop_phi_updates_before_continue(
                    *body, updates,
                )),
            },
            other => other,
        }
    }

    fn inject_loop_phi_backedge_updates(body: Stmt, updates: &Stmt) -> Stmt {
        if Self::stmt_is_empty(updates) || Self::stmt_contains_continue(&body) {
            return body;
        }
        let body = Self::inject_loop_phi_updates_before_continue(body, updates);
        if Self::stmt_may_fallthrough(&body) {
            Self::stmt_sequence(vec![body, updates.clone()])
        } else {
            body
        }
    }

    fn collect_render_jump_targets(
        &self,
        stmt: &StructuredStatement,
        materialized_blocks: &HashSet<usize>,
        targets: &mut HashSet<usize>,
    ) {
        match stmt {
            StructuredStatement::Sequence(stmts) => {
                for stmt in stmts {
                    self.collect_render_jump_targets(stmt, materialized_blocks, targets);
                }
            }
            StructuredStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_render_jump_targets(then_branch, materialized_blocks, targets);
                if let Some(else_branch) = else_branch {
                    self.collect_render_jump_targets(else_branch, materialized_blocks, targets);
                }
            }
            StructuredStatement::Loop { body, .. } => {
                self.collect_render_jump_targets(body, materialized_blocks, targets);
            }
            StructuredStatement::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    self.collect_render_jump_targets(body, materialized_blocks, targets);
                }
                if let Some(default) = default {
                    self.collect_render_jump_targets(default, materialized_blocks, targets);
                }
            }
            StructuredStatement::UnstructuredJump { to_block_id, .. } => {
                targets.insert(
                    self.resolve_render_jump_target(*to_block_id, materialized_blocks)
                        .unwrap_or(*to_block_id),
                );
            }
            StructuredStatement::BasicBlock { .. }
            | StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::Empty => {}
        }
    }

    fn resolve_render_jump_target(
        &self,
        target_block_id: usize,
        materialized_blocks: &HashSet<usize>,
    ) -> Option<usize> {
        let mut current = target_block_id;
        let mut seen = HashSet::new();

        loop {
            if materialized_blocks.contains(&current) {
                return Some(current);
            }

            if !self.block_is_phi_connector(current) || !seen.insert(current) {
                return None;
            }

            let mut successors = self.cfg_successor_block_ids(current);
            successors.sort_unstable();
            successors.dedup();
            if successors.len() != 1 {
                return None;
            }
            current = successors[0];
        }
    }

    fn bind_missing_goto_labels(&self, stmt: Stmt) -> Stmt {
        let mut labels = BTreeSet::new();
        Self::collect_label_names(&stmt, &mut labels);
        let mut goto_targets = BTreeSet::new();
        Self::collect_goto_label_names(&stmt, &mut goto_targets);

        let mut insert_before = BTreeMap::<String, Vec<String>>::new();
        let mut tail_labels = Vec::new();

        for label in goto_targets.into_iter().filter(|label| !labels.contains(label)) {
            if let Some(anchor) = self.resolve_existing_label_anchor(&label, &labels) {
                insert_before.entry(anchor).or_default().push(label);
            } else {
                tail_labels.push(label);
            }
        }

        if insert_before.is_empty() && tail_labels.is_empty() {
            return stmt;
        }

        for labels in insert_before.values_mut() {
            labels.sort();
            labels.dedup();
        }
        tail_labels.sort();
        tail_labels.dedup();

        let stmt = Self::insert_missing_labels_before_anchors(stmt, &insert_before);
        if tail_labels.is_empty() {
            stmt
        } else {
            let mut out = vec![stmt];
            out.extend(tail_labels.into_iter().map(Self::empty_label_stmt));
            Self::stmt_sequence(out)
        }
    }

    fn resolve_existing_label_anchor(
        &self,
        label: &str,
        existing_labels: &BTreeSet<String>,
    ) -> Option<String> {
        let target_block_id = label.strip_prefix("BB")?.parse::<usize>().ok()?;
        let mut current = target_block_id;
        let mut seen = HashSet::new();

        loop {
            let current_label = format!("BB{}", current);
            if existing_labels.contains(&current_label) {
                return Some(current_label);
            }
            if !seen.insert(current) {
                return None;
            }

            let mut successors = self.cfg_successor_block_ids(current);
            successors.sort_unstable();
            successors.dedup();
            if successors.len() != 1 {
                return None;
            }
            current = successors[0];
        }
    }

    fn collect_label_names(stmt: &Stmt, out: &mut BTreeSet<String>) {
        match stmt {
            Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
                for stmt in stmts {
                    Self::collect_label_names(stmt, out);
                }
            }
            Stmt::Label { name, body } => {
                out.insert(name.clone());
                Self::collect_label_names(body, out);
            }
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                Self::collect_label_names(then_branch, out);
                if let Some(else_branch) = else_branch {
                    Self::collect_label_names(else_branch, out);
                }
            }
            Stmt::Loop { body, .. } => Self::collect_label_names(body, out),
            Stmt::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    Self::collect_label_names(body, out);
                }
                if let Some(default) = default {
                    Self::collect_label_names(default, out);
                }
            }
            Stmt::Break
            | Stmt::Continue
            | Stmt::Return(_)
            | Stmt::Assign { .. }
            | Stmt::ExprStmt(_)
            | Stmt::Goto(_)
            | Stmt::Empty => {}
        }
    }

    fn collect_goto_label_names(stmt: &Stmt, out: &mut BTreeSet<String>) {
        match stmt {
            Stmt::Sequence(stmts) | Stmt::Block(stmts) => {
                for stmt in stmts {
                    Self::collect_goto_label_names(stmt, out);
                }
            }
            Stmt::Label { body, .. } => Self::collect_goto_label_names(body, out),
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                Self::collect_goto_label_names(then_branch, out);
                if let Some(else_branch) = else_branch {
                    Self::collect_goto_label_names(else_branch, out);
                }
            }
            Stmt::Loop { body, .. } => Self::collect_goto_label_names(body, out),
            Stmt::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    Self::collect_goto_label_names(body, out);
                }
                if let Some(default) = default {
                    Self::collect_goto_label_names(default, out);
                }
            }
            Stmt::Goto(label) => {
                out.insert(label.clone());
            }
            Stmt::Break
            | Stmt::Continue
            | Stmt::Return(_)
            | Stmt::Assign { .. }
            | Stmt::ExprStmt(_)
            | Stmt::Empty => {}
        }
    }

    fn insert_missing_labels_before_anchors(
        stmt: Stmt,
        insert_before: &BTreeMap<String, Vec<String>>,
    ) -> Stmt {
        match stmt {
            Stmt::Sequence(stmts) => {
                Self::rebuild_sequence_with_inserted_labels(stmts, false, insert_before)
            }
            Stmt::Block(stmts) => {
                Self::rebuild_sequence_with_inserted_labels(stmts, true, insert_before)
            }
            Stmt::Label { name, body } => {
                let label = Stmt::Label {
                    name: name.clone(),
                    body: Box::new(Self::insert_missing_labels_before_anchors(
                        *body,
                        insert_before,
                    )),
                };
                if let Some(missing) = insert_before.get(&name) {
                    let mut out = missing
                        .iter()
                        .cloned()
                        .map(Self::empty_label_stmt)
                        .collect::<Vec<_>>();
                    out.push(label);
                    Stmt::Sequence(out)
                } else {
                    label
                }
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => Stmt::If {
                condition,
                then_branch: Box::new(Self::insert_missing_labels_before_anchors(
                    *then_branch,
                    insert_before,
                )),
                else_branch: else_branch.map(|branch| {
                    Box::new(Self::insert_missing_labels_before_anchors(
                        *branch,
                        insert_before,
                    ))
                }),
            },
            Stmt::Loop {
                kind,
                condition,
                body,
            } => Stmt::Loop {
                kind,
                condition,
                body: Box::new(Self::insert_missing_labels_before_anchors(
                    *body,
                    insert_before,
                )),
            },
            Stmt::Switch {
                discriminant,
                cases,
                default,
            } => Stmt::Switch {
                discriminant,
                cases: cases
                    .into_iter()
                    .map(|(label, body)| {
                        (
                            label,
                            Self::insert_missing_labels_before_anchors(body, insert_before),
                        )
                    })
                    .collect(),
                default: default.map(|body| {
                    Box::new(Self::insert_missing_labels_before_anchors(
                        *body,
                        insert_before,
                    ))
                }),
            },
            other => other,
        }
    }

    fn rebuild_sequence_with_inserted_labels(
        stmts: Vec<Stmt>,
        as_block: bool,
        insert_before: &BTreeMap<String, Vec<String>>,
    ) -> Stmt {
        let mut out = Vec::new();
        for stmt in stmts {
            let stmt = Self::insert_missing_labels_before_anchors(stmt, insert_before);
            match stmt {
                Stmt::Label { name, body } => {
                    if let Some(missing) = insert_before.get(&name) {
                        out.extend(missing.iter().cloned().map(Self::empty_label_stmt));
                    }
                    out.push(Stmt::Label { name, body });
                }
                other => out.push(other),
            }
        }
        if as_block {
            Stmt::Block(out)
        } else {
            Stmt::Sequence(out)
        }
    }

    fn empty_label_stmt(name: String) -> Stmt {
        Stmt::Label {
            name,
            body: Box::new(Stmt::Empty),
        }
    }

    fn collect_structured_block_ids(stmt: &StructuredStatement, blocks: &mut HashSet<usize>) {
        match stmt {
            StructuredStatement::BasicBlock { block_id, .. } => {
                blocks.insert(*block_id);
            }
            StructuredStatement::Sequence(stmts) => {
                for stmt in stmts {
                    Self::collect_structured_block_ids(stmt, blocks);
                }
            }
            StructuredStatement::If {
                condition_block_id,
                then_branch,
                else_branch,
                ..
            } => {
                blocks.insert(*condition_block_id);
                Self::collect_structured_block_ids(then_branch, blocks);
                if let Some(else_branch) = else_branch {
                    Self::collect_structured_block_ids(else_branch, blocks);
                }
            }
            StructuredStatement::Loop {
                header_block_id,
                body,
                ..
            } => {
                if let Some(header_block_id) = header_block_id {
                    blocks.insert(*header_block_id);
                }
                Self::collect_structured_block_ids(body, blocks);
            }
            StructuredStatement::Switch {
                header_block_id,
                cases,
                default,
                ..
            } => {
                blocks.insert(*header_block_id);
                for (_, body) in cases {
                    Self::collect_structured_block_ids(body, blocks);
                }
                if let Some(default) = default {
                    Self::collect_structured_block_ids(default, blocks);
                }
            }
            StructuredStatement::UnstructuredJump { from_block_id, .. } => {
                blocks.insert(*from_block_id);
            }
            StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::Empty => {}
        }
    }

    fn entry_block_id(stmt: &StructuredStatement) -> Option<usize> {
        match stmt {
            StructuredStatement::BasicBlock { block_id, .. } => Some(*block_id),
            StructuredStatement::Sequence(stmts) => stmts.iter().find_map(Self::entry_block_id),
            StructuredStatement::If {
                condition_block_id, ..
            } => Some(*condition_block_id),
            StructuredStatement::Loop {
                loop_type,
                header_block_id,
                body,
                ..
            } => match loop_type {
                LoopType::DoWhile => Self::entry_block_id(body).or(*header_block_id),
                LoopType::While | LoopType::Endless => {
                    header_block_id.or_else(|| Self::entry_block_id(body))
                }
            },
            StructuredStatement::Switch {
                header_block_id, ..
            } => Some(*header_block_id),
            StructuredStatement::Empty
            | StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::UnstructuredJump { .. } => None,
        }
    }

    fn ends_with_unconditional_return(stmt: &StructuredStatement) -> bool {
        match stmt {
            StructuredStatement::Return(_) => true,
            StructuredStatement::BasicBlock { stmts, .. } => stmts.iter().any(|stmt| {
                matches!(
                    &stmt.value,
                    RValue::Op { opcode, .. } if Self::is_return_opcode(opcode) && stmt.pred.is_none()
                )
            }),
            StructuredStatement::Sequence(children) => children
                .iter()
                .rev()
                .find(|child| !matches!(child, StructuredStatement::Empty))
                .is_some_and(Self::ends_with_unconditional_return),
            _ => false,
        }
    }
}

pub fn lower_structured_function(
    structured: &StructuredStatement,
    cfg: &ControlFlowGraph,
    function_ir: &FunctionIR,
    analysis: &FunctionAnalysis,
) -> crate::ast::StructuredFunction {
    StructuredLowering::new(cfg, function_ir, analysis).lower_function(structured)
}

pub fn lower_structured_stmt(
    structured: &StructuredStatement,
    cfg: &ControlFlowGraph,
    function_ir: &FunctionIR,
    analysis: &FunctionAnalysis,
) -> Stmt {
    StructuredLowering::new(cfg, function_ir, analysis).lower_structured_stmt(structured)
}

fn negate_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Unary { op, arg } if op == "!" => *arg,
        other => Expr::Unary {
            op: "!".to_string(),
            arg: Box::new(other),
        },
    }
}

pub fn lower_memory_load_expr(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<Expr> {
    if let Some(special) = lower_constmem_load_expr(access, analysis) {
        return Some(special);
    }
    let (base, index) = lower_base_and_index(stmt, access, analysis)?;
    Some(Expr::Index {
        base: Box::new(base),
        index: Box::new(index),
    })
}

pub fn lower_memory_store_lvalue(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<LValue> {
    let (base, index) = lower_base_and_index(stmt, access, analysis)?;
    Some(LValue::Indexed {
        base: Box::new(base),
        index: Box::new(index),
    })
}

fn lookup_mem_access<'a>(
    analysis: &'a FunctionAnalysis,
    block_id: usize,
    stmt_idx: usize,
) -> Option<&'a MemAccessInfo> {
    analysis
        .mem_accesses
        .iter()
        .find(|access| access.block_id == block_id && access.stmt_idx == stmt_idx)
}

fn lower_constmem_load_expr(access: &MemAccessInfo, analysis: &FunctionAnalysis) -> Option<Expr> {
    let stmt_ref = crate::abi::StatementRef {
        block_id: access.block_id,
        stmt_idx: access.stmt_idx,
    };
    let annotation = analysis
        .abi_annotations
        .constmem_by_stmt
        .get(&stmt_ref)
        .and_then(|annotations| annotations.first())?;
    Some(constmem_semantic_expr(&annotation.semantic, analysis))
}

fn lower_base_and_index(
    stmt: &IRStatement,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<(Expr, Expr)> {
    let mem_expr = stmt.mem_addr_args.as_ref()?.first()?;
    let IRExpr::Mem {
        base: addr_base,
        offset,
        width,
    } = mem_expr
    else {
        return None;
    };
    let base = match access.space {
        CudaMemorySpace::Shared => Expr::Builtin(match &access.root {
            AddressRoot::SharedObject(name) => name.clone(),
            _ => "shmem".to_string(),
        }),
        CudaMemorySpace::Local => Expr::Reg(match &access.root {
            AddressRoot::LocalObject(name) => name.clone(),
            _ => "local_mem".to_string(),
        }),
        CudaMemorySpace::Global => match &access.root {
            AddressRoot::ParamWord(param_idx) => {
                Expr::Reg(global_param_base_name(*param_idx, analysis))
            }
            AddressRoot::RegisterBase(reg) => lower_reg_expr(reg),
            _ => lower_scalar_expr_with_analysis(mem_expr, Some(analysis)),
        },
        CudaMemorySpace::Const => match &access.root {
            AddressRoot::ConstSymbol(name) => Expr::ConstMemSymbol(name.clone()),
            _ => lower_scalar_expr_with_analysis(mem_expr, Some(analysis)),
        },
        CudaMemorySpace::Param | CudaMemorySpace::Generic => {
            lower_scalar_expr_with_analysis(mem_expr, Some(analysis))
        }
    };
    let index = match access.space {
        CudaMemorySpace::Shared | CudaMemorySpace::Local => lower_rooted_element_index_expr(
            addr_base.as_ref(),
            offset.as_deref(),
            access,
            analysis,
            Some(space_backing_bit_width(access.space)),
        ),
        CudaMemorySpace::Global => lower_global_index_expr(
            addr_base.as_ref(),
            offset.as_deref(),
            access,
            analysis,
            access.bit_width.or(*width),
        ),
        _ => lower_index_expr(
            offset.as_deref(),
            access.bit_width.or(*width),
            Some(analysis),
        ),
    };
    Some((normalize_index_base(base, access, analysis), index))
}

fn lower_explicit_local_stmt(stmt: &IRStatement, access: &MemAccessInfo) -> Option<LoweredStmt> {
    let byte_expr = lower_local_byte_offset_expr(stmt)?;
    match access.kind {
        MemAccessKind::Load => {
            let dst = stmt.defs.first()?.get_reg()?;
            let src = Expr::CallLike {
                func: local_space_helper_name("load", access),
                args: vec![byte_expr],
            };
            Some(LoweredStmt {
                stmt: Stmt::Assign {
                    dst: LValue::Var(lower_reg_name(dst)),
                    src,
                },
                predicate_def_indices: vec![0],
            })
        }
        MemAccessKind::Store => {
            let src = store_value_expr(stmt, None)?;
            Some(LoweredStmt {
                stmt: Stmt::ExprStmt(Expr::CallLike {
                    func: local_space_helper_name("store", access),
                    args: vec![byte_expr, src],
                }),
                predicate_def_indices: Vec::new(),
            })
        }
        MemAccessKind::Atomic | MemAccessKind::Reduction => None,
    }
}

fn lower_local_byte_offset_expr(stmt: &IRStatement) -> Option<Expr> {
    let mem_expr = stmt.mem_addr_args.as_ref()?.first()?;
    let IRExpr::Mem { base, offset, .. } = mem_expr else {
        return None;
    };
    Some(combine_byte_offset_expr(
        base.as_ref(),
        offset.as_deref(),
        None,
    ))
}

fn local_space_helper_name(prefix: &str, access: &MemAccessInfo) -> String {
    let width = access.bit_width.unwrap_or(32);
    let lanes = access.vector_width.unwrap_or(1);
    if lanes > 1 {
        format!("local_{}_bits{}_x{}", prefix, width, lanes)
    } else {
        format!("local_{}_bits{}", prefix, width)
    }
}

fn local_space_requires_explicit_ops(analysis: &FunctionAnalysis) -> bool {
    analysis.mem_accesses.iter().any(|access| {
        access.space == CudaMemorySpace::Local
            && (access.has_dynamic_offset
                || access.constant_byte_offset.is_none()
                || access.constant_byte_offset.is_some_and(|offset| offset < 0))
    })
}

fn space_backing_bit_width(space: CudaMemorySpace) -> u32 {
    match space {
        CudaMemorySpace::Shared | CudaMemorySpace::Local => 32,
        _ => 32,
    }
}

fn lower_index_expr(
    offset: Option<&IRExpr>,
    bit_width: Option<u32>,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    match offset {
        None => Expr::Imm("0".to_string()),
        Some(IRExpr::ImmI(value)) => {
            let bytes = bit_width
                .and_then(|bits| bits.checked_div(8))
                .filter(|bytes| *bytes > 0);
            if let Some(bytes) = bytes {
                if value % i64::from(bytes) == 0 {
                    return Expr::Imm((value / i64::from(bytes)).to_string());
                }
            }
            Expr::Imm(value.to_string())
        }
        Some(expr) => lower_scalar_expr_with_analysis(expr, analysis),
    }
}

fn lower_element_index_expr(
    base: &IRExpr,
    offset: Option<&IRExpr>,
    bit_width: Option<u32>,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    let byte_expr = combine_byte_offset_expr(base, offset, analysis);
    scale_index_expr(byte_expr, bit_width)
}

fn lower_rooted_element_index_expr(
    base: &IRExpr,
    offset: Option<&IRExpr>,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
    bit_width: Option<u32>,
) -> Expr {
    if let Some(rooted_offset) = rooted_space_byte_offset(base, access, analysis) {
        let byte_expr = combine_byte_offset_expr(&rooted_offset, offset, Some(analysis));
        return scale_index_expr(byte_expr, bit_width);
    }
    lower_element_index_expr(base, offset, bit_width, Some(analysis))
}

fn combine_byte_offset_expr(
    base: &IRExpr,
    offset: Option<&IRExpr>,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    let mut terms = Vec::new();
    if !ir_expr_is_zero(base) {
        terms.push(lower_scalar_expr_with_analysis(base, analysis));
    }
    if let Some(offset) = offset.filter(|expr| !ir_expr_is_zero(expr)) {
        terms.push(lower_scalar_expr_with_analysis(offset, analysis));
    }
    match terms.len() {
        0 => Expr::Imm("0".to_string()),
        1 => terms.remove(0),
        _ => Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(terms.remove(0)),
            rhs: Box::new(terms.remove(0)),
        },
    }
}

fn scale_index_expr(expr: Expr, bit_width: Option<u32>) -> Expr {
    let bytes = bit_width
        .and_then(|bits| bits.checked_div(8))
        .filter(|bytes| *bytes > 1);
    let Some(bytes) = bytes else {
        return expr;
    };
    if let Expr::Imm(value) = &expr {
        if let Ok(value) = value.parse::<i64>() {
            if value % i64::from(bytes) == 0 {
                return Expr::Imm((value / i64::from(bytes)).to_string());
            }
        }
    }
    if let Some(simplified) = divide_expr_by_const(&expr, i64::from(bytes)) {
        return simplified;
    }
    Expr::Binary {
        op: "/".to_string(),
        lhs: Box::new(expr),
        rhs: Box::new(Expr::Imm(bytes.to_string())),
    }
}

fn divide_expr_by_const(expr: &Expr, bytes: i64) -> Option<Expr> {
    match expr {
        Expr::Imm(text) => text
            .parse::<i64>()
            .ok()
            .filter(|value| value % bytes == 0)
            .map(|value| Expr::Imm((value / bytes).to_string())),
        Expr::Binary { op, lhs, rhs } if op == "*" => {
            if expr_is_imm_i64(lhs, bytes) {
                return Some((**rhs).clone());
            }
            if expr_is_imm_i64(rhs, bytes) {
                return Some((**lhs).clone());
            }
            None
        }
        Expr::Binary { op, lhs, rhs } if op == "+" || op == "-" => Some(Expr::Binary {
            op: op.clone(),
            lhs: Box::new(divide_expr_by_const(lhs, bytes)?),
            rhs: Box::new(divide_expr_by_const(rhs, bytes)?),
        }),
        _ => None,
    }
}

fn expr_is_imm_i64(expr: &Expr, expected: i64) -> bool {
    matches!(expr, Expr::Imm(text) if text.parse::<i64>().ok() == Some(expected))
}

fn lower_global_index_expr(
    addr_base: &IRExpr,
    offset: Option<&IRExpr>,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
    bit_width: Option<u32>,
) -> Expr {
    if let Some(rooted_offset) = rooted_space_byte_offset(addr_base, access, analysis) {
        let byte_expr = combine_byte_offset_expr(&rooted_offset, offset, Some(analysis));
        return scale_index_expr(byte_expr, bit_width);
    }
    if let Some(root_relative) =
        lower_root_relative_global_index_expr(addr_base, offset, access, analysis, bit_width)
    {
        return root_relative;
    }
    lower_index_expr(offset, bit_width, Some(analysis))
}

fn lower_root_relative_global_index_expr(
    addr_base: &IRExpr,
    offset: Option<&IRExpr>,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
    bit_width: Option<u32>,
) -> Option<Expr> {
    let AddressRoot::ParamWord(param_idx) = access.root else {
        return None;
    };
    let rooted_base_name = global_param_base_name(param_idx, analysis);
    let lowered_addr = lower_scalar_expr_with_analysis(addr_base, Some(analysis));
    if let Expr::WidePtr {
        base,
        offset: rooted,
    } = lowered_addr
    {
        if matches!(base.as_ref(), Expr::Reg(name) if name == &rooted_base_name) {
            let byte_expr = match offset {
                Some(offset) if !ir_expr_is_zero(offset) => add_expr(
                    *rooted,
                    lower_scalar_expr_with_analysis(offset, Some(analysis)),
                ),
                _ => *rooted,
            };
            return Some(scale_index_expr(byte_expr, bit_width));
        }
    }
    let absolute_addr = match addr_base {
        IRExpr::Addr64 { .. } => combine_byte_offset_expr(addr_base, offset, Some(analysis)),
        _ => return None,
    };
    let rooted_base = Expr::Cast {
        ty: "uintptr_t".to_string(),
        expr: Box::new(Expr::Reg(global_param_base_name(param_idx, analysis))),
    };
    Some(scale_index_expr(
        Expr::Binary {
            op: "-".to_string(),
            lhs: Box::new(absolute_addr),
            rhs: Box::new(rooted_base),
        },
        bit_width,
    ))
}

fn global_param_base_name(param_idx: u32, analysis: &FunctionAnalysis) -> String {
    if let Some(alias) = analysis.abi_aliases.by_param.get(&param_idx) {
        if alias.kind == crate::abi::ArgAliasKind::Ptr64 {
            return format!("arg{}_ptr", param_idx);
        }
    }
    format!("param_{}", param_idx)
}

fn normalize_index_base(base: Expr, access: &MemAccessInfo, analysis: &FunctionAnalysis) -> Expr {
    if !index_base_requires_pointer_cast(access, &base) {
        return base;
    }
    Expr::Cast {
        ty: format!("{}*", unresolved_access_scalar_ty(access, analysis)),
        expr: Box::new(base),
    }
}

fn index_base_requires_pointer_cast(access: &MemAccessInfo, base: &Expr) -> bool {
    matches!(
        access.space,
        CudaMemorySpace::Global
            | CudaMemorySpace::Const
            | CudaMemorySpace::Param
            | CudaMemorySpace::Generic
    ) && !expr_is_indexable_base(base, access)
}

fn expr_is_indexable_base(expr: &Expr, access: &MemAccessInfo) -> bool {
    match expr {
        Expr::Reg(text) | Expr::Raw(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            text.ends_with("_ptr")
                || text == "shmem"
                || text == "shmem_u8"
                || (matches!(access.root, AddressRoot::ParamWord(_)) && text.starts_with("param_"))
        }
        Expr::Cast { ty, .. } => ty.ends_with('*'),
        _ => false,
    }
}

fn unresolved_access_scalar_ty(
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> &'static str {
    match access.space {
        CudaMemorySpace::Shared => analysis.shared_pointee_ty.unwrap_or("uint32_t"),
        _ => scalar_ty_for_bit_width(access.bit_width),
    }
}

fn scalar_ty_for_bit_width(bit_width: Option<u32>) -> &'static str {
    match bit_width.unwrap_or(32) {
        8 => "uint8_t",
        16 => "uint16_t",
        32 => "uint32_t",
        64 => "uint64_t",
        128 => "uint4",
        _ => "uint32_t",
    }
}

fn store_value_expr(stmt: &IRStatement, analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    stmt_args(stmt)
        .iter()
        .find(|expr| !matches!(expr, IRExpr::Mem { .. }))
        .map(|expr| lower_scalar_expr_with_analysis(expr, analysis))
}

fn stmt_args(stmt: &IRStatement) -> &[IRExpr] {
    match &stmt.value {
        RValue::Op { args, .. } => args,
        RValue::Phi(_) | RValue::ImmI(_) | RValue::ImmF(_) => &[],
    }
}

fn stmt_opcode(stmt: &IRStatement) -> &str {
    match &stmt.value {
        RValue::Op { opcode, .. } => opcode,
        RValue::Phi(_) => "phi",
        RValue::ImmI(_) => "immi",
        RValue::ImmF(_) => "immf",
    }
}

fn atomic_func_name(opcode: &str) -> &'static str {
    if opcode.contains(".POPC.INC") || opcode.contains(".ADD") {
        "atomicAdd"
    } else if opcode.contains(".MIN") {
        "atomicMin"
    } else if opcode.contains(".MAX") {
        "atomicMax"
    } else if opcode.contains(".CAS") {
        "atomicCAS"
    } else if opcode.contains(".EXCH") {
        "atomicExch"
    } else if opcode.contains(".AND") {
        "atomicAnd"
    } else if opcode.contains(".OR") {
        "atomicOr"
    } else if opcode.contains(".XOR") {
        "atomicXor"
    } else {
        "atomicOp"
    }
}

fn lower_scalar_expr(expr: &IRExpr) -> Expr {
    lower_scalar_expr_with_analysis(expr, None)
}

fn lower_scalar_expr_with_analysis(expr: &IRExpr, analysis: Option<&FunctionAnalysis>) -> Expr {
    match expr {
        IRExpr::Reg(reg) => analysis
            .and_then(|facts| facts.builtin_by_reg.get(reg))
            .map(|name| Expr::Builtin(name.clone()))
            .unwrap_or_else(|| lower_reg_expr(reg)),
        IRExpr::ImmI(value) => Expr::Imm(value.to_string()),
        IRExpr::ImmF(value) => Expr::Imm(value.to_string()),
        IRExpr::Addr64 { lo, hi } => analysis
            .and_then(|facts| lower_rooted_wide_expr(lo, hi, facts))
            .unwrap_or_else(|| {
                let lo = lower_scalar_expr_with_analysis(lo, analysis);
                let hi = lower_scalar_expr_with_analysis(hi, analysis);
                lower_named_pointer_lane_wide_expr(lo.clone(), hi.clone()).unwrap_or(Expr::Addr64 {
                    lo: Box::new(lo),
                    hi: Box::new(hi),
                })
            }),
        IRExpr::Mem { .. } => Expr::Raw("/*mem*/".to_string()),
        IRExpr::Op { op, args } => lower_ir_op_expr_with_analysis(op, args, analysis),
    }
}

fn lower_basic_stmt(
    block_id: usize,
    stmt_idx: usize,
    stmt: &IRStatement,
    analysis: &FunctionAnalysis,
) -> Stmt {
    let lowered = lower_memory_stmt_detail(block_id, stmt_idx, stmt, analysis)
        .unwrap_or_else(|| lower_non_memory_stmt_with_analysis(stmt, Some(analysis)));
    apply_stmt_predicate(stmt, lowered)
}

#[cfg(test)]
fn lower_non_memory_stmt(stmt: &IRStatement) -> LoweredStmt {
    lower_non_memory_stmt_with_analysis(stmt, None)
}

fn lower_non_memory_stmt_with_analysis(
    stmt: &IRStatement,
    analysis: Option<&FunctionAnalysis>,
) -> LoweredStmt {
    if let Some(lowered) = lower_structural_control_stmt(stmt) {
        return lowered;
    }
    if let Some(lowered) = lower_multi_def_imad_wide_stmt(stmt, analysis) {
        return lowered;
    }
    if let Some(lowered) = lower_multi_def_lea_stmt(stmt, analysis) {
        return lowered;
    }
    if let Some(lowered) = lower_multi_def_lop3_stmt(stmt, analysis) {
        return lowered;
    }
    match &stmt.value {
        RValue::Op { opcode, args } => {
            let rhs = lower_op_expr_with_analysis(opcode, args, analysis);
            if let Some((def_idx, def)) = select_non_memory_result_def(stmt) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src: rhs,
                    },
                    predicate_def_indices: vec![def_idx],
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(rhs),
                    predicate_def_indices: Vec::new(),
                }
            }
        }
        RValue::Phi(args) => {
            let rhs = Expr::CallLike {
                func: "phi".to_string(),
                args: args
                    .iter()
                    .map(|arg| lower_scalar_expr_with_analysis(arg, analysis))
                    .collect(),
            };
            if let Some((def_idx, def)) = select_non_memory_result_def(stmt) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src: rhs,
                    },
                    predicate_def_indices: vec![def_idx],
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(rhs),
                    predicate_def_indices: Vec::new(),
                }
            }
        }
        RValue::ImmI(value) => {
            let src = Expr::Imm(value.to_string());
            if let Some((def_idx, def)) = select_non_memory_result_def(stmt) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src,
                    },
                    predicate_def_indices: vec![def_idx],
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(src),
                    predicate_def_indices: Vec::new(),
                }
            }
        }
        RValue::ImmF(value) => {
            let src = Expr::Imm(value.to_string());
            if let Some((def_idx, def)) = select_non_memory_result_def(stmt) {
                LoweredStmt {
                    stmt: Stmt::Assign {
                        dst: LValue::Var(lower_reg_name(def)),
                        src,
                    },
                    predicate_def_indices: vec![def_idx],
                }
            } else {
                LoweredStmt {
                    stmt: Stmt::ExprStmt(src),
                    predicate_def_indices: Vec::new(),
                }
            }
        }
    }
}

fn lower_structural_control_stmt(stmt: &IRStatement) -> Option<LoweredStmt> {
    let RValue::Op { opcode, args } = &stmt.value else {
        return None;
    };
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    let lowered_stmt = match mnem {
        "EXIT" | "RET" => Stmt::Return(None),
        "BAR" if opcode.contains("SYNC") => Stmt::ExprStmt(Expr::CallLike {
            func: "__syncthreads".to_string(),
            args: Vec::new(),
        }),
        "WARPSYNC" => Stmt::ExprStmt(Expr::CallLike {
            func: "__syncwarp".to_string(),
            args: Vec::new(),
        }),
        "BRX" => Stmt::Switch {
            discriminant: args.first().map(lower_scalar_expr),
            cases: Vec::new(),
            default: None,
        },
        "BRA" | "BSSY" | "BSYNC" | "SSY" | "SYNC" | "NOP" | "DEPBAR" | "MEMBAR" | "FENCE" => {
            Stmt::Empty
        }
        _ => return None,
    };
    Some(LoweredStmt {
        stmt: lowered_stmt,
        predicate_def_indices: Vec::new(),
    })
}

#[cfg(test)]
fn lower_op_expr(opcode: &str, args: &[IRExpr]) -> Expr {
    lower_op_expr_with_analysis(opcode, args, None)
}

fn lower_op_expr_with_analysis(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    if let Some(expr) = lower_semantic_op_expr(opcode, args, analysis) {
        return expr;
    }
    if let Some(expr) = lower_simple_op_expr(opcode, args, analysis) {
        return expr;
    }
    Expr::CallLike {
        func: opcode.to_string(),
        args: args
            .iter()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis))
            .collect(),
    }
}

fn lower_semantic_op_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    lower_setp_expr(opcode, args, analysis)
        .or_else(|| lower_fsel_expr(opcode, args, analysis))
        .or_else(|| lower_lop3_expr(opcode, args, analysis))
        .or_else(|| lower_plop3_expr(opcode, args))
        .or_else(|| lower_lea_hi_expr(opcode, args, analysis))
        .or_else(|| lower_lea_expr(opcode, args, analysis))
        .or_else(|| lower_imad_hi_expr(opcode, args, analysis))
        .or_else(|| lower_imad_wide_expr(opcode, args, analysis))
        .or_else(|| lower_imad_x_expr(opcode, args, analysis))
        .or_else(|| lower_imad_expr(opcode, args, analysis))
        .or_else(|| lower_iadd3_x_expr(opcode, args, analysis))
        .or_else(|| lower_wide_add_lo_expr(opcode, args, analysis))
        .or_else(|| lower_shf_expr(opcode, args, analysis))
        .or_else(|| lower_shfl_expr(opcode, args, analysis))
        .or_else(|| lower_iabs_expr(opcode, args, analysis))
        .or_else(|| lower_i2f_expr(opcode, args, analysis))
        .or_else(|| lower_f2i_expr(opcode, args, analysis))
        .or_else(|| lower_frnd_expr(opcode, args, analysis))
        .or_else(|| lower_fadd_expr(opcode, args, analysis))
        .or_else(|| lower_fmul_expr(opcode, args, analysis))
        .or_else(|| lower_ffma_expr(opcode, args, analysis))
        .or_else(|| lower_fmnmx_expr(opcode, args, analysis))
        .or_else(|| lower_mufu_expr(opcode, args, analysis))
}

fn lower_simple_op_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    match mnem {
        "S2R" | "CS2R" | "S2UR" => args
            .first()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis)),
        "MOV" | "UMOV" | "FMOV" => args
            .first()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis)),
        "IADD" | "IADD3" | "UIADD" | "UIADD3" | "FADD" if !opcode.contains('.') => {
            lower_add_expr(args, analysis)
        }
        _ => None,
    }
}

fn lower_setp_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "ISETP" | "UISETP" | "FSETP" | "DSETP" | "HSETP2") || args.len() < 2 {
        return None;
    }
    let parts = opcode.split('.').collect::<Vec<_>>();
    let cmp_token = parts
        .iter()
        .skip(1)
        .find_map(|part| setp_comparison_token(part))?;
    let mut lhs = lower_scalar_expr_with_analysis(&args[0], analysis);
    let mut rhs = lower_scalar_expr_with_analysis(&args[1], analysis);
    if mnem == "ISETP"
        && !parts.iter().any(|part| *part == "U32")
        && ordered_setp_comparison_op(cmp_token).is_some_and(|cmp| !matches!(cmp, "==" | "!="))
    {
        lhs = Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(lhs),
        };
        rhs = Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(rhs),
        };
    }
    let cmp_expr = if matches!(mnem, "FSETP" | "DSETP" | "HSETP2") {
        lower_float_setp_compare_expr(cmp_token, &args[0], &args[1], lhs, rhs, analysis)?
    } else {
        Expr::Binary {
            op: ordered_setp_comparison_op(cmp_token)?.to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    };
    let Some(combine) = args.get(2) else {
        return Some(cmp_expr);
    };
    if ir_expr_is_true_pred(combine) {
        return Some(cmp_expr);
    }
    if ir_expr_is_false_pred(combine) {
        return Some(Expr::Imm("false".to_string()));
    }
    let combine_op = if parts.iter().any(|part| *part == "OR") {
        "||"
    } else {
        "&&"
    };
    Some(Expr::Binary {
        op: combine_op.to_string(),
        lhs: Box::new(cmp_expr),
        rhs: Box::new(lower_scalar_expr_with_analysis(combine, analysis)),
    })
}

fn setp_comparison_token(part: &str) -> Option<&'static str> {
    match part {
        "LT" => Some("LT"),
        "LTU" => Some("LTU"),
        "LE" => Some("LE"),
        "LEU" => Some("LEU"),
        "GT" => Some("GT"),
        "GTU" => Some("GTU"),
        "GE" => Some("GE"),
        "GEU" => Some("GEU"),
        "EQ" => Some("EQ"),
        "EQU" => Some("EQU"),
        "NE" => Some("NE"),
        "NEU" => Some("NEU"),
        "NUM" => Some("NUM"),
        "NAN" => Some("NAN"),
        _ => None,
    }
}

fn ordered_setp_comparison_op(part: &str) -> Option<&'static str> {
    match part {
        "LT" | "LTU" => Some("<"),
        "LE" | "LEU" => Some("<="),
        "GT" | "GTU" => Some(">"),
        "GE" | "GEU" => Some(">="),
        "EQ" | "EQU" => Some("=="),
        "NE" | "NEU" => Some("!="),
        _ => None,
    }
}

fn lower_float_setp_compare_expr(
    cmp_token: &str,
    lhs_ir: &IRExpr,
    rhs_ir: &IRExpr,
    lhs: Expr,
    rhs: Expr,
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if cmp_token == "NUM" {
        let Some(nan_any) = float_setp_nan_any_expr(lhs_ir, rhs_ir, analysis) else {
            return Some(Expr::Imm("true".to_string()));
        };
        return Some(Expr::Unary {
            op: "!".to_string(),
            arg: Box::new(nan_any),
        });
    }
    if cmp_token == "NAN" {
        return Some(
            float_setp_nan_any_expr(lhs_ir, rhs_ir, analysis)
                .unwrap_or_else(|| Expr::Imm("false".to_string())),
        );
    }
    let ordered = Expr::Binary {
        op: ordered_setp_comparison_op(cmp_token)?.to_string(),
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    };
    if !is_unordered_float_cmp_token(cmp_token) {
        return Some(ordered);
    }
    let Some(nan_any) = float_setp_nan_any_expr(lhs_ir, rhs_ir, analysis) else {
        return Some(ordered);
    };
    Some(or_expr(ordered, nan_any))
}

fn is_unordered_float_cmp_token(cmp_token: &str) -> bool {
    matches!(cmp_token, "LTU" | "LEU" | "GTU" | "GEU" | "EQU" | "NEU")
}

fn float_setp_nan_any_expr(
    lhs_ir: &IRExpr,
    rhs_ir: &IRExpr,
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mut checks = Vec::new();
    if ir_expr_may_be_nan(lhs_ir) {
        checks.push(Expr::CallLike {
            func: "isnan".to_string(),
            args: vec![lower_scalar_expr_with_analysis(lhs_ir, analysis)],
        });
    }
    if ir_expr_may_be_nan(rhs_ir) {
        checks.push(Expr::CallLike {
            func: "isnan".to_string(),
            args: vec![lower_scalar_expr_with_analysis(rhs_ir, analysis)],
        });
    }
    checks.into_iter().reduce(or_expr)
}

fn ir_expr_may_be_nan(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::ImmI(_) => false,
        IRExpr::ImmF(value) => value.is_nan(),
        IRExpr::Op { op, .. } if matches!(op.as_str(), "+QNAN" | "-QNAN") => true,
        IRExpr::Op { op, .. } if matches!(op.as_str(), "+INF" | "-INF") => false,
        IRExpr::Reg(_) | IRExpr::Addr64 { .. } | IRExpr::Mem { .. } | IRExpr::Op { .. } => true,
    }
}

fn or_expr(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op: "||".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn lower_fsel_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "SEL" | "FSEL") || args.len() != 3 {
        return None;
    }
    Some(Expr::Ternary {
        cond: Box::new(lower_scalar_expr_with_analysis(&args[2], analysis)),
        then_expr: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
        else_expr: Box::new(lower_scalar_expr_with_analysis(&args[1], analysis)),
    })
}

fn lower_add_expr(args: &[IRExpr], analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    let mut terms = args
        .iter()
        .filter(|expr| !ir_expr_is_zero(expr))
        .map(|expr| lower_scalar_expr_with_analysis(expr, analysis));
    let first = terms.next()?;
    Some(terms.fold(first, |lhs, rhs| Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }))
}

fn lower_multi_def_lop3_stmt(
    stmt: &IRStatement,
    analysis: Option<&FunctionAnalysis>,
) -> Option<LoweredStmt> {
    let RValue::Op { opcode, args } = &stmt.value else {
        return None;
    };
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "LOP3" | "ULOP3") || stmt.defs.len() < 2 {
        return None;
    }
    let (pred_def_idx, pred_def) = stmt
        .defs
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr.get_reg().map(|reg| (idx, reg)))
        .find(|(_, reg)| matches!(reg.class.as_str(), "P" | "UP"))?;
    let (data_def_idx, data_def) = stmt
        .defs
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr.get_reg().map(|reg| (idx, reg)))
        .find(|(_, reg)| matches!(reg.class.as_str(), "R" | "UR"))?;
    let data_expr = lower_lop3_expr(opcode, args, analysis)?;
    let pred_expr = Expr::Binary {
        op: "!=".to_string(),
        lhs: Box::new(data_expr.clone()),
        rhs: Box::new(Expr::Imm("0".to_string())),
    };
    Some(LoweredStmt {
        stmt: Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var(lower_reg_name(data_def)),
                src: data_expr,
            },
            Stmt::Assign {
                dst: LValue::Var(lower_reg_name(pred_def)),
                src: pred_expr,
            },
        ]),
        predicate_def_indices: vec![data_def_idx, pred_def_idx],
    })
}

fn lower_multi_def_imad_wide_stmt(
    stmt: &IRStatement,
    analysis: Option<&FunctionAnalysis>,
) -> Option<LoweredStmt> {
    let RValue::Op { opcode, args } = &stmt.value else {
        return None;
    };
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if !matches!(mnem, "IMAD" | "UIMAD") || !parts.iter().any(|part| *part == "WIDE") {
        return None;
    }
    let (lo_def_idx, lo_def) = stmt
        .defs
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr.get_reg().map(|reg| (idx, reg)))
        .find(|(_, reg)| matches!(reg.class.as_str(), "R" | "UR"))?;
    let hi_def = stmt
        .defs
        .iter()
        .filter_map(|expr| expr.get_reg())
        .find(|reg| reg.class == lo_def.class && reg.idx == lo_def.idx.saturating_add(1))?;
    let wide = lower_imad_wide_value_expr(opcode, args, analysis)?;
    let hi_def_idx = stmt.defs.iter().enumerate().find_map(|(idx, expr)| {
        let reg = expr.get_reg()?;
        (reg == hi_def).then_some(idx)
    })?;
    Some(LoweredStmt {
        stmt: Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var(lower_reg_name(lo_def)),
                src: Expr::LaneExtract {
                    value: Box::new(wide.clone()),
                    lane: PointerLane::Lo32,
                },
            },
            Stmt::Assign {
                dst: LValue::Var(lower_reg_name(hi_def)),
                src: Expr::LaneExtract {
                    value: Box::new(wide),
                    lane: PointerLane::Hi32,
                },
            },
        ]),
        predicate_def_indices: vec![lo_def_idx, hi_def_idx],
    })
}

fn lower_multi_def_lea_stmt(
    stmt: &IRStatement,
    analysis: Option<&FunctionAnalysis>,
) -> Option<LoweredStmt> {
    let RValue::Op { opcode, args } = &stmt.value else {
        return None;
    };
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "LEA" | "ULEA") || opcode.split('.').any(|part| part == "HI") {
        return None;
    }
    let (data_def_idx, data_def) = stmt
        .defs
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr.get_reg().map(|reg| (idx, reg)))
        .find(|(_, reg)| matches!(reg.class.as_str(), "R" | "UR"))?;
    let (pred_def_idx, pred_def) = stmt
        .defs
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr.get_reg().map(|reg| (idx, reg)))
        .find(|(_, reg)| matches!(reg.class.as_str(), "P" | "UP"))?;
    let data_expr = lower_lea_expr(opcode, args, analysis)?;
    let carry_expr = lower_lea_carry_expr(args, analysis)?;
    Some(LoweredStmt {
        stmt: Stmt::Sequence(vec![
            Stmt::Assign {
                dst: LValue::Var(lower_reg_name(data_def)),
                src: data_expr,
            },
            Stmt::Assign {
                dst: LValue::Var(lower_reg_name(pred_def)),
                src: carry_expr,
            },
        ]),
        predicate_def_indices: vec![data_def_idx, pred_def_idx],
    })
}

fn is_pred_control_expr(expr: &IRExpr) -> bool {
    ir_expr_is_true_pred(expr) || ir_expr_is_false_pred(expr)
}

fn is_predicate_expr(expr: &IRExpr) -> bool {
    is_pred_control_expr(expr)
        || expr
            .get_reg()
            .is_some_and(|reg| matches!(reg.class.as_str(), "P" | "UP"))
}

fn lower_lop3_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "LOP3" | "ULOP3") || args.len() != 5 || !is_pred_control_expr(&args[4]) {
        return None;
    }
    let imm = match args[3] {
        IRExpr::ImmI(value) => (value & 0xff) as u8,
        _ => return None,
    };
    if let Some(expr) = lower_lop3_float_sign_inject_expr(imm, args, analysis) {
        return Some(expr);
    }
    let a0z = ir_expr_is_zero(&args[0]);
    let a1z = ir_expr_is_zero(&args[1]);
    let a2z = ir_expr_is_zero(&args[2]);

    if a0z && a1z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 0, 0, 1) as u8) << 1);
        return lower_lop3_unary_expr(lower_scalar_expr_with_analysis(&args[2], analysis), bits);
    }
    if a0z && a2z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 0, 1, 0) as u8) << 1);
        return lower_lop3_unary_expr(lower_scalar_expr_with_analysis(&args[1], analysis), bits);
    }
    if a1z && a2z {
        let bits = ((lop3_bit(imm, 0, 0, 0) as u8) << 0) | ((lop3_bit(imm, 1, 0, 0) as u8) << 1);
        return lower_lop3_unary_expr(lower_scalar_expr_with_analysis(&args[0], analysis), bits);
    }
    if a2z {
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 1, 0) as u8) << 1)
            | ((lop3_bit(imm, 1, 0, 0) as u8) << 2)
            | ((lop3_bit(imm, 1, 1, 0) as u8) << 3);
        return lower_lop3_binary_expr(
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[1], analysis),
            nibble,
        );
    }
    if a0z {
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 0, 1) as u8) << 1)
            | ((lop3_bit(imm, 0, 1, 0) as u8) << 2)
            | ((lop3_bit(imm, 0, 1, 1) as u8) << 3);
        return lower_lop3_binary_expr(
            lower_scalar_expr_with_analysis(&args[1], analysis),
            lower_scalar_expr_with_analysis(&args[2], analysis),
            nibble,
        );
    }
    if a1z {
        let nibble = ((lop3_bit(imm, 0, 0, 0) as u8) << 0)
            | ((lop3_bit(imm, 0, 0, 1) as u8) << 1)
            | ((lop3_bit(imm, 1, 0, 0) as u8) << 2)
            | ((lop3_bit(imm, 1, 0, 1) as u8) << 3);
        return lower_lop3_binary_expr(
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[2], analysis),
            nibble,
        );
    }
    None
}

fn lower_plop3_expr(opcode: &str, args: &[IRExpr]) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "PLOP3" | "UPLOP3") || args.len() != 6 {
        return None;
    }
    if args[..4].iter().all(ir_expr_is_true_pred) {
        match args[4] {
            IRExpr::ImmI(8 | 128) => return Some(Expr::Imm("false".to_string())),
            _ => {}
        }
    }
    None
}

fn lower_lop3_float_sign_inject_expr(
    imm: u8,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    match imm {
        0xf8 if is_sign_mask_imm(&args[1])
            && expr_is_floatish(&args[0], analysis)
            && expr_is_floatish(&args[2], analysis) =>
        {
            Some(Expr::CallLike {
                func: "copysignf".to_string(),
                args: vec![
                    lower_scalar_expr_with_analysis(&args[0], analysis),
                    lower_scalar_expr_with_analysis(&args[2], analysis),
                ],
            })
        }
        0xf8 if is_sign_mask_imm(&args[2])
            && expr_is_floatish(&args[0], analysis)
            && expr_is_floatish(&args[1], analysis) =>
        {
            Some(Expr::CallLike {
                func: "copysignf".to_string(),
                args: vec![
                    lower_scalar_expr_with_analysis(&args[0], analysis),
                    lower_scalar_expr_with_analysis(&args[1], analysis),
                ],
            })
        }
        0xea if is_sign_mask_imm(&args[0])
            && expr_is_floatish(&args[1], analysis)
            && expr_is_floatish(&args[2], analysis) =>
        {
            Some(Expr::CallLike {
                func: "copysignf".to_string(),
                args: vec![
                    lower_scalar_expr_with_analysis(&args[2], analysis),
                    lower_scalar_expr_with_analysis(&args[1], analysis),
                ],
            })
        }
        0xea if is_sign_mask_imm(&args[1])
            && expr_is_floatish(&args[0], analysis)
            && expr_is_floatish(&args[2], analysis) =>
        {
            Some(Expr::CallLike {
                func: "copysignf".to_string(),
                args: vec![
                    lower_scalar_expr_with_analysis(&args[2], analysis),
                    lower_scalar_expr_with_analysis(&args[0], analysis),
                ],
            })
        }
        _ => None,
    }
}

fn is_sign_mask_imm(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::ImmI(value) if (*value as u32) == 0x8000_0000)
}

fn expr_is_floatish(expr: &IRExpr, analysis: Option<&FunctionAnalysis>) -> bool {
    match expr {
        IRExpr::ImmF(_) => true,
        IRExpr::Reg(reg) => analysis
            .and_then(|facts| facts.scalar_type_by_reg.get(reg).copied())
            .is_some_and(is_floatish_scalar_type),
        _ => false,
    }
}

fn is_floatish_scalar_type(ty: InferredType) -> bool {
    matches!(
        ty,
        InferredType::F16 | InferredType::F32 | InferredType::AnyFloat
    )
}

fn lop3_bit(imm: u8, a: u8, b: u8, c: u8) -> bool {
    let idx = (a << 2) | (b << 1) | c;
    ((imm >> idx) & 1) != 0
}

fn lower_lop3_unary_expr(x: Expr, bits: u8) -> Option<Expr> {
    match bits & 0x3 {
        0x0 => Some(Expr::Imm("0".to_string())),
        0x1 => Some(Expr::Unary {
            op: "~".to_string(),
            arg: Box::new(x),
        }),
        0x2 => Some(x),
        0x3 => Some(Expr::Imm("0xffffffff".to_string())),
        _ => None,
    }
}

fn lower_lop3_binary_expr(x: Expr, y: Expr, nibble: u8) -> Option<Expr> {
    let x_not = || Expr::Unary {
        op: "~".to_string(),
        arg: Box::new(x.clone()),
    };
    let y_not = || Expr::Unary {
        op: "~".to_string(),
        arg: Box::new(y.clone()),
    };
    let x_and_y = || Expr::Binary {
        op: "&".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    let x_or_y = || Expr::Binary {
        op: "|".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    let x_xor_y = || Expr::Binary {
        op: "^".to_string(),
        lhs: Box::new(x.clone()),
        rhs: Box::new(y.clone()),
    };
    match nibble & 0xf {
        0x0 => Some(Expr::Imm("0".to_string())),
        0x1 => Some(Expr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_or_y()),
        }),
        0x2 => Some(Expr::Binary {
            op: "&".to_string(),
            lhs: Box::new(x_not()),
            rhs: Box::new(y.clone()),
        }),
        0x3 => Some(x_not()),
        0x4 => Some(Expr::Binary {
            op: "&".to_string(),
            lhs: Box::new(x.clone()),
            rhs: Box::new(y_not()),
        }),
        0x5 => Some(y_not()),
        0x6 => Some(x_xor_y()),
        0x7 => Some(Expr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_and_y()),
        }),
        0x8 => Some(x_and_y()),
        0x9 => Some(Expr::Unary {
            op: "~".to_string(),
            arg: Box::new(x_xor_y()),
        }),
        0xa => Some(y.clone()),
        0xb => Some(Expr::Binary {
            op: "|".to_string(),
            lhs: Box::new(x_not()),
            rhs: Box::new(y.clone()),
        }),
        0xc => Some(x.clone()),
        0xd => Some(Expr::Binary {
            op: "|".to_string(),
            lhs: Box::new(x.clone()),
            rhs: Box::new(y_not()),
        }),
        0xe => Some(x_or_y()),
        0xf => Some(Expr::Imm("0xffffffff".to_string())),
        _ => None,
    }
}

fn lower_imad_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if !matches!(mnem, "IMAD" | "UIMAD") || args.len() != 3 {
        return None;
    }
    if parts
        .iter()
        .any(|part| matches!(*part, "WIDE" | "HI" | "X"))
    {
        return None;
    }
    if parts.iter().any(|part| *part == "MOV") {
        if ir_expr_is_zero(&args[0]) && ir_expr_is_zero(&args[1]) {
            return Some(lower_scalar_expr_with_analysis(&args[2], analysis));
        }
        return None;
    }
    if parts.iter().any(|part| *part == "SHL") {
        if let IRExpr::ImmI(value) = args[1] {
            let factor = value as u64;
            if factor > 0 && factor.is_power_of_two() {
                return Some(Expr::Binary {
                    op: "<<".to_string(),
                    lhs: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
                    rhs: Box::new(Expr::Imm(factor.trailing_zeros().to_string())),
                });
            }
        }
        return None;
    }
    if parts.iter().any(|part| *part == "IADD") {
        if is_imm_i(&args[1], 1) {
            return Some(add_expr(
                lower_scalar_expr_with_analysis(&args[0], analysis),
                lower_scalar_expr_with_analysis(&args[2], analysis),
            ));
        }
        if is_imm_i(&args[0], 1) {
            return Some(add_expr(
                lower_scalar_expr_with_analysis(&args[1], analysis),
                lower_scalar_expr_with_analysis(&args[2], analysis),
            ));
        }
    }
    Some(add_expr(
        Expr::Binary {
            op: "*".to_string(),
            lhs: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
            rhs: Box::new(lower_scalar_expr_with_analysis(&args[1], analysis)),
        },
        lower_scalar_expr_with_analysis(&args[2], analysis),
    ))
}

fn lower_imad_wide_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if !matches!(mnem, "IMAD" | "UIMAD")
        || !parts.iter().any(|part| *part == "WIDE")
        || args.len() != 3
    {
        return None;
    }
    Some(Expr::LaneExtract {
        value: Box::new(lower_imad_wide_value_expr(opcode, args, analysis)?),
        lane: PointerLane::Lo32,
    })
}

fn lower_imad_wide_value_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if args.len() != 3 {
        return None;
    }
    Some(add_expr(
        Expr::Binary {
            op: "*".to_string(),
            lhs: Box::new(lower_imad_wide_operand(opcode, &args[0], analysis)),
            rhs: Box::new(lower_imad_wide_operand(opcode, &args[1], analysis)),
        },
        lower_imad_wide_base_expr(&args[2], analysis),
    ))
}

fn lower_imad_wide_operand(
    opcode: &str,
    arg: &IRExpr,
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    if opcode.starts_with("UIMAD") || opcode.split('.').any(|part| part == "U32") {
        widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(arg, analysis))
    } else {
        widen_sx32_to_i64_expr(lower_scalar_expr_with_analysis(arg, analysis))
    }
}

fn lower_imad_wide_base_expr(arg: &IRExpr, analysis: Option<&FunctionAnalysis>) -> Expr {
    if let Some(reg) = arg.get_reg() {
        let hi = crate::ir::RegId::new(&reg.class, reg.idx.saturating_add(1), reg.sign)
            .with_ssa(reg.ssa.unwrap_or(0));
        return lower_wide_input_expr(arg, &IRExpr::Reg(hi), analysis);
    }
    match lower_scalar_expr_with_analysis(arg, analysis) {
        Expr::PtrLane {
            base,
            lane: PointerLane::Lo32,
        } => Expr::Addr64 {
            lo: Box::new(Expr::PtrLane {
                base: base.clone(),
                lane: PointerLane::Lo32,
            }),
            hi: Box::new(Expr::PtrLane {
                base,
                lane: PointerLane::Hi32,
            }),
        },
        expr => widen_u32_to_u64_expr(expr),
    }
}

fn lower_imad_x_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if !matches!(mnem, "IMAD" | "UIMAD") || !parts.iter().any(|part| *part == "X") || args.len() < 4
    {
        return None;
    }
    let carry = lower_carry_inc_expr(&args[3], analysis)?;
    let mut expr = if ir_expr_is_zero(&args[0]) && ir_expr_is_zero(&args[1]) {
        lower_scalar_expr_with_analysis(&args[2], analysis)
    } else if is_imm_i(&args[1], 1) {
        add_expr(
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[2], analysis),
        )
    } else if is_imm_i(&args[0], 1) {
        add_expr(
            lower_scalar_expr_with_analysis(&args[1], analysis),
            lower_scalar_expr_with_analysis(&args[2], analysis),
        )
    } else {
        add_expr(
            Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
                rhs: Box::new(lower_scalar_expr_with_analysis(&args[1], analysis)),
            },
            lower_scalar_expr_with_analysis(&args[2], analysis),
        )
    };
    if !matches!(carry, Expr::Imm(ref text) if text == "0") {
        expr = add_expr(expr, carry);
    }
    Some(expr)
}

fn lower_iadd3_x_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if !matches!(opcode, "IADD3.X" | "UIADD3.X") {
        return None;
    }
    let (a0, a1, a2, carry_pred) = extract_addx_operands(args)?;
    let mut sum = add_expr(
        lower_scalar_expr_with_analysis(a0, analysis),
        lower_scalar_expr_with_analysis(a1, analysis),
    );
    if !ir_expr_is_zero(a2) {
        sum = add_expr(sum, lower_scalar_expr_with_analysis(a2, analysis));
    }
    Some(add_expr(sum, lower_carry_inc_expr(carry_pred, analysis)?))
}

fn extract_addx_operands(args: &[IRExpr]) -> Option<(&IRExpr, &IRExpr, &IRExpr, &IRExpr)> {
    if args.len() < 4 {
        return None;
    }
    let mut start = 0usize;
    let mut end = args.len();
    while end.saturating_sub(start) > 5 {
        let Some(reg) = args[start].get_reg() else {
            break;
        };
        if !matches!(reg.class.as_str(), "P" | "UP") {
            break;
        }
        start += 1;
    }
    while end.saturating_sub(start) > 4 && is_pred_control_expr(&args[end - 1]) {
        end -= 1;
    }
    (end.saturating_sub(start) == 4).then_some((
        &args[start],
        &args[start + 1],
        &args[start + 2],
        &args[start + 3],
    ))
}

fn lower_carry_inc_expr(expr: &IRExpr, analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    if ir_expr_is_true_pred(expr) {
        return Some(Expr::Imm("1".to_string()));
    }
    if ir_expr_is_false_pred(expr) {
        return Some(Expr::Imm("0".to_string()));
    }
    let reg = expr.get_reg()?;
    if !matches!(reg.class.as_str(), "P" | "UP") {
        return None;
    }
    Some(Expr::Ternary {
        cond: Box::new(lower_scalar_expr_with_analysis(expr, analysis)),
        then_expr: Box::new(Expr::Imm("1".to_string())),
        else_expr: Box::new(Expr::Imm("0".to_string())),
    })
}

fn lower_shf_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if args.len() != 3 {
        return None;
    }
    let parts = opcode.split('.').collect::<Vec<_>>();
    let is_right = parts.contains(&"R");
    let is_left = parts.contains(&"L");
    let is_hi = parts.contains(&"HI");
    if is_right && is_hi {
        if !ir_expr_is_zero(&args[0]) {
            return None;
        }
        let lhs = lower_scalar_expr_with_analysis(&args[2], analysis);
        let rhs = lower_scalar_expr_with_analysis(&args[1], analysis);
        let lhs = if parts.contains(&"S32") {
            Expr::Cast {
                ty: "int32_t".to_string(),
                expr: Box::new(lhs),
            }
        } else {
            lhs
        };
        return Some(Expr::Binary {
            op: ">>".to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        });
    }
    if is_left && is_hi && parts.contains(&"U64") {
        return Some(Expr::LaneExtract {
            value: Box::new(Expr::Binary {
                op: "<<".to_string(),
                lhs: Box::new(lower_wide_input_expr(&args[0], &args[2], analysis)),
                rhs: Box::new(lower_scalar_expr_with_analysis(&args[1], analysis)),
            }),
            lane: PointerLane::Hi32,
        });
    }
    None
}

fn lower_shfl_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if args.len() != 3 {
        return None;
    }
    let src = lower_scalar_expr_with_analysis(&args[0], analysis);
    let lane = lower_scalar_expr_with_analysis(&args[1], analysis);
    let clamp = lower_scalar_expr_with_analysis(&args[2], analysis);
    if opcode.starts_with("SHFL.DOWN") {
        return Some(Expr::CallLike {
            func: "__shfl_down_sync".to_string(),
            args: vec![Expr::Imm("0xffffffff".to_string()), src, lane],
        });
    }
    if opcode.starts_with("SHFL.UP") {
        return Some(Expr::CallLike {
            func: "__shfl_up_sync".to_string(),
            args: vec![Expr::Imm("0xffffffff".to_string()), src, lane],
        });
    }
    if opcode.starts_with("SHFL.BFLY") || opcode.starts_with("SHFL.XOR") {
        return Some(Expr::CallLike {
            func: "__shfl_xor_sync".to_string(),
            args: vec![Expr::Imm("0xffffffff".to_string()), src, lane],
        });
    }
    if opcode.starts_with("SHFL") || opcode.starts_with("USHFL") {
        return Some(Expr::CallLike {
            func: "__shfl_sync".to_string(),
            args: vec![Expr::Imm("0xffffffff".to_string()), src, lane, clamp],
        });
    }
    None
}

fn lower_iabs_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if !opcode.starts_with("IABS") || args.len() != 1 {
        return None;
    }
    Some(Expr::CallLike {
        func: "abs".to_string(),
        args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
    })
}

fn lower_i2f_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if !matches!(mnem, "I2F" | "UI2F") || args.len() != 1 {
        return None;
    }
    let signed = mnem == "I2F";
    let mut func = None;
    for part in parts.iter().skip(1) {
        match *part {
            "RP" => {
                func = Some(if signed {
                    "__int2float_ru"
                } else {
                    "__uint2float_ru"
                })
            }
            "RM" => {
                func = Some(if signed {
                    "__int2float_rd"
                } else {
                    "__uint2float_rd"
                })
            }
            "RN" => {
                func = Some(if signed {
                    "__int2float_rn"
                } else {
                    "__uint2float_rn"
                })
            }
            "RZ" => {
                func = Some(if signed {
                    "__int2float_rz"
                } else {
                    "__uint2float_rz"
                })
            }
            _ => return None,
        }
    }
    Some(if let Some(func) = func {
        Expr::CallLike {
            func: func.to_string(),
            args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
        }
    } else {
        Expr::Cast {
            ty: "float".to_string(),
            expr: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
        }
    })
}

fn lower_f2i_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if mnem != "F2I" || args.len() != 1 {
        return None;
    }
    let unsigned = parts.iter().any(|part| *part == "U32");
    let signed = parts.iter().any(|part| *part == "S32");
    let trunc = parts.iter().any(|part| matches!(*part, "TRUNC" | "RZ"));
    if !trunc || (!unsigned && !signed) {
        return None;
    }
    let func = match unsigned {
        true => "__float2uint_rz",
        false => "__float2int_rz",
    };
    Some(Expr::CallLike {
        func: func.to_string(),
        args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
    })
}

fn lower_frnd_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    if parts.first().copied() != Some("FRND") || args.len() != 1 {
        return None;
    }
    let func = if parts.iter().any(|part| *part == "FLOOR") {
        "floorf"
    } else if parts.iter().any(|part| *part == "TRUNC") {
        "truncf"
    } else if parts.iter().any(|part| *part == "CEIL") {
        "ceilf"
    } else if parts.iter().any(|part| matches!(*part, "NEAR" | "RN")) {
        "rintf"
    } else {
        return None;
    };
    Some(Expr::CallLike {
        func: func.to_string(),
        args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
    })
}

fn lower_fadd_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    if parts.first().copied() != Some("FADD") || args.len() < 2 {
        return None;
    }
    let mut ftz = false;
    let mut rounding_mode = None;
    for part in parts.iter().skip(1) {
        match *part {
            "FTZ" => ftz = true,
            "RM" | "RP" | "RZ" | "RN" => rounding_mode = Some(*part),
            _ => return None,
        }
    }
    if ftz || rounding_mode.is_some() {
        if args.len() != 2 {
            return None;
        }
        return Some(Expr::CallLike {
            // CUDA exposes rounding-mode intrinsics, not FTZ-specific spellings.
            func: match rounding_mode.unwrap_or("RN") {
                "RM" => "__fadd_rd".to_string(),
                "RP" => "__fadd_ru".to_string(),
                "RZ" => "__fadd_rz".to_string(),
                "RN" => "__fadd_rn".to_string(),
                _ => unreachable!(),
            },
            args: vec![
                lower_scalar_expr_with_analysis(&args[0], analysis),
                lower_scalar_expr_with_analysis(&args[1], analysis),
            ],
        });
    }
    lower_add_expr(args, analysis)
}

fn lower_fmul_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    if parts.first().copied() != Some("FMUL") || args.len() != 2 {
        return None;
    }
    let mut ftz = false;
    let mut rounding_mode = None;
    for part in parts.iter().skip(1) {
        match *part {
            "FTZ" => ftz = true,
            "RM" | "RP" | "RZ" | "RN" => rounding_mode = Some(*part),
            _ => return None,
        }
    }
    if ftz || rounding_mode.is_some() {
        return Some(Expr::CallLike {
            func: match rounding_mode.unwrap_or("RN") {
                "RM" => "__fmul_rd".to_string(),
                "RP" => "__fmul_ru".to_string(),
                "RZ" => "__fmul_rz".to_string(),
                "RN" => "__fmul_rn".to_string(),
                _ => unreachable!(),
            },
            args: vec![
                lower_scalar_expr_with_analysis(&args[0], analysis),
                lower_scalar_expr_with_analysis(&args[1], analysis),
            ],
        });
    }
    Some(Expr::Binary {
        op: "*".to_string(),
        lhs: Box::new(lower_scalar_expr_with_analysis(&args[0], analysis)),
        rhs: Box::new(lower_scalar_expr_with_analysis(&args[1], analysis)),
    })
}

fn lower_ffma_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    if parts.first().copied() != Some("FFMA") || args.len() != 3 {
        return None;
    }
    let lowered_args = vec![
        lower_scalar_expr_with_analysis(&args[0], analysis),
        lower_scalar_expr_with_analysis(&args[1], analysis),
        lower_scalar_expr_with_analysis(&args[2], analysis),
    ];
    let base = Expr::CallLike {
        func: "fmaf".to_string(),
        args: lowered_args.clone(),
    };
    let mut ftz = false;
    let mut rounding_mode = None;
    let mut saturate = false;
    for part in parts.iter().skip(1) {
        match *part {
            "FTZ" => ftz = true,
            "SAT" => saturate = true,
            "RM" | "RP" | "RZ" | "RN" => rounding_mode = Some(*part),
            _ => return None,
        }
    }
    let rounded = if let Some(mode) = rounding_mode {
        Expr::CallLike {
            func: match mode {
                "RM" => "__fmaf_rd".to_string(),
                "RP" => "__fmaf_ru".to_string(),
                "RZ" => "__fmaf_rz".to_string(),
                "RN" => "__fmaf_rn".to_string(),
                _ => unreachable!(),
            },
            args: lowered_args,
        }
    } else if ftz {
        Expr::CallLike {
            func: "__fmaf_rn".to_string(),
            args: lowered_args,
        }
    } else {
        base
    };
    if saturate {
        return Some(Expr::CallLike {
            func: "__saturatef".to_string(),
            args: vec![rounded],
        });
    }
    Some(rounded)
}

fn lower_fmnmx_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if opcode != "FMNMX" || args.len() != 3 {
        return None;
    }
    let fmin = Expr::CallLike {
        func: "fminf".to_string(),
        args: vec![
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[1], analysis),
        ],
    };
    let fmax = Expr::CallLike {
        func: "fmaxf".to_string(),
        args: vec![
            lower_scalar_expr_with_analysis(&args[0], analysis),
            lower_scalar_expr_with_analysis(&args[1], analysis),
        ],
    };
    if ir_expr_is_true_pred(&args[2]) {
        return Some(fmin);
    }
    if ir_expr_is_false_pred(&args[2]) {
        return Some(fmax);
    }
    Some(Expr::Ternary {
        cond: Box::new(lower_scalar_expr_with_analysis(&args[2], analysis)),
        then_expr: Box::new(fmin),
        else_expr: Box::new(fmax),
    })
}

fn lower_mufu_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if args.len() != 1 {
        return None;
    }
    let func = if opcode.starts_with("MUFU.RCP") {
        "rcp_approx"
    } else if opcode.starts_with("MUFU.RSQ") {
        "rsqrtf"
    } else if opcode.starts_with("MUFU.EX2") {
        "exp2f"
    } else if opcode.starts_with("MUFU.LG2") {
        "log2f"
    } else if opcode.starts_with("MUFU.SIN") {
        "sinf"
    } else if opcode.starts_with("MUFU.COS") {
        "cosf"
    } else if opcode.starts_with("MUFU.SQRT") {
        "sqrtf"
    } else {
        return None;
    };
    Some(Expr::CallLike {
        func: func.to_string(),
        args: vec![lower_scalar_expr_with_analysis(&args[0], analysis)],
    })
}

fn lower_wide_add_lo_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let wide_value = if opcode == "IADD.64" && args.len() == 4 {
        Some(add_expr(
            lower_wide_input_expr(&args[0], &args[2], analysis),
            lower_wide_input_expr(&args[1], &args[3], analysis),
        ))
    } else if matches!(opcode, "IADD3.64" | "UIADD3.64") && args.len() == 6 {
        Some(add_expr(
            add_expr(
                lower_wide_input_expr(&args[0], &args[3], analysis),
                lower_wide_input_expr(&args[1], &args[4], analysis),
            ),
            lower_wide_input_expr(&args[2], &args[5], analysis),
        ))
    } else {
        None
    }?;
    Some(Expr::LaneExtract {
        value: Box::new(wide_value),
        lane: PointerLane::Lo32,
    })
}

fn lower_wide_input_expr(lo: &IRExpr, hi: &IRExpr, analysis: Option<&FunctionAnalysis>) -> Expr {
    if let Some(rooted) = analysis.and_then(|facts| lower_rooted_wide_expr(lo, hi, facts)) {
        return rooted;
    }
    if ir_expr_is_zero(hi) {
        return Expr::Cast {
            ty: "uint64_t".to_string(),
            expr: Box::new(Expr::Cast {
                ty: "uint32_t".to_string(),
                expr: Box::new(lower_scalar_expr_with_analysis(lo, analysis)),
            }),
        };
    }
    let lo = lower_scalar_expr_with_analysis(lo, analysis);
    let hi = lower_scalar_expr_with_analysis(hi, analysis);
    lower_named_pointer_lane_wide_expr(lo.clone(), hi.clone()).unwrap_or(Expr::Addr64 {
        lo: Box::new(lo),
        hi: Box::new(hi),
    })
}

fn lower_rooted_wide_expr(lo: &IRExpr, hi: &IRExpr, analysis: &FunctionAnalysis) -> Option<Expr> {
    let lo_reg = lo.get_reg()?;
    let root = analysis.root_by_reg.get(lo_reg)?;
    let base = rooted_wide_base_expr(root, analysis)?;
    let hi_matches_root = match hi {
        IRExpr::Reg(hi_reg) => analysis
            .root_by_reg
            .get(hi_reg)
            .is_some_and(|hi_root| compatible_wide_roots(root, hi_root)),
        other => ir_expr_is_zero(other),
    };
    if !hi_matches_root {
        return None;
    }
    let byte_offset = analysis.byte_offset_by_reg.get(lo_reg)?;
    let expanded = expand_rooted_offset_expr(byte_offset, root, analysis, &mut BTreeSet::new());
    Some(Expr::WidePtr {
        base: Box::new(base),
        offset: Box::new(lower_scalar_expr_with_analysis(&expanded, Some(analysis))),
    })
}

fn rooted_wide_base_expr(root: &AddressRoot, analysis: &FunctionAnalysis) -> Option<Expr> {
    match root {
        AddressRoot::ParamWord(param_idx) => {
            Some(Expr::Reg(global_param_base_name(*param_idx, analysis)))
        }
        AddressRoot::SharedObject(name) => Some(Expr::Builtin(name.clone())),
        AddressRoot::LocalObject(name) => Some(Expr::Reg(name.clone())),
        AddressRoot::ConstSymbol(name) => Some(Expr::ConstMemSymbol(name.clone())),
        AddressRoot::RegisterBase(_) | AddressRoot::Generic => None,
    }
}

fn compatible_wide_roots(lo_root: &AddressRoot, hi_root: &AddressRoot) -> bool {
    matches!(
        (lo_root, hi_root),
        (AddressRoot::ParamWord(lo_idx), AddressRoot::ParamWord(hi_idx))
            if *hi_idx == lo_idx.saturating_add(1)
    ) || lo_root == hi_root
}

fn lower_named_pointer_lane_wide_expr(lo: Expr, hi: Expr) -> Option<Expr> {
    match (lo, hi) {
        (
            Expr::PtrLane {
                base: lo_base,
                lane: PointerLane::Lo32,
            },
            Expr::PtrLane {
                base: hi_base,
                lane: PointerLane::Hi32,
            },
        ) if lo_base == hi_base => Some(Expr::WidePtr {
            base: Box::new(Expr::Reg(lo_base)),
            offset: Box::new(Expr::Imm("0".to_string())),
        }),
        _ => None,
    }
}

fn lower_imad_hi_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    let mnem = parts.first().copied().unwrap_or(opcode);
    if !matches!(mnem, "IMAD" | "UIMAD")
        || !parts.iter().any(|part| *part == "HI")
        || args.len() != 3
    {
        return None;
    }
    let wide_lhs = if parts.iter().any(|part| *part == "U32") {
        widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(&args[0], analysis))
    } else {
        widen_sx32_to_i64_expr(lower_scalar_expr_with_analysis(&args[0], analysis))
    };
    let wide_rhs = if parts.iter().any(|part| *part == "U32") {
        widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(&args[1], analysis))
    } else {
        widen_sx32_to_i64_expr(lower_scalar_expr_with_analysis(&args[1], analysis))
    };
    let hi = Expr::LaneExtract {
        value: Box::new(Expr::Binary {
            op: "*".to_string(),
            lhs: Box::new(wide_lhs),
            rhs: Box::new(wide_rhs),
        }),
        lane: PointerLane::Hi32,
    };
    if ir_expr_is_zero(&args[2]) {
        return Some(hi);
    }
    Some(add_expr(
        hi,
        lower_scalar_expr_with_analysis(&args[2], analysis),
    ))
}

fn add_expr(lhs: Expr, rhs: Expr) -> Expr {
    if matches!(lhs, Expr::Imm(ref text) if text == "0") {
        return rhs;
    }
    if matches!(rhs, Expr::Imm(ref text) if text == "0") {
        return lhs;
    }
    match (lhs, rhs) {
        (Expr::WidePtr { base, offset }, rhs) if !matches!(rhs, Expr::WidePtr { .. }) => {
            Expr::WidePtr {
                base,
                offset: Box::new(add_expr(*offset, rhs)),
            }
        }
        (lhs, Expr::WidePtr { base, offset }) if !matches!(lhs, Expr::WidePtr { .. }) => {
            Expr::WidePtr {
                base,
                offset: Box::new(add_expr(lhs, *offset)),
            }
        }
        (lhs, rhs) => Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
    }
}

fn lower_lea_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "LEA" | "ULEA") || opcode.split('.').any(|part| part == "HI") {
        return None;
    }
    let [base, offset, shift] = args else {
        return None;
    };
    let scaled = lower_shifted_u32_expr(base, shift, analysis)?;
    Some(add_expr(
        scaled,
        lower_scalar_expr_with_analysis(offset, analysis),
    ))
}

fn lower_lea_hi_expr(
    opcode: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !matches!(mnem, "LEA" | "ULEA") || !opcode.split('.').any(|part| part == "HI") {
        return None;
    }
    let sx32 = opcode.split('.').any(|part| part == "SX32");
    match args {
        [base, offset, accum, shift] if matches!(shift, IRExpr::ImmI(_)) => {
            let wide_sum = add_expr(
                add_expr(
                    widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(base, analysis)),
                    lower_shifted_wide_expr(offset, shift, sx32, analysis)?,
                ),
                widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(accum, analysis)),
            );
            Some(Expr::LaneExtract {
                value: Box::new(wide_sum),
                lane: PointerLane::Hi32,
            })
        }
        [offset, ptr_hi, shift, carry]
            if matches!(shift, IRExpr::ImmI(_)) && is_predicate_expr(carry) =>
        {
            Some(add_lea_hi_terms(vec![
                Expr::LaneExtract {
                    value: Box::new(lower_shifted_wide_expr(offset, shift, sx32, analysis)?),
                    lane: PointerLane::Hi32,
                },
                lower_scalar_expr_with_analysis(ptr_hi, analysis),
                lower_carry_inc_expr(carry, analysis)?,
            ]))
        }
        [offset, ptr_hi, accum_hi, shift, carry]
            if matches!(shift, IRExpr::ImmI(_)) && is_predicate_expr(carry) =>
        {
            Some(add_lea_hi_terms(vec![
                Expr::LaneExtract {
                    value: Box::new(lower_shifted_wide_expr(offset, shift, sx32, analysis)?),
                    lane: PointerLane::Hi32,
                },
                lower_scalar_expr_with_analysis(ptr_hi, analysis),
                lower_scalar_expr_with_analysis(accum_hi, analysis),
                lower_carry_inc_expr(carry, analysis)?,
            ]))
        }
        _ => None,
    }
}

fn lower_lea_carry_expr(args: &[IRExpr], analysis: Option<&FunctionAnalysis>) -> Option<Expr> {
    let [base, offset, shift] = args else {
        return None;
    };
    let wide_sum = add_expr(
        widen_u32_to_u64_expr(lower_shifted_u32_expr(base, shift, analysis)?),
        widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(offset, analysis)),
    );
    Some(Expr::Binary {
        op: "!=".to_string(),
        lhs: Box::new(Expr::LaneExtract {
            value: Box::new(wide_sum),
            lane: PointerLane::Hi32,
        }),
        rhs: Box::new(Expr::Imm("0".to_string())),
    })
}

fn lower_shifted_u32_expr(
    value: &IRExpr,
    shift: &IRExpr,
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let shift = match shift {
        IRExpr::ImmI(value) => *value,
        _ => return None,
    };
    let base = lower_scalar_expr_with_analysis(value, analysis);
    if shift == 0 {
        return Some(base);
    }
    Some(Expr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(base),
        rhs: Box::new(Expr::Imm(shift.to_string())),
    })
}

fn lower_shifted_wide_expr(
    value: &IRExpr,
    shift: &IRExpr,
    sx32: bool,
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    let shift = match shift {
        IRExpr::ImmI(value) => *value,
        _ => return None,
    };
    let base = if sx32 {
        widen_sx32_to_i64_expr(lower_scalar_expr_with_analysis(value, analysis))
    } else {
        widen_u32_to_u64_expr(lower_scalar_expr_with_analysis(value, analysis))
    };
    if shift == 0 {
        return Some(base);
    }
    Some(Expr::Binary {
        op: "<<".to_string(),
        lhs: Box::new(base),
        rhs: Box::new(Expr::Imm(shift.to_string())),
    })
}

fn widen_u32_to_u64_expr(expr: Expr) -> Expr {
    Expr::Cast {
        ty: "uint64_t".to_string(),
        expr: Box::new(Expr::Cast {
            ty: "uint32_t".to_string(),
            expr: Box::new(expr),
        }),
    }
}

fn widen_sx32_to_i64_expr(expr: Expr) -> Expr {
    Expr::Cast {
        ty: "int64_t".to_string(),
        expr: Box::new(Expr::Cast {
            ty: "int32_t".to_string(),
            expr: Box::new(expr),
        }),
    }
}

fn add_lea_hi_terms(mut terms: Vec<Expr>) -> Expr {
    terms.retain(|expr| !matches!(expr, Expr::Imm(text) if text == "0"));
    let mut iter = terms.into_iter();
    let first = iter.next().unwrap_or_else(|| Expr::Imm("0".to_string()));
    iter.fold(first, add_expr)
}

fn is_imm_i(expr: &IRExpr, value: i64) -> bool {
    matches!(expr, IRExpr::ImmI(actual) if *actual == value)
}

fn lower_ir_op_expr_with_analysis(
    op: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Expr {
    if let Some(expr) = lower_constmem_ir_op_expr(op, args, analysis) {
        return expr;
    }
    if let Some(expr) = lower_builtin_expr(op, args) {
        return expr;
    }
    if let Some(expr) = lower_simple_ir_op_expr(op, args, analysis) {
        return expr;
    }
    Expr::CallLike {
        func: op.to_string(),
        args: args
            .iter()
            .map(|arg| lower_scalar_expr_with_analysis(arg, analysis))
            .collect(),
    }
}

fn lower_constmem_ir_op_expr(
    op: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    if op != "ConstMem" || args.len() != 2 {
        return None;
    }
    let bank = expr_i64(&args[0]).and_then(|value| u32::try_from(value).ok())?;
    let offset = expr_i64(&args[1]).and_then(|value| u32::try_from(value).ok())?;
    let semantic = analysis
        .and_then(|facts| facts.abi_profile)
        .map(|profile| profile.classify_constmem(bank, offset))
        .unwrap_or(ConstMemSemantic::Unknown { bank, offset });
    Some(match analysis {
        Some(facts) => constmem_semantic_expr(&semantic, facts),
        None => Expr::ConstMemSymbol(format!("c[0x{:x}][0x{:x}]", bank, offset)),
    })
}

fn lower_builtin_expr(op: &str, args: &[IRExpr]) -> Option<Expr> {
    if !args.is_empty() {
        return None;
    }
    let name = match op {
        "PT" | "UPT" => return Some(Expr::Imm("true".to_string())),
        "!PT" | "!UPT" => return Some(Expr::Imm("false".to_string())),
        "+INF" => return Some(Expr::Raw("INFINITY".to_string())),
        "-INF" => return Some(Expr::Raw("-INFINITY".to_string())),
        "+QNAN" => return Some(Expr::Raw("NAN".to_string())),
        "-QNAN" => return Some(Expr::Raw("-NAN".to_string())),
        "SR_CTAID.X" => "blockIdx.x",
        "SR_CTAID.Y" => "blockIdx.y",
        "SR_CTAID.Z" => "blockIdx.z",
        "SR_TID.X" => "threadIdx.x",
        "SR_TID.Y" => "threadIdx.y",
        "SR_TID.Z" => "threadIdx.z",
        "SR_NTID.X" => "blockDim.x",
        "SR_NTID.Y" => "blockDim.y",
        "SR_NTID.Z" => "blockDim.z",
        "SR_GRIDID" => "gridId",
        "SR_LANEID" => "laneId",
        "SRZ" => "0",
        _ => return None,
    };
    Some(Expr::Builtin(name.to_string()))
}

fn lower_simple_ir_op_expr(
    op: &str,
    args: &[IRExpr],
    analysis: Option<&FunctionAnalysis>,
) -> Option<Expr> {
    match (op, args) {
        ("!", [arg]) | ("-", [arg]) => Some(Expr::Unary {
            op: op.to_string(),
            arg: Box::new(lower_scalar_expr_with_analysis(arg, analysis)),
        }),
        (binary_op, [lhs, rhs]) if is_simple_binary_op(binary_op) => Some(Expr::Binary {
            op: binary_op.to_string(),
            lhs: Box::new(lower_scalar_expr_with_analysis(lhs, analysis)),
            rhs: Box::new(lower_scalar_expr_with_analysis(rhs, analysis)),
        }),
        _ => None,
    }
}

fn is_simple_binary_op(op: &str) -> bool {
    matches!(
        op,
        "+" | "-"
            | "*"
            | "/"
            | "%"
            | "&"
            | "|"
            | "^"
            | "<<"
            | ">>"
            | "&&"
            | "||"
            | "=="
            | "!="
            | "<"
            | "<="
            | ">"
            | ">="
    )
}

fn lower_reg_expr(reg: &crate::ir::RegId) -> Expr {
    match reg.class.as_str() {
        "RZ" | "URZ" => Expr::Imm("0".to_string()),
        "PT" | "UPT" => Expr::Imm("true".to_string()),
        _ if reg.sign < 0 => Expr::Unary {
            op: "-".to_string(),
            arg: Box::new(Expr::Reg(lower_reg_name(reg))),
        },
        _ => Expr::Reg(lower_reg_name(reg)),
    }
}

fn lower_reg_name(reg: &crate::ir::RegId) -> String {
    canonical_reg_ident(reg)
}

fn ir_expr_is_zero(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::ImmI(0) => true,
        IRExpr::Reg(reg) => matches!(reg.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn ir_expr_is_true_pred(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::Reg(reg) if matches!(reg.class.as_str(), "PT" | "UPT"))
        || matches!(expr, IRExpr::Op { op, args } if args.is_empty() && matches!(op.as_str(), "PT" | "UPT"))
}

fn ir_expr_is_false_pred(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::Op { op, args } if args.is_empty() && matches!(op.as_str(), "!PT" | "!UPT"))
        || matches!(expr, IRExpr::Op { op, args } if op == "!" && args.len() == 1 && ir_expr_is_true_pred(&args[0]))
}

fn named_param_word_expr(name: &str) -> Expr {
    if let Some((base, lane)) = PointerLane::parse_named(name) {
        Expr::PtrLane { base, lane }
    } else {
        Expr::Reg(name.to_string())
    }
}

fn rooted_space_byte_offset(
    addr_base: &IRExpr,
    access: &MemAccessInfo,
    analysis: &FunctionAnalysis,
) -> Option<IRExpr> {
    let expected_root = match &access.root {
        AddressRoot::ParamWord(param_idx) => AddressRoot::ParamWord(*param_idx),
        AddressRoot::RegisterBase(reg) => analysis
            .root_by_reg
            .get(reg)
            .cloned()
            .unwrap_or_else(|| AddressRoot::RegisterBase(reg.clone())),
        AddressRoot::SharedObject(name) => AddressRoot::SharedObject(name.clone()),
        AddressRoot::LocalObject(name) => AddressRoot::LocalObject(name.clone()),
        AddressRoot::ConstSymbol(name) => AddressRoot::ConstSymbol(name.clone()),
        AddressRoot::Generic => AddressRoot::Generic,
    };
    let byte_offset = match addr_base {
        IRExpr::Reg(reg) => analysis.byte_offset_by_reg.get(reg).cloned(),
        IRExpr::Addr64 { lo, .. } => lo
            .get_reg()
            .and_then(|reg| analysis.byte_offset_by_reg.get(reg))
            .cloned(),
        _ => None,
    }?;
    Some(expand_rooted_offset_expr(
        &byte_offset,
        &expected_root,
        analysis,
        &mut BTreeSet::new(),
    ))
}

fn expand_rooted_offset_expr(
    expr: &IRExpr,
    expected_root: &AddressRoot,
    analysis: &FunctionAnalysis,
    seen: &mut BTreeSet<crate::ir::RegId>,
) -> IRExpr {
    match expr {
        IRExpr::Reg(reg) if can_expand_rooted_reg(reg, expected_root, analysis) => {
            if !seen.insert(reg.clone()) {
                return IRExpr::Reg(reg.clone());
            }
            let expanded = analysis
                .byte_offset_by_reg
                .get(reg)
                .map(|inner| expand_rooted_offset_expr(inner, expected_root, analysis, seen))
                .unwrap_or_else(|| IRExpr::Reg(reg.clone()));
            seen.remove(reg);
            expanded
        }
        IRExpr::Addr64 { lo, hi } => IRExpr::Addr64 {
            lo: Box::new(expand_rooted_offset_expr(lo, expected_root, analysis, seen)),
            hi: Box::new(expand_rooted_offset_expr(hi, expected_root, analysis, seen)),
        },
        IRExpr::Mem {
            base,
            offset,
            width,
        } => IRExpr::Mem {
            base: Box::new(expand_rooted_offset_expr(
                base,
                expected_root,
                analysis,
                seen,
            )),
            offset: offset.as_ref().map(|expr| {
                Box::new(expand_rooted_offset_expr(
                    expr,
                    expected_root,
                    analysis,
                    seen,
                ))
            }),
            width: *width,
        },
        IRExpr::Op { op, args } => IRExpr::Op {
            op: op.clone(),
            args: args
                .iter()
                .map(|arg| expand_rooted_offset_expr(arg, expected_root, analysis, seen))
                .collect(),
        },
        IRExpr::ImmI(_) | IRExpr::ImmF(_) | IRExpr::Reg(_) => expr.clone(),
    }
}

fn can_expand_rooted_reg(
    reg: &crate::ir::RegId,
    expected_root: &AddressRoot,
    analysis: &FunctionAnalysis,
) -> bool {
    matches!(reg.class.as_str(), "R" | "UR")
        && analysis
            .root_by_reg
            .get(reg)
            .is_some_and(|root| root == expected_root)
}

fn select_memory_result_def(stmt: &IRStatement) -> Option<(usize, &crate::ir::RegId)> {
    select_non_memory_result_def(stmt)
}

fn select_non_memory_result_def(stmt: &IRStatement) -> Option<(usize, &crate::ir::RegId)> {
    stmt.defs
        .iter()
        .enumerate()
        .filter_map(|(idx, def)| {
            let reg = def.get_reg()?;
            (!is_sink_reg(reg)).then_some((idx, reg))
        })
        .next()
}

fn is_sink_reg(reg: &crate::ir::RegId) -> bool {
    matches!(reg.class.as_str(), "PT" | "UPT" | "RZ" | "URZ")
}

fn apply_stmt_predicate(stmt: &IRStatement, lowered: LoweredStmt) -> Stmt {
    let LoweredStmt {
        stmt: lowered_stmt,
        predicate_def_indices,
    } = lowered;
    if matches!(lowered_stmt, Stmt::Empty) {
        return Stmt::Empty;
    }
    let Some(pred) = &stmt.pred else {
        return lowered_stmt;
    };
    if predicate_def_indices.len() == 1 {
        if let Some(def_idx) = predicate_def_indices.first().copied() {
            if let Some(old_def) = stmt.pred_old_defs.get(def_idx) {
                if let Stmt::Assign { dst, src } = lowered_stmt {
                    return Stmt::Assign {
                        dst,
                        src: Expr::Ternary {
                            cond: Box::new(lower_scalar_expr(pred)),
                            then_expr: Box::new(src),
                            else_expr: Box::new(lower_scalar_expr(old_def)),
                        },
                    };
                }
            }
        }
    } else if !predicate_def_indices.is_empty() {
        if let Stmt::Sequence(stmts) = &lowered_stmt {
            let rewritten = stmts
                .iter()
                .cloned()
                .zip(predicate_def_indices.into_iter())
                .map(|(lowered_stmt, def_idx)| {
                    let old_def = stmt.pred_old_defs.get(def_idx)?;
                    let Stmt::Assign { dst, src } = lowered_stmt else {
                        return None;
                    };
                    Some(Stmt::Assign {
                        dst,
                        src: Expr::Ternary {
                            cond: Box::new(lower_scalar_expr(pred)),
                            then_expr: Box::new(src),
                            else_expr: Box::new(lower_scalar_expr(old_def)),
                        },
                    })
                })
                .collect::<Option<Vec<_>>>();
            if let Some(rewritten) = rewritten {
                return Stmt::Sequence(rewritten);
            }
        }
    }
    Stmt::If {
        condition: lower_scalar_expr(pred),
        then_branch: Box::new(lowered_stmt),
        else_branch: None,
    }
}

fn constmem_semantic_expr(semantic: &ConstMemSemantic, analysis: &FunctionAnalysis) -> Expr {
    match semantic {
        ConstMemSemantic::ParamWord {
            param_idx,
            word_idx,
        } => {
            let rendered = analysis
                .abi_aliases
                .render_param_word(*param_idx, *word_idx)
                .unwrap_or_else(|| format!("param_{}", param_idx.saturating_add(*word_idx)));
            named_param_word_expr(&rendered)
        }
        ConstMemSemantic::Builtin(name) => Expr::Builtin((*name).to_string()),
        ConstMemSemantic::AbiInternal(offset) => {
            Expr::ConstMemSymbol(format!("abi_internal_0x{:x}", offset))
        }
        ConstMemSemantic::Unknown { bank, offset } => {
            Expr::ConstMemSymbol(format!("c[0x{:x}][0x{:x}]", bank, offset))
        }
    }
}

fn expr_i64(expr: &IRExpr) -> Option<i64> {
    match expr {
        IRExpr::ImmI(value) => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{analyze_function_ir, build_cfg, build_ssa, decode_sass};

    fn analyze_stmt(
        sass: &str,
        pred: impl Fn(&IRStatement) -> bool,
    ) -> (FunctionAnalysis, usize, usize, IRStatement) {
        let instrs = decode_sass(sass);
        let cfg = build_cfg(instrs.clone());
        let fir = build_ssa(&cfg);
        let analysis = analyze_function_ir(&fir, &instrs, None);
        for block in &fir.blocks {
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                if pred(stmt) {
                    return (analysis, block.id, stmt_idx, stmt.clone());
                }
            }
        }
        panic!("statement not found");
    }

    fn analyze_sass(sass: &str) -> FunctionAnalysis {
        let instrs = decode_sass(sass);
        let cfg = build_cfg(instrs.clone());
        let fir = build_ssa(&cfg);
        analyze_function_ir(&fir, &instrs, None)
    }

    #[test]
    fn lowers_global_param_rooted_loads_to_index_exprs() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E R6, [R4.64+0x4] ;
            /*0030*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("LDG"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(src.render(), "arg0_ptr[1]");
    }

    #[test]
    fn lowers_pointer_arith_global_loads_from_rooted_base_offsets() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ IADD3 R4, R4, 0x4, RZ ;
            /*0030*/ LDG.E R6, [R4.64] ;
            /*0040*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("LDG"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(src.render(), "arg0_ptr[1]");
    }

    #[test]
    fn expands_nested_rooted_offsets_in_global_index_lowering() {
        let rooted = crate::ir::RegId::new("R", 10, 1).with_ssa(0);
        let nested = crate::ir::RegId::new("R", 8, 1).with_ssa(0);
        let mem_expr = IRExpr::Mem {
            base: Box::new(IRExpr::Reg(nested.clone())),
            offset: Some(Box::new(IRExpr::ImmI(4))),
            width: Some(32),
        };
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "LDG.E".to_string(),
                args: vec![mem_expr.clone()],
            },
            pred: None,
            mem_addr_args: Some(vec![mem_expr]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis
            .root_by_reg
            .insert(rooted.clone(), AddressRoot::ParamWord(0));
        analysis.byte_offset_by_reg.insert(
            rooted.clone(),
            IRExpr::Op {
                op: "+".to_string(),
                args: vec![IRExpr::ImmI(32), IRExpr::ImmI(2)],
            },
        );
        analysis
            .root_by_reg
            .insert(nested.clone(), AddressRoot::ParamWord(0));
        analysis.byte_offset_by_reg.insert(
            nested,
            IRExpr::Op {
                op: "+".to_string(),
                args: vec![IRExpr::Reg(rooted), IRExpr::ImmI(64)],
            },
        );
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Global,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(4),
            has_dynamic_offset: true,
            root: AddressRoot::ParamWord(0),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        let rendered = src.render();
        assert!(
            rendered.contains("arg0_ptr[") || rendered.contains("param_0["),
            "expected a rooted param base, got: {rendered}"
        );
        assert!(
            !rendered.contains("r8_0"),
            "nested rooted temp leaked: {rendered}"
        );
        assert!(
            !rendered.contains("r10_0"),
            "rooted temp leaked: {rendered}"
        );
    }

    #[test]
    fn expands_rooted_shared_offsets_before_scaling_indices() {
        let rooted = crate::ir::RegId::new("R", 19, 1).with_ssa(0);
        let mem_expr = IRExpr::Mem {
            base: Box::new(IRExpr::Reg(rooted.clone())),
            offset: Some(Box::new(IRExpr::ImmI(528))),
            width: Some(32),
        };
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "LDS".to_string(),
                args: vec![mem_expr.clone()],
            },
            pred: None,
            mem_addr_args: Some(vec![mem_expr]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.root_by_reg.insert(
            rooted.clone(),
            AddressRoot::SharedObject("shmem".to_string()),
        );
        analysis.byte_offset_by_reg.insert(
            rooted,
            IRExpr::Op {
                op: "*".to_string(),
                args: vec![
                    IRExpr::Op {
                        op: "SR_TID.X".to_string(),
                        args: Vec::new(),
                    },
                    IRExpr::ImmI(4),
                ],
            },
        );
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(528),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "shmem[threadIdx.x + 132]");
    }

    #[test]
    fn lowers_shared_store_to_indexed_lvalue() {
        let sass = r#"
            /*0000*/ STS [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(dst.render(), "shmem[(r2_0 + 8) / 4]");
    }

    #[test]
    fn lowers_scaled_shared_thread_indices_to_element_offsets() {
        let sass = r#"
            /*0000*/ S2R R7, SR_TID.X ;
            /*0010*/ STS [R7.X4+0x10], R4 ;
            /*0020*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, .. } = lowered else {
            panic!("expected assign");
        };
        assert_eq!(dst.render(), "shmem[threadIdx.x + 4]");
    }

    #[test]
    fn lowers_dynamic_local_stores_to_explicit_helpers() {
        let sass = r#"
            /*0000*/ STL [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("STL"));
        let lowered = lower_basic_stmt(block_id, stmt_idx, &stmt, &analysis);
        let Stmt::ExprStmt(expr) = lowered else {
            panic!("expected explicit helper call");
        };
        assert_eq!(expr.render(), "local_store_bits32(r2_0 + 8, r4_0)");
    }

    #[test]
    fn lowers_shared_atomics_to_explicit_calls() {
        let sass = r#"
            /*0000*/ ATOMS.ADD R0, [R2], R4 ;
            /*0010*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) =
            analyze_stmt(sass, |stmt| stmt_opcode(stmt).starts_with("ATOMS"));
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { dst, src } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(dst.render(), "r0_0");
        assert!(src.render().contains("atomicAdd"));
        assert!(src.render().contains("&shmem[r2_0 / 4]"));
    }

    #[test]
    fn lowers_popc_inc_shared_atomics_to_atomic_add_one() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "ATOMS.POPC.INC".to_string(),
                args: vec![IRExpr::Mem {
                    base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                    offset: None,
                    width: Some(32),
                }],
            },
            pred: None,
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                offset: None,
                width: Some(32),
            }]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Atomic,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "atomicAdd(&shmem[r2 / 4], 1)");
    }

    #[test]
    fn lowers_param_window_loads_to_kernel_param_symbols() {
        let sass = r#"
            /*0000*/ LDC R4, c[0x0][0x160] ;
            /*0010*/ LDC R5, c[0x0][0x164] ;
            /*0020*/ EXIT ;
        "#;
        let (analysis, block_id, stmt_idx, stmt) = analyze_stmt(sass, |stmt| {
            stmt_opcode(stmt).starts_with("LDC")
                && stmt
                    .defs
                    .first()
                    .and_then(IRExpr::get_reg)
                    .is_some_and(|reg| reg.idx == 4)
        });
        let lowered = lower_memory_stmt(block_id, stmt_idx, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert!(src.render().starts_with("arg0"));
        assert!(!src.render().contains("ConstMem"));
    }

    #[test]
    fn lowers_predicated_defs_to_ternary_assignments() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(2))],
            value: RValue::Op {
                opcode: "MOV".to_string(),
                args: vec![IRExpr::ImmI(1)],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: None,
            pred_old_defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(1))],
        };
        let lowered = lower_basic_stmt(0, 0, &stmt, &FunctionAnalysis::default());
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "p0 ? 1 : r4_1");
    }

    #[test]
    fn lowers_predicated_atomic_results_with_false_path_value() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(1))],
            value: RValue::Op {
                opcode: "ATOMS.ADD".to_string(),
                args: vec![
                    IRExpr::Mem {
                        base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                        offset: None,
                        width: Some(32),
                    },
                    IRExpr::Reg(crate::ir::RegId::new("R", 4, 1)),
                ],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 2, 1))),
                offset: None,
                width: Some(32),
            }]),
            pred_old_defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Atomic,
            space: CudaMemorySpace::Shared,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::SharedObject("shmem".to_string()),
        });
        let lowered = lower_basic_stmt(0, 0, &stmt, &analysis);
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "p0 ? (atomicAdd(&shmem[r2 / 4], r4)) : r0_0");
    }

    #[test]
    fn lowers_multi_def_atomic_results_to_data_reg_and_preserves_false_path() {
        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(1)),
            ],
            value: RValue::Op {
                opcode: "ATOMG.E.ADD.STRONG.GPU".to_string(),
                args: vec![
                    IRExpr::Mem {
                        base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1))),
                        offset: None,
                        width: Some(64),
                    },
                    IRExpr::Reg(crate::ir::RegId::new("R", 13, 1)),
                ],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 2, 1))),
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1))),
                offset: None,
                width: Some(64),
            }]),
            pred_old_defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
            ],
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Atomic,
            space: CudaMemorySpace::Global,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::RegisterBase(crate::ir::RegId::new("R", 4, 1)),
        });
        let lowered = lower_basic_stmt(0, 0, &stmt, &analysis);
        let Stmt::Assign { dst, src } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(dst.render(), "r7_1");
        assert_eq!(
            src.render(),
            "p2 ? (atomicAdd(&((uint32_t*)(r4))[0], r13)) : r7_0"
        );
    }

    #[test]
    fn lowers_exit_barrier_and_indirect_branch_ops_without_raw_helpers() {
        let exit_stmt = IRStatement {
            defs: Vec::new(),
            value: RValue::Op {
                opcode: "EXIT".to_string(),
                args: Vec::new(),
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let exit_lowered = lower_basic_stmt(0, 0, &exit_stmt, &FunctionAnalysis::default());
        assert_eq!(exit_lowered.render_with_indent(0), "if (p0) return;\n");

        let barrier_stmt = IRStatement {
            defs: Vec::new(),
            value: RValue::Op {
                opcode: "BAR.SYNC".to_string(),
                args: Vec::new(),
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let barrier_lowered = lower_basic_stmt(0, 0, &barrier_stmt, &FunctionAnalysis::default());
        assert_eq!(barrier_lowered.render_with_indent(0), "__syncthreads();\n");

        let brx_stmt = IRStatement {
            defs: Vec::new(),
            value: RValue::Op {
                opcode: "BRX".to_string(),
                args: vec![IRExpr::Reg(crate::ir::RegId::new("R", 8, 1).with_ssa(0))],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 1, 1))),
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let brx_lowered = lower_basic_stmt(0, 0, &brx_stmt, &FunctionAnalysis::default());
        let rendered = brx_lowered.render_with_indent(0);
        assert!(rendered.contains("if (p1)"), "got:\n{rendered}");
        assert!(rendered.contains("switch (r8_0)"), "got:\n{rendered}");
        assert!(!rendered.contains("BRX("), "got:\n{rendered}");
    }

    #[test]
    fn lowers_structured_if_and_memory_load_body() {
        let analysis = FunctionAnalysis::default();
        let structured = StructuredStatement::If {
            condition_block_id: 0,
            condition_expr: IRExpr::Reg(crate::ir::RegId::new("P", 0, 1)),
            then_branch: Box::new(StructuredStatement::BasicBlock {
                block_id: 1,
                stmts: vec![IRStatement {
                    defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1))],
                    value: RValue::Op {
                        opcode: "MOV".to_string(),
                        args: vec![IRExpr::ImmI(1)],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                }],
            }),
            else_branch: None,
        };
        let lowered = lower_structured_stmt(
            &structured,
            &crate::cfg::ControlFlowGraph::new(),
            &crate::ir::FunctionIR { blocks: Vec::new() },
            &analysis,
        );
        let rendered = lowered.render_with_indent(0);
        assert!(rendered.contains("if (p0)"));
        assert!(rendered.contains("r4 = 1;"));
    }

    #[test]
    fn lowers_phi_connectors_structurally_across_if_edges() {
        let mut cfg = crate::cfg::ControlFlowGraph::new();
        let bb0 = cfg.add_node(crate::cfg::BasicBlock {
            id: 0,
            start: 0x00,
            instrs: Vec::new(),
        });
        let bb1 = cfg.add_node(crate::cfg::BasicBlock {
            id: 1,
            start: 0x10,
            instrs: Vec::new(),
        });
        let bb2 = cfg.add_node(crate::cfg::BasicBlock {
            id: 2,
            start: 0x20,
            instrs: Vec::new(),
        });
        let bb3 = cfg.add_node(crate::cfg::BasicBlock {
            id: 3,
            start: 0x30,
            instrs: Vec::new(),
        });
        cfg.add_edge(bb0, bb1, crate::cfg::EdgeKind::CondBranch);
        cfg.add_edge(bb0, bb2, crate::cfg::EdgeKind::FallThrough);
        cfg.add_edge(bb1, bb3, crate::cfg::EdgeKind::UncondBranch);
        cfg.add_edge(bb2, bb3, crate::cfg::EdgeKind::UncondBranch);

        let pred = crate::ir::RegId::new("P", 0, 1);
        let then_val = crate::ir::RegId::new("R", 4, 1).with_ssa(1);
        let else_val = crate::ir::RegId::new("R", 4, 1).with_ssa(2);
        let phi_val = crate::ir::RegId::new("R", 5, 1).with_ssa(1);
        let use_val = crate::ir::RegId::new("R", 6, 1).with_ssa(1);
        let fir = crate::ir::FunctionIR {
            blocks: vec![
                crate::ir::IRBlock {
                    id: 0,
                    start_addr: 0x00,
                    irdst: vec![
                        (
                            Some(crate::ir::IRCond::Pred {
                                reg: pred.clone(),
                                sense: true,
                            }),
                            0x10,
                        ),
                        (
                            Some(crate::ir::IRCond::Pred {
                                reg: pred.clone(),
                                sense: false,
                            }),
                            0x20,
                        ),
                    ],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(pred.clone())],
                        value: RValue::ImmI(1),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
                crate::ir::IRBlock {
                    id: 1,
                    start_addr: 0x10,
                    irdst: vec![(Some(crate::ir::IRCond::True), 0x30)],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(then_val.clone())],
                        value: RValue::ImmI(11),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
                crate::ir::IRBlock {
                    id: 2,
                    start_addr: 0x20,
                    irdst: vec![(Some(crate::ir::IRCond::True), 0x30)],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(else_val.clone())],
                        value: RValue::ImmI(22),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
                crate::ir::IRBlock {
                    id: 3,
                    start_addr: 0x30,
                    irdst: vec![],
                    stmts: vec![
                        IRStatement {
                            defs: vec![IRExpr::Reg(phi_val.clone())],
                            value: RValue::Phi(vec![
                                IRExpr::Reg(then_val.clone()),
                                IRExpr::Reg(else_val.clone()),
                            ]),
                            pred: None,
                            mem_addr_args: None,
                            pred_old_defs: Vec::new(),
                        },
                        IRStatement {
                            defs: vec![IRExpr::Reg(use_val)],
                            value: RValue::Op {
                                opcode: "MOV".to_string(),
                                args: vec![IRExpr::Reg(phi_val)],
                            },
                            pred: None,
                            mem_addr_args: None,
                            pred_old_defs: Vec::new(),
                        },
                    ],
                },
            ],
        };
        let structured = StructuredStatement::Sequence(vec![
            StructuredStatement::If {
                condition_block_id: 0,
                condition_expr: IRExpr::Reg(pred),
                then_branch: Box::new(StructuredStatement::BasicBlock {
                    block_id: 1,
                    stmts: fir.blocks[1].stmts.clone(),
                }),
                else_branch: Some(Box::new(StructuredStatement::BasicBlock {
                    block_id: 2,
                    stmts: fir.blocks[2].stmts.clone(),
                })),
            },
            StructuredStatement::BasicBlock {
                block_id: 3,
                stmts: fir.blocks[3].stmts.clone(),
            },
        ]);

        let lowered = lower_structured_stmt(&structured, &cfg, &fir, &FunctionAnalysis::default());
        let rendered = lowered.render_with_indent(0);
        assert!(rendered.contains("p0 = 1;"), "got:\n{rendered}");
        assert!(rendered.contains("r5_1 = r4_1;"), "got:\n{rendered}");
        assert!(rendered.contains("r5_1 = r4_2;"), "got:\n{rendered}");
        assert!(rendered.contains("r6_1 = r5_1;"), "got:\n{rendered}");
        assert!(!rendered.contains("phi("), "got:\n{rendered}");
    }

    #[test]
    fn lowers_if_entry_phi_inside_jump_target_label() {
        let mut cfg = crate::cfg::ControlFlowGraph::new();
        let bb0 = cfg.add_node(crate::cfg::BasicBlock {
            id: 0,
            start: 0x00,
            instrs: Vec::new(),
        });
        let bb1 = cfg.add_node(crate::cfg::BasicBlock {
            id: 1,
            start: 0x10,
            instrs: Vec::new(),
        });
        let bb2 = cfg.add_node(crate::cfg::BasicBlock {
            id: 2,
            start: 0x20,
            instrs: Vec::new(),
        });
        let bb4 = cfg.add_node(crate::cfg::BasicBlock {
            id: 4,
            start: 0x40,
            instrs: Vec::new(),
        });
        cfg.add_edge(bb0, bb1, crate::cfg::EdgeKind::CondBranch);
        cfg.add_edge(bb0, bb2, crate::cfg::EdgeKind::FallThrough);
        cfg.add_edge(bb4, bb1, crate::cfg::EdgeKind::UncondBranch);

        let pred = crate::ir::RegId::new("P", 0, 1);
        let phi_dst = crate::ir::RegId::new("R", 5, 1).with_ssa(1);
        let use_dst = crate::ir::RegId::new("R", 6, 1).with_ssa(1);
        let fir = crate::ir::FunctionIR {
            blocks: vec![
                crate::ir::IRBlock {
                    id: 0,
                    start_addr: 0x00,
                    irdst: vec![
                        (
                            Some(crate::ir::IRCond::Pred {
                                reg: pred.clone(),
                                sense: true,
                            }),
                            0x10,
                        ),
                        (
                            Some(crate::ir::IRCond::Pred {
                                reg: pred.clone(),
                                sense: false,
                            }),
                            0x20,
                        ),
                    ],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(pred.clone())],
                        value: RValue::ImmI(1),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
                crate::ir::IRBlock {
                    id: 1,
                    start_addr: 0x10,
                    irdst: vec![],
                    stmts: vec![
                        IRStatement {
                            defs: vec![IRExpr::Reg(phi_dst.clone())],
                            value: RValue::Phi(vec![IRExpr::ImmI(11), IRExpr::ImmI(99)]),
                            pred: None,
                            mem_addr_args: None,
                            pred_old_defs: Vec::new(),
                        },
                        IRStatement {
                            defs: vec![IRExpr::Reg(use_dst)],
                            value: RValue::Op {
                                opcode: "MOV".to_string(),
                                args: vec![IRExpr::Reg(phi_dst)],
                            },
                            pred: None,
                            mem_addr_args: None,
                            pred_old_defs: Vec::new(),
                        },
                    ],
                },
                crate::ir::IRBlock {
                    id: 2,
                    start_addr: 0x20,
                    irdst: vec![],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(1))],
                        value: RValue::ImmI(22),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
                crate::ir::IRBlock {
                    id: 4,
                    start_addr: 0x40,
                    irdst: vec![(Some(crate::ir::IRCond::True), 0x10)],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 8, 1).with_ssa(1))],
                        value: RValue::ImmI(33),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
            ],
        };
        let structured = StructuredStatement::Sequence(vec![
            StructuredStatement::If {
                condition_block_id: 0,
                condition_expr: IRExpr::Reg(pred),
                then_branch: Box::new(StructuredStatement::BasicBlock {
                    block_id: 1,
                    stmts: fir.blocks[1].stmts.clone(),
                }),
                else_branch: Some(Box::new(StructuredStatement::BasicBlock {
                    block_id: 2,
                    stmts: fir.blocks[2].stmts.clone(),
                })),
            },
            StructuredStatement::BasicBlock {
                block_id: 4,
                stmts: fir.blocks[3].stmts.clone(),
            },
            StructuredStatement::UnstructuredJump {
                from_block_id: 4,
                to_block_id: 1,
                condition: None,
            },
        ]);

        let lowered = lower_structured_stmt(&structured, &cfg, &fir, &FunctionAnalysis::default());
        let rendered = lowered.render_with_indent(0);
        let label_pos = rendered.find("BB1:").expect("branch entry label");
        let phi_pos = rendered.find("r5_1 = 11;").expect("phi prelude");
        assert!(label_pos < phi_pos, "got:\n{rendered}");
        assert!(rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(rendered.contains("r6_1 = r5_1;"), "got:\n{rendered}");
    }

    #[test]
    fn redirects_jumps_away_from_elided_empty_connectors() {
        let mut cfg = crate::cfg::ControlFlowGraph::new();
        let bb0 = cfg.add_node(crate::cfg::BasicBlock {
            id: 0,
            start: 0x00,
            instrs: Vec::new(),
        });
        let bb1 = cfg.add_node(crate::cfg::BasicBlock {
            id: 1,
            start: 0x10,
            instrs: Vec::new(),
        });
        let bb2 = cfg.add_node(crate::cfg::BasicBlock {
            id: 2,
            start: 0x20,
            instrs: Vec::new(),
        });
        cfg.add_edge(bb0, bb1, crate::cfg::EdgeKind::UncondBranch);
        cfg.add_edge(bb1, bb2, crate::cfg::EdgeKind::FallThrough);

        let fir = crate::ir::FunctionIR {
            blocks: vec![
                crate::ir::IRBlock {
                    id: 0,
                    start_addr: 0x00,
                    irdst: vec![(Some(crate::ir::IRCond::True), 0x10)],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(1))],
                        value: RValue::ImmI(7),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
                crate::ir::IRBlock {
                    id: 1,
                    start_addr: 0x10,
                    irdst: vec![(Some(crate::ir::IRCond::True), 0x20)],
                    stmts: Vec::new(),
                },
                crate::ir::IRBlock {
                    id: 2,
                    start_addr: 0x20,
                    irdst: vec![],
                    stmts: vec![IRStatement {
                        defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(1))],
                        value: RValue::ImmI(9),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: Vec::new(),
                    }],
                },
            ],
        };
        let structured = StructuredStatement::Sequence(vec![
            StructuredStatement::BasicBlock {
                block_id: 0,
                stmts: fir.blocks[0].stmts.clone(),
            },
            StructuredStatement::UnstructuredJump {
                from_block_id: 0,
                to_block_id: 1,
                condition: None,
            },
            StructuredStatement::BasicBlock {
                block_id: 2,
                stmts: fir.blocks[2].stmts.clone(),
            },
        ]);

        let lowered = lower_structured_stmt(&structured, &cfg, &fir, &FunctionAnalysis::default());
        let rendered = lowered.render_with_indent(0);
        assert!(rendered.contains("goto BB2;"), "got:\n{rendered}");
        assert!(rendered.contains("BB2:"), "got:\n{rendered}");
        assert!(!rendered.contains("goto BB1;"), "got:\n{rendered}");
        assert!(!rendered.contains("BB1:"), "got:\n{rendered}");
    }

    #[test]
    fn lowers_iadd3_to_addition_chain() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "IADD3".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                    IRExpr::ImmI(4),
                    IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let Stmt::Assign { src, .. } = lowered.stmt else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "r4_0 + 4");
    }

    #[test]
    fn lowers_wide_add_helpers_without_calllike_leaks() {
        let iadd64 = lower_op_expr(
            "IADD.64",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
            ],
        );
        assert!(
            !iadd64.render().contains("IADD.64("),
            "got: {}",
            iadd64.render()
        );

        let uiadd3_64 = lower_op_expr(
            "UIADD3.64",
            &[
                IRExpr::Reg(crate::ir::RegId::new("UR", 18, 1).with_ssa(0)),
                IRExpr::ImmI(16),
                IRExpr::Reg(crate::ir::RegId::new("URZ", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("UR", 19, 1).with_ssa(0)),
                IRExpr::ImmI(0),
                IRExpr::Reg(crate::ir::RegId::new("URZ", 0, 1)),
            ],
        );
        assert!(
            !uiadd3_64.render().contains("UIADD3.64("),
            "got: {}",
            uiadd3_64.render()
        );
    }

    #[test]
    fn keeps_modifier_bearing_add_ops_explicit() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "IADD3.X".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                    IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                    IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let Stmt::Assign { src, .. } = lowered.stmt else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "IADD3.X(r2_0, r3_0, 0)");
    }

    #[test]
    fn lowers_special_register_and_setp_ops_semantically() {
        let s2r = lower_op_expr(
            "S2R",
            &[IRExpr::Op {
                op: "SR_TID.X".to_string(),
                args: Vec::new(),
            }],
        );
        assert_eq!(s2r.render(), "threadIdx.x");

        let setp = lower_op_expr(
            "ISETP.GE.AND",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::ImmI(1),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(setp.render(), "(int32_t)(r2_0) >= (int32_t)(1)");
    }

    #[test]
    fn lowers_shfl_down_to_cuda_intrinsic() {
        let shfl = lower_op_expr(
            "SHFL.DOWN",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                IRExpr::ImmI(16),
                IRExpr::ImmI(31),
            ],
        );
        assert_eq!(shfl.render(), "__shfl_down_sync(0xffffffff, r3_0, 16)");
    }

    #[test]
    fn lowers_shf_hi_right_patterns_to_shifts() {
        let shf = lower_op_expr(
            "SHF.R.S32.HI",
            &[
                IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                IRExpr::ImmI(31),
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0)),
            ],
        );
        assert_eq!(shf.render(), "(int32_t)(r6_0) >> 31");

        let ushf = lower_op_expr(
            "USHF.R.U32.HI",
            &[
                IRExpr::Reg(crate::ir::RegId::new("URZ", 0, 1)),
                IRExpr::ImmI(5),
                IRExpr::Op {
                    op: "SR_TID.X".to_string(),
                    args: Vec::new(),
                },
            ],
        );
        assert_eq!(ushf.render(), "threadIdx.x >> 5");
    }

    #[test]
    fn non_memory_lowering_prefers_data_def_over_sink_predicate() {
        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0)),
            ],
            value: RValue::Op {
                opcode: "SHFL.DOWN".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                    IRExpr::ImmI(16),
                    IRExpr::ImmI(31),
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let Stmt::Assign { dst, src } = lowered.stmt else {
            panic!("expected assignment");
        };
        assert_eq!(dst.render(), "r0_0");
        assert_eq!(src.render(), "__shfl_down_sync(0xffffffff, r3_0, 16)");
    }

    #[test]
    fn lowers_lop3_and_mask_to_native_bitwise_expr() {
        let expr = lower_op_expr(
            "LOP3.LUT",
            &[
                IRExpr::Op {
                    op: "SR_TID.X".to_string(),
                    args: Vec::new(),
                },
                IRExpr::ImmI(31),
                IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                IRExpr::ImmI(0xc0),
                IRExpr::Op {
                    op: "!PT".to_string(),
                    args: Vec::new(),
                },
            ],
        );
        assert_eq!(expr.render(), "threadIdx.x & 31");
    }

    #[test]
    fn lowers_multi_def_lop3_to_data_and_predicate_assignments() {
        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("P", 0, 1).with_ssa(1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 8, 1).with_ssa(0)),
            ],
            value: RValue::Op {
                opcode: "LOP3.LUT".to_string(),
                args: vec![
                    IRExpr::Op {
                        op: "SR_TID.X".to_string(),
                        args: Vec::new(),
                    },
                    IRExpr::ImmI(31),
                    IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                    IRExpr::ImmI(0xc0),
                    IRExpr::Op {
                        op: "!PT".to_string(),
                        args: Vec::new(),
                    },
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let rendered = lowered.stmt.render_with_indent(0);
        assert!(
            rendered.contains("r8_0 = threadIdx.x & 31;"),
            "got:\n{rendered}"
        );
        assert!(
            rendered.contains("p0_1 = (threadIdx.x & 31) != 0;"),
            "got:\n{rendered}"
        );
    }

    #[test]
    fn lowers_plop3_true_inputs_to_false_predicate() {
        let expr = lower_op_expr(
            "PLOP3.LUT",
            &[
                IRExpr::Op {
                    op: "PT".to_string(),
                    args: Vec::new(),
                },
                IRExpr::Op {
                    op: "PT".to_string(),
                    args: Vec::new(),
                },
                IRExpr::Op {
                    op: "PT".to_string(),
                    args: Vec::new(),
                },
                IRExpr::Op {
                    op: "PT".to_string(),
                    args: Vec::new(),
                },
                IRExpr::ImmI(8),
                IRExpr::ImmI(0),
            ],
        );
        assert_eq!(expr.render(), "false");
    }

    #[test]
    fn lowers_lea_hi_and_shift_hi_structurally() {
        let lea_hi = lower_op_expr(
            "LEA.HI.X",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
                IRExpr::ImmI(2),
                IRExpr::Reg(crate::ir::RegId::new("P", 1, 1).with_ssa(0)),
            ],
        );
        let rendered = lea_hi.render();
        assert!(!rendered.contains("LEA.HI.X("), "got: {rendered}");
        assert!(
            rendered.contains("r6_0") && rendered.contains("r7_0"),
            "got: {rendered}"
        );

        let lea_hi_sx32 = lower_op_expr(
            "LEA.HI.X.SX32",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 9, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 10, 1).with_ssa(0)),
                IRExpr::ImmI(1),
                IRExpr::Reg(crate::ir::RegId::new("P", 2, 1).with_ssa(0)),
            ],
        );
        let rendered = lea_hi_sx32.render();
        assert!(!rendered.contains("LEA.HI.X.SX32("), "got: {rendered}");
        assert!(rendered.contains("(p2_0 ? 1 : 0)"), "got: {rendered}");

        let shf_hi = lower_op_expr(
            "SHF.L.U64.HI",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 11, 1).with_ssa(0)),
                IRExpr::ImmI(2),
                IRExpr::Reg(crate::ir::RegId::new("R", 12, 1).with_ssa(0)),
            ],
        );
        let rendered = shf_hi.render();
        assert!(!rendered.contains("SHF.L.U64.HI("), "got: {rendered}");
        assert!(rendered.contains(">> 32"), "got: {rendered}");
    }

    #[test]
    fn lowers_constmem_scalars_to_param_aliases_and_builtin_symbols() {
        let analysis = analyze_sass(
            "/*0000*/ IABS R5, c[0x0][0x164] ;\n\
             /*0010*/ EXIT ;\n",
        );
        let param = lower_ir_op_expr_with_analysis(
            "ConstMem",
            &[IRExpr::ImmI(0), IRExpr::ImmI(0x164)],
            Some(&analysis),
        );
        assert_eq!(param.render(), "arg1");

        let legacy_analysis = FunctionAnalysis {
            abi_profile: Some(crate::abi::AbiProfile::legacy_param_140()),
            ..FunctionAnalysis::default()
        };
        let builtin = lower_ir_op_expr_with_analysis(
            "ConstMem",
            &[IRExpr::ImmI(0), IRExpr::ImmI(0x0)],
            Some(&legacy_analysis),
        );
        assert_eq!(builtin.render(), "blockDim.x");
    }

    #[test]
    fn wraps_unresolved_register_bases_before_indexing() {
        let stmt = IRStatement {
            defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0))],
            value: RValue::Op {
                opcode: "LDG.E".to_string(),
                args: vec![IRExpr::Mem {
                    base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))),
                    offset: None,
                    width: Some(32),
                }],
            },
            pred: None,
            mem_addr_args: Some(vec![IRExpr::Mem {
                base: Box::new(IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))),
                offset: None,
                width: Some(32),
            }]),
            pred_old_defs: Vec::new(),
        };
        let mut analysis = FunctionAnalysis::default();
        analysis.mem_accesses.push(MemAccessInfo {
            block_id: 0,
            stmt_idx: 0,
            kind: MemAccessKind::Load,
            space: CudaMemorySpace::Global,
            bit_width: Some(32),
            vector_width: None,
            constant_byte_offset: Some(0),
            has_dynamic_offset: true,
            root: AddressRoot::RegisterBase(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
        });
        let lowered = lower_memory_stmt(0, 0, &stmt, &analysis).expect("lowered");
        let Stmt::Assign { src, .. } = lowered else {
            panic!("expected assignment");
        };
        assert_eq!(src.render(), "((uint32_t*)(r4_0))[0]");
    }

    #[test]
    fn lowers_ffma_fmnmx_and_mufu_ops_semantically() {
        let ffma = lower_op_expr(
            "FFMA",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
            ],
        );
        assert_eq!(ffma.render(), "fmaf(r1_0, r2_0, r3_0)");

        let fmnmx = lower_op_expr(
            "FMNMX",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Op {
                    op: "!PT".to_string(),
                    args: Vec::new(),
                },
            ],
        );
        assert_eq!(fmnmx.render(), "fmaxf(r1_0, r2_0)");

        let mufu = lower_op_expr(
            "MUFU.EX2",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0))],
        );
        assert_eq!(mufu.render(), "exp2f(r4_0)");

        let i2f = lower_op_expr(
            "I2F.RP",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0))],
        );
        assert_eq!(i2f.render(), "__int2float_ru(r5_0)");

        let f2i = lower_op_expr(
            "F2I.FTZ.U32.TRUNC.NTZ",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0))],
        );
        assert_eq!(f2i.render(), "__float2uint_rz(r6_0)");

        let imad_hi = lower_op_expr(
            "IMAD.HI.U32",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 8, 1).with_ssa(0)),
                IRExpr::ImmI(0),
            ],
        );
        let rendered = imad_hi.render();
        assert!(!rendered.contains("IMAD.HI.U32("), "got: {rendered}");
        assert!(rendered.contains(">> 32"), "got: {rendered}");
    }

    #[test]
    fn lowers_fsel_fmul_and_imad_family_semantically() {
        let fsel = lower_op_expr(
            "FSEL",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
                IRExpr::ImmI(1),
                IRExpr::Reg(crate::ir::RegId::new("P", 1, 1).with_ssa(0)),
            ],
        );
        assert_eq!(fsel.render(), "p1_0 ? r7_0 : 1");

        let fmul = lower_op_expr(
            "FMUL",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
            ],
        );
        assert_eq!(fmul.render(), "r1_0 * r2_0");

        let fmul_ftz = lower_op_expr(
            "FMUL.FTZ",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::ImmF(0.5),
            ],
        );
        assert_eq!(fmul_ftz.render(), "__fmul_rn(r1_0, 0.5)");

        let fmul_rm = lower_op_expr(
            "FMUL.RM",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
            ],
        );
        assert_eq!(fmul_rm.render(), "__fmul_rd(r1_0, r2_0)");

        let fadd_ftz = lower_op_expr(
            "FADD.FTZ",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                IRExpr::ImmI(1),
            ],
        );
        assert_eq!(fadd_ftz.render(), "__fadd_rn(r3_0, 1)");

        let imad = lower_op_expr(
            "IMAD",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
            ],
        );
        assert_eq!(imad.render(), "r1_0 * r2_0 + r3_0");

        let imad_wide = lower_op_expr(
            "IMAD.WIDE",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 9, 1).with_ssa(0)),
                IRExpr::ImmI(4),
                IRExpr::Reg(crate::ir::RegId::new("R", 10, 1).with_ssa(0)),
            ],
        );
        let rendered = imad_wide.render();
        assert!(
            rendered.contains("(int64_t)((int32_t)(r9_0))"),
            "got: {rendered}"
        );
        assert!(
            rendered.contains("(int64_t)((int32_t)(4))"),
            "got: {rendered}"
        );
        assert!(
            rendered.contains("((uintptr_t)(((uint64_t)(r11_0) << 32) | (uint32_t)(r10_0)))"),
            "got: {rendered}"
        );

        let imad_mov = lower_op_expr(
            "IMAD.MOV.U32",
            &[
                IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("RZ", 0, 1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
            ],
        );
        assert_eq!(imad_mov.render(), "r4_0");

        let imad_iadd = lower_op_expr(
            "IMAD.IADD",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0)),
                IRExpr::ImmI(1),
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0)),
            ],
        );
        assert_eq!(imad_iadd.render(), "r5_0 + r6_0");

        let imad_shl = lower_op_expr(
            "IMAD.SHL.U32",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 8, 1).with_ssa(0)),
                IRExpr::ImmI(16),
                IRExpr::ImmI(0),
            ],
        );
        assert_eq!(imad_shl.render(), "r8_0 << 4");
    }

    #[test]
    fn lowers_iadd3_x_carry_chains_semantically() {
        let expr = lower_op_expr(
            "UIADD3.X",
            &[
                IRExpr::ImmI(0),
                IRExpr::Reg(crate::ir::RegId::new("UR", 7, 1).with_ssa(0)),
                IRExpr::ImmI(0),
                IRExpr::Reg(crate::ir::RegId::new("UP", 0, 1).with_ssa(0)),
                IRExpr::Op {
                    op: "!UPT".to_string(),
                    args: Vec::new(),
                },
            ],
        );
        assert_eq!(expr.render(), "ur7_0 + (up0_0 ? 1 : 0)");
    }

    #[test]
    fn lowers_imad_x_and_multidef_lea_semantically() {
        let imadx = lower_op_expr(
            "IMAD.X",
            &[
                IRExpr::ImmI(0),
                IRExpr::ImmI(0),
                IRExpr::Reg(crate::ir::RegId::new("R", 15, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("P", 3, 1).with_ssa(0)),
            ],
        );
        assert_eq!(imadx.render(), "r15_0 + (p3_0 ? 1 : 0)");

        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("P", 0, 1).with_ssa(1)),
            ],
            value: RValue::Op {
                opcode: "LEA".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 0, 1).with_ssa(0)),
                    IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                    IRExpr::ImmI(2),
                ],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: Vec::new(),
        };
        let lowered = lower_non_memory_stmt(&stmt);
        let rendered = lowered.stmt.render_with_indent(0);
        assert!(
            rendered.contains("r2_0 = (r0_0 << 2) + r4_0;"),
            "got:\n{rendered}"
        );
        assert!(rendered.contains("p0_1 ="), "got:\n{rendered}");
        assert!(rendered.contains(">> 32"), "got:\n{rendered}");
        assert!(rendered.contains("!= 0;"), "got:\n{rendered}");
    }

    #[test]
    fn lowers_predicated_multidef_lea_with_false_path_values() {
        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(2)),
                IRExpr::Reg(crate::ir::RegId::new("P", 5, 1).with_ssa(1)),
            ],
            value: RValue::Op {
                opcode: "LEA".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                    IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                    IRExpr::ImmI(2),
                ],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: None,
            pred_old_defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(1)),
                IRExpr::Reg(crate::ir::RegId::new("P", 5, 1).with_ssa(0)),
            ],
        };
        let lowered = lower_basic_stmt(0, 0, &stmt, &FunctionAnalysis::default());
        let Stmt::Sequence(stmts) = lowered else {
            panic!("expected sequence");
        };
        assert_eq!(stmts.len(), 2);
        let Stmt::Assign { dst, src } = &stmts[0] else {
            panic!("expected data assignment");
        };
        assert_eq!(dst.render(), "r4_2");
        assert_eq!(src.render(), "p0 ? ((r1_0 << 2) + r2_0) : r4_1");
        let Stmt::Assign { dst, src } = &stmts[1] else {
            panic!("expected carry assignment");
        };
        assert_eq!(dst.render(), "p5_1");
        let rendered = src.render();
        assert!(rendered.starts_with("p0 ? "), "got: {rendered}");
        assert!(rendered.ends_with(" : p5_0"), "got: {rendered}");
    }

    #[test]
    fn lowers_predicated_multidef_imad_wide_with_widened_value_and_false_path_values() {
        let stmt = IRStatement {
            defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(2)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(2)),
            ],
            value: RValue::Op {
                opcode: "UIMAD.WIDE.U32".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                    IRExpr::ImmI(16),
                    IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                ],
            },
            pred: Some(IRExpr::Reg(crate::ir::RegId::new("P", 0, 1))),
            mem_addr_args: None,
            pred_old_defs: vec![
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(1)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(1)),
            ],
        };
        let lowered = lower_basic_stmt(0, 0, &stmt, &FunctionAnalysis::default());
        let Stmt::Sequence(stmts) = lowered else {
            panic!("expected sequence");
        };
        assert_eq!(stmts.len(), 2);
        let Stmt::Assign { dst, src } = &stmts[0] else {
            panic!("expected low-lane assignment");
        };
        assert_eq!(dst.render(), "r6_2");
        let rendered = src.render();
        assert!(rendered.starts_with("p0 ? "), "got: {rendered}");
        assert!(
            rendered.contains("(uint64_t)((uint32_t)(r1_0))"),
            "got: {rendered}"
        );
        assert!(
            rendered.contains("(uint64_t)((uint32_t)(16))"),
            "got: {rendered}"
        );
        assert!(rendered.ends_with(" : r6_1"), "got: {rendered}");
        let Stmt::Assign { dst, src } = &stmts[1] else {
            panic!("expected high-lane assignment");
        };
        assert_eq!(dst.render(), "r7_2");
        let rendered = src.render();
        assert!(rendered.starts_with("p0 ? "), "got: {rendered}");
        assert!(rendered.contains(">> 32"), "got: {rendered}");
        assert!(rendered.ends_with(" : r7_1"), "got: {rendered}");
    }

    #[test]
    fn lowers_float_setp_unordered_compare_tokens_to_native_ops() {
        let expr = lower_op_expr(
            "FSETP.GEU.AND",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::ImmI(0),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(expr.render(), "r1_0 >= 0 || isnan(r1_0)");

        let num = lower_op_expr(
            "FSETP.NUM.AND",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(num.render(), "!(isnan(r2_0) || isnan(r3_0))");

        let nan = lower_op_expr(
            "FSETP.NAN.AND",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(nan.render(), "isnan(r4_0) || isnan(r5_0)");
    }

    #[test]
    fn lowers_special_pseudo_constants_and_predicates() {
        let qnan = lower_scalar_expr(&IRExpr::Op {
            op: "+QNAN".to_string(),
            args: Vec::new(),
        });
        assert_eq!(qnan.render(), "NAN");

        let pinf = lower_scalar_expr(&IRExpr::Op {
            op: "+INF".to_string(),
            args: Vec::new(),
        });
        assert_eq!(pinf.render(), "INFINITY");

        let pred = lower_scalar_expr(&IRExpr::Op {
            op: "!UPT".to_string(),
            args: Vec::new(),
        });
        assert_eq!(pred.render(), "false");

        let expr = lower_op_expr(
            "FSETP.GEU.AND",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::ImmI(0),
                IRExpr::Op {
                    op: "PT".to_string(),
                    args: Vec::new(),
                },
            ],
        );
        assert_eq!(expr.render(), "r1_0 >= 0 || isnan(r1_0)");
    }

    #[test]
    fn lowers_float_sign_lop3_to_copysignf() {
        let (analysis, _, _, stmt) = analyze_stmt(
            "/*0000*/ MOV R4, c[0x0][0x160] ;\n\
             /*0010*/ MOV R5, c[0x0][0x164] ;\n\
             /*0020*/ LDG.E.CONSTANT R8, [R4.64] ;\n\
             /*0030*/ FADD R9, R8, 0f00000000 ;\n\
             /*0040*/ LOP3.LUT R10, R9, 0x80000000, R8, 0xf8, !PT ;\n\
             /*0050*/ EXIT ;\n",
            |stmt| stmt_opcode(stmt).starts_with("LOP3"),
        );
        let lowered = lower_non_memory_stmt_with_analysis(&stmt, Some(&analysis));
        let rendered = lowered.stmt.render_with_indent(0);
        assert!(
            rendered.contains("r10_0 = copysignf(r9_0, r8_0);"),
            "got:\n{rendered}"
        );
    }

    #[test]
    fn lowers_iabs_and_mufu_rcp_semantically() {
        let analysis = analyze_sass(
            "/*0000*/ IABS R5, c[0x0][0x164] ;\n\
             /*0010*/ MUFU.RCP R6, R5 ;\n\
             /*0020*/ EXIT ;\n",
        );
        let iabs = lower_op_expr_with_analysis(
            "IABS",
            &[IRExpr::Op {
                op: "ConstMem".to_string(),
                args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x164)],
            }],
            Some(&analysis),
        );
        assert_eq!(iabs.render(), "abs(arg1)");

        let rcp = lower_op_expr_with_analysis(
            "MUFU.RCP",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0))],
            Some(&analysis),
        );
        assert_eq!(rcp.render(), "rcp_approx(r5_0)");
    }

    #[test]
    fn lowers_modeled_float_modifier_ops_and_keeps_unknown_fmnmx_explicit() {
        let ffma = lower_op_expr(
            "FFMA.RM",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
            ],
        );
        assert_eq!(ffma.render(), "__fmaf_rd(r1_0, r2_0, r3_0)");

        let ffma_sat = lower_op_expr(
            "FFMA.SAT",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 4, 1).with_ssa(0)),
                IRExpr::ImmF(0.5),
                IRExpr::ImmF(0.5),
            ],
        );
        assert_eq!(ffma_sat.render(), "__saturatef(fmaf(r4_0, 0.5, 0.5))");

        let ffma_ftz = lower_op_expr(
            "FFMA.FTZ",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 6, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 7, 1).with_ssa(0)),
                IRExpr::ImmF(1.0),
            ],
        );
        assert_eq!(ffma_ftz.render(), "__fmaf_rn(r6_0, r7_0, 1)");

        let frnd = lower_op_expr(
            "FRND.FLOOR",
            &[IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(0))],
        );
        assert_eq!(frnd.render(), "floorf(r5_0)");

        let fmnmx = lower_op_expr(
            "FMNMX.NAN",
            &[
                IRExpr::Reg(crate::ir::RegId::new("R", 1, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                IRExpr::Reg(crate::ir::RegId::new("PT", 0, 1)),
            ],
        );
        assert_eq!(fmnmx.render(), "FMNMX.NAN(r1_0, r2_0, true)");
    }

    #[test]
    fn lowers_ir_binary_ops_to_ast_binary_exprs() {
        let expr = IRExpr::Op {
            op: "*".to_string(),
            args: vec![
                IRExpr::Op {
                    op: "+".to_string(),
                    args: vec![
                        IRExpr::Reg(crate::ir::RegId::new("R", 2, 1).with_ssa(0)),
                        IRExpr::ImmI(1),
                    ],
                },
                IRExpr::ImmI(4),
            ],
        };
        assert_eq!(lower_scalar_expr(&expr).render(), "(r2_0 + 1) * 4");
    }
}
