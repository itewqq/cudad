//! CFG structurization pass used by the decompiler pipeline.
//! This pass is intentionally conservative: when a region cannot be safely
//! converted into structured control flow, it falls back to explicit goto.

use crate::ast::{Expr as AstExpr, LValue as AstLValue, LoopKind as AstLoopKind, Stmt as AstStmt};
use crate::ast_passes::{ast_apply_token_map, ast_cleanup};
#[cfg(test)]
use crate::ast_passes::ast_simplify;
use crate::cfg::ControlFlowGraph;
use crate::ir::{DisplayCtx, FunctionIR, IRBlock, IRCond, IRExpr, IRStatement, RValue, RegId};
use crate::name_recovery::apply_token_map_to_rendered;
use crate::semantic_lift::{DefRef, SemanticLiftResult};

use petgraph::algo::dominators::{simple_fast, Dominators};
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// --- Structured AST Definition ---
#[derive(Debug, Clone, PartialEq)]
pub enum LoopType {
    While,
    DoWhile, // Detection not fully implemented
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
    /// Multi-way branch (switch/case), recognized from nodes with >2
    /// outgoing edges (e.g. branch table / cascaded if-else chains).
    Switch {
        header_block_id: usize,
        /// The expression being switched on (if recoverable).
        discriminant: Option<IRExpr>,
        /// Each case arm: (case label/index, body).
        cases: Vec<(usize, StructuredStatement)>,
        /// Optional default arm.
        default: Option<Box<StructuredStatement>>,
    },
    Empty,
}

impl fmt::Display for StructuredStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                write!(f, "BB{}[{} stmts]", block_id, stmts.len())
            }
            StructuredStatement::Sequence(s) => {
                write!(f, "Sequence(")?;
                for (i, stmt) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    stmt.fmt(f)?;
                }
                write!(f, ")")
            }
            StructuredStatement::If {
                condition_block_id,
                condition_expr,
                then_branch,
                else_branch,
                ..
            } => {
                write!(
                    f,
                    "If(BB{}, Cond: {:?}, Then: {}",
                    condition_block_id, condition_expr, then_branch
                )?;
                if let Some(eb) = else_branch {
                    write!(f, ", Else: {}", eb)?;
                }
                write!(f, ")")
            }
            StructuredStatement::Loop {
                header_block_id,
                loop_type,
                condition_expr,
                body,
                ..
            } => {
                write!(
                    f,
                    "{:?}-Loop(Hdr: BB{:?}, Cond: {:?}, Body: {})",
                    loop_type,
                    header_block_id.unwrap_or(usize::MAX),
                    condition_expr,
                    body
                )
            }
            StructuredStatement::Break(_) => write!(f, "Break"),
            StructuredStatement::Continue(_) => write!(f, "Continue"),
            StructuredStatement::Return(_) => write!(f, "Return"),
            StructuredStatement::UnstructuredJump {
                from_block_id,
                to_block_id,
                ..
            } => {
                write!(f, "Goto(BB{}->BB{})", from_block_id, to_block_id)
            }
            StructuredStatement::Switch {
                header_block_id,
                cases,
                default,
                ..
            } => {
                write!(f, "Switch(BB{}, {} cases", header_block_id, cases.len())?;
                if default.is_some() {
                    write!(f, " +default")?;
                }
                write!(f, ")")
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
    addr_to_node_index: HashMap<u32, NodeIndex>,
    /// Maps IRBlock.id → index into function_ir.blocks for O(1) lookup.
    block_id_to_idx: HashMap<usize, usize>,
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

    /// Checks if a structured statement ends with an unconditional return/exit,
    /// meaning any subsequent siblings in a Sequence are dead code.
    fn ends_with_unconditional_return(stmt: &StructuredStatement) -> bool {
        match stmt {
            StructuredStatement::Return(_) => true,
            StructuredStatement::BasicBlock { stmts, .. } => {
                // Look for an unconditional RET/EXIT in the block's stmts.
                stmts.iter().any(|s| {
                    if let RValue::Op { opcode, .. } = &s.value {
                        Self::is_return_opcode(opcode) && s.pred.is_none()
                    } else {
                        false
                    }
                })
            }
            StructuredStatement::Sequence(children) => {
                // A sequence ends with return if its last non-empty child does.
                children
                    .iter()
                    .rev()
                    .find(|c| !matches!(c, StructuredStatement::Empty))
                    .map_or(false, Self::ends_with_unconditional_return)
            }
            _ => false,
        }
    }

    fn is_setp_opcode(opcode: &str) -> bool {
        opcode.starts_with("ISETP") || opcode.starts_with("FSETP")
    }

    fn is_barrier_opcode(opcode: &str) -> bool {
        opcode.starts_with("BAR.SYNC")
    }

    /// Returns true if the register is one of the "sink" registers (RZ, PT, URZ, UPT)
    /// whose writes have no observable effect.
    fn is_zero_or_true_reg(r: &IRExpr) -> bool {
        match r {
            IRExpr::Reg(reg) => matches!(reg.class.as_str(), "RZ" | "PT" | "URZ" | "UPT"),
            _ => false,
        }
    }

    /// Render an IR expression for decompiled output, replacing hardware-zero
    /// and always-true predicate registers with literals ("0" and "true").
    fn decompile_expr(ctx: &dyn DisplayCtx, e: &IRExpr) -> String {
        match e {
            IRExpr::Reg(r) => match r.class.as_str() {
                "RZ" | "URZ" => "0".to_string(),
                "PT" | "UPT" => "true".to_string(),
                _ => ctx.expr(e),
            },
            IRExpr::Op { op, args } => {
                // Recursively decompile arguments so nested RZ/PT are replaced.
                if op == "-" && args.len() == 1 {
                    let inner = Self::decompile_expr(ctx, &args[0]);
                    let simple = match &args[0] {
                        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => true,
                        IRExpr::Op { op, .. } if op == "ConstMem" => true,
                        _ => false,
                    };
                    if simple {
                        return format!("-{}", inner);
                    }
                    return format!("-({})", inner);
                }
                if op == "!" && args.len() == 1 {
                    let inner = Self::decompile_expr(ctx, &args[0]);
                    // !true → false, !(expr) → !(expr)
                    if inner == "true" {
                        return "false".to_string();
                    }
                    return format!("!({})", inner);
                }
                let list = args
                    .iter()
                    .map(|a| Self::decompile_expr(ctx, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", op, list)
            }
            _ => ctx.expr(e),
        }
    }

    /// Returns true if the instruction has memory side effects that must be preserved
    /// even when all destination registers are RZ/PT.
    fn is_memory_side_effect_opcode(value: &RValue) -> bool {
        let opcode = match value {
            RValue::Op { opcode, .. } => opcode.as_str(),
            _ => return false,
        };
        // Store instructions, atomics, and shared-memory operations
        opcode.starts_with("STG")
            || opcode.starts_with("STS")
            || opcode.starts_with("STL")
            || opcode.starts_with("ST.")
            || opcode.starts_with("ATOM")
            || opcode.starts_with("RED")
            || opcode.starts_with("CCTL")
            || opcode.starts_with("MEMBAR")
            || opcode.starts_with("BAR")
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

    fn node_is_dominated_by(
        &self,
        dom_results: &Dominators<NodeIndex>,
        node_to_check: NodeIndex,
        potential_dominator: NodeIndex,
    ) -> bool {
        if node_to_check == potential_dominator {
            return true;
        }
        let mut current = node_to_check;
        while let Some(idom) = dom_results.immediate_dominator(current) {
            if idom == potential_dominator {
                return true;
            }
            if idom == current {
                break;
            }
            current = idom;
            if current == NodeIndex::end() {
                break;
            }
        }
        false
    }

    pub fn new(cfg: &'a ControlFlowGraph, function_ir: &'a FunctionIR) -> Self {
        let entry_node = Self::choose_entry_node(cfg).unwrap_or_else(|| NodeIndex::new(0));
        let dom = simple_fast(cfg, entry_node);

        let mut addr_to_node_index = HashMap::new();
        for node_idx in cfg.node_indices() {
            let summary = cfg
                .node_weight(node_idx)
                .expect("CFG node should have a weight");
            addr_to_node_index.insert(summary.start, node_idx);
        }

        // Pre-compute block.id → index for O(1) lookup.
        let block_id_to_idx: HashMap<usize, usize> = function_ir
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();

        Structurizer {
            cfg,
            function_ir,
            dom,
            addr_to_node_index,
            block_id_to_idx,
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

    pub fn structure_function(&mut self) -> Option<StructuredStatement> {
        // New collapse-based structurizer. Iteratively rewrites a region graph
        // built from the CFG by pattern-matching collapse rules until one
        // region remains (or the goto fallback fires).
        let mut graph = collapse::RegionGraph::build_from_cfg(self);
        collapse::run_collapse(&mut graph, self);
        Some(collapse::emit_root(&graph, self))
    }

    fn get_ir_block_by_cfg_node(&self, cfg_node: NodeIndex) -> Option<&'a IRBlock> {
        let summary = self.cfg.node_weight(cfg_node)?;
        let &idx = self.block_id_to_idx.get(&summary.id)?;
        Some(&self.function_ir.blocks[idx])
    }

    /// O(1) IR block lookup by block id.
    fn get_ir_block(&self, block_id: usize) -> Option<&'a IRBlock> {
        let &idx = self.block_id_to_idx.get(&block_id)?;
        Some(&self.function_ir.blocks[idx])
    }

    fn extract_if_targets_and_condition(
        &self,
        ir_block: &'a IRBlock,
    ) -> Option<(IRExpr, NodeIndex, Option<NodeIndex>)> {
        if ir_block.irdst.is_empty() || ir_block.irdst.len() > 2 {
            return None;
        }

        if ir_block.irdst.len() == 2 {
            let (cond1_opt, addr1) = &ir_block.irdst[0];
            let (cond2_opt, addr2) = &ir_block.irdst[1];
            let node1 = self.addr_to_node_index.get(addr1).copied()?;
            let node2 = self.addr_to_node_index.get(addr2).copied()?;

            match (cond1_opt, cond2_opt) {
                (
                    Some(IRCond::Pred { reg: r1, sense: s1 }),
                    Some(IRCond::Pred { reg: r2, sense: s2 }),
                ) => {
                    if r1.class == r2.class && r1.idx == r2.idx && *s1 != *s2 {
                        let cond_expr = IRExpr::Reg(r1.clone());
                        return if *s1 {
                            Some((cond_expr, node1, Some(node2)))
                        } else {
                            Some((cond_expr, node2, Some(node1)))
                        };
                    }
                }
                (Some(IRCond::Pred { reg, sense }), Some(IRCond::True)) => {
                    let cond_expr_val = IRExpr::Reg(reg.clone());
                    return if *sense {
                        Some((cond_expr_val, node1, Some(node2)))
                    } else {
                        Some((negate_condition(cond_expr_val), node1, Some(node2)))
                    };
                }
                (Some(IRCond::True), Some(IRCond::Pred { reg, sense })) => {
                    let cond_expr_val = IRExpr::Reg(reg.clone());
                    return if *sense {
                        Some((cond_expr_val, node2, Some(node1)))
                    } else {
                        Some((negate_condition(cond_expr_val), node2, Some(node1)))
                    };
                }
                _ => {}
            }
        } else if ir_block.irdst.len() == 1 {
            let (cond_opt, addr) = &ir_block.irdst[0];
            if let Some(IRCond::Pred { reg, sense }) = cond_opt {
                let target_node = self.addr_to_node_index.get(addr).copied()?;
                let base_reg_expr = IRExpr::Reg(reg.clone());
                let cond_expr = if *sense {
                    base_reg_expr
                } else {
                    negate_condition(base_reg_expr)
                };
                return Some((cond_expr, target_node, None));
            }
        }
        None
    }

    fn is_convergence_barrier_opcode(opcode: &str) -> bool {
        // SM 100+ (Blackwell) annotates convergence barriers with reliability
        // hints: `BSSY.RECONVERGENT`, `BSYNC.RELIABLE`, etc.  Compare against
        // the base mnemonic so both the plain and annotated forms match.
        let base = opcode.split('.').next().unwrap_or(opcode);
        matches!(base, "BSSY" | "BSYNC" | "SSY" | "SYNC" | "WARPSYNC")
    }

    fn ast_sequence(stmts: Vec<AstStmt>) -> AstStmt {
        let mut flat = Vec::new();
        for stmt in stmts {
            match stmt {
                AstStmt::Empty => {}
                AstStmt::Sequence(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => AstStmt::Empty,
            1 => flat.into_iter().next().unwrap(),
            _ => AstStmt::Sequence(flat),
        }
    }

    fn ast_is_empty(stmt: &AstStmt) -> bool {
        match stmt {
            AstStmt::Empty => true,
            AstStmt::Block(stmts) | AstStmt::Sequence(stmts) => {
                stmts.iter().all(Self::ast_is_empty)
            }
            _ => false,
        }
    }

    fn ast_expr_from_display(ctx: &dyn DisplayCtx, expr: &IRExpr) -> AstExpr {
        match expr {
            IRExpr::Reg(r) => {
                if matches!(r.class.as_str(), "RZ" | "URZ") {
                    AstExpr::Imm("0".to_string())
                } else if matches!(r.class.as_str(), "PT" | "UPT") {
                    AstExpr::Imm("true".to_string())
                } else {
                    AstExpr::Reg(ctx.expr(expr))
                }
            }
            IRExpr::ImmI(i) => AstExpr::Imm(i.to_string()),
            IRExpr::ImmF(f) => AstExpr::Imm(f.to_string()),
            _ => AstExpr::Raw(Self::decompile_expr(ctx, expr)),
        }
    }

    fn ast_lvalue_from_display(ctx: &dyn DisplayCtx, expr: &IRExpr) -> AstLValue {
        match expr {
            IRExpr::Reg(r) => {
                if matches!(r.class.as_str(), "RZ" | "URZ") {
                    AstLValue::Raw("0".to_string())
                } else if matches!(r.class.as_str(), "PT" | "UPT") {
                    AstLValue::Raw("true".to_string())
                } else {
                    AstLValue::Var(ctx.expr(expr))
                }
            }
            _ => AstLValue::Raw(Self::decompile_expr(ctx, expr)),
        }
    }

    fn ast_dest_from_defs(ctx: &dyn DisplayCtx, defs: &[IRExpr]) -> Option<AstLValue> {
        if defs.is_empty() {
            return None;
        }
        if defs.len() == 1 {
            return Some(Self::ast_lvalue_from_display(ctx, &defs[0]));
        }
        let rendered = defs
            .iter()
            .filter(|d| !Self::is_zero_or_true_reg(d))
            .map(|d| ctx.expr(d))
            .collect::<Vec<_>>();
        if rendered.is_empty() {
            None
        } else {
            Some(AstLValue::Raw(format!("({})", rendered.join(", "))))
        }
    }

    fn ast_value_from_rvalue(ctx: &dyn DisplayCtx, value: &RValue) -> AstExpr {
        match value {
            RValue::Op { opcode, args } => {
                let args_s = args
                    .iter()
                    .map(|a| Self::decompile_expr(ctx, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                AstExpr::Raw(format!("{}({})", opcode, args_s))
            }
            RValue::Phi(args) => {
                let args_s = args
                    .iter()
                    .map(|a| Self::decompile_expr(ctx, a))
                    .collect::<Vec<_>>()
                    .join(", ");
                AstExpr::Raw(format!("phi({})", args_s))
            }
            RValue::ImmI(i) => AstExpr::Imm(i.to_string()),
            RValue::ImmF(f) => AstExpr::Imm(f.to_string()),
        }
    }

    fn ast_is_trivial_self_assign(stmt: &AstStmt) -> bool {
        match stmt {
            AstStmt::Assign {
                dst: AstLValue::Var(dst),
                src: AstExpr::Reg(src),
            }
            | AstStmt::Assign {
                dst: AstLValue::Var(dst),
                src: AstExpr::Raw(src),
            }
            | AstStmt::Assign {
                dst: AstLValue::Raw(dst),
                src: AstExpr::Reg(src),
            }
            | AstStmt::Assign {
                dst: AstLValue::Raw(dst),
                src: AstExpr::Raw(src),
            } => dst == src,
            _ => false,
        }
    }

    fn ast_guarded(condition: Option<AstExpr>, stmt: AstStmt) -> AstStmt {
        if matches!(stmt, AstStmt::Empty) || Self::ast_is_trivial_self_assign(&stmt) {
            return AstStmt::Empty;
        }
        match condition {
            Some(condition) => AstStmt::If {
                condition,
                then_branch: Box::new(stmt),
                else_branch: None,
            },
            None => stmt,
        }
    }

    fn lower_lifted_stmt_ast(lifted_stmt: &crate::semantic_lift::LiftedStmt) -> AstStmt {
        if lifted_stmt.dest.is_sink_literal() {
            return Self::ast_guarded(
                lifted_stmt.pred.clone(),
                AstStmt::ExprStmt(lifted_stmt.rhs.clone()),
            );
        }
        match (&lifted_stmt.pred, &lifted_stmt.pred_old_val) {
            (Some(pred), Some(old)) => AstStmt::Assign {
                dst: lifted_stmt.dest.clone(),
                src: AstExpr::Ternary {
                    cond: Box::new(pred.clone()),
                    then_expr: Box::new(lifted_stmt.rhs.clone()),
                    else_expr: Box::new(old.clone()),
                },
            },
            (Some(pred), None) => Self::ast_guarded(
                Some(pred.clone()),
                AstStmt::Assign {
                    dst: lifted_stmt.dest.clone(),
                    src: lifted_stmt.rhs.clone(),
                },
            ),
            (None, _) => AstStmt::Assign {
                dst: lifted_stmt.dest.clone(),
                src: lifted_stmt.rhs.clone(),
            },
        }
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
            .find(|(_, s)| s.defs == stmt.defs && s.value == stmt.value && s.pred == stmt.pred)
        {
            *fir_search_from = found_idx + 1;
            found_idx
        } else {
            fallback_idx
        }
    }

    fn lower_non_control_stmt_ast(
        &self,
        block_id: usize,
        stmt_idx: usize,
        ir_s: &IRStatement,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> Vec<AstStmt> {
        if let Some(res) = lifted {
            let mut lowered = Vec::new();
            for def_idx in Self::lifted_def_emit_order(ir_s) {
                let def_ref = DefRef {
                    block_id,
                    stmt_idx,
                    def_idx,
                };
                let Some(lifted_stmt) = res.by_def.get(&def_ref) else {
                    continue;
                };
                lowered.push(Self::lower_lifted_stmt_ast(lifted_stmt));
            }
            if !lowered.is_empty() {
                return lowered;
            }
        }

        let side_effect_only =
            ir_s.defs.is_empty() || ir_s.defs.iter().all(Self::is_zero_or_true_reg);
        let value_expr = Self::ast_value_from_rvalue(ctx, &ir_s.value);
        let pred_expr = ir_s.pred.as_ref().map(|pred| {
            self.resolve_stmt_predicate_expr_ast(block_id, stmt_idx, pred, ctx, lifted)
        });

        if side_effect_only {
            return vec![Self::ast_guarded(pred_expr, AstStmt::ExprStmt(value_expr))];
        }

        let dest = Self::ast_dest_from_defs(ctx, &ir_s.defs)
            .unwrap_or_else(|| AstLValue::Raw("_".to_string()));

        if ir_s.pred.is_some() && !ir_s.pred_old_defs.is_empty() && ir_s.defs.len() == 1 {
            let pred = pred_expr.unwrap();
            let old = Self::ast_expr_from_display(ctx, &ir_s.pred_old_defs[0]);
            return vec![AstStmt::Assign {
                dst: dest,
                src: AstExpr::Ternary {
                    cond: Box::new(pred),
                    then_expr: Box::new(value_expr),
                    else_expr: Box::new(old),
                },
            }];
        }

        vec![Self::ast_guarded(
            pred_expr,
            AstStmt::Assign {
                dst: dest,
                src: value_expr,
            },
        )]
    }

    fn lower_stmt_list_ast(
        &self,
        block_id: usize,
        stmts: &[IRStatement],
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
        emit_returns: bool,
    ) -> AstStmt {
        let fir_block = self.get_ir_block(block_id);
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
        let mut fir_search_from = 0usize;
        let mut out = Vec::new();

        for (stmt_idx, ir_s) in stmts.iter().enumerate() {
            let lookup_stmt_idx =
                Self::resolve_stmt_lookup_idx(fir_block, &mut fir_search_from, stmt_idx, ir_s);

            if matches!(ir_s.value, RValue::Phi(_)) {
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
                if Self::is_branch_only_opcode(opcode)
                    || Self::is_convergence_barrier_opcode(opcode)
                    || opcode == "NOP"
                {
                    continue;
                }
                if Self::is_barrier_opcode(opcode) {
                    out.push(Self::ast_guarded(
                        ir_s.pred.as_ref().map(|pred| {
                            self.resolve_stmt_predicate_expr_ast(
                                block_id,
                                lookup_stmt_idx,
                                pred,
                                ctx,
                                lifted,
                            )
                        }),
                        AstStmt::ExprStmt(AstExpr::CallLike {
                            func: "__syncthreads".to_string(),
                            args: vec![],
                        }),
                    ));
                    continue;
                }
                if Self::is_return_opcode(opcode) {
                    if !emit_returns {
                        continue;
                    }
                    let guarded = Self::ast_guarded(
                        ir_s.pred.as_ref().map(|pred| {
                            self.resolve_stmt_predicate_expr_ast(
                                block_id,
                                lookup_stmt_idx,
                                pred,
                                ctx,
                                lifted,
                            )
                        }),
                        AstStmt::Return(None),
                    );
                    let terminate = ir_s.pred.is_none();
                    out.push(guarded);
                    if terminate {
                        break;
                    }
                    continue;
                }
            }

            if !ir_s.defs.is_empty()
                && ir_s.defs.iter().all(Self::is_zero_or_true_reg)
                && !Self::is_memory_side_effect_opcode(&ir_s.value)
            {
                continue;
            }

            out.extend(self.lower_non_control_stmt_ast(
                block_id,
                lookup_stmt_idx,
                ir_s,
                ctx,
                lifted,
            ));
        }

        Self::ast_sequence(out)
    }

    fn cfg_node_for_block_id(&self, block_id: usize) -> Option<NodeIndex> {
        self.cfg
            .node_indices()
            .find(|&node| self.cfg[node].id == block_id)
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

    fn lower_loop_phi_prelude_ast(
        &self,
        header_block_id: usize,
        body: &StructuredStatement,
        ctx: &dyn DisplayCtx,
    ) -> AstStmt {
        let Some(header_node) = self.cfg_node_for_block_id(header_block_id) else {
            return AstStmt::Empty;
        };
        let Some(header_block) = self.get_ir_block(header_block_id) else {
            return AstStmt::Empty;
        };

        let mut loop_blocks = HashSet::new();
        loop_blocks.insert(header_block_id);
        Self::collect_structured_block_ids(body, &mut loop_blocks);

        let preds: Vec<_> = self
            .cfg
            .neighbors_directed(header_node, Direction::Incoming)
            .collect();
        let external_preds = preds
            .iter()
            .copied()
            .filter(|pred| !loop_blocks.contains(&self.cfg[*pred].id))
            .collect::<Vec<_>>();
        if external_preds.len() != 1 {
            return AstStmt::Empty;
        }
        let Some(pred_idx) = preds.iter().position(|&pred| pred == external_preds[0]) else {
            return AstStmt::Empty;
        };

        let mut out = Vec::new();
        for stmt in header_block
            .stmts
            .iter()
            .filter(|stmt| matches!(stmt.value, RValue::Phi(_)))
        {
            let Some(dst) = Self::ast_dest_from_defs(ctx, &stmt.defs) else {
                continue;
            };
            if dst.is_sink_literal() {
                continue;
            }
            let RValue::Phi(args) = &stmt.value else {
                continue;
            };
            let Some(src_expr) = args.get(pred_idx) else {
                continue;
            };
            let assign = AstStmt::Assign {
                dst,
                src: Self::ast_expr_from_display(ctx, src_expr),
            };
            if !Self::ast_is_trivial_self_assign(&assign) {
                out.push(assign);
            }
        }

        Self::ast_sequence(out)
    }

    fn lower_condition_prelude_ast(
        &self,
        block_id: usize,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstStmt {
        let Some(block) = self.get_ir_block(block_id) else {
            return AstStmt::Empty;
        };
        match self.lower_stmt_list_ast(block_id, &block.stmts, ctx, lifted, false) {
            AstStmt::Empty => AstStmt::Empty,
            AstStmt::Sequence(stmts) => AstStmt::Block(stmts),
            stmt => AstStmt::Block(vec![stmt]),
        }
    }

    fn resolve_predicate_condition_expr_ast(
        &self,
        block_id: usize,
        max_stmt_idx: Option<usize>,
        pred: &RegId,
        lifted: Option<&SemanticLiftResult>,
    ) -> Option<AstExpr> {
        let lifted = lifted?;
        let block = self.get_ir_block(block_id)?;
        let end = max_stmt_idx
            .map(|idx| idx.saturating_add(1).min(block.stmts.len()))
            .unwrap_or(block.stmts.len());
        for (stmt_idx, stmt) in block.stmts[..end].iter().enumerate().rev() {
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
                    return Some(ls.rhs.clone());
                }
            }
        }

        let want_ssa = pred.ssa?;
        let mut resolved = None;
        for other_block in &self.function_ir.blocks {
            for (stmt_idx, stmt) in other_block.stmts.iter().enumerate() {
                for (def_idx, def) in stmt.defs.iter().enumerate() {
                    let IRExpr::Reg(dst) = def else {
                        continue;
                    };
                    if dst.class != pred.class || dst.idx != pred.idx || dst.ssa != Some(want_ssa) {
                        continue;
                    }
                    let def_ref = DefRef {
                        block_id: other_block.id,
                        stmt_idx,
                        def_idx,
                    };
                    let Some(ls) = lifted.by_def.get(&def_ref) else {
                        continue;
                    };
                    if let Some(existing) = &resolved {
                        if existing != &ls.rhs {
                            return None;
                        }
                    } else {
                        resolved = Some(ls.rhs.clone());
                    }
                }
            }
        }
        resolved
    }

    fn resolve_stmt_predicate_expr_ast(
        &self,
        block_id: usize,
        stmt_idx: usize,
        pred_expr: &IRExpr,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstExpr {
        match pred_expr {
            IRExpr::Op { op, args } if op == "!" && args.len() == 1 => AstExpr::Unary {
                op: "!".to_string(),
                arg: Box::new(
                    self.resolve_stmt_predicate_expr_ast(block_id, stmt_idx, &args[0], ctx, lifted),
                ),
            },
            IRExpr::Reg(pred) => self
                .resolve_predicate_condition_expr_ast(block_id, Some(stmt_idx), pred, lifted)
                .unwrap_or_else(|| Self::ast_expr_from_display(ctx, pred_expr)),
            _ => Self::ast_expr_from_display(ctx, pred_expr),
        }
    }

    fn lower_condition_expr_ast(
        &self,
        condition_block_id: usize,
        condition_expr: &IRExpr,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstExpr {
        match condition_expr {
            IRExpr::Op { op, args } if op == "!" && args.len() == 1 => AstExpr::Unary {
                op: "!".to_string(),
                arg: Box::new(self.lower_condition_expr_ast(
                    condition_block_id,
                    &args[0],
                    ctx,
                    lifted,
                )),
            },
            IRExpr::Reg(pred) => self
                .resolve_predicate_condition_expr_ast(condition_block_id, None, pred, lifted)
                .unwrap_or_else(|| Self::ast_expr_from_display(ctx, condition_expr)),
            _ => Self::ast_expr_from_display(ctx, condition_expr),
        }
    }

    fn lower_structured_stmt_ast(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        lifted: Option<&SemanticLiftResult>,
    ) -> AstStmt {
        match stmt {
            StructuredStatement::BasicBlock { block_id, stmts } => {
                self.lower_stmt_list_ast(*block_id, stmts, ctx, lifted, true)
            }
            StructuredStatement::Sequence(stmts) => {
                let mut out = Vec::new();
                for child in stmts {
                    if matches!(child, StructuredStatement::Empty) {
                        continue;
                    }
                    out.push(self.lower_structured_stmt_ast(child, ctx, lifted));
                    if Self::ends_with_unconditional_return(child) {
                        break;
                    }
                }
                Self::ast_sequence(out)
            }
            StructuredStatement::If {
                condition_block_id,
                condition_expr,
                then_branch,
                else_branch,
            } => {
                let prelude = self.lower_condition_prelude_ast(*condition_block_id, ctx, lifted);
                let then_empty = Self::is_effectively_empty(then_branch);
                let else_ref = else_branch.as_deref();
                let else_empty = else_ref.map(Self::is_effectively_empty).unwrap_or(true);
                let core = if then_empty && else_empty {
                    AstStmt::Empty
                } else if then_empty && !else_empty {
                    AstStmt::If {
                        condition: self.lower_condition_expr_ast(
                            *condition_block_id,
                            &negate_condition(condition_expr.clone()),
                            ctx,
                            lifted,
                        ),
                        then_branch: Box::new(self.lower_structured_stmt_ast(
                            else_ref.unwrap(),
                            ctx,
                            lifted,
                        )),
                        else_branch: None,
                    }
                } else {
                    let else_ast =
                        else_ref.map(|stmt| self.lower_structured_stmt_ast(stmt, ctx, lifted));
                    AstStmt::If {
                        condition: self.lower_condition_expr_ast(
                            *condition_block_id,
                            condition_expr,
                            ctx,
                            lifted,
                        ),
                        then_branch: Box::new(self.lower_structured_stmt_ast(
                            then_branch,
                            ctx,
                            lifted,
                        )),
                        else_branch: else_ast
                            .filter(|stmt| !Self::ast_is_empty(stmt))
                            .map(Box::new),
                    }
                };
                Self::ast_sequence(vec![prelude, core])
            }
            StructuredStatement::Loop {
                loop_type,
                header_block_id,
                condition_expr,
                body,
            } => {
                let printable_body = Self::without_redundant_loop_tail_continue(body);
                let phi_prelude = header_block_id
                    .map(|hid| self.lower_loop_phi_prelude_ast(hid, &printable_body, ctx))
                    .unwrap_or(AstStmt::Empty);
                let condition_prelude = if *loop_type != LoopType::DoWhile {
                    header_block_id
                        .map(|hid| self.lower_condition_prelude_ast(hid, ctx, lifted))
                        .unwrap_or(AstStmt::Empty)
                } else {
                    AstStmt::Empty
                };
                let loop_stmt = AstStmt::Loop {
                    kind: match loop_type {
                        LoopType::While => AstLoopKind::While,
                        LoopType::DoWhile => AstLoopKind::DoWhile,
                        LoopType::Endless => AstLoopKind::Endless,
                    },
                    condition: condition_expr.as_ref().map(|expr| {
                        if let Some(hid) = header_block_id {
                            self.lower_condition_expr_ast(*hid, expr, ctx, lifted)
                        } else {
                            Self::ast_expr_from_display(ctx, expr)
                        }
                    }),
                    body: Box::new(self.lower_structured_stmt_ast(&printable_body, ctx, lifted)),
                };
                Self::ast_sequence(vec![phi_prelude, condition_prelude, loop_stmt])
            }
            StructuredStatement::Break(_) => AstStmt::Break,
            StructuredStatement::Continue(_) => AstStmt::Continue,
            StructuredStatement::Return(expr_opt) => AstStmt::Return(
                expr_opt
                    .as_ref()
                    .map(|expr| Self::ast_expr_from_display(ctx, expr)),
            ),
            StructuredStatement::UnstructuredJump {
                to_block_id,
                condition,
                ..
            } => Self::ast_guarded(
                condition
                    .as_ref()
                    .map(|expr| Self::ast_expr_from_display(ctx, expr)),
                AstStmt::Goto(format!("BB{}", to_block_id)),
            ),
            StructuredStatement::Switch {
                header_block_id,
                discriminant,
                cases,
                default,
            } => {
                let prelude = self.lower_condition_prelude_ast(*header_block_id, ctx, lifted);
                let switch_stmt = AstStmt::Switch {
                    discriminant: discriminant
                        .as_ref()
                        .map(|expr| Self::ast_expr_from_display(ctx, expr)),
                    cases: cases
                        .iter()
                        .map(|(case_idx, body)| {
                            (*case_idx, self.lower_structured_stmt_ast(body, ctx, lifted))
                        })
                        .collect(),
                    default: default
                        .as_ref()
                        .map(|body| Box::new(self.lower_structured_stmt_ast(body, ctx, lifted))),
                };
                Self::ast_sequence(vec![prelude, switch_stmt])
            }
            StructuredStatement::Empty => AstStmt::Empty,
        }
    }

    #[cfg(test)]
    pub(crate) fn pretty_print(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
    ) -> String {
        self.pretty_print_with_lift(stmt, ctx, indent_level, None)
    }

    #[cfg(test)]
    pub(crate) fn pretty_print_with_lift(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        ast_simplify(self.lower_structured_stmt_ast(stmt, ctx, lifted))
            .render_with_indent(indent_level)
    }

    pub fn pretty_print_with_lift_cleanup(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
    ) -> String {
        ast_cleanup(self.lower_structured_stmt_ast(stmt, ctx, lifted))
            .render_with_indent(indent_level)
    }

    pub fn pretty_print_with_lift_cleanup_and_names(
        &self,
        stmt: &StructuredStatement,
        ctx: &dyn DisplayCtx,
        indent_level: usize,
        lifted: Option<&SemanticLiftResult>,
        token_map: &HashMap<String, String>,
    ) -> String {
        let rendered = ast_cleanup(ast_apply_token_map(
            self.lower_structured_stmt_ast(stmt, ctx, lifted),
            token_map,
        ))
        .render_with_indent(indent_level);
        apply_token_map_to_rendered(&rendered, token_map)
    }
}

fn negate_condition(expr: IRExpr) -> IRExpr {
    match expr {
        IRExpr::Op { op, mut args } if op == "!" && args.len() == 1 => args.remove(0),
        _ => IRExpr::Op {
            op: "!".to_string(),
            args: vec![expr],
        },
    }
}

// ---------------------------------------------------------------------------
// Ghidra-style collapse-based structurizer
// ---------------------------------------------------------------------------
//
// The region graph is built once from the CFG and then iteratively rewritten
// by pattern-matching collapse rules. Unlike petgraph, region indices never
// shift: regions are stored in `Vec<Option<...>>`-like storage and marked
// `Tombstone` when consumed. This keeps `RegionId` stable for the entire run
// and avoids the stale-NodeIndex bug that killed the old `src/region.rs`.
//
// Phase A covers: region construction, sequence collapse, a minimal goto
// fallback, and the root emitter.
mod collapse {
    use super::*;

    /// Stable index into the region vector. Never invalidated.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub(super) struct RegionId(pub(super) usize);

    /// What a region currently represents.
    #[derive(Clone, Debug)]
    pub(super) enum RegionKind {
        /// A still-unstructured CFG basic block.
        Basic { block_id: usize },
        /// A region that has already been folded into a structured statement.
        /// `primary_block_id` is the IR block id whose `irdst` governs the
        /// region's *current* outgoing branch — for a Sequence, it's the tail
        /// block; for an If, it's the merge-or-head block id.
        Composite {
            stmt: StructuredStatement,
            primary_block_id: usize,
        },
        /// Slot is dead. Callers must not traverse it.
        Tombstone,
    }

    /// A directed edge between two regions with the metadata collapse rules
    /// care about.
    #[derive(Clone, Debug)]
    pub(super) struct RegionEdge {
        pub from: RegionId,
        pub to: RegionId,
        /// Which arm of the source's branch this edge represents. `Some(true)`
        /// = the oriented "true" arm, `Some(false)` = the "false" arm, `None`
        /// = an unconditional/fallthrough edge.
        pub cond_arm: Option<bool>,
        /// True iff the source-to-target edge is a back edge in the original
        /// CFG. Computed once at construction and never updated.
        pub is_back_edge: bool,
        /// True once `select_goto_edge` has marked this edge as "give up, emit
        /// it as an UnstructuredJump". Collapse rules skip goto-marked edges.
        pub is_goto: bool,
    }

    /// Stable-index region graph.
    pub(super) struct RegionGraph {
        regions: Vec<RegionKind>,
        edges: Vec<Option<RegionEdge>>,
        out_edges: Vec<Vec<usize>>,
        in_edges: Vec<Vec<usize>>,
        entry: RegionId,
    }

    impl RegionGraph {
        // ------------- construction -------------

        pub(super) fn build_from_cfg(s: &Structurizer<'_>) -> RegionGraph {
            // Deterministic node ordering by BasicBlock.id so results don't
            // depend on petgraph's internal ordering.
            let mut ordered: Vec<NodeIndex> = s.cfg.node_indices().collect();
            ordered.sort_by_key(|&n| s.cfg[n].id);

            let mut regions: Vec<RegionKind> = Vec::with_capacity(ordered.len());
            let mut cfg_to_region: HashMap<NodeIndex, RegionId> = HashMap::new();

            for cfg_node in &ordered {
                let block_id = s.cfg[*cfg_node].id;
                let rid = RegionId(regions.len());
                cfg_to_region.insert(*cfg_node, rid);

                // Seed return blocks as Composite right away so the collapse
                // loop sees them as terminal regions. Mirrors the legacy
                // behavior at structurizer.rs:461-498.
                let mut seeded = false;
                if let Some(ir_block) = s.get_ir_block_by_cfg_node(*cfg_node) {
                    if Structurizer::is_block_return(ir_block) {
                        let stmt = build_return_stmt(ir_block);
                        regions.push(RegionKind::Composite {
                            stmt,
                            primary_block_id: block_id,
                        });
                        seeded = true;
                    }
                }
                if !seeded {
                    regions.push(RegionKind::Basic { block_id });
                }
            }

            let n = regions.len();
            let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut in_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut edges: Vec<Option<RegionEdge>> = Vec::new();

            for cfg_node in &ordered {
                let from = cfg_to_region[cfg_node];
                // Orient the arms using the existing helper — it handles all
                // four irdst shapes (Pred/Pred, Pred/True, True/Pred, Pred).
                let ir_block = match s.get_ir_block_by_cfg_node(*cfg_node) {
                    Some(b) => b,
                    None => continue,
                };
                let targets = s.extract_if_targets_and_condition(ir_block);

                // Deterministic succs ordering: sort by target block id.
                let mut succs: Vec<NodeIndex> = s
                    .cfg
                    .neighbors_directed(*cfg_node, Direction::Outgoing)
                    .collect();
                succs.sort_by_key(|&n| s.cfg[n].id);

                // Single-CFG-edge blocks never represent a real CFG-level
                // branch from the structurizer's perspective: any embedded
                // predicated jump/return is rendered inside the block by the
                // pretty printer (`if (P0) return;`), so the outgoing edge is
                // effectively unconditional.
                let single_edge = succs.len() == 1;

                for succ in succs.iter().copied() {
                    let to = cfg_to_region[&succ];
                    let cond_arm = if single_edge {
                        None
                    } else {
                        match &targets {
                            Some((_, t, Some(f))) => {
                                if succ == *t {
                                    Some(true)
                                } else if succ == *f {
                                    Some(false)
                                } else {
                                    None
                                }
                            }
                            Some((_, t, None)) => {
                                if succ == *t {
                                    Some(true)
                                } else {
                                    Some(false)
                                }
                            }
                            None => None,
                        }
                    };
                    let is_back_edge = s.node_is_dominated_by(&s.dom, *cfg_node, succ);
                    let edge_idx = edges.len();
                    edges.push(Some(RegionEdge {
                        from,
                        to,
                        cond_arm,
                        is_back_edge,
                        is_goto: false,
                    }));
                    out_edges[from.0].push(edge_idx);
                    in_edges[to.0].push(edge_idx);
                }
            }

            // Entry region: same entry-picking logic as the old code.
            let entry_cfg =
                Structurizer::choose_entry_node(s.cfg).unwrap_or_else(|| NodeIndex::new(0));
            let entry = cfg_to_region
                .get(&entry_cfg)
                .copied()
                .unwrap_or(RegionId(0));

            let mut graph = RegionGraph {
                regions,
                edges,
                out_edges,
                in_edges,
                entry,
            };

            // Heuristic preprocessing (mirrors the OLD `select_if_merge_node`
            // behavior at structurizer.rs:274-295): for every 2-way head whose
            // true arm reaches the false arm in the CFG (or vice versa) but
            // not the other way around, the reached arm is the local "join"
            // — the reaching arm is the body of a one-armed if. Drop the
            // direct edge from head to merge (the "shortcut") so the collapse
            // engine sees a 1-way conditional head, and let `try_if_then_one_arm`
            // pick it up. This is intentionally lossy in the same way the old
            // structurizer was: it favors structured output over preserving
            // every CFG edge of an irreducible diamond.
            graph.drop_shortcut_branch_edges(s);

            graph
        }

        /// See `build_from_cfg`'s comment about the OLD heuristic.
        fn drop_shortcut_branch_edges(&mut self, s: &Structurizer<'_>) {
            let rids: Vec<RegionId> = (0..self.regions.len()).map(RegionId).collect();
            for r in rids {
                if !self.is_alive(r) {
                    continue;
                }
                let succs = self.live_succs(r);
                if succs.len() != 2 {
                    continue;
                }
                // Identify true/false arms by cond_arm (set by build_from_cfg).
                let (t_idx, t_e, f_idx, f_e) = match (&succs[0], &succs[1]) {
                    ((i0, e0), (i1, e1))
                        if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) =>
                    {
                        (*i0, e0.clone(), *i1, e1.clone())
                    }
                    ((i0, e0), (i1, e1))
                        if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) =>
                    {
                        (*i1, e1.clone(), *i0, e0.clone())
                    }
                    _ => continue,
                };
                if t_e.is_back_edge || f_e.is_back_edge {
                    continue;
                }
                // Map regions back to CFG nodes for reachability lookups.
                let head_block_id = match &self.regions[r.0] {
                    RegionKind::Basic { block_id, .. } => *block_id,
                    _ => continue,
                };
                let head_cfg = match s.cfg.node_indices().find(|&n| s.cfg[n].id == head_block_id) {
                    Some(n) => n,
                    None => continue,
                };
                let t_block_id = primary_block_id_of(self, t_e.to);
                let f_block_id = primary_block_id_of(self, f_e.to);
                let t_cfg = match s.cfg.node_indices().find(|&n| s.cfg[n].id == t_block_id) {
                    Some(n) => n,
                    None => continue,
                };
                let f_cfg = match s.cfg.node_indices().find(|&n| s.cfg[n].id == f_block_id) {
                    Some(n) => n,
                    None => continue,
                };
                // CFG reachability that doesn't loop back through the head.
                let t_reaches_f = has_cfg_path_excluding(s, t_cfg, f_cfg, head_cfg);
                let f_reaches_t = has_cfg_path_excluding(s, f_cfg, t_cfg, head_cfg);
                // Only act when exactly one direction reaches.
                if t_reaches_f && !f_reaches_t {
                    self.remove_edge(f_idx);
                } else if f_reaches_t && !t_reaches_f {
                    self.remove_edge(t_idx);
                }
            }
        }

        // ------------- queries -------------

        pub(super) fn region(&self, r: RegionId) -> &RegionKind {
            &self.regions[r.0]
        }

        pub(super) fn is_alive(&self, r: RegionId) -> bool {
            !matches!(self.regions[r.0], RegionKind::Tombstone)
        }

        pub(super) fn entry(&self) -> RegionId {
            self.entry
        }

        pub(super) fn active_ids(&self) -> Vec<RegionId> {
            (0..self.regions.len())
                .filter(|&i| !matches!(self.regions[i], RegionKind::Tombstone))
                .map(RegionId)
                .collect()
        }

        pub(super) fn active_count(&self) -> usize {
            self.regions
                .iter()
                .filter(|r| !matches!(r, RegionKind::Tombstone))
                .count()
        }

        /// All live (non-goto, non-removed) successor edges of `r`.
        /// Returns `(edge_index, &edge)` pairs so callers can mutate.
        pub(super) fn live_succs(&self, r: RegionId) -> Vec<(usize, RegionEdge)> {
            self.out_edges[r.0]
                .iter()
                .filter_map(|&ei| {
                    let e = self.edges[ei].as_ref()?;
                    if e.is_goto {
                        return None;
                    }
                    Some((ei, e.clone()))
                })
                .collect()
        }

        /// All live predecessor edges of `r`.
        pub(super) fn live_preds(&self, r: RegionId) -> Vec<(usize, RegionEdge)> {
            self.in_edges[r.0]
                .iter()
                .filter_map(|&ei| {
                    let e = self.edges[ei].as_ref()?;
                    if e.is_goto {
                        return None;
                    }
                    Some((ei, e.clone()))
                })
                .collect()
        }

        // ------------- mutation -------------

        pub(super) fn add_region(&mut self, kind: RegionKind) -> RegionId {
            let rid = RegionId(self.regions.len());
            self.regions.push(kind);
            self.out_edges.push(Vec::new());
            self.in_edges.push(Vec::new());
            rid
        }

        /// Mark an edge as removed from the graph. Adjacency entries are
        /// cleaned lazily by the edge-iteration helpers (they skip `None`).
        pub(super) fn remove_edge(&mut self, edge_idx: usize) {
            self.edges[edge_idx] = None;
        }

        fn push_edge(&mut self, e: RegionEdge) -> usize {
            let idx = self.edges.len();
            let from = e.from;
            let to = e.to;
            self.edges.push(Some(e));
            self.out_edges[from.0].push(idx);
            self.in_edges[to.0].push(idx);
            idx
        }

        pub(super) fn add_edge(
            &mut self,
            from: RegionId,
            to: RegionId,
            cond_arm: Option<bool>,
            is_back_edge: bool,
        ) -> usize {
            self.push_edge(RegionEdge {
                from,
                to,
                cond_arm,
                is_back_edge,
                is_goto: false,
            })
        }

        /// Tombstone a region. Caller is responsible for having already
        /// removed or rewired every edge incident to it.
        pub(super) fn tombstone(&mut self, r: RegionId) {
            self.regions[r.0] = RegionKind::Tombstone;
            self.out_edges[r.0].clear();
            self.in_edges[r.0].clear();
        }

        /// Rewire every live incoming edge `X → old` to `X → new`. Back-edge
        /// status and cond_arm are preserved.
        pub(super) fn redirect_in_edges(&mut self, old: RegionId, new: RegionId) {
            let in_edge_idxs: Vec<usize> = self.in_edges[old.0].clone();
            for ei in in_edge_idxs {
                if let Some(edge) = self.edges[ei].as_mut() {
                    edge.to = new;
                    self.in_edges[new.0].push(ei);
                }
            }
            self.in_edges[old.0].clear();
        }

        /// Rewire every live outgoing edge `old → Y` to `new → Y`.
        pub(super) fn redirect_out_edges(&mut self, old: RegionId, new: RegionId) {
            let out_edge_idxs: Vec<usize> = self.out_edges[old.0].clone();
            for ei in out_edge_idxs {
                if let Some(edge) = self.edges[ei].as_mut() {
                    edge.from = new;
                    self.out_edges[new.0].push(ei);
                }
            }
            self.out_edges[old.0].clear();
        }

        pub(super) fn set_entry(&mut self, r: RegionId) {
            self.entry = r;
        }
    }

    // ------------- helpers -------------

    /// BFS reachability from `start` to `goal` in the static CFG, treating
    /// `excluded` as removed (so paths that would otherwise loop back through
    /// the original branch head don't count as "reaching").
    fn has_cfg_path_excluding(
        s: &Structurizer<'_>,
        start: NodeIndex,
        goal: NodeIndex,
        excluded: NodeIndex,
    ) -> bool {
        if start == goal {
            return true;
        }
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        let mut work: VecDeque<NodeIndex> = VecDeque::new();
        seen.insert(start);
        seen.insert(excluded);
        work.push_back(start);
        while let Some(n) = work.pop_front() {
            for succ in s.cfg.neighbors_directed(n, Direction::Outgoing) {
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

    /// Build a Return-terminated statement for an IR block whose last stmt
    /// is an unconditional RET/EXIT. Mirrors structurizer.rs:461-498.
    fn build_return_stmt(ir_block: &IRBlock) -> StructuredStatement {
        let ret_val = ir_block.stmts.last().and_then(|s| {
            if let RValue::Op { opcode, args } = &s.value {
                if (opcode.starts_with("RET") || opcode == "EXIT") && !args.is_empty() {
                    return Some(args[0].clone());
                }
            }
            None
        });
        let mut pre_return_stmts = ir_block.stmts.clone();
        if let Some(IRStatement {
            value: RValue::Op { opcode, .. },
            pred,
            ..
        }) = pre_return_stmts.last()
        {
            if pred.is_none() && Structurizer::is_return_opcode(opcode) {
                pre_return_stmts.pop();
            }
        }
        let has_renderable_pre_return_stmt = pre_return_stmts.iter().any(|s| {
            if matches!(s.value, RValue::Phi(_)) {
                return false;
            }
            match &s.value {
                RValue::Op { opcode, .. } => {
                    !Structurizer::is_branch_only_opcode(opcode)
                        && !Structurizer::is_return_opcode(opcode)
                }
                _ => true,
            }
        });
        let mut seq = Vec::new();
        if has_renderable_pre_return_stmt {
            seq.push(StructuredStatement::BasicBlock {
                block_id: ir_block.id,
                stmts: pre_return_stmts,
            });
        }
        seq.push(StructuredStatement::Return(ret_val));
        if seq.len() == 1 {
            seq.into_iter().next().unwrap()
        } else {
            StructuredStatement::Sequence(seq)
        }
    }

    /// Materialize a region's StructuredStatement — cloning for Composite or
    /// building a fresh `BasicBlock` for Basic regions.
    fn stmt_for_region(
        graph: &RegionGraph,
        r: RegionId,
        s: &Structurizer<'_>,
    ) -> StructuredStatement {
        match graph.region(r) {
            RegionKind::Basic { block_id, .. } => {
                let stmts = s
                    .get_ir_block(*block_id)
                    .map(|b| b.stmts.clone())
                    .unwrap_or_default();
                StructuredStatement::BasicBlock {
                    block_id: *block_id,
                    stmts,
                }
            }
            RegionKind::Composite { stmt, .. } => stmt.clone(),
            RegionKind::Tombstone => StructuredStatement::Empty,
        }
    }

    fn primary_block_id_of(graph: &RegionGraph, r: RegionId) -> usize {
        match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            RegionKind::Composite {
                primary_block_id, ..
            } => *primary_block_id,
            RegionKind::Tombstone => usize::MAX,
        }
    }

    /// Concatenate two statements into a flattened Sequence, dropping Empty
    /// children and collapsing nested Sequences.
    fn concat_stmts(a: StructuredStatement, b: StructuredStatement) -> StructuredStatement {
        let mut out: Vec<StructuredStatement> = Vec::new();
        match a {
            StructuredStatement::Empty => {}
            StructuredStatement::Sequence(v) => {
                for s in v {
                    if !matches!(s, StructuredStatement::Empty) {
                        out.push(s);
                    }
                }
            }
            other => out.push(other),
        }
        match b {
            StructuredStatement::Empty => {}
            StructuredStatement::Sequence(v) => {
                for s in v {
                    if !matches!(s, StructuredStatement::Empty) {
                        out.push(s);
                    }
                }
            }
            other => out.push(other),
        }
        match out.len() {
            0 => StructuredStatement::Empty,
            1 => out.into_iter().next().unwrap(),
            _ => StructuredStatement::Sequence(out),
        }
    }

    // ------------- collapse rules -------------

    /// Sequence collapse: fold `A → B` when neither end has a branch structure
    /// still pending. Forbidden from absorbing into a 2-way head or a
    /// back-edge tail (see plan §7).
    pub(super) fn try_seq(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let succs = graph.live_succs(r);
        if succs.len() != 1 {
            return false;
        }
        let (_, a_to_b) = &succs[0];
        if a_to_b.cond_arm.is_some() || a_to_b.is_back_edge {
            return false;
        }
        let b = a_to_b.to;
        if b == r {
            return false;
        }
        let preds_b = graph.live_preds(b);
        if preds_b.len() != 1 {
            return false;
        }
        let succs_b = graph.live_succs(b);
        match succs_b.len() {
            0 => {}
            1 => {
                let (_, out_b) = &succs_b[0];
                if out_b.cond_arm.is_some() || out_b.is_back_edge {
                    return false;
                }
            }
            2 => {
                // Multi-block do-while shape: A → B(test), B → A (back-edge)
                // and B → X (forward exit). After folding A and B together,
                // the back-edge to A becomes a self-back-edge on the new
                // composite, which `try_do_while` then matches. Refuse any
                // other 2-successor pattern (the partition rule keeps 2-way
                // forward branches in the hands of `try_if_*` / `try_while_do`).
                let back_to_a = succs_b.iter().any(|(_, e)| e.to == r && e.is_back_edge);
                let forward_other = succs_b.iter().any(|(_, e)| e.to != r && !e.is_back_edge);
                if !(back_to_a && forward_other) {
                    return false;
                }
            }
            _ => return false,
        }

        // Build the new Composite statement.
        let a_stmt = stmt_for_region(graph, r, s);
        let b_stmt = stmt_for_region(graph, b, s);
        let new_stmt = concat_stmts(a_stmt, b_stmt);
        let new_primary = primary_block_id_of(graph, b);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: new_stmt,
            primary_block_id: new_primary,
        });

        // Rewire: X→A edges (other than A→B) now go X→NEW.
        // The only incoming edge of B is A→B; remove it before redirecting.
        // The A→B edge itself is removed.
        let a_to_b_idx = succs[0].0;
        graph.remove_edge(a_to_b_idx);

        // Pull all live X→A edges onto NEW.
        graph.redirect_in_edges(r, new_rid);
        // Pull all live B→Y edges onto NEW.
        graph.redirect_out_edges(b, new_rid);

        // Entry update.
        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        // Note: B can't have been entry if r had the only pred of B and r is
        // alive, but handle it defensively.
        if graph.entry() == b {
            graph.set_entry(new_rid);
        }

        graph.tombstone(r);
        graph.tombstone(b);
        true
    }

    // ------------- if/else collapse -------------

    /// Try to collapse `A` as an `if-then-else`: `A` has two oriented arms to
    /// distinct regions `T` and `F`, both arms are forward and converge at a
    /// common merge `M` (or both terminate). The condition expression comes
    /// from `A` itself (which must remain Basic so the prelude printer can
    /// emit its statements before the `if`).
    pub(super) fn try_if_else(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        // Heads must be Basic so condition_block_id refers to a real IR block
        // and the prelude renderer can emit A's statements before the if.
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };

        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }
        // Identify the true/false arms.
        let (true_idx, true_e, false_idx, false_e) = match (&succs[0], &succs[1]) {
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) => {
                (*i0, e0.clone(), *i1, e1.clone())
            }
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) => {
                (*i1, e1.clone(), *i0, e0.clone())
            }
            _ => return false,
        };
        if true_e.is_back_edge || false_e.is_back_edge {
            return false;
        }
        let t = true_e.to;
        let f = false_e.to;
        if t == r || f == r || t == f {
            return false;
        }
        // Both branches must be single-entry from `r`.
        let preds_t = graph.live_preds(t);
        let preds_f = graph.live_preds(f);
        if preds_t.len() != 1 || preds_f.len() != 1 {
            return false;
        }
        let succs_t = graph.live_succs(t);
        let succs_f = graph.live_succs(f);
        // Each branch is either terminal (no live succs) or has a single
        // unconditional forward edge to a common merge.
        if succs_t.len() > 1 || succs_f.len() > 1 {
            return false;
        }
        let t_merge = succs_t.first().map(|(_, e)| e.clone());
        let f_merge = succs_f.first().map(|(_, e)| e.clone());
        let merge: Option<RegionId> = match (&t_merge, &f_merge) {
            (None, None) => None,
            (Some(et), Some(ef)) => {
                if et.to != ef.to
                    || et.cond_arm.is_some()
                    || ef.cond_arm.is_some()
                    || et.is_back_edge
                    || ef.is_back_edge
                {
                    return false;
                }
                Some(et.to)
            }
            // Asymmetric merges (one terminal, one not) — fall through to
            // try_if_then if appropriate.
            _ => return false,
        };

        // Build statement.
        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let then_stmt = stmt_for_region(graph, t, s);
        let else_stmt = stmt_for_region(graph, f, s);
        let if_stmt = StructuredStatement::If {
            condition_block_id: head_block_id,
            condition_expr: cond_expr,
            then_branch: Box::new(then_stmt),
            else_branch: Some(Box::new(else_stmt)),
        };

        // Build the new Composite. Its primary block id is the merge if there
        // is one, else the head's id.
        let new_primary = if let Some(m) = merge {
            primary_block_id_of(graph, m)
        } else {
            head_block_id
        };
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: if_stmt,
            primary_block_id: new_primary,
        });

        // Edge surgery.
        graph.remove_edge(true_idx);
        graph.remove_edge(false_idx);
        if let Some((tm_idx, _)) = succs_t.first() {
            graph.remove_edge(*tm_idx);
        }
        if let Some((fm_idx, _)) = succs_f.first() {
            graph.remove_edge(*fm_idx);
        }
        graph.redirect_in_edges(r, new_rid);
        if let Some(m) = merge {
            // The new region falls through to the merge.
            graph.add_edge(new_rid, m, None, false);
        }
        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(t);
        graph.tombstone(f);
        true
    }

    /// Try to collapse `A` as a one-armed `if`: `A` has two oriented arms,
    /// one going to a body `T` that re-joins via the *other* arm's target as
    /// the implicit merge. If the body is on the false arm, we negate.
    pub(super) fn try_if_then(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };
        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }
        let (true_idx, true_e, false_idx, false_e) = match (&succs[0], &succs[1]) {
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) => {
                (*i0, e0.clone(), *i1, e1.clone())
            }
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) => {
                (*i1, e1.clone(), *i0, e0.clone())
            }
            _ => return false,
        };
        if true_e.is_back_edge || false_e.is_back_edge {
            return false;
        }
        let t = true_e.to;
        let f = false_e.to;
        if t == r || f == r || t == f {
            return false;
        }

        // Try with the true arm as the body, then the false arm as the body.
        // body_side: which arm holds the body, the other is the implicit merge.
        // A body is acceptable if it is single-entry from `r` AND either
        //   (a) terminates with no live successors (e.g. a seeded return
        //       composite), or
        //   (b) has a single unconditional forward edge back to `merge`.
        let attempt = |graph: &RegionGraph, body: RegionId, merge: RegionId| -> bool {
            let preds_b = graph.live_preds(body);
            if preds_b.len() != 1 {
                return false;
            }
            let succs_b = graph.live_succs(body);
            if succs_b.is_empty() {
                // Terminal body (e.g. seeded return). Only safe when this
                // is not actually a while-loop shape: reject if the merge
                // arm has a back-edge to `r`, signaling that `r` is a loop
                // header and `try_while_do` should run instead.
                for (_, me) in graph.live_succs(merge) {
                    if me.is_back_edge && me.to == r {
                        return false;
                    }
                }
                return true;
            }
            if succs_b.len() != 1 {
                return false;
            }
            let (_, eb) = &succs_b[0];
            if eb.cond_arm.is_some() || eb.is_back_edge || eb.to != merge {
                return false;
            }
            true
        };

        let (body, body_idx, body_edge_to_merge_idx, negate) = if attempt(graph, t, f) {
            let mi = graph.live_succs(t).first().map(|(i, _)| *i);
            (t, true_idx, mi, false)
        } else if attempt(graph, f, t) {
            let mi = graph.live_succs(f).first().map(|(i, _)| *i);
            (f, false_idx, mi, true)
        } else {
            return false;
        };
        let merge = if negate { t } else { f };
        let other_arm_idx = if negate { true_idx } else { false_idx };

        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let cond_oriented = if negate {
            negate_condition(cond_expr)
        } else {
            cond_expr
        };
        let then_stmt = stmt_for_region(graph, body, s);
        let if_stmt = StructuredStatement::If {
            condition_block_id: head_block_id,
            condition_expr: cond_oriented,
            then_branch: Box::new(then_stmt),
            else_branch: None,
        };

        let new_primary = primary_block_id_of(graph, merge);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: if_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(body_idx);
        graph.remove_edge(other_arm_idx);
        if let Some(mi) = body_edge_to_merge_idx {
            graph.remove_edge(mi);
        }
        graph.redirect_in_edges(r, new_rid);
        graph.add_edge(new_rid, merge, None, false);

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(body);
        true
    }

    /// One-arm if-then collapse: handles 2-way Basic heads where one arm has
    /// already been stripped by `drop_shortcut_branch_edges`. The remaining
    /// edge still carries `cond_arm = Some(_)`, the body is reached only via
    /// that conditional edge, and the body's natural exit is the implicit
    /// merge — exactly the shape the OLD heuristic produced for diamonds with
    /// a "shortcut" edge.
    pub(super) fn try_if_then_one_arm(
        graph: &mut RegionGraph,
        r: RegionId,
        s: &Structurizer<'_>,
    ) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };
        let succs = graph.live_succs(r);
        if succs.len() != 1 {
            return false;
        }
        let (body_idx, body_e) = (succs[0].0, succs[0].1.clone());
        let arm = match body_e.cond_arm {
            Some(a) => a,
            None => return false,
        };
        if body_e.is_back_edge {
            return false;
        }
        let body = body_e.to;
        if body == r {
            return false;
        }
        // Body must be single-entry from us so we can absorb it.
        let preds_b = graph.live_preds(body);
        if preds_b.len() != 1 {
            return false;
        }
        // The body must terminate or fall into a single forward merge.
        let succs_b = graph.live_succs(body);
        let merge_opt: Option<RegionId> = match succs_b.len() {
            0 => None,
            1 => {
                let (_, eb) = &succs_b[0];
                if eb.cond_arm.is_some() || eb.is_back_edge {
                    return false;
                }
                Some(eb.to)
            }
            _ => return false,
        };

        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let cond_oriented = if arm {
            cond_expr
        } else {
            negate_condition(cond_expr)
        };
        let body_stmt = stmt_for_region(graph, body, s);
        let if_stmt = StructuredStatement::If {
            condition_block_id: head_block_id,
            condition_expr: cond_oriented,
            then_branch: Box::new(body_stmt),
            else_branch: None,
        };

        let new_primary = match merge_opt {
            Some(m) => primary_block_id_of(graph, m),
            None => head_block_id,
        };
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: if_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(body_idx);
        if let Some((mi, _)) = succs_b.first() {
            graph.remove_edge(*mi);
        }
        graph.redirect_in_edges(r, new_rid);
        if let Some(m) = merge_opt {
            graph.add_edge(new_rid, m, None, false);
        }

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(body);
        true
    }

    // ------------- loop collapse -------------

    /// While-do loop: head `H` is a Basic 2-way branch where one arm goes to
    /// a body `B` (which then back-edges to `H`) and the other arm exits.
    ///
    /// The structurizer used to set `Loop::header_block_id = Some(H)`, which
    /// drove the printer to render `H`'s IR statements as a *prelude* before
    /// the loop construct *and* the body Sequence still embedded `H`,
    /// duplicating it. We avoid that by emitting a Sequence with `H` as a
    /// plain BasicBlock followed by the Loop with `header_block_id = None`.
    pub(super) fn try_while_do(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };
        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }
        let (true_idx, true_e, false_idx, false_e) = match (&succs[0], &succs[1]) {
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) => {
                (*i0, e0.clone(), *i1, e1.clone())
            }
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) => {
                (*i1, e1.clone(), *i0, e0.clone())
            }
            _ => return false,
        };
        // For a while-do, neither arm of the head is the back edge — the back
        // edge is from the body region.
        if true_e.is_back_edge || false_e.is_back_edge {
            return false;
        }

        // Try body=true,exit=false; then swap.
        let try_arm = |graph: &RegionGraph, body: RegionId| -> bool {
            if body == r {
                return false;
            }
            let preds_b = graph.live_preds(body);
            if preds_b.len() != 1 {
                return false;
            }
            let succs_b = graph.live_succs(body);
            if succs_b.len() != 1 {
                return false;
            }
            let (_, eb) = &succs_b[0];
            // Body's only successor must be the head, via a back edge.
            if eb.to != r || !eb.is_back_edge {
                return false;
            }
            true
        };

        let (body, exit, negate) = if try_arm(graph, true_e.to) {
            (true_e.to, false_e.to, false)
        } else if try_arm(graph, false_e.to) {
            (false_e.to, true_e.to, true)
        } else {
            return false;
        };

        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let cond_oriented = if negate {
            negate_condition(cond_expr)
        } else {
            cond_expr
        };
        let body_stmt = stmt_for_region(graph, body, s);
        let head_stmt = stmt_for_region(graph, r, s);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::While,
            header_block_id: None,
            condition_expr: Some(cond_oriented),
            body: Box::new(body_stmt),
        };
        let new_stmt = concat_stmts(head_stmt, loop_stmt);

        // Find the body→head back edge index so we can remove it.
        let back_edge_idx: Option<usize> = graph
            .live_succs(body)
            .iter()
            .find(|(_, e)| e.to == r && e.is_back_edge)
            .map(|(i, _)| *i);

        let new_primary = primary_block_id_of(graph, exit);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: new_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(true_idx);
        graph.remove_edge(false_idx);
        if let Some(bi) = back_edge_idx {
            graph.remove_edge(bi);
        }
        // Note: redirect_in_edges only moves still-live entries; the body's
        // back-edge from r is already removed.
        graph.redirect_in_edges(r, new_rid);
        graph.add_edge(new_rid, exit, None, false);

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(body);
        true
    }

    /// Do-while loop: a region `R` whose only outgoing edges are a self
    /// back-edge (`R → R`, conditional) and a forward exit (`R → X`,
    /// conditional, opposite arm). After `try_seq` folds a multi-block loop
    /// body, the back edge becomes this self-edge.
    pub(super) fn try_do_while(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }

        // Identify the self back-edge and the forward exit.
        let mut back: Option<(usize, RegionEdge)> = None;
        let mut exit: Option<(usize, RegionEdge)> = None;
        for (ei, e) in &succs {
            if e.to == r && e.is_back_edge {
                back = Some((*ei, e.clone()));
            } else if !e.is_back_edge {
                exit = Some((*ei, e.clone()));
            }
        }
        let (back_idx, back_e) = match back {
            Some(x) => x,
            None => return false,
        };
        let (exit_idx, exit_e) = match exit {
            Some(x) => x,
            None => return false,
        };
        // Both arms must be predicate-driven, opposite.
        let (b_arm, x_arm) = (back_e.cond_arm, exit_e.cond_arm);
        match (b_arm, x_arm) {
            (Some(true), Some(false)) | (Some(false), Some(true)) => {}
            _ => return false,
        }

        // The condition expression lives on the *tail* basic block (the one
        // whose irdst spawned this 2-way branch). Look it up by primary id.
        let tail_bid = primary_block_id_of(graph, r);
        let ir_block = match s.get_ir_block(tail_bid) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        // Orient so that "true" means keep looping (i.e. take the back edge).
        let cond_oriented = if b_arm == Some(true) {
            cond_expr
        } else {
            negate_condition(cond_expr)
        };

        let body_stmt = stmt_for_region(graph, r, s);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::DoWhile,
            header_block_id: Some(tail_bid),
            condition_expr: Some(cond_oriented),
            body: Box::new(body_stmt),
        };

        let exit_target = exit_e.to;
        let new_primary = primary_block_id_of(graph, exit_target);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: loop_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(back_idx);
        graph.remove_edge(exit_idx);
        graph.redirect_in_edges(r, new_rid);
        graph.add_edge(new_rid, exit_target, None, false);

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        true
    }

    /// Degenerate self-loop: `R` has exactly one live successor which is the
    /// self back-edge (no exit). Emits an `Endless` loop.
    pub(super) fn try_self_loop(
        graph: &mut RegionGraph,
        r: RegionId,
        s: &Structurizer<'_>,
    ) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let succs = graph.live_succs(r);
        if succs.len() != 1 {
            return false;
        }
        let (back_idx, back_e) = (succs[0].0, succs[0].1.clone());
        if back_e.to != r || !back_e.is_back_edge {
            return false;
        }
        let body_stmt = stmt_for_region(graph, r, s);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::Endless,
            header_block_id: None,
            condition_expr: None,
            body: Box::new(body_stmt),
        };
        let new_primary = primary_block_id_of(graph, r);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: loop_stmt,
            primary_block_id: new_primary,
        });
        graph.remove_edge(back_idx);
        graph.redirect_in_edges(r, new_rid);
        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        true
    }

    // ------------- switch recognition -------------

    /// Try to fold a node with 3+ outgoing edges into a Switch statement.
    ///
    /// A switch candidate is a region with ≥3 live successors (from a branch
    /// table or cascaded conditional pattern), where all successors converge
    /// to a common post-dominator. Each successor becomes a case arm.
    ///
    /// This is intentionally conservative: it only fires when ALL case arms
    /// are leaf nodes (no further control flow) that converge to the same
    /// merge point, avoiding complex nesting issues.
    fn try_switch(graph: &mut RegionGraph, head: RegionId, s: &Structurizer<'_>) -> bool {
        let succs = graph.live_succs(head);
        if succs.len() < 3 {
            return false;
        }

        // Check that all successors are leaves: each has at most 1 successor,
        // and all share the same single successor (the merge point).
        let mut merge_target: Option<RegionId> = None;
        let mut leaf_succs: Vec<(RegionId, usize)> = Vec::new(); // (succ_id, edge_idx)

        for (ei, e) in &succs {
            let succ_succs = graph.live_succs(e.to);
            if succ_succs.len() > 1 {
                return false; // case arm has branches — too complex
            }
            if succ_succs.len() == 1 {
                let merge = succ_succs[0].1.to;
                match merge_target {
                    None => merge_target = Some(merge),
                    Some(m) if m == merge => {}
                    Some(_) => return false, // different merge points
                }
            }
            // succ_succs.len() == 0 means terminal (return/exit) — allowed
            leaf_succs.push((e.to, *ei));
        }

        // At least one arm must have a merge target (otherwise they're all terminals,
        // which wouldn't benefit from switch folding).
        // Actually, even all-terminal is fine for a switch — we just don't need a merge.

        // Build the switch statement.
        let header_bid = primary_block_id_of(graph, head);
        let head_stmt = stmt_for_region(graph, head, s);

        let mut cases = Vec::new();
        for (idx, (succ_rid, _edge_idx)) in leaf_succs.iter().enumerate() {
            let case_body = stmt_for_region(graph, *succ_rid, s);
            cases.push((idx, case_body));
        }

        let switch_stmt = StructuredStatement::Switch {
            header_block_id: header_bid,
            discriminant: None, // Could be extracted from BRX operand in future
            cases,
            default: None,
        };

        // If there's a head statement (prelude code), wrap it in a sequence.
        let final_stmt = match head_stmt {
            StructuredStatement::Empty => switch_stmt,
            _ => StructuredStatement::Sequence(vec![head_stmt, switch_stmt]),
        };

        // Replace head with the switch composite.
        graph.regions[head.0] = RegionKind::Composite {
            stmt: final_stmt,
            primary_block_id: header_bid,
        };

        // Remove all edges from head to succs
        for (ei, _) in &succs {
            graph.remove_edge(*ei);
        }

        // Tombstone all case-arm regions
        for (succ_rid, _) in &leaf_succs {
            // Remove outgoing edges from the case arm to the merge target
            for (sei, _) in graph.live_succs(*succ_rid) {
                graph.remove_edge(sei);
            }
            // Remove incoming edges to the case arm
            for (pei, _) in graph.live_preds(*succ_rid) {
                graph.remove_edge(pei);
            }
            graph.tombstone(*succ_rid);
        }

        // If there's a merge target, add an edge from head to merge
        if let Some(merge) = merge_target {
            graph.add_edge(head, merge, None, false);
        }

        true
    }

    // ------------- node splitting for irreducible CFG -------------

    /// Maximum IR statements in a Basic region that we're willing to duplicate.
    const SPLIT_MAX_STMTS: usize = 8;
    /// Maximum number of node splits per structurization to avoid blowup.
    const SPLIT_BUDGET: usize = 12;

    /// Try to make the region graph reducible by splitting a small node that
    /// has multiple non-back-edge predecessors.
    ///
    /// An irreducible region typically has a node reachable via two different
    /// forward paths — it has >1 non-back-edge predecessor from distinct
    /// regions. By duplicating the target and redirecting one predecessor to
    /// the copy, the region becomes reducible and the normal fold rules can
    /// proceed.
    ///
    /// We only split Basic regions with ≤ `SPLIT_MAX_STMTS` IR statements to
    /// keep the output compact. Returns true if a split happened.
    fn try_split_for_reducibility(
        graph: &mut RegionGraph,
        s: &Structurizer<'_>,
        split_count: &mut usize,
    ) -> bool {
        if *split_count >= SPLIT_BUDGET {
            return false;
        }

        // Find a candidate: a live region with ≥2 non-back-edge, non-goto predecessors.
        let mut best: Option<(RegionId, Vec<(usize, RegionEdge)>)> = None;
        for rid in graph.active_ids() {
            if rid == graph.entry() {
                continue; // never split the entry node
            }
            let preds = graph.live_preds(rid);
            let non_back: Vec<_> = preds
                .into_iter()
                .filter(|(_, e)| !e.is_back_edge && !e.is_goto)
                .collect();
            if non_back.len() < 2 {
                continue;
            }
            // Only split small Basic regions.
            let stmt_count = match graph.region(rid) {
                RegionKind::Basic { block_id } => s
                    .function_ir
                    .blocks
                    .iter()
                    .find(|b| b.id == *block_id)
                    .map(|b| b.stmts.len())
                    .unwrap_or(usize::MAX),
                _ => usize::MAX,
            };
            if stmt_count > SPLIT_MAX_STMTS {
                continue;
            }
            // Prefer the candidate with the smallest region id (deterministic).
            if best.as_ref().map_or(true, |(cur, _)| rid.0 < cur.0) {
                best = Some((rid, non_back));
            }
        }

        let Some((target, non_back_preds)) = best else {
            return false;
        };

        // Split: duplicate the target region.
        let new_kind = graph.region(target).clone();
        let copy = graph.add_region(new_kind);

        // Duplicate all outgoing edges from `target` to `copy`.
        let out_edges: Vec<(RegionId, Option<bool>, bool)> = graph
            .live_succs(target)
            .into_iter()
            .map(|(_, e)| (e.to, e.cond_arm, e.is_back_edge))
            .collect();
        for (to, arm, be) in out_edges {
            graph.add_edge(copy, to, arm, be);
        }

        // Redirect one of the non-back-edge predecessors to point to the copy
        // instead of the original. Pick the one with the largest from-region-id
        // (deterministic, and avoids touching the entry-side path).
        let redirect_pred = non_back_preds
            .iter()
            .max_by_key(|(_, e)| e.from.0)
            .map(|(ei, _)| *ei);
        if let Some(ei) = redirect_pred {
            if let Some(edge) = graph.edges[ei].as_mut() {
                // Remove old in-edge association
                graph.in_edges[target.0].retain(|&x| x != ei);
                edge.to = copy;
                graph.in_edges[copy.0].push(ei);
            }
        }

        *split_count += 1;
        true
    }

    // ------------- goto fallback -------------

    /// Minimal Phase A fallback: if the collapse loop stalled, walk the graph
    /// and mark the first live edge as goto. Later phases will pick more
    /// intelligently.
    pub(super) fn select_goto_edge(graph: &mut RegionGraph, s: &Structurizer<'_>) -> bool {
        // Deterministic selection: smallest (from, to) region id pair.
        let mut candidate: Option<(RegionId, RegionId, usize, Option<bool>)> = None;
        for rid in graph.active_ids() {
            for (ei, e) in graph.live_succs(rid) {
                let key = (e.from, e.to, ei, e.cond_arm);
                candidate = Some(match candidate {
                    None => key,
                    Some(cur) => {
                        if (key.0 .0, key.1 .0) < (cur.0 .0, cur.1 .0) {
                            key
                        } else {
                            cur
                        }
                    }
                });
            }
        }
        let Some((from_r, to_r, edge_idx, cond_arm)) = candidate else {
            return false;
        };

        // Materialize the goto into the source region's stmt. Convert the
        // source into a Composite that ends with an UnstructuredJump, so the
        // printer can emit "if (P0) goto BBxx;" or bare "goto BBxx;".
        let cur_stmt = stmt_for_region(graph, from_r, s);
        let from_bid = primary_block_id_of(graph, from_r);
        let to_bid = primary_block_id_of(graph, to_r);
        let cond_ir = cond_for_arm(graph, from_r, cond_arm, s);
        let jump = StructuredStatement::UnstructuredJump {
            from_block_id: from_bid,
            to_block_id: to_bid,
            condition: cond_ir,
        };
        let new_stmt = concat_stmts(cur_stmt, jump);
        // In place, update the source region to Composite carrying this stmt.
        // We only care to rewrite the stmt, not the edges.
        match &mut graph.regions[from_r.0] {
            RegionKind::Basic { .. } | RegionKind::Composite { .. } => {
                graph.regions[from_r.0] = RegionKind::Composite {
                    stmt: new_stmt,
                    primary_block_id: from_bid,
                };
            }
            RegionKind::Tombstone => return false,
        }
        // Flag the edge so future collapse rules skip it.
        if let Some(edge) = graph.edges[edge_idx].as_mut() {
            edge.is_goto = true;
        }
        true
    }

    /// Look up the branch predicate for the source's selected arm, so the
    /// goto fallback can emit `if (pred) goto BBxx;` rather than an
    /// unconditional jump (which would change semantics).
    fn cond_for_arm(
        graph: &RegionGraph,
        from: RegionId,
        cond_arm: Option<bool>,
        s: &Structurizer<'_>,
    ) -> Option<IRExpr> {
        let arm = cond_arm?;
        let bid = primary_block_id_of(graph, from);
        let ir_block = s.get_ir_block(bid)?;
        let (cond_expr, _, _) = s.extract_if_targets_and_condition(ir_block)?;
        Some(if arm {
            cond_expr
        } else {
            negate_condition(cond_expr)
        })
    }

    // ------------- main driver -------------

    pub(super) fn run_collapse(graph: &mut RegionGraph, s: &Structurizer<'_>) {
        let initial = graph.active_count().max(1);
        let max_iter = 4 * initial * initial + 100;
        let mut iter = 0usize;
        let mut split_count = 0usize;
        loop {
            iter += 1;
            if iter > max_iter {
                eprintln!(
                    "structurizer: collapse exceeded {} iterations; bailing",
                    max_iter
                );
                break;
            }
            let mut changed = false;
            for rid in graph.active_ids() {
                if !graph.is_alive(rid) {
                    continue;
                }
                if try_seq(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_if_else(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_if_then(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_if_then_one_arm(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_while_do(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_do_while(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_self_loop(graph, rid, s) {
                    changed = true;
                    break;
                }
                if try_switch(graph, rid, s) {
                    changed = true;
                    break;
                }
            }
            if changed {
                continue;
            }
            if graph.active_count() <= 1 {
                break;
            }
            // Before resorting to goto, try node splitting for irreducible CFG.
            if try_split_for_reducibility(graph, s, &mut split_count) {
                continue;
            }
            if !select_goto_edge(graph, s) {
                break;
            }
        }
    }

    // ------------- root emission -------------

    /// Walk reachable regions from `entry` and concatenate them. In the happy
    /// path exactly one region survives; in the fallback path we thread
    /// leftover regions together with explicit gotos already materialized
    /// into their statements.
    pub(super) fn emit_root(graph: &RegionGraph, s: &Structurizer<'_>) -> StructuredStatement {
        // DFS from entry, preorder, deterministic.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut order: Vec<RegionId> = Vec::new();
        let mut stack: Vec<RegionId> = vec![graph.entry()];
        while let Some(rid) = stack.pop() {
            if !visited.insert(rid.0) {
                continue;
            }
            if !graph.is_alive(rid) {
                continue;
            }
            order.push(rid);
            // Push successors in reverse-sorted order so we visit in
            // ascending (from, to) region-id order.
            let mut out: Vec<RegionId> = graph.out_edges[rid.0]
                .iter()
                .filter_map(|&ei| graph.edges[ei].as_ref().map(|e| e.to))
                .collect();
            out.sort_by_key(|r| r.0);
            out.dedup();
            for r in out.into_iter().rev() {
                if !visited.contains(&r.0) {
                    stack.push(r);
                }
            }
        }

        // Collect statements from the walked regions.
        let mut parts: Vec<StructuredStatement> = Vec::new();
        for rid in order {
            match graph.region(rid) {
                RegionKind::Basic { .. } => {
                    // A Basic region that survived to emission means the
                    // collapse loop stalled before folding it in. Emit its
                    // raw IR stmts via the shared lookup helper.
                    parts.push(stmt_for_region(graph, rid, s));
                }
                RegionKind::Composite { stmt, .. } => parts.push(stmt.clone()),
                RegionKind::Tombstone => {}
            }
        }
        match parts.len() {
            0 => StructuredStatement::Empty,
            1 => parts.into_iter().next().unwrap(),
            _ => StructuredStatement::Sequence(parts),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock as CfgBasicBlock, ControlFlowGraph, EdgeKind};
    use crate::ir::DefaultDisplay;
    use petgraph::graph::DiGraph;
    use std::collections::HashMap;

    fn stmt(opcode: &str) -> IRStatement {
        IRStatement {
            defs: vec![],
            value: RValue::Op {
                opcode: opcode.to_string(),
                args: vec![],
            },
            pred: None,
            mem_addr_args: None,
            pred_old_defs: vec![],
        }
    }

    fn predicated_stmt(opcode: &str, pred: IRExpr) -> IRStatement {
        IRStatement {
            defs: vec![],
            value: RValue::Op {
                opcode: opcode.to_string(),
                args: vec![],
            },
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
            StructuredStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                contains_loop(then_branch)
                    || else_branch.as_deref().map(contains_loop).unwrap_or(false)
            }
            _ => false,
        }
    }

    fn contains_goto(s: &StructuredStatement) -> bool {
        match s {
            StructuredStatement::UnstructuredJump { .. } => true,
            StructuredStatement::Sequence(v) => v.iter().any(contains_goto),
            StructuredStatement::Loop { body, .. } => contains_goto(body),
            StructuredStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                contains_goto(then_branch)
                    || else_branch.as_deref().map(contains_goto).unwrap_or(false)
            }
            _ => false,
        }
    }

    #[test]
    fn is_convergence_barrier_opcode_accepts_legacy_and_blackwell_forms() {
        // Legacy plain mnemonics must still match.
        assert!(Structurizer::is_convergence_barrier_opcode("BSSY"));
        assert!(Structurizer::is_convergence_barrier_opcode("BSYNC"));
        assert!(Structurizer::is_convergence_barrier_opcode("SSY"));
        assert!(Structurizer::is_convergence_barrier_opcode("SYNC"));
        assert!(Structurizer::is_convergence_barrier_opcode("WARPSYNC"));
        // Blackwell (SM 100+) reliability-annotated variants.
        assert!(Structurizer::is_convergence_barrier_opcode(
            "BSSY.RECONVERGENT"
        ));
        assert!(Structurizer::is_convergence_barrier_opcode("BSSY.RELIABLE"));
        assert!(Structurizer::is_convergence_barrier_opcode(
            "BSYNC.RECONVERGENT"
        ));
        assert!(Structurizer::is_convergence_barrier_opcode(
            "BSYNC.RELIABLE"
        ));
        // Unrelated opcodes must NOT match.
        assert!(!Structurizer::is_convergence_barrier_opcode("BRA"));
        assert!(!Structurizer::is_convergence_barrier_opcode(
            "BREAK.RELIABLE"
        ));
        assert!(!Structurizer::is_convergence_barrier_opcode("BREAK"));
        assert!(!Structurizer::is_convergence_barrier_opcode("IMAD"));
    }

    #[test]
    fn recovers_sequence() {
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x10)], vec![stmt("OP0")]),
            (1, 0x10, vec![(Some(IRCond::True), 0x20)], vec![stmt("OP1")]),
            (2, 0x20, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![(0, 1, EdgeKind::FallThrough), (1, 2, EdgeKind::FallThrough)];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(matches!(
            out,
            StructuredStatement::Sequence(_)
                | StructuredStatement::BasicBlock { .. }
                | StructuredStatement::Return(_)
        ));
    }

    #[test]
    fn recovers_if_then() {
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (
                0,
                0x00,
                vec![(
                    Some(IRCond::Pred {
                        reg: p0,
                        sense: true,
                    }),
                    0x10,
                )],
                vec![stmt("CMP")],
            ),
            (
                1,
                0x10,
                vec![(Some(IRCond::True), 0x20)],
                vec![stmt("THEN")],
            ),
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
    fn collapse_recovers_if_then_with_terminal_return() {
        // Shape: `if (P) return; /* fallthrough */`
        //   BB0 (cond) → BB1 (RET)      [true arm]
        //   BB0        → BB2 (TAIL→RET) [false arm / fallthrough]
        // BB1 is a seeded return composite with zero successors; the old
        // try_if_then rejected this because succs_b was empty.
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (
                0,
                0x00,
                vec![
                    (
                        Some(IRCond::Pred {
                            reg: p0.clone(),
                            sense: true,
                        }),
                        0x10,
                    ),
                    (
                        Some(IRCond::Pred {
                            reg: p0,
                            sense: false,
                        }),
                        0x20,
                    ),
                ],
                vec![stmt("CMP")],
            ),
            (1, 0x10, vec![], vec![stmt("RET")]),
            (2, 0x20, vec![], vec![stmt("TAIL"), stmt("RET")]),
        ];
        let edges = vec![(0, 1, EdgeKind::CondBranch), (0, 2, EdgeKind::FallThrough)];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(contains_if(&out), "expected if structure, got: {:?}", out);
        assert!(
            !contains_goto(&out),
            "expected no goto fallback, got: {:?}",
            out
        );
    }

    #[test]
    fn recovers_if_then_else() {
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (
                0,
                0x00,
                vec![
                    (
                        Some(IRCond::Pred {
                            reg: p0.clone(),
                            sense: true,
                        }),
                        0x10,
                    ),
                    (
                        Some(IRCond::Pred {
                            reg: p0,
                            sense: false,
                        }),
                        0x20,
                    ),
                ],
                vec![stmt("CMP")],
            ),
            (
                1,
                0x10,
                vec![(Some(IRCond::True), 0x30)],
                vec![stmt("THEN")],
            ),
            (
                2,
                0x20,
                vec![(Some(IRCond::True), 0x30)],
                vec![stmt("ELSE")],
            ),
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
                    (
                        Some(IRCond::Pred {
                            reg: p0.clone(),
                            sense: true,
                        }),
                        0x10,
                    ),
                    (
                        Some(IRCond::Pred {
                            reg: p0,
                            sense: false,
                        }),
                        0x20,
                    ),
                ],
                vec![stmt("CMP")],
            ),
            (
                1,
                0x10,
                vec![(Some(IRCond::True), 0x00)],
                vec![stmt("BODY")],
            ),
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
        assert!(!contains_goto(&out));
    }

    #[test]
    fn collapse_recovers_do_while() {
        // Single-block tail-test loop:
        //   BB0 (entry) → BB1 (loop body + test) → BB1 (back-edge) / BB2 (exit)
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x10)], vec![stmt("PRE")]),
            (
                1,
                0x10,
                vec![
                    (
                        Some(IRCond::Pred {
                            reg: p0.clone(),
                            sense: true,
                        }),
                        0x10,
                    ),
                    (
                        Some(IRCond::Pred {
                            reg: p0,
                            sense: false,
                        }),
                        0x20,
                    ),
                ],
                vec![stmt("BODY")],
            ),
            (2, 0x20, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![
            (0, 1, EdgeKind::FallThrough),
            (1, 1, EdgeKind::CondBranch),
            (1, 2, EdgeKind::FallThrough),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(contains_loop(&out));
        assert!(!contains_goto(&out));
    }

    #[test]
    fn collapse_recovers_multiblock_do_while() {
        // Tail-test loop with a straight-line body block folded into the
        // region before the test fires: BB0 → BB1 → BB2(test) → BB1 (back) / BB3.
        let p0 = RegId::new("P", 0, 1);
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x10)], vec![stmt("PRE")]),
            (
                1,
                0x10,
                vec![(Some(IRCond::True), 0x20)],
                vec![stmt("BODY_A")],
            ),
            (
                2,
                0x20,
                vec![
                    (
                        Some(IRCond::Pred {
                            reg: p0.clone(),
                            sense: true,
                        }),
                        0x10,
                    ),
                    (
                        Some(IRCond::Pred {
                            reg: p0,
                            sense: false,
                        }),
                        0x30,
                    ),
                ],
                vec![stmt("BODY_B")],
            ),
            (3, 0x30, vec![], vec![stmt("RET")]),
        ];
        let edges = vec![
            (0, 1, EdgeKind::FallThrough),
            (1, 2, EdgeKind::FallThrough),
            (2, 1, EdgeKind::CondBranch),
            (2, 3, EdgeKind::FallThrough),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        let out = structurizer.structure_function().unwrap();
        assert!(contains_loop(&out));
        assert!(!contains_goto(&out));
    }

    #[test]
    fn collapse_falls_back_to_goto_on_irreducible() {
        // Irreducible diamond: two entries (0 and 1) into the same body 2,
        // which loops back to 2. No reducible structure — goto fallback.
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x20)], vec![stmt("A")]),
            (1, 0x10, vec![(Some(IRCond::True), 0x20)], vec![stmt("B")]),
            (
                2,
                0x20,
                vec![(Some(IRCond::True), 0x20)],
                vec![stmt("LOOP")],
            ),
        ];
        let edges = vec![
            (0, 2, EdgeKind::UncondBranch),
            (1, 2, EdgeKind::UncondBranch),
            (2, 2, EdgeKind::UncondBranch),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let mut structurizer = Structurizer::new(&cfg, &fir);
        // Should complete without panicking. The graph is irreducible so the
        // output may include gotos or multiple residual regions; we just
        // require totality.
        let _ = structurizer.structure_function();
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
        let specs = vec![(
            0,
            0x00,
            vec![],
            vec![predicated_stmt("EXIT", IRExpr::Reg(RegId::new("P", 0, 1)))],
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
        assert!(rendered.contains("if (P0) return;"));
        assert!(!rendered.contains("EXIT("));
    }

    #[test]
    fn pretty_print_omits_phi_statements_without_summary_comment() {
        let specs = vec![(
            0,
            0x00,
            vec![],
            vec![
                IRStatement {
                    defs: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(1))],
                    value: RValue::Phi(vec![
                        IRExpr::Reg(RegId::new("R", 2, 1).with_ssa(1)),
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
        assert!(!rendered.contains("phi node(s) omitted"));
        assert!(rendered.contains("IADD3("));
    }

    #[test]
    fn pretty_print_inserts_loop_phi_entry_prelude() {
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x10)], vec![stmt("PRE")]),
            (
                1,
                0x10,
                vec![(Some(IRCond::True), 0x10)],
                vec![
                    IRStatement {
                        defs: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(2))],
                        value: RValue::Phi(vec![
                            IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(1)),
                            IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(3)),
                        ]),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                    IRStatement {
                        defs: vec![IRExpr::Reg(RegId::new("R", 2, 1).with_ssa(1))],
                        value: RValue::Op {
                            opcode: "MOV".to_string(),
                            args: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(2))],
                        },
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                ],
            ),
        ];
        let edges = vec![
            (0, 1, EdgeKind::UncondBranch),
            (1, 1, EdgeKind::UncondBranch),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let structurizer = Structurizer::new(&cfg, &fir);
        let body_stmt = StructuredStatement::BasicBlock {
            block_id: 1,
            stmts: fir.blocks[1].stmts.clone(),
        };
        let phi_prelude = structurizer.lower_loop_phi_prelude_ast(1, &body_stmt, &DefaultDisplay);
        let rendered_prelude = phi_prelude.render_with_indent(0);
        assert!(rendered_prelude.contains("R1.2 ="));
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::DoWhile,
            header_block_id: Some(1),
            condition_expr: Some(IRExpr::Reg(RegId::new("P", 0, 1))),
            body: Box::new(body_stmt),
        };
        let rendered =
            structurizer.pretty_print_with_lift_cleanup(&loop_stmt, &DefaultDisplay, 0, None);
        assert!(rendered.contains("R1.2 ="));
        assert!(rendered.contains("MOV(R1.2)"));
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
        assert!(rendered.contains("BODY()"));
        assert!(!rendered.contains("continue;"));
    }
}
