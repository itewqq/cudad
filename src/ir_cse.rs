//! IR-level common subexpression elimination for SSA-form `FunctionIR`.
//!
//! This pass runs on the SSA IR before structurization. It walks the
//! dominator tree in DFS order, maintaining a scoped set of available
//! expressions. When a statement computes a value identical to one
//! already available from a dominating definition, the new def is
//! replaced with a copy of the existing def.
//!
//! Because we operate on SSA form, value identity is straightforward:
//! two expressions are the same if their opcode and SSA-named operands
//! match exactly.

use std::collections::HashMap;

use crate::cfg::ControlFlowGraph;
use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue, RegId};

use petgraph::algo::dominators::simple_fast;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::BTreeMap;

/// A hashable key that represents the value computed by a statement.
/// Two keys are equal iff the statements compute the same value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ExprKey {
    Op { opcode: String, args: Vec<ExprAtom> },
}

/// A hashable representation of an IRExpr leaf or sub-expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ExprAtom {
    Reg {
        class: String,
        idx: i32,
        sign: i32,
        ssa: Option<usize>,
    },
    ImmI(i64),
    /// Float bits for exact comparison (avoids f64 Hash issues).
    ImmF(u64),
    Mem {
        base: Box<ExprAtom>,
        offset: Option<Box<ExprAtom>>,
        width: Option<u32>,
    },
    Op {
        op: String,
        args: Vec<ExprAtom>,
    },
}

fn expr_to_atom(e: &IRExpr) -> ExprAtom {
    match e {
        IRExpr::Reg(r) => ExprAtom::Reg {
            class: r.class.clone(),
            idx: r.idx,
            sign: r.sign,
            ssa: r.ssa,
        },
        IRExpr::ImmI(v) => ExprAtom::ImmI(*v),
        IRExpr::ImmF(v) => ExprAtom::ImmF(v.to_bits()),
        IRExpr::Mem {
            base,
            offset,
            width,
        } => ExprAtom::Mem {
            base: Box::new(expr_to_atom(base)),
            offset: offset.as_ref().map(|o| Box::new(expr_to_atom(o))),
            width: *width,
        },
        IRExpr::Op { op, args } => ExprAtom::Op {
            op: op.clone(),
            args: args.iter().map(expr_to_atom).collect(),
        },
    }
}

fn stmt_to_key(stmt: &IRStatement) -> Option<ExprKey> {
    // Only CSE pure Op statements with exactly one def and no predicate.
    if stmt.defs.len() != 1 || stmt.pred.is_some() {
        return None;
    }
    match &stmt.value {
        RValue::Op { opcode, args } => {
            // Skip side-effectful ops and loads (they may read different memory).
            if stmt.is_side_effectful() || is_memory_load(opcode) {
                return None;
            }
            let mut atoms: Vec<ExprAtom> = args.iter().map(expr_to_atom).collect();
            // Canonicalize commutative operations: sort operands so that
            // IADD3(R0, 0x5, RZ) and IADD3(0x5, R0, RZ) get the same key.
            if is_fully_commutative(opcode) {
                atoms.sort();
            } else if is_partially_commutative(opcode) {
                // IMAD: a*b+c — first two args (multiplicands) are commutative,
                // third (addend) stays in place.
                if atoms.len() >= 2 {
                    if atoms[0] > atoms[1] {
                        atoms.swap(0, 1);
                    }
                }
            }
            Some(ExprKey::Op {
                opcode: opcode.clone(),
                args: atoms,
            })
        }
        _ => None,
    }
}

/// Fully commutative: all operands are interchangeable.
///
/// Note: LOP3 is NOT commutative — permuting operands A,B,C requires
/// adjusting the LUT truth table to get the same result.
/// IADD3 is fully commutative ONLY when not using carry (.X suffix).
fn is_fully_commutative(opcode: &str) -> bool {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    // IADD3.X has carry semantics on the third operand — exclude it.
    if mnem == "IADD3" || mnem == "UIADD3" {
        return !opcode.contains(".X");
    }
    matches!(mnem, "FADD" | "FMUL")
}

/// Partially commutative: first two operands are commutative (multiply).
fn is_partially_commutative(opcode: &str) -> bool {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    matches!(mnem, "IMAD" | "FFMA")
}

fn is_memory_load(opcode: &str) -> bool {
    opcode.starts_with("LD") || opcode.starts_with("S2R") || opcode.starts_with("CS2R")
}

/// Run dominator-based CSE on `fir`. Requires the original CFG for
/// dominator computation.
pub fn ir_cse(fir: &FunctionIR, cfg: &ControlFlowGraph) -> FunctionIR {
    if cfg.node_count() == 0 || fir.blocks.is_empty() {
        return fir.clone();
    }

    // Build dominator tree.
    let entry = find_entry_node(cfg);
    let doms = simple_fast(cfg, entry);

    // Build dom children.
    let mut children: BTreeMap<NodeIndex, Vec<NodeIndex>> = BTreeMap::new();
    for n in cfg.node_indices() {
        if let Some(idom) = doms.immediate_dominator(n) {
            children.entry(idom).or_default().push(n);
        }
    }

    // Map block_id → index in fir.blocks for O(1) lookup.
    let mut bid_to_idx: HashMap<usize, usize> = HashMap::new();
    for (idx, block) in fir.blocks.iter().enumerate() {
        bid_to_idx.insert(block.id, idx);
    }

    // available_exprs: ExprKey → (def RegId) — scoped by dominator depth.
    // We use a Vec of HashMaps as a scope stack.
    let mut scope_stack: Vec<HashMap<ExprKey, RegId>> = Vec::new();
    // The replacement map: (block_idx, stmt_idx) → replacement RegId
    let mut replacements: HashMap<(usize, usize), RegId> = HashMap::new();

    fn walk(
        node: NodeIndex,
        cfg: &ControlFlowGraph,
        fir: &FunctionIR,
        children: &BTreeMap<NodeIndex, Vec<NodeIndex>>,
        bid_to_idx: &HashMap<usize, usize>,
        scope_stack: &mut Vec<HashMap<ExprKey, RegId>>,
        replacements: &mut HashMap<(usize, usize), RegId>,
    ) {
        scope_stack.push(HashMap::new());

        let block_id = cfg[node].id;
        if let Some(&block_idx) = bid_to_idx.get(&block_id) {
            let block = &fir.blocks[block_idx];
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                if let Some(key) = stmt_to_key(stmt) {
                    // Search all scopes for this key.
                    let existing = scope_stack.iter().rev().find_map(|scope| scope.get(&key));
                    if let Some(existing_reg) = existing {
                        // Replace this def with the existing one.
                        replacements.insert((block_idx, stmt_idx), existing_reg.clone());
                    } else {
                        // Record this def as available.
                        if let Some(def_reg) = stmt.defs[0].get_reg() {
                            scope_stack.last_mut().unwrap().insert(key, def_reg.clone());
                        }
                    }
                }
            }
        }

        // Recurse into dominator children.
        if let Some(kids) = children.get(&node) {
            for &child in kids {
                walk(
                    child,
                    cfg,
                    fir,
                    children,
                    bid_to_idx,
                    scope_stack,
                    replacements,
                );
            }
        }

        scope_stack.pop();
    }

    walk(
        entry,
        cfg,
        fir,
        &children,
        &bid_to_idx,
        &mut scope_stack,
        &mut replacements,
    );

    if replacements.is_empty() {
        return fir.clone();
    }

    // Apply replacements: for each replaced statement, rewrite it as a
    // copy (phi-like assignment) from the existing def.
    let mut new_blocks = Vec::with_capacity(fir.blocks.len());
    for (block_idx, block) in fir.blocks.iter().enumerate() {
        let mut new_stmts = Vec::with_capacity(block.stmts.len());
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            if let Some(replacement_reg) = replacements.get(&(block_idx, stmt_idx)) {
                // Rewrite: Rdst = Rsrc (copy from CSE'd register)
                new_stmts.push(IRStatement {
                    defs: stmt.defs.clone(),
                    value: RValue::Op {
                        opcode: "IMAD.MOV.U32".to_string(),
                        args: vec![
                            IRExpr::Reg(RegId::new("RZ", 0, 1)),
                            IRExpr::Reg(RegId::new("RZ", 0, 1)),
                            IRExpr::Reg(replacement_reg.clone()),
                        ],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: vec![],
                });
            } else {
                new_stmts.push(stmt.clone());
            }
        }
        new_blocks.push(IRBlock {
            id: block.id,
            start_addr: block.start_addr,
            irdst: block.irdst.clone(),
            stmts: new_stmts,
        });
    }

    FunctionIR { blocks: new_blocks }
}

fn find_entry_node(cfg: &ControlFlowGraph) -> NodeIndex {
    let mut candidates: Vec<_> = cfg
        .node_indices()
        .filter(|&n| cfg.neighbors_directed(n, Direction::Incoming).count() == 0)
        .collect();
    if candidates.is_empty() {
        return cfg.node_indices().next().unwrap_or(NodeIndex::new(0));
    }
    candidates.sort_by_key(|&n| cfg[n].id);
    candidates[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, decode_sass};

    #[test]
    fn cse_eliminates_duplicate_pure_computation() {
        // Two identical IADD3 R?, R0, 0x1, RZ in the same block.
        // After SSA each gets a unique def, but the computation is identical.
        let sass = r#"
            /*0000*/ IADD3 R1, R0, 0x1, RZ ;
            /*0010*/ IADD3 R2, R0, 0x1, RZ ;
            /*0020*/ STG.E [R1.64], R2 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let cse_fir = ir_cse(&fir, &cfg);

        // The second IADD3 should be rewritten to a copy (IMAD.MOV.U32)
        let block = &cse_fir.blocks[0];
        let copy_count = block.stmts.iter()
            .filter(|s| matches!(&s.value, RValue::Op { opcode, args } if opcode == "IMAD.MOV.U32" && args.len() == 3))
            .count();
        // At least one IMAD.MOV.U32 copy should exist from CSE
        assert!(
            copy_count >= 1,
            "CSE should create at least one copy, found {}",
            copy_count
        );
    }

    #[test]
    fn cse_does_not_eliminate_loads() {
        // Two identical LDG loads should NOT be CSE'd (memory may change).
        let sass = r#"
            /*0000*/ LDG.E R1, [R0.64] ;
            /*0010*/ LDG.E R2, [R0.64] ;
            /*0020*/ IADD3 R3, R1, R2, RZ ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let cse_fir = ir_cse(&fir, &cfg);

        // Both loads should survive
        let load_count = cse_fir
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .filter(|s| matches!(&s.value, RValue::Op { opcode, .. } if opcode.starts_with("LDG")))
            .count();
        assert_eq!(load_count, 2, "loads should not be CSE'd");
    }

    #[test]
    fn cse_canonicalizes_commutative_operands() {
        // IADD3 R1, R0, 0x5, RZ  and  IADD3 R2, 0x5, R0, RZ
        // are the same computation (IADD3 is commutative).
        // CSE should recognize them as identical after canonicalization.
        let sass = r#"
            /*0000*/ IADD3 R1, R0, 0x5, RZ ;
            /*0010*/ IADD3 R2, 0x5, R0, RZ ;
            /*0020*/ IADD3 R3, R1, R2, RZ ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let cse_fir = ir_cse(&fir, &cfg);

        // After CSE, R2's computation should be replaced with a copy of R1
        let copy_count = cse_fir
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .filter(|s| {
                matches!(&s.value, RValue::Op { opcode, .. }
                    if opcode == "IMAD.MOV.U32")
            })
            .count();
        assert!(
            copy_count >= 1,
            "CSE should eliminate one commutative duplicate, found {} copies",
            copy_count
        );
    }
}
