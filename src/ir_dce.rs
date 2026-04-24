//! IR-level dead code elimination for SSA-form `FunctionIR`.
//!
//! This pass operates on the SSA IR *before* structurization and
//! rendering, using def-use chains to identify and remove statements
//! whose results are never consumed by any side-effectful operation.
//!
//! Algorithm (reverse-reachability worklist):
//!   1. Identify all *essential* statements — those with observable side
//!      effects (memory stores, atomics, barriers, control flow).
//!   2. Mark them as live.
//!   3. For each live statement, mark the definitions of all its
//!      operands as live (walk backwards through the def-use chain).
//!   4. Repeat until the worklist is empty.
//!   5. Remove all non-live, non-phi statements.
//!
//! Phi nodes are kept unconditionally because removing them would require
//! rebuilding SSA; later structural cleanup handles redundant rendered forms.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ir::{FunctionIR, RegId};

/// A unique identifier for a statement within a `FunctionIR`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StmtId {
    block_idx: usize,
    stmt_idx: usize,
}

/// Run IR-level dead code elimination on `fir`, returning a new
/// `FunctionIR` with dead statements removed.
pub fn ir_dce(fir: &FunctionIR) -> FunctionIR {
    // ---- Step 1: Build def map (SSA reg → defining statement) ----
    let mut def_map: HashMap<RegId, StmtId> = HashMap::new();

    for (block_idx, block) in fir.blocks.iter().enumerate() {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let sid = StmtId {
                block_idx,
                stmt_idx,
            };
            for def_expr in &stmt.defs {
                if let Some(r) = def_expr.get_reg() {
                    if !is_immutable_reg(r) {
                        def_map.insert(normalize_reg_key(r), sid);
                    }
                }
            }
        }
    }

    // ---- Step 2: Seed worklist with essential statements ----
    let mut live: HashSet<StmtId> = HashSet::new();
    let mut worklist: VecDeque<StmtId> = VecDeque::new();

    for (block_idx, block) in fir.blocks.iter().enumerate() {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            if stmt.is_side_effectful() {
                let sid = StmtId {
                    block_idx,
                    stmt_idx,
                };
                if live.insert(sid) {
                    worklist.push_back(sid);
                }
            }
        }
    }

    // ---- Step 3: Backward walk — mark operand defs as live ----
    while let Some(sid) = worklist.pop_front() {
        let stmt = &fir.blocks[sid.block_idx].stmts[sid.stmt_idx];
        let uses = stmt.collect_all_uses();
        for used_reg in uses {
            if is_immutable_reg(&used_reg) {
                continue;
            }
            if let Some(&def_sid) = def_map.get(&normalize_reg_key(&used_reg)) {
                if live.insert(def_sid) {
                    worklist.push_back(def_sid);
                }
            }
        }
    }

    // ---- Step 4: Rebuild FunctionIR keeping only live + phi stmts ----
    let mut new_blocks = Vec::with_capacity(fir.blocks.len());
    for (block_idx, block) in fir.blocks.iter().enumerate() {
        let mut new_stmts = Vec::new();
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let sid = StmtId {
                block_idx,
                stmt_idx,
            };
            let is_phi = matches!(stmt.value, crate::ir::RValue::Phi(_));
            if live.contains(&sid) || is_phi {
                new_stmts.push(stmt.clone());
            }
        }
        new_blocks.push(crate::ir::IRBlock {
            id: block.id,
            start_addr: block.start_addr,
            irdst: block.irdst.clone(),
            stmts: new_stmts,
        });
    }

    FunctionIR { blocks: new_blocks }
}

fn is_immutable_reg(r: &RegId) -> bool {
    matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
}

fn normalize_reg_key(r: &RegId) -> RegId {
    let mut normalized = r.clone();
    normalized.sign = 1;
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, decode_sass, IRBlock, IRExpr, IRStatement, RValue};

    #[test]
    fn ir_dce_removes_dead_assignment() {
        // R1 = 1 (dead); R2 = 2 (dead); EXIT (live)
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x1 ;
            /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, 0x2 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let orig_stmts: usize = fir.blocks.iter().map(|b| b.stmts.len()).sum();

        let cleaned = ir_dce(&fir);
        let new_stmts: usize = cleaned.blocks.iter().map(|b| b.stmts.len()).sum();

        // The two IMAD.MOV.U32 should be removed (dead), EXIT is side-effectful but
        // it's handled as control flow — in our model EXIT is not in stmts but
        // the IMAD.MOV.U32 instructions have no users.
        assert!(
            new_stmts < orig_stmts,
            "DCE should remove dead stmts: orig={}, new={}",
            orig_stmts,
            new_stmts
        );
    }

    #[test]
    fn ir_dce_preserves_store_chain() {
        // R1 = addr ; STG [R1] = R2  →  R1 is used by the store, must be kept
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ STG.E [R4.64], R8 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let cleaned = ir_dce(&fir);

        // Store and its address computation must survive
        let total_stmts: usize = cleaned.blocks.iter().map(|b| b.stmts.len()).sum();
        assert!(
            total_stmts >= 2,
            "store chain must survive DCE, got {} stmts",
            total_stmts
        );
    }

    #[test]
    fn ir_dce_preserves_phi_nodes() {
        let sass = r#"
            /*000*/ IADD R0, R0, 0x1 ;
            /*010*/ ISETP.LT.AND P0, PT, R0, 0x5, PT ;
            /*020*/ @P0 BRA 0x000 ;
            /*030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let cleaned = ir_dce(&fir);

        let phi_count: usize = cleaned
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .filter(|s| matches!(s.value, crate::ir::RValue::Phi(_)))
            .count();
        assert!(phi_count > 0, "phi nodes must survive IR-DCE");
    }

    #[test]
    fn ir_dce_keeps_defs_used_with_negative_sign() {
        let def = RegId::new("R", 7, 1).with_ssa(1);
        let pred = RegId::new("P", 0, 1).with_ssa(1);
        let fir = FunctionIR {
            blocks: vec![IRBlock {
                id: 0,
                start_addr: 0,
                irdst: vec![],
                stmts: vec![
                    IRStatement {
                        defs: vec![IRExpr::Reg(def.clone())],
                        value: RValue::Op {
                            opcode: "I2FP.F32.S32".to_string(),
                            args: vec![IRExpr::ImmI(42)],
                        },
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                    IRStatement {
                        defs: vec![IRExpr::Reg(pred.clone())],
                        value: RValue::Op {
                            opcode: "FSETP.GT.AND".to_string(),
                            args: vec![
                                IRExpr::Reg(RegId::new("R", 7, -1).with_ssa(1)),
                                IRExpr::ImmF(0.0),
                                IRExpr::Reg(RegId::new("PT", 0, 1)),
                            ],
                        },
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                    IRStatement {
                        defs: vec![],
                        value: RValue::Op {
                            opcode: "EXIT".to_string(),
                            args: vec![],
                        },
                        pred: Some(IRExpr::Reg(pred)),
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                ],
            }],
        };

        let cleaned = ir_dce(&fir);
        let total_stmts: usize = cleaned.blocks.iter().map(|b| b.stmts.len()).sum();
        assert_eq!(total_stmts, 3, "negative-signed use must keep its defining stmt");
    }

    #[test]
    fn ir_dce_keeps_uniform_defs_used_with_negative_sign() {
        let def = RegId::new("UR", 5, 1).with_ssa(0);
        let pred = RegId::new("P", 0, 1).with_ssa(1);
        let fir = FunctionIR {
            blocks: vec![IRBlock {
                id: 0,
                start_addr: 0,
                irdst: vec![],
                stmts: vec![
                    IRStatement {
                        defs: vec![IRExpr::Reg(def.clone())],
                        value: RValue::Op {
                            opcode: "UMOV".to_string(),
                            args: vec![IRExpr::ImmI(1)],
                        },
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                    IRStatement {
                        defs: vec![IRExpr::Reg(pred.clone())],
                        value: RValue::Op {
                            opcode: "UISETP.GE.U32.AND".to_string(),
                            args: vec![
                                IRExpr::Reg(RegId::new("UR", 5, -1).with_ssa(0)),
                                IRExpr::ImmI(0),
                                IRExpr::Reg(RegId::new("PT", 0, 1)),
                            ],
                        },
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                    IRStatement {
                        defs: vec![],
                        value: RValue::Op {
                            opcode: "EXIT".to_string(),
                            args: vec![],
                        },
                        pred: Some(IRExpr::Reg(pred)),
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                ],
            }],
        };

        let cleaned = ir_dce(&fir);
        let total_stmts: usize = cleaned.blocks.iter().map(|b| b.stmts.len()).sum();
        assert_eq!(total_stmts, 3, "negative-signed uniform use must keep its defining stmt");
    }
}
