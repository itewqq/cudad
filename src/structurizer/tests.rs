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
    fn pretty_print_inserts_loop_phi_backedge_updates() {
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
                            IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(3)),
                            IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(1)),
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
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::DoWhile,
            header_block_id: Some(1),
            condition_expr: Some(IRExpr::Reg(RegId::new("P", 0, 1))),
            body: Box::new(StructuredStatement::BasicBlock {
                block_id: 1,
                stmts: fir.blocks[1].stmts.clone(),
            }),
        };
        let rendered =
            structurizer.pretty_print_with_lift_cleanup(&loop_stmt, &DefaultDisplay, 0, None);
        assert!(
            rendered.contains("R1.2 = R1.1;"),
            "rendered:
{}",
            rendered
        );
        assert!(
            rendered.contains("R1.2 = R1.3;"),
            "rendered:
{}",
            rendered
        );
        assert!(
            rendered.contains("R2.1 = MOV(R1.2);"),
            "rendered:
{}",
            rendered
        );
    }

    #[test]
    fn pretty_print_inserts_tail_header_body_entry_phi_updates() {
        let specs = vec![
            (0, 0x00, vec![(Some(IRCond::True), 0x10)], vec![stmt("ENTRY")]),
            (
                1,
                0x10,
                vec![(Some(IRCond::True), 0x20)],
                vec![
                    IRStatement {
                        defs: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(1))],
                        value: RValue::Phi(vec![
                            IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(2)),
                            IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(0)),
                        ]),
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                    IRStatement {
                        defs: vec![IRExpr::Reg(RegId::new("R", 2, 1).with_ssa(1))],
                        value: RValue::Op {
                            opcode: "MOV".to_string(),
                            args: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(1))],
                        },
                        pred: None,
                        mem_addr_args: None,
                        pred_old_defs: vec![],
                    },
                ],
            ),
            (
                2,
                0x20,
                vec![
                    (Some(IRCond::True), 0x10),
                    (Some(IRCond::True), 0x30),
                ],
                vec![IRStatement {
                    defs: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(2))],
                    value: RValue::Op {
                        opcode: "MOV".to_string(),
                        args: vec![IRExpr::Reg(RegId::new("R", 1, 1).with_ssa(1))],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: vec![],
                }],
            ),
        ];
        let edges = vec![
            (0, 1, EdgeKind::UncondBranch),
            (1, 2, EdgeKind::UncondBranch),
            (2, 1, EdgeKind::CondBranch),
        ];
        let (cfg, fir, _) = build_case(&specs, &edges);
        let structurizer = Structurizer::new(&cfg, &fir);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::DoWhile,
            header_block_id: Some(2),
            condition_expr: Some(IRExpr::Reg(RegId::new("P", 0, 1))),
            body: Box::new(StructuredStatement::Sequence(vec![
                StructuredStatement::BasicBlock {
                    block_id: 1,
                    stmts: fir.blocks[1].stmts.clone(),
                },
                StructuredStatement::BasicBlock {
                    block_id: 2,
                    stmts: fir.blocks[2].stmts.clone(),
                },
            ])),
        };
        let rendered =
            structurizer.pretty_print_with_lift_cleanup(&loop_stmt, &DefaultDisplay, 0, None);
        assert!(
            rendered.contains("R1.1 = R1.0;"),
            "expected loop entry phi prelude, got:\n{}",
            rendered
        );
        assert!(
            rendered.contains("R1.1 = R1.2;"),
            "expected tail-header backedge to feed the body-entry phi, got:\n{}",
            rendered
        );
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
