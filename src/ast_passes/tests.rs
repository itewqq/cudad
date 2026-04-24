use super::*;
use crate::ast::{LoopKind, PointerLane};

#[test]
fn simplify_drops_self_assignment() {
    let stmt = Stmt::Assign {
        dst: LValue::Var("v0".to_string()),
        src: Expr::Reg("v0".to_string()),
    };
    assert_eq!(ast_simplify(stmt), Stmt::Empty);
}

#[test]
fn simplify_inverts_empty_then_if() {
    let stmt = Stmt::If {
        condition: Expr::Reg("b0".to_string()),
        then_branch: Box::new(Stmt::Empty),
        else_branch: Some(Box::new(Stmt::Return(None))),
    };
    assert_eq!(
        ast_simplify(stmt),
        Stmt::If {
            condition: Expr::Unary {
                op: "!".to_string(),
                arg: Box::new(Expr::Reg("b0".to_string())),
            },
            then_branch: Box::new(Stmt::Return(None)),
            else_branch: None,
        }
    );
}

#[test]
fn recover_plain_symbolic_pointer_index_strips_u32_offset_casts() {
    let addr = Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(Expr::Raw("arg0_ptr".to_string())),
        rhs: Box::new(Expr::Cast {
            ty: "uint32_t".to_string(),
            expr: Box::new(Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("idx".to_string())),
                rhs: Box::new(Expr::Imm("1".to_string())),
            }),
        }),
    };
    let recovered = recover_plain_symbolic_pointer_index_from_wide_addr(&addr)
        .expect("u32 index casts should not block recovered pointer indexing");
    assert_eq!(recovered.render(), "arg0_ptr + (idx + 1)");
}

#[test]
fn cleanup_does_not_redivide_existing_symbolic_element_indices() {
    let stmt = Stmt::Assign {
        dst: LValue::Deref {
            ty: Some("uint32_t".to_string()),
            addr: Box::new(Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Raw("arg2_ptr".to_string())),
                rhs: Box::new(Expr::Imm("4".to_string())),
            }),
        },
        src: Expr::Reg("v0".to_string()),
    };

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("arg2_ptr + 4"),
        "cleanup should treat existing symbolic indices as elements, got:\n{}",
        rendered
    );
    assert!(
        !rendered.contains("arg2_ptr + 1"),
        "cleanup should not re-divide existing symbolic indices, got:\n{}",
        rendered
    );
}

#[test]
fn seeded_wide_addr_fold_keeps_full_width_delta_when_hi_moves_off_base_seed() {
    let mut seeded = SeededWideAddrMaps::default();
    let info = SeededWideAddrInfo {
        base: Expr::Raw("arg2_ptr".to_string()),
        seed_lo: Expr::PtrLane {
            base: "arg2_ptr".to_string(),
            lane: PointerLane::Lo32,
        },
    };
    seeded.lo_by_name.insert("ptr_lo".to_string(), info.clone());
    seeded.hi_by_name.insert("ptr_hi".to_string(), info);

    let folded = fold_seeded_wide_addr_use(
        &Expr::Reg("ptr_lo".to_string()),
        &Expr::Reg("ptr_hi".to_string()),
        &seeded,
    )
    .expect("seeded pair should fold");

    let rendered = folded.render();
    assert!(
        rendered.contains("(uint64_t)((uint32_t)(ptr_hi)) << 32")
            && rendered.contains("(uintptr_t)(arg2_ptr)")
            && !rendered.contains("ptr_lo - arg2_ptr.lo32"),
        "expected seeded fold to preserve the high lane when the pair moves off the base seed, got:\n{}",
        rendered
    );
}

#[test]
fn simplify_flattens_nested_sequences() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Empty,
        Stmt::Sequence(vec![Stmt::ExprStmt(Expr::Reg("a".to_string()))]),
        Stmt::ExprStmt(Expr::Reg("b".to_string())),
    ]);
    assert_eq!(
        ast_simplify(stmt),
        Stmt::Sequence(vec![
            Stmt::ExprStmt(Expr::Reg("a".to_string())),
            Stmt::ExprStmt(Expr::Reg("b".to_string())),
        ])
    );
}

#[test]
fn simplify_preserves_loop_structure() {
    let stmt = Stmt::Loop {
        kind: LoopKind::While,
        condition: Some(Expr::Reg("b0".to_string())),
        body: Box::new(Stmt::Sequence(vec![Stmt::Empty, Stmt::Continue])),
    };
    assert_eq!(
        ast_simplify(stmt),
        Stmt::Loop {
            kind: LoopKind::While,
            condition: Some(Expr::Reg("b0".to_string())),
            body: Box::new(Stmt::Continue),
        }
    );
}

#[test]
fn dce_removes_dead_pure_assignment() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v0".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("v1".to_string())),
                rhs: Box::new(Expr::Imm("1".to_string())),
            },
        },
        Stmt::Return(None),
    ]);
    assert_eq!(ast_cleanup(stmt), Stmt::Return(None));
}

#[test]
fn dce_removes_dead_pure_helper_call_assignment() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v0".to_string()),
            src: Expr::CallLike {
                func: "abs".to_string(),
                args: vec![Expr::Reg("v1".to_string())],
            },
        },
        Stmt::Return(None),
    ]);
    assert_eq!(ast_cleanup(stmt), Stmt::Return(None));
}

#[test]
fn dce_removes_dead_carry_helper_assignment() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("b0".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Reg("v1".to_string()),
                    Expr::Reg("v2".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Return(None),
    ]);
    assert_eq!(ast_cleanup(stmt), Stmt::Return(None));
}

#[test]
fn collapse_typed_addr64_pair_recovers_widened_scalar_sum() {
    let mut defs = std::collections::HashMap::new();
    defs.insert(
        "sign".to_string(),
        Expr::Binary {
            op: ">>".to_string(),
            lhs: Box::new(Expr::Cast {
                ty: "int32_t".to_string(),
                expr: Box::new(Expr::Reg("base".to_string())),
            }),
            rhs: Box::new(Expr::Imm("31".to_string())),
        },
    );
    defs.insert(
        "lo".to_string(),
        Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(Expr::Reg("base".to_string())),
            rhs: Box::new(Expr::Reg("idx".to_string())),
        },
    );
    defs.insert(
        "hi".to_string(),
        Expr::Cast {
            ty: "uint32_t".to_string(),
            expr: Box::new(Expr::Binary {
                op: ">>".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Cast {
                        ty: "uintptr_t".to_string(),
                        expr: Box::new(Expr::Binary {
                            op: "|".to_string(),
                            lhs: Box::new(Expr::Cast {
                                ty: "uint64_t".to_string(),
                                expr: Box::new(Expr::Cast {
                                    ty: "uint32_t".to_string(),
                                    expr: Box::new(Expr::Reg("idx".to_string())),
                                }),
                            }),
                            rhs: Box::new(Expr::Binary {
                                op: "<<".to_string(),
                                lhs: Box::new(Expr::Cast {
                                    ty: "uint64_t".to_string(),
                                    expr: Box::new(Expr::Cast {
                                        ty: "uint32_t".to_string(),
                                        expr: Box::new(Expr::Reg("sign".to_string())),
                                    }),
                                }),
                                rhs: Box::new(Expr::Imm("32".to_string())),
                            }),
                        }),
                    }),
                    rhs: Box::new(Expr::Cast {
                        ty: "int64_t".to_string(),
                        expr: Box::new(Expr::Cast {
                            ty: "int32_t".to_string(),
                            expr: Box::new(Expr::Reg("base".to_string())),
                        }),
                    }),
                }),
                rhs: Box::new(Expr::Imm("32".to_string())),
            }),
        },
    );

    let collapsed = collapse_typed_addr64_pair(
        &Expr::Reg("lo".to_string()),
        &Expr::Reg("hi".to_string()),
        &defs,
    )
    .expect("split widened scalar pair should collapse");
    let expected = Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(widen_i32_expr_ast(Expr::Reg("base".to_string()))),
        rhs: Box::new(widen_u32_expr_ast(Expr::Reg("idx".to_string()))),
    };
    assert!(
        same_match_expr(&collapsed, &expected),
        "expected widened scalar collapse, got: {}",
        collapsed.render()
    );
}

#[test]
fn fold_resolved_typed_scaled_hi_expr_recovers_base_relative_hi_lane() {
    let defs = std::collections::HashMap::new();
    let hi_lane = Expr::Cast {
        ty: "uint32_t".to_string(),
        expr: Box::new(Expr::Binary {
            op: ">>".to_string(),
            lhs: Box::new(Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Cast {
                    ty: "uint8_t*".to_string(),
                    expr: Box::new(Expr::Raw("arg2_ptr".to_string())),
                }),
                rhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("base_idx".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                }),
            }),
            rhs: Box::new(Expr::Imm("32".to_string())),
        }),
    };
    let carry = Expr::Ternary {
        cond: Box::new(Expr::Intrinsic {
            op: IntrinsicOp::CarryU32Add3,
            args: vec![
                Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("idx".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                },
                Expr::Cast {
                    ty: "uint32_t".to_string(),
                    expr: Box::new(Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Cast {
                            ty: "uint8_t*".to_string(),
                            expr: Box::new(Expr::Raw("arg2_ptr".to_string())),
                        }),
                        rhs: Box::new(Expr::Binary {
                            op: "*".to_string(),
                            lhs: Box::new(Expr::Reg("base_idx".to_string())),
                            rhs: Box::new(Expr::Imm("4".to_string())),
                        }),
                    }),
                },
                Expr::Imm("0".to_string()),
            ],
        }),
        then_expr: Box::new(Expr::Imm("1".to_string())),
        else_expr: Box::new(Expr::Imm("0".to_string())),
    };
    let mul_hi = Expr::CallLike {
        func: "mul_hi_u32".to_string(),
        args: vec![Expr::Reg("idx".to_string()), Expr::Imm("4".to_string())],
    };
    let expr = Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(mul_hi.clone()),
        rhs: Box::new(Expr::Binary {
            op: "+".to_string(),
            lhs: Box::new(hi_lane.clone()),
            rhs: Box::new(carry.clone()),
        }),
    };

    assert!(
        resolve_typed_lane_source(&hi_lane, &defs).is_some(),
        "hi lane should resolve"
    );
    let carry_lane = if let Expr::Ternary { cond, .. } = &carry {
        if let Expr::Intrinsic { args, .. } = cond.as_ref() {
            args[1].clone()
        } else {
            Expr::Imm("0".to_string())
        }
    } else {
        Expr::Imm("0".to_string())
    };
    assert!(
        resolve_typed_lane_source(&carry_lane, &defs).is_some(),
        "carry low lane should resolve"
    );
    assert!(
        match_typed_carry_increment(&carry, &defs).is_some(),
        "carry increment should resolve"
    );
    assert!(
        expr_matches_step_hi_term(
            &mul_hi,
            &Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(Expr::Reg("idx".to_string())),
                rhs: Box::new(Expr::Imm("4".to_string())),
            }
        ),
        "mul_hi term should match the scaled low step"
    );

    let folded = fold_resolved_typed_scaled_hi_expr(&expr, &defs)
        .expect("scaled hi expression should collapse");
    let expected = Expr::LaneExtract {
        value: Box::new(Expr::WidePtr {
            base: Box::new(Expr::Raw("arg2_ptr".to_string())),
            offset: Box::new(Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("base_idx".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                }),
                rhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("idx".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                }),
            }),
        }),
        lane: PointerLane::Hi32,
    };
    assert!(
        same_match_expr(&folded, &expected),
        "expected scaled hi collapse, got: {}",
        folded.render()
    );
}

#[test]
fn dce_preserves_live_assignment_flow() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v0".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("v1".to_string())),
                rhs: Box::new(Expr::Imm("1".to_string())),
            },
        },
        Stmt::Return(Some(Expr::Reg("v0".to_string()))),
    ]);
    assert_eq!(ast_cleanup(stmt.clone()), stmt);
}

#[test]
fn dce_keeps_branch_defs_live_out_of_if() {
    let stmt = Stmt::Sequence(vec![
        Stmt::If {
            condition: Expr::Reg("b0".to_string()),
            then_branch: Box::new(Stmt::Assign {
                dst: LValue::Var("v0".to_string()),
                src: Expr::Reg("v1".to_string()),
            }),
            else_branch: None,
        },
        Stmt::Return(Some(Expr::Reg("v0".to_string()))),
    ]);
    assert_eq!(ast_cleanup(stmt.clone()), stmt);
}

#[test]
fn dce_preserves_memory_store_even_when_result_unused() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Deref {
                ty: Some("uint32_t".to_string()),
                addr: Box::new(Expr::Reg("ptr".to_string())),
            },
            src: Expr::Reg("v0".to_string()),
        },
        Stmt::Return(None),
    ]);
    assert_eq!(ast_cleanup(stmt.clone()), stmt);
}

#[test]
fn dce_keeps_unknown_calllike_assignment() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v0".to_string()),
            src: Expr::CallLike {
                func: "mystery".to_string(),
                args: vec![Expr::Reg("v1".to_string())],
            },
        },
        Stmt::Return(None),
    ]);
    assert_eq!(ast_cleanup(stmt.clone()), stmt);
}

#[test]
fn predicate_cleanup_removes_duplicate_return_guard() {
    let guard = Stmt::If {
        condition: Expr::Reg("b0".to_string()),
        then_branch: Box::new(Stmt::Return(None)),
        else_branch: None,
    };
    let stmt = Stmt::Sequence(vec![
        guard.clone(),
        Stmt::ExprStmt(Expr::CallLike {
            func: "touch".to_string(),
            args: vec![Expr::Reg("v0".to_string())],
        }),
        guard,
    ]);
    assert_eq!(
        ast_cleanup(stmt),
        Stmt::Sequence(vec![
            Stmt::If {
                condition: Expr::Reg("b0".to_string()),
                then_branch: Box::new(Stmt::Return(None)),
                else_branch: None,
            },
            Stmt::ExprStmt(Expr::CallLike {
                func: "touch".to_string(),
                args: vec![Expr::Reg("v0".to_string())],
            }),
        ])
    );
}

#[test]
fn predicate_cleanup_keeps_guard_after_nested_reassignment() {
    let guard = Stmt::If {
        condition: Expr::Reg("b0".to_string()),
        then_branch: Box::new(Stmt::Return(None)),
        else_branch: None,
    };
    let stmt = Stmt::Sequence(vec![
        guard.clone(),
        Stmt::If {
            condition: Expr::Reg("v1".to_string()),
            then_branch: Box::new(Stmt::Assign {
                dst: LValue::Var("b0".to_string()),
                src: Expr::Reg("v2".to_string()),
            }),
            else_branch: None,
        },
        guard.clone(),
    ]);
    assert_eq!(ast_cleanup(stmt.clone()), stmt);
}

#[test]
fn addr64_fold_resolves_copy_aliased_ptr_lanes() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("UR8.1".to_string()),
            src: Expr::ConstMemSymbol("arg0_ptr.lo32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("UR9.1".to_string()),
            src: Expr::ConstMemSymbol("arg0_ptr.hi32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("P0.5".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Binary {
                        op: "*".to_string(),
                        lhs: Box::new(Expr::Reg("R4.2".to_string())),
                        rhs: Box::new(Expr::Imm("4".to_string())),
                    },
                    Expr::Reg("UR8.1".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R18.2".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("R4.2".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                }),
                rhs: Box::new(Expr::Reg("UR8.1".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R19.2".to_string()),
            src: Expr::CallLike {
                func: "lea_hi_x".to_string(),
                args: vec![
                    Expr::Reg("R4.2".to_string()),
                    Expr::Reg("UR9.1".to_string()),
                    Expr::Imm("2".to_string()),
                    Expr::Reg("P0.5".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R11.2".to_string()),
            src: Expr::Load {
                ty: None,
                addr: Box::new(Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("R18.2".to_string())),
                        hi: Box::new(Expr::Reg("R19.2".to_string())),
                    }),
                    rhs: Box::new(Expr::Imm("0".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("arg0_ptr + (int64_t)R4.2 * 4"),
        "got:\n{}",
        rendered
    );
    assert!(!rendered.contains("UR8.1"), "got:\n{}", rendered);
    assert!(!rendered.contains("UR9.1"), "got:\n{}", rendered);
}

#[test]
fn addr64_fold_collects_defs_from_wrapping_blocks() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Block(vec![
            Stmt::Assign {
                dst: LValue::Var("R6.0".to_string()),
                src: Expr::ConstMemSymbol("arg2_ptr.lo32".to_string()),
            },
            Stmt::Assign {
                dst: LValue::Var("R7.0".to_string()),
                src: Expr::ConstMemSymbol("arg2_ptr.hi32".to_string()),
            },
            Stmt::Assign {
                dst: LValue::Var("R6.1".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Binary {
                        op: "*".to_string(),
                        lhs: Box::new(Expr::Reg("R4.0".to_string())),
                        rhs: Box::new(Expr::Imm("4".to_string())),
                    }),
                    rhs: Box::new(Expr::Reg("R6.0".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("R7.1".to_string()),
                src: Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::CallLike {
                            func: "mul_hi_u32".to_string(),
                            args: vec![
                                Expr::Reg("R4.0".to_string()),
                                Expr::Imm("4".to_string()),
                            ],
                        }),
                        rhs: Box::new(Expr::Reg("R7.0".to_string())),
                    }),
                    rhs: Box::new(Expr::Ternary {
                        cond: Box::new(Expr::CallLike {
                            func: "carry_u32_add3".to_string(),
                            args: vec![
                                Expr::Binary {
                                    op: "*".to_string(),
                                    lhs: Box::new(Expr::Reg("R4.0".to_string())),
                                    rhs: Box::new(Expr::Imm("4".to_string())),
                                },
                                Expr::Reg("R6.0".to_string()),
                                Expr::Imm("0".to_string()),
                            ],
                        }),
                        then_expr: Box::new(Expr::Imm("1".to_string())),
                        else_expr: Box::new(Expr::Imm("0".to_string())),
                    }),
                },
            },
        ]),
        Stmt::Loop {
            kind: LoopKind::DoWhile,
            condition: Some(Expr::Reg("P1.0".to_string())),
            body: Box::new(Stmt::Block(vec![Stmt::Assign {
                dst: LValue::Deref {
                    ty: None,
                    addr: Box::new(Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Addr64 {
                            lo: Box::new(Expr::Reg("R6.1".to_string())),
                            hi: Box::new(Expr::Reg("R7.1".to_string())),
                        }),
                        rhs: Box::new(Expr::Raw("(int64_t)R8.3 * 4".to_string())),
                    }),
                },
                src: Expr::Reg("R21.2".to_string()),
            }])),
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(rendered.contains("*(arg2_ptr + (int64_t)R4.0 * 4 + (int64_t)R8.3 * 4) = R21.2;"));
    assert!(!rendered.contains("((uintptr_t)(((uint64_t)(R7.1) << 32) | (uint32_t)(R6.1)))"));
}

#[test]
fn addr64_fold_propagates_outer_ptr_lane_aliases_into_nested_loops() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("UR8.1".to_string()),
            src: Expr::ConstMemSymbol("arg0_ptr.lo32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("UR9.1".to_string()),
            src: Expr::ConstMemSymbol("arg0_ptr.hi32".to_string()),
        },
        Stmt::Loop {
            kind: LoopKind::While,
            condition: Some(Expr::Reg("p0".to_string())),
            body: Box::new(Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("P0.5".to_string()),
                    src: Expr::CallLike {
                        func: "carry_u32_add3".to_string(),
                        args: vec![
                            Expr::Binary {
                                op: "*".to_string(),
                                lhs: Box::new(Expr::Reg("R4.2".to_string())),
                                rhs: Box::new(Expr::Imm("4".to_string())),
                            },
                            Expr::Reg("UR8.1".to_string()),
                            Expr::Imm("0".to_string()),
                        ],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("R18.2".to_string()),
                    src: Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Binary {
                            op: "*".to_string(),
                            lhs: Box::new(Expr::Reg("R4.2".to_string())),
                            rhs: Box::new(Expr::Imm("4".to_string())),
                        }),
                        rhs: Box::new(Expr::Reg("UR8.1".to_string())),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("R19.2".to_string()),
                    src: Expr::CallLike {
                        func: "lea_hi_x".to_string(),
                        args: vec![
                            Expr::Reg("R4.2".to_string()),
                            Expr::Reg("UR9.1".to_string()),
                            Expr::Imm("2".to_string()),
                            Expr::Reg("P0.5".to_string()),
                        ],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("R11.2".to_string()),
                    src: Expr::Load {
                        ty: None,
                        addr: Box::new(Expr::Addr64 {
                            lo: Box::new(Expr::Reg("R18.2".to_string())),
                            hi: Box::new(Expr::Reg("R19.2".to_string())),
                        }),
                    },
                },
            ])),
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("R11.2 = *(arg0_ptr + (int64_t)R4.2 * 4);"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("UR8.1"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("UR9.1"),
        "got:
{}",
        rendered
    );
}

#[test]
fn addr64_fold_rewrites_helper_chain_structurally() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v5".to_string()),
            src: Expr::Binary {
                op: "&".to_string(),
                lhs: Box::new(Expr::Reg("v4".to_string())),
                rhs: Box::new(Expr::Imm("255".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("b0".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Reg("v5".to_string()),
                    Expr::ConstMemSymbol("arg0_ptr.lo32".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v6".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("v5".to_string())),
                rhs: Box::new(Expr::ConstMemSymbol("arg0_ptr.lo32".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v7".to_string()),
            src: Expr::CallLike {
                func: "lea_hi_x_sx32".to_string(),
                args: vec![
                    Expr::Reg("v5".to_string()),
                    Expr::ConstMemSymbol("arg0_ptr.hi32".to_string()),
                    Expr::Imm("1".to_string()),
                    Expr::Reg("b0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v8".to_string()),
            src: Expr::Load {
                ty: Some("uint8_t".to_string()),
                addr: Box::new(Expr::Addr64 {
                    lo: Box::new(Expr::Reg("v6".to_string())),
                    hi: Box::new(Expr::Reg("v7".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(rendered.contains("v5 = v4 & 255;"));
    assert!(rendered.contains("v8 = *((uint8_t*)(arg0_ptr + (int64_t)v5));"));
    assert!(!rendered.contains("addr64("));
    assert!(!rendered.contains("lea_hi_x_sx32("));
    assert!(!rendered.contains("arg0_ptr.lo32"));
    assert!(!rendered.contains("arg0_ptr.hi32"));
}

#[test]
fn addr64_fold_resolves_guard_selected_pointer_pair() {
    let guard = Expr::Unary {
        op: "!".to_string(),
        arg: Box::new(Expr::Reg("p0".to_string())),
    };
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v5".to_string()),
            src: Expr::Binary {
                op: "&".to_string(),
                lhs: Box::new(Expr::Reg("v4".to_string())),
                rhs: Box::new(Expr::Imm("255".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("b0".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Reg("v5".to_string()),
                    Expr::ConstMemSymbol("arg4_ptr.lo32".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("lo_live".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("v5".to_string())),
                rhs: Box::new(Expr::ConstMemSymbol("arg4_ptr.lo32".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("hi_live".to_string()),
            src: Expr::CallLike {
                func: "lea_hi_x_sx32".to_string(),
                args: vec![
                    Expr::Reg("v5".to_string()),
                    Expr::ConstMemSymbol("arg4_ptr.hi32".to_string()),
                    Expr::Imm("1".to_string()),
                    Expr::Reg("b0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("lo_selected".to_string()),
            src: Expr::Ternary {
                cond: Box::new(guard.clone()),
                then_expr: Box::new(Expr::Reg("lo_live".to_string())),
                else_expr: Box::new(Expr::Reg("old_lo".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("hi_selected".to_string()),
            src: Expr::Ternary {
                cond: Box::new(guard.clone()),
                then_expr: Box::new(Expr::Reg("hi_live".to_string())),
                else_expr: Box::new(Expr::Reg("old_hi".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("out".to_string()),
            src: Expr::Ternary {
                cond: Box::new(guard),
                then_expr: Box::new(Expr::Load {
                    ty: Some("uint8_t".to_string()),
                    addr: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("lo_selected".to_string())),
                        hi: Box::new(Expr::Reg("hi_selected".to_string())),
                    }),
                }),
                else_expr: Box::new(Expr::Reg("out_old".to_string())),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(rendered.contains("out = !p0 ? (*((uint8_t*)(arg4_ptr + (int64_t)v5))) : out_old;"));
    assert!(!rendered.contains("addr64("));
    assert!(!rendered.contains("lea_hi_x_sx32("));
    assert!(!rendered.contains("lo_selected"));
    assert!(!rendered.contains("hi_selected"));
    assert!(!rendered.contains("arg4_ptr.lo32"));
    assert!(!rendered.contains("arg4_ptr.hi32"));
}

#[test]
fn match_carry_expr_accepts_reversed_ptr_lo_operands() {
    let carry = match_carry_expr(&Expr::CallLike {
        func: "carry_u32_add3".to_string(),
        args: vec![
            Expr::Reg("R8.9".to_string()),
            Expr::Imm("32".to_string()),
            Expr::Imm("0".to_string()),
        ],
    })
    .expect("expected carry match");
    assert_eq!(carry.ptr_lo, "R8.9");
    assert_eq!(carry.offset.render(), "32");
}

#[test]
fn addr64_fold_handles_split_pair_hi_carry_with_reversed_operands() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("UR4.9".to_string()),
            src: Expr::ConstMemSymbol("arg2_ptr.lo32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("UR5.2".to_string()),
            src: Expr::ConstMemSymbol("arg2_ptr.hi32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("R8.9".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("R4.0".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                }),
                rhs: Box::new(Expr::Reg("UR4.9".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("P0.15".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Binary {
                        op: "*".to_string(),
                        lhs: Box::new(Expr::Reg("R4.0".to_string())),
                        rhs: Box::new(Expr::Imm("4".to_string())),
                    },
                    Expr::Reg("UR4.9".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R9.10".to_string()),
            src: Expr::CallLike {
                func: "lea_hi_x".to_string(),
                args: vec![
                    Expr::Reg("R4.0".to_string()),
                    Expr::Reg("UR5.2".to_string()),
                    Expr::Imm("2".to_string()),
                    Expr::Reg("P0.15".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R8.10".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("R8.9".to_string())),
                rhs: Box::new(Expr::Imm("32".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R9.11".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("R9.10".to_string())),
                rhs: Box::new(Expr::Ternary {
                    cond: Box::new(Expr::CallLike {
                        func: "carry_u32_add3".to_string(),
                        args: vec![
                            Expr::Reg("R8.9".to_string()),
                            Expr::Imm("32".to_string()),
                            Expr::Imm("0".to_string()),
                        ],
                    }),
                    then_expr: Box::new(Expr::Imm("1".to_string())),
                    else_expr: Box::new(Expr::Imm("0".to_string())),
                }),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R11.11".to_string()),
            src: Expr::Reg("R9.11".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("R10.11".to_string()),
            src: Expr::Reg("R8.10".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("v33".to_string()),
            src: Expr::Load {
                ty: Some("uint32_t".to_string()),
                addr: Box::new(Expr::Binary {
                    op: "-".to_string(),
                    lhs: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("R10.11".to_string())),
                        hi: Box::new(Expr::Reg("R11.11".to_string())),
                    }),
                    rhs: Box::new(Expr::Imm("28".to_string())),
                }),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v34".to_string()),
            src: Expr::Load {
                ty: Some("uint32_t".to_string()),
                addr: Box::new(Expr::Addr64 {
                    lo: Box::new(Expr::Reg("R10.11".to_string())),
                    hi: Box::new(Expr::Reg("R11.11".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("arg2_ptr + (int64_t)R4.0 * 4"),
        "got:
{}",
        rendered
    );
    assert!(
        rendered.contains("v34 = *((uint32_t*)"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("((uintptr_t)"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("R10.11"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("R11.11"),
        "got:
{}",
        rendered
    );
}

#[test]
fn addr64_fold_resolves_pair_defs_initialized_inside_single_arm_if() {
    let stmt = Stmt::Sequence(vec![
        Stmt::If {
            condition: Expr::Reg("p0".to_string()),
            then_branch: Box::new(Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("base_lo".to_string()),
                    src: Expr::ConstMemSymbol("arg0_ptr.lo32".to_string()),
                },
                Stmt::Assign {
                    dst: LValue::Var("base_hi".to_string()),
                    src: Expr::ConstMemSymbol("arg0_ptr.hi32".to_string()),
                },
                Stmt::Assign {
                    dst: LValue::Var("lo1".to_string()),
                    src: Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Binary {
                            op: "*".to_string(),
                            lhs: Box::new(Expr::Reg("idx".to_string())),
                            rhs: Box::new(Expr::Imm("4".to_string())),
                        }),
                        rhs: Box::new(Expr::Reg("base_lo".to_string())),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("carry0".to_string()),
                    src: Expr::CallLike {
                        func: "carry_u32_add3".to_string(),
                        args: vec![
                            Expr::Binary {
                                op: "*".to_string(),
                                lhs: Box::new(Expr::Reg("idx".to_string())),
                                rhs: Box::new(Expr::Imm("4".to_string())),
                            },
                            Expr::Reg("base_lo".to_string()),
                            Expr::Imm("0".to_string()),
                        ],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("hi1".to_string()),
                    src: Expr::CallLike {
                        func: "lea_hi_x".to_string(),
                        args: vec![
                            Expr::Reg("idx".to_string()),
                            Expr::Reg("base_hi".to_string()),
                            Expr::Imm("2".to_string()),
                            Expr::Reg("carry0".to_string()),
                        ],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("lo2".to_string()),
                    src: Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Reg("lo1".to_string())),
                        rhs: Box::new(Expr::Imm("32".to_string())),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("hi2".to_string()),
                    src: Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Reg("hi1".to_string())),
                        rhs: Box::new(Expr::Ternary {
                            cond: Box::new(Expr::CallLike {
                                func: "carry_u32_add3".to_string(),
                                args: vec![
                                    Expr::Reg("lo1".to_string()),
                                    Expr::Imm("32".to_string()),
                                    Expr::Imm("0".to_string()),
                                ],
                            }),
                            then_expr: Box::new(Expr::Imm("1".to_string())),
                            else_expr: Box::new(Expr::Imm("0".to_string())),
                        }),
                    },
                },
            ])),
            else_branch: None,
        },
        Stmt::Assign {
            dst: LValue::Var("lo3".to_string()),
            src: Expr::Reg("lo2".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("hi3".to_string()),
            src: Expr::Reg("hi2".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("out".to_string()),
            src: Expr::Load {
                ty: Some("uint32_t".to_string()),
                addr: Box::new(Expr::Binary {
                    op: "-".to_string(),
                    lhs: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("lo3".to_string())),
                        hi: Box::new(Expr::Reg("hi3".to_string())),
                    }),
                    rhs: Box::new(Expr::Imm("32".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("arg0_ptr"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("((uintptr_t)"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("lo3"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("hi3"),
        "got:
{}",
        rendered
    );
}

#[test]
fn addr64_fold_handles_pre_offset_pair_added_to_pointer_base() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v30".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("v4".to_string())),
                    rhs: Box::new(Expr::Imm("4".to_string())),
                }),
                rhs: Box::new(Expr::Imm("16".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("b2".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Binary {
                        op: "*".to_string(),
                        lhs: Box::new(Expr::Reg("v4".to_string())),
                        rhs: Box::new(Expr::Imm("4".to_string())),
                    },
                    Expr::Imm("16".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v31".to_string()),
            src: Expr::CallLike {
                func: "lea_hi_x".to_string(),
                args: vec![
                    Expr::Reg("v4".to_string()),
                    Expr::Imm("0".to_string()),
                    Expr::Imm("2".to_string()),
                    Expr::Reg("b2".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v7".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("v30".to_string())),
                rhs: Box::new(Expr::ConstMemSymbol("arg0_ptr.lo32".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v3".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v31".to_string())),
                    rhs: Box::new(Expr::ConstMemSymbol("arg0_ptr.hi32".to_string())),
                }),
                rhs: Box::new(Expr::Ternary {
                    cond: Box::new(Expr::CallLike {
                        func: "carry_u32_add3".to_string(),
                        args: vec![
                            Expr::Reg("v30".to_string()),
                            Expr::ConstMemSymbol("arg0_ptr.lo32".to_string()),
                            Expr::Imm("0".to_string()),
                        ],
                    }),
                    then_expr: Box::new(Expr::Imm("1".to_string())),
                    else_expr: Box::new(Expr::Imm("0".to_string())),
                }),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("out".to_string()),
            src: Expr::Load {
                ty: Some("uint32_t".to_string()),
                addr: Box::new(Expr::Binary {
                    op: "-".to_string(),
                    lhs: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("v7".to_string())),
                        hi: Box::new(Expr::Reg("v3".to_string())),
                    }),
                    rhs: Box::new(Expr::Imm("16".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("out = *((uint32_t*)"),
        "got:
{}",
        rendered
    );
    assert!(
        rendered.contains("arg0_ptr"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("((uintptr_t)"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("arg0_ptr.lo32"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("arg0_ptr.hi32"),
        "got:
{}",
        rendered
    );
}

#[test]
fn addr64_fold_preserves_outer_pointer_offset_shape() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("v5".to_string()),
            src: Expr::Binary {
                op: "&".to_string(),
                lhs: Box::new(Expr::Reg("v4".to_string())),
                rhs: Box::new(Expr::Imm("255".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("b0".to_string()),
            src: Expr::CallLike {
                func: "carry_u32_add3".to_string(),
                args: vec![
                    Expr::Reg("v5".to_string()),
                    Expr::ConstMemSymbol("arg4_ptr.lo32".to_string()),
                    Expr::Imm("0".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v6".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("v5".to_string())),
                rhs: Box::new(Expr::ConstMemSymbol("arg4_ptr.lo32".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v8".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Reg("v5".to_string())),
                    rhs: Box::new(Expr::ConstMemSymbol("arg4_ptr.hi32".to_string())),
                }),
                rhs: Box::new(Expr::Ternary {
                    cond: Box::new(Expr::Reg("b0".to_string())),
                    then_expr: Box::new(Expr::Imm("1".to_string())),
                    else_expr: Box::new(Expr::Imm("0".to_string())),
                }),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v9".to_string()),
            src: Expr::Load {
                ty: Some("uint8_t".to_string()),
                addr: Box::new(Expr::Binary {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::Addr64 {
                        lo: Box::new(Expr::Reg("v6".to_string())),
                        hi: Box::new(Expr::Reg("v8".to_string())),
                    }),
                    rhs: Box::new(Expr::Imm("1".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("v9 = *((uint8_t*)((arg4_ptr + (int64_t)v5) + 1));")
            || rendered.contains("v9 = *((uint8_t*)(arg4_ptr + (int64_t)v5 + 1));")
    );
    assert!(!rendered.contains("addr64("));
    assert!(!rendered.contains("arg4_ptr.lo32"));
    assert!(!rendered.contains("arg4_ptr.hi32"));
}

#[test]
fn addr64_fold_keeps_loop_carried_pointer_pair_symbolic() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("base_lo".to_string()),
            src: Expr::ConstMemSymbol("arg2_ptr.lo32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("base_hi".to_string()),
            src: Expr::ConstMemSymbol("arg2_ptr.hi32".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("seed_lo".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::ConstMemSymbol("arg2_ptr.lo32".to_string())),
                rhs: Box::new(Expr::Imm("16".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("seed_hi".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::ConstMemSymbol("arg2_ptr.hi32".to_string())),
                rhs: Box::new(Expr::Ternary {
                    cond: Box::new(Expr::CallLike {
                        func: "carry_u32_add3".to_string(),
                        args: vec![
                            Expr::ConstMemSymbol("arg2_ptr.lo32".to_string()),
                            Expr::Imm("16".to_string()),
                            Expr::Imm("0".to_string()),
                        ],
                    }),
                    then_expr: Box::new(Expr::Imm("1".to_string())),
                    else_expr: Box::new(Expr::Imm("0".to_string())),
                }),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("phi_lo".to_string()),
            src: Expr::Reg("seed_lo".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("phi_hi".to_string()),
            src: Expr::Reg("seed_hi".to_string()),
        },
        Stmt::Loop {
            kind: LoopKind::DoWhile,
            condition: Some(Expr::Reg("p0".to_string())),
            body: Box::new(Stmt::Sequence(vec![
                Stmt::Assign {
                    dst: LValue::Var("out".to_string()),
                    src: Expr::Load {
                        ty: Some("uint32_t".to_string()),
                        addr: Box::new(Expr::Binary {
                            op: "-".to_string(),
                            lhs: Box::new(Expr::Addr64 {
                                lo: Box::new(Expr::Reg("phi_lo".to_string())),
                                hi: Box::new(Expr::Reg("phi_hi".to_string())),
                            }),
                            rhs: Box::new(Expr::Imm("16".to_string())),
                        }),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("next_lo".to_string()),
                    src: Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Reg("phi_lo".to_string())),
                        rhs: Box::new(Expr::Imm("32".to_string())),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("next_hi".to_string()),
                    src: Expr::Binary {
                        op: "+".to_string(),
                        lhs: Box::new(Expr::Reg("phi_hi".to_string())),
                        rhs: Box::new(Expr::Ternary {
                            cond: Box::new(Expr::CallLike {
                                func: "carry_u32_add3".to_string(),
                                args: vec![
                                    Expr::Reg("phi_lo".to_string()),
                                    Expr::Imm("32".to_string()),
                                    Expr::Imm("0".to_string()),
                                ],
                            }),
                            then_expr: Box::new(Expr::Imm("1".to_string())),
                            else_expr: Box::new(Expr::Imm("0".to_string())),
                        }),
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("phi_lo".to_string()),
                    src: Expr::CallLike {
                        func: "__loop_phi".to_string(),
                        args: vec![Expr::Reg("next_lo".to_string())],
                    },
                },
                Stmt::Assign {
                    dst: LValue::Var("phi_hi".to_string()),
                    src: Expr::CallLike {
                        func: "__loop_phi".to_string(),
                        args: vec![Expr::Reg("next_hi".to_string())],
                    },
                },
            ])),
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    let out_load = regex::Regex::new(
        r"out = \*\(\(uint32_t\*\)\(\(\(uint8_t\*\)arg2_ptr\) \+ \(int64_t\)phi_lo - \(\(uint32_t\)\(uintptr_t\)arg2_ptr\) - 16\)\);",
    )
    .expect("valid loop-carried pointer regex");
    assert!(out_load.is_match(&rendered), "got:\n{}", rendered);
    assert!(
        !rendered.contains("arg2_ptr + (int64_t)16 - 16"),
        "got:
{}",
        rendered
    );
    assert!(
        !rendered.contains("arg2_ptr.lo32"),
        "got:
{}",
        rendered
    );
}

#[test]
fn addr64_fold_recovers_explicit_lanes_from_byte_cast_pointer_base() {
    let byte_addr = Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(Expr::Cast {
            ty: "uint8_t*".to_string(),
            expr: Box::new(Expr::Raw("arg0_ptr".to_string())),
        }),
        rhs: Box::new(Expr::Binary {
            op: "*".to_string(),
            lhs: Box::new(Expr::Reg("idx".to_string())),
            rhs: Box::new(Expr::Imm("4".to_string())),
        }),
    };
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("lo".to_string()),
            src: Expr::Cast {
                ty: "uint32_t".to_string(),
                expr: Box::new(byte_addr.clone()),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("hi".to_string()),
            src: Expr::Cast {
                ty: "uint32_t".to_string(),
                expr: Box::new(Expr::Binary {
                    op: ">>".to_string(),
                    lhs: Box::new(Expr::Cast {
                        ty: "uint64_t".to_string(),
                        expr: Box::new(byte_addr),
                    }),
                    rhs: Box::new(Expr::Imm("32".to_string())),
                }),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("out".to_string()),
            src: Expr::Load {
                ty: Some("float".to_string()),
                addr: Box::new(Expr::Addr64 {
                    lo: Box::new(Expr::Reg("lo".to_string())),
                    hi: Box::new(Expr::Reg("hi".to_string())),
                }),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(rendered.contains("out = *(arg0_ptr + idx);"), "got:\n{}", rendered);
    assert!(!rendered.contains("addr64("), "got:\n{}", rendered);
}

#[test]
fn addr64_fold_prefers_lo_backed_pointer_when_hi_tracks_next_byte() {
    let byte_addr = Expr::Binary {
        op: "+".to_string(),
        lhs: Box::new(Expr::Cast {
            ty: "uint8_t*".to_string(),
            expr: Box::new(Expr::Raw("arg6_ptr".to_string())),
        }),
        rhs: Box::new(Expr::Reg("idx".to_string())),
    };
    let stmt = Stmt::Sequence(vec![Stmt::Assign {
        dst: LValue::Var("out".to_string()),
        src: Expr::Load {
            ty: Some("uint8_t".to_string()),
            addr: Box::new(Expr::Addr64 {
                lo: Box::new(Expr::Cast {
                    ty: "uint32_t".to_string(),
                    expr: Box::new(byte_addr.clone()),
                }),
                hi: Box::new(Expr::Cast {
                    ty: "uint32_t".to_string(),
                    expr: Box::new(Expr::Binary {
                        op: ">>".to_string(),
                        lhs: Box::new(Expr::Cast {
                            ty: "uint64_t".to_string(),
                            expr: Box::new(Expr::Binary {
                                op: "+".to_string(),
                                lhs: Box::new(byte_addr),
                                rhs: Box::new(Expr::Imm("1".to_string())),
                            }),
                        }),
                        rhs: Box::new(Expr::Imm("32".to_string())),
                    }),
                }),
            }),
        },
    }]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(
        rendered.contains("out = *(arg6_ptr + idx);")
            || rendered.contains("out = *((uint8_t*)(arg6_ptr + (int64_t)idx));"),
        "got:\n{}",
        rendered
    );
    assert!(!rendered.contains("((uintptr_t)(((uint64_t)"), "got:\n{}", rendered);
}

#[test]
fn simplify_recovers_division_from_fchk_guarded_rcp_refine() {
    let seq = vec![
        Stmt::Assign {
            dst: LValue::Var("v2".to_string()),
            src: Expr::Reg("arg0".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("v10".to_string()),
            src: Expr::CallLike {
                func: "rcp_approx".to_string(),
                args: vec![Expr::Reg("v9".to_string())],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("b4".to_string()),
            src: Expr::CallLike {
                func: "FCHK".to_string(),
                args: vec![Expr::Reg("v2".to_string()), Expr::Reg("v9".to_string())],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v11".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Unary {
                        op: "-".to_string(),
                        arg: Box::new(Expr::Reg("v9".to_string())),
                    }),
                    rhs: Box::new(Expr::Reg("v10".to_string())),
                }),
                rhs: Box::new(Expr::Imm("1".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v12".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("v10".to_string())),
                    rhs: Box::new(Expr::Reg("v11".to_string())),
                }),
                rhs: Box::new(Expr::Reg("v10".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v13".to_string()),
            src: Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(Expr::Reg("v12".to_string())),
                rhs: Box::new(Expr::Reg("arg0".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v14".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Unary {
                        op: "-".to_string(),
                        arg: Box::new(Expr::Reg("v9".to_string())),
                    }),
                    rhs: Box::new(Expr::Reg("v13".to_string())),
                }),
                rhs: Box::new(Expr::Reg("arg0".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("v15".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("v12".to_string())),
                    rhs: Box::new(Expr::Reg("v14".to_string())),
                }),
                rhs: Box::new(Expr::Reg("v13".to_string())),
            },
        },
        Stmt::If {
            condition: Expr::Reg("b4".to_string()),
            then_branch: Box::new(Stmt::Assign {
                dst: LValue::Var("out".to_string()),
                src: Expr::Reg("v12".to_string()),
            }),
            else_branch: Some(Box::new(Stmt::Assign {
                dst: LValue::Var("out".to_string()),
                src: Expr::Reg("v15".to_string()),
            })),
        },
        Stmt::Return(Some(Expr::Reg("out".to_string()))),
    ];

    let mut defs = HashMap::new();
    for stmt in seq.iter().take(seq.len() - 2) {
        update_linear_defs(stmt, &mut defs);
    }
    assert!(match_rcp_refine_expr(&Expr::Reg("v12".to_string()), &defs).is_some());
    assert!(match_newton_correction_expr(&Expr::Reg("v14".to_string()), &defs).is_some());
    assert!(
        match_rcp_division_expr(&Expr::Reg("v15".to_string()), &defs).is_some(),
        "div match failed: {:?}",
        defs
    );
    let recovered = recover_fchk_division_stmt(&seq[seq.len() - 2], &defs, &BTreeSet::new());
    assert!(recovered.is_some(), "defs={:?}", defs);

    let stmt = Stmt::Sequence(seq);
    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert!(rendered.contains("out = arg0 / v9;"), "got:\n{}", rendered);
    assert!(!rendered.contains("FCHK("), "got:\n{}", rendered);
}

#[test]
fn simplify_recovers_division_from_fchk_guarded_rcp_refine_with_aliased_denominator() {
    let seq = vec![
        Stmt::Assign {
            dst: LValue::Var("R2.1".to_string()),
            src: Expr::Reg("arg0".to_string()),
        },
        Stmt::Assign {
            dst: LValue::Var("R9.2".to_string()),
            src: Expr::Cast {
                ty: "float".to_string(),
                expr: Box::new(Expr::Reg("UR4.2".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R0.3".to_string()),
            src: Expr::CallLike {
                func: "rcp_approx".to_string(),
                args: vec![Expr::Reg("R9.2".to_string())],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("P0.4".to_string()),
            src: Expr::CallLike {
                func: "FCHK".to_string(),
                args: vec![
                    Expr::Reg("R2.1".to_string()),
                    Expr::Reg("R9.2".to_string()),
                ],
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R3.2".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Unary {
                        op: "-".to_string(),
                        arg: Box::new(Expr::Reg("R9.2".to_string())),
                    }),
                    rhs: Box::new(Expr::Reg("R0.3".to_string())),
                }),
                rhs: Box::new(Expr::Imm("1".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R0.4".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("R0.3".to_string())),
                    rhs: Box::new(Expr::Reg("R3.2".to_string())),
                }),
                rhs: Box::new(Expr::Reg("R0.3".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R3.3".to_string()),
            src: Expr::Binary {
                op: "*".to_string(),
                lhs: Box::new(Expr::Reg("R0.4".to_string())),
                rhs: Box::new(Expr::Reg("arg0".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R8.2".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Unary {
                        op: "-".to_string(),
                        arg: Box::new(Expr::Reg("R9.2".to_string())),
                    }),
                    rhs: Box::new(Expr::Reg("R3.3".to_string())),
                }),
                rhs: Box::new(Expr::Reg("arg0".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("R8.3".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Binary {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::Reg("R0.4".to_string())),
                    rhs: Box::new(Expr::Reg("R8.2".to_string())),
                }),
                rhs: Box::new(Expr::Reg("R3.3".to_string())),
            },
        },
        Stmt::If {
            condition: Expr::Reg("P0.4".to_string()),
            then_branch: Box::new(Stmt::Assign {
                dst: LValue::Var("R8.6".to_string()),
                src: Expr::Reg("R0.4".to_string()),
            }),
            else_branch: Some(Box::new(Stmt::Assign {
                dst: LValue::Var("R8.6".to_string()),
                src: Expr::Reg("R8.3".to_string()),
            })),
        },
        Stmt::Return(Some(Expr::Reg("R8.6".to_string()))),
    ];

    let mut defs = HashMap::new();
    for stmt in seq.iter().take(seq.len() - 2) {
        update_linear_defs(stmt, &mut defs);
    }

    let recovered = recover_fchk_division_stmt(&seq[seq.len() - 2], &defs, &BTreeSet::new());
    assert!(recovered.is_some(), "defs={:?}", defs);

    let rendered = ast_cleanup(Stmt::Sequence(seq)).render_with_indent(0);
    assert!(
        rendered.contains("R8.6 = arg0 / (float)(UR4.2);"),
        "got:\n{}",
        rendered
    );
    assert!(!rendered.contains("FCHK("), "got:\n{}", rendered);
}

#[test]
fn simplify_dedups_consecutive_identical_pure_assignments() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("out".to_string()),
            src: Expr::Binary {
                op: "/".to_string(),
                lhs: Box::new(Expr::Reg("arg0".to_string())),
                rhs: Box::new(Expr::Reg("arg1".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("tmp".to_string()),
            src: Expr::Binary {
                op: "+".to_string(),
                lhs: Box::new(Expr::Reg("arg2".to_string())),
                rhs: Box::new(Expr::Imm("1".to_string())),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("out".to_string()),
            src: Expr::Binary {
                op: "/".to_string(),
                lhs: Box::new(Expr::Reg("arg0".to_string())),
                rhs: Box::new(Expr::Reg("arg1".to_string())),
            },
        },
        Stmt::Return(Some(Expr::Reg("out".to_string()))),
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert_eq!(rendered.matches("out = arg0 / arg1;").count(), 1, "got:\n{}", rendered);
}

#[test]
fn simplify_keeps_consecutive_impure_assignments() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Assign {
            dst: LValue::Var("x".to_string()),
            src: Expr::Load {
                addr: Box::new(Expr::Reg("ptr".to_string())),
                ty: Some("uint32_t".to_string()),
            },
        },
        Stmt::Assign {
            dst: LValue::Var("x".to_string()),
            src: Expr::Load {
                addr: Box::new(Expr::Reg("ptr".to_string())),
                ty: Some("uint32_t".to_string()),
            },
        },
    ]);

    let rendered = ast_cleanup(stmt).render_with_indent(0);
    assert_eq!(rendered.matches("x = *((uint32_t*)ptr);").count(), 2, "got:\n{}", rendered);
}

#[test]
fn guard_select_specialize_sees_loop_local_select_defs() {
    let stmt = Stmt::Loop {
        kind: LoopKind::While,
        condition: Some(Expr::Imm("true".to_string())),
        body: Box::new(Stmt::Block(vec![
            Stmt::Assign {
                dst: LValue::Var("v38".to_string()),
                src: Expr::Ternary {
                    cond: Box::new(Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("b6".to_string())),
                    }),
                    then_expr: Box::new(Expr::Imm("4".to_string())),
                    else_expr: Box::new(Expr::Reg("ctaid_x".to_string())),
                },
            },
            Stmt::Assign {
                dst: LValue::Var("out".to_string()),
                src: Expr::Ternary {
                    cond: Box::new(Expr::Unary {
                        op: "!".to_string(),
                        arg: Box::new(Expr::Reg("b6".to_string())),
                    }),
                    then_expr: Box::new(Expr::Load {
                        ty: Some("float".to_string()),
                        addr: Box::new(Expr::Binary {
                            op: "+".to_string(),
                            lhs: Box::new(Expr::Reg("arg2_ptr".to_string())),
                            rhs: Box::new(Expr::Binary {
                                op: "*".to_string(),
                                lhs: Box::new(Expr::Reg("idx".to_string())),
                                rhs: Box::new(Expr::Reg("v38".to_string())),
                            }),
                        }),
                    }),
                    else_expr: Box::new(Expr::Imm("0".to_string())),
                },
            },
        ])),
    };

    let rendered = ast_guard_select_specialize(stmt).render_with_indent(0);
    assert!(rendered.contains("idx * 4"), "got:\n{}", rendered);
    assert!(
        !rendered.contains("idx * v38"),
        "loop-local guarded select should have been specialized in-place:\n{}",
        rendered
    );
}
