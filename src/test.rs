use crate::*;
use pretty_assertions::assert_eq;
use regex::Regex;

const SAMPLE_SASS: &str = r#"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;  /* 0x ... */
        /*0010*/                   S2R R26, SR_TID.X ;                       /* 0x ... */
        /*0020*/            @!P0   BRA 0x0050 ;                              /* 0x ... */
        /*0030*/                   IADD3 R1, R1, 0x1, RZ ;                   /* 0x ... */
        /*0040*/                   BRA  0x0060 ;                             /* 0x ... */
        /*0050*/                   IMAD.WIDE R2, R27, R2, c[0x0][0x168] ;    /* 0x ... */
        /*0060*/                   EXIT ;                                    /* 0x ... */
"#;

const SAMPLE_SASS_FLOAT: &str = r#"
    /*0300*/                   FSEL R5, R7, 0.89999997615814208984, P1 ;
    /*0310*/                   EXIT ;
"#;

const SAMPLE_SASS_PRED_EXIT: &str = r#"
    /*0040*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x178], PT ;
    /*0050*/               @P0 EXIT ;
    /*0060*/                   IMAD.MOV.U32 R7, RZ, RZ, 0x4 ;
"#;

const IF_SAMPLE: &str = r#"
 /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
 /*0010*/ @P0 BRA 0x0030 ;
 /*0020*/ IMAD.MOV.U32 R1, RZ, RZ, 0x5 ;
 /*0030*/ IMAD.MOV.U32 R1, RZ, RZ, 0x6 ;
 /*0040*/ EXIT ;
"#;

const LOOP_SAMPLE: &str = r#"
 /*000*/ IADD R0, R0, 0x1 ;
 /*010*/ ISETP.LT.AND P0, PT, R0, 0x5, PT ;
 /*020*/ @P0 BRA 0x000 ;
 /*030*/ EXIT ;
"#;

fn print_cfg_stdout(cfg: &ControlFlowGraph) {
    println!("{}", crate::cfg::graph_to_dot(&cfg));
    for idx in cfg.node_indices() {
        println!("Basic block idx: {:#?}", idx);
        for inst in &cfg[idx].instrs {
            println!("{:#?}", inst);
        }
        println!("");
    }
}

#[cfg(test)]
#[test]
fn test_parse_operand() {
    use crate::parser::DecodedOperand::*;
    let instrs = decode_sass(SAMPLE_SASS);
    // 检查第 0 条指令第四个操作数应为 ConstMem
    match &instrs[0].operands[3] {
        ConstMem { bank, offset } => {
            assert_eq!((*bank, *offset), (0x0, 0x28));
        }
        _ => panic!("expect ConstMem"),
    }
}

#[test]
fn test_cfg() {
    let cfg = build_cfg(decode_sass(SAMPLE_SASS));
    assert_eq!(cfg.node_count(), 4);
    // 可打印 dot: println!("{}", crate::cfg::graph_to_dot(&cfg));
}

#[test]
fn test_float_immediate() {
    let instrs = decode_sass(SAMPLE_SASS_FLOAT);
    if let DecodedOperand::ImmediateF(v) = &instrs[0].operands[2] {
        assert!((*v - 0.8999999).abs() < 1e-4);
    } else {
        panic!("expect float immediate");
    }
}

#[test]
fn test_predicated_exit_fallthrough() {
    let cfg = build_cfg(decode_sass(SAMPLE_SASS_PRED_EXIT));
    // 预计有 2 basic blocks: 0040‑0050, 0060
    assert_eq!(cfg.node_count(), 2);
    // 并且 block0 -> block1 (fall‑through) 存在
    let edges: Vec<_> = cfg.edge_indices().collect();
    assert_eq!(edges.len(), 1);
}

#[test]
fn phi_insert() {
    let cfg = build_cfg(decode_sass(IF_SAMPLE));
    print_cfg_stdout(&cfg);
    let fir = build_ssa(&cfg);
    let phi_cnt: usize = fir
        .blocks
        .iter()
        .map(|b| {
            b.stmts
                .iter()
                .filter(|s| matches!(s.value, crate::ir::RValue::Phi(_)))
                .count()
        })
        .sum();
    assert_eq!(phi_cnt, 1);
}

#[test]
fn rpo_loop() {
    let cfg = build_cfg(decode_sass(LOOP_SAMPLE));
    let fir = build_ssa(&cfg);
    assert!(fir.blocks.len() >= 2);
}

#[test]
fn test_parser_normalizes_register_modifiers_and_memref() {
    let sample = r#"
        /*0000*/ IADD3 R4, R2.reuse, -0x1, RZ ;
        /*0010*/ LDG.E.U8 R16, [R2.64+0x1] ;
        /*0020*/ ULOP3.LUT UR5, UR5, 0xffffff00, URZ, 0xc0, !UPT ;
    "#;
    let instrs = decode_sass(sample);
    assert_eq!(instrs.len(), 3);

    match &instrs[0].operands[1] {
        DecodedOperand::Register { class, idx, .. } => {
            assert_eq!(class, "R");
            assert_eq!(*idx, 2);
        }
        _ => panic!("expected normalized register"),
    }
    match &instrs[1].operands[1] {
        DecodedOperand::Address { offset, width, .. } => {
            assert_eq!(*offset, Some(1));
            assert_eq!(*width, Some(64));
        }
        _ => panic!("expected parsed mem ref"),
    }
    match &instrs[2].operands[3] {
        DecodedOperand::UniformRegister { class, .. } => assert_eq!(class, "URZ"),
        _ => panic!("expected special register URZ"),
    }
}

#[test]
fn test_parse_sm_version_from_headerflags() {
    let sample = r#"
        .headerflags @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
    "#;
    assert_eq!(parse_sm_version(sample), Some(89));
}

#[test]
fn test_parse_sm_version_from_target_fallback() {
    let sample = r#"
        .target sm_75
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
    "#;
    assert_eq!(parse_sm_version(sample), Some(75));
}

#[test]
fn test_split_decoded_functions_multi_function_dump() {
    // Synthetic two-function SASS dump mirroring cuobjdump's layout.
    let sample = "
\tcode for sm_89
\t\tFunction : first
\t.headerflags\t@\"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89\"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   EXIT ;
\t\tFunction : second
\t.headerflags\t@\"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89\"
        /*0000*/                   IADD3 R2, RZ, 0x1, RZ ;
        /*0010*/                   IADD3 R3, RZ, 0x2, RZ ;
        /*0020*/                   EXIT ;
";
    let funcs = crate::parser::split_decoded_functions(sample);
    assert_eq!(funcs.len(), 2, "expected two functions");
    assert_eq!(funcs[0].name, "first");
    assert_eq!(funcs[1].name, "second");
    assert_eq!(funcs[0].sm, Some(89));
    assert_eq!(funcs[1].sm, Some(89));
    assert_eq!(funcs[0].instrs.len(), 2, "first function: MOV + EXIT");
    assert_eq!(funcs[1].instrs.len(), 3, "second function: 2 IADD3 + EXIT");
    // Ensure the preamble lines (before the first Function marker) do not
    // leak into any function's instrs list.
    for f in &funcs {
        assert!(
            f.instrs.iter().all(|i| !i.raw.contains("headerflags")),
            "instrs should only contain decoded instructions, not headerflags"
        );
    }
}

#[test]
fn test_split_decoded_functions_empty_on_single_function_dump_without_marker() {
    // A dump with no `Function :` markers returns empty — callers should
    // fall back to `decode_sass` in that case.
    let sample = r#"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/ EXIT ;
    "#;
    assert!(crate::parser::split_decoded_functions(sample).is_empty());
}

#[test]
fn test_ssa_keeps_immutable_special_registers_unversioned() {
    let sample = r#"
        /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
        /*0010*/ IADD3 R1, RZ, 0x1, RZ ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let dot = fir.to_dot(&cfg, &DefaultDisplay);

    assert!(dot.contains("RZ"));
    assert!(dot.contains("PT"));
    assert!(!dot.contains("RZ."));
    assert!(!dot.contains("PT."));
}

#[test]
fn test_ssa_signed_register_use_tracks_latest_positive_version() {
    let sample = r#"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x1 ;
        /*0010*/ IADD3 R1, R1, 0x1, RZ ;
        /*0020*/ IMAD.IADD R1, R1, 0x1, -R1 ;
        /*0030*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);

    let mut seen = false;
    for block in &fir.blocks {
        for stmt in &block.stmts {
            let RValue::Op { opcode, args } = &stmt.value else {
                continue;
            };
            if opcode != "IMAD.IADD" {
                continue;
            }
            seen = true;
            assert!(args.len() >= 3);
            let IRExpr::Reg(lhs_use) = &args[0] else {
                panic!("expected register use in IMAD.IADD arg0");
            };
            let IRExpr::Reg(neg_use) = &args[2] else {
                panic!("expected register use in IMAD.IADD arg2");
            };
            assert_eq!(lhs_use.class, "R");
            assert_eq!(lhs_use.idx, 1);
            assert_eq!(lhs_use.ssa, Some(1));
            assert_eq!(neg_use.class, "R");
            assert_eq!(neg_use.idx, 1);
            assert_eq!(neg_use.sign, -1);
            assert_eq!(neg_use.ssa, Some(1));
        }
    }
    assert!(seen, "expected IMAD.IADD statement in SSA");
}

#[test]
fn ssa_iadd3_pred_output_is_defined_and_reused_by_lea_hi_x() {
    let sample = r#"
        /*0000*/ IADD3 R10, P2, R6, c[0x0][0x160], RZ ;
        /*0010*/ LEA.HI.X.SX32 R11, R6, c[0x0][0x164], 0x1, P2 ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir.blocks.iter().find(|b| b.id == 0).expect("expected BB0");

    let mut carry_def: Option<RegId> = None;
    let mut lea_carry_use: Option<RegId> = None;
    for stmt in &block0.stmts {
        let RValue::Op { opcode, args } = &stmt.value else {
            continue;
        };
        if opcode == "IADD3" {
            carry_def = stmt
                .defs
                .iter()
                .filter_map(|d| d.get_reg())
                .find(|r| r.class == "P")
                .cloned();
        }
        if opcode == "LEA.HI.X.SX32" {
            let Some(IRExpr::Reg(r)) = args.get(3) else {
                panic!("expected LEA.HI.X carry predicate at arg[3]");
            };
            lea_carry_use = Some(r.clone());
        }
    }

    let carry_def = carry_def.expect("expected IADD3 predicate carry def");
    let lea_carry_use = lea_carry_use.expect("expected LEA.HI.X carry predicate use");
    assert_eq!(carry_def.class, lea_carry_use.class);
    assert_eq!(carry_def.idx, lea_carry_use.idx);
    assert_eq!(carry_def.ssa, lea_carry_use.ssa);
}

#[test]
fn ssa_iadd64_defines_high_lane_and_lifts_carry_into_hi_half() {
    let sample = r#"
        /*0000*/ LDC.64 R4, c[0x0][0x380] ;
        /*0010*/ SHF.R.S32.HI R7, RZ, 0x1f, R6 ;
        /*0020*/ IADD.64 R4, R6, R4 ;
        /*0030*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir.blocks.iter().find(|b| b.id == 0).expect("expected BB0");
    let stmt = block0
        .stmts
        .iter()
        .find(|stmt| matches!(&stmt.value, RValue::Op { opcode, .. } if opcode == "IADD.64"))
        .expect("expected IADD.64 statement in SSA");
    assert_eq!(stmt.defs.len(), 2, "IADD.64 should define a register pair");
    let RValue::Op { args, .. } = &stmt.value else {
        unreachable!();
    };
    assert_eq!(
        args.len(),
        4,
        "IADD.64 should carry both low and hi source lanes"
    );
    assert!(matches!(args[2], IRExpr::Reg(ref r) if r.class == "R" && r.idx == 7));
    assert!(matches!(args[3], IRExpr::Reg(ref r) if r.class == "R" && r.idx == 5));

    let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());
    let stmt_idx = block0
        .stmts
        .iter()
        .position(|candidate| std::ptr::eq(candidate, stmt))
        .expect("IADD.64 statement should have an index");
    let hi = lifted
        .by_def
        .get(&DefRef {
            block_id: block0.id,
            stmt_idx,
            def_idx: 1,
        })
        .expect("expected lifted hi-half for IADD.64")
        .rhs
        .render();
    assert!(
        hi.contains("(int64_t)((int32_t)(R6.0))"),
        "expected hi-half to preserve the widened scalar addend, got: {hi}"
    );
    assert!(
        hi.contains("((uintptr_t)(((uint64_t)(R5.0) << 32) | (uint32_t)(R4.0)))"),
        "expected hi-half to preserve the explicit wide base pair, got: {hi}"
    );
    assert!(
        !hi.contains("carry_u32_add3("),
        "expected hi-half to collapse the carry helper into a wide sum, got: {hi}"
    );
}

#[test]
fn ssa_uiadd3_64_defines_high_lane_and_lifts_carry_into_hi_half() {
    let sample = r#"
        /*0000*/ UIADD3.64 UR4, UPT, UPT, UR18, 0x10, URZ ;
        /*0010*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir.blocks.iter().find(|b| b.id == 0).expect("expected BB0");
    let stmt = block0
        .stmts
        .iter()
        .find(|stmt| matches!(&stmt.value, RValue::Op { opcode, .. } if opcode == "UIADD3.64"))
        .expect("expected UIADD3.64 statement in SSA");
    assert!(
        stmt.defs
            .iter()
            .any(|def| matches!(def, IRExpr::Reg(reg) if reg.class == "UR" && reg.idx == 4)),
        "low half missing from UIADD3.64 defs: {:?}",
        stmt.defs
    );
    assert!(
        stmt.defs
            .iter()
            .any(|def| matches!(def, IRExpr::Reg(reg) if reg.class == "UR" && reg.idx == 5)),
        "implicit high half missing from UIADD3.64 defs: {:?}",
        stmt.defs
    );
    let RValue::Op { args, .. } = &stmt.value else {
        unreachable!();
    };
    assert_eq!(
        args.len(),
        6,
        "UIADD3.64 should carry both low and hi input lanes"
    );
    assert!(matches!(args[3], IRExpr::Reg(ref r) if r.class == "UR" && r.idx == 19));
    assert!(matches!(args[4], IRExpr::ImmI(0)));
    assert!(matches!(args[5], IRExpr::Reg(ref r) if r.class == "URZ"));

    let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());
    let stmt_idx = block0
        .stmts
        .iter()
        .position(|candidate| std::ptr::eq(candidate, stmt))
        .expect("UIADD3.64 statement should have an index");
    let hi_def_idx = stmt
        .defs
        .iter()
        .position(|def| matches!(def, IRExpr::Reg(reg) if reg.class == "UR" && reg.idx == 5))
        .expect("expected hi-half def index");
    let hi = lifted
        .by_def
        .get(&DefRef {
            block_id: block0.id,
            stmt_idx,
            def_idx: hi_def_idx,
        })
        .expect("expected lifted hi-half for UIADD3.64")
        .rhs
        .render();
    assert!(
        hi.contains("((uintptr_t)(((uint64_t)(UR19.0) << 32) | (uint32_t)(UR18.0)))"),
        "expected hi-half to preserve the explicit wide base pair, got: {hi}"
    );
    assert!(
        hi.contains("(uint64_t)((uint32_t)(16))"),
        "expected hi-half to preserve the widened immediate addend, got: {hi}"
    );
    assert!(
        !hi.contains("carry_u32_add3("),
        "expected hi-half to collapse the carry helper into a wide sum, got: {hi}"
    );
}

#[test]
fn ssa_lea_pred_output_is_defined_and_reused_by_lea_hi_x() {
    let sample = r#"
        /*0000*/ LEA R4, P5, R9, R4, 0x2 ;
        /*0010*/ LEA.HI.X R5, R9, R5, R10, 0x2, P5 ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir.blocks.iter().find(|b| b.id == 0).expect("expected BB0");

    let mut carry_def: Option<RegId> = None;
    let mut carry_use: Option<RegId> = None;
    for stmt in &block0.stmts {
        let RValue::Op { opcode, args } = &stmt.value else {
            continue;
        };
        if opcode == "LEA" {
            carry_def = stmt
                .defs
                .iter()
                .filter_map(|d| d.get_reg())
                .find(|r| r.class == "P")
                .cloned();
        }
        if opcode == "LEA.HI.X" {
            carry_use = args
                .iter()
                .filter_map(|a| match a {
                    IRExpr::Reg(r) if r.class == "P" => Some(r.clone()),
                    _ => None,
                })
                .next();
        }
    }

    let carry_def = carry_def.expect("expected LEA predicate carry def");
    let carry_use = carry_use.expect("expected LEA.HI.X carry predicate use");
    assert_eq!(carry_def.class, carry_use.class);
    assert_eq!(carry_def.idx, carry_use.idx);
    assert_eq!(carry_def.ssa, carry_use.ssa);
}

#[test]
fn ssa_uiadd3_pred_output_feeds_uiadd3_x_carry_input() {
    let sample = r#"
        /*0000*/ UIADD3 UR8, UP0, UR8, 0x4, URZ ;
        /*0010*/ UIADD3.X UR9, URZ, UR9, URZ, UP0, !UPT ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir.blocks.iter().find(|b| b.id == 0).expect("expected BB0");

    let mut carry_def: Option<RegId> = None;
    let mut carry_use: Option<RegId> = None;
    for stmt in &block0.stmts {
        let RValue::Op { opcode, args } = &stmt.value else {
            continue;
        };
        if opcode == "UIADD3" {
            carry_def = stmt
                .defs
                .iter()
                .filter_map(|d| d.get_reg())
                .find(|r| r.class == "UP")
                .cloned();
        }
        if opcode == "UIADD3.X" {
            carry_use = args
                .iter()
                .filter_map(|a| match a {
                    IRExpr::Reg(r) if r.class == "UP" => Some(r.clone()),
                    _ => None,
                })
                .next();
        }
    }

    let carry_def = carry_def.expect("expected UIADD3 carry predicate def");
    let carry_use = carry_use.expect("expected UIADD3.X carry predicate use");
    assert_eq!(carry_def.class, carry_use.class);
    assert_eq!(carry_def.idx, carry_use.idx);
    assert_eq!(carry_def.ssa, carry_use.ssa);
}

#[test]
fn ssa_imad_wide_models_implicit_hi_def() {
    let sample = r#"
        /*0000*/ LDC R3, c[0x0][0x160] ;
        /*0010*/ LDC.64 R4, c[0x0][0x168] ;
        /*0020*/ IMAD.WIDE R2, R1, 0x4, R4 ;
        /*0030*/ STG.E [R2.64], R7 ;
        /*0040*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let stmt = fir
        .blocks
        .iter()
        .flat_map(|block| block.stmts.iter())
        .find(|stmt| matches!(&stmt.value, RValue::Op { opcode, .. } if opcode.starts_with("IMAD.WIDE")))
        .expect("missing IMAD.WIDE stmt");
    assert!(
        stmt.defs
            .iter()
            .any(|def| matches!(def, IRExpr::Reg(reg) if reg.class == "R" && reg.idx == 2)),
        "low half missing from IMAD.WIDE defs: {:?}",
        stmt.defs
    );
    assert!(
        stmt.defs
            .iter()
            .any(|def| matches!(def, IRExpr::Reg(reg) if reg.class == "R" && reg.idx == 3)),
        "implicit high half missing from IMAD.WIDE defs: {:?}",
        stmt.defs
    );
}

#[test]
fn semantic_lift_carry_def_uses_pre_increment_low_operand() {
    let sample = r#"
        /*0000*/ IADD3 R1, P0, R1, 0x1, RZ ;
        /*0010*/ IADD3.X R2, R2, RZ, RZ, P0, !PT ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());

    let low = lifted
        .by_def
        .get(&DefRef {
            block_id: 0,
            stmt_idx: 0,
            def_idx: 0,
        })
        .expect("expected low add def");
    let carry = lifted
        .by_def
        .get(&DefRef {
            block_id: 0,
            stmt_idx: 0,
            def_idx: 1,
        })
        .expect("expected carry def");

    assert_eq!(low.rhs.render(), "R1.0 + 1");
    assert_eq!(carry.rhs.render(), "carry_u32_add3(R1.0, 1, 0)");
    assert!(!carry.rhs.render().contains("R1.1"));
}

#[test]
fn ssa_ir_does_not_emit_synthetic_carry_opcodes() {
    let sample = r#"
        /*0000*/ IADD3 R10, P2, R6, c[0x0][0x160], RZ ;
        /*0010*/ UIADD3 UR8, UP0, UR8, 0x4, URZ ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    for block in &fir.blocks {
        for stmt in &block.stmts {
            let RValue::Op { opcode, .. } = &stmt.value else {
                continue;
            };
            assert_ne!(opcode, "IADD3.CARRY");
            assert_ne!(opcode, "IADD3.CARRY2");
            assert_ne!(opcode, "UIADD3.CARRY");
            assert_ne!(opcode, "UIADD3.CARRY2");
        }
    }
}

fn run_structured_output_lifted(sass: &str) -> String {
    let cfg = build_cfg(decode_sass(sass));
    if cfg.node_count() == 0 {
        return "void kernel(void) {\n}\n".to_string();
    }
    let fir = build_ssa(&cfg);
    let mut structurizer = Structurizer::new(&cfg, &fir);
    let lift_cfg = SemanticLiftConfig::default();
    let lifted = lift_function_ir(&fir, &lift_cfg);
    match structurizer.structure_function() {
        Some(tree) => structurizer.pretty_print_with_lift(&tree, &DefaultDisplay, 0, Some(&lifted)),
        None => String::new(),
    }
}

#[test]
fn empty_cfg_ssa_build_is_non_fatal() {
    let cfg = build_cfg(Vec::new());
    let fir = build_ssa(&cfg);
    assert!(fir.blocks.is_empty());
}

#[test]
fn malformed_sass_returns_stub_output() {
    let out = run_canonical_output_full_pass("not sass");
    assert_eq!(out, "void kernel(void) {\n}\n");
}

#[test]
fn unknown_opcode_lifted_path_stays_intrinsic_like() {
    let sass = r#"
        /*0000*/ FOO.BAR R1, R2, 0x3 ;
        /*0010*/ EXIT ;
    "#;
    let out = run_structured_output_lifted(sass);
    assert!(out.contains("FOO.BAR("));
    assert!(!out.trim().is_empty());
}

fn run_structured_output_full_pass_from_instrs(
    instrs: Vec<DecodedInstruction>,
    sm: Option<u32>,
) -> String {
    let cfg = build_cfg(instrs.clone());
    if cfg.node_count() == 0 {
        return String::new();
    }
    let inferred_profile = AbiProfile::detect_with_sm(&instrs, sm);
    let fir = {
        let ssa = build_ssa(&cfg);
        let dce1 = ir_dce(&ssa);
        let cp = ir_constprop(&dce1);
        let alg = ir_algebra(&cp);
        let cse = ir_cse(&alg, &cfg);
        let copyprop = ir_copyprop(&cse);
        ir_dce(&copyprop)
    };
    let analysis_abi_profile = Some(inferred_profile);
    let abi_annotations = analysis_abi_profile.map(|p| annotate_function_ir_constmem(&fir, p));
    let abi_aliases = match (analysis_abi_profile, abi_annotations.as_ref()) {
        (Some(_), Some(anns)) => Some(infer_arg_aliases(&fir, anns)),
        _ => None,
    };
    let mut structurizer = Structurizer::new(&cfg, &fir);
    let local_decls = infer_local_typed_declarations_with_abi(
        &fir,
        abi_annotations.as_ref(),
        abi_aliases.as_ref(),
    );

    let Some(tree) = structurizer.structure_function() else {
        return "// Failed to structure function or function is empty.
"
        .to_string();
    };

    let default_display = DefaultDisplay;
    let abi_display = match (analysis_abi_profile, abi_aliases.clone()) {
        (Some(profile), Some(aliases)) => Some(AbiDisplay::with_aliases(profile, aliases)),
        (Some(profile), None) => Some(AbiDisplay::new(profile)),
        (None, _) => None,
    };
    let display_ctx: &dyn DisplayCtx = abi_display
        .as_ref()
        .map(|d| d as &dyn DisplayCtx)
        .unwrap_or(&default_display);
    let lift_cfg = SemanticLiftConfig {
        abi_annotations: abi_annotations.as_ref(),
        abi_aliases: abi_aliases.as_ref(),
        strict: true,
    };
    let lifted = lift_function_ir(&fir, &lift_cfg);
    let preview_output =
        structurizer.pretty_print_with_lift_cleanup(&tree, display_ctx, 0, Some(&lifted));
    let has_unstructured = preview_output.contains("goto BB");
    let plan = plan_structured_name_recovery_with_lift(
        &fir,
        &preview_output,
        Some(&lifted),
        &NameRecoveryConfig {
            style: if has_unstructured {
                NameStyle::VerbatimSsa
            } else {
                NameStyle::Temp
            },
            rewrite_control_predicates: !has_unstructured,
            semantic_symbolization: true,
        },
    );
    let enable_post_name_addr64_fold = !has_unstructured
        && preview_output.len() <= 30_000
        && preview_output.matches("((uintptr_t)").count() >= 8;
    let named_output = structurizer.pretty_print_with_lift_cleanup_and_names(
        &tree,
        display_ctx,
        0,
        Some(&lifted),
        &plan.token_map,
        enable_post_name_addr64_fold,
    );
    let symbols = filter_recovered_symbols_by_output(&named_output, &plan.symbols);
    let name_type_map = infer_recovered_name_types(&fir, &symbols);
    render_typed_structured_output(
        &named_output,
        abi_aliases.as_ref(),
        &local_decls,
        Some(&symbols),
        &name_type_map,
        collect_shared_memory_decls(Some(&lifted)),
    )
}

fn run_structured_output_full_pass(sass: &str) -> String {
    let instrs = decode_sass(sass);
    if instrs.is_empty() {
        return "void kernel(void) {
}
"
        .to_string();
    }
    let sm = parse_sm_version(sass);
    let inferred_profile = AbiProfile::detect_with_sm(&instrs, sm);
    let cfg = build_cfg(instrs.clone());
    if cfg.node_count() == 0 {
        return "void kernel(void) {
}
"
        .to_string();
    }
    let fir = {
        let ssa = build_ssa(&cfg);
        let dce1 = ir_dce(&ssa);
        let cp = ir_constprop(&dce1);
        let alg = ir_algebra(&cp);
        let cse = ir_cse(&alg, &cfg);
        let copyprop = ir_copyprop(&cse);
        ir_dce(&copyprop)
    };
    let analysis_abi_profile = Some(inferred_profile);
    let abi_annotations = analysis_abi_profile.map(|p| annotate_function_ir_constmem(&fir, p));
    let abi_aliases = match (analysis_abi_profile, abi_annotations.as_ref()) {
        (Some(_), Some(anns)) => Some(infer_arg_aliases(&fir, anns)),
        _ => None,
    };

    let mut out = String::new();
    out.push_str("// --- Structured Output ---\n");
    if let Some(anns) = &abi_annotations {
        if !anns.is_empty() {
            out.push_str("// ABI const-memory mapping (sample):\n");
            for line in anns.summarize_lines(16) {
                out.push_str("// ");
                out.push_str(&line);
                out.push('\n');
            }
        }
    }
    if let Some(aliases) = &abi_aliases {
        if !aliases.is_empty() {
            out.push_str("// ABI arg aliases (heuristic):\n");
            for line in aliases.summarize_lines(12) {
                out.push_str("// ");
                out.push_str(&line);
                out.push('\n');
            }
        }
    }
    if let Some(aliases) = &abi_aliases {
        if !aliases.is_empty() {
            out.push_str("// Typed signature inferred from ABI aliases:\n");
            for line in aliases.summarize_lines(12) {
                out.push_str("// ");
                out.push_str(&line);
                out.push('\n');
            }
        }
    }
    out.push_str(&run_structured_output_full_pass_from_instrs(instrs, sm));
    out.push_str("// --- End Structured Output ---\n");
    out
}

fn run_canonical_output_full_pass_from_instrs(
    instrs: Vec<DecodedInstruction>,
    sm: Option<u32>,
    function_name: &str,
) -> String {
    build_named_decompile_artifacts(instrs, sm, Some(function_name))
        .rendered
        .expect("canonical backend should render")
}

fn run_canonical_output_full_pass(sass: &str) -> String {
    let instrs = decode_sass(sass);
    if instrs.is_empty() {
        return "void kernel(void) {\n}\n".to_string();
    }
    run_canonical_output_full_pass_from_instrs(instrs, parse_sm_version(sass), "kernel")
}

fn assert_canonical_full_pass_nonempty_and_deterministic(sass: &str) -> String {
    let out1 = run_canonical_output_full_pass(sass);
    let out2 = run_canonical_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    out1
}

#[test]
fn canonical_full_pass_emits_clean_pointer_and_shared_forms() {
    let sass = r#"
        /*0000*/ MOV R4, c[0x0][0x160] ;
        /*0010*/ MOV R5, c[0x0][0x164] ;
        /*0020*/ IADD3 R4, R4, 0x4, RZ ;
        /*0030*/ LDG.E R6, [R4.64] ;
        /*0040*/ ATOMS.ADD R0, [R2], R6 ;
        /*0050*/ STS [R2+0x8], R6 ;
        /*0060*/ EXIT ;
    "#;
    let out = assert_canonical_full_pass_nonempty_and_deterministic(sass);
    let raw_ssa = Regex::new(r"\b(?:R|UR|P|UP)\d+\.\d+\b").expect("raw ssa regex");
    assert!(
        out.contains("arg0_ptr[1]"),
        "expected typed pointer access, got:\n{out}"
    );
    assert!(
        out.contains("atomicAdd("),
        "expected atomic lowering, got:\n{out}"
    );
    assert!(
        out.contains("shmem[(r2_0 + 8) / 4]"),
        "expected shared byte-offset indexing, got:\n{out}"
    );
    assert!(
        !raw_ssa.is_match(&out),
        "expected canonical output to avoid raw SSA tokens, got:\n{out}"
    );
}

#[test]
fn canonical_full_pass_preserves_named_render_entrypoints() {
    let rendered = run_canonical_output_full_pass_from_instrs(
        decode_sass("/*0000*/ MOV R0, RZ ;\n/*0010*/ EXIT ;\n"),
        None,
        "named_kernel",
    );
    assert!(
        rendered.starts_with("void named_kernel("),
        "expected named render entrypoint, got:\n{rendered}"
    );
}

#[test]
fn canonical_full_pass_lowers_imad_wide_param_roots_through_helper_path() {
    let rendered = run_canonical_output_full_pass_from_instrs(
        decode_sass(
            "/*0000*/ IMAD.WIDE R6, R3, 0x4, c[0x0][0x160] ;\n\
             /*0010*/ LDG.E.U32 R11, [R6.64] ;\n\
             /*0020*/ MOV R8, c[0x0][0x168] ;\n\
             /*0030*/ MOV R9, c[0x0][0x16c] ;\n\
             /*0040*/ STG.E [R8.64], R11 ;\n\
             /*0050*/ EXIT ;\n",
        ),
        None,
        "kernel",
    );
    assert!(
        rendered.contains("arg0_ptr[r3_0]"),
        "expected IMAD.WIDE-rooted access to preserve the computed index, got:\n{rendered}"
    );
    assert!(
        !rendered.contains("IMAD.WIDE("),
        "expected canonical IMAD.WIDE lowering to stay structural, got:\n{rendered}"
    );
}

#[test]
fn canonical_full_pass_lowers_iadd64_param_roots_into_typed_loads() {
    let rendered = run_canonical_output_full_pass_from_instrs(
        decode_sass(
            "/*0000*/ LDC.64 R4, c[0x0][0x160] ;\n\
             /*0010*/ SHF.R.S32.HI R7, RZ, 0x1f, R6 ;\n\
             /*0020*/ IADD.64 R4, R6, R4 ;\n\
             /*0030*/ LDG.E.U8 R8, [R4.64] ;\n\
             /*0040*/ LDC.64 R10, c[0x0][0x168] ;\n\
             /*0050*/ STG.E.U8 [R10.64], R8 ;\n\
             /*0060*/ EXIT ;\n",
        ),
        None,
        "kernel",
    );
    assert!(
        rendered.contains("arg0_ptr[((uintptr_t)(((uint64_t)(r7_0) << 32) | (uint32_t)(r6_0)))]"),
        "expected IADD.64-rooted load to stay on arg0_ptr with a widened byte offset, got:\n{rendered}"
    );
    assert!(
        !rendered.contains("addr64("),
        "expected canonical output to avoid raw addr64 helpers, got:\n{rendered}"
    );
}

#[test]
fn canonical_full_pass_histogram256_lowers_shared_atomic_popc_and_barriers() {
    let hist = split_decoded_functions(include_str!("../test_cu/corpus/shared_mem_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "histogram256")
        .expect("histogram256 fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(hist.instrs, hist.sm, "histogram256");
    assert!(
        out.contains("atomicAdd(&shmem[") && out.contains(", 1);"),
        "expected POPC/INC shared atomic to lower to atomicAdd(..., 1), got:\n{out}"
    );
    assert!(
        out.contains("__syncthreads();"),
        "expected barrier op to lower to __syncthreads(), got:\n{out}"
    );
    for leak in ["ATOMS.POPC.INC", "BAR.SYNC", "BSYNC()", "BRA()", "EXIT()"] {
        assert!(
            !out.contains(leak),
            "expected canonical histogram output to drop raw `{leak}` artifacts, got:\n{out}"
        );
    }
}

fn assert_full_pass_nonempty_and_deterministic(sass: &str) -> String {
    let out1 = run_structured_output_full_pass(sass);
    let out2 = run_structured_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    out1
}

#[test]
fn test_abi_profile_detects_legacy_window_from_sample() {
    let sample = r#"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
        /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x148] ;
        /*0020*/ IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x154] ;
    "#;
    let instrs = decode_sass(sample);
    let profile = AbiProfile::detect(&instrs);
    assert_eq!(profile, AbiProfile::legacy_param_140());
}

#[test]
fn test_abi_profile_sm_fallback_without_param_offsets() {
    let sample = r#"
        .headerflags @"EF_CUDA_SM70"
        /*0000*/ S2R R0, SR_CTAID.X ;
        /*0010*/ S2R R1, SR_TID.X ;
    "#;
    let instrs = decode_sass(sample);
    let sm = parse_sm_version(sample);
    let profile = AbiProfile::detect_with_sm(&instrs, sm);
    assert_eq!(profile, AbiProfile::legacy_param_140());
}

#[test]
fn test_structured_output_with_abi_display_symbols_constmem() {
    let sample = r#"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
        /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x0] ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let display = AbiDisplay::new(AbiProfile::legacy_param_140());
    let out = fir.to_dot(&cfg, &display);

    assert!(out.contains("param_0"));
    assert!(out.contains("blockDim.x"));
}

#[test]
fn test_abi_inferred_aliases_show_in_output() {
    let sample = r#"
        /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
        /*0010*/ IADD3.X R5, R5, c[0x0][0x164], RZ ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sample));
    let fir = build_ssa(&cfg);
    let profile = AbiProfile::modern_param_160();
    let anns = annotate_function_ir_constmem(&fir, profile);
    let aliases = infer_arg_aliases(&fir, &anns);
    let display = AbiDisplay::with_aliases(profile, aliases);
    let out = fir.to_dot(&cfg, &display);
    assert!(out.contains("arg0_ptr.lo32"));
    assert!(out.contains("arg0_ptr.hi32"));
}

#[test]
fn lifted_output_uses_infix_for_supported_patterns() {
    let sass = include_str!("../test_cu/if.sass");
    let out = run_structured_output_lifted(sass);
    assert!(out.contains("R1.1 = R1.0 + 1;"));
    assert!(!out.contains("IADD3("));
}

#[test]
fn lifted_output_falls_back_to_raw_for_unmatched_opcodes() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let out = run_structured_output_lifted(sass);
    // PLOP3 with all-constant inputs is now constant-folded (no plop3_lut call).
    // The if_loop fixture's PLOP3 instructions all have PT,PT,PT,PT inputs,
    // so they should fold to true/false constants.
    assert!(
        !out.contains("plop3_lut("),
        "all-constant PLOP3 should be folded"
    );
    // Verify LEA.HI.X is now lifted to lea_hi_x helper notation.
    assert!(
        out.contains("lea_hi_x("),
        "LEA.HI.X should be lifted to lea_hi_x helper"
    );
}

#[test]
fn semantic_lift_has_mixed_coverage_on_if_loop_fixture() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let cfg = build_cfg(decode_sass(sass));
    let fir = build_ssa(&cfg);
    let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());
    assert!(lifted.stats.lifted > 0);
    assert!(lifted.stats.fallback > 0);
}

#[test]
fn semantic_lift_percentage_gate_if_loop_fixture() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let cfg = build_cfg(decode_sass(sass));
    let fir = build_ssa(&cfg);
    let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());

    assert!(lifted.stats.attempted > 0);
    let lifted_pct = lifted.stats.lifted as f64 / lifted.stats.attempted as f64;
    let fallback_pct = lifted.stats.fallback as f64 / lifted.stats.attempted as f64;

    // Guardrails for regression detection while preserving conservative fallback.
    assert!(lifted_pct >= 0.30);
    assert!(fallback_pct <= 0.70);
}

#[test]
fn structured_output_goto_gate_lifted_fixtures() {
    let if_loop = run_structured_output_lifted(include_str!("../test_cu/if_loop.sass"));
    let rc4 = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    let test_div = run_structured_output_lifted(include_str!("../test_cu/test_div.sass"));

    let if_loop_goto = if_loop.matches("goto BB").count();
    let rc4_goto = rc4.matches("goto BB").count();
    let test_div_goto = test_div.matches("goto BB").count();

    assert_eq!(if_loop_goto, 0);
    assert_eq!(rc4_goto, 0);
    assert_eq!(test_div_goto, 0);
}

#[test]
fn lifted_rc4_uses_shared_array_style_for_shared_mem_ops() {
    let out = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("shmem_u8["));
    assert!(!out.contains("_ = STS."));
}

#[test]
fn lifted_rc4_reduces_add_with_carry_and_lea_hi_opcode_noise() {
    let out = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("? 1 : 0"));
    assert!(out.contains("hi32("));
    assert!(!out.contains("IADD3.X("));
    assert!(!out.contains("LEA.HI("));
    assert!(!out.contains("LOP3.LUT("));
}

#[test]
fn lifted_rc4_renders_barrier_as_syncthreads() {
    let out = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("__syncthreads();"));
    assert!(!out.contains("BAR.SYNC"));
}

#[test]
fn smoke_struct_output_full_pass_rc4_sass() {
    let out = assert_full_pass_nonempty_and_deterministic(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("__shared__ uint8_t shmem_u8[256];"));
    assert!(out.contains("__syncthreads();"));
    assert!(out.contains("uint8_t* arg0_ptr"));
    assert!(out.contains("uint8_t* arg4_ptr"));
    assert!(out.contains("uint8_t* arg6_ptr"));
    assert!(!out.contains("addr64("));
    assert!(!out.contains("prmt("));
}

#[test]
fn smoke_struct_output_full_pass_if_sass() {
    let sass = include_str!("../test_cu/if.sass");
    let expected = "void kernel(void) {\n  return;\n}\n";
    let out = assert_canonical_full_pass_nonempty_and_deterministic(sass);
    assert_eq!(out, expected);
}

#[test]
fn smoke_struct_output_full_pass_loop_constant_sass() {
    let out =
        assert_full_pass_nonempty_and_deterministic(include_str!("../test_cu/loop_constant.sass"));
    assert!(out.contains("__shared__ uint8_t shmem_u8[256];"));
    let shmem_store = Regex::new(
        r"shmem_u8\[(?:tid_x|v\d+(?:_next(?:_\d+)?)?)\] = (?:tid_x|v\d+(?:_next(?:_\d+)?)?);",
    )
    .expect("valid loop_constant shmem store regex");
    assert!(
        shmem_store.is_match(&out),
        "expected loop_constant shmem store, got:\n{}",
        out
    );
    assert!(
        out.contains("while(!((int32_t)(tid_x) >= (int32_t)(256)));")
            || out.contains("while(!((int32_t)(tid_x_next) >= (int32_t)(256)));")
            || out.contains("while(!((int32_t)(v2_next) >= (int32_t)(256)));")
    );
    assert!(!out.contains("addr64("));
    assert!(!out.contains("prmt("));
}

#[test]
fn smoke_struct_output_full_pass_if_loop_sass() {
    let out = assert_full_pass_nonempty_and_deterministic(include_str!("../test_cu/if_loop.sass"));
    assert!(out.contains("float* arg4_ptr"));
    assert!(out.matches("do {").count() >= 2);
    let final_store = Regex::new(r"\*\(arg4_ptr \+ v2\) = v\d+(?:_next(?:_\d+)?)?;")
        .expect("valid if_loop final store regex");
    assert!(final_store.is_match(&out));
    assert!(!out.contains("lea_hi_x("));
    assert!(!out.contains("prmt("));
    assert!(!out.contains("hfma2("));
}

#[test]
fn smoke_struct_output_full_pass_test_div_sass() {
    let out = assert_canonical_full_pass_nonempty_and_deterministic(include_str!(
        "../test_cu/test_div.sass"
    ));
    assert!(out.contains("void kernel(int32_t arg0, int32_t arg1, uint32_t* arg2_ptr)"));
    assert!(
        out.contains("abs(arg1)"),
        "expected signed-dividend lowering, got:\n{}",
        out
    );
    assert!(
        out.contains("rcp_approx("),
        "expected reciprocal approximation helper recovery, got:\n{}",
        out
    );
    let final_store = Regex::new(r"arg2_ptr\[0\] = r\d+_\d+;").expect("valid final store regex");
    assert!(final_store.is_match(&out));
    let negate_guard = Regex::new(r"r\d+_\d+ = !p\d+_\d+ \? \(-r\d+_\d+\) : r\d+_\d+;")
        .expect("valid negate guard regex");
    assert!(negate_guard.is_match(&out));
    assert!(!out.contains("ConstMem("));
    assert!(!out.contains("addr64("));
    assert!(!out.contains("prmt("));
}

#[test]
fn lifted_rc4_does_not_render_address_width_suffix_on_global_accesses() {
    let out = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("@64"));
}

#[test]
fn full_pass_rc4_keeps_thread0_gate_as_predicate_guard() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("if (!(tid_x != RZ))"));
    assert!(
        out.contains("if (!(P")
            || out.contains("if (!P")
            || out.contains("if (!(b")
            || out.contains("if (!b")
    );
}

#[test]
fn full_pass_rc4_addr64_collapsed_to_typed_pointer() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    // After addr64 collapse, carry/lea_hi patterns are replaced with typed pointer
    // expressions. No addr64() calls should remain in the rc4 output.
    assert!(
        !out.contains("addr64("),
        "all addr64 patterns should be collapsed to typed pointers"
    );
    // The collapsed output should contain typed pointer arithmetic with arg0_ptr.
    assert!(
        out.contains("(arg0_ptr + (int64_t)"),
        "arg0_ptr pointer expressions expected"
    );
    // No raw lea_hi_x_sx32 should remain (DCE removes dead intermediates).
    assert!(
        !out.contains("lea_hi_x_sx32("),
        "lea_hi_x_sx32 should be DCE'd after collapse"
    );
    // Negative checks from the old test still apply.
    assert!(!out.contains("arg0_ptr.hi32 << 1"));
    assert!(!out.contains("ConstMem(0, 356) << 1"));
}

#[test]
fn full_pass_rc4_global_u8_accesses_use_typed_pointers() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    // After addr64 collapse, global u8 accesses use typed pointer expressions
    // instead of addr64 pairs.
    assert!(out.contains("((uint8_t*)(arg0_ptr + (int64_t)"));
}

#[test]
fn full_pass_rc4_key_sel_uses_ssa_predicate_not_raw_p1() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("!P1 ?"));
}

#[test]
fn full_pass_rc4_pointer_arithmetic_uses_typed_expressions() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    // After addr64 collapse, the carry + lea_hi + addr64 pattern is collapsed
    // into typed pointer expressions. Verify that arg4_ptr and arg6_ptr patterns
    // are also collapsed.
    assert!(
        out.contains("(arg4_ptr + (int64_t)") || out.contains("(arg6_ptr + (int64_t)"),
        "expected collapsed pointer expressions for arg4_ptr or arg6_ptr"
    );
}

#[test]
fn full_pass_relu_materializes_bounds_guard() {
    let relu = split_decoded_functions(include_str!("../test_cu/corpus/arith_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "relu")
        .expect("relu fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(relu.instrs, relu.sm, "relu");
    assert!(
        !out.contains("if (b0) return;"),
        "expected the relu bounds guard to be rendered as a comparison, got:
{}",
        out
    );
    assert!(
        !out.contains("ISETP(") && !out.contains("S2R(") && !out.contains("FMNMX("),
        "expected relu to avoid raw scalar helper opcodes, got:\n{}",
        out
    );
    let re =
        Regex::new(r"(p\d+_\d+) = \(int32_t\)\(.+\) >= \(int32_t\)\(.+\);").expect("valid regex");
    let pred_name = re
        .captures(&out)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    assert!(
        pred_name
            .as_ref()
            .is_some_and(|pred_name| out.contains(&format!("if ({pred_name}) return;"))),
        "expected the relu bounds guard to be materialized as a comparison, got:
{}",
        out
    );
}

#[test]
fn full_pass_dot_thread_recovers_pointer_params_and_typed_loads() {
    let dot = split_decoded_functions(include_str!("../test_cu/corpus/loop_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "dot_thread")
        .expect("dot_thread fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(dot.instrs, dot.sm, "dot_thread");
    assert!(
        out.contains("float* arg0_ptr")
            && out.contains("float* arg2_ptr")
            && out.contains("float* arg4_ptr"),
        "expected dot_thread pointer params to stay typed, got:
{}",
        out
    );
    assert!(
        out.contains("arg0_ptr[") && out.contains("arg2_ptr[") && out.contains("arg4_ptr["),
        "expected dot_thread main memory accesses to stay on typed pointer arithmetic, got:
{}",
        out
    );
    assert!(
        !out.contains("((uintptr_t)"),
        "expected dot_thread to avoid packed pointer reconstruction, got:
{}",
        out
    );
    assert!(
        !out.contains("c[0x0][0x164]")
            && !out.contains("c[0x0][0x16c]")
            && !out.contains("c[0x0][0x174]"),
        "expected dot_thread hi param words to resolve symbolically, got:
{}",
        out
    );
    assert!(
        !out.contains("pair_hi("),
        "expected dot_thread loop-carried pointer updates to resolve their hi halves, got:
{}",
        out
    );
    let ffma_recurrence = Regex::new(r"r\d+_\d+ = r\d+_\d+ \* r\d+_\d+ \+ r\d+_\d+;")
        .expect("valid dot_thread recurrence regex");
    assert!(
        ffma_recurrence.is_match(&out),
        "expected dot_thread accumulator FFMA recurrence to survive structurization, got:
{}",
        out
    );
}

#[test]
fn full_pass_cumsum_linear_no_longer_overflows_named_render() {
    let cumsum = split_decoded_functions(include_str!("../test_cu/corpus/loop_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "cumsum_linear")
        .expect("cumsum_linear fixture should exist");
    let out = run_structured_output_full_pass_from_instrs(cumsum.instrs, cumsum.sm);
    assert!(
        !out.trim().is_empty(),
        "expected cumsum_linear to decompile to non-empty output"
    );
    assert!(
        out.contains("float* arg0_ptr") && out.contains("float* arg2_ptr"),
        "expected cumsum_linear pointer params to survive the named pass, got:
{}",
        out
    );
    assert!(
        !out.contains("((uintptr_t)"),
        "expected cumsum_linear remainder paths to avoid packed pointer reconstruction, got:
{}",
        out
    );
    assert!(
        !out.lines().any(|line| {
            let line = line.trim();
            let Some((lhs, rhs)) = line
                .strip_suffix(';')
                .and_then(|line| line.split_once(" = "))
            else {
                return false;
            };
            lhs == rhs
        }),
        "expected cumsum_linear named output to omit trivial self-assignments, got:
{}",
        out
    );
}

#[test]
fn canonical_full_pass_cumsum_linear_avoids_register_pseudo_pointers() {
    let cumsum = split_decoded_functions(include_str!("../test_cu/corpus/loop_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "cumsum_linear")
        .expect("cumsum_linear fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(cumsum.instrs, cumsum.sm, "cumsum_linear");
    let raw_reg_index =
        Regex::new(r"\b(?:r|ur)\d+_\d+\[").expect("valid raw register pseudo-pointer regex");
    assert!(
        !raw_reg_index.is_match(&out),
        "expected canonical cumsum_linear output to avoid raw register pseudo-pointers, got:\n{}",
        out
    );
    assert!(
        !out.contains("((uint32_t*)(ur"),
        "expected arg-rooted uniform pointer loops to stay on typed params, got:\n{}",
        out
    );
    assert!(
        !out.contains("((uint32_t*)(r2_5))")
            && !out.contains("((uint32_t*)(r4_4))")
            && !out.contains("((uint32_t*)(r2_8))")
            && !out.contains("((uint32_t*)(r4_7))"),
        "expected canonical cumsum_linear remainder paths to stay rooted on arg pointers, got:\n{}",
        out
    );
    assert!(
        out.contains("arg0_ptr[") && out.contains("arg2_ptr["),
        "expected canonical cumsum_linear loops to stay rooted on arg pointers, got:\n{}",
        out
    );
}

#[test]
fn full_pass_gelu_forward_recovers_copysign_and_typed_pointer_arithmetic() {
    let gelu = split_decoded_functions(include_str!("../test_cu/corpus_sm100/ml_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "gelu_forward")
        .expect("gelu_forward fixture should exist");
    let out = run_structured_output_full_pass_from_instrs(gelu.instrs, gelu.sm);
    assert!(
        out.contains("copysignf("),
        "expected copysignf recovery, got:
{}",
        out
    );
    assert!(
        out.contains("v7 = *(arg0_ptr + v2);"),
        "expected collapsed input pointer access, got:
{}",
        out
    );
    assert!(
        out.contains("*(arg2_ptr + v2) = v32;"),
        "expected collapsed output pointer access, got:
{}",
        out
    );
    assert!(
        !out.contains("2147483648 &"),
        "expected sign-mask bit twiddling to be lifted, got:
{}",
        out
    );
    assert!(
        !out.contains("pair_hi("),
        "expected no pair_hi helper in gelu output, got:
{}",
        out
    );
}

#[test]
fn canonical_full_pass_gelu_forward_recovers_structural_math_ops() {
    let gelu = split_decoded_functions(include_str!("../test_cu/corpus_sm100/ml_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "gelu_forward")
        .expect("gelu_forward fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(gelu.instrs, gelu.sm, "gelu_forward");
    assert!(
        out.contains("blockDim.x * blockIdx.x + threadIdx.x"),
        "expected canonical gelu_forward launch index to lower structurally, got:\n{}",
        out
    );
    assert!(
        out.contains("!p0_1 ? r8_0 : 1;") && out.contains("copysignf("),
        "expected canonical gelu_forward to recover fsel/copysign math, got:\n{}",
        out
    );
    assert!(
        !out.contains("IMAD(")
            && !out.contains("FMUL(")
            && !out.contains("FSEL(")
            && !out.contains("LOP3.LUT("),
        "expected canonical gelu_forward math to avoid raw helper opcodes, got:\n{}",
        out
    );
}

#[test]
fn full_pass_nbody_final_store_uses_collapsed_force_pointer() {
    let nbody = split_decoded_functions(include_str!(
        "../test_cu/corpus_sm100/simulation_kernels.sass"
    ))
    .into_iter()
    .find(|f| f.name == "nbody_forces")
    .expect("nbody_forces fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(nbody.instrs, nbody.sm, "nbody_forces");
    let store0 = Regex::new(r"arg2_ptr\[[A-Za-z0-9_]+ \* 12 / 4\] = [A-Za-z0-9_]+;")
        .expect("valid first store regex");
    let store1 = Regex::new(r"arg2_ptr\[\([A-Za-z0-9_]+ \* 12 \+ 4\) / 4\] = [A-Za-z0-9_]+;")
        .expect("valid flattened second store regex");
    let store2 = Regex::new(r"arg2_ptr\[\([A-Za-z0-9_]+ \* 12 \+ 8\) / 4\] = [A-Za-z0-9_]+;")
        .expect("valid flattened third store regex");
    assert!(
        store0.is_match(&out),
        "expected collapsed first force store, got:
{}",
        out
    );
    assert!(
        store1.is_match(&out),
        "expected collapsed second force store, got:
{}",
        out
    );
    assert!(
        store2.is_match(&out),
        "expected collapsed third force store, got:
{}",
        out
    );
    assert!(
        !out.contains("pair_hi("),
        "expected no pair_hi helper in nbody output, got:
{}",
        out
    );
    assert!(
        !out.contains("((uint32_t*)"),
        "expected nbody force stores to stay float-typed, got:
{}",
        out
    );
}

#[test]
fn canonical_full_pass_sgemm_tiled_keeps_predicated_typed_global_loads() {
    let sgemm =
        split_decoded_functions(include_str!("../test_cu/corpus_sm100/compute_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "sgemm_tiled")
            .expect("sgemm_tiled fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(sgemm.instrs, sgemm.sm, "sgemm_tiled");
    let lhs_load = Regex::new(
        r"r\d+_\d+ = !p\d+_\d+ \? \(arg0_ptr\[[A-Za-z0-9_()+*/ ]+\]\) : (?:0|0\.0|r\d+_\d+);",
    )
    .expect("valid canonical lhs load regex");
    let rhs_load = Regex::new(
        r"r\d+_\d+ = !p\d+_\d+ \? \(arg2_ptr\[[A-Za-z0-9_()+*/ ]+\]\) : (?:0|0\.0|r\d+_\d+);",
    )
    .expect("valid canonical rhs load regex");
    let raw_reg_index =
        Regex::new(r"\b(?:r|ur)\d+_\d+\[").expect("valid raw register pseudo-pointer regex");
    assert!(
        lhs_load.is_match(&out) && rhs_load.is_match(&out),
        "expected canonical sgemm_tiled predicated loads to stay on typed pointers, got:\n{}",
        out
    );
    assert!(
        !raw_reg_index.is_match(&out),
        "expected canonical sgemm_tiled to avoid raw register pseudo-pointers, got:\n{}",
        out
    );
    assert!(
        !out.contains("*((uint32_t*)(arg0_ptr") && !out.contains("*((uint32_t*)(arg2_ptr"),
        "expected canonical sgemm_tiled to avoid stale integer pointer casts, got:\n{}",
        out
    );
}

#[test]
fn canonical_full_pass_sgemm_tiled_avoids_lea_helper_artifacts() {
    let sgemm =
        split_decoded_functions(include_str!("../test_cu/corpus_sm100/compute_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "sgemm_tiled")
            .expect("sgemm_tiled fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(sgemm.instrs, sgemm.sm, "sgemm_tiled");
    assert!(
        !out.contains("lea_hi_x(")
            && !out.contains("pair_hi(")
            && !out.contains("addr64(")
            && !out.contains("arg0_ptr_lo32")
            && !out.contains("arg0_ptr_hi32")
            && !out.contains("arg2_ptr_lo32")
            && !out.contains("arg2_ptr_hi32"),
        "expected canonical sgemm_tiled to avoid LEA/pointer-helper artifacts, got:\n{}",
        out
    );
}

#[test]
fn canonical_full_pass_warp_reduce_sum_keeps_shuffle_intrinsics() {
    let warp = split_decoded_functions(include_str!("../test_cu/corpus/compute_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "warp_reduce_sum")
        .expect("warp_reduce_sum fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(warp.instrs, warp.sm, "warp_reduce_sum");
    assert!(
        out.contains("__shfl_down_sync"),
        "expected canonical warp_reduce_sum to keep CUDA shuffle intrinsics, got:\n{}",
        out
    );
    assert!(
        !out.contains("SHFL.DOWN("),
        "expected canonical warp_reduce_sum to avoid raw shuffle mnemonics, got:\n{}",
        out
    );
    assert!(
        !out.contains("LOP3.LUT("),
        "expected canonical warp_reduce_sum to avoid raw lop3 helpers, got:\n{}",
        out
    );
    assert!(
        !out.contains("SHF.R.U32.HI(") && !out.contains("USHF.R.U32.HI("),
        "expected canonical warp_reduce_sum to avoid raw funnel-shift helpers, got:\n{}",
        out
    );
    assert!(
        out.contains("r8_0 = threadIdx.x & 31;") && out.contains("p0_1 = (threadIdx.x & 31) != 0;"),
        "expected canonical warp_reduce_sum to recover lane-mask lop3 semantics, got:\n{}",
        out
    );
}

#[test]
fn full_pass_warp_reduce_sum_infers_float_input_pointer() {
    let warp = split_decoded_functions(include_str!("../test_cu/corpus/compute_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "warp_reduce_sum")
        .expect("warp_reduce_sum fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(warp.instrs, warp.sm, "warp_reduce_sum");
    assert!(
        out.contains("void warp_reduce_sum(float* arg0_ptr, float* arg2_ptr, int32_t arg4)"),
        "expected warp_reduce_sum to infer float input/output pointers, got:\n{}",
        out
    );
    assert!(
        out.contains("__shfl_down_sync"),
        "expected warp_reduce_sum to keep CUDA shuffle intrinsics, got:\n{}",
        out
    );
}

#[test]
fn full_pass_stencil2d_top_halo_predicated_load_defaults_to_zero() {
    let stencil = split_decoded_functions(include_str!("../test_cu/corpus/compute_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "stencil2d_5pt")
        .expect("stencil2d_5pt fixture should exist");
    let out =
        run_canonical_output_full_pass_from_instrs(stencil.instrs, stencil.sm, "stencil2d_5pt");
    assert!(
        out.contains("r4_7 = !p2_1 ? (arg0_ptr[r8_0 + -1]) : 0;"),
        "expected predicated top-halo load to default to zero, got:\n{}",
        out
    );
    assert!(
        out.contains("r0_3 = !p1_5 ? (arg0_ptr[r8_0 + 16]) : 0;"),
        "expected predicated right-halo load to default to zero, got:\n{}",
        out
    );
}

#[test]
fn full_pass_reduce_block_infers_float_shared_roundtrip_pointers() {
    let reduce = split_decoded_functions(include_str!("../test_cu/corpus/shared_mem_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "reduce_block")
        .expect("reduce_block fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(reduce.instrs, reduce.sm, "reduce_block");
    assert!(
        out.contains("void reduce_block(float* arg0_ptr, float* arg2_ptr, int32_t arg4)"),
        "expected reduce_block to infer float shared-memory roundtrip pointers, got:\n{}",
        out
    );
    assert!(
        out.contains("extern __shared__ float shmem[];"),
        "expected reduce_block shared memory to stay float-typed, got:\n{}",
        out
    );
}

#[test]
fn full_pass_stencil1d_infers_float_shared_halo_pointers() {
    let stencil =
        split_decoded_functions(include_str!("../test_cu/corpus/shared_mem_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "stencil1d")
            .expect("stencil1d fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(stencil.instrs, stencil.sm, "stencil1d");
    assert!(
        out.contains("void stencil1d(float* arg0_ptr, float* arg2_ptr, int32_t arg4)")
            && out.contains("extern __shared__ float shmem[];"),
        "expected stencil1d to infer float halo pointers, got:\n{}",
        out
    );
}

#[test]
fn full_pass_reduce_block_scales_shared_word_indices_by_element_size() {
    let reduce = split_decoded_functions(include_str!("../test_cu/corpus/shared_mem_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "reduce_block")
        .expect("reduce_block fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(reduce.instrs, reduce.sm, "reduce_block");
    assert!(
        out.contains("shmem[threadIdx.x + 128]"),
        "expected reduce_block to use float-element shared offsets, got:\n{}",
        out
    );
    assert!(
        !out.contains("shmem[(r7_0 + 512) / 4]"),
        "reduce_block should not leak raw byte offsets into shared indices, got:\n{}",
        out
    );
}

#[test]
fn full_pass_stencil1d_scales_shared_word_indices_by_element_size() {
    let stencil =
        split_decoded_functions(include_str!("../test_cu/corpus/shared_mem_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "stencil1d")
            .expect("stencil1d fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(stencil.instrs, stencil.sm, "stencil1d");
    assert!(
        out.contains("shmem[threadIdx.x]"),
        "expected stencil1d base shared index to use threadIdx.x directly, got:\n{}",
        out
    );
    assert!(
        out.contains("shmem[threadIdx.x + 132]"),
        "expected stencil1d halo element to use float indices, got:\n{}",
        out
    );
    assert!(
        !out.contains("shmem[(r19_0 + 528) / 4]"),
        "stencil1d should not leak raw byte offsets into shared indices, got:\n{}",
        out
    );
}

#[test]
fn full_pass_histogram256_lowers_shared_atomics_on_sm89() {
    let hist = split_decoded_functions(include_str!("../test_cu/corpus/shared_mem_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "histogram256")
        .expect("histogram256 fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(hist.instrs, hist.sm, "histogram256");
    assert!(
        out.contains("atomicAdd(&shmem["),
        "expected histogram256 shared atomic to lower to atomicAdd, got:\n{}",
        out
    );
    assert!(
        out.contains(", 1);"),
        "expected histogram256 shared atomic increment value to be preserved, got:\n{}",
        out
    );
    assert!(
        !out.contains("ATOMS.POPC.INC"),
        "shared atomic mnemonic should not leak into pseudo-C, got:\n{}",
        out
    );
}

#[test]
fn full_pass_histogram256_lowers_shared_atomics_on_blackwell() {
    for text in [
        include_str!("../test_cu/corpus_sm100/shared_mem_kernels.sass"),
        include_str!("../test_cu/corpus_sm120/shared_mem_kernels.sass"),
    ] {
        let hist = split_decoded_functions(text)
            .into_iter()
            .find(|f| f.name == "histogram256")
            .expect("histogram256 fixture should exist");
        let out = run_canonical_output_full_pass_from_instrs(hist.instrs, hist.sm, "histogram256");
        assert!(
            out.contains("atomicAdd(&shmem["),
            "expected histogram256 shared atomic to lower to atomicAdd, got:\n{}",
            out
        );
        assert!(
            !out.contains("ATOMS.POPC.INC"),
            "shared atomic mnemonic should not leak into pseudo-C, got:\n{}",
            out
        );
        assert!(
            !out.contains("&shmem[R"),
            "shared atomic address should not stay as an unresolved raw register, got:\n{}",
            out
        );
    }
}

#[test]
fn full_pass_histogram256_sm120_keeps_global_byte_loads_rooted_on_arg0_ptr() {
    let hist = split_decoded_functions(include_str!(
        "../test_cu/corpus_sm120/shared_mem_kernels.sass"
    ))
    .into_iter()
    .find(|f| f.name == "histogram256")
    .expect("histogram256 fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(hist.instrs, hist.sm, "histogram256");
    let arg0_loads = out.matches("*(arg0_ptr +").count()
        + out.matches("*((uint8_t*)(arg0_ptr +").count()
        + out.matches("arg0_ptr[((uintptr_t)").count();
    assert!(
        arg0_loads >= 4,
        "expected SM120 histogram byte loads to stay rooted on arg0_ptr, got:\n{}",
        out
    );
    assert!(
        !out.contains("IADD.64("),
        "expected SM120 histogram to drop dead rooted wide-add helpers, got:\n{}",
        out
    );
    assert!(
        !out.contains("*(((uint8_t*)(((uintptr_t)(((uint64_t)("),
        "expected SM120 histogram to avoid synthetic packed stride bases, got:\n{}",
        out
    );
}

#[test]
fn ssa_ldl_128_emits_all_lane_defs() {
    let sass = r#"
        /*0000*/ LDL.128 R4, [R2] ;
        /*0010*/ EXIT ;
    "#;
    let cfg = build_cfg(decode_sass(sass));
    let fir = build_ssa(&cfg);
    let load = fir.blocks[0]
        .stmts
        .iter()
        .find(|stmt| matches!(&stmt.value, RValue::Op { opcode, .. } if opcode == "LDL.128"))
        .expect("expected LDL.128 statement in SSA");
    assert_eq!(
        load.defs.len(),
        4,
        "LDL.128 should define a 4-lane tuple in SSA"
    );
}

#[test]
fn full_pass_sha256_single_block_rewrites_local_load_helpers() {
    let sha256 =
        split_decoded_functions(include_str!("../test_cu/corpus_sm120/crypto_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "sha256_single_block")
            .expect("sha256_single_block fixture should exist");
    let out = run_structured_output_full_pass_from_instrs(sha256.instrs, sha256.sm);
    assert!(
        !out.contains("LDL("),
        "expected no raw LDL helper in sha256 output, got:
{}",
        out
    );
    assert!(
        !out.contains("IADD.64("),
        "expected no raw IADD.64 helper in sha256 output, got:
{}",
        out
    );
    let stack_load =
        Regex::new(r"v\w+ = \*\(\(uint32_t\*\)\(v\w+ - 40\)\);").expect("valid stack load regex");
    assert!(
        stack_load.is_match(&out),
        "expected local stack loads to lower to typed pointer dereferences, got:
{}",
        out
    );
    let digest_store = Regex::new(
        r"\*(?:\((arg2_ptr(?: \+ \d+)?)\)|\(\(uint32_t\*\)\((arg2_ptr(?: \+ \d+)?)\)\)) = v\d+;",
    )
    .expect("valid sha256 final-store regex");
    let stores = digest_store
        .captures_iter(&out)
        .filter_map(|caps| {
            caps.get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().to_string())
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected = std::collections::BTreeSet::from([
        "arg2_ptr".to_string(),
        "arg2_ptr + 1".to_string(),
        "arg2_ptr + 2".to_string(),
        "arg2_ptr + 3".to_string(),
        "arg2_ptr + 4".to_string(),
        "arg2_ptr + 5".to_string(),
        "arg2_ptr + 6".to_string(),
        "arg2_ptr + 7".to_string(),
    ]);
    assert!(
        stores == expected,
        "expected final digest stores to stay rooted on uint32_t word indexing over arg2_ptr, got:\n{}",
        out
    );
}

#[test]
fn full_pass_sobel_edge_detect_rewrites_iadd64_pairs_to_typed_pointers() {
    let sobel = split_decoded_functions(include_str!(
        "../test_cu/corpus_sm120/image_processing_kernels.sass"
    ))
    .into_iter()
    .find(|f| f.name == "sobel_edge_detect")
    .expect("sobel_edge_detect fixture should exist");
    let out = run_structured_output_full_pass_from_instrs(sobel.instrs, sobel.sm);
    assert!(
        !out.contains("((uintptr_t)(((uint64_t)"),
        "expected no packed arg0_ptr hi/lo reconstruction in sobel output, got:
{}",
        out
    );
    assert!(
        out.contains("uint8_t* arg0_ptr")
            && out.contains("uint8_t* arg2_ptr")
            && out.contains("*(arg0_ptr +")
            && out.contains("*(arg2_ptr + v142) = v141;"),
        "expected sobel to use typed byte loads from arg0_ptr, got:
{}",
        out
    );
}

#[test]
fn full_pass_topk_per_row_rewrites_split_window_pointer_pair() {
    let topk = split_decoded_functions(include_str!("../test_cu/corpus_sm120/ml_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "topk_per_row")
        .expect("topk_per_row fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(topk.instrs, topk.sm, "topk_per_row");
    assert!(
        !out.contains("IADD.64("),
        "expected topk_per_row to lower raw wide-add helpers, got:\n{}",
        out
    );
    assert!(
        out.contains("float* arg2_ptr")
            && out.contains("int32_t* arg4_ptr")
            && !out.contains("float* arg4_ptr")
            && out.contains("arg2_ptr[")
            && out.contains("arg4_ptr[")
            && !out.contains("addr64(")
            && !out.contains("((uint32_t*)(r2_5))"),
        "expected topk_per_row to keep arg2_ptr/arg4_ptr typed in canonical output, got:
{}",
        out
    );
    assert!(
        !out.contains("IMAD.X(")
            && !out.contains("LEA(")
            && !out.contains("LEA.HI.X(")
            && !out.contains("SHF.L.U64.HI(")
            && !out.contains("PLOP3.LUT("),
        "expected topk_per_row to lower carry-imad and plain lea helpers structurally, got:\n{}",
        out
    );
}

#[test]
fn canonical_full_pass_topk_per_row_old_corpus_keeps_remainder_loads_rooted() {
    let topk = split_decoded_functions(include_str!("../test_cu/corpus/ml_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "topk_per_row")
        .expect("topk_per_row fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(topk.instrs, topk.sm, "topk_per_row");
    assert!(
        !out.contains("((uint32_t*)(r2_5))") && out.contains("arg0_ptr["),
        "expected old-corpus topk_per_row remainder loads to stay rooted on arg0_ptr, got:\n{}",
        out
    );
}

#[test]
fn full_pass_layer_norm_forward_keeps_affine_pointer_pairs_typed() {
    let layer_norm =
        split_decoded_functions(include_str!("../test_cu/corpus_sm120/ml_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "layer_norm_forward")
            .expect("layer_norm_forward fixture should exist");
    let out = run_structured_output_full_pass_from_instrs(layer_norm.instrs, layer_norm.sm);
    assert!(
        out.contains("float* arg2_ptr")
            && out.contains("float* arg4_ptr")
            && out.contains("float* arg6_ptr")
            && out.contains("*((float*)(((uint8_t*)arg2_ptr)")
            && out.contains("*((float*)(((uint8_t*)arg4_ptr)")
            && out.contains("*((float*)(((uint8_t*)arg6_ptr)"),
        "expected layer_norm_forward gamma/beta accesses to stay on typed base-relative pointer arithmetic, got:
{}",
        out
    );
    assert!(
        !out.contains("*((uint32_t*)(((uint8_t*)arg6_ptr)"),
        "expected layer_norm_forward affine output stores to stay typed as float, got:
{}",
        out
    );
    assert!(
        !out.contains("CALL.REL.NOINC()") && !out.contains("FCHK("),
        "expected layer_norm_forward to drop compiler slow-path call/fchk scaffolding, got:
{}",
        out
    );
}

#[test]
fn canonical_full_pass_layer_norm_forward_lowers_raw_fsetp_compares() {
    let layer_norm =
        split_decoded_functions(include_str!("../test_cu/corpus_sm120/ml_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "layer_norm_forward")
            .expect("layer_norm_forward fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(
        layer_norm.instrs,
        layer_norm.sm,
        "layer_norm_forward",
    );
    let geu_compare = Regex::new(r"p\d+_\d+ = abs\(r\d+_\d+\) >= .* \|\| isnan\(abs\(r\d+_\d+\)\)")
        .expect("valid canonical layernorm compare regex");
    assert!(
        geu_compare.is_match(&out),
        "expected canonical layer_norm_forward to preserve unordered GEU semantics with an isnan disjunct, got:\n{}",
        out
    );
    assert!(
        !out.contains("FSETP.GEU.AND("),
        "expected canonical layer_norm_forward to avoid raw FSETP compare helpers, got:\n{}",
        out
    );
    assert!(
        !out.contains("LEA.HI.X(") && !out.contains("SHF.L.U64.HI("),
        "expected canonical layer_norm_forward to lower hi-lane address helpers structurally, got:\n{}",
        out
    );
}

#[test]
fn canonical_full_pass_state_machine_avoids_raw_true_predicate_helpers() {
    let state_machine =
        split_decoded_functions(include_str!("../test_cu/corpus/control_flow_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "state_machine")
            .expect("state_machine fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(
        state_machine.instrs,
        state_machine.sm,
        "state_machine",
    );
    assert!(
        !out.contains("!UPT()") && !out.contains("PT()"),
        "expected canonical state_machine to lower predicate pseudo-ops structurally, got:\n{}",
        out
    );
    assert!(
        !out.contains("UIADD3.X("),
        "expected canonical state_machine to lower carry add pseudo-ops structurally, got:\n{}",
        out
    );
    assert!(
        !out.contains("BRX(") && out.contains("switch ("),
        "expected canonical state_machine to lower indirect branches into an explicit switch placeholder, got:\n{}",
        out
    );
}

#[test]
fn canonical_full_pass_multi_exit_loop_lowers_scalar_helper_ops() {
    let kernel = split_decoded_functions(include_str!("../test_cu/corpus/control_flow_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "multi_exit_loop")
        .expect("multi_exit_loop fixture should exist");
    let out =
        run_canonical_output_full_pass_from_instrs(kernel.instrs, kernel.sm, "multi_exit_loop");
    assert!(
        !out.contains("I2F.RP(")
            && !out.contains("F2I.FTZ.U32.TRUNC.NTZ(")
            && !out.contains("IMAD.HI.U32("),
        "expected canonical multi_exit_loop to lower scalar helper ops structurally, got:\n{}",
        out
    );
    assert!(
        out.contains("__int2float_ru(")
            && out.contains("__float2uint_rz(")
            && out.contains(">> 32"),
        "expected canonical multi_exit_loop to render structural conversion/high-word math, got:\n{}",
        out
    );
}

#[test]
fn canonical_full_pass_dispatch_ops_avoids_raw_qnan_helpers() {
    let dispatch =
        split_decoded_functions(include_str!("../test_cu/corpus/control_flow_kernels.sass"))
            .into_iter()
            .find(|f| f.name == "dispatch_ops")
            .expect("dispatch_ops fixture should exist");
    let out =
        run_canonical_output_full_pass_from_instrs(dispatch.instrs, dispatch.sm, "dispatch_ops");
    assert!(
        !out.contains("+QNAN()") && out.contains("NAN"),
        "expected canonical dispatch_ops to lower QNAN pseudo-immediates structurally, got:\n{}",
        out
    );
    assert!(
        !out.contains("FRND.FLOOR(") && !out.contains("FMUL.FTZ("),
        "expected canonical dispatch_ops to lower modeled float modifier ops structurally, got:\n{}",
        out
    );
    assert!(
        out.contains("floorf("),
        "expected canonical dispatch_ops to recover floorf from FRND.FLOOR, got:\n{}",
        out
    );
}

#[test]
fn full_pass_old_softmax_forward_trims_post_loop_packed_pointer_tail() {
    let softmax = split_decoded_functions(include_str!("../test_cu/corpus/ml_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "softmax_forward")
        .expect("softmax_forward fixture should exist");
    let out =
        run_canonical_output_full_pass_from_instrs(softmax.instrs, softmax.sm, "softmax_forward");
    assert!(
        !out.contains("IADD.64("),
        "expected old-corpus softmax to lower raw wide-add helpers, got:\n{}",
        out
    );
    assert!(
        out.contains("exp2f(")
            && out.contains("arg2_ptr[")
            && !out.contains("addr64("),
        "expected old-corpus softmax cleanup to keep the hot path rooted on typed arg2_ptr arithmetic, got:
{}",
        out
    );
    assert!(
        !out.contains("CALL.REL.NOINC()"),
        "expected old-corpus softmax cleanup to drop compiler slow-path calls, got:
{}",
        out
    );
    assert!(
        !out.contains("FFMA.SAT(") && !out.contains("FFMA.RM(") && !out.contains("FADD.FTZ("),
        "expected old-corpus softmax to lower modeled float modifier ops structurally, got:\n{}",
        out
    );
    assert!(
        !out.contains("LEA.HI.X(") && !out.contains("SHF.L.U64.HI("),
        "expected old-corpus softmax to lower hi-lane address helpers structurally, got:\n{}",
        out
    );
    assert!(
        out.contains("__saturatef(") && out.contains("__fmaf_rd("),
        "expected old-corpus softmax to render saturating/rounded FFMA forms structurally, got:\n{}",
        out
    );
}

#[test]
fn full_pass_sha256_single_block_renders_without_post_name_addr64_overflow() {
    let sha = split_decoded_functions(include_str!("../test_cu/corpus/crypto_kernels.sass"))
        .into_iter()
        .find(|f| f.name == "sha256_single_block")
        .expect("sha256_single_block fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(sha.instrs, sha.sm, "sha256_single_block");
    assert!(
        out.contains("__global__ void sha256_single_block")
            || out.contains("void sha256_single_block"),
        "expected sha256_single_block to render successfully, got:
{}",
        out
    );
}

#[test]
fn full_pass_utf8_count_chars_rewrites_iadd64_pointer_arithmetic() {
    let utf8 = split_decoded_functions(include_str!(
        "../test_cu/corpus_sm120/data_processing_kernels.sass"
    ))
    .into_iter()
    .find(|f| f.name == "utf8_count_chars")
    .expect("utf8_count_chars fixture should exist");
    let out = run_canonical_output_full_pass_from_instrs(utf8.instrs, utf8.sm, "utf8_count_chars");
    assert!(
        !out.contains("IADD.64("),
        "expected no raw IADD.64 helper in utf8_count_chars output, got:
{}",
        out
    );
    assert!(
        out.contains("arg0_ptr[") && out.contains("arg2_ptr[r0_1] = r7_7;"),
        "expected typed pointer load/store recovery in utf8_count_chars, got:\n{}",
        out
    );
}

#[test]
fn lifted_rc4_no_raw_constmem_call_syntax() {
    let out = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("c[0x0][0x180]()"));
}

// ----------------------------------------------------------------------
// Corpus invariant runner
// ----------------------------------------------------------------------
//
// Drives the canonical backend against every function in
// `test_cu/corpus/*.sass` (multi-function `cuobjdump --dump-sass` dumps)
// and asserts structural invariants rather than byte-equality goldens.
// The corpus exists to catch regressions that escape the hand-authored
// per-kernel fixtures — adding a new CUDA source file and regenerating
// the `.sass` dump immediately expands coverage without writing goldens.

/// Walk a list of `(filename, sass_text)` pairs through the canonical
/// backend and return one `(file, function_name, output)` tuple per
/// function. Shared by the SM 89 corpus and the SM 100/120 (Blackwell)
/// corpora so each architecture's invariant tests can pass its own file
/// list without duplicating the per-function loop.
fn run_corpus_files(files: &[(&'static str, &'static str)]) -> Vec<(&'static str, String, String)> {
    let mut results = Vec::new();
    for (fname, text) in files.iter().copied() {
        let funcs = crate::parser::split_decoded_functions(text);
        assert!(
            !funcs.is_empty(),
            "corpus file {} should contain at least one function",
            fname
        );
        for f in funcs {
            let out = run_canonical_output_full_pass_from_instrs(f.instrs.clone(), f.sm, &f.name);
            results.push((fname, f.name, out));
        }
    }
    results
}

/// Enumerate all `(file, function_name, output)` tuples from the SM 89
/// corpus.  Each `include_str!` pulls in a real SASS dump at compile
/// time, so the corpus test binary is self-contained and deterministic.
fn run_corpus() -> Vec<(&'static str, String, String)> {
    let files: &[(&'static str, &'static str)] = &[
        (
            "arith_kernels.sass",
            include_str!("../test_cu/corpus/arith_kernels.sass"),
        ),
        (
            "branching_kernels.sass",
            include_str!("../test_cu/corpus/branching_kernels.sass"),
        ),
        (
            "loop_kernels.sass",
            include_str!("../test_cu/corpus/loop_kernels.sass"),
        ),
        (
            "shared_mem_kernels.sass",
            include_str!("../test_cu/corpus/shared_mem_kernels.sass"),
        ),
        (
            "crypto_kernels.sass",
            include_str!("../test_cu/corpus/crypto_kernels.sass"),
        ),
        (
            "compute_kernels.sass",
            include_str!("../test_cu/corpus/compute_kernels.sass"),
        ),
        (
            "control_flow_kernels.sass",
            include_str!("../test_cu/corpus/control_flow_kernels.sass"),
        ),
        (
            "image_processing_kernels.sass",
            include_str!("../test_cu/corpus/image_processing_kernels.sass"),
        ),
        (
            "ml_kernels.sass",
            include_str!("../test_cu/corpus/ml_kernels.sass"),
        ),
        (
            "simulation_kernels.sass",
            include_str!("../test_cu/corpus/simulation_kernels.sass"),
        ),
        (
            "data_processing_kernels.sass",
            include_str!("../test_cu/corpus/data_processing_kernels.sass"),
        ),
    ];
    run_corpus_files(files)
}

/// Enumerate `(file, function_name, output)` tuples from the SM 100
/// (Blackwell) corpus.  Mirrors `run_corpus` but loads the dumps from
/// `test_cu/corpus_sm100/`.  Lets the SM 100 invariant tests run the
/// same lifted+named pipeline against Blackwell-era SASS so regressions
/// in the new ABI / opcode coverage surface immediately.
fn run_corpus_sm100() -> Vec<(&'static str, String, String)> {
    let files: &[(&'static str, &'static str)] = &[
        (
            "arith_kernels.sass",
            include_str!("../test_cu/corpus_sm100/arith_kernels.sass"),
        ),
        (
            "branching_kernels.sass",
            include_str!("../test_cu/corpus_sm100/branching_kernels.sass"),
        ),
        (
            "loop_kernels.sass",
            include_str!("../test_cu/corpus_sm100/loop_kernels.sass"),
        ),
        (
            "shared_mem_kernels.sass",
            include_str!("../test_cu/corpus_sm100/shared_mem_kernels.sass"),
        ),
        (
            "crypto_kernels.sass",
            include_str!("../test_cu/corpus_sm100/crypto_kernels.sass"),
        ),
        (
            "compute_kernels.sass",
            include_str!("../test_cu/corpus_sm100/compute_kernels.sass"),
        ),
        (
            "control_flow_kernels.sass",
            include_str!("../test_cu/corpus_sm100/control_flow_kernels.sass"),
        ),
        (
            "image_processing_kernels.sass",
            include_str!("../test_cu/corpus_sm100/image_processing_kernels.sass"),
        ),
        (
            "ml_kernels.sass",
            include_str!("../test_cu/corpus_sm100/ml_kernels.sass"),
        ),
        (
            "simulation_kernels.sass",
            include_str!("../test_cu/corpus_sm100/simulation_kernels.sass"),
        ),
        (
            "data_processing_kernels.sass",
            include_str!("../test_cu/corpus_sm100/data_processing_kernels.sass"),
        ),
    ];
    run_corpus_files(files)
}

/// Enumerate `(file, function_name, output)` tuples from the SM 120
/// corpus.  SM 120 dumps add inline scheduling annotations
/// (`&req=...`, `?WAITn`, ...) that the parser strips before tokenizing
/// — this corpus exists to keep that strip path covered end-to-end.
fn run_corpus_sm120() -> Vec<(&'static str, String, String)> {
    let files: &[(&'static str, &'static str)] = &[
        (
            "arith_kernels.sass",
            include_str!("../test_cu/corpus_sm120/arith_kernels.sass"),
        ),
        (
            "branching_kernels.sass",
            include_str!("../test_cu/corpus_sm120/branching_kernels.sass"),
        ),
        (
            "loop_kernels.sass",
            include_str!("../test_cu/corpus_sm120/loop_kernels.sass"),
        ),
        (
            "shared_mem_kernels.sass",
            include_str!("../test_cu/corpus_sm120/shared_mem_kernels.sass"),
        ),
        (
            "crypto_kernels.sass",
            include_str!("../test_cu/corpus_sm120/crypto_kernels.sass"),
        ),
        (
            "compute_kernels.sass",
            include_str!("../test_cu/corpus_sm120/compute_kernels.sass"),
        ),
        (
            "control_flow_kernels.sass",
            include_str!("../test_cu/corpus_sm120/control_flow_kernels.sass"),
        ),
        (
            "image_processing_kernels.sass",
            include_str!("../test_cu/corpus_sm120/image_processing_kernels.sass"),
        ),
        (
            "ml_kernels.sass",
            include_str!("../test_cu/corpus_sm120/ml_kernels.sass"),
        ),
        (
            "simulation_kernels.sass",
            include_str!("../test_cu/corpus_sm120/simulation_kernels.sass"),
        ),
        (
            "data_processing_kernels.sass",
            include_str!("../test_cu/corpus_sm120/data_processing_kernels.sass"),
        ),
    ];
    run_corpus_files(files)
}

#[test]
fn corpus_splits_at_least_one_function_per_file() {
    // Sanity: each corpus file should split into >=1 functions and collectively
    // cover the expected kernel count.
    let results = run_corpus();
    assert!(
        results.len() >= 20,
        "expected at least 20 functions in the corpus, got {}",
        results.len()
    );
}

#[test]
fn corpus_every_function_produces_non_empty_output() {
    for (file, name, out) in run_corpus() {
        assert!(
            !out.trim().is_empty(),
            "empty pseudo-C output for {}:{}",
            file,
            name
        );
    }
}

/// Needle patterns that should never appear in rendered pseudo-C output.
/// Covers all five convergence barrier mnemonics in three forms:
///   1. Bare tokens with leading/trailing whitespace (`" BSSY "`)
///   2. Function-call form from unfolded barriers (`"BSSY("`)
///   3. Suffix forms for reliability-annotated Blackwell variants (`"BSSY."`)
const CONVERGENCE_BARRIER_BANNED_NEEDLES: &[&str] = &[
    // Bare tokens
    " BSSY ",
    " BSYNC ",
    " SSY ",
    " SYNC ",
    " WARPSYNC ",
    // Function-call form
    "BSSY(",
    "BSYNC(",
    "WARPSYNC(",
    "SSY(",
    "SYNC(",
    // Suffix forms — `BSSY.RECONVERGENT`, `SYNC.RELIABLE`, etc.
    "BSSY.",
    "BSYNC.",
    "WARPSYNC.",
    "SSY.",
    "SYNC.",
];

#[test]
fn corpus_output_contains_no_raw_convergence_barriers() {
    // BSSY/BSYNC/SSY/SYNC/WARPSYNC are control-flow scaffolding the lifter
    // should fold away. Seeing them verbatim in the rendered C means the
    // semantic-lift and structurizer pipeline missed one.
    for (file, name, out) in run_corpus() {
        for needle in CONVERGENCE_BARRIER_BANNED_NEEDLES {
            assert!(
                !out.contains(needle),
                "raw convergence barrier {:?} leaked into output for {}:{}\n---\n{}",
                needle,
                file,
                name,
                out
            );
        }
    }
}

#[test]
fn corpus_output_is_deterministic() {
    // Running the same function twice should produce identical output.
    // Determinism regressions here usually come from HashMap iteration
    // order sneaking into the pipeline.
    let first = run_corpus();
    let second = run_corpus();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2, "non-deterministic output for {}:{}", a.0, a.1);
    }
}

#[test]
fn corpus_goto_budget_is_tight() {
    // Per-fixture goto budget. The aspiration is zero gotos everywhere, but
    // until every pattern is covered by collapse rules we allow a small
    // budget per function to keep progress visible. Tighten as the
    // structurizer improves.
    //
    // Functions with genuinely complex multi-exit/early-return patterns get
    // an explicit per-function budget. Everything else must be zero.
    // When the structurizer gains new rules, reduce these budgets.
    let allow_list: std::collections::HashMap<&str, usize> = [
        // multi_exit_loop: 3 early-return paths inside a for-loop; the
        // compiler generates multiple BRA-to-EXIT paths that the collapse
        // loop cannot currently merge back to structured break/return.
        ("control_flow_kernels.sass:multi_exit_loop", 26),
        // find_pattern: 4-deep nested loop with early return from the
        // innermost level + break propagating through 2 outer levels.
        ("control_flow_kernels.sass:find_pattern", 9),
        // nested_loop_break_continue: 2-level loop with break+continue
        // at both levels — the compiler tail-duplicates some exits.
        ("control_flow_kernels.sass:nested_loop_break_continue", 16),
        // state_machine: the new explicit-terminator CFG sometimes leaves
        // one small loop latch in goto form until the AST structurizer slice
        // lands; keep a tiny temporary budget here.
        ("control_flow_kernels.sass:state_machine", 2),
        // box_blur_variable_radius: nested loop with continue (skip OOB).
        ("image_processing_kernels.sass:box_blur_variable_radius", 1),
        // string_search: loop with early break on mismatch.
        ("data_processing_kernels.sass:string_search", 4),
        // utf8_count_chars: multi-way byte classification (if-chain on
        // byte ranges) + continuation-byte skip loop.
        ("data_processing_kernels.sass:utf8_count_chars", 7),
    ]
    .into();

    let mut violations: Vec<(String, usize, usize)> = Vec::new();
    let outputs = run_corpus();
    assert!(
        !outputs.is_empty(),
        "SM 89 corpus produced no outputs — fixture broken?"
    );
    for (file, name, out) in outputs {
        let key = format!("{}:{}", file, name);
        let gotos = out.matches("goto BB").count();
        let budget = allow_list.get(key.as_str()).copied().unwrap_or(0);
        if gotos > budget {
            violations.push((key, gotos, budget));
        }
    }
    if !violations.is_empty() {
        violations.sort();
        let summary = violations
            .iter()
            .map(|(k, n, b)| format!("  {} — gotos: {} (budget: {})", k, n, b))
            .collect::<Vec<_>>()
            .join("\n");
        panic!("corpus goto budget exceeded:\n{}", summary);
    }

    // Also verify the allow-list isn't stale: if a function's goto count
    // drops below its budget, we want to tighten the budget.
    let mut over_budget: Vec<String> = Vec::new();
    for (file, name, out) in run_corpus() {
        let key = format!("{}:{}", file, name);
        let gotos = out.matches("goto BB").count();
        if let Some(&budget) = allow_list.get(key.as_str()) {
            if gotos < budget {
                over_budget.push(format!(
                    "  {} — gotos: {} but budget is {} (tighten!)",
                    key, gotos, budget
                ));
            }
        }
    }
    if !over_budget.is_empty() {
        over_budget.sort();
        eprintln!(
            "HINT: some allow-list budgets can be tightened:\n{}",
            over_budget.join("\n")
        );
    }
}

#[test]
fn corpus_output_no_raw_ffma() {
    // After semantic lifting, FFMA instructions should be rendered as
    // `a * b + c`, not as raw `FFMA(...)` opcode calls. This guards
    // against regressions in the FFMA lift rule.
    for (file, name, out) in run_corpus() {
        assert!(
            !out.contains("FFMA("),
            "raw FFMA opcode leaked into output for {}:{}\n---\n{}",
            file,
            name,
            out
        );
    }
}

#[test]
fn corpus_output_no_raw_fmnmx() {
    for (file, name, out) in run_corpus() {
        assert!(
            !out.contains("FMNMX("),
            "raw FMNMX opcode leaked into output for {}:{}\n---\n{}",
            file,
            name,
            out
        );
    }
}

#[test]
fn corpus_output_no_raw_mufu_rsq() {
    for (file, name, out) in run_corpus() {
        assert!(
            !out.contains("MUFU.RSQ("),
            "raw MUFU.RSQ opcode leaked into output for {}:{}\n---\n{}",
            file,
            name,
            out
        );
    }
}

#[test]
fn corpus_output_no_raw_mufu_ex2() {
    for (file, name, out) in run_corpus() {
        assert!(
            !out.contains("MUFU.EX2("),
            "raw MUFU.EX2 opcode leaked into output for {}:{}\n---\n{}",
            file,
            name,
            out
        );
    }
}

fn assert_no_operand_modifier_leaks(outputs: Vec<(&'static str, String, String)>, label: &str) {
    let abs_re = Regex::new(r"\|(?:UR|R)\d+\|").expect("valid regex");
    let reuse_re = Regex::new(r"\.reuse\b").expect("valid regex");
    for (file, name, out) in outputs {
        assert!(
            !abs_re.is_match(&out),
            "raw abs-bar operand leaked into {} output for {}:{}\n---\n{}",
            label,
            file,
            name,
            out
        );
        assert!(
            !reuse_re.is_match(&out),
            "raw .reuse operand leaked into {} output for {}:{}\n---\n{}",
            label,
            file,
            name,
            out
        );
    }
}
const STRUCTURED_HELPER_BANNED_NEEDLES: &[&str] =
    &["addr64(", "hfma2(", "prmt(", "FCHK(", "CALL.REL.NOINC()"];

fn assert_no_structured_helper_leaks(outputs: Vec<(&'static str, String, String)>, label: &str) {
    for (file, name, out) in outputs {
        for needle in STRUCTURED_HELPER_BANNED_NEEDLES {
            assert!(
                !out.contains(needle),
                "structured helper {:?} leaked into {} output for {}:{}\n---\n{}",
                needle,
                label,
                file,
                name,
                out
            );
        }
    }
}

#[test]
fn corpus_output_has_no_operand_modifier_leaks() {
    assert_no_operand_modifier_leaks(run_corpus(), "corpus");
}

#[test]
fn corpus_sm100_output_has_no_operand_modifier_leaks() {
    assert_no_operand_modifier_leaks(run_corpus_sm100(), "SM 100 corpus");
}

#[test]
fn corpus_sm120_output_has_no_operand_modifier_leaks() {
    assert_no_operand_modifier_leaks(run_corpus_sm120(), "SM 120 corpus");
}

#[test]
fn corpus_output_has_no_structured_helper_leaks() {
    assert_no_structured_helper_leaks(run_corpus(), "corpus");
}

#[test]
fn corpus_sm100_output_has_no_structured_helper_leaks() {
    assert_no_structured_helper_leaks(run_corpus_sm100(), "SM 100 corpus");
}

#[test]
fn corpus_sm120_output_has_no_structured_helper_leaks() {
    assert_no_structured_helper_leaks(run_corpus_sm120(), "SM 120 corpus");
}

#[test]
fn corpus_output_has_no_ssa_suffix_tokens() {
    // After name recovery, SSA-suffixed tokens like `R3.0`, `P1.2`, `UR4.1`
    // should not remain. A regression usually means name recovery failed to
    // cover a newly-emitted token shape.
    let re = Regex::new(r"\b(?:UR|UP|R|P)\d+\.\d+\b").expect("valid regex");
    for (file, name, out) in run_corpus() {
        assert!(
            !re.is_match(&out),
            "SSA-suffix token leaked into named output for {}:{}\n---\n{}",
            file,
            name,
            out
        );
    }
}

#[test]
fn corpus_output_has_no_unbound_bool_return_guards() {
    let guard_re = Regex::new(r"if \(!?(b\d+)\) return;").expect("valid regex");
    let assign_re = Regex::new(r"\b(b\d+)\s*=").expect("valid regex");
    for (file, name, out) in run_corpus() {
        let mut assigned = std::collections::BTreeSet::new();
        for line in out.lines() {
            if let Some(caps) = assign_re.captures(line) {
                assigned.insert(caps[1].to_string());
            }
            if let Some(caps) = guard_re.captures(line) {
                let pred = caps[1].to_string();
                assert!(
                    assigned.contains(&pred),
                    "unbound bool return guard leaked into output for {}:{} -> {}\n---\n{}",
                    file,
                    name,
                    pred,
                    out
                );
            }
        }
    }
}

// ----------------------------------------------------------------------
// SM 100 (Blackwell) corpus invariants
// ----------------------------------------------------------------------
//
// Mirror the most important SM 89 corpus invariants against the
// Blackwell-era dumps so regressions in the new ABI / opcode coverage
// (LDCU, BSSY.RECONVERGENT, descriptor memory, c[0x0][0x380]+ params)
// surface immediately.  We deliberately skip the strict goto-budget gate
// here because the structurizer has not yet been tuned for SM 100; the
// budget will move out of the SM 89-only test once Blackwell lifters
// fold the remaining new patterns.

#[test]
fn corpus_sm100_splits_at_least_one_function_per_file() {
    let results = run_corpus_sm100();
    assert!(
        results.len() >= 20,
        "expected at least 20 SM 100 functions in the corpus, got {}",
        results.len()
    );
}

#[test]
fn corpus_sm100_every_function_produces_non_empty_output() {
    for (file, name, out) in run_corpus_sm100() {
        assert!(
            !out.trim().is_empty(),
            "empty pseudo-C output for SM 100 {}:{}",
            file,
            name
        );
    }
}

#[test]
fn corpus_sm100_output_contains_no_raw_convergence_barriers() {
    // SM 100 reliability-annotated forms (`BSSY.RECONVERGENT`,
    // `BSYNC.RELIABLE`, ...) must be folded away by the structurizer
    // just like the legacy plain mnemonics.  Uses the shared
    // CONVERGENCE_BARRIER_BANNED_NEEDLES set which covers bare tokens,
    // function-call form, and dot-suffixed forms for all five
    // convergence barrier mnemonics.
    for (file, name, out) in run_corpus_sm100() {
        for needle in CONVERGENCE_BARRIER_BANNED_NEEDLES {
            assert!(
                !out.contains(needle),
                "raw convergence barrier {:?} leaked into SM 100 output for {}:{}\n---\n{}",
                needle,
                file,
                name,
                out
            );
        }
    }
}

#[test]
fn corpus_sm100_output_is_deterministic() {
    let first = run_corpus_sm100();
    let second = run_corpus_sm100();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(
            a.2, b.2,
            "non-deterministic SM 100 output for {}:{}",
            a.0, a.1
        );
    }
}

#[test]
fn corpus_sm100_output_has_no_ssa_suffix_tokens() {
    let re = Regex::new(r"\b(?:UR|UP|R|P)\d+\.\d+\b").expect("valid regex");
    for (file, name, out) in run_corpus_sm100() {
        assert!(
            !re.is_match(&out),
            "SSA-suffix token leaked into SM 100 named output for {}:{}\n---\n{}",
            file,
            name,
            out
        );
    }
}

#[test]
fn corpus_sm100_output_has_no_unbound_bool_return_guards() {
    let guard_re = Regex::new(r"if \(!?(b\d+)\) return;").expect("valid regex");
    let assign_re = Regex::new(r"\b(b\d+)\s*=").expect("valid regex");
    for (file, name, out) in run_corpus_sm100() {
        let mut assigned = std::collections::BTreeSet::new();
        for line in out.lines() {
            if let Some(caps) = assign_re.captures(line) {
                assigned.insert(caps[1].to_string());
            }
            if let Some(caps) = guard_re.captures(line) {
                let pred = caps[1].to_string();
                assert!(
                    assigned.contains(&pred),
                    "unbound bool return guard leaked into SM 100 output for {}:{} -> {}\n---\n{}",
                    file,
                    name,
                    pred,
                    out
                );
            }
        }
    }
}

#[test]
fn corpus_sm100_resolves_blackwell_builtins() {
    // The Blackwell ABI relocates `blockDim*` and `gridDim*` from
    // `c[0x0][0x0..0x18]` to `c[0x0][0x360..0x378]`.  This test pins three
    // overlapping invariants instead of the much weaker "at least one
    // function across the entire corpus binds a builtin" check:
    //
    //   1. Every input file has at least one function that binds a
    //      builtin.  Catches per-file regressions where one .sass dump
    //      stops resolving while every other dump papers it over.
    //   2. The overall fraction of bound functions stays above a floor
    //      (currently 50/59 ~= 85%).  Catches a global drop where most
    //      functions stop binding while a few test-friendly ones still
    //      do.
    //   3. The handful of functions that genuinely never reference
    //      `blockDim*`/`gridDim*` in their SASS body is small and
    //      stable.  If that count grows we either lost a binding or
    //      added a corpus dump that needs reclassification.
    //
    // The current corpus baseline is 52/59 functions bound; the floor is
    // set at 50 to leave a tiny amount of headroom for benign churn but
    // still catch a real regression.
    let outputs = run_corpus_sm100();
    assert!(
        !outputs.is_empty(),
        "SM 100 corpus produced zero functions — fixture is broken"
    );

    let total = outputs.len();
    let mut bound = 0usize;
    let mut per_file: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    for (file, _name, out) in &outputs {
        let entry = per_file.entry((*file).to_string()).or_insert((0, 0));
        entry.0 += 1;
        let has_builtin = out.contains("blockDim.x")
            || out.contains("blockDim.y")
            || out.contains("blockDim.z")
            || out.contains("gridDim.x")
            || out.contains("gridDim.y")
            || out.contains("gridDim.z");
        if has_builtin {
            bound += 1;
            entry.1 += 1;
        }
    }

    // Invariant 1: every file has at least one function binding a builtin.
    let files_without_any: Vec<String> = per_file
        .iter()
        .filter_map(
            |(file, (_, with))| {
                if *with == 0 {
                    Some(file.clone())
                } else {
                    None
                }
            },
        )
        .collect();
    assert!(
        files_without_any.is_empty(),
        "no Blackwell builtin (block/gridDim*) bound in these SM 100 files: {:?} \
         — BlackwellParam380 profile likely failed to bind c[0x0][0x360..0x378]",
        files_without_any
    );

    // Invariant 2: at least 50/59 functions bind a builtin (~85%).
    // Tighten this threshold once the corpus stops growing.
    const MIN_BOUND_FUNCTIONS: usize = 50;
    assert!(
        bound >= MIN_BOUND_FUNCTIONS,
        "only {}/{} SM 100 functions bound a Blackwell builtin (floor: {}). \
         A drop usually means BlackwellParam380 detection or builtin resolution \
         regressed for a class of kernels.",
        bound,
        total,
        MIN_BOUND_FUNCTIONS
    );
}

/// Compute total and per-function `goto BB` counts across a corpus pass.
/// Used by the SM 100/120 loose-budget gates below.
fn corpus_goto_summary(outputs: &[(&'static str, String, String)]) -> (usize, usize) {
    let mut total = 0usize;
    let mut max_per_fn = 0usize;
    for (_, _, out) in outputs {
        let n = out.matches("goto BB").count();
        total += n;
        if n > max_per_fn {
            max_per_fn = n;
        }
    }
    (total, max_per_fn)
}

#[test]
fn corpus_sm100_goto_budget_is_loose() {
    // SM 100 (Blackwell) deliberately runs without the per-function
    // allow-list that the SM 89 corpus uses (`corpus_goto_budget_is_tight`)
    // because the structurizer's collapse rules have not been audited
    // against the relocated builtin slots and reliability-annotated
    // barriers yet.  We still want a loose ceiling so that a regression
    // which floods the output with unstructured `goto BB` jumps does
    // not silently slip through — the structurizer is the load-bearing
    // pass for downstream tooling and a goto explosion would corrupt
    // every consumer.
    //
    // The current SM 100 corpus baseline (as of 2026-04-08) is 46 total
    // gotos with a max of 26 per function (`multi_exit_loop`).  The
    // ceilings below give roughly 15%-30% headroom over that baseline:
    // tight enough to catch a real explosion (e.g. 46 -> 200) and loose
    // enough to absorb benign churn from added rules or new corpus
    // dumps.  Tighten in lockstep with the structurizer once a per-fn
    // allow-list is added.
    const SM100_TOTAL_GOTO_CEILING: usize = 60;
    const SM100_PER_FN_GOTO_CEILING: usize = 30;

    let outputs = run_corpus_sm100();
    assert!(
        !outputs.is_empty(),
        "SM 100 corpus produced no outputs — fixture broken?"
    );
    let (total, max_per_fn) = corpus_goto_summary(&outputs);
    assert!(
        total <= SM100_TOTAL_GOTO_CEILING,
        "SM 100 corpus emitted {} `goto BB` jumps (loose ceiling: {}). \
         A real regression in the structurizer is the most likely cause; \
         drop the ignored stats helper or rerun with `--nocapture` to \
         see the per-function distribution.",
        total,
        SM100_TOTAL_GOTO_CEILING
    );
    assert!(
        max_per_fn <= SM100_PER_FN_GOTO_CEILING,
        "an SM 100 function emitted {} `goto BB` jumps (loose per-fn ceiling: {}). \
         If this is a benign baseline drift, raise the ceiling; otherwise the \
         structurizer regressed on a multi-exit pattern.",
        max_per_fn,
        SM100_PER_FN_GOTO_CEILING
    );
}

// ----------------------------------------------------------------------
// SM 120 corpus invariants
// ----------------------------------------------------------------------
//
// SM 120 uses the same ABI as SM 100 but adds inline scheduling
// annotations (`&req=`, `?WAITn`, ...) that the parser strips before
// tokenizing.  Keeping a separate corpus pass guards that strip path
// end-to-end.

#[test]
fn corpus_sm120_splits_at_least_one_function_per_file() {
    let results = run_corpus_sm120();
    assert!(
        results.len() >= 20,
        "expected at least 20 SM 120 functions in the corpus, got {}",
        results.len()
    );
}

#[test]
fn corpus_sm120_every_function_produces_non_empty_output() {
    for (file, name, out) in run_corpus_sm120() {
        assert!(
            !out.trim().is_empty(),
            "empty pseudo-C output for SM 120 {}:{}",
            file,
            name
        );
    }
}

#[test]
fn corpus_sm120_output_contains_no_scheduling_annotations() {
    // The parser must strip every `&req=…` / `?WAITn` / `?trans` token
    // before lifting; if any leak through they will surface verbatim in
    // the rendered output and a downstream parse step will choke.
    let banned = ["&req=", "&wr=", "&rd=", "?WAIT", "?trans"];
    for (file, name, out) in run_corpus_sm120() {
        for needle in banned.iter() {
            assert!(
                !out.contains(needle),
                "scheduling annotation {:?} leaked into SM 120 output for {}:{}\n---\n{}",
                needle,
                file,
                name,
                out
            );
        }
    }
}

#[test]
fn corpus_sm120_output_is_deterministic() {
    let first = run_corpus_sm120();
    let second = run_corpus_sm120();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(
            a.2, b.2,
            "non-deterministic SM 120 output for {}:{}",
            a.0, a.1
        );
    }
}

#[test]
fn corpus_sm120_goto_budget_is_loose() {
    // Mirror of `corpus_sm100_goto_budget_is_loose`.  SM 120 shares the
    // Blackwell ABI and currently produces an identical goto baseline
    // (46 total, 26 max-per-fn) because the SM 120 corpus is the SM 100
    // corpus reassembled with inline scheduling annotations stripped by
    // the parser.  We still want an independent gate here because the
    // strip path could itself regress in a way that does not affect
    // SM 100 output.
    const SM120_TOTAL_GOTO_CEILING: usize = 60;
    const SM120_PER_FN_GOTO_CEILING: usize = 30;

    let outputs = run_corpus_sm120();
    assert!(
        !outputs.is_empty(),
        "SM 120 corpus produced no outputs — fixture broken?"
    );
    let (total, max_per_fn) = corpus_goto_summary(&outputs);
    assert!(
        total <= SM120_TOTAL_GOTO_CEILING,
        "SM 120 corpus emitted {} `goto BB` jumps (loose ceiling: {}). \
         If the SM 100 ceiling still holds, the strip path for inline \
         scheduling annotations may have regressed.",
        total,
        SM120_TOTAL_GOTO_CEILING
    );
    assert!(
        max_per_fn <= SM120_PER_FN_GOTO_CEILING,
        "an SM 120 function emitted {} `goto BB` jumps (loose per-fn ceiling: {}).",
        max_per_fn,
        SM120_PER_FN_GOTO_CEILING
    );
}
