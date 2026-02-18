use pretty_assertions::assert_eq;
use regex::Regex;
use crate::*;

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

const IF_SAMPLE:&str=r#"
 /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
 /*0010*/ @P0 BRA 0x0030 ;
 /*0020*/ IMAD.MOV.U32 R1, RZ, RZ, 0x5 ;
 /*0030*/ IMAD.MOV.U32 R1, RZ, RZ, 0x6 ;
 /*0040*/ EXIT ;
"#;

const LOOP_SAMPLE:&str=r#"
 /*000*/ IADD R0, R0, 0x1 ;
 /*010*/ ISETP.LT.AND P0, PT, R0, 0x5, PT ;
 /*020*/ @P0 BRA 0x000 ;
 /*030*/ EXIT ;
"#;

fn print_cfg_stdout(cfg: &ControlFlowGraph){
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
    use crate::parser::Operand::*;
    let instrs = parse_sass(SAMPLE_SASS);
    // 检查第 0 条指令第四个操作数应为 ConstMem
    match &instrs[0].operands[3] {
        ConstMem { bank, offset } => {
            assert_eq!((*bank, *offset), (0x0, 0x28));
        },
        _ => panic!("expect ConstMem"),
    }
}

#[test]
fn test_cfg() {
    let cfg = build_cfg(parse_sass(SAMPLE_SASS));
    assert_eq!(cfg.node_count(), 4);
    // 可打印 dot: println!("{}", crate::cfg::graph_to_dot(&cfg));
}

#[test]
fn test_float_immediate() {
    let instrs = parse_sass(SAMPLE_SASS_FLOAT);
    if let Operand::ImmediateF(v) = &instrs[0].operands[2] {
        assert!((*v - 0.8999999).abs() < 1e-4);
    } else { panic!("expect float immediate"); }
}

#[test]
fn test_predicated_exit_fallthrough() {
    let cfg = build_cfg(parse_sass(SAMPLE_SASS_PRED_EXIT));
    // 预计有 2 basic blocks: 0040‑0050, 0060
    assert_eq!(cfg.node_count(), 2);
    // 并且 block0 -> block1 (fall‑through) 存在
    let edges: Vec<_> = cfg.edge_indices().collect();
    assert_eq!(edges.len(), 1);
}

#[test]
fn phi_insert(){
    let cfg=build_cfg(parse_sass(IF_SAMPLE));
    print_cfg_stdout(&cfg);
    let fir=build_ssa(&cfg);
    let phi_cnt:usize=fir.blocks.iter().map(|b|b.stmts.iter().filter(|s|matches!(s.value,crate::ir::RValue::Phi(_))).count()).sum();
    assert_eq!(phi_cnt,1);
}

#[test]
fn rpo_loop(){
    let cfg=build_cfg(parse_sass(LOOP_SAMPLE));
    let fir=build_ssa(&cfg);
    assert!(fir.blocks.len()>=2);
}

#[test]
fn test_parser_normalizes_register_modifiers_and_memref() {
    let sample = r#"
        /*0000*/ IADD3 R4, R2.reuse, -0x1, RZ ;
        /*0010*/ LDG.E.U8 R16, [R2.64+0x1] ;
        /*0020*/ ULOP3.LUT UR5, UR5, 0xffffff00, URZ, 0xc0, !UPT ;
    "#;
    let instrs = parse_sass(sample);
    assert_eq!(instrs.len(), 3);

    match &instrs[0].operands[1] {
        Operand::Register { class, idx, .. } => {
            assert_eq!(class, "R");
            assert_eq!(*idx, 2);
        }
        _ => panic!("expected normalized register"),
    }
    match &instrs[1].operands[1] {
        Operand::MemRef { offset, width, .. } => {
            assert_eq!(*offset, Some(1));
            assert_eq!(*width, Some(64));
        }
        _ => panic!("expected parsed mem ref"),
    }
    match &instrs[2].operands[3] {
        Operand::Register { class, .. } => assert_eq!(class, "URZ"),
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
fn test_ssa_keeps_immutable_special_registers_unversioned() {
    let sample = r#"
        /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
        /*0010*/ IADD3 R1, RZ, 0x1, RZ ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(parse_sass(sample));
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
    let cfg = build_cfg(parse_sass(sample));
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
    let cfg = build_cfg(parse_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir
        .blocks
        .iter()
        .find(|b| b.id == 0)
        .expect("expected BB0");

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
fn ssa_uiadd3_pred_output_feeds_uiadd3_x_carry_input() {
    let sample = r#"
        /*0000*/ UIADD3 UR8, UP0, UR8, 0x4, URZ ;
        /*0010*/ UIADD3.X UR9, URZ, UR9, URZ, UP0, !UPT ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(parse_sass(sample));
    let fir = build_ssa(&cfg);
    let block0 = fir
        .blocks
        .iter()
        .find(|b| b.id == 0)
        .expect("expected BB0");

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
fn semantic_lift_carry_def_uses_pre_increment_low_operand() {
    let sample = r#"
        /*0000*/ IADD3 R1, P0, R1, 0x1, RZ ;
        /*0010*/ IADD3.X R2, R2, RZ, RZ, P0, !PT ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(parse_sass(sample));
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
    let cfg = build_cfg(parse_sass(sample));
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

#[test]
fn test_predicated_instruction_preserves_negated_sense() {
    let sample = r#"
        /*0000*/ ISETP.NE.AND P0, PT, R1, RZ, PT ;
        /*0010*/ @!P0 IADD3 R2, R2, 0x1, RZ ;
        /*0020*/ EXIT ;
    "#;
    let out = run_structured_output(sample);
    // In the raw SSA path (without name recovery), predicated instructions
    // with old values always render as ternary selects.  The self-referencing
    // ternary → if-guard conversion happens later, during name recovery.
    assert!(
        out.contains("!(P0.0) ? (IADD3(") && out.contains(") : R2.0"),
        "expected ternary select with negated predicate, got:\n{}",
        out
    );
}

fn run_structured_output(sass: &str) -> String {
    let cfg = build_cfg(parse_sass(sass));
    if cfg.node_count() == 0 {
        return "void kernel(void) {\n}\n".to_string();
    }
    let fir = build_ssa(&cfg);
    let mut structurizer = Structurizer::new(&cfg, &fir);
    match structurizer.structure_function() {
        Some(tree) => structurizer.pretty_print(&tree, &DefaultDisplay, 0),
        None => String::new(),
    }
}

fn run_structured_output_lifted(sass: &str) -> String {
    let cfg = build_cfg(parse_sass(sass));
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

fn run_structured_output_lifted_named(sass: &str) -> String {
    let cfg = build_cfg(parse_sass(sass));
    if cfg.node_count() == 0 {
        return "void kernel(void) {\n}\n".to_string();
    }
    let fir = build_ssa(&cfg);
    let mut structurizer = Structurizer::new(&cfg, &fir);
    let lift_cfg = SemanticLiftConfig::default();
    let lifted = lift_function_ir(&fir, &lift_cfg);
    let rendered = match structurizer.structure_function() {
        Some(tree) => structurizer.pretty_print_with_lift(&tree, &DefaultDisplay, 0, Some(&lifted)),
        None => String::new(),
    };
    recover_structured_output_names(
        &fir,
        &rendered,
        &NameRecoveryConfig {
            style: NameStyle::Temp,
            rewrite_control_predicates: true,
            emit_phi_merge_comments: false,
            semantic_symbolization: false,
        },
    )
    .output
}

#[test]
fn empty_cfg_ssa_build_is_non_fatal() {
    let cfg = build_cfg(Vec::new());
    let fir = build_ssa(&cfg);
    assert!(fir.blocks.is_empty());
}

#[test]
fn malformed_sass_returns_stub_output() {
    let out = run_structured_output("not sass");
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

fn run_structured_output_lifted_named_with_phi_comments(sass: &str) -> String {
    let cfg = build_cfg(parse_sass(sass));
    let fir = build_ssa(&cfg);
    let mut structurizer = Structurizer::new(&cfg, &fir);
    let lift_cfg = SemanticLiftConfig::default();
    let lifted = lift_function_ir(&fir, &lift_cfg);
    let rendered = match structurizer.structure_function() {
        Some(tree) => structurizer.pretty_print_with_lift(&tree, &DefaultDisplay, 0, Some(&lifted)),
        None => String::new(),
    };
    recover_structured_output_names(
        &fir,
        &rendered,
        &NameRecoveryConfig {
            style: NameStyle::Temp,
            rewrite_control_predicates: true,
            emit_phi_merge_comments: true,
            semantic_symbolization: false,
        },
    )
    .output
}

fn render_typed_structured_output_for_test(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
    local_decls: &[String],
) -> String {
    let params = aliases
        .map(|a| a.render_typed_param_list())
        .unwrap_or_default();
    let sig = if params.is_empty() {
        "void kernel(void)".to_string()
    } else {
        format!("void kernel({})", params.join(", "))
    };

    let mut out = String::new();
    out.push_str(&sig);
    out.push_str(" {\n");
    for d in local_decls {
        out.push_str("  ");
        out.push_str(d);
        out.push('\n');
    }
    let extra_decls = infer_self_contained_locals_for_test(code_output, aliases, local_decls);
    for d in &extra_decls {
        out.push_str("  ");
        out.push_str(d);
        out.push('\n');
    }
    if !local_decls.is_empty() || !extra_decls.is_empty() {
        out.push('\n');
    }
    for line in code_output.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

fn infer_self_contained_locals_for_test(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
    local_decls: &[String],
) -> Vec<String> {
    let ident_re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").expect("valid regex");
    let temp_re = Regex::new(
        r"\b(?:v\d+|u\d+|b\d+|abi_internal_0x[0-9A-Fa-f]+|arg\d+_ptr_(?:lo32|hi32)|arg\d+_word\d+)\b",
    )
    .expect("valid regex");
    let assign_re = Regex::new(
        r"^\s*(?:if\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=",
    )
    .expect("valid regex");
    let mut declared = std::collections::BTreeSet::<String>::new();
    for d in local_decls {
        for m in ident_re.find_iter(d) {
            declared.insert(m.as_str().to_string());
        }
    }
    if let Some(a) = aliases {
        for p in a.render_typed_param_list() {
            for m in ident_re.find_iter(&p) {
                declared.insert(m.as_str().to_string());
            }
        }
    }

    let mut seen = std::collections::BTreeSet::<String>::new();
    let mut ordered = Vec::<String>::new();
    let mut defined = std::collections::BTreeSet::<String>::new();
    let mut live_ins = std::collections::BTreeSet::<String>::new();
    for raw_line in code_output.lines() {
        let line = raw_line.split("//").next().unwrap_or("");
        let lhs_span = assign_re.captures(line).and_then(|c| {
            let whole = c.get(0)?;
            let after_eq = line.get(whole.end()..).unwrap_or("").trim_start();
            if after_eq.starts_with('=') {
                return None;
            }
            c.get(1)
                .map(|m| (m.as_str().to_string(), m.start(), m.end()))
        });
        for m in temp_re.find_iter(line) {
            let t = m.as_str();
            let is_lhs = lhs_span
                .as_ref()
                .map(|(_, s, e)| m.start() == *s && m.end() == *e)
                .unwrap_or(false);
            if !is_lhs && !defined.contains(t) {
                live_ins.insert(t.to_string());
            }
            if declared.contains(t) {
                continue;
            }
            if seen.insert(t.to_string()) {
                ordered.push(t.to_string());
            }
        }
        if let Some((lhs, _, _)) = lhs_span {
            if temp_re.is_match(&lhs) {
                defined.insert(lhs);
            }
        }
    }

    let mut out = Vec::<String>::new();
    if code_output.contains("shmem_u8[") && !declared.contains("shmem_u8") {
        out.push("__shared__ uint8_t shmem_u8[256];".to_string());
    }
    for name in ordered {
        let is_bool = name.starts_with('b')
            && name[1..].chars().all(|c| c.is_ascii_digit());
        let is_live_in = live_ins.contains(&name);
        if is_bool {
            if is_live_in {
                out.push(format!("bool {}; // live-in", name));
            } else {
                out.push(format!("bool {};", name));
            }
        } else {
            if is_live_in {
                out.push(format!("uint32_t {}; // live-in", name));
            } else {
                out.push(format!("uint32_t {};", name));
            }
        }
    }
    out
}

fn run_structured_output_full_pass(sass: &str) -> String {
    let instrs = parse_sass(sass);
    let cfg = build_cfg(instrs.clone());
    if cfg.node_count() == 0 {
        return "void kernel(void) {\n}\n".to_string();
    }
    let sm = parse_sm_version(sass);
    let inferred_profile = AbiProfile::detect_with_sm(&instrs, sm);
    let fir = build_ssa(&cfg);
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

    if let Some(tree) = structurizer.structure_function() {
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
        let code_output = structurizer.pretty_print_with_lift(&tree, display_ctx, 0, Some(&lifted));
        let recovered_output = recover_structured_output_names(
            &fir,
            &code_output,
            &NameRecoveryConfig {
                style: NameStyle::Temp,
                rewrite_control_predicates: true,
                emit_phi_merge_comments: true,
                semantic_symbolization: true,
            },
        )
        .output;
        let final_output = render_typed_structured_output_for_test(
            &recovered_output,
            abi_aliases.as_ref(),
            &local_decls,
        );
        out.push_str(&final_output);
    } else {
        out.push_str("// Failed to structure function or function is empty.\n");
    }
    out.push_str("// --- End Structured Output ---\n");
    out
}

fn split_call_metrics(rendered: &str) -> (usize, usize) {
    let raw_re = Regex::new(r"\b[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*\(").expect("valid regex");
    let helper_re = Regex::new(
        r"\b(?:f2i_trunc_u32_ftz_ntz|i2f_rp|rcp_approx|mul_hi_u32|uldc64|abs)\(",
    )
    .expect("valid regex");
    (raw_re.find_iter(rendered).count(), helper_re.find_iter(rendered).count())
}

#[test]
fn test_abi_profile_detects_legacy_window_from_sample() {
    let sample = r#"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
        /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x148] ;
        /*0020*/ IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x154] ;
    "#;
    let instrs = parse_sass(sample);
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
    let instrs = parse_sass(sample);
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
    let cfg = build_cfg(parse_sass(sample));
    let fir = build_ssa(&cfg);
    let display = AbiDisplay::new(AbiProfile::legacy_param_140());
    let out = fir.to_dot(&cfg, &display);

    assert!(out.contains("param_0"));
    assert!(out.contains("blockDimX"));
}

#[test]
fn test_abi_inferred_aliases_show_in_output() {
    let sample = r#"
        /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
        /*0010*/ IADD3.X R5, R5, c[0x0][0x164], RZ ;
        /*0020*/ EXIT ;
    "#;
    let cfg = build_cfg(parse_sass(sample));
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
fn smoke_struct_output_if_sass() {
    let sass = include_str!("../test_cu/if.sass");
    let expected = include_str!("../test_cu/golden/if.pseudo.c");
    let out1 = run_structured_output(sass);
    let out2 = run_structured_output(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_loop_constant_sass() {
    let sass = include_str!("../test_cu/loop_constant.sass");
    let expected = include_str!("../test_cu/golden/loop_constant.pseudo.c");
    let out1 = run_structured_output(sass);
    let out2 = run_structured_output(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_if_loop_sass() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let expected = include_str!("../test_cu/golden/if_loop.pseudo.c");
    let out1 = run_structured_output(sass);
    let out2 = run_structured_output(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_test_div_sass() {
    let sass = include_str!("../test_cu/test_div.sass");
    let expected = include_str!("../test_cu/golden/test_div.pseudo.c");
    let out1 = run_structured_output(sass);
    let out2 = run_structured_output(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_if_sass() {
    let sass = include_str!("../test_cu/if.sass");
    let expected = include_str!("../test_cu/golden_lifted/if.pseudo.c");
    let out1 = run_structured_output_lifted(sass);
    let out2 = run_structured_output_lifted(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_loop_constant_sass() {
    let sass = include_str!("../test_cu/loop_constant.sass");
    let expected = include_str!("../test_cu/golden_lifted/loop_constant.pseudo.c");
    let out1 = run_structured_output_lifted(sass);
    let out2 = run_structured_output_lifted(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_if_loop_sass() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let expected = include_str!("../test_cu/golden_lifted/if_loop.pseudo.c");
    let out1 = run_structured_output_lifted(sass);
    let out2 = run_structured_output_lifted(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_test_div_sass() {
    let sass = include_str!("../test_cu/test_div.sass");
    let expected = include_str!("../test_cu/golden_lifted/test_div.pseudo.c");
    let out1 = run_structured_output_lifted(sass);
    let out2 = run_structured_output_lifted(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_rc4_sass() {
    let sass = include_str!("../test_cu/rc4.sass");
    let expected = include_str!("../test_cu/golden_lifted/rc4.pseudo.c");
    let out1 = run_structured_output_lifted(sass);
    let out2 = run_structured_output_lifted(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
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
    let sass = include_str!("../test_cu/loop_constant.sass");
    let out = run_structured_output_lifted(sass);
    assert!(out.contains("BSYNC("));
}

#[test]
fn semantic_lift_has_mixed_coverage_on_if_loop_fixture() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let cfg = build_cfg(parse_sass(sass));
    let fir = build_ssa(&cfg);
    let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());
    assert!(lifted.stats.lifted > 0);
    assert!(lifted.stats.fallback > 0);
}

#[test]
fn semantic_lift_percentage_gate_if_loop_fixture() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let cfg = build_cfg(parse_sass(sass));
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
fn smoke_struct_output_lifted_named_if_sass() {
    let sass = include_str!("../test_cu/if.sass");
    let expected = include_str!("../test_cu/golden_lifted_named/if.pseudo.c");
    let out1 = run_structured_output_lifted_named(sass);
    let out2 = run_structured_output_lifted_named(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_named_loop_constant_sass() {
    let sass = include_str!("../test_cu/loop_constant.sass");
    let expected = include_str!("../test_cu/golden_lifted_named/loop_constant.pseudo.c");
    let out1 = run_structured_output_lifted_named(sass);
    let out2 = run_structured_output_lifted_named(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_named_if_loop_sass() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let expected = include_str!("../test_cu/golden_lifted_named/if_loop.pseudo.c");
    let out1 = run_structured_output_lifted_named(sass);
    let out2 = run_structured_output_lifted_named(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_named_test_div_sass() {
    let sass = include_str!("../test_cu/test_div.sass");
    let expected = include_str!("../test_cu/golden_lifted_named/test_div.pseudo.c");
    let out1 = run_structured_output_lifted_named(sass);
    let out2 = run_structured_output_lifted_named(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_lifted_named_rc4_sass() {
    let sass = include_str!("../test_cu/rc4.sass");
    let expected = include_str!("../test_cu/golden_lifted_named/rc4.pseudo.c");
    let out1 = run_structured_output_lifted_named(sass);
    let out2 = run_structured_output_lifted_named(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_full_pass_rc4_sass() {
    let sass = include_str!("../test_cu/rc4.sass");
    let expected = include_str!("../test_cu/golden_full_pass/rc4.pseudo.c");
    let out1 = run_structured_output_full_pass(sass);
    let out2 = run_structured_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_full_pass_if_sass() {
    let sass = include_str!("../test_cu/if.sass");
    let expected = include_str!("../test_cu/golden_full_pass/if.pseudo.c");
    let out1 = run_structured_output_full_pass(sass);
    let out2 = run_structured_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_full_pass_loop_constant_sass() {
    let sass = include_str!("../test_cu/loop_constant.sass");
    let expected = include_str!("../test_cu/golden_full_pass/loop_constant.pseudo.c");
    let out1 = run_structured_output_full_pass(sass);
    let out2 = run_structured_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_full_pass_if_loop_sass() {
    let sass = include_str!("../test_cu/if_loop.sass");
    let expected = include_str!("../test_cu/golden_full_pass/if_loop.pseudo.c");
    let out1 = run_structured_output_full_pass(sass);
    let out2 = run_structured_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn smoke_struct_output_full_pass_test_div_sass() {
    let sass = include_str!("../test_cu/test_div.sass");
    let expected = include_str!("../test_cu/golden_full_pass/test_div.pseudo.c");
    let out1 = run_structured_output_full_pass(sass);
    let out2 = run_structured_output_full_pass(sass);
    assert!(!out1.trim().is_empty());
    assert_eq!(out1, out2);
    assert_eq!(out1.trim_end(), expected.trim_end());
}

#[test]
fn named_output_has_no_ssa_suffix_tokens_rc4() {
    let out = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    let re = Regex::new(r"\b(?:UR|UP|R|P)\d+\.\d+\b").expect("valid regex");
    assert!(!re.is_match(&out));
}

#[test]
fn named_output_deterministic_rc4() {
    let o1 = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    let o2 = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    assert_eq!(o1, o2);
}

#[test]
fn named_output_preserves_no_goto_gate() {
    let if_loop = run_structured_output_lifted_named(include_str!("../test_cu/if_loop.sass"));
    let rc4 = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    let test_div = run_structured_output_lifted_named(include_str!("../test_cu/test_div.sass"));

    assert_eq!(if_loop.matches("goto BB").count(), 0);
    assert_eq!(rc4.matches("goto BB").count(), 0);
    assert_eq!(test_div.matches("goto BB").count(), 0);
}

#[test]
fn named_output_with_semantic_lift_still_removes_opcode_noise() {
    let out = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("hi32("));
    assert!(out.contains("? 1 : 0"));
    assert!(!out.contains("IADD3.X("));
    assert!(!out.contains("LEA.HI("));
    assert!(!out.contains("LOP3.LUT("));
}

#[test]
fn lifted_named_call_metrics_split_rc4() {
    let out = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    let (raw, helper) = split_call_metrics(&out);
    assert_eq!(raw, 0);
    assert!(helper > 0);
}

#[test]
fn lifted_named_call_metrics_split_test_div() {
    let out = run_structured_output_lifted_named(include_str!("../test_cu/test_div.sass"));
    let (raw, helper) = split_call_metrics(&out);
    assert!(raw <= 1);
    assert!(helper > 0);
}

#[test]
fn named_rc4_uses_parenthesized_pointer_offset_syntax() {
    let out = run_structured_output_lifted_named(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("*v9+1"));
    assert!(!out.contains("*v164+1"));
    let re = Regex::new(r"addr64\([^\)]+\) \+ 1").expect("valid regex");
    assert!(re.is_match(&out));
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
    assert!(out.contains("if (!(P") || out.contains("if (!P") || out.contains("if (!(b") || out.contains("if (!b"));
}

#[test]
fn full_pass_rc4_avoids_overclaimed_lea_hi_x_pointer_formula() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("lea_hi_x_sx32("));
    assert!(!out.contains("arg0_ptr.hi32 << 1"));
    assert!(!out.contains("ConstMem(0, 356) << 1"));
}

#[test]
fn full_pass_rc4_global_u8_accesses_use_addr64_pairs() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    assert!(out.contains("((uint8_t*)addr64("));
}

#[test]
fn full_pass_rc4_key_sel_uses_ssa_predicate_not_raw_p1() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("!P1 ?"));
}

#[test]
fn full_pass_rc4_lea_hi_x_uses_add_carry_predicates() {
    let out = run_structured_output_full_pass(include_str!("../test_cu/rc4.sass"));
    let carry_defs = Regex::new(r"(b\d+)\s*=\s*carry_u32_add3\(").expect("valid regex");
    let mut found_lea_using_carry = false;
    for cap in carry_defs.captures_iter(&out) {
        let b = cap.get(1).expect("capture 1").as_str();
        let needle = format!(", {})", b);
        if out.contains("lea_hi_x_sx32(") && out.contains(&needle) {
            found_lea_using_carry = true;
            break;
        }
    }
    assert!(
        found_lea_using_carry,
        "expected at least one lea_hi_x_sx32(..., bN) to use carry_u32_add3-derived predicate"
    );
}

#[test]
fn lifted_rc4_no_raw_constmem_call_syntax() {
    let out = run_structured_output_lifted(include_str!("../test_cu/rc4.sass"));
    assert!(!out.contains("c[0x0][0x180]()"));
}

#[test]
fn named_phi_merge_comments_are_opt_in() {
    let off = run_structured_output_lifted_named(include_str!("../test_cu/if_loop.sass"));
    let on = run_structured_output_lifted_named_with_phi_comments(include_str!("../test_cu/if_loop.sass"));
    assert!(!off.contains("phi merge:"));
    assert!(on.contains("phi merge:"));
}
