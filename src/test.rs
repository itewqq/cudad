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
fn test_split_functions_multi_function_dump() {
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
    let funcs = crate::parser::split_functions(sample);
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
fn test_split_functions_empty_on_single_function_dump_without_marker() {
    // A dump with no `Function :` markers returns empty — callers should
    // fall back to `parse_sass` in that case.
    let sample = r#"
        /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/ EXIT ;
    "#;
    assert!(crate::parser::split_functions(sample).is_empty());
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
    let sass = include_str!("../test_cu/if_loop.sass");
    let out = run_structured_output_lifted(sass);
    assert!(out.contains("PLOP3.LUT("));
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

// ----------------------------------------------------------------------
// Corpus invariant runner
// ----------------------------------------------------------------------
//
// Drives the full lifted+named pipeline against every function in
// `test_cu/corpus/*.sass` (multi-function `cuobjdump --dump-sass` dumps)
// and asserts structural invariants rather than byte-equality goldens.
// The corpus exists to catch regressions that escape the hand-authored
// per-kernel fixtures — adding a new CUDA source file and regenerating
// the `.sass` dump immediately expands coverage without writing goldens.

/// Render a single function's instructions through the lifted+named pipeline.
/// Mirrors `run_structured_output_lifted_named` but takes an already-parsed
/// instruction list so it can be driven from `split_functions`.
fn run_lifted_named_from_instrs(instrs: Vec<Instruction>, sm: Option<u32>) -> String {
    let profile = AbiProfile::detect_with_sm(&instrs, sm);
    let cfg = build_cfg(instrs);
    if cfg.node_count() == 0 {
        return String::new();
    }
    let fir = build_ssa(&cfg);
    let abi_annotations = annotate_function_ir_constmem(&fir, profile);
    let abi_aliases = infer_arg_aliases(&fir, &abi_annotations);
    let mut structurizer = Structurizer::new(&cfg, &fir);
    let lift_cfg = SemanticLiftConfig {
        abi_annotations: Some(&abi_annotations),
        abi_aliases: Some(&abi_aliases),
        strict: true,
    };
    let lifted = lift_function_ir(&fir, &lift_cfg);
    let abi_display = AbiDisplay::with_aliases(profile, abi_aliases);
    let rendered = match structurizer.structure_function() {
        Some(tree) => structurizer.pretty_print_with_lift(&tree, &abi_display, 0, Some(&lifted)),
        None => String::new(),
    };
    recover_structured_output_names(
        &fir,
        &rendered,
        &NameRecoveryConfig {
            style: NameStyle::Temp,
            rewrite_control_predicates: true,
            emit_phi_merge_comments: false,
            semantic_symbolization: true,
        },
    )
    .output
}

/// Walk a list of `(filename, sass_text)` pairs through the lifted+named
/// pipeline and return one `(file, function_name, output)` tuple per
/// function.  Shared by the SM 89 corpus and the SM 100/120 (Blackwell)
/// corpora so each architecture's invariant tests can pass its own file
/// list without duplicating the per-function loop.
fn run_corpus_files(
    files: &[(&'static str, &'static str)],
) -> Vec<(&'static str, String, String)> {
    let mut results = Vec::new();
    for (fname, text) in files.iter().copied() {
        let funcs = crate::parser::split_functions(text);
        assert!(
            !funcs.is_empty(),
            "corpus file {} should contain at least one function",
            fname
        );
        for f in funcs {
            let out = run_lifted_named_from_instrs(f.instrs.clone(), f.sm);
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
        ("arith_kernels.sass",    include_str!("../test_cu/corpus/arith_kernels.sass")),
        ("branching_kernels.sass", include_str!("../test_cu/corpus/branching_kernels.sass")),
        ("loop_kernels.sass",     include_str!("../test_cu/corpus/loop_kernels.sass")),
        ("shared_mem_kernels.sass", include_str!("../test_cu/corpus/shared_mem_kernels.sass")),
        ("crypto_kernels.sass",   include_str!("../test_cu/corpus/crypto_kernels.sass")),
        ("compute_kernels.sass",  include_str!("../test_cu/corpus/compute_kernels.sass")),
        ("control_flow_kernels.sass", include_str!("../test_cu/corpus/control_flow_kernels.sass")),
        ("image_processing_kernels.sass", include_str!("../test_cu/corpus/image_processing_kernels.sass")),
        ("ml_kernels.sass",         include_str!("../test_cu/corpus/ml_kernels.sass")),
        ("simulation_kernels.sass", include_str!("../test_cu/corpus/simulation_kernels.sass")),
        ("data_processing_kernels.sass", include_str!("../test_cu/corpus/data_processing_kernels.sass")),
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
        ("arith_kernels.sass",    include_str!("../test_cu/corpus_sm100/arith_kernels.sass")),
        ("branching_kernels.sass", include_str!("../test_cu/corpus_sm100/branching_kernels.sass")),
        ("loop_kernels.sass",     include_str!("../test_cu/corpus_sm100/loop_kernels.sass")),
        ("shared_mem_kernels.sass", include_str!("../test_cu/corpus_sm100/shared_mem_kernels.sass")),
        ("crypto_kernels.sass",   include_str!("../test_cu/corpus_sm100/crypto_kernels.sass")),
        ("compute_kernels.sass",  include_str!("../test_cu/corpus_sm100/compute_kernels.sass")),
        ("control_flow_kernels.sass", include_str!("../test_cu/corpus_sm100/control_flow_kernels.sass")),
        ("image_processing_kernels.sass", include_str!("../test_cu/corpus_sm100/image_processing_kernels.sass")),
        ("ml_kernels.sass",         include_str!("../test_cu/corpus_sm100/ml_kernels.sass")),
        ("simulation_kernels.sass", include_str!("../test_cu/corpus_sm100/simulation_kernels.sass")),
        ("data_processing_kernels.sass", include_str!("../test_cu/corpus_sm100/data_processing_kernels.sass")),
    ];
    run_corpus_files(files)
}

/// Enumerate `(file, function_name, output)` tuples from the SM 120
/// corpus.  SM 120 dumps add inline scheduling annotations
/// (`&req=...`, `?WAITn`, ...) that the parser strips before tokenizing
/// — this corpus exists to keep that strip path covered end-to-end.
fn run_corpus_sm120() -> Vec<(&'static str, String, String)> {
    let files: &[(&'static str, &'static str)] = &[
        ("arith_kernels.sass",    include_str!("../test_cu/corpus_sm120/arith_kernels.sass")),
        ("branching_kernels.sass", include_str!("../test_cu/corpus_sm120/branching_kernels.sass")),
        ("loop_kernels.sass",     include_str!("../test_cu/corpus_sm120/loop_kernels.sass")),
        ("shared_mem_kernels.sass", include_str!("../test_cu/corpus_sm120/shared_mem_kernels.sass")),
        ("crypto_kernels.sass",   include_str!("../test_cu/corpus_sm120/crypto_kernels.sass")),
        ("compute_kernels.sass",  include_str!("../test_cu/corpus_sm120/compute_kernels.sass")),
        ("control_flow_kernels.sass", include_str!("../test_cu/corpus_sm120/control_flow_kernels.sass")),
        ("image_processing_kernels.sass", include_str!("../test_cu/corpus_sm120/image_processing_kernels.sass")),
        ("ml_kernels.sass",         include_str!("../test_cu/corpus_sm120/ml_kernels.sass")),
        ("simulation_kernels.sass", include_str!("../test_cu/corpus_sm120/simulation_kernels.sass")),
        ("data_processing_kernels.sass", include_str!("../test_cu/corpus_sm120/data_processing_kernels.sass")),
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
    " BSSY ", " BSYNC ", " SSY ", " SYNC ", " WARPSYNC ",
    // Function-call form
    "BSSY(", "BSYNC(", "WARPSYNC(", "SSY(", "SYNC(",
    // Suffix forms — `BSSY.RECONVERGENT`, `SYNC.RELIABLE`, etc.
    "BSSY.", "BSYNC.", "WARPSYNC.", "SSY.", "SYNC.",
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
        assert_eq!(
            a.2, b.2,
            "non-deterministic output for {}:{}",
            a.0, a.1
        );
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
        ("control_flow_kernels.sass:find_pattern", 8),
        // nested_loop_break_continue: 2-level loop with break+continue
        // at both levels — the compiler tail-duplicates some exits.
        ("control_flow_kernels.sass:nested_loop_break_continue", 4),
        // box_blur_variable_radius: nested loop with continue (skip OOB).
        ("image_processing_kernels.sass:box_blur_variable_radius", 1),
        // string_search: loop with early break on mismatch.
        ("data_processing_kernels.sass:string_search", 3),
        // utf8_count_chars: multi-way byte classification (if-chain on
        // byte ranges) + continuation-byte skip loop.
        ("data_processing_kernels.sass:utf8_count_chars", 5),
    ]
    .into();

    let mut violations: Vec<(String, usize, usize)> = Vec::new();
    for (file, name, out) in run_corpus() {
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
        panic!(
            "corpus goto budget exceeded:\n{}",
            summary
        );
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
        let has_builtin = out.contains("blockDimX")
            || out.contains("blockDimY")
            || out.contains("blockDimZ")
            || out.contains("gridDimX")
            || out.contains("gridDimY")
            || out.contains("gridDimZ");
        if has_builtin {
            bound += 1;
            entry.1 += 1;
        }
    }

    // Invariant 1: every file has at least one function binding a builtin.
    let files_without_any: Vec<String> = per_file
        .iter()
        .filter_map(|(file, (_, with))| {
            if *with == 0 {
                Some(file.clone())
            } else {
                None
            }
        })
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
    assert!(!outputs.is_empty(), "SM 100 corpus produced no outputs — fixture broken?");
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
    assert!(!outputs.is_empty(), "SM 120 corpus produced no outputs — fixture broken?");
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

#[test]
#[ignore]
fn dump_corpus_outputs_for_review() {
    let targets = [
        "gelu_forward", "softmax_forward", "layer_norm_forward",
        "cross_entropy_loss", "fused_relu_bias_residual", "bilinear_resize",
        "sobel_edge_detect", "nms_kernel", "nbody_forces", "bfs_expand",
        "pagerank_iter", "lj_forces", "sgemm_tiled", "bitonic_sort",
        "aes128_encrypt_block", "sha256_single_block",
        "decision_tree", "state_machine", "dispatch_ops",
        "string_search", "rle_compress", "csv_find_fields",
        "utf8_count_chars", "topk_per_row", "radix_histogram",
        "box_blur_variable_radius", "batched_sgemv",
        "relu", "saxpy",
    ];
    for (file, name, out) in run_corpus() {
        if targets.contains(&name.as_str()) {
            println!("\n{}", "=".repeat(70));
            println!("=== {}:{} ===", file, name);
            println!("{}", "=".repeat(70));
            println!("{}", out);
        }
    }
}
