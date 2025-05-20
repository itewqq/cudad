use pretty_assertions::assert_eq;
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