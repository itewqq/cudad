use cudad::*;
use crate::{cfg::build_cfg, ir::build_ssa, structurizer::*};
use clap::Parser;
use std::fs;

const SAMPLE_SASS: &str = r#"
.text._Z13verify_kernelPbPKhS1_mS1_Ph:
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R26, SR_TID.X ;
        /*0020*/                   IMAD.MOV.U32 R2, RZ, RZ, 0x978 ;
        /*0030*/                   ULDC.64 UR8, c[0x0][0x118] ;
        /*0040*/                   IADD3 R1, R1, -0x130, RZ ;
        /*0050*/                   S2R R27, SR_CTAID.X ;
        /*0060*/                   ISETP.GT.U32.AND P0, PT, R26, 0x3, PT ;
        /*0070*/                   IMAD.WIDE R2, R27, R2, c[0x0][0x168] ;
        /*0080*/              @!P0 IMAD.WIDE.U32 R4, R26, 0x8, R2 ;
        /*0090*/              @!P0 LDG.E.64 R4, [R4.64] ;
        /*00a0*/                   IMAD.MOV.U32 R18, RZ, RZ, 0x520 ;
        /*00b0*/                   SHF.R.S32.HI R16, RZ, 0x1f, R27 ;
        /*00c0*/                   IMAD.MOV.U32 R52, RZ, RZ, 0x4000 ;
        /*00d0*/                   IMAD.MOV.U32 R17, RZ, RZ, 0x1 ;
        /*00e0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;
        /*00f0*/                   IMAD.SHL.U32 R83, R26, 0x8, RZ ;
        /*0100*/                   IMAD.WIDE R18, R27, R18, c[0x0][0x180] ;
        /*0110*/                   IMAD.WIDE R52, R27, R52, c[0x0][0x188] ;
        /*0120*/              @!P0 STS.64 [R26.X8+0x3440], R4 ;
.L_x_3:
        /*0130*/                   ISETP.GT.U32.AND P0, PT, R26, 0x3f, PT ;
        /*0140*/                   BSSY B0, `(.L_x_0) ;
        /*0150*/                   IMAD.MOV.U32 R22, RZ, RZ, 0x1 ;
        /*0160*/               @P0 BRA `(.L_x_1) ;
        /*0170*/                   IMAD R4, R0, 0x240, RZ ;
        /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, RZ ;
        /*0190*/                   IMAD.MOV.U32 R22, RZ, RZ, 0x1 ;
        /*01a0*/                   IMAD.MOV.U32 R23, RZ, RZ, R26 ;
        /*01b0*/                   IMAD.WIDE R20, R27, 0x978, R4 ;
.L_x_2:
        /*01c0*/                   IMAD R25, R23, 0x9, RZ ;
        /*01d0*/                   IADD3 R9, R25.reuse, 0x2, RZ ;
        /*01e0*/                   IADD3 R7, R25, 0x1, RZ ;
        /*01f0*/                   IADD3 R8, P4, P5, R20.reuse, c[0x0][0x168], R9 ;
        /*0200*/                   IADD3 R4, P0, P1, R20.reuse, c[0x0][0x168], R25 ;
        /*0210*/                   IADD3 R6, P2, P3, R20, c[0x0][0x168], R7 ;
        /*0220*/                   IADD3.X R9, R21, c[0x0][0x16c], RZ, P4, P5 ;
        /*0230*/                   IADD3.X R5, R21.reuse, c[0x0][0x16c], RZ, P0, P1 ;
        /*0240*/                   IADD3.X R7, R21, c[0x0][0x16c], RZ, P2, P3 ;
        /*0250*/                   LDG.E.U8 R28, [R8.64+0x20] ;
        /*0260*/                   IADD3 R11, R25, 0x3, RZ ;
        /*0270*/                   LDG.E.U8 R24, [R4.64+0x20] ;
        /*0280*/                   LDG.E.U8 R29, [R6.64+0x20] ;
        /*0290*/                   IADD3 R10, P0, P1, R20, c[0x0][0x168], R11 ;
        /*02a0*/                   IADD3 R15, R25.reuse, 0x5, RZ ;
        /*02b0*/                   IADD3 R13, R25, 0x4, RZ ;
        /*02c0*/                   IADD3.X R11, R21, c[0x0][0x16c], RZ, P0, P1 ;
        /*02d0*/                   IADD3 R14, P0, P1, R20.reuse, c[0x0][0x168], R15 ;
        /*02e0*/                   IADD3 R12, P2, P3, R20, c[0x0][0x168], R13 ;
        /*02f0*/                   LDG.E.U8 R10, [R10.64+0x20] ;
        /*0300*/                   IADD3 R7, R25.reuse, 0x7, RZ ;
        /*0310*/                   IADD3 R31, R25, 0x6, RZ ;
        /*0320*/                   IADD3.X R15, R21.reuse, c[0x0][0x16c], RZ, P0, P1 ;
        /*0330*/                   IADD3.X R13, R21, c[0x0][0x16c], RZ, P2, P3 ;
        /*0340*/                   IADD3 R6, P0, P1, R20.reuse, c[0x0][0x168], R7 ;
        /*0350*/                   LDG.E.U8 R14, [R14.64+0x20] ;
        /*0360*/                   IADD3 R4, P2, P3, R20.reuse, c[0x0][0x168], R31 ;
        /*0370*/                   IADD3 R9, R25, 0x8, RZ ;
        /*0380*/                   LDG.E.U8 R13, [R12.64+0x20] ;
        /*0390*/                   IADD3.X R7, R21.reuse, c[0x0][0x16c], RZ, P0, P1 ;
        /*03a0*/                   IADD3.X R5, R21, c[0x0][0x16c], RZ, P2, P3 ;
        /*03b0*/                   IADD3 R8, P0, P1, R20, c[0x0][0x168], R9 ;
        /*03c0*/                   LDG.E.U8 R7, [R6.64+0x20] ;
        /*03d0*/                   IADD3.X R9, R21, c[0x0][0x16c], RZ, P0, P1 ;
        /*03e0*/                   LDG.E.U8 R25, [R4.64+0x20] ;
        /*03f0*/                   LDG.E.U8 R9, [R8.64+0x20] ;
        /*0400*/                   IMAD.U32 R11, R28, 0x10000, RZ ;
        /*0410*/                   PRMT R24, R29, 0x7604, R24 ;
        /*0420*/                   LOP3.LUT R11, R24, 0x30000, R11, 0xf8, !PT ;
        /*0430*/                   IADD3 R12, -R11, 0x20000, RZ ;
        /*0440*/                   SHF.R.S32.HI R11, RZ, 0x1f, R12 ;
        /*0450*/                   IMAD.SHL.U32 R24, R12, 0x2, RZ ;
        /*0460*/                   SHF.R.U32.HI R28, RZ, 0x2, R28 ;
        /*0470*/                   IMAD.SHL.U32 R5, R10, 0x40, RZ ;
        /*0480*/                   LOP3.LUT R11, R11, R24, RZ, 0xc0, !PT ;
        /*0490*/                   LOP3.LUT R5, R5, 0xffff, R28, 0xf8, !PT ;
        /*04a0*/                   IMAD.SHL.U32 R14, R14, 0x10, RZ ;
        /*04b0*/                   IMAD.SHL.U32 R4, R13, 0x4000, RZ ;
        /*04c0*/                   SHF.R.U32.HI R13, RZ, 0x4, R13 ;
        /*04d0*/                   IMAD.IADD R11, R12, 0x1, -R11 ;
        /*04e0*/                   LOP3.LUT R13, R14, 0xffff, R13, 0xf8, !PT ;
        /*04f0*/                   IMAD.SHL.U32 R8, R7, 0x4, RZ ;
        /*0500*/                   LOP3.LUT R4, R5, 0x3c000, R4, 0xf8, !PT ;
        /*0510*/                   IMAD.SHL.U32 R6, R25, 0x1000, RZ ;
        /*0520*/                   SHF.R.U32.HI R25, RZ, 0x6, R25 ;
        /*0530*/                   ISETP.GE.U32.AND P0, PT, R11, 0x1ffb2, PT ;
        /*0540*/                   IMAD.SHL.U32 R9, R9, 0x400, RZ ;
        /*0550*/                   LOP3.LUT R6, R13, 0x3f000, R6, 0xf8, !PT ;
        /*0560*/                   IADD3 R13, -R4, 0x20000, RZ ;
        /*0570*/                   LOP3.LUT R8, R8, 0xffff, R25, 0xf8, !PT ;
        /*0580*/                   SEL R11, RZ, 0x1, P0 ;
        /*0590*/                   IMAD.SHL.U32 R5, R13, 0x2, RZ ;
        /*05a0*/                   LOP3.LUT R8, R8, R9, RZ, 0xfc, !PT ;
        /*05b0*/                   IADD3 R14, -R6, 0x20000, RZ ;
        /*05c0*/                   LOP3.LUT R11, R22, R11, RZ, 0xc0, !PT ;
        /*05d0*/                   SHF.R.S32.HI R4, RZ, 0x1f, R13 ;
        /*05e0*/                   IMAD R9, R0, 0x40, R23 ;
        /*05f0*/                   IADD3 R15, -R8, 0x20000, RZ ;
        /*0600*/                   IMAD.SHL.U32 R6, R14, 0x2, RZ ;
        /*0610*/                   PRMT R7, R11, 0x9910, RZ ;
        /*0620*/                   LOP3.LUT R4, R4, R5, RZ, 0xc0, !PT ;
        /*0630*/                   SHF.R.S32.HI R5, RZ, 0x1f, R14 ;
        /*0640*/                   STS.128 [R9.X16], R12 ;
        /*0650*/                   ISETP.NE.AND P0, PT, R7, RZ, PT ;
        /*0660*/                   IMAD.SHL.U32 R8, R15, 0x2, RZ ;
        /*0670*/                   IADD3 R23, R23, c[0x0][0x0], RZ ;
        /*0680*/                   IMAD.IADD R4, R13, 0x1, -R4 ;
        /*0690*/                   SHF.R.S32.HI R7, RZ, 0x1f, R15 ;
        /*06a0*/                   LOP3.LUT R5, R5, R6, RZ, 0xc0, !PT ;
        /*06b0*/                   ISETP.GE.U32.AND P1, PT, R23, 0x40, PT ;
        /*06c0*/                   LOP3.LUT R8, R7, R8, RZ, 0xc0, !PT ;
        /*06d0*/                   IMAD.IADD R5, R14, 0x1, -R5 ;
        /*06e0*/                   ISETP.LT.U32.AND P0, PT, R4, 0x1ffb2, P0 ;
        /*06f0*/                   IMAD.IADD R8, R15, 0x1, -R8 ;
        /*0700*/                   ISETP.LT.U32.AND P0, PT, R5, 0x1ffb2, P0 ;
        /*0710*/                   ISETP.LT.U32.AND P0, PT, R8, 0x1ffb2, P0 ;
        /*0720*/                   SEL R22, RZ, 0x1, !P0 ;
        /*0730*/              @!P1 BRA `(.L_x_2) ;
.L_x_1:
        /*0740*/                   BSYNC B0 ;
.L_x_0:
        /*0750*/                   IADD3 R0, R0, 0x1, RZ ;
        /*0760*/                   LOP3.LUT R17, R22, R17, RZ, 0xc0, !PT ;
        /*0770*/                   ISETP.GE.U32.AND P0, PT, R0, 0x4, PT ;
        /*0780*/              @!P0 BRA `(.L_x_3) ;
        /*0790*/                   LDG.E.U8 R9, [R2.64+0x970] ;
        /*07a0*/                   BSSY B0, `(.L_x_4) ;
        /*07b0*/                   IMAD.MOV.U32 R0, RZ, RZ, R26 ;
.L_x_5:
        /*07c0*/                   STS [R0.X4+0x2000], RZ ;
        /*07d0*/                   IADD3 R0, R0, c[0x0][0x0], RZ ;
        /*07e0*/                   ISETP.GE.U32.AND P0, PT, R0, 0x100, PT ;
        /*07f0*/              @!P0 BRA `(.L_x_5) ;
        /*0800*/                   BSYNC B0 ;
.L_x_4:
        /*0810*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0820*/                   LDG.E.U8 R11, [R2.64+0x971] ;
        /*0830*/                   ISETP.GE.U32.AND P1, PT, R26, R9, PT ;
        /*0840*/                   BSSY B0, `(.L_x_6) ;
        /*0850*/                   PRMT R0, R17, 0x9910, RZ ;
        /*0860*/                   ISETP.GE.U32.AND P0, PT, R9, 0x51, PT ;
        /*0870*/                   ISETP.NE.AND P0, PT, R0, RZ, !P0 ;
        /*0880*/                   SEL R0, RZ, 0x1, !P0 ;
        /*0890*/               @P1 BRA `(.L_x_7) ;
        /*08a0*/                   IMAD.MOV.U32 R8, RZ, RZ, R26 ;
.L_x_8:
        /*08b0*/                   ISETP.NE.AND P1, PT, R8.reuse, RZ, PT ;
        /*08c0*/                   IADD3 R5, R8, 0x920, RZ ;
        /*08d0*/                   IADD3 R4, P0, R2, R5, RZ ;
        /*08e0*/                   IMAD.X R5, RZ, RZ, R3, P0 ;
        /*08f0*/               @P1 IADD3 R7, R8, 0x91f, RZ ;
        /*0900*/                   LDG.E.U8 R5, [R4.64] ;
        /*0910*/               @P1 IADD3 R6, P0, R2, R7, RZ ;
        /*0920*/               @P1 IMAD.X R7, RZ, RZ, R3, P0 ;
        /*0930*/               @P1 LDG.E.U8 R6, [R6.64] ;
        /*0940*/                   IMAD.MOV.U32 R12, RZ, RZ, 0x1 ;
        /*0950*/                   IADD3 R8, R8, c[0x0][0x0], RZ ;
        /*0960*/                   ISETP.GE.U32.AND P2, PT, R8, R9, PT ;
        /*0970*/                   STS [R5.X4+0x2000], R12 ;
        /*0980*/               @P1 ISETP.GT.U32.AND P0, PT, R5, R6, PT ;
        /*0990*/               @P1 SEL R10, R0, RZ, P0 ;
        /*09a0*/               @P1 PRMT R0, R10, 0x7610, R0 ;
        /*09b0*/              @!P2 BRA `(.L_x_8) ;
.L_x_7:
        /*09c0*/                   BSYNC B0 ;
.L_x_6:
        /*09d0*/                   BSSY B0, `(.L_x_9) ;
        /*09e0*/                   IMAD.MOV.U32 R4, RZ, RZ, R26 ;
.L_x_10:
        /*09f0*/                   STS [R4.X4+0x2400], RZ ;
        /*0a00*/                   IADD3 R4, R4, c[0x0][0x0], RZ ;
        /*0a10*/                   ISETP.GE.U32.AND P0, PT, R4, 0x100, PT ;
        /*0a20*/              @!P0 BRA `(.L_x_10) ;
        /*0a30*/                   BSYNC B0 ;
.L_x_9:
        /*0a40*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0a50*/                   LDG.E.U8 R12, [R2.64+0x972] ;
        /*0a60*/                   IMAD.IADD R4, R9.reuse, 0x1, R26 ;
        /*0a70*/                   ISETP.GT.U32.AND P0, PT, R9, R11.reuse, PT ;
        /*0a80*/                   BSSY B0, `(.L_x_11) ;
        /*0a90*/                   ISETP.GE.U32.AND P1, PT, R4, R11, PT ;
        /*0aa0*/                   ISETP.GT.U32.OR P0, PT, R11, 0x50, P0 ;
        /*0ab0*/                   SEL R0, R0, RZ, !P0 ;
        /*0ac0*/               @P1 BRA `(.L_x_12) ;
        /*0ad0*/                   IMAD.MOV.U32 R8, RZ, RZ, R4 ;
"#;

#[derive(clap::ValueEnum, Clone, Debug)]
enum AbiProfileMode {
    Auto,
    Legacy140,
    Modern160,
}

#[derive(Parser, Debug)]
/// CLI for CUDA SASS SSA/CFG tool
struct Args {
    /// Input SASS file (if not given, use SAMPLE_SASS)
    #[clap(short, long)]
    input: Option<String>,
    /// Output SSA DOT file (if not given, print to stdout)
    #[clap(short, long)]
    output: Option<String>,
    /// Dump CFG as DOT
    #[clap(long)]
    cfg_dot: bool,
    /// Dump SSA IR as DOT
    #[clap(long)]
    ssa_dot: bool,
    /// Dump structured block tree as text
    #[clap(long)]
    struct_code: bool,
    /// Resolve known ABI constant-memory slots (block/grid dims, params)
    #[clap(long)]
    abi_map: bool,
    /// Force ABI profile used by `--abi-map` (`auto|legacy140|modern160`)
    #[clap(long, value_enum, default_value = "auto")]
    abi_profile: AbiProfileMode,
}



fn main() {
    let args = Args::parse();
    let sass = if let Some(path) = args.input {
        fs::read_to_string(path).expect("Failed to read input file")
    } else {
        SAMPLE_SASS.to_string()
    };
    let instrs = parse_sass(&sass);
    let sm_version = parse_sm_version(&sass);
    let abi_profile = if args.abi_map || !matches!(args.abi_profile, AbiProfileMode::Auto) {
        let auto_profile = AbiProfile::detect_with_sm(&instrs, sm_version);
        let selected = match args.abi_profile {
            AbiProfileMode::Auto => auto_profile,
            AbiProfileMode::Legacy140 => AbiProfile::legacy_param_140(),
            AbiProfileMode::Modern160 => AbiProfile::modern_param_160(),
        };
        Some(selected)
    } else {
        None
    };
    let cfg = build_cfg(instrs);
    if args.cfg_dot {
        println!("{}", graph_to_dot(&cfg));
    }
    if args.ssa_dot {
        let fir = build_ssa(&cfg);
        let default_display = DefaultDisplay;
        let abi_display = abi_profile.map(AbiDisplay::new);
        let display_ctx: &dyn DisplayCtx = abi_display
            .as_ref()
            .map(|d| d as &dyn DisplayCtx)
            .unwrap_or(&default_display);
        let dot = fir.to_dot(&cfg, display_ctx);
        if let Some(ref path) = args.output {
            fs::write(path, dot).expect("Failed to write DOT file");
        } else {
            println!("{}", dot);
        }
    }
    if args.struct_code {
        let fir = build_ssa(&cfg); // build_ssa returns FunctionIR
        // Create the structurizer instance
        let mut structurizer_instance = structurizer::Structurizer::new(&cfg, &fir);
            
        println!("// --- Structured Output ---");
        if let Some(structured_func_body) = structurizer_instance.structure_function() {
            let default_display = DefaultDisplay;
            let abi_display = abi_profile.map(AbiDisplay::new);
            let display_ctx: &dyn DisplayCtx = abi_display
                .as_ref()
                .map(|d| d as &dyn DisplayCtx)
                .unwrap_or(&default_display);
            let code_output = structurizer_instance.pretty_print(&structured_func_body, display_ctx, 0);
            
            if let Some(ref path) = args.output {
                 // Potentially overwrite if also doing other dots to same file
                fs::write(path, code_output).expect("Failed to write structured code file");
                println!("Structured code written to {}", path);
            } else {
                println!("{}", code_output);
            }
        } else {
            println!("// Failed to structure function or function is empty.");
        }
        println!("// --- End Structured Output ---");
    }
    // for idx in cfg.node_indices() {
    //     println!("{:?}", &cfg[idx]);
    // }
    // // 期望 4 个基本块:
    // // 0x0000‑0010, 0x0020, 0x0030‑0040, 0x0050, 0x0060 => 实际根据 leader 划分 4
    // // assert_eq!(cfg.node_count(), 4);
    // // DOT 输出可人工验证： println!("{}", graph_to_dot(&cfg));
    // println!("{}", graph_to_dot(&cfg));
}
