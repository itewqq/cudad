use clap::Parser;
use cudad::*;
use std::fs;

/// Embedded demo SASS for quick testing without --input.
const SAMPLE_SASS: &str = include_str!("../../test_cu/sample_verify_kernel.sass");

#[derive(clap::ValueEnum, Clone, Debug)]
enum AbiProfileMode {
    Auto,
    Legacy140,
    Modern160,
}

#[derive(Parser, Debug)]
/// CLI for the canonical CUDA SASS decompiler pipeline
struct Args {
    /// Input SASS file (if not given, use SAMPLE_SASS)
    #[clap(short, long)]
    input: Option<String>,
    /// Output file for structured output or SSA DOT
    #[clap(short, long)]
    output: Option<String>,
    /// Dump CFG as DOT to stdout
    #[clap(long)]
    cfg_dot: bool,
    /// Dump optimized SSA IR as DOT
    #[clap(long)]
    ssa_dot: bool,
    /// Force ABI profile (`auto|legacy140|modern160`)
    #[clap(long, value_enum, default_value = "auto")]
    abi_profile: AbiProfileMode,
}

fn load_sass(input: Option<&str>) -> String {
    match input {
        Some(path) => fs::read_to_string(path).expect("Failed to read input file"),
        None => SAMPLE_SASS.to_string(),
    }
}

fn resolve_abi_profile(
    mode: &AbiProfileMode,
    instrs: &[DecodedInstruction],
    sm_version: Option<u32>,
) -> AbiProfile {
    match mode {
        AbiProfileMode::Auto => AbiProfile::detect_with_sm(instrs, sm_version),
        AbiProfileMode::Legacy140 => AbiProfile::legacy_param_140(),
        AbiProfileMode::Modern160 => AbiProfile::modern_param_160(),
    }
}

fn optimize_ssa(cfg: &ControlFlowGraph) -> FunctionIR {
    let ssa = build_ssa(cfg);
    let dce1 = ir_dce(&ssa);
    let cp = ir_constprop(&dce1);
    let alg = ir_algebra(&cp);
    let cse = ir_cse(&alg, cfg);
    let copyprop = ir_copyprop(&cse);
    ir_dce(&copyprop)
}

fn emit_ssa_dot(cfg: &ControlFlowGraph, abi_profile: AbiProfile, output: Option<&str>) {
    let fir = optimize_ssa(cfg);
    let anns = annotate_function_ir_constmem(&fir, abi_profile);
    let aliases = infer_arg_aliases(&fir, &anns);
    let display = AbiDisplay::with_aliases(abi_profile, aliases);
    let dot = fir.to_dot(cfg, &display);
    emit_output(&dot, output);
}

fn emit_struct_code(cfg: &ControlFlowGraph, abi_profile: AbiProfile, output: Option<&str>) {
    let fir = optimize_ssa(cfg);
    let abi_annotations = annotate_function_ir_constmem(&fir, abi_profile);
    let abi_aliases = infer_arg_aliases(&fir, &abi_annotations);
    let local_decls = infer_local_typed_declarations_with_abi(
        &fir,
        Some(&abi_annotations),
        Some(&abi_aliases),
    );

    println!("// --- Structured Output ---");
    print_abi_summary(&abi_annotations, &abi_aliases);
    if !abi_aliases.is_empty() {
        println!("// Typed signature inferred from ABI aliases:");
        for line in abi_aliases.summarize_lines(12) {
            println!("// {}", line);
        }
    }

    let mut structurizer = Structurizer::new(cfg, &fir);
    let Some(structured_body) = structurizer.structure_function() else {
        println!("// Failed to structure function or function is empty.");
        println!("// --- End Structured Output ---");
        return;
    };

    let display = AbiDisplay::with_aliases(abi_profile, abi_aliases.clone());
    let lift_cfg = SemanticLiftConfig {
        abi_annotations: Some(&abi_annotations),
        abi_aliases: Some(&abi_aliases),
        strict: true,
    };
    let lifted = lift_function_ir(&fir, &lift_cfg);
    let preview_output =
        structurizer.pretty_print_with_lift_cleanup(&structured_body, &display, 0, Some(&lifted));
    let plan = plan_structured_name_recovery_with_lift(
        &fir,
        &preview_output,
        Some(&lifted),
        &NameRecoveryConfig {
            style: NameStyle::Temp,
            rewrite_control_predicates: true,
            semantic_symbolization: true,
        },
    );
    let named_output = structurizer.pretty_print_with_lift_cleanup_and_names(
        &structured_body,
        &display,
        0,
        Some(&lifted),
        &plan.token_map,
    );
    let symbols = filter_recovered_symbols_by_output(&named_output, &plan.symbols);
    let name_type_map = infer_recovered_name_types(&fir, &symbols);
    let final_output = render_typed_structured_output(
        &named_output,
        Some(&abi_aliases),
        &local_decls,
        Some(&symbols),
        &name_type_map,
        collect_shared_memory_decls(Some(&lifted)),
    );

    emit_output(&final_output, output);
    println!("// --- End Structured Output ---");
}

fn print_abi_summary(abi_annotations: &AbiAnnotations, abi_aliases: &AbiArgAliases) {
    if !abi_annotations.is_empty() {
        println!("// ABI const-memory mapping (sample):");
        for line in abi_annotations.summarize_lines(16) {
            println!("// {}", line);
        }
    }
    if !abi_aliases.is_empty() {
        println!("// ABI arg aliases (heuristic):");
        for line in abi_aliases.summarize_lines(12) {
            println!("// {}", line);
        }
    }
}

fn emit_output(content: &str, output: Option<&str>) {
    match output {
        Some(path) => {
            fs::write(path, content).expect("Failed to write output file");
            println!("Output written to {}", path);
        }
        None => println!("{}", content),
    }
}

fn emit_empty_stub(output: Option<&str>) {
    let stub = "// Warning: no parseable SASS instruction lines were found.\n// Returning an empty stub to keep the pipeline non-fatal.\n__global__ void kernel(void) {\n}\n";
    println!("// --- Structured Output ---");
    emit_output(stub, output);
    println!("// --- End Structured Output ---");
}

fn main() {
    let args = Args::parse();
    let sass = load_sass(args.input.as_deref());
    let instrs = decode_sass(&sass);
    let sm_version = parse_sm_version(&sass);
    let abi_profile = resolve_abi_profile(&args.abi_profile, &instrs, sm_version);
    let cfg = build_cfg(instrs);

    if args.cfg_dot {
        println!("{}", graph_to_dot(&cfg));
        return;
    }

    if args.ssa_dot {
        emit_ssa_dot(&cfg, abi_profile, args.output.as_deref());
        return;
    }

    if cfg.node_count() == 0 {
        emit_empty_stub(args.output.as_deref());
    } else {
        emit_struct_code(&cfg, abi_profile, args.output.as_deref());
    }
}
