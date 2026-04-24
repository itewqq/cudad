use clap::Parser;
use cudad::*;
use std::{fs, process};

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
    /// Select one function from a multi-function dump by `Function :` name
    #[clap(long)]
    function: Option<String>,
    /// Force ABI profile (`auto|legacy140|modern160`)
    #[clap(long, value_enum, default_value = "auto")]
    abi_profile: AbiProfileMode,
}

#[derive(Clone, Debug, PartialEq)]
struct InputFunction {
    name: Option<String>,
    sm: Option<u32>,
    instrs: Vec<DecodedInstruction>,
}

impl InputFunction {
    fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("kernel")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputMode {
    Structured,
    CfgDot,
    SsaDot,
}

fn load_sass(input: Option<&str>) -> String {
    match input {
        Some(path) => fs::read_to_string(path).expect("Failed to read input file"),
        None => SAMPLE_SASS.to_string(),
    }
}

fn load_input_functions(sass: &str) -> Vec<InputFunction> {
    let funcs = split_decoded_functions(sass);
    if funcs.is_empty() {
        return vec![InputFunction {
            name: None,
            sm: parse_sm_version(sass),
            instrs: decode_sass(sass),
        }];
    }

    funcs
        .into_iter()
        .map(|f| InputFunction {
            name: Some(f.name),
            sm: f.sm,
            instrs: f.instrs,
        })
        .collect()
}

fn format_available_functions(functions: &[InputFunction]) -> String {
    let names = functions
        .iter()
        .filter_map(|f| f.name.as_deref())
        .collect::<Vec<_>>();
    if names.is_empty() {
        "input has no named `Function :` sections".to_string()
    } else {
        names.join(", ")
    }
}

fn select_functions(
    functions: Vec<InputFunction>,
    requested: Option<&str>,
) -> Result<Vec<InputFunction>, String> {
    if let Some(name) = requested {
        let available = format_available_functions(&functions);
        let selected = functions
            .into_iter()
            .filter(|f| f.name.as_deref() == Some(name))
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Err(format!(
                "Function `{}` not found. Available functions: {}",
                name, available
            ));
        }
        return Ok(selected);
    }

    Ok(functions)
}

fn require_single_function<'a>(
    functions: &'a [InputFunction],
    mode: OutputMode,
) -> Result<&'a InputFunction, String> {
    if functions.len() == 1 {
        return Ok(&functions[0]);
    }

    let flag = match mode {
        OutputMode::Structured => "structured output",
        OutputMode::CfgDot => "--cfg-dot",
        OutputMode::SsaDot => "--ssa-dot",
    };
    Err(format!(
        "{} requires a single function, but the input resolved to {} functions. Use `--function <name>` to choose one.",
        flag,
        functions.len()
    ))
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

fn build_ssa_dot(cfg: &ControlFlowGraph, abi_profile: AbiProfile) -> String {
    let fir = optimize_ssa(cfg);
    let anns = annotate_function_ir_constmem(&fir, abi_profile);
    let aliases = infer_arg_aliases(&fir, &anns);
    let display = AbiDisplay::with_aliases(abi_profile, aliases);
    fir.to_dot(cfg, &display)
}

fn append_abi_summary(
    out: &mut String,
    abi_annotations: &AbiAnnotations,
    abi_aliases: &AbiArgAliases,
) {
    if !abi_annotations.is_empty() {
        out.push_str("// ABI const-memory mapping (sample):\n");
        for line in abi_annotations.summarize_lines(16) {
            out.push_str("// ");
            out.push_str(&line);
            out.push('\n');
        }
    }
    if !abi_aliases.is_empty() {
        out.push_str("// ABI arg aliases (heuristic):\n");
        for line in abi_aliases.summarize_lines(12) {
            out.push_str("// ");
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("// Typed signature inferred from ABI aliases:\n");
        for line in abi_aliases.summarize_lines(12) {
            out.push_str("// ");
            out.push_str(&line);
            out.push('\n');
        }
    }
}

fn build_empty_stub(function_name: &str, warning: &str) -> String {
    let mut out = String::new();
    out.push_str("// --- Structured Output ---\n");
    out.push_str("// Warning: ");
    out.push_str(warning);
    out.push('\n');
    out.push_str(&format!("void {}(void) {{\n}}\n", function_name));
    out.push_str("// --- End Structured Output ---\n");
    out
}

fn build_canonical_structured_output(
    function: &InputFunction,
    abi_profile: AbiProfile,
) -> String {
    let function_name = function.display_name();
    let artifacts = build_named_decompile_artifacts_with_profile(
        function.instrs.clone(),
        function.sm,
        Some(function_name),
        Some(abi_profile),
    );
    let Some(analysis) = artifacts.analysis.as_ref() else {
        return build_empty_stub(
            function_name,
            "no parseable SASS instruction lines were found; returning an empty canonical stub.",
        );
    };
    let Some(rendered) = artifacts.rendered.as_ref() else {
        return build_empty_stub(
            function_name,
            "canonical backend did not produce structured output for this function.",
        );
    };
    let mut out = String::new();
    out.push_str("// --- Structured Output ---\n");
    append_abi_summary(&mut out, &analysis.abi_annotations, &analysis.abi_aliases);
    out.push_str(rendered);
    if !rendered.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("// --- End Structured Output ---\n");
    out
}

fn build_function_output(
    function: &InputFunction,
    mode: OutputMode,
    abi_mode: &AbiProfileMode,
) -> String {
    let abi_profile = resolve_abi_profile(abi_mode, &function.instrs, function.sm);

    match mode {
        OutputMode::Structured => build_canonical_structured_output(function, abi_profile),
        OutputMode::CfgDot => {
            let cfg = build_cfg(function.instrs.clone());
            graph_to_dot(&cfg)
        }
        OutputMode::SsaDot => {
            let cfg = build_cfg(function.instrs.clone());
            build_ssa_dot(&cfg, abi_profile)
        }
    }
}

fn build_multi_function_output(functions: &[InputFunction], abi_mode: &AbiProfileMode) -> String {
    if functions.len() == 1 {
        return build_function_output(&functions[0], OutputMode::Structured, abi_mode);
    }

    let mut out = String::new();
    for (idx, function) in functions.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str("// === Function: ");
        out.push_str(function.display_name());
        out.push_str(" ===\n");
        out.push_str(&build_function_output(
            function,
            OutputMode::Structured,
            abi_mode,
        ));
    }
    out
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

fn exit_with_error(message: &str) -> ! {
    eprintln!("error: {}", message);
    process::exit(2);
}

fn main() {
    let args = Args::parse();
    let sass = load_sass(args.input.as_deref());
    let functions = load_input_functions(&sass);
    let functions = select_functions(functions, args.function.as_deref())
        .unwrap_or_else(|e| exit_with_error(&e));

    let mode = if args.cfg_dot {
        OutputMode::CfgDot
    } else if args.ssa_dot {
        OutputMode::SsaDot
    } else {
        OutputMode::Structured
    };

    match mode {
        OutputMode::Structured => {
            let content = build_multi_function_output(&functions, &args.abi_profile);
            emit_output(&content, args.output.as_deref());
        }
        OutputMode::CfgDot | OutputMode::SsaDot => {
            let function =
                require_single_function(&functions, mode).unwrap_or_else(|e| exit_with_error(&e));
            let content = build_function_output(function, mode, &args.abi_profile);
            emit_output(&content, args.output.as_deref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_input_functions_falls_back_for_single_function_input() {
        let sample = "/*0000*/ MOV R0, RZ ;\n/*0010*/ EXIT ;\n";
        let funcs = load_input_functions(sample);
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, None);
        assert_eq!(funcs[0].instrs.len(), 2);
    }

    #[test]
    fn load_input_functions_splits_multi_function_dump() {
        let sample = r#"
        code for sm_89
        Function : first
        /*0000*/ MOV R0, RZ ;
        /*0010*/ EXIT ;
        Function : second
        /*0000*/ MOV R1, RZ ;
        /*0010*/ EXIT ;
        "#;
        let funcs = load_input_functions(sample);
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].name.as_deref(), Some("first"));
        assert_eq!(funcs[1].name.as_deref(), Some("second"));
    }

    #[test]
    fn select_functions_filters_requested_name() {
        let funcs = vec![
            InputFunction {
                name: Some("first".to_string()),
                sm: None,
                instrs: Vec::new(),
            },
            InputFunction {
                name: Some("second".to_string()),
                sm: None,
                instrs: Vec::new(),
            },
        ];
        let selected = select_functions(funcs, Some("second")).expect("selected function");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name.as_deref(), Some("second"));
    }

    #[test]
    fn select_functions_reports_available_names() {
        let funcs = vec![
            InputFunction {
                name: Some("first".to_string()),
                sm: None,
                instrs: Vec::new(),
            },
            InputFunction {
                name: Some("second".to_string()),
                sm: None,
                instrs: Vec::new(),
            },
        ];
        let err =
            select_functions(funcs, Some("missing")).expect_err("missing function should error");
        assert!(err.contains("missing"));
        assert!(err.contains("first"));
        assert!(err.contains("second"));
    }

    #[test]
    fn require_single_function_rejects_multi_function_cfg_dot() {
        let funcs = vec![
            InputFunction {
                name: Some("first".to_string()),
                sm: None,
                instrs: Vec::new(),
            },
            InputFunction {
                name: Some("second".to_string()),
                sm: None,
                instrs: Vec::new(),
            },
        ];
        let err = require_single_function(&funcs, OutputMode::CfgDot)
            .expect_err("cfg dot should require one function");
        assert!(err.contains("--function <name>"));
    }

    #[test]
    fn structured_output_uses_canonical_driver_for_named_functions() {
        let sample = r#"
        code for sm_89
        Function : first
        /*0000*/ MOV R0, RZ ;
        /*0010*/ EXIT ;
        Function : second
        /*0000*/ MOV R1, RZ ;
        /*0010*/ EXIT ;
        "#;
        let funcs = load_input_functions(sample);
        let rendered = build_multi_function_output(&funcs, &AbiProfileMode::Auto);
        assert!(rendered.contains("void first("), "missing first function:\n{rendered}");
        assert!(rendered.contains("void second("), "missing second function:\n{rendered}");
        assert!(
            !rendered.contains("__global__ void kernel"),
            "CLI structured output should use the canonical renderer, got:\n{rendered}"
        );
    }

    #[test]
    fn structured_output_keeps_canonical_pointer_rendering() {
        let function = InputFunction {
            name: Some("kernel".to_string()),
            sm: None,
            instrs: decode_sass(
                "/*0000*/ MOV R4, c[0x0][0x160] ;\n\
                 /*0010*/ MOV R5, c[0x0][0x164] ;\n\
                 /*0020*/ IADD3 R4, R4, 0x4, RZ ;\n\
                 /*0030*/ LDG.E R6, [R4.64] ;\n\
                 /*0040*/ MOV R8, c[0x0][0x168] ;\n\
                 /*0050*/ MOV R9, c[0x0][0x16c] ;\n\
                 /*0060*/ STG.E [R8.64], R6 ;\n\
                 /*0070*/ EXIT ;\n",
            ),
        };
        let rendered = build_function_output(&function, OutputMode::Structured, &AbiProfileMode::Auto);
        assert!(rendered.contains("arg0_ptr[1]"), "missing canonical pointer index:\n{rendered}");
        assert!(rendered.contains("// ABI arg aliases"), "missing ABI summary:\n{rendered}");
    }
}
