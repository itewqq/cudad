use cudad::*;
use clap::Parser;
use regex::Regex;
use std::fs;
use std::sync::OnceLock;

/// Embedded demo SASS for quick testing without --input.
const SAMPLE_SASS: &str = include_str!("../../test_cu/sample_verify_kernel.sass");

// ---------------------------------------------------------------------------
// Cached regex objects
// ---------------------------------------------------------------------------

fn main_reg_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b((?:U?R|U?P)\d+)\b").unwrap())
}

fn main_ident_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap())
}

fn main_temp_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\b(?:v\d+|u\d+|b\d+|abi_internal_0x[0-9A-Fa-f]+|arg\d+_(?:ptr_)?(?:lo32|hi32|word\d+)(?:_\d+)?|tid_[xyz](?:_\d+)?|ctaid_[xyz](?:_\d+)?|param_\d+|block_dim_[xyz](?:_\d+)?|grid_dim_[xyz](?:_\d+)?|lane_id(?:_\d+)?|cga_cta_id(?:_\d+)?)\b",
        )
        .unwrap()
    })
}

fn main_assign_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:if\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=").unwrap()
    })
}

fn render_typed_structured_output(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
    local_decls: &[String],
    name_type_map: &std::collections::HashMap<String, &str>,
) -> String {
    let params = aliases
        .map(|a| a.render_typed_param_list())
        .unwrap_or_default();
    let sig = if params.is_empty() {
        "__global__ void kernel(void)".to_string()
    } else {
        format!("__global__ void kernel({})", params.join(", "))
    };

    // Filter out declarations for registers that no longer appear in the
    // code output (e.g., raw R0/P0 after name recovery renamed them).
    let reg_name = main_reg_name_re();
    let used_decls: Vec<&str> = local_decls
        .iter()
        .filter(|d| {
            if let Some(cap) = reg_name.captures(d) {
                let reg_name = cap.get(1).unwrap().as_str();
                // Check if this register name appears in the code body (as a word boundary match)
                let pat = format!(r"\b{}\b", regex::escape(reg_name));
                Regex::new(&pat).map_or(false, |re| re.is_match(code_output))
            } else {
                true // keep non-register declarations
            }
        })
        .map(|s| s.as_str())
        .collect();

    let mut out = String::new();
    out.push_str(&sig);
    out.push_str(" {\n");
    for d in &used_decls {
        out.push_str("  ");
        out.push_str(d);
        out.push('\n');
    }
    let extra_decls = infer_self_contained_locals(code_output, aliases, local_decls, name_type_map);
    for d in &extra_decls {
        out.push_str("  ");
        out.push_str(d);
        out.push('\n');
    }
    if !used_decls.is_empty() || !extra_decls.is_empty() {
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

fn infer_self_contained_locals(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
    local_decls: &[String],
    name_type_map: &std::collections::HashMap<String, &str>,
) -> Vec<String> {
    let ident = main_ident_re();
    let temp = main_temp_re();
    let assign = main_assign_re();
    let mut declared = std::collections::BTreeSet::<String>::new();
    for d in local_decls {
        for m in ident.find_iter(d) {
            declared.insert(m.as_str().to_string());
        }
    }
    if let Some(a) = aliases {
        for p in a.render_typed_param_list() {
            for m in ident.find_iter(&p) {
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
        let lhs_span = assign.captures(line).and_then(|c| {
            let whole = c.get(0)?;
            let after_eq = line.get(whole.end()..).unwrap_or("").trim_start();
            if after_eq.starts_with('=') {
                return None;
            }
            c.get(1)
                .map(|m| (m.as_str().to_string(), m.start(), m.end()))
        });
        // Collect RHS uses of temp variables.
        for m in temp.find_iter(line) {
            let t = m.as_str();
            let is_lhs = lhs_span
                .as_ref()
                .map(|(_, s, e)| m.start() == *s && m.end() == *e)
                .unwrap_or(false);
            if !is_lhs && !defined.contains(t) {
                // Only mark as live-in if this use is NOT on a line that also
                // defines this same variable (i.e. self-referential assignments
                // like `v21 = -v21` are really loop-carried phi definitions,
                // not genuine live-in reads).
                let lhs_name = lhs_span.as_ref().map(|(n, _, _)| n.as_str());
                if lhs_name != Some(t) {
                    live_ins.insert(t.to_string());
                }
            }
            if declared.contains(t) {
                continue;
            }
            if seen.insert(t.to_string()) {
                ordered.push(t.to_string());
            }
        }
        if let Some((lhs, _, _)) = lhs_span {
            if temp.is_match(&lhs) {
                defined.insert(lhs);
            }
        }
    }

    let mut out = Vec::<String>::new();
    if code_output.contains("shmem_u8[") && !declared.contains("shmem_u8") {
        out.push("__shared__ uint8_t shmem_u8[256];".to_string());
    }
    if code_output.contains("shmem[") && !declared.contains("shmem") {
        out.push("__shared__ uint32_t shmem[];".to_string());
    }
    for name in ordered {
        let is_bool = name.starts_with('b')
            && name[1..].chars().all(|c| c.is_ascii_digit());
        let is_live_in = live_ins.contains(&name);
        // Use IR-level type inference if available for this recovered name.
        let ty = if is_bool {
            "bool"
        } else if let Some(inferred) = name_type_map.get(name.as_str()) {
            inferred
        } else {
            "uint32_t"
        };
        if is_live_in {
            out.push(format!("{} {}; // live-in", ty, name));
        } else {
            out.push(format!("{} {};", ty, name));
        }
    }
    out
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum AbiProfileMode {
    Auto,
    Legacy140,
    Modern160,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum NameStyleMode {
    Temp,
    Reg,
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
    /// Enable conservative semantic expression lifting in structured output
    #[clap(long)]
    semantic_lift: bool,
    /// Emit heuristic typed argument/local declarations in structured output
    #[clap(long)]
    typed_decls: bool,
    /// Recover SSA-style names into C-like temporary names in structured output
    #[clap(long)]
    recover_names: bool,
    /// Name style used by --recover-names (`temp|reg`)
    #[clap(long, value_enum, default_value = "temp")]
    name_style: NameStyleMode,
    /// Emit comment-only phi/live-in merge annotations in recovered-name mode
    #[clap(long)]
    phi_merge_comments: bool,
    /// Rewrite recovered temp vars using semantic seed names when possible
    #[clap(long)]
    semantic_symbolize: bool,
    /// Force ABI profile used by `--abi-map` (`auto|legacy140|modern160`)
    #[clap(long, value_enum, default_value = "auto")]
    abi_profile: AbiProfileMode,
}



// ---------------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------------

/// Load SASS source from file or fall back to the embedded demo sample.
fn load_sass(input: Option<&str>) -> String {
    match input {
        Some(path) => fs::read_to_string(path).expect("Failed to read input file"),
        None => SAMPLE_SASS.to_string(),
    }
}

/// Select the ABI profile from explicit CLI choice or auto-detection.
fn resolve_abi_profile(
    args: &Args,
    instrs: &[Instruction],
    sm_version: Option<u32>,
) -> Option<AbiProfile> {
    let inferred = AbiProfile::detect_with_sm(instrs, sm_version);
    if args.abi_map || !matches!(args.abi_profile, AbiProfileMode::Auto) {
        Some(match args.abi_profile {
            AbiProfileMode::Auto => inferred,
            AbiProfileMode::Legacy140 => AbiProfile::legacy_param_140(),
            AbiProfileMode::Modern160 => AbiProfile::modern_param_160(),
        })
    } else {
        None
    }
}

/// Emit an SSA DOT graph to `--output` or stdout.
fn emit_ssa_dot(cfg: &ControlFlowGraph, abi_profile: Option<AbiProfile>, output: Option<&str>) {
    let fir = {
        let ssa = build_ssa(cfg);
        let dce1 = ir_dce(&ssa);
        let cp = ir_constprop(&dce1);
        let cse = ir_cse(&cp, cfg);
        ir_dce(&cse)
    };
    let default_display = DefaultDisplay;
    let abi_display = abi_profile.map(|profile| {
        let anns = annotate_function_ir_constmem(&fir, profile);
        let aliases = infer_arg_aliases(&fir, &anns);
        AbiDisplay::with_aliases(profile, aliases)
    });
    let display_ctx: &dyn DisplayCtx = abi_display
        .as_ref()
        .map(|d| d as &dyn DisplayCtx)
        .unwrap_or(&default_display);
    let dot = fir.to_dot(cfg, display_ctx);
    emit_output(&dot, output);
}

/// Run the full structured-code pipeline: SSA → structurize → lift → name → typed output.
fn emit_struct_code(cfg: &ControlFlowGraph, args: &Args, abi_profile: Option<AbiProfile>) {
    let fir = {
        let ssa = build_ssa(cfg);
        let dce1 = ir_dce(&ssa);
        let cp = ir_constprop(&dce1);
        let cse = ir_cse(&cp, cfg);
        ir_dce(&cse)
    };

    // Resolve ABI annotations & aliases (needed for --abi-map, --typed-decls, --semantic-lift).
    let inferred = AbiProfile::detect_with_sm(&[], None);
    let analysis_profile = if abi_profile.is_some() || args.typed_decls {
        abi_profile.or(Some(inferred))
    } else {
        None
    };
    let abi_annotations = analysis_profile.map(|p| annotate_function_ir_constmem(&fir, p));
    let abi_aliases = match (analysis_profile, abi_annotations.as_ref()) {
        (Some(_), Some(anns)) => Some(infer_arg_aliases(&fir, anns)),
        _ => None,
    };

    println!("// --- Structured Output ---");

    // Optional: print ABI mapping summary.
    if args.abi_map {
        print_abi_summary(&abi_annotations, &abi_aliases);
    }
    // Only print typed signature summary if --abi-map didn't already print the aliases.
    if args.typed_decls && !args.abi_map {
        if let Some(aliases) = &abi_aliases {
            if !aliases.is_empty() {
                println!("// Typed signature inferred from ABI aliases:");
                for line in aliases.summarize_lines(12) {
                    println!("// {}", line);
                }
            }
        }
    }

    // Structurize.
    let mut structurizer_instance = structurizer::Structurizer::new(cfg, &fir);
    let Some(structured_body) = structurizer_instance.structure_function() else {
        println!("// Failed to structure function or function is empty.");
        println!("// --- End Structured Output ---");
        return;
    };

    // Build display context.
    let default_display = DefaultDisplay;
    let abi_display = match (abi_profile, abi_aliases.clone()) {
        (Some(profile), Some(aliases)) => Some(AbiDisplay::with_aliases(profile, aliases)),
        (Some(profile), None) => Some(AbiDisplay::new(profile)),
        (None, _) => None,
    };
    let display_ctx: &dyn DisplayCtx = abi_display
        .as_ref()
        .map(|d| d as &dyn DisplayCtx)
        .unwrap_or(&default_display);

    // Semantic lift (optional).
    let lift_result = if args.semantic_lift {
        let lift_cfg = SemanticLiftConfig {
            abi_annotations: abi_annotations.as_ref(),
            abi_aliases: abi_aliases.as_ref(),
            strict: true,
        };
        Some(lift_function_ir(&fir, &lift_cfg))
    } else {
        None
    };

    // Pretty-print → name recovery → typed declarations.
    let code_output = structurizer_instance.pretty_print_with_lift(
        &structured_body,
        display_ctx,
        0,
        lift_result.as_ref(),
    );
    let (recovered_output, name_to_reg_base) = if args.recover_names {
        let style = match args.name_style {
            NameStyleMode::Temp => NameStyle::Temp,
            NameStyleMode::Reg => NameStyle::RegisterFamily,
        };
        let recover_cfg = NameRecoveryConfig {
            style,
            rewrite_control_predicates: true,
            emit_phi_merge_comments: args.phi_merge_comments,
            semantic_symbolization: args.semantic_symbolize,
        };
        let result = recover_structured_output_names(&fir, &code_output, &recover_cfg);
        // Dead-code elimination: remove assignments whose LHS is never used.
        let output = eliminate_dead_code(&result.output);
        (output, result.name_to_reg_base)
    } else {
        (code_output, std::collections::HashMap::new())
    };
    let local_decls = if args.typed_decls {
        infer_local_typed_declarations_with_abi(
            &fir,
            abi_annotations.as_ref(),
            abi_aliases.as_ref(),
        )
    } else {
        Vec::new()
    };
    let final_output = if args.typed_decls {
        // Build a type map from recovered names to C type strings using
        // the name→register mapping from name recovery + IR type inference.
        let df_types = type_inference::infer_types(&fir);
        let name_type_map: std::collections::HashMap<String, &str> = name_to_reg_base
            .iter()
            .filter_map(|(name, key)| {
                df_types.get(key).map(|t| (name.clone(), t.to_c_type()))
            })
            .collect();
        render_typed_structured_output(&recovered_output, abi_aliases.as_ref(), &local_decls, &name_type_map)
    } else {
        recovered_output
    };

    emit_output(&final_output, args.output.as_deref());
    println!("// --- End Structured Output ---");
}

fn print_abi_summary(
    abi_annotations: &Option<AbiAnnotations>,
    abi_aliases: &Option<AbiArgAliases>,
) {
    if let Some(anns) = abi_annotations {
        if !anns.is_empty() {
            println!("// ABI const-memory mapping (sample):");
            for line in anns.summarize_lines(16) {
                println!("// {}", line);
            }
        }
    }
    if let Some(aliases) = abi_aliases {
        if !aliases.is_empty() {
            println!("// ABI arg aliases (heuristic):");
            for line in aliases.summarize_lines(12) {
                println!("// {}", line);
            }
        }
    }
}

/// Write `content` to a file or stdout.
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
    let stub = "\
// Warning: no parseable SASS instruction lines were found.\n\
// Returning an empty stub to keep the pipeline non-fatal.\n\
__global__ void kernel(void) {\n\
}\n";
    println!("// --- Structured Output ---");
    emit_output(stub, output);
    println!("// --- End Structured Output ---");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let mut args = Args::parse();

    // Default to full decompilation when no mode flag is given.
    if !args.cfg_dot && !args.ssa_dot && !args.struct_code {
        args.struct_code = true;
        args.abi_map = true;
        args.semantic_lift = true;
        args.typed_decls = true;
        args.recover_names = true;
        args.semantic_symbolize = true;
    }

    let sass = load_sass(args.input.as_deref());
    let instrs = parse_sass(&sass);
    let sm_version = parse_sm_version(&sass);
    let abi_profile = resolve_abi_profile(&args, &instrs, sm_version);
    let cfg = build_cfg(instrs);

    if args.cfg_dot {
        println!("{}", graph_to_dot(&cfg));
    }
    if args.ssa_dot {
        emit_ssa_dot(&cfg, abi_profile, args.output.as_deref());
    }
    if args.struct_code {
        if cfg.node_count() == 0 {
            emit_empty_stub(args.output.as_deref());
        } else {
            emit_struct_code(&cfg, &args, abi_profile);
        }
    }
}
