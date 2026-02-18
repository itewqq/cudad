/// Regenerate all golden files from current code.
/// Run with: cargo run --example regen_goldens

use cudad::*;
use regex::Regex;
use std::collections::BTreeSet;
use std::fs;

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

fn render_typed_structured_output(
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
    let extra_decls = infer_self_contained_locals(code_output, aliases, local_decls);
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

fn infer_self_contained_locals(
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
    let mut declared = BTreeSet::<String>::new();
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

    let mut seen = BTreeSet::<String>::new();
    let mut ordered = Vec::<String>::new();
    let mut defined = BTreeSet::<String>::new();
    let mut live_ins = BTreeSet::<String>::new();
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
            #[allow(clippy::collapsible_else_if)]
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
        let final_output = render_typed_structured_output(
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

fn main() {
    let cases = ["if", "if_loop", "loop_constant", "test_div", "rc4"];
    for name in &cases {
        let sass_path = format!("test_cu/{}.sass", name);
        let sass = fs::read_to_string(&sass_path).unwrap_or_else(|e| panic!("read {}: {}", sass_path, e));

        // golden/ (raw structured)
        let out = run_structured_output(&sass);
        let golden_path = format!("test_cu/golden/{}.pseudo.c", name);
        fs::write(&golden_path, &out).unwrap();
        eprintln!("wrote {}", golden_path);

        // golden_lifted/
        let out = run_structured_output_lifted(&sass);
        let golden_path = format!("test_cu/golden_lifted/{}.pseudo.c", name);
        fs::write(&golden_path, &out).unwrap();
        eprintln!("wrote {}", golden_path);

        // golden_lifted_named/
        let out = run_structured_output_lifted_named(&sass);
        let golden_path = format!("test_cu/golden_lifted_named/{}.pseudo.c", name);
        fs::write(&golden_path, &out).unwrap();
        eprintln!("wrote {}", golden_path);

        // golden_full_pass/
        let out = run_structured_output_full_pass(&sass);
        let golden_path = format!("test_cu/golden_full_pass/{}.pseudo.c", name);
        fs::write(&golden_path, &out).unwrap();
        eprintln!("wrote {}", golden_path);
    }
    eprintln!("All golden files regenerated.");
}
