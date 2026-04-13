/// Regenerate curated golden files from the canonical full-pass backend.
/// Run with: cargo run --example regen_goldens
use cudad::*;
use std::fs;

fn run_structured_output_full_pass(sass: &str) -> String {
    let instrs = decode_sass(sass);
    if instrs.is_empty() {
        return "void kernel(void) {\n}\n".to_string();
    }

    let sm = parse_sm_version(sass);
    let cfg = build_cfg(instrs.clone());
    if cfg.node_count() == 0 {
        return "void kernel(void) {\n}\n".to_string();
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
            out.push_str("// Typed signature inferred from ABI aliases:\n");
            for line in aliases.summarize_lines(12) {
                out.push_str("// ");
                out.push_str(&line);
                out.push('\n');
            }
        }
    }

    let Some(tree) = structurizer.structure_function() else {
        out.push_str("// Failed to structure function or function is empty.\n");
        out.push_str("// --- End Structured Output ---\n");
        return out;
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
    let preview_output = structurizer.pretty_print_with_lift_cleanup(&tree, display_ctx, 0, Some(&lifted));
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
        &tree,
        display_ctx,
        0,
        Some(&lifted),
        &plan.token_map,
    );
    let symbols = filter_recovered_symbols_by_output(&named_output, &plan.symbols);
    let name_type_map = infer_recovered_name_types(&fir, &symbols);
    let final_output = render_typed_structured_output(
        &named_output,
        abi_aliases.as_ref(),
        &local_decls,
        Some(&symbols),
        &name_type_map,
        collect_shared_memory_decls(Some(&lifted)),
    );
    out.push_str(&final_output);
    out.push_str("// --- End Structured Output ---\n");
    out
}

fn main() {
    let cases = ["if", "if_loop", "loop_constant", "test_div", "rc4"];
    for name in &cases {
        let sass_path = format!("test_cu/{}.sass", name);
        let sass =
            fs::read_to_string(&sass_path).unwrap_or_else(|e| panic!("read {}: {}", sass_path, e));

        let out = run_structured_output_full_pass(&sass);
        let golden_path = format!("test_cu/golden_full_pass/{}.pseudo.c", name);
        fs::write(&golden_path, &out).unwrap();
        eprintln!("wrote {}", golden_path);
    }
    eprintln!("Regenerated canonical golden files.");
}
