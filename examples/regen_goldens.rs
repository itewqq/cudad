/// Regenerate curated golden files from the canonical full-pass backend.
/// Run with: cargo run --example regen_goldens
use cudad::{build_named_decompile_artifacts, decode_sass, parse_sm_version};
use std::fs;

fn render_canonical_output(sass: &str) -> String {
    let instrs = decode_sass(sass);
    if instrs.is_empty() {
        return "void kernel(void) {\n}\n".to_string();
    }
    build_named_decompile_artifacts(instrs, parse_sm_version(sass), Some("kernel"))
        .rendered
        .unwrap_or_else(|| "void kernel(void) {\n}\n".to_string())
}

fn main() {
    let cases = ["if", "if_loop", "loop_constant", "test_div", "rc4"];
    for name in &cases {
        let sass_path = format!("test_cu/{}.sass", name);
        let sass =
            fs::read_to_string(&sass_path).unwrap_or_else(|e| panic!("read {}: {}", sass_path, e));

        let out = render_canonical_output(&sass);
        let golden_path = format!("test_cu/golden_full_pass/{}.pseudo.c", name);
        fs::write(&golden_path, &out).unwrap_or_else(|e| panic!("write {}: {}", golden_path, e));
        eprintln!("wrote {}", golden_path);
    }
    eprintln!("Regenerated canonical golden files.");
}
