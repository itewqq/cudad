//! Canonical full-pass backend driver scaffolding.
//!
//! Purpose:
//! - define the single owner of full-pass pipeline orchestration
//! - make later cutover possible without duplicating the full-pass backend in
//!   the CLI, tests, and examples
//!
//! Inputs:
//! - decoded instructions and optional SM version
//!
//! Outputs:
//! - stitched pipeline artifacts for the canonical backend
//!
//! Invariants:
//! - this module owns orchestration only; it must not embed ad hoc rendering
//!   heuristics or output-text repair logic
//! - later stages must consume analysis artifacts directly instead of
//!   rediscovering facts from rendered output
//!
//! Algorithm:
//! - phase-by-phase orchestration around CFG, SSA, analysis, structurization,
//!   AST lowering, symbol planning, and rendering
//!
//! This module must not:
//! - parse rendered output
//! - apply regex-based semantic fixes
//! - duplicate the old post-struct pipeline

use crate::ast::StructuredFunction;
use crate::ast_lowering::lower_structured_function;
use crate::abi::AbiProfile;
use crate::cfg::ControlFlowGraph;
use crate::function_analysis::FunctionAnalysis;
use crate::ir::FunctionIR;
use crate::parser::DecodedInstruction;
use crate::structurizer::{StructuredStatement, Structurizer};

#[derive(Clone, Debug, Default)]
pub struct DecompileArtifacts {
    pub function_name: Option<String>,
    pub cfg: Option<ControlFlowGraph>,
    pub optimized_ir: Option<FunctionIR>,
    pub analysis: Option<FunctionAnalysis>,
    pub structured: Option<StructuredStatement>,
    pub ast: Option<StructuredFunction>,
    pub rendered: Option<String>,
}

pub fn build_decompile_artifacts(
    instrs: Vec<DecodedInstruction>,
    sm: Option<u32>,
) -> DecompileArtifacts {
    build_named_decompile_artifacts_with_profile(instrs, sm, None, None)
}

pub fn build_named_decompile_artifacts(
    instrs: Vec<DecodedInstruction>,
    sm: Option<u32>,
    function_name: Option<&str>,
) -> DecompileArtifacts {
    build_named_decompile_artifacts_with_profile(instrs, sm, function_name, None)
}

pub fn build_named_decompile_artifacts_with_profile(
    instrs: Vec<DecodedInstruction>,
    sm: Option<u32>,
    function_name: Option<&str>,
    abi_profile: Option<AbiProfile>,
) -> DecompileArtifacts {
    let mut artifacts = DecompileArtifacts::default();
    if instrs.is_empty() {
        return artifacts;
    }
    let function_name = function_name.unwrap_or("kernel");
    let cfg = crate::build_cfg(instrs.clone());
    if cfg.node_count() == 0 {
        artifacts.function_name = Some(function_name.to_string());
        artifacts.cfg = Some(cfg);
        return artifacts;
    }
    let optimized_ir = {
        let ssa = crate::build_ssa(&cfg);
        let dce1 = crate::ir_dce(&ssa);
        let cp = crate::ir_constprop(&dce1);
        let alg = crate::ir_algebra(&cp);
        let cse = crate::ir_cse(&alg, &cfg);
        let copy = crate::ir_copyprop(&cse);
        crate::ir_dce(&copy)
    };
    let analysis = crate::analyze_function_ir_with_profile(&optimized_ir, &instrs, abi_profile, sm);
    let mut structurizer = Structurizer::new(&cfg, &optimized_ir);
    let structured = structurizer.structure_function();
    let ast = structured
        .as_ref()
        .map(|structured| lower_structured_function(structured, &analysis));
    let rendered = ast.as_ref().map(|function| function.render(function_name));
    artifacts.function_name = Some(function_name.to_string());
    artifacts.cfg = Some(cfg);
    artifacts.optimized_ir = Some(optimized_ir);
    artifacts.analysis = Some(analysis);
    artifacts.structured = structured;
    artifacts.ast = ast;
    artifacts.rendered = rendered;
    artifacts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_sass, AbiProfile};
    use regex::Regex;

    #[test]
    fn builds_pipeline_artifacts_for_memory_aware_samples() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E R6, [R4.64+0x4] ;
            /*0030*/ STS [R2+0x8], R6 ;
            /*0040*/ EXIT ;
        "#;
        let artifacts = build_decompile_artifacts(decode_sass(sass), None);
        assert!(artifacts.cfg.is_some());
        assert!(artifacts.optimized_ir.is_some());
        assert!(artifacts.analysis.is_some());
        assert!(artifacts.structured.is_some());
        assert!(artifacts.ast.is_some());
        assert!(artifacts.rendered.is_some());
    }

    #[test]
    fn rendered_output_is_produced_alongside_memory_analysis() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E R6, [R4.64+0x4] ;
            /*0030*/ MOV R8, c[0x0][0x168] ;
            /*0040*/ MOV R9, c[0x0][0x16c] ;
            /*0050*/ STG.E [R8.64], R6 ;
            /*0060*/ EXIT ;
        "#;
        let artifacts = build_decompile_artifacts(decode_sass(sass), None);
        let analysis = artifacts.analysis.expect("analysis");
        assert!(
            analysis
                .mem_accesses
                .iter()
                .any(|access| matches!(access.space, crate::memory_model::CudaMemorySpace::Global))
        );
        assert!(artifacts.rendered.is_some());
    }

    #[test]
    fn rendered_output_uses_param_symbols_without_raw_ssa_tokens() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ IADD3 R4, R4, 0x4, RZ ;
            /*0030*/ LDG.E R6, [R4.64] ;
            /*0040*/ STS [R2], R6 ;
            /*0050*/ EXIT ;
        "#;
        let rendered = build_decompile_artifacts(decode_sass(sass), None)
            .rendered
            .expect("rendered output");
        let raw_ssa = Regex::new(r"\b(?:R|UR|P|UP)\d+\.\d+\b").expect("raw ssa regex");
        assert!(rendered.contains("arg0_ptr[1]"));
        assert!(!raw_ssa.is_match(&rendered), "rendered output leaked raw SSA tokens:\n{rendered}");
    }

    #[test]
    fn rendered_output_keeps_shared_byte_offsets_as_indices() {
        let sass = r#"
            /*0000*/ STS [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let rendered = build_decompile_artifacts(decode_sass(sass), None)
            .rendered
            .expect("rendered output");
        assert!(rendered.contains("shmem[(r2_0 + 8) / 4] = r4_0;"));
    }

    #[test]
    fn renders_with_explicit_function_names_when_provided() {
        let sass = r#"
            /*0000*/ MOV R0, RZ ;
            /*0010*/ EXIT ;
        "#;
        let rendered = build_named_decompile_artifacts(decode_sass(sass), None, Some("named_kernel"))
            .rendered
            .expect("rendered output");
        assert!(rendered.starts_with("void named_kernel("), "unexpected render:\n{rendered}");
    }

    #[test]
    fn explicit_abi_profile_override_is_threaded_into_analysis() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
            /*0010*/ EXIT ;
        "#;
        let artifacts = build_named_decompile_artifacts_with_profile(
            decode_sass(sass),
            None,
            Some("kernel"),
            Some(AbiProfile::modern_param_160()),
        );
        let analysis = artifacts.analysis.expect("analysis");
        assert_eq!(analysis.abi_profile, Some(AbiProfile::modern_param_160()));
    }
}
