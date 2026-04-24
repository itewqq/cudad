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
use crate::cfg::ControlFlowGraph;
use crate::function_analysis::FunctionAnalysis;
use crate::ir::FunctionIR;
use crate::parser::DecodedInstruction;
use crate::structurizer::StructuredStatement;

#[derive(Clone, Debug, Default)]
pub struct DecompileArtifacts {
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
    let mut artifacts = DecompileArtifacts::default();
    if instrs.is_empty() {
        return artifacts;
    }
    let cfg = crate::build_cfg(instrs.clone());
    if cfg.node_count() == 0 {
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
    let analysis = crate::analyze_function_ir(&optimized_ir, &instrs, sm);
    artifacts.cfg = Some(cfg);
    artifacts.optimized_ir = Some(optimized_ir);
    artifacts.analysis = Some(analysis);
    artifacts
}
