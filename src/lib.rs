//! Top-level re-export for external callers.

pub mod abi;
pub mod ast;
pub mod ast_lowering;
pub mod backend_names;
pub mod ast_passes;
pub mod backend_pipeline;
pub mod canonical_ast_passes;
pub mod cfg;
pub mod cfg_analysis;
pub mod debug_util;
pub mod function_analysis;
pub mod ir;
pub mod ir_algebra;
pub mod ir_constprop;
pub mod ir_copyprop;
pub mod ir_cse;
pub mod ir_dce;
pub mod name_recovery;
pub mod memory_model;
pub mod op_semantics;
pub mod parser;
pub mod ptr_verification;
pub mod semantic_lift;
pub mod semantic_propagation;
pub mod symbol_plan;
pub mod structurizer;
pub mod type_inference;
pub mod typed_output;

#[cfg(test)]
mod test;

pub use abi::*;
pub use ast::*;
pub use ast_lowering::*;
pub use backend_names::*;
pub use ast_passes::*;
pub use backend_pipeline::*;
pub use canonical_ast_passes::*;
pub use cfg::*;
pub use cfg_analysis::*;
pub use function_analysis::*;
pub use ir::*;
pub use ir_algebra::*;
pub use ir_constprop::*;
pub use ir_copyprop::*;
pub use ir_cse::*;
pub use ir_dce::*;
pub use name_recovery::*;
pub use memory_model::*;
pub use op_semantics::*;
pub use parser::{
    decode_instruction_line, decode_sass, parse_sm_version, split_decoded_functions,
    DecodedFunction, DecodedInstruction, DecodedOperand, Predicate, PredicateUse, SchedulingInfo,
    TerminatorKind,
};
pub use ptr_verification::*;
pub use semantic_lift::*;
pub use semantic_propagation::*;
pub use symbol_plan::*;
pub use structurizer::*;
pub use type_inference::*;
pub use typed_output::*;
