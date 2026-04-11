//! 顶层 re‑export，方便外部调用。

pub mod parser;
pub mod cfg;
pub mod ir;
pub mod abi;
pub mod debug_util;
pub mod structurizer;
pub mod cfg_analysis;
pub mod semantic_lift;
pub mod name_recovery;
pub mod op_semantics;
pub mod dce;
pub mod ir_dce;
pub mod ir_constprop;
pub mod ir_copyprop;
pub mod ir_cse;
pub mod ir_algebra;
pub mod type_inference;
pub mod semantic_propagation;
pub mod ptr_verification;

#[cfg(test)]
mod test;

pub use parser::{
    parse_instruction_line, parse_sass, parse_sm_version, split_functions, Instruction, Operand,
    PredicateUse, SassFunction,
};
pub use cfg::*;
pub use ir::*;
pub use abi::*;
pub use structurizer::*;
pub use cfg_analysis::*;
pub use semantic_lift::*;
pub use name_recovery::*;
pub use op_semantics::*;
pub use dce::*;
pub use ir_dce::*;
pub use ir_constprop::*;
pub use ir_copyprop::*;
pub use ir_cse::*;
pub use ir_algebra::*;
pub use type_inference::*;
pub use semantic_propagation::*;
pub use ptr_verification::*;
