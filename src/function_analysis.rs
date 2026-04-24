//! Memory-aware SSA analysis contract for the rewritten backend.
//!
//! Purpose:
//! - own all post-SSA facts needed by structurization consumers and AST
//!   lowering
//! - centralize CUDA memory-space classification, pointer-root recovery,
//!   ABI/profile facts, and declaration-driving type facts
//!
//! Inputs:
//! - optimized `FunctionIR`
//! - optional decoded-function metadata such as SM version
//!
//! Outputs:
//! - `FunctionAnalysis`, the canonical fact base for the post-SSA backend
//!
//! Invariants:
//! - later stages must consume these facts directly
//! - no later stage may parse rendered text to recover memory, type, or naming
//!   facts already represented here
//!
//! Algorithm summary:
//! - seed memory spaces from opcode families and operand forms
//! - refine with ABI/profile facts
//! - propagate address roots and pointer facts with SSA worklists
//! - synthesize shared/local objects from compatible proven roots
//!
//! This module must not:
//! - render pseudo-C
//! - mutate CFG/SSA structure
//! - depend on post-render cleanup behavior

use crate::abi::{AbiAnnotations, AbiArgAliases, AbiProfile};
use crate::ir::FunctionIR;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CudaMemorySpace {
    Param,
    Const,
    Global,
    Local,
    Shared,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemAccessKind {
    Load,
    Store,
    Atomic,
    Reduction,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddressRoot {
    ParamWord(u32),
    ConstSymbol(String),
    SharedObject(String),
    LocalObject(String),
    RegisterBase(String),
    Generic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemAccessInfo {
    pub block_id: usize,
    pub stmt_idx: usize,
    pub kind: MemAccessKind,
    pub space: CudaMemorySpace,
    pub bit_width: Option<u32>,
    pub vector_width: Option<u8>,
    pub root: AddressRoot,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionAnalysis {
    pub abi_profile: Option<AbiProfile>,
    pub abi_annotations: AbiAnnotations,
    pub abi_aliases: AbiArgAliases,
    pub mem_accesses: Vec<MemAccessInfo>,
}

pub fn analyze_function_ir(_function_ir: &FunctionIR, _sm: Option<u32>) -> FunctionAnalysis {
    FunctionAnalysis::default()
}
