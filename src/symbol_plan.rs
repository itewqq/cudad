//! Symbol and declaration planning scaffolding for the rewritten backend.
//!
//! Purpose:
//! - own deterministic parameter/local/shared declaration planning from the
//!   structured AST and analysis facts
//!
//! Inputs:
//! - `StructuredFunction`
//! - `FunctionAnalysis`
//!
//! Outputs:
//! - `SymbolPlan`, containing deterministic declaration and naming decisions
//!
//! Invariants:
//! - declarations are produced structurally, not inferred from rendered text
//! - naming is deterministic under identical AST traversal order
//!
//! This module must not:
//! - scan rendered output
//! - use regex fallback declaration synthesis

use crate::ast::Decl;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolPlan {
    pub params: Vec<Decl>,
    pub locals: Vec<Decl>,
}
