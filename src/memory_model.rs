//! Shared CUDA memory model vocabulary for the rewritten backend.
//!
//! Purpose:
//! - define stable enums shared by SSA analysis, AST lowering, declaration
//!   planning, and rendering
//!
//! Invariants:
//! - this module contains vocabulary only
//! - it must not depend on rendering, regexes, or backend repair logic

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
