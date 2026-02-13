//! 顶层 re‑export，方便外部调用。

pub mod parser;
pub mod cfg;
pub mod ir;
pub mod abi;
pub mod debug_util;
pub mod ast;
pub mod region;
pub mod structurizer;
pub mod cfg_analysis;
pub mod high_il;

#[cfg(test)]
mod test;

pub use parser::*;
pub use cfg::*;
pub use ir::*;
pub use abi::*;
pub use structurizer::*;
pub use cfg_analysis::*;
pub use high_il::*;
