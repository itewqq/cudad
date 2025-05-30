//! 顶层 re‑export，方便外部调用。

pub mod parser;
pub mod cfg;
pub mod ir;
pub mod debug_util;
pub mod structurizer;
pub mod cfg_analysis;

#[cfg(test)]
mod test;

pub use parser::*;
pub use cfg::*;
pub use ir::*;
pub use structurizer::*;