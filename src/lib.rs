//! 顶层 re‑export，方便外部调用。

pub mod parser;
pub mod cfg;
pub mod ir;

#[cfg(test)]
mod test;

pub use parser::*;
pub use cfg::*;
pub use ir::*;