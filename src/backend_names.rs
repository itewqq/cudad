//! Canonical identifier planning for the rewritten backend.
//!
//! Purpose:
//! - provide deterministic, C/CUDA-safe names for SSA registers emitted by the
//!   new memory-aware pipeline
//! - centralize predicate-name classification so symbol planning and lowering
//!   stay aligned
//!
//! Inputs:
//! - SSA `RegId`s
//! - rendered identifier strings that may correspond to canonical predicate
//!   registers
//!
//! Outputs:
//! - legal identifiers that preserve register class/index/SSA version
//! - predicate-name classification helpers for declaration typing
//!
//! Invariants:
//! - identical `RegId`s always map to identical identifiers
//! - the mapping is syntax-safe for C/CUDA source
//! - predicate registers remain recognizable after renaming
//!
//! Algorithm summary:
//! - normalize the register class to a lowercase ASCII identifier fragment
//! - append the hardware register index and optional SSA version
//! - classify predicate names structurally instead of by rendered text regex
//!
//! This module must not:
//! - inspect rendered pseudo-C output
//! - infer semantic names from usage

use crate::ir::RegId;

pub fn canonical_reg_ident(reg: &RegId) -> String {
    let class = sanitize_class(&reg.class);
    let base = match reg.class.as_str() {
        "RZ" | "PT" | "URZ" | "UPT" => class,
        _ if reg.idx < 0 => format!("{}n{}", class, reg.idx.unsigned_abs()),
        _ => format!("{}{}", class, reg.idx),
    };
    match reg.ssa {
        Some(ssa) => format!("{}_{}", base, ssa),
        None => base,
    }
}

pub fn is_predicate_ident(name: &str) -> bool {
    predicate_suffix(name.strip_prefix("p")).is_some()
        || predicate_suffix(name.strip_prefix("up")).is_some()
}

fn predicate_suffix(rest: Option<&str>) -> Option<()> {
    let rest = rest?;
    let (head, tail) = rest.split_once('_').unwrap_or((rest, ""));
    if head.is_empty() || !head.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    if tail.is_empty() {
        return Some(());
    }
    tail.chars().all(|ch| ch.is_ascii_digit()).then_some(())
}

fn sanitize_class(class: &str) -> String {
    let mut out = String::with_capacity(class.len() + 1);
    for ch in class.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || !out.chars().next().is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_') {
        out.insert(0, 'r');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_ssa_registers_to_c_identifiers() {
        let reg = RegId::new("R", 4, 1).with_ssa(2);
        assert_eq!(canonical_reg_ident(&reg), "r4_2");
    }

    #[test]
    fn canonicalizes_uniform_predicates() {
        let reg = RegId::new("UP", 3, 1);
        assert_eq!(canonical_reg_ident(&reg), "up3");
        assert!(is_predicate_ident("up3"));
    }

    #[test]
    fn recognizes_canonical_predicate_names_only() {
        assert!(is_predicate_ident("p0_1"));
        assert!(is_predicate_ident("up7"));
        assert!(!is_predicate_ident("param_0"));
        assert!(!is_predicate_ident("r4_0"));
    }
}
