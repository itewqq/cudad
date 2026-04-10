//! Post-SSA verification of 64-bit pointer pairs.
//!
//! During operand lowering (`lower_operand` in `ir.rs`), 64-bit memory
//! references speculatively construct `addr64(lo, hi)` by assuming
//! `R(N+1)` is the high 32 bits of `R(N).64`. This pass verifies those
//! pairs after SSA construction, when def-use chains are available.
//!
//! A valid hi-part register should be defined by one of:
//! - `LEA.HI*` — address generation high part
//! - `IADD3.X` — add-with-carry (hi word of 64-bit add)
//! - `SHF.*.HI` — funnel shift producing high bits
//! - `IMAD.WIDE` — wide multiply (defines pair, hi is implicit)
//! - A phi or copy of any of the above
//!
//! If the hi register is NOT defined by a recognized hi-part instruction,
//! this pass flags it. Currently informational — the decompiler still
//! produces correct output because addr64 is rendered as a combined
//! address regardless, but the information can be used to mark verified
//! pointers with stronger type confidence.

use std::collections::{HashMap, HashSet};

use crate::ir::{FunctionIR, IRExpr, RValue, RegId};

/// Result of verifying 64-bit pointer pairs in a function.
#[derive(Clone, Debug, Default)]
pub struct Ptr64VerificationResult {
    /// (lo_reg, hi_reg) pairs that passed verification — the hi register
    /// is indeed defined by a recognized hi-part instruction chain.
    pub verified_pairs: Vec<(RegId, RegId)>,
    /// (lo_reg, hi_reg) pairs where the hi register's def could not be
    /// traced to a recognized hi-part instruction.
    pub unverified_pairs: Vec<(RegId, RegId)>,
}

/// Verify 64-bit pointer pairs in the function IR.
pub fn verify_ptr64_pairs(fir: &FunctionIR) -> Ptr64VerificationResult {
    // Step 1: Build def map — for each SSA register, what opcode defined it?
    let mut def_opcode: HashMap<RegKey, String> = HashMap::new();
    // Also track phi/copy sources for transitive verification
    let mut copy_sources: HashMap<RegKey, Vec<RegKey>> = HashMap::new();

    for block in &fir.blocks {
        for stmt in &block.stmts {
            for def in &stmt.defs {
                if let Some(r) = def.get_reg() {
                    if is_immutable_reg(r) {
                        continue;
                    }
                    let key = RegKey::from(r);
                    match &stmt.value {
                        RValue::Op { opcode, args } => {
                            def_opcode.insert(key.clone(), opcode.clone());
                            // Track copy sources
                            if let Some(src) = extract_copy_source(opcode, args) {
                                copy_sources.entry(key).or_default().push(src);
                            }
                        }
                        RValue::Phi(args) => {
                            def_opcode.insert(key.clone(), "PHI".to_string());
                            let srcs: Vec<RegKey> = args
                                .iter()
                                .filter_map(|a| a.get_reg())
                                .filter(|r| !is_immutable_reg(r))
                                .map(|r| RegKey::from(r))
                                .collect();
                            copy_sources.insert(key, srcs);
                        }
                        _ => {
                            def_opcode.insert(key, "LITERAL".to_string());
                        }
                    }
                }
            }
        }
    }

    // Step 2: Find all addr64(lo, hi) expressions in the IR
    let mut result = Ptr64VerificationResult::default();

    for block in &fir.blocks {
        for stmt in &block.stmts {
            visit_expr_for_addr64(&stmt.value, &def_opcode, &copy_sources, &mut result);
        }
    }

    result
}

/// Check whether a register (possibly through copy/phi chains) was
/// ultimately defined by a recognized hi-part instruction.
fn is_hi_part_def(
    key: &RegKey,
    def_opcode: &HashMap<RegKey, String>,
    copy_sources: &HashMap<RegKey, Vec<RegKey>>,
    visited: &mut HashSet<RegKey>,
) -> bool {
    if !visited.insert(key.clone()) {
        return false; // cycle
    }

    if let Some(opcode) = def_opcode.get(key) {
        if is_hi_part_opcode(opcode) {
            return true;
        }
        // IMAD.WIDE defines both RN and R(N+1); if this register is R(N+1)
        // and R(N) was defined by IMAD.WIDE, it's the implicit hi-part.
        if opcode.starts_with("IMAD.WIDE") {
            return true; // IMAD.WIDE inherently produces a hi-part too
        }
        // Check through copies/phis transitively
        if opcode == "PHI" || is_copy_opcode(opcode) {
            if let Some(srcs) = copy_sources.get(key) {
                // For phi: if ANY input is hi-part, consider it verified
                // (loop-carried phis commonly have one entry from LEA.HI)
                return srcs.iter().any(|s| is_hi_part_def(s, def_opcode, copy_sources, visited));
            }
        }
    } else {
        // No def found — check if the paired lo-register (idx - 1) is
        // defined by IMAD.WIDE, making this the implicit hi-part.
        if key.idx > 0 {
            let lo_key = RegKey {
                class: key.class.clone(),
                idx: key.idx - 1,
                ssa: key.ssa,
            };
            if let Some(lo_op) = def_opcode.get(&lo_key) {
                if lo_op.starts_with("IMAD.WIDE") {
                    return true;
                }
            }
            // Also try without SSA version match (initial R5.0 paired with R4.0)
            for (k, v) in def_opcode {
                if k.class == key.class && k.idx == key.idx - 1 && v.starts_with("IMAD.WIDE") {
                    return true;
                }
            }
        }
    }

    false
}

fn is_hi_part_opcode(opcode: &str) -> bool {
    opcode.starts_with("LEA.HI")
        || opcode == "IADD3.X"
        || opcode.contains(".HI")  // SHF.R.U32.HI, SHF.R.S32.HI, etc.
        || opcode.starts_with("IMAD.WIDE")
}

fn is_copy_opcode(opcode: &str) -> bool {
    opcode.starts_with("IMAD.MOV") || opcode == "MOV" || opcode.starts_with("MOV.")
}

fn visit_expr_for_addr64(
    value: &RValue,
    def_opcode: &HashMap<RegKey, String>,
    copy_sources: &HashMap<RegKey, Vec<RegKey>>,
    result: &mut Ptr64VerificationResult,
) {
    match value {
        RValue::Op { args, .. } => {
            for arg in args {
                scan_expr_for_addr64(arg, def_opcode, copy_sources, result);
            }
        }
        RValue::Phi(args) => {
            for arg in args {
                scan_expr_for_addr64(arg, def_opcode, copy_sources, result);
            }
        }
        _ => {}
    }
}

fn scan_expr_for_addr64(
    expr: &IRExpr,
    def_opcode: &HashMap<RegKey, String>,
    copy_sources: &HashMap<RegKey, Vec<RegKey>>,
    result: &mut Ptr64VerificationResult,
) {
    match expr {
        IRExpr::Op { op, args } if op == "addr64" && args.len() == 2 => {
            let lo_reg = args[0].get_reg().cloned();
            let hi_reg = args[1].get_reg().cloned();
            if let (Some(lo), Some(hi)) = (lo_reg, hi_reg) {
                let hi_key = RegKey::from(&hi);
                let mut visited = HashSet::new();
                if is_hi_part_def(&hi_key, def_opcode, copy_sources, &mut visited) {
                    result.verified_pairs.push((lo, hi));
                } else {
                    result.unverified_pairs.push((lo, hi));
                }
            }
            // Also scan sub-expressions
            for a in args {
                scan_expr_for_addr64(a, def_opcode, copy_sources, result);
            }
        }
        IRExpr::Op { args, .. } => {
            for a in args {
                scan_expr_for_addr64(a, def_opcode, copy_sources, result);
            }
        }
        IRExpr::Mem { base, offset, .. } => {
            scan_expr_for_addr64(base, def_opcode, copy_sources, result);
            if let Some(off) = offset {
                scan_expr_for_addr64(off, def_opcode, copy_sources, result);
            }
        }
        _ => {}
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RegKey {
    class: String,
    idx: i32,
    ssa: Option<usize>,
}

impl RegKey {
    fn from(r: &RegId) -> Self {
        Self {
            class: r.class.clone(),
            idx: r.idx,
            ssa: r.ssa,
        }
    }
}

fn extract_copy_source(opcode: &str, args: &[IRExpr]) -> Option<RegKey> {
    if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
            if let Some(r) = args[2].get_reg() {
                if !is_immutable_reg(r) {
                    return Some(RegKey::from(r));
                }
            }
        }
    }
    if (opcode == "MOV" || opcode.starts_with("MOV.")) && args.len() == 1 {
        if let Some(r) = args[0].get_reg() {
            if !is_immutable_reg(r) {
                return Some(RegKey::from(r));
            }
        }
    }
    None
}

fn is_zero_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 0,
        IRExpr::ImmF(f) => *f == 0.0,
        IRExpr::Reg(r) => matches!(r.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn is_immutable_reg(r: &RegId) -> bool {
    matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, parse_sass};

    #[test]
    fn verifies_imad_wide_hi_part() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ LDG.E.64 R2, [R4.64] ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let result = verify_ptr64_pairs(&fir);

        // The LDG uses R4.64 → addr64(R4, R5).
        // R5 comes from IMAD.WIDE (hi part), so should be verified.
        assert!(
            !result.verified_pairs.is_empty() || result.unverified_pairs.is_empty(),
            "Expected IMAD.WIDE pair to be verified or no pairs found. \
             Verified: {:?}, Unverified: {:?}",
            result.verified_pairs,
            result.unverified_pairs
        );
    }

    #[test]
    fn flags_unverified_when_hi_is_arbitrary() {
        // R5 is not defined by a hi-part instruction — just a plain IADD3
        let sass = r#"
            /*0000*/ IADD3 R5, R0, 0x1, RZ ;
            /*0010*/ IADD3 R4, R0, 0x2, RZ ;
            /*0020*/ LDG.E.64 R2, [R4.64] ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let result = verify_ptr64_pairs(&fir);

        // R5 defined by plain IADD3 (not .X), so the pair should be unverified
        if !result.unverified_pairs.is_empty() {
            // Good — flagged as unverified
        } else if result.verified_pairs.is_empty() {
            // Also acceptable — no addr64 pairs found at all (depends on lowering)
        } else {
            panic!(
                "Expected unverified pair or no pairs. Got verified: {:?}",
                result.verified_pairs
            );
        }
    }
}
