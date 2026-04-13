//! SSA-level semantic label propagation.
//!
//! This pass walks the SSA IR to identify "seed" definitions that have
//! well-known meanings (e.g. `S2R SR_TID.X` → `tid_x`) and propagates
//! those labels along def-use chains through phi nodes and copies.
//!
//! The result is a map from (register class, index, SSA version) to a
//! semantic label string.  Name recovery can use this to produce better
//! names than the default `v0, v1, …` scheme.

use std::collections::{HashMap, VecDeque};

use crate::ir::{FunctionIR, IRExpr, RValue, RegId};

/// Unique SSA register identity (class + index + SSA version).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SsaRegKey {
    pub class: String,
    pub idx: i32,
    pub ssa: Option<usize>,
}

impl SsaRegKey {
    fn from_reg(r: &RegId) -> Self {
        Self {
            class: r.class.clone(),
            idx: r.idx,
            ssa: r.ssa,
        }
    }
}

/// Run semantic label propagation on the function IR.
///
/// Returns a map from SSA register → semantic label name (e.g. "tid_x",
/// "ctaid_y", "block_dim_x", "lane_id", etc.).
pub fn propagate_semantic_labels(fir: &FunctionIR) -> HashMap<SsaRegKey, String> {
    let mut labels: HashMap<SsaRegKey, String> = HashMap::new();
    let mut worklist: VecDeque<SsaRegKey> = VecDeque::new();

    // ---- Phase 1: Seed from S2R / CS2R / S2UR instructions ----
    for block in &fir.blocks {
        for stmt in &block.stmts {
            if let RValue::Op { opcode, args } = &stmt.value {
                if let Some(sem) = extract_semantic_seed(opcode, args) {
                    for def in &stmt.defs {
                        if let Some(r) = def.get_reg() {
                            if is_immutable_reg(r) || is_predicate_reg(r) {
                                continue;
                            }
                            let key = SsaRegKey::from_reg(r);
                            if !labels.contains_key(&key) {
                                labels.insert(key.clone(), sem.clone());
                                worklist.push_back(key);
                            }
                        }
                    }
                }
            }
        }
    }

    if labels.is_empty() {
        return labels;
    }

    // ---- Phase 2: Build phi/copy dependency graph ----
    struct PhiInfo {
        def_key: SsaRegKey,
        input_keys: Vec<SsaRegKey>,
    }
    struct CopyInfo {
        def_key: SsaRegKey,
        src_key: SsaRegKey,
    }

    let mut phis: Vec<PhiInfo> = Vec::new();
    let mut copies: Vec<CopyInfo> = Vec::new();

    // Also build reverse map: src → list of copy/phi defs that use it
    let mut src_to_copy_defs: HashMap<SsaRegKey, Vec<usize>> = HashMap::new();
    let mut src_to_phi_defs: HashMap<SsaRegKey, Vec<usize>> = HashMap::new();

    for block in &fir.blocks {
        for stmt in &block.stmts {
            match &stmt.value {
                RValue::Phi(args) => {
                    if let Some(def) = stmt.defs.first().and_then(|d| d.get_reg()) {
                        if is_immutable_reg(def) || is_predicate_reg(def) {
                            continue;
                        }
                        let inputs: Vec<SsaRegKey> = args
                            .iter()
                            .filter_map(|a| a.get_reg())
                            .filter(|r| !is_immutable_reg(r))
                            .map(|r| SsaRegKey::from_reg(r))
                            .collect();
                        let phi_idx = phis.len();
                        let def_key = SsaRegKey::from_reg(def);
                        for inp in &inputs {
                            src_to_phi_defs
                                .entry(inp.clone())
                                .or_default()
                                .push(phi_idx);
                        }
                        phis.push(PhiInfo {
                            def_key,
                            input_keys: inputs,
                        });
                    }
                }
                RValue::Op { opcode, args } => {
                    if stmt.defs.len() == 1 && stmt.pred.is_none() {
                        if let Some(def) = stmt.defs[0].get_reg() {
                            if is_immutable_reg(def) || is_predicate_reg(def) {
                                continue;
                            }
                            if let Some(src) = extract_copy_source(opcode, args) {
                                let copy_idx = copies.len();
                                let def_key = SsaRegKey::from_reg(def);
                                src_to_copy_defs
                                    .entry(src.clone())
                                    .or_default()
                                    .push(copy_idx);
                                copies.push(CopyInfo {
                                    def_key,
                                    src_key: src,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // ---- Phase 3: Worklist-based propagation ----
    // When a label is assigned to a register, propagate through copies and phis.
    let mut iterations = 0;
    while let Some(key) = worklist.pop_front() {
        iterations += 1;
        if iterations > 10000 {
            break; // Safety bound
        }

        let label = match labels.get(&key) {
            Some(l) => l.clone(),
            None => continue,
        };

        // Forward: propagate through copies where `key` is the source
        if let Some(copy_indices) = src_to_copy_defs.get(&key) {
            for &ci in copy_indices {
                let copy = &copies[ci];
                if !labels.contains_key(&copy.def_key) {
                    labels.insert(copy.def_key.clone(), label.clone());
                    worklist.push_back(copy.def_key.clone());
                }
            }
        }

        // Forward: propagate through phis where `key` is an input
        if let Some(phi_indices) = src_to_phi_defs.get(&key) {
            for &pi in phi_indices {
                let phi = &phis[pi];
                if labels.contains_key(&phi.def_key) {
                    continue;
                }
                // All inputs must have the same label for the phi output to inherit
                let all_same = phi
                    .input_keys
                    .iter()
                    .all(|inp| labels.get(inp).map(|l| l == &label).unwrap_or(false));
                if all_same {
                    labels.insert(phi.def_key.clone(), label.clone());
                    worklist.push_back(phi.def_key.clone());
                }
            }
        }
    }

    // ---- Phase 4: Backward propagation through phis ----
    // If a phi's def has a label, propagate to inputs that don't have one yet.
    let mut changed = true;
    let mut back_iters = 0;
    while changed && back_iters < 50 {
        changed = false;
        back_iters += 1;

        for phi in &phis {
            if let Some(def_label) = labels.get(&phi.def_key).cloned() {
                for inp in &phi.input_keys {
                    if !labels.contains_key(inp) {
                        labels.insert(inp.clone(), def_label.clone());
                        changed = true;
                    }
                }
            }
        }

        // Also forward through copies again after backward propagation
        for copy in &copies {
            if let Some(src_label) = labels.get(&copy.src_key).cloned() {
                if !labels.contains_key(&copy.def_key) {
                    labels.insert(copy.def_key.clone(), src_label);
                    changed = true;
                }
            }
        }
    }

    labels
}

/// Check if the opcode+args represent a special register read.
fn extract_semantic_seed(opcode: &str, args: &[IRExpr]) -> Option<String> {
    // S2R / CS2R / S2UR — read from special register
    if !(opcode.starts_with("S2R") || opcode.starts_with("CS2R") || opcode.starts_with("S2UR")) {
        return None;
    }
    if args.len() != 1 {
        return None;
    }
    // The argument is typically IRExpr::Op { op: "SR_TID.X", args: [] }
    let sr_name = match &args[0] {
        IRExpr::Op { op, args } if args.is_empty() => op.as_str(),
        IRExpr::Reg(r) if r.class.starts_with("SR") => return sr_to_semantic(&r.class),
        _ => return None,
    };
    sr_to_semantic(sr_name)
}

fn sr_to_semantic(sr: &str) -> Option<String> {
    let name = match sr {
        "SR_TID.X" => "tid_x",
        "SR_TID.Y" => "tid_y",
        "SR_TID.Z" => "tid_z",
        "SR_CTAID.X" => "ctaid_x",
        "SR_CTAID.Y" => "ctaid_y",
        "SR_CTAID.Z" => "ctaid_z",
        "SR_NTID.X" => "block_dim_x",
        "SR_NTID.Y" => "block_dim_y",
        "SR_NTID.Z" => "block_dim_z",
        "SR_NCTAID.X" => "grid_dim_x",
        "SR_NCTAID.Y" => "grid_dim_y",
        "SR_NCTAID.Z" => "grid_dim_z",
        "SR_LANEID" => "lane_id",
        "SR_CgaCtaId" => "cga_cta_id",
        _ => return None,
    };
    Some(name.to_string())
}

/// Detect copy-like instructions at the IR level.
fn extract_copy_source(opcode: &str, args: &[IRExpr]) -> Option<SsaRegKey> {
    // IMAD.MOV.U32 Rdst, RZ, RZ, Rsrc → copy
    if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
            if let Some(r) = args[2].get_reg() {
                if !is_immutable_reg(r) {
                    return Some(SsaRegKey::from_reg(r));
                }
            }
        }
    }
    // MOV Rdst, Rsrc
    if (opcode == "MOV" || opcode.starts_with("MOV.")) && args.len() == 1 {
        if let Some(r) = args[0].get_reg() {
            if !is_immutable_reg(r) {
                return Some(SsaRegKey::from_reg(r));
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

fn is_predicate_reg(r: &RegId) -> bool {
    matches!(r.class.as_str(), "P" | "UP")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, decode_sass};

    #[test]
    fn seeds_s2r_tid_x() {
        let sass = r#"
            /*0000*/ S2R R2, SR_TID.X ;
            /*0010*/ IADD3 R3, R2, R0, RZ ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let labels = propagate_semantic_labels(&fir);

        // R2 should have label "tid_x"
        let has_tid_x = labels.values().any(|v| v == "tid_x");
        assert!(has_tid_x, "Expected tid_x label, got {:?}", labels);
    }

    #[test]
    fn propagates_through_copy() {
        let sass = r#"
            /*0000*/ S2R R2, SR_TID.X ;
            /*0010*/ IMAD.MOV.U32 R3, RZ, RZ, R2 ;
            /*0020*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let labels = propagate_semantic_labels(&fir);

        // Both R2 and R3 should have label "tid_x"
        let tid_x_count = labels.values().filter(|v| v.as_str() == "tid_x").count();
        assert!(
            tid_x_count >= 2,
            "Expected tid_x on both R2 and R3, got {} labels: {:?}",
            tid_x_count,
            labels
        );
    }

    #[test]
    fn propagates_through_phi() {
        // R2 is assigned by S2R on one path and loops back through a phi
        let sass = r#"
            /*0000*/ S2R R2, SR_TID.X ;
            /*0010*/ ISETP.LT.AND P0, PT, R3, 0x5, PT ;
            /*0020*/ @P0 BRA 0x000 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(decode_sass(sass));
        let fir = build_ssa(&cfg);
        let labels = propagate_semantic_labels(&fir);

        // All SSA versions of "R2" should have label "tid_x"
        let r2_labels: Vec<_> = labels
            .iter()
            .filter(|(k, _)| k.class == "R" && k.idx == 2)
            .collect();
        assert!(
            !r2_labels.is_empty(),
            "Expected some R2 labels, got {:?}",
            labels
        );
        for (key, val) in &r2_labels {
            assert_eq!(
                val.as_str(),
                "tid_x",
                "R2 version {:?} should be tid_x, got {}",
                key.ssa,
                val
            );
        }
    }
}
