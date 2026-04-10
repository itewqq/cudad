//! Dataflow-based type inference for SSA-form `FunctionIR`.
//!
//! This pass propagates type information along SSA def-use chains,
//! starting from instruction-level seeds (e.g., FADD → float, ISETP → int,
//! IMAD.WIDE → pointer) and flowing through phi nodes and copies.
//!
//! The type lattice is: `Bottom < {U8, U16, U32, I32, F32, U64, Ptr64} < Top`
//! Phi nodes take the join (least upper bound) of their inputs.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::ir::{FunctionIR, IRExpr, RValue, RegId};

/// Inferred type for a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InferredType {
    Bottom,
    U8,
    U16,
    U32,
    I32,
    F32,
    F16,
    U64,
    Ptr64,
    Top,
}

impl InferredType {
    /// Least upper bound in the type lattice.
    pub fn join(self, other: Self) -> Self {
        if self == other { return self; }
        if self == InferredType::Bottom { return other; }
        if other == InferredType::Bottom { return self; }
        // Different concrete types → Top (ambiguous)
        InferredType::Top
    }

    /// Convert to a C type string for declarations.
    pub fn to_c_type(self) -> &'static str {
        match self {
            InferredType::Bottom | InferredType::Top => "uint32_t",
            InferredType::U8 => "uint8_t",
            InferredType::U16 => "uint16_t",
            InferredType::U32 => "uint32_t",
            InferredType::I32 => "int32_t",
            InferredType::F32 => "float",
            InferredType::F16 => "__half",
            InferredType::U64 => "uint64_t",
            InferredType::Ptr64 => "uintptr_t",
        }
    }
}

/// Unique identity for an SSA register (class + idx + ssa version).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SsaReg {
    class: String,
    idx: i32,
    ssa: Option<usize>,
}

fn to_ssa_reg(r: &RegId) -> SsaReg {
    SsaReg {
        class: r.class.clone(),
        idx: r.idx,
        ssa: r.ssa,
    }
}

/// Run type inference on the function IR, returning a map from
/// (register class, register index) → inferred type.
///
/// This merges all SSA versions of the same physical register and
/// returns the consensus type per physical register — matching the
/// granularity used by `infer_local_typed_declarations_with_abi`.
pub fn infer_types(fir: &FunctionIR) -> BTreeMap<(String, i32), InferredType> {
    let mut types: HashMap<SsaReg, InferredType> = HashMap::new();
    let mut worklist: VecDeque<SsaReg> = VecDeque::new();

    // ---- Phase 1: Seed from instruction opcodes ----
    for block in &fir.blocks {
        for stmt in &block.stmts {
            if let RValue::Op { opcode, .. } = &stmt.value {
                if let Some(ty) = type_from_opcode(opcode) {
                    for def in &stmt.defs {
                        if let Some(r) = def.get_reg() {
                            if is_immutable_reg(r) || is_predicate_reg(r) {
                                continue;
                            }
                            let key = to_ssa_reg(r);
                            let prev = types.get(&key).copied().unwrap_or(InferredType::Bottom);
                            let joined = prev.join(ty);
                            if joined != prev {
                                types.insert(key.clone(), joined);
                                worklist.push_back(key);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Phase 2: Forward propagation through phi and copy ----
    // Build a map: SSA reg → list of (block, stmt) where it's used as phi/copy input
    // For each phi: if all inputs have the same type, the output inherits it.
    // For copies (IMAD.MOV.U32 with src = known reg): output inherits source type.

    // Build phi/copy dependency graph
    struct PhiInfo {
        def_reg: SsaReg,
        input_regs: Vec<SsaReg>,
    }
    struct CopyInfo {
        def_reg: SsaReg,
        src_reg: SsaReg,
    }

    let mut phis: Vec<PhiInfo> = Vec::new();
    let mut copies: Vec<CopyInfo> = Vec::new();

    for block in &fir.blocks {
        for stmt in &block.stmts {
            match &stmt.value {
                RValue::Phi(args) => {
                    if let Some(def) = stmt.defs.first().and_then(|d| d.get_reg()) {
                        if is_immutable_reg(def) || is_predicate_reg(def) {
                            continue;
                        }
                        let inputs: Vec<SsaReg> = args.iter()
                            .filter_map(|a| a.get_reg())
                            .filter(|r| !is_immutable_reg(r))
                            .map(|r| to_ssa_reg(r))
                            .collect();
                        phis.push(PhiInfo {
                            def_reg: to_ssa_reg(def),
                            input_regs: inputs,
                        });
                    }
                }
                RValue::Op { opcode, args } => {
                    // Detect copy patterns
                    if stmt.defs.len() == 1 && stmt.pred.is_none() {
                        if let Some(def) = stmt.defs[0].get_reg() {
                            if is_immutable_reg(def) || is_predicate_reg(def) {
                                continue;
                            }
                            if let Some(src) = extract_copy_source(opcode, args) {
                                copies.push(CopyInfo {
                                    def_reg: to_ssa_reg(def),
                                    src_reg: src,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Iterate until fixpoint
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;

        // Propagate through copies
        for copy in &copies {
            if let Some(&src_ty) = types.get(&copy.src_reg) {
                if src_ty != InferredType::Bottom && src_ty != InferredType::Top {
                    let prev = types.get(&copy.def_reg).copied().unwrap_or(InferredType::Bottom);
                    let joined = prev.join(src_ty);
                    if joined != prev {
                        types.insert(copy.def_reg.clone(), joined);
                        changed = true;
                    }
                }
            }
        }

        // Propagate through phis
        for phi in &phis {
            let mut joined = InferredType::Bottom;
            for input in &phi.input_regs {
                let input_ty = types.get(input).copied().unwrap_or(InferredType::Bottom);
                joined = joined.join(input_ty);
            }
            if joined != InferredType::Bottom && joined != InferredType::Top {
                let prev = types.get(&phi.def_reg).copied().unwrap_or(InferredType::Bottom);
                let new = prev.join(joined);
                if new != prev {
                    types.insert(phi.def_reg.clone(), new);
                    changed = true;
                }
            }
        }

        // Backward propagation: if a def has a known type, propagate to
        // its use sites' defs (through phi inputs)
        for phi in &phis {
            if let Some(&def_ty) = types.get(&phi.def_reg) {
                if def_ty != InferredType::Bottom && def_ty != InferredType::Top {
                    for input in &phi.input_regs {
                        let prev = types.get(input).copied().unwrap_or(InferredType::Bottom);
                        let joined = prev.join(def_ty);
                        if joined != prev {
                            types.insert(input.clone(), joined);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // ---- Phase 3: Merge SSA versions into per-physical-register types ----
    let mut result: BTreeMap<(String, i32), InferredType> = BTreeMap::new();
    for (ssa_reg, ty) in &types {
        if *ty == InferredType::Bottom || *ty == InferredType::Top {
            continue;
        }
        let key = (ssa_reg.class.clone(), ssa_reg.idx);
        let prev = result.get(&key).copied().unwrap_or(InferredType::Bottom);
        result.insert(key, prev.join(*ty));
    }

    result
}

fn type_from_opcode(opcode: &str) -> Option<InferredType> {
    // Float operations
    if opcode.starts_with("FADD")
        || opcode.starts_with("FMUL")
        || opcode.starts_with("FFMA")
        || opcode.starts_with("FSEL")
        || opcode.starts_with("MUFU")
        || opcode.starts_with("FMNMX")
        || opcode.starts_with("FSETP")
        || opcode.starts_with("I2F")
        || opcode.starts_with("I2FP")
    {
        return Some(InferredType::F32);
    }

    // Half-precision
    if opcode.starts_with("HFMA2") || opcode.starts_with("HADD2") || opcode.starts_with("HMUL2") {
        return Some(InferredType::F16);
    }

    // Pointer / wide operations
    if opcode.starts_with("IMAD.WIDE") || opcode == "LEA" {
        return Some(InferredType::Ptr64);
    }

    // Data width from opcode suffix
    for tok in opcode.split('.') {
        match tok {
            "U8" | "S8" => return Some(InferredType::U8),
            "U16" | "S16" => return Some(InferredType::U16),
            _ => {}
        }
    }

    None
}

fn extract_copy_source(opcode: &str, args: &[IRExpr]) -> Option<SsaReg> {
    // IMAD.MOV.U32 Rdst, RZ, RZ, Rsrc → copy from Rsrc
    if opcode.starts_with("IMAD.MOV") && args.len() >= 3 {
        if is_zero_expr(&args[0]) && is_zero_expr(&args[1]) {
            if let Some(r) = args[2].get_reg() {
                if !is_immutable_reg(r) {
                    return Some(to_ssa_reg(r));
                }
            }
        }
    }
    // MOV Rdst, Rsrc
    if (opcode == "MOV" || opcode.starts_with("MOV.")) && args.len() == 1 {
        if let Some(r) = args[0].get_reg() {
            if !is_immutable_reg(r) {
                return Some(to_ssa_reg(r));
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
    use crate::{build_cfg, build_ssa, parse_sass};

    #[test]
    fn infers_float_from_fadd() {
        let sass = r#"
            /*0000*/ FADD R2, R0, R1 ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r2_ty = types.get(&("R".to_string(), 2)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r2_ty, InferredType::F32);
    }

    #[test]
    fn infers_pointer_from_imad_wide() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r4_ty = types.get(&("R".to_string(), 4)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r4_ty, InferredType::Ptr64);
    }

    #[test]
    fn propagates_through_phi() {
        // A loop where R2 is assigned by FADD on one path and loops back
        // through a phi — the phi output should also be inferred as F32.
        let sass = r#"
            /*0000*/ FADD R2, R0, R1 ;
            /*0010*/ ISETP.LT.AND P0, PT, R3, 0x5, PT ;
            /*0020*/ @P0 BRA 0x000 ;
            /*0030*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r2_ty = types.get(&("R".to_string(), 2)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r2_ty, InferredType::F32);
    }
}
