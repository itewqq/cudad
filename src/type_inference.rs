//! Dataflow-based type inference for SSA-form `FunctionIR`.
//!
//! This pass propagates type information along SSA def-use chains,
//! starting from instruction-level seeds (e.g., FADD → float, ISETP → int,
//! IMAD.WIDE → pointer) and flowing through phi nodes and copies.
//!
//! The type lattice is:
//!   `Bottom < {U8, U16, U32, I32, F16, F32} < AnyInt/AnyFloat < Top`
//!   `Bottom < {U64, Ptr64} < Top`
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
    /// Any integer type (U8, U16, U32, I32 merged). Renders as "uint32_t".
    AnyInt,
    /// Any float type (F16, F32 merged). Renders as "float".
    AnyFloat,
    Top,
}

impl InferredType {
    /// Least upper bound in the type lattice.
    pub fn join(self, other: Self) -> Self {
        use InferredType::*;
        if self == other { return self; }
        if self == Bottom { return other; }
        if other == Bottom { return self; }
        if self == Top || other == Top { return Top; }

        // Check if both are in the integer family.
        let is_int = |t: InferredType| matches!(t, U8 | U16 | U32 | I32 | AnyInt);
        // Check if both are in the float family.
        let is_float = |t: InferredType| matches!(t, F16 | F32 | AnyFloat);

        if is_int(self) && is_int(other) {
            return AnyInt;
        }
        if is_float(self) && is_float(other) {
            return AnyFloat;
        }

        // Different families (int vs float, int vs ptr, etc.) → Top
        Top
    }

    /// Convert to a C type string for declarations.
    pub fn to_c_type(self) -> &'static str {
        match self {
            InferredType::Bottom | InferredType::Top | InferredType::AnyInt => "uint32_t",
            InferredType::U8 => "uint8_t",
            InferredType::U16 => "uint16_t",
            InferredType::U32 => "uint32_t",
            InferredType::I32 => "int32_t",
            InferredType::F32 | InferredType::AnyFloat => "float",
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

    // ---- Phase 1.5: Use-site back-propagation ----
    // When a register is used as an operand to an instruction with known type
    // requirements, seed that type on the input register.
    for block in &fir.blocks {
        for stmt in &block.stmts {
            if let RValue::Op { opcode, args } = &stmt.value {
                let arg_types = infer_arg_types_from_opcode(opcode, args.len());
                for (arg, ty) in args.iter().zip(arg_types.iter()) {
                    if let Some(ty) = ty {
                        if let Some(r) = arg.get_reg() {
                            if is_immutable_reg(r) || is_predicate_reg(r) {
                                continue;
                            }
                            let key = to_ssa_reg(r);
                            let prev = types.get(&key).copied().unwrap_or(InferredType::Bottom);
                            let joined = prev.join(*ty);
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
    // Float operations → F32 for the DEF
    if opcode.starts_with("FADD")
        || opcode.starts_with("FMUL")
        || opcode.starts_with("FFMA")
        || opcode.starts_with("FSEL")
        || opcode.starts_with("MUFU")
        || opcode.starts_with("FMNMX")
        || opcode.starts_with("FSETP")
    {
        return Some(InferredType::F32);
    }

    // I2F / I2FP → float output
    if opcode.starts_with("I2F") {
        return Some(InferredType::F32);
    }

    // F2I → integer output; check signedness suffix
    if opcode.starts_with("F2I") {
        if opcode.contains(".S32") || opcode.contains(".S16") || opcode.contains(".S8") {
            return Some(InferredType::I32);
        }
        return Some(InferredType::U32);
    }

    // Half-precision
    if opcode.starts_with("HFMA2") || opcode.starts_with("HADD2") || opcode.starts_with("HMUL2") {
        return Some(InferredType::F16);
    }

    // Pointer / wide operations
    if opcode.starts_with("IMAD.WIDE") || opcode == "LEA" {
        return Some(InferredType::Ptr64);
    }

    // IABS produces signed output (absolute value of signed input)
    if opcode.starts_with("IABS") {
        return Some(InferredType::I32);
    }

    // IMAD.IADD → integer (unsigned by default)
    if opcode.starts_with("IMAD.IADD") {
        return Some(InferredType::U32);
    }

    // Signed vs unsigned comparison predicates from ISETP
    // ISETP.GE.AND, ISETP.GT.AND → signed comparison (no U suffix)
    // ISETP.GE.U32.AND, ISETP.GT.U32.AND → unsigned comparison
    if opcode.starts_with("ISETP") {
        // The comparison is on the INPUTS, not the output (which is a predicate).
        // We don't seed the predicate register here; use-site propagation handles inputs.
        return None;
    }

    // SHF (funnel shift): check for signed suffix
    if opcode.starts_with("SHF") {
        if opcode.contains(".S32") || opcode.contains(".S16") {
            return Some(InferredType::I32);
        }
        return Some(InferredType::U32);
    }

    // IADD3.X → unsigned carry chain
    if opcode == "IADD3.X" || opcode.starts_with("IADD3.X.") {
        return Some(InferredType::U32);
    }

    // Data width from opcode suffix
    for tok in opcode.split('.') {
        match tok {
            "U8" => return Some(InferredType::U8),
            "S8" => return Some(InferredType::I32), // signed byte → treat as signed int
            "U16" => return Some(InferredType::U16),
            "S16" => return Some(InferredType::I32), // signed short → treat as signed int
            "S32" => return Some(InferredType::I32),
            _ => {}
        }
    }

    None
}

/// Infer type constraints for each argument position of an instruction,
/// based on how the instruction interprets its inputs.
///
/// Returns a Vec the same length as `num_args`, where each element is
/// `Some(ty)` if the opcode constrains that argument to type `ty`, or
/// `None` if no constraint is known.
///
/// This enables **use-site back-propagation**: if ISETP.GE.AND compares
/// two signed integers, the input registers get seeded as I32.
fn infer_arg_types_from_opcode(opcode: &str, num_args: usize) -> Vec<Option<InferredType>> {
    let mut result = vec![None; num_args];
    if num_args == 0 {
        return result;
    }

    // ---- ISETP: integer set-predicate (comparison) ----
    // Only ordered comparisons (.LT, .LE, .GT, .GE) imply signedness:
    //   ISETP.<ord>.<logic>       → signed (inputs are I32)
    //   ISETP.<ord>.U32.<logic>   → unsigned (inputs are U32)
    // Equality comparisons (.EQ, .NE) are signedness-agnostic — skip them.
    // Note: The arg layout from the parser may include predicate operands
    // (e.g., PT) as uses. We seed ALL non-predicate args with the comparison
    // type; predicate/immutable registers will be filtered out by the caller.
    if opcode.starts_with("ISETP") {
        let is_ordered = opcode.contains(".LT")
            || opcode.contains(".LE")
            || opcode.contains(".GT")
            || opcode.contains(".GE");
        if !is_ordered {
            // .EQ / .NE — signedness-agnostic, no constraint
            return result;
        }
        let input_ty = if opcode.contains(".U32") {
            InferredType::U32
        } else {
            InferredType::I32
        };
        for slot in result.iter_mut() {
            *slot = Some(input_ty);
        }
        return result;
    }

    // ---- I2F / I2FP: integer-to-float conversion ----
    // The source integer arg inherits signedness from the opcode suffix.
    // I2F.U16 → source is U16;  I2FP.F32.S32 → source is I32
    // Note: starts_with("I2F") deliberately matches both I2F and I2FP —
    // they use the same suffix scheme (.S32, .U32, .U16, etc.).
    if opcode.starts_with("I2F") {
        let src_ty = if opcode.contains(".S32") {
            InferredType::I32
        } else if opcode.contains(".S16") {
            InferredType::I32 // signed 16-bit, widen to signed int
        } else if opcode.contains(".S8") {
            InferredType::I32
        } else if opcode.contains(".U16") {
            InferredType::U16
        } else if opcode.contains(".U8") {
            InferredType::U8
        } else if opcode.contains(".U32") {
            InferredType::U32
        } else {
            // Default: I2F without explicit width → treat as signed (NVIDIA default)
            InferredType::I32
        };
        // First arg is the integer source
        if num_args >= 1 { result[0] = Some(src_ty); }
        return result;
    }

    // ---- F2I: float-to-integer conversion ----
    // The source is always float; we seed the float input.
    if opcode.starts_with("F2I") {
        if num_args >= 1 { result[0] = Some(InferredType::F32); }
        return result;
    }

    // ---- FADD, FMUL, FFMA, FSEL, FMNMX, FSETP: float operations ----
    // All inputs are float.
    if opcode.starts_with("FADD")
        || opcode.starts_with("FMUL")
        || opcode.starts_with("FFMA")
        || opcode.starts_with("FSEL")
        || opcode.starts_with("FMNMX")
        || opcode.starts_with("FSETP")
    {
        for slot in result.iter_mut() {
            *slot = Some(InferredType::F32);
        }
        return result;
    }

    // ---- MUFU: multi-function unit (sin, cos, rsqrt, etc.) ----
    // Input is float.
    if opcode.starts_with("MUFU") {
        if num_args >= 1 { result[0] = Some(InferredType::F32); }
        return result;
    }

    // ---- HFMA2, HADD2, HMUL2: half-precision operations ----
    if opcode.starts_with("HFMA2") || opcode.starts_with("HADD2") || opcode.starts_with("HMUL2") {
        for slot in result.iter_mut() {
            *slot = Some(InferredType::F16);
        }
        return result;
    }

    // ---- IABS: absolute value (input is signed) ----
    if opcode.starts_with("IABS") {
        if num_args >= 1 { result[0] = Some(InferredType::I32); }
        return result;
    }

    // ---- IMAD with explicit signedness suffix ----
    // Plain IMAD performs truncated 32×32→32 multiply-add where the low 32
    // bits are identical regardless of signedness — no constraint.
    // Only IMAD.HI (which produces the *high* 32 bits) is signedness-sensitive:
    //   IMAD.HI.U32 → unsigned;  IMAD.HI (no U32) → signed.
    // IMAD.U32 / IMAD.SHL.U32 also explicitly mark unsigned.
    if opcode.starts_with("IMAD") && !opcode.starts_with("IMAD.MOV") && !opcode.starts_with("IMAD.WIDE") {
        // Only infer signedness when the opcode has an explicit signedness marker
        let has_hi = opcode.contains(".HI");
        let has_u32 = opcode.contains(".U32");
        if has_hi || has_u32 {
            let input_ty = if has_u32 {
                InferredType::U32
            } else {
                // .HI without .U32 → signed
                InferredType::I32
            };
            for slot in result.iter_mut() {
                *slot = Some(input_ty);
            }
        }
        // Plain IMAD without .HI or .U32 → no signedness constraint
        return result;
    }

    // ---- SHF: funnel shift ----
    // SHF.R.S32.HI → signed shift; SHF.L.U32 → unsigned shift
    if opcode.starts_with("SHF") {
        let input_ty = if opcode.contains(".S32") || opcode.contains(".S16") {
            InferredType::I32
        } else {
            InferredType::U32
        };
        // arg[0] = data_lo, arg[1] = shift_amount (unsigned), arg[2] = data_hi
        if num_args >= 1 { result[0] = Some(input_ty); }
        // shift amount is always unsigned
        if num_args >= 2 { result[1] = Some(InferredType::U32); }
        if num_args >= 3 { result[2] = Some(input_ty); }
        return result;
    }

    // ---- LDG/LDS with width suffix: propagate width to address register ----
    // We don't seed the address itself (it's a pointer), but for loads with
    // explicit width like LDG.E.U8, the *loaded value* width is known from
    // type_from_opcode. No arg-side propagation needed.

    result
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

    #[test]
    fn back_propagates_signed_from_isetp() {
        // ISETP.GE.AND (no U32) → signed comparison → inputs should be I32.
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, R1, PT ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        // R0 and R1 should be inferred as I32 (signed comparison inputs)
        let r0_ty = types.get(&("R".to_string(), 0)).copied().unwrap_or(InferredType::Bottom);
        let r1_ty = types.get(&("R".to_string(), 1)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r0_ty, InferredType::I32, "ISETP.GE.AND input R0 should be I32");
        assert_eq!(r1_ty, InferredType::I32, "ISETP.GE.AND input R1 should be I32");
    }

    #[test]
    fn back_propagates_unsigned_from_isetp_u32() {
        // ISETP.GT.U32.AND → unsigned comparison → inputs should be U32.
        let sass = r#"
            /*0000*/ ISETP.GT.U32.AND P0, PT, R0, R1, PT ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r0_ty = types.get(&("R".to_string(), 0)).copied().unwrap_or(InferredType::Bottom);
        let r1_ty = types.get(&("R".to_string(), 1)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r0_ty, InferredType::U32, "ISETP.GT.U32 input R0 should be U32");
        assert_eq!(r1_ty, InferredType::U32, "ISETP.GT.U32 input R1 should be U32");
    }

    #[test]
    fn isetp_eq_ne_does_not_seed_signedness() {
        // ISETP.NE.AND is an equality comparison — signedness-agnostic.
        // Inputs should NOT be seeded with any type.
        let sass = r#"
            /*0000*/ ISETP.NE.AND P0, PT, R0, R1, PT ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r0_ty = types.get(&("R".to_string(), 0)).copied().unwrap_or(InferredType::Bottom);
        let r1_ty = types.get(&("R".to_string(), 1)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r0_ty, InferredType::Bottom, "ISETP.NE should not seed signedness on R0");
        assert_eq!(r1_ty, InferredType::Bottom, "ISETP.NE should not seed signedness on R1");
    }

    #[test]
    fn back_propagates_float_from_fadd_inputs() {
        // FADD R2, R0, R1 → both inputs should be seeded as F32
        let sass = r#"
            /*0000*/ FADD R2, R0, R1 ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r0_ty = types.get(&("R".to_string(), 0)).copied().unwrap_or(InferredType::Bottom);
        let r1_ty = types.get(&("R".to_string(), 1)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r0_ty, InferredType::F32, "FADD input R0 should be F32");
        assert_eq!(r1_ty, InferredType::F32, "FADD input R1 should be F32");
    }

    #[test]
    fn infers_signed_from_shf_s32() {
        // SHF.R.S32.HI → output and data inputs should be I32
        let sass = r#"
            /*0000*/ SHF.R.S32.HI R2, RZ, 0x1f, R0 ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        // R2 (output) should be I32 from type_from_opcode
        let r2_ty = types.get(&("R".to_string(), 2)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r2_ty, InferredType::I32, "SHF.R.S32.HI output should be I32");

        // R0 (data_hi input) should also be I32 from back-propagation
        let r0_ty = types.get(&("R".to_string(), 0)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r0_ty, InferredType::I32, "SHF.R.S32.HI data input should be I32");
    }

    #[test]
    fn infers_signed_from_iabs() {
        // IABS R1, R0 → output is I32, input should be I32
        let sass = r#"
            /*0000*/ IABS R1, R0 ;
            /*0010*/ EXIT ;
        "#;
        let cfg = build_cfg(parse_sass(sass));
        let fir = build_ssa(&cfg);
        let types = infer_types(&fir);

        let r1_ty = types.get(&("R".to_string(), 1)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r1_ty, InferredType::I32, "IABS output should be I32");

        let r0_ty = types.get(&("R".to_string(), 0)).copied().unwrap_or(InferredType::Bottom);
        assert_eq!(r0_ty, InferredType::I32, "IABS input should be I32 (back-propagated)");
    }
}
