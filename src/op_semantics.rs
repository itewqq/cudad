use crate::parser::DecodedOperand;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefRole {
    Data,
    Pred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UseRole {
    Data,
    PredIn,
    PredCtrl,
    Addr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpSemantics {
    pub def_operand_indices: Vec<usize>,
    pub def_roles: Vec<DefRole>,
    pub use_operand_indices: Vec<usize>,
    pub use_roles: Vec<UseRole>,
}

fn is_pred_operand(op: &DecodedOperand) -> bool {
    matches!(
        op,
        DecodedOperand::PredicateRegister { class, .. }
            if matches!(class.as_str(), "P" | "UP" | "PT" | "UPT")
    )
}

fn is_mem_operand(op: &DecodedOperand) -> bool {
    matches!(
        op,
        DecodedOperand::Address { .. } | DecodedOperand::DescriptorMem { .. }
    )
}

fn opcode_mnemonic(opcode: &str) -> &str {
    opcode.split('.').next().unwrap_or(opcode)
}

fn opcode_has_mod(opcode: &str, needle: &str) -> bool {
    opcode.split('.').skip(1).any(|m| m == needle)
}

// ---------------------------------------------------------------------------
// Table-driven instruction descriptor
// ---------------------------------------------------------------------------

/// Compact descriptor for an instruction class's operand semantics.
///
/// `defs` lists how many leading operands are defs (with roles),
/// and `use_overrides` lists special use-role overrides indexed from
/// the end of the use list (e.g. the last use may be PredIn).
#[derive(Clone, Debug)]
struct OpDesc {
    /// Number of data-def operands (always starting at operand 0).
    data_defs: u8,
    /// If true, scan operands 1.. for predicate defs (IADD3 pattern).
    pred_defs_after_first: bool,
    /// Overrides for use roles, applied from the end.
    /// E.g. `vec![(0, PredIn)]` means "last use is PredIn".
    use_tail_overrides: Vec<(usize, UseRole)>,
}

impl Default for OpDesc {
    fn default() -> Self {
        Self {
            data_defs: 1,
            pred_defs_after_first: false,
            use_tail_overrides: Vec::new(),
        }
    }
}

/// Table of known instruction patterns. Checked in order; first match wins.
/// Each entry is (match_fn, OpDesc).
fn build_op_table() -> Vec<(fn(&str) -> bool, OpDesc)> {
    vec![
        // IADD3 / UIADD3 (non-.X): first operand is data def,
        // subsequent pred operands are also defs.
        (
            (|op: &str| {
                let m = opcode_mnemonic(op);
                (m == "IADD3" || m == "UIADD3") && !opcode_has_mod(op, "X")
            }) as fn(&str) -> bool,
            OpDesc {
                data_defs: 1,
                pred_defs_after_first: true,
                use_tail_overrides: Vec::new(),
            },
        ),
        // IADD3.X / UIADD3.X: last two uses are PredIn, PredCtrl
        (
            |op: &str| {
                let m = opcode_mnemonic(op);
                m == "IADD3" && opcode_has_mod(op, "X") || m == "UIADD3" && opcode_has_mod(op, "X")
            },
            OpDesc {
                data_defs: 1,
                pred_defs_after_first: false,
                use_tail_overrides: vec![(0, UseRole::PredCtrl), (1, UseRole::PredIn)],
            },
        ),
        // LEA / ULEA (non-.HI): first operand is the data def, and an
        // optional predicate operand immediately after it is the carry-out def.
        (
            |op: &str| {
                let m = opcode_mnemonic(op);
                matches!(m, "LEA" | "ULEA") && !opcode_has_mod(op, "HI")
            },
            OpDesc {
                data_defs: 1,
                pred_defs_after_first: true,
                use_tail_overrides: Vec::new(),
            },
        ),
        // LEA.HI.X / ULEA.HI.X: last use is PredIn (carry)
        (
            |op: &str| op.starts_with("LEA.HI.X") || op.starts_with("ULEA.HI.X"),
            OpDesc {
                data_defs: 1,
                pred_defs_after_first: false,
                use_tail_overrides: vec![(0, UseRole::PredIn)],
            },
        ),
        // ISETP / FSETP / DSETP / HSETP2 / UISETP: first two operands
        // are pred defs, rest are uses.
        (
            |op: &str| {
                let m = opcode_mnemonic(op);
                matches!(m, "ISETP" | "FSETP" | "DSETP" | "HSETP2" | "UISETP")
            },
            OpDesc {
                data_defs: 0, // handled specially below
                pred_defs_after_first: false,
                use_tail_overrides: Vec::new(),
            },
        ),
    ]
}

pub fn derive_op_semantics(
    opcode: &str,
    operands: &[DecodedOperand],
    is_mem_load: bool,
    is_mem_store: bool,
) -> OpSemantics {
    // Memory loads: first operand is def, rest are addr uses.
    if is_mem_load {
        let mut def_operand_indices = Vec::new();
        let mut def_roles = Vec::new();
        if !operands.is_empty() {
            def_operand_indices.push(0);
            def_roles.push(DefRole::Data);
        }
        let use_operand_indices = (1..operands.len()).collect::<Vec<_>>();
        let use_roles = vec![UseRole::Addr; use_operand_indices.len()];
        return OpSemantics {
            def_operand_indices,
            def_roles,
            use_operand_indices,
            use_roles,
        };
    }

    // Memory stores: all operands are uses (first is addr, rest are data).
    if is_mem_store {
        let use_operand_indices = (0..operands.len()).collect::<Vec<_>>();
        let mut use_roles = Vec::with_capacity(use_operand_indices.len());
        for (idx, _) in use_operand_indices.iter().enumerate() {
            use_roles.push(if idx == 0 {
                UseRole::Addr
            } else {
                UseRole::Data
            });
        }
        return OpSemantics {
            def_operand_indices: Vec::new(),
            def_roles: Vec::new(),
            use_operand_indices,
            use_roles,
        };
    }

    let mnem = opcode_mnemonic(opcode);
    if mnem.starts_with("RED") {
        let mut use_operand_indices = Vec::new();
        let mut use_roles = Vec::new();
        let addr_idx = operands.iter().position(is_mem_operand);
        for idx in 0..operands.len() {
            use_operand_indices.push(idx);
            use_roles.push(if Some(idx) == addr_idx {
                UseRole::Addr
            } else {
                UseRole::Data
            });
        }
        return OpSemantics {
            def_operand_indices: Vec::new(),
            def_roles: Vec::new(),
            use_operand_indices,
            use_roles,
        };
    }

    if mnem.starts_with("ATOM") {
        let mut def_operand_indices = Vec::new();
        let mut def_roles = Vec::new();
        let mut use_operand_indices = Vec::new();
        let mut use_roles = Vec::new();

        let addr_idx = operands.iter().position(is_mem_operand);
        let mut cursor = 0;
        if operands.first().is_some_and(is_pred_operand) {
            def_operand_indices.push(0);
            def_roles.push(DefRole::Pred);
            cursor = 1;
        }
        if let Some(addr_idx) = addr_idx {
            if cursor < addr_idx {
                def_operand_indices.push(cursor);
                def_roles.push(DefRole::Data);
            }
            for idx in 0..operands.len() {
                if def_operand_indices.contains(&idx) {
                    continue;
                }
                use_operand_indices.push(idx);
                use_roles.push(if idx == addr_idx {
                    UseRole::Addr
                } else {
                    UseRole::Data
                });
            }
            return OpSemantics {
                def_operand_indices,
                def_roles,
                use_operand_indices,
                use_roles,
            };
        }
    }

    // SHFL.* family: optional predicate output in operand 0, data output in
    // operand 1, then the source/lane/clamp inputs.
    if matches!(mnem, "SHFL" | "USHFL") {
        let mut def_operand_indices = Vec::new();
        let mut def_roles = Vec::new();
        let mut first_use_idx = 0usize;

        if operands.first().is_some_and(is_pred_operand) {
            def_operand_indices.push(0);
            def_roles.push(DefRole::Pred);
            first_use_idx = 1;
        }
        if operands.len() > first_use_idx {
            def_operand_indices.push(first_use_idx);
            def_roles.push(DefRole::Data);
            first_use_idx += 1;
        }

        let use_operand_indices = (first_use_idx..operands.len()).collect::<Vec<_>>();
        let use_roles = vec![UseRole::Data; use_operand_indices.len()];
        return OpSemantics {
            def_operand_indices,
            def_roles,
            use_operand_indices,
            use_roles,
        };
    }

    // LOP3.* family: some forms expose an optional predicate def in operand 0
    // and a data def in operand 1, followed by the logic inputs, LUT
    // immediate, and predicate control operand.
    if matches!(mnem, "LOP3" | "ULOP3") {
        let mut def_operand_indices = Vec::new();
        let mut def_roles = Vec::new();
        let mut first_use_idx = 0usize;

        if operands.first().is_some_and(is_pred_operand) {
            def_operand_indices.push(0);
            def_roles.push(DefRole::Pred);
            first_use_idx = 1;
        }
        if operands.len() > first_use_idx {
            def_operand_indices.push(first_use_idx);
            def_roles.push(DefRole::Data);
            first_use_idx += 1;
        }

        let use_operand_indices = (first_use_idx..operands.len()).collect::<Vec<_>>();
        let use_roles = vec![UseRole::Data; use_operand_indices.len()];
        return OpSemantics {
            def_operand_indices,
            def_roles,
            use_operand_indices,
            use_roles,
        };
    }

    // SETP family: special case — first two operands are pred defs.
    if matches!(mnem, "ISETP" | "FSETP" | "DSETP" | "HSETP2" | "UISETP") {
        return derive_setp_semantics(operands);
    }

    // Table lookup
    let table = build_op_table();
    for (matcher, desc) in &table {
        if matcher(opcode) {
            return apply_op_desc(desc, opcode, operands);
        }
    }

    // Default: first operand is data def, rest are data uses.
    apply_op_desc(&OpDesc::default(), opcode, operands)
}

fn derive_setp_semantics(operands: &[DecodedOperand]) -> OpSemantics {
    let mut def_operand_indices = Vec::new();
    let mut def_roles = Vec::new();
    // First two operands are predicate defs (result, overflow/carry)
    for i in 0..operands.len().min(2) {
        if is_pred_operand(&operands[i]) {
            def_operand_indices.push(i);
            def_roles.push(DefRole::Pred);
        }
    }
    if def_operand_indices.is_empty() && !operands.is_empty() {
        // Fallback: treat first as data def
        def_operand_indices.push(0);
        def_roles.push(DefRole::Data);
    }
    let use_operand_indices: Vec<_> = (0..operands.len())
        .filter(|i| !def_operand_indices.contains(i))
        .collect();
    let use_roles = vec![UseRole::Data; use_operand_indices.len()];
    OpSemantics {
        def_operand_indices,
        def_roles,
        use_operand_indices,
        use_roles,
    }
}

fn apply_op_desc(desc: &OpDesc, _opcode: &str, operands: &[DecodedOperand]) -> OpSemantics {
    let mut def_operand_indices = Vec::new();
    let mut def_roles = Vec::new();

    // Data defs
    for i in 0..(desc.data_defs as usize).min(operands.len()) {
        def_operand_indices.push(i);
        def_roles.push(DefRole::Data);
    }

    // Pred defs after first operand (IADD3 pattern)
    if desc.pred_defs_after_first {
        for (idx, op) in operands.iter().enumerate().skip(1) {
            if is_pred_operand(op) {
                def_operand_indices.push(idx);
                def_roles.push(DefRole::Pred);
            } else {
                break;
            }
        }
    }

    // Uses: everything not a def
    let mut use_operand_indices = Vec::new();
    for idx in 0..operands.len() {
        if !def_operand_indices.contains(&idx) {
            use_operand_indices.push(idx);
        }
    }
    let mut use_roles = vec![UseRole::Data; use_operand_indices.len()];

    // Apply tail overrides (indexed from end of use list)
    for &(from_end, role) in &desc.use_tail_overrides {
        if use_roles.len() > from_end {
            let idx = use_roles.len() - 1 - from_end;
            use_roles[idx] = role;
        }
    }

    OpSemantics {
        def_operand_indices,
        def_roles,
        use_operand_indices,
        use_roles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(class: &str, idx: i32) -> DecodedOperand {
        match class {
            "P" | "UP" | "PT" | "UPT" => DecodedOperand::PredicateRegister {
                class: class.to_string(),
                idx,
                sense: true,
            },
            "UR" | "URZ" => DecodedOperand::UniformRegister {
                class: class.to_string(),
                idx,
                sign: 1,
                abs: false,
                reuse: false,
                ty: None,
            },
            _ => DecodedOperand::Register {
                class: class.to_string(),
                idx,
                sign: 1,
                abs: false,
                reuse: false,
                ty: None,
            },
        }
    }

    fn imm(v: i64) -> DecodedOperand {
        DecodedOperand::ImmediateI(v)
    }

    #[test]
    fn iadd3_defs_include_predicates() {
        let ops = vec![
            reg("R", 4),
            reg("P", 0),
            reg("P", 1),
            reg("R", 2),
            imm(1),
            reg("RZ", 0),
        ];
        let sem = derive_op_semantics("IADD3", &ops, false, false);
        assert!(sem.def_operand_indices.contains(&0)); // R4 data def
        assert!(sem.def_operand_indices.contains(&1)); // P0 pred def
        assert!(sem.def_operand_indices.contains(&2)); // P1 pred def
    }

    #[test]
    fn iadd3_x_last_uses_are_pred() {
        let ops = vec![
            reg("R", 3),
            reg("R", 0),
            reg("R", 1),
            reg("RZ", 0),
            reg("P", 2),
        ];
        let sem = derive_op_semantics("IADD3.X", &ops, false, false);
        // Last use should be PredCtrl, second-to-last PredIn
        assert!(sem.use_roles.contains(&UseRole::PredCtrl));
        assert!(sem.use_roles.contains(&UseRole::PredIn));
    }

    #[test]
    fn load_first_operand_is_def() {
        let ops = vec![reg("R", 1), reg("R", 0)];
        let sem = derive_op_semantics("LDG.E", &ops, true, false);
        assert_eq!(sem.def_operand_indices, vec![0]);
        assert_eq!(sem.use_roles, vec![UseRole::Addr]);
    }

    #[test]
    fn store_has_no_defs() {
        let ops = vec![reg("R", 0), reg("R", 1)];
        let sem = derive_op_semantics("STG.E", &ops, false, true);
        assert!(sem.def_operand_indices.is_empty());
        assert_eq!(sem.use_roles[0], UseRole::Addr);
        assert_eq!(sem.use_roles[1], UseRole::Data);
    }

    #[test]
    fn atomg_tracks_pred_and_data_defs_with_address_use() {
        let ops = vec![
            DecodedOperand::PredicateRegister {
                class: "PT".into(),
                idx: 0,
                sense: true,
            },
            reg("R", 7),
            DecodedOperand::Address {
                base: Box::new(reg("R", 4)),
                offset: None,
                width: Some(64),
                scale: None,
                raw: "[R4.64]".into(),
            },
            reg("R", 13),
        ];
        let sem = derive_op_semantics("ATOMG.E.ADD.STRONG.GPU", &ops, false, false);
        assert_eq!(sem.def_operand_indices, vec![0, 1]);
        assert_eq!(sem.def_roles, vec![DefRole::Pred, DefRole::Data]);
        assert_eq!(sem.use_operand_indices, vec![2, 3]);
        assert_eq!(sem.use_roles, vec![UseRole::Addr, UseRole::Data]);
    }

    #[test]
    fn red_has_no_defs_and_marks_memory_operand_as_address() {
        let ops = vec![
            DecodedOperand::Address {
                base: Box::new(reg("R", 2)),
                offset: None,
                width: Some(64),
                scale: None,
                raw: "[R2.64]".into(),
            },
            reg("R", 7),
        ];
        let sem = derive_op_semantics("RED.E.ADD.F32.FTZ.RN.STRONG.GPU", &ops, false, false);
        assert!(sem.def_operand_indices.is_empty());
        assert_eq!(sem.use_operand_indices, vec![0, 1]);
        assert_eq!(sem.use_roles, vec![UseRole::Addr, UseRole::Data]);
    }

    #[test]
    fn default_first_is_def_rest_are_uses() {
        let ops = vec![reg("R", 2), reg("R", 0), reg("R", 1)];
        let sem = derive_op_semantics("FADD", &ops, false, false);
        assert_eq!(sem.def_operand_indices, vec![0]);
        assert_eq!(sem.use_operand_indices, vec![1, 2]);
        assert_eq!(sem.use_roles, vec![UseRole::Data, UseRole::Data]);
    }

    #[test]
    fn lea_hi_x_last_use_is_pred_in() {
        let ops = vec![reg("R", 5), reg("R", 0), reg("R", 1), imm(2), reg("P", 0)];
        let sem = derive_op_semantics("LEA.HI.X", &ops, false, false);
        assert_eq!(*sem.use_roles.last().unwrap(), UseRole::PredIn);
    }

    #[test]
    fn lea_predicate_output_is_modeled_as_a_def() {
        let ops = vec![reg("R", 4), reg("P", 5), reg("R", 9), reg("R", 4), imm(2)];
        let sem = derive_op_semantics("LEA", &ops, false, false);
        assert_eq!(sem.def_operand_indices, vec![0, 1]);
        assert_eq!(sem.def_roles, vec![DefRole::Data, DefRole::Pred]);
        assert_eq!(sem.use_operand_indices, vec![2, 3, 4]);
        assert_eq!(
            sem.use_roles,
            vec![UseRole::Data, UseRole::Data, UseRole::Data]
        );
    }

    #[test]
    fn shfl_tracks_predicate_and_data_outputs() {
        let ops = vec![reg("PT", 0), reg("R", 0), reg("R", 3), imm(16), imm(31)];
        let sem = derive_op_semantics("SHFL.DOWN", &ops, false, false);
        assert_eq!(sem.def_operand_indices, vec![0, 1]);
        assert_eq!(sem.def_roles, vec![DefRole::Pred, DefRole::Data]);
        assert_eq!(sem.use_operand_indices, vec![2, 3, 4]);
        assert_eq!(
            sem.use_roles,
            vec![UseRole::Data, UseRole::Data, UseRole::Data]
        );
    }

    #[test]
    fn lop3_tracks_predicate_and_data_outputs() {
        let ops = vec![
            reg("P", 0),
            reg("R", 8),
            reg("R", 6),
            imm(31),
            reg("RZ", 0),
            imm(0xc0),
            reg("PT", 0),
        ];
        let sem = derive_op_semantics("LOP3.LUT", &ops, false, false);
        assert_eq!(sem.def_operand_indices, vec![0, 1]);
        assert_eq!(sem.def_roles, vec![DefRole::Pred, DefRole::Data]);
        assert_eq!(sem.use_operand_indices, vec![2, 3, 4, 5, 6]);
        assert_eq!(
            sem.use_roles,
            vec![
                UseRole::Data,
                UseRole::Data,
                UseRole::Data,
                UseRole::Data,
                UseRole::Data
            ]
        );
    }
}
