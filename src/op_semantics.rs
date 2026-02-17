use crate::parser::Operand;

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

fn is_pred_operand(op: &Operand) -> bool {
    matches!(
        op,
        Operand::Register { class, .. } if class == "P" || class == "UP"
    )
}

fn opcode_mnemonic(opcode: &str) -> &str {
    opcode.split('.').next().unwrap_or(opcode)
}

fn opcode_has_mod(opcode: &str, needle: &str) -> bool {
    opcode.split('.').skip(1).any(|m| m == needle)
}

pub fn derive_op_semantics(opcode: &str, operands: &[Operand], is_mem_load: bool, is_mem_store: bool) -> OpSemantics {
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

    if is_mem_store {
        let use_operand_indices = (0..operands.len()).collect::<Vec<_>>();
        let mut use_roles = Vec::with_capacity(use_operand_indices.len());
        for (idx, _) in use_operand_indices.iter().enumerate() {
            use_roles.push(if idx == 0 { UseRole::Addr } else { UseRole::Data });
        }
        return OpSemantics {
            def_operand_indices: Vec::new(),
            def_roles: Vec::new(),
            use_operand_indices,
            use_roles,
        };
    }

    let mnem = opcode_mnemonic(opcode);
    let mut def_operand_indices = Vec::new();
    let mut def_roles = Vec::new();
    if !operands.is_empty() {
        def_operand_indices.push(0);
        def_roles.push(DefRole::Data);
    }

    if matches!(mnem, "IADD3" | "UIADD3") && !opcode_has_mod(opcode, "X") {
        for (idx, op) in operands.iter().enumerate().skip(1) {
            if is_pred_operand(op) {
                def_operand_indices.push(idx);
                def_roles.push(DefRole::Pred);
            } else {
                break;
            }
        }
    }

    let mut use_operand_indices = Vec::new();
    for idx in 0..operands.len() {
        if !def_operand_indices.contains(&idx) {
            use_operand_indices.push(idx);
        }
    }
    let mut use_roles = Vec::with_capacity(use_operand_indices.len());
    for _ in &use_operand_indices {
        use_roles.push(UseRole::Data);
    }

    if matches!(mnem, "IADD3.X" | "UIADD3.X") && use_roles.len() >= 2 {
        if let Some(last) = use_roles.last_mut() {
            *last = UseRole::PredCtrl;
        }
        if use_roles.len() >= 2 {
            let idx = use_roles.len() - 2;
            use_roles[idx] = UseRole::PredIn;
        }
    }

    if opcode.starts_with("LEA.HI.X") || opcode.starts_with("ULEA.HI.X") {
        if let Some(last) = use_roles.last_mut() {
            *last = UseRole::PredIn;
        }
    }

    OpSemantics {
        def_operand_indices,
        def_roles,
        use_operand_indices,
        use_roles,
    }
}
