//! Memory-aware SSA analysis contract for the rewritten backend.
//!
//! Purpose:
//! - own all post-SSA facts needed by structurization consumers and AST
//!   lowering
//! - centralize CUDA memory-space classification, pointer-root recovery,
//!   ABI/profile facts, and declaration-driving type facts
//!
//! Inputs:
//! - optimized `FunctionIR`
//! - decoded instructions for ABI/profile recovery
//! - optional decoded-function metadata such as SM version
//!
//! Outputs:
//! - `FunctionAnalysis`, the canonical fact base for the post-SSA backend
//!
//! Invariants:
//! - later stages must consume these facts directly
//! - no later stage may parse rendered text to recover memory, type, or naming
//!   facts already represented here
//!
//! Algorithm summary:
//! - seed memory spaces from opcode families and operand forms
//! - refine with ABI/profile facts
//! - propagate address roots and pointer facts with SSA worklists
//! - synthesize shared/local objects from compatible proven roots
//!
//! This module must not:
//! - render pseudo-C
//! - mutate CFG/SSA structure
//! - depend on post-render cleanup behavior

use std::collections::{BTreeMap, VecDeque};

use crate::abi::{
    annotate_function_ir_constmem, infer_arg_aliases, AbiAnnotations, AbiArgAliases, AbiProfile,
    ConstMemSemantic, StatementRef,
};
use crate::ir::{FunctionIR, IRExpr, IRStatement, RValue, RegId};
use crate::memory_model::{CudaMemorySpace, MemAccessKind};
use crate::parser::DecodedInstruction;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddressRoot {
    ParamWord(u32),
    ConstSymbol(String),
    SharedObject(String),
    LocalObject(String),
    RegisterBase(RegId),
    Generic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemAccessInfo {
    pub block_id: usize,
    pub stmt_idx: usize,
    pub kind: MemAccessKind,
    pub space: CudaMemorySpace,
    pub bit_width: Option<u32>,
    pub vector_width: Option<u8>,
    pub root: AddressRoot,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionAnalysis {
    pub abi_profile: Option<AbiProfile>,
    pub abi_annotations: AbiAnnotations,
    pub abi_aliases: AbiArgAliases,
    pub root_by_reg: BTreeMap<RegId, AddressRoot>,
    pub mem_accesses: Vec<MemAccessInfo>,
}

pub fn analyze_function_ir(
    function_ir: &FunctionIR,
    instructions: &[DecodedInstruction],
    sm: Option<u32>,
) -> FunctionAnalysis {
    let abi_profile = (!instructions.is_empty()).then(|| AbiProfile::detect_with_sm(instructions, sm));
    let abi_annotations = abi_profile
        .map(|profile| annotate_function_ir_constmem(function_ir, profile))
        .unwrap_or_default();
    let abi_aliases = if abi_annotations.is_empty() {
        AbiArgAliases::default()
    } else {
        infer_arg_aliases(function_ir, &abi_annotations)
    };

    let root_by_reg = propagate_address_roots(function_ir, &abi_annotations, &abi_aliases);
    let mem_accesses = collect_memory_accesses(function_ir, &root_by_reg);

    FunctionAnalysis {
        abi_profile,
        abi_annotations,
        abi_aliases,
        root_by_reg,
        mem_accesses,
    }
}

fn propagate_address_roots(
    function_ir: &FunctionIR,
    abi_annotations: &AbiAnnotations,
    abi_aliases: &AbiArgAliases,
) -> BTreeMap<RegId, AddressRoot> {
    let mut roots = BTreeMap::<RegId, AddressRoot>::new();
    let mut queue = VecDeque::<RegId>::new();

    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            let Some(annotations) = abi_annotations.constmem_by_stmt.get(&stmt_ref) else {
                continue;
            };
            for (def_idx, def) in stmt.defs.iter().enumerate() {
                let Some(reg) = def.get_reg() else {
                    continue;
                };
                let Some(root) = root_from_annotation(def_idx, annotations, abi_aliases) else {
                    continue;
                };
                if insert_root_if_changed(&mut roots, reg.clone(), root) {
                    queue.push_back(reg.clone());
                }
            }
        }
    }

    while queue.pop_front().is_some() {
        for block in &function_ir.blocks {
            for stmt in &block.stmts {
                let propagated = match &stmt.value {
                    RValue::Phi(args) => {
                        let merged = merge_roots(args.iter().filter_map(|expr| expr_root(expr, &roots)));
                        merged.map(|root| (root, stmt.defs.clone()))
                    }
                    RValue::Op { opcode, args } if is_root_copy_like(opcode) => {
                        args.iter()
                            .find_map(|expr| expr_root(expr, &roots))
                            .map(|source| (source, stmt.defs.clone()))
                    }
                    RValue::Op { opcode, args } if is_pointer_arith_like(opcode) => {
                        args.iter()
                            .find_map(|expr| expr_root(expr, &roots))
                            .map(|source| (source, stmt.defs.clone()))
                    }
                    _ => None,
                };

                let Some((root, defs)) = propagated else {
                    continue;
                };
                for def in defs {
                    let Some(reg) = def.get_reg() else {
                        continue;
                    };
                    if insert_root_if_changed(&mut roots, reg.clone(), root.clone()) {
                        queue.push_back(reg.clone());
                    }
                }
            }
        }
    }

    roots
}

fn root_from_annotation(
    def_idx: usize,
    annotations: &[crate::abi::ConstMemAnnotation],
    abi_aliases: &AbiArgAliases,
) -> Option<AddressRoot> {
    for ann in annotations {
        match &ann.semantic {
            ConstMemSemantic::ParamWord { param_idx, .. } => {
                let lane_param = param_idx.checked_add(def_idx as u32)?;
                if abi_aliases.is_ptr_param(*param_idx) || abi_aliases.is_ptr_param(lane_param) {
                    return Some(AddressRoot::ParamWord(*param_idx));
                }
                return Some(AddressRoot::ParamWord(lane_param));
            }
            ConstMemSemantic::Builtin(name) => {
                return Some(AddressRoot::ConstSymbol((*name).to_string()));
            }
            ConstMemSemantic::AbiInternal(offset) => {
                return Some(AddressRoot::ConstSymbol(format!("abi_internal_0x{:x}", offset)));
            }
            ConstMemSemantic::Unknown { bank, offset } => {
                return Some(AddressRoot::ConstSymbol(format!("c[0x{:x}][0x{:x}]", bank, offset)));
            }
        }
    }
    None
}

fn insert_root_if_changed(
    roots: &mut BTreeMap<RegId, AddressRoot>,
    reg: RegId,
    root: AddressRoot,
) -> bool {
    match roots.get(&reg) {
        Some(existing) if existing == &root => false,
        _ => {
            roots.insert(reg, root);
            true
        }
    }
}

fn merge_roots(roots: impl Iterator<Item = AddressRoot>) -> Option<AddressRoot> {
    let mut iter = roots.peekable();
    let first = iter.peek()?.clone();
    if iter.all(|root| root == first) {
        Some(first)
    } else {
        None
    }
}

fn expr_root(expr: &IRExpr, roots: &BTreeMap<RegId, AddressRoot>) -> Option<AddressRoot> {
    match expr {
        IRExpr::Reg(reg) => roots.get(reg).cloned().or_else(|| Some(AddressRoot::RegisterBase(reg.clone()))),
        IRExpr::Addr64 { lo, hi } => {
            let lo_root = expr_root(lo, roots)?;
            let hi_root = expr_root(hi, roots)?;
            match (&lo_root, &hi_root) {
                (AddressRoot::ParamWord(lo_idx), AddressRoot::ParamWord(hi_idx))
                    if *hi_idx == lo_idx.saturating_add(1) =>
                {
                    Some(AddressRoot::ParamWord(*lo_idx))
                }
                _ if lo_root == hi_root => Some(lo_root),
                _ => None,
            }
        }
        IRExpr::Mem { base, .. } => expr_root(base, roots),
        IRExpr::Op { args, .. } => args.iter().find_map(|arg| expr_root(arg, roots)),
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => None,
    }
}

fn collect_memory_accesses(
    function_ir: &FunctionIR,
    root_by_reg: &BTreeMap<RegId, AddressRoot>,
) -> Vec<MemAccessInfo> {
    let mut out = Vec::new();
    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let Some((kind, space)) = classify_stmt_memory(stmt) else {
                continue;
            };
            let root = stmt
                .mem_addr_args
                .as_ref()
                .and_then(|args| args.first())
                .and_then(|expr| expr_root(expr, root_by_reg))
                .unwrap_or_else(|| default_root_for_space(space, stmt));
            let bit_width = opcode_memory_bit_width(stmt_opcode(stmt));
            out.push(MemAccessInfo {
                block_id: block.id,
                stmt_idx,
                kind,
                space,
                bit_width,
                vector_width: opcode_vector_width(stmt_opcode(stmt)),
                root,
            });
        }
    }
    out
}

fn classify_stmt_memory(stmt: &IRStatement) -> Option<(MemAccessKind, CudaMemorySpace)> {
    let opcode = stmt_opcode(stmt);
    let mnem = opcode.split('.').next().unwrap_or(opcode);

    if matches!(mnem, "LDC" | "ULDC" | "LDCU") {
        return Some((MemAccessKind::Load, CudaMemorySpace::Const));
    }
    if mnem.starts_with("LDG") {
        return Some((MemAccessKind::Load, CudaMemorySpace::Global));
    }
    if mnem.starts_with("STG") {
        return Some((MemAccessKind::Store, CudaMemorySpace::Global));
    }
    if mnem.starts_with("LDS") {
        return Some((MemAccessKind::Load, CudaMemorySpace::Shared));
    }
    if mnem.starts_with("STS") {
        return Some((MemAccessKind::Store, CudaMemorySpace::Shared));
    }
    if mnem.starts_with("LDL") {
        return Some((MemAccessKind::Load, CudaMemorySpace::Local));
    }
    if mnem.starts_with("STL") {
        return Some((MemAccessKind::Store, CudaMemorySpace::Local));
    }
    if mnem.starts_with("ATOMS") {
        return Some((MemAccessKind::Atomic, CudaMemorySpace::Shared));
    }
    if mnem.starts_with("ATOM") || mnem.starts_with("ATOMG") {
        return Some((MemAccessKind::Atomic, CudaMemorySpace::Global));
    }
    if mnem.starts_with("RED") {
        return Some((MemAccessKind::Reduction, CudaMemorySpace::Global));
    }
    None
}

fn default_root_for_space(space: CudaMemorySpace, stmt: &IRStatement) -> AddressRoot {
    match space {
        CudaMemorySpace::Shared => AddressRoot::SharedObject("shmem".to_string()),
        CudaMemorySpace::Local => AddressRoot::LocalObject("local_mem".to_string()),
        _ => stmt
            .mem_addr_args
            .as_ref()
            .and_then(|args| args.first())
            .and_then(first_reg_in_expr)
            .map(AddressRoot::RegisterBase)
            .unwrap_or(AddressRoot::Generic),
    }
}

fn stmt_opcode(stmt: &IRStatement) -> &str {
    match &stmt.value {
        RValue::Op { opcode, .. } => opcode,
        RValue::Phi(_) => "phi",
        RValue::ImmI(_) => "immi",
        RValue::ImmF(_) => "immf",
    }
}

fn is_root_copy_like(opcode: &str) -> bool {
    opcode.starts_with("MOV") || opcode.starts_with("UMOV") || opcode.starts_with("IMAD.MOV")
}

fn is_pointer_arith_like(opcode: &str) -> bool {
    opcode.starts_with("LEA")
        || opcode.starts_with("ULEA")
        || opcode.starts_with("IADD.64")
        || opcode.starts_with("IADD3.64")
        || opcode.starts_with("UIADD3.64")
        || opcode.starts_with("IMAD.WIDE")
}

fn first_reg_in_expr(expr: &IRExpr) -> Option<RegId> {
    match expr {
        IRExpr::Reg(reg) => Some(reg.clone()),
        IRExpr::Addr64 { lo, hi } => first_reg_in_expr(lo).or_else(|| first_reg_in_expr(hi)),
        IRExpr::Mem { base, offset, .. } => {
            first_reg_in_expr(base).or_else(|| offset.as_ref().and_then(|expr| first_reg_in_expr(expr)))
        }
        IRExpr::Op { args, .. } => args.iter().find_map(first_reg_in_expr),
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => None,
    }
}

fn opcode_vector_width(opcode: &str) -> Option<u8> {
    if opcode.split('.').any(|part| part == "128") {
        Some(4)
    } else if opcode.split('.').any(|part| part == "64") {
        Some(2)
    } else {
        None
    }
}

fn opcode_memory_bit_width(opcode: &str) -> Option<u32> {
    let parts = opcode.split('.').collect::<Vec<_>>();
    if parts.iter().any(|part| matches!(*part, "U8" | "S8" | "B8")) {
        return Some(8);
    }
    if parts.iter().any(|part| matches!(*part, "U16" | "S16" | "B16" | "F16")) {
        return Some(16);
    }
    if parts.iter().any(|part| matches!(*part, "64" | "U64" | "S64")) {
        return Some(64);
    }
    if parts.iter().any(|part| matches!(*part, "128")) {
        return Some(128);
    }
    if stmt_like_memory_opcode(opcode) {
        return Some(32);
    }
    None
}

fn stmt_like_memory_opcode(opcode: &str) -> bool {
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    mnem.starts_with("LD")
        || mnem.starts_with("ST")
        || mnem.starts_with("ATOM")
        || mnem.starts_with("RED")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, decode_sass};

    fn analyze(sass: &str) -> FunctionAnalysis {
        let instrs = decode_sass(sass);
        let cfg = build_cfg(instrs.clone());
        let fir = build_ssa(&cfg);
        analyze_function_ir(&fir, &instrs, None)
    }

    #[test]
    fn classifies_shared_and_local_memory_ops() {
        let sass = r#"
            /*0000*/ LDS R4, [R2] ;
            /*0010*/ STS [R3], R4 ;
            /*0020*/ LDL R5, [R6] ;
            /*0030*/ STL [R7], R5 ;
            /*0040*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        assert_eq!(analysis.mem_accesses.len(), 4);
        assert_eq!(analysis.mem_accesses[0].space, CudaMemorySpace::Shared);
        assert_eq!(analysis.mem_accesses[1].space, CudaMemorySpace::Shared);
        assert_eq!(analysis.mem_accesses[2].space, CudaMemorySpace::Local);
        assert_eq!(analysis.mem_accesses[3].space, CudaMemorySpace::Local);
    }

    #[test]
    fn classifies_const_and_global_memory_ops() {
        let sass = r#"
            /*0000*/ LDC R4, c[0x0][0x160] ;
            /*0010*/ LDG.E R6, [R8.64] ;
            /*0020*/ STG.E [R10.64], R6 ;
            /*0030*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        assert_eq!(analysis.mem_accesses.len(), 3);
        assert_eq!(analysis.mem_accesses[0].space, CudaMemorySpace::Const);
        assert_eq!(analysis.mem_accesses[1].space, CudaMemorySpace::Global);
        assert_eq!(analysis.mem_accesses[2].space, CudaMemorySpace::Global);
    }

    #[test]
    fn propagates_pointer_roots_from_param_window_loads() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E R6, [R4.64] ;
            /*0030*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        let global = analysis
            .mem_accesses
            .iter()
            .find(|access| access.space == CudaMemorySpace::Global)
            .expect("global access");
        assert_eq!(global.root, AddressRoot::ParamWord(0));
    }

    #[test]
    fn tracks_shared_atomic_accesses() {
        let sass = r#"
            /*0000*/ ATOMS.ADD R0, [R2], R4 ;
            /*0010*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        assert_eq!(analysis.mem_accesses.len(), 1);
        assert_eq!(analysis.mem_accesses[0].kind, MemAccessKind::Atomic);
        assert_eq!(analysis.mem_accesses[0].space, CudaMemorySpace::Shared);
    }
}
