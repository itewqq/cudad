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
    pub constant_byte_offset: Option<i64>,
    pub has_dynamic_offset: bool,
    pub root: AddressRoot,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FunctionAnalysis {
    pub abi_profile: Option<AbiProfile>,
    pub abi_annotations: AbiAnnotations,
    pub abi_aliases: AbiArgAliases,
    pub root_by_reg: BTreeMap<RegId, AddressRoot>,
    pub byte_offset_by_reg: BTreeMap<RegId, IRExpr>,
    pub mem_accesses: Vec<MemAccessInfo>,
}

pub fn analyze_function_ir(
    function_ir: &FunctionIR,
    instructions: &[DecodedInstruction],
    sm: Option<u32>,
) -> FunctionAnalysis {
    analyze_function_ir_with_profile(function_ir, instructions, None, sm)
}

pub fn analyze_function_ir_with_profile(
    function_ir: &FunctionIR,
    instructions: &[DecodedInstruction],
    abi_profile_override: Option<AbiProfile>,
    sm: Option<u32>,
) -> FunctionAnalysis {
    let abi_profile = abi_profile_override
        .or_else(|| (!instructions.is_empty()).then(|| AbiProfile::detect_with_sm(instructions, sm)));
    let abi_annotations = abi_profile
        .map(|profile| annotate_function_ir_constmem(function_ir, profile))
        .unwrap_or_default();
    let abi_aliases = if abi_annotations.is_empty() {
        AbiArgAliases::default()
    } else {
        infer_arg_aliases(function_ir, &abi_annotations)
    };

    let (root_by_reg, byte_offset_by_reg) =
        propagate_address_facts(function_ir, &abi_annotations, &abi_aliases);
    let mem_accesses =
        collect_memory_accesses(function_ir, &abi_annotations, &abi_aliases, &root_by_reg);

    FunctionAnalysis {
        abi_profile,
        abi_annotations,
        abi_aliases,
        root_by_reg,
        byte_offset_by_reg,
        mem_accesses,
    }
}

fn propagate_address_facts(
    function_ir: &FunctionIR,
    abi_annotations: &AbiAnnotations,
    abi_aliases: &AbiArgAliases,
) -> (BTreeMap<RegId, AddressRoot>, BTreeMap<RegId, IRExpr>) {
    let mut roots = BTreeMap::<RegId, AddressRoot>::new();
    let mut byte_offsets = BTreeMap::<RegId, IRExpr>::new();
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
                    insert_offset_if_changed(&mut byte_offsets, reg.clone(), IRExpr::ImmI(0));
                    queue.push_back(reg.clone());
                }
            }
        }
    }

    while queue.pop_front().is_some() {
        for block in &function_ir.blocks {
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                let stmt_ref = StatementRef {
                    block_id: block.id,
                    stmt_idx,
                };
                let annotated_root = abi_annotations
                    .constmem_by_stmt
                    .get(&stmt_ref)
                    .and_then(|annotations| root_from_annotation(0, annotations, abi_aliases));
                let propagated = match &stmt.value {
                    RValue::Phi(args) => {
                        let merged = merge_roots(args.iter().filter_map(|expr| expr_root(expr, &roots)));
                        let byte_offset = merge_offsets(
                            args.iter()
                                .filter_map(|expr| expr_byte_offset(expr, &roots, &byte_offsets)),
                        );
                        match (merged, byte_offset) {
                            (Some(root), Some(byte_offset)) => Some((root, byte_offset, stmt.defs.clone())),
                            _ => None,
                        }
                    }
                    RValue::Op { opcode, args } if is_root_copy_like(opcode) => {
                        args.iter().find_map(|expr| {
                            let root = expr_root(expr, &roots)?;
                            let byte_offset = expr_byte_offset(expr, &roots, &byte_offsets)?;
                            Some((root, byte_offset, stmt.defs.clone()))
                        })
                    }
                    RValue::Op { opcode, args } if is_pointer_arith_like(opcode) => {
                        propagate_pointer_arith(
                            opcode,
                            args,
                            annotated_root.as_ref(),
                            &roots,
                            &byte_offsets,
                        )
                            .map(|(root, byte_offset)| (root, byte_offset, stmt.defs.clone()))
                    }
                    _ => None,
                };

                let Some((root, byte_offset, defs)) = propagated else {
                    continue;
                };
                for def in defs {
                    let Some(reg) = def.get_reg() else {
                        continue;
                    };
                    let changed_root = insert_root_if_changed(&mut roots, reg.clone(), root.clone());
                    let changed_offset =
                        insert_offset_if_changed(&mut byte_offsets, reg.clone(), byte_offset.clone());
                    if changed_root || changed_offset {
                        queue.push_back(reg.clone());
                    }
                }
            }
        }
    }

    (roots, byte_offsets)
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

fn merge_offsets(offsets: impl Iterator<Item = IRExpr>) -> Option<IRExpr> {
    let mut iter = offsets.peekable();
    let first = iter.peek()?.clone();
    if iter.all(|offset| offset == first) {
        Some(first)
    } else {
        None
    }
}

fn expr_root(expr: &IRExpr, roots: &BTreeMap<RegId, AddressRoot>) -> Option<AddressRoot> {
    match expr {
        IRExpr::Reg(reg) => {
            if matches!(reg.class.as_str(), "RZ" | "URZ" | "PT" | "UPT") {
                None
            } else {
                roots.get(reg).cloned()
            }
        }
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

fn expr_byte_offset(
    expr: &IRExpr,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<IRExpr> {
    match expr {
        IRExpr::Reg(reg) => roots.get(reg).and_then(|_| byte_offsets.get(reg)).cloned(),
        IRExpr::Addr64 { lo, hi } => {
            let lo_reg = lo.get_reg()?;
            let hi_reg = hi.get_reg()?;
            let lo_root = roots.get(lo_reg)?;
            let hi_root = roots.get(hi_reg)?;
            if compatible_addr64_roots(lo_root, hi_root) {
                byte_offsets.get(lo_reg).cloned()
            } else {
                None
            }
        }
        _ => None,
    }
}

fn compatible_addr64_roots(lo_root: &AddressRoot, hi_root: &AddressRoot) -> bool {
    matches!(
        (lo_root, hi_root),
        (AddressRoot::ParamWord(lo_idx), AddressRoot::ParamWord(hi_idx))
            if *hi_idx == lo_idx.saturating_add(1)
    ) || lo_root == hi_root
}

fn propagate_pointer_arith(
    opcode: &str,
    args: &[IRExpr],
    annotated_root: Option<&AddressRoot>,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, IRExpr)> {
    if opcode.starts_with("IMAD.WIDE") || opcode.starts_with("UIMAD.WIDE") {
        return propagate_imad_wide_pointer_arith(args, annotated_root, roots, byte_offsets);
    }
    propagate_additive_pointer_arith(args, annotated_root, roots, byte_offsets)
}

fn propagate_additive_pointer_arith(
    args: &[IRExpr],
    annotated_root: Option<&AddressRoot>,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, IRExpr)> {
    let rooted_args = args
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr_root(expr, roots).map(|root| (idx, root)))
        .collect::<Vec<_>>();

    let (root_idx, root, mut byte_offset) = match rooted_args.as_slice() {
        [(root_idx, root)] => (
            *root_idx,
            root.clone(),
            expr_byte_offset(&args[*root_idx], roots, byte_offsets)?,
        ),
        [] => {
            let root_idx = args.iter().position(is_constmem_like_expr)?;
            (root_idx, annotated_root?.clone(), IRExpr::ImmI(0))
        }
        _ => return None,
    };
    for (idx, expr) in args.iter().enumerate() {
        if idx == root_idx || is_zero_like_expr(expr) {
            continue;
        }
        byte_offset = add_offset_expr(byte_offset, expr.clone());
    }
    Some((root, byte_offset))
}

fn propagate_imad_wide_pointer_arith(
    args: &[IRExpr],
    annotated_root: Option<&AddressRoot>,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, IRExpr)> {
    let base_expr = args.get(2)?;
    let (root, base_offset) = if let Some(root) = expr_root(base_expr, roots) {
        let byte_offset = expr_byte_offset(base_expr, roots, byte_offsets)?;
        (root, byte_offset)
    } else if is_constmem_like_expr(base_expr) {
        (annotated_root?.clone(), IRExpr::ImmI(0))
    } else {
        return None;
    };
    let product = mul_offset_expr(args.first()?.clone(), args.get(1)?.clone());
    Some((root, add_offset_expr(base_offset, product)))
}

fn collect_memory_accesses(
    function_ir: &FunctionIR,
    abi_annotations: &AbiAnnotations,
    abi_aliases: &AbiArgAliases,
    root_by_reg: &BTreeMap<RegId, AddressRoot>,
) -> Vec<MemAccessInfo> {
    let mut out = Vec::new();
    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            let Some((kind, seed_space)) = classify_stmt_memory(stmt) else {
                continue;
            };
            let stmt_ref = StatementRef {
                block_id: block.id,
                stmt_idx,
            };
            let annotated_root = abi_annotations
                .constmem_by_stmt
                .get(&stmt_ref)
                .and_then(|annotations| root_from_annotation(0, annotations, abi_aliases));
            let space = refine_memory_space(seed_space, annotated_root.as_ref());
            let root = stmt
                .mem_addr_args
                .as_ref()
                .and_then(|args| args.first())
                .and_then(|expr| expr_root(expr, root_by_reg))
                .or(annotated_root)
                .unwrap_or_else(|| default_root_for_space(space, stmt));
            let bit_width = opcode_memory_bit_width(stmt_opcode(stmt));
            out.push(MemAccessInfo {
                block_id: block.id,
                stmt_idx,
                kind,
                space,
                bit_width,
                vector_width: opcode_vector_width(stmt_opcode(stmt)),
                constant_byte_offset: stmt
                    .mem_addr_args
                    .as_ref()
                    .and_then(|args| args.first())
                    .and_then(constant_byte_offset),
                has_dynamic_offset: stmt
                    .mem_addr_args
                    .as_ref()
                    .and_then(|args| args.first())
                    .is_some_and(has_dynamic_offset_term),
                root,
            });
        }
    }
    out
}

fn refine_memory_space(
    seed_space: CudaMemorySpace,
    annotated_root: Option<&AddressRoot>,
) -> CudaMemorySpace {
    match (seed_space, annotated_root) {
        (CudaMemorySpace::Const, Some(AddressRoot::ParamWord(_))) => CudaMemorySpace::Param,
        _ => seed_space,
    }
}

fn insert_offset_if_changed(
    offsets: &mut BTreeMap<RegId, IRExpr>,
    reg: RegId,
    byte_offset: IRExpr,
) -> bool {
    match offsets.get(&reg) {
        Some(existing) if existing == &byte_offset => false,
        _ => {
            offsets.insert(reg, byte_offset);
            true
        }
    }
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
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    matches!(mnem, "LEA" | "ULEA" | "IADD" | "IADD3" | "UIADD" | "UIADD3")
        || opcode.starts_with("IMAD.WIDE")
}

fn add_offset_expr(lhs: IRExpr, rhs: IRExpr) -> IRExpr {
    if is_zero_like_expr(&lhs) {
        return rhs;
    }
    if is_zero_like_expr(&rhs) {
        return lhs;
    }
    IRExpr::Op {
        op: "+".to_string(),
        args: vec![lhs, rhs],
    }
}

fn mul_offset_expr(lhs: IRExpr, rhs: IRExpr) -> IRExpr {
    if is_zero_like_expr(&lhs) || is_zero_like_expr(&rhs) {
        return IRExpr::ImmI(0);
    }
    if is_one_like_expr(&lhs) {
        return rhs;
    }
    if is_one_like_expr(&rhs) {
        return lhs;
    }
    IRExpr::Op {
        op: "*".to_string(),
        args: vec![lhs, rhs],
    }
}

fn is_zero_like_expr(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::ImmI(0) => true,
        IRExpr::Reg(reg) => matches!(reg.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

fn is_one_like_expr(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::ImmI(1))
}

fn is_constmem_like_expr(expr: &IRExpr) -> bool {
    matches!(expr, IRExpr::Op { op, args } if op == "ConstMem" && args.len() == 2)
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

fn constant_byte_offset(expr: &IRExpr) -> Option<i64> {
    match expr {
        IRExpr::Mem { base, offset, .. } => Some(
            static_immediate_component(base.as_ref())
                + offset
                    .as_deref()
                    .map(static_immediate_component)
                    .unwrap_or(0),
        ),
        IRExpr::Op { op, args } if op == "ConstMem" && args.len() == 2 => match args[1] {
            IRExpr::ImmI(offset) => Some(offset),
            _ => None,
        },
        _ => None,
    }
}

fn static_immediate_component(expr: &IRExpr) -> i64 {
    match expr {
        IRExpr::ImmI(value) => *value,
        IRExpr::Op { op, args } if op == "+" => args.iter().map(static_immediate_component).sum(),
        IRExpr::Op { op, args } if op == "-" && args.len() == 2 => {
            static_immediate_component(&args[0]) - static_immediate_component(&args[1])
        }
        IRExpr::Addr64 { lo, hi } => {
            static_immediate_component(lo)
                + static_immediate_component(hi).checked_shl(32).unwrap_or(0)
        }
        IRExpr::Reg(_) | IRExpr::ImmF(_) | IRExpr::Mem { .. } | IRExpr::Op { .. } => 0,
    }
}

fn has_dynamic_offset_term(expr: &IRExpr) -> bool {
    match expr {
        IRExpr::Mem { base, offset, .. } => {
            has_dynamic_offset_term(base.as_ref())
                || offset
                    .as_deref()
                    .is_some_and(has_dynamic_offset_term)
        }
        IRExpr::Reg(reg) => !matches!(reg.class.as_str(), "RZ" | "URZ"),
        IRExpr::Addr64 { lo, hi } => {
            has_dynamic_offset_term(lo.as_ref()) || has_dynamic_offset_term(hi.as_ref())
        }
        IRExpr::Op { args, .. } => args.iter().any(has_dynamic_offset_term),
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => false,
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
        assert_eq!(analysis.mem_accesses[0].space, CudaMemorySpace::Param);
        assert_eq!(analysis.mem_accesses[0].root, AddressRoot::ParamWord(0));
        assert!(!analysis.mem_accesses[0].has_dynamic_offset);
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

    #[test]
    fn propagates_param_roots_through_iadd3_address_arithmetic() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ IADD3 R4, R4, 0x4, RZ ;
            /*0030*/ LDG.E R6, [R4.64] ;
            /*0040*/ EXIT ;
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
    fn propagates_param_roots_through_imad_wide_inline_constmem_bases() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R6, R3, 0x4, c[0x0][0x160] ;
            /*0010*/ LDG.E.U32 R11, [R6.64] ;
            /*0020*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        let rooted_reg = crate::ir::RegId::new("R", 6, 1).with_ssa(0);
        assert_eq!(
            analysis.root_by_reg.get(&rooted_reg),
            Some(&AddressRoot::ParamWord(0))
        );
        let byte_offset = analysis
            .byte_offset_by_reg
            .get(&rooted_reg)
            .expect("missing rooted byte offset");
        assert_eq!(
            byte_offset,
            &IRExpr::Op {
                op: "*".to_string(),
                args: vec![IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)), IRExpr::ImmI(4)],
            }
        );
        let global = analysis
            .mem_accesses
            .iter()
            .find(|access| access.space == CudaMemorySpace::Global)
            .expect("global access");
        assert_eq!(global.root, AddressRoot::ParamWord(0));
    }

    #[test]
    fn tracks_constant_byte_offsets_for_space_objects() {
        let sass = r#"
            /*0000*/ STS [R2+0x8], R4 ;
            /*0010*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        assert_eq!(analysis.mem_accesses[0].constant_byte_offset, Some(8));
        assert!(analysis.mem_accesses[0].has_dynamic_offset);
    }
}
