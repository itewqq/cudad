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
    annotate_function_ir_constmem, infer_arg_aliases, infer_shared_word_pointee_type_for_function,
    AbiAnnotations, AbiArgAliases, AbiProfile, ConstMemSemantic, StatementRef,
};
use crate::ir::{FunctionIR, IRBlock, IRExpr, IRStatement, RValue, RegId};
use crate::memory_model::{CudaMemorySpace, MemAccessKind};
use crate::parser::DecodedInstruction;
use crate::type_inference::{infer_ssa_types, InferredType};

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
    pub shared_pointee_ty: Option<&'static str>,
    pub scalar_type_by_reg: BTreeMap<RegId, InferredType>,
    pub builtin_by_reg: BTreeMap<RegId, String>,
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
    let abi_profile = abi_profile_override.or_else(|| {
        (!instructions.is_empty()).then(|| AbiProfile::detect_with_sm(instructions, sm))
    });
    let abi_annotations = abi_profile
        .map(|profile| annotate_function_ir_constmem(function_ir, profile))
        .unwrap_or_default();
    let abi_aliases = if abi_annotations.is_empty() {
        AbiArgAliases::default()
    } else {
        infer_arg_aliases(function_ir, &abi_annotations)
    };
    let shared_pointee_ty = infer_shared_word_pointee_type_for_function(function_ir);
    let scalar_type_by_reg = infer_ssa_types(function_ir);
    let builtin_by_reg = recover_builtin_regs(function_ir, &abi_annotations);

    let (root_by_reg, byte_offset_by_reg) =
        propagate_address_facts(function_ir, &abi_annotations, &abi_aliases);
    let mem_accesses =
        collect_memory_accesses(function_ir, &abi_annotations, &abi_aliases, &root_by_reg);

    FunctionAnalysis {
        abi_profile,
        abi_annotations,
        abi_aliases,
        shared_pointee_ty,
        scalar_type_by_reg,
        builtin_by_reg,
        root_by_reg,
        byte_offset_by_reg,
        mem_accesses,
    }
}

fn recover_builtin_regs(
    function_ir: &FunctionIR,
    abi_annotations: &AbiAnnotations,
) -> BTreeMap<RegId, String> {
    let mut builtins = BTreeMap::<RegId, String>::new();
    let mut changed = true;

    while changed {
        changed = false;
        for block in &function_ir.blocks {
            for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
                let stmt_ref = StatementRef {
                    block_id: block.id,
                    stmt_idx,
                };
                let propagated = builtin_seed_from_stmt(stmt, &builtins)
                    .or_else(|| stmt_builtin_load_name(stmt, &stmt_ref, abi_annotations));
                let Some(name) = propagated else {
                    continue;
                };
                for def in &stmt.defs {
                    let Some(reg) = def.get_reg() else {
                        continue;
                    };
                    if !matches!(reg.class.as_str(), "R" | "UR") {
                        continue;
                    }
                    match builtins.get(reg) {
                        Some(existing) if existing == &name => {}
                        _ => {
                            builtins.insert(reg.clone(), name.clone());
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    builtins
}

fn builtin_seed_from_stmt(
    stmt: &IRStatement,
    builtins: &BTreeMap<RegId, String>,
) -> Option<String> {
    match &stmt.value {
        RValue::Phi(args) => merge_builtins(
            args.iter()
                .filter_map(|expr| expr_builtin_name(expr, builtins)),
        ),
        RValue::Op { opcode, args } if is_root_copy_like(opcode) => args
            .iter()
            .find_map(|expr| expr_builtin_name(expr, builtins)),
        RValue::Op { opcode, args }
            if matches!(
                opcode.split('.').next().unwrap_or(opcode),
                "S2R" | "CS2R" | "S2UR"
            ) =>
        {
            args.first()
                .and_then(|expr| expr_builtin_name(expr, builtins))
        }
        _ => None,
    }
}

fn builtin_name_from_annotations(
    def_idx: usize,
    annotations: &[crate::abi::ConstMemAnnotation],
) -> Option<String> {
    annotations.iter().find_map(|ann| match &ann.semantic {
        ConstMemSemantic::Builtin(name) if def_idx == 0 => Some((*name).to_string()),
        _ => None,
    })
}

fn stmt_builtin_load_name(
    stmt: &IRStatement,
    stmt_ref: &StatementRef,
    abi_annotations: &AbiAnnotations,
) -> Option<String> {
    let RValue::Op { opcode, .. } = &stmt.value else {
        return None;
    };
    let mnem = opcode.split('.').next().unwrap_or(opcode);
    if !(matches!(mnem, "LDC" | "ULDC" | "LDCU") || is_root_copy_like(opcode)) {
        return None;
    }
    abi_annotations
        .constmem_by_stmt
        .get(stmt_ref)
        .and_then(|annotations| builtin_name_from_annotations(0, annotations))
}

fn merge_builtins(builtins: impl Iterator<Item = String>) -> Option<String> {
    let mut iter = builtins;
    let first = iter.next()?;
    iter.all(|name| name == first).then_some(first)
}

fn expr_builtin_name(expr: &IRExpr, builtins: &BTreeMap<RegId, String>) -> Option<String> {
    match expr {
        IRExpr::Reg(reg) => builtins.get(reg).cloned(),
        IRExpr::Op { op, args } if args.is_empty() => {
            special_register_builtin_name(op).map(str::to_string)
        }
        _ => None,
    }
}

fn special_register_builtin_name(op: &str) -> Option<&'static str> {
    match op {
        "SR_CTAID.X" => Some("blockIdx.x"),
        "SR_CTAID.Y" => Some("blockIdx.y"),
        "SR_CTAID.Z" => Some("blockIdx.z"),
        "SR_TID.X" => Some("threadIdx.x"),
        "SR_TID.Y" => Some("threadIdx.y"),
        "SR_TID.Z" => Some("threadIdx.z"),
        "SR_NTID.X" => Some("blockDim.x"),
        "SR_NTID.Y" => Some("blockDim.y"),
        "SR_NTID.Z" => Some("blockDim.z"),
        "SR_GRIDID" => Some("gridId"),
        "SR_LANEID" => Some("laneId"),
        _ => None,
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
    let def_sites = collect_reg_def_sites(function_ir);

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
                if !can_carry_pointer_facts(reg) {
                    continue;
                }
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
                    RValue::Phi(args) => Some((
                        merge_roots(args.iter().filter_map(|expr| expr_root(expr, &roots))),
                        args.iter()
                            .map(|expr| expr_byte_offset(expr, &roots, &byte_offsets))
                            .collect::<Option<Vec<_>>>()
                            .and_then(|offsets| merge_offsets(offsets.into_iter()))
                            .or_else(|| {
                                let root = merge_roots(
                                    args.iter().filter_map(|expr| expr_root(expr, &roots)),
                                )?;
                                recover_pointer_phi_offset(
                                    function_ir,
                                    block,
                                    stmt,
                                    args,
                                    &root,
                                    &roots,
                                    &byte_offsets,
                                    &def_sites,
                                )
                            }),
                        stmt.defs.clone(),
                    )),
                    RValue::Op { opcode, args } if is_root_copy_like(opcode) => Some((
                        args.iter().find_map(|expr| expr_root(expr, &roots)),
                        args.iter()
                            .find_map(|expr| expr_byte_offset(expr, &roots, &byte_offsets)),
                        stmt.defs.clone(),
                    )),
                    RValue::Op { opcode, args } if is_pointer_arith_like(opcode) => {
                        propagate_pointer_arith(
                            opcode,
                            args,
                            annotated_root.as_ref(),
                            &roots,
                            &byte_offsets,
                        )
                        .map(|(root, byte_offset)| (Some(root), byte_offset, stmt.defs.clone()))
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
                    if !can_carry_pointer_facts(reg) {
                        continue;
                    }
                    let changed_root = root
                        .clone()
                        .is_some_and(|root| insert_root_if_changed(&mut roots, reg.clone(), root));
                    let changed_offset = byte_offset.clone().is_some_and(|byte_offset| {
                        insert_offset_if_changed(&mut byte_offsets, reg.clone(), byte_offset)
                    });
                    if changed_root || changed_offset {
                        queue.push_back(reg.clone());
                    }
                }
            }
        }
    }

    (roots, byte_offsets)
}

fn collect_reg_def_sites(function_ir: &FunctionIR) -> BTreeMap<RegId, (usize, usize)> {
    let mut def_sites = BTreeMap::new();
    for (block_idx, block) in function_ir.blocks.iter().enumerate() {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            for def in &stmt.defs {
                let Some(reg) = def.get_reg() else {
                    continue;
                };
                def_sites.insert(reg.clone(), (block_idx, stmt_idx));
            }
        }
    }
    def_sites
}

fn recover_pointer_phi_offset(
    function_ir: &FunctionIR,
    block: &IRBlock,
    stmt: &IRStatement,
    args: &[IRExpr],
    root: &AddressRoot,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
    def_sites: &BTreeMap<RegId, (usize, usize)>,
) -> Option<IRExpr> {
    let ptr_phi = stmt
        .defs
        .iter()
        .find_map(|def| def.get_reg())
        .filter(|reg| can_carry_pointer_facts(reg))?;
    let ptr_args = args
        .iter()
        .map(|arg| arg.get_reg().cloned())
        .collect::<Option<Vec<_>>>()?;

    for candidate in &block.stmts {
        if std::ptr::eq(candidate, stmt) {
            continue;
        }
        let RValue::Phi(idx_args) = &candidate.value else {
            continue;
        };
        if idx_args.len() != args.len() {
            continue;
        }
        let idx_phi = candidate
            .defs
            .iter()
            .find_map(|def| def.get_reg())
            .filter(|reg| can_carry_pointer_facts(reg))?;
        let idx_arg_regs = idx_args
            .iter()
            .map(|arg| arg.get_reg().cloned())
            .collect::<Option<Vec<_>>>()?;

        let mut scale = None;
        let mut bias = None;
        let matches = ptr_args
            .iter()
            .zip(idx_arg_regs.iter())
            .all(|(ptr_src, idx_src)| {
                let Some(match_info) = pointer_phi_source_scale(
                    function_ir,
                    ptr_src,
                    ptr_phi,
                    idx_src,
                    idx_phi,
                    root,
                    roots,
                    byte_offsets,
                    def_sites,
                ) else {
                    return false;
                };
                match scale {
                    Some(existing) if existing != match_info.scale => false,
                    None => {
                        scale = Some(match_info.scale);
                        if let Some(src_bias) = match_info.bias.clone() {
                            bias = Some(src_bias);
                        }
                        true
                    }
                    Some(_) => {
                        if let Some(src_bias) = match_info.bias {
                            match &bias {
                                Some(existing) => same_ir_expr(existing, &src_bias),
                                None => {
                                    bias = Some(src_bias);
                                    true
                                }
                            }
                        } else {
                            true
                        }
                    }
                }
            });
        let Some(scale) = scale.filter(|scale| *scale > 0) else {
            continue;
        };
        if matches {
            let idx_offset = mul_offset_expr(IRExpr::Reg(idx_phi.clone()), IRExpr::ImmI(scale));
            return Some(match bias {
                Some(bias) => add_offset_expr(bias, idx_offset),
                None => idx_offset,
            });
        }
    }

    None
}

fn pointer_phi_source_scale(
    function_ir: &FunctionIR,
    ptr_src: &RegId,
    ptr_phi: &RegId,
    idx_src: &RegId,
    idx_phi: &RegId,
    root: &AddressRoot,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
    def_sites: &BTreeMap<RegId, (usize, usize)>,
) -> Option<PointerPhiMatch> {
    let (block_idx, stmt_idx) = *def_sites.get(ptr_src)?;
    let ptr_stmt = &function_ir.blocks[block_idx].stmts[stmt_idx];
    let (step, scale, base) = imad_wide_source_parts(ptr_stmt)?;

    if base.get_reg() == Some(ptr_phi) {
        scalar_reg_matches_additive_update(function_ir, idx_src, idx_phi, step, def_sites)
            .then_some(PointerPhiMatch { scale, bias: None })
    } else if roots.get(ptr_src) == Some(root) && same_ir_expr(step, &IRExpr::Reg(idx_src.clone()))
    {
        Some(PointerPhiMatch {
            scale,
            bias: pointer_seed_bias(ptr_src, idx_src, scale, byte_offsets)?,
        })
    } else {
        None
    }
}

#[derive(Clone)]
struct PointerPhiMatch {
    scale: i64,
    bias: Option<IRExpr>,
}

fn pointer_seed_bias(
    ptr_src: &RegId,
    idx_src: &RegId,
    scale: i64,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<Option<IRExpr>> {
    let offset = byte_offsets.get(ptr_src)?;
    let scaled_idx = mul_offset_expr(IRExpr::Reg(idx_src.clone()), IRExpr::ImmI(scale));
    strip_offset_term(offset, &scaled_idx)
}

fn strip_offset_term(total: &IRExpr, term: &IRExpr) -> Option<Option<IRExpr>> {
    if same_ir_expr(total, term) {
        return Some(None);
    }
    let IRExpr::Op { op, args } = total else {
        return None;
    };
    if op != "+" || args.len() != 2 {
        return None;
    }
    if same_ir_expr(&args[0], term) {
        return Some(Some(args[1].clone()));
    }
    if same_ir_expr(&args[1], term) {
        return Some(Some(args[0].clone()));
    }
    None
}

fn imad_wide_source_parts(stmt: &IRStatement) -> Option<(&IRExpr, i64, &IRExpr)> {
    let RValue::Op { opcode, args } = &stmt.value else {
        return None;
    };
    if !(opcode.starts_with("IMAD.WIDE") || opcode.starts_with("UIMAD.WIDE")) || args.len() != 3 {
        return None;
    }
    let scale = match args.get(1)? {
        IRExpr::ImmI(value) => *value,
        _ => return None,
    };
    Some((args.first()?, scale, args.get(2)?))
}

fn scalar_reg_matches_additive_update(
    function_ir: &FunctionIR,
    update_reg: &RegId,
    phi_reg: &RegId,
    step: &IRExpr,
    def_sites: &BTreeMap<RegId, (usize, usize)>,
) -> bool {
    let Some((block_idx, stmt_idx)) = def_sites.get(update_reg).copied() else {
        return false;
    };
    let stmt = &function_ir.blocks[block_idx].stmts[stmt_idx];
    let phi_expr = IRExpr::Reg(phi_reg.clone());
    match &stmt.value {
        RValue::Op { opcode, args }
            if opcode.starts_with("IMAD") || opcode.starts_with("UIMAD") =>
        {
            matches!(
                args.as_slice(),
                [lhs, IRExpr::ImmI(1), rhs]
                    if same_ir_expr(lhs, step) && same_ir_expr(rhs, &phi_expr)
            ) || matches!(
                args.as_slice(),
                [IRExpr::ImmI(1), lhs, rhs]
                    if same_ir_expr(lhs, step) && same_ir_expr(rhs, &phi_expr)
            )
        }
        RValue::Op { opcode, args }
            if opcode.starts_with("IADD") || opcode.starts_with("UIADD") =>
        {
            scalar_add_args_match(args, &phi_expr, step)
        }
        _ => false,
    }
}

fn scalar_add_args_match(args: &[IRExpr], phi_expr: &IRExpr, step: &IRExpr) -> bool {
    let terms = args
        .iter()
        .filter(|expr| !is_zero_like_expr(expr))
        .cloned()
        .collect::<Vec<_>>();
    match terms.as_slice() {
        [lhs, rhs] => {
            (same_ir_expr(lhs, step) && same_ir_expr(rhs, phi_expr))
                || (same_ir_expr(rhs, step) && same_ir_expr(lhs, phi_expr))
        }
        _ => false,
    }
}

fn same_ir_expr(lhs: &IRExpr, rhs: &IRExpr) -> bool {
    lhs == rhs
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
                return Some(AddressRoot::ConstSymbol(format!(
                    "abi_internal_0x{:x}",
                    offset
                )));
            }
            ConstMemSemantic::Unknown { bank, offset } => {
                return Some(AddressRoot::ConstSymbol(format!(
                    "c[0x{:x}][0x{:x}]",
                    bank, offset
                )));
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
        IRExpr::Reg(reg) => can_carry_pointer_facts(reg)
            .then(|| roots.get(reg))
            .flatten()
            .cloned(),
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
        IRExpr::Reg(reg) => can_carry_pointer_facts(reg)
            .then(|| roots.get(reg).and_then(|_| byte_offsets.get(reg)))
            .flatten()
            .cloned(),
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
) -> Option<(AddressRoot, Option<IRExpr>)> {
    if opcode.starts_with("IMAD.WIDE") || opcode.starts_with("UIMAD.WIDE") {
        return propagate_imad_wide_pointer_arith(args, annotated_root, roots, byte_offsets);
    }
    if opcode.starts_with("IADD.64") {
        return propagate_iadd64_pointer_arith(args, roots, byte_offsets);
    }
    if opcode.starts_with("IADD3.64") || opcode.starts_with("UIADD3.64") {
        return propagate_iadd3_64_pointer_arith(args, roots, byte_offsets);
    }
    propagate_additive_pointer_arith(args, annotated_root, roots, byte_offsets)
}

fn propagate_additive_pointer_arith(
    args: &[IRExpr],
    annotated_root: Option<&AddressRoot>,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, Option<IRExpr>)> {
    let rooted_args = args
        .iter()
        .enumerate()
        .filter_map(|(idx, expr)| expr_root(expr, roots).map(|root| (idx, root)))
        .collect::<Vec<_>>();

    let (root_idx, root, mut byte_offset) = match rooted_args.as_slice() {
        [(root_idx, root)] => (
            *root_idx,
            root.clone(),
            expr_byte_offset(&args[*root_idx], roots, byte_offsets),
        ),
        [] => {
            let root_idx = args.iter().position(is_constmem_like_expr)?;
            (root_idx, annotated_root?.clone(), Some(IRExpr::ImmI(0)))
        }
        _ => return None,
    };
    for (idx, expr) in args.iter().enumerate() {
        if idx == root_idx || is_zero_like_expr(expr) {
            continue;
        }
        byte_offset = byte_offset.map(|offset| add_offset_expr(offset, expr.clone()));
    }
    Some((root, byte_offset))
}

fn propagate_imad_wide_pointer_arith(
    args: &[IRExpr],
    annotated_root: Option<&AddressRoot>,
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, Option<IRExpr>)> {
    let base_expr = args.get(2)?;
    let (root, base_offset) = if let Some(root) = expr_root(base_expr, roots) {
        let byte_offset = expr_byte_offset(base_expr, roots, byte_offsets);
        (root, byte_offset)
    } else if is_constmem_like_expr(base_expr) {
        (annotated_root?.clone(), Some(IRExpr::ImmI(0)))
    } else {
        return None;
    };
    let product = mul_offset_expr(args.first()?.clone(), args.get(1)?.clone());
    Some((
        root,
        base_offset.map(|offset| add_offset_expr(offset, product)),
    ))
}

fn propagate_iadd64_pointer_arith(
    args: &[IRExpr],
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, Option<IRExpr>)> {
    let lhs = wide_pair_expr(args.first()?.clone(), args.get(2)?.clone());
    let rhs = wide_pair_expr(args.get(1)?.clone(), args.get(3)?.clone());
    let lhs_root = expr_root(&lhs, roots).zip(expr_byte_offset(&lhs, roots, byte_offsets));
    let rhs_root = expr_root(&rhs, roots).zip(expr_byte_offset(&rhs, roots, byte_offsets));
    match (lhs_root, rhs_root) {
        (Some((root, base_offset)), None) => Some((
            root,
            Some(add_offset_expr(
                base_offset,
                wide_pair_offset_expr(args.get(1)?, args.get(3)?),
            )),
        )),
        (None, Some((root, base_offset))) => Some((
            root,
            Some(add_offset_expr(
                base_offset,
                wide_pair_offset_expr(args.first()?, args.get(2)?),
            )),
        )),
        (Some((root, _)), Some(_)) => Some((root, None)),
        _ => None,
    }
}

fn propagate_iadd3_64_pointer_arith(
    args: &[IRExpr],
    roots: &BTreeMap<RegId, AddressRoot>,
    byte_offsets: &BTreeMap<RegId, IRExpr>,
) -> Option<(AddressRoot, Option<IRExpr>)> {
    let wide_inputs = [(0usize, 3usize), (1usize, 4usize), (2usize, 5usize)];
    let rooted_inputs = wide_inputs
        .iter()
        .filter_map(|(lo_idx, hi_idx)| {
            let value = wide_pair_expr(args.get(*lo_idx)?.clone(), args.get(*hi_idx)?.clone());
            let root = expr_root(&value, roots)?;
            let offset = expr_byte_offset(&value, roots, byte_offsets)?;
            Some(((*lo_idx, *hi_idx), root, offset))
        })
        .collect::<Vec<_>>();
    let [((root_lo_idx, root_hi_idx), root, base_offset)] = rooted_inputs.as_slice() else {
        return None;
    };
    let mut byte_offset = base_offset.clone();
    for (lo_idx, hi_idx) in wide_inputs {
        if lo_idx == *root_lo_idx && hi_idx == *root_hi_idx {
            continue;
        }
        byte_offset = add_offset_expr(
            byte_offset,
            wide_pair_offset_expr(args.get(lo_idx)?, args.get(hi_idx)?),
        );
    }
    Some((root.clone(), Some(byte_offset)))
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
        || opcode.starts_with("UIMAD.WIDE")
        || opcode.starts_with("IADD.64")
        || opcode.starts_with("IADD3.64")
        || opcode.starts_with("UIADD3.64")
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

fn can_carry_pointer_facts(reg: &RegId) -> bool {
    matches!(reg.class.as_str(), "R" | "UR")
}

fn wide_pair_expr(lo: IRExpr, hi: IRExpr) -> IRExpr {
    IRExpr::Addr64 {
        lo: Box::new(lo),
        hi: Box::new(hi),
    }
}

fn wide_pair_offset_expr(lo: &IRExpr, hi: &IRExpr) -> IRExpr {
    if is_zero_like_expr(hi) {
        return lo.clone();
    }
    wide_pair_expr(lo.clone(), hi.clone())
}

fn first_reg_in_expr(expr: &IRExpr) -> Option<RegId> {
    match expr {
        IRExpr::Reg(reg) => can_carry_pointer_facts(reg).then_some(reg.clone()),
        IRExpr::Addr64 { lo, hi } => first_reg_in_expr(lo).or_else(|| first_reg_in_expr(hi)),
        IRExpr::Mem { base, offset, .. } => first_reg_in_expr(base)
            .or_else(|| offset.as_ref().and_then(|expr| first_reg_in_expr(expr))),
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
                || offset.as_deref().is_some_and(has_dynamic_offset_term)
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
    if parts
        .iter()
        .any(|part| matches!(*part, "U16" | "S16" | "B16" | "F16"))
    {
        return Some(16);
    }
    if parts
        .iter()
        .any(|part| matches!(*part, "64" | "U64" | "S64"))
    {
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
    use crate::{
        build_cfg, build_ssa, decode_sass, ir_algebra, ir_constprop, ir_copyprop, ir_cse, ir_dce,
        split_decoded_functions,
    };

    fn analyze(sass: &str) -> FunctionAnalysis {
        let instrs = decode_sass(sass);
        let cfg = build_cfg(instrs.clone());
        let fir = build_ssa(&cfg);
        analyze_function_ir(&fir, &instrs, None)
    }

    fn analyze_optimized_instrs(
        instrs: Vec<DecodedInstruction>,
        sm: Option<u32>,
    ) -> FunctionAnalysis {
        let cfg = build_cfg(instrs.clone());
        let ssa = build_ssa(&cfg);
        let dce1 = ir_dce(&ssa);
        let cp = ir_constprop(&dce1);
        let alg = ir_algebra(&cp);
        let cse = ir_cse(&alg, &cfg);
        let copy = ir_copyprop(&cse);
        let optimized = ir_dce(&copy);
        analyze_function_ir(&optimized, &instrs, sm)
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
    fn infers_float_shared_pointee_type_from_roundtrip() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R4, RZ, RZ, c[0x0][0x160] ;
            /*0010*/ IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x164] ;
            /*0020*/ LDG.E.CONSTANT R8, [R4.64] ;
            /*0030*/ STS [R0], R8 ;
            /*0040*/ LDS R9, [R0] ;
            /*0050*/ FADD R10, R9, 0f00000000 ;
            /*0060*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        assert_eq!(analysis.shared_pointee_ty, Some("float"));
    }

    #[test]
    fn carries_scalar_ssa_types_for_post_ssa_lowering() {
        let sass = r#"
            /*0000*/ MOV R4, c[0x0][0x160] ;
            /*0010*/ MOV R5, c[0x0][0x164] ;
            /*0020*/ LDG.E.CONSTANT R8, [R4.64] ;
            /*0030*/ FMUL R9, R8, 0f3f800000 ;
            /*0040*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        let load = crate::ir::RegId::new("R", 8, 1).with_ssa(0);
        let mul = crate::ir::RegId::new("R", 9, 1).with_ssa(0);
        assert_eq!(
            analysis.scalar_type_by_reg.get(&load).copied(),
            Some(InferredType::F32)
        );
        assert_eq!(
            analysis.scalar_type_by_reg.get(&mul).copied(),
            Some(InferredType::F32)
        );
    }

    #[test]
    fn recovers_special_register_builtins_through_copy_chains() {
        let sass = r#"
            /*0000*/ S2R R7, SR_TID.X ;
            /*0010*/ MOV R8, R7 ;
            /*0020*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        let tid = crate::ir::RegId::new("R", 7, 1).with_ssa(0);
        let tid_copy = crate::ir::RegId::new("R", 8, 1).with_ssa(0);
        assert_eq!(
            analysis.builtin_by_reg.get(&tid).map(String::as_str),
            Some("threadIdx.x")
        );
        assert_eq!(
            analysis.builtin_by_reg.get(&tid_copy).map(String::as_str),
            Some("threadIdx.x")
        );
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
    fn propagates_param_roots_through_iadd64_pairs() {
        let sass = r#"
            /*0000*/ LDC.64 R10, c[0x0][0x160] ;
            /*0010*/ SHF.R.S32.HI R13, RZ, 0x1f, R5 ;
            /*0020*/ IADD.64 R12, R5, R10 ;
            /*0030*/ LDG.E.U8 R4, [R12.64] ;
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
    fn propagates_param_roots_through_uiadd3_64_pairs() {
        let sass = r#"
            /*0000*/ LDC.64 UR18, c[0x0][0x160] ;
            /*0010*/ UIADD3.64 UR4, UPT, UPT, UR18, 0x10, URZ ;
            /*0020*/ LDG.E.U32 R0, [UR4.64] ;
            /*0030*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        let rooted_reg = crate::ir::RegId::new("UR", 4, 1).with_ssa(0);
        assert_eq!(
            analysis.root_by_reg.get(&rooted_reg),
            Some(&AddressRoot::ParamWord(0))
        );
        assert_eq!(
            analysis.byte_offset_by_reg.get(&rooted_reg),
            Some(&IRExpr::ImmI(16))
        );
        let global = analysis
            .mem_accesses
            .iter()
            .find(|access| access.space == CudaMemorySpace::Global)
            .expect("global access");
        assert_eq!(global.root, AddressRoot::ParamWord(0));
    }

    #[test]
    fn ignores_predicate_carry_defs_during_pointer_propagation() {
        let hist = split_decoded_functions(include_str!(
            "../test_cu/corpus_sm100/shared_mem_kernels.sass"
        ))
        .into_iter()
        .find(|func| func.name == "histogram256")
        .expect("histogram256 fixture should exist");
        let analysis = analyze_optimized_instrs(hist.instrs, hist.sm);
        assert!(
            analysis
                .root_by_reg
                .keys()
                .all(|reg| matches!(reg.class.as_str(), "R" | "UR")),
            "predicate/carry defs must not participate in pointer propagation: {:?}",
            analysis.root_by_reg.keys().collect::<Vec<_>>()
        );
        assert!(
            analysis
                .mem_accesses
                .iter()
                .any(|access| access.space == CudaMemorySpace::Global),
            "expected global accesses to remain analyzable after skipping carry predicates"
        );
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
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 3, 1).with_ssa(0)),
                    IRExpr::ImmI(4)
                ],
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
    fn propagates_param_roots_through_uimad_wide_pointer_bases() {
        let sass = r#"
            /*0000*/ MOV R8, c[0x0][0x160] ;
            /*0010*/ MOV R9, c[0x0][0x164] ;
            /*0020*/ UIMAD.WIDE UR6, UR4, 0x4, R8 ;
            /*0030*/ LDG.E.U32 R11, [UR6.64] ;
            /*0040*/ EXIT ;
        "#;
        let analysis = analyze(sass);
        let rooted_reg = crate::ir::RegId::new("UR", 6, 1).with_ssa(0);
        assert_eq!(
            analysis.root_by_reg.get(&rooted_reg),
            Some(&AddressRoot::ParamWord(0))
        );
        let global = analysis
            .mem_accesses
            .iter()
            .find(|access| access.space == CudaMemorySpace::Global)
            .expect("global access");
        assert_eq!(global.root, AddressRoot::ParamWord(0));
    }

    #[test]
    fn propagates_loop_carried_param_roots_even_when_offsets_diverge() {
        let cumsum = split_decoded_functions(include_str!("../test_cu/corpus/loop_kernels.sass"))
            .into_iter()
            .find(|func| func.name == "cumsum_linear")
            .expect("cumsum_linear fixture should exist");
        let analysis = analyze_optimized_instrs(cumsum.instrs, cumsum.sm);
        assert_eq!(
            analysis
                .root_by_reg
                .get(&crate::ir::RegId::new("R", 2, 1).with_ssa(5)),
            Some(&AddressRoot::ParamWord(0))
        );
        assert_eq!(
            analysis
                .root_by_reg
                .get(&crate::ir::RegId::new("R", 4, 1).with_ssa(4)),
            Some(&AddressRoot::ParamWord(2))
        );
    }

    #[test]
    fn recovers_loop_phi_byte_offsets_from_companion_index_phis() {
        let dot = split_decoded_functions(include_str!("../test_cu/corpus/loop_kernels.sass"))
            .into_iter()
            .find(|func| func.name == "dot_thread")
            .expect("dot_thread fixture should exist");
        let analysis = analyze_optimized_instrs(dot.instrs, dot.sm);
        let ptr_phi = crate::ir::RegId::new("R", 6, 1).with_ssa(4);
        assert_eq!(
            analysis.byte_offset_by_reg.get(&ptr_phi),
            Some(&IRExpr::Op {
                op: "*".to_string(),
                args: vec![
                    IRExpr::Reg(crate::ir::RegId::new("R", 5, 1).with_ssa(5)),
                    IRExpr::ImmI(4),
                ],
            })
        );
    }

    #[test]
    fn recovers_loop_phi_byte_offsets_with_entry_bias() {
        let idx_seed = crate::ir::RegId::new("R", 5, 1).with_ssa(0);
        let idx_phi = crate::ir::RegId::new("R", 5, 1).with_ssa(1);
        let idx_step = crate::ir::RegId::new("R", 5, 1).with_ssa(2);
        let base = crate::ir::RegId::new("R", 10, 1).with_ssa(0);
        let ptr_seed = crate::ir::RegId::new("R", 6, 1).with_ssa(0);
        let ptr_phi = crate::ir::RegId::new("R", 6, 1).with_ssa(1);
        let ptr_step = crate::ir::RegId::new("R", 6, 1).with_ssa(2);

        let block = IRBlock {
            id: 0,
            start_addr: 0,
            irdst: Vec::new(),
            stmts: vec![
                IRStatement {
                    defs: vec![IRExpr::Reg(ptr_seed.clone())],
                    value: RValue::Op {
                        opcode: "IMAD.WIDE".to_string(),
                        args: vec![
                            IRExpr::Reg(idx_seed.clone()),
                            IRExpr::ImmI(4),
                            IRExpr::Reg(base.clone()),
                        ],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                },
                IRStatement {
                    defs: vec![IRExpr::Reg(idx_step.clone())],
                    value: RValue::Op {
                        opcode: "IADD3".to_string(),
                        args: vec![
                            IRExpr::Reg(idx_phi.clone()),
                            IRExpr::ImmI(1),
                            IRExpr::ImmI(0),
                        ],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                },
                IRStatement {
                    defs: vec![IRExpr::Reg(ptr_step.clone())],
                    value: RValue::Op {
                        opcode: "IMAD.WIDE".to_string(),
                        args: vec![
                            IRExpr::ImmI(1),
                            IRExpr::ImmI(4),
                            IRExpr::Reg(ptr_phi.clone()),
                        ],
                    },
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                },
                IRStatement {
                    defs: vec![IRExpr::Reg(idx_phi.clone())],
                    value: RValue::Phi(vec![
                        IRExpr::Reg(idx_seed.clone()),
                        IRExpr::Reg(idx_step.clone()),
                    ]),
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                },
                IRStatement {
                    defs: vec![IRExpr::Reg(ptr_phi.clone())],
                    value: RValue::Phi(vec![
                        IRExpr::Reg(ptr_seed.clone()),
                        IRExpr::Reg(ptr_step.clone()),
                    ]),
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: Vec::new(),
                },
            ],
        };
        let function_ir = FunctionIR {
            blocks: vec![block],
        };
        let def_sites = collect_reg_def_sites(&function_ir);
        let mut roots = BTreeMap::new();
        roots.insert(base.clone(), AddressRoot::ParamWord(0));
        roots.insert(ptr_seed.clone(), AddressRoot::ParamWord(0));
        roots.insert(ptr_phi.clone(), AddressRoot::ParamWord(0));
        roots.insert(ptr_step.clone(), AddressRoot::ParamWord(0));
        let mut byte_offsets = BTreeMap::new();
        byte_offsets.insert(
            ptr_seed.clone(),
            IRExpr::Op {
                op: "+".to_string(),
                args: vec![
                    IRExpr::Op {
                        op: "*".to_string(),
                        args: vec![IRExpr::Reg(idx_seed.clone()), IRExpr::ImmI(4)],
                    },
                    IRExpr::ImmI(8),
                ],
            },
        );
        let ptr_phi_stmt = &function_ir.blocks[0].stmts[4];
        let recovered = recover_pointer_phi_offset(
            &function_ir,
            &function_ir.blocks[0],
            ptr_phi_stmt,
            match &ptr_phi_stmt.value {
                RValue::Phi(args) => args.as_slice(),
                _ => unreachable!("ptr phi should stay a phi"),
            },
            &AddressRoot::ParamWord(0),
            &roots,
            &byte_offsets,
            &def_sites,
        );
        assert_eq!(
            recovered,
            Some(IRExpr::Op {
                op: "+".to_string(),
                args: vec![
                    IRExpr::ImmI(8),
                    IRExpr::Op {
                        op: "*".to_string(),
                        args: vec![IRExpr::Reg(idx_phi), IRExpr::ImmI(4)],
                    },
                ],
            })
        );
    }

    #[test]
    fn does_not_classify_mismatched_addr64_pairs_from_lo_root_only() {
        let lo = crate::ir::RegId::new("R", 6, 1).with_ssa(0);
        let hi = crate::ir::RegId::new("R", 7, 1).with_ssa(0);
        let function_ir = FunctionIR {
            blocks: vec![IRBlock {
                id: 0,
                start_addr: 0,
                irdst: Vec::new(),
                stmts: vec![IRStatement {
                    defs: vec![IRExpr::Reg(crate::ir::RegId::new("R", 8, 1).with_ssa(0))],
                    value: RValue::Op {
                        opcode: "LDG.E.U32".to_string(),
                        args: vec![IRExpr::Addr64 {
                            lo: Box::new(IRExpr::Reg(lo.clone())),
                            hi: Box::new(IRExpr::Reg(hi.clone())),
                        }],
                    },
                    pred: None,
                    mem_addr_args: Some(vec![IRExpr::Addr64 {
                        lo: Box::new(IRExpr::Reg(lo.clone())),
                        hi: Box::new(IRExpr::Reg(hi.clone())),
                    }]),
                    pred_old_defs: Vec::new(),
                }],
            }],
        };
        let roots = BTreeMap::from([
            (lo.clone(), AddressRoot::ParamWord(0)),
            (hi.clone(), AddressRoot::ParamWord(6)),
        ]);
        let accesses = collect_memory_accesses(
            &function_ir,
            &AbiAnnotations::default(),
            &AbiArgAliases::default(),
            &roots,
        );
        assert_eq!(accesses.len(), 1);
        assert_ne!(accesses[0].root, AddressRoot::ParamWord(0));
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
