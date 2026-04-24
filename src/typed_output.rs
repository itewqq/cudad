use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use regex::Regex;

use crate::abi::{AbiArgAliases, ArgAliasKind, ArgScalarKind};
use crate::ast::{Expr, LValue};
use crate::ir::FunctionIR;
use crate::name_recovery::RecoveredSymbol;
use crate::semantic_lift::SemanticLiftResult;
use crate::type_inference::{infer_types, InferredType};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedMemoryDecls {
    pub shmem_u8: bool,
    pub shmem_words: bool,
}

pub fn collect_shared_memory_decls(lifted: Option<&SemanticLiftResult>) -> SharedMemoryDecls {
    let Some(lifted) = lifted else {
        return SharedMemoryDecls::default();
    };

    let mut usage = SharedMemoryDecls::default();
    for stmt in lifted.by_def.values() {
        visit_lvalue(&stmt.dest, &mut usage);
        if let Some(pred) = &stmt.pred {
            visit_expr(pred, &mut usage);
        }
        visit_expr(&stmt.rhs, &mut usage);
        if let Some(old) = &stmt.pred_old_val {
            visit_expr(old, &mut usage);
        }
    }
    usage
}

pub fn infer_recovered_name_types(
    function_ir: &FunctionIR,
    recovered_symbols: &[RecoveredSymbol],
) -> HashMap<String, &'static str> {
    let inferred = infer_types(function_ir);
    recovered_symbols
        .iter()
        .filter_map(|symbol| {
            symbol
                .ty_hint
                .map(|ty| (symbol.name.clone(), ty))
                .or_else(|| {
                    inferred.get(&symbol.reg_base).and_then(|ty| {
                        fallback_recovered_symbol_type(*ty)
                            .map(|ty| (symbol.name.clone(), ty))
                    })
                })
        })
        .collect()
}

fn fallback_recovered_symbol_type(ty: InferredType) -> Option<&'static str> {
    match ty {
        // Per-physical-register pointer inference is too coarse once SSA
        // versions of the same register are reused for low halves and scalar
        // temporaries. Only trust the richer per-symbol `ty_hint` for pointer
        // locals; otherwise fall back to conservative scalar defaults.
        InferredType::Ptr64 | InferredType::Bottom | InferredType::Top => None,
        other => Some(other.to_c_type()),
    }
}

pub fn render_typed_structured_output(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
    local_decls: &[String],
    recovered_symbols: Option<&[RecoveredSymbol]>,
    name_type_map: &HashMap<String, &'static str>,
    shared_memory: SharedMemoryDecls,
) -> String {
    let code_output = normalize_param_pointer_derefs(code_output, aliases);
    let params = aliases
        .map(|a| a.render_typed_param_list())
        .unwrap_or_default();
    let sig = if params.is_empty() {
        "__global__ void kernel(void)".to_string()
    } else {
        format!("__global__ void kernel({})", params.join(", "))
    };

    let mut decls = Vec::new();
    let mut declared_names = BTreeSet::new();
    let mut declared_types = HashMap::<String, String>::new();
    record_param_decls(aliases, &mut declared_names, &mut declared_types);
    let usage_type_map = infer_symbol_types_from_usage(&code_output, &declared_types);
    let mut shared_decl_known_types = declared_types.clone();
    if let Some(symbols) = recovered_symbols {
        for symbol in symbols {
            shared_decl_known_types.insert(
                symbol.name.clone(),
                preferred_symbol_type(symbol, name_type_map, &usage_type_map),
            );
        }
    }

    if shared_memory.shmem_u8 {
        decls.push("__shared__ uint8_t shmem_u8[256];".to_string());
    }
    if shared_memory.shmem_words {
        let shmem_ty = infer_shared_word_decl_type(
            &code_output,
            name_type_map,
            &usage_type_map,
            &shared_decl_known_types,
        );
        decls.push(format!("__shared__ {} shmem[];", shmem_ty));
    }
    record_explicit_decl_names(&decls, &mut declared_names, &mut declared_types);

    match recovered_symbols {
        None => decls.extend(local_decls.iter().cloned()),
        Some(symbols) => {
            for symbol in symbols {
                let ty = preferred_symbol_type(symbol, name_type_map, &usage_type_map);
                declared_names.insert(symbol.name.clone());
                declared_types.insert(symbol.name.clone(), ty.clone());
                decls.push(render_decl(&ty, &symbol.name, symbol.live_in));
            }
        }
    }
    if recovered_symbols.is_none() {
        record_explicit_decl_names(local_decls, &mut declared_names, &mut declared_types);
    }
    decls.extend(synthesize_fallback_decls(
        &code_output,
        &declared_names,
        &declared_types,
        name_type_map,
        &usage_type_map,
    ));

    let mut out = String::new();
    out.push_str(&sig);
    out.push_str(" {\n");
    for decl in &decls {
        out.push_str("  ");
        out.push_str(decl);
        out.push('\n');
    }
    if !decls.is_empty() {
        out.push('\n');
    }
    for line in code_output.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

fn normalize_param_pointer_derefs(
    code_output: &str,
    aliases: Option<&AbiArgAliases>,
) -> String {
    let Some(aliases) = aliases else {
        return code_output.to_string();
    };

    let param_widths = aliases
        .by_param
        .iter()
        .filter_map(|(idx, alias)| {
            (alias.kind == ArgAliasKind::Ptr64)
                .then_some(alias.pointee_ty?)
                .and_then(|ty| scalar_type_width_bytes(ty).map(|width| (format!("arg{}_ptr", idx), width)))
        })
        .collect::<HashMap<_, _>>();
    if param_widths.is_empty() {
        return code_output.to_string();
    }

    let bytes = code_output.as_bytes();
    let mut out = String::with_capacity(code_output.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx..].starts_with(b"*((") {
            if let Some((replacement, consumed)) =
                try_normalize_param_pointer_deref(&code_output[idx..], &param_widths)
            {
                out.push_str(&replacement);
                idx += consumed;
                continue;
            }
        }
        out.push(bytes[idx] as char);
        idx += 1;
    }
    out
}

fn try_normalize_param_pointer_deref(
    text: &str,
    param_widths: &HashMap<String, usize>,
) -> Option<(String, usize)> {
    let type_end = text[3..].find("*)")? + 3;
    let cast_ty = text[3..type_end].trim();
    let cast_width = scalar_type_width_bytes(cast_ty)?;

    let addr_open = type_end + 2;
    if text.as_bytes().get(addr_open).copied()? != b'(' {
        return None;
    }

    let mut depth = 0i32;
    let mut addr_close = None;
    for (offset, byte) in text.as_bytes()[addr_open..].iter().enumerate() {
        match *byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    addr_close = Some(addr_open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let addr_close = addr_close?;
    if text.as_bytes().get(addr_close + 1).copied()? != b')' {
        return None;
    }

    let addr = text[addr_open + 1..addr_close].trim();
    let base_end = addr
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .unwrap_or(addr.len());
    let base = addr[..base_end].trim();
    let remainder = addr[base_end..].trim_start();
    if !remainder.is_empty() && !matches!(remainder.as_bytes().first().copied(), Some(b'+') | Some(b'-')) {
        return None;
    }

    let expected_width = *param_widths.get(base)?;
    if cast_width != expected_width {
        return None;
    }
    if remainder.contains(base) || remainder.contains("uintptr_t") {
        return None;
    }

    Some((format!("*({})", addr), addr_close + 2))
}

fn scalar_type_width_bytes(ty: &str) -> Option<usize> {
    match ty.trim() {
        "uint8_t" | "int8_t" | "bool" => Some(1),
        "uint16_t" | "int16_t" | "__half" => Some(2),
        "uint32_t" | "int32_t" | "float" => Some(4),
        "uint64_t" | "int64_t" | "uintptr_t" | "intptr_t" | "double" => Some(8),
        _ => None,
    }
}

fn render_decl(ty: &str, name: &str, live_in: bool) -> String {
    if live_in {
        format!("{} {}; // live-in", ty, name)
    } else {
        format!("{} {};", ty, name)
    }
}

fn preferred_symbol_type(
    symbol: &RecoveredSymbol,
    name_type_map: &HashMap<String, &'static str>,
    usage_type_map: &HashMap<String, String>,
) -> String {
    match symbol.reg_base.0.as_str() {
        "P" | "UP" => return "bool".to_string(),
        _ => {}
    }
    if is_semantic_u32_name(&symbol.name) || is_param_lane_name(&symbol.name) {
        return "uint32_t".to_string();
    }
    if let Some(ty) = usage_type_map.get(&symbol.name) {
        return ty.clone();
    }
    if let Some(ty) = propagated_next_usage_type(&symbol.name, usage_type_map) {
        return ty;
    }
    if let Some(ty) = symbol.ty_hint {
        return ty.to_string();
    }
    name_type_map
        .get(&symbol.name)
        .copied()
        .unwrap_or("uint32_t")
        .to_string()
}

fn propagated_next_usage_type(
    name: &str,
    usage_type_map: &HashMap<String, String>,
) -> Option<String> {
    let prefix = format!("{}_next", name);
    let mut matches = usage_type_map
        .iter()
        .filter_map(|(other, ty)| other.starts_with(&prefix).then_some(ty.as_str()));
    let first = matches.next()?;
    if matches.any(|ty| ty != first) {
        return None;
    }
    Some(first.to_string())
}

fn is_semantic_u32_name(name: &str) -> bool {
    const SEEDS: &[&str] = &[
        "tid_x",
        "tid_y",
        "tid_z",
        "ctaid_x",
        "ctaid_y",
        "ctaid_z",
        "block_dim_x",
        "block_dim_y",
        "block_dim_z",
        "grid_dim_x",
        "grid_dim_y",
        "grid_dim_z",
        "lane_id",
        "cga_cta_id",
    ];
    SEEDS.iter().any(|seed| {
        name == *seed
            || name.strip_prefix(seed).is_some_and(|suffix| {
                suffix.starts_with('_') && suffix[1..].chars().all(|c| c.is_ascii_digit())
            })
    })
}

fn is_param_lane_name(name: &str) -> bool {
    (name.starts_with("arg") && name.ends_with("_ptr_lo32"))
        || (name.starts_with("arg") && name.ends_with("_ptr_hi32"))
        || (name.starts_with("arg") && name.ends_with("_u64_lo32"))
        || (name.starts_with("arg") && name.ends_with("_u64_hi32"))
}

fn fallback_ident_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b").unwrap())
}

fn record_param_decls(
    aliases: Option<&AbiArgAliases>,
    declared_names: &mut BTreeSet<String>,
    declared_types: &mut HashMap<String, String>,
) {
    let Some(aliases) = aliases else {
        return;
    };
    for alias in aliases.by_param.values() {
        match alias.kind {
            ArgAliasKind::Ptr64 => {
                let name = format!("arg{}_ptr", alias.param_idx);
                let ty = alias
                    .pointee_ty
                    .map(|elem| format!("{}*", elem))
                    .unwrap_or_else(|| "uintptr_t".to_string());
                declared_names.insert(name.clone());
                declared_types.insert(name, ty);
            }
            ArgAliasKind::U64 => {
                let name = format!("arg{}_u64", alias.param_idx);
                declared_names.insert(name.clone());
                declared_types.insert(name, "uint64_t".to_string());
            }
            ArgAliasKind::Word32 => {
                let name = format!("arg{}", alias.param_idx);
                let ty = match alias.scalar_kind.unwrap_or_else(|| {
                    if alias.signed_words.contains(&0) {
                        ArgScalarKind::I32
                    } else {
                        ArgScalarKind::U32
                    }
                }) {
                    ArgScalarKind::U32 => "uint32_t",
                    ArgScalarKind::I32 => "int32_t",
                    ArgScalarKind::F32 => "float",
                };
                declared_names.insert(name.clone());
                declared_types.insert(name, ty.to_string());
            }
        }
    }
}

fn record_explicit_decl_names(
    decls: &[String],
    declared_names: &mut BTreeSet<String>,
    declared_types: &mut HashMap<String, String>,
) {
    for decl in decls {
        let code = decl.split("//").next().unwrap_or("").trim();
        let Some(code) = code.strip_suffix(';') else {
            continue;
        };
        let mut parts = code.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let raw_name = parts.pop().unwrap().trim();
        let name = raw_name
            .split('[')
            .next()
            .unwrap_or(raw_name)
            .trim()
            .to_string();
        let ty = parts.join(" ");
        if name.is_empty() || ty.is_empty() {
            continue;
        }
        declared_names.insert(name.clone());
        declared_types.insert(name, ty);
    }
}

fn synthesize_fallback_decls(
    code_output: &str,
    declared_names: &BTreeSet<String>,
    declared_types: &HashMap<String, String>,
    name_type_map: &HashMap<String, &'static str>,
    usage_type_map: &HashMap<String, String>,
) -> Vec<String> {
    let ident_re = fallback_ident_re();
    let mut seen = declared_names.clone();
    let mut out = Vec::new();
    for line in code_output.lines() {
        let code = line.split("//").next().unwrap_or("");
        for mat in ident_re.find_iter(code) {
            let name = mat.as_str();
            if !should_synthesize_decl(code, mat.start(), mat.end(), name, &seen) {
                continue;
            }
            let ty = infer_fallback_decl_type(name, declared_types, name_type_map, usage_type_map);
            seen.insert(name.to_string());
            out.push(format!("{} {};", ty, name));
        }
    }
    out
}

fn should_synthesize_decl(
    line: &str,
    start: usize,
    end: usize,
    name: &str,
    declared_names: &BTreeSet<String>,
) -> bool {
    if declared_names.contains(name) || is_reserved_identifier(name) {
        return false;
    }
    let prev = line[..start].chars().next_back();
    if matches!(prev, Some('.')) {
        return false;
    }
    let next = line[end..].chars().find(|ch| !ch.is_whitespace());
    if matches!(next, Some('(' | '.' | ':')) {
        return false;
    }
    if name.starts_with("BB") && name[2..].chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    true
}

fn is_reserved_identifier(name: &str) -> bool {
    matches!(
        name,
        "if"
            | "else"
            | "do"
            | "while"
            | "for"
            | "switch"
            | "case"
            | "default"
            | "break"
            | "continue"
            | "return"
            | "goto"
            | "true"
            | "false"
            | "blockIdx"
            | "threadIdx"
            | "blockDim"
            | "gridDim"
            | "laneId"
            | "uintptr_t"
            | "uint64_t"
            | "int64_t"
            | "uint32_t"
            | "int32_t"
            | "uint16_t"
            | "int16_t"
            | "uint8_t"
            | "int8_t"
            | "float"
            | "double"
            | "bool"
    )
}

fn infer_fallback_decl_type(
    name: &str,
    declared_types: &HashMap<String, String>,
    name_type_map: &HashMap<String, &'static str>,
    usage_type_map: &HashMap<String, String>,
) -> String {
    if let Some(ty) = declared_types.get(name) {
        return ty.clone();
    }
    if let Some(ty) = usage_type_map.get(name) {
        return ty.clone();
    }
    if let Some(ty) = name_type_map.get(name) {
        return (*ty).to_string();
    }
    if let Some(base) = next_name_base(name) {
        if let Some(ty) = declared_types.get(&base) {
            return ty.clone();
        }
        if let Some(ty) = usage_type_map.get(&base) {
            return ty.clone();
        }
        if let Some(ty) = name_type_map.get(&base) {
            return (*ty).to_string();
        }
    }
    if is_boolish_name(name)
        || next_name_base(name)
            .as_deref()
            .is_some_and(is_boolish_name)
    {
        return "bool".to_string();
    }
    if is_semantic_u32_name(name)
        || next_name_base(name)
            .as_deref()
            .is_some_and(is_semantic_u32_name)
        || is_param_lane_name(name)
    {
        return "uint32_t".to_string();
    }
    "uint32_t".to_string()
}

fn usage_assign_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);$").expect("valid usage assign regex")
    })
}

fn typed_deref_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\*\(\(([^)]+)\*\)")
            .expect("valid typed dereference regex")
    })
}

fn arg_ptr_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(arg\d+_ptr)\b").expect("valid arg ptr regex"))
}

fn float_literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?x)
            (?:
                \b\d+\.\d*(?:[eE][+-]?\d+)?\b |
                \b\d*\.\d+(?:[eE][+-]?\d+)?\b
            )
        ")
        .expect("valid float literal regex")
    })
}

fn infer_symbol_types_from_usage(
    code_output: &str,
    declared_types: &HashMap<String, String>,
) -> HashMap<String, String> {
    let assign_re = usage_assign_re();
    let mut assigns = Vec::<(String, String)>::new();
    let mut assign_counts = HashMap::<String, usize>::new();
    let mut direct_votes = HashMap::<String, BTreeSet<String>>::new();
    for line in code_output.lines() {
        let code = line.split("//").next().unwrap_or("").trim();
        let Some(caps) = assign_re.captures(code) else {
            continue;
        };
        let lhs = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
        let rhs = caps.get(2).map(|m| m.as_str().trim().to_string()).unwrap_or_default();
        if !lhs.is_empty() && !rhs.is_empty() {
            *assign_counts.entry(lhs.clone()).or_insert(0) += 1;
            if let Some(ty) = infer_usage_expr_type_direct(&rhs, declared_types) {
                direct_votes.entry(lhs.clone()).or_default().insert(ty);
            }
            assigns.push((lhs, rhs));
        }
    }

    let mut known = declared_types.clone();
    for (lhs, votes) in &direct_votes {
        if votes.len() == 1 {
            if let Some(ty) = votes.iter().next() {
                known.insert(lhs.clone(), ty.clone());
            }
        }
    }
    for (lhs, rhs) in &assigns {
        let Some(ty) = infer_usage_expr_type_direct(rhs, &known) else {
            continue;
        };
        if assign_counts.get(lhs).copied().unwrap_or(0) == 1 {
            known.insert(lhs.clone(), ty);
        }
    }

    let mut changed = true;
    let mut passes = 0;
    while changed && passes < 8 {
        changed = false;
        passes += 1;
        for (lhs, rhs) in &assigns {
            if known.contains_key(lhs) || assign_counts.get(lhs).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(ty) = infer_usage_expr_type_from_known(rhs, &known) else {
                continue;
            };
            known.insert(lhs.clone(), ty);
            changed = true;
        }
        let mut consistent_votes = HashMap::<String, BTreeSet<String>>::new();
        for (lhs, rhs) in &assigns {
            if known.contains_key(lhs) {
                continue;
            }
            let Some(ty) = infer_usage_expr_type_from_known(rhs, &known) else {
                continue;
            };
            consistent_votes.entry(lhs.clone()).or_default().insert(ty);
        }
        for (lhs, votes) in consistent_votes {
            if votes.len() != 1 {
                continue;
            }
            let ty = votes.iter().next().cloned().unwrap_or_default();
            if ty.is_empty() {
                continue;
            }
            known.insert(lhs, ty);
            changed = true;
        }
    }

    known
        .into_iter()
        .filter(|(name, _)| !declared_types.contains_key(name))
        .collect()
}

fn infer_usage_expr_type_direct(
    expr: &str,
    known_types: &HashMap<String, String>,
) -> Option<String> {
    let expr = expr.trim();
    if let Some((_, then_expr, else_expr)) = split_top_level_ternary(expr) {
        let then_ty = infer_usage_expr_type_direct(then_expr, known_types);
        let else_ty = infer_usage_expr_type_direct(else_expr, known_types);
        return choose_usage_branch_type(then_ty, else_ty);
    }
    if let Some(caps) = typed_deref_re().captures(expr) {
        return Some(
            caps.get(1)?
                .as_str()
                .trim()
                .trim_start_matches('(')
                .trim()
                .to_string(),
        );
    }
    if expr.starts_with('*') {
        if let Some(arg_name) = arg_ptr_re()
            .captures(expr)
            .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
        {
            return declared_pointee_type(known_types.get(&arg_name)?);
        }
    }
    if let Some(cast_ty) = leading_scalar_cast_type(expr) {
        return Some(cast_ty.to_string());
    }
    if expr.starts_with("(float)")
        || expr.starts_with("((float)")
        || expr.contains("copysignf(")
        || expr.contains("exp2f(")
        || expr.contains("rsqrtf(")
        || expr.contains("rcp_approx(")
        || float_literal_re().is_match(expr)
    {
        return Some("float".to_string());
    }
    if expr.starts_with("(__half)") || expr.starts_with("((__half)") {
        return Some("__half".to_string());
    }
    None
}

fn split_top_level_ternary(expr: &str) -> Option<(&str, &str, &str)> {
    let mut depth = 0i32;
    let mut ternary_depth = 0i32;
    let mut question_idx = None;
    for (idx, ch) in expr.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '?' if depth == 0 => {
                ternary_depth += 1;
                question_idx.get_or_insert(idx);
            }
            ':' if depth == 0 && ternary_depth > 0 => {
                ternary_depth -= 1;
                if ternary_depth == 0 {
                    let q_idx = question_idx?;
                    let cond = expr[..q_idx].trim();
                    let then_expr = expr[q_idx + 1..idx].trim();
                    let else_expr = expr[idx + 1..].trim();
                    if cond.is_empty() || then_expr.is_empty() || else_expr.is_empty() {
                        return None;
                    }
                    return Some((cond, trim_wrapping_parens(then_expr), trim_wrapping_parens(else_expr)));
                }
            }
            _ => {}
        }
    }
    None
}

fn trim_wrapping_parens(expr: &str) -> &str {
    let mut current = expr.trim();
    loop {
        if !current.starts_with('(') || !current.ends_with(')') {
            return current;
        }
        let mut depth = 0i32;
        let mut wraps_entire = true;
        for (idx, ch) in current.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 && idx + ch.len_utf8() != current.len() {
                        wraps_entire = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if !wraps_entire || depth != 0 {
            return current;
        }
        current = current[1..current.len() - 1].trim();
    }
}

fn choose_usage_branch_type(
    then_ty: Option<String>,
    else_ty: Option<String>,
) -> Option<String> {
    match (then_ty, else_ty) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
        (Some(lhs), None) | (None, Some(lhs)) => Some(lhs),
        (Some(lhs), Some(rhs)) if lhs == "float" || rhs == "float" => Some("float".to_string()),
        (Some(lhs), Some(rhs)) if lhs == "__half" || rhs == "__half" => Some("__half".to_string()),
        _ => None,
    }
}

fn infer_shared_word_decl_type(
    code_output: &str,
    name_type_map: &HashMap<String, &'static str>,
    usage_type_map: &HashMap<String, String>,
    declared_types: &HashMap<String, String>,
) -> &'static str {
    let assign_re = usage_assign_re();
    let mut known_types = HashMap::new();
    for (name, ty) in name_type_map {
        known_types.insert(name.clone(), (*ty).to_string());
    }
    for (name, ty) in declared_types {
        known_types.insert(name.clone(), ty.clone());
    }
    for (name, ty) in usage_type_map {
        known_types.insert(name.clone(), ty.clone());
    }
    let mut saw_float = false;
    for line in code_output.lines() {
        let code = line.split("//").next().unwrap_or("").trim();
        let Some(caps) = assign_re.captures(code) else {
            continue;
        };
        let lhs = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        let rhs = caps.get(2).map(|m| m.as_str().trim()).unwrap_or_default();
        let ty = if lhs.starts_with("shmem[") {
            infer_usage_expr_type_from_known(rhs, &known_types)
        } else if rhs.contains("shmem[") {
            usage_type_map
                .get(lhs)
                .cloned()
                .or_else(|| name_type_map.get(lhs).map(|ty| (*ty).to_string()))
                .or_else(|| declared_types.get(lhs).cloned())
        } else {
            None
        };
        match ty {
            Some(ref ty) if ty == "float" => saw_float = true,
            Some(ref ty) if ty == "__half" => return "uint32_t",
            Some(_) => {}
            None => {}
        }
    }
    if saw_float {
        "float"
    } else {
        "uint32_t"
    }
}

fn leading_scalar_cast_type(expr: &str) -> Option<&'static str> {
    for ty in [
        "__half",
        "float",
        "bool",
        "uint8_t",
        "int8_t",
        "uint16_t",
        "int16_t",
        "uint32_t",
        "int32_t",
        "uint64_t",
        "int64_t",
        "uintptr_t",
        "intptr_t",
    ] {
        let single = format!("({})", ty);
        let double = format!("(({})", ty);
        if expr.starts_with(&single) || expr.starts_with(&double) {
            return Some(ty);
        }
    }
    None
}

fn infer_usage_expr_type_from_known(
    expr: &str,
    known_types: &HashMap<String, String>,
) -> Option<String> {
    let expr = expr.trim();
    if let Some(ty) = infer_usage_expr_type_direct(expr, known_types) {
        return Some(ty);
    }
    let mut saw_float = false;
    let mut saw_half = false;
    let mut saw_scalar: Option<String> = None;
    for mat in fallback_ident_re().find_iter(expr) {
        let name = mat.as_str();
        let Some(ty) = known_types.get(name) else {
            continue;
        };
        match ty.as_str() {
            "float" => saw_float = true,
            "__half" => saw_half = true,
            ty if !ty.ends_with('*') => {
                if let Some(prev) = &saw_scalar {
                    if prev != ty {
                        saw_scalar = Some("uint32_t".to_string());
                    }
                } else {
                    saw_scalar = Some(ty.to_string());
                }
            }
            _ => {}
        }
    }
    if saw_float {
        return Some("float".to_string());
    }
    if saw_half {
        return Some("__half".to_string());
    }
    saw_scalar
}

fn declared_pointee_type(ty: &str) -> Option<String> {
    ty.strip_suffix('*').map(|base| base.trim().to_string())
}

fn next_name_base(name: &str) -> Option<String> {
    name.split_once("_next")
        .map(|(base, _)| base.to_string())
        .filter(|base| !base.is_empty())
}

fn is_boolish_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('b') else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit() || ch == '_')
}

fn visit_lvalue(lvalue: &LValue, usage: &mut SharedMemoryDecls) {
    match lvalue {
        LValue::Raw(_) | LValue::Var(_) | LValue::PtrLane { .. } => {}
        LValue::Deref { addr, .. } => visit_expr(addr, usage),
        LValue::Indexed { base, index } => {
            visit_expr(base, usage);
            visit_expr(index, usage);
        }
    }
}

fn visit_expr(expr: &Expr, usage: &mut SharedMemoryDecls) {
    match expr {
        Expr::Builtin(name) if name == "shmem_u8" => usage.shmem_u8 = true,
        Expr::Builtin(name) if name == "shmem" => usage.shmem_words = true,
        Expr::Unary { arg, .. } => visit_expr(arg, usage),
        Expr::Binary { lhs, rhs, .. } => {
            visit_expr(lhs, usage);
            visit_expr(rhs, usage);
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            visit_expr(cond, usage);
            visit_expr(then_expr, usage);
            visit_expr(else_expr, usage);
        }
        Expr::CallLike { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                visit_expr(arg, usage);
            }
        }
        Expr::Load { addr, .. } => visit_expr(addr, usage),
        Expr::WidePtr { base, offset } => {
            visit_expr(base, usage);
            visit_expr(offset, usage);
        }
        Expr::Addr64 { lo, hi } => {
            visit_expr(lo, usage);
            visit_expr(hi, usage);
        }
        Expr::Cast { expr, .. } => visit_expr(expr, usage),
        Expr::Index { base, index } => {
            visit_expr(base, usage);
            visit_expr(index, usage);
        }
        Expr::Raw(_)
        | Expr::Imm(_)
        | Expr::Reg(_)
        | Expr::PtrLane { .. }
        | Expr::ConstMemSymbol(_)
        | Expr::Builtin(_) => {}
        Expr::LaneExtract { value, .. } => visit_expr(value, usage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{AliasConfidence, ArgAlias};
    use crate::ast::{Expr, LValue};
    use crate::semantic_lift::{DefRef, LiftedStmt, SemanticLiftResult};
    use std::collections::BTreeSet;

    fn pointer_alias(param_idx: u32, pointee_ty: &'static str) -> AbiArgAliases {
        AbiArgAliases {
            by_param: HashMap::from([(
                param_idx,
                ArgAlias {
                    param_idx,
                    kind: ArgAliasKind::Ptr64,
                    confidence: AliasConfidence::High,
                    observed_words: BTreeSet::new(),
                    scalar_kind: Some(ArgScalarKind::F32),
                    signed_words: BTreeSet::new(),
                    pointee_ty: Some(pointee_ty),
                },
            )])
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn render_uses_metadata_driven_decls() {
        let output = render_typed_structured_output(
            "tid_x = threadIdx.x;\nif (b0) return;",
            None,
            &[],
            Some(&[
                RecoveredSymbol {
                    name: "tid_x".to_string(),
                    reg_base: ("R".to_string(), 0),
                    ty_hint: None,
                    live_in: false,
                    order: 0,
                },
                RecoveredSymbol {
                    name: "b0".to_string(),
                    reg_base: ("P".to_string(), 0),
                    ty_hint: None,
                    live_in: true,
                    order: 1,
                },
            ]),
            &HashMap::from([(String::from("tid_x"), "uint32_t")]),
            SharedMemoryDecls::default(),
        );

        assert!(output.contains("uint32_t tid_x;"));
        assert!(output.contains("bool b0; // live-in"));
    }

    #[test]
    fn collect_shared_memory_decls_detects_lifted_shared_refs() {
        let mut lifted = SemanticLiftResult::default();
        lifted.by_def.insert(
            DefRef {
                block_id: 0,
                stmt_idx: 0,
                def_idx: 0,
            },
            LiftedStmt {
                dest: LValue::Indexed {
                    base: Box::new(Expr::Builtin("shmem_u8".to_string())),
                    index: Box::new(Expr::Imm("0".to_string())),
                },
                pred: None,
                rhs: Expr::Index {
                    base: Box::new(Expr::Builtin("shmem".to_string())),
                    index: Box::new(Expr::Imm("1".to_string())),
                },
                pred_old_val: None,
            },
        );

        let usage = collect_shared_memory_decls(Some(&lifted));
        assert!(usage.shmem_u8);
        assert!(usage.shmem_words);
    }

    #[test]
    fn render_adds_fallback_decls_for_post_cleanup_next_names() {
        let output = render_typed_structured_output(
            "v1_next = v1 + 1;\nif (b0_next) return;",
            None,
            &[],
            Some(&[
                RecoveredSymbol {
                    name: "v1".to_string(),
                    reg_base: ("R".to_string(), 1),
                    ty_hint: None,
                    live_in: false,
                    order: 0,
                },
                RecoveredSymbol {
                    name: "b0".to_string(),
                    reg_base: ("P".to_string(), 0),
                    ty_hint: None,
                    live_in: false,
                    order: 1,
                },
            ]),
            &HashMap::from([
                (String::from("v1"), "uint32_t"),
                (String::from("b0"), "bool"),
            ]),
            SharedMemoryDecls::default(),
        );

        assert!(output.contains("uint32_t v1_next;"));
        assert!(output.contains("bool b0_next;"));
    }

    #[test]
    fn usage_inference_honors_explicit_scalar_casts() {
        let inferred = infer_symbol_types_from_usage(
            "lo = ((uint32_t)((((uint8_t*)arg0_ptr) + (int64_t)idx)));\nhi = ((uint32_t)(((uint64_t)((((uint8_t*)arg0_ptr) + (int64_t)idx))) >> 32));",
            &HashMap::from([(String::from("arg0_ptr"), String::from("uint8_t*"))]),
        );

        assert_eq!(inferred.get("lo").map(String::as_str), Some("uint32_t"));
        assert_eq!(inferred.get("hi").map(String::as_str), Some("uint32_t"));
    }

    #[test]
    fn preferred_symbol_type_uses_next_family_usage_type() {
        let symbol = RecoveredSymbol {
            name: "v28".to_string(),
            reg_base: ("R".to_string(), 28),
            ty_hint: Some("uintptr_t"),
            live_in: false,
            order: 0,
        };
        let usage = HashMap::from([
            (String::from("v28_next_1"), String::from("uint32_t")),
            (String::from("v28_next_2"), String::from("uint32_t")),
        ]);

        assert_eq!(
            preferred_symbol_type(&symbol, &HashMap::new(), &usage),
            "uint32_t"
        );
    }

    #[test]
    fn usage_inference_recovers_pointer_load_type_from_ternary_guard() {
        let inferred = infer_symbol_types_from_usage(
            "v74 = !b6 ? (*(arg2_ptr + v15_next)) : 0;",
            &HashMap::from([(String::from("arg2_ptr"), String::from("float*"))]),
        );

        assert_eq!(inferred.get("v74").map(String::as_str), Some("float"));
    }

    #[test]
    fn usage_inference_handles_consistent_multi_assign_float_values() {
        let inferred = infer_symbol_types_from_usage(
            "v181 = v51 * v41 + v69;\n\
             v69_next = v52 * v73 + v166;\n\
             v38 = v181;\n\
             v38 = v69_next;",
            &HashMap::from([
                (String::from("v51"), String::from("float")),
                (String::from("v41"), String::from("float")),
                (String::from("v69"), String::from("float")),
                (String::from("v52"), String::from("float")),
                (String::from("v73"), String::from("float")),
                (String::from("v166"), String::from("float")),
            ]),
        );

        assert_eq!(inferred.get("v181").map(String::as_str), Some("float"));
        assert_eq!(inferred.get("v69_next").map(String::as_str), Some("float"));
        assert_eq!(inferred.get("v38").map(String::as_str), Some("float"));
    }

    #[test]
    fn render_infers_float_shared_decl_from_float_consumers() {
        let output = render_typed_structured_output(
            "v0 = shmem[idx];\nshmem[idx] = v1;",
            None,
            &[],
            Some(&[
                RecoveredSymbol {
                    name: "v0".to_string(),
                    reg_base: ("R".to_string(), 0),
                    ty_hint: Some("float"),
                    live_in: false,
                    order: 0,
                },
                RecoveredSymbol {
                    name: "v1".to_string(),
                    reg_base: ("R".to_string(), 1),
                    ty_hint: Some("uint32_t"),
                    live_in: false,
                    order: 1,
                },
            ]),
            &HashMap::from([
                (String::from("v0"), "float"),
                (String::from("v1"), "uint32_t"),
            ]),
            SharedMemoryDecls {
                shmem_u8: false,
                shmem_words: true,
            },
        );

        assert!(output.contains("__shared__ float shmem[];"), "got:\n{}", output);
    }

    #[test]
    fn render_infers_float_shared_decl_from_guarded_shared_reads() {
        let output = render_typed_structured_output(
            "v0 = !b0 ? (shmem[idx]) : 0.0;",
            None,
            &[],
            Some(&[RecoveredSymbol {
                name: "v0".to_string(),
                reg_base: ("R".to_string(), 0),
                ty_hint: Some("float"),
                live_in: false,
                order: 0,
            }]),
            &HashMap::from([(String::from("v0"), "float")]),
            SharedMemoryDecls {
                shmem_u8: false,
                shmem_words: true,
            },
        );

        assert!(output.contains("__shared__ float shmem[];"), "got:\n{}", output);
    }

    #[test]
    fn render_does_not_synthesize_scalar_decl_for_shared_array_name() {
        let output = render_typed_structured_output(
            "v0 = shmem[idx];",
            None,
            &[],
            Some(&[RecoveredSymbol {
                name: "v0".to_string(),
                reg_base: ("R".to_string(), 0),
                ty_hint: Some("float"),
                live_in: false,
                order: 0,
            }]),
            &HashMap::from([(String::from("v0"), "float")]),
            SharedMemoryDecls {
                shmem_u8: false,
                shmem_words: true,
            },
        );

        assert!(output.contains("__shared__ float shmem[];"), "got:\n{}", output);
        assert!(!output.contains("uint32_t shmem;"), "got:\n{}", output);
    }

    #[test]
    fn normalize_param_pointer_derefs_rewrites_matching_width_loads() {
        let aliases = pointer_alias(0, "float");
        let normalized =
            normalize_param_pointer_derefs("v0 = *((uint32_t*)(arg0_ptr + idx));", Some(&aliases));

        assert_eq!(normalized, "v0 = *(arg0_ptr + idx);");
    }

    #[test]
    fn normalize_param_pointer_derefs_keeps_byte_offset_forms() {
        let aliases = pointer_alias(2, "float");
        let normalized = normalize_param_pointer_derefs(
            "v0 = *((float*)(arg2_ptr + (int64_t)idx - ((uint32_t)(uintptr_t)arg2_ptr)));",
            Some(&aliases),
        );

        assert_eq!(
            normalized,
            "v0 = *((float*)(arg2_ptr + (int64_t)idx - ((uint32_t)(uintptr_t)arg2_ptr)));"
        );
    }
}
