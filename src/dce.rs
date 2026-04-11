//! Text-level dead-code elimination for structured pseudocode output.
//!
//! This pass operates on the *rendered* pseudocode string (after name
//! recovery) rather than on the IR.  It identifies simple assignments
//! `ident = expr;` whose LHS is never referenced anywhere else in the
//! output and removes them.
//!
//! The pass iterates until a fixpoint because removing one dead assignment
//! may render another assignment dead.

use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Cached regex objects — compiled once, reused across all calls.
// ---------------------------------------------------------------------------

fn assign_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);$").unwrap())
}

fn assign_lhs_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=").unwrap())
}

fn ident_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*)\b").unwrap())
}

fn side_effect_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:__syncthreads|ATOMS|STS|STG|STS\.U|STG\.E|RED|ATOM|BAR|MEMBAR|DEPBAR|FENCE)",
        )
        .unwrap()
    })
}

fn constant_rhs_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^(?:threadIdx\.[xyz]|blockIdx\.[xyz]|blockDim\.[xyz]|gridDim\.[xyz]|cgaCtaId|laneId|abi_internal_0x[0-9A-Fa-f]+|c\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]|arg\d+_ptr\.(?:lo32|hi32)|arg\d+_word\d+\.(?:lo32|hi32)|arg\d+|param_\d+)$",
        )
        .unwrap()
    })
}

fn literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^-?\d+$|^true$|^false$|^0x[0-9A-Fa-f]+$").unwrap())
}

fn guard_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*if\s*\(([^)]+)\)\s*return\s*;").unwrap())
}

fn plus_zero_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r" \+ 0([;),\s]|$)").unwrap())
}

fn zero_plus_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b0 \+ ").unwrap())
}

fn minus_zero_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r" - 0([;),\s]|$)").unwrap())
}

fn times_one_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r" \* 1([;),\s]|$)").unwrap())
}

fn one_times_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b1 \* ").unwrap())
}

// addr64 collapsing regexes
fn addr64_use_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"addr64\(([A-Za-z_]\w*),\s*([A-Za-z_]\w*)\)").unwrap())
}

fn lo_add_def_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Match: vLO = OFFSET + PTR.lo32;  or  vLO = OFFSET + PTR_lo32;
        Regex::new(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\+\s*([A-Za-z_]\w*[._]lo32)\s*;")
            .unwrap()
    })
}

fn lea_hi_def_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*([A-Za-z_]\w*)\s*=\s*lea_hi_x(?:_sx32)?\(([A-Za-z_]\w*),\s*([A-Za-z_]\w*[._]hi32),\s*\d+,\s*([A-Za-z_]\w*)\)\s*;")
            .unwrap()
    })
}

/// Match: vHI = EXPR + PTR.hi32 + (bN ? 1 : 0);
/// or:    vHI = EXPR + PTR_hi32 + (bN ? 1 : 0);
/// This is the non-lea_hi alternative for the hi-part of addr64 construction.
fn hi_add_carry_def_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\+\s*([A-Za-z_]\w*[._]hi32)\s*\+\s*\(([A-Za-z_]\w*)\s*\?\s*1\s*:\s*0\)\s*;")
            .unwrap()
    })
}

fn carry_def_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Match: bN = carry_u32_add3(OFFSET, PTR.lo32, 0);  or  PTR_lo32
        Regex::new(r"^\s*([A-Za-z_]\w*)\s*=\s*carry_u32_add3\(([A-Za-z_]\w*),\s*([A-Za-z_]\w*[._]lo32),\s*0\)\s*;")
            .unwrap()
    })
}

/// Run text-level dead-code elimination on `input`.
///
/// Returns the cleaned output with dead assignments removed.
pub fn eliminate_dead_code(input: &str) -> String {
    let mut text = input.to_string();
    // CSE: unify redundant loads of the same pure expression.
    text = eliminate_common_subexpressions(&text);
    // Constant propagation: inline trivial integer/boolean constants.
    text = propagate_constants(&text);
    // Algebraic simplification: fold X+0, 0+X, X*1, etc.
    text = simplify_algebra(&text);
    // Collapse addr64(lo, hi) patterns into typed pointer expressions.
    text = collapse_addr64_patterns(&text);
    // Iterate to fixpoint — removing one line may make another variable dead.
    loop {
        let next = dce_one_pass(&text);
        if next == text {
            break;
        }
        text = next;
    }
    // Remove duplicate `if (X) return;` guards.
    text = eliminate_duplicate_guards(&text);
    text
}

/// One round of DCE.  Returns the text with dead assignments removed.
fn dce_one_pass(input: &str) -> String {
    let assign = assign_re();
    let side_effect = side_effect_re();
    let ident = ident_re();

    let lines: Vec<&str> = input.lines().collect();

    // Count how many times each identifier appears as a non-LHS reference.
    // We'll build: for each line, what is the LHS identifier (if pure assignment)?
    // Then for each identifier, count occurrences across all lines excluding
    // the assignment-LHS position of lines that define it.

    struct LineInfo<'a> {
        text: &'a str,
        lhs: Option<String>,      // The LHS identifier if this is a pure assignment
        removable: bool,          // Whether this line is a candidate for removal
    }

    let mut line_infos: Vec<LineInfo> = Vec::with_capacity(lines.len());

    for &line in &lines {
        let trimmed = line.trim();
        // Skip empty, comments, control flow
        if trimmed.is_empty()
            || trimmed.starts_with("//")
            || trimmed.starts_with("if ")
            || trimmed.starts_with("if(")
            || trimmed.starts_with("} else")
            || trimmed.starts_with("else")
            || trimmed.starts_with('}')
            || trimmed.starts_with('{')
            || trimmed.starts_with("return")
            || trimmed.starts_with("break")
            || trimmed.starts_with("continue")
            || trimmed.starts_with("do ")
            || trimmed.starts_with("do{")
            || trimmed.starts_with("} while")
            || trimmed.starts_with("while")
            || trimmed.starts_with("for ")
            || trimmed.starts_with("*")       // memory store: *addr = ...
            || trimmed.starts_with("__sync")
        {
            line_infos.push(LineInfo { text: line, lhs: None, removable: false });
            continue;
        }

        if let Some(caps) = assign.captures(trimmed) {
            let lhs_name = caps.get(1).unwrap().as_str().to_string();
            let rhs = caps.get(2).unwrap().as_str();

            // Don't remove if RHS has side effects
            let has_side_effect = side_effect.is_match(rhs)
                || rhs.contains("__syncthreads")
                || rhs.contains("atomicInc")
                || rhs.contains("atomicAdd");

            if has_side_effect {
                line_infos.push(LineInfo { text: line, lhs: None, removable: false });
            } else {
                line_infos.push(LineInfo { text: line, lhs: Some(lhs_name), removable: true });
            }
        } else {
            // Not a simple assignment — side-effecting statement, etc.
            line_infos.push(LineInfo { text: line, lhs: None, removable: false });
        }
    }

    // Count all identifier occurrences across the entire text.
    let mut total_count: HashMap<String, usize> = HashMap::new();
    for info in &line_infos {
        for m in ident.find_iter(info.text) {
            *total_count.entry(m.as_str().to_string()).or_insert(0) += 1;
        }
    }

    // For each removable assignment, count how many times LHS appears as its
    // own definition.  If total references == definition count, the variable
    // is dead (only appears on the LHS of its own assignments).
    let mut def_count: HashMap<String, usize> = HashMap::new();
    for info in &line_infos {
        if info.removable {
            if let Some(ref lhs) = info.lhs {
                *def_count.entry(lhs.clone()).or_insert(0) += 1;
            }
        }
    }

    // An identifier is dead if every occurrence is as a definition LHS.
    let dead_vars: BTreeSet<String> = def_count
        .iter()
        .filter(|(name, &defs)| {
            let total = total_count.get(name.as_str()).copied().unwrap_or(0);
            total <= defs
        })
        .map(|(name, _)| name.clone())
        .collect();

    if dead_vars.is_empty() {
        return input.to_string();
    }

    // Remove lines that define dead variables.
    let result: Vec<&str> = line_infos
        .iter()
        .filter(|info| {
            if let Some(ref lhs) = info.lhs {
                if info.removable && dead_vars.contains(lhs) {
                    return false; // Remove this line
                }
            }
            true
        })
        .map(|info| info.text)
        .collect();

    result.join("\n")
}

/// Text-level common-subexpression elimination.
///
/// When `X = <pure_rhs>;` and later `Y = <pure_rhs>;` appear (same RHS string),
/// and X has not been reassigned between the two, replace all uses of Y with X
/// and remove the Y assignment.
///
/// Only applies to "kernel-constant" RHS expressions:
/// - Special registers: `threadIdx.x`, `blockIdx.x`, `blockDimX`, `cgaCtaId`
/// - ABI constants: `abi_internal_0x*`, `c[0x*][0x*]`
/// - Argument components: `arg*_ptr.lo32`, `arg*_ptr.hi32`, `arg*`
/// - Explicit parameters: `param_*`
fn eliminate_common_subexpressions(input: &str) -> String {
    let assign = assign_re();
    let ident = ident_re();
    let constant_rhs = constant_rhs_re();

    let lines: Vec<&str> = input.lines().collect();

    // First pass: count how many times each variable appears as an LHS.
    let mut lhs_count: HashMap<String, usize> = HashMap::new();
    for &line in &lines {
        let trimmed = line.trim();
        if let Some(caps) = assign.captures(trimmed) {
            let lhs = caps.get(1).unwrap().as_str().to_string();
            *lhs_count.entry(lhs).or_insert(0) += 1;
        }
    }

    // Second pass: find all assignments of constant RHS and build a rename map.
    // For each constant RHS, keep the first variable that was assigned to it.
    let mut rhs_to_first_var: HashMap<String, String> = HashMap::new();
    // Map from variable name to what it should be renamed to.
    let mut rename_map: HashMap<String, String> = HashMap::new();
    // Track lines to remove (index).
    let mut lines_to_remove: BTreeSet<usize> = BTreeSet::new();

    for (line_idx, &line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if let Some(caps) = assign.captures(trimmed) {
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let rhs = caps.get(2).unwrap().as_str().trim().to_string();

            if constant_rhs.is_match(&rhs) {
                // Only CSE if:
                // 1. The anchor variable (first_var) is never reassigned
                // 2. The variable being renamed (lhs) is never reassigned
                let lhs_is_single_def = lhs_count.get(&lhs).copied().unwrap_or(0) == 1;
                if let Some(first_var) = rhs_to_first_var.get(&rhs) {
                    let anchor_is_single_def =
                        lhs_count.get(first_var).copied().unwrap_or(0) == 1;
                    if anchor_is_single_def && lhs_is_single_def && lhs != *first_var {
                        rename_map.insert(lhs.clone(), first_var.clone());
                        lines_to_remove.insert(line_idx);
                        continue;
                    }
                } else if lhs_is_single_def {
                    rhs_to_first_var.insert(rhs, lhs.clone());
                }
            }
        }
    }

    if rename_map.is_empty() {
        return input.to_string();
    }

    // Second pass: apply the rename map and remove redundant lines.
    let mut result = Vec::with_capacity(lines.len());
    for (line_idx, &line) in lines.iter().enumerate() {
        if lines_to_remove.contains(&line_idx) {
            continue;
        }
        // Apply renames in this line.
        let new_line = ident.replace_all(line, |caps: &regex::Captures| {
            let name = caps.get(1).unwrap().as_str();
            if let Some(replacement) = rename_map.get(name) {
                replacement.clone()
            } else {
                name.to_string()
            }
        });
        result.push(new_line.into_owned());
    }

    result.join("\n")
}

/// Text-level constant propagation.
///
/// When `X = <small_literal>;` (integer literal, true, false) and X is
/// defined exactly once, replace all uses of X with the literal and remove
/// the assignment line.
///
/// Only propagates constants that are "small" (≤ 10 chars) to avoid bloating
/// the output — large hex constants are better left in named variables.
fn propagate_constants(input: &str) -> String {
    let assign = assign_re();
    let ident = ident_re();
    let literal = literal_re();

    let lines: Vec<&str> = input.lines().collect();

    // Count how many times each variable appears as LHS of an assignment.
    let mut lhs_count: HashMap<String, usize> = HashMap::new();
    for &line in &lines {
        let trimmed = line.trim();
        if let Some(caps) = assign.captures(trimmed) {
            let lhs = caps.get(1).unwrap().as_str().to_string();
            *lhs_count.entry(lhs).or_insert(0) += 1;
        }
    }

    // Find single-def assignments of small literals.
    let mut const_map: HashMap<String, String> = HashMap::new();
    let mut lines_to_remove: BTreeSet<usize> = BTreeSet::new();

    for (line_idx, &line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(caps) = assign.captures(trimmed) {
            let lhs = caps.get(1).unwrap().as_str().to_string();
            let rhs = caps.get(2).unwrap().as_str().trim().to_string();

            // Only propagate if:
            // 1. Single definition
            // 2. RHS is a small literal
            // 3. Literal is short enough to inline (≤ 10 chars)
            if lhs_count.get(&lhs).copied().unwrap_or(0) == 1
                && literal.is_match(&rhs)
                && rhs.len() <= 10
            {
                const_map.insert(lhs, rhs);
                lines_to_remove.insert(line_idx);
            }
        }
    }

    if const_map.is_empty() {
        return input.to_string();
    }

    // Apply the constant map and remove assignment lines.
    let mut result = Vec::with_capacity(lines.len());
    for (line_idx, &line) in lines.iter().enumerate() {
        if lines_to_remove.contains(&line_idx) {
            continue;
        }
        let new_line = ident.replace_all(line, |caps: &regex::Captures| {
            let name = caps.get(1).unwrap().as_str();
            if let Some(literal) = const_map.get(name) {
                literal.clone()
            } else {
                name.to_string()
            }
        });
        result.push(new_line.into_owned());
    }

    result.join("\n")
}

/// Text-level algebraic simplification.
///
/// Folds trivial identity patterns in rendered expressions:
///   - `<expr> + 0` → `<expr>`
///   - `0 + <expr>` → `<expr>`
///   - `<expr> - 0` → `<expr>`
///   - `<expr> * 1` → `<expr>`
///   - `1 * <expr>` → `<expr>`
fn simplify_algebra(input: &str) -> String {
    let mut text = input.to_string();
    loop {
        let prev = text.clone();
        // Process line-by-line so regex context stays narrow.
        let lines: Vec<String> = text
            .lines()
            .map(|line| simplify_algebra_line(line))
            .collect();
        text = lines.join("\n");
        if text == prev {
            break;
        }
    }
    text
}

fn simplify_algebra_line(line: &str) -> String {
    let mut s = line.to_string();

    s = plus_zero_re().replace_all(&s, "$1").into_owned();
    s = zero_plus_re().replace_all(&s, "").into_owned();
    s = minus_zero_re().replace_all(&s, "$1").into_owned();
    s = times_one_re().replace_all(&s, "$1").into_owned();
    s = one_times_re().replace_all(&s, "").into_owned();

    s
}

/// Remove duplicate `if (X) return;` lines where the predicate is not
/// reassigned between the occurrences.
fn eliminate_duplicate_guards(input: &str) -> String {
    let guard = guard_re();
    let assign = assign_lhs_re();

    let lines: Vec<&str> = input.lines().collect();
    let mut result: Vec<&str> = Vec::with_capacity(lines.len());
    // Track active guard predicates that have already been checked.
    let mut active_guards: BTreeSet<String> = BTreeSet::new();

    for &line in &lines {
        let trimmed = line.trim();

        // If a variable is reassigned, invalidate it from active guards.
        if let Some(caps) = assign.captures(trimmed) {
            let lhs = caps.get(1).unwrap().as_str();
            active_guards.remove(lhs);
        }

        // Check if this is an `if (X) return;` line.
        if let Some(caps) = guard.captures(trimmed) {
            let pred = caps.get(1).unwrap().as_str().to_string();
            if active_guards.contains(&pred) {
                // Duplicate guard — skip this line.
                continue;
            }
            active_guards.insert(pred);
        }

        // Scope changes (braces) or control-flow changes should clear
        // the guard set to be conservative. But since the guard implies
        // early return, code after the first `if (X) return;` is only
        // reached when X is false, so any later `if (X) return;` with
        // the same unmodified X is dead regardless of scope depth.

        result.push(line);
    }

    result.join("\n")
}

/// Collapse addr64(lo, hi) pointer arithmetic patterns into typed pointer
/// expressions.
///
/// Recognizes the stereotyped SASS pointer construction pattern:
/// ```text
///   bN = carry_u32_add3(OFFSET, PTR.lo32, 0);
///   vLO = OFFSET + PTR.lo32;
///   vHI = lea_hi_x_sx32(OFFSET, PTR.hi32, SCALE, bN);
///   ... = *((TYPE*)addr64(vLO, vHI));
/// ```
/// and rewrites `addr64(vLO, vHI)` → `(PTR + (int64_t)OFFSET)`.
///
/// The carry/lo/hi intermediate assignments become dead and are removed by
/// the subsequent DCE fixpoint.
/// Extract base pointer name from a lo32/hi32 qualified name.
/// Handles both dot notation ("arg0_ptr.lo32" → "arg0_ptr")
/// and underscore notation ("arg4_ptr_lo32" → "arg4_ptr").
fn strip_ptr_suffix(name: &str, suffix_dot: &str, suffix_underscore: &str) -> Option<String> {
    if let Some(base) = name.strip_suffix(suffix_dot) {
        return Some(base.to_string());
    }
    if let Some(base) = name.strip_suffix(suffix_underscore) {
        return Some(base.to_string());
    }
    None
}

fn collapse_addr64_patterns(input: &str) -> String {
    let addr64 = addr64_use_re();
    let lo_add = lo_add_def_re();
    let lea_hi = lea_hi_def_re();
    let hi_add_carry = hi_add_carry_def_re();
    let carry = carry_def_re();

    // Quick check: if no addr64 in the text, skip entirely.
    if !input.contains("addr64(") {
        return input.to_string();
    }

    let lines: Vec<&str> = input.lines().collect();

    // Phase 1: Collect definitions of the form we recognize.
    //
    // lo_defs: var → (offset_var, ptr_lo32_name)
    //   e.g. "v33" → ("v32", "arg0_ptr.lo32")
    //
    // lea_hi_defs: var → (offset_var, ptr_hi32_name, carry_var)
    //   Pattern: vHI = lea_hi_x_sx32(OFFSET, PTR.hi32, SCALE, bN);
    //   e.g. "v34" → ("v32", "arg0_ptr.hi32", "b9")
    //
    // add_carry_hi_defs: var → (hi_offset_var, ptr_hi32_name, carry_var)
    //   Pattern: vHI = EXPR + PTR.hi32 + (bN ? 1 : 0);
    //   e.g. "v159" → ("v12", "arg4_ptr_hi32", "b42")
    //
    // carry_defs: var → (offset_var, ptr_lo32_name)
    //   e.g. "b9" → ("v32", "arg0_ptr.lo32")
    let mut lo_defs: HashMap<String, (String, String)> = HashMap::new();
    let mut lea_hi_defs: HashMap<String, (String, String, String)> = HashMap::new();
    let mut add_carry_hi_defs: HashMap<String, (String, String, String)> = HashMap::new();
    let mut carry_defs: HashMap<String, (String, String)> = HashMap::new();

    for &line in &lines {
        let trimmed = line.trim();
        if let Some(caps) = lo_add.captures(trimmed) {
            let dest = caps.get(1).unwrap().as_str().to_string();
            let offset = caps.get(2).unwrap().as_str().to_string();
            let ptr_lo = caps.get(3).unwrap().as_str().to_string();
            lo_defs.insert(dest, (offset, ptr_lo));
        }
        if let Some(caps) = lea_hi.captures(trimmed) {
            let dest = caps.get(1).unwrap().as_str().to_string();
            let offset = caps.get(2).unwrap().as_str().to_string();
            let ptr_hi = caps.get(3).unwrap().as_str().to_string();
            let carry_var = caps.get(4).unwrap().as_str().to_string();
            lea_hi_defs.insert(dest, (offset, ptr_hi, carry_var));
        }
        if let Some(caps) = hi_add_carry.captures(trimmed) {
            let dest = caps.get(1).unwrap().as_str().to_string();
            let hi_offset = caps.get(2).unwrap().as_str().to_string();
            let ptr_hi = caps.get(3).unwrap().as_str().to_string();
            let carry_var = caps.get(4).unwrap().as_str().to_string();
            add_carry_hi_defs.insert(dest, (hi_offset, ptr_hi, carry_var));
        }
        if let Some(caps) = carry.captures(trimmed) {
            let dest = caps.get(1).unwrap().as_str().to_string();
            let offset = caps.get(2).unwrap().as_str().to_string();
            let ptr_lo = caps.get(3).unwrap().as_str().to_string();
            carry_defs.insert(dest, (offset, ptr_lo));
        }
    }

    // Phase 2: For each addr64(vLO, vHI), check if vLO and vHI match a
    // consistent pointer pattern.
    // Build a replacement map: "addr64(vLO, vHI)" → "(PTR + (int64_t)OFFSET)"
    let mut replacements: HashMap<String, String> = HashMap::new();

    // Scan for addr64 occurrences
    for &line in &lines {
        for caps in addr64.captures_iter(line) {
            let lo_var = caps.get(1).unwrap().as_str();
            let hi_var = caps.get(2).unwrap().as_str();
            let key = format!("addr64({}, {})", lo_var, hi_var);

            if replacements.contains_key(&key) {
                continue;
            }

            // We need a lo-part definition.
            let lo_info = match lo_defs.get(lo_var) {
                Some(info) => info,
                None => continue,
            };
            let (lo_offset, ptr_lo) = lo_info;

            // Extract base pointer from lo32 name.
            let base_lo = match strip_ptr_suffix(ptr_lo, ".lo32", "_lo32") {
                Some(b) => b,
                None => continue,
            };

            // Try to match hi-part: first lea_hi, then add_carry_hi.
            let hi_match = if let Some((hi_offset, ptr_hi, carry_var)) =
                lea_hi_defs.get(hi_var)
            {
                // Pattern 1: lea_hi_x_sx32(OFFSET, PTR.hi32, SCALE, bN)
                // The offset in lea_hi must match the lo offset.
                if hi_offset == lo_offset {
                    Some((ptr_hi.clone(), carry_var.clone(), lo_offset.clone()))
                } else {
                    None
                }
            } else if let Some((hi_offset_var, ptr_hi, carry_var)) =
                add_carry_hi_defs.get(hi_var)
            {
                // Pattern 2: EXPR + PTR.hi32 + (bN ? 1 : 0)
                // The carry variable tells us the real lo-offset.
                // The hi_offset_var might be a sign extension of lo_offset
                // (e.g., v12 = (int32_t)v13 >> 31), or it might equal lo_offset.
                // We trust the carry_var to identify the true offset.
                if let Some((carry_offset, _)) = carry_defs.get(carry_var) {
                    if carry_offset == lo_offset {
                        Some((ptr_hi.clone(), carry_var.clone(), lo_offset.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let (ptr_hi, carry_var, offset) = match hi_match {
                Some(m) => m,
                None => continue,
            };

            // Extract base pointer from hi32 name and verify consistency.
            let base_hi = match strip_ptr_suffix(&ptr_hi, ".hi32", "_hi32") {
                Some(b) => b,
                None => continue,
            };

            if base_lo != base_hi {
                continue;
            }

            // Verify carry variable consistency (optional — if carry exists)
            if let Some((carry_offset, carry_ptr_lo)) = carry_defs.get(&carry_var) {
                if carry_offset != &offset || carry_ptr_lo != ptr_lo {
                    continue;
                }
            }

            // Match! Build the replacement expression.
            replacements.insert(key, format!("({} + (int64_t){})", base_lo, offset));
        }
    }

    if replacements.is_empty() {
        return input.to_string();
    }

    // Phase 3: Apply replacements to all lines.
    let mut result = Vec::with_capacity(lines.len());
    for &line in &lines {
        let mut new_line = line.to_string();
        for (from, to) in &replacements {
            if new_line.contains(from.as_str()) {
                new_line = new_line.replace(from.as_str(), to.as_str());
            }
        }
        result.push(new_line);
    }

    result.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_simple_dead_assignment() {
        let input = "  v0 = 42;\n  v1 = v0 + 1;\n  return;";
        let output = eliminate_dead_code(input);
        // v1 is dead (never used), and once v1 is removed, v0 is also dead.
        assert!(!output.contains("v1 ="), "v1 should be removed (dead)");
        assert!(!output.contains("v0 = 42"), "v0 should be removed (transitively dead)");
        assert!(output.contains("return;"));
    }

    #[test]
    fn iterates_to_fixpoint() {
        // v2 uses v1, v1 uses v0. If v2 is dead, removing it makes v1 dead,
        // which makes v0 dead.
        let input = "  v0 = 42;\n  v1 = v0 + 1;\n  v2 = v1 + 2;\n  return;";
        let output = eliminate_dead_code(input);
        assert!(!output.contains("v0"), "v0 should be removed (transitively dead)");
        assert!(!output.contains("v1"), "v1 should be removed");
        assert!(!output.contains("v2"), "v2 should be removed");
        assert!(output.contains("return;"));
    }

    #[test]
    fn preserves_memory_stores() {
        let input = "  v0 = 42;\n  *addr64(v0, v1) = 0;\n  return;";
        let output = eliminate_dead_code(input);
        // v0 = 42 is a single-def constant — it gets propagated into the store.
        assert!(output.contains("*addr64(42, v1)"), "constant should be inlined into store");
        assert!(!output.contains("v0 = 42"), "v0 assignment should be removed after propagation");
    }

    #[test]
    fn preserves_control_flow() {
        let input = "  b0 = x > 1;\n  if (b0) return;\n  v1 = 10;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("b0 = x > 1"), "b0 is used in if");
        assert!(output.contains("if (b0) return;"));
    }

    #[test]
    fn preserves_side_effect_calls() {
        let input = "  v0 = __syncthreads();\n  return;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("__syncthreads"), "side effect must be preserved");
    }

    #[test]
    fn does_not_remove_used_variables() {
        let input = "  v0 = threadIdx.x;\n  v1 = v0 * blockDim.x;\n  *ptr = v1;\n  return;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("v0 = threadIdx.x"));
        assert!(output.contains("v1 = v0 * blockDim.x"));
        assert!(output.contains("*ptr = v1"));
    }

    #[test]
    fn removes_multiple_dead_vars_in_one_pass() {
        let input = "  a = 1;\n  b = 2;\n  c = 3;\n  d = a + 1;\n  *ptr = d;";
        let output = eliminate_dead_code(input);
        assert!(!output.contains("b = 2"), "b is dead");
        assert!(!output.contains("c = 3"), "c is dead");
        // a=1 is const-propagated into d, so a=1 is removed.
        assert!(!output.contains("a = 1"), "a should be propagated into d");
        assert!(output.contains("d = 1 + 1"), "d should have a inlined");
        assert!(output.contains("*ptr = d"), "d is used in store");
    }

    #[test]
    fn removes_duplicate_return_guard() {
        let input = "  b1 = x > 1;\n  if (b1) return;\n  *ptr = 42;\n  if (b1) return;\n  return;";
        let output = eliminate_dead_code(input);
        // First guard should remain, second should be removed.
        let count = output.matches("if (b1) return;").count();
        assert_eq!(count, 1, "duplicate guard should be removed");
        assert!(output.contains("*ptr = 42"));
    }

    #[test]
    fn keeps_guard_after_reassignment() {
        let input = "  b1 = x > 1;\n  if (b1) return;\n  b1 = y > 2;\n  if (b1) return;\n  return;";
        let output = eliminate_dead_code(input);
        // b1 is reassigned between the two guards, so both should remain.
        let count = output.matches("if (b1) return;").count();
        assert_eq!(count, 2, "guard after reassignment should be kept");
    }

    #[test]
    fn cse_unifies_redundant_blockdimx() {
        let input = "  u5 = blockDim.x;\n  v15 = blockDim.x;\n  v1 = u5 + v15;\n  *ptr = v1;";
        let output = eliminate_dead_code(input);
        // v15 should be renamed to u5.
        assert!(!output.contains("v15"), "v15 should be unified with u5");
        assert!(output.contains("u5 = blockDim.x"));
        assert!(output.contains("v1 = u5 + u5"));
    }

    #[test]
    fn cse_does_not_unify_reassigned_var() {
        let input = "  x = blockDim.x;\n  x = 42;\n  y = blockDim.x;\n  *ptr = y;";
        let output = eliminate_dead_code(input);
        // x is reassigned, so y should NOT be renamed to x.
        assert!(output.contains("y = blockDim.x"), "y must remain since x is reassigned");
    }

    #[test]
    fn const_prop_inlines_small_literals() {
        let input = "  v0 = 0;\n  v1 = 1;\n  *ptr = v0;\n  *ptr2 = v1;";
        let output = eliminate_dead_code(input);
        assert!(!output.contains("v0 = 0"), "v0 should be propagated");
        assert!(!output.contains("v1 = 1"), "v1 should be propagated");
        assert!(output.contains("*ptr = 0"), "v0 inlined to 0");
        assert!(output.contains("*ptr2 = 1"), "v1 inlined to 1");
    }

    #[test]
    fn const_prop_does_not_inline_multi_def() {
        // v0 is assigned twice — should NOT be propagated.
        let input = "  v0 = 0;\n  *ptr = v0;\n  v0 = 1;\n  *ptr2 = v0;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("v0 = 0"), "v0 multi-def should not be propagated");
        assert!(output.contains("v0 = 1"), "v0 multi-def should not be propagated");
    }

    #[test]
    fn const_prop_does_not_inline_expressions() {
        // "v0 + 1" is not a literal — should not be propagated.
        let input = "  x = v0 + 1;\n  *ptr = x;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("x = v0 + 1"), "expression should not be propagated");
    }

    #[test]
    fn algebra_folds_plus_zero() {
        let input = "  v10 = mul_hi_u32(v5, v8) + 0;\n  *ptr = v10;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("mul_hi_u32(v5, v8)"), "should contain the call");
        assert!(!output.contains("+ 0"), "+ 0 should be folded away");
    }

    #[test]
    fn algebra_folds_zero_plus() {
        let input = "  v1 = 0 + v0;\n  *ptr = v1;";
        let output = eliminate_dead_code(input);
        assert!(!output.contains("0 + "), "0 + should be folded away");
    }

    #[test]
    fn algebra_folds_times_one() {
        let input = "  v1 = v0 * 1;\n  *ptr = v1;";
        let output = eliminate_dead_code(input);
        assert!(!output.contains("* 1"), "* 1 should be folded away");
    }

    #[test]
    fn algebra_does_not_fold_times_ten() {
        // "* 10" should NOT match the "* 1" rule.
        let input = "  v1 = v0 * 10;\n  *ptr = v1;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("v0 * 10"), "* 10 must not be folded");
    }

    #[test]
    fn collapses_addr64_lea_hi_pattern() {
        // Pattern 1: carry + add + lea_hi_x_sx32 + addr64 → typed pointer
        let input = "\
  b9 = carry_u32_add3(v32, arg0_ptr.lo32, 0);
  v33 = v32 + arg0_ptr.lo32;
  v34 = lea_hi_x_sx32(v32, arg0_ptr.hi32, 1, b9);
  v35 = *((uint8_t*)addr64(v33, v34));
  return v35;";
        let output = eliminate_dead_code(input);
        assert!(
            output.contains("(arg0_ptr + (int64_t)v32)"),
            "addr64 should be collapsed to typed pointer: {}", output
        );
        assert!(!output.contains("addr64("), "no addr64 should remain");
        // carry/add/lea_hi intermediates should be DCE'd
        assert!(!output.contains("lea_hi_x_sx32("), "lea_hi_x_sx32 should be DCE'd");
    }

    #[test]
    fn collapses_addr64_hi_add_carry_pattern() {
        // Pattern 2: carry + add + hi_add_carry + addr64 → typed pointer
        let input = "\
  b42 = carry_u32_add3(v13, arg4_ptr_lo32, 0);
  v158 = v13 + arg4_ptr_lo32;
  v159 = v12 + arg4_ptr_hi32 + (b42 ? 1 : 0);
  v160 = *((uint8_t*)addr64(v158, v159));
  return v160;";
        let output = eliminate_dead_code(input);
        assert!(
            output.contains("(arg4_ptr + (int64_t)v13)"),
            "addr64 should be collapsed to typed pointer: {}", output
        );
        assert!(!output.contains("addr64("), "no addr64 should remain");
    }

    #[test]
    fn collapses_addr64_with_dot_suffix() {
        // Pattern using .lo32/.hi32 suffix (dot notation)
        let input = "\
  b1 = carry_u32_add3(v5, arg2_ptr.lo32, 0);
  v6 = v5 + arg2_ptr.lo32;
  v7 = v4 + arg2_ptr.hi32 + (b1 ? 1 : 0);
  *((uint32_t*)addr64(v6, v7)) = v8;";
        let output = eliminate_dead_code(input);
        assert!(
            output.contains("(arg2_ptr + (int64_t)v5)"),
            "addr64 should be collapsed with dot suffix: {}", output
        );
    }

    #[test]
    fn addr64_collapse_preserves_unmatched_patterns() {
        // addr64 where lo/hi don't follow recognized patterns should remain
        let input = "\
  v10 = v3 * 4 + v1;
  v11 = v2 + v4;
  v12 = *addr64(v10, v11);
  return v12;";
        let output = eliminate_dead_code(input);
        assert!(
            output.contains("addr64(v10, v11)"),
            "unrecognized addr64 should be preserved: {}", output
        );
    }
}
