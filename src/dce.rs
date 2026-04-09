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

/// Run text-level dead-code elimination on `input`.
///
/// Returns the cleaned output with dead assignments removed.
pub fn eliminate_dead_code(input: &str) -> String {
    let mut text = input.to_string();
    // Iterate to fixpoint — removing one line may make another variable dead.
    loop {
        let next = dce_one_pass(&text);
        if next == text {
            break;
        }
        text = next;
    }
    text
}

/// One round of DCE.  Returns the text with dead assignments removed.
fn dce_one_pass(input: &str) -> String {
    // Regex to detect simple assignment lines:
    //   <indent> <ident> = <expr>;
    // We must NOT remove:
    //   - Lines with side effects: memory stores (*addr = ...), function calls
    //     that have effects (__syncthreads, ATOMS, STS, etc.)
    //   - Conditional assignments: if (...) <ident> = ...
    //   - Control flow: return, if, do, while, break, continue
    //   - Lines that are purely comments
    //   - Memory dereference LHS: *addr64(...) = ...
    let assign_re =
        Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);$").expect("valid regex");

    // Side-effect function patterns — never remove these even if LHS looks dead.
    let side_effect_re = Regex::new(
        r"(?:__syncthreads|ATOMS|STS|STG|STS\.U|STG\.E|RED|ATOM|BAR|MEMBAR|DEPBAR|FENCE)",
    )
    .expect("valid regex");

    let lines: Vec<&str> = input.lines().collect();

    // First pass: collect all identifiers and count their non-LHS occurrences.
    // An identifier is "used" if it appears anywhere other than as the LHS of
    // its own assignment.
    let ident_re = Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*)\b").expect("valid regex");

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

        if let Some(caps) = assign_re.captures(trimmed) {
            let lhs_name = caps.get(1).unwrap().as_str().to_string();
            let rhs = caps.get(2).unwrap().as_str();

            // Don't remove if RHS has side effects
            let has_side_effect = side_effect_re.is_match(rhs)
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
        for m in ident_re.find_iter(info.text) {
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
        assert!(output.contains("v0 = 42"), "v0 used in store");
        assert!(output.contains("*addr64"));
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
        let input = "  v0 = threadIdx.x;\n  v1 = v0 * blockDimX;\n  *ptr = v1;\n  return;";
        let output = eliminate_dead_code(input);
        assert!(output.contains("v0 = threadIdx.x"));
        assert!(output.contains("v1 = v0 * blockDimX"));
        assert!(output.contains("*ptr = v1"));
    }

    #[test]
    fn removes_multiple_dead_vars_in_one_pass() {
        let input = "  a = 1;\n  b = 2;\n  c = 3;\n  d = a + 1;\n  *ptr = d;";
        let output = eliminate_dead_code(input);
        assert!(!output.contains("b = 2"), "b is dead");
        assert!(!output.contains("c = 3"), "c is dead");
        assert!(output.contains("a = 1"), "a is used by d");
        assert!(output.contains("d = a + 1"), "d is used in store");
    }
}
