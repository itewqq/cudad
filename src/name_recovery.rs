//! Optional post-render name recovery for SSA-style pseudocode.
//! This stage is intentionally conservative and non-structural:
//! it rewrites variable tokens in rendered output only.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use regex::Regex;

use crate::ir::{FunctionIR, IRCond, IRExpr, RValue, RegId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameStyle {
    Temp,
    RegisterFamily,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameRecoveryConfig {
    pub style: NameStyle,
    pub rewrite_control_predicates: bool,
    pub emit_phi_merge_comments: bool,
    pub semantic_symbolization: bool,
}

impl Default for NameRecoveryConfig {
    fn default() -> Self {
        Self {
            style: NameStyle::Temp,
            rewrite_control_predicates: true,
            emit_phi_merge_comments: false,
            semantic_symbolization: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NameRecoveryStats {
    pub ssa_tokens_seen: usize,
    pub rewritten_tokens: usize,
    pub output_vars: usize,
    pub split_components: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NameRecoveryResult {
    pub output: String,
    pub stats: NameRecoveryStats,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RegBase {
    class: String,
    idx: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RegSsa {
    base: RegBase,
    ssa: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Loc {
    block_id: usize,
    stmt_idx: usize,
    order_in_stmt: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Group {
    Data = 0,
    Uniform = 1,
    Pred = 2,
}

#[derive(Debug, Default)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn make_set(&mut self) -> usize {
        let idx = self.parent.len();
        self.parent.push(idx);
        self.rank.push(0);
        idx
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.find(p);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let rank_a = self.rank[ra];
        let rank_b = self.rank[rb];
        if rank_a < rank_b {
            self.parent[ra] = rb;
        } else if rank_a > rank_b {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

pub fn recover_structured_output_names(
    function_ir: &FunctionIR,
    rendered: &str,
    config: &NameRecoveryConfig,
) -> NameRecoveryResult {
    let (tokens, first_seen, mut uf, idx_of) = collect_tokens(function_ir);

    // Build conservative congruence edges.
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            for def in &stmt.defs {
                let IRExpr::Reg(dest) = def else {
                    continue;
                };
                let Some(dest_key) = reg_ssa(dest) else {
                    continue;
                };
                let Some(dest_uf_idx) = idx_of.get(&dest_key).copied() else {
                    continue;
                };

                if let RValue::Phi(args) = &stmt.value {
                    for arg in args {
                        if let Some(arg_key) = reg_ssa_from_expr(arg) {
                            if arg_key.base == dest_key.base {
                                if let Some(arg_idx) = idx_of.get(&arg_key).copied() {
                                    uf.union(dest_uf_idx, arg_idx);
                                }
                            }
                        }
                    }
                } else if let RValue::Op { opcode, args } = &stmt.value {
                    if let Some(src_key) = conservative_copy_source(opcode, args) {
                        if src_key.base == dest_key.base {
                            if let Some(src_idx) = idx_of.get(&src_key).copied() {
                                uf.union(dest_uf_idx, src_idx);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut components: BTreeMap<usize, Vec<RegSsa>> = BTreeMap::new();
    for tok in &tokens {
        let Some(idx) = idx_of.get(tok).copied() else {
            continue;
        };
        let root = uf.find(idx);
        components.entry(root).or_default().push(tok.clone());
    }

    let mut comp_rows = Vec::new();
    for members in components.values() {
        let mut min_loc = Loc {
            block_id: usize::MAX,
            stmt_idx: usize::MAX,
            order_in_stmt: u8::MAX,
        };
        let mut min_tok = String::new();
        for m in members {
            if let Some(loc) = first_seen.get(m).copied() {
                min_loc = min_loc.min(loc);
            }
            let t = reg_token_core(m);
            if min_tok.is_empty() || t < min_tok {
                min_tok = t;
            }
        }
        let base = members[0].base.clone();
        comp_rows.push((base_group(&base.class), min_loc, min_tok, base, members.clone()));
    }
    comp_rows.sort_by(|a, b| {
        (a.0, a.1, &a.2, &a.3).cmp(&(b.0, b.1, &b.2, &b.3))
    });

    let mut fam_count: BTreeMap<RegBase, usize> = BTreeMap::new();
    for row in &comp_rows {
        *fam_count.entry(row.3.clone()).or_insert(0) += 1;
    }

    let mut fam_seen: BTreeMap<RegBase, usize> = BTreeMap::new();
    let mut v_ctr = 0usize;
    let mut u_ctr = 0usize;
    let mut b_ctr = 0usize;

    let mut component_name: Vec<String> = Vec::new();
    for row in &comp_rows {
        let grp = row.0;
        let base = &row.3;
        let name = match config.style {
            NameStyle::Temp => match grp {
                Group::Data => {
                    let s = format!("v{}", v_ctr);
                    v_ctr += 1;
                    s
                }
                Group::Uniform => {
                    let s = format!("u{}", u_ctr);
                    u_ctr += 1;
                    s
                }
                Group::Pred => {
                    let s = format!("b{}", b_ctr);
                    b_ctr += 1;
                    s
                }
            },
            NameStyle::RegisterFamily => {
                let base_name = format!("{}{}", base.class.to_ascii_lowercase(), base.idx);
                let total = *fam_count.get(base).unwrap_or(&1usize);
                if total > 1 {
                    let seen = fam_seen.entry(base.clone()).or_insert(0);
                    let s = format!("{}_{}", base_name, *seen);
                    *seen += 1;
                    s
                } else {
                    base_name
                }
            }
        };
        component_name.push(name);
    }

    let mut token_map = HashMap::<String, String>::new();
    let mut ambiguous = BTreeSet::<String>::new();
    for (row_idx, row) in comp_rows.iter().enumerate() {
        for m in &row.4 {
            let token = reg_token_core(m);
            let out = &component_name[row_idx];
            if let Some(prev) = token_map.get(&token) {
                if prev != out {
                    ambiguous.insert(token.clone());
                }
            } else {
                token_map.insert(token, out.clone());
            }
        }
    }
    for bad in ambiguous {
        token_map.remove(&bad);
    }

    // For predicated instructions, the dest and its pred_old_def frequently
    // denote the same logical variable (dest keeps old value when predicate is
    // false). Prefer mapping old-value tokens to the same recovered name as the
    // dest, but never overwrite an existing conflicting mapping: preserving a
    // stable one-to-one token map is correctness-critical.
    //
    // We do this as a post-map fixup rather than a Union-Find edge because
    // unioning them would alter the global component count and shift the
    // sequential name counters for every subsequent component.
    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            if stmt.pred.is_none() || stmt.pred_old_defs.is_empty() {
                continue;
            }
            for (def_idx, def) in stmt.defs.iter().enumerate() {
                let Some(dest_key) = reg_ssa_from_expr(def) else {
                    continue;
                };
                let dest_tok = reg_token_core(&dest_key);
                let Some(dest_name) = token_map.get(&dest_tok).cloned() else {
                    continue;
                };
                if let Some(old_expr) = stmt.pred_old_defs.get(def_idx) {
                    if let Some(old_key) = reg_ssa_from_expr(old_expr) {
                        let old_tok = reg_token_core(&old_key);
                        merge_pred_old_token_mapping(&mut token_map, old_tok, &dest_name);
                    }
                }
            }
        }
    }

    let re = Regex::new(r"\b(?:UR|UP|R|P)\d+\.\d+\b").expect("valid regex");
    let mut ssa_tokens_seen = 0usize;
    let mut rewritten_tokens = 0usize;
    let mut output = re
        .replace_all(rendered, |caps: &regex::Captures<'_>| {
            ssa_tokens_seen += 1;
            let t = caps.get(0).expect("match").as_str();
            if let Some(rep) = token_map.get(t) {
                rewritten_tokens += 1;
                rep.clone()
            } else {
                t.to_string()
            }
        })
        .into_owned();

    // Post-pass: convert self-referencing ternaries to if-guards.
    // After name recovery, patterns like `v17 = b3 ? (v17 + 1) : v17;` have
    // identical dest and old-value names.  These are clearer as
    // `if (b3) v17 = v17 + 1;`.  Genuine selects (dest != old) are left alone.
    output = simplify_predicated_ternaries(&output);

    if config.rewrite_control_predicates {
        let mut pred_map = HashMap::<String, String>::new();
        for (row_idx, row) in comp_rows.iter().enumerate() {
            let base = &row.3;
            if !matches!(base.class.as_str(), "P" | "UP") {
                continue;
            }
            if fam_count.get(base).copied().unwrap_or(0) != 1 {
                continue;
            }
            pred_map.insert(
                format!("{}{}", base.class, base.idx),
                component_name[row_idx].clone(),
            );
        }
        output = rewrite_control_guard_predicates(&output, &pred_map);
    }

    if config.semantic_symbolization {
        output = rewrite_semantic_seed_names(&output);
    }

    if config.emit_phi_merge_comments {
        output = append_phi_merge_comments(function_ir, &output, &token_map);
    }

    let split_components = fam_count
        .values()
        .copied()
        .filter(|n| *n > 1)
        .map(|n| n - 1)
        .sum::<usize>();

    NameRecoveryResult {
        output,
        stats: NameRecoveryStats {
            ssa_tokens_seen,
            rewritten_tokens,
            output_vars: component_name.len(),
            split_components,
        },
    }
}

fn rewrite_semantic_seed_names(output: &str) -> String {
    let assign_re = Regex::new(
        r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<rhs>[A-Za-z_][A-Za-z0-9_.]*)\s*;\s*$",
    )
    .expect("valid regex");
    let ident_re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").expect("valid regex");
    let mut used = BTreeSet::<String>::new();
    for m in ident_re.find_iter(output) {
        used.insert(m.as_str().to_string());
    }

    let mut rename = HashMap::<String, String>::new();
    for line in output.lines() {
        let Some(cap) = assign_re.captures(line) else {
            continue;
        };
        let lhs = cap.name("lhs").expect("lhs").as_str().to_string();
        let rhs = cap.name("rhs").expect("rhs").as_str();
        let Some(seed) = semantic_name_seed(rhs) else {
            continue;
        };
        if rename.contains_key(&lhs) {
            continue;
        }
        let unique = alloc_unique_name(seed, &mut used);
        rename.insert(lhs, unique);
    }
    if rename.is_empty() {
        return output.to_string();
    }

    let mut out = output.to_string();
    for (from, to) in rename {
        let pat = Regex::new(&format!(r"\b{}\b", regex::escape(&from))).expect("valid regex");
        out = pat.replace_all(&out, to.as_str()).into_owned();
    }
    out
}

fn alloc_unique_name(seed: String, used: &mut BTreeSet<String>) -> String {
    if used.insert(seed.clone()) {
        return seed;
    }
    let mut idx = 1usize;
    loop {
        let candidate = format!("{}_{}", seed, idx);
        if used.insert(candidate.clone()) {
            return candidate;
        }
        idx += 1;
    }
}

fn semantic_name_seed(rhs: &str) -> Option<String> {
    let fixed = match rhs {
        "threadIdx.x" => Some("tid_x"),
        "threadIdx.y" => Some("tid_y"),
        "threadIdx.z" => Some("tid_z"),
        "blockIdx.x" => Some("ctaid_x"),
        "blockIdx.y" => Some("ctaid_y"),
        "blockIdx.z" => Some("ctaid_z"),
        "blockDim.x" => Some("block_dim_x"),
        "blockDim.y" => Some("block_dim_y"),
        "blockDim.z" => Some("block_dim_z"),
        "gridDim.x" => Some("grid_dim_x"),
        "gridDim.y" => Some("grid_dim_y"),
        "gridDim.z" => Some("grid_dim_z"),
        _ => None,
    };
    if let Some(name) = fixed {
        return Some(name.to_string());
    }
    let arg_ptr_lane_re = Regex::new(r"^arg(?P<idx>\d+)_ptr\.(?P<lane>lo32|hi32)$").expect("valid regex");
    if let Some(cap) = arg_ptr_lane_re.captures(rhs) {
        return Some(format!(
            "arg{}_ptr_{}",
            cap.name("idx").expect("idx").as_str(),
            cap.name("lane").expect("lane").as_str()
        ));
    }
    None
}

/// Simplify predicated-instruction ternary selects in the recovered output.
///
/// Two classes of ternary are converted to cleaner if-guarded assignments:
///
/// 1. **Self-referencing**: `v17 = b3 ? (v17 + 1) : v17;` → `if (b3) v17 = v17 + 1;`
///    The dest and old-value are the same variable after name recovery; the
///    else branch just keeps the current value — an if-guard is clearer.
///
/// 2. **Phi-marked old value**:
///    `v165 = !b41 ? (load(...)) : /*phi*/v17;` → `if (!b41) v165 = load(...);`
///    A `/*phi*/` old arm denotes loop-carried old state. Emitting it as an
///    if-guard avoids fabricating a meaningful else-value in pseudocode dataflow.
///
/// Genuine conditional selects (different dest/old) are left
/// as ternaries.
fn simplify_predicated_ternaries(output: &str) -> String {
    let out = simplify_predicated_ternaries_primary(output);
    let out = simplify_predicated_only_ternaries_with_gated_uses(&out);
    simplify_predicated_ternaries_primary(&out)
}

fn simplify_predicated_ternaries_primary(output: &str) -> String {
    // Match ternaries, optionally with a /*phi*/ tag before the old-value.
    //
    // Capture groups:
    //  1 – leading whitespace (indent)
    //  2 – dest variable name
    //  3 – predicate expression
    //  4 – RHS expression (inside the outermost parens after `?`)
    //  5 – optional `/*phi*/` marker (ignored for rewrite legality)
    //  6 – old-value variable name
    let re = Regex::new(
        r"^(\s*)(\S+) = (.+?) \? \((.*)\) : (/\*phi\*/)?(\S+);$"
    ).expect("valid regex");
    let if_assign_re = Regex::new(
        r"^\s*if \((.+)\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=.*;$",
    )
    .expect("valid regex");
    let plain_assign_re = Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*;$").expect("valid regex");
    let ident_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b").expect("valid regex");

    // Track variables that are currently known to be conditionally assigned on
    // a specific predicate guard (`if (pred) v = ...;`).
    //
    // This allows a conservative rewrite of suspicious forms like
    // `v177 = !b41 ? (shmem_u8[v176]) : v176;` where `v176` was only assigned
    // under `if (!b41)` right above. Emitting an if-guard avoids fabricating
    // else-path dataflow from a value that does not exist on that path.
    let mut conditional_assign_pred = HashMap::<String, String>::new();

    let mut out = String::with_capacity(output.len());
    for line in output.lines() {
        if let Some(caps) = re.captures(line) {
            let indent = &caps[1];
            let dest = &caps[2];
            let pred = &caps[3];
            let rhs = &caps[4];
            let has_phi_marker = caps.get(5).is_some();
            let old = &caps[6];
            let pred_norm = normalize_predicate_text(pred);

            let old_is_cond_defined_on_same_pred = conditional_assign_pred
                .get(old)
                .map_or(false, |p| p == &pred_norm);
            let old_is_cond_defined_any = conditional_assign_pred.contains_key(old);
            let rhs_uses_cond_defined_same_pred = ident_re.find_iter(rhs).any(|m| {
                conditional_assign_pred
                    .get(m.as_str())
                    .map_or(false, |p| p == &pred_norm)
            });
            let rhs_uses_cond_defined_any = ident_re
                .find_iter(rhs)
                .any(|m| conditional_assign_pred.contains_key(m.as_str()));

            if dest == old
                || has_phi_marker
                || old_is_cond_defined_on_same_pred
                || rhs_uses_cond_defined_same_pred
                || old_is_cond_defined_any
                || rhs_uses_cond_defined_any
            {
                // Self-referencing or phi-marked ternary → if-guard.
                out.push_str(&format!(
                    "{}if ({}) {} = {};\n",
                    indent, pred, dest, rhs
                ));
                conditional_assign_pred.insert(dest.to_string(), pred_norm);
                continue;
            }

            // Keep genuine select; this defines dest unconditionally.
            conditional_assign_pred.remove(dest);
        }

        if let Some(caps) = if_assign_re.captures(line) {
            let pred_norm = normalize_predicate_text(caps.get(1).expect("pred").as_str());
            let dest = caps.get(2).expect("dest").as_str().to_string();
            conditional_assign_pred.insert(dest, pred_norm);
        } else if let Some(caps) = plain_assign_re.captures(line) {
            let dest = caps.get(1).expect("dest").as_str();
            conditional_assign_pred.remove(dest);
        }

        out.push_str(line);
        out.push('\n');
    }
    // Remove trailing newline if the original didn't end with one.
    if !output.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

fn simplify_predicated_only_ternaries_with_gated_uses(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.is_empty() {
        return output.to_string();
    }

    let ternary_re = Regex::new(r"^(\s*)(\S+) = (.+?) \? \((.*)\) : (/\*phi\*/)?(\S+);$")
        .expect("valid regex");
    let if_assign_re = Regex::new(
        r"^\s*if \((.+)\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=.*;$",
    )
    .expect("valid regex");
    let plain_assign_re = Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*;$").expect("valid regex");
    let ident_re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b").expect("valid regex");

    let mut rewrite_line = vec![false; lines.len()];

    for i in 0..lines.len() {
        let Some(caps) = ternary_re.captures(lines[i]) else {
            continue;
        };
        let dest = caps.get(2).expect("dest").as_str();
        let pred_norm = normalize_predicate_text(caps.get(3).expect("pred").as_str());

        let mut saw_use = false;
        let mut only_gated_uses = true;

        for line in lines.iter().skip(i + 1) {
            if let Some(c) = if_assign_re.captures(line) {
                if c.get(2).expect("dest").as_str() == dest {
                    break;
                }
            } else if let Some(c) = plain_assign_re.captures(line) {
                if c.get(1).expect("dest").as_str() == dest {
                    break;
                }
            }

            let uses_dest = ident_re.find_iter(line).any(|m| m.as_str() == dest);
            if !uses_dest {
                continue;
            }
            saw_use = true;

            let gated_by_same_pred_if = if_assign_re.captures(line).map_or(false, |c| {
                normalize_predicate_text(c.get(1).expect("pred").as_str()) == pred_norm
            });
            let gated_by_same_pred_ternary = ternary_re.captures(line).map_or(false, |c| {
                normalize_predicate_text(c.get(3).expect("pred").as_str()) == pred_norm
            });

            if !(gated_by_same_pred_if || gated_by_same_pred_ternary) {
                only_gated_uses = false;
                break;
            }
        }

        if saw_use && only_gated_uses {
            rewrite_line[i] = true;
        }
    }

    let mut out = String::with_capacity(output.len());
    for (i, line) in lines.iter().enumerate() {
        if rewrite_line[i] {
            if let Some(caps) = ternary_re.captures(line) {
                let indent = caps.get(1).expect("indent").as_str();
                let dest = caps.get(2).expect("dest").as_str();
                let pred = caps.get(3).expect("pred").as_str();
                let rhs = caps.get(4).expect("rhs").as_str();
                out.push_str(&format!("{}if ({}) {} = {};\n", indent, pred, dest, rhs));
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }

    if !output.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

fn normalize_predicate_text(pred: &str) -> String {
    pred.split_whitespace().collect::<String>()
}

fn merge_pred_old_token_mapping(
    token_map: &mut HashMap<String, String>,
    old_tok: String,
    dest_name: &str,
) {
    match token_map.get(&old_tok) {
        None => {
            token_map.insert(old_tok, dest_name.to_string());
        }
        Some(existing) if existing == dest_name => {
            // Already mapped consistently.
        }
        Some(_) => {
            // Keep existing mapping. Overwriting would conflate different
            // recovered names and can create unsound dataflow in output.
        }
    }
}

fn rewrite_control_guard_predicates(output: &str, pred_map: &HashMap<String, String>) -> String {
    let pred_re = Regex::new(r"\b(?:P|UP)\d+\b").expect("valid regex");
    let mut out = String::new();
    for line in output.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("if (") || trimmed.starts_with("while (") {
            let replaced = pred_re.replace_all(line, |caps: &regex::Captures<'_>| {
                let t = caps.get(0).expect("match").as_str();
                pred_map.get(t).cloned().unwrap_or_else(|| t.to_string())
            });
            out.push_str(&replaced);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn append_phi_merge_comments(
    function_ir: &FunctionIR,
    output: &str,
    token_map: &HashMap<String, String>,
) -> String {
    let mut by_block = BTreeMap::<usize, Vec<String>>::new();
    let mut live_ins = BTreeSet::<String>::new();

    for block in &function_ir.blocks {
        let mut merges = Vec::new();
        for stmt in &block.stmts {
            let (Some(IRExpr::Reg(dst)), RValue::Phi(args)) = (stmt.defs.first(), &stmt.value) else {
                continue;
            };
            let Some(dst_ssa) = reg_ssa(dst) else {
                continue;
            };
            let dst_tok = reg_token_core(&dst_ssa);
            let dst_name = token_map.get(&dst_tok).cloned().unwrap_or(dst_tok);
            let mut arg_names = Vec::new();
            for a in args {
                if let Some(a_ssa) = reg_ssa_from_expr(a) {
                    let a_tok = reg_token_core(&a_ssa);
                    let a_name = token_map.get(&a_tok).cloned().unwrap_or(a_tok);
                    if a_name != dst_name {
                        live_ins.insert(a_name.clone());
                    }
                    arg_names.push(a_name);
                } else {
                    arg_names.push(render_phi_fallback(a));
                }
            }
            merges.push(format!("phi merge: {} <- phi({})", dst_name, arg_names.join(", ")));
        }
        if !merges.is_empty() {
            by_block.insert(block.id, merges);
        }
    }

    if by_block.is_empty() {
        return output.to_string();
    }

    let bb_re = Regex::new(r"^\s*BB(\d+)\s*\{").expect("valid regex");
    let mut current_block: Option<usize> = None;
    let mut rendered = String::new();

    if !live_ins.is_empty() {
        rendered.push_str("// live-ins: ");
        rendered.push_str(&live_ins.into_iter().collect::<Vec<_>>().join(", "));
        rendered.push('\n');
    }

    for line in output.lines() {
        if let Some(cap) = bb_re.captures(line) {
            if let Some(m) = cap.get(1) {
                current_block = m.as_str().parse::<usize>().ok();
            }
        }
        rendered.push_str(line);
        rendered.push('\n');

        if line.contains("phi node(s) omitted") {
            if let Some(block_id) = current_block {
                if let Some(merges) = by_block.get(&block_id) {
                    let indent = line
                        .chars()
                        .take_while(|c| c.is_ascii_whitespace())
                        .collect::<String>();
                    for m in merges {
                        rendered.push_str(&indent);
                        rendered.push_str("// ");
                        rendered.push_str(m);
                        rendered.push('\n');
                    }
                }
            }
        }
    }
    rendered
}

fn render_phi_fallback(e: &IRExpr) -> String {
    match e {
        IRExpr::Reg(r) => r.display(),
        IRExpr::ImmI(i) => i.to_string(),
        IRExpr::ImmF(f) => f.to_string(),
        IRExpr::Mem { .. } => "<mem>".to_string(),
        IRExpr::Op { op, .. } => op.clone(),
    }
}

fn collect_tokens(
    function_ir: &FunctionIR,
) -> (
    BTreeSet<RegSsa>,
    HashMap<RegSsa, Loc>,
    UnionFind,
    HashMap<RegSsa, usize>,
) {
    let mut tokens = BTreeSet::<RegSsa>::new();
    let mut first_seen = HashMap::<RegSsa, Loc>::new();

    for block in &function_ir.blocks {
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            for def in &stmt.defs {
                collect_expr_tokens(
                    def,
                    block.id,
                    stmt_idx,
                    0,
                    &mut tokens,
                    &mut first_seen,
                );
            }
            if let Some(pred) = &stmt.pred {
                collect_expr_tokens(
                    pred,
                    block.id,
                    stmt_idx,
                    1,
                    &mut tokens,
                    &mut first_seen,
                );
            }
            match &stmt.value {
                RValue::Op { args, .. } => {
                    for arg in args {
                        collect_expr_tokens(
                            arg,
                            block.id,
                            stmt_idx,
                            2,
                            &mut tokens,
                            &mut first_seen,
                        );
                    }
                }
                RValue::Phi(args) => {
                    for arg in args {
                        collect_expr_tokens(
                            arg,
                            block.id,
                            stmt_idx,
                            2,
                            &mut tokens,
                            &mut first_seen,
                        );
                    }
                }
                RValue::ImmI(_) | RValue::ImmF(_) => {}
            }
            if let Some(mem) = &stmt.mem_addr_args {
                for arg in mem {
                    collect_expr_tokens(
                        arg,
                        block.id,
                        stmt_idx,
                        3,
                        &mut tokens,
                        &mut first_seen,
                    );
                }
            }
        }
        for (cond, _) in &block.irdst {
            if let Some(IRCond::Pred { reg, .. }) = cond {
                if let Some(key) = reg_ssa(reg) {
                    let loc = Loc {
                        block_id: block.id,
                        stmt_idx: usize::MAX - 1,
                        order_in_stmt: 4,
                    };
                    tokens.insert(key.clone());
                    first_seen
                        .entry(key)
                        .and_modify(|v| *v = (*v).min(loc))
                        .or_insert(loc);
                }
            }
        }
    }

    let mut uf = UnionFind::default();
    let mut idx_of = HashMap::new();
    for t in &tokens {
        let idx = uf.make_set();
        idx_of.insert(t.clone(), idx);
    }
    (tokens, first_seen, uf, idx_of)
}

fn collect_expr_tokens(
    e: &IRExpr,
    block_id: usize,
    stmt_idx: usize,
    order_in_stmt: u8,
    tokens: &mut BTreeSet<RegSsa>,
    first_seen: &mut HashMap<RegSsa, Loc>,
) {
    match e {
        IRExpr::Reg(r) => {
            if let Some(key) = reg_ssa(r) {
                let loc = Loc {
                    block_id,
                    stmt_idx,
                    order_in_stmt,
                };
                tokens.insert(key.clone());
                first_seen
                    .entry(key)
                    .and_modify(|v| *v = (*v).min(loc))
                    .or_insert(loc);
            }
        }
        IRExpr::Mem { base, offset, .. } => {
            collect_expr_tokens(base, block_id, stmt_idx, order_in_stmt, tokens, first_seen);
            if let Some(off) = offset {
                collect_expr_tokens(off, block_id, stmt_idx, order_in_stmt, tokens, first_seen);
            }
        }
        IRExpr::Op { args, .. } => {
            for a in args {
                collect_expr_tokens(a, block_id, stmt_idx, order_in_stmt, tokens, first_seen);
            }
        }
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
    }
}

fn conservative_copy_source(opcode: &str, args: &[IRExpr]) -> Option<RegSsa> {
    if (opcode.starts_with("MOV") || opcode.starts_with("UMOV")) && args.len() == 1 {
        return reg_ssa_from_expr(&args[0]);
    }
    if opcode.starts_with("IMAD.MOV")
        && args.len() == 3
        && is_zero_expr(&args[0])
        && is_zero_expr(&args[1])
    {
        return reg_ssa_from_expr(&args[2]);
    }
    None
}

fn reg_ssa_from_expr(e: &IRExpr) -> Option<RegSsa> {
    let IRExpr::Reg(r) = e else {
        return None;
    };
    reg_ssa(r)
}

fn reg_ssa(r: &RegId) -> Option<RegSsa> {
    if is_immutable_reg(r) {
        return None;
    }
    let ssa = r.ssa?;
    Some(RegSsa {
        base: RegBase {
            class: r.class.clone(),
            idx: r.idx,
        },
        ssa,
    })
}

fn reg_token_core(r: &RegSsa) -> String {
    format!("{}{}.{}", r.base.class, r.base.idx, r.ssa)
}

fn base_group(class: &str) -> Group {
    match class {
        "UR" => Group::Uniform,
        "P" | "UP" => Group::Pred,
        _ => Group::Data,
    }
}

fn is_immutable_reg(r: &RegId) -> bool {
    matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
}

fn is_zero_expr(e: &IRExpr) -> bool {
    match e {
        IRExpr::ImmI(i) => *i == 0,
        IRExpr::ImmF(f) => *f == 0.0,
        IRExpr::Reg(r) => matches!(r.class.as_str(), "RZ" | "URZ"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cfg, build_ssa, parse_sass};

    fn build_fir(sass: &str) -> FunctionIR {
        build_ssa(&build_cfg(parse_sass(sass)))
    }

    #[test]
    fn phi_connected_versions_merge() {
        let sass = r#"
            /*0000*/ IADD R0, R0, 0x1 ;
            /*0010*/ ISETP.LT.AND P0, PT, R0, 0x5, PT ;
            /*0020*/ @P0 BRA 0x000 ;
            /*0030*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let (tokens, _, _, _) = collect_tokens(&fir);
        let mut r0 = tokens
            .iter()
            .filter(|t| t.base.class == "R" && t.base.idx == 0)
            .map(reg_token_core)
            .collect::<Vec<_>>();
        r0.sort();
        r0.dedup();
        assert!(r0.len() >= 2);
        let rendered = format!("{} = {} + 1;", r0[0], r0[1]);
        let out = recover_structured_output_names(&fir, &rendered, &NameRecoveryConfig::default());
        for t in &r0 {
            assert!(!out.output.contains(t));
        }
    }

    #[test]
    fn non_congruent_versions_split() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, 0x1 ;
            /*0010*/ IMAD.MOV.U32 R1, RZ, RZ, 0x2 ;
            /*0020*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let rendered = "R1.0 = 1; R1.1 = 2; R2.0 = R1.0 + R1.1;";
        let out = recover_structured_output_names(&fir, rendered, &NameRecoveryConfig::default());
        // Distinct components should use different recovered names.
        assert!(out.output.contains("v0 = 1;"));
        assert!(out.output.contains("v1 = 2;"));
    }

    #[test]
    fn class_prefixes_temp_style() {
        let sass = r#"
            /*0000*/ IADD3 R1, R1, 0x1, RZ ;
            /*0010*/ UIADD3 UR4, UR4, 0x1, URZ ;
            /*0020*/ ISETP.GE.AND P0, PT, R1, 0x1, PT ;
            /*0030*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let rendered = "R1.0 = R1.1 + 1; UR4.0 = UR4.1 + 1; P0.0 = R1.0 >= 1;";
        let out = recover_structured_output_names(&fir, rendered, &NameRecoveryConfig::default());
        assert!(out.output.contains("v"));
        assert!(out.output.contains("u"));
        assert!(out.output.contains("b"));
        assert!(!out.output.contains("R1."));
        assert!(!out.output.contains("UR4."));
        assert!(!out.output.contains("P0."));
    }

    #[test]
    fn token_rewrite_is_safe() {
        let sass = r#"
            /*0000*/ IADD3 R1, R1, 0x1, RZ ;
            /*0010*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let rendered = "BB10 {\n  R1.0 = IADD3(R1.1, 1, RZ);\n}";
        let out = recover_structured_output_names(&fir, rendered, &NameRecoveryConfig::default());
        assert!(out.output.contains("BB10"));
        assert!(out.output.contains("IADD3("));
    }

    #[test]
    fn deterministic_mapping() {
        let sass = include_str!("../test_cu/if_loop.sass");
        let fir = build_fir(sass);
        let rendered = "R1.0 = R1.1 + R2.0; P0.1 = R1.0 >= 0;";
        let o1 = recover_structured_output_names(&fir, rendered, &NameRecoveryConfig::default());
        let o2 = recover_structured_output_names(&fir, rendered, &NameRecoveryConfig::default());
        assert_eq!(o1.output, o2.output);
        assert_eq!(o1.stats, o2.stats);
    }

    #[test]
    fn rewrites_control_predicate_when_unambiguous() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let out = recover_structured_output_names(
            &fir,
            "if (P0) {\n}\n",
            &NameRecoveryConfig::default(),
        );
        assert!(out.output.contains("if (b"));
    }

    #[test]
    fn keeps_control_predicate_raw_when_ambiguous() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ ISETP.LT.AND P0, PT, R1, 0x2, PT ;
            /*0020*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let out = recover_structured_output_names(
            &fir,
            "if (P0) {\n}\n",
            &NameRecoveryConfig::default(),
        );
        assert!(out.output.contains("if (P0)"));
    }

    #[test]
    fn emits_phi_merge_comments_when_enabled() {
        let sass = r#"
            /*000*/ IADD R0, R0, 0x1 ;
            /*010*/ ISETP.LT.AND P0, PT, R0, 0x5, PT ;
            /*020*/ @P0 BRA 0x000 ;
            /*030*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let phi_block = fir
            .blocks
            .iter()
            .find(|b| b.stmts.iter().any(|s| matches!(s.value, RValue::Phi(_))))
            .map(|b| b.id)
            .expect("phi block");
        let rendered = format!("BB{} {{\n  // 1 phi node(s) omitted\n}}\n", phi_block);
        let out = recover_structured_output_names(
            &fir,
            &rendered,
            &NameRecoveryConfig {
                style: NameStyle::Temp,
                rewrite_control_predicates: true,
                emit_phi_merge_comments: true,
                semantic_symbolization: false,
            },
        );
        assert!(out.output.contains("phi merge:"));
        // live-ins summary is emitted only when cross-name phi inputs exist.
        // Keep phi-merge comments as the strict opt-in contract here.
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_only_self_reference() {
        let input = "  v1 = !b0 ? (v1 + 1) : v1;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(out, "  if (!b0) v1 = v1 + 1;\n");
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_phi_marked_non_self() {
        let input = "  v177 = !b41 ? (v175 & 255) : /*phi*/v176;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(out, "  if (!b41) v177 = v175 & 255;\n");
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_when_old_only_exists_under_same_predicate() {
        let input =
            "  if (!b41) v176 = v175 & 255;\n  v177 = !b41 ? (shmem_u8[v176]) : v176;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(
            out,
            "  if (!b41) v176 = v175 & 255;\n  if (!b41) v177 = shmem_u8[v176];\n"
        );
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_when_rhs_depends_on_conditional_value() {
        let input =
            "  if (!b47) v217 = *((uint8_t*)addr64(v215, v216));\n  v8 = !b47 ? (v217 ^ v6) : v217;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(
            out,
            "  if (!b47) v217 = *((uint8_t*)addr64(v215, v216));\n  if (!b47) v8 = v217 ^ v6;\n"
        );
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_when_old_is_any_conditional_value() {
        let input =
            "  if (!b43) v179 = *((uint8_t*)(addr64(v163, v164) + 1));\n  v193 = !b4 ? (*((uint8_t*)(addr64(v163, v164) + 2))) : v179;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(
            out,
            "  if (!b43) v179 = *((uint8_t*)(addr64(v163, v164) + 1));\n  if (!b4) v193 = *((uint8_t*)(addr64(v163, v164) + 2));\n"
        );
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_predicated_only_chain() {
        let input = "  v201 = !b4 ? (v200 & 255) : v200;\n  v202 = !b4 ? (shmem_u8[v201]) : v201;\n  if (!b4) v16 = v193 ^ v202;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(
            out,
            "  if (!b4) v201 = v200 & 255;\n  if (!b4) v202 = shmem_u8[v201];\n  if (!b4) v16 = v193 ^ v202;\n"
        );
    }

    #[test]
    fn simplify_predicated_ternary_rewrites_bb13_v225_else_use_after_chain_rewrite() {
        let input =
            "  v225 = !b47 ? (v224 & 255) : v224;\n  if (!b47) v6 = shmem_u8[v225];\n  v3 = !b47 ? (v213) : v225;\n";
        let out = simplify_predicated_ternaries(input);
        assert_eq!(
            out,
            "  if (!b47) v225 = v224 & 255;\n  if (!b47) v6 = shmem_u8[v225];\n  if (!b47) v3 = v213;\n"
        );
    }

    #[test]
    fn merge_pred_old_token_mapping_does_not_overwrite_conflict() {
        let mut token_map = HashMap::new();
        token_map.insert("R2.7".to_string(), "v9".to_string());

        merge_pred_old_token_mapping(&mut token_map, "R2.7".to_string(), "v3");

        assert_eq!(token_map.get("R2.7").map(String::as_str), Some("v9"));
    }

    #[test]
    fn semantic_symbolization_renames_seeded_dimensions() {
        let sass = r#"
            /*0000*/ S2R R2, SR_TID.X ;
            /*0010*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let rendered = "v0 = threadIdx.x;\nif (v0 > 1) {\n}\n";
        let out = recover_structured_output_names(
            &fir,
            rendered,
            &NameRecoveryConfig {
                style: NameStyle::Temp,
                rewrite_control_predicates: true,
                emit_phi_merge_comments: false,
                semantic_symbolization: true,
            },
        );
        assert!(out.output.contains("tid_x = threadIdx.x;"));
        assert!(out.output.contains("if (tid_x > 1)"));
    }
}
