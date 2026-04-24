//! Structural name-recovery planning for the canonical backend.
//! This stage computes SSA-to-symbol mappings and semantic renames that are
//! applied before final rendering rather than by post-render text surgery.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use regex::Regex;

use crate::ast::Expr;
use crate::ir::{FunctionIR, IRCond, IRExpr, IRStatement, RValue, RegId};
use crate::semantic_lift::SemanticLiftResult;
use crate::semantic_propagation::{propagate_semantic_labels, SsaRegKey};
use crate::type_inference::{infer_ssa_types, InferredType};

// ---------------------------------------------------------------------------
// Cached regex objects — compiled once, reused across all calls.
// ---------------------------------------------------------------------------

fn ssa_token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(?:UR|UP|R|P)\d+\.\d+\b").unwrap())
}

fn nr_ident_word_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b").unwrap())
}

fn nr_ternary_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(\s*)(\S+) = (.+?) \? \((.*)\) : (/\*phi\*/)?(\S+);$").unwrap())
}

fn nr_if_assign_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*if \((.+)\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=.*;$").unwrap())
}

fn nr_plain_assign_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*;$").unwrap())
}

fn nr_pred_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(?:P|UP)\d+\b").unwrap())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameStyle {
    Temp,
    RegisterFamily,
    VerbatimSsa,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameRecoveryConfig {
    pub style: NameStyle,
    pub rewrite_control_predicates: bool,
    pub semantic_symbolization: bool,
}

impl Default for NameRecoveryConfig {
    fn default() -> Self {
        Self {
            style: NameStyle::Temp,
            rewrite_control_predicates: true,
            semantic_symbolization: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecoveredSymbol {
    pub name: String,
    pub reg_base: (String, i32),
    pub ty_hint: Option<&'static str>,
    pub live_in: bool,
    pub order: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StructuralNameRecoveryPlan {
    /// Final token map used during AST renaming before rendering.
    pub token_map: HashMap<String, String>,
    /// Recovered symbol metadata before output-based dead-symbol filtering.
    pub symbols: Vec<RecoveredSymbol>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RegBase {
    class: String,
    idx: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RegSsa {
    base: RegBase,
    ssa: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Loc {
    block_id: usize,
    stmt_idx: usize,
    order_in_stmt: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Group {
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

/// Build the SSA-token → recovered-name mapping from the IR without
/// mutating rendered text. The canonical backend now applies this plan
/// structurally before the final pretty-print.
#[allow(clippy::type_complexity)]
pub(crate) fn build_name_map(
    function_ir: &FunctionIR,
    config: &NameRecoveryConfig,
) -> (
    HashMap<String, String>,
    Vec<(Group, Loc, String, RegBase, Vec<RegSsa>)>,
    Vec<String>,
    BTreeMap<RegBase, usize>,
) {
    let (tokens, first_seen, mut uf, idx_of) = collect_tokens(function_ir);
    let ssa_types = infer_ssa_types(function_ir);
    let semantic_labels = propagate_semantic_labels(function_ir);

    if !matches!(config.style, NameStyle::VerbatimSsa) {
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
                                        if !ssa_name_union_compatible(
                                            &dest_key,
                                            &arg_key,
                                            &ssa_types,
                                            &semantic_labels,
                                            NameUnionEdge::Phi,
                                        ) {
                                            continue;
                                        }
                                        if let Some(arg_idx) = idx_of.get(&arg_key).copied() {
                                            uf.union(dest_uf_idx, arg_idx);
                                        }
                                    }
                                }
                        }
                    } else if let RValue::Op { opcode, args } = &stmt.value {
                        if let Some(src_key) = conservative_copy_source(opcode, args) {
                            // For true copy instructions (MOV, IMAD.MOV) we can
                            // safely unify across different register bases since
                            // the value is identical. For other ops we keep the
                            // same-base restriction as a safety measure.
                            let is_true_copy = opcode.starts_with("MOV")
                                || opcode.starts_with("UMOV")
                                || opcode.starts_with("IMAD.MOV");
                            if is_true_copy || src_key.base == dest_key.base {
                                if !ssa_name_union_compatible(
                                    &dest_key,
                                    &src_key,
                                    &ssa_types,
                                    &semantic_labels,
                                    NameUnionEdge::Copy,
                                ) {
                                    continue;
                                }
                                if let Some(src_idx) = idx_of.get(&src_key).copied() {
                                    uf.union(dest_uf_idx, src_idx);
                                }
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
        comp_rows.push((
            base_group(&base.class),
            min_loc,
            min_tok,
            base,
            members.clone(),
        ));
    }
    comp_rows.sort_by(|a, b| (a.0, a.1, &a.2, &a.3).cmp(&(b.0, b.1, &b.2, &b.3)));

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
            NameStyle::VerbatimSsa => {
                let token = &row.2;
                token
                    .chars()
                    .map(|c| match c {
                        '.' => '_',
                        other => other.to_ascii_lowercase(),
                    })
                    .collect()
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
    if !matches!(config.style, NameStyle::VerbatimSsa) {
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
    }

    (token_map, comp_rows, component_name, fam_count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NameUnionEdge {
    Phi,
    Copy,
}

fn ssa_name_union_compatible(
    lhs: &RegSsa,
    rhs: &RegSsa,
    ssa_types: &BTreeMap<RegId, InferredType>,
    semantic_labels: &HashMap<SsaRegKey, String>,
    edge: NameUnionEdge,
) -> bool {
    let lhs_label = semantic_labels.get(&SsaRegKey {
        class: lhs.base.class.clone(),
        idx: lhs.base.idx,
        ssa: Some(lhs.ssa),
    });
    let rhs_label = semantic_labels.get(&SsaRegKey {
        class: rhs.base.class.clone(),
        idx: rhs.base.idx,
        ssa: Some(rhs.ssa),
    });
    match (lhs_label, rhs_label) {
        (Some(lhs), Some(rhs)) if lhs != rhs => return false,
        (Some(_), None) | (None, Some(_)) if matches!(edge, NameUnionEdge::Phi) => return false,
        _ => {}
    }

    let lhs_ty = ssa_types.get(&RegId {
        class: lhs.base.class.clone(),
        idx: lhs.base.idx,
        sign: 1,
        ssa: Some(lhs.ssa),
    });
    let rhs_ty = ssa_types.get(&RegId {
        class: rhs.base.class.clone(),
        idx: rhs.base.idx,
        sign: 1,
        ssa: Some(rhs.ssa),
    });
    match (lhs_ty.copied(), rhs_ty.copied()) {
        (Some(lhs), Some(rhs)) => inferred_name_family(lhs) == inferred_name_family(rhs),
        (Some(_), None) | (None, Some(_)) => matches!(edge, NameUnionEdge::Copy),
        (None, None) => true,
    }
}

fn inferred_name_family(ty: InferredType) -> u8 {
    match ty {
        InferredType::F16 | InferredType::F32 | InferredType::AnyFloat => 0,
        InferredType::Ptr64 | InferredType::U64 => 1,
        InferredType::U8
        | InferredType::U16
        | InferredType::U32
        | InferredType::I32
        | InferredType::AnyInt => 2,
        InferredType::Bottom | InferredType::Top => 3,
    }
}

pub fn plan_structured_name_recovery_with_lift(
    function_ir: &FunctionIR,
    rendered: &str,
    lifted: Option<&SemanticLiftResult>,
    config: &NameRecoveryConfig,
) -> StructuralNameRecoveryPlan {
    let (mut token_map, comp_rows, component_name, fam_count) = build_name_map(function_ir, config);
    let mut recovered_names = component_name.clone();
    let mut preview = apply_token_map_to_rendered(rendered, &token_map);
    preview = simplify_predicated_ternaries(&preview);

    if config.rewrite_control_predicates {
        preview = rewrite_control_guard_predicates(
            &preview,
            &build_control_predicate_map(&comp_rows, &fam_count, &recovered_names),
        );
    }

    if config.semantic_symbolization {
        let (symbolized_output, symbolized_names) =
            semantic_symbolize_output(function_ir, &preview, lifted, &comp_rows, &recovered_names);
        preview = symbolized_output;
        recovered_names = symbolized_names;
    }

    let rename = component_name
        .iter()
        .cloned()
        .zip(recovered_names.iter().cloned())
        .filter(|(from, to)| from != to)
        .collect::<BTreeMap<_, _>>();
    for name in token_map.values_mut() {
        if let Some(replacement) = rename.get(name) {
            *name = replacement.clone();
        }
    }
    token_map.extend(build_control_predicate_map(
        &comp_rows,
        &fam_count,
        &recovered_names,
    ));

    let _ = preview;
    StructuralNameRecoveryPlan {
        token_map,
        symbols: collect_recovered_symbol_metadata(function_ir, &comp_rows, &recovered_names),
    }
}

fn semantic_symbolize_output(
    function_ir: &FunctionIR,
    output: &str,
    lifted: Option<&SemanticLiftResult>,
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
    component_name: &[String],
) -> (String, Vec<String>) {
    let mut result = output.to_string();
    let mut names = component_name.to_vec();

    let ir_rename = build_ir_semantic_rename_map(function_ir, &result, comp_rows, &names);
    result = apply_identifier_renames(&result, &ir_rename);
    apply_name_renames(&mut names, &ir_rename);

    let seed_rename =
        build_semantic_seed_rename_map(function_ir, lifted, &result, comp_rows, &names);
    result = apply_identifier_renames(&result, &seed_rename);
    apply_name_renames(&mut names, &seed_rename);

    (result, names)
}

/// SSA-graph-based semantic symbolization.
///
/// Uses `propagate_semantic_labels` to find SSA registers with well-known
/// meanings (tid_x, ctaid_y, etc.) and renames the corresponding recovered
/// temp names (v0, v1, …) to their semantic names.
fn build_ir_semantic_rename_map(
    function_ir: &FunctionIR,
    output: &str,
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
    component_name: &[String],
) -> BTreeMap<String, String> {
    let ir_labels = propagate_semantic_labels(function_ir);
    let ssa_types = infer_ssa_types(function_ir);
    let mut used = collect_code_identifiers(output);
    let mut rename = BTreeMap::<String, String>::new();

    for (row_idx, row) in comp_rows.iter().enumerate() {
        let temp_name = &component_name[row_idx];
        // Skip names that are already semantic (from a previous pass or manual)
        if !temp_name.starts_with('v') && !temp_name.starts_with('u') {
            continue;
        }
        // Check if any SSA member of this component has an IR-level label.
        let mut label: Option<&str> = None;
        let mut conflict = false;
        for member in &row.4 {
            let key = SsaRegKey {
                class: member.base.class.clone(),
                idx: member.base.idx,
                ssa: Some(member.ssa),
            };
            if let Some(candidate) = ir_labels.get(&key) {
                match label {
                    None => label = Some(candidate.as_str()),
                    Some(prev) if prev == candidate.as_str() => {}
                    Some(_) => {
                        conflict = true;
                        break;
                    }
                }
            }
        }
        if conflict {
            continue;
        }
        let Some(seed) = label else {
            continue;
        };
        if !component_accepts_semantic_seed(&row.4, seed, &ssa_types) {
            continue;
        }
        if rename.contains_key(temp_name) {
            continue;
        }
        let unique = alloc_unique_name(seed.to_string(), &mut used);
        rename.insert(temp_name.clone(), unique);
    }

    rename
}

fn build_semantic_seed_rename_map(
    function_ir: &FunctionIR,
    lifted: Option<&SemanticLiftResult>,
    output: &str,
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
    component_name: &[String],
) -> BTreeMap<String, String> {
    let Some(lifted) = lifted else {
        return BTreeMap::new();
    };
    build_lifted_seed_rename_map(function_ir, lifted, output, comp_rows, component_name)
}

fn build_lifted_seed_rename_map(
    function_ir: &FunctionIR,
    lifted: &SemanticLiftResult,
    output: &str,
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
    component_name: &[String],
) -> BTreeMap<String, String> {
    let row_of = build_row_index_map(comp_rows);
    let ssa_types = infer_ssa_types(function_ir);
    let mut used = collect_code_identifiers(output);
    let mut rename = BTreeMap::<String, String>::new();

    for (def_ref, stmt) in &lifted.by_def {
        let Some(def_expr) = lookup_stmt_def(
            function_ir,
            def_ref.block_id,
            def_ref.stmt_idx,
            def_ref.def_idx,
        ) else {
            continue;
        };
        let Some(row_idx) = row_index_for_expr(def_expr, &row_of) else {
            continue;
        };
        let Some(current_name) = component_name.get(row_idx) else {
            continue;
        };
        if rename.contains_key(current_name) {
            continue;
        }
        let Some(seed) = semantic_seed_from_lifted_expr(&stmt.rhs) else {
            continue;
        };
        if !component_accepts_semantic_seed(&comp_rows[row_idx].4, &seed, &ssa_types) {
            continue;
        }
        if name_matches_seed(current_name, &seed) {
            continue;
        }
        let unique = alloc_unique_name(seed, &mut used);
        rename.insert(current_name.clone(), unique);
    }

    rename
}

fn build_row_index_map(
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
) -> HashMap<RegSsa, usize> {
    let mut row_of = HashMap::<RegSsa, usize>::new();
    for (row_idx, row) in comp_rows.iter().enumerate() {
        for member in &row.4 {
            row_of.insert(member.clone(), row_idx);
        }
    }
    row_of
}

fn lookup_stmt_def<'a>(
    function_ir: &'a FunctionIR,
    block_id: usize,
    stmt_idx: usize,
    def_idx: usize,
) -> Option<&'a IRExpr> {
    function_ir
        .blocks
        .iter()
        .find(|block| block.id == block_id)
        .and_then(|block| block.stmts.get(stmt_idx))
        .and_then(|stmt| stmt.defs.get(def_idx))
}

fn semantic_seed_from_lifted_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Raw(text) | Expr::ConstMemSymbol(text) | Expr::Builtin(text) => {
            semantic_name_seed(text)
        }
        _ => None,
    }
}

fn name_matches_seed(name: &str, seed: &str) -> bool {
    name == seed
        || name.strip_prefix(seed).is_some_and(|suffix| {
            suffix.starts_with('_') && suffix[1..].chars().all(|c| c.is_ascii_digit())
        })
}

fn component_accepts_semantic_seed(
    members: &[RegSsa],
    seed: &str,
    ssa_types: &BTreeMap<RegId, InferredType>,
) -> bool {
    let Some(expected_family) = semantic_seed_name_family(seed) else {
        return true;
    };
    members.iter().all(|member| {
        let key = RegId {
            class: member.base.class.clone(),
            idx: member.base.idx,
            sign: 1,
            ssa: Some(member.ssa),
        };
        ssa_types
            .get(&key)
            .copied()
            .map(inferred_name_family)
            .is_none_or(|family| family == expected_family)
    })
}

fn semantic_seed_name_family(seed: &str) -> Option<u8> {
    match seed {
        "tid_x"
        | "tid_y"
        | "tid_z"
        | "ctaid_x"
        | "ctaid_y"
        | "ctaid_z"
        | "block_dim_x"
        | "block_dim_y"
        | "block_dim_z"
        | "grid_dim_x"
        | "grid_dim_y"
        | "grid_dim_z"
        | "lane_id"
        | "cga_cta_id" => Some(inferred_name_family(InferredType::U32)),
        _ => None,
    }
}

fn apply_identifier_renames(output: &str, rename: &BTreeMap<String, String>) -> String {
    if rename.is_empty() {
        return output.to_string();
    }

    let mut out = output.to_string();
    for (from, to) in rename {
        let pat = Regex::new(&format!(r"\b{}\b", regex::escape(from))).expect("valid regex");
        out = pat.replace_all(&out, to.as_str()).into_owned();
    }
    out
}

fn apply_name_renames(names: &mut [String], rename: &BTreeMap<String, String>) {
    for name in names {
        if let Some(replacement) = rename.get(name) {
            *name = replacement.clone();
        }
    }
}

fn collect_recovered_symbol_metadata(
    function_ir: &FunctionIR,
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
    recovered_names: &[String],
) -> Vec<RecoveredSymbol> {
    let ssa_types = infer_ssa_types(function_ir);
    let mut row_of = HashMap::<RegSsa, usize>::new();
    for (row_idx, row) in comp_rows.iter().enumerate() {
        for member in &row.4 {
            row_of.insert(member.clone(), row_idx);
        }
    }

    let mut defined = BTreeSet::<usize>::new();
    let mut live_in = BTreeSet::<usize>::new();
    let mut first_touch = BTreeMap::<usize, usize>::new();
    let mut next_order = 0usize;

    let mut record_touch = |row_idx: usize| {
        first_touch.entry(row_idx).or_insert_with(|| {
            let order = next_order;
            next_order += 1;
            order
        });
    };

    for block in &function_ir.blocks {
        for stmt in &block.stmts {
            let defs_in_stmt = stmt
                .defs
                .iter()
                .filter_map(|def| row_index_for_expr(def, &row_of))
                .collect::<BTreeSet<_>>();
            for row_idx in collect_stmt_use_rows(stmt, &row_of) {
                record_touch(row_idx);
                if !defined.contains(&row_idx) && !defs_in_stmt.contains(&row_idx) {
                    live_in.insert(row_idx);
                }
            }
            for row_idx in defs_in_stmt {
                record_touch(row_idx);
                defined.insert(row_idx);
            }
        }
        for (cond, _) in &block.irdst {
            let Some(IRCond::Pred { reg, .. }) = cond else {
                continue;
            };
            let Some(row_idx) = row_index_for_reg_id(reg, &row_of) else {
                continue;
            };
            record_touch(row_idx);
            if !defined.contains(&row_idx) {
                live_in.insert(row_idx);
            }
        }
    }

    let mut symbols = comp_rows
        .iter()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            let name = recovered_names.get(row_idx)?.clone();
            let mut row_ty = InferredType::Bottom;
            for member in &row.4 {
                let key = RegId {
                    class: member.base.class.clone(),
                    idx: member.base.idx,
                    sign: 1,
                    ssa: Some(member.ssa),
                };
                if let Some(ty) = ssa_types.get(&key).copied() {
                    row_ty = row_ty.join(ty);
                }
            }
            Some(RecoveredSymbol {
                name,
                reg_base: (row.3.class.clone(), row.3.idx),
                ty_hint: match row_ty {
                    InferredType::Bottom | InferredType::Top => None,
                    other => Some(other.to_c_type()),
                },
                live_in: live_in.contains(&row_idx),
                order: first_touch.get(&row_idx).copied().unwrap_or(usize::MAX),
            })
        })
        .collect::<Vec<_>>();
    symbols.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.name.cmp(&b.name)));
    symbols
}

pub fn filter_recovered_symbols_by_output(
    output: &str,
    symbols: &[RecoveredSymbol],
) -> Vec<RecoveredSymbol> {
    let used_names = collect_code_identifiers(output);
    symbols
        .iter()
        .filter(|symbol| {
            used_names.contains(&symbol.name) && !is_reserved_rendered_builtin_name(&symbol.name)
        })
        .cloned()
        .collect()
}

fn is_reserved_rendered_builtin_name(name: &str) -> bool {
    matches!(name, "shmem" | "shmem_u8")
}

fn row_index_for_reg_id(reg: &RegId, row_of: &HashMap<RegSsa, usize>) -> Option<usize> {
    reg_ssa(reg).and_then(|key| row_of.get(&key).copied())
}

fn row_index_for_expr(expr: &IRExpr, row_of: &HashMap<RegSsa, usize>) -> Option<usize> {
    let IRExpr::Reg(reg) = expr else {
        return None;
    };
    row_index_for_reg_id(reg, row_of)
}

fn collect_stmt_use_rows(stmt: &IRStatement, row_of: &HashMap<RegSsa, usize>) -> Vec<usize> {
    let mut out = Vec::new();
    if let Some(pred) = &stmt.pred {
        collect_expr_rows(pred, row_of, &mut out);
    }
    match &stmt.value {
        RValue::Op { args, .. } | RValue::Phi(args) => {
            for arg in args {
                collect_expr_rows(arg, row_of, &mut out);
            }
        }
        RValue::ImmI(_) | RValue::ImmF(_) => {}
    }
    if let Some(mem) = &stmt.mem_addr_args {
        for arg in mem {
            collect_expr_rows(arg, row_of, &mut out);
        }
    }
    for old in &stmt.pred_old_defs {
        collect_expr_rows(old, row_of, &mut out);
    }
    out
}

fn collect_expr_rows(expr: &IRExpr, row_of: &HashMap<RegSsa, usize>, out: &mut Vec<usize>) {
    match expr {
        IRExpr::Reg(reg) => {
            if let Some(row_idx) = row_index_for_reg_id(reg, row_of) {
                out.push(row_idx);
            }
        }
        IRExpr::Addr64 { lo, hi } => {
            collect_expr_rows(lo, row_of, out);
            collect_expr_rows(hi, row_of, out);
        }
        IRExpr::Mem { base, offset, .. } => {
            collect_expr_rows(base, row_of, out);
            if let Some(offset) = offset {
                collect_expr_rows(offset, row_of, out);
            }
        }
        IRExpr::Op { args, .. } => {
            for arg in args {
                collect_expr_rows(arg, row_of, out);
            }
        }
        IRExpr::ImmI(_) | IRExpr::ImmF(_) => {}
    }
}

fn collect_code_identifiers(output: &str) -> BTreeSet<String> {
    let ident = nr_ident_word_re();
    let mut used = BTreeSet::<String>::new();
    for line in output.lines() {
        let code = line.split("//").next().unwrap_or("");
        if code.trim().is_empty() {
            continue;
        }
        for m in ident.find_iter(code) {
            used.insert(m.as_str().to_string());
        }
    }
    used
}

pub fn apply_token_map_to_rendered(rendered: &str, token_map: &HashMap<String, String>) -> String {
    let re = ssa_token_re();
    re.replace_all(rendered, |caps: &regex::Captures<'_>| {
        let token = caps.get(0).expect("match").as_str();
        token_map
            .get(token)
            .cloned()
            .unwrap_or_else(|| token.to_string())
    })
    .into_owned()
}

fn build_control_predicate_map(
    comp_rows: &[(Group, Loc, String, RegBase, Vec<RegSsa>)],
    fam_count: &BTreeMap<RegBase, usize>,
    recovered_names: &[String],
) -> HashMap<String, String> {
    let mut pred_map = HashMap::<String, String>::new();
    for (row_idx, row) in comp_rows.iter().enumerate() {
        let base = &row.3;
        if !matches!(base.class.as_str(), "P" | "UP") {
            continue;
        }
        if fam_count.get(base).copied().unwrap_or(0) != 1 {
            continue;
        }
        if let Some(name) = recovered_names.get(row_idx) {
            pred_map.insert(format!("{}{}", base.class, base.idx), name.clone());
        }
    }
    pred_map
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
    let re = nr_ternary_re();
    let if_assign = nr_if_assign_re();
    let plain_assign = nr_plain_assign_re();
    let ident = nr_ident_word_re();

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
            let rhs_uses_cond_defined_same_pred = ident.find_iter(rhs).any(|m| {
                conditional_assign_pred
                    .get(m.as_str())
                    .map_or(false, |p| p == &pred_norm)
            });
            let rhs_uses_cond_defined_any = ident
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
                out.push_str(&format!("{}if ({}) {} = {};\n", indent, pred, dest, rhs));
                conditional_assign_pred.insert(dest.to_string(), pred_norm);
                continue;
            }

            // Keep genuine select; this defines dest unconditionally.
            conditional_assign_pred.remove(dest);
        }

        if let Some(caps) = if_assign.captures(line) {
            let pred_norm = normalize_predicate_text(caps.get(1).expect("pred").as_str());
            let dest = caps.get(2).expect("dest").as_str().to_string();
            conditional_assign_pred.insert(dest, pred_norm);
        } else if let Some(caps) = plain_assign.captures(line) {
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

    let ternary = nr_ternary_re();
    let if_assign = nr_if_assign_re();
    let plain_assign = nr_plain_assign_re();
    let ident = nr_ident_word_re();

    let mut rewrite_line = vec![false; lines.len()];

    for i in 0..lines.len() {
        let Some(caps) = ternary.captures(lines[i]) else {
            continue;
        };
        let dest = caps.get(2).expect("dest").as_str();
        let pred_norm = normalize_predicate_text(caps.get(3).expect("pred").as_str());

        let mut saw_use = false;
        let mut only_gated_uses = true;

        for line in lines.iter().skip(i + 1) {
            if let Some(c) = if_assign.captures(line) {
                if c.get(2).expect("dest").as_str() == dest {
                    break;
                }
            } else if let Some(c) = plain_assign.captures(line) {
                if c.get(1).expect("dest").as_str() == dest {
                    break;
                }
            }

            let uses_dest = ident.find_iter(line).any(|m| m.as_str() == dest);
            if !uses_dest {
                continue;
            }
            saw_use = true;

            let gated_by_same_pred_if = if_assign.captures(line).map_or(false, |c| {
                normalize_predicate_text(c.get(1).expect("pred").as_str()) == pred_norm
            });
            let gated_by_same_pred_ternary = ternary.captures(line).map_or(false, |c| {
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
            if let Some(caps) = ternary.captures(line) {
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
    let pred = nr_pred_re();
    let mut out = String::new();
    for line in output.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("if (") || trimmed.starts_with("while (") {
            let replaced = pred.replace_all(line, |caps: &regex::Captures<'_>| {
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
                collect_expr_tokens(def, block.id, stmt_idx, 0, &mut tokens, &mut first_seen);
            }
            if let Some(pred) = &stmt.pred {
                collect_expr_tokens(pred, block.id, stmt_idx, 1, &mut tokens, &mut first_seen);
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
                    collect_expr_tokens(arg, block.id, stmt_idx, 3, &mut tokens, &mut first_seen);
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
        IRExpr::Addr64 { lo, hi } => {
            collect_expr_tokens(lo, block_id, stmt_idx, order_in_stmt, tokens, first_seen);
            collect_expr_tokens(hi, block_id, stmt_idx, order_in_stmt, tokens, first_seen);
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
    use crate::semantic_lift::{lift_function_ir, SemanticLiftConfig};
    use crate::{build_cfg, build_ssa, decode_sass};

    fn build_fir(sass: &str) -> FunctionIR {
        build_ssa(&build_cfg(decode_sass(sass)))
    }

    fn apply_plan(
        fir: &FunctionIR,
        rendered: &str,
        config: &NameRecoveryConfig,
    ) -> (String, StructuralNameRecoveryPlan) {
        let plan = plan_structured_name_recovery_with_lift(fir, rendered, None, config);
        let output = apply_token_map_to_rendered(rendered, &plan.token_map);
        (output, plan)
    }

    fn apply_plan_with_lift(
        fir: &FunctionIR,
        rendered: &str,
        lifted: &SemanticLiftResult,
        config: &NameRecoveryConfig,
    ) -> (String, StructuralNameRecoveryPlan) {
        let plan = plan_structured_name_recovery_with_lift(fir, rendered, Some(lifted), config);
        let output = apply_token_map_to_rendered(rendered, &plan.token_map);
        (output, plan)
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
        let (out, _) = apply_plan(&fir, &rendered, &NameRecoveryConfig::default());
        for t in &r0 {
            assert!(!out.contains(t));
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
        let (out, _) = apply_plan(&fir, rendered, &NameRecoveryConfig::default());
        assert!(out.contains("v0 = 1;"));
        assert!(out.contains("v1 = 2;"));
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
        let (out, _) = apply_plan(&fir, rendered, &NameRecoveryConfig::default());
        assert!(out.contains('v'));
        assert!(out.contains('u'));
        assert!(out.contains('b'));
        assert!(!out.contains("R1."));
        assert!(!out.contains("UR4."));
        assert!(!out.contains("P0."));
    }

    #[test]
    fn token_rewrite_is_safe() {
        let sass = r#"
            /*0000*/ IADD3 R1, R1, 0x1, RZ ;
            /*0010*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let rendered = "BB10 {\n  R1.0 = IADD3(R1.1, 1, RZ);\n}";
        let (out, _) = apply_plan(&fir, rendered, &NameRecoveryConfig::default());
        assert!(out.contains("BB10"));
        assert!(out.contains("IADD3("));
    }

    #[test]
    fn deterministic_mapping() {
        let sass = include_str!("../test_cu/if_loop.sass");
        let fir = build_fir(sass);
        let rendered = "R1.0 = R1.1 + R2.0; P0.1 = R1.0 >= 0;";
        let (out1, plan1) = apply_plan(&fir, rendered, &NameRecoveryConfig::default());
        let (out2, plan2) = apply_plan(&fir, rendered, &NameRecoveryConfig::default());
        assert_eq!(out1, out2);
        assert_eq!(plan1.token_map, plan2.token_map);
        assert_eq!(plan1.symbols, plan2.symbols);
    }

    #[test]
    fn rewrites_control_predicate_when_unambiguous() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let (_, plan) = apply_plan(&fir, "if (P0) {\n}\n", &NameRecoveryConfig::default());
        assert!(plan
            .token_map
            .get("P0")
            .is_some_and(|name| name.starts_with('b')));
    }

    #[test]
    fn keeps_control_predicate_raw_when_ambiguous() {
        let sass = r#"
            /*0000*/ ISETP.GE.AND P0, PT, R0, 0x1, PT ;
            /*0010*/ ISETP.LT.AND P0, PT, R1, 0x2, PT ;
            /*0020*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let (_, plan) = apply_plan(&fir, "if (P0) {\n}\n", &NameRecoveryConfig::default());
        assert!(!plan.token_map.contains_key("P0"));
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
        let input = "  if (!b41) v176 = v175 & 255;\n  v177 = !b41 ? (shmem_u8[v176]) : v176;\n";
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
        let rendered = "R2.0 = threadIdx.x;\nif (R2.0 > 1) {\n}\n";
        let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());
        let (out, _) = apply_plan_with_lift(
            &fir,
            rendered,
            &lifted,
            &NameRecoveryConfig {
                style: NameStyle::Temp,
                rewrite_control_predicates: true,
                semantic_symbolization: true,
            },
        );
        assert!(out.contains("tid_x = threadIdx.x;"));
        assert!(out.contains("if (tid_x > 1)"));
    }

    #[test]
    fn semantic_symbolization_updates_symbol_metadata() {
        let sass = r#"
            /*0000*/ S2R R2, SR_TID.X ;
            /*0010*/ EXIT ;
        "#;
        let fir = build_fir(sass);
        let rendered = "R2.0 = threadIdx.x;\nif (R2.0 > 1) {\n}\n";
        let lifted = lift_function_ir(&fir, &SemanticLiftConfig::default());
        let (_, plan) = apply_plan_with_lift(
            &fir,
            rendered,
            &lifted,
            &NameRecoveryConfig {
                style: NameStyle::Temp,
                rewrite_control_predicates: true,
                semantic_symbolization: true,
            },
        );
        let name_to_reg_base = plan
            .symbols
            .iter()
            .map(|symbol| (symbol.name.clone(), symbol.reg_base.clone()))
            .collect::<HashMap<_, _>>();
        assert!(name_to_reg_base.contains_key("tid_x"));
        assert!(!name_to_reg_base.contains_key("v0"));
        assert!(plan.symbols.iter().any(|symbol| symbol.name == "tid_x"));
    }

    #[test]
    fn filter_recovered_symbols_drops_shared_memory_builtins() {
        let symbols = vec![RecoveredSymbol {
            name: "shmem".to_string(),
            reg_base: ("R".to_string(), 0),
            ty_hint: Some("uint32_t"),
            live_in: false,
            order: 0,
        }];

        let filtered = filter_recovered_symbols_by_output("v0 = shmem[idx];\n", &symbols);
        assert!(filtered.is_empty(), "got {:?}", filtered);
    }
}
