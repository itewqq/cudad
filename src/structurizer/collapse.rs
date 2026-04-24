    use super::*;

    /// Stable index into the region vector. Never invalidated.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub(super) struct RegionId(pub(super) usize);

    /// What a region currently represents.
    #[derive(Clone, Debug)]
    pub(super) enum RegionKind {
        /// A still-unstructured CFG basic block.
        Basic { block_id: usize },
        /// A region that has already been folded into a structured statement.
        /// `primary_block_id` is the IR block id whose `irdst` governs the
        /// region's *current* outgoing branch — for a Sequence, it's the tail
        /// block; for an If, it's the merge-or-head block id.
        Composite {
            stmt: StructuredStatement,
            primary_block_id: usize,
        },
        /// Slot is dead. Callers must not traverse it.
        Tombstone,
    }

    /// A directed edge between two regions with the metadata collapse rules
    /// care about.
    #[derive(Clone, Debug)]
    pub(super) struct RegionEdge {
        pub from: RegionId,
        pub to: RegionId,
        /// Which arm of the source's branch this edge represents. `Some(true)`
        /// = the oriented "true" arm, `Some(false)` = the "false" arm, `None`
        /// = an unconditional/fallthrough edge.
        pub cond_arm: Option<bool>,
        /// True iff the source-to-target edge is a back edge in the original
        /// CFG. Computed once at construction and never updated.
        pub is_back_edge: bool,
        /// True once `select_goto_edge` has marked this edge as "give up, emit
        /// it as an UnstructuredJump". Collapse rules skip goto-marked edges.
        pub is_goto: bool,
    }

    /// Stable-index region graph.
    pub(super) struct RegionGraph {
        regions: Vec<RegionKind>,
        edges: Vec<Option<RegionEdge>>,
        out_edges: Vec<Vec<usize>>,
        in_edges: Vec<Vec<usize>>,
        entry: RegionId,
    }

    impl RegionGraph {
        // ------------- construction -------------

        pub(super) fn build_from_cfg(s: &Structurizer<'_>) -> RegionGraph {
            // Deterministic node ordering by BasicBlock.id so results don't
            // depend on petgraph's internal ordering.
            let mut ordered: Vec<NodeIndex> = s.cfg.node_indices().collect();
            ordered.sort_by_key(|&n| s.cfg[n].id);

            let mut regions: Vec<RegionKind> = Vec::with_capacity(ordered.len());
            let mut cfg_to_region: HashMap<NodeIndex, RegionId> = HashMap::new();

            for cfg_node in &ordered {
                let block_id = s.cfg[*cfg_node].id;
                let rid = RegionId(regions.len());
                cfg_to_region.insert(*cfg_node, rid);

                // Seed return blocks as Composite right away so the collapse
                // loop sees them as terminal regions. Mirrors the legacy
                // behavior at structurizer.rs:461-498.
                let mut seeded = false;
                if let Some(ir_block) = s.get_ir_block_by_cfg_node(*cfg_node) {
                    if Structurizer::is_block_return(ir_block) {
                        let stmt = build_return_stmt(ir_block);
                        regions.push(RegionKind::Composite {
                            stmt,
                            primary_block_id: block_id,
                        });
                        seeded = true;
                    }
                }
                if !seeded {
                    regions.push(RegionKind::Basic { block_id });
                }
            }

            let n = regions.len();
            let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut in_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut edges: Vec<Option<RegionEdge>> = Vec::new();

            for cfg_node in &ordered {
                let from = cfg_to_region[cfg_node];
                // Orient the arms using the existing helper — it handles all
                // four irdst shapes (Pred/Pred, Pred/True, True/Pred, Pred).
                let ir_block = match s.get_ir_block_by_cfg_node(*cfg_node) {
                    Some(b) => b,
                    None => continue,
                };
                let targets = s.extract_if_targets_and_condition(ir_block);

                // Deterministic succs ordering: sort by target block id.
                let mut succs: Vec<NodeIndex> = s
                    .cfg
                    .neighbors_directed(*cfg_node, Direction::Outgoing)
                    .collect();
                succs.sort_by_key(|&n| s.cfg[n].id);

                // Single-CFG-edge blocks never represent a real CFG-level
                // branch from the structurizer's perspective: any embedded
                // predicated jump/return is rendered inside the block by the
                // pretty printer (`if (P0) return;`), so the outgoing edge is
                // effectively unconditional.
                let single_edge = succs.len() == 1;

                for succ in succs.iter().copied() {
                    let to = cfg_to_region[&succ];
                    let cond_arm = if single_edge {
                        None
                    } else {
                        match &targets {
                            Some((_, t, Some(f))) => {
                                if succ == *t {
                                    Some(true)
                                } else if succ == *f {
                                    Some(false)
                                } else {
                                    None
                                }
                            }
                            Some((_, t, None)) => {
                                if succ == *t {
                                    Some(true)
                                } else {
                                    Some(false)
                                }
                            }
                            None => None,
                        }
                    };
                    let is_back_edge = s.node_is_dominated_by(&s.dom, *cfg_node, succ);
                    let edge_idx = edges.len();
                    edges.push(Some(RegionEdge {
                        from,
                        to,
                        cond_arm,
                        is_back_edge,
                        is_goto: false,
                    }));
                    out_edges[from.0].push(edge_idx);
                    in_edges[to.0].push(edge_idx);
                }
            }

            // Entry region: same entry-picking logic as the old code.
            let entry_cfg =
                Structurizer::choose_entry_node(s.cfg).unwrap_or_else(|| NodeIndex::new(0));
            let entry = cfg_to_region
                .get(&entry_cfg)
                .copied()
                .unwrap_or(RegionId(0));

            let graph = RegionGraph {
                regions,
                edges,
                out_edges,
                in_edges,
                entry,
            };

            graph
        }

        // ------------- queries -------------

        pub(super) fn region(&self, r: RegionId) -> &RegionKind {
            &self.regions[r.0]
        }

        pub(super) fn is_alive(&self, r: RegionId) -> bool {
            !matches!(self.regions[r.0], RegionKind::Tombstone)
        }

        pub(super) fn entry(&self) -> RegionId {
            self.entry
        }

        pub(super) fn active_ids(&self) -> Vec<RegionId> {
            (0..self.regions.len())
                .filter(|&i| !matches!(self.regions[i], RegionKind::Tombstone))
                .map(RegionId)
                .collect()
        }

        pub(super) fn active_count(&self) -> usize {
            self.regions
                .iter()
                .filter(|r| !matches!(r, RegionKind::Tombstone))
                .count()
        }

        /// All live (non-goto, non-removed) successor edges of `r`.
        /// Returns `(edge_index, &edge)` pairs so callers can mutate.
        pub(super) fn live_succs(&self, r: RegionId) -> Vec<(usize, RegionEdge)> {
            self.out_edges[r.0]
                .iter()
                .filter_map(|&ei| {
                    let e = self.edges[ei].as_ref()?;
                    if e.is_goto {
                        return None;
                    }
                    Some((ei, e.clone()))
                })
                .collect()
        }

        /// All live predecessor edges of `r`.
        pub(super) fn live_preds(&self, r: RegionId) -> Vec<(usize, RegionEdge)> {
            self.in_edges[r.0]
                .iter()
                .filter_map(|&ei| {
                    let e = self.edges[ei].as_ref()?;
                    if e.is_goto {
                        return None;
                    }
                    Some((ei, e.clone()))
                })
                .collect()
        }

        // ------------- mutation -------------

        pub(super) fn add_region(&mut self, kind: RegionKind) -> RegionId {
            let rid = RegionId(self.regions.len());
            self.regions.push(kind);
            self.out_edges.push(Vec::new());
            self.in_edges.push(Vec::new());
            rid
        }

        /// Mark an edge as removed from the graph. Adjacency entries are
        /// cleaned lazily by the edge-iteration helpers (they skip `None`).
        pub(super) fn remove_edge(&mut self, edge_idx: usize) {
            self.edges[edge_idx] = None;
        }

        fn push_edge(&mut self, e: RegionEdge) -> usize {
            let idx = self.edges.len();
            let from = e.from;
            let to = e.to;
            self.edges.push(Some(e));
            self.out_edges[from.0].push(idx);
            self.in_edges[to.0].push(idx);
            idx
        }

        pub(super) fn add_edge(
            &mut self,
            from: RegionId,
            to: RegionId,
            cond_arm: Option<bool>,
            is_back_edge: bool,
        ) -> usize {
            self.push_edge(RegionEdge {
                from,
                to,
                cond_arm,
                is_back_edge,
                is_goto: false,
            })
        }

        /// Tombstone a region. Caller is responsible for having already
        /// removed or rewired every edge incident to it.
        pub(super) fn tombstone(&mut self, r: RegionId) {
            self.regions[r.0] = RegionKind::Tombstone;
            self.out_edges[r.0].clear();
            self.in_edges[r.0].clear();
        }

        /// Rewire every live incoming edge `X → old` to `X → new`. Back-edge
        /// status and cond_arm are preserved.
        pub(super) fn redirect_in_edges(&mut self, old: RegionId, new: RegionId) {
            let in_edge_idxs: Vec<usize> = self.in_edges[old.0].clone();
            for ei in in_edge_idxs {
                if let Some(edge) = self.edges[ei].as_mut() {
                    edge.to = new;
                    self.in_edges[new.0].push(ei);
                }
            }
            self.in_edges[old.0].clear();
        }

        /// Rewire every live outgoing edge `old → Y` to `new → Y`.
        pub(super) fn redirect_out_edges(&mut self, old: RegionId, new: RegionId) {
            let out_edge_idxs: Vec<usize> = self.out_edges[old.0].clone();
            for ei in out_edge_idxs {
                if let Some(edge) = self.edges[ei].as_mut() {
                    edge.from = new;
                    self.out_edges[new.0].push(ei);
                }
            }
            self.out_edges[old.0].clear();
        }

        pub(super) fn set_entry(&mut self, r: RegionId) {
            self.entry = r;
        }
    }

    // ------------- helpers -------------

    /// Build a Return-terminated statement for an IR block whose last stmt
    /// is an unconditional RET/EXIT. Mirrors structurizer.rs:461-498.
    fn build_return_stmt(ir_block: &IRBlock) -> StructuredStatement {
        let ret_val = ir_block.stmts.last().and_then(|s| {
            if let RValue::Op { opcode, args } = &s.value {
                if (opcode.starts_with("RET") || opcode == "EXIT") && !args.is_empty() {
                    return Some(args[0].clone());
                }
            }
            None
        });
        let mut pre_return_stmts = ir_block.stmts.clone();
        if let Some(IRStatement {
            value: RValue::Op { opcode, .. },
            pred,
            ..
        }) = pre_return_stmts.last()
        {
            if pred.is_none() && Structurizer::is_return_opcode(opcode) {
                pre_return_stmts.pop();
            }
        }
        let has_renderable_pre_return_stmt = pre_return_stmts.iter().any(|s| {
            if matches!(s.value, RValue::Phi(_)) {
                return false;
            }
            match &s.value {
                RValue::Op { opcode, .. } => {
                    !Structurizer::is_branch_only_opcode(opcode)
                        && !Structurizer::is_return_opcode(opcode)
                }
                _ => true,
            }
        });
        let mut seq = Vec::new();
        if has_renderable_pre_return_stmt {
            seq.push(StructuredStatement::BasicBlock {
                block_id: ir_block.id,
                stmts: pre_return_stmts,
            });
        }
        seq.push(StructuredStatement::Return(ret_val));
        if seq.len() == 1 {
            seq.into_iter().next().unwrap()
        } else {
            StructuredStatement::Sequence(seq)
        }
    }

    /// Materialize a region's StructuredStatement — cloning for Composite or
    /// building a fresh `BasicBlock` for Basic regions.
    fn stmt_for_region(
        graph: &RegionGraph,
        r: RegionId,
        s: &Structurizer<'_>,
    ) -> StructuredStatement {
        match graph.region(r) {
            RegionKind::Basic { block_id, .. } => {
                let stmts = s
                    .get_ir_block(*block_id)
                    .map(|b| b.stmts.clone())
                    .unwrap_or_default();
                StructuredStatement::BasicBlock {
                    block_id: *block_id,
                    stmts,
                }
            }
            RegionKind::Composite { stmt, .. } => stmt.clone(),
            RegionKind::Tombstone => StructuredStatement::Empty,
        }
    }

    fn primary_block_id_of(graph: &RegionGraph, r: RegionId) -> usize {
        match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            RegionKind::Composite {
                primary_block_id, ..
            } => *primary_block_id,
            RegionKind::Tombstone => usize::MAX,
        }
    }

    fn target_block_id_for_edge(
        graph: &RegionGraph,
        from: RegionId,
        cond_arm: Option<bool>,
        s: &Structurizer<'_>,
    ) -> Option<usize> {
        let from_bid = primary_block_id_of(graph, from);
        let ir_block = s.get_ir_block(from_bid)?;
        let from_cfg = s.cfg_node_for_block_id(from_bid)?;

        let outgoing = || {
            s.cfg
                .neighbors_directed(from_cfg, Direction::Outgoing)
                .map(|node| s.cfg[node].id)
                .collect::<Vec<_>>()
        };

        if let Some((_, true_node, false_node)) = s.extract_if_targets_and_condition(ir_block) {
            let true_bid = s.cfg[true_node].id;
            match cond_arm {
                Some(true) => return Some(true_bid),
                Some(false) => {
                    if let Some(false_node) = false_node {
                        return Some(s.cfg[false_node].id);
                    }
                    let mut others = outgoing()
                        .into_iter()
                        .filter(|bid| *bid != true_bid)
                        .collect::<Vec<_>>();
                    others.sort_unstable();
                    others.dedup();
                    if others.len() == 1 {
                        return others.into_iter().next();
                    }
                }
                None => {}
            }
        }

        if ir_block.irdst.len() == 1 {
            let (_, addr) = ir_block.irdst.first()?;
            if let Some(node) = s.addr_to_node_index.get(addr) {
                return Some(s.cfg[*node].id);
            }
        }

        let mut succs = outgoing();
        succs.sort_unstable();
        succs.dedup();
        if succs.len() == 1 {
            succs.into_iter().next()
        } else {
            None
        }
    }

    /// Concatenate two statements into a flattened Sequence, dropping Empty
    /// children and collapsing nested Sequences.
    fn concat_stmts(a: StructuredStatement, b: StructuredStatement) -> StructuredStatement {
        let mut out: Vec<StructuredStatement> = Vec::new();
        match a {
            StructuredStatement::Empty => {}
            StructuredStatement::Sequence(v) => {
                for s in v {
                    if !matches!(s, StructuredStatement::Empty) {
                        out.push(s);
                    }
                }
            }
            other => out.push(other),
        }
        match b {
            StructuredStatement::Empty => {}
            StructuredStatement::Sequence(v) => {
                for s in v {
                    if !matches!(s, StructuredStatement::Empty) {
                        out.push(s);
                    }
                }
            }
            other => out.push(other),
        }
        match out.len() {
            0 => StructuredStatement::Empty,
            1 => out.into_iter().next().unwrap(),
            _ => StructuredStatement::Sequence(out),
        }
    }

    // ------------- collapse rules -------------

    /// Sequence collapse: fold `A → B` when neither end has a branch structure
    /// still pending. Forbidden from absorbing into a 2-way head or a
    /// back-edge tail (see plan §7).
    pub(super) fn try_seq(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let succs = graph.live_succs(r);
        if succs.len() != 1 {
            return false;
        }
        let (_, a_to_b) = &succs[0];
        if a_to_b.cond_arm.is_some() || a_to_b.is_back_edge {
            return false;
        }
        let b = a_to_b.to;
        if b == r {
            return false;
        }
        let preds_b = graph.live_preds(b);
        if preds_b.len() != 1 {
            return false;
        }
        let succs_b = graph.live_succs(b);
        match succs_b.len() {
            0 => {}
            1 => {
                let (_, out_b) = &succs_b[0];
                if out_b.cond_arm.is_some() || out_b.is_back_edge {
                    return false;
                }
            }
            2 => {
                // Multi-block do-while shape: A → B(test), B → A (back-edge)
                // and B → X (forward exit). After folding A and B together,
                // the back-edge to A becomes a self-back-edge on the new
                // composite, which `try_do_while` then matches. Refuse any
                // other 2-successor pattern (the partition rule keeps 2-way
                // forward branches in the hands of `try_if_*` / `try_while_do`).
                let back_to_a = succs_b.iter().any(|(_, e)| e.to == r && e.is_back_edge);
                let forward_other = succs_b.iter().any(|(_, e)| e.to != r && !e.is_back_edge);
                if !(back_to_a && forward_other) {
                    return false;
                }
            }
            _ => return false,
        }

        // Build the new Composite statement.
        let a_stmt = stmt_for_region(graph, r, s);
        let b_stmt = stmt_for_region(graph, b, s);
        let new_stmt = concat_stmts(a_stmt, b_stmt);
        let new_primary = primary_block_id_of(graph, b);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: new_stmt,
            primary_block_id: new_primary,
        });

        // Rewire: X→A edges (other than A→B) now go X→NEW.
        // The only incoming edge of B is A→B; remove it before redirecting.
        // The A→B edge itself is removed.
        let a_to_b_idx = succs[0].0;
        graph.remove_edge(a_to_b_idx);

        // Pull all live X→A edges onto NEW.
        graph.redirect_in_edges(r, new_rid);
        // Pull all live B→Y edges onto NEW.
        graph.redirect_out_edges(b, new_rid);

        // Entry update.
        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        // Note: B can't have been entry if r had the only pred of B and r is
        // alive, but handle it defensively.
        if graph.entry() == b {
            graph.set_entry(new_rid);
        }

        graph.tombstone(r);
        graph.tombstone(b);
        true
    }

    // ------------- if/else collapse -------------

    /// Try to collapse `A` as an `if-then-else`: `A` has two oriented arms to
    /// distinct regions `T` and `F`, both arms are forward and converge at a
    /// common merge `M` (or both terminate). The condition expression comes
    /// from `A` itself (which must remain Basic so the prelude printer can
    /// emit its statements before the `if`).
    pub(super) fn try_if_else(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        // Heads must be Basic so condition_block_id refers to a real IR block
        // and the prelude renderer can emit A's statements before the if.
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };

        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }
        // Identify the true/false arms.
        let (true_idx, true_e, false_idx, false_e) = match (&succs[0], &succs[1]) {
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) => {
                (*i0, e0.clone(), *i1, e1.clone())
            }
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) => {
                (*i1, e1.clone(), *i0, e0.clone())
            }
            _ => return false,
        };
        if true_e.is_back_edge || false_e.is_back_edge {
            return false;
        }
        let t = true_e.to;
        let f = false_e.to;
        if t == r || f == r || t == f {
            return false;
        }
        // Both branches must be single-entry from `r`.
        let preds_t = graph.live_preds(t);
        let preds_f = graph.live_preds(f);
        if preds_t.len() != 1 || preds_f.len() != 1 {
            return false;
        }
        let succs_t = graph.live_succs(t);
        let succs_f = graph.live_succs(f);
        // Each branch is either terminal (no live succs) or has a single
        // unconditional forward edge to a common merge.
        if succs_t.len() > 1 || succs_f.len() > 1 {
            return false;
        }
        let t_merge = succs_t.first().map(|(_, e)| e.clone());
        let f_merge = succs_f.first().map(|(_, e)| e.clone());
        let merge: Option<RegionId> = match (&t_merge, &f_merge) {
            (None, None) => None,
            (Some(et), Some(ef)) => {
                if et.to != ef.to
                    || et.cond_arm.is_some()
                    || ef.cond_arm.is_some()
                    || et.is_back_edge
                    || ef.is_back_edge
                {
                    return false;
                }
                Some(et.to)
            }
            // Asymmetric merges (one terminal, one not) — fall through to
            // try_if_then if appropriate.
            _ => return false,
        };

        // Build statement.
        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let then_stmt = stmt_for_region(graph, t, s);
        let else_stmt = stmt_for_region(graph, f, s);
        let if_stmt = StructuredStatement::If {
            condition_block_id: head_block_id,
            condition_expr: cond_expr,
            then_branch: Box::new(then_stmt),
            else_branch: Some(Box::new(else_stmt)),
        };

        // Build the new Composite. Its primary block id is the merge if there
        // is one, else the head's id.
        let new_primary = if let Some(m) = merge {
            primary_block_id_of(graph, m)
        } else {
            head_block_id
        };
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: if_stmt,
            primary_block_id: new_primary,
        });

        // Edge surgery.
        graph.remove_edge(true_idx);
        graph.remove_edge(false_idx);
        if let Some((tm_idx, _)) = succs_t.first() {
            graph.remove_edge(*tm_idx);
        }
        if let Some((fm_idx, _)) = succs_f.first() {
            graph.remove_edge(*fm_idx);
        }
        graph.redirect_in_edges(r, new_rid);
        if let Some(m) = merge {
            // The new region falls through to the merge.
            graph.add_edge(new_rid, m, None, false);
        }
        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(t);
        graph.tombstone(f);
        true
    }

    /// Try to collapse `A` as a one-armed `if`: `A` has two oriented arms,
    /// one going to a body `T` that re-joins via the *other* arm's target as
    /// the implicit merge. If the body is on the false arm, we negate.
    pub(super) fn try_if_then(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };
        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }
        let (true_idx, true_e, false_idx, false_e) = match (&succs[0], &succs[1]) {
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) => {
                (*i0, e0.clone(), *i1, e1.clone())
            }
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) => {
                (*i1, e1.clone(), *i0, e0.clone())
            }
            _ => return false,
        };
        if true_e.is_back_edge || false_e.is_back_edge {
            return false;
        }
        let t = true_e.to;
        let f = false_e.to;
        if t == r || f == r || t == f {
            return false;
        }

        // Try with the true arm as the body, then the false arm as the body.
        // body_side: which arm holds the body, the other is the implicit merge.
        // A body is acceptable if it is single-entry from `r` AND either
        //   (a) terminates with no live successors (e.g. a seeded return
        //       composite), or
        //   (b) has a single unconditional forward edge back to `merge`.
        let attempt = |graph: &RegionGraph, body: RegionId, merge: RegionId| -> bool {
            let preds_b = graph.live_preds(body);
            if preds_b.len() != 1 {
                return false;
            }
            let succs_b = graph.live_succs(body);
            if succs_b.is_empty() {
                // Terminal body (e.g. seeded return). Only safe when this
                // is not actually a while-loop shape: reject if the merge
                // arm has a back-edge to `r`, signaling that `r` is a loop
                // header and `try_while_do` should run instead.
                for (_, me) in graph.live_succs(merge) {
                    if me.is_back_edge && me.to == r {
                        return false;
                    }
                }
                return true;
            }
            if succs_b.len() != 1 {
                return false;
            }
            let (_, eb) = &succs_b[0];
            if eb.cond_arm.is_some() || eb.is_back_edge || eb.to != merge {
                return false;
            }
            true
        };

        let (body, body_idx, body_edge_to_merge_idx, negate) = if attempt(graph, t, f) {
            let mi = graph.live_succs(t).first().map(|(i, _)| *i);
            (t, true_idx, mi, false)
        } else if attempt(graph, f, t) {
            let mi = graph.live_succs(f).first().map(|(i, _)| *i);
            (f, false_idx, mi, true)
        } else {
            return false;
        };
        let merge = if negate { t } else { f };
        let other_arm_idx = if negate { true_idx } else { false_idx };

        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let cond_oriented = if negate {
            negate_condition(cond_expr)
        } else {
            cond_expr
        };
        let then_stmt = stmt_for_region(graph, body, s);
        let if_stmt = StructuredStatement::If {
            condition_block_id: head_block_id,
            condition_expr: cond_oriented,
            then_branch: Box::new(then_stmt),
            else_branch: None,
        };

        let new_primary = primary_block_id_of(graph, merge);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: if_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(body_idx);
        graph.remove_edge(other_arm_idx);
        if let Some(mi) = body_edge_to_merge_idx {
            graph.remove_edge(mi);
        }
        graph.redirect_in_edges(r, new_rid);
        graph.add_edge(new_rid, merge, None, false);

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(body);
        true
    }

    /// One-arm if-then collapse: handles 2-way Basic heads where one arm has
    /// already been stripped by `drop_shortcut_branch_edges`. The remaining
    /// edge still carries `cond_arm = Some(_)`, the body is reached only via
    /// that conditional edge, and the body's natural exit is the implicit
    /// merge — exactly the shape the OLD heuristic produced for diamonds with
    /// a "shortcut" edge.
    pub(super) fn try_if_then_one_arm(
        graph: &mut RegionGraph,
        r: RegionId,
        s: &Structurizer<'_>,
    ) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };
        let succs = graph.live_succs(r);
        if succs.len() != 1 {
            return false;
        }
        let (body_idx, body_e) = (succs[0].0, succs[0].1.clone());
        let arm = match body_e.cond_arm {
            Some(a) => a,
            None => return false,
        };
        if body_e.is_back_edge {
            return false;
        }
        let body = body_e.to;
        if body == r {
            return false;
        }
        // Body must be single-entry from us so we can absorb it.
        let preds_b = graph.live_preds(body);
        if preds_b.len() != 1 {
            return false;
        }
        // The body must terminate or fall into a single forward merge.
        let succs_b = graph.live_succs(body);
        let merge_opt: Option<RegionId> = match succs_b.len() {
            0 => None,
            1 => {
                let (_, eb) = &succs_b[0];
                if eb.cond_arm.is_some() || eb.is_back_edge {
                    return false;
                }
                Some(eb.to)
            }
            _ => return false,
        };

        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let cond_oriented = if arm {
            cond_expr
        } else {
            negate_condition(cond_expr)
        };
        let body_stmt = stmt_for_region(graph, body, s);
        let if_stmt = StructuredStatement::If {
            condition_block_id: head_block_id,
            condition_expr: cond_oriented,
            then_branch: Box::new(body_stmt),
            else_branch: None,
        };

        let new_primary = match merge_opt {
            Some(m) => primary_block_id_of(graph, m),
            None => head_block_id,
        };
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: if_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(body_idx);
        if let Some((mi, _)) = succs_b.first() {
            graph.remove_edge(*mi);
        }
        graph.redirect_in_edges(r, new_rid);
        if let Some(m) = merge_opt {
            graph.add_edge(new_rid, m, None, false);
        }

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(body);
        true
    }

    // ------------- loop collapse -------------

    /// While-do loop: head `H` is a Basic 2-way branch where one arm goes to
    /// a body `B` (which then back-edges to `H`) and the other arm exits.
    ///
    /// The structurizer used to set `Loop::header_block_id = Some(H)`, which
    /// drove the printer to render `H`'s IR statements as a *prelude* before
    /// the loop construct *and* the body Sequence still embedded `H`,
    /// duplicating it. We avoid that by emitting a Sequence with `H` as a
    /// plain BasicBlock followed by the Loop with `header_block_id = None`.
    pub(super) fn try_while_do(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let head_block_id = match graph.region(r) {
            RegionKind::Basic { block_id, .. } => *block_id,
            _ => return false,
        };
        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }
        let (true_idx, true_e, false_idx, false_e) = match (&succs[0], &succs[1]) {
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(true) && e1.cond_arm == Some(false) => {
                (*i0, e0.clone(), *i1, e1.clone())
            }
            ((i0, e0), (i1, e1)) if e0.cond_arm == Some(false) && e1.cond_arm == Some(true) => {
                (*i1, e1.clone(), *i0, e0.clone())
            }
            _ => return false,
        };
        // For a while-do, neither arm of the head is the back edge — the back
        // edge is from the body region.
        if true_e.is_back_edge || false_e.is_back_edge {
            return false;
        }

        // Try body=true,exit=false; then swap.
        let try_arm = |graph: &RegionGraph, body: RegionId| -> bool {
            if body == r {
                return false;
            }
            let preds_b = graph.live_preds(body);
            if preds_b.len() != 1 {
                return false;
            }
            let succs_b = graph.live_succs(body);
            if succs_b.len() != 1 {
                return false;
            }
            let (_, eb) = &succs_b[0];
            // Body's only successor must be the head, via a back edge.
            if eb.to != r || !eb.is_back_edge {
                return false;
            }
            true
        };

        let (body, exit, negate) = if try_arm(graph, true_e.to) {
            (true_e.to, false_e.to, false)
        } else if try_arm(graph, false_e.to) {
            (false_e.to, true_e.to, true)
        } else {
            return false;
        };

        let ir_block = match s.get_ir_block(head_block_id) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        let cond_oriented = if negate {
            negate_condition(cond_expr)
        } else {
            cond_expr
        };
        let body_stmt = stmt_for_region(graph, body, s);
        let head_stmt = stmt_for_region(graph, r, s);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::While,
            header_block_id: None,
            condition_expr: Some(cond_oriented),
            body: Box::new(body_stmt),
        };
        let new_stmt = concat_stmts(head_stmt, loop_stmt);

        // Find the body→head back edge index so we can remove it.
        let back_edge_idx: Option<usize> = graph
            .live_succs(body)
            .iter()
            .find(|(_, e)| e.to == r && e.is_back_edge)
            .map(|(i, _)| *i);

        let new_primary = primary_block_id_of(graph, exit);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: new_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(true_idx);
        graph.remove_edge(false_idx);
        if let Some(bi) = back_edge_idx {
            graph.remove_edge(bi);
        }
        // Note: redirect_in_edges only moves still-live entries; the body's
        // back-edge from r is already removed.
        graph.redirect_in_edges(r, new_rid);
        graph.add_edge(new_rid, exit, None, false);

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        graph.tombstone(body);
        true
    }

    /// Do-while loop: a region `R` whose only outgoing edges are a self
    /// back-edge (`R → R`, conditional) and a forward exit (`R → X`,
    /// conditional, opposite arm). After `try_seq` folds a multi-block loop
    /// body, the back edge becomes this self-edge.
    pub(super) fn try_do_while(graph: &mut RegionGraph, r: RegionId, s: &Structurizer<'_>) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let succs = graph.live_succs(r);
        if succs.len() != 2 {
            return false;
        }

        // Identify the self back-edge and the forward exit.
        let mut back: Option<(usize, RegionEdge)> = None;
        let mut exit: Option<(usize, RegionEdge)> = None;
        for (ei, e) in &succs {
            if e.to == r && e.is_back_edge {
                back = Some((*ei, e.clone()));
            } else if !e.is_back_edge {
                exit = Some((*ei, e.clone()));
            }
        }
        let (back_idx, back_e) = match back {
            Some(x) => x,
            None => return false,
        };
        let (exit_idx, exit_e) = match exit {
            Some(x) => x,
            None => return false,
        };
        // Both arms must be predicate-driven, opposite.
        let (b_arm, x_arm) = (back_e.cond_arm, exit_e.cond_arm);
        match (b_arm, x_arm) {
            (Some(true), Some(false)) | (Some(false), Some(true)) => {}
            _ => return false,
        }

        // The condition expression lives on the *tail* basic block (the one
        // whose irdst spawned this 2-way branch). Look it up by primary id.
        let tail_bid = primary_block_id_of(graph, r);
        let ir_block = match s.get_ir_block(tail_bid) {
            Some(b) => b,
            None => return false,
        };
        let (cond_expr, _, _) = match s.extract_if_targets_and_condition(ir_block) {
            Some(t) => t,
            None => return false,
        };
        // Orient so that "true" means keep looping (i.e. take the back edge).
        let cond_oriented = if b_arm == Some(true) {
            cond_expr
        } else {
            negate_condition(cond_expr)
        };

        let body_stmt = stmt_for_region(graph, r, s);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::DoWhile,
            header_block_id: Some(tail_bid),
            condition_expr: Some(cond_oriented),
            body: Box::new(body_stmt),
        };

        let exit_target = exit_e.to;
        let new_primary = primary_block_id_of(graph, exit_target);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: loop_stmt,
            primary_block_id: new_primary,
        });

        graph.remove_edge(back_idx);
        graph.remove_edge(exit_idx);
        graph.redirect_in_edges(r, new_rid);
        graph.add_edge(new_rid, exit_target, None, false);

        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        true
    }

    /// Degenerate self-loop: `R` has exactly one live successor which is the
    /// self back-edge (no exit). Emits an `Endless` loop.
    pub(super) fn try_self_loop(
        graph: &mut RegionGraph,
        r: RegionId,
        s: &Structurizer<'_>,
    ) -> bool {
        if !graph.is_alive(r) {
            return false;
        }
        let succs = graph.live_succs(r);
        if succs.len() != 1 {
            return false;
        }
        let (back_idx, back_e) = (succs[0].0, succs[0].1.clone());
        if back_e.to != r || !back_e.is_back_edge {
            return false;
        }
        let body_stmt = stmt_for_region(graph, r, s);
        let loop_stmt = StructuredStatement::Loop {
            loop_type: LoopType::Endless,
            header_block_id: None,
            condition_expr: None,
            body: Box::new(body_stmt),
        };
        let new_primary = primary_block_id_of(graph, r);
        let new_rid = graph.add_region(RegionKind::Composite {
            stmt: loop_stmt,
            primary_block_id: new_primary,
        });
        graph.remove_edge(back_idx);
        graph.redirect_in_edges(r, new_rid);
        if graph.entry() == r {
            graph.set_entry(new_rid);
        }
        graph.tombstone(r);
        true
    }

    // ------------- switch recognition -------------

    /// Try to fold a node with 3+ outgoing edges into a Switch statement.
    ///
    /// A switch candidate is a region with ≥3 live successors (from a branch
    /// table or cascaded conditional pattern), where all successors converge
    /// to a common post-dominator. Each successor becomes a case arm.
    ///
    /// This is intentionally conservative: it only fires when ALL case arms
    /// are leaf nodes (no further control flow) that converge to the same
    /// merge point, avoiding complex nesting issues.
    fn try_switch(graph: &mut RegionGraph, head: RegionId, s: &Structurizer<'_>) -> bool {
        let succs = graph.live_succs(head);
        if succs.len() < 3 {
            return false;
        }

        // Check that all successors are leaves: each has at most 1 successor,
        // and all share the same single successor (the merge point).
        let mut merge_target: Option<RegionId> = None;
        let mut leaf_succs: Vec<(RegionId, usize)> = Vec::new(); // (succ_id, edge_idx)

        for (ei, e) in &succs {
            let succ_succs = graph.live_succs(e.to);
            if succ_succs.len() > 1 {
                return false; // case arm has branches — too complex
            }
            if succ_succs.len() == 1 {
                let merge = succ_succs[0].1.to;
                match merge_target {
                    None => merge_target = Some(merge),
                    Some(m) if m == merge => {}
                    Some(_) => return false, // different merge points
                }
            }
            // succ_succs.len() == 0 means terminal (return/exit) — allowed
            leaf_succs.push((e.to, *ei));
        }

        // At least one arm must have a merge target (otherwise they're all terminals,
        // which wouldn't benefit from switch folding).
        // Actually, even all-terminal is fine for a switch — we just don't need a merge.

        // Build the switch statement.
        let header_bid = primary_block_id_of(graph, head);
        let head_stmt = stmt_for_region(graph, head, s);

        let mut cases = Vec::new();
        for (idx, (succ_rid, _edge_idx)) in leaf_succs.iter().enumerate() {
            let case_body = stmt_for_region(graph, *succ_rid, s);
            cases.push((idx, case_body));
        }

        let switch_stmt = StructuredStatement::Switch {
            header_block_id: header_bid,
            discriminant: None, // Could be extracted from BRX operand in future
            cases,
            default: None,
        };

        // If there's a head statement (prelude code), wrap it in a sequence.
        let final_stmt = match head_stmt {
            StructuredStatement::Empty => switch_stmt,
            _ => StructuredStatement::Sequence(vec![head_stmt, switch_stmt]),
        };

        // Replace head with the switch composite.
        graph.regions[head.0] = RegionKind::Composite {
            stmt: final_stmt,
            primary_block_id: header_bid,
        };

        // Remove all edges from head to succs
        for (ei, _) in &succs {
            graph.remove_edge(*ei);
        }

        // Tombstone all case-arm regions
        for (succ_rid, _) in &leaf_succs {
            // Remove outgoing edges from the case arm to the merge target
            for (sei, _) in graph.live_succs(*succ_rid) {
                graph.remove_edge(sei);
            }
            // Remove incoming edges to the case arm
            for (pei, _) in graph.live_preds(*succ_rid) {
                graph.remove_edge(pei);
            }
            graph.tombstone(*succ_rid);
        }

        // If there's a merge target, add an edge from head to merge
        if let Some(merge) = merge_target {
            graph.add_edge(head, merge, None, false);
        }

        true
    }

    // ------------- node splitting for irreducible CFG -------------

    /// Maximum IR statements in a Basic region that we're willing to duplicate.
    const SPLIT_MAX_STMTS: usize = 8;
    /// Maximum number of basic-block headers/blocks represented by a
    /// composite region that we're willing to duplicate to recover a
    /// reducible shape. Keeping this small avoids exploding code size while
    /// still handling common shared-tail patterns like nested remainder
    /// ladders.
    const SPLIT_MAX_COMPOSITE_BLOCKS: usize = 3;
    /// Maximum number of node splits per structurization to avoid blowup.
    const SPLIT_BUDGET: usize = 12;

    fn structured_stmt_block_count(stmt: &StructuredStatement) -> usize {
        match stmt {
            StructuredStatement::BasicBlock { .. } => 1,
            StructuredStatement::Sequence(stmts) => {
                stmts.iter().map(structured_stmt_block_count).sum()
            }
            StructuredStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                1 + structured_stmt_block_count(then_branch)
                    + else_branch
                        .as_deref()
                        .map(structured_stmt_block_count)
                        .unwrap_or(0)
            }
            StructuredStatement::Loop {
                header_block_id, body, ..
            } => {
                usize::from(header_block_id.is_some()) + structured_stmt_block_count(body)
            }
            StructuredStatement::Switch { cases, default, .. } => {
                1 + cases
                    .iter()
                    .map(|(_, body)| structured_stmt_block_count(body))
                    .sum::<usize>()
                    + default
                        .as_deref()
                        .map(structured_stmt_block_count)
                        .unwrap_or(0)
            }
            StructuredStatement::Break(_)
            | StructuredStatement::Continue(_)
            | StructuredStatement::Return(_)
            | StructuredStatement::UnstructuredJump { .. }
            | StructuredStatement::Empty => 0,
        }
    }

    /// Try to make the region graph reducible by splitting a small node that
    /// has multiple non-back-edge predecessors.
    ///
    /// An irreducible region typically has a node reachable via two different
    /// forward paths — it has >1 non-back-edge predecessor from distinct
    /// regions. By duplicating the target and redirecting one predecessor to
    /// the copy, the region becomes reducible and the normal fold rules can
    /// proceed.
    ///
    /// We only split Basic regions with ≤ `SPLIT_MAX_STMTS` IR statements to
    /// keep the output compact. Returns true if a split happened.
    fn try_split_for_reducibility(
        graph: &mut RegionGraph,
        s: &Structurizer<'_>,
        split_count: &mut usize,
    ) -> bool {
        if *split_count >= SPLIT_BUDGET {
            return false;
        }

        // Find a candidate: a live region with ≥2 non-back-edge, non-goto predecessors.
        let mut best: Option<(RegionId, Vec<(usize, RegionEdge)>)> = None;
        for rid in graph.active_ids() {
            if rid == graph.entry() {
                continue; // never split the entry node
            }
            let preds = graph.live_preds(rid);
            let non_back: Vec<_> = preds
                .into_iter()
                .filter(|(_, e)| !e.is_back_edge && !e.is_goto)
                .collect();
            if non_back.len() < 2 {
                continue;
            }
            // Only split regions whose duplicated payload is small enough to
            // stay readable. Small composites are especially useful for
            // shared-tail ladders where multiple heads flow into the same
            // cleanup block sequence.
            let duplicable = match graph.region(rid) {
                RegionKind::Basic { block_id } => s
                    .function_ir
                    .blocks
                    .iter()
                    .find(|b| b.id == *block_id)
                    .map(|b| b.stmts.len())
                    .map(|n| n <= SPLIT_MAX_STMTS)
                    .unwrap_or(false),
                RegionKind::Composite { stmt, .. } => {
                    structured_stmt_block_count(stmt) <= SPLIT_MAX_COMPOSITE_BLOCKS
                }
                RegionKind::Tombstone => false,
            };
            if !duplicable {
                continue;
            }
            // Prefer the candidate with the smallest region id (deterministic).
            if best.as_ref().map_or(true, |(cur, _)| rid.0 < cur.0) {
                best = Some((rid, non_back));
            }
        }

        let Some((target, non_back_preds)) = best else {
            return false;
        };

        // Split: duplicate the target region.
        let new_kind = graph.region(target).clone();
        let copy = graph.add_region(new_kind);

        // Duplicate all outgoing edges from `target` to `copy`.
        let out_edges: Vec<(RegionId, Option<bool>, bool)> = graph
            .live_succs(target)
            .into_iter()
            .map(|(_, e)| (e.to, e.cond_arm, e.is_back_edge))
            .collect();
        for (to, arm, be) in out_edges {
            graph.add_edge(copy, to, arm, be);
        }

        // Redirect one of the non-back-edge predecessors to point to the copy
        // instead of the original. Pick the one with the largest from-region-id
        // (deterministic, and avoids touching the entry-side path).
        let redirect_pred = non_back_preds
            .iter()
            .max_by_key(|(_, e)| e.from.0)
            .map(|(ei, _)| *ei);
        if let Some(ei) = redirect_pred {
            if let Some(edge) = graph.edges[ei].as_mut() {
                // Remove old in-edge association
                graph.in_edges[target.0].retain(|&x| x != ei);
                edge.to = copy;
                graph.in_edges[copy.0].push(ei);
            }
        }

        *split_count += 1;
        true
    }

    // ------------- goto fallback -------------

    /// Minimal Phase A fallback: if the collapse loop stalled, walk the graph
    /// and mark the first live edge as goto. Later phases will pick more
    /// intelligently.
    pub(super) fn select_goto_edge(graph: &mut RegionGraph, s: &Structurizer<'_>) -> bool {
        // Deterministic selection: smallest (from, to) region id pair.
        let mut candidate: Option<(RegionId, RegionId, usize, Option<bool>)> = None;
        for rid in graph.active_ids() {
            for (ei, e) in graph.live_succs(rid) {
                let key = (e.from, e.to, ei, e.cond_arm);
                candidate = Some(match candidate {
                    None => key,
                    Some(cur) => {
                        if (key.0 .0, key.1 .0) < (cur.0 .0, cur.1 .0) {
                            key
                        } else {
                            cur
                        }
                    }
                });
            }
        }
        let Some((from_r, _to_r, edge_idx, cond_arm)) = candidate else {
            return false;
        };

        // Materialize the goto into the source region's stmt. Convert the
        // source into a Composite that ends with an UnstructuredJump, so the
        // printer can emit "if (P0) goto BBxx;" or bare "goto BBxx;".
        let cur_stmt = stmt_for_region(graph, from_r, s);
        let from_bid = primary_block_id_of(graph, from_r);
        let Some(to_bid) = target_block_id_for_edge(graph, from_r, cond_arm, s) else {
            return false;
        };
        let cond_ir = cond_for_arm(graph, from_r, cond_arm, s);
        let jump = StructuredStatement::UnstructuredJump {
            from_block_id: from_bid,
            to_block_id: to_bid,
            condition: cond_ir,
        };
        let new_stmt = concat_stmts(cur_stmt, jump);
        // In place, update the source region to Composite carrying this stmt.
        // We only care to rewrite the stmt, not the edges.
        match &mut graph.regions[from_r.0] {
            RegionKind::Basic { .. } | RegionKind::Composite { .. } => {
                graph.regions[from_r.0] = RegionKind::Composite {
                    stmt: new_stmt,
                    primary_block_id: from_bid,
                };
            }
            RegionKind::Tombstone => return false,
        }
        // Flag the edge so future collapse rules skip it.
        if let Some(edge) = graph.edges[edge_idx].as_mut() {
            edge.is_goto = true;
        }
        true
    }

    /// Look up the branch predicate for the source's selected arm, so the
    /// goto fallback can emit `if (pred) goto BBxx;` rather than an
    /// unconditional jump (which would change semantics).
    fn cond_for_arm(
        graph: &RegionGraph,
        from: RegionId,
        cond_arm: Option<bool>,
        s: &Structurizer<'_>,
    ) -> Option<IRExpr> {
        let arm = cond_arm?;
        let bid = primary_block_id_of(graph, from);
        let ir_block = s.get_ir_block(bid)?;
        let (cond_expr, _, _) = s.extract_if_targets_and_condition(ir_block)?;
        Some(if arm {
            cond_expr
        } else {
            negate_condition(cond_expr)
        })
    }

    // ------------- main driver -------------

    pub(super) fn run_collapse(graph: &mut RegionGraph, s: &Structurizer<'_>) {
        let initial = graph.active_count().max(1);
        let max_iter = 4 * initial * initial + 100;
        let mut iter = 0usize;
        let mut split_count = 0usize;
        let debug_collapse = std::env::var("DEBUG_COLLAPSE").is_ok();
        loop {
            iter += 1;
            if iter > max_iter {
                eprintln!(
                    "structurizer: collapse exceeded {} iterations; bailing",
                    max_iter
                );
                break;
            }
            let mut changed = false;
            for rid in graph.active_ids() {
                if !graph.is_alive(rid) {
                    continue;
                }
                if try_seq(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: seq on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_if_else(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: if_else on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_if_then(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: if_then on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_if_then_one_arm(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: if_then_one_arm on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_while_do(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: while_do on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_do_while(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: do_while on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_self_loop(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: self_loop on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
                if try_switch(graph, rid, s) {
                    if debug_collapse {
                        eprintln!("collapse iter {iter}: switch on R{}", rid.0);
                    }
                    changed = true;
                    break;
                }
            }
            if changed {
                continue;
            }
            if graph.active_count() <= 1 {
                break;
            }
            // Before resorting to goto, try node splitting for irreducible CFG.
            if try_split_for_reducibility(graph, s, &mut split_count) {
                if debug_collapse {
                    eprintln!("collapse iter {iter}: split");
                }
                continue;
            }
            if debug_collapse {
                eprintln!("collapse iter {iter}: fallback goto");
                for rid in graph.active_ids() {
                    let succs = graph
                        .live_succs(rid)
                        .into_iter()
                        .map(|(_, e)| {
                            format!(
                                "R{}->R{} arm={:?} back={} goto={}",
                                e.from.0, e.to.0, e.cond_arm, e.is_back_edge, e.is_goto
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "  R{} {:?} succs=[{}]",
                        rid.0,
                        graph.region(rid),
                        succs.join(", ")
                    );
                }
            }
            if !select_goto_edge(graph, s) {
                break;
            }
        }
    }

    // ------------- root emission -------------

    /// Walk reachable regions from `entry` and concatenate them. In the happy
    /// path exactly one region survives; in the fallback path we thread
    /// leftover regions together with explicit gotos already materialized
    /// into their statements.
    pub(super) fn emit_root(graph: &RegionGraph, s: &Structurizer<'_>) -> StructuredStatement {
        // DFS from entry, preorder, deterministic.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut order: Vec<RegionId> = Vec::new();
        let mut stack: Vec<RegionId> = vec![graph.entry()];
        while let Some(rid) = stack.pop() {
            if !visited.insert(rid.0) {
                continue;
            }
            if !graph.is_alive(rid) {
                continue;
            }
            order.push(rid);
            // Push successors in reverse-sorted order so we visit in
            // ascending (from, to) region-id order.
            let mut out: Vec<RegionId> = graph.out_edges[rid.0]
                .iter()
                .filter_map(|&ei| graph.edges[ei].as_ref().map(|e| e.to))
                .collect();
            out.sort_by_key(|r| r.0);
            out.dedup();
            for r in out.into_iter().rev() {
                if !visited.contains(&r.0) {
                    stack.push(r);
                }
            }
        }

        // Collect statements from the walked regions.
        let mut parts: Vec<StructuredStatement> = Vec::new();
        for rid in order {
            match graph.region(rid) {
                RegionKind::Basic { .. } => {
                    // A Basic region that survived to emission means the
                    // collapse loop stalled before folding it in. Emit its
                    // raw IR stmts via the shared lookup helper.
                    parts.push(stmt_for_region(graph, rid, s));
                }
                RegionKind::Composite { stmt, .. } => parts.push(stmt.clone()),
                RegionKind::Tombstone => {}
            }
        }
        match parts.len() {
            0 => StructuredStatement::Empty,
            1 => parts.into_iter().next().unwrap(),
            _ => StructuredStatement::Sequence(parts),
        }
    }
