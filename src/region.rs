//! region.rs  –  Region graph + helpers used by the structuriser
//! -------------------------------------------------------------
//! The `RegionForest` is a thin abstraction layer that sits between the
//! original *Control‑Flow Graph* (CFG) and the high‑level *AST* emitted
//! by `structurizer`.  Each CFG basic‑block is wrapped in a `RegionKind::Basic`
//! node; whenever we detect a structured pattern (if/loop/seq) we collapse
//! the corresponding sub‑graph into a single `RegionKind::Composite` node
//! holding the freshly built `AstNode`.  The surrounding edges are rewired
//! so the rest of the algorithm can continue pattern matching on a smaller
//! graph.
//!
//! The implementation purposefully keeps the surface small – just enough
//! for the structuriser – without tying ourselves to any particular IR
//! detail.

use crate::ast::AstNode;
use crate::cfg::ControlFlowGraph;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

/// A *region* is either a plain basic‑block (`Basic`) or a higher‑level
/// structured construct that already has an `AstNode` representation
/// (`Composite`).  Collapsing always creates a `Composite` node.
#[derive(Clone, Debug)]
pub enum RegionKind {
    Basic(NodeIndex),      // points back into the original CFG
    Composite(AstNode),
}

impl RegionKind {
    #[inline]
    pub fn is_basic(&self) -> bool {
        matches!(self, RegionKind::Basic(_))
    }
}

/// RegionForest wraps an *ever‑shrinking* graph whose nodes are the current
/// regions.  `blk_map` keeps track of which region the **original** CFG node
/// currently belongs to – this is handy when later passes need to query the
/// region of some basic block.
pub struct RegionForest<'a> {
    /// Region graph (directed, same direction as CFG edges)
    pub rg: Graph<RegionKind, ()>,
    /// Region node that corresponds to the CFG entry – treated as the root
    /// when printing/debugging but algorithmically not special.
    pub root: NodeIndex,
    /// Mapping *original CFG node* ➜ *current region node*.
    pub blk_map: HashMap<NodeIndex, NodeIndex>,
    /// Borrowed reference to the immutable original CFG.  Used only for
    /// debugging and for anchoring back to block addresses when emitting
    /// pseudocode comments.
    _cfg: &'a ControlFlowGraph,
}

impl<'a> RegionForest<'a> {
    /// Construct an initial forest where each basic block forms its own
    /// region.  All CFG edges are mirrored 1‑for‑1 in the region graph.
    pub fn new(cfg: &'a ControlFlowGraph) -> Self {
        let mut rg = Graph::<RegionKind, ()>::new();
        let mut blk_map = HashMap::new();

        for n in cfg.node_indices() {
            let idx = rg.add_node(RegionKind::Basic(n));
            blk_map.insert(n, idx);
        }
        // copy all edges (direction preserved)
        for e in cfg.edge_references() {
            let s = blk_map[&e.source()];
            let d = blk_map[&e.target()];
            rg.add_edge(s, d, ());
        }
        let root = blk_map[&cfg.node_indices().next().expect("empty CFG?")];
        Self { rg, root, blk_map, _cfg: cfg }
    }

    /// Collapse an *internal, connected* set of region nodes into a new
    /// `Composite` node that owns the supplied `ast`.  Incoming edges from
    /// outside the set now point to the new node; outgoing edges from the
    /// set now originate from the new node.  All inner edges are discarded.
    ///
    /// *Panics* if `sub_nodes` is empty or contains nodes not in this forest.
    pub fn collapse(&mut self, sub_nodes: &HashSet<NodeIndex>, ast: AstNode) -> NodeIndex {
        assert!(!sub_nodes.is_empty(), "collapse: empty set");
        // 1. create the replacement node first
        let new_region = self.rg.add_node(RegionKind::Composite(ast));

        // 2. record all incoming / outgoing edges that cross the boundary
        let mut incomings = Vec::<(NodeIndex, NodeIndex)>::new();
        let mut outgoings = Vec::<(NodeIndex, NodeIndex)>::new();

        for &old in sub_nodes {
            // incoming
            for e in self.rg.edges_directed(old, Direction::Incoming) {
                if !sub_nodes.contains(&e.source()) {
                    incomings.push((e.source(), new_region));
                }
            }
            // outgoing
            for e in self.rg.edges_directed(old, Direction::Outgoing) {
                if !sub_nodes.contains(&e.target()) {
                    outgoings.push((new_region, e.target()));
                }
            }
        }

        // 3. remove the old region nodes (edges are removed automatically)
        for &old in sub_nodes {
            self.rg.remove_node(old);
        }

        // 4. re‑add external edges towards the new node
        for (s, d) in incomings { self.rg.add_edge(s, d, ()); }
        for (s, d) in outgoings { self.rg.add_edge(s, d, ()); }

        new_region
    }

    /// Helper: membership test of original CFG node ➜ current region.
    pub fn region_of_block(&self, blk: NodeIndex) -> NodeIndex {
        self.blk_map[&blk]
    }

    /// Replace mapping of a basic block to its new composite region after
    /// collapsing.  Callers **must** ensure that all basic blocks inside the
    /// collapsed region get remapped; otherwise future look‑ups may panic.
    pub fn remap_blocks<I>(&mut self, new_r: NodeIndex, blocks: I)
    where
        I: IntoIterator<Item = NodeIndex>,
    {
        for b in blocks {
            self.blk_map.insert(b, new_r);
        }
    }
}
