# cudad 改进计划

> 本文档为 Claude Code 准备，包含完整上下文、问题根因、修复方案和验证步骤。
> 所有 212 条现有测试必须在每次修改后继续通过（`cargo test`）。

---

## 背景

`cudad` 是一个实验性 CUDA SASS 反编译器，流水线为：

```
parse_sass → build_cfg → build_ssa → structurize → semantic_lift → name_recovery → abi_render
```

代码审计发现了两类严重问题：

1. **工程层面：该用索引/高效数据结构的地方使用了 O(n) 线性扫描**——在大函数上形成性能瓶颈。
2. **架构层面：名称恢复和局部变量推断在渲染后文本上用正则做**——绕过了 AST/IR 级别操作，难以维护且容易出错。

下面按优先级从高到低排列每个问题的完整上下文和修复方案。

---

## 第一类：工程层面性能问题

### Issue 1.1: CFG fall-through 边查找使用 O(n) HashMap 遍历

**文件**: `src/cfg.rs:93-98`

**现状**:
```rust
// 对每个基本块，遍历整个 addr2node HashMap 来查找 "地址 > 当前块 start 的最小地址"
if let Some((&_next_addr, &nidx)) = addr2node.iter()
    .filter(|(&a, _)| a > bb_start)
    .min_by_key(|(&a, _)| a) {
    ...
}
```

**根因**: `addr2node` 是 `HashMap<u32, NodeIndex>`（第 66 行），不支持有序查询。对每个基本块做一次 O(n) 全扫描，整个 edge-building 阶段的总复杂度为 O(n²)。

**修复方案**:
1. 将 `addr2node` 从 `HashMap<u32, NodeIndex>` 改为 `BTreeMap<u32, NodeIndex>`。
2. 将 fall-through 查找改为 `addr2node.range((Excluded(bb_start), Unbounded)).next()`，O(log n)。

**具体步骤**:
```
文件: src/cfg.rs
第 66 行: 将 `let mut addr2node = std::collections::HashMap::<u32, NodeIndex>::new();`
改为:     `let mut addr2node = std::collections::BTreeMap::<u32, NodeIndex>::new();`

第 93-98 行: 将整个 filter+min_by_key 链
改为:
    use std::ops::Bound::*;
    if let Some((&_next_addr, &nidx)) = addr2node.range((Excluded(bb_start), Unbounded)).next() {
        if g.find_edge(idx, nidx).is_none() {
            g.update_edge(idx, nidx, EdgeKind::FallThrough);
        }
    }
```

**验证**: `cargo test` 全部通过。无需新增测试，行为不变。

---

### Issue 1.2: Structurizer 中 IR 块查找使用 O(n) 线性扫描

**文件**: `src/structurizer.rs:303-306`

**现状**:
```rust
fn get_ir_block_by_cfg_node(&self, cfg_node: NodeIndex) -> Option<&'a IRBlock> {
    self.cfg.node_weight(cfg_node).and_then(|summary| {
        self.function_ir.blocks.iter().find(|b| b.id == summary.id)
    })
}
```

**根因**: `FunctionIR.blocks` 是 `Vec<IRBlock>`，查找 `id == summary.id` 是 O(n)。此函数在 structurizer 中被高频调用（几乎每个 collapse rule 都会调用）。

**修复方案**:
1. 在 `Structurizer::new()` 中预建一个 `HashMap<usize, usize>` 将 block.id → blocks 数组索引的映射。
2. 修改 `get_ir_block_by_cfg_node` 使用该映射做 O(1) 查找。

**具体步骤**:
```
文件: src/structurizer.rs

在 Structurizer 结构体中（约第 88 行）新增字段:
    block_id_to_idx: HashMap<usize, usize>,

在 Structurizer::new()（约第 248 行）中初始化:
    let block_id_to_idx: HashMap<usize, usize> = function_ir.blocks.iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

修改 get_ir_block_by_cfg_node:
    fn get_ir_block_by_cfg_node(&self, cfg_node: NodeIndex) -> Option<&'a IRBlock> {
        let summary = self.cfg.node_weight(cfg_node)?;
        let &idx = self.block_id_to_idx.get(&summary.id)?;
        Some(&self.function_ir.blocks[idx])
    }
```

**验证**: `cargo test` 全部通过。行为不变。

---

### Issue 1.3: 支配前沿计算中不必要的 BTreeSet clone

**文件**: `src/ir.rs:748-759`

**现状**:
```rust
fn up(n, child, df, idom) {
    // 递归子树
    for &c in child.get(&n).unwrap_or(&Vec::new()) {
        let set = df.entry(c).or_default().clone();   // ← 整个 BTreeSet clone
        for w in set {
            if idom.get(&w).copied() != Some(n) {
                df.entry(n).or_default().insert(w);
            }
        }
    }
}
```

**根因**: 由于 borrow checker 限制，用 `.clone()` 规避了同时 immutable 和 mutable 借用的问题。但每次 clone 一个 BTreeSet 的开销为 O(|DF(c)|)，总开销在病态 CFG 上可达 O(n²)。

**修复方案**:
先收集所有需要传播的元素到一个临时 Vec，再批量 insert，避免 clone：
```rust
for &c in child.get(&n).unwrap_or(&Vec::new()) {
    let propagate: Vec<NodeIndex> = df.get(&c).cloned().unwrap_or_default()
        .into_iter()
        .filter(|w| idom.get(w).copied() != Some(n))
        .collect();
    for w in propagate {
        df.entry(n).or_default().insert(w);
    }
}
```

注意: `.cloned().unwrap_or_default()` 仍然需要 clone，但可以用更底层的方式优化。一个更彻底的方案是使用 Cytron 原始论文的迭代 DF 计算（不递归 up-propagation），但这属于大重构，当前修复足以消除 hot path 上的冗余 clone。

**验证**: `cargo test` 全部通过。

---

### Issue 1.4: `loop_of()` 查询使用线性扫描

**文件**: `src/cfg_analysis.rs:135-137`

**现状**:
```rust
pub fn loop_of(&self, n: NodeIndex) -> Option<&NaturalLoop> {
    self.loops.iter().find(|lp| lp.body.contains(&n))
}
```

**根因**: 对每次查询都线性扫描所有 loops，每个 loop 的 `body.contains` 是 O(log |body|)。如果频繁查询，总复杂度为 O(L × log B)。

**修复方案**:
在 `CFGAnalysis::new()` 中预建 `HashMap<NodeIndex, usize>` (node → loop index)：

```
文件: src/cfg_analysis.rs

在 CFGAnalysis 结构体新增字段:
    node_to_loop: HashMap<NodeIndex, usize>,

在 CFGAnalysis::new() 中初始化:
    let mut node_to_loop = HashMap::new();
    for (i, lp) in loops.iter().enumerate() {
        for &n in &lp.body {
            node_to_loop.entry(n).or_insert(i);
        }
    }

修改 loop_of:
    pub fn loop_of(&self, n: NodeIndex) -> Option<&NaturalLoop> {
        self.node_to_loop.get(&n).map(|&i| &self.loops[i])
    }
```

**注意**: 如果一个节点属于多个嵌套循环，`entry().or_insert()` 只保留第一个匹配。这与原始 `.find()` 行为一致（取第一个匹配的 loop）。如果将来需要支持嵌套循环查询，可改为 `Vec<usize>`。

**验证**: `cargo test` 全部通过。

---

### Issue 1.5: 后支配树 exit 节点选择不完整（语义正确性问题）

**文件**: `src/cfg_analysis.rs:51-55`

**现状**:
```rust
let exit = cfg
    .node_indices()
    .find(|&n| cfg.neighbors_directed(n, Direction::Outgoing).next().is_none())
    .unwrap_or(NodeIndex::new(0));
```

**根因**: 简单地取"第一个没有后继的节点"作为反向图入口。如果 CFG 有多个 EXIT/RET 节点（即多个出口），只有一个被选为反向图入口，导致其他出口节点不可达，后支配树结果不完整/不正确。

**这是一个语义正确性问题**，不仅是性能问题。

**修复方案**:
添加一个虚拟 unique-exit 节点，将所有出口节点连接到它，再对反向图计算支配：

```rust
fn compute_postdom(cfg: &ControlFlowGraph) -> BTreeMap<NodeIndex, NodeIndex> {
    // 收集所有出口节点（无后继 + 含无条件 EXIT/RET）
    let exits: Vec<NodeIndex> = cfg
        .node_indices()
        .filter(|&n| cfg.neighbors_directed(n, Direction::Outgoing).next().is_none())
        .collect();

    if exits.len() <= 1 {
        // 只有一个出口，原逻辑正确
        let exit = exits.first().copied().unwrap_or(NodeIndex::new(0));
        let rev = petgraph::visit::Reversed(cfg);
        let doms = simple_fast(&rev, exit);
        let mut out = BTreeMap::new();
        for n in cfg.node_indices() {
            if let Some(i) = doms.immediate_dominator(n) {
                out.insert(n, i);
            }
        }
        return out;
    }

    // 多出口：克隆 CFG，添加虚拟 unique-exit 节点
    let mut aug = cfg.clone();
    let virtual_exit = aug.add_node(BasicBlock {
        id: usize::MAX,
        start: u32::MAX,
        instrs: Vec::new(),
    });
    for exit in &exits {
        aug.add_edge(*exit, virtual_exit, EdgeKind::FallThrough);
    }
    let rev = petgraph::visit::Reversed(&aug);
    let doms = simple_fast(&rev, virtual_exit);
    let mut out = BTreeMap::new();
    for n in cfg.node_indices() {
        if let Some(i) = doms.immediate_dominator(n) {
            if i != virtual_exit {
                out.insert(n, i);
            }
        }
    }
    out
}
```

**注意**: 这需要 `use crate::cfg::{BasicBlock, EdgeKind};`，以及确保 `BasicBlock` 和 `ControlFlowGraph` 支持 `Clone`（已经 derive 了）。virtual_exit 使用 `usize::MAX` 作为 id 避免与真实块冲突。

**验证**: `cargo test` 全部通过。可能需要更新少量 golden 输出——如果后支配树之前就算错了的话。如果 golden 不变，说明当前测试用例碰巧只有单出口。

---

## 第二类：架构层面——文本级 regex 操作应提升到 AST/IR 级别

### Issue 2.1: 名称恢复在渲染后文本上做 token 替换

**文件**: `src/name_recovery.rs:293-307`

**现状**:
`recover_structured_output_names()` 的输入是 `rendered: &str`（已渲染为文本的伪代码），输出也是 `String`。核心逻辑在第 293-307 行：

```rust
let re = Regex::new(r"\b(?:UR|UP|R|P)\d+\.\d+\b").expect("valid regex");
let mut output = re.replace_all(rendered, |caps: &regex::Captures<'_>| {
    ssa_tokens_seen += 1;
    let t = caps.get(0).expect("match").as_str();
    if let Some(rep) = token_map.get(t) {
        rewritten_tokens += 1;
        rep.clone()
    } else {
        t.to_string()
    }
}).into_owned();
```

**根因**:
1. Union-Find 同余分析（第 122-159 行）是在 IR 上做的（正确），但最终的替换是在文本上做的。
2. 这意味着注释中的 SSA token 也会被替换（可能不期望）。
3. 更严重的是，替换后的后处理（`simplify_predicated_ternaries`，第 457-510 行）用了更复杂的正则在文本上做条件赋值 → if-guard 的变换——这实质上是在文本上重新解析并修改 AST 结构。

**修复方案（分阶段）**:

#### 阶段 A: 将 token_map 传入 pretty_print 阶段（中等工作量）

不需要重构整个管线。核心思路：

1. `recover_structured_output_names()` 的前半部分（Union-Find 分析 + 组件命名，第 122-239 行）已经在 IR 上运行，可以独立提取为一个纯函数 `build_name_map(function_ir, config) -> HashMap<String, String>`。
2. 将这个 name map 传入 `Structurizer::pretty_print_with_lift()` 作为一个可选的 `name_map: Option<&HashMap<String, String>>`。
3. 在 `pretty_print` 内部渲染 `IRExpr::Reg(r)` 时，如果 name_map 中有映射就用映射名，否则用原名。这样名称替换发生在渲染时而非渲染后。

**具体步骤**:

```
1. 在 src/name_recovery.rs 中新增函数:
   pub fn build_name_map(function_ir: &FunctionIR, config: &NameRecoveryConfig)
       -> (HashMap<String, String>, NameRecoveryStats)
   提取现有 recover_structured_output_names() 的第 122-259 行逻辑。

2. 新增一个 NameAwareDisplay 实现 DisplayCtx trait:
   struct NameAwareDisplay<'a> {
       inner: &'a dyn DisplayCtx,
       name_map: &'a HashMap<String, String>,
   }
   impl DisplayCtx for NameAwareDisplay {
       fn reg(&self, r: &RegId) -> String {
           let raw = self.inner.reg(r);
           self.name_map.get(&raw).cloned().unwrap_or(raw)
       }
   }

3. 在 src/bin/main.rs 的 emit_struct_code() 中:
   - 在 structurizer pretty_print 之前调用 build_name_map() 获得 name_map
   - 用 NameAwareDisplay 包装 display_ctx
   - 将包装后的 ctx 传入 pretty_print_with_lift()

4. 保留旧的 recover_structured_output_names() 作为 fallback，
   用 #[deprecated] 标注，未来移除。
```

#### 阶段 B: 将 ternary 简化提升到 StructuredStatement AST 级别（较大工作量）

当前 `simplify_predicated_ternaries()`（第 457-560 行）在文本上用正则做如下变换：
```
v17 = b3 ? (v17 + 1) : v17;  →  if (b3) v17 = v17 + 1;
```

这本质上是一个 AST 变换，应该在 `StructuredStatement` 层面做。

**具体步骤**:
```
1. 在 src/structurizer.rs 的 pretty_print 阶段之前，
   新增一个 AST pass: simplify_predicated_ternaries_ast()

2. 遍历 StructuredStatement 树，对每个 BasicBlock 中的 IRStatement:
   - 如果 stmt.pred.is_some() 且 stmt.pred_old_defs 中的旧值 == 同一 component 的 dest
   - 则标记该 stmt 为 "emit as if-guard" 而不是 ternary

3. 这需要 name_map 来判断 "旧值和 dest 是否属于同一命名组件"，
   所以依赖阶段 A 的 build_name_map() 输出。
```

**验证**: 每个阶段完成后，`cargo test` 必须全部通过。golden 输出可能需要 `cargo run --example regen_goldens` 重新生成（如果输出格式有微调）。

---

### Issue 2.2: 局部变量声明推断在渲染后文本上做

**文件**: `src/bin/main.rs:49-138`

**现状**:
`infer_self_contained_locals()` 接受 `code_output: &str`（渲染后文本），用多个正则在文本中扫描变量名出现、赋值左侧、live-in 信息，最终生成 `uint32_t v0; bool b1;` 等声明。

**根因**: 在流水线最末端操作文本，而这些信息（哪些变量被定义、哪些是 live-in、哪些是 bool 类型）在 IR/SSA 中本来就有。

**修复方案**:

这个功能已经有一个 IR 级别的版本：`abi.rs` 中的 `infer_local_typed_declarations_with_abi()`（在 `main.rs:335-339` 中调用）。`infer_self_contained_locals()` 是一个补充层，捕捉前者遗漏的变量。

**具体步骤**:
```
1. 在 src/abi.rs 的 infer_local_typed_declarations_with_abi() 中扩展逻辑，
   使其能覆盖当前 infer_self_contained_locals() 检测的所有 case：
   - 使用 name_map（来自阶段 A）将 SSA 名映射到恢复后的名称
   - 从 FunctionIR 的 def/use 信息判断 live-in（而不是在文本上扫描）
   - 从 RegId.class 判断类型（P/UP → bool, R → uint32_t）

2. 一旦 IR 级别版本完全覆盖了文本级版本的功能，
   删除 infer_self_contained_locals()。

3. 如果无法一步到位，可先保留文本级版本作为 fallback，
   但在 IR 级别版本能处理的情况下优先使用 IR 版本。
```

**验证**: golden 输出应完全一致（声明内容和顺序不变）。`cargo test` 全部通过。

---

## 附加清理项

### Issue 3.1: `high_il.rs` 未被流水线使用

**文件**: `src/high_il.rs` (85 行)

**现状**: 定义了 `HExpr`/`HNode` 高层 IL，但没有任何代码使用它。`lib.rs` re-export 了它但无人消费。

**修复方案**: 两个选择：
- (a) 如果将来打算用它替代 `StructuredStatement`，保留但标注 `#[allow(dead_code)]` 并在 README 说明。
- (b) 如果不打算用，删除 `src/high_il.rs` 并从 `lib.rs` 移除相关 `pub mod` 和 `pub use`。

**建议选 (b)** 以减少维护负担。将来需要时可从 git 历史恢复。

---

## 执行顺序建议

```
优先级 1（低风险、高收益）:
  1.1 CFG fall-through BTreeMap        ← 3 行改动
  1.2 Structurizer block lookup index  ← 10 行改动
  1.4 loop_of() index                  ← 8 行改动
  3.1 删除 high_il.rs                  ← 3 行删除

优先级 2（中风险、高收益）:
  1.5 后支配树 virtual exit            ← 20 行改动，可能影响 golden

优先级 3（中等工作量、架构改进）:
  2.1-A NameAwareDisplay               ← 约 50 行新增
  1.3 DF 计算 clone 消除              ← 5 行改动

优先级 4（较大工作量、长期改进）:
  2.1-B ternary 简化 AST 化           ← 约 100 行新增/重构
  2.2 局部变量声明 IR 化              ← 约 80 行新增/重构
```

每个 issue 完成后立即运行 `cargo test`，确认 212 条测试全部通过。如果涉及输出变化，运行 `cargo run --example regen_goldens` 重新生成 golden 文件，并 review diff 确认变化合理。
