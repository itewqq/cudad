//!  当前进度：极简 SSA (RPO 遍历 + 简单 Φ 插入，可处理循环)

use crate::cfg::ControlFlowGraph;
use crate::parser::Operand;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{visit::DfsPostOrder, Direction};
use std::collections::{HashMap, HashSet};
use crate::cfg::EdgeKind;
use std::env;
use petgraph::algo::dominators::simple_fast;
use std::cell::RefCell;
use crate::debug_log;

/* ---------- 数据结构 ---------- */
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegType {
    BitWidth(u32),
    // 可扩展更多类型修饰符
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RegId {
    pub class: String,  // "R", "P", "UR"等
    pub idx: i32,
    pub sign: i32,      // +1/-1
    pub ssa: Option<usize>, // SSA版本号
}

impl RegId {
    pub fn new(class: &str, idx: i32, sign: i32) -> Self {
        RegId {
            class: class.to_string(),
            idx,
            sign,
            ssa: None,
        }
    }

    pub fn display(&self) -> String {
        let base = format!("{}{}", self.class, self.idx);
        let ssa = self.ssa.map(|v| format!(".{}", v)).unwrap_or_default();
        let sign = if self.sign < 0 { "-" } else { "" };
        format!("{}{}{}", sign, base, ssa)
    }

    pub fn with_ssa(&self, version: usize) -> Self {
        let mut new_reg = self.clone();
        new_reg.ssa = Some(version);
        new_reg
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IRCond {
    True,
    Pred { reg: RegId, sense: bool }, // 例如P0.3==1
}

#[derive(Clone, Debug, PartialEq)]
pub enum IRExpr {
    Reg(RegId),
    ImmI(i64),
    ImmF(f64),
    Mem { base: Box<IRExpr>, offset: Option<Box<IRExpr>>, width: Option<u32> },
    Op { op: String, args: Vec<IRExpr> },
}

impl IRExpr {
    pub fn display(&self) -> String {
        match self {
            IRExpr::Reg(r) => r.display(),
            IRExpr::ImmI(i) => format!("{}", i),
            IRExpr::ImmF(f) => format!("{}", f),
            IRExpr::Mem { base, offset, width } => {
                let off = offset.as_ref().map(|o| format!(", {}", o.display())).unwrap_or_default();
                let w = width.map(|w| format!(":{}", w)).unwrap_or_default();
                format!("Mem[{}{}{}]", base.display(), off, w)
            }
            IRExpr::Op { op, args } => {
                let args_str = args.iter().map(|a| a.display()).collect::<Vec<_>>().join(", ");
                format!("{}({})", op, args_str)
            }
        }
    }

    pub fn get_reg(&self) -> Option<&RegId> {
        match self {
            IRExpr::Reg(r) => Some(r),
            _ => None
        }
    }

    pub fn get_reg_mut(&mut self) -> Option<&mut RegId> {
        match self {
            IRExpr::Reg(r) => Some(r),
            _ => None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RValue {
    Op { opcode: String, args: Vec<IRExpr> },
    Phi(Vec<IRExpr>),
    ImmI(i64),
    ImmF(f64),
    Undef,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRStatement {
    pub dest: Option<IRExpr>,
    pub value: RValue,
    /// SSA变量，表示该语句的谓词条件（如@P0）
    pub pred: Option<IRExpr>,
    pub mem_addr_args: Option<Vec<IRExpr>>, // 新增：内存地址相关操作数
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRBlock {
    pub id: usize,
    pub start_addr: u32, // 当前块起始地址
    pub irdst: Vec<(Option<IRCond>, u32)>, // (条件, 目标地址)列表
    pub stmts: Vec<IRStatement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionIR {
    pub blocks: Vec<IRBlock>,
}

/* ---------- PhysReg helper ---------- */
fn phys_reg_of(op: &Operand) -> Option<RegId> {
    match op {
        Operand::Register { class, idx, sign, ty } => {
            if class == "PT" || class == "RZ" {
                Some(RegId::new(class, 0, *sign))
            } else if class.starts_with("P") {
                // P0-P6
                if (0..=6).contains(idx) {
                    Some(RegId::new("P", *idx, *sign))
                } else {
                    None
                }
            } else {
                Some(RegId::new(class, *idx, *sign))
            }
        }
        Operand::Uniform { idx } => Some(RegId::new("UR", *idx, 1)),
        _ => None,
    }
}

fn is_mem_store(opcode: &str) -> bool {
    opcode.starts_with("STS") || opcode.starts_with("STG") || opcode.starts_with("ST")
}

fn is_mem_load(opcode: &str) -> bool {
    opcode.starts_with("LDG") || opcode.starts_with("LD")
}

/* ---------- 顶层工具函数 ---------- */
// 递归替换IRExpr中的寄存器为最新SSA版本
fn rename_expr(expr: &IRExpr, stacks: &HashMap<RegId, Vec<IRExpr>>) -> IRExpr {
    match expr {
        IRExpr::Reg(reg) => {
            if let Some(stack) = stacks.get(reg) {
                if let Some(top) = stack.last() {
                    return top.clone();
                }
            }
            expr.clone()
        },
        IRExpr::Mem { base, offset, width } => {
            let new_base = Box::new(rename_expr(base, stacks));
            let new_offset = offset.as_ref().map(|o| Box::new(rename_expr(o, stacks)));
            IRExpr::Mem { base: new_base, offset: new_offset, width: *width }
        },
        IRExpr::Op { op, args } => {
            IRExpr::Op { 
                op: op.clone(), 
                args: args.iter().map(|a| rename_expr(a, stacks)).collect() 
            }
        },
        _ => expr.clone(),
    }
}

fn regtype_from_parser(pty: &Option<crate::parser::RegType>) -> Option<RegType> {
    pty.as_ref().map(|t| match t {
        crate::parser::RegType::BitWidth(w) => RegType::BitWidth(*w),
    })
}

pub fn lower_operand(op: &crate::parser::Operand) -> IRExpr {
    match op {
        Operand::Register { .. } => {
            if let Some(reg) = phys_reg_of(op) {
                IRExpr::Reg(reg)
            } else {
                IRExpr::ImmI(0)
            }
        }
        Operand::Uniform { .. } => {
            if let Some(reg) = phys_reg_of(op) {
                IRExpr::Reg(reg)
            } else {
                IRExpr::ImmI(0)
            }
        }
        Operand::ImmediateI(i) => IRExpr::ImmI(*i),
        Operand::ImmediateF(f) => IRExpr::ImmF(*f),
        Operand::ConstMem { bank, offset } => {
            IRExpr::Op { 
                op: "ConstMem".to_string(), 
                args: vec![
                    IRExpr::ImmI(*bank as i64), 
                    IRExpr::ImmI(*offset as i64)
                ] 
            }
        }
        Operand::Raw(s) => {
            // 简单处理Raw字符串
            if let Ok(i) = s.parse::<i64>() {
                IRExpr::ImmI(i)
            } else if let Ok(f) = s.parse::<f64>() {
                IRExpr::ImmF(f)
            } else {
                IRExpr::Op { op: s.clone(), args: vec![] }
            }
        }
    }
}

/* ---------- SSA 主流程 ---------- */
pub fn build_ssa(cfg: &ControlFlowGraph) -> FunctionIR {
    use petgraph::graph::NodeIndex;
    use petgraph::Direction;
    use petgraph::algo::dominators::simple_fast;
    use std::collections::{HashMap, HashSet};

    let entry = NodeIndex::new(0);
    let doms = simple_fast(cfg, entry);
    
    // 1. 收集所有变量的定义点（A(V)）
    let mut defsites: HashMap<RegId, HashSet<NodeIndex>> = HashMap::new();
    let mut ir_blocks = HashMap::<usize, IRBlock>::new();
    for n in cfg.node_indices() {
        let bb = &cfg[n];
        let mut stmts = Vec::<IRStatement>::new();
        for ins in &bb.instrs {
            debug_log!("ins: {:?}", ins);
            let mut args = Vec::<IRExpr>::new();
            let mut mem_addr_args = None;
            if is_mem_load(&ins.opcode) {
                // 加载指令：第一个操作数为dest，其余为地址
                let mut addr_args = Vec::new();
                for (i, op) in ins.operands.iter().enumerate() {
                    if i == 0 {
                        // dest
                        if let Some(reg) = phys_reg_of(op) {
                            // dest 赋值在下方
                        }
                    } else {
                        // 地址相关操作数
                        addr_args.push(lower_operand(op));
                        args.push(lower_operand(op));
                    }
                }
                if !addr_args.is_empty() {
                    mem_addr_args = Some(addr_args);
                }
            } else if is_mem_store(&ins.opcode) {
                // 存储指令：第一个操作数为地址，其余为数据
                let mut addr_args = Vec::new();
                for (i, op) in ins.operands.iter().enumerate() {
                    if i == 0 {
                        // 地址相关
                        addr_args.push(lower_operand(op));
                        args.push(lower_operand(op));
                    } else {
                        // 数据相关
                        args.push(lower_operand(op));
                    }
                }
                if !addr_args.is_empty() {
                    mem_addr_args = Some(addr_args);
                }
            } else {
                // 普通指令
                for (i, op) in ins.operands.iter().enumerate() {
                    if i != 0 {
                        args.push(lower_operand(op));
                    }
                }
            }
            let pred_var = if let Some(pred) = &ins.pred {
                if pred.reg.starts_with("P") {
                    if let Ok(idx) = pred.reg[1..].parse::<i32>() {
                        Some(IRExpr::Reg(RegId::new("P", idx, 1)))
                    } else { 
                        None 
                    }
                } else { 
                    None 
                }
            } else { 
                None 
            };
            let mut dest = None;
            if is_mem_load(&ins.opcode) {
                if let Some(reg) = ins.operands.first().and_then(phys_reg_of) {
                    dest = Some(IRExpr::Reg(reg.clone()));
                    defsites.entry(reg).or_default().insert(n);
                }
            } else if !is_mem_store(&ins.opcode) {
                if let Some(reg) = ins.operands.first().and_then(phys_reg_of) {
                    dest = Some(IRExpr::Reg(reg.clone()));
                    defsites.entry(reg).or_default().insert(n);
                }
            }
            let rv = RValue::Op {
                opcode: ins.opcode.clone(),
                args,
            };
            stmts.push(IRStatement { dest, value: rv, pred: pred_var, mem_addr_args });
            debug_log!("stmt: {:?}", stmts.last().unwrap());
        }
        // 计算所有出边的条件跳转表
        let mut irdst = Vec::new();
        let this_idx = n;
        let mut cond_branch_info = None;
        if let Some(last) = bb.instrs.last() {
            if let Some(pred) = &last.pred {
                if pred.reg.starts_with("P") {
                    if let Ok(idx) = pred.reg[1..].parse::<i32>() {
                        cond_branch_info = Some((format!("P{}", idx), 0, pred.sense));
                    }
                }
            }
        }
        for edge in cfg.edges(this_idx) {
            let target_idx = edge.target();
            let target_addr = cfg[target_idx].start;
            match *edge.weight() {
                EdgeKind::CondBranch => {
                    if let Some((ref name, id, sense)) = cond_branch_info {
                        irdst.push((Some(IRCond::Pred { reg: RegId::new("P", name[1..].parse().unwrap_or(0), 1), sense }), target_addr));
                    } else {
                        irdst.push((Some(IRCond::True), target_addr));
                    }
                }
                EdgeKind::FallThrough => {
                    if let Some((ref name, id, sense)) = cond_branch_info {
                        irdst.push((Some(IRCond::Pred { reg: RegId::new("P", name[1..].parse().unwrap_or(0), 1), sense: !sense }), target_addr));
                    } else {
                        irdst.push((Some(IRCond::True), target_addr));
                    }
                }
                EdgeKind::UncondBranch => {
                    irdst.push((Some(IRCond::True), target_addr));
                }
            }
        }
        ir_blocks.insert(bb.id, IRBlock {
            id: bb.id,
            start_addr: bb.start,
            irdst,
            stmts,
        });
    }
    debug_log!("defsites: {:?}", defsites);

    // 2. 计算支配前沿 DF(X)
    let mut df = HashMap::<NodeIndex, HashSet<NodeIndex>>::new();
    for y in cfg.node_indices() {
        let preds: Vec<_> = cfg.neighbors_directed(y, Direction::Incoming).collect();
        if preds.len() < 2 { continue; }
        let idom_y = doms.immediate_dominator(y);
        for &p in &preds {
            let mut runner = p;
            while Some(runner) != idom_y {
                df.entry(runner).or_default().insert(y);
                if let Some(idom_r) = doms.immediate_dominator(runner) {
                    runner = idom_r;
                } else {
                    break;
                }
            }
        }
    }
    debug_log!("DF: {:?}", df);

    // 3. Cytron算法插入phi节点
    let mut phi_blocks: HashMap<RegId, HashSet<NodeIndex>> = HashMap::new();
    let mut iter_count = 0usize;
    let all_nodes: Vec<_> = cfg.node_indices().collect();
    for (reg, defset) in &defsites {
        // 只对 P0-P6、R0-R255、UR* 做SSA
        let is_ssa_reg = (reg.class == "P" && (0..=6).contains(&reg.idx)) ||
                         (reg.class == "R" && (0..=255).contains(&reg.idx)) ||
                         (reg.class == "UR");
        if !is_ssa_reg { continue; }
        iter_count += 1;
        let mut has_already: HashMap<NodeIndex, usize> = HashMap::new();
        let mut work: HashMap<NodeIndex, usize> = HashMap::new();
        for &x in &all_nodes {
            has_already.insert(x, 0);
            work.insert(x, 0);
        }
        let mut W: Vec<NodeIndex> = Vec::new();
        for &x in defset {
            work.insert(x, iter_count);
            if !W.contains(&x) { W.push(x); }
        }
        while !W.is_empty() {
            let x = W.pop().unwrap();
            if let Some(dfset) = df.get(&x) {
                for &y in dfset {
                    if has_already.get(&y).copied().unwrap_or(0) < iter_count {
                        phi_blocks.entry(reg.clone()).or_default().insert(y);
                        has_already.insert(y, iter_count);
                        if work.get(&y).copied().unwrap_or(0) < iter_count {
                            work.insert(y, iter_count);
                            if !W.contains(&y) { W.push(y); }
                        }
                    }
                }
            }
        }
    }
    debug_log!("phi_blocks: {:?}", phi_blocks);
    // 插入phi节点
    let mut phi_map: HashMap<(usize, usize), RegId> = HashMap::new();
    for (reg, blocks) in &phi_blocks {
        for &n in blocks {
            let preds: Vec<_> = cfg.neighbors_directed(n, Direction::Incoming).collect();
            if preds.len() > 1 {
                let block = ir_blocks.get_mut(&cfg[n].id).unwrap();
                let phi = IRStatement {
                    dest: Some(IRExpr::Reg(reg.clone())), // phi节点dest为变量本体（无SSA版本）
                    value: RValue::Phi(vec![]),
                    pred: None,
                    mem_addr_args: None,
                };
                block.stmts.insert(0, phi);
                phi_map.insert((block.id, 0), reg.clone());
                debug_log!("insert phi for {:?} at block {}", reg, block.id);
            }
        }
    }
    debug_log!("phi_map: {:?}", phi_map);

    // 4. Cytron算法变量重命名
    let mut domtree_children: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for n in cfg.node_indices() {
        if let Some(idom) = doms.immediate_dominator(n) {
            domtree_children.entry(idom).or_default().push(n);
        }
    }
    let mut stacks: HashMap<RegId, Vec<IRExpr>> = HashMap::new();
    let mut reg_version: HashMap<RegId, usize> = HashMap::new();
    fn new_ssa_var(reg: &RegId, reg_version: &mut HashMap<RegId, usize>, ty: Option<RegType>) -> IRExpr {
        if reg.class == "PT" || reg.class == "RZ" {
            IRExpr::Reg(reg.clone())
        } else {
            let v = reg_version.entry(reg.clone()).or_insert(0);
            let ssa_var = IRExpr::Reg(reg.with_ssa(*v));
            *v += 1;
            ssa_var
        }
    }
    #[derive(Debug)]
    struct RenameRecord {
        block_id: usize,
        phi_dest: Vec<(usize, IRExpr)>,
        phi_args: Vec<(usize, usize, IRExpr)>,
        normal_renames: Vec<(usize, IRExpr)>,
        op_arg_renames: Vec<(usize, usize, IRExpr)>,
        irdst_renames: Vec<(usize, IRCond)>,
        pred_renames: Vec<(usize, IRExpr)>,
        mem_addr_args_renames: Vec<(usize, usize, IRExpr)>, // 新增：mem_addr_args的SSA重命名
    }
    fn collect_rename(
        n: NodeIndex,
        cfg: &ControlFlowGraph,
        domtree_children: &HashMap<NodeIndex, Vec<NodeIndex>>,
        stacks: &mut HashMap<RegId, Vec<IRExpr>>,
        reg_version: &mut HashMap<RegId, usize>,
        phi_map: &HashMap<(usize, usize), RegId>,
        ir_blocks: &HashMap<usize, IRBlock>,
        out: &mut Vec<RenameRecord>,
    ) {
        let block = ir_blocks.get(&cfg[n].id).unwrap();
        let mut phi_dest = Vec::new();
        let mut normal_renames = Vec::new();
        let mut op_arg_renames = Vec::new();
        let mut irdst_renames = Vec::new();
        let mut phi_args = Vec::new();
        let mut pred_renames = Vec::new();
        let mut mem_addr_args_renames: Vec<(usize, usize, IRExpr)> = Vec::new();
        // 1. 处理Phi节点，分配新SSA版本，push栈，并记录新Var
        for (i, stmt) in block.stmts.iter().enumerate() {
            if let Some(dest) = &stmt.dest {
                if let RValue::Phi(_) = stmt.value {
                    if let Some(reg) = phi_map.get(&(block.id, i)) {
                        let v = new_ssa_var(reg, reg_version, None);
                        stacks.entry(reg.clone()).or_default().push(v.clone());
                        let v_debug = v.clone();
                        debug_log!("phi rename: block {} phi {} -> {:?}", block.id, i, v_debug);
                        phi_dest.push((i, v));
                    }
                }
            }
        }
        // 2. 重命名普通语句的参数
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            match &stmt.value {
                RValue::Op { args, .. } => {
                    for (arg_idx, arg) in args.iter().enumerate() {
                        if let IRExpr::Reg(reg) = arg {
                            let reg = reg.clone();
                            if let Some(stack) = stacks.get(&reg) {
                                if let Some(top) = stack.last() {
                                    op_arg_renames.push((stmt_idx, arg_idx, top.clone()));
                                    debug_log!("op arg rename: block {} stmt {} arg {} -> {:?}", block.id, stmt_idx, arg_idx, top);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            // 新增：重命名mem_addr_args中的SSA变量
            if let Some(mem_addr_args) = &stmt.mem_addr_args {
                for (arg_idx, arg) in mem_addr_args.iter().enumerate() {
                    if let IRExpr::Reg(reg) = arg {
                        let reg = reg.clone();
                        if let Some(stack) = stacks.get(&reg) {
                            if let Some(top) = stack.last() {
                                mem_addr_args_renames.push((stmt_idx, arg_idx, top.clone()));
                                debug_log!("mem_addr_args rename: block {} stmt {} arg {} -> {:?}", block.id, stmt_idx, arg_idx, top);
                            }
                        }
                    }
                }
            }
        }
        // 3. 普通赋值分配新SSA版本，push栈
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            if let Some(dest) = &stmt.dest {
                if !matches!(stmt.value, RValue::Phi(_)) {
                    let reg = dest.get_reg().unwrap().clone();
                    let v = new_ssa_var(&reg, reg_version, None);
                    stacks.entry(reg.clone()).or_default().push(v.clone());
                    let v_debug = v.clone();
                    debug_log!("assign rename: block {} stmt {} -> {:?}", block.id, stmt_idx, v_debug);
                    normal_renames.push((stmt_idx, v));
                }
            }
        }
        // 4. 收集phi填充信息
        let succs: Vec<_> = cfg.neighbors(n).collect();
        for &succ in &succs {
            let succ_block_id = cfg[succ].id;
            let phi_indices: Vec<usize> = phi_map.iter()
                .filter(|(&(bid, _), _)| bid == succ_block_id)
                .map(|(&(bid, idx), _)| idx)
                .collect();
            for &i in &phi_indices {
                if let Some(reg) = phi_map.get(&(succ_block_id, i)) {
                    if let Some(stack) = stacks.get(reg) {
                        if let Some(top) = stack.last() {
                            phi_args.push((succ_block_id, i, top.clone()));
                            debug_log!("phi param: succ block {} phi {} <- {:?}", succ_block_id, i, top);
                        }
                    }
                }
            }
        }
        // 5. IRDst条件SSA重命名
        let mut irdst_renames_local = Vec::new();
        for (i, (cond, _addr)) in block.irdst.iter().enumerate() {
            if let Some(IRCond::Pred { reg, sense }) = cond {
                // 只处理P寄存器
                if reg.class == "P" {
                    let reg = reg.clone();
                    if let Some(stack) = stacks.get(&reg) {
                        if let Some(top) = stack.last() {
                            // 修正：用parse_var_physreg和模式匹配
                            if let IRExpr::Reg(ref top_reg) = top {
                                let new_cond = IRCond::Pred { reg: top_reg.clone(), sense: *sense };
                                irdst_renames_local.push((i, new_cond.clone()));
                                debug_log!("irdst rename: block {} irdst {} -> {:?}", block.id, i, new_cond);
                            }
                        }
                    }
                }
            }
        }
        irdst_renames = irdst_renames_local;
        // 6. pred字段SSA重命名
        for (stmt_idx, stmt) in block.stmts.iter().enumerate() {
            if let Some(pred_var) = &stmt.pred {
                let reg = pred_var.get_reg().unwrap().clone();
                if let Some(stack) = stacks.get(&reg) {
                    if let Some(top) = stack.last() {
                        pred_renames.push((stmt_idx, top.clone()));
                        debug_log!("pred rename: block {} stmt {} -> {:?}", block.id, stmt_idx, top);
                    }
                }
            }
        }
        out.push(RenameRecord {
            block_id: block.id,
            phi_dest,
            phi_args,
            normal_renames,
            op_arg_renames,
            irdst_renames,
            pred_renames,
            mem_addr_args_renames,
        });
        // 7. 递归遍历支配树
        if let Some(children) = domtree_children.get(&n) {
            for &child in children {
                collect_rename(child, cfg, domtree_children, stacks, reg_version, phi_map, ir_blocks, out);
            }
        }
        // 8. pop本块分配的SSA版本
        for stmt in &block.stmts {
            if let Some(dest) = &stmt.dest {
                let reg = dest.get_reg().unwrap().clone();
                if let Some(stack) = stacks.get_mut(&reg) {
                    stack.pop();
                }
            }
        }
    }
    // 收集所有重命名和phi参数填充信息
    let mut rename_records = Vec::new();
    collect_rename(entry, cfg, &domtree_children, &mut stacks, &mut reg_version, &phi_map, &ir_blocks, &mut rename_records);
    debug_log!("rename_records: {:?}", rename_records);
    // 统一批量可变借用ir_blocks，应用所有重命名和phi参数填充
    for rec in &rename_records {
        if let Some(block) = ir_blocks.get_mut(&rec.block_id) {
            // phi dest重命名
            for (i, v) in &rec.phi_dest {
                if let Some(stmt) = block.stmts.get_mut(*i) {
                    stmt.dest = Some(v.clone());
                }
            }
            // 普通赋值重命名
            for (stmt_idx, v) in &rec.normal_renames {
                if let Some(stmt) = block.stmts.get_mut(*stmt_idx) {
                    stmt.dest = Some(v.clone());
                }
            }
            // 递归重命名所有参数（RValue::Op、RValue::Phi、mem_addr_args、pred）
            for stmt in &mut block.stmts {
                match &mut stmt.value {
                    RValue::Op { args, .. } => {
                        for arg in args.iter_mut() {
                            *arg = rename_expr(arg, &stacks);
                        }
                    },
                    RValue::Phi(args) => {
                        for arg in args.iter_mut() {
                            *arg = rename_expr(arg, &stacks);
                        }
                    },
                    _ => {}
                }
                if let Some(ref mut mem_addr_args) = stmt.mem_addr_args {
                    for arg in mem_addr_args.iter_mut() {
                        *arg = rename_expr(arg, &stacks);
                    }
                }
                if let Some(ref mut pred) = stmt.pred {
                    *pred = rename_expr(pred, &stacks);
                }
            }
            // IRDst条件SSA重命名
            for (i, new_cond) in &rec.irdst_renames {
                if let Some((cond, _addr)) = block.irdst.get_mut(*i) {
                    *cond = Some(new_cond.clone());
                }
            }
        }
        // phi参数填充
        for (succ_block_id, i, val) in &rec.phi_args {
            if let Some(succ_block) = ir_blocks.get_mut(succ_block_id) {
                if let Some(IRStatement { value: RValue::Phi(ref mut args), .. }) = succ_block.stmts.get_mut(*i) {
                    args.push(val.clone());
                }
            }
        }
    }
    // 清理空phi节点
    for block in ir_blocks.values_mut() {
        block.stmts.retain(|stmt| {
            if let RValue::Phi(ref args) = stmt.value {
                !args.is_empty()
            } else {
                true
            }
        });
    }
    debug_log!("final ir_blocks: {:?}", ir_blocks);
    let mut blocks: Vec<_> = ir_blocks.into_iter().map(|(_, b)| b).collect();
    blocks.sort_by_key(|b| b.id);
    FunctionIR { blocks }
}

/// Output the SSA IR as a Graphviz DOT graph for debugging
pub fn ssa_to_dot(cfg: &ControlFlowGraph, fir: &FunctionIR) -> String {
    use std::fmt::Write;
    let mut s = String::from("digraph SSA_IR {\n");
    // Use rectangle shape for nodes
    s.push_str("  node [shape=box];\n");
    // Map block id to node index
    let mut id2idx = std::collections::HashMap::new();
    for idx in cfg.node_indices() {
        let bb = &cfg[idx];
        id2idx.insert(bb.id, idx);
    }
    // Emit nodes with IR
    for block in &fir.blocks {
        let mut label = format!("BB{} | Start: 0x{:x}\\n\\n",
            block.id,
            block.start_addr,
        );

        for stmt in &block.stmts {
            // 跳过分支类指令（BRA/JMP/RET/EXIT等）和imm常量（如果是分支/跳转类）
            let is_branch = match &stmt.value {
                RValue::Op { opcode, .. } => {
                    let op = opcode.to_ascii_uppercase();
                    op == "BRA" || op == "JMP" || op == "JMPP" || op == "RET" || op == "EXIT"
                },
                _ => false,
            };
            // imm常量行只在不是分支/跳转类时输出
            let is_imm = matches!(&stmt.value, RValue::ImmI(_) | RValue::ImmF(_));
            if is_branch || (is_imm && stmt.pred.is_some()) {
                continue;
            }
            let dest = stmt.dest.as_ref().map(|v| {
                if let IRExpr::Reg(r) = v {
                    if r.class == "PT" || r.class == "RZ" { format!("{} = ", r.display()) } else { format!("{} = ", r.display()) }
                } else {
                    String::new()
                }
            }).unwrap_or_default();
            let val = match &stmt.value {
                RValue::Op { opcode, args } => {
                    let args_str = args.iter().map(|a| match a {
                        IRExpr::Reg(r) => {
                            if r.class == "PT" || r.class == "RZ" { r.display() } else { r.display() }
                        },
                        IRExpr::ImmI(i) => format!("imm {}", i),
                        IRExpr::ImmF(f) => format!("imm {}", f),
                        IRExpr::Mem { base, offset, width } => {
                            let base_str = base.display();
                            let offset_str = offset.as_ref().map(|o| o.display()).unwrap_or_default();
                            let width_str = width.map(|w| w.to_string()).unwrap_or_default();
                            format!("{} + {} + {}", base_str, offset_str, width_str)
                        },
                        IRExpr::Op { op, args } => {
                            let args_str = args.iter().map(|a| a.display()).collect::<Vec<_>>().join(", ");
                            format!("{}({})", op, args_str)
                        },
                    }).collect::<Vec<_>>().join(", ");
                    format!("{}({})", opcode, args_str)
                },
                RValue::Phi(vars) => {
                    let args_str = vars.iter().map(|v| v.display()).collect::<Vec<_>>().join(", ");
                    format!("phi({})", args_str)
                },
                RValue::ImmI(i) => format!("imm {}", i),
                RValue::ImmF(f) => format!("imm {}", f),
                RValue::Undef => "undef".to_string(),
            };
            let pred = stmt.pred.as_ref().map(|v| {
                if let IRExpr::Reg(r) = v {
                    if r.class == "PT" || r.class == "RZ" { format!(" [@{}]", r.display()) } else { format!(" [@{}]", r.display()) }
                } else {
                    String::new()
                }
            }).unwrap_or_default();
            // 新增：输出mem_addr_args
            let addr_str = if let Some(mem_addr_args) = &stmt.mem_addr_args {
                let s = mem_addr_args.iter().map(|a| match a {
                    IRExpr::Reg(r) => {
                        if r.class == "PT" || r.class == "RZ" { r.display() } else { r.display() }
                    },
                    IRExpr::ImmI(i) => format!("imm {}", i),
                    IRExpr::ImmF(f) => format!("imm {}", f),
                    IRExpr::Mem { base, offset, width } => {
                        let base_str = base.display();
                        let offset_str = offset.as_ref().map(|o| o.display()).unwrap_or_default();
                        let width_str = width.map(|w| w.to_string()).unwrap_or_default();
                        format!("{} + {} + {}", base_str, offset_str, width_str)
                    },
                    IRExpr::Op { op, args } => {
                        let args_str = args.iter().map(|a| a.display()).collect::<Vec<_>>().join(", ");
                        format!("{}({})", op, args_str)
                    },
                }).collect::<Vec<_>>().join(", ");
                format!(" Addr: [{}]", s)
            } else { String::new() };
            label.push_str(&format!("{}{}{}{}\\n", dest, val, pred, addr_str));
        }
        label.push_str("\\n");
        // 显示所有IRDst分支
        for (cond, addr) in &block.irdst {
            let cond_str = match cond {
                Some(IRCond::True) => "(uncond)".to_string(),
                Some(IRCond::Pred { reg, sense }) => {
                    if reg.class == "PT" || reg.class == "RZ" {
                        format!("({}{})", if *sense { "" } else { "!" }, reg.display())
                    } else {
                        format!("({}{})", if *sense { "" } else { "!" }, reg.display())
                    }
                },
                None => "(unknown)".to_string(),
            };
            label.push_str(&format!("IRDst: {} -> 0x{:x}\\n", cond_str, addr));
        }
        writeln!(s, "  {} [label=\"{}\"];", block.id, label.replace("\"", "\\\"")).unwrap();
    }
    // Emit edges
    for e in cfg.edge_indices() {
        let (sidx, didx) = cfg.edge_endpoints(e).unwrap();
        let sid = &cfg[sidx].id;
        let did = &cfg[didx].id;
        writeln!(s, "  {} -> {};", sid, did).unwrap();
    }
    s.push('}');
    s
}

// 工具函数：格式化Var为R4.2或R4.2@64格式（仅Mem时带@WIDTH）
fn format_var(v: &IRExpr) -> String {
    match v {
        IRExpr::Reg(r) => r.display(),
        IRExpr::ImmI(i) => format!("imm{}", i),
        IRExpr::ImmF(f) => format!("imm{}", f),
        IRExpr::Mem { base, offset, width } => {
            let base_str = match (&**base, width) {
                (IRExpr::Reg(r), Some(bits)) => format!("{}@{}", r.display(), bits),
                _ => format_var(base),
            };
            let offset_str = offset.as_ref().map(|o| format_var(o)).unwrap_or_default();
            let width_str = width.map(|w| w.to_string()).unwrap_or_default();
            format!("mem[{}+{}+{}]", base_str, offset_str, width_str)
        },
        IRExpr::Op { op, args } => {
            let args_str = args.iter().map(|a| format_var(a)).collect::<Vec<_>>().join(", ");
            format!("{}({})", op, args_str)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Instruction, Operand, PredicateUse};
    use crate::cfg::{BasicBlock, ControlFlowGraph, EdgeKind};
    use petgraph::graph::Graph;

    #[test]
    fn test_irdst_conditional() {
        // 构造一个简单的条件跳转CFG: BB0 --(P0=1)--> BB1, BB0 --(P0=0)--> BB2
        // BB0: @P0 BRA 0x20; (fallthrough 0x10)
        let bb0 = BasicBlock {
            id: 0,
            start: 0x0,
            instrs: vec![Instruction {
                addr: 0x0,
                pred: Some(PredicateUse { reg: "P0".to_string(), sense: true }),
                opcode: "BRA".to_string(),
                operands: vec![Operand::ImmediateI(0x20)],
                raw: "@P0 BRA 0x20;".to_string(),
            }],
        };
        let bb1 = BasicBlock { id: 1, start: 0x10, instrs: vec![] };
        let bb2 = BasicBlock { id: 2, start: 0x20, instrs: vec![] };
        let mut g: ControlFlowGraph = Graph::new();
        let n0 = g.add_node(bb0);
        let n1 = g.add_node(bb1);
        let n2 = g.add_node(bb2);
        g.add_edge(n0, n1, EdgeKind::FallThrough); // fallthrough
        g.add_edge(n0, n2, EdgeKind::CondBranch); // branch
        let fir = build_ssa(&g);
        // 检查irdst
        let block0 = &fir.blocks[0];
        assert_eq!(block0.irdst.len(), 2);
        // 必须有一个是条件跳转到0x20, 一个是无条件到0x10
        let mut found_cond = false;
        let mut found_uncond = false;
        for (cond, addr) in &block0.irdst {
            if let Some(IRCond::Pred { reg, sense, .. }) = cond {
                assert_eq!(reg.display(), "P0");
                if *sense {
                    assert_eq!(*addr, 0x20);
                    found_cond = true;
                } else {
                    assert_eq!(*addr, 0x10);
                    found_uncond = true;
                }
            }
        }
        assert!(found_cond && found_uncond);
    }
}

