//! ir.rs  –  SSA construction + DOT export
//! fully implements Cytron 91 algorithm (minimal Φ + rename)
//! no string cached inside nodes; all printing via DisplayCtx

use std::collections::{BTreeMap, BTreeSet, HashMap};
use crate::parser::Operand;
use crate::cfg::{ControlFlowGraph, EdgeKind};
use crate::op_semantics::derive_op_semantics;
use petgraph::{
    graph::NodeIndex,
    Direction,
};
use petgraph::visit::EdgeRef;

/* =======================================================================
   Section 0 – Printing context
======================================================================= */
/// An external formatter that decides how to show registers / expressions
pub trait DisplayCtx {
    fn reg(&self, r:&RegId)->String;
    fn expr(&self, e:&IRExpr)->String {
        match e {
            IRExpr::Reg(r) => self.reg(r),
            IRExpr::ImmI(i)=>format!("{}",i),
            IRExpr::ImmF(f)=>format!("{}",f),
            IRExpr::Mem{base,offset,width}=>{
                let s=if let Some(off)=offset{
                    format!("*({} + {})",self.expr(base),self.expr(off))
                } else {
                    format!("*{}",self.expr(base))
                };
                let _ = width;
                s
            }
            IRExpr::Op{op,args}=>{
                if op=="-" && args.len()==1 {
                    let inner = self.expr(&args[0]);
                    let simple = match &args[0] {
                        IRExpr::Reg(_) | IRExpr::ImmI(_) | IRExpr::ImmF(_) => true,
                        IRExpr::Op { op, .. } if op == "ConstMem" => true,
                        _ => false,
                    };
                    if simple {
                        return format!("-{}", inner);
                    }
                    return format!("-({})", inner);
                }
                let list=args.iter().map(|a|self.expr(a)).collect::<Vec<_>>().join(", ");
                format!("{}({})",op,list)
            }
        }
    }
}
/// Default formatter ⟶ 原先 display() 效果
pub struct DefaultDisplay;
impl DisplayCtx for DefaultDisplay {
    fn reg(&self, r:&RegId)->String { r.display() }
}

/* =======================================================================
   Section 1 – Core IR data structures  (unchanged except `DisplayCtx`)
======================================================================= */
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegType { BitWidth(u32) }

#[derive(Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct RegId {
    pub class:String, pub idx:i32, pub sign:i32, pub ssa:Option<usize>
}
impl RegId {
    pub fn new(class:&str, idx:i32, sign:i32)->Self{
        Self{class:class.into(), idx, sign, ssa:None}
    }
    pub fn with_ssa(&self, v:usize)->Self{
        let mut r=self.clone(); r.ssa=Some(v); r
    }
    pub fn display(&self)->String{
        let base = match self.class.as_str() {
            // Immutable pseudo-registers print without numeric suffix.
            "RZ" | "PT" | "URZ" | "UPT" => self.class.clone(),
            _ => format!("{}{}", self.class, self.idx),
        };
        let ssa=self.ssa.map(|v|format!(".{}",v)).unwrap_or_default();
        let sign=if self.sign<0{"-"}else{""};
        format!("{}{}{}",sign,base,ssa)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IRCond{
    True,
    Pred{reg:RegId, sense:bool},
}

#[derive(Clone, Debug, PartialEq)]
pub enum IRExpr{
    Reg(RegId),
    ImmI(i64),
    ImmF(f64),
    Mem{base:Box<IRExpr>,offset:Option<Box<IRExpr>>,width:Option<u32>},
    Op { op:String, args:Vec<IRExpr>},
}
impl IRExpr{
    pub fn get_reg(&self)->Option<&RegId>{ if let IRExpr::Reg(r)=self{Some(r)}else{None}}
    pub fn get_reg_mut(&mut self)->Option<&mut RegId>{ if let IRExpr::Reg(r)=self{Some(r)}else{None}}
}

#[derive(Clone, Debug, PartialEq)]
pub enum RValue{
    Op{opcode:String,args:Vec<IRExpr>},
    Phi(Vec<IRExpr>),
    ImmI(i64),
    ImmF(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRStatement{
    pub defs:Vec<IRExpr>,
    pub value:RValue,
    pub pred:Option<IRExpr>,
    pub mem_addr_args:Option<Vec<IRExpr>>,
    /// For predicated non-branch instructions: the *previous* SSA version of
    /// each def register (i.e. the value the register keeps when the predicate
    /// is false).  Populated during SSA renaming so the renderer can emit
    /// `dest = pred ? rhs : old;` instead of `if (pred) dest = rhs;`.
    pub pred_old_defs:Vec<IRExpr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IRBlock{
    pub id:usize,
    pub start_addr:u32,
    pub irdst:Vec<(Option<IRCond>,u32)>,
    pub stmts:Vec<IRStatement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionIR{
    pub blocks:Vec<IRBlock>,
}

/* =======================================================================
   Section 2 – Helpers: parser→IR lowering
======================================================================= */
fn phys_reg_of(op:&Operand)->Option<RegId>{
    match op{
        Operand::Register{class,idx,sign,..} => {
            match class.as_str() {
                "RZ" | "PT" | "URZ" | "UPT" => Some(RegId::new(class, 0, *sign)),
                "UP" | "P" => Some(RegId::new(class, *idx, *sign)),
                _ => Some(RegId::new(class, *idx, *sign)),
            }
        }
        Operand::Uniform{idx} => Some(RegId::new("UR",*idx,1)),
        _=>None
    }
}
fn lower_operand(op:&Operand)->IRExpr{
    match op{
        Operand::Register{..}|Operand::Uniform{..}=>{
            if let Some(r)=phys_reg_of(op){ IRExpr::Reg(r)} else{IRExpr::ImmI(0)}
        }
        Operand::ImmediateI(i)=>IRExpr::ImmI(*i),
        Operand::ImmediateF(f)=>IRExpr::ImmF(*f),
        Operand::ConstMem{bank,offset}=>{
            IRExpr::Op{op:"ConstMem".into(),args:vec![IRExpr::ImmI(*bank as i64),IRExpr::ImmI(*offset as i64)]}
        }
        Operand::MemRef { base, offset, width, .. } => {
            let mut base_expr = lower_operand(base.as_ref());
            if matches!(width, Some(64)) {
                if let Some(hi_expr) = infer_64bit_pair_hi_expr(base.as_ref()) {
                    base_expr = IRExpr::Op {
                        op: "addr64".into(),
                        args: vec![base_expr, hi_expr],
                    };
                }
            }
            let off_expr = offset.as_ref().map(|v| Box::new(IRExpr::ImmI(*v)));
            IRExpr::Mem {
                base: Box::new(base_expr),
                offset: off_expr,
                width: *width,
            }
        }
        Operand::Raw(s)=>{
            if let Some(pred) = parse_raw_predicate_token(s) {
                return pred;
            }
            if let Some((neg, bank, off)) = parse_raw_constmem_token(s) {
                let cm = IRExpr::Op {
                    op: "ConstMem".into(),
                    args: vec![IRExpr::ImmI(bank as i64), IRExpr::ImmI(off as i64)],
                };
                if neg {
                    return IRExpr::Op {
                        op: "-".into(),
                        args: vec![cm],
                    };
                }
                return cm;
            }
            if let Ok(i)=s.parse::<i64>() { IRExpr::ImmI(i)}
            else if let Ok(f)=s.parse::<f64>(){ IRExpr::ImmF(f)}
            else{ IRExpr::Op{op:s.clone(),args:vec![]}}
        }
    }
}

fn infer_64bit_pair_hi_expr(base: &Operand) -> Option<IRExpr> {
    match base {
        Operand::Register { class, idx, .. } if class == "R" || class == "UR" => {
            Some(IRExpr::Reg(RegId::new(class, idx + 1, 1)))
        }
        Operand::Uniform { idx } => Some(IRExpr::Reg(RegId::new("UR", idx + 1, 1))),
        _ => None,
    }
}

fn parse_raw_constmem_token(s: &str) -> Option<(bool, u32, u32)> {
    let t = s.trim();
    let (neg, core) = if let Some(rest) = t.strip_prefix('-') {
        (true, rest)
    } else {
        (false, t)
    };
    if !(core.starts_with("c[") && core.ends_with(']')) {
        return None;
    }
    let mut parts = core.split('[');
    let head = parts.next()?;
    if head != "c" {
        return None;
    }
    let bank_part = parts.next()?.strip_suffix(']')?;
    let off_part = parts.next()?.strip_suffix(']')?;
    if parts.next().is_some() {
        return None;
    }
    if !bank_part.starts_with("0x") || !off_part.starts_with("0x") {
        return None;
    }
    let bank = u32::from_str_radix(&bank_part[2..], 16).ok()?;
    let off = u32::from_str_radix(&off_part[2..], 16).ok()?;
    Some((neg, bank, off))
}

fn parse_raw_predicate_token(s: &str) -> Option<IRExpr> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let (neg, core) = if let Some(rest) = t.strip_prefix('!') {
        (true, rest)
    } else {
        (false, t)
    };

    let base = if let Some(num) = core.strip_prefix('P') {
        let idx = num.parse::<i32>().ok()?;
        IRExpr::Reg(RegId::new("P", idx, 1))
    } else if let Some(num) = core.strip_prefix("UP") {
        let idx = num.parse::<i32>().ok()?;
        IRExpr::Reg(RegId::new("UP", idx, 1))
    } else {
        return None;
    };

    if neg {
        Some(IRExpr::Op {
            op: "!".into(),
            args: vec![base],
        })
    } else {
        Some(base)
    }
}
/* mem load/store heuristics */
fn is_mem_load(op:&str)->bool{op.starts_with("LD")}
fn is_mem_store(op:&str)->bool{op.starts_with("ST")}

/* =======================================================================
   Section 3 – Build SSA (Cytron algorithm)
======================================================================= */

/// 构建最小 Φ + 重命名后的 SSA IR
pub fn build_ssa(cfg: &ControlFlowGraph) -> FunctionIR {
    use petgraph::algo::dominators::simple_fast;
    use petgraph::graph::NodeIndex;
    use petgraph::Direction;
    use std::collections::{HashMap, HashSet};

    if cfg.node_count() == 0 {
        return FunctionIR { blocks: Vec::new() };
    }

    /* ---------- helpers ---------- */
    fn base_reg(r: &RegId) -> RegId {
        // SSA identity is register family + index only.
        // Unary sign is an expression-level property and must not fork SSA chains.
        let mut out = r.clone();
        out.ssa = None;
        out.sign = 1;
        out
    }
    fn is_immutable_reg(r: &RegId) -> bool {
        matches!(r.class.as_str(), "RZ" | "PT" | "URZ" | "UPT")
    }
    fn new_ssa(r: &RegId, cnt: &mut HashMap<RegId, usize>) -> RegId {
        let key = base_reg(r);
        let v = cnt.entry(key.clone()).or_insert(0);
        let out = key.with_ssa(*v);
        *v += 1;
        out
    }
    fn top_or_new<'a>(
        key: &RegId,
        stack: &'a mut HashMap<RegId, Vec<RegId>>,
        cnt: &mut HashMap<RegId, usize>,
    ) -> &'a RegId {
        let slot = stack.entry(key.clone()).or_default();
        if slot.is_empty() {
            let tmp = key.with_ssa(*cnt.entry(key.clone()).or_insert(0));
            *cnt.get_mut(key).unwrap() += 1;
            slot.push(tmp);
        }
        slot.last().unwrap()
    }
    fn find_entry_node(cfg: &ControlFlowGraph) -> Option<NodeIndex> {
        let mut candidates = cfg
            .node_indices()
            .filter(|&n| cfg.neighbors_directed(n, Direction::Incoming).count() == 0)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return cfg.node_indices().next();
        }
        candidates.sort_by_key(|&n| cfg[n].id);
        candidates.first().copied()
    }
    fn rename_expr(
        e: &mut IRExpr,
        stack: &mut HashMap<RegId, Vec<RegId>>,
        cnt: &mut HashMap<RegId, usize>,
    ) {
        match e {
            IRExpr::Reg(r) => {
                if is_immutable_reg(r) {
                    return;
                }
                let use_sign = r.sign;
                let mut top = top_or_new(&base_reg(r), stack, cnt).clone();
                top.sign = use_sign;
                *r = top;
            }
            IRExpr::Mem { base, offset, .. } => {
                rename_expr(base, stack, cnt);
                if let Some(off) = offset {
                    rename_expr(off, stack, cnt);
                }
            }
            IRExpr::Op { args, .. } => {
                for a in args {
                    rename_expr(a, stack, cnt);
                }
            }
            _ => {}
        }
    }
    fn parse_pred_reg_name(name: &str) -> Option<RegId> {
        if let Some(num) = name.strip_prefix("UP") {
            let idx = num.parse::<i32>().ok()?;
            return Some(RegId::new("UP", idx, 1));
        }
        if let Some(num) = name.strip_prefix('P') {
            let idx = num.parse::<i32>().ok()?;
            return Some(RegId::new("P", idx, 1));
        }
        None
    }

    /* ---------- step-0: build IR blocks & defsites ---------- */
    let mut ir_blocks = HashMap::<usize, IRBlock>::new();
    let mut defsites=BTreeMap::<RegId,BTreeSet<NodeIndex>>::new();

    for n in cfg.node_indices() {
        let bb = &cfg[n];
        let mut stmts = Vec::<IRStatement>::new();

        for ins in &bb.instrs {
            let mut defs = Vec::<IRExpr>::new();
            let mut args = Vec::<IRExpr>::new();
            let mut mem_addr_args = None::<Vec<IRExpr>>;
            let sem = derive_op_semantics(
                &ins.opcode,
                &ins.operands,
                is_mem_load(&ins.opcode),
                is_mem_store(&ins.opcode),
            );

            for idx in &sem.def_operand_indices {
                if let Some(r) = ins.operands.get(*idx).and_then(phys_reg_of) {
                    defs.push(IRExpr::Reg(r.clone()));
                    if !is_immutable_reg(&r) {
                        defsites.entry(base_reg(&r)).or_default().insert(n);
                    }
                }
            }
            // ULDC/LDC .64 define a register pair (lo in dst, hi in dst+1).
            // Model the second def explicitly so high halves are not treated as live-ins.
            let mnem = ins.opcode.split('.').next().unwrap_or(ins.opcode.as_str());
            let is_64 = ins.opcode.split('.').skip(1).any(|t| t == "64");
            if is_64 && matches!(mnem, "ULDC" | "LDC") {
                if let Some(mut hi) = ins.operands.first().and_then(phys_reg_of) {
                    hi.idx += 1;
                    hi.sign = 1;
                    defs.push(IRExpr::Reg(hi.clone()));
                    if !is_immutable_reg(&hi) {
                        defsites.entry(base_reg(&hi)).or_default().insert(n);
                    }
                }
            }
            for idx in &sem.use_operand_indices {
                if let Some(op) = ins.operands.get(*idx) {
                    args.push(lower_operand(op));
                }
            }

            if is_mem_load(&ins.opcode) {
                let mut a = Vec::new();
                for op in ins.operands.iter().skip(1) {
                    a.push(lower_operand(op));
                }
                mem_addr_args = Some(a);
            } else if is_mem_store(&ins.opcode) {
                let mut a = Vec::new();
                if let Some(op0) = ins.operands.first() {
                    a.push(lower_operand(op0));
                }
                mem_addr_args = Some(a);
            }

            let pred_expr = ins.pred.as_ref().and_then(|p| {
                let base = IRExpr::Reg(parse_pred_reg_name(&p.reg)?);
                if p.sense {
                    Some(base)
                } else {
                    Some(IRExpr::Op {
                        op: "!".into(),
                        args: vec![base],
                    })
                }
            });

            stmts.push(IRStatement {
                defs,
                value: RValue::Op {
                    opcode: ins.opcode.clone(),
                    args,
                },
                pred: pred_expr,
                mem_addr_args,
                pred_old_defs: vec![],
            });
        }

        /* IRDst */
        let mut irdst = Vec::<(Option<IRCond>, u32)>::new();
        let last_pred = bb.instrs.last().and_then(|i| i.pred.as_ref());
        for e in cfg.edges(n) {
            let tgt_addr = cfg[e.target()].start;
            match *e.weight() {
                EdgeKind::CondBranch => {
                    if let Some(p) = last_pred {
                        if let Some(pred_reg) = parse_pred_reg_name(&p.reg) {
                            irdst.push((
                                Some(IRCond::Pred {
                                    reg: pred_reg,
                                    sense: p.sense,
                                }),
                                tgt_addr,
                            ));
                        }
                    }
                }
                EdgeKind::FallThrough => {
                    if let Some(p) = last_pred {
                        if let Some(pred_reg) = parse_pred_reg_name(&p.reg) {
                            irdst.push((
                                Some(IRCond::Pred {
                                    reg: pred_reg,
                                    sense: !p.sense,
                                }),
                                tgt_addr,
                            ));
                        }
                    } else {
                        irdst.push((Some(IRCond::True), tgt_addr));
                    }
                }
                EdgeKind::UncondBranch => irdst.push((Some(IRCond::True), tgt_addr)),
            }
        }

        ir_blocks.insert(
            bb.id,
            IRBlock {
                id: bb.id,
                start_addr: bb.start,
                irdst,
                stmts,
            },
        );
    }

    /* ---------- step-1: DomTree + DF ---------- */
    let entry = find_entry_node(cfg).unwrap_or_else(|| NodeIndex::new(0));
    let doms = simple_fast(cfg, entry);
    let mut idom = BTreeMap::<NodeIndex, NodeIndex>::new();
    for n in cfg.node_indices() {
        if let Some(i) = doms.immediate_dominator(n) {
            idom.insert(n, i);
        }
    }
    let df = compute_df(cfg, &idom, entry);

    /* ---------- step-2: Φ placement ---------- */
    let mut phi_needed = BTreeSet::<(NodeIndex, RegId)>::new();
    for (reg, defs) in &defsites {
        let mut work: Vec<_> = defs.iter().copied().collect();
        let mut seen = HashSet::<NodeIndex>::new();
        while let Some(x) = work.pop() {
            for &y in df.get(&x).unwrap_or(&BTreeSet::new()) {
                if seen.insert(y) {
                    phi_needed.insert((y, reg.clone()));
                    if !defs.contains(&y) {
                        work.push(y);
                    }
                }
            }
        }
    }

    /* 在块头插入 Φ，占位向量长度 = succ.in_edges() 长度 */
    for (blk_node, reg) in &phi_needed {
        let block_id = cfg[*blk_node].id;
        let preds: Vec<_> = cfg.neighbors_directed(*blk_node, Direction::Incoming).collect();
        let placeholder = vec![IRExpr::ImmI(0); preds.len()];
        ir_blocks
            .get_mut(&block_id)
            .unwrap()
            .stmts
            .insert(
                0,
                IRStatement {
                    defs: vec![IRExpr::Reg(reg.clone())],
                    value: RValue::Phi(placeholder),
                    pred: None,
                    mem_addr_args: None,
                    pred_old_defs: vec![],
                },
            );
    }

    /* ---------- step-3: Rename (Cytron WHICH-PRED) ---------- */
    // dom children
    let mut children = BTreeMap::<NodeIndex, Vec<NodeIndex>>::new();
    for (&b, &p) in &idom {
        children.entry(p).or_default().push(b);
    }

    let mut stack = HashMap::<RegId, Vec<RegId>>::new();
    let mut counter = HashMap::<RegId, usize>::new();

    fn rename(
        n: NodeIndex,
        cfg: &ControlFlowGraph,
        children: &BTreeMap<NodeIndex, Vec<NodeIndex>>,
        ir_blocks: &mut HashMap<usize, IRBlock>,
        stack: &mut HashMap<RegId, Vec<RegId>>,
        counter: &mut HashMap<RegId, usize>,
    ) {
        let bid = cfg[n].id;

        /* 1. 处理当前块 */
        {
            let blk = ir_blocks.get_mut(&bid).unwrap();

            /* φ 左值 */
            for stmt in blk
                .stmts
                .iter_mut()
                .filter(|s| matches!(s.value, RValue::Phi(_)))
            {
                let key = base_reg(stmt.defs.first().unwrap().get_reg().unwrap());
                let new = IRExpr::Reg(new_ssa(&key, counter));
                stack.entry(key.clone()).or_default().push(new.get_reg().unwrap().clone());
                stmt.defs = vec![new];
            }

            /* 普通语句 */
            for stmt in &mut blk.stmts {
                match &mut stmt.value {
                    RValue::Op { args, .. } => {
                        for a in args {
                            rename_expr(a, stack, counter);
                        }
                    }
                    RValue::Phi(_)
                    | RValue::ImmI(_)
                    | RValue::ImmF(_) => {}
                }
                if let Some(ma) = &mut stmt.mem_addr_args {
                    for a in ma {
                        rename_expr(a, stack, counter);
                    }
                }
                if let Some(p) = &mut stmt.pred {
                    rename_expr(p, stack, counter);
                }

                if !matches!(stmt.value, RValue::Phi(_)) {
                    let is_predicated = stmt.pred.is_some();
                    for def in &mut stmt.defs {
                        let cur = def.get_reg().unwrap().clone();
                        if is_immutable_reg(&cur) {
                            continue;
                        }
                        let k = base_reg(&cur);
                        // For predicated instructions, record the *previous*
                        // SSA version so the renderer can emit a conditional
                        // select instead of an unconditional assignment.
                        if is_predicated {
                            let old = top_or_new(&k, stack, counter).clone();
                            stmt.pred_old_defs.push(IRExpr::Reg(old));
                        }
                        let new = IRExpr::Reg(new_ssa(&k, counter));
                        stack
                            .entry(k.clone())
                            .or_default()
                            .push(new.get_reg().unwrap().clone());
                        *def = new;
                    }
                }
            }

            /* IRDst 条件 */
            for (cond, _) in &mut blk.irdst {
                if let Some(IRCond::Pred { reg, .. }) = cond {
                    *reg = top_or_new(&base_reg(reg), stack, counter).clone();
                }
            }
        } // blk borrow drop

        /* 2. 为后继块按 WhichPred 填充 Φ 参数 */
        for succ in cfg.neighbors_directed(n, Direction::Outgoing) {
            let succ_id = cfg[succ].id;

            // 获取 succ 的前驱列表（固定顺序）
            let preds: Vec<_> = cfg.neighbors_directed(succ, Direction::Incoming).collect();
            let idx_in_succ = preds
                .iter()
                .position(|&p| p == n)
                .expect("predecessor not found");

            let blk_succ = ir_blocks.get_mut(&succ_id).unwrap();
            for stmt in blk_succ
                .stmts
                .iter_mut()
                .filter(|s| matches!(s.value, RValue::Phi(_)))
            {
                let key = base_reg(stmt.defs.first().unwrap().get_reg().unwrap());
                let src = top_or_new(&key, stack, counter).clone();
                if let RValue::Phi(ref mut vec) = stmt.value {
                    vec[idx_in_succ] = IRExpr::Reg(src);
                }
            }
        }

        /* 3. 递归 */
        if let Some(chs) = children.get(&n) {
            for &c in chs {
                rename(c, cfg, children, ir_blocks, stack, counter);
            }
        }

        /* 4. pop */
        {
            let blk = ir_blocks.get(&bid).unwrap();
            for stmt in &blk.stmts {
                for def in &stmt.defs {
                    let d = def.get_reg().unwrap();
                    if is_immutable_reg(d) {
                        continue;
                    }
                    let k = base_reg(d);
                    if let Some(v) = stack.get_mut(&k) {
                        v.pop();
                    }
                }
            }
        }
    }

    rename(
        entry,
        cfg,
        &children,
        &mut ir_blocks,
        &mut stack,
        &mut counter,
    );

    /* ---------- collect ---------- */
    let mut blocks: Vec<_> = ir_blocks.into_iter().map(|(_, b)| b).collect();
    blocks.sort_by_key(|b| b.id);
    FunctionIR { blocks }
}

/* ==================== DF helper ==================== */
fn compute_df(cfg:&ControlFlowGraph,idom:&BTreeMap<NodeIndex,NodeIndex>, root: NodeIndex)
->HashMap<NodeIndex,BTreeSet<NodeIndex>>
{
    let mut local=HashMap::<NodeIndex,BTreeSet<NodeIndex>>::new();
    for n in cfg.node_indices(){
        for succ in cfg.neighbors_directed(n,Direction::Outgoing){
            if idom.get(&succ).copied()!=Some(n){
                local.entry(n).or_default().insert(succ);
            }
        }
    }
    let mut children:BTreeMap<NodeIndex,Vec<NodeIndex>>=BTreeMap::new();
    for (&b,&p) in idom{ children.entry(p).or_default().push(b); }

    fn up(n:NodeIndex,child:&BTreeMap<NodeIndex,Vec<NodeIndex>>,
          df:&mut HashMap<NodeIndex,BTreeSet<NodeIndex>>,idom:&BTreeMap<NodeIndex,NodeIndex>)
    {
        if let Some(ch)=child.get(&n){ for &c in ch{ up(c,child,df,idom); } }
        for &c in child.get(&n).unwrap_or(&Vec::new()){
            let set=df.entry(c).or_default().clone();
            for w in set{
                if idom.get(&w).copied()!=Some(n){
                    df.entry(n).or_default().insert(w);
                }
            }
        }
    }
    let mut df=local;
    up(root,&children,&mut df,idom);
    df
}

/* =======================================================================
   Section 4 – DOT Debugging
======================================================================= */
impl FunctionIR {
    pub fn to_dot(&self, cfg: &ControlFlowGraph, ctx: &dyn DisplayCtx) -> String {
        use std::fmt::Write;

        /// 将 IRCond 显示成 “(P0.3)” / “(!P1.7)” / “(uncond)”
        fn cond_str(c: &Option<IRCond>, ctx:&dyn DisplayCtx) -> String {
            match c {
                Some(IRCond::True) | None => "(uncond)".into(),
                Some(IRCond::Pred { reg, sense }) => {
                    let s = ctx.reg(reg);
                    if *sense { format!("({})", s) } else { format!("(!{})", s) }
                }
            }
        }

        let mut dot = String::from("digraph SSA {\n  node[shape=box];\n");

        /* ----------- 节点 ----------- */
        for b in &self.blocks {
            let mut label = format!("BB{} | Start: 0x{:x}\\l", b.id, b.start_addr);

            /* 语句 */
            for stmt in &b.stmts {
                let line = match &stmt.value {
                    RValue::Op { opcode, args } => {
                        let dst = if stmt.defs.is_empty() {
                            "_".to_string()
                        } else if stmt.defs.len() == 1 {
                            ctx.expr(&stmt.defs[0])
                        } else {
                            let list = stmt
                                .defs
                                .iter()
                                .map(|d| ctx.expr(d))
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("({})", list)
                        };
                        let a = args
                            .iter()
                            .map(|e| ctx.expr(e))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{} = {}({})", dst, opcode, a)
                    }
                    RValue::Phi(vars) => {
                        let dst = ctx.expr(stmt.defs.first().unwrap());
                        let list = vars
                            .iter()
                            .map(|v| ctx.expr(v))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{} = phi({})", dst, list)
                    }
                    RValue::ImmI(i) => {
                        format!("{} = {}", ctx.expr(stmt.defs.first().unwrap()), i)
                    }
                    RValue::ImmF(f) => {
                        format!("{} = {}", ctx.expr(stmt.defs.first().unwrap()), f)
                    }
                };
                label.push_str(&line);
                label.push_str("\\l");
            }

            /* IRDst 列表 */
            label.push_str("\\l");
            for (cond, addr) in &b.irdst {
                let cstr = cond_str(cond, ctx);
                label.push_str(&format!("IRDst: {} -> 0x{:x}\\l", cstr, addr));
            }

            // 写入节点
            writeln!(
                dot,
                "  {} [label=\"{}\"];",
                b.id,
                label.replace('\"', "\\\"")
            )
            .unwrap();
        }

        /* ----------- 边 ----------- */
        for e in cfg.edge_references() {
            let (sid, did) = (cfg[e.source()].id, cfg[e.target()].id);
            writeln!(dot, "  {} -> {};", sid, did).unwrap();
        }
        dot.push_str("}");
        dot
    }
}
