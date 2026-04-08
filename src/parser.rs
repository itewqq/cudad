//! 极简 SASS 行解析（面向 SM_89 / cuobjdump 输出）
//! 目标：把形如
//! /*0000*/  @!P0 IMAD.MOV.U32 R1, RZ, c[0x0][0x28] ; /* machine-code */
//! 解析为 `Instruction` 结构体。
//! * 解析地址 / 谓词 / 操作码 / 操作数。
//! * Operand 细分：寄存器 / 常量内存 / 立即数 / 原始字符串。

use nom::{
    bytes::complete::{tag, take_until, take_while},
    character::complete::multispace0,
    combinator::{map, opt},
    sequence::{delimited, tuple},
    IResult,
};

use lazy_static::lazy_static;
use regex::Regex;

/* --------------------------------- 数据结构 -------------------------------- */

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateUse {
    pub reg: String, // P0 / P1
    pub sense: bool, // true = @P0, false = @!P0
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegType {
    BitWidth(u32),
    // 可扩展更多类型修饰符
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register { class: String, idx: i32, sign: i32, ty: Option<RegType> },      // R0 / RZ / R255 / -R2 / R4.64
    Uniform  { idx: i32 },                      // UR*
    ImmediateI(i64),                            // 整数立即数
    ImmediateF(f64),                            // 浮点立即数
    ConstMem { bank: u32, offset: u32 },        // c[bank][off]
    MemRef {                                     // [R4.64+0x20]
        base: Box<Operand>,
        offset: Option<i64>,
        width: Option<u32>,
        raw: String,
    },
    Raw(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub addr: u32,
    pub pred: Option<PredicateUse>,
    pub opcode: String,
    pub operands: Vec<Operand>,
    pub raw: String,
}

/* --------------------------------- 正则预编译 -------------------------------- */

lazy_static! {
    static ref RE_REG_NUM: Regex = Regex::new(r"^(?P<cls>UP|UR|R|P)(?P<idx>\d+)$").unwrap();
    static ref RE_CONST : Regex = Regex::new(r"^c\[(?P<bank>0x[0-9A-Fa-f]+)\]\[(?P<off>0x[0-9A-Fa-f]+)\]$").unwrap();
    // Blackwell descriptor-qualified memory: `desc[URx][...]`.  The inner
    // bracket is restricted to a single non-nested group to prevent the
    // greedy quantifier from crossing bracket boundaries.  Only `UR`
    // descriptor registers are known to appear in SM 100+ SASS today.
    static ref RE_DESC  : Regex = Regex::new(r"^desc\[UR(?P<desc>\d+)\](?P<mem>\[[^\[\]]+\])$").unwrap();
    static ref RE_HEX_I : Regex = Regex::new(r"^-?0x[0-9A-Fa-f]+$").unwrap();
    static ref RE_DEC_I : Regex = Regex::new(r"^-?\d+$").unwrap();
    static ref RE_FLOAT : Regex = Regex::new(r"^-?\d+\.\d+(e[+-]?\d+)?$").unwrap();
    static ref RE_HEADER_SM: Regex = Regex::new(r"EF_CUDA_SM(?P<sm>\d+)").unwrap();
    static ref RE_TARGET_SM: Regex = Regex::new(r"\bsm_(?P<sm>\d+)\b").unwrap();
}

/* --------------------------------- 基础解析器 -------------------------------- */

/// 十六进制数字 0xaabb
fn parse_hex(s: &str) -> Option<i64> {
    let s = s.trim();
    
    // Check if string starts with "0x" or "-0x"
    let (is_negative, num_str) = if s.starts_with("-0x") {
        (true, &s[3..])
    } else if s.starts_with("0x") {
        (false, &s[2..])
    } else {
        return None;
    };

    // Parse hex string to i64
    i64::from_str_radix(num_str, 16)
        .ok()
        .map(|num| if is_negative { -num } else { num })
}

/// 十六进制地址 /*0040*/
fn parse_addr(input: &str) -> IResult<&str, u32> {
    map(
        delimited(tag("/*"), take_while(|c: char| c.is_ascii_hexdigit()), tag("*/")),
        |hex: &str| u32::from_str_radix(hex, 16).unwrap(),
    )(input)
}

/// @P0 / @!P1 前缀
fn parse_pred(input: &str) -> IResult<&str, PredicateUse> {
    map(
        tuple((tag("@"), opt(tag("!")), take_while(|c: char| c.is_ascii_alphanumeric()))),
        |(_, neg, reg): (_, Option<&str>, &str)| PredicateUse { reg: reg.to_owned(), sense: neg.is_none() },
    )(input)
}

fn is_opcode_char(c: char) -> bool { !(c.is_whitespace() || c == ';') }
fn parse_opcode(input: &str) -> IResult<&str, &str> { take_while(is_opcode_char)(input) }

fn parse_int_literal(s: &str) -> Option<i64> {
    if RE_HEX_I.is_match(s) {
        return parse_hex(s);
    }
    if RE_DEC_I.is_match(s) {
        return s.parse::<i64>().ok();
    }
    None
}

fn parse_register_operand(tok: &str) -> Option<Operand> {
    let mut sign = 1;
    let mut core = tok.trim();
    if let Some(rest) = core.strip_prefix('-') {
        sign = -1;
        core = rest;
    }

    // Treat predicate negation token like "!UPT" as raw expression.
    if core.starts_with('!') {
        return None;
    }

    let mut ty = None;
    let mut parts = core.split('.');
    let base = parts.next()?.to_ascii_uppercase();
    for suffix in parts {
        let s = suffix.to_ascii_lowercase();
        if let Ok(bits) = s.parse::<u32>() {
            ty = Some(RegType::BitWidth(bits));
        }
        // ".reuse", ".x8" and other suffixes are syntax decorations for now.
    }

    let (class, idx) = match base.as_str() {
        "RZ" => ("RZ".to_string(), 0),
        "PT" => ("PT".to_string(), 0),
        "URZ" => ("URZ".to_string(), 0),
        "UPT" => ("UPT".to_string(), 0),
        _ => {
            if let Some(cap) = RE_REG_NUM.captures(base.as_str()) {
                let class = cap["cls"].to_string();
                let idx = cap["idx"].parse::<i32>().ok()?;
                (class, idx)
            } else {
                return None;
            }
        }
    };

    Some(Operand::Register { class, idx, sign, ty })
}

fn parse_mem_ref_operand(tok: &str) -> Option<Operand> {
    let t = tok.trim();
    if !(t.starts_with('[') && t.ends_with(']')) {
        return None;
    }
    let inner = &t[1..t.len() - 1];
    let (base_str, offset) = if let Some((lhs, rhs)) = inner.split_once('+') {
        (lhs.trim(), parse_int_literal(rhs.trim()))
    } else {
        (inner.trim(), None)
    };

    let base = classify_operand(base_str);
    let width = match &base {
        Operand::Register { ty: Some(RegType::BitWidth(bits)), .. } => Some(*bits),
        _ => None,
    };

    Some(Operand::MemRef {
        base: Box::new(base),
        offset,
        width,
        raw: t.to_string(),
    })
}

/* ---------- Operand 解析 ---------- */
fn classify_operand(tok: &str) -> Operand {
    let t = tok.trim();
    // 1) [base+off] memory reference
    if let Some(mem) = parse_mem_ref_operand(t) {
        return mem;
    }
    // 2) floating literal
    if RE_FLOAT.is_match(t) {
        let v: f64 = t.parse().unwrap();
        return Operand::ImmediateF(v);
    }
    // 3) integer literal
    if RE_HEX_I.is_match(t) {
        return Operand::ImmediateI(parse_hex(t).unwrap());
    }
    if RE_DEC_I.is_match(t) {
        return Operand::ImmediateI(t.parse::<i64>().unwrap());
    }
    // 4) register-like operand (R*/P*/UR*/UP* plus RZ/PT/URZ/UPT)
    if let Some(r) = parse_register_operand(t) {
        return r;
    }
    // 5) constant memory c[bank][off]
    if let Some(cap) = RE_CONST.captures(t) {
        let bank = u32::from_str_radix(&cap["bank"][2..], 16).unwrap();
        let off  = u32::from_str_radix(&cap["off"][2..], 16).unwrap();
        return Operand::ConstMem { bank, offset: off };
    }
    // 5b) descriptor-based memory: desc[URx][Ry[.width][+off]]
    //     Strip the desc[URx] qualifier and parse the inner [...] as a MemRef.
    //     TODO: the descriptor register index (URx) is currently discarded —
    //     two instructions loading through different descriptors produce
    //     indistinguishable MemRef nodes.  Sufficient for present decompilation
    //     needs but should become a dedicated Operand variant if we ever care
    //     about round-tripping or descriptor-level differential analysis.
    if let Some(cap) = RE_DESC.captures(t) {
        let mem_part = &cap["mem"];
        if let Some(mem) = parse_mem_ref_operand(mem_part) {
            return mem;
        }
    }
    // 6) fallback raw token
    Operand::Raw(t.to_string())
}

/// Return a prefix of `s` with SM 120 scheduling annotations removed.
///
/// SM 120 SASS appends tokens like `&wr=0x0`, `&req={1,0}`, `?trans1`,
/// and `?WAIT5_END_GROUP` after the last operand but before the `;`.
/// We cut at the first `&` or `?` that is both preceded by whitespace **and**
/// followed by an ASCII letter, then trim trailing whitespace from the result.
///
/// Assumptions (verified against the SM 100/120 corpus):
/// * `&` inside bracketed sub-expressions like `c[0x0][0x28]` is never
///   preceded by whitespace, so the scan leaves it intact.
/// * Real scheduling keywords always start with an ASCII letter (`wr`,
///   `req`, `trans`, `WAIT`), so tokens like `&0x1` or `?5` are preserved.
/// * SASS never emits `?`-prefixed operands in current architectures; if
///   NVIDIA ever introduces one, this heuristic would need revisiting.
fn strip_scheduling_annotations(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Only consider `&` or `?` preceded by at least one whitespace character.
        if (bytes[i] == b'&' || bytes[i] == b'?')
            && i > 0
            && bytes[i - 1].is_ascii_whitespace()
        {
            // The next character must be an ASCII letter (scheduling keywords).
            if i + 1 < bytes.len() && bytes[i + 1].is_ascii_alphabetic() {
                // Walk backwards to strip trailing whitespace before the token.
                let end = s[..i].trim_end().len();
                return &s[..end];
            }
        }
        i += 1;
    }
    s
}

fn parse_operands(input: &str) -> IResult<&str, Vec<Operand>> {
    let (rest, list) = take_until(";")(input)?;
    let list = strip_scheduling_annotations(list);
    let ops = list
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(classify_operand)
        .collect();
    Ok((rest, ops))
}

/* ---------- 单行指令 ---------- */
pub fn parse_instruction_line(line: &str) -> Option<Instruction> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("/*") { return None; }

    let parsed = tuple((
        parse_addr,                   // /*0040*/
        multispace0,
        opt(tag("code")),            // 略过 "code" 行
        opt(tuple((multispace0, parse_pred, multispace0))),
        parse_opcode,
        parse_operands,
    ))(trimmed);

    let (_, (addr, _, _, pred_opt, opcode, operands)) = match parsed.ok() {
        Some(v) => v,
        None => return None,
    };
    let pred = pred_opt.map(|(_, p, _)| p);
    Some(Instruction {
        addr,
        pred,
        opcode: opcode.to_string(),
        operands,
        raw: line.to_owned(),
    })
}

/* ---------- 解析整段 ---------- */
pub fn parse_sass(text: &str) -> Vec<Instruction> {
    text.lines().filter_map(parse_instruction_line).collect()
}

/// One parsed function extracted from a multi-function SASS dump.
#[derive(Debug, Clone)]
pub struct SassFunction {
    pub name: String,
    pub sm: Option<u32>,
    pub instrs: Vec<Instruction>,
}

/// Split a multi-function SASS dump (as produced by `cuobjdump --dump-sass`)
/// into its constituent functions. Each function is delimited by a
/// `Function : <name>` header line. The SM version is extracted from the
/// per-function `.headerflags` line when present, otherwise from any
/// `sm_xx` marker found before the first function.
pub fn split_functions(text: &str) -> Vec<SassFunction> {
    lazy_static! {
        static ref RE_FUNCTION: Regex =
            Regex::new(r"^\s*Function\s*:\s*(?P<name>\S+)\s*$").unwrap();
    }

    // Collect (name, start_line_idx) pairs, then slice the line vector.
    let lines: Vec<&str> = text.lines().collect();
    let mut markers: Vec<(String, usize)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(cap) = RE_FUNCTION.captures(line) {
            markers.push((cap["name"].to_string(), i));
        }
    }

    if markers.is_empty() {
        return Vec::new();
    }

    // Default SM from the preamble (everything before the first function).
    let preamble = lines[..markers[0].1].join("\n");
    let default_sm = parse_sm_version(&preamble);

    let mut out = Vec::with_capacity(markers.len());
    for (mi, (name, start)) in markers.iter().enumerate() {
        let end = if mi + 1 < markers.len() {
            markers[mi + 1].1
        } else {
            lines.len()
        };
        let region = lines[*start..end].join("\n");
        let sm = parse_sm_version(&region).or(default_sm);
        let instrs: Vec<Instruction> = lines[*start..end]
            .iter()
            .filter_map(|l| parse_instruction_line(l))
            .collect();
        out.push(SassFunction {
            name: name.clone(),
            sm,
            instrs,
        });
    }
    out
}

/// Parse SM generation from SASS metadata lines when available.
/// Priority:
/// 1) `.headerflags` entries containing `EF_CUDA_SMxx`
/// 2) `.target sm_xx` fallback
pub fn parse_sm_version(text: &str) -> Option<u32> {
    for line in text.lines() {
        if !line.contains("headerflags") {
            continue;
        }
        if let Some(cap) = RE_HEADER_SM.captures(line) {
            if let Ok(sm) = cap["sm"].parse::<u32>() {
                return Some(sm);
            }
        }
    }

    if let Some(cap) = RE_TARGET_SM.captures(text) {
        if let Ok(sm) = cap["sm"].parse::<u32>() {
            return Some(sm);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── strip_scheduling_annotations ──────────────────────────────────

    #[test]
    fn strip_sched_single_wr() {
        let input = " R1, c[0x0][0x37c]                                 &wr=0x0          ?trans8";
        assert_eq!(strip_scheduling_annotations(input), " R1, c[0x0][0x37c]");
    }

    #[test]
    fn strip_sched_req_and_wr() {
        let input = " R12, desc[UR6][R4.64]                  &req={0} &wr=0x2 ?trans4";
        assert_eq!(strip_scheduling_annotations(input), " R12, desc[UR6][R4.64]");
    }

    #[test]
    fn strip_sched_wait_end_group() {
        let input = " R7, R7, UR4, R0  &req={1}  ?WAIT5_END_GROUP";
        assert_eq!(strip_scheduling_annotations(input), " R7, R7, UR4, R0");
    }

    #[test]
    fn strip_sched_question_only() {
        // Some SM 120 lines have only ?trans with no & token.
        let input = " R0, R1                                                             ?WAIT12_END_GROUP";
        assert_eq!(strip_scheduling_annotations(input), " R0, R1");
    }

    #[test]
    fn strip_sched_preserves_brackets() {
        // `&` inside brackets like c[0x0][0x28] must NOT be touched.
        let input = " R1, c[0x0][0x28]";
        assert_eq!(strip_scheduling_annotations(input), " R1, c[0x0][0x28]");
    }

    #[test]
    fn strip_sched_no_annotations() {
        let input = " R1, R2, R3";
        assert_eq!(strip_scheduling_annotations(input), " R1, R2, R3");
    }

    // ── Full instruction parsing with annotations ─────────────────────

    #[test]
    fn parse_sm120_imad_with_annotations() {
        let line =
            "        /*0030*/                   IADD3 R1, PT, PT, R1, -0x100, RZ                      &req={0}         ?WAIT2_END_GROUP;  /* 0xffffff0001017810 */";
        let instr = parse_instruction_line(line).expect("should parse");
        assert_eq!(instr.opcode, "IADD3");
        assert_eq!(instr.operands.len(), 6);
        assert_eq!(instr.operands[0], Operand::Register { class: "R".into(), idx: 1, ty: None, sign: 1 });
        assert_eq!(instr.operands[5], Operand::Register { class: "RZ".into(), idx: 0, ty: None, sign: 1 });
    }

    #[test]
    fn parse_sm120_ldg_desc_with_multi_annotations() {
        // This has both &req={0} &wr=0x2 plus ?trans4.
        let line =
            "        /*00d0*/                   LDG.E.CONSTANT R12, desc[UR6][R4.64]                  &req={0} &wr=0x2 ?trans4;           /* 0x00000006040c7981 */";
        let instr = parse_instruction_line(line).expect("should parse");
        assert_eq!(instr.opcode, "LDG.E.CONSTANT");
        assert_eq!(instr.operands.len(), 2);
        assert_eq!(instr.operands[0], Operand::Register { class: "R".into(), idx: 12, ty: None, sign: 1 });
        // desc[UR6][R4.64] → MemRef (descriptor prefix stripped)
        assert!(matches!(
            &instr.operands[1],
            Operand::MemRef { base, offset: None, width: Some(64), .. }
            if matches!(base.as_ref(), Operand::Register { class, idx: 4, .. } if class == "R")
        ));
    }

    // ── Descriptor operand parsing ────────────────────────────────────

    #[test]
    fn classify_desc_operand_no_offset() {
        let op = classify_operand("desc[UR6][R4.64]");
        assert!(matches!(
            &op,
            Operand::MemRef { base, offset: None, width: Some(64), .. }
            if matches!(base.as_ref(), Operand::Register { class, idx: 4, .. } if class == "R")
        ));
    }

    #[test]
    fn classify_desc_operand_with_offset() {
        let op = classify_operand("desc[UR6][R2.64+0x4]");
        assert!(matches!(
            &op,
            Operand::MemRef { base, offset: Some(4), width: Some(64), .. }
            if matches!(base.as_ref(), Operand::Register { class, idx: 2, .. } if class == "R")
        ));
    }

    #[test]
    fn classify_desc_operand_large_offset() {
        let op = classify_operand("desc[UR6][R4.64+0x3c]");
        assert!(matches!(
            &op,
            Operand::MemRef { base, offset: Some(0x3c), width: Some(64), .. }
            if matches!(base.as_ref(), Operand::Register { class, idx: 4, .. } if class == "R")
        ));
    }

    #[test]
    fn classify_desc_operand_negative_offset() {
        // Observed in SM 100 corpus: `+-0x1c` means base + (-28).
        let op = classify_operand("desc[UR6][R4.64+-0x1c]");
        assert!(matches!(
            &op,
            Operand::MemRef { base, offset: Some(-0x1c), width: Some(64), .. }
            if matches!(base.as_ref(), Operand::Register { class, idx: 4, .. } if class == "R")
        ));
    }

    #[test]
    fn classify_desc_operand_large_descriptor_index() {
        // Descriptor index `UR15` should still parse — the index itself is
        // currently discarded (see TODO in classify_operand).
        let op = classify_operand("desc[UR15][R4.64]");
        assert!(matches!(
            &op,
            Operand::MemRef { base, offset: None, width: Some(64), .. }
            if matches!(base.as_ref(), Operand::Register { class, idx: 4, .. } if class == "R")
        ));
    }

    // ── strip_scheduling_annotations edge cases ───────────────────────

    #[test]
    fn strip_sched_empty_input() {
        assert_eq!(strip_scheduling_annotations(""), "");
    }

    #[test]
    fn strip_sched_ampersand_at_index_zero_is_kept() {
        // No preceding whitespace → the `&` is assumed to be part of an
        // operand, not a scheduling annotation.
        assert_eq!(strip_scheduling_annotations("&literal"), "&literal");
    }

    #[test]
    fn strip_sched_ampersand_followed_by_digit_is_kept() {
        // Scheduling annotations always start with a letter (`wr`, `req`,
        // `trans`, `WAIT`), so `&` followed by a digit should not trigger.
        assert_eq!(strip_scheduling_annotations(" &0x1"), " &0x1");
    }
}
