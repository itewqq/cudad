//! Canonical SASS decoding front-end.

use nom::{
    bytes::complete::{tag, take_while},
    character::complete::multispace0,
    combinator::{map, opt},
    sequence::{delimited, tuple},
    IResult,
};

use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateUse {
    pub reg: String,
    pub sense: bool,
}

pub type Predicate = PredicateUse;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegType {
    BitWidth(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SchedulingInfo {
    pub annotations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedOperand {
    Register {
        class: String,
        idx: i32,
        sign: i32,
        ty: Option<RegType>,
    },
    UniformRegister {
        class: String,
        idx: i32,
        sign: i32,
        ty: Option<RegType>,
    },
    PredicateRegister {
        class: String,
        idx: i32,
        sense: bool,
    },
    ImmediateI(i64),
    ImmediateF(f64),
    ConstMem {
        bank: u32,
        offset: u32,
    },
    DescriptorMem {
        descriptor: Box<DecodedOperand>,
        addr: Box<DecodedOperand>,
        raw: String,
    },
    Address {
        base: Box<DecodedOperand>,
        offset: Option<i64>,
        width: Option<u32>,
        raw: String,
    },
    Raw(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminatorKind {
    None,
    FallthroughOnly,
    CondBranch {
        taken: Option<u32>,
        fallthrough: Option<u32>,
    },
    Jump {
        target: u32,
    },
    Return,
    IndirectOrUnknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedInstruction {
    pub addr: u32,
    pub pred: Option<Predicate>,
    pub opcode: String,
    pub operands: Vec<DecodedOperand>,
    pub scheduling: SchedulingInfo,
    pub terminator: TerminatorKind,
    pub raw: String,
}

#[derive(Debug, Clone)]
pub struct DecodedFunction {
    pub name: String,
    pub sm: Option<u32>,
    pub instrs: Vec<DecodedInstruction>,
}




lazy_static! {
    static ref RE_REG_NUM: Regex = Regex::new(r"^(?P<cls>UP|UR|R|P)(?P<idx>\d+)$").unwrap();
    static ref RE_CONST: Regex =
        Regex::new(r"^c\[(?P<bank>0x[0-9A-Fa-f]+)\]\[(?P<off>0x[0-9A-Fa-f]+)\]$").unwrap();
    static ref RE_HEX_I: Regex = Regex::new(r"^-?0x[0-9A-Fa-f]+$").unwrap();
    static ref RE_DEC_I: Regex = Regex::new(r"^-?\d+$").unwrap();
    static ref RE_FLOAT: Regex = Regex::new(r"^-?\d+\.\d+(e[+-]?\d+)?$").unwrap();
    static ref RE_HEADER_SM: Regex = Regex::new(r"EF_CUDA_SM(?P<sm>\d+)").unwrap();
    static ref RE_TARGET_SM: Regex = Regex::new(r"\bsm_(?P<sm>\d+)\b").unwrap();
    static ref RE_FUNCTION: Regex = Regex::new(r"^\s*Function\s*:\s*(?P<name>\S+)\s*$").unwrap();
}



fn parse_hex(s: &str) -> Option<i64> {
    let s = s.trim();
    let (is_negative, num_str) = if let Some(rest) = s.strip_prefix("-0x") {
        (true, rest)
    } else if let Some(rest) = s.strip_prefix("0x") {
        (false, rest)
    } else {
        return None;
    };

    i64::from_str_radix(num_str, 16)
        .ok()
        .map(|num| if is_negative { -num } else { num })
}

fn parse_addr(input: &str) -> IResult<&str, u32> {
    map(
        delimited(
            tag("/*"),
            take_while(|c: char| c.is_ascii_hexdigit()),
            tag("*/"),
        ),
        |hex: &str| u32::from_str_radix(hex, 16).unwrap(),
    )(input)
}

fn parse_pred(input: &str) -> IResult<&str, PredicateUse> {
    map(
        tuple((
            tag("@"),
            opt(tag("!")),
            take_while(|c: char| c.is_ascii_alphanumeric()),
        )),
        |(_, neg, reg): (_, Option<&str>, &str)| PredicateUse {
            reg: reg.to_owned(),
            sense: neg.is_none(),
        },
    )(input)
}

fn is_opcode_char(c: char) -> bool {
    !(c.is_whitespace() || c == ';')
}

fn parse_opcode(input: &str) -> IResult<&str, &str> {
    take_while(is_opcode_char)(input)
}

fn parse_instruction_prefix(input: &str) -> IResult<&str, (u32, Option<PredicateUse>, &str)> {
    let (input, addr) = parse_addr(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = opt(tag("code"))(input)?;
    let (input, pred_opt) = opt(tuple((multispace0, parse_pred, multispace0)))(input)?;
    let (input, opcode) = parse_opcode(input)?;
    Ok((input, (addr, pred_opt.map(|(_, pred, _)| pred), opcode)))
}

fn parse_int_literal(s: &str) -> Option<i64> {
    if RE_HEX_I.is_match(s) {
        return parse_hex(s);
    }
    if RE_DEC_I.is_match(s) {
        return s.parse::<i64>().ok();
    }
    None
}

fn parse_signed_offset_literal(s: &str) -> Option<i64> {
    if let Some(rest) = s.strip_prefix('+') {
        return parse_int_literal(rest.trim());
    }
    parse_int_literal(s.trim())
}

fn split_top_level(input: &str, delimiter: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut square = 0u32;
    let mut curly = 0u32;
    let mut round = 0u32;

    for (idx, ch) in input.char_indices() {
        match ch {
            '[' => square += 1,
            ']' => square = square.saturating_sub(1),
            '{' => curly += 1,
            '}' => curly = curly.saturating_sub(1),
            '(' => round += 1,
            ')' => round = round.saturating_sub(1),
            _ => {}
        }

        if ch == delimiter && square == 0 && curly == 0 && round == 0 {
            out.push(input[start..idx].trim());
            start = idx + ch.len_utf8();
        }
    }

    out.push(input[start..].trim());
    out
}

fn find_schedule_start(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut square = 0u32;
    let mut curly = 0u32;
    let mut round = 0u32;

    for (idx, ch) in s.char_indices() {
        match ch {
            '[' => square += 1,
            ']' => square = square.saturating_sub(1),
            '{' => curly += 1,
            '}' => curly = curly.saturating_sub(1),
            '(' => round += 1,
            ')' => round = round.saturating_sub(1),
            _ => {}
        }

        if square == 0
            && curly == 0
            && round == 0
            && (bytes[idx] == b'&' || bytes[idx] == b'?')
            && idx > 0
            && bytes[idx - 1].is_ascii_whitespace()
        {
            let next = s[idx + ch.len_utf8()..].chars().next();
            if next.is_some_and(|c| c.is_ascii_alphabetic()) {
                return Some(idx);
            }
        }
    }

    None
}

fn strip_scheduling_annotations(s: &str) -> &str {
    if let Some(idx) = find_schedule_start(s) {
        let end = s[..idx].trim_end().len();
        &s[..end]
    } else {
        s
    }
}

fn extract_scheduling_info(s: &str) -> SchedulingInfo {
    if let Some(idx) = find_schedule_start(s) {
        SchedulingInfo {
            annotations: s[idx..].split_whitespace().map(str::to_string).collect(),
        }
    } else {
        SchedulingInfo::default()
    }
}

fn parse_register_like_components(core: &str) -> Option<(String, i32, Option<RegType>)> {
    let mut ty = None;
    let mut parts = core.split('.');
    let base = parts.next()?.to_ascii_uppercase();
    for suffix in parts {
        if let Ok(bits) = suffix.to_ascii_lowercase().parse::<u32>() {
            ty = Some(RegType::BitWidth(bits));
        }
    }

    let (class, idx) = match base.as_str() {
        "RZ" => ("RZ".to_string(), 0),
        "PT" => ("PT".to_string(), 0),
        "URZ" => ("URZ".to_string(), 0),
        "UPT" => ("UPT".to_string(), 0),
        _ => {
            let cap = RE_REG_NUM.captures(base.as_str())?;
            let class = cap["cls"].to_string();
            let idx = cap["idx"].parse::<i32>().ok()?;
            (class, idx)
        }
    };

    Some((class, idx, ty))
}

fn decode_predicate_operand(tok: &str) -> Option<DecodedOperand> {
    let t = tok.trim();
    if t.is_empty() {
        return None;
    }
    let (sense, core) = if let Some(rest) = t.strip_prefix('!') {
        (false, rest)
    } else {
        (true, t)
    };

    let upper = core.to_ascii_uppercase();
    match upper.as_str() {
        "PT" => Some(DecodedOperand::PredicateRegister {
            class: "PT".into(),
            idx: 0,
            sense,
        }),
        "UPT" => Some(DecodedOperand::PredicateRegister {
            class: "UPT".into(),
            idx: 0,
            sense,
        }),
        _ => {
            if let Some(num) = upper.strip_prefix("UP") {
                let idx = num.parse::<i32>().ok()?;
                return Some(DecodedOperand::PredicateRegister {
                    class: "UP".into(),
                    idx,
                    sense,
                });
            }
            if let Some(num) = upper.strip_prefix('P') {
                let idx = num.parse::<i32>().ok()?;
                return Some(DecodedOperand::PredicateRegister {
                    class: "P".into(),
                    idx,
                    sense,
                });
            }
            None
        }
    }
}

fn decode_register_operand(tok: &str) -> Option<DecodedOperand> {
    let mut sign = 1;
    let mut core = tok.trim();
    if let Some(rest) = core.strip_prefix('-') {
        sign = -1;
        core = rest;
    }
    if core.starts_with('!') {
        return None;
    }

    let (class, idx, ty) = parse_register_like_components(core)?;
    match class.as_str() {
        "UR" | "URZ" => Some(DecodedOperand::UniformRegister {
            class,
            idx,
            sign,
            ty,
        }),
        "R" | "RZ" => Some(DecodedOperand::Register {
            class,
            idx,
            sign,
            ty,
        }),
        _ => None,
    }
}

fn decode_constmem_operand(tok: &str) -> Option<DecodedOperand> {
    let captures = RE_CONST.captures(tok.trim())?;
    let bank = u32::from_str_radix(&captures["bank"][2..], 16).ok()?;
    let offset = u32::from_str_radix(&captures["off"][2..], 16).ok()?;
    Some(DecodedOperand::ConstMem { bank, offset })
}

fn decoded_width(base: &DecodedOperand) -> Option<u32> {
    match base {
        DecodedOperand::Register {
            ty: Some(RegType::BitWidth(bits)),
            ..
        }
        | DecodedOperand::UniformRegister {
            ty: Some(RegType::BitWidth(bits)),
            ..
        } => Some(*bits),
        _ => None,
    }
}

fn split_address_base_offset(inner: &str) -> (&str, Option<i64>) {
    let mut square = 0u32;
    let mut curly = 0u32;
    let mut round = 0u32;

    for (idx, ch) in inner.char_indices().skip(1) {
        match ch {
            '[' => square += 1,
            ']' => square = square.saturating_sub(1),
            '{' => curly += 1,
            '}' => curly = curly.saturating_sub(1),
            '(' => round += 1,
            ')' => round = round.saturating_sub(1),
            _ => {}
        }

        if square == 0 && curly == 0 && round == 0 && (ch == '+' || ch == '-') {
            if let Some(offset) = parse_signed_offset_literal(&inner[idx..]) {
                return (inner[..idx].trim(), Some(offset));
            }
        }
    }

    (inner.trim(), None)
}

fn decode_address_operand(tok: &str) -> Option<DecodedOperand> {
    let t = tok.trim();
    if !(t.starts_with('[') && t.ends_with(']')) {
        return None;
    }
    let inner = &t[1..t.len() - 1];
    let (base_str, offset) = split_address_base_offset(inner);
    let base = decode_operand(base_str);
    Some(DecodedOperand::Address {
        width: decoded_width(&base),
        base: Box::new(base),
        offset,
        raw: t.to_string(),
    })
}

fn decode_descriptor_operand(tok: &str) -> Option<DecodedOperand> {
    let t = tok.trim();
    let rest = t.strip_prefix("desc[")?;
    let end_desc = rest.find(']')?;
    let descriptor = decode_operand(&rest[..end_desc]);
    let addr_part = rest[end_desc + 1..].trim();
    let addr = decode_address_operand(addr_part)?;
    Some(DecodedOperand::DescriptorMem {
        descriptor: Box::new(descriptor),
        addr: Box::new(addr),
        raw: t.to_string(),
    })
}



fn decode_operand(tok: &str) -> DecodedOperand {
    let t = tok.trim();
    if t.is_empty() {
        return DecodedOperand::Raw(String::new());
    }
    if let Some(op) = decode_descriptor_operand(t) {
        return op;
    }
    if let Some(op) = decode_address_operand(t) {
        return op;
    }
    if let Some(op) = decode_predicate_operand(t) {
        return op;
    }
    if RE_FLOAT.is_match(t) {
        return DecodedOperand::ImmediateF(t.parse::<f64>().unwrap());
    }
    if let Some(v) = parse_int_literal(t) {
        return DecodedOperand::ImmediateI(v);
    }
    if let Some(op) = decode_register_operand(t) {
        return op;
    }
    if let Some(op) = decode_constmem_operand(t) {
        return op;
    }
    DecodedOperand::Raw(t.to_string())
}

fn branch_target_addr(operands: &[DecodedOperand]) -> Option<u32> {
    operands.first().and_then(|op| match op {
        DecodedOperand::ImmediateI(v) if *v >= 0 => Some(*v as u32),
        DecodedOperand::Raw(s) => {
            let trimmed = s.trim();
            if let Some(rest) = trimmed.strip_prefix("0x") {
                u32::from_str_radix(rest, 16).ok()
            } else {
                trimmed.parse::<u32>().ok()
            }
        }
        _ => None,
    })
}

fn classify_terminator(
    opcode: &str,
    pred: Option<&PredicateUse>,
    operands: &[DecodedOperand],
    next_addr: Option<u32>,
) -> TerminatorKind {
    let base = opcode.split('.').next().unwrap_or(opcode);
    match base {
        "EXIT" | "RET" => {
            if pred.is_some() {
                TerminatorKind::CondBranch {
                    taken: None,
                    fallthrough: next_addr,
                }
            } else {
                TerminatorKind::Return
            }
        }
        "BRA" | "JMP" | "JMPP" => {
            let target = branch_target_addr(operands);
            if pred.is_some() {
                TerminatorKind::CondBranch {
                    taken: target,
                    fallthrough: next_addr,
                }
            } else if let Some(target) = target {
                TerminatorKind::Jump { target }
            } else {
                TerminatorKind::IndirectOrUnknown
            }
        }
        "BRX" => {
            if pred.is_some() {
                TerminatorKind::CondBranch {
                    taken: None,
                    fallthrough: next_addr,
                }
            } else {
                TerminatorKind::IndirectOrUnknown
            }
        }
        "CAL" | "JCAL" | "PRET" => TerminatorKind::FallthroughOnly,
        _ => TerminatorKind::None,
    }
}

fn finalize_terminators(instrs: &mut [DecodedInstruction]) {
    for idx in 0..instrs.len() {
        let next_addr = instrs.get(idx + 1).map(|next| next.addr);
        let terminator = classify_terminator(
            &instrs[idx].opcode,
            instrs[idx].pred.as_ref(),
            &instrs[idx].operands,
            next_addr,
        );
        instrs[idx].terminator = terminator;
    }
}

pub fn decode_instruction_line(line: &str) -> Option<DecodedInstruction> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("/*") {
        return None;
    }

    let (rest, (addr, pred, opcode)) = parse_instruction_prefix(trimmed).ok()?;
    let (operand_field, _) = rest.split_once(';')?;
    let operand_text = strip_scheduling_annotations(operand_field);
    let scheduling = extract_scheduling_info(operand_field);
    let operands = split_top_level(operand_text, ',')
        .into_iter()
        .filter(|part| !part.is_empty())
        .map(decode_operand)
        .collect::<Vec<_>>();
    let terminator = classify_terminator(opcode, pred.as_ref(), &operands, None);

    Some(DecodedInstruction {
        addr,
        pred,
        opcode: opcode.to_string(),
        operands,
        scheduling,
        terminator,
        raw: line.to_string(),
    })
}


pub fn decode_sass(text: &str) -> Vec<DecodedInstruction> {
    let mut instrs = text
        .lines()
        .filter_map(decode_instruction_line)
        .collect::<Vec<_>>();
    finalize_terminators(&mut instrs);
    instrs
}


pub fn split_decoded_functions(text: &str) -> Vec<DecodedFunction> {
    let lines: Vec<&str> = text.lines().collect();
    let mut markers: Vec<(String, usize)> = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if let Some(cap) = RE_FUNCTION.captures(line) {
            markers.push((cap["name"].to_string(), idx));
        }
    }

    if markers.is_empty() {
        return Vec::new();
    }

    let preamble = lines[..markers[0].1].join(
        "
",
    );
    let default_sm = parse_sm_version(&preamble);
    let mut out = Vec::with_capacity(markers.len());

    for (marker_idx, (name, start)) in markers.iter().enumerate() {
        let end = if marker_idx + 1 < markers.len() {
            markers[marker_idx + 1].1
        } else {
            lines.len()
        };
        let region = lines[*start..end].join(
            "
",
        );
        let sm = parse_sm_version(&region).or(default_sm);
        let mut instrs = lines[*start..end]
            .iter()
            .filter_map(|line| decode_instruction_line(line))
            .collect::<Vec<_>>();
        finalize_terminators(&mut instrs);
        out.push(DecodedFunction {
            name: name.clone(),
            sm,
            instrs,
        });
    }

    out
}


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

    #[test]
    fn strip_sched_single_wr() {
        let input = " R1, c[0x0][0x37c]                                 &wr=0x0          ?trans8";
        assert_eq!(strip_scheduling_annotations(input), " R1, c[0x0][0x37c]");
        assert_eq!(
            extract_scheduling_info(input).annotations,
            vec!["&wr=0x0", "?trans8"]
        );
    }

    #[test]
    fn strip_sched_req_and_wr() {
        let input = " R12, desc[UR6][R4.64]                  &req={0} &wr=0x2 ?trans4";
        assert_eq!(
            strip_scheduling_annotations(input),
            " R12, desc[UR6][R4.64]"
        );
    }

    #[test]
    fn strip_sched_wait_end_group() {
        let input = " R7, R7, UR4, R0  &req={1}  ?WAIT5_END_GROUP";
        assert_eq!(strip_scheduling_annotations(input), " R7, R7, UR4, R0");
    }

    #[test]
    fn strip_sched_question_only() {
        let input =
            " R0, R1                                                             ?WAIT12_END_GROUP";
        assert_eq!(strip_scheduling_annotations(input), " R0, R1");
    }

    #[test]
    fn strip_sched_preserves_brackets() {
        let input = " R1, c[0x0][0x28]";
        assert_eq!(strip_scheduling_annotations(input), " R1, c[0x0][0x28]");
    }

    #[test]
    fn strip_sched_no_annotations() {
        let input = " R1, R2, R3";
        assert_eq!(strip_scheduling_annotations(input), " R1, R2, R3");
        assert!(extract_scheduling_info(input).annotations.is_empty());
    }

    #[test]
    fn parse_sm120_imad_with_annotations() {
        let line =
            "        /*0030*/                   IADD3 R1, PT, PT, R1, -0x100, RZ                      &req={0}         ?WAIT2_END_GROUP;  /* 0xffffff0001017810 */";
        let instr = decode_instruction_line(line).expect("should parse");
        assert_eq!(instr.opcode, "IADD3");
        assert_eq!(instr.operands.len(), 6);
        assert_eq!(
            instr.operands[0],
            DecodedOperand::Register {
                class: "R".into(),
                idx: 1,
                ty: None,
                sign: 1,
            }
        );
        assert_eq!(
            instr.operands[5],
            DecodedOperand::Register {
                class: "RZ".into(),
                idx: 0,
                ty: None,
                sign: 1,
            }
        );
        assert_eq!(
            instr.scheduling.annotations,
            vec!["&req={0}", "?WAIT2_END_GROUP"]
        );
    }

    #[test]
    fn parse_sm120_ldg_desc_with_multi_annotations() {
        let line =
            "        /*00d0*/                   LDG.E.CONSTANT R12, desc[UR6][R4.64]                  &req={0} &wr=0x2 ?trans4;           /* 0x00000006040c7981 */";
        let instr = decode_instruction_line(line).expect("should parse");
        assert_eq!(instr.opcode, "LDG.E.CONSTANT");
        assert_eq!(instr.operands.len(), 2);
        assert_eq!(
            instr.operands[0],
            DecodedOperand::Register {
                class: "R".into(),
                idx: 12,
                ty: None,
                sign: 1,
            }
        );
        assert!(matches!(
            &instr.operands[1],
            DecodedOperand::DescriptorMem { addr, .. }
                if matches!(addr.as_ref(), DecodedOperand::Address {
                    base,
                    offset: None,
                    width: Some(64),
                    ..
                } if matches!(base.as_ref(), DecodedOperand::Register { class, idx: 4, .. } if class == "R"))
        ));
        assert_eq!(
            instr.scheduling.annotations,
            vec!["&req={0}", "&wr=0x2", "?trans4"]
        );
    }

    #[test]
    fn classify_desc_operand_no_offset() {
        let op = decode_operand("desc[UR6][R4.64]");
        assert!(matches!(
            &op,
            DecodedOperand::DescriptorMem { addr, .. }
                if matches!(addr.as_ref(), DecodedOperand::Address {
                    base,
                    offset: None,
                    width: Some(64),
                    ..
                } if matches!(base.as_ref(), DecodedOperand::Register { class, idx: 4, .. } if class == "R"))
        ));
    }

    #[test]
    fn classify_desc_operand_with_offset() {
        let op = decode_operand("desc[UR6][R2.64+0x4]");
        assert!(matches!(
            &op,
            DecodedOperand::DescriptorMem { addr, .. }
                if matches!(addr.as_ref(), DecodedOperand::Address {
                    base,
                    offset: Some(4),
                    width: Some(64),
                    ..
                } if matches!(base.as_ref(), DecodedOperand::Register { class, idx: 2, .. } if class == "R"))
        ));
    }

    #[test]
    fn classify_desc_operand_large_offset() {
        let op = decode_operand("desc[UR6][R4.64+0x3c]");
        assert!(matches!(
            &op,
            DecodedOperand::DescriptorMem { addr, .. }
                if matches!(addr.as_ref(), DecodedOperand::Address {
                    base,
                    offset: Some(0x3c),
                    width: Some(64),
                    ..
                } if matches!(base.as_ref(), DecodedOperand::Register { class, idx: 4, .. } if class == "R"))
        ));
    }

    #[test]
    fn classify_desc_operand_negative_offset() {
        let op = decode_operand("desc[UR6][R4.64+-0x1c]");
        assert!(matches!(
            &op,
            DecodedOperand::DescriptorMem { addr, .. }
                if matches!(addr.as_ref(), DecodedOperand::Address {
                    base,
                    offset: Some(-0x1c),
                    width: Some(64),
                    ..
                } if matches!(base.as_ref(), DecodedOperand::Register { class, idx: 4, .. } if class == "R"))
        ));
    }

    #[test]
    fn classify_desc_operand_large_descriptor_index() {
        let op = decode_operand("desc[UR15][R4.64]");
        assert!(matches!(
            &op,
            DecodedOperand::DescriptorMem { descriptor, addr, .. }
                if matches!(descriptor.as_ref(), DecodedOperand::UniformRegister { class, idx: 15, .. } if class == "UR")
                && matches!(addr.as_ref(), DecodedOperand::Address {
                    base,
                    offset: None,
                    width: Some(64),
                    ..
                } if matches!(base.as_ref(), DecodedOperand::Register { class, idx: 4, .. } if class == "R"))
        ));
    }

    #[test]
    fn classify_bracketed_operand_negative_offset_without_plus() {
        let decoded = decode_operand("[R8.64-0x20]");
        assert!(matches!(
            decoded,
            DecodedOperand::Address {
                offset: Some(-0x20),
                width: Some(64),
                ..
            }
        ));
    }

    #[test]
    fn split_top_level_preserves_nested_commas() {
        let parts = split_top_level("R1, desc[UR6][R4.64+0x4], &req={1,0}", ',');
        assert_eq!(parts, vec!["R1", "desc[UR6][R4.64+0x4]", "&req={1,0}"]);
    }

    #[test]
    fn decode_predicated_branch_terminator() {
        let sample = r#"
            /*0000*/ @P0 BRA 0x0020 ;
            /*0010*/ IADD3 R1, R1, 0x1, RZ ;
            /*0020*/ EXIT ;
        "#;
        let instrs = decode_sass(sample);
        assert_eq!(
            instrs[0].terminator,
            TerminatorKind::CondBranch {
                taken: Some(0x20),
                fallthrough: Some(0x10),
            }
        );
        assert_eq!(instrs[2].terminator, TerminatorKind::Return);
    }

    #[test]
    fn decode_predicated_exit_terminator() {
        let sample = r#"
            /*0040*/ @P0 EXIT ;
            /*0050*/ IADD3 R7, RZ, 0x4, RZ ;
        "#;
        let instrs = decode_sass(sample);
        assert_eq!(
            instrs[0].terminator,
            TerminatorKind::CondBranch {
                taken: None,
                fallthrough: Some(0x50),
            }
        );
    }

    #[test]
    fn decode_indirect_branch_terminator() {
        let sample = r#"
            /*0000*/ BRX R0 ;
            /*0010*/ EXIT ;
        "#;
        let instrs = decode_sass(sample);
        assert_eq!(instrs[0].terminator, TerminatorKind::IndirectOrUnknown);
    }

    #[test]
    fn decode_call_as_fallthrough_only() {
        let sample = r#"
            /*0000*/ CAL 0x0040 ;
            /*0010*/ EXIT ;
        "#;
        let instrs = decode_sass(sample);
        assert_eq!(instrs[0].terminator, TerminatorKind::FallthroughOnly);
    }

    #[test]
    fn test_parse_sm_version_from_headerflags() {
        let sample = r#"
            .headerflags @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        "#;
        assert_eq!(parse_sm_version(sample), Some(89));
    }

    #[test]
    fn test_parse_sm_version_from_target_fallback() {
        let sample = r#"
            .target sm_75
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        "#;
        assert_eq!(parse_sm_version(sample), Some(75));
    }

    #[test]
    fn test_split_decoded_functions_multi_function_dump() {
        let sample = r#"
	code for sm_89
		Function : first
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   EXIT ;
		Function : second
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89"
        /*0000*/                   IADD3 R2, RZ, 0x1, RZ ;
        /*0010*/                   IADD3 R3, RZ, 0x2, RZ ;
        /*0020*/                   EXIT ;
"#;
        let funcs = split_decoded_functions(sample);
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].name, "first");
        assert_eq!(funcs[1].name, "second");
        assert_eq!(funcs[0].sm, Some(89));
        assert_eq!(funcs[1].sm, Some(89));
        assert_eq!(funcs[0].instrs.len(), 2);
        assert_eq!(funcs[1].instrs.len(), 3);
        for func in &funcs {
            assert!(func
                .instrs
                .iter()
                .all(|instr| !instr.raw.contains("headerflags")));
        }
    }

    #[test]
    fn test_split_decoded_functions_empty_on_single_function_dump_without_marker() {
        let sample = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
            /*0010*/ EXIT ;
        "#;
        assert!(split_decoded_functions(sample).is_empty());
    }

    #[test]
    fn strip_sched_empty_input() {
        assert_eq!(strip_scheduling_annotations(""), "");
    }

    #[test]
    fn strip_sched_ampersand_at_index_zero_is_kept() {
        assert_eq!(strip_scheduling_annotations("&literal"), "&literal");
    }

    #[test]
    fn strip_sched_ampersand_followed_by_digit_is_kept() {
        assert_eq!(strip_scheduling_annotations(" &0x1"), " &0x1");
    }
}
