//! ABI profile and constant-memory parameter mapping for SASS decompilation.
//! This module is intentionally conservative: it maps well-known offsets to
//! symbolic names and falls back to raw `ConstMem(bank, off)` semantics.

use crate::ir::{DisplayCtx, IRExpr, RegId};
use crate::parser::{Instruction, Operand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiGeneration {
    /// Common older layout with parameter window beginning near c[0x0][0x140].
    LegacyParam140,
    /// Common newer layout with parameter window beginning near c[0x0][0x160].
    ModernParam160,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiProfile {
    pub generation: AbiGeneration,
    pub const_bank: u32,
    pub param_base: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstMemKind {
    Builtin(&'static str),
    ParamWord { param_idx: u32, word_idx: u32 },
    AbiInternal(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedConstMem {
    pub symbol: String,
    pub kind: ConstMemKind,
}

impl AbiProfile {
    pub fn legacy_param_140() -> Self {
        Self {
            generation: AbiGeneration::LegacyParam140,
            const_bank: 0,
            param_base: 0x140,
        }
    }

    pub fn modern_param_160() -> Self {
        Self {
            generation: AbiGeneration::ModernParam160,
            const_bank: 0,
            param_base: 0x160,
        }
    }

    /// Detect the best-effort ABI profile from observed constant-memory offsets.
    /// Heuristic:
    /// - Prefer layouts with denser hits in the first parameter window.
    /// - Fall back to modern profile, which matches current fixtures.
    pub fn detect(instructions: &[Instruction]) -> Self {
        let mut near_140_hits = 0usize;
        let mut near_160_hits = 0usize;

        for ins in instructions {
            for op in &ins.operands {
                collect_constmem_hits(op, &mut |bank, off| {
                    if bank != 0 {
                        return;
                    }
                    if (0x140..0x160).contains(&off) && off % 4 == 0 {
                        near_140_hits += 1;
                    }
                    if (0x160..0x180).contains(&off) && off % 4 == 0 {
                        near_160_hits += 1;
                    }
                });
            }
        }

        if near_160_hits > 0 && near_160_hits >= near_140_hits {
            return Self::modern_param_160();
        }
        if near_140_hits > 0 {
            return Self::legacy_param_140();
        }
        Self::modern_param_160()
    }

    /// Resolve `c[bank][offset]` into a symbolic ABI meaning when possible.
    pub fn resolve_constmem(&self, bank: u32, offset: u32) -> Option<ResolvedConstMem> {
        if bank != self.const_bank {
            return None;
        }

        let builtin = match offset {
            0x8 => Some("blockDimX"),
            0xc => Some("blockDimY"),
            0x10 => Some("blockDimZ"),
            0x14 => Some("gridDimX"),
            0x18 => Some("gridDimY"),
            0x1c => Some("gridDimZ"),
            _ => None,
        };
        if let Some(name) = builtin {
            return Some(ResolvedConstMem {
                symbol: name.to_string(),
                kind: ConstMemKind::Builtin(name),
            });
        }

        if offset >= self.param_base && (offset - self.param_base) % 4 == 0 {
            let word = (offset - self.param_base) / 4;
            let param_idx = word / 2;
            let word_idx = word % 2;
            return Some(ResolvedConstMem {
                symbol: format!("param_{}[{}]", param_idx, word_idx),
                kind: ConstMemKind::ParamWord { param_idx, word_idx },
            });
        }

        if matches!(offset, 0x28 | 0x44) {
            return Some(ResolvedConstMem {
                symbol: format!("abi_internal_0x{:x}", offset),
                kind: ConstMemKind::AbiInternal(offset),
            });
        }

        None
    }
}

pub struct AbiDisplay {
    profile: AbiProfile,
}

impl AbiDisplay {
    pub fn new(profile: AbiProfile) -> Self {
        Self { profile }
    }

    pub fn profile(&self) -> AbiProfile {
        self.profile
    }

    fn try_constmem_symbol(&self, op: &str, args: &[IRExpr]) -> Option<String> {
        if op != "ConstMem" || args.len() != 2 {
            return None;
        }
        let bank = imm_as_u32(&args[0])?;
        let offset = imm_as_u32(&args[1])?;
        let resolved = self.profile.resolve_constmem(bank, offset)?;
        Some(resolved.symbol)
    }
}

impl DisplayCtx for AbiDisplay {
    fn reg(&self, r: &RegId) -> String {
        r.display()
    }

    fn expr(&self, e: &IRExpr) -> String {
        match e {
            IRExpr::Reg(r) => self.reg(r),
            IRExpr::ImmI(i) => format!("{}", i),
            IRExpr::ImmF(f) => format!("{}", f),
            IRExpr::Mem { base, offset, width } => {
                let mut s = format!("*{}", self.expr(base));
                if let Some(off) = offset {
                    s.push_str(&format!("+{}", self.expr(off)));
                }
                if let Some(w) = width {
                    s.push_str(&format!("@{}", w));
                }
                s
            }
            IRExpr::Op { op, args } => {
                if let Some(sym) = self.try_constmem_symbol(op, args) {
                    return sym;
                }
                let list = args.iter().map(|a| self.expr(a)).collect::<Vec<_>>().join(", ");
                format!("{}({})", op, list)
            }
        }
    }
}

fn imm_as_u32(e: &IRExpr) -> Option<u32> {
    match e {
        IRExpr::ImmI(i) if *i >= 0 => u32::try_from(*i).ok(),
        _ => None,
    }
}

fn collect_constmem_hits<F>(op: &Operand, f: &mut F)
where
    F: FnMut(u32, u32),
{
    match op {
        Operand::ConstMem { bank, offset } => f(*bank, *offset),
        Operand::MemRef { base, .. } => collect_constmem_hits(base.as_ref(), f),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_sass;

    #[test]
    fn detects_legacy_profile_from_param_window() {
        let sass = r#"
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x140] ;
            /*0010*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x148] ;
            /*0020*/ IADD3 R3, R3, c[0x0][0x154], RZ ;
        "#;
        let instrs = parse_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::legacy_param_140());
    }

    #[test]
    fn detects_modern_profile_from_param_window() {
        let sass = r#"
            /*0000*/ IMAD.WIDE R4, R0, R7, c[0x0][0x160] ;
            /*0010*/ IMAD.WIDE R6, R0, R7, c[0x0][0x168] ;
            /*0020*/ IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x17c] ;
        "#;
        let instrs = parse_sass(sass);
        let p = AbiProfile::detect(&instrs);
        assert_eq!(p, AbiProfile::modern_param_160());
    }

    #[test]
    fn resolves_user_asked_offsets_in_legacy_profile() {
        let p = AbiProfile::legacy_param_140();
        assert_eq!(p.resolve_constmem(0, 0x8).unwrap().symbol, "blockDimX");
        assert_eq!(p.resolve_constmem(0, 0x28).unwrap().symbol, "abi_internal_0x28");
        assert_eq!(p.resolve_constmem(0, 0x44).unwrap().symbol, "abi_internal_0x44");
        assert_eq!(p.resolve_constmem(0, 0x140).unwrap().symbol, "param_0[0]");
        assert_eq!(p.resolve_constmem(0, 0x144).unwrap().symbol, "param_0[1]");
        assert_eq!(p.resolve_constmem(0, 0x148).unwrap().symbol, "param_1[0]");
        assert_eq!(p.resolve_constmem(0, 0x14c).unwrap().symbol, "param_1[1]");
        assert_eq!(p.resolve_constmem(0, 0x150).unwrap().symbol, "param_2[0]");
        assert_eq!(p.resolve_constmem(0, 0x154).unwrap().symbol, "param_2[1]");
    }

    #[test]
    fn abi_display_formats_constmem_symbolically() {
        let display = AbiDisplay::new(AbiProfile::legacy_param_140());
        let expr = IRExpr::Op {
            op: "ConstMem".to_string(),
            args: vec![IRExpr::ImmI(0), IRExpr::ImmI(0x148)],
        };
        assert_eq!(display.expr(&expr), "param_1[0]");
    }
}
