use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::abi::StatementRef;
use crate::ir::IRExpr;

use super::op_sig::OpSig;
use super::{rules, LiftedExpr, SemanticLiftConfig};

pub(super) type RuleFn = fn(
    sig: &OpSig,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr>;

#[derive(Clone)]
pub(super) struct RuleEntry {
    #[cfg_attr(not(test), allow(dead_code))]
    pub name: &'static str,
    pub apply: RuleFn,
}

#[derive(Default)]
pub(super) struct RuleRegistry {
    by_mnemonic: BTreeMap<&'static str, Vec<RuleEntry>>,
}

impl RuleRegistry {
    pub fn register(&mut self, mnemonic: &'static str, name: &'static str, apply: RuleFn) {
        self.by_mnemonic
            .entry(mnemonic)
            .or_default()
            .push(RuleEntry { name, apply });
    }

    pub fn dispatch(
        &self,
        opcode: &str,
        args: &[IRExpr],
        stmt_ref: StatementRef,
        config: &SemanticLiftConfig<'_>,
    ) -> Option<LiftedExpr> {
        let sig = OpSig::parse(opcode);
        let candidates = self.by_mnemonic.get(sig.mnemonic.as_str())?;
        for candidate in candidates {
            if let Some(expr) = (candidate.apply)(&sig, args, stmt_ref, config) {
                return Some(expr);
            }
        }
        None
    }

    #[cfg(test)]
    pub fn entries_for(&self, mnemonic: &str) -> usize {
        self.by_mnemonic.get(mnemonic).map_or(0, Vec::len)
    }

    #[cfg(test)]
    pub fn entry_name(&self, mnemonic: &str, idx: usize) -> Option<&'static str> {
        self.by_mnemonic
            .get(mnemonic)
            .and_then(|v| v.get(idx))
            .map(|e| e.name)
    }
}

fn build_registry() -> RuleRegistry {
    let mut registry = RuleRegistry::default();
    rules::register_all(&mut registry);
    registry
}

fn global_registry() -> &'static RuleRegistry {
    static REGISTRY: OnceLock<RuleRegistry> = OnceLock::new();
    REGISTRY.get_or_init(build_registry)
}

pub(super) fn dispatch_opcode(
    opcode: &str,
    args: &[IRExpr],
    stmt_ref: StatementRef,
    config: &SemanticLiftConfig<'_>,
) -> Option<LiftedExpr> {
    global_registry().dispatch(opcode, args, stmt_ref, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_rule(
        _sig: &OpSig,
        _args: &[IRExpr],
        _stmt_ref: StatementRef,
        _config: &SemanticLiftConfig<'_>,
    ) -> Option<LiftedExpr> {
        Some(LiftedExpr::Imm("1".to_string()))
    }

    fn second_rule(
        _sig: &OpSig,
        _args: &[IRExpr],
        _stmt_ref: StatementRef,
        _config: &SemanticLiftConfig<'_>,
    ) -> Option<LiftedExpr> {
        Some(LiftedExpr::Imm("2".to_string()))
    }

    #[test]
    fn dispatch_prefers_first_registered_rule() {
        let mut reg = RuleRegistry::default();
        reg.register("IADD3", "first", first_rule);
        reg.register("IADD3", "second", second_rule);

        let out = reg
            .dispatch(
                "IADD3",
                &[],
                StatementRef {
                    block_id: 0,
                    stmt_idx: 0,
                },
                &SemanticLiftConfig::default(),
            )
            .expect("rule should match");
        assert_eq!(out.render(), "1");
        assert_eq!(reg.entries_for("IADD3"), 2);
        assert_eq!(reg.entry_name("IADD3", 0), Some("first"));
        assert_eq!(reg.entry_name("IADD3", 1), Some("second"));
    }
}
