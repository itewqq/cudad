mod address;
mod basic;
mod bitwise;
mod memory;
mod predicate;

use crate::semantic_lift::registry::RuleRegistry;

pub(super) fn register_all(registry: &mut RuleRegistry) {
    basic::register(registry);
    memory::register(registry);
    predicate::register(registry);
    bitwise::register(registry);
    address::register(registry);
}
