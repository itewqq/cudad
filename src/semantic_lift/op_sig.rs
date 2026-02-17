#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct OpSig {
    pub raw_opcode: String,
    pub mnemonic: String,
    pub mods: Vec<String>,
}

impl OpSig {
    pub fn parse(opcode: &str) -> Self {
        let mut parts = opcode.split('.');
        let mnemonic = parts.next().unwrap_or_default().to_string();
        let mods = parts.map(|m| m.to_string()).collect::<Vec<_>>();
        Self {
            raw_opcode: opcode.to_string(),
            mnemonic,
            mods,
        }
    }

    #[allow(dead_code)]
    pub fn has_mod(&self, needle: &str) -> bool {
        self.mods.iter().any(|m| m == needle)
    }
}
