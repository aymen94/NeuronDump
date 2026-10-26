//! Token types and classes, tokens, special tokens, vocabulary summary, tokenizer model kinds.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    pub(super) fn from_id(v: i64) -> Self {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenClass {
    Special,
    Reserved,
    Byte,
    Whitespace,
    Digit,
    Punctuation,
    Emoji,
    Cjk,
    Code,
    Word,
    Subword,
    Mixed,
}

impl TokenClass {
    pub const ALL: [TokenClass; 12] = [
        TokenClass::Special,
        TokenClass::Reserved,
        TokenClass::Byte,
        TokenClass::Whitespace,
        TokenClass::Digit,
        TokenClass::Punctuation,
        TokenClass::Emoji,
        TokenClass::Cjk,
        TokenClass::Code,
        TokenClass::Word,
        TokenClass::Subword,
        TokenClass::Mixed,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TokenClass::Special => "special",
            TokenClass::Reserved => "reserved",
            TokenClass::Byte => "byte",
            TokenClass::Whitespace => "whitespace",
            TokenClass::Digit => "digit",
            TokenClass::Punctuation => "punctuation",
            TokenClass::Emoji => "emoji",
            TokenClass::Cjk => "cjk",
            TokenClass::Code => "code",
            TokenClass::Word => "word",
            TokenClass::Subword => "subword",
            TokenClass::Mixed => "mixed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_str() == s.to_ascii_lowercase())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Token {
    pub id: u32,
    /// Exactly as stored in the GGUF.
    pub raw: String,
    /// Decoded text (`Ġ`→space, `Ċ`→newline, `▁`→space, `<0x0A>`→newline, ...).
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    pub token_type: TokenType,
    pub class: TokenClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script: Option<&'static str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub code_langs: Vec<&'static str>,
    /// Starts a new word (leading space / `▁`).
    pub word_start: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpecialToken {
    pub role: String,
    pub id: u32,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VocabSummary {
    pub model: String,
    pub pre: Option<String>,
    pub size: usize,
    pub merges: usize,
    pub add_bos: Option<bool>,
    pub add_eos: Option<bool>,
    pub add_space_prefix: Option<bool>,
    pub class_counts: BTreeMap<String, usize>,
    pub script_counts: BTreeMap<String, usize>,
    pub code_lang_counts: BTreeMap<String, usize>,
    pub type_counts: BTreeMap<String, usize>,
    pub specials: Vec<SpecialToken>,
    /// All control / user-defined tokens (chat markers, tool tokens, FIM, reserved...).
    pub control_tokens: Vec<SpecialToken>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// Byte-level BPE (`gpt2`).
    Bpe,
    /// SentencePiece (`llama`).
    Spm,
    /// WordPiece (`bert`).
    Wpm,
    Other,
}
