//! Vocabulary extraction: readable token text, classification, special tokens, search.

mod encode;
mod keywords;

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use anyhow::{bail, Result};
use regex::Regex;
use serde::Serialize;

use neurondump_gguf::GgufFile;

pub use encode::{Encoder, Encoding};

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
    fn from_id(v: i64) -> Self {
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

pub struct Vocab {
    pub model_name: String,
    pub model: TokenizerModel,
    pub tokens: Vec<Token>,
    pub summary: VocabSummary,
    pub merges: Vec<String>,
    pub by_raw: HashMap<String, u32>,
}

/// GPT-2 `bytes_to_unicode` inverse: printable stand-in char -> byte.
fn byte_decoder() -> &'static HashMap<char, u8> {
    static M: OnceLock<HashMap<char, u8>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let c = if printable {
                b
            } else {
                n += 1;
                255 + n
            };
            if let Some(ch) = char::from_u32(c) {
                m.insert(ch, b as u8);
            }
        }
        m
    })
}

pub fn byte_encoder() -> &'static HashMap<u8, char> {
    static M: OnceLock<HashMap<u8, char>> = OnceLock::new();
    M.get_or_init(|| byte_decoder().iter().map(|(c, b)| (*b, *c)).collect())
}

fn parse_byte_token(raw: &str) -> Option<u8> {
    let hex = raw.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 {
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// Decodes a raw vocabulary entry into readable text.
pub fn decode_raw(model: TokenizerModel, raw: &str) -> String {
    match model {
        TokenizerModel::Bpe => {
            let dec = byte_decoder();
            let mut bytes = Vec::with_capacity(raw.len());
            for ch in raw.chars() {
                match dec.get(&ch) {
                    Some(b) => bytes.push(*b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(e) => e.as_bytes().iter().map(|b| if b.is_ascii() { (*b as char).to_string() } else { format!("<0x{b:02X}>") }).collect(),
            }
        }
        TokenizerModel::Spm | TokenizerModel::Wpm | TokenizerModel::Other => {
            if let Some(b) = parse_byte_token(raw) {
                if b.is_ascii() {
                    return (b as char).to_string();
                }
                return raw.to_string();
            }
            raw.replace('\u{2581}', " ")
        }
    }
}

fn reserved_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)^(<\|reserved[_a-z]*_?\d*\|>|\[unused\d+\]|<unused\d+>|<\|extra_?\d+\|>|<extra_id_\d+>|<\|?placeholder\d*\|?>|<\|dummy_?\d+\|>|<pad_?\d+>|<\|unused_?\d+\|>)$").unwrap()
    })
}

fn special_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^(<\|[^|]{1,64}\|>|<[A-Za-z_/][A-Za-z0-9_\-/:.]{0,40}>|\[[A-Z_/]{2,24}\]|<｜[^｜]{1,40}｜>)$").unwrap())
}

pub fn script_of(c: char) -> Option<&'static str> {
    let u = c as u32;
    Some(match u {
        0x41..=0x5A | 0x61..=0x7A | 0xC0..=0x24F | 0x1E00..=0x1EFF => "latin",
        0x370..=0x3FF | 0x1F00..=0x1FFF => "greek",
        0x400..=0x52F => "cyrillic",
        0x530..=0x58F => "armenian",
        0x590..=0x5FF => "hebrew",
        0x600..=0x6FF | 0x750..=0x77F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => "arabic",
        0x900..=0x97F => "devanagari",
        0x980..=0x9FF => "bengali",
        0xA00..=0xA7F => "gurmukhi",
        0xA80..=0xAFF => "gujarati",
        0xB80..=0xBFF => "tamil",
        0xC00..=0xC7F => "telugu",
        0xC80..=0xCFF => "kannada",
        0xD00..=0xD7F => "malayalam",
        0xE00..=0xE7F => "thai",
        0xE80..=0xEFF => "lao",
        0x10A0..=0x10FF => "georgian",
        0x1200..=0x137F => "ethiopic",
        0x1780..=0x17FF => "khmer",
        0x1000..=0x109F => "myanmar",
        0x3040..=0x309F => "hiragana",
        0x30A0..=0x30FF | 0x31F0..=0x31FF => "katakana",
        0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => "hangul",
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F => "han",
        _ => return None,
    })
}

fn is_cjk(c: char) -> bool {
    matches!(script_of(c), Some("han" | "hiragana" | "katakana" | "hangul"))
        || matches!(c as u32, 0x3000..=0x303F | 0xFF00..=0xFFEF)
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32, 0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x1F000..=0x1F2FF | 0x2B00..=0x2BFF | 0xFE0F | 0x200D)
}

fn code_langs_of(core: &str) -> Vec<&'static str> {
    let mut v = Vec::new();
    for (lang, words) in keywords::LANGS {
        if words.contains(&core) {
            v.push(*lang);
        }
    }
    if v.is_empty() && keywords::CODE_SYMBOLS.contains(&core) {
        v.push("symbol");
    }
    v
}

fn classify(model: TokenizerModel, raw: &str, text: &str, ty: TokenType) -> (TokenClass, Option<&'static str>, Vec<&'static str>, bool) {
    if ty == TokenType::Byte || parse_byte_token(raw).is_some() {
        return (TokenClass::Byte, None, vec![], false);
    }
    if ty == TokenType::Unused || reserved_re().is_match(raw) {
        return (TokenClass::Reserved, None, vec![], false);
    }
    if matches!(ty, TokenType::Control | TokenType::Unknown) || (ty == TokenType::UserDefined && special_re().is_match(raw)) {
        return (TokenClass::Special, None, vec![], false);
    }
    let word_start = match model {
        TokenizerModel::Wpm => !raw.starts_with("##") && raw.starts_with('\u{2581}'),
        _ => text.starts_with(' '),
    };
    if text.is_empty() {
        return (TokenClass::Mixed, None, vec![], false);
    }
    if text.chars().all(char::is_whitespace) {
        return (TokenClass::Whitespace, None, vec![], false);
    }
    let core = text.trim();
    let core = core.strip_prefix("##").unwrap_or(core);
    if core.chars().any(is_emoji) {
        return (TokenClass::Emoji, None, vec![], word_start);
    }
    if core.chars().any(is_cjk) {
        let script = core.chars().find_map(script_of);
        return (TokenClass::Cjk, script, vec![], word_start);
    }
    if core.chars().all(|c| c.is_numeric()) {
        return (TokenClass::Digit, None, vec![], word_start);
    }
    let langs = code_langs_of(core);
    if !langs.is_empty() {
        let script = core.chars().find_map(script_of);
        return (TokenClass::Code, script, langs, word_start);
    }
    if core.chars().all(|c| !c.is_alphanumeric()) {
        return (TokenClass::Punctuation, None, vec![], word_start);
    }
    if core.chars().all(char::is_alphabetic) {
        let script = core.chars().find_map(script_of).or(Some("other"));
        let class = if word_start || model == TokenizerModel::Wpm && !raw.starts_with("##") && raw.starts_with('\u{2581}') {
            TokenClass::Word
        } else {
            TokenClass::Subword
        };
        return (class, script, vec![], word_start);
    }
    (TokenClass::Mixed, core.chars().find_map(script_of), vec![], word_start)
}

const SPECIAL_KEYS: &[(&str, &str)] = &[
    ("bos", "tokenizer.ggml.bos_token_id"),
    ("eos", "tokenizer.ggml.eos_token_id"),
    ("eot", "tokenizer.ggml.eot_token_id"),
    ("eom", "tokenizer.ggml.eom_token_id"),
    ("unk", "tokenizer.ggml.unknown_token_id"),
    ("sep", "tokenizer.ggml.seperator_token_id"),
    ("sep", "tokenizer.ggml.separator_token_id"),
    ("pad", "tokenizer.ggml.padding_token_id"),
    ("cls", "tokenizer.ggml.cls_token_id"),
    ("mask", "tokenizer.ggml.mask_token_id"),
    ("fim_pre", "tokenizer.ggml.fim_pre_token_id"),
    ("fim_suf", "tokenizer.ggml.fim_suf_token_id"),
    ("fim_mid", "tokenizer.ggml.fim_mid_token_id"),
    ("fim_pad", "tokenizer.ggml.fim_pad_token_id"),
    ("fim_rep", "tokenizer.ggml.fim_rep_token_id"),
    ("fim_sep", "tokenizer.ggml.fim_sep_token_id"),
    ("fim_pre", "tokenizer.ggml.prefix_token_id"),
    ("fim_suf", "tokenizer.ggml.suffix_token_id"),
    ("fim_mid", "tokenizer.ggml.middle_token_id"),
];

/// Every metadata key holding a token id (needed when remapping ids in slimming).
pub fn special_token_keys(g: &GgufFile) -> Vec<String> {
    g.metadata.keys().filter(|k| k.starts_with("tokenizer.ggml.") && k.ends_with("_token_id")).cloned().collect()
}

impl Vocab {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model_name = g.get_str("tokenizer.ggml.model").unwrap_or("none").to_string();
        let model = match model_name.as_str() {
            "gpt2" => TokenizerModel::Bpe,
            "llama" => TokenizerModel::Spm,
            "bert" => TokenizerModel::Wpm,
            _ => TokenizerModel::Other,
        };
        let Some(raws) = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).and_then(|a| a.as_strings()) else {
            bail!("model has no tokenizer.ggml.tokens vocabulary");
        };
        let scores = g.get("tokenizer.ggml.scores").and_then(|v| v.as_array()).and_then(|a| a.as_f32s());
        let types = g.get("tokenizer.ggml.token_type").and_then(|v| v.as_array());
        let merges: Vec<String> = g
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_strings())
            .map(|m| m.to_vec())
            .unwrap_or_default();
        let mut warnings = Vec::new();
        if model == TokenizerModel::Other {
            warnings.push(format!("tokenizer model {model_name:?}: text decoding uses generic rules"));
        }

        let tokens: Vec<Token> = {
            use rayon::prelude::*;
            raws.par_iter()
                .enumerate()
                .map(|(i, raw)| {
                    let ty = types.and_then(|t| t.get_i64(i)).map_or(TokenType::Undefined, TokenType::from_id);
                    let text = decode_raw(model, raw);
                    let (class, script, code_langs, word_start) = classify(model, raw, &text, ty);
                    Token {
                        id: i as u32,
                        raw: raw.clone(),
                        text,
                        score: scores.and_then(|s| s.get(i).copied()),
                        token_type: ty,
                        class,
                        script,
                        code_langs,
                        word_start,
                    }
                })
                .collect()
        };

        let mut class_counts = BTreeMap::new();
        let mut script_counts = BTreeMap::new();
        let mut code_lang_counts = BTreeMap::new();
        let mut type_counts = BTreeMap::new();
        let mut control_tokens = Vec::new();
        for t in &tokens {
            *class_counts.entry(t.class.as_str().to_string()).or_insert(0) += 1;
            if let Some(s) = t.script {
                *script_counts.entry(s.to_string()).or_insert(0) += 1;
            }
            for l in &t.code_langs {
                *code_lang_counts.entry(l.to_string()).or_insert(0) += 1;
            }
            let tn = serde_json::to_value(t.token_type).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
            *type_counts.entry(tn).or_insert(0) += 1;
            if t.class == TokenClass::Special {
                let role = if t.raw.to_lowercase().contains("tool") || t.raw.contains("python_tag") {
                    "tool"
                } else if t.raw.contains("fim") || t.raw.contains("FIM") {
                    "fim"
                } else if t.raw.contains("think") {
                    "reasoning"
                } else if t.raw.contains("image") || t.raw.contains("vision") || t.raw.contains("video") || t.raw.contains("audio") {
                    "multimodal"
                } else {
                    "control"
                };
                control_tokens.push(SpecialToken { role: role.into(), id: t.id, text: Some(t.raw.clone()) });
            }
        }

        let mut specials = Vec::new();
        for (role, key) in SPECIAL_KEYS {
            if let Some(id) = g.get_u64(key) {
                if specials.iter().any(|s: &SpecialToken| s.role == *role && s.id as u64 == id) {
                    continue;
                }
                let text = tokens.get(id as usize).map(|t| t.raw.clone());
                if text.is_none() {
                    warnings.push(format!("{key} = {id} is outside the vocabulary"));
                }
                specials.push(SpecialToken { role: role.to_string(), id: id as u32, text });
            }
        }

        let by_raw = tokens.iter().map(|t| (t.raw.clone(), t.id)).collect();
        let summary = VocabSummary {
            model: model_name.clone(),
            pre: g.get_str("tokenizer.ggml.pre").map(str::to_string),
            size: tokens.len(),
            merges: merges.len(),
            add_bos: g.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()),
            add_eos: g.get("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()),
            add_space_prefix: g.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.as_bool()),
            class_counts,
            script_counts,
            code_lang_counts,
            type_counts,
            specials,
            control_tokens,
            warnings,
        };
        Ok(Self { model_name, model, tokens, summary, merges, by_raw })
    }

    pub fn search(&self, q: &VocabQuery) -> Result<VocabPage> {
        let re = match (&q.q, q.regex) {
            (Some(pat), true) => Some(Regex::new(pat)?),
            _ => None,
        };
        let needle = q.q.as_ref().filter(|_| !q.regex).map(|s| s.to_lowercase());
        let class = match &q.class {
            Some(c) if !c.is_empty() => Some(TokenClass::parse(c).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown token class {c:?} (classes: {})",
                    TokenClass::ALL.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(", ")
                )
            })?),
            _ => None,
        };
        let script = q.script.as_deref().filter(|s| !s.is_empty());
        let lang = q.lang.as_deref().filter(|s| !s.is_empty());
        let matches = |t: &&Token| {
            if let Some(c) = class {
                if t.class != c {
                    return false;
                }
            }
            if let Some(s) = script {
                if t.script != Some(s) {
                    return false;
                }
            }
            if let Some(l) = lang {
                if !t.code_langs.contains(&l) {
                    return false;
                }
            }
            if let Some(re) = &re {
                return re.is_match(&t.text) || re.is_match(&t.raw);
            }
            if let Some(n) = &needle {
                if let Ok(id) = n.parse::<u32>() {
                    if t.id == id {
                        return true;
                    }
                }
                return t.text.to_lowercase().contains(n.as_str()) || t.raw.to_lowercase().contains(n.as_str());
            }
            true
        };
        let mut total = 0usize;
        let mut out = Vec::new();
        let limit = q.limit.unwrap_or(100);
        let offset = q.offset.unwrap_or(0);
        for t in self.tokens.iter().filter(matches) {
            if total >= offset && out.len() < limit {
                out.push(t.clone());
            }
            total += 1;
        }
        Ok(VocabPage { total, offset, limit, tokens: out })
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize, Serialize)]
pub struct VocabQuery {
    pub q: Option<String>,
    #[serde(default)]
    pub regex: bool,
    pub class: Option<String>,
    pub script: Option<String>,
    pub lang: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VocabPage {
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
    pub tokens: Vec<Token>,
}
