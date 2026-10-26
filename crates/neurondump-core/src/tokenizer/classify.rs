//! Token classification (word, code, emoji, script, special...) and the special-token metadata keys.

use super::*;

pub(super) fn reserved_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)^(<\|reserved[_a-z]*_?\d*\|>|\[unused\d+\]|<unused\d+>|<\|extra_?\d+\|>|<extra_id_\d+>|<\|?placeholder\d*\|?>|<\|dummy_?\d+\|>|<pad_?\d+>|<\|unused_?\d+\|>)$").unwrap()
    })
}

pub(super) fn special_re() -> &'static Regex {
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

pub(super) fn is_cjk(c: char) -> bool {
    matches!(script_of(c), Some("han" | "hiragana" | "katakana" | "hangul"))
        || matches!(c as u32, 0x3000..=0x303F | 0xFF00..=0xFFEF)
}

pub(super) fn is_emoji(c: char) -> bool {
    matches!(c as u32, 0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x1F000..=0x1F2FF | 0x2B00..=0x2BFF | 0xFE0F | 0x200D)
}

pub(super) fn code_langs_of(core: &str) -> Vec<&'static str> {
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

pub(super) fn classify(model: TokenizerModel, raw: &str, text: &str, ty: TokenType) -> (TokenClass, Option<&'static str>, Vec<&'static str>, bool) {
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

pub(super) const SPECIAL_KEYS: &[(&str, &str)] = &[
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
