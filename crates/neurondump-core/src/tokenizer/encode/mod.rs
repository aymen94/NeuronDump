//! Text -> token ids for BPE (byte-level), SPM and WPM vocabularies, following llama.cpp's
//! `llama-vocab.cpp` algorithms. Used for corpus-driven vocabulary trimming.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{bail, Result};
use regex::Regex;
use serde::Serialize;

use super::{byte_encoder, TokenType, TokenizerModel, Vocab};

mod algos;

const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";
const LLAMA3_RE: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";
const QWEN2_RE: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

#[derive(Debug, Clone, Serialize)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub pieces: Vec<String>,
}

pub struct Encoder<'a> {
    vocab: &'a Vocab,
    ranks: HashMap<&'a str, u32>,
    pre_re: Option<Regex>,
    ignore_merges: bool,
    specials_re: Option<Regex>,
    add_space_prefix: bool,
    pub warnings: Vec<String>,
}

#[derive(PartialEq)]
struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    /// Max-heap on score; ties go to the leftmost pair.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal).then_with(|| other.left.cmp(&self.left))
    }
}

impl<'a> Encoder<'a> {
    pub fn new(vocab: &'a Vocab) -> Result<Self> {
        let mut warnings = Vec::new();
        let pre = vocab.summary.pre.clone().unwrap_or_default();
        let (pre_re, ignore_merges) = match vocab.model {
            TokenizerModel::Bpe => {
                let (pat, ignore) = match pre.as_str() {
                    "llama3" | "llama-bpe" | "llama-v3" | "falcon3" | "pixtral" | "dbrx" | "tekken" | "deepseek-llm"
                    | "deepseek-coder" | "deepseek-v3" | "smaug-bpe" | "gpt-4o" | "superbpe" => (LLAMA3_RE, true),
                    "qwen2" | "qwen35" | "qwen3" | "deepseek-r1-qwen" | "stablelm2" | "hunyuan" | "megrez" => (QWEN2_RE, false),
                    "" | "default" | "gpt-2" | "gpt2" | "starcoder" | "refact" | "olmo" | "smollm" | "jais" => (GPT2_RE, false),
                    other => {
                        warnings.push(format!("pre-tokenizer {other:?} not implemented exactly; using the llama3 rules"));
                        (LLAMA3_RE, false)
                    }
                };
                (Some(Regex::new(pat)?), ignore)
            }
            TokenizerModel::Spm | TokenizerModel::Wpm => (None, false),
            TokenizerModel::Other => bail!("tokenization for tokenizer model {:?} is not implemented", vocab.model_name),
        };
        let ranks = vocab.merges.iter().enumerate().map(|(i, m)| (m.as_str(), i as u32)).collect();
        let mut specials: Vec<&str> = vocab
            .tokens
            .iter()
            .filter(|t| matches!(t.token_type, TokenType::Control | TokenType::UserDefined) && !t.raw.is_empty())
            .map(|t| t.raw.as_str())
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let specials_re = if specials.is_empty() {
            None
        } else {
            Some(Regex::new(&specials.iter().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|"))?)
        };
        Ok(Self {
            vocab,
            ranks,
            pre_re,
            ignore_merges,
            specials_re,
            add_space_prefix: vocab.summary.add_space_prefix.unwrap_or(vocab.model == TokenizerModel::Spm),
            warnings,
        })
    }

    /// Encodes `text`. Every token touched along the way (including intermediate BPE/SPM
    /// merge results) is added to `used` when given.
    pub fn encode(&self, text: &str, add_special: bool, mut used: Option<&mut HashSet<u32>>) -> Encoding {
        let mut ids = Vec::new();
        let special = |role: &str| self.vocab.summary.specials.iter().find(|s| s.role == role).map(|s| s.id);
        // llama.cpp defaults: BOS for SPM, WPM ([CLS]) and llama3-style BPE; EOS ([SEP]) for WPM.
        let pre = self.vocab.summary.pre.as_deref().unwrap_or("");
        let default_bos = match self.vocab.model {
            TokenizerModel::Spm | TokenizerModel::Wpm => true,
            TokenizerModel::Bpe => self.ignore_merges || pre.starts_with("llama") || pre == "falcon3",
            TokenizerModel::Other => false,
        };
        let add_bos = add_special && self.vocab.summary.add_bos.unwrap_or(default_bos);
        let add_eos = add_special && self.vocab.summary.add_eos.unwrap_or(self.vocab.model == TokenizerModel::Wpm);
        if add_bos {
            if let Some(b) = special("bos").or_else(|| special("cls")) {
                ids.push(b);
            }
        }
        let mut first = true;
        let mut last = 0;
        let mut segments: Vec<(bool, &str)> = Vec::new();
        if let Some(re) = &self.specials_re {
            for m in re.find_iter(text) {
                if m.start() > last {
                    segments.push((false, &text[last..m.start()]));
                }
                segments.push((true, m.as_str()));
                last = m.end();
            }
        }
        if last < text.len() {
            segments.push((false, &text[last..]));
        }
        for (special, seg) in segments {
            if special {
                if let Some(&id) = self.vocab.by_raw.get(seg) {
                    ids.push(id);
                }
                continue;
            }
            match self.vocab.model {
                TokenizerModel::Bpe => self.bpe(seg, &mut ids, used.as_deref_mut()),
                TokenizerModel::Spm => self.spm(seg, first, &mut ids, used.as_deref_mut()),
                TokenizerModel::Wpm => self.wpm(seg, &mut ids),
                TokenizerModel::Other => {}
            }
            first = false;
        }
        if add_eos {
            if let Some(e) = special("eos").or_else(|| special("sep")) {
                ids.push(e);
            }
        }
        if let Some(u) = used {
            u.extend(ids.iter().copied());
        }
        let pieces = ids.iter().map(|&i| self.vocab.tokens.get(i as usize).map(|t| t.text.clone()).unwrap_or_default()).collect();
        Encoding { ids, pieces }
    }

    fn pre_tokenize<'t>(&self, text: &'t str) -> Vec<&'t str> {
        let Some(re) = &self.pre_re else { return vec![text] };
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let Some(m) = re.find_at(text, pos) else { break };
            if m.start() > pos {
                out.push(&text[pos..m.start()]);
            }
            let mut end = m.end();
            let s = m.as_str();
            // Emulate `\s+(?!\S)`: a whitespace run followed by non-space leaves its last
            // char to prefix the next word.
            if s.chars().all(char::is_whitespace) && !s.ends_with(['\r', '\n']) && s.chars().count() > 1 {
                if let Some(next) = text[end..].chars().next() {
                    if !next.is_whitespace() {
                        let last_len = s.chars().last().map_or(0, char::len_utf8);
                        end -= last_len;
                    }
                }
            }
            if end == m.start() {
                end = m.end();
            }
            out.push(&text[m.start()..end]);
            pos = end;
        }
        out
    }

}
