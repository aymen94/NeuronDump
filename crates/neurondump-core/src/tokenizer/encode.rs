//! Text -> token ids for BPE (byte-level), SPM and WPM vocabularies, following llama.cpp's
//! `llama-vocab.cpp` algorithms. Used for corpus-driven vocabulary trimming.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{bail, Result};
use regex::Regex;
use serde::Serialize;

use super::{byte_encoder, TokenType, TokenizerModel, Vocab};

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

    fn bpe(&self, text: &str, ids: &mut Vec<u32>, mut used: Option<&mut HashSet<u32>>) {
        let enc = byte_encoder();
        for word in self.pre_tokenize(text) {
            let mapped: String = word.bytes().map(|b| enc[&b]).collect();
            if self.ignore_merges {
                if let Some(&id) = self.vocab.by_raw.get(mapped.as_str()) {
                    ids.push(id);
                    continue;
                }
            }
            let mut syms: Vec<String> = mapped.chars().map(String::from).collect();
            loop {
                let mut best: Option<(u32, usize)> = None;
                for i in 0..syms.len().saturating_sub(1) {
                    let key = format!("{} {}", syms[i], syms[i + 1]);
                    if let Some(&r) = self.ranks.get(key.as_str()) {
                        if best.is_none_or(|(br, _)| r < br) {
                            best = Some((r, i));
                        }
                    }
                }
                let Some((_, i)) = best else { break };
                let merged = format!("{}{}", syms[i], syms[i + 1]);
                if let (Some(u), Some(&id)) = (used.as_deref_mut(), self.vocab.by_raw.get(merged.as_str())) {
                    u.insert(id);
                }
                syms[i] = merged;
                syms.remove(i + 1);
            }
            for s in syms {
                match self.vocab.by_raw.get(s.as_str()) {
                    Some(&id) => ids.push(id),
                    None => {
                        for ch in s.chars() {
                            if let Some(&id) = self.vocab.by_raw.get(ch.to_string().as_str()) {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    fn spm(&self, text: &str, first: bool, ids: &mut Vec<u32>, mut used: Option<&mut HashSet<u32>>) {
        let mut s = String::with_capacity(text.len() + 3);
        if first && self.add_space_prefix {
            s.push('\u{2581}');
        }
        s.push_str(&text.replace(' ', "\u{2581}"));
        // Symbols as (start, len, prev, next) over `s`.
        let mut start: Vec<usize> = Vec::new();
        let mut len: Vec<usize> = Vec::new();
        for (i, c) in s.char_indices() {
            start.push(i);
            len.push(c.len_utf8());
        }
        let n = start.len();
        if n == 0 {
            return;
        }
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 < n as isize { i + 1 } else { -1 }).collect();
        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Bigram>, l: isize, r: isize, start: &[usize], len: &[usize]| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            let piece = &s[start[l]..start[l] + len[l] + len[r]];
            if let Some(&id) = self.vocab.by_raw.get(piece) {
                let score = self.vocab.tokens[id as usize].score.unwrap_or(0.0);
                heap.push(Bigram { score, left: l, right: r, size: piece.len() });
            }
        };
        for i in 1..n {
            try_add(&mut heap, i as isize - 1, i as isize, &start, &len);
        }
        while let Some(b) = heap.pop() {
            if len[b.left] == 0 || len[b.right] == 0 || len[b.left] + len[b.right] != b.size {
                continue;
            }
            len[b.left] += len[b.right];
            len[b.right] = 0;
            next[b.left] = next[b.right];
            if next[b.right] >= 0 {
                prev[next[b.right] as usize] = b.left as isize;
            }
            if let (Some(u), Some(&id)) = (used.as_deref_mut(), self.vocab.by_raw.get(&s[start[b.left]..start[b.left] + len[b.left]])) {
                u.insert(id);
            }
            try_add(&mut heap, prev[b.left], b.left as isize, &start, &len);
            try_add(&mut heap, b.left as isize, next[b.left], &start, &len);
        }
        let mut i: isize = 0;
        while i >= 0 {
            let iu = i as usize;
            let piece = &s[start[iu]..start[iu] + len[iu]];
            match self.vocab.by_raw.get(piece) {
                Some(&id) => ids.push(id),
                None => {
                    for b in piece.bytes() {
                        if let Some(&id) = self.vocab.by_raw.get(format!("<0x{b:02X}>").as_str()) {
                            ids.push(id);
                        } else if let Some(u) = self.vocab.summary.specials.iter().find(|s| s.role == "unk") {
                            ids.push(u.id);
                        }
                    }
                }
            }
            i = next[iu];
        }
    }

    fn wpm(&self, text: &str, ids: &mut Vec<u32>) {
        let unk = self.vocab.summary.specials.iter().find(|s| s.role == "unk").map(|s| s.id);
        let lower = text.to_lowercase();
        let mut words: Vec<String> = Vec::new();
        let mut cur = String::new();
        for c in lower.chars() {
            if c.is_whitespace() {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            } else if c.is_ascii_punctuation() || (!c.is_alphanumeric() && !c.is_whitespace()) || super::is_cjk(c) {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
                words.push(c.to_string());
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            words.push(cur);
        }
        for w in words {
            let chars: Vec<(usize, char)> = w.char_indices().collect();
            let mut i = 0;
            let mut out = Vec::new();
            let mut ok = true;
            while i < chars.len() {
                let mut found = None;
                let mut j = chars.len();
                while j > i {
                    let end = chars.get(j).map_or(w.len(), |c| c.0);
                    let sub = &w[chars[i].0..end];
                    let key = if i == 0 { format!("\u{2581}{sub}") } else { sub.to_string() };
                    if let Some(&id) = self.vocab.by_raw.get(key.as_str()) {
                        found = Some((id, j));
                        break;
                    }
                    j -= 1;
                }
                match found {
                    Some((id, j)) => {
                        out.push(id);
                        i = j;
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                ids.extend(out);
            } else if let Some(u) = unk {
                ids.push(u);
            }
        }
    }
}
