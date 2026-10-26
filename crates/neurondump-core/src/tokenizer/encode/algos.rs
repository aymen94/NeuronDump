//! The three segmentation algorithms: byte-level BPE merges, SentencePiece (SPM) and WordPiece (WPM).

use super::*;

impl<'a> Encoder<'a> {
    pub(super) fn bpe(&self, text: &str, ids: &mut Vec<u32>, mut used: Option<&mut HashSet<u32>>) {
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

    pub(super) fn spm(&self, text: &str, first: bool, ids: &mut Vec<u32>, mut used: Option<&mut HashSet<u32>>) {
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

    pub(super) fn wpm(&self, text: &str, ids: &mut Vec<u32>) {
        let unk = self.vocab.summary.specials.iter().find(|s| s.role == "unk").map(|s| s.id);
        let lower = text.to_lowercase();
        let mut words: Vec<String> = Vec::new();
        let mut cur = String::new();
        for c in lower.chars() {
            if c.is_whitespace() {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            } else if c.is_ascii_punctuation() || (!c.is_alphanumeric() && !c.is_whitespace()) || super::super::is_cjk(c) {
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
