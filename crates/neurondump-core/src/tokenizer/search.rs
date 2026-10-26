//! Vocabulary search: query, paging and filters.

use super::*;

impl Vocab {
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
