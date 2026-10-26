//! `tokenize`: tokens of a prompt (with the chat template), their spans, norms and neighbours.

use super::*;

/// Aligns decoded token texts to `text`; tokens that do not match get no span.
pub(super) fn spans(vocab: &Vocab, ids: &[u32], text: &str) -> Vec<Option<(usize, usize)>> {
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0;
    let mut lost = false;
    for (i, &id) in ids.iter().enumerate() {
        let t = vocab.tokens.get(id as usize).map(|t| t.text.as_str()).unwrap_or("");
        if lost || t.is_empty() {
            out.push(None);
            continue;
        }
        let rest = &text[pos..];
        if rest.starts_with(t) {
            out.push(Some((pos, pos + t.len())));
            pos += t.len();
        } else if let Some(stripped) = t.strip_prefix(' ').filter(|s| rest.starts_with(*s) && !s.is_empty()) {
            // SentencePiece space prefix that is not in the text.
            out.push(Some((pos, pos + stripped.len())));
            pos += stripped.len();
        } else if i == 0 {
            // Added BOS / CLS.
            out.push(None);
        } else {
            lost = true;
            out.push(None);
        }
    }
    out
}

pub fn tokenize(m: &ModelHandle, vocab: &Vocab, req: &TokenizeRequest, banned: &[(u32, u32)], p: &dyn Progress) -> Result<TokenizeReport> {
    let started = Instant::now();
    let g = &m.gguf;
    let enc = Encoder::new(vocab)?;
    let mut warnings = enc.warnings.clone();
    let chat = if req.chat {
        Some(render_chat(m, Some(vocab), &ChatRequest { prompt: req.text.clone(), system: req.system.clone(), template: req.template.clone() })?)
    } else {
        None
    };
    if chat.as_ref().is_some_and(|c| c.source == "none") {
        warnings.push("this model has no recognizable chat template: the text is used as is".into());
    }
    let full_text = chat.as_ref().map(|c| c.text.clone()).unwrap_or_else(|| req.text.clone());
    let e = enc.encode(&full_text, true, None);
    let sp = spans(vocab, &e.ids, &full_text);
    let user = chat.as_ref().and_then(|c| c.prompt_span).unwrap_or((0, full_text.len()));
    let emb = g.tensor("token_embd.weight");
    let out_t = g.tensor("output.weight");
    let tied = out_t.is_none() && emb.is_some();
    let has_head = out_t.is_some() || (tied && !matches!(m.architecture(), "bert" | "nomic-bert" | "jina-bert-v2"));
    let uniq: Vec<u32> = e.ids.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let mut nn = HashMap::new();
    if let (Some(t), true) = (emb, req.neighbors > 0) {
        let q: Vec<u32> = uniq.iter().copied().take(64).collect();
        if uniq.len() > 64 {
            warnings.push(format!("nearest neighbours computed for the first 64 of {} distinct tokens", uniq.len()));
        }
        nn = nearest_rows(g, t, &q, req.neighbors.min(32), p)?;
    }
    let mut emb_norm: HashMap<u32, f32> = HashMap::new();
    let mut out_norm: HashMap<u32, f32> = HashMap::new();
    for &id in &uniq {
        if let Some(t) = emb.filter(|t| (id as u64) < t.n_rows()) {
            emb_norm.insert(id, norm(&row(g, t, id)?));
        }
        let ot = if tied { emb } else { out_t };
        if let Some(t) = ot.filter(|t| (id as u64) < t.n_rows() && has_head) {
            out_norm.insert(id, if tied { emb_norm.get(&id).copied().unwrap_or(0.0) } else { norm(&row(g, t, id)?) });
        }
    }
    let text_of = |id: u32| vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default();
    let tokens = e
        .ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let tok = &vocab.tokens[id as usize];
            let special = matches!(tok.token_type, TokenType::Control | TokenType::UserDefined);
            let span = sp[i];
            let added = match span {
                None => true,
                Some((a, b)) => chat.is_some() && (b <= user.0 || a >= user.1),
            };
            let is_banned = banned.iter().any(|(a, b)| id >= *a && id <= *b);
            TokenRow {
                index: i,
                id,
                raw: tok.raw.clone(),
                text: tok.text.clone(),
                class: tok.class,
                token_type: tok.token_type,
                special,
                added,
                span,
                embedding_norm: emb_norm.get(&id).copied(),
                output_norm: out_norm.get(&id).copied(),
                in_output: has_head && out_norm.get(&id).is_some_and(|n| *n > 0.0) && !is_banned,
                banned: is_banned,
                neighbors: nn.get(&id).map(|v| v.iter().map(|(n, c)| Neighbor { id: *n, text: text_of(*n), cosine: *c }).collect()).unwrap_or_default(),
            }
        })
        .collect();
    Ok(TokenizeReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        tokenizer: vocab.model_name.clone(),
        input: req.text.clone(),
        chat,
        full_text,
        tokens,
        tied_embeddings: tied,
        has_output_head: has_head,
        embedding_tensor: emb.map(|t| t.name.clone()),
        warnings,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

// --- embedding map ------------------------------------------------------------------------
