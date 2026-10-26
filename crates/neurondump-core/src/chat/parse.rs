//! Lexing and parsing Go `text/template` source into nodes.

use super::*;

pub(super) fn lex(src: &str) -> Result<Vec<Tok>> {
    let mut out = Vec::new();
    let mut rest = src;
    let mut trim_next = false;
    while !rest.is_empty() {
        let Some(i) = rest.find("{{") else {
            let t = if trim_next { rest.trim_start() } else { rest };
            out.push(Tok::Text(t.to_string()));
            break;
        };
        let mut text = &rest[..i];
        if trim_next {
            text = text.trim_start();
        }
        let after = &rest[i + 2..];
        let trim_left = after.starts_with("- ") || after.starts_with("-\n") || after.starts_with("-\t") || after.starts_with("-\r");
        if trim_left {
            text = text.trim_end();
        }
        if !text.is_empty() {
            out.push(Tok::Text(text.to_string()));
        }
        let body_start = if trim_left { 1 } else { 0 };
        let body = &after[body_start..];
        // Comments may contain `}}`-free text only; actions end at the first `}}` outside strings.
        let end = find_close(body).ok_or_else(|| anyhow!("unclosed action in template"))?;
        let mut inner = &body[..end];
        trim_next = false;
        if inner.ends_with(" -") || inner.ends_with("\n-") || inner.ends_with("\t-") || inner == "-" {
            inner = &inner[..inner.len() - 1];
            trim_next = true;
        }
        let inner = inner.trim();
        if !(inner.starts_with("/*") && inner.ends_with("*/")) {
            out.push(Tok::Action(inner.to_string()));
        }
        rest = &body[end + 2..];
    }
    Ok(out)
}

pub(super) fn find_close(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut q: Option<u8> = None;
    while i < b.len() {
        let c = b[i];
        match q {
            Some(qc) => {
                if c == b'\\' && qc == b'"' {
                    i += 1;
                } else if c == qc {
                    q = None;
                }
            }
            None => {
                if c == b'"' || c == b'`' {
                    q = Some(c);
                } else if c == b'}' && i + 1 < b.len() && b[i + 1] == b'}' {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

pub(super) fn words(s: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '"' || c == '`' {
            let mut j = i + 1;
            while j < b.len() && b[j] != c {
                if c == '"' && b[j] == '\\' {
                    j += 1;
                }
                j += 1;
            }
            out.push(b[i..(j + 1).min(b.len())].iter().collect());
            i = j + 1;
            continue;
        }
        if c == '(' {
            let mut depth = 0;
            let mut j = i;
            while j < b.len() {
                match b[j] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    '"' | '`' => {
                        let q = b[j];
                        j += 1;
                        while j < b.len() && b[j] != q {
                            j += 1;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if j >= b.len() {
                bail!("unbalanced parenthesis in template action {s:?}");
            }
            out.push(b[i..=j].iter().collect());
            i = j + 1;
            continue;
        }
        if c == '|' || c == ',' {
            out.push(c.to_string());
            i += 1;
            continue;
        }
        let mut j = i;
        while j < b.len() && !b[j].is_whitespace() && !matches!(b[j], '(' | ')' | '|' | ',') {
            j += 1;
        }
        out.push(b[i..j].iter().collect());
        i = j;
    }
    Ok(out)
}

pub(super) fn unquote(s: &str) -> String {
    if let Some(r) = s.strip_prefix('`').and_then(|r| r.strip_suffix('`')) {
        return r.to_string();
    }
    serde_json::from_str::<String>(s).unwrap_or_else(|_| s.trim_matches('"').to_string())
}

pub(super) fn path(s: &str) -> Vec<String> {
    s.split('.').filter(|p| !p.is_empty()).map(String::from).collect()
}

pub(super) fn parse_arg(w: &str) -> Result<Arg> {
    Ok(if w.starts_with('(') {
        Arg::Sub(Box::new(parse_pipe(&w[1..w.len() - 1])?))
    } else if w.starts_with('"') || w.starts_with('`') {
        Arg::Str(unquote(w))
    } else if w == "." {
        Arg::Dot(vec![])
    } else if let Some(r) = w.strip_prefix('.') {
        Arg::Dot(path(r))
    } else if let Some(r) = w.strip_prefix('$') {
        let mut parts = r.splitn(2, '.');
        let name = parts.next().unwrap_or("").to_string();
        Arg::Var(name, parts.next().map(path).unwrap_or_default())
    } else if w == "true" || w == "false" {
        Arg::Bool(w == "true")
    } else if w == "nil" {
        Arg::Nil
    } else if let Ok(i) = w.parse::<i64>() {
        Arg::Int(i)
    } else {
        Arg::Ident(w.to_string())
    })
}

pub(super) fn parse_pipe(s: &str) -> Result<Pipe> {
    let mut ws = words(s)?;
    let mut decl = Vec::new();
    if let Some(pos) = ws.iter().position(|w| w == ":=" || w == "=") {
        decl = ws[..pos].iter().filter(|w| *w != ",").map(|w| w.trim_start_matches('$').to_string()).collect();
        ws = ws[pos + 1..].to_vec();
    }
    let mut cmds = Vec::new();
    let mut cur = Vec::new();
    for w in ws {
        if w == "|" {
            cmds.push(std::mem::take(&mut cur));
        } else {
            cur.push(parse_arg(&w)?);
        }
    }
    cmds.push(cur);
    Ok(Pipe { cmds, decl })
}

/// Parses until one of `stops` (`else`, `end`) and returns the nodes and the stop action.
pub(super) fn parse_list(toks: &[Tok], i: &mut usize, stops: &[&str]) -> Result<(Vec<Node>, Option<String>)> {
    let mut out = Vec::new();
    while *i < toks.len() {
        let t = &toks[*i];
        *i += 1;
        match t {
            Tok::Text(s) => out.push(Node::Text(s.clone())),
            Tok::Action(a) => {
                let (kw, rest) = match a.split_once(char::is_whitespace) {
                    Some((k, r)) => (k, r.trim()),
                    None => (a.as_str(), ""),
                };
                if stops.contains(&kw) {
                    return Ok((out, Some(a.clone())));
                }
                match kw {
                    "if" => {
                        let mut branches = vec![];
                        let mut cond = parse_pipe(rest)?;
                        let mut els = None;
                        loop {
                            let (body, stop) = parse_list(toks, i, &["else", "end"])?;
                            branches.push((cond.clone(), body));
                            let stop = stop.ok_or_else(|| anyhow!("missing {{{{ end }}}} for if"))?;
                            if stop == "end" {
                                break;
                            }
                            let r = stop.trim_start_matches("else").trim();
                            if let Some(c) = r.strip_prefix("if ") {
                                cond = parse_pipe(c)?;
                                continue;
                            }
                            let (body, stop) = parse_list(toks, i, &["end"])?;
                            stop.ok_or_else(|| anyhow!("missing {{{{ end }}}} for else"))?;
                            els = Some(body);
                            break;
                        }
                        out.push(Node::If(branches, els));
                    }
                    "range" | "with" => {
                        let p = parse_pipe(rest)?;
                        let (body, stop) = parse_list(toks, i, &["else", "end"])?;
                        let stop = stop.ok_or_else(|| anyhow!("missing {{{{ end }}}} for {kw}"))?;
                        let els = if stop == "end" {
                            None
                        } else {
                            let (b, s2) = parse_list(toks, i, &["end"])?;
                            s2.ok_or_else(|| anyhow!("missing {{{{ end }}}}"))?;
                            Some(b)
                        };
                        out.push(if kw == "range" { Node::Range(p, body, els) } else { Node::With(p, body, els) });
                    }
                    "end" | "else" => bail!("unexpected {{{{ {kw} }}}} in template"),
                    "define" | "block" | "template" => bail!("{{{{ {kw} }}}} is not supported by this template renderer"),
                    _ => out.push(Node::Out(parse_pipe(a)?)),
                }
            }
        }
    }
    Ok((out, None))
}

pub(super) fn parse(src: &str) -> Result<Vec<Node>> {
    let toks = lex(src)?;
    let mut i = 0;
    let (nodes, stop) = parse_list(&toks, &mut i, &[])?;
    if let Some(s) = stop {
        bail!("unexpected {{{{ {s} }}}}");
    }
    Ok(nodes)
}

// --- evaluation -------------------------------------------------------------------------
