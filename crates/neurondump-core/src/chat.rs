//! Chat prompt rendering: a small evaluator for the subset of Go `text/template` that
//! Ollama templates use (`if`/`else if`/`range`/`with`, variables, `eq`/`ne`/`and`/`or`/
//! `not`/`len`/`slice`/`index`, trim markers), plus built-in templates for the common
//! families when a model has no Ollama template (raw GGUF files only carry Jinja).

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use crate::model::ModelHandle;
use crate::tokenizer::Vocab;

/// What the chat formatting is built from.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    /// Overrides the model's default system prompt (empty string = none).
    #[serde(default)]
    pub system: Option<String>,
    /// Go template text replacing the model's template.
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRender {
    /// `ollama`, `builtin:<family>`, `custom` or `none`.
    pub source: String,
    pub template: String,
    pub system: Option<String>,
    pub text: String,
    /// Byte range of the user prompt inside `text`.
    pub prompt_span: Option<(usize, usize)>,
}

const CHATML: &str = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n";
const LLAMA3: &str = "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
const GEMMA: &str = "<start_of_turn>user\n{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}<end_of_turn>\n<start_of_turn>model\n";
const PHI3: &str = "{{ if .System }}<|system|>\n{{ .System }}<|end|>\n{{ end }}<|user|>\n{{ .Prompt }}<|end|>\n<|assistant|>\n";

/// Built-in template for the model's family, detected from its special tokens.
pub fn builtin_template(vocab: &Vocab) -> Option<(&'static str, &'static str)> {
    let has = |s: &str| vocab.by_raw.contains_key(s);
    if has("<|start_header_id|>") {
        Some(("llama3", LLAMA3))
    } else if has("<|im_start|>") {
        Some(("chatml", CHATML))
    } else if has("<start_of_turn>") {
        Some(("gemma", GEMMA))
    } else if has("<|user|>") && has("<|assistant|>") {
        Some(("phi3", PHI3))
    } else {
        None
    }
}

/// Renders the prompt as the model's chat format.
pub fn render_chat(m: &ModelHandle, vocab: Option<&Vocab>, req: &ChatRequest) -> Result<ChatRender> {
    let (source, template) = match &req.template {
        Some(t) => ("custom".to_string(), t.clone()),
        None => match (&m.ollama.template, vocab.and_then(builtin_template)) {
            (Some(t), _) if !t.trim().is_empty() => ("ollama".to_string(), t.clone()),
            (_, Some((name, t))) => (format!("builtin:{name}"), t.to_string()),
            _ => ("none".to_string(), "{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}".to_string()),
        },
    };
    let system = match &req.system {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s.clone()),
        None => m.ollama.system.clone(),
    };
    let text = render_template(&template, system.as_deref(), &req.prompt)?;
    let prompt_span = (!req.prompt.is_empty()).then(|| text.rfind(&req.prompt).map(|i| (i, i + req.prompt.len()))).flatten();
    Ok(ChatRender { source, template, system, text, prompt_span })
}

/// Executes a Go template with Ollama's values for a single user turn.
pub fn render_template(template: &str, system: Option<&str>, prompt: &str) -> Result<String> {
    let msg = |role: &str, content: &str| {
        let mut m = BTreeMap::new();
        m.insert("Role".into(), V::Str(role.into()));
        m.insert("Content".into(), V::Str(content.into()));
        V::Map(m)
    };
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(msg("system", s));
    }
    messages.push(msg("user", prompt));
    let mut root = BTreeMap::new();
    root.insert("System".into(), V::Str(system.unwrap_or("").into()));
    root.insert("Prompt".into(), V::Str(prompt.into()));
    root.insert("Response".into(), V::Str(String::new()));
    root.insert("Suffix".into(), V::Str(String::new()));
    root.insert("Messages".into(), V::List(messages));
    let nodes = parse(template)?;
    let mut out = String::new();
    let root = V::Map(root);
    let mut ev = Eval { root: &root, vars: vec![BTreeMap::new()] };
    ev.exec(&nodes, &root, &mut out)?;
    Ok(out)
}

// --- values -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum V {
    Nil,
    Bool(bool),
    Int(i64),
    Str(String),
    List(Vec<V>),
    Map(BTreeMap<String, V>),
}

impl V {
    fn truthy(&self) -> bool {
        match self {
            V::Nil => false,
            V::Bool(b) => *b,
            V::Int(i) => *i != 0,
            V::Str(s) => !s.is_empty(),
            V::List(l) => !l.is_empty(),
            V::Map(m) => !m.is_empty(),
        }
    }

    fn render(&self) -> String {
        match self {
            V::Nil => String::new(),
            V::Bool(b) => b.to_string(),
            V::Int(i) => i.to_string(),
            V::Str(s) => s.clone(),
            V::List(l) => format!("[{}]", l.iter().map(V::render).collect::<Vec<_>>().join(" ")),
            V::Map(m) => format!("map[{}]", m.iter().map(|(k, v)| format!("{k}:{}", v.render())).collect::<Vec<_>>().join(" ")),
        }
    }

    fn field(&self, name: &str) -> V {
        match self {
            V::Map(m) => m.get(name).cloned().unwrap_or(V::Nil),
            _ => V::Nil,
        }
    }

    fn json(&self) -> serde_json::Value {
        match self {
            V::Nil => serde_json::Value::Null,
            V::Bool(b) => (*b).into(),
            V::Int(i) => (*i).into(),
            V::Str(s) => s.clone().into(),
            V::List(l) => l.iter().map(V::json).collect(),
            V::Map(m) => m.iter().map(|(k, v)| (k.clone(), v.json())).collect::<serde_json::Map<_, _>>().into(),
        }
    }
}

// --- parsing ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Arg {
    /// `.A.B` (empty = `.`), relative to dot.
    Dot(Vec<String>),
    /// `$x.A` (`$` alone = root).
    Var(String, Vec<String>),
    Str(String),
    Int(i64),
    Bool(bool),
    Nil,
    Ident(String),
    Sub(Box<Pipe>),
}

/// Commands separated by `|`; each command is its arguments.
#[derive(Debug, Clone)]
struct Pipe {
    cmds: Vec<Vec<Arg>>,
    /// `$a := …` / `$a, $b := …` (range) / `$a = …`.
    decl: Vec<String>,
}

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Out(Pipe),
    If(Vec<(Pipe, Vec<Node>)>, Option<Vec<Node>>),
    Range(Pipe, Vec<Node>, Option<Vec<Node>>),
    With(Pipe, Vec<Node>, Option<Vec<Node>>),
}

enum Tok {
    Text(String),
    Action(String),
}

fn lex(src: &str) -> Result<Vec<Tok>> {
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

fn find_close(s: &str) -> Option<usize> {
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

fn words(s: &str) -> Result<Vec<String>> {
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

fn unquote(s: &str) -> String {
    if let Some(r) = s.strip_prefix('`').and_then(|r| r.strip_suffix('`')) {
        return r.to_string();
    }
    serde_json::from_str::<String>(s).unwrap_or_else(|_| s.trim_matches('"').to_string())
}

fn path(s: &str) -> Vec<String> {
    s.split('.').filter(|p| !p.is_empty()).map(String::from).collect()
}

fn parse_arg(w: &str) -> Result<Arg> {
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

fn parse_pipe(s: &str) -> Result<Pipe> {
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
fn parse_list(toks: &[Tok], i: &mut usize, stops: &[&str]) -> Result<(Vec<Node>, Option<String>)> {
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

fn parse(src: &str) -> Result<Vec<Node>> {
    let toks = lex(src)?;
    let mut i = 0;
    let (nodes, stop) = parse_list(&toks, &mut i, &[])?;
    if let Some(s) = stop {
        bail!("unexpected {{{{ {s} }}}}");
    }
    Ok(nodes)
}

// --- evaluation -------------------------------------------------------------------------

struct Eval<'a> {
    root: &'a V,
    vars: Vec<BTreeMap<String, V>>,
}

impl Eval<'_> {
    fn var(&self, name: &str) -> V {
        if name.is_empty() {
            return self.root.clone();
        }
        for scope in self.vars.iter().rev() {
            if let Some(v) = scope.get(name) {
                return v.clone();
            }
        }
        V::Nil
    }

    fn set(&mut self, name: &str, v: V) {
        // `=` assigns an existing variable; `:=` declares in the current scope.
        for scope in self.vars.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.into(), v);
                return;
            }
        }
        if let Some(s) = self.vars.last_mut() {
            s.insert(name.into(), v);
        }
    }

    fn arg(&mut self, a: &Arg, dot: &V) -> Result<V> {
        let walk = |mut v: V, p: &[String]| {
            for f in p {
                v = v.field(f);
            }
            v
        };
        Ok(match a {
            Arg::Dot(p) => walk(dot.clone(), p),
            Arg::Var(n, p) => walk(self.var(n), p),
            Arg::Str(s) => V::Str(s.clone()),
            Arg::Int(i) => V::Int(*i),
            Arg::Bool(b) => V::Bool(*b),
            Arg::Nil => V::Nil,
            Arg::Sub(p) => self.pipe(p, dot)?,
            Arg::Ident(f) => self.call(f, &[], dot)?,
        })
    }

    fn call(&mut self, f: &str, args: &[Arg], dot: &V) -> Result<V> {
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.arg(a, dot)?);
        }
        let ord = |a: &V, b: &V| -> Option<std::cmp::Ordering> {
            match (a, b) {
                (V::Int(x), V::Int(y)) => Some(x.cmp(y)),
                (V::Str(x), V::Str(y)) => Some(x.cmp(y)),
                _ => None,
            }
        };
        Ok(match f {
            "eq" => {
                let (first, rest) = vals.split_first().ok_or_else(|| anyhow!("eq needs arguments"))?;
                V::Bool(rest.iter().any(|v| v == first))
            }
            "ne" => V::Bool(vals.first() != vals.get(1)),
            "lt" | "le" | "gt" | "ge" => {
                let o = match (vals.first(), vals.get(1)) {
                    (Some(a), Some(b)) => ord(a, b).ok_or_else(|| anyhow!("{f}: incomparable values"))?,
                    _ => bail!("{f} needs two arguments"),
                };
                V::Bool(match f {
                    "lt" => o.is_lt(),
                    "le" => o.is_le(),
                    "gt" => o.is_gt(),
                    _ => o.is_ge(),
                })
            }
            "and" => {
                let mut last = V::Bool(true);
                for v in vals {
                    if !v.truthy() {
                        return Ok(v);
                    }
                    last = v;
                }
                last
            }
            "or" => {
                let mut last = V::Bool(false);
                for v in vals {
                    if v.truthy() {
                        return Ok(v);
                    }
                    last = v;
                }
                last
            }
            "not" => V::Bool(!vals.first().is_some_and(V::truthy)),
            "len" => V::Int(match vals.first() {
                Some(V::List(l)) => l.len() as i64,
                Some(V::Str(s)) => s.len() as i64,
                Some(V::Map(m)) => m.len() as i64,
                _ => 0,
            }),
            "slice" => {
                let idx = |i: usize| match vals.get(i) {
                    Some(V::Int(n)) => Some(*n as usize),
                    _ => None,
                };
                match vals.first() {
                    Some(V::List(l)) => {
                        let a = idx(1).unwrap_or(0).min(l.len());
                        let b = idx(2).unwrap_or(l.len()).min(l.len()).max(a);
                        V::List(l[a..b].to_vec())
                    }
                    Some(V::Str(s)) => {
                        let a = idx(1).unwrap_or(0).min(s.len());
                        let b = idx(2).unwrap_or(s.len()).min(s.len()).max(a);
                        V::Str(s.get(a..b).unwrap_or("").to_string())
                    }
                    _ => V::Nil,
                }
            }
            "index" => {
                let mut v = vals.first().cloned().unwrap_or(V::Nil);
                for k in vals.iter().skip(1) {
                    v = match (&v, k) {
                        (V::List(l), V::Int(i)) => l.get(*i as usize).cloned().unwrap_or(V::Nil),
                        (V::Map(m), V::Str(s)) => m.get(s).cloned().unwrap_or(V::Nil),
                        _ => V::Nil,
                    };
                }
                v
            }
            "print" => V::Str(vals.iter().map(V::render).collect::<Vec<_>>().join("")),
            "println" => V::Str(vals.iter().map(V::render).collect::<Vec<_>>().join(" ") + "\n"),
            "printf" => {
                let fmt = vals.first().map(V::render).unwrap_or_default();
                let mut out = String::new();
                let mut args = vals.iter().skip(1);
                let mut chars = fmt.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '%' {
                        match chars.next() {
                            Some('%') => out.push('%'),
                            Some('q') => out.push_str(&serde_json::to_string(&args.next().map(V::render).unwrap_or_default()).unwrap_or_default()),
                            Some(_) => out.push_str(&args.next().map(V::render).unwrap_or_default()),
                            None => out.push('%'),
                        }
                    } else {
                        out.push(c);
                    }
                }
                V::Str(out)
            }
            "json" => V::Str(vals.first().map(|v| v.json().to_string()).unwrap_or_default()),
            "html" | "js" | "urlquery" => V::Str(vals.iter().map(V::render).collect()),
            "currentDate" => V::Str(String::new()),
            "toTypeScriptType" => V::Str(String::new()),
            other => bail!("template function {other:?} is not supported"),
        })
    }

    fn pipe(&mut self, p: &Pipe, dot: &V) -> Result<V> {
        let mut prev: Option<V> = None;
        for cmd in &p.cmds {
            let Some((head, rest)) = cmd.split_first() else { continue };
            let v = match head {
                Arg::Ident(f) => {
                    let mut args: Vec<Arg> = rest.to_vec();
                    if let Some(pv) = prev.take() {
                        // The previous command's result becomes the last argument.
                        self.vars.push(BTreeMap::from([("\u{0}pipe".to_string(), pv)]));
                        args.push(Arg::Var("\u{0}pipe".into(), vec![]));
                        let r = self.call(f, &args, dot);
                        self.vars.pop();
                        r?
                    } else {
                        self.call(f, &args, dot)?
                    }
                }
                a => self.arg(a, dot)?,
            };
            prev = Some(v);
        }
        let v = prev.unwrap_or(V::Nil);
        if let Some(name) = p.decl.first() {
            self.set(name, v.clone());
        }
        Ok(v)
    }

    fn exec(&mut self, nodes: &[Node], dot: &V, out: &mut String) -> Result<()> {
        for n in nodes {
            match n {
                Node::Text(t) => out.push_str(t),
                Node::Out(p) => {
                    let v = self.pipe(p, dot)?;
                    if p.decl.is_empty() {
                        out.push_str(&v.render());
                    }
                }
                Node::If(branches, els) => {
                    let mut done = false;
                    for (cond, body) in branches {
                        self.vars.push(BTreeMap::new());
                        let c = self.pipe(cond, dot)?;
                        if c.truthy() {
                            self.exec(body, dot, out)?;
                            self.vars.pop();
                            done = true;
                            break;
                        }
                        self.vars.pop();
                    }
                    if !done {
                        if let Some(b) = els {
                            self.exec(b, dot, out)?;
                        }
                    }
                }
                Node::With(p, body, els) => {
                    self.vars.push(BTreeMap::new());
                    let v = self.pipe(p, dot)?;
                    if v.truthy() {
                        self.exec(body, &v, out)?;
                    } else if let Some(b) = els {
                        self.exec(b, dot, out)?;
                    }
                    self.vars.pop();
                }
                Node::Range(p, body, els) => {
                    let decl = p.decl.clone();
                    let inner = Pipe { cmds: p.cmds.clone(), decl: vec![] };
                    let v = self.pipe(&inner, dot)?;
                    let items: Vec<(V, V)> = match v {
                        V::List(l) => l.into_iter().enumerate().map(|(i, x)| (V::Int(i as i64), x)).collect(),
                        V::Map(m) => m.into_iter().map(|(k, x)| (V::Str(k), x)).collect(),
                        V::Int(n) => (0..n).map(|i| (V::Int(i), V::Int(i))).collect(),
                        _ => vec![],
                    };
                    if items.is_empty() {
                        if let Some(b) = els {
                            self.exec(b, dot, out)?;
                        }
                        continue;
                    }
                    for (k, x) in items {
                        let mut scope = BTreeMap::new();
                        match decl.as_slice() {
                            [one] => {
                                scope.insert(one.clone(), x.clone());
                            }
                            [a, b, ..] => {
                                scope.insert(a.clone(), k);
                                scope.insert(b.clone(), x.clone());
                            }
                            [] => {}
                        }
                        self.vars.push(scope);
                        self.exec(body, &x, out)?;
                        self.vars.pop();
                    }
                }
            }
        }
        Ok(())
    }
}
