//! Chat templates as decision trees: Jinja (`tokenizer.chat_template*`) and Ollama's
//! Go `text/template` layer. Not an evaluator — a structural parser that extracts the
//! branches, the variables they read and the special tokens they emit.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use regex::Regex;
use serde::Serialize;

use neurondump_gguf::GgufFile;

use crate::model::ModelHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Syntax {
    Jinja,
    GoTemplate,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateNode {
    /// root, if, branch, for, range, with, set, macro, call, define, template, emit, filter, ...
    pub kind: String,
    /// For branches: `if`, `elif`, `else` (Go: `if`, `else if`, `else`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vars: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub emits: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TemplateNode>,
}

impl TemplateNode {
    fn new(kind: &str) -> Self {
        Self { kind: kind.into(), branch: None, condition: None, vars: vec![], emits: vec![], text: None, children: vec![] }
    }

    pub fn count_decisions(&self) -> usize {
        let own = usize::from(matches!(self.kind.as_str(), "branch" | "for" | "range" | "with"));
        own + self.children.iter().map(TemplateNode::count_decisions).sum::<usize>()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateTree {
    pub name: String,
    pub syntax: Syntax,
    pub source: String,
    pub root: TemplateNode,
    pub variables: BTreeSet<String>,
    pub special_tokens: BTreeSet<String>,
    pub decisions: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateReport {
    pub templates: Vec<TemplateTree>,
    pub ollama_system: Option<String>,
    pub ollama_params: Option<serde_json::Value>,
}

enum Piece<'a> {
    Text(&'a str),
    Expr(&'a str),
    Stmt(&'a str),
}

fn lex<'a>(src: &'a str, open_expr: &str, open_stmt: Option<&str>, comment: (&str, &str), close_expr: &str, close_stmt: &str) -> Vec<Piece<'a>> {
    let mut out = Vec::new();
    let mut rest = src;
    loop {
        let candidates = [Some(open_expr), open_stmt, Some(comment.0)];
        let next = candidates
            .iter()
            .flatten()
            .filter_map(|o| rest.find(o).map(|i| (i, *o)))
            .min_by_key(|(i, o)| (*i, std::cmp::Reverse(o.len())));
        let Some((i, open)) = next else {
            if !rest.is_empty() {
                out.push(Piece::Text(rest));
            }
            break;
        };
        if i > 0 {
            out.push(Piece::Text(&rest[..i]));
        }
        let after = &rest[i + open.len()..];
        let close = if open == comment.0 {
            comment.1
        } else if Some(open) == open_stmt {
            close_stmt
        } else {
            close_expr
        };
        let Some(j) = after.find(close) else {
            out.push(Piece::Text(&rest[i..]));
            break;
        };
        let inner = after[..j].trim_start_matches(['-', '+', '~']).trim_end_matches(['-', '+', '~']).trim();
        if open == comment.0 {
            // dropped
        } else if Some(open) == open_stmt {
            out.push(Piece::Stmt(inner));
        } else {
            out.push(Piece::Expr(inner));
        }
        rest = &after[j + close.len()..];
    }
    out
}

fn strings_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#""((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|`([^`]*)`"#).unwrap())
}

fn generic_special_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"<\|[^|<>\s]{1,48}\|>|<｜[^｜]{1,48}｜>|\[/?[A-Z_]{2,24}\]|</?(?:s|think|tool_call|tool_response|tools|start_of_turn|end_of_turn|bos|eos|image|answer|reasoning)>").unwrap())
}

fn ident_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r#"\$?\.?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\s*['"][^'"]+['"]\s*\])*|\$\.[A-Za-z_][A-Za-z0-9_.]*|\$[A-Za-z_][A-Za-z0-9_]*"#).unwrap()
    })
}

const JINJA_WORDS: &[&str] = &[
    "and", "or", "not", "in", "is", "if", "else", "elif", "none", "None", "true", "false", "True", "False",
    "defined", "undefined", "string", "mapping", "iterable", "number", "sequence", "equalto", "boolean",
    "for", "recursive", "loop", "namespace", "range", "raise_exception", "length", "trim", "tojson", "upper",
    "lower", "items", "strip", "split", "startswith", "endswith", "rstrip", "lstrip", "join", "replace",
    "default", "first", "last", "reverse", "selectattr", "rejectattr", "map", "list", "count", "unique",
    "safe", "format", "capitalize", "title", "indent", "with", "without", "context", "keys", "values",
    "get", "strftime_now", "lstrip_blocks", "find", "abs", "int", "float", "dictsort", "batch", "slice",
    "attribute", "tojson",
];

const GO_WORDS: &[&str] = &[
    "if", "else", "end", "range", "with", "define", "template", "block", "and", "or", "not", "eq", "ne", "lt",
    "le", "gt", "ge", "len", "index", "slice", "print", "printf", "println", "json", "html", "js", "call",
    "urlquery", "nil", "true", "false", "toJson", "currentDate", "break", "continue",
];

struct Ctx<'a> {
    known_specials: &'a [String],
    variables: BTreeSet<String>,
    specials: BTreeSet<String>,
    locals: BTreeSet<String>,
}

impl Ctx<'_> {
    fn emits_in(&mut self, text: &str) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for s in self.known_specials {
            if s.len() > 1 && text.contains(s.as_str()) && !found.contains(s) {
                found.push(s.clone());
            }
        }
        for m in generic_special_re().find_iter(text) {
            let s = m.as_str().to_string();
            if !found.contains(&s) {
                found.push(s);
            }
        }
        for f in &found {
            self.specials.insert(f.clone());
        }
        found
    }

    fn vars_in(&mut self, expr: &str, words: &[&str]) -> Vec<String> {
        let no_strings = strings_re().replace_all(expr, "\"\"");
        let mut out: Vec<String> = Vec::new();
        let mut after_pipe = false;
        for m in ident_re().find_iter(&no_strings) {
            let before = no_strings[..m.start()].trim_end();
            // `x | filter` and `x.method(`: skip filters.
            after_pipe = before.ends_with('|') || (after_pipe && before.ends_with('('));
            if after_pipe {
                continue;
            }
            let raw = m.as_str();
            let norm = Regex::new(r#"\[\s*['"]([^'"]+)['"]\s*\]"#).unwrap().replace_all(raw, ".$1").to_string();
            let norm = norm.trim_start_matches('$').to_string();
            let norm = if norm.starts_with('.') { norm } else { norm.clone() };
            let head = norm.trim_start_matches('.').split('.').next().unwrap_or("");
            if head.is_empty() || words.contains(&head) || head.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
            // Strip trailing method calls (`.startswith`, `.strip`...).
            let parts: Vec<&str> = norm.split('.').collect();
            let mut keep: Vec<&str> = Vec::new();
            for p in parts {
                if words.contains(&p) && !keep.is_empty() {
                    break;
                }
                keep.push(p);
            }
            let v = keep.join(".");
            if !out.contains(&v) {
                out.push(v.clone());
            }
            if !self.locals.contains(head) {
                self.variables.insert(v);
            }
        }
        for lit in strings_re().captures_iter(expr) {
            let s = lit.get(1).or(lit.get(2)).or(lit.get(3)).map_or("", |m| m.as_str());
            self.emits_in(s);
        }
        out
    }
}

fn first_word(s: &str) -> (&str, &str) {
    let s = s.trim();
    match s.find(|c: char| c.is_whitespace() || c == '(') {
        Some(i) => (&s[..i], s[i..].trim()),
        None => (s, ""),
    }
}

fn emit_node(ctx: &mut Ctx, text: &str, expr: bool, words: &[&str]) -> Option<TemplateNode> {
    if !expr && text.trim().is_empty() {
        return None;
    }
    let mut n = TemplateNode::new("emit");
    if expr {
        n.vars = ctx.vars_in(text, words);
        n.text = Some(format!("{{{{ {} }}}}", text));
        for lit in strings_re().captures_iter(text) {
            let s = lit.get(1).or(lit.get(2)).or(lit.get(3)).map_or("", |m| m.as_str());
            n.emits.extend(ctx.emits_in(s));
        }
    } else {
        n.emits = ctx.emits_in(text);
        let t: String = text.chars().take(120).collect();
        n.text = Some(t);
    }
    Some(n)
}

/// Stack-based tree builder shared by both syntaxes.
struct Builder {
    stack: Vec<TemplateNode>,
    errors: Vec<String>,
}

impl Builder {
    fn new() -> Self {
        Self { stack: vec![TemplateNode::new("root")], errors: vec![] }
    }

    fn push_child(&mut self, n: TemplateNode) {
        if let Some(top) = self.stack.last_mut() {
            // Merge consecutive literal emits.
            if n.kind == "emit" && n.vars.is_empty() {
                if let Some(last) = top.children.last_mut() {
                    if last.kind == "emit" && last.vars.is_empty() {
                        let mut t = last.text.take().unwrap_or_default();
                        t.push_str(n.text.as_deref().unwrap_or(""));
                        last.text = Some(t.chars().take(160).collect());
                        for e in n.emits {
                            if !last.emits.contains(&e) {
                                last.emits.push(e);
                            }
                        }
                        return;
                    }
                }
            }
            top.children.push(n);
        }
    }

    fn open(&mut self, n: TemplateNode) {
        self.stack.push(n);
    }

    /// Opens an if-chain: a container node plus its first branch.
    fn open_if(&mut self, kind: &str, branch: &str, cond: String, vars: Vec<String>) {
        self.stack.push(TemplateNode::new(kind));
        let mut b = TemplateNode::new("branch");
        b.branch = Some(branch.into());
        b.condition = Some(cond);
        b.vars = vars;
        self.stack.push(b);
    }

    /// `elif` / `else`: closes the current branch and opens a sibling.
    fn next_branch(&mut self, branch: &str, cond: Option<String>, vars: Vec<String>) {
        if self.stack.len() >= 3 && self.stack.last().is_some_and(|n| n.kind == "branch") {
            let b = self.stack.pop().unwrap();
            self.stack.last_mut().unwrap().children.push(b);
            let mut nb = TemplateNode::new("branch");
            nb.branch = Some(branch.into());
            nb.condition = cond;
            nb.vars = vars;
            self.stack.push(nb);
        } else if self.stack.len() >= 2 && matches!(self.stack.last().map(|n| n.kind.as_str()), Some("for" | "range" | "with")) {
            // `for ... else` / `range ... else` / `with ... else`
            let mut nb = TemplateNode::new("branch");
            nb.branch = Some(branch.into());
            nb.condition = cond;
            self.stack.push(nb);
        } else {
            self.errors.push(format!("`{branch}` outside of an if block"));
        }
    }

    fn close(&mut self, what: &str) {
        if self.stack.len() <= 1 {
            self.errors.push(format!("unbalanced `{what}`"));
            return;
        }
        let n = self.stack.pop().unwrap();
        if n.kind == "branch" && self.stack.len() > 1 {
            self.stack.last_mut().unwrap().children.push(n);
            let c = self.stack.pop().unwrap();
            self.push_child(c);
        } else {
            self.push_child(n);
        }
    }

    fn finish(mut self) -> (TemplateNode, Vec<String>) {
        if self.stack.len() > 1 {
            self.errors.push(format!("{} unclosed block(s)", self.stack.len() - 1));
            while self.stack.len() > 1 {
                self.close("eof");
            }
        }
        (self.stack.pop().unwrap(), self.errors)
    }
}

pub fn parse_jinja(name: &str, src: &str, known_specials: &[String]) -> TemplateTree {
    let mut ctx = Ctx { known_specials, variables: BTreeSet::new(), specials: BTreeSet::new(), locals: BTreeSet::new() };
    let mut b = Builder::new();
    let words = JINJA_WORDS;
    let mut raw_mode = false;
    for p in lex(src, "{{", Some("{%"), ("{#", "#}"), "}}", "%}") {
        match p {
            Piece::Text(t) => {
                if let Some(n) = emit_node(&mut ctx, t, false, words) {
                    b.push_child(n);
                }
            }
            Piece::Expr(e) => {
                if let Some(n) = emit_node(&mut ctx, e, true, words) {
                    b.push_child(n);
                }
            }
            Piece::Stmt(s) => {
                let (kw, rest) = first_word(s);
                if raw_mode && kw != "endraw" {
                    continue;
                }
                match kw {
                    "if" => {
                        let v = ctx.vars_in(rest, words);
                        b.open_if("if", "if", rest.into(), v);
                    }
                    "elif" => {
                        let v = ctx.vars_in(rest, words);
                        b.next_branch("elif", Some(rest.into()), v);
                    }
                    "else" => b.next_branch("else", None, vec![]),
                    "endif" | "endfor" | "endmacro" | "endcall" | "endfilter" | "endset" | "endgeneration" | "endblock" => b.close(kw),
                    "for" => {
                        let mut n = TemplateNode::new("for");
                        if let Some((target, iter)) = rest.split_once(" in ") {
                            for t in target.split(',') {
                                ctx.locals.insert(t.trim().to_string());
                            }
                            n.vars = ctx.vars_in(iter, words);
                        }
                        n.condition = Some(rest.into());
                        b.open(n);
                    }
                    "set" => {
                        let mut n = TemplateNode::new("set");
                        let (lhs, rhs) = rest.split_once('=').unwrap_or((rest, ""));
                        let lhs = lhs.trim();
                        if !lhs.contains('.') {
                            ctx.locals.insert(lhs.to_string());
                        }
                        n.vars = ctx.vars_in(rhs, words);
                        n.condition = Some(rest.into());
                        if rhs.is_empty() {
                            b.open(n); // block set
                        } else {
                            b.push_child(n);
                        }
                    }
                    "macro" => {
                        let mut n = TemplateNode::new("macro");
                        n.condition = Some(rest.into());
                        if let Some((_, args)) = rest.split_once('(') {
                            for a in args.trim_end_matches(')').split(',') {
                                let a = a.split('=').next().unwrap_or("").trim();
                                if !a.is_empty() {
                                    ctx.locals.insert(a.to_string());
                                }
                            }
                        }
                        b.open(n);
                    }
                    "call" | "filter" | "generation" | "block" => {
                        let mut n = TemplateNode::new(kw);
                        n.condition = Some(rest.into());
                        b.open(n);
                    }
                    "raw" => raw_mode = true,
                    "endraw" => raw_mode = false,
                    "break" | "continue" | "do" | "include" | "import" | "from" | "extends" => {
                        let mut n = TemplateNode::new(kw);
                        n.condition = Some(rest.into());
                        b.push_child(n);
                    }
                    other => {
                        b.errors.push(format!("unknown statement `{other}`"));
                    }
                }
            }
        }
    }
    let (root, errors) = b.finish();
    TemplateTree {
        name: name.into(),
        syntax: Syntax::Jinja,
        source: src.into(),
        decisions: root.count_decisions(),
        root,
        variables: ctx.variables,
        special_tokens: ctx.specials,
        errors,
    }
}

pub fn parse_go(name: &str, src: &str, known_specials: &[String]) -> TemplateTree {
    let mut ctx = Ctx { known_specials, variables: BTreeSet::new(), specials: BTreeSet::new(), locals: BTreeSet::new() };
    let mut b = Builder::new();
    let words = GO_WORDS;
    for p in lex(src, "{{", None, ("{{/*", "*/}}"), "}}", "}}") {
        match p {
            Piece::Text(t) => {
                if let Some(n) = emit_node(&mut ctx, t, false, words) {
                    b.push_child(n);
                }
            }
            Piece::Stmt(_) => {}
            Piece::Expr(e) => {
                let e = e.trim();
                if e.starts_with("/*") {
                    continue;
                }
                let (kw, rest) = first_word(e);
                // `$x := ...` declarations
                if let Some((lhs, _)) = e.split_once(":=") {
                    for v in lhs.split(',') {
                        ctx.locals.insert(v.trim().trim_start_matches('$').to_string());
                    }
                }
                match kw {
                    "if" => {
                        let v = ctx.vars_in(rest, words);
                        b.open_if("if", "if", rest.into(), v);
                    }
                    "else" => {
                        let (k2, r2) = first_word(rest);
                        if k2 == "if" {
                            let v = ctx.vars_in(r2, words);
                            b.next_branch("else if", Some(r2.into()), v);
                        } else if k2 == "with" {
                            let v = ctx.vars_in(r2, words);
                            b.next_branch("else with", Some(r2.into()), v);
                        } else {
                            b.next_branch("else", None, vec![]);
                        }
                    }
                    "end" => b.close("end"),
                    "range" | "with" => {
                        let mut n = TemplateNode::new(kw);
                        n.vars = ctx.vars_in(rest, words);
                        n.condition = Some(rest.into());
                        b.open(n);
                    }
                    "define" | "block" => {
                        let mut n = TemplateNode::new(kw);
                        n.condition = Some(rest.into());
                        b.open(n);
                    }
                    "template" => {
                        let mut n = TemplateNode::new("template");
                        n.condition = Some(rest.into());
                        b.push_child(n);
                    }
                    "break" | "continue" => b.push_child(TemplateNode::new(kw)),
                    _ => {
                        if let Some(n) = emit_node(&mut ctx, e, true, words) {
                            b.push_child(n);
                        }
                    }
                }
            }
        }
    }
    let (root, errors) = b.finish();
    TemplateTree {
        name: name.into(),
        syntax: Syntax::GoTemplate,
        source: src.into(),
        decisions: root.count_decisions(),
        root,
        variables: ctx.variables,
        special_tokens: ctx.specials,
        errors,
    }
}

/// Raw special-token strings from the vocabulary (control + user-defined).
pub fn vocab_specials(g: &GgufFile) -> Vec<String> {
    let Some(tokens) = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).and_then(|a| a.as_strings()) else {
        return vec![];
    };
    let types = g.get("tokenizer.ggml.token_type").and_then(|v| v.as_array());
    let mut out: Vec<String> = tokens
        .iter()
        .enumerate()
        .filter(|(i, t)| types.and_then(|ty| ty.get_i64(*i)).is_some_and(|ty| ty == 3 || ty == 4) && t.len() > 2)
        .map(|(_, t)| t.clone())
        .collect();
    out.sort_by_key(|s| std::cmp::Reverse(s.len()));
    out
}

pub fn template_report(m: &ModelHandle) -> TemplateReport {
    let specials = vocab_specials(&m.gguf);
    let mut templates = Vec::new();
    for (k, v) in &m.gguf.metadata {
        if k == "tokenizer.chat_template" || k.starts_with("tokenizer.chat_template.") {
            if let Some(src) = v.as_str() {
                let name = k.strip_prefix("tokenizer.chat_template.").unwrap_or("default");
                templates.push(parse_jinja(name, src, &specials));
            }
        }
    }
    if let Some(t) = &m.ollama.template {
        templates.push(parse_go("ollama", t, &specials));
    }
    TemplateReport { templates, ollama_system: m.ollama.system.clone(), ollama_params: m.ollama.params.clone() }
}
