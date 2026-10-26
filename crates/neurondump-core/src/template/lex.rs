//! Splitting template source into text / tag pieces; reading variables and special tokens of a tag.

use super::*;

pub(super) enum Piece<'a> {
    Text(&'a str),
    Expr(&'a str),
    Stmt(&'a str),
}

pub(super) fn lex<'a>(src: &'a str, open_expr: &str, open_stmt: Option<&str>, comment: (&str, &str), close_expr: &str, close_stmt: &str) -> Vec<Piece<'a>> {
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

pub(super) fn strings_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#""((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|`([^`]*)`"#).unwrap())
}

pub(super) fn generic_special_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"<\|[^|<>\s]{1,48}\|>|<｜[^｜]{1,48}｜>|\[/?[A-Z_]{2,24}\]|</?(?:s|think|tool_call|tool_response|tools|start_of_turn|end_of_turn|bos|eos|image|answer|reasoning)>").unwrap())
}

pub(super) fn ident_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r#"\$?\.?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\s*['"][^'"]+['"]\s*\])*|\$\.[A-Za-z_][A-Za-z0-9_.]*|\$[A-Za-z_][A-Za-z0-9_]*"#).unwrap()
    })
}

pub(super) const JINJA_WORDS: &[&str] = &[
    "and", "or", "not", "in", "is", "if", "else", "elif", "none", "None", "true", "false", "True", "False",
    "defined", "undefined", "string", "mapping", "iterable", "number", "sequence", "equalto", "boolean",
    "for", "recursive", "loop", "namespace", "range", "raise_exception", "length", "trim", "tojson", "upper",
    "lower", "items", "strip", "split", "startswith", "endswith", "rstrip", "lstrip", "join", "replace",
    "default", "first", "last", "reverse", "selectattr", "rejectattr", "map", "list", "count", "unique",
    "safe", "format", "capitalize", "title", "indent", "with", "without", "context", "keys", "values",
    "get", "strftime_now", "lstrip_blocks", "find", "abs", "int", "float", "dictsort", "batch", "slice",
    "attribute", "tojson",
];

pub(super) const GO_WORDS: &[&str] = &[
    "if", "else", "end", "range", "with", "define", "template", "block", "and", "or", "not", "eq", "ne", "lt",
    "le", "gt", "ge", "len", "index", "slice", "print", "printf", "println", "json", "html", "js", "call",
    "urlquery", "nil", "true", "false", "toJson", "currentDate", "break", "continue",
];

pub(super) struct Ctx<'a> {
    pub(super) known_specials: &'a [String],
    pub(super) variables: BTreeSet<String>,
    pub(super) specials: BTreeSet<String>,
    pub(super) locals: BTreeSet<String>,
}

impl Ctx<'_> {
    pub(super) fn emits_in(&mut self, text: &str) -> Vec<String> {
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

    pub(super) fn vars_in(&mut self, expr: &str, words: &[&str]) -> Vec<String> {
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

pub(super) fn first_word(s: &str) -> (&str, &str) {
    let s = s.trim();
    match s.find(|c: char| c.is_whitespace() || c == '(') {
        Some(i) => (&s[..i], s[i..].trim()),
        None => (s, ""),
    }
}

pub(super) fn emit_node(ctx: &mut Ctx, text: &str, expr: bool, words: &[&str]) -> Option<TemplateNode> {
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
