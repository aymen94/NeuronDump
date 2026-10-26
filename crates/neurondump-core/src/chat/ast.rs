//! Template values (`V`) and the parsed template tree: arguments, pipelines, nodes, tokens.

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum V {
    Nil,
    Bool(bool),
    Int(i64),
    Str(String),
    List(Vec<V>),
    Map(BTreeMap<String, V>),
}

impl V {
    pub(super) fn truthy(&self) -> bool {
        match self {
            V::Nil => false,
            V::Bool(b) => *b,
            V::Int(i) => *i != 0,
            V::Str(s) => !s.is_empty(),
            V::List(l) => !l.is_empty(),
            V::Map(m) => !m.is_empty(),
        }
    }

    pub(super) fn render(&self) -> String {
        match self {
            V::Nil => String::new(),
            V::Bool(b) => b.to_string(),
            V::Int(i) => i.to_string(),
            V::Str(s) => s.clone(),
            V::List(l) => format!("[{}]", l.iter().map(V::render).collect::<Vec<_>>().join(" ")),
            V::Map(m) => format!("map[{}]", m.iter().map(|(k, v)| format!("{k}:{}", v.render())).collect::<Vec<_>>().join(" ")),
        }
    }

    pub(super) fn field(&self, name: &str) -> V {
        match self {
            V::Map(m) => m.get(name).cloned().unwrap_or(V::Nil),
            _ => V::Nil,
        }
    }

    pub(super) fn json(&self) -> serde_json::Value {
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
pub(super) enum Arg {
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
pub(super) struct Pipe {
    pub(super) cmds: Vec<Vec<Arg>>,
    /// `$a := …` / `$a, $b := …` (range) / `$a = …`.
    pub(super) decl: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) enum Node {
    Text(String),
    Out(Pipe),
    If(Vec<(Pipe, Vec<Node>)>, Option<Vec<Node>>),
    Range(Pipe, Vec<Node>, Option<Vec<Node>>),
    With(Pipe, Vec<Node>, Option<Vec<Node>>),
}

pub(super) enum Tok {
    Text(String),
    Action(String),
}
