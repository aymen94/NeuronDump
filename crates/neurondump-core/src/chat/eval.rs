//! Evaluating a parsed template: variables, functions (`eq`, `len`, `slice`...), pipelines, control flow.

use super::*;

pub(super) struct Eval<'a> {
    pub(super) root: &'a V,
    pub(super) vars: Vec<BTreeMap<String, V>>,
}

impl Eval<'_> {
    pub(super) fn var(&self, name: &str) -> V {
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

    pub(super) fn set(&mut self, name: &str, v: V) {
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

    pub(super) fn arg(&mut self, a: &Arg, dot: &V) -> Result<V> {
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

    pub(super) fn call(&mut self, f: &str, args: &[Arg], dot: &V) -> Result<V> {
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

    pub(super) fn pipe(&mut self, p: &Pipe, dot: &V) -> Result<V> {
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

    pub(super) fn exec(&mut self, nodes: &[Node], dot: &V, out: &mut String) -> Result<()> {
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
