//! Jinja and Go template parsers producing a `TemplateTree`.

use super::*;

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
