//! Text exports of a view: tree, Graphviz DOT and Mermaid.

use super::*;

pub(super) fn removable_color(r: Removable) -> &'static str {
    match r {
        Removable::Safe => "#c8f0c8",
        Removable::Risky => "#fde9b6",
        Removable::BreaksModel => "#f7c4c0",
    }
}

pub(super) fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

pub(super) fn short_size(n: &Node) -> String {
    if n.params == 0 && n.bytes == 0 {
        String::new()
    } else {
        format!("{} · {}", human_count(n.params), human_bytes(n.bytes))
    }
}

impl GraphView {
    pub(super) fn children_map(&self) -> (Vec<&Node>, HashMap<&str, Vec<&Node>>) {
        let ids: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        let mut kids: HashMap<&str, Vec<&Node>> = HashMap::new();
        let mut roots = Vec::new();
        for n in &self.nodes {
            match n.parent.as_deref().filter(|p| ids.contains(p)) {
                Some(p) => kids.entry(p).or_default().push(n),
                None => roots.push(n),
            }
        }
        (roots, kids)
    }

    pub fn to_tree(&self) -> String {
        let (roots, kids) = self.children_map();
        let mut s = format!(
            "{} ({}, {}) · level {} · group by {}\n",
            self.model,
            self.architecture,
            enum_str(&self.kind),
            self.level.as_str(),
            self.group_by.as_str()
        );
        fn walk(s: &mut String, n: &Node, kids: &HashMap<&str, Vec<&Node>>, depth: usize) {
            let imp = if n.importance.source.starts_with("structural: not") || n.kind == NodeKind::Group { String::new() } else { format!("  imp {:.2}", n.importance.score) };
            let q = if n.quant.is_empty() { String::new() } else { format!("  [{}]", n.quant.join(",")) };
            let _ = writeln!(
                s,
                "{}{} {}  ({}){}{}  {}  #{}",
                "  ".repeat(depth),
                if kids.contains_key(n.id.as_str()) { "▾" } else if n.lazy { "▸" } else { "•" },
                n.label,
                n.kind.as_str(),
                if short_size(n).is_empty() { String::new() } else { format!("  {}", short_size(n)) },
                q,
                n.removable.as_str(),
                n.id
            );
            let _ = imp.is_empty() || {
                s.pop();
                let _ = writeln!(s, "{imp}");
                true
            };
            if let Some(cs) = kids.get(n.id.as_str()) {
                for c in cs {
                    walk(s, c, kids, depth + 1);
                }
            }
        }
        for r in roots {
            walk(&mut s, r, &kids, 0);
        }
        let flows = self.edges.iter().filter(|e| e.kind != EdgeKind::Contains).count();
        let _ = writeln!(s, "{} nodes, {} connections (▸ = expandable with --expand <id>)", self.nodes.len(), flows);
        s
    }

    pub fn to_dot(&self) -> String {
        let (roots, kids) = self.children_map();
        let mut s = String::from("digraph neurondump {\n  rankdir=TB;\n  compound=true;\n  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10];\n  edge [fontname=\"Helvetica\", fontsize=8];\n");
        let _ = writeln!(s, "  label=\"{} ({})\";", dot_escape(&self.model), dot_escape(&self.architecture));
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            idx.insert(&n.id, i);
        }
        fn emit(s: &mut String, n: &Node, kids: &HashMap<&str, Vec<&Node>>, idx: &HashMap<&str, usize>, depth: usize) {
            let pad = "  ".repeat(depth + 1);
            let size = short_size(n);
            let label = if size.is_empty() { dot_escape(&n.label) } else { format!("{}\\n{}", dot_escape(&n.label), size) };
            match kids.get(n.id.as_str()) {
                Some(cs) => {
                    let _ = writeln!(s, "{pad}subgraph cluster_{} {{", idx[n.id.as_str()]);
                    let _ = writeln!(s, "{pad}  label=\"{label}\"; style=\"rounded\"; color=\"#888888\";");
                    // An anchor node so edges to the cluster have an endpoint.
                    let _ = writeln!(s, "{pad}  n{} [label=\"{}\", shape=plaintext, style=\"\"];", idx[n.id.as_str()], dot_escape(&n.label));
                    for c in cs {
                        emit(s, c, kids, idx, depth + 1);
                    }
                    let _ = writeln!(s, "{pad}}}");
                }
                None => {
                    let shape = match n.kind {
                        NodeKind::Op => ", shape=ellipse",
                        NodeKind::Token => ", shape=note",
                        _ => "",
                    };
                    let _ = writeln!(s, "{pad}n{} [label=\"{label}\", fillcolor=\"{}\"{shape}];", idx[n.id.as_str()], removable_color(n.removable));
                }
            }
        }
        for r in roots {
            emit(&mut s, r, &kids, &idx, 0);
        }
        for e in &self.edges {
            if e.kind == EdgeKind::Contains {
                continue;
            }
            let (Some(f), Some(t)) = (idx.get(e.from.as_str()), idx.get(e.to.as_str())) else { continue };
            let style = match e.kind {
                EdgeKind::Dataflow => "",
                EdgeKind::Tied => ", style=dashed, color=\"#c0392b\", dir=both",
                EdgeKind::Uses => ", style=dotted, color=\"#555555\"",
                EdgeKind::Contains => "",
            };
            let label = e.label.as_ref().map(|l| format!(", label=\"{}\"", dot_escape(l))).unwrap_or_default();
            let _ = writeln!(s, "  n{f} -> n{t} [{}{style}{label}];", if e.label.as_deref() == Some("residual") { "color=\"#999999\"" } else { "" }.trim_start_matches(", "));
        }
        s.push_str("}\n");
        s.replace("[, ", "[")
    }

    pub fn to_mermaid(&self) -> String {
        let (roots, kids) = self.children_map();
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            idx.insert(&n.id, i);
        }
        let mut s = String::from("flowchart TD\n");
        let mlabel = |n: &Node| -> String {
            let size = short_size(n);
            let text = if size.is_empty() { n.label.clone() } else { format!("{}<br/>{}", n.label, size) };
            text.replace('"', "#quot;")
        };
        fn emit(s: &mut String, n: &Node, kids: &HashMap<&str, Vec<&Node>>, idx: &HashMap<&str, usize>, depth: usize, mlabel: &dyn Fn(&Node) -> String) {
            let pad = "  ".repeat(depth + 1);
            match kids.get(n.id.as_str()) {
                Some(cs) => {
                    let _ = writeln!(s, "{pad}subgraph g{}[\"{}\"]", idx[n.id.as_str()], mlabel(n));
                    for c in cs {
                        emit(s, c, kids, idx, depth + 1, mlabel);
                    }
                    let _ = writeln!(s, "{pad}end");
                }
                None => {
                    let (o, c) = match n.kind {
                        NodeKind::Op => ("((", "))"),
                        NodeKind::Token => ("[/", "/]"),
                        _ => ("[", "]"),
                    };
                    let _ = writeln!(s, "{pad}n{}{o}\"{}\"{c}:::{}", idx[n.id.as_str()], mlabel(n), n.removable.as_str());
                }
            }
        }
        for r in roots {
            emit(&mut s, r, &kids, &idx, 0, &mlabel);
        }
        for e in &self.edges {
            if e.kind == EdgeKind::Contains {
                continue;
            }
            let (Some(f), Some(t)) = (idx.get(e.from.as_str()), idx.get(e.to.as_str())) else { continue };
            let pref = |i: usize| if kids.contains_key(self.nodes[i].id.as_str()) { format!("g{i}") } else { format!("n{i}") };
            let arrow = match e.kind {
                EdgeKind::Dataflow => "-->",
                EdgeKind::Tied => "<-.->",
                EdgeKind::Uses => "-.->",
                EdgeKind::Contains => "---",
            };
            match &e.label {
                Some(l) => {
                    let _ = writeln!(s, "  {} {arrow}|{}| {}", pref(*f), l.replace('"', "'"), pref(*t));
                }
                None => {
                    let _ = writeln!(s, "  {} {arrow} {}", pref(*f), pref(*t));
                }
            }
        }
        s.push_str("  classDef safe fill:#c8f0c8,stroke:#3a7d3a;\n  classDef risky fill:#fde9b6,stroke:#a77b00;\n  classDef breaks_model fill:#f7c4c0,stroke:#b42318;\n");
        s
    }
}
