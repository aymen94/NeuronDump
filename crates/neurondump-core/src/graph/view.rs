//! `ModelGraph::view`: a level / grouping / expansion view with edges lifted to visible nodes.

use super::*;

impl ModelGraph {
    pub fn view(&self, o: &ViewOptions) -> GraphView {
        let lr = o.level.rank();
        let mut visible: IndexMap<String, Node> = IndexMap::new();
        for n in self.nodes.values() {
            if n.kind != NodeKind::Model && self.rank(n) <= lr {
                visible.insert(n.id.clone(), n.clone());
            }
        }
        if o.level == Level::Fine {
            let blocks: Vec<String> = self.nodes.values().filter(|n| n.lazy).map(|n| n.id.clone()).collect();
            for b in blocks {
                for c in self.fine_children(&b) {
                    visible.insert(c.id.clone(), c);
                }
            }
        }
        let mut expanded = Vec::new();
        for e in &o.expand {
            let Some(id) = self.resolve_id(e) else { continue };
            expanded.push(id.clone());
            for a in self.ancestors(&id) {
                if a != "model" {
                    if let Some(n) = self.get(&a) {
                        visible.entry(a).or_insert(n);
                    }
                }
            }
            if let Some(n) = self.get(&id) {
                visible.entry(id.clone()).or_insert(n);
            }
            for c in self.children(&id) {
                visible.entry(c.id.clone()).or_insert(c);
            }
        }

        let mut nodes: Vec<Node>;
        let mut parent_of: HashMap<String, Option<String>> = HashMap::new();
        if o.group_by == GroupBy::Section {
            nodes = visible.values().cloned().collect();
            for n in &mut nodes {
                if n.parent.as_deref() == Some("model") {
                    n.parent = None;
                }
                parent_of.insert(n.id.clone(), n.parent.clone());
            }
        } else {
            // Units: the deepest visible kind of the requested level; everything structural
            // above them is replaced by synthetic group nodes.
            let unit_rank = visible.values().map(|n| self.rank(n)).max().unwrap_or(1).min(lr.max(2));
            let mut groups: IndexMap<String, Node> = IndexMap::new();
            nodes = Vec::new();
            for n in visible.values() {
                let r = self.rank(n);
                let is_unit = r == unit_rank || (r < unit_rank && n.kind == NodeKind::Section && !visible.values().any(|c| c.parent.as_deref() == Some(&n.id)));
                let side = n.layer.is_none() && n.kind != NodeKind::Group && !matches!(o.group_by, GroupBy::Quant | GroupBy::Importance);
                if !is_unit && !(r > unit_rank) {
                    continue;
                }
                let mut c = n.clone();
                if r > unit_rank {
                    // Deeper nodes from `expand` keep their structural parent.
                    parent_of.insert(c.id.clone(), c.parent.clone());
                    nodes.push(c);
                    continue;
                }
                let key: Option<(String, String)> = match o.group_by {
                    GroupBy::Section => None,
                    GroupBy::Layer => n.layer.map(|l| (format!("group.layer.{l}"), format!("layer {l}"))),
                    GroupBy::Block => n.block.clone().map(|b| (format!("group.block.{b}"), format!("all {}", block_label(&b)))),
                    GroupBy::Quant => Some({
                        let q = if n.quant.is_empty() { "none".to_string() } else { n.quant.join("+") };
                        (format!("group.quant.{q}"), format!("quant {q}"))
                    }),
                    GroupBy::Kind => n.layer_kind.clone().map(|k| (format!("group.kind.{k}"), k)),
                    GroupBy::Importance => Some({
                        let s = n.importance.score;
                        let b = if n.importance.source.starts_with("structural") && n.removable != Removable::BreaksModel {
                            "unmeasured"
                        } else if s >= 0.66 {
                            "high"
                        } else if s >= 0.33 {
                            "medium"
                        } else {
                            "low"
                        };
                        (format!("group.importance.{b}"), format!("importance {b}"))
                    }),
                };
                match key {
                    Some((gid, glabel)) => {
                        let gn = groups.entry(gid.clone()).or_insert_with(|| {
                            let mut g = new_node(&gid, NodeKind::Group, &glabel, None);
                            g.removable = Removable::Safe;
                            g.importance = Importance { score: 0.0, source: "group".into() };
                            g
                        });
                        gn.params += n.params;
                        gn.bytes += n.bytes;
                        gn.removable = gn.removable.max(n.removable);
                        gn.contains.push(ChildSummary { id: n.id.clone(), kind: n.kind, label: n.label.clone(), params: n.params, bytes: n.bytes });
                        c.parent = Some(gid);
                    }
                    None => {
                        c.parent = if side { n.parent.clone().filter(|p| p != "model" && visible.contains_key(p)) } else { None };
                        if let Some(p) = &c.parent {
                            if !nodes.iter().any(|x| &x.id == p) {
                                if let Some(pn) = visible.get(p) {
                                    let mut pn = pn.clone();
                                    pn.parent = None;
                                    parent_of.insert(pn.id.clone(), None);
                                    nodes.push(pn);
                                }
                            }
                        }
                    }
                }
                parent_of.insert(c.id.clone(), c.parent.clone());
                nodes.push(c);
            }
            for (gid, g) in groups {
                parent_of.insert(gid, None);
                nodes.insert(0, g);
            }
        }
        let vis: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();

        // Lift edges to the nearest visible ancestor.
        let lift = |x: &str| -> Option<String> {
            if vis.contains(x) {
                return Some(x.to_string());
            }
            self.ancestors(x).into_iter().find(|a| vis.contains(a))
        };
        let mut seen: HashSet<(String, String, EdgeKind)> = HashSet::new();
        let mut edges = Vec::new();
        for e in &self.edges {
            let (Some(f), Some(t)) = (lift(&e.from), lift(&e.to)) else { continue };
            if f == t {
                continue;
            }
            // Edges into a node's own ancestor are containment, not flow.
            if self.ancestors(&f).contains(&t) || self.ancestors(&t).contains(&f) {
                continue;
            }
            if seen.insert((f.clone(), t.clone(), e.kind)) {
                let label = if e.from == f && e.to == t { e.label.clone() } else { e.label.clone().filter(|l| l != "residual") };
                edges.push(Edge { from: f, to: t, kind: e.kind, label });
            }
        }
        for n in &nodes {
            if let Some(Some(p)) = parent_of.get(&n.id) {
                if vis.contains(p) {
                    edges.push(Edge { from: p.clone(), to: n.id.clone(), kind: EdgeKind::Contains, label: None });
                }
            }
        }
        GraphView {
            schema_version: SCHEMA_VERSION,
            model: self.model.clone(),
            architecture: self.architecture.clone(),
            kind: self.kind,
            level: o.level,
            group_by: o.group_by,
            expand: expanded,
            nodes,
            edges,
            notes: self.notes.clone(),
        }
    }
}

// --- exports -----------------------------------------------------------------------------
