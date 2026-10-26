//! Node lookups: get, children, ancestors, tensors under a node, node detail, ids.

use super::*;

impl ModelGraph {
    /// Structural or fine node.
    pub fn get(&self, id: &str) -> Option<Node> {
        self.nodes.get(id).cloned().or_else(|| self.fine_node(id))
    }

    /// Children of a node, including generated fine children.
    pub fn children(&self, id: &str) -> Vec<Node> {
        let mut out: Vec<Node> = self.nodes.values().filter(|n| n.parent.as_deref() == Some(id)).cloned().collect();
        if let Some(n) = self.nodes.get(id) {
            if n.lazy {
                out.extend(self.fine_children(id).into_iter().filter(|c| c.parent.as_deref() == Some(id)));
            }
        } else if Self::fine_parent_block(id).is_some() {
            let block = Self::fine_parent_block(id).unwrap_or_default();
            out.extend(self.fine_children(&block).into_iter().filter(|c| c.parent.as_deref() == Some(id)));
        }
        out
    }

    /// Ancestor chain (closest first), for structural and fine nodes.
    pub fn ancestors(&self, id: &str) -> Vec<String> {
        if let Some(n) = self.fine_node(id) {
            let mut out = vec![];
            let mut cur = n.parent.clone();
            while let Some(p) = cur {
                cur = self.get(&p).and_then(|x| x.parent);
                out.push(p);
            }
            return out;
        }
        ancestors_of(&self.nodes, id)
    }

    /// Every tensor node under `id` (or `id` itself when it is a tensor).
    pub fn tensors_under(&self, id: &str) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.kind == NodeKind::Tensor && (n.id == id || ancestors_of(&self.nodes, &n.id).iter().any(|a| a == id)))
            .collect()
    }

    pub fn detail(&self, id: &str) -> Option<NodeDetail> {
        let node = self.get(id)?;
        let mut node = node;
        let children = self.children(id);
        if node.lazy {
            node.contains = children.iter().filter(|c| c.kind.is_fine()).map(|c| ChildSummary { id: c.id.clone(), kind: c.kind, label: c.label.clone(), params: c.params, bytes: c.bytes }).collect::<Vec<_>>();
            node.contains.extend(self.nodes.get(id).map(|n| n.contains.clone()).unwrap_or_default());
        }
        let ancestors = self
            .ancestors(id)
            .iter()
            .filter_map(|a| self.get(a))
            .map(|a| ChildSummary { id: a.id.clone(), kind: a.kind, label: a.label.clone(), params: a.params, bytes: a.bytes })
            .collect();
        // Connections: base edges with exactly one endpoint inside the subtree, the other
        // end lifted to the ancestor at the node's own depth.
        let inside = |x: &str| x == id || self.ancestors(x).iter().any(|a| a == id);
        let depth = self.ancestors(id).len();
        let lift = |x: &str| -> String {
            let anc = self.ancestors(x);
            if anc.len() <= depth {
                return x.to_string();
            }
            let mut chain = vec![x.to_string()];
            chain.extend(anc);
            // chain[k] has depth len-1-k
            let target = chain.len() - 1 - depth;
            chain[target].clone()
        };
        let mut incoming = BTreeSet::new();
        let mut outgoing = BTreeSet::new();
        for e in &self.edges {
            let (fi, ti) = (inside(&e.from), inside(&e.to));
            if fi && !ti {
                outgoing.insert((id.to_string(), lift(&e.to), e.kind as u8, e.label.clone()));
            } else if ti && !fi {
                incoming.insert((lift(&e.from), id.to_string(), e.kind as u8, e.label.clone()));
            }
        }
        let to_edge = |(f, t, k, l): (String, String, u8, Option<String>)| Edge {
            from: f,
            to: t,
            kind: match k {
                0 => EdgeKind::Dataflow,
                1 => EdgeKind::Contains,
                2 => EdgeKind::Tied,
                _ => EdgeKind::Uses,
            },
            label: l,
        };
        Some(NodeDetail {
            node,
            ancestors,
            children,
            incoming: incoming.into_iter().map(to_edge).collect(),
            outgoing: outgoing.into_iter().map(to_edge).collect(),
        })
    }

    /// All node ids down to tensor level (for completion).
    pub fn ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    // --- views -------------------------------------------------------------------------

    pub(super) fn rank(&self, n: &Node) -> u8 {
        match n.kind {
            NodeKind::Model => 0,
            NodeKind::Section => 1,
            NodeKind::Layer => 2,
            NodeKind::Tensor | NodeKind::Token if n.layer.is_none() => {
                // Input/output tensors are few and part of the main path; side sections
                // (vision encoder, MTP head, unknown tensors) can hold hundreds.
                if n.kind == NodeKind::Token || !matches!(n.parent.as_deref(), Some("input" | "output")) {
                    4
                } else {
                    2
                }
            }
            NodeKind::Block | NodeKind::Op if n.layer.is_some() => 3,
            NodeKind::Op => 2,
            NodeKind::Block => 3,
            NodeKind::Tensor | NodeKind::Token => 4,
            NodeKind::Group => 1,
            _ => 5,
        }
    }

}
