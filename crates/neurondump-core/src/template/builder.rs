//! Building the nested decision tree from opened / closed branches.

use super::*;

/// Stack-based tree builder shared by both syntaxes.
pub(super) struct Builder {
    pub(super) stack: Vec<TemplateNode>,
    pub(super) errors: Vec<String>,
}

impl Builder {
    pub(super) fn new() -> Self {
        Self { stack: vec![TemplateNode::new("root")], errors: vec![] }
    }

    pub(super) fn push_child(&mut self, n: TemplateNode) {
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

    pub(super) fn open(&mut self, n: TemplateNode) {
        self.stack.push(n);
    }

    /// Opens an if-chain: a container node plus its first branch.
    pub(super) fn open_if(&mut self, kind: &str, branch: &str, cond: String, vars: Vec<String>) {
        self.stack.push(TemplateNode::new(kind));
        let mut b = TemplateNode::new("branch");
        b.branch = Some(branch.into());
        b.condition = Some(cond);
        b.vars = vars;
        self.stack.push(b);
    }

    /// `elif` / `else`: closes the current branch and opens a sibling.
    pub(super) fn next_branch(&mut self, branch: &str, cond: Option<String>, vars: Vec<String>) {
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

    pub(super) fn close(&mut self, what: &str) {
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

    pub(super) fn finish(mut self) -> (TemplateNode, Vec<String>) {
        if self.stack.len() > 1 {
            self.errors.push(format!("{} unclosed block(s)", self.stack.len() - 1));
            while self.stack.len() > 1 {
                self.close("eof");
            }
        }
        (self.stack.pop().unwrap(), self.errors)
    }
}
