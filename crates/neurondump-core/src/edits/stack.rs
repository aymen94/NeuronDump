//! The edit stack: push / toggle / remove, undo / redo / restore.

use super::*;

/// Edit stack with undo / redo (whole-stack checkpoints).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditStack {
    pub edits: Vec<Edit>,
    #[serde(skip)]
    pub(super) undo: Vec<Vec<Edit>>,
    #[serde(skip)]
    pub(super) redo: Vec<Vec<Edit>>,
    pub(super) next_id: u64,
}

impl EditStack {
    pub(super) fn checkpoint(&mut self) {
        self.undo.push(self.edits.clone());
        if self.undo.len() > 200 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn push(&mut self, op: EditOp, label: Option<String>) -> u64 {
        self.checkpoint();
        self.next_id = self.next_id.max(self.edits.iter().map(|e| e.id).max().unwrap_or(0)) + 1;
        let id = self.next_id;
        let label = label.filter(|l| !l.is_empty()).unwrap_or_else(|| op.describe());
        self.edits.push(Edit { id, enabled: true, label, created: now_unix(), op });
        id
    }

    pub fn toggle(&mut self, id: u64) -> Result<bool> {
        let i = self.edits.iter().position(|e| e.id == id).ok_or_else(|| anyhow!("no edit #{id}"))?;
        self.checkpoint();
        self.edits[i].enabled = !self.edits[i].enabled;
        Ok(self.edits[i].enabled)
    }

    pub fn remove(&mut self, id: u64) -> Result<()> {
        let i = self.edits.iter().position(|e| e.id == id).ok_or_else(|| anyhow!("no edit #{id}"))?;
        self.checkpoint();
        self.edits.remove(i);
        Ok(())
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.edits, prev));
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.edits, next));
                true
            }
            None => false,
        }
    }

    /// Back to the original model (undoable).
    pub fn restore(&mut self) {
        if !self.edits.is_empty() {
            self.checkpoint();
            self.edits.clear();
        }
    }

    /// Replaces the whole stack (load / import), renumbering ids (undoable).
    pub fn replace(&mut self, edits: Vec<Edit>) {
        self.checkpoint();
        self.edits = edits
            .into_iter()
            .enumerate()
            .map(|(i, mut e)| {
                e.id = i as u64 + 1;
                if e.label.is_empty() {
                    e.label = e.op.describe();
                }
                e
            })
            .collect();
        self.next_id = self.edits.len() as u64;
    }

    pub fn enabled(&self) -> Vec<Edit> {
        self.edits.iter().filter(|e| e.enabled).cloned().collect()
    }
}
