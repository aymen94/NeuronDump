//! Per-model live edit stacks: editing, view, snapshots, imports, compiling, prompt text, sampling.

use super::*;

impl App {
    /// A copy of the model's edit stack.
    pub fn edit_stack(&self, m: &ModelHandle) -> EditStack {
        self.edits.lock().unwrap().get(&m.entry.id).cloned().unwrap_or_default()
    }

    /// Changes the stack; a change whose result does not compile is rolled back.
    pub fn edit<T>(&self, m: &ModelHandle, f: impl FnOnce(&mut EditStack) -> Result<T>) -> Result<T> {
        let g = self.graph(m)?;
        let v = self.vocab(m).ok();
        let mut all = self.edits.lock().unwrap();
        let stack = all.entry(m.entry.id.clone()).or_default();
        let before = stack.clone();
        let r = f(stack)?;
        if let Err(e) = compile(&g, v.as_deref(), &stack.edits) {
            *stack = before;
            return Err(e);
        }
        Ok(r)
    }

    pub fn push_edit(&self, m: &ModelHandle, op: EditOp, label: Option<String>) -> Result<u64> {
        self.edit(m, |s| Ok(s.push(op, label)))
    }

    pub fn edit_view(&self, m: &ModelHandle) -> Result<EditView> {
        let st = self.edit_stack(m);
        let g = self.graph(m)?;
        let v = self.vocab(m).ok();
        let (compiled, error) = match compile(&g, v.as_deref(), &st.edits) {
            Ok(c) => (Some(c), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };
        Ok(EditView {
            model: m.entry.name.clone(),
            digest: m.entry.digest.clone(),
            active: st.edits.iter().filter(|e| e.enabled).count(),
            can_undo: st.can_undo(),
            can_redo: st.can_redo(),
            edits: st.edits,
            compiled,
            error,
            snapshots: crate::edits::list_snapshots(&self.config.output_dir, m),
        })
    }

    pub fn save_snapshot(&self, m: &ModelHandle, name: &str) -> Result<PathBuf> {
        crate::edits::save_snapshot(&self.config.output_dir, m, name, &self.edit_stack(m).edits)
    }

    pub fn load_snapshot(&self, m: &ModelHandle, name: &str) -> Result<Vec<String>> {
        let f = crate::edits::load_snapshot(&self.config.output_dir, m, name)?;
        self.import_edits(m, f)
    }

    /// Replaces the stack with a portable `edits.json`; returns warnings.
    pub fn import_edits(&self, m: &ModelHandle, f: crate::edits::EditsFile) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        if !f.digest.is_empty() && f.digest != m.entry.digest {
            warnings.push(format!("these edits were made for {} ({}); node ids may not match", f.model, f.digest));
        }
        self.edit(m, |s| {
            s.replace(f.edits);
            Ok(())
        })?;
        Ok(warnings)
    }

    pub(super) fn compile_for(&self, m: &ModelHandle, edits: &[Edit], baseline: bool) -> Result<Compiled> {
        if baseline {
            return Ok(Compiled::default());
        }
        let g = self.graph(m)?;
        let v = self.vocab(m).ok();
        compile(&g, v.as_deref(), edits)
    }

    /// The text fed to the model: raw, or chat-templated (always when a prompt edit is active).
    pub fn prompt_text(&self, m: &ModelHandle, pp: &PromptParams, c: &Compiled) -> Result<String> {
        if !pp.chat && !c.prompt_edited() {
            return Ok(pp.prompt.clone());
        }
        let v = self.vocab(m).ok();
        let req = ChatRequest {
            prompt: pp.prompt.clone(),
            system: c.system.clone().or_else(|| pp.system.clone()),
            template: c.template.clone().or_else(|| pp.template.clone()),
        };
        Ok(render_chat(m, v.as_deref(), &req)?.text)
    }

    /// Sampling of the current stack (for display).
    pub fn current_sampling(&self, m: &ModelHandle) -> Option<Sampling> {
        self.edit_view(m).ok().and_then(|v| v.compiled).and_then(|c| c.sampling)
    }

}
