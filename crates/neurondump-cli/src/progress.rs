//! indicatif progress bar (stderr) implementing core's `Progress`.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use neurondump_core::Progress;

pub struct Bar {
    bar: ProgressBar,
    cancel: Arc<AtomicBool>,
}

impl Bar {
    pub fn new(label: &str, quiet: bool) -> Self {
        let bar = ProgressBar::new(1000);
        if quiet || !std::io::stderr().is_terminal() {
            bar.set_draw_target(ProgressDrawTarget::hidden());
        } else {
            bar.set_draw_target(ProgressDrawTarget::stderr());
        }
        bar.set_style(
            ProgressStyle::with_template("{prefix:.bold} [{bar:30}] {percent:>3}% {elapsed_precise} {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        bar.set_prefix(label.to_string());
        Self { bar, cancel: Arc::new(AtomicBool::new(false)) }
    }

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    pub fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

impl Progress for Bar {
    fn set(&self, percent: f32, step: &str) {
        self.bar.set_position((percent.clamp(0.0, 100.0) * 10.0) as u64);
        let s: String = step.chars().take(60).collect();
        self.bar.set_message(s);
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}
