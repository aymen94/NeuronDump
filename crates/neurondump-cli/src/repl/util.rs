//! Argument splitting, paging long output, picking the current model.

use super::*;

/// Minimal shell-like splitting with quotes.
pub(super) fn split_args(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                any = true;
            }
            None if c.is_whitespace() => {
                if !cur.is_empty() || any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            None => cur.push(c),
        }
    }
    if !cur.is_empty() || any {
        out.push(cur);
    }
    out
}

pub(super) fn page(text: &str) {
    let lines: Vec<&str> = text.lines().collect();
    let height = 50;
    if lines.len() <= height || !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
        return;
    }
    let stdin = std::io::stdin();
    for (i, chunk) in lines.chunks(height).enumerate() {
        for l in chunk {
            println!("{l}");
        }
        if (i + 1) * height < lines.len() {
            print!("-- more ({}/{}) Enter: next, q: quit -- ", ((i + 1) * height).min(lines.len()), lines.len());
            let _ = std::io::stdout().flush();
            let mut s = String::new();
            if stdin.lock().read_line(&mut s).is_err() || s.trim() == "q" {
                break;
            }
        }
    }
}

pub(super) fn pick_model(ctx: &Ctx) -> Option<String> {
    let names: Vec<String> = ctx.app.library().entries.iter().map(|e| e.name.clone()).collect();
    if names.is_empty() {
        eprintln!("no models found; use `use <path.gguf>` or start with --scan <dir>");
        return None;
    }
    if !std::io::stdin().is_terminal() {
        return names.first().cloned();
    }
    let sel = dialoguer::FuzzySelect::new().with_prompt("Pick a model (type to filter)").items(&names).default(0).interact_opt().ok()??;
    Some(names[sel].clone())
}
