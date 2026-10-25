//! User configuration (`~/.config/neurondump/config.toml`, `%APPDATA%\neurondump\config.toml`).
//!
//! Only flat `key = value` TOML is needed, so it is read/written by hand.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum UiMode {
    Cli,
    Web,
}

impl UiMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cli" => Some(UiMode::Cli),
            "web" => Some(UiMode::Web),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            UiMode::Cli => "cli",
            UiMode::Web => "web",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub default_ui: Option<UiMode>,
    pub port: u16,
    pub scan_dirs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub threads: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_ui: None,
            port: 7878,
            scan_dirs: Vec::new(),
            output_dir: default_output_dir(),
            threads: None,
        }
    }
}

pub fn default_output_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join("neurondump")
}

pub fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("NEURONDUMP_CONFIG") {
        return PathBuf::from(p);
    }
    let base = if cfg!(windows) {
        dirs::config_dir()
    } else {
        dirs::home_dir().map(|h| h.join(".config"))
    };
    base.unwrap_or_else(|| PathBuf::from(".")).join("neurondump").join("config.toml")
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    if v.len() >= 2 && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\''))) {
        v[1..v.len() - 1].replace("\\\\", "\\").replace("\\\"", "\"")
    } else {
        v.to_string()
    }
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

impl Config {
    pub fn load() -> Self {
        let path = config_path();
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text).unwrap_or_else(|e| {
                tracing::warn!("ignoring invalid config {}: {e}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut c = Self::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                bail!("line {}: expected key = value", n + 1);
            };
            c.set(k.trim(), v.trim()).with_context(|| format!("line {}", n + 1))?;
        }
        Ok(c)
    }

    /// Sets a key from its textual (TOML-ish) value.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "default_ui" => {
                let v = unquote(value);
                self.default_ui = if v.is_empty() || v == "none" {
                    None
                } else {
                    Some(UiMode::parse(&v).context("default_ui must be cli, web or none")?)
                };
            }
            "port" => self.port = unquote(value).parse().context("port must be a number")?,
            "output_dir" => self.output_dir = PathBuf::from(unquote(value)),
            "threads" => {
                let v = unquote(value);
                self.threads = if v.is_empty() || v == "0" || v == "auto" { None } else { Some(v.parse()?) };
            }
            "scan_dirs" => {
                let v = value.trim();
                let inner = v.strip_prefix('[').and_then(|v| v.strip_suffix(']')).unwrap_or(v);
                self.scan_dirs = inner
                    .split(',')
                    .map(unquote)
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect();
            }
            _ => bail!("unknown config key {key:?} (known: default_ui, port, scan_dirs, output_dir, threads)"),
        }
        Ok(())
    }

    pub fn to_toml(&self) -> String {
        let dirs: Vec<String> = self.scan_dirs.iter().map(|d| quote(&d.to_string_lossy())).collect();
        format!(
            "# NeuronDump configuration\ndefault_ui = {}\nport = {}\nscan_dirs = [{}]\noutput_dir = {}\nthreads = {}\n",
            quote(self.default_ui.map_or("none", UiMode::as_str)),
            self.port,
            dirs.join(", "),
            quote(&self.output_dir.to_string_lossy()),
            self.threads.unwrap_or(0),
        )
    }

    pub fn save(&self) -> Result<PathBuf> {
        let path = config_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, self.to_toml()).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}
