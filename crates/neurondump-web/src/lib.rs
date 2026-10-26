//! Local web interface: a JSON API mirroring `neurondump-core` plus an embedded,
//! framework-free frontend. Contains no analysis logic.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::convert::Infallible;
use std::hash::{BuildHasher, Hasher};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;
use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream};
use rust_embed::RustEmbed;
use serde::Deserialize;
use serde_json::{json, Value};

use neurondump_core::api;
use neurondump_core::compare::compare;
use neurondump_core::export::{render as export_render, Format, Table};
use neurondump_core::inspect::{inspect, InspectOptions, InspectReport};
use neurondump_core::stats::{tensor_stats, TensorStats};
use neurondump_core::template::template_report;
use neurondump_core::edits::EditOp;
use neurondump_core::tokenizer::VocabQuery;
use neurondump_core::{App, JobKind, ModelSummary, NoProgress};

mod state;
mod assets;
mod models;
mod jobs;
mod graph;
mod edits;
pub use state::*;
use assets::*;
use models::*;
use jobs::*;
use graph::*;
use edits::*;

pub const TOKEN_HEADER: &str = "x-neurondump-token";

pub struct ServeOptions {
    pub host: String,
    pub port: u16,
    pub open: bool,
}

pub struct AppState {
    pub app: Arc<App>,
    pub token: String,
    summaries: Mutex<Option<Vec<ModelSummary>>>,
    reports: Mutex<HashMap<String, Arc<InspectReport>>>,
    tensor_stats: Mutex<HashMap<(String, String), Arc<TensorStats>>>,
}

type St = State<Arc<AppState>>;

/// 128-bit random token from the OS-seeded SipHash keys (no extra RNG dependency).
pub fn random_token() -> String {
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(i);
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        h.write_u32(std::process::id());
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// The full router (exposed for tests and embedding).
pub fn router(state: Arc<AppState>) -> Router {
    let api = Router::new()
        .route("/info", get(info))
        .route("/models", get(models))
        .route("/models/rescan", post(rescan))
        .route("/models/{id}", get(model))
        .route("/models/{id}/metadata", get(metadata))
        .route("/models/{id}/vocab", get(vocab))
        .route("/models/{id}/tensors", get(tensors))
        .route("/models/{id}/layers", get(layers))
        .route("/models/{id}/template", get(template))
        .route("/models/{id}/tensors/{name}/stats", get(one_tensor_stats))
        .route("/models/{id}/graph", get(graph))
        .route("/models/{id}/nodes/{node}", get(graph_node))
        .route("/models/{id}/impact", post(impact))
        .route("/models/{id}/plan-from-ablation", post(plan_from_ablation))
        .route("/models/{id}/tokenize", post(tokenize))
        .route("/models/{id}/embedding-map", get(embedding_map))
        .route("/models/{id}/vocab/texts", get(vocab_texts))
        .route("/models/{id}/edits", get(get_edits).post(post_edit).delete(delete_edits))
        .route("/models/{id}/edits/export", get(export_edits))
        .route("/models/{id}/edits/undo", post(undo_edits))
        .route("/models/{id}/edits/redo", post(redo_edits))
        .route("/models/{id}/edits/restore", post(restore_edits))
        .route("/models/{id}/edits/{n}", delete(delete_edit))
        .route("/models/{id}/edits/{n}/toggle", post(toggle_edit))
        .route("/models/{id}/snapshots", get(list_snapshots).post(post_snapshot))
        .route("/models/{id}/bake", post(post_bake))
        .route("/jobs", post(create_job).get(list_jobs))
        .route("/jobs/{id}", get(get_job))
        .route("/jobs/{id}", delete(cancel_job))
        .route("/jobs/{id}/events", get(job_events))
        .route("/jobs/{id}/result", get(job_result))
        .route("/search", get(search))
        .route("/compare", get(compare_models))
        .route("/export/{id}", get(export))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state.clone());
    Router::new()
        .route("/", get(index))
        .route("/assets/{*path}", get(static_file))
        .nest("/api", api)
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            header::REFERRER_POLICY,
            HeaderValue::from_static("no-referrer"),
        ))
}

fn open_browser(url: &str) {
    let r = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if let Err(e) = r {
        eprintln!("could not open a browser: {e}");
    }
}

/// Runs the server until Ctrl-C (blocking).
pub fn serve(app: Arc<App>, opts: ServeOptions) -> anyhow::Result<()> {
    let token = random_token();
    let state = AppState::new(app, token.clone());
    let addr: SocketAddr = format!("{}:{}", opts.host, opts.port).parse().context("invalid --host/--port")?;
    if !addr.ip().is_loopback() {
        eprintln!(
            "WARNING: binding to {} exposes NeuronDump (which reads and writes files on this machine) to your network.",
            addr.ip()
        );
    }
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("binding {addr}"))?;
        let shown = if addr.ip().is_unspecified() { format!("127.0.0.1:{}", opts.port) } else { addr.to_string() };
        let url = format!("http://{shown}/?token={token}");
        eprintln!("NeuronDump web UI: {url}");
        eprintln!("API token header: {TOKEN_HEADER}: {token}");
        if opts.open {
            open_browser(&url);
        }
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}
