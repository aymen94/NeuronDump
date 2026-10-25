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

use neurondump_core::compare::compare;
use neurondump_core::export::{render as export_render, Format, Table};
use neurondump_core::graph::{GroupBy, Level, ViewOptions};
use neurondump_core::inspect::{inspect, tensor_rows, InspectOptions, InspectReport};
use neurondump_core::stats::{tensor_stats, TensorStats};
use neurondump_core::template::template_report;
use neurondump_core::tokenizer::VocabQuery;
use neurondump_core::{App, JobKind, ModelSummary, NoProgress};

pub const TOKEN_HEADER: &str = "x-neurondump-token";

#[derive(RustEmbed)]
#[folder = "frontend/"]
struct Assets;

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

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        let msg = format!("{e:#}");
        let code = if msg.contains("unknown model id") || msg.contains("no model named") || msg.contains("no tensor") || msg.contains("no node") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::BAD_REQUEST
        };
        ApiError(code, msg)
    }
}

fn not_found(what: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("{what} not found"))
}

type ApiResult<T> = Result<T, ApiError>;

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> ApiResult<T> {
    tokio::task::spawn_blocking(f).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?.map_err(Into::into)
}

impl AppState {
    pub fn new(app: Arc<App>, token: String) -> Arc<Self> {
        Arc::new(Self {
            app,
            token,
            summaries: Mutex::new(None),
            reports: Mutex::new(HashMap::new()),
            tensor_stats: Mutex::new(HashMap::new()),
        })
    }

    fn invalidate(&self) {
        *self.summaries.lock().unwrap() = None;
        self.reports.lock().unwrap().clear();
        self.tensor_stats.lock().unwrap().clear();
    }

    /// Header-only model summaries, cached until a rescan or a change on disk.
    fn summaries(&self) -> Vec<ModelSummary> {
        if self.app.refresh_if_changed() {
            self.invalidate();
        }
        let cached = self.summaries.lock().unwrap().clone();
        match cached {
            Some(r) => r,
            None => {
                let r = self.app.library().summaries();
                *self.summaries.lock().unwrap() = Some(r.clone());
                r
            }
        }
    }

    /// Cheap inspect report (no stats) per model id, cached in memory.
    fn report(&self, id: &str) -> anyhow::Result<Arc<InspectReport>> {
        if let Some(r) = self.reports.lock().unwrap().get(id) {
            return Ok(r.clone());
        }
        let m = self.app.open_id(id)?;
        let r = Arc::new(inspect(&m, &InspectOptions::default(), &NoProgress)?);
        self.reports.lock().unwrap().insert(id.to_string(), r.clone());
        Ok(r)
    }
}

async fn auth(State(s): St, req: Request, next: Next) -> Response {
    let header_ok = req.headers().get(TOKEN_HEADER).and_then(|v| v.to_str().ok()) == Some(s.token.as_str());
    // EventSource and downloads cannot set headers: accept ?token= as well.
    let query_ok = req
        .uri()
        .query()
        .map(|q| q.split('&').any(|kv| kv.strip_prefix("token=") == Some(s.token.as_str())))
        .unwrap_or(false);
    if header_ok || query_ok {
        next.run(req).await
    } else {
        ApiError(StatusCode::UNAUTHORIZED, "missing or invalid session token".into()).into_response()
    }
}

async fn static_file(Path(path): Path<String>) -> Response {
    serve_asset(&path)
}

async fn index() -> Response {
    serve_asset("index.html")
}

fn serve_asset(path: &str) -> Response {
    if path.contains("..") {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match Assets::get(path) {
        Some(f) => {
            let mime = match path.rsplit('.').next() {
                Some("html") => "text/html; charset=utf-8",
                Some("js") => "text/javascript; charset=utf-8",
                Some("css") => "text/css; charset=utf-8",
                Some("svg") => "image/svg+xml",
                Some("wasm") => "application/wasm",
                Some("json") => "application/json",
                _ => "application/octet-stream",
            };
            ([(header::CONTENT_TYPE, mime)], f.data.into_owned()).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn info(State(s): St) -> Json<Value> {
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "schema_version": neurondump_core::util::SCHEMA_VERSION,
        "output_dir": s.app.config.output_dir,
        "trace_architectures": neurondump_core::trace::supported_architectures(),
    }))
}

async fn models(State(s): St) -> ApiResult<Json<Value>> {
    let st = s.clone();
    let rows = blocking(move || Ok(st.summaries())).await?;
    let warnings = s.app.library().warnings.clone();
    // Paths are informational only; the API never accepts them back.
    Ok(Json(json!({ "models": rows, "warnings": warnings })))
}

async fn rescan(State(s): St) -> ApiResult<Json<Value>> {
    let st = s.clone();
    let n = blocking(move || {
        st.app.rescan();
        st.invalidate();
        Ok(st.app.library().entries.len())
    })
    .await?;
    Ok(Json(json!({ "models": n })))
}

async fn model(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let r = blocking(move || s.report(&id)).await?;
    Ok(Json(json!({
        "model": r.model,
        "file": r.file,
        "architecture": {
            "name": r.architecture.name,
            "kind": r.architecture.kind,
            "hyperparams": r.architecture.hyperparams,
            "keys": r.architecture.keys,
            "total_params": r.architecture.total_params,
            "params_by_component": r.architecture.params_by_component,
            "bytes_by_component": r.architecture.bytes_by_component,
            "types": r.architecture.types,
            "layer_kinds": r.architecture.layer_kinds,
            "unknown_roles": r.architecture.unknown_roles,
            "tied_embeddings": r.architecture.tied_embeddings,
        },
        "tokenizer": r.tokenizer,
        "ollama": r.ollama,
        "projector": r.projector.as_ref().map(|p| json!({
            "path": p.path, "architecture": p.architecture.name, "kind": p.architecture.kind,
            "params": p.architecture.total_params, "tensors": p.file.n_tensors, "size": p.file.file_size,
        })),
        "warnings": r.warnings,
        "timings": r.timings,
    })))
}

#[derive(Deserialize)]
struct MetaQuery {
    #[serde(default)]
    full: bool,
}

async fn metadata(State(s): St, Path(id): Path<String>, Query(q): Query<MetaQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        if q.full {
            let m = s.app.open_id(&id)?;
            Ok(serde_json::to_value(neurondump_core::inspect::metadata_entries(&m.gguf, true))?)
        } else {
            Ok(serde_json::to_value(&s.report(&id)?.metadata)?)
        }
    })
    .await?;
    Ok(Json(v))
}

async fn vocab(State(s): St, Path(id): Path<String>, Query(q): Query<VocabQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        let v = s.app.vocab(&m)?;
        let mut q = q;
        q.limit = Some(q.limit.unwrap_or(100).min(1000));
        let page = v.search(&q)?;
        Ok(json!({ "summary": v.summary, "page": page }))
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
struct TensorQuery {
    layer: Option<u32>,
    role: Option<String>,
    q: Option<String>,
    sort: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}

async fn tensors(State(s): St, Path(id): Path<String>, Query(q): Query<TensorQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        let rows = tensor_rows(&m.gguf);
        let mut sel: Vec<_> = rows
            .into_iter()
            .filter(|r| q.layer.is_none_or(|l| r.layer == Some(l)))
            .filter(|r| q.role.as_ref().is_none_or(|ro| ro.is_empty() || r.role.contains(ro.as_str())))
            .filter(|r| q.q.as_ref().is_none_or(|n| n.is_empty() || r.name.contains(n.as_str())))
            .collect();
        match q.sort.as_deref().unwrap_or("offset") {
            "size" => sel.sort_by(|a, b| b.n_bytes.cmp(&a.n_bytes)),
            "type" => sel.sort_by(|a, b| a.ggml_type.cmp(&b.ggml_type).then(a.name.cmp(&b.name))),
            "layer" => sel.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.name.cmp(&b.name))),
            "name" => sel.sort_by(|a, b| a.name.cmp(&b.name)),
            _ => sel.sort_by_key(|r| r.offset),
        }
        let total = sel.len();
        let offset = q.offset.unwrap_or(0).min(total);
        let limit = q.limit.unwrap_or(200).min(2000);
        let page: Vec<_> = sel.into_iter().skip(offset).take(limit).collect();
        Ok(json!({ "total": total, "offset": offset, "limit": limit, "tensors": page }))
    })
    .await?;
    Ok(Json(v))
}

async fn layers(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let r = blocking(move || s.report(&id)).await?;
    Ok(Json(json!({ "layers": r.architecture.layers, "layer_kinds": r.architecture.layer_kinds })))
}

async fn template(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        Ok(serde_json::to_value(template_report(&m))?)
    })
    .await?;
    Ok(Json(v))
}

async fn one_tensor_stats(State(s): St, Path((id, name)): Path<(String, String)>) -> ApiResult<Json<Value>> {
    let key = (id.clone(), name.clone());
    if let Some(t) = s.tensor_stats.lock().unwrap().get(&key) {
        return Ok(Json(serde_json::to_value(t.as_ref()).unwrap_or_default()));
    }
    let st = s.clone();
    let t = blocking(move || {
        let m = st.app.open_id(&id)?;
        let info = m.gguf.tensor(&name).ok_or_else(|| anyhow::anyhow!("no tensor {name:?}"))?;
        tensor_stats(&m.gguf, info, &NoProgress, &|_| {})
    })
    .await?;
    let t = Arc::new(t);
    s.tensor_stats.lock().unwrap().insert(key, t.clone());
    Ok(Json(serde_json::to_value(t.as_ref()).unwrap_or_default()))
}

#[derive(Deserialize)]
struct JobRequest {
    kind: String,
    model: String,
    #[serde(default)]
    params: Value,
}

async fn create_job(State(s): St, Json(req): Json<JobRequest>) -> ApiResult<(StatusCode, Json<Value>)> {
    let kind = JobKind::parse(&req.kind).ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, format!("unknown job kind {:?}", req.kind)))?;
    // Only opaque ids from discovery are accepted, never paths or names.
    if s.app.library().by_id(&req.model).is_none() {
        return Err(not_found("model"));
    }
    let mut params = req.params;
    if kind == JobKind::Analyze {
        // Corpus files are paths on the server's disk; the web UI sends text instead.
        if let Some(o) = params.as_object_mut() {
            o.remove("corpus_files");
        }
    }
    if kind == JobKind::Slim {
        if let Some(out) = params.get("out").and_then(Value::as_str) {
            neurondump_core::slim::safe_file_name(out).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;
        }
        if let Some(o) = params.as_object_mut() {
            o.remove("perplexity_corpus");
        }
    }
    if matches!(kind, JobKind::Trace | JobKind::Ablate | JobKind::Sweep) {
        // Corpus files are paths on the server's disk; the web UI sends texts instead.
        if let Some(o) = params.as_object_mut() {
            o.remove("corpus_file");
        }
    }
    let id = s.app.submit(kind, &req.model, params)?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "id": id }))))
}

async fn list_jobs(State(s): St) -> Json<Value> {
    Json(serde_json::to_value(s.app.jobs.list()).unwrap_or_default())
}

async fn get_job(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let j = s.app.jobs.get(&id).ok_or_else(|| not_found("job"))?;
    Ok(Json(serde_json::to_value(j).unwrap_or_default()))
}

async fn cancel_job(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    if s.app.jobs.get(&id).is_none() {
        return Err(not_found("job"));
    }
    let ok = s.app.jobs.cancel(&id);
    Ok(Json(json!({ "cancelling": ok })))
}

async fn job_result(State(s): St, Path(id): Path<String>) -> ApiResult<Response> {
    let j = s.app.jobs.get(&id).ok_or_else(|| not_found("job"))?;
    if !j.status.is_finished() {
        return Err(ApiError(StatusCode::CONFLICT, "job not finished".into()));
    }
    let Some(p) = j.result_path else {
        return Err(ApiError(StatusCode::NOT_FOUND, j.error.unwrap_or_else(|| "no result".into())));
    };
    let bytes = tokio::fs::read(p).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(([(header::CONTENT_TYPE, "application/json")], bytes).into_response())
}

async fn job_events(State(s): St, Path(id): Path<String>) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    if s.app.jobs.get(&id).is_none() {
        return Err(not_found("job"));
    }
    let st = stream::unfold((s, id, u64::MAX, false), |(s, id, last, done)| async move {
        if done {
            return None;
        }
        loop {
            let j = s.app.jobs.get(&id)?;
            if j.version != last || j.status.is_finished() {
                let finished = j.status.is_finished();
                let ev = Event::default().event(if finished { "done" } else { "progress" }).json_data(&j).unwrap_or_default();
                return Some((Ok(ev), (s, id, j.version, finished)));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    });
    Ok(Sse::new(st).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    /// Model id to also search tokens, tensors and metadata in.
    model: Option<String>,
    limit: Option<usize>,
}

async fn search(State(s): St, Query(q): Query<SearchQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let models = s.summaries();
        let limit = q.limit.unwrap_or(10).min(200);
        let r = match q.model.as_deref().filter(|m| !m.is_empty()) {
            Some(id) => {
                let m = s.app.open_id(id)?;
                let vocab = s.app.vocab(&m).ok();
                neurondump_core::search::search(&q.q, &models, Some((&m, vocab.as_deref())), limit)
            }
            None => neurondump_core::search::search(&q.q, &models, None, limit),
        };
        Ok(serde_json::to_value(r)?)
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
struct CompareQuery {
    models: String,
}

async fn compare_models(State(s): St, Query(q): Query<CompareQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let ids: Vec<&str> = q.models.split(',').filter(|x| !x.is_empty()).collect();
        if ids.len() < 2 {
            anyhow::bail!("compare needs at least two model ids");
        }
        let hs = ids.iter().map(|i| s.app.open_id(i)).collect::<anyhow::Result<Vec<_>>>()?;
        let refs: Vec<_> = hs.iter().map(|h| h.as_ref()).collect();
        Ok(serde_json::to_value(compare(&refs))?)
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
struct ExportQuery {
    format: Option<String>,
    table: Option<String>,
}

async fn export(State(s): St, Path(id): Path<String>, Query(q): Query<ExportQuery>) -> ApiResult<Response> {
    let (text, fmt, name) = blocking(move || {
        let f = Format::parse(q.format.as_deref().unwrap_or("json"))?;
        let t = Table::parse(q.table.as_deref().unwrap_or("tensors"))?;
        let m = s.app.open_id(&id)?;
        let r = inspect(&m, &InspectOptions { full: f == Format::Json, stats: false }, &NoProgress)?;
        let tokens = if t == Table::Vocab { Some(s.app.vocab(&m)?) } else { None };
        let text = export_render(&r, tokens.as_ref().map(|v| v.tokens.as_slice()), f, t)?;
        let base: String = m.entry.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
        let suffix = if f == Format::Json { "inspect".to_string() } else { q.table.clone().unwrap_or_else(|| "tensors".into()) };
        Ok((text, f, format!("{base}.{suffix}.{}", f.extension())))
    })
    .await?;
    let mut resp = Response::new(Body::from(text));
    resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(fmt.content_type()));
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}\"")) {
        resp.headers_mut().insert(header::CONTENT_DISPOSITION, v);
    }
    Ok(resp)
}

// --- graph / ablation ---------------------------------------------------------------

#[derive(Deserialize)]
struct GraphQuery {
    level: Option<String>,
    group_by: Option<String>,
    /// Comma-separated node ids to expand (lazy fine level).
    expand: Option<String>,
}

async fn graph(State(s): St, Path(id): Path<String>, Query(q): Query<GraphQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        let g = s.app.graph(&m)?;
        let level = match q.level.as_deref().filter(|l| !l.is_empty()) {
            Some(l) => Level::parse(l).ok_or_else(|| anyhow::anyhow!("level must be section, layer, block, tensor or fine"))?,
            None => Level::Layer,
        };
        let group_by = match q.group_by.as_deref().filter(|l| !l.is_empty()) {
            Some(l) => GroupBy::parse(l).ok_or_else(|| anyhow::anyhow!("group_by must be section, layer, block, quant, kind or importance"))?,
            None => GroupBy::Section,
        };
        let expand: Vec<String> = q.expand.unwrap_or_default().split(',').map(str::trim).filter(|x| !x.is_empty()).map(String::from).collect();
        Ok(serde_json::to_value(g.view(&ViewOptions { level, group_by, expand }))?)
    })
    .await?;
    Ok(Json(v))
}

async fn graph_node(State(s): St, Path((id, node)): Path<(String, String)>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        let g = s.app.graph(&m)?;
        let rid = g.resolve_id(&node).ok_or_else(|| anyhow::anyhow!("no node {node:?}"))?;
        let d = g.detail(&rid).ok_or_else(|| anyhow::anyhow!("no node {node:?}"))?;
        Ok(serde_json::to_value(d)?)
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
struct CutBody {
    #[serde(default)]
    cut: Vec<String>,
    #[serde(default)]
    force: bool,
    /// plan-from-ablation: use a finished sweep job instead of `cut`.
    #[serde(default)]
    sweep_job: Option<String>,
    /// plan-from-ablation with a sweep: number of least damaging entries.
    #[serde(default)]
    top: Option<usize>,
}

async fn impact(State(s): St, Path(id): Path<String>, Json(b): Json<CutBody>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        Ok(serde_json::to_value(s.app.impact(&m, &b.cut, b.force)?)?)
    })
    .await?;
    Ok(Json(v))
}

async fn plan_from_ablation(State(s): St, Path(id): Path<String>, Json(b): Json<CutBody>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        if let Some(job) = &b.sweep_job {
            let r = s.app.jobs.result(job).ok_or_else(|| anyhow::anyhow!("no finished sweep job {job:?}"))?;
            let sw: neurondump_core::trace::SweepReport = serde_json::from_value(r).map_err(|_| anyhow::anyhow!("job {job:?} is not a sweep"))?;
            let (plan, impact) = s.app.plan_from_sweep(&m, &sw, b.top.unwrap_or(1), b.force)?;
            return Ok(json!({ "plan": plan, "impact": impact }));
        }
        let plan = s.app.plan_from_cut(&m, &b.cut, b.force)?;
        Ok(json!({ "plan": plan }))
    })
    .await?;
    Ok(Json(v))
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
