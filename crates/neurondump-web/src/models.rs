//! Read-only model endpoints: info, list / rescan, overview, metadata, vocab, tensors, layers, template, stats, search, compare, export, tokenize, embedding map.

use super::*;

pub(super) async fn info(State(s): St) -> Json<Value> {
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "schema_version": neurondump_core::util::SCHEMA_VERSION,
        "output_dir": s.app.config.output_dir,
        "trace_architectures": neurondump_core::trace::supported_architectures(),
        "job_kinds": ["inspect", "stats", "analyze", "slim", "trace", "ablate", "sweep", "journey", "generate", "compare", "bake", "diff"],
    }))
}

pub(super) async fn models(State(s): St) -> ApiResult<Json<Value>> {
    let st = s.clone();
    let rows = blocking(move || Ok(st.summaries())).await?;
    let warnings = s.app.library().warnings.clone();
    // Paths are informational only; the API never accepts them back.
    Ok(Json(json!({ "models": rows, "warnings": warnings })))
}

pub(super) async fn rescan(State(s): St) -> ApiResult<Json<Value>> {
    let st = s.clone();
    let n = blocking(move || {
        st.app.rescan();
        st.invalidate();
        Ok(st.app.library().entries.len())
    })
    .await?;
    Ok(Json(json!({ "models": n })))
}

pub(super) async fn model(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let r = blocking(move || s.report(&id)).await?;
    Ok(Json(api::model(&r)))
}

#[derive(Deserialize)]
pub(super) struct MetaQuery {
    #[serde(default)]
    pub(super) full: bool,
}

pub(super) async fn metadata(State(s): St, Path(id): Path<String>, Query(q): Query<MetaQuery>) -> ApiResult<Json<Value>> {
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

pub(super) async fn vocab(State(s): St, Path(id): Path<String>, Query(q): Query<VocabQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        api::vocab(&*s.app.vocab(&m)?, &q)
    })
    .await?;
    Ok(Json(v))
}

pub(super) async fn tensors(State(s): St, Path(id): Path<String>, Query(q): Query<api::TensorQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || Ok(api::tensors(&s.app.open_id(&id)?.gguf, &q))).await?;
    Ok(Json(v))
}

pub(super) async fn layers(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let r = blocking(move || s.report(&id)).await?;
    Ok(Json(api::layers(&r)))
}

pub(super) async fn template(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        Ok(serde_json::to_value(template_report(&m))?)
    })
    .await?;
    Ok(Json(v))
}

pub(super) async fn one_tensor_stats(State(s): St, Path((id, name)): Path<(String, String)>) -> ApiResult<Json<Value>> {
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
pub(super) struct SearchQuery {
    pub(super) q: String,
    /// Model id to also search tokens, tensors and metadata in.
    pub(super) model: Option<String>,
    pub(super) limit: Option<usize>,
}

pub(super) async fn search(State(s): St, Query(q): Query<SearchQuery>) -> ApiResult<Json<Value>> {
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
pub(super) struct CompareQuery {
    pub(super) models: String,
}

pub(super) async fn compare_models(State(s): St, Query(q): Query<CompareQuery>) -> ApiResult<Json<Value>> {
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
pub(super) struct ExportQuery {
    pub(super) format: Option<String>,
    pub(super) table: Option<String>,
}

pub(super) async fn export(State(s): St, Path(id): Path<String>, Query(q): Query<ExportQuery>) -> ApiResult<Response> {
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

pub(super) async fn tokenize(State(s): St, Path(id): Path<String>, Json(req): Json<neurondump_core::tokens::TokenizeRequest>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        let vocab = s.app.vocab(&m)?;
        let banned = s.app.edit_view(&m).ok().and_then(|v| v.compiled).map(|c| c.intervention.mask_tokens).unwrap_or_default();
        Ok(serde_json::to_value(neurondump_core::tokens::tokenize(&m, &vocab, &req, &banned, &NoProgress)?)?)
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
pub(super) struct MapQuery {
    pub(super) format: Option<String>,
}

/// Binary Float32Array (x, y, z, class per token); `?format=json` describes it.
pub(super) async fn embedding_map(State(s): St, Path(id): Path<String>, Query(q): Query<MapQuery>) -> ApiResult<Response> {
    let json_only = q.format.as_deref() == Some("json");
    let map = blocking(move || {
        let m = s.app.open_id(&id)?;
        s.app.embedding_map(&m, &NoProgress)
    })
    .await?;
    if json_only {
        return Ok(Json(serde_json::to_value(map.info()).unwrap_or_default()).into_response());
    }
    let info = map.info();
    let bytes = tokio::task::spawn_blocking(move || map.to_f32_bytes()).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut resp = Response::new(Body::from(bytes));
    let h = resp.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
    for (k, v) in [("x-neurondump-count", info.count.to_string()), ("x-neurondump-stride", info.stride.to_string()), ("x-neurondump-explained", format!("{:.4},{:.4},{:.4}", info.explained[0], info.explained[1], info.explained[2]))] {
        if let Ok(v) = HeaderValue::from_str(&v) {
            h.insert(k, v);
        }
    }
    Ok(resp)
}

/// Every token's decoded text, by id (for map labels and search).
pub(super) async fn vocab_texts(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        Ok(api::vocab_texts(&*s.app.vocab(&m)?))
    })
    .await?;
    Ok(Json(v))
}

// --- edits ------------------------------------------------------------------------------
