//! Job endpoints: create (with parameter checks), list, get, cancel, result, live progress events.

use super::*;

#[derive(Deserialize)]
pub(super) struct JobRequest {
    pub(super) kind: String,
    pub(super) model: String,
    #[serde(default)]
    pub(super) params: Value,
}

pub(super) async fn create_job(State(s): St, Json(req): Json<JobRequest>) -> ApiResult<(StatusCode, Json<Value>)> {
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
    if kind == JobKind::Bake {
        let out = params.get("out").and_then(Value::as_str).unwrap_or("");
        neurondump_core::slim::safe_file_name(out).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;
    }
    if kind == JobKind::Diff {
        let other = params.get("other").and_then(Value::as_str).unwrap_or("");
        if s.app.library().by_id(other).is_none() {
            return Err(not_found("model `other`"));
        }
    }
    if matches!(kind, JobKind::Trace | JobKind::Ablate | JobKind::Sweep | JobKind::Compare) {
        // Corpus files are paths on the server's disk; the web UI sends texts instead.
        if let Some(o) = params.as_object_mut() {
            o.remove("corpus_file");
        }
    }
    let id = s.app.submit(kind, &req.model, params)?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "id": id }))))
}

pub(super) async fn list_jobs(State(s): St) -> Json<Value> {
    Json(serde_json::to_value(s.app.jobs.list()).unwrap_or_default())
}

pub(super) async fn get_job(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let j = s.app.jobs.get(&id).ok_or_else(|| not_found("job"))?;
    Ok(Json(serde_json::to_value(j).unwrap_or_default()))
}

pub(super) async fn cancel_job(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    if s.app.jobs.get(&id).is_none() {
        return Err(not_found("job"));
    }
    let ok = s.app.jobs.cancel(&id);
    Ok(Json(json!({ "cancelling": ok })))
}

pub(super) async fn job_result(State(s): St, Path(id): Path<String>) -> ApiResult<Response> {
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

pub(super) async fn job_events(State(s): St, Path(id): Path<String>) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
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
