//! Shared server state (cached summaries / reports / tensor stats), API errors, blocking helper, token auth.

use super::*;

pub struct ApiError(pub(super) StatusCode, pub(super) String);

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

pub(super) fn not_found(what: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("{what} not found"))
}

pub(super) type ApiResult<T> = Result<T, ApiError>;

pub(super) async fn blocking<T: Send + 'static>(f: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> ApiResult<T> {
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

    pub(super) fn invalidate(&self) {
        *self.summaries.lock().unwrap() = None;
        self.reports.lock().unwrap().clear();
        self.tensor_stats.lock().unwrap().clear();
    }

    /// Header-only model summaries, cached until a rescan or a change on disk.
    pub(super) fn summaries(&self) -> Vec<ModelSummary> {
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
    pub(super) fn report(&self, id: &str) -> anyhow::Result<Arc<InspectReport>> {
        if let Some(r) = self.reports.lock().unwrap().get(id) {
            return Ok(r.clone());
        }
        let m = self.app.open_id(id)?;
        let r = Arc::new(inspect(&m, &InspectOptions::default(), &NoProgress)?);
        self.reports.lock().unwrap().insert(id.to_string(), r.clone());
        Ok(r)
    }
}

pub(super) async fn auth(State(s): St, req: Request, next: Next) -> Response {
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
