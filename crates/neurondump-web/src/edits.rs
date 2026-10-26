//! Live edit endpoints: the stack, push / toggle / delete, undo / redo / restore, export, snapshots, bake.

use super::*;

pub(super) fn edit_view(s: &AppState, id: &str) -> anyhow::Result<Value> {
    let m = s.app.open_id(id)?;
    Ok(serde_json::to_value(s.app.edit_view(&m)?)?)
}

pub(super) async fn get_edits(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(blocking(move || edit_view(&s, &id)).await?))
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum EditBody {
    /// Import / replace: `{"edits": [...]}` (the edits.json format).
    Replace(neurondump_core::edits::EditsFile),
    Push {
        #[serde(flatten)]
        op: EditOp,
        #[serde(default)]
        label: Option<String>,
    },
}

pub(super) async fn post_edit(State(s): St, Path(id): Path<String>, Json(b): Json<EditBody>) -> ApiResult<Json<Value>> {
    Ok(Json(
        blocking(move || {
            let m = s.app.open_id(&id)?;
            let mut warnings = Vec::new();
            match b {
                EditBody::Replace(f) => warnings = s.app.import_edits(&m, f)?,
                EditBody::Push { op, label } => {
                    s.app.push_edit(&m, op, label)?;
                }
            }
            let mut v = edit_view(&s, &id)?;
            v["warnings"] = json!(warnings);
            Ok(v)
        })
        .await?,
    ))
}

/// Restore all (back to the original model; undoable).
pub(super) async fn delete_edits(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(blocking(move || edit_action(&s, &id, "restore")).await?))
}

pub(super) async fn delete_edit(State(s): St, Path((id, n)): Path<(String, u64)>) -> ApiResult<Json<Value>> {
    Ok(Json(
        blocking(move || {
            let m = s.app.open_id(&id)?;
            s.app.edit(&m, |st| st.remove(n))?;
            edit_view(&s, &id)
        })
        .await?,
    ))
}

pub(super) async fn toggle_edit(State(s): St, Path((id, n)): Path<(String, u64)>) -> ApiResult<Json<Value>> {
    Ok(Json(
        blocking(move || {
            let m = s.app.open_id(&id)?;
            s.app.edit(&m, |st| st.toggle(n))?;
            edit_view(&s, &id)
        })
        .await?,
    ))
}

pub(super) fn edit_action(s: &AppState, id: &str, action: &str) -> anyhow::Result<Value> {
    let m = s.app.open_id(id)?;
    let changed = s.app.edit(&m, |st| {
        Ok(match action {
            "undo" => st.undo(),
            "redo" => st.redo(),
            _ => {
                st.restore();
                true
            }
        })
    })?;
    let mut v = edit_view(s, id)?;
    v["changed"] = json!(changed);
    Ok(v)
}

pub(super) async fn undo_edits(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(blocking(move || edit_action(&s, &id, "undo")).await?))
}

pub(super) async fn redo_edits(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(blocking(move || edit_action(&s, &id, "redo")).await?))
}

pub(super) async fn restore_edits(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(blocking(move || edit_action(&s, &id, "restore")).await?))
}

/// The stack as a portable edits.json download.
pub(super) async fn export_edits(State(s): St, Path(id): Path<String>) -> ApiResult<Response> {
    let (text, name) = blocking(move || {
        let m = s.app.open_id(&id)?;
        let f = neurondump_core::edits::EditsFile::new(&m, None, s.app.edit_stack(&m).edits);
        let base: String = m.entry.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
        Ok((serde_json::to_string_pretty(&f)?, format!("{base}.edits.json")))
    })
    .await?;
    let mut resp = Response::new(Body::from(text));
    resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}\"")) {
        resp.headers_mut().insert(header::CONTENT_DISPOSITION, v);
    }
    Ok(resp)
}

pub(super) async fn list_snapshots(State(s): St, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(
        blocking(move || {
            let m = s.app.open_id(&id)?;
            Ok(json!({ "snapshots": neurondump_core::edits::list_snapshots(&s.app.config.output_dir, &m) }))
        })
        .await?,
    ))
}

#[derive(Deserialize)]
pub(super) struct SnapshotBody {
    pub(super) name: String,
    /// `save` (default) or `load`.
    #[serde(default)]
    pub(super) action: Option<String>,
}

pub(super) async fn post_snapshot(State(s): St, Path(id): Path<String>, Json(b): Json<SnapshotBody>) -> ApiResult<Json<Value>> {
    Ok(Json(
        blocking(move || {
            let m = s.app.open_id(&id)?;
            let mut warnings = Vec::new();
            match b.action.as_deref().unwrap_or("save") {
                "save" => {
                    s.app.save_snapshot(&m, &b.name)?;
                }
                "load" => warnings = s.app.load_snapshot(&m, &b.name)?,
                other => anyhow::bail!("unknown snapshot action {other:?} (save, load)"),
            }
            let mut v = edit_view(&s, &id)?;
            v["warnings"] = json!(warnings);
            Ok(v)
        })
        .await?,
    ))
}

#[derive(Deserialize)]
pub(super) struct BakeBody {
    pub(super) out: String,
}

/// Writes the enabled edits into a new GGUF under `<output_dir>/bake/` (a job).
pub(super) async fn post_bake(State(s): St, Path(id): Path<String>, Json(b): Json<BakeBody>) -> ApiResult<(StatusCode, Json<Value>)> {
    neurondump_core::slim::safe_file_name(&b.out).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;
    if s.app.library().by_id(&id).is_none() {
        return Err(not_found("model"));
    }
    let jid = s.app.submit(JobKind::Bake, &id, json!({ "out": b.out }))?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "id": jid }))))
}
