//! Graph endpoints: views, node details, static impact of a cut, plan from an ablation.

use super::*;

pub(super) async fn graph(State(s): St, Path(id): Path<String>, Query(q): Query<api::GraphQuery>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        api::graph(&*s.app.graph(&m)?, &q)
    })
    .await?;
    Ok(Json(v))
}

pub(super) async fn graph_node(State(s): St, Path((id, node)): Path<(String, String)>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        api::node(&*s.app.graph(&m)?, &node)
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
pub(super) struct CutBody {
    #[serde(default)]
    pub(super) cut: Vec<String>,
    #[serde(default)]
    pub(super) force: bool,
    /// plan-from-ablation: use a finished sweep job instead of `cut`.
    #[serde(default)]
    pub(super) sweep_job: Option<String>,
    /// plan-from-ablation with a sweep: number of least damaging entries.
    #[serde(default)]
    pub(super) top: Option<usize>,
}

pub(super) async fn impact(State(s): St, Path(id): Path<String>, Json(b): Json<CutBody>) -> ApiResult<Json<Value>> {
    let v = blocking(move || {
        let m = s.app.open_id(&id)?;
        Ok(serde_json::to_value(s.app.impact(&m, &b.cut, b.force)?)?)
    })
    .await?;
    Ok(Json(v))
}

pub(super) async fn plan_from_ablation(State(s): St, Path(id): Path<String>, Json(b): Json<CutBody>) -> ApiResult<Json<Value>> {
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


// --- token inspector / embedding map --------------------------------------------------
