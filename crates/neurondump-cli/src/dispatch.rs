//! `run_command`: executes one subcommand by calling into neurondump-core and rendering the result.

use super::*;

/// Executes one command and returns its output text.
pub fn run_command(ctx: &Ctx, cmd: &Cmd) -> Result<String> {
    let json = ctx.global.json;
    let app = &ctx.app;
    match cmd {
        Cmd::Cli | Cmd::Web { .. } => bail!("not available here"),
        Cmd::Config { action } => {
            let mut c = Config::load();
            match action.clone().unwrap_or(ConfigAction::Show) {
                ConfigAction::Show => Ok(if json { to_json(&c)? } else { format!("# {}\n{}", config_path().display(), c.to_toml()) }),
                ConfigAction::Set { key, value } => {
                    c.set(&key, &value)?;
                    let p = c.save()?;
                    Ok(format!("saved {}\n", p.display()))
                }
                ConfigAction::Reset => {
                    let p = Config::default().save()?;
                    Ok(format!("reset {}\n", p.display()))
                }
                ConfigAction::Path => Ok(format!("{}\n", config_path().display())),
            }
        }
        Cmd::List => {
            let lib = app.library();
            let rows = lib.summaries();
            if json {
                to_json(&json!({ "schema_version": 1, "models": rows, "warnings": lib.warnings }))
            } else {
                Ok(render::list(&rows, &lib.warnings))
            }
        }
        Cmd::Inspect { model, all, stats, full, out, metadata } => {
            let opts = InspectOptions { full: *full, stats: *stats };
            if *all {
                return inspect_all(ctx, &opts, out.as_ref());
            }
            let Some(model) = model else { bail!("give a model name or --all") };
            let t = Instant::now();
            let r = inspect_one(ctx, model, &opts)?;
            if ctx.global.timings {
                eprintln!(
                    "timings: architecture {} ms, tokenizer {} ms, templates {} ms, stats {} ms, total {} ms",
                    r.timings.architecture_ms, r.timings.tokenizer_ms, r.timings.template_ms, r.timings.stats_ms, t.elapsed().as_millis()
                );
            }
            let v = serde_json::to_value(&r)?;
            if let Some(dir) = out {
                std::fs::create_dir_all(dir)?;
                let p = dir.join(format!("{}.inspect.json", file_safe(&r.model.name)));
                std::fs::write(&p, serde_json::to_vec_pretty(&v)?)?;
                if !ctx.global.quiet {
                    eprintln!("wrote {}", p.display());
                }
            }
            if json {
                return to_json(&v);
            }
            let mut s = render::inspect(&r);
            if *metadata {
                s.push_str(&render::metadata(&r));
            }
            Ok(s)
        }
        Cmd::Search { query, model, limit } => {
            let models = app.library().summaries();
            let r = match model {
                Some(m) => {
                    let h = app.open(m)?;
                    let v = app.vocab(&h).ok();
                    neurondump_core::search::search(query, &models, Some((&h, v.as_deref())), *limit)
                }
                None => neurondump_core::search::search(query, &models, None, *limit),
            };
            if json {
                to_json(&r)
            } else {
                Ok(render::search(&r))
            }
        }
        Cmd::Vocab { model, search, regex, class, script, lang, limit, offset, tokenize } => {
            let m = app.open(model)?;
            let v = app.vocab(&m)?;
            if let Some(text) = tokenize {
                let enc = Encoder::new(&v)?;
                let e = enc.encode(text, true, None);
                if json {
                    return to_json(&e);
                }
                let mut s = String::new();
                for w in &enc.warnings {
                    s.push_str(&format!("warning: {w}\n"));
                }
                s.push_str(&format!("{} tokens\n", e.ids.len()));
                for (id, p) in e.ids.iter().zip(&e.pieces) {
                    s.push_str(&format!("{id:>8} -> '{}'\n", render::esc(p)));
                }
                return Ok(s);
            }
            let q = VocabQuery {
                q: search.clone(),
                regex: *regex,
                class: class.clone(),
                script: script.clone(),
                lang: lang.clone(),
                offset: Some(*offset),
                limit: Some(*limit),
            };
            let page = v.search(&q)?;
            if json {
                to_json(&json!({ "summary": v.summary, "page": page }))
            } else {
                Ok(render::vocab(&page))
            }
        }
        Cmd::Tensors { model, layer, role, sort, limit } => {
            let m = app.open(model)?;
            let rows = neurondump_core::inspect::tensor_rows(&m.gguf);
            let mut sel: Vec<_> = rows
                .iter()
                .filter(|r| layer.is_none_or(|l| r.layer == Some(l)))
                .filter(|r| role.as_ref().is_none_or(|ro| r.role == *ro || r.role.contains(ro.as_str())))
                .collect();
            match sort.as_str() {
                "size" => sel.sort_by(|a, b| b.n_bytes.cmp(&a.n_bytes)),
                "type" => sel.sort_by(|a, b| a.ggml_type.cmp(&b.ggml_type).then(a.name.cmp(&b.name))),
                "layer" => sel.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.name.cmp(&b.name))),
                "name" => sel.sort_by(|a, b| a.name.cmp(&b.name)),
                "offset" => sel.sort_by_key(|r| r.offset),
                other => bail!("unknown sort {other:?} (size, type, layer, name, offset)"),
            }
            if let Some(l) = limit {
                sel.truncate(*l);
            }
            if json {
                to_json(&sel)
            } else {
                Ok(render::tensors(&sel))
            }
        }
        Cmd::Layers { model } => {
            let m = app.open(model)?;
            let r = inspect(&m, &InspectOptions::default(), &NoProgress)?;
            if json {
                to_json(&r.architecture.layers)
            } else {
                Ok(render::layers(&r))
            }
        }
        Cmd::Template { model, raw } => {
            let m = app.open(model)?;
            let t = template_report(&m);
            if json {
                to_json(&t)
            } else {
                Ok(render::template(&t, *raw))
            }
        }
        Cmd::Stats { model, tensor, layer, limit } => {
            let v = ctx.run_job(JobKind::Stats, model, json!({ "tensors": tensor, "layers": layer }))?;
            if json {
                return to_json(&v);
            }
            let r: StatsReport = serde_json::from_value(v)?;
            Ok(render::stats(&r, *limit))
        }
        Cmd::Analyze { model, corpus, out, stats, fast, include_edge_layers } => {
            let params = json!({
                "corpus_files": corpus,
                "compute_stats": stats,
                "skip_layer_similarity": fast,
                "include_edge_layers": include_edge_layers,
            });
            let v = ctx.run_job(JobKind::Analyze, model, params)?;
            let plan: PrunePlan = serde_json::from_value(v.clone())?;
            let out = out.clone().unwrap_or_else(|| PathBuf::from(format!("{}.prune.json", file_safe(&plan.model))));
            std::fs::write(&out, serde_json::to_vec_pretty(&v)?).with_context(|| format!("writing {}", out.display()))?;
            if json {
                return to_json(&v);
            }
            Ok(format!("{}\nplan written to {} (edit `selected` flags or use `slim --only`)\n", render::plan(&plan), out.display()))
        }
        Cmd::Slim { model, plan, only, out, perplexity } => {
            let plan = App::load_plan(plan)?;
            let params = json!({ "plan": plan, "only": only, "out": out, "perplexity_corpus": perplexity });
            let v = ctx.run_job(JobKind::Slim, model, params)?;
            if json {
                return to_json(&v);
            }
            let r: SlimReport = serde_json::from_value(v)?;
            Ok(render::slim(&r))
        }
        Cmd::Trace { model, prompt, corpus, top_k, max_tokens } => {
            let params = json!({ "prompt": prompt, "corpus_file": corpus, "top_k": top_k, "max_tokens": max_tokens });
            let v = ctx.run_job(JobKind::Trace, model, params)?;
            if json {
                return to_json(&v);
            }
            let r: TraceReport = serde_json::from_value(v)?;
            Ok(render::trace(&r))
        }
        Cmd::Diff { a, b, top, metadata_only } => {
            let params = json!({ "other": b, "metadata_only": metadata_only, "top": if json { *top } else { None } });
            let v = ctx.run_job(JobKind::Diff, a, params)?;
            if json {
                return to_json(&v);
            }
            Ok(render::diff(&serde_json::from_value(v)?, top.unwrap_or(20)))
        }
        Cmd::Compare { models } => {
            let handles: Vec<_> = models.iter().map(|m| app.open(m)).collect::<Result<_>>()?;
            let refs: Vec<_> = handles.iter().map(|h| h.as_ref()).collect();
            let r = compare(&refs);
            if json {
                to_json(&r)
            } else {
                Ok(render::compare(&r))
            }
        }
        Cmd::Export { model, format, table, out } => {
            let f = Format::parse(format)?;
            let t = Table::parse(table)?;
            let m = app.open(model)?;
            let r = inspect(&m, &InspectOptions { full: f == Format::Json, stats: false }, &NoProgress)?;
            let tokens = if t == Table::Vocab { Some(app.vocab(&m)?) } else { None };
            let text = export_render(&r, tokens.as_ref().map(|v| v.tokens.as_slice()), f, t)?;
            match out {
                Some(p) => {
                    std::fs::write(p, &text)?;
                    Ok(format!("wrote {} ({} bytes)\n", p.display(), text.len()))
                }
                None => Ok(text),
            }
        }
        Cmd::Jobs => {
            let jobs = app.jobs.list();
            if json {
                to_json(&jobs)
            } else {
                Ok(render::jobs(&jobs))
            }
        }
        Cmd::Graph { model, level, group_by, format, expand, out } => {
            let m = app.open(model)?;
            let g = app.graph(&m)?;
            let level = Level::parse(level).ok_or_else(|| anyhow::anyhow!("--level must be section, layer, block, tensor or fine"))?;
            let group_by = GroupBy::parse(group_by).ok_or_else(|| anyhow::anyhow!("--group-by must be section, layer, block, quant, kind or importance"))?;
            for e in expand {
                if g.resolve_id(e).is_none() {
                    bail!("unknown node {e:?} (ids are shown after # in `graph`)");
                }
            }
            let v = g.view(&ViewOptions { level, group_by, expand: expand.clone() });
            let text = match (format.as_str(), json) {
                ("json", _) | (_, true) => to_json(&v)?,
                ("tree", _) => v.to_tree(),
                ("dot", _) => v.to_dot(),
                ("mermaid", _) => v.to_mermaid(),
                (other, _) => bail!("unknown --format {other:?} (tree, json, dot, mermaid)"),
            };
            match out {
                Some(p) => {
                    std::fs::write(p, &text).with_context(|| format!("writing {}", p.display()))?;
                    Ok(format!("wrote {} ({} nodes, {} edges)\n", p.display(), v.nodes.len(), v.edges.len()))
                }
                None => Ok(text),
            }
        }
        Cmd::Node { model, id } => {
            let m = app.open(model)?;
            let g = app.graph(&m)?;
            let rid = g.resolve_id(id).ok_or_else(|| anyhow::anyhow!("unknown node {id:?} (try `graph {model} --level tensor`)"))?;
            let d = g.detail(&rid).ok_or_else(|| anyhow::anyhow!("unknown node {id:?}"))?;
            if json {
                to_json(&d)
            } else {
                Ok(render::node(&d))
            }
        }
        Cmd::Ablate { model, cut, static_only, prompt, corpus, tokens, force, save_plan, max_tokens } => {
            let m = app.open(model)?;
            let impact = app.impact(&m, cut, *force)?;
            let mut s = render::impact(&impact);
            if impact.cut.is_empty() {
                bail!("{s}nothing to cut: unknown id(s) {}", impact.unknown.join(", "));
            }
            if impact.refused {
                if json {
                    write_stdout(&to_json(&json!({ "impact": impact }))?);
                } else {
                    write_stdout(&s);
                }
                bail!("refused: cutting this is `breaks_model`; pass --force to simulate or plan it anyway");
            }
            let mut plan_path = None;
            if let Some(p) = save_plan {
                let plan = app.plan_from_cut(&m, cut, *force)?;
                std::fs::write(p, serde_json::to_vec_pretty(&plan)?).with_context(|| format!("writing {}", p.display()))?;
                s.push_str(&format!("plan written to {} (apply with `slim {model} --plan {} --out <name>.gguf`)\n", p.display(), p.display()));
                plan_path = Some(p.clone());
            }
            let mut simulation = None;
            if !*static_only {
                if impact.simulation_supported {
                    let params = json!({
                        "cut": cut, "force": force, "prompt": prompt, "corpus_file": corpus, "tokens": tokens, "max_tokens": max_tokens,
                    });
                    let v = ctx.run_job(JobKind::Ablate, model, params)?;
                    let r: neurondump_core::ablate::AblateResult = serde_json::from_value(v)?;
                    if let Some(sim) = &r.simulation {
                        s.push('\n');
                        s.push_str(&render::simulation(sim));
                    }
                    simulation = r.simulation;
                } else {
                    s.push_str("(no simulation: see the notes above; showing static impact only)\n");
                }
            }
            if json {
                return to_json(&json!({ "impact": impact, "simulation": simulation, "plan": plan_path }));
            }
            Ok(s)
        }
        Cmd::Sweep { model, level, layer, corpus, out, max_tokens, save_plan, top, force } => {
            let params = json!({ "level": level, "layer": layer, "corpus_file": corpus, "max_tokens": max_tokens });
            let v = ctx.run_job(JobKind::Sweep, model, params)?;
            let r: neurondump_core::trace::SweepReport = serde_json::from_value(v.clone())?;
            let out = out.clone().unwrap_or_else(|| PathBuf::from(format!("{}.sweep.json", file_safe(&r.model))));
            std::fs::write(&out, serde_json::to_vec_pretty(&v)?).with_context(|| format!("writing {}", out.display()))?;
            let mut s = render::sweep(&r);
            s.push_str(&format!("sweep written to {}\n", out.display()));
            if let Some(p) = save_plan {
                let m = app.open(model)?;
                let (plan, impact) = app.plan_from_sweep(&m, &r, *top, *force)?;
                std::fs::write(p, serde_json::to_vec_pretty(&plan)?).with_context(|| format!("writing {}", p.display()))?;
                s.push_str(&format!(
                    "plan with the {top} least damaging {}(s) written to {} — {} → {} ({})\n",
                    r.level,
                    p.display(),
                    neurondump_core::util::human_bytes(impact.original_size),
                    neurondump_core::util::human_bytes(impact.new_size),
                    impact.verdict.as_str()
                ));
            }
            if json {
                return to_json(&v);
            }
            Ok(s)
        }
        Cmd::Tokens { model, text, chat, system, neighbors } => {
            let m = app.open(model)?;
            let v = app.vocab(&m)?;
            let req = TokenizeRequest { text: text.clone(), chat: *chat || system.is_some(), system: system.clone(), template: None, neighbors: *neighbors };
            let banned = app.edit_view(&m).ok().and_then(|v| v.compiled).map(|c| c.intervention.mask_tokens).unwrap_or_default();
            let bar = ctx.bar("tokens");
            let r = neurondump_core::tokens::tokenize(&m, &v, &req, &banned, &bar);
            bar.finish();
            let r = r?;
            if json {
                to_json(&r)
            } else {
                Ok(render::tokens(&r))
            }
        }
        Cmd::Journey { model, prompt, token, edits, top_k } => {
            let e = load_edits(edits.as_ref())?;
            let params = json!({
                "prompt": prompt.prompt, "chat": prompt.chat || prompt.system.is_some(), "system": prompt.system,
                "token": token, "top_k": top_k, "edits": e.unwrap_or_default(),
            });
            let v = ctx.run_job(JobKind::Journey, model, params)?;
            if json {
                return to_json(&v);
            }
            Ok(render::journey(&serde_json::from_value(v)?))
        }
        Cmd::Generate { model, prompt, steps, explain, temperature, top_k, top_p, seed, edits } => {
            let e = load_edits(edits.as_ref())?;
            let params = json!({
                "prompt": prompt.prompt, "chat": prompt.chat || prompt.system.is_some(), "system": prompt.system,
                "steps": steps, "explain": explain, "temperature": temperature, "sample_top_k": top_k, "top_p": top_p, "seed": seed,
                "edits": e.unwrap_or_default(),
            });
            let v = ctx.run_job(JobKind::Generate, model, params)?;
            if json {
                return to_json(&v);
            }
            Ok(render::generate(&serde_json::from_value(v)?, *explain))
        }
        Cmd::Edit { model, apply, prompt, chat, corpus, tokens, bake, no_compare } => {
            let e = load_edits(Some(apply))?.unwrap_or_default();
            let mut out = String::new();
            let mut jv = json!({});
            {
                // Validate the edits (and show what they compile to) before running anything.
                let m = app.open(model)?;
                let g = app.graph(&m)?;
                let v = app.vocab(&m).ok();
                let c = neurondump_core::edits::compile(&g, v.as_deref(), &e)?;
                out.push_str(&render::edits_list(&e, Some(&c)));
            }
            if !*no_compare {
                let params = json!({ "prompts": prompt, "chat": chat, "corpus_file": corpus, "tokens": tokens, "edits": e });
                let v = ctx.run_job(JobKind::Compare, model, params)?;
                let r: neurondump_core::app::CompareResult = serde_json::from_value(v.clone())?;
                out.push('\n');
                out.push_str(&render::simulation(&r.simulation));
                out.push_str(&render::compare_decisions(&r.simulation));
                jv["compare"] = v;
            }
            if let Some(path) = bake {
                let abs = std::path::absolute(path).unwrap_or_else(|_| path.clone());
                let v = ctx.run_job(JobKind::Bake, model, json!({ "edits": e, "out": abs }))?;
                let r: neurondump_core::edits::BakeReport = serde_json::from_value(v.clone())?;
                out.push('\n');
                out.push_str(&render::bake(&r));
                jv["bake"] = v;
            }
            if json {
                return to_json(&jv);
            }
            Ok(out)
        }
    }
}
