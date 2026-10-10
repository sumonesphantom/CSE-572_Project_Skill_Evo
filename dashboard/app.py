"""SkillEvo dashboard: is the skill library evolving in a positive or negative way?

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skillevo.analysis import LABELS, Run, list_runs, monitor_audit, summarize  # noqa: E402

st.set_page_config(page_title="SkillEvo monitor", layout="wide")

# Fixed categorical slots per arm (color follows the entity, never its rank).
ARM_COLOR = {"guarded": "#2a78d6", "unguarded": "#eb6834", "append_only": "#1baf7a", "noskill": "#898781"}
# Reserved status colors for verdicts / health, always shown with a glyph + label.
STATUS = {"positive": "#0ca30c", "helpful": "#0ca30c", "negative": "#d03b3b", "harmful": "#d03b3b",
          "neutral": "#898781", "uncertain": "#898781", "inert": "#c3c2b7", "unused": "#c3c2b7"}
GLYPH = {"positive": "▲", "helpful": "▲", "negative": "▼", "harmful": "▼", "neutral": "●",
         "uncertain": "●", "inert": "○", "unused": "○"}
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
              hovermode="x unified", font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))


def badge(label: str) -> str:
    return f"{GLYPH.get(label, '')} {label}"


@st.cache_data(ttl=20)
def load(paths: tuple[str, ...]):
    runs = [Run(p) for p in paths]
    return runs, pd.DataFrame([summarize(r) for r in runs])


runs_dir = ROOT / "runs"
available = [str(p) for p in list_runs(runs_dir)]
st.sidebar.title("SkillEvo")
if not available:
    st.info(f"No runs yet in {runs_dir}. Start one with `python run.py --arm guarded`.")
    st.stop()
chosen = st.sidebar.multiselect("Runs to compare", available, default=available,
                                format_func=lambda p: Path(p).name)
if st.sidebar.button("Refresh"):
    st.cache_data.clear()
if not chosen:
    st.stop()
runs, summary = load(tuple(chosen))
by_name = {r.name: r for r in runs}
color_of = {r.name: ARM_COLOR.get(r.arm, "#4a3aa7") for r in runs}

tab_over, tab_edits, tab_skills, tab_audit, tab_eps = st.tabs(
    ["Overview", "Edits & drift", "Skills & lineage", "Monitor audit", "Episodes"])

# ------------------------------------------------------------------ overview
with tab_over:
    st.subheader("Is evolution helping?")
    cols = st.columns(min(len(runs), 4))
    for c, r in zip(cols, runs):
        s = summary.set_index("run").loc[r.name]
        test_key = next((k for k in s.index if k.endswith("_final") and not k.startswith("val")), None)
        with c:
            st.markdown(f"**{r.name}**  \n<span style='color:#52514e'>{r.arm}</span>", unsafe_allow_html=True)
            if test_key:
                split = test_key[:-6]
                st.metric(f"{split} accuracy", f"{s[test_key]:.1%}", f"{s[split + '_gain']:+.1%} vs round 0")
            dd = s.get("val_max_drawdown", 0) or 0
            st.caption(f"val max drawdown {dd:.1%} · harmful edits kept {int(s.get('kept_negative', 0) or 0)}"
                       f" · rollbacks {int(s.get('rollbacks', 0) or 0)} · skills {int(s.get('library_final', 0) or 0)}")

    acc = pd.concat([r.accuracy().assign(run=r.name) for r in runs], ignore_index=True)
    splits = list(dict.fromkeys(acc.split))
    grid = st.columns(len(splits))
    for g, split in zip(grid, splits):
        d = acc[acc.split == split]
        fig = px.line(d, x="round", y="accuracy", color="run", markers=True, color_discrete_map=color_of,
                      title=f"{split} accuracy by round")
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_layout(**LAYOUT, yaxis_tickformat=".0%")
        g.plotly_chart(fig, width="stretch")
        rb = pd.concat([r.rounds.assign(run=r.name) for r in runs])
        rb = rb[rb.rolled_back == True]  # noqa: E712
        if len(rb) and split == "val":
            g.caption("Rollbacks at rounds: " + ", ".join(f"{x.run} r{x['round']}" for _, x in rb.iterrows()))

    size = pd.concat([r.rounds[["round", "library_size"]].assign(run=r.name) for r in runs])
    fig = px.line(size, x="round", y="library_size", color="run", markers=True, color_discrete_map=color_of,
                  title="Skill library size")
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Summary table (all metrics)"):
        st.dataframe(summary.set_index("run").T.astype(str), width="stretch")

# ------------------------------------------------------------ edits & drift
with tab_edits:
    name = st.selectbox("Run", list(by_name), key="edits_run")
    r = by_name[name]
    e = r.edits
    if e.empty:
        st.info("This run has no edits (no-skill baseline or still at round 0).")
    else:
        c1, c2 = st.columns(2)
        counts = e.groupby(["round", "v_label"]).size().reset_index(name="edits")
        fig = go.Figure()
        for lbl in LABELS:
            d = counts[counts.v_label == lbl]
            fig.add_bar(x=d["round"], y=d["edits"], name=badge(lbl), marker_color=STATUS[lbl],
                        marker_line_color="white", marker_line_width=2)
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, barmode="stack", title="Edit verdicts per round (validation probe)")
        c1.plotly_chart(fig, width="stretch")

        kept = e[e.accepted].sort_values("edit_id")
        drift = kept.groupby("round")[["v_net"] + (["a_net"] if "a_net" in kept else [])].sum().cumsum().reset_index()
        fig = go.Figure()
        fig.add_scatter(x=drift["round"], y=drift["v_net"], name="validation probe", mode="lines+markers",
                        line=dict(color="#2a78d6", width=2), marker_size=8)
        if "a_net" in drift:
            fig.add_scatter(x=drift["round"], y=drift["a_net"], name="held-out audit", mode="lines+markers",
                            line=dict(color="#eb6834", width=2), marker_size=8)
        fig.add_hline(y=0, line_color="#c3c2b7")
        fig.update_layout(**LAYOUT, title="Cumulative net tasks gained by kept edits", yaxis_title="tasks gained - lost")
        c2.plotly_chart(fig, width="stretch")

        show = e.assign(verdict=e.v_label.map(badge), audit=e.get("a_label", pd.Series(dtype=str)).map(badge),
                        kept=e.accepted.map({True: "kept", False: "rejected"}))
        cols = ["edit_id", "round", "summary", "source", "verdict", "v_net", "v_p_value", "v_n_probe", "v_n_relevant",
                "audit", "a_net", "kept", "rationale"]
        st.dataframe(show[[c for c in cols if c in show]], width="stretch", hide_index=True,
                     column_config={"v_p_value": st.column_config.NumberColumn("p (McNemar)", format="%.3f"),
                                    "v_net": "val net", "a_net": "audit net"})
        if not r.events.empty:
            st.warning("Rollbacks: " + "; ".join(f"round {x['round']} → snapshot of round {x['to_round']}"
                                                 for _, x in r.events[r.events.type == "rollback"].iterrows()))

# --------------------------------------------------------- skills & lineage
with tab_skills:
    name = st.selectbox("Run", list(by_name), key="skills_run")
    r = by_name[name]
    if r.stats.empty:
        st.info("No skill statistics yet.")
    else:
        rounds = sorted(r.stats["round"].unique())
        rd = st.select_slider("Round", rounds, value=rounds[-1])
        s = r.stats[r.stats["round"] == rd].copy().sort_values("lift")
        s["label"] = s.skill_id + " " + s.name.str.slice(0, 40)
        fig = go.Figure()
        for h in ("helpful", "uncertain", "harmful", "unused"):
            d = s[s.health == h]
            if d.empty:
                continue
            fig.add_bar(y=d.label, x=d.lift, orientation="h", name=badge(h), marker_color=STATUS[h],
                        error_x=dict(type="data", symmetric=False, array=d.ci_hi - d.lift, arrayminus=d.lift - d.ci_lo,
                                     color="#52514e", thickness=1),
                        customdata=d[["n", "loo", "success"]].values,
                        hovertemplate="%{y}<br>lift %{x:+.2f}<br>uses %{customdata[0]}<br>leave-one-out %{customdata[1]:+.2f}<extra></extra>")
        fig.add_vline(x=0, line_color="#c3c2b7")
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, title="Skill lift vs no-skill baseline (95% bootstrap CI)",
                          height=max(300, 40 * len(s) + 120), xaxis_tickformat="+.0%")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(s.assign(health=s.health.map(badge))[["skill_id", "version", "name", "health", "n", "success", "lift",
                                                           "ci_lo", "ci_hi", "loo", "loo_n"]],
                     width="stretch", hide_index=True)

        st.markdown("#### Lineage")
        v = r.versions
        if not v.empty:
            sid = st.selectbox("Skill", sorted(v.skill_id.unique()),
                               format_func=lambda x: f"{x} · {v[v.skill_id == x].name.iloc[-1]}")
            hist = r.stats[r.stats.skill_id == sid]
            if len(hist) > 1:
                fig = px.line(hist, x="round", y="lift", markers=True, title=f"{sid} lift over rounds")
                fig.update_traces(line_color="#2a78d6", line_width=2, marker_size=8)
                fig.add_hline(y=0, line_color="#c3c2b7")
                fig.update_layout(**LAYOUT, yaxis_tickformat="+.0%")
                st.plotly_chart(fig, width="stretch")
            edits = r.edits.set_index("edit_id") if not r.edits.empty else pd.DataFrame()
            for _, row in v[v.skill_id == sid].iterrows():
                ed = edits.loc[row.edit_id] if row.get("edit_id") in edits.index else None
                verdict = f" — {badge(ed.v_label)} (val net {ed.v_net:+d}, p={ed.v_p_value:.2f})" if ed is not None else ""
                head = f"Round {row['round']} · {row['event']} · {row['key']}{verdict}"
                with st.expander(head, expanded=row["event"] != "retired"):
                    if row["event"] == "retired":
                        st.write(f"Retired by {row.get('by')} → {row.get('into') or 'removed'}")
                        continue
                    if isinstance(row.get("parents"), list) and row["parents"]:
                        st.caption("parents: " + ", ".join(row["parents"]))
                    st.markdown(f"**When to use:** {row['description']}")
                    st.code(row["content"], language="markdown")
                    if isinstance(row.get("diff"), str) and row["diff"]:
                        st.code(row["diff"], language="diff")

# ------------------------------------------------------------ monitor audit
with tab_audit:
    st.subheader("Do validation verdicts hold on held-out tasks?")
    allE = pd.concat([r.edits.assign(run=r.name) for r in runs if not r.edits.empty], ignore_index=True) \
        if any(not r.edits.empty for r in runs) else pd.DataFrame()
    if allE.empty or "a_net" not in allE:
        st.info("Audit needs `monitor.audit_split` in the config and at least one edit.")
    else:
        m = monitor_audit(allE)
        c = st.columns(4)
        c[0].metric("Audited edits", m.get("n_audited", 0))
        c[1].metric("Sign agreement", f"{m.get('sign_agreement', float('nan')):.0%}")
        c[2].metric("Cohen's κ", f"{m.get('kappa', float('nan')):.2f}")
        c[3].metric("Precision of ▲ positive", f"{m.get('precision_positive', float('nan')):.0%}")
        fig = go.Figure()
        for lbl in LABELS:
            d = allE[allE.v_label == lbl]
            if d.empty:
                continue
            fig.add_scatter(x=d.v_net, y=d.a_net, mode="markers", name=badge(lbl),
                            marker=dict(color=STATUS[lbl], size=11, line=dict(color="white", width=2)),
                            text=d.run + " " + d.edit_id + ": " + d.summary,
                            hovertemplate="%{text}<br>val net %{x:+d}, audit net %{y:+d}<extra></extra>")
        fig.add_hline(y=0, line_color="#c3c2b7")
        fig.add_vline(x=0, line_color="#c3c2b7")
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, title="Edit effect: validation probe vs held-out audit",
                          xaxis_title="validation net tasks", yaxis_title="held-out net tasks")
        st.plotly_chart(fig, width="stretch")
        conf = pd.crosstab(allE.v_label.map(badge), allE.a_label.map(badge), rownames=["validation"], colnames=["held-out"])
        st.dataframe(conf, width="stretch")

# ------------------------------------------------------------------ episodes
with tab_eps:
    name = st.selectbox("Run", list(by_name), key="eps_run")
    t = by_name[name].trajectories
    if t.empty:
        st.info("No episodes logged yet.")
    else:
        c = st.columns(4)
        phase = c[0].selectbox("Phase", sorted(t.phase.unique()))
        rnd = c[1].selectbox("Round", sorted(t[t.phase == phase]["round"].unique(), reverse=True))
        outcome = c[2].selectbox("Outcome", ["all", "correct", "wrong", "regressed vs no-skill", "fixed vs no-skill"])
        d = t[(t.phase == phase) & (t["round"] == rnd)]
        if outcome == "correct":
            d = d[d.correct]
        elif outcome == "wrong":
            d = d[~d.correct]
        elif outcome == "regressed vs no-skill":
            d = d[~d.correct & d.baseline]
        elif outcome == "fixed vs no-skill":
            d = d[d.correct & ~d.baseline]
        c[3].metric("Episodes", len(d))
        d = d.assign(skills=d.skills.map(lambda x: ", ".join(x)))
        st.dataframe(d[["task_id", "difficulty", "correct", "baseline", "skills", "sql", "error", "repaired"]],
                     width="stretch", hide_index=True)
