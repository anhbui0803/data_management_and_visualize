"""Stage 8 — Visualization: one self-contained HTML dashboard + one HTML file per figure.

Every figure is registered with its USE CASE (user · question · decision), which is
printed above the chart in the dashboard. Figures whose source table is missing
(stage skipped) are omitted with a note instead of failing.

Colours: validated reference categorical palette (first three slots validate all-pairs
for colour-vision deficiency), single-hue blue ramp for magnitude, text in neutral ink.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .config import Config
from .utils import LOG

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


@dataclass
class Figure:
    key: str
    title: str
    user: str
    question: str
    decision: str
    fig: go.Figure | None
    table: str
    note: str = ""


def base_layout(fig: go.Figure, height=420, legend=True, **kw) -> go.Figure:
    margin = kw.pop("margin", dict(l=70, r=30, t=30, b=60))
    fig.update_layout(
        template="plotly_white", height=height, font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, margin=margin,
        showlegend=legend, legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color=INK2))
    return fig


def _read(cfg: Config, name: str) -> pd.DataFrame | None:
    p = cfg.tables / name
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------------------- V1
def fig_cohort_flow(cfg):
    t = _read(cfg, "cohort_flow.csv")
    if t is None:
        return None
    labels = [f"{r.step} · {r.rule}" + (f" (−{r.rows_removed:,})" if r.rows_removed else "") for r in t.itertuples()]
    fig = go.Figure(go.Bar(
        y=labels, x=t["rows_after"], orientation="h", marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
        text=[f"{r.rows_after:,} rows · {r.patients_after:,} patients" for r in t.itertuples()],
        textposition="outside", textfont=dict(color=INK2),
        hovertemplate="%{y}<br>rows after: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Encounters remaining", range=[0, t["rows_after"].max() * 1.6])
    return base_layout(fig, height=320, legend=False)


# --------------------------------------------------------------------------- V2
def fig_dq_scorecard(cfg):
    sc = _read(cfg, "dq_scorecard.csv")
    if sc is None:
        return None
    sc = sc.sort_values("completeness")
    z = sc[["completeness", "validity"]].T.values
    text = [[("—" if pd.isna(v) else f"{v:.0%}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=sc["column"], y=["Completeness", "Validity"], colorscale=[[i / 6, c] for i, c in enumerate(SEQ)],
        zmin=0, zmax=1, text=text, texttemplate="", xgap=2, ygap=2,
        colorbar=dict(title="share", tickformat=".0%", outlinewidth=0),
        customdata=[sc["verdict"].tolist()] * 2,
        hovertemplate="%{x}<br>%{y}: %{text}<br>verdict: %{customdata}<extra></extra>"))
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=10))
    return base_layout(fig, height=360, legend=False, margin=dict(l=110, r=30, t=20, b=150))


# --------------------------------------------------------------------------- V3
def fig_missing_patterns(cfg, top=10):
    pat = _read(cfg, "missing_patterns.csv")
    if pat is None:
        return None
    pat = pat.head(top)
    cols = [c for c in pat.columns if c not in ("n", "share")]
    ylab = [f"{r.share:.1%} of patients" for r in pat.itertuples()]
    fig = go.Figure(go.Heatmap(
        z=pat[cols].values, x=cols, y=ylab, colorscale=[[0, "#f0efec"], [1, S1]], showscale=False,
        xgap=2, ygap=2, zmin=0, zmax=1,
        hovertemplate="pattern share: %{y}<br>%{x}: %{z} (1 = missing / not measured)<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, height=380, legend=False, margin=dict(l=140, r=30, t=20, b=60))


# --------------------------------------------------------------------------- V4
def fig_segments(cfg):
    seg = _read(cfg, "readmit_by_segment.csv")
    kpi = _read(cfg, "kpi.csv")
    if seg is None:
        return None
    seg = seg[seg["cohort"] == "all_encounters"]
    variables = ["discharge_grp", "diag1_grp", "age_group", "adm_source_grp", "a1c_group", "insulin_status",
                 "race_f", "specialty_grp"]
    nice = {"discharge_grp": "Discharge destination", "diag1_grp": "Primary diagnosis", "age_group": "Age group",
            "adm_source_grp": "Admission source", "a1c_group": "HbA1c group", "insulin_status": "Insulin",
            "race_f": "Race", "specialty_grp": "Admitting specialty"}
    fig = go.Figure()
    for i, v in enumerate(variables):
        s = seg[seg["variable"] == v].sort_values("rate")
        fig.add_trace(go.Bar(
            x=s["rate"], y=s["level"], orientation="h", visible=(i == 0), name=nice[v],
            marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, array=s["ci_high"] - s["rate"],
                         arrayminus=s["rate"] - s["ci_low"], color=INK2, thickness=1.2, width=3),
            customdata=s[["n", "events"]].values,
            hovertemplate="%{y}<br>rate %{x:.1%}<br>%{customdata[1]:,} / %{customdata[0]:,}<extra></extra>"))
    buttons = [dict(label=nice[v], method="update",
                    args=[{"visible": [j == i for j in range(len(variables))]}]) for i, v in enumerate(variables)]
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.15, xanchor="left",
                                        bgcolor="white", bordercolor=GRID)])
    if kpi is not None:
        overall = float(kpi.loc[kpi["cohort"] == "all_encounters", "readmit30_rate"].iloc[0])
        fig.add_vline(x=overall, line=dict(color=MUTED, dash="dot", width=1.5),
                      annotation_text=f"overall {overall:.1%}", annotation_position="bottom right",
                      annotation_font_color=INK2, annotation_bgcolor=SURFACE)
    fig.update_xaxes(title="30-day readmission rate (95% Wilson CI)", tickformat=".0%")
    return base_layout(fig, height=440, legend=False, margin=dict(l=200, r=30, t=60, b=60))


# --------------------------------------------------------------------------- static versions for print
def fig_segments_static(cfg, variables=("discharge_grp", "adm_source_grp", "age_group", "a1c_group"), min_n=100):
    """Small multiples of V4 for the printed report (no dropdown; four panels on one shared x-axis)."""
    from plotly.subplots import make_subplots
    seg = _read(cfg, "readmit_by_segment.csv")
    if seg is None:
        return None
    seg = seg[(seg["cohort"] == "all_encounters") & (seg["n"] >= min_n)]  # tiny levels are in the CSV / dashboard
    nice = {"discharge_grp": "Discharge destination", "adm_source_grp": "Admission source",
            "age_group": "Age group", "a1c_group": "HbA1c group", "diag1_grp": "Primary diagnosis"}
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, horizontal_spacing=0.22, vertical_spacing=0.12,
                        subplot_titles=[nice[v] for v in variables])
    for i, v in enumerate(variables):
        s = seg[seg["variable"] == v].sort_values("rate")
        fig.add_trace(go.Bar(x=s["rate"], y=s["level"], orientation="h", marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
                             error_x=dict(type="data", symmetric=False, array=s["ci_high"] - s["rate"],
                                          arrayminus=s["rate"] - s["ci_low"], color=INK2, thickness=1.2, width=3),
                             showlegend=False), row=i // 2 + 1, col=i % 2 + 1)
    xmax = float(seg[seg["variable"].isin(variables)]["ci_high"].max()) * 1.08
    fig.update_xaxes(tickformat=".0%", range=[0, xmax])
    fig.update_xaxes(title="30-day readmission rate (95% CI)", row=2)
    fig.update_yaxes(tickfont=dict(size=10))
    return base_layout(fig, height=620, legend=False, margin=dict(l=150, r=30, t=50, b=60))


def fig_dq_scorecard_static(cfg):
    """V2 for print: only fields below 100% completeness or validity, value labels shown.
    The full per-field table is tables/dq_scorecard.csv (report appendix)."""
    sc = _read(cfg, "dq_scorecard.csv")
    if sc is None:
        return None
    sc = sc[(sc["completeness"] < 1) | (sc["validity"].fillna(1) < 1)].sort_values("completeness")
    z = sc[["completeness", "validity"]].T.values
    text = [[("—" if pd.isna(v) else f"{v:.1%}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=sc["column"], y=["Completeness", "Validity"], colorscale=[[i / 6, c] for i, c in enumerate(SEQ)],
        zmin=0, zmax=1, text=text, texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2,
        colorbar=dict(title="share", tickformat=".0%", outlinewidth=0)))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=11))
    return base_layout(fig, height=320, legend=False, margin=dict(l=110, r=30, t=20, b=110))


STATIC_OVERRIDES = {"V4_segments": fig_segments_static, "V2_dq_scorecard": fig_dq_scorecard_static}


# --------------------------------------------------------------------------- V5
def fig_diag_hba1c(cfg):
    seg = _read(cfg, "readmit_by_segment.csv")
    first_path = cfg.interim / "cohort_first.parquet"
    if seg is None or not first_path.exists():
        return None
    d = pd.read_parquet(first_path, columns=["diag1_grp", "hba1c_tested", "readmit30"])
    d = d[d["diag1_grp"] != "Missing"]          # a handful of patients; CI would dominate the axis
    g = d.groupby(["diag1_grp", "hba1c_tested"])["readmit30"].agg(["mean", "count", "sum"]).reset_index()
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
    g["lo"], g["hi"] = lo, hi
    order = g[g["hba1c_tested"] == 0].sort_values("mean", ascending=False)["diag1_grp"].tolist()
    fig = go.Figure()
    for val, name, color in ((0, "HbA1c not tested", S2), (1, "HbA1c tested", S1)):
        s = g[g["hba1c_tested"] == val].set_index("diag1_grp").reindex(order).reset_index()
        fig.add_trace(go.Bar(
            x=s["diag1_grp"], y=s["mean"], name=name, marker=dict(color=color, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=s["hi"] - s["mean"], arrayminus=s["mean"] - s["lo"],
                         color=INK2, thickness=1.2, width=3),
            customdata=s[["count", "sum"]].values,
            hovertemplate="%{x} · " + name + "<br>rate %{y:.1%}<br>%{customdata[1]:,} / %{customdata[0]:,}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title="30-day readmission (first encounter, 95% CI)", tickformat=".0%")
    return base_layout(fig, height=420)


# --------------------------------------------------------------------------- V6
def fig_forest(cfg):
    ms = _read(cfg, "model_exposure_summary.csv")
    mi = _read(cfg, "missing_strategy_compare.csv")
    rows = []
    if ms is not None:
        rows += [(r.model, r.or_, r.or_low, r.or_high) for r in ms.rename(columns={"or": "or_"}).itertuples()]
    if mi is not None:
        rows += [(f"Missing data: {r.strategy}", r.or_, r.or_low, r.or_high)
                 for r in mi.rename(columns={"or": "or_"}).itertuples()]
    if not rows:
        return None
    f = pd.DataFrame(rows, columns=["label", "or", "lo", "hi"])
    fig = go.Figure(go.Scatter(
        x=f["or"], y=f["label"], mode="markers", marker=dict(color=S1, size=10, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, array=f["hi"] - f["or"], arrayminus=f["or"] - f["lo"],
                     color=S1, thickness=2, width=0),
        hovertemplate="%{y}<br>OR %{x:.3f}<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=MUTED, width=1.5))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Adjusted odds ratio for HbA1c tested (95% CI) — log scale", type="log")
    return base_layout(fig, height=60 + 42 * len(f), legend=False, margin=dict(l=330, r=30, t=20, b=60))


def fig_forest_diag(cfg):
    t = _read(cfg, "model_hba1c_by_diag.csv")
    if t is None:
        return None
    t = t.sort_values("or")
    fig = go.Figure(go.Scatter(
        x=t["or"], y=[f"{g} (n={n:,})" for g, n in zip(t["diag_group"], t["n"])], mode="markers",
        marker=dict(color=S1, size=10, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, array=t["or_high"] - t["or"], arrayminus=t["or"] - t["or_low"],
                     color=S1, thickness=2, width=0),
        customdata=t["p"], hovertemplate="%{y}<br>OR %{x:.3f}<br>p = %{customdata:.3g}<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=MUTED, width=1.5))
    fig.update_xaxes(title="OR for HbA1c tested within primary-diagnosis group (95% CI) — log scale", type="log")
    return base_layout(fig, height=60 + 40 * len(t), legend=False, margin=dict(l=220, r=30, t=20, b=60))


# --------------------------------------------------------------------------- V7
def fig_caseload(cfg):
    c = _read(cfg, "risk_caseload.csv")
    m = _read(cfg, "risk_metrics.csv")
    if c is None:
        return None
    fig = go.Figure()
    auc = dict(zip(m["model"], m["auc"])) if m is not None else {}
    names = {"logit": "Logistic regression", "hgb": "Gradient boosting"}
    for (model, color) in (("logit", S1), ("hgb", S2)):
        s = c[c["model"] == model]
        if s.empty:
            continue
        label = f"{names[model]} (AUC {auc.get(model, float('nan')):.3f})"
        fig.add_trace(go.Scatter(x=s["top_k_share"], y=s["capture_rate"], mode="lines+markers", name=label,
                                 line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                 customdata=s[["patients_followed", "ppv"]].values,
                                 hovertemplate=label + "<br>follow top %{x:.0%} (%{customdata[0]:,} discharges)"
                                 "<br>catch %{y:.1%} of readmissions<br>PPV %{customdata[1]:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", name="Random selection",
                             line=dict(color=MUTED, dash="dot", width=1.5), hoverinfo="skip"))
    fig.update_xaxes(title="Share of discharges followed up (highest predicted risk first)", tickformat=".0%")
    fig.update_yaxes(title="Share of 30-day readmissions caught", tickformat=".0%")
    return base_layout(fig, height=420, hovermode="x unified")


def fig_calibration(cfg):
    c = _read(cfg, "risk_calibration.csv")
    if c is None:
        return None
    fig = go.Figure()
    names = {"logit": "Logistic regression", "hgb": "Gradient boosting"}
    for model, color in (("logit", S1), ("hgb", S2)):
        s = c[c["model"] == model]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s["predicted"], y=s["observed"], mode="lines+markers", name=names[model],
                                 line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                 customdata=s[["decile", "n"]].values,
                                 hovertemplate=names[model] + "<br>decile %{customdata[0]} (n=%{customdata[1]:,})"
                                 "<br>predicted %{x:.1%} · observed %{y:.1%}<extra></extra>"))
    mx = float(c[["predicted", "observed"]].max().max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect calibration",
                             line=dict(color=MUTED, dash="dot", width=1.5), hoverinfo="skip"))
    fig.update_xaxes(title="Mean predicted risk (decile)", tickformat=".0%")
    fig.update_yaxes(title="Observed readmission rate", tickformat=".0%")
    return base_layout(fig, height=400)


# --------------------------------------------------------------------------- V8
def fig_power(cfg):
    pc = _read(cfg, "power_curve.csv")
    if pc is None:
        return None
    fig = go.Figure()
    for frac, color in ((1.0, S1), (0.5, S2), (0.25, S3)):
        s = pc[pc["sample_fraction"] == frac]
        name = f"{int(frac * 100)}% of available patients"
        fig.add_trace(go.Scatter(x=s["abs_difference"], y=s["power"], mode="lines", name=name,
                                 line=dict(color=color, width=2),
                                 hovertemplate=name + "<br>difference %{x:.2%} → power %{y:.0%}<extra></extra>"))
    fig.add_hline(y=0.8, line=dict(color=MUTED, dash="dot", width=1.5), annotation_text="80% power",
                  annotation_font_color=INK2)
    fig.update_xaxes(title="Absolute difference in readmission rate to detect (tested vs not tested)", tickformat=".1%")
    fig.update_yaxes(title="Power (α = 0.05, two-sided)", tickformat=".0%", range=[0, 1.02])
    return base_layout(fig, height=400, hovermode="x unified")


# --------------------------------------------------------------------------- registry
def build_figures(cfg: Config) -> list[Figure]:
    return [
        Figure("V1_cohort_flow", "V1 · How the analytic cohort was built",
               "Data steward; report reader", "How many records survive each cleaning rule, and why are they removed?",
               "Accept the analytic cohort and document exclusions (RECORD 6.3).", fig_cohort_flow(cfg),
               "cohort_flow.csv"),
        Figure("V2_dq_scorecard", "V2 · Data-quality scorecard (completeness × validity per field)",
               "Data steward / data owner", "Which fields are fit for use, and which need fixing or exclusion?",
               "Exclude 'not fit' fields (e.g. weight), plan missing-data handling, request upstream fixes.",
               fig_dq_scorecard(cfg), "dq_scorecard.csv · dq_checks.csv"),
        Figure("V3_missing_patterns", "V3 · Joint missingness patterns (top 10)",
               "Analyst", "Is missingness concentrated in a few fields and patterns (plausibly MAR) or scattered?",
               "Choose the missing-data strategy (indicator vs MICE) and which variables to impute.",
               fig_missing_patterns(cfg), "missing_patterns.csv · missing_vs_outcome.csv"),
        Figure("V4_segments", "V4 · 30-day readmission by patient segment",
               "Hospital quality manager", "Which patient segments have the highest readmission rate?",
               "Target discharge-planning and follow-up programmes at the highest-rate segments.",
               fig_segments(cfg), "readmit_by_segment.csv"),
        Figure("V5_diag_hba1c", "V5 · Readmission by primary diagnosis and HbA1c testing (crude)",
               "Diabetes clinical lead", "Is readmission lower when HbA1c was measured, and does it differ by diagnosis?",
               "Decide where an HbA1c-testing protocol is worth piloting (crude view; see V6 for adjusted).",
               fig_diag_hba1c(cfg), "readmit_by_segment.csv"),
        Figure("V6_forest_methods", "V6 · Robustness of the HbA1c association across methods",
               "Clinical lead; analyst", "Does the adjusted association survive different correlation and missing-data methods?",
               "Judge how much trust to place in the finding before acting on it.", fig_forest(cfg),
               "model_exposure_summary.csv · missing_strategy_compare.csv"),
        Figure("V6b_forest_diag", "V6b · Adjusted HbA1c association within each primary-diagnosis group",
               "Clinical lead", "For which diagnosis groups is testing associated with lower readmission?",
               "Prioritise diagnosis groups for a protocol study.", fig_forest_diag(cfg), "model_hba1c_by_diag.csv"),
        Figure("V7_caseload", "V7 · Follow-up caseload vs readmissions caught",
               "Hospital quality manager", "If we follow up the top-k% riskiest discharges, what share of readmissions do we reach?",
               "Set the follow-up threshold and staffing level for a transitional-care programme.",
               fig_caseload(cfg), "risk_caseload.csv · risk_metrics.csv"),
        Figure("V7b_calibration", "V7b · Calibration of the risk models",
               "Analyst; quality manager", "Do predicted risks match observed readmission rates?",
               "Decide whether predicted probabilities can be shown to clinicians as risks or only as ranks.",
               fig_calibration(cfg), "risk_calibration.csv"),
        Figure("V8_power", "V8 · Statistical power for the HbA1c comparison",
               "Analyst; lecturer", "Is the sample large enough to detect a 1-percentage-point difference?",
               "Confirm sample adequacy; avoid over-interpreting non-significant subgroup results.",
               fig_power(cfg), "power_analysis.csv · power_curve.csv"),
    ]


KPI_FIELDS = [("encounters", "Eligible encounters", "{:,.0f}"), ("patients", "Patients", "{:,.0f}"),
              ("readmit30_rate", "30-day readmission", "{:.1%}"), ("hba1c_tested_rate", "HbA1c tested", "{:.1%}"),
              ("repeat_patient_share", "Patients with ≥2 stays", "{:.1%}")]


def _kpi_html(cfg: Config) -> str:
    kpi = _read(cfg, "kpi.csv")
    if kpi is None:
        return ""
    r = kpi[kpi["cohort"] == "all_encounters"].iloc[0]
    tiles = "".join(f'<div class="tile"><div class="tv">{fmt.format(r[k])}</div><div class="tl">{html.escape(lbl)}</div></div>'
                    for k, lbl, fmt in KPI_FIELDS)
    return f'<section class="tiles">{tiles}</section>'


CSS = f"""
.viz-root {{ color-scheme: light; --surface-1:{SURFACE}; --text-primary:{INK}; --text-secondary:{INK2};
  --border:{GRID}; font-family:{FONT}; color:var(--text-primary); background:#f5f4f0; margin:0; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:28px 20px 60px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} .sub {{ color:var(--text-secondary); margin:0 0 20px; font-size:14px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:24px; }}
.tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.tv {{ font-size:26px; font-weight:650; }} .tl {{ color:var(--text-secondary); font-size:13px; margin-top:2px; }}
.card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:18px 18px 8px; margin-bottom:22px; }}
.card h2 {{ font-size:17px; margin:0 0 10px; }}
.uc {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:8px; font-size:13px; }}
.uc div {{ background:#f5f4f0; border-radius:8px; padding:8px 10px; }} .uc b {{ display:block; color:var(--text-secondary);
  font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:2px; }}
.tbl {{ color:var(--text-secondary); font-size:12px; margin:4px 0 8px; }} code {{ font-size:12px; }}
.missing {{ color:var(--text-secondary); font-style:italic; padding:16px 0; }}
@media (max-width:760px) {{ .uc {{ grid-template-columns:1fr; }} }}
"""


def run(cfg: Config) -> Path:
    figs = build_figures(cfg)
    cfg.figures.mkdir(parents=True, exist_ok=True)
    cards = []
    config = {"displaylogo": False, "responsive": True}
    for f in figs:
        uc = (f'<div class="uc"><div><b>User</b>{html.escape(f.user)}</div><div><b>Question</b>{html.escape(f.question)}</div>'
              f'<div><b>Decision supported</b>{html.escape(f.decision)}</div></div>')
        if f.fig is None:
            body = '<div class="missing">Source table not found — run the corresponding pipeline stage.</div>'
        else:
            f.fig.write_html(cfg.figures / f"{f.key}.html", include_plotlyjs="cdn", config=config)
            body = f.fig.to_html(full_html=False, include_plotlyjs=False, config=config, div_id=f.key)
        cards.append(f'<section class="card"><h2>{html.escape(f.title)}</h2>{uc}{body}'
                     f'<div class="tbl">Table view: <code>tables/{html.escape(f.table)}</code></div></section>')
    mode = "SMOKE TEST (sampled patients — not for interpretation)" if cfg.smoke else "Full dataset"
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Diabetic readmission — decision dashboard</title>
<style>{CSS}</style><script type="text/javascript">{get_plotlyjs()}</script></head>
<body class="viz-root"><div class="wrap"><h1>30-day readmission of diabetic inpatients</h1>
<p class="sub">UCI Diabetes 130-US Hospitals, 1999–2008 · {html.escape(mode)} · environment: {html.escape(cfg.environment)}</p>
{_kpi_html(cfg)}{''.join(cards)}</div></body></html>"""
    out = cfg.out_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    LOG.info("Dashboard written: %s (%d figures, %d skipped)", out, sum(f.fig is not None for f in figs),
             sum(f.fig is None for f in figs))
    return out
