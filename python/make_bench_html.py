#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np


DEFAULT_CONSTRUCTIONS_INPUT = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "compacted_constructions.csv"
)
DEFAULT_SEARCHES_INPUT = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "compacted_searches.csv"
)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def build_constructions_table(df: pd.DataFrame) -> str:
    # Add readable time if present
    dfv = df.copy()
    if "timestamp" in dfv.columns and "when" not in dfv.columns:
        dfv["when"] = dfv["timestamp"].apply(
            lambda s: datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
        )
    cols = [
        c
        for c in [
            "when",
            "dataset",
            "host",
            "name",
            "clique",
            "n",
            "primary",
            "secondary",
            "build_secs",
            "reps_nn_in_clique",
            "reps_nn_pct",
            "footprint_mb",
        ]
        if c in dfv.columns
    ]
    dfv = dfv[cols]
    style = (
        dfv.style.background_gradient(
            subset=[c for c in ["build_secs"] if c in dfv.columns], cmap="Reds_r"
        )
        .background_gradient(
            subset=[
                c
                for c in [
                    "n",
                    "primary",
                    "secondary",
                    "reps_nn_in_clique",
                    "reps_nn_pct",
                    "footprint_mb",
                ]
                if c in dfv.columns
            ],
            cmap="Blues",
        )
        .set_table_attributes('class="bench-table"')
    )
    return style.to_html()


def build_interactive_dashboard(df_search: pd.DataFrame) -> str:
    """Build an overview and per-dataset plots with Pareto highlighting, from searches CSV.

    Adds a dropdown to switch between grouping by query method (phases) and by construction (index name).
    """
    if df_search.empty:
        return "<p>No search rows to plot.</p>"

    # Prepare
    df = df_search.copy()
    if "timestamp" in df.columns and "when" not in df.columns:
        df["when"] = df["timestamp"].apply(
            lambda s: datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
        )

    # Map methods to labels and treat them as phases for plotting
    method_labels = {
        "baseline": "Baseline",
        "postexp": "Post-expansion",
        "expandvis": "ExpandVis",
        "primary": "Primary",
        "exhaustive_primary": "Exhaustive Primary",
        "exhaustive_secondary": "Exhaustive Secondary",
        "primary_specific": "Primary Specific",
    }
    df["phase_key"] = df["method"].astype(str) if "method" in df.columns else ""
    df["phase"] = df["phase_key"].map(method_labels)
    df["phase"] = df["phase"].fillna(df["phase_key"]).astype(str)

    # Construction grouping: use index/construction name if present
    if "name" in df.columns:
        df["construction_key"] = df["name"].astype(str)
        df["construction"] = df["construction_key"].astype(str)
    else:
        # Fallback to a synthesized construction key to avoid errors
        df["construction_key"] = df.index.astype(str)
        df["construction"] = df["construction_key"]

    # Clique grouping: use clique method/label if present
    has_clique = False
    if "clique" in df.columns:
        df["clique_key"] = df["clique"].astype(str)
        df["clique_label"] = df["clique_key"].astype(str)
        has_clique = True
    else:
        df["clique_key"] = ""
        df["clique_label"] = ""

    # Available groups
    available_phases: List[Tuple[str, str]] = [
        (k, method_labels.get(k, k)) for k in sorted(df["phase_key"].unique())
    ]
    available_constructions: List[Tuple[str, str]] = [
        (k, k) for k in sorted(df["construction_key"].unique())
    ]
    available_cliques: List[Tuple[str, str]] = [
        (k, k) for k in (sorted(df["clique_key"].unique()) if has_clique else [])
    ]

    # Build consistent color maps per grouping
    def build_color_map_from_groups(groups: List[Tuple[str, str]]) -> Dict[str, str]:
        palette = px.colors.qualitative.Plotly
        color_map: Dict[str, str] = {}
        for idx, (_gkey, glabel) in enumerate(groups):
            color_map[glabel] = palette[idx % len(palette)]
        return color_map

    def compute_pareto_mask(
        values: np.ndarray, maximize: Tuple[bool, bool]
    ) -> np.ndarray:
        """Return boolean mask of non-dominated points for 2D values.
        values: array shape (n, 2)
        maximize: (max_x, max_y) True if we want larger-is-better for that axis
        """
        if values.size == 0:
            return np.array([], dtype=bool)
        # Normalize directions: transform so that we maximize both by flipping signs where needed
        x = values[:, 0]
        y = values[:, 1]
        if not maximize[0]:
            x = -x
        if not maximize[1]:
            y = -y
        order = np.argsort(-x, kind="mergesort")  # sort by x descending
        x_sorted = x[order]
        y_sorted = y[order]
        is_nd_sorted = np.zeros_like(x_sorted, dtype=bool)
        best_y = -np.inf
        for i in range(len(x_sorted)):
            if y_sorted[i] > best_y + 1e-12:
                is_nd_sorted[i] = True
                best_y = y_sorted[i]
        is_nd = np.zeros_like(is_nd_sorted)
        is_nd[order] = is_nd_sorted
        return is_nd

    def make_overview(
        df_long: pd.DataFrame,
        color_col: str,
        title_suffix: str,
        color_map: Dict[str, str],
    ) -> go.Figure:
        fig = px.scatter(
            df_long,
            x="qps",
            y="recall",
            color=color_col if color_col in df_long.columns else None,
            symbol=None,
            facet_col="dataset" if "dataset" in df_long.columns else None,
            facet_col_wrap=2
            if ("dataset" in df_long.columns and df_long["dataset"].nunique() > 1)
            else None,
            hover_data=[
                c
                for c in [
                    "dataset",
                    "name",
                    "host",
                    "when",
                    "phase",
                    "construction",
                    "comp_q",
                    "beam",
                ]
                if c in df_long.columns
            ],
            title=f"Recall vs QPS — Overview by Dataset ({title_suffix})",
            template="simple_white",
            color_discrete_map=color_map if color_map else None,
        )
        fig.update_traces(
            marker=dict(size=7, opacity=0.85, line=dict(width=0.5, color="#ffffff"))
        )
        fig.update_layout(legend_title_text="", margin=dict(l=40, r=20, t=60, b=40))
        fig.update_xaxes(title_text="QPS (higher is better)")
        fig.update_yaxes(title_text="Recall (higher is better)")
        return fig

    def make_dataset_figs(
        df_long: pd.DataFrame,
        dataset_value: str,
        group_key: str,
        group_label_col: str,
        available_groups: List[Tuple[str, str]],
        color_map: Dict[str, str],
    ) -> List[go.Figure]:
        figs: List[go.Figure] = []
        ddf = (
            df_long[df_long["dataset"] == dataset_value]
            if "dataset" in df_long.columns
            else df_long
        )

        # Figure A: Recall vs QPS with per-phase traces and Pareto per phase
        fig_a = go.Figure()
        for key, label in available_groups:
            pdf = ddf[ddf[group_key] == key]
            if pdf.empty:
                continue
            fig_a.add_trace(
                go.Scatter(
                    x=pdf["qps"],
                    y=pdf["recall"],
                    mode="markers",
                    name=f"{label}",
                    marker=dict(size=8, opacity=0.9, color=color_map.get(label)),
                    hovertemplate=(
                        "group=%{fullData.name}<br>qps=%{x:.3f}<br>recall=%{y:.5f}"
                        + (
                            "<br>comp/q=%{customdata[0]:.3f}"
                            if "comp_q" in pdf.columns
                            else ""
                        )
                        + "<br>%{text}<extra></extra>"
                    ),
                    text=[
                        " | ".join(
                            [
                                *([f"name={n}"] if "name" in pdf.columns else []),
                                *([f"method={p}"] if "phase" in pdf.columns else []),
                                *([f"beam={b}"] if "beam" in pdf.columns else []),
                                *([f"clique={c}"] if "clique" in pdf.columns else []),
                                *([f"host={h}"] if "host" in pdf.columns else []),
                                *([f"when={w}"] if "when" in pdf.columns else []),
                            ]
                        )
                        for n, p, b, c, h, w in zip(
                            pdf.get("name", pd.Series([None] * len(pdf))),
                            pdf.get("phase", pd.Series([None] * len(pdf))),
                            pdf.get("beam", pd.Series([None] * len(pdf))),
                            pdf.get("clique", pd.Series([None] * len(pdf))),
                            pdf.get("host", pd.Series([None] * len(pdf))),
                            pdf.get("when", pd.Series([None] * len(pdf))),
                        )
                    ],
                    customdata=pdf[["comp_q"]].values
                    if "comp_q" in pdf.columns
                    else None,
                )
            )

            # Pareto front for recall (max) vs qps (max)
            vals = pdf[["qps", "recall"]].to_numpy(dtype=float)
            mask = compute_pareto_mask(vals, maximize=(True, True))
            if mask.any():
                pareto_pts = pdf.loc[mask, ["qps", "recall"]].sort_values(
                    ["qps", "recall"], ascending=[True, True]
                )
                fig_a.add_trace(
                    go.Scatter(
                        x=pareto_pts["qps"],
                        y=pareto_pts["recall"],
                        mode="lines+markers",
                        name=f"{label} Pareto",
                        line=dict(width=2, color=color_map.get(label)),
                        marker=dict(
                            size=9,
                            line=dict(width=1, color="#000"),
                            color=color_map.get(label),
                        ),
                        hoverinfo="skip",
                    )
                )

        title_suffix_a = (
            "Method"
            if group_key == "phase_key"
            else (
                "Construction" if group_key == "construction_key" else "Clique Method"
            )
        )
        fig_a.update_layout(
            title=(f"{dataset_value}: Recall vs QPS by {title_suffix_a}")
            if "dataset" in df_long.columns
            else (f"Recall vs QPS by {title_suffix_a}"),
            template="simple_white",
            legend_title_text="",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig_a.update_xaxes(title_text="QPS (higher is better)")
        fig_a.update_yaxes(title_text="Recall (higher is better)")
        figs.append(fig_a)

        # Figure B: Recall vs Comp/Q (if comp_q available)
        if "comp_q" in ddf.columns and not ddf["comp_q"].isna().all():
            fig_b = go.Figure()
            for key, label in available_groups:
                pdf = ddf[(ddf[group_key] == key) & (~ddf["comp_q"].isna())]
                if pdf.empty:
                    continue
                fig_b.add_trace(
                    go.Scatter(
                        x=pdf["comp_q"],
                        y=pdf["recall"],
                        mode="markers",
                        name=f"{label}",
                        marker=dict(size=8, opacity=0.9, color=color_map.get(label)),
                        hovertemplate=(
                            "group=%{fullData.name}<br>comp/q=%{x:.3f}<br>recall=%{y:.5f}<br>%{text}<extra></extra>"
                        ),
                        text=[
                            " | ".join(
                                [
                                    *([f"name={n}"] if "name" in pdf.columns else []),
                                    *(
                                        [f"method={p}"]
                                        if "phase" in pdf.columns
                                        else []
                                    ),
                                    *([f"beam={b}"] if "beam" in pdf.columns else []),
                                    *(
                                        [f"clique={c}"]
                                        if "clique" in pdf.columns
                                        else []
                                    ),
                                    *([f"host={h}"] if "host" in pdf.columns else []),
                                    *([f"when={w}"] if "when" in pdf.columns else []),
                                ]
                            )
                            for n, p, b, c, h, w in zip(
                                pdf.get("name", pd.Series([None] * len(pdf))),
                                pdf.get("phase", pd.Series([None] * len(pdf))),
                                pdf.get("beam", pd.Series([None] * len(pdf))),
                                pdf.get("clique", pd.Series([None] * len(pdf))),
                                pdf.get("host", pd.Series([None] * len(pdf))),
                                pdf.get("when", pd.Series([None] * len(pdf))),
                            )
                        ],
                    )
                )

                # Pareto front for recall (max) vs comp_q (min)
                vals = pdf[["comp_q", "recall"]].to_numpy(dtype=float)
                mask = compute_pareto_mask(vals, maximize=(False, True))
                if mask.any():
                    pareto_pts = pdf.loc[mask, ["comp_q", "recall"]].sort_values(
                        ["comp_q", "recall"], ascending=[True, True]
                    )
                    fig_b.add_trace(
                        go.Scatter(
                            x=pareto_pts["comp_q"],
                            y=pareto_pts["recall"],
                            mode="lines+markers",
                            name=f"{label} Pareto",
                            line=dict(width=2, color=color_map.get(label)),
                            marker=dict(
                                size=9,
                                line=dict(width=1, color="#000"),
                                color=color_map.get(label),
                            ),
                            hoverinfo="skip",
                        )
                    )

            title_suffix_b = (
                "Method"
                if group_key == "phase_key"
                else (
                    "Construction"
                    if group_key == "construction_key"
                    else "Clique Method"
                )
            )
            fig_b.update_layout(
                title=(f"{dataset_value}: Recall vs Comp/Q by {title_suffix_b}")
                if "dataset" in df_long.columns
                else (f"Recall vs Comp/Q by {title_suffix_b}"),
                template="simple_white",
                legend_title_text="",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            fig_b.update_xaxes(title_text="Comp/Q (lower is better)", type="log")
            fig_b.update_yaxes(title_text="Recall (higher is better)")
            figs.append(fig_b)

        # Figure C: QPS vs Comp/Q (if comp_q available)
        if "comp_q" in ddf.columns and not ddf["comp_q"].isna().all():
            fig_c = go.Figure()
            for key, label in available_groups:
                pdf = ddf[(ddf[group_key] == key) & (~ddf["comp_q"].isna())]
                if pdf.empty:
                    continue
                fig_c.add_trace(
                    go.Scatter(
                        x=pdf["comp_q"],
                        y=pdf["qps"],
                        mode="markers",
                        name=f"{label}",
                        marker=dict(size=8, opacity=0.9, color=color_map.get(label)),
                        hovertemplate=(
                            "group=%{fullData.name}<br>comp/q=%{x:.3f}<br>qps=%{y:.3f}<br>%{text}<extra></extra>"
                        ),
                        text=[
                            " | ".join(
                                [
                                    *([f"name={n}"] if "name" in pdf.columns else []),
                                    *(
                                        [f"method={p}"]
                                        if "phase" in pdf.columns
                                        else []
                                    ),
                                    *([f"beam={b}"] if "beam" in pdf.columns else []),
                                    *(
                                        [f"clique={c}"]
                                        if "clique" in pdf.columns
                                        else []
                                    ),
                                    *([f"host={h}"] if "host" in pdf.columns else []),
                                    *([f"when={w}"] if "when" in pdf.columns else []),
                                ]
                            )
                            for n, p, b, c, h, w in zip(
                                pdf.get("name", pd.Series([None] * len(pdf))),
                                pdf.get("phase", pd.Series([None] * len(pdf))),
                                pdf.get("beam", pd.Series([None] * len(pdf))),
                                pdf.get("clique", pd.Series([None] * len(pdf))),
                                pdf.get("host", pd.Series([None] * len(pdf))),
                                pdf.get("when", pd.Series([None] * len(pdf))),
                            )
                        ],
                    )
                )

                # Pareto front for qps (max) vs comp_q (min)
                vals = pdf[["comp_q", "qps"]].to_numpy(dtype=float)
                mask = compute_pareto_mask(vals, maximize=(False, True))
                if mask.any():
                    pareto_pts = pdf.loc[mask, ["comp_q", "qps"]].sort_values(
                        ["comp_q", "qps"], ascending=[True, True]
                    )
                    fig_c.add_trace(
                        go.Scatter(
                            x=pareto_pts["comp_q"],
                            y=pareto_pts["qps"],
                            mode="lines+markers",
                            name=f"{label} Pareto",
                            line=dict(width=2, color=color_map.get(label)),
                            marker=dict(
                                size=9,
                                line=dict(width=1, color="#000"),
                                color=color_map.get(label),
                            ),
                            hoverinfo="skip",
                        )
                    )

            title_suffix_c = (
                "Method"
                if group_key == "phase_key"
                else (
                    "Construction"
                    if group_key == "construction_key"
                    else "Clique Method"
                )
            )
            fig_c.update_layout(
                title=(f"{dataset_value}: QPS vs Comp/Q by {title_suffix_c}")
                if "dataset" in df_long.columns
                else (f"QPS vs Comp/Q by {title_suffix_c}"),
                template="simple_white",
                legend_title_text="",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            fig_c.update_xaxes(title_text="Comp/Q (lower is better)", type="log")
            fig_c.update_yaxes(title_text="QPS (higher is better)")
            figs.append(fig_c)

        return figs

    # Data already long-form
    df_long = df

    html_parts: List[str] = []
    include_js = True
    plotly_config = {"responsive": True}

    # Dropdown to toggle grouping
    options_html = [
        '<option value="method" selected>Query Method</option>',
        '<option value="construction">Construction</option>',
    ]
    if has_clique:
        options_html.append('<option value="clique">Clique Method</option>')
    html_parts.append(
        """
<div style="margin: 10px 0;">
  <label for="grouping-mode" style="margin-right:8px; font-weight:600;">Color & Pareto by:</label>
  <select id="grouping-mode">{opts}</select>
</div>
""".format(opts="\n    ".join(options_html))
    )

    # Method (phase) section
    method_section: List[str] = []
    method_section.append("<h2>Overview</h2>")
    color_map_method = build_color_map_from_groups(available_phases)
    overview_fig_method = make_overview(
        df_long,
        color_col="phase",
        title_suffix="colored by Method",
        color_map=color_map_method,
    )
    method_section.append(
        pio.to_html(
            overview_fig_method,
            full_html=False,
            include_plotlyjs="cdn" if include_js else False,
            config=plotly_config,
        )
    )
    include_js = False

    if "dataset" in df_long.columns and df_long["dataset"].nunique() > 0:
        for dataset_value in sorted(df_long["dataset"].dropna().unique()):
            method_section.append(f"<h3>Dataset: {dataset_value}</h3>")
            figs = make_dataset_figs(
                df_long,
                dataset_value,
                group_key="phase_key",
                group_label_col="phase",
                available_groups=available_phases,
                color_map=color_map_method,
            )
            for fig in figs:
                method_section.append(
                    pio.to_html(
                        fig,
                        full_html=False,
                        include_plotlyjs=False,
                        config=plotly_config,
                    )
                )
    else:
        method_section.append("<h3>Details</h3>")
        for fig in make_dataset_figs(
            df_long,
            dataset_value="",
            group_key="phase_key",
            group_label_col="phase",
            available_groups=available_phases,
            color_map=color_map_method,
        ):
            method_section.append(
                pio.to_html(
                    fig, full_html=False, include_plotlyjs=False, config=plotly_config
                )
            )

    html_parts.append(
        '<div id="mode-method" class="mode-section">'
        + "\n".join(method_section)
        + "</div>"
    )

    # Construction section
    construction_section: List[str] = []
    construction_section.append("<h2>Overview</h2>")
    color_map_construction = build_color_map_from_groups(available_constructions)
    overview_fig_constr = make_overview(
        df_long,
        color_col="construction",
        title_suffix="colored by Construction",
        color_map=color_map_construction,
    )
    construction_section.append(
        pio.to_html(
            overview_fig_constr,
            full_html=False,
            include_plotlyjs=False,
            config=plotly_config,
        )
    )

    if "dataset" in df_long.columns and df_long["dataset"].nunique() > 0:
        for dataset_value in sorted(df_long["dataset"].dropna().unique()):
            construction_section.append(f"<h3>Dataset: {dataset_value}</h3>")
            figs = make_dataset_figs(
                df_long,
                dataset_value,
                group_key="construction_key",
                group_label_col="construction",
                available_groups=available_constructions,
                color_map=color_map_construction,
            )
            for fig in figs:
                construction_section.append(
                    pio.to_html(
                        fig,
                        full_html=False,
                        include_plotlyjs=False,
                        config=plotly_config,
                    )
                )
    else:
        construction_section.append("<h3>Details</h3>")
        for fig in make_dataset_figs(
            df_long,
            dataset_value="",
            group_key="construction_key",
            group_label_col="construction",
            available_groups=available_constructions,
            color_map=color_map_construction,
        ):
            construction_section.append(
                pio.to_html(
                    fig, full_html=False, include_plotlyjs=False, config=plotly_config
                )
            )

    html_parts.append(
        '<div id="mode-construction" class="mode-section" style="display:none;">'
        + "\n".join(construction_section)
        + "</div>"
    )

    # Clique section (optional)
    if has_clique:
        clique_section: List[str] = []
        clique_section.append("<h2>Overview</h2>")
        color_map_clique = build_color_map_from_groups(available_cliques)
        overview_fig_clique = make_overview(
            df_long,
            color_col="clique_label",
            title_suffix="colored by Clique Method",
            color_map=color_map_clique,
        )
        clique_section.append(
            pio.to_html(
                overview_fig_clique,
                full_html=False,
                include_plotlyjs=False,
                config=plotly_config,
            )
        )

        if "dataset" in df_long.columns and df_long["dataset"].nunique() > 0:
            for dataset_value in sorted(df_long["dataset"].dropna().unique()):
                clique_section.append(f"<h3>Dataset: {dataset_value}</h3>")
                figs = make_dataset_figs(
                    df_long,
                    dataset_value,
                    group_key="clique_key",
                    group_label_col="clique_label",
                    available_groups=available_cliques,
                    color_map=color_map_clique,
                )
                for fig in figs:
                    clique_section.append(
                        pio.to_html(
                            fig,
                            full_html=False,
                            include_plotlyjs=False,
                            config=plotly_config,
                        )
                    )
        else:
            clique_section.append("<h3>Details</h3>")
            for fig in make_dataset_figs(
                df_long,
                dataset_value="",
                group_key="clique_key",
                group_label_col="clique_label",
                available_groups=available_cliques,
                color_map=color_map_clique,
            ):
                clique_section.append(
                    pio.to_html(
                        fig,
                        full_html=False,
                        include_plotlyjs=False,
                        config=plotly_config,
                    )
                )

        html_parts.append(
            '<div id="mode-clique" class="mode-section" style="display:none;">'
            + "\n".join(clique_section)
            + "</div>"
        )

    # Toggle script
    html_parts.append(
        """
<script>
(function(){
  function resizePlots(container){
    if (!window.Plotly) return;
    var plots = container.querySelectorAll('.plotly-graph-div');
    plots.forEach(function(div){ try { Plotly.Plots.resize(div); } catch(e){} });
  }
  function setMode(mode){
    var sections = ['mode-method','mode-construction','mode-clique'];
    var shown = null;
    sections.forEach(function(id){
      var el = document.getElementById(id);
      if (!el) return;
      if (id === 'mode-' + mode) { el.style.display = ''; shown = el; }
      else { el.style.display = 'none'; }
    });
    if (shown) resizePlots(shown);
  }
  document.addEventListener('DOMContentLoaded', function(){
    var sel = document.getElementById('grouping-mode');
    if (!sel) return;
    sel.addEventListener('change', function(){ setMode(sel.value); });
    setMode(sel.value || 'method');
  });
})();
</script>
"""
    )

    return "\n".join(html_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Render constructions table and interactive plots from split CSVs"
    )
    parser.add_argument(
        "--constructions-input",
        default=DEFAULT_CONSTRUCTIONS_INPUT,
        help="Path to compacted_constructions.csv",
    )
    parser.add_argument(
        "--searches-input",
        default=DEFAULT_SEARCHES_INPUT,
        help="Path to compacted_searches.csv",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for HTML files"
    )
    parser.add_argument(
        "--out-name",
        default="compacted_benchmarks_report.html",
        help="Output HTML filename",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_con = (
        pd.read_csv(args.constructions_input)
        if os.path.exists(args.constructions_input)
        else pd.DataFrame()
    )
    if not df_con.empty and "timestamp" in df_con.columns:
        df_con = df_con.copy()
        df_con["when"] = df_con["timestamp"].apply(
            lambda s: datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
        )
        df_con = df_con.sort_values(
            by=[
                c
                for c in ["timestamp", "dataset", "name", "clique"]
                if c in df_con.columns
            ]
        )

    df_srch = (
        pd.read_csv(args.searches_input)
        if os.path.exists(args.searches_input)
        else pd.DataFrame()
    )
    if not df_srch.empty and "timestamp" in df_srch.columns:
        df_srch = df_srch.sort_values(
            by=[
                c
                for c in ["timestamp", "dataset", "name", "clique", "beam", "method"]
                if c in df_srch.columns
            ]
        )

    table_html = (
        build_constructions_table(df_con)
        if not df_con.empty
        else "<p>No constructions data found.</p>"
    )
    plots_html = build_interactive_dashboard(df_srch)

    html = (
        r"""
<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<title>Compacted Benchmarks Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
.table-container { overflow-x: auto; }

/* Table base */
table.bench-table { border-collapse: collapse; width: 100%; }
table.bench-table th, table.bench-table td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }

/* Ensure Plotly plots fill horizontally */
.mode-section { width: 100%; }
.mode-section .plotly-graph-div, .mode-section .js-plotly-plot { width: 100% !important; max-width: 100% !important; }

/* Sticky, clickable headers with subtle bg */
table.bench-table thead th {
  position: sticky; top: 0;
  background: #fafafa;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
}

/* Arrow indicator (injected by JS) */
table.bench-table thead th .arrow {
  margin-left: 6px;
  opacity: .6;
}

/* Active sort state */
table.bench-table thead th.sort-asc .arrow::after { content: "↑"; }
table.bench-table thead th.sort-desc .arrow::after { content: "↓"; }

/* Row hover */
table.bench-table tbody tr:hover td { background: #fcfcfc; }

/* Right-align numeric-looking cells (optional) */
table.bench-table td.num { text-align: right; }
</style>
"""
        + f"""</head>
<body>
<h1>Compacted Benchmarks Report</h1>
<p><strong>Constructions:</strong> {os.path.abspath(args.constructions_input)}<br/>
<strong>Searches:</strong> {os.path.abspath(args.searches_input)}</p>
<div class=\"table-container\">
{table_html}
</div>
{plots_html}
"""
        + r"""<script>
(function () {
  function textOf(cell) {
    if (!cell) return "";
    return (cell.textContent || cell.innerText || "").trim();
  }

  function guessType(rows, colIdx) {
    let n = 0, numHits = 0, dateHits = 0;
    for (let i = 0; i < rows.length && n < 32; i++) {
      const t = textOf(rows[i].cells[colIdx]);
      if (!t) { n++; continue; }
      const num = Number(t.replace(/[%,$\u00A0\s,]/g, "")); // strip NBSP too
      if (!Number.isNaN(num) && /^[-+]?[\d.,\s$%eE]+$/.test(t)) numHits++;
      else if (!Number.isNaN(Date.parse(t))) dateHits++;
      n++;
    }
    if (numHits >= Math.max(3, n * 0.6)) return "number";
    if (dateHits >= Math.max(3, n * 0.6)) return "date";
    return "text";
  }

  function parseByType(type, value) {
    if (type === "number") {
      const v = Number(value.replace(/[%,$\u00A0\s,]/g, "").replace(/\u2212/g, "-")); // handle Unicode minus
      return Number.isNaN(v) ? NaN : v;
    }
    if (type === "date") {
      const t = Date.parse(value);
      return Number.isNaN(t) ? NaN : t;
    }
    return value.toLowerCase();
  }

  function ensureArrows(ths) {
    ths.forEach(th => {
      if (!th.querySelector(".arrow")) {
        const s = document.createElement("span");
        s.className = "arrow";
        th.appendChild(s);
      }
    });
  }

  function clearSortStates(ths) {
    ths.forEach(th => th.classList.remove("sort-asc", "sort-desc"));
  }

  function makeSortable(table) {
    if (!table) return;

    const thead = table.tHead;
    const bodies = Array.from(table.tBodies || []);
    if (!thead || bodies.length === 0) return;

    // Effective header row is the last row of THEAD (handles multi-index headers)
    const headerRow = thead.querySelector("tr:last-child");
    const ths = Array.from(headerRow ? headerRow.cells : []);
    ensureArrows(ths);

    // Stable sort key (original index across all TBODY rows)
    const allRows = bodies.flatMap(tb => Array.from(tb.rows));
    allRows.forEach((tr, i) => tr.dataset.idx = i);

    ths.forEach((th, colIdx) => {
      th.addEventListener("click", () => {
        const rows = bodies.flatMap(tb => Array.from(tb.rows));
        const type = th.dataset.type || guessType(rows, colIdx);
        const asc = !th.classList.contains("sort-asc");

        clearSortStates(ths);
        th.classList.add(asc ? "sort-asc" : "sort-desc");

        const coll = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
        rows.sort((a, b) => {
          const avRaw = textOf(a.cells[colIdx]);
          const bvRaw = textOf(b.cells[colIdx]);
          const av = parseByType(type, avRaw);
          const bv = parseByType(type, bvRaw);

          if (type === "text") {
            const cmp = coll.compare(av, bv);
            if (cmp !== 0) return asc ? cmp : -cmp;
          } else {
            const aNaN = Number.isNaN(av), bNaN = Number.isNaN(bv);
            if (aNaN && bNaN) { /* keep stable */ }
            else if (aNaN) return 1;
            else if (bNaN) return -1;
            else if (av !== bv) return asc ? (av - bv) : (bv - av);
          }
          return (+a.dataset.idx) - (+b.dataset.idx);
        });

        // Reattach into the first TBODY (works for single-TBODY Styler tables)
        const frag = document.createDocumentFragment();
        rows.forEach(r => frag.appendChild(r));
        bodies[0].appendChild(frag);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Select by class to avoid Styler's auto id like id="T_xxx"
    const table = document.querySelector("table.bench-table");
    makeSortable(table);
  });
})();
</script>
"""
    )

    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
