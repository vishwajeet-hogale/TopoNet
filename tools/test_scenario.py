import argparse
import os
import os.path as osp
import warnings
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.utils import get_logger

from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test


# ---------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------

METRIC_LABELS = {
    'OpenLane-V2 Score': 'OLV2 Score',
    'DET_l': 'DET$_l$',
    'DET_t': 'DET$_t$',
    'TOP_ll': 'TOP$_{ll}$',
    'TOP_lt': 'TOP$_{lt}$',
}
METRIC_COLS = list(METRIC_LABELS.keys())
SCENARIO_TITLES = {
    'curvature': 'Road Curvature',
    'lighting': 'Lighting Condition',
    'occlusion': 'Occlusion Level',
    'topology_complexity': 'Topology Complexity',
}
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']


def _bar_chart_per_scenario(df, scenario_type, global_score, run_dir):
    """Grouped bar chart: all 5 metrics per category, with global baseline line."""
    categories = df.index.tolist()
    n_cats = len(categories)
    n_metrics = len(METRIC_COLS)
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(7, n_cats * 2.2), 5))
    for j, metric in enumerate(METRIC_COLS):
        vals = df[metric].values.astype(float)
        bars = ax.bar(x + j * width, vals, width, label=METRIC_LABELS[metric],
                      color=PALETTE[j], edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=global_score, color='grey', linestyle='--', linewidth=1,
               label=f'Global OLV2 ({global_score:.3f})')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Scenario: {SCENARIO_TITLES.get(scenario_type, scenario_type)}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.0, df[METRIC_COLS].max().max() + 0.12))
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f'bar_{scenario_type}.png'), dpi=200)
    fig.savefig(osp.join(run_dir, f'bar_{scenario_type}.pdf'))
    plt.close(fig)


def _score_drop_chart(scenario_tables, global_score, run_dir):
    """Horizontal bar chart showing OLV2 Score delta from global for every category."""
    rows = []
    for stype, df in scenario_tables.items():
        for cat in df.index:
            delta = float(df.loc[cat, 'OpenLane-V2 Score']) - global_score
            rows.append({'scenario': SCENARIO_TITLES.get(stype, stype),
                         'category': cat, 'delta': delta,
                         'samples': int(df.loc[cat, 'samples'])})
    if not rows:
        return
    delta_df = pd.DataFrame(rows).sort_values('delta')

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.55)))
    colors = ['#C44E52' if d < 0 else '#55A868' for d in delta_df['delta']]
    labels = [f"{r['category']}  (n={r['samples']})" for _, r in delta_df.iterrows()]
    bars = ax.barh(range(len(delta_df)), delta_df['delta'], color=colors,
                   edgecolor='white', height=0.6)
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='grey', linewidth=0.8)
    ax.set_xlabel('$\\Delta$ OLV2 Score (vs. global)', fontsize=11)
    ax.set_title('Performance Gap by Scenario Category', fontsize=13, fontweight='bold')
    for bar, v in zip(bars, delta_df['delta']):
        ax.text(v + (0.003 if v >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f'{v:+.3f}', ha='left' if v >= 0 else 'right',
                va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'score_delta.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'score_delta.pdf'))
    plt.close(fig)


def _heatmap(scenario_tables, run_dir):
    """Heatmap: rows = all categories across scenario types, cols = metrics."""
    rows = []
    labels = []
    for stype, df in scenario_tables.items():
        prefix = SCENARIO_TITLES.get(stype, stype)
        for cat in df.index:
            labels.append(f'{prefix} / {cat}')
            rows.append(df.loc[cat, METRIC_COLS].values.astype(float))
    if not rows:
        return
    mat = np.array(rows)

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=max(0.6, mat.max() + 0.05))
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if mat[i, j] < 0.2 else 'black')
    ax.set_title('Metric Heatmap Across All Scenarios', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Score')
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'heatmap.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'heatmap.pdf'))
    plt.close(fig)


def _sample_distribution(scenario_tables, run_dir):
    """Pie charts showing sample counts per category for each scenario type."""
    n_types = len(scenario_tables)
    if n_types == 0:
        return
    fig, axes = plt.subplots(1, n_types, figsize=(4.5 * n_types, 4))
    if n_types == 1:
        axes = [axes]
    for ax, (stype, df) in zip(axes, scenario_tables.items()):
        counts = df['samples'].astype(int)
        cat_labels = [f'{cat}\n(n={int(c)})' for cat, c in zip(df.index, counts)]
        ax.pie(counts, labels=cat_labels, autopct='%1.0f%%', startangle=90,
               colors=PALETTE[:len(counts)], textprops={'fontsize': 8})
        ax.set_title(SCENARIO_TITLES.get(stype, stype), fontsize=11, fontweight='bold')
    fig.suptitle('Sample Distribution by Scenario', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'sample_distribution.png'), dpi=200,
                bbox_inches='tight')
    fig.savefig(osp.join(run_dir, 'sample_distribution.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def _radar_chart(scenario_tables, global_row, run_dir):
    """One radar chart per scenario type overlaying all categories + global."""
    angles = np.linspace(0, 2 * np.pi, len(METRIC_COLS), endpoint=False).tolist()
    angles += angles[:1]

    for stype, df in scenario_tables.items():
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        gvals = [float(global_row.get(m, 0)) for m in METRIC_COLS]
        gvals += gvals[:1]
        ax.plot(angles, gvals, 'k--', linewidth=1.2, label='Global')
        ax.fill(angles, gvals, alpha=0.05, color='grey')
        for idx, cat in enumerate(df.index):
            vals = [float(df.loc[cat, m]) for m in METRIC_COLS]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.5,
                    label=f'{cat} (n={int(df.loc[cat, "samples"])})',
                    color=PALETTE[idx % len(PALETTE)])
            ax.fill(angles, vals, alpha=0.08,
                    color=PALETTE[idx % len(PALETTE)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], fontsize=9)
        ax.set_ylim(0, min(1.0, df[METRIC_COLS].max().max() + 0.15))
        ax.set_title(f'{SCENARIO_TITLES.get(stype, stype)}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig.savefig(osp.join(run_dir, f'radar_{stype}.png'), dpi=200,
                    bbox_inches='tight')
        fig.savefig(osp.join(run_dir, f'radar_{stype}.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


def generate_visualizations(scenario_tables, global_row, run_dir):
    """Generate all visualization plots and save to run_dir."""
    global_score = float(global_row.get('OpenLane-V2 Score', 0))

    for stype, df in scenario_tables.items():
        _bar_chart_per_scenario(df, stype, global_score, run_dir)

    _score_drop_chart(scenario_tables, global_score, run_dir)
    _heatmap(scenario_tables, run_dir)
    _sample_distribution(scenario_tables, run_dir)
    _radar_chart(scenario_tables, global_row, run_dir)

    print(f"\nVisualizations saved to: {run_dir}")
    print("  bar_<scenario>.png/pdf        : grouped bar charts per scenario")
    print("  score_delta.png/pdf           : performance gap vs. global")
    print("  heatmap.png/pdf               : metric heatmap across all categories")
    print("  sample_distribution.png/pdf   : pie charts of sample counts")
    print("  radar_<scenario>.png/pdf      : radar overlay per scenario")


def parse_args():
    parser = argparse.ArgumentParser(description="TopoNet evaluation")

    parser.add_argument("config", help="config file")
    parser.add_argument("checkpoint", help="checkpoint file")

    parser.add_argument("--out", action="store_true")
    parser.add_argument("--out-dir", default="work_dirs/results")

    parser.add_argument(
        "--eval",
        nargs="+",
        help="evaluation metrics"
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction
    )

    args = parser.parse_args()
    return args


# ---------------------------------------------------
# Scenario Evaluation
# ---------------------------------------------------

def evaluate_by_scenario(dataset, outputs, eval_kwargs, out_dir):

    metric_cols = ['OpenLane-V2 Score', 'DET_l', 'DET_t', 'TOP_ll', 'TOP_lt']
    all_rows = []  # collect every row for the combined CSV

    # ---- Global ----
    print("\n==============================")
    print("GLOBAL METRICS")
    print("==============================")

    global_metrics = dataset.evaluate(outputs, **eval_kwargs)
    print(global_metrics)

    global_row = {'scenario_type': 'global', 'category': 'all', 'samples': len(outputs)}
    global_row.update({k: global_metrics.get(k) for k in metric_cols})
    all_rows.append(global_row)

    # ---- Build scenario groups ----
    scenarios = {
        "curvature": {},
        "lighting": {},
        "occlusion": {},
        "topology_complexity": {}
    }

    for i in range(len(dataset)):

        info = dataset.data_infos[i]
        if "scenario_meta" not in info:
            continue

        scenario_meta = info["scenario_meta"]
        mapping = {}

        if "curvature" in scenario_meta:
            curve_value = scenario_meta["curvature"].get("value_m_inv", None)
            thresholds = scenario_meta["curvature"].get("thresholds_m_inv", {})
            if curve_value is not None:
                if curve_value <= thresholds.get("straight", 0.003):
                    mapping["curvature"] = "straight"
                elif curve_value <= thresholds.get("low", 0.008):
                    mapping["curvature"] = "low curvature"
                elif curve_value <= thresholds.get("medium", 0.02):
                    mapping["curvature"] = "medium curvature"
                else:
                    mapping["curvature"] = "high curvature"

        if "topology_complexity" in scenario_meta:
            topo_value = scenario_meta["topology_complexity"].get("value", None)
            if topo_value is not None:
                if topo_value <= 0.3:
                    mapping["topology_complexity"] = "low topology complexity"
                elif topo_value <= 0.6:
                    mapping["topology_complexity"] = "medium topology complexity"
                else:
                    mapping["topology_complexity"] = "high topology complexity"

        if "lighting" in scenario_meta:
            lighting_label = scenario_meta["lighting"].get("label", None)
            if lighting_label:
                mapping["lighting"] = lighting_label

        if "occlusion" in scenario_meta:
            occlusion_label = scenario_meta["occlusion"].get("label", None)
            if occlusion_label:
                mapping["occlusion"] = occlusion_label

        for key, value in mapping.items():
            if value is None:
                continue
            scenarios[key].setdefault(value, []).append(i)

    # ---- Per-scenario evaluation ----
    print("\n==============================")
    print("SCENARIO BREAKDOWN")
    print("==============================")

    scenario_tables = {}

    for scenario_type in scenarios:

        print("\n---", scenario_type.upper(), "---")

        if len(scenarios[scenario_type]) == 0:
            print(f"  No data available")
            continue

        rows = []
        for category in sorted(scenarios[scenario_type].keys()):

            indices = scenarios[scenario_type][category]
            subset_outputs = [outputs[i] for i in indices]

            original_data_infos = dataset.data_infos
            dataset.data_infos = [dataset.data_infos[i] for i in indices]
            metrics = dataset.evaluate(subset_outputs, **eval_kwargs)
            dataset.data_infos = original_data_infos

            row = {'category': category, 'samples': len(indices)}
            row.update({k: metrics.get(k) for k in metric_cols})
            rows.append(row)

            combined_row = {'scenario_type': scenario_type, **row}
            all_rows.append(combined_row)

            print(f"  {category}: {len(indices)} samples")
            print(f"    Metrics: {metrics}")

        df = pd.DataFrame(rows).set_index('category')
        scenario_tables[scenario_type] = df

    # ---- Save everything ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(out_dir, f"eval_{timestamp}")
    mmcv.mkdir_or_exist(run_dir)

    # Global metrics
    global_df = pd.DataFrame([global_row]).set_index('scenario_type')
    global_df.to_csv(osp.join(run_dir, "global_metrics.csv"))

    # Per-scenario tables
    for scenario_type, df in scenario_tables.items():
        df.to_csv(osp.join(run_dir, f"scenario_{scenario_type}.csv"))

    # Combined table (all rows in one file)
    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(osp.join(run_dir, "all_metrics.csv"), index=False)

    # Pretty-print summary
    print("\n==============================")
    print("SAVED RESULTS")
    print("==============================")
    print(f"Directory: {run_dir}")
    print(f"\nGlobal:")
    print(global_df.to_string())
    for scenario_type, df in scenario_tables.items():
        print(f"\n{scenario_type}:")
        print(df.to_string())

    # ---- Generate visualizations ----
    generate_visualizations(scenario_tables, global_row, run_dir)

    return run_dir



# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    logger = get_logger(name="mmdet")

    mmcv.mkdir_or_exist(osp.abspath(args.out_dir))

    set_random_seed(args.seed)
    # -------------------------
    # Model
    # -------------------------
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # -------------------------
    # Dataset
    # -------------------------
    print("Building dataset")
    dataset = build_dataset(cfg.data.test)
    
    # Limit to first N samples for faster testing on limited GPU/memory
    max_samples = 100
    if len(dataset) > max_samples:
        print(f"Limiting evaluation to first {max_samples} samples (original: {len(dataset)})")
        dataset.data_infos = dataset.data_infos[:max_samples]
    
    print("Loading data")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,  # Disable multiprocessing workers to avoid memory issues on Windows
        dist=False,
        shuffle=False,
    )



    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])

    # -------------------------
    # Inference
    # -------------------------

    outputs = single_gpu_test(model, data_loader)

    # -------------------------
    # Save results
    # -------------------------

    if args.out:
        out_file = osp.join(args.out_dir, "results.pkl")
        mmcv.dump(outputs, out_file)
        print("Saved:", out_file)

    # -------------------------
    # Evaluation
    # -------------------------

    if args.eval:

        eval_kwargs = cfg.get("evaluation", {}).copy()

        evaluate_by_scenario(dataset, outputs, eval_kwargs, args.out_dir)


if __name__ == "__main__":
    main()