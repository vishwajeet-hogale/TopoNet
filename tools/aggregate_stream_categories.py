import argparse
import concurrent.futures
import glob
import json
import os
import os.path as osp
import warnings
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd

from projects.toponet.core.lane.util import fix_pts_interpolate
from projects.toponet.utils.openlanev2_eval_stream import evaluate_centerline_stream

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


METRIC_COLS = ["OpenLane-V2 Score", "DET_l", "DET_t", "TOP_ll", "TOP_lt"]


def build_submission(results_dict):
    return {
        "method": "TopoNet",
        "authors": ["TopoNet"],
        "e-mail": "toponet@example.com",
        "institution / company": "OpenDriveLab",
        "country / region": "CN",
        "results": results_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate streamed TopoNet outputs by scenario categories"
    )
    parser.add_argument(
        "--stream-dir",
        default="work_dirs/results/stream_outputs",
        help="Directory containing streamed .pkl files and optional manifest.txt",
    )
    parser.add_argument(
        "--data-root",
        default="data/OpenLane-V2",
        help="Root directory of OpenLane-V2 data",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split used to produce the streamed outputs",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/results",
        help="Directory where outputs will be saved",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of samples. -1 means all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker threads for loading/processing samples. 0 uses auto.",
    )
    return parser.parse_args()


def _resolve_manifest_path(stream_dir, manifest_path, raw_path):
    normalized = osp.normpath(raw_path)
    if osp.isabs(normalized):
        return normalized

    candidates = [
        osp.normpath(osp.join(stream_dir, normalized)),
        osp.normpath(osp.join(osp.dirname(manifest_path), normalized)),
        osp.normpath(normalized),
        osp.normpath(osp.join(stream_dir, osp.basename(normalized))),
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if osp.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve manifest entry to an existing pickle file: {raw_path}"
    )


def load_stream_paths(stream_dir):
    manifest = osp.join(stream_dir, "manifest.txt")
    paths = []

    if osp.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                fp = line.strip()
                if fp:
                    paths.append(_resolve_manifest_path(stream_dir, manifest, fp))

    if not paths:
        paths = sorted(glob.glob(osp.join(stream_dir, "*.pkl")))

    if not paths:
        raise FileNotFoundError(f"No streamed .pkl files found in: {stream_dir}")

    return paths


def load_json_paths(data_root, split):
    split_dir = osp.join(data_root, split)
    if not osp.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    paths = []
    segments = sorted(glob.glob(osp.join(split_dir, "*")))
    for seg in segments:
        info_dir = osp.join(seg, "info")
        if not osp.isdir(info_dir):
            continue
        paths.extend(sorted(glob.glob(osp.join(info_dir, "*.json"))))

    if not paths:
        raise FileNotFoundError(f"No info JSON files found under: {split_dir}")

    return paths


def get_curvature_label(meta):
    if "curvature" not in meta:
        return None
    value = meta["curvature"].get("value_m_inv")
    thresholds = meta["curvature"].get("thresholds_m_inv", {})
    if value is None:
        return None
    if value <= thresholds.get("straight", 0.003):
        return "straight"
    if value <= thresholds.get("low", 0.008):
        return "low curvature"
    if value <= thresholds.get("medium", 0.02):
        return "medium curvature"
    return "high curvature"


def get_sample_labels(scenario_meta):
    labels = {}

    curvature = get_curvature_label(scenario_meta)
    if curvature:
        labels["curvature"] = curvature

    lighting = scenario_meta.get("lighting", {}).get("label")
    if lighting:
        labels["lighting"] = lighting

    occlusion = scenario_meta.get("occlusion", {}).get("label")
    if occlusion:
        labels["occlusion"] = occlusion

    return labels


def get_prediction_stats(pred):
    lane_scores = []
    te_scores = []

    lane_results = pred.get("lane_results")
    if lane_results is not None and len(lane_results) > 1 and lane_results[1] is not None:
        lane_scores = np.asarray(lane_results[1], dtype=np.float32)

    bbox_results = pred.get("bbox_results")
    if bbox_results is not None and len(bbox_results) > 1 and bbox_results[1] is not None:
        te_scores = np.asarray(bbox_results[1], dtype=np.float32)

    n_lanes = int(len(lane_scores))
    n_traffic_elements = int(len(te_scores))

    lane_conf_mean = float(np.mean(lane_scores)) if n_lanes > 0 else np.nan
    te_conf_mean = float(np.mean(te_scores)) if n_traffic_elements > 0 else np.nan

    return {
        "n_lanes": n_lanes,
        "n_traffic_elements": n_traffic_elements,
        "lane_conf_mean": lane_conf_mean,
        "te_conf_mean": te_conf_mean,
    }


def prepare_eval_annotation(annotation):
    prepared = {
        "lane_centerline": [],
        "traffic_element": [],
        "topology_lclc": np.array(annotation.get("topology_lclc", []), dtype=np.float32),
        "topology_lcte": np.array(annotation.get("topology_lcte", []), dtype=np.float32),
    }

    for lane in annotation.get("lane_centerline", []):
        lane_out = dict(lane)
        points = np.array(lane.get("points", []), dtype=np.float32)
        if len(points) == 201:
            points = points[::20]
        lane_out["points"] = points
        prepared["lane_centerline"].append(lane_out)

    for te in annotation.get("traffic_element", []):
        te_out = dict(te)
        te_out["points"] = np.array(te.get("points", []), dtype=np.float32)
        prepared["traffic_element"].append(te_out)

    return prepared


def convert_prediction_to_openlane(raw_pred):
    pred_info = {
        "lane_centerline": [],
        "traffic_element": [],
        "topology_lclc": None,
        "topology_lcte": None,
    }

    valid_indices = None
    lane_results = raw_pred.get("lane_results")
    if lane_results is not None:
        scores = lane_results[1]
        if scores is not None:
            valid_indices = np.argsort(-scores)
            lanes = lane_results[0][valid_indices]
            lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)
            sorted_scores = scores[valid_indices]
            for pred_idx, (lane, score) in enumerate(zip(lanes, sorted_scores)):
                points = fix_pts_interpolate(lane, 11)
                pred_info["lane_centerline"].append(
                    {
                        "id": 10000 + pred_idx,
                        "points": points.astype(np.float32),
                        "confidence": float(score),
                    }
                )

    te_valid_indices = None
    bbox_results = raw_pred.get("bbox_results")
    if bbox_results is not None:
        scores = bbox_results[1]
        if scores is not None:
            te_valid_indices = np.argsort(-scores)
            tes = bbox_results[0][te_valid_indices]
            sorted_scores = scores[te_valid_indices]
            class_idxs = bbox_results[2][te_valid_indices]
            for pred_idx, (te, score, class_idx) in enumerate(zip(tes, sorted_scores, class_idxs)):
                class_idx = int(class_idx)
                pred_info["traffic_element"].append(
                    {
                        "id": 20000 + pred_idx,
                        "category": 1 if class_idx < 4 else 2,
                        "attribute": class_idx,
                        "points": te.reshape(2, 2).astype(np.float32),
                        "confidence": float(score),
                    }
                )

    lclc = raw_pred.get("lclc_results")
    if lclc is not None and valid_indices is not None:
        pred_info["topology_lclc"] = lclc.astype(np.float32)[valid_indices][:, valid_indices]
    else:
        n = len(pred_info["lane_centerline"])
        pred_info["topology_lclc"] = np.zeros((n, n), dtype=np.float32)

    lcte = raw_pred.get("lcte_results")
    if lcte is not None and valid_indices is not None and te_valid_indices is not None:
        pred_info["topology_lcte"] = lcte.astype(np.float32)[valid_indices][:, te_valid_indices]
    else:
        pred_info["topology_lcte"] = np.zeros(
            (len(pred_info["lane_centerline"]), len(pred_info["traffic_element"])),
            dtype=np.float32,
        )

    return pred_info


def extract_token(json_path, split):
    timestamp = osp.splitext(osp.basename(json_path))[0]
    info_dir = osp.dirname(json_path)
    segment_id = osp.basename(osp.dirname(info_dir))
    return (split, segment_id, str(timestamp))


def update_bucket(bucket, stats):
    bucket["samples"] += 1
    bucket["sum_lanes"] += stats["n_lanes"]
    bucket["sum_traffic_elements"] += stats["n_traffic_elements"]

    if not np.isnan(stats["lane_conf_mean"]):
        bucket["sum_lane_conf_mean"] += stats["lane_conf_mean"]
        bucket["count_lane_conf_mean"] += 1

    if not np.isnan(stats["te_conf_mean"]):
        bucket["sum_te_conf_mean"] += stats["te_conf_mean"]
        bucket["count_te_conf_mean"] += 1


def finalize_bucket(bucket):
    samples = max(bucket["samples"], 1)
    lane_conf_count = max(bucket["count_lane_conf_mean"], 1)
    te_conf_count = max(bucket["count_te_conf_mean"], 1)

    return {
        "samples": bucket["samples"],
        "avg_lanes": bucket["sum_lanes"] / samples,
        "avg_traffic_elements": bucket["sum_traffic_elements"] / samples,
        "avg_lane_conf_mean": bucket["sum_lane_conf_mean"] / lane_conf_count,
        "avg_te_conf_mean": bucket["sum_te_conf_mean"] / te_conf_count,
    }


def _resolve_workers(workers, total_items):
    if total_items <= 1:
        return 1
    if workers and workers > 0:
        return min(workers, total_items)
    cpu = os.cpu_count() or 1
    return min(max(4, cpu), 16, total_items)


def _process_sample(pkl_path, json_path, split):
    pred = mmcv.load(pkl_path)
    with open(json_path, "r", encoding="utf-8") as handle:
        sample_json = json.load(handle)

    scenario_meta = sample_json.get("scenario_meta", {})
    labels = get_sample_labels(scenario_meta)
    stats = get_prediction_stats(pred)
    token = extract_token(json_path, split)

    annotation = sample_json.get("annotation")
    if annotation is None:
        raise KeyError(f"Missing annotation in JSON: {json_path}")

    gt_entry = {"annotation": prepare_eval_annotation(annotation)}
    pred_entry = {"predictions": convert_prediction_to_openlane(pred)}
    return token, labels, stats, gt_entry, pred_entry


def aggregate(stream_paths, json_paths, split, workers=0):
    scenario_buckets = {
        "curvature": defaultdict(lambda: {
            "samples": 0,
            "sum_lanes": 0,
            "sum_traffic_elements": 0,
            "sum_lane_conf_mean": 0.0,
            "count_lane_conf_mean": 0,
            "sum_te_conf_mean": 0.0,
            "count_te_conf_mean": 0,
        }),
        "lighting": defaultdict(lambda: {
            "samples": 0,
            "sum_lanes": 0,
            "sum_traffic_elements": 0,
            "sum_lane_conf_mean": 0.0,
            "count_lane_conf_mean": 0,
            "sum_te_conf_mean": 0.0,
            "count_te_conf_mean": 0,
        }),
        "occlusion": defaultdict(lambda: {
            "samples": 0,
            "sum_lanes": 0,
            "sum_traffic_elements": 0,
            "sum_lane_conf_mean": 0.0,
            "count_lane_conf_mean": 0,
            "sum_te_conf_mean": 0.0,
            "count_te_conf_mean": 0,
        }),
    }

    token_to_labels = {
        "curvature": defaultdict(list),
        "lighting": defaultdict(list),
        "occlusion": defaultdict(list),
    }

    gt_dict = {}
    pred_results = {}

    total = min(len(stream_paths), len(json_paths))
    samples = list(zip(stream_paths, json_paths))
    resolved_workers = _resolve_workers(workers, total)

    if resolved_workers <= 1:
        iterator = samples
        if tqdm is not None:
            iterator = tqdm(iterator, total=total, desc="Aggregating samples", unit="sample")

        for pkl_path, json_path in iterator:
            token, labels, stats, gt_entry, pred_entry = _process_sample(pkl_path, json_path, split)
            gt_dict[token] = gt_entry
            pred_results[token] = pred_entry
            for scenario_type, label in labels.items():
                update_bucket(scenario_buckets[scenario_type][label], stats)
                token_to_labels[scenario_type][label].append(token)
    else:
        progress = None
        if tqdm is not None:
            progress = tqdm(total=total, desc=f"Aggregating samples ({resolved_workers} workers)", unit="sample")

        with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            futures = [
                executor.submit(_process_sample, pkl_path, json_path, split)
                for pkl_path, json_path in samples
            ]
            for future in concurrent.futures.as_completed(futures):
                token, labels, stats, gt_entry, pred_entry = future.result()
                gt_dict[token] = gt_entry
                pred_results[token] = pred_entry
                for scenario_type, label in labels.items():
                    update_bucket(scenario_buckets[scenario_type][label], stats)
                    token_to_labels[scenario_type][label].append(token)
                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

    final = {}
    for scenario_type, category_map in scenario_buckets.items():
        final[scenario_type] = {
            category: finalize_bucket(bucket)
            for category, bucket in category_map.items()
        }
    return final, token_to_labels, gt_dict, pred_results


def _extract_olv_metrics(metric_results):
    score = metric_results["OpenLane-V2 Score"]
    return {
        "OpenLane-V2 Score": float(score["score"]),
        "DET_l": float(score["DET_l"]),
        "DET_t": float(score["DET_t"]),
        "TOP_ll": float(score["TOP_ll"]),
        "TOP_lt": float(score["TOP_lt"]),
    }


def compute_olv_tables(token_to_labels, gt_dict, pred_results):
    global_metrics = _extract_olv_metrics(
        evaluate_centerline_stream(gt_dict, build_submission(pred_results), verbose=False)
    )

    scenario_tables = {}
    for scenario_type in ["curvature", "lighting", "occlusion"]:
        rows = []
        categories = sorted(token_to_labels[scenario_type].keys())
        iterator = categories
        if tqdm is not None and categories:
            iterator = tqdm(categories, desc=f"OLV metrics: {scenario_type}", unit="category")

        for category in iterator:
            tokens = token_to_labels[scenario_type][category]
            if not tokens:
                continue
            sub_gt = {token: gt_dict[token] for token in tokens}
            sub_pred = {token: pred_results[token] for token in tokens}
            metrics = _extract_olv_metrics(
                evaluate_centerline_stream(sub_gt, build_submission(sub_pred), verbose=False)
            )
            row = {
                "scenario_type": scenario_type,
                "category": category,
                "samples": len(tokens),
            }
            row.update(metrics)
            rows.append(row)

        scenario_tables[scenario_type] = pd.DataFrame(rows)

    return global_metrics, scenario_tables


def to_dataframe(category_dict):
    rows = []
    for category, values in sorted(category_dict.items()):
        row = {"category": category}
        row.update(values)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "category",
            "samples",
            "avg_lanes",
            "avg_traffic_elements",
            "avg_lane_conf_mean",
            "avg_te_conf_mean",
        ])
    return pd.DataFrame(rows)


def plot_scenario(df, scenario_type, run_dir):
    if df.empty:
        return

    categories = [f"{cat}\n(n={int(n)})" for cat, n in zip(df["category"], df["samples"]) ]
    x = np.arange(len(categories))

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(categories) * 1.4), 8), sharex=True)

    bars_lanes = axes[0].bar(x - 0.2, df["avg_lanes"], width=0.4, label="Avg Lanes / sample", color="#2a9d8f")
    bars_te = axes[0].bar(x + 0.2, df["avg_traffic_elements"], width=0.4, label="Avg Traffic Elements / sample", color="#457b9d")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{scenario_type.title()} - Prediction Density By Category")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for bar in bars_lanes:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, height + 0.03, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_te:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, height + 0.03, f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    bars_lane_conf = axes[1].bar(x - 0.2, df["avg_lane_conf_mean"], width=0.4, label="Mean lane confidence", color="#f4a261")
    bars_te_conf = axes[1].bar(x + 0.2, df["avg_te_conf_mean"], width=0.4, label="Mean traffic-element confidence", color="#e76f51")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"{scenario_type.title()} - Confidence Quality By Category")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    for bar in bars_lane_conf:
        height = bar.get_height()
        if np.isnan(height):
            continue
        axes[1].text(bar.get_x() + bar.get_width() / 2, min(height + 0.015, 0.99), f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_te_conf:
        height = bar.get_height()
        if np.isnan(height):
            continue
        axes[1].text(bar.get_x() + bar.get_width() / 2, min(height + 0.015, 0.99), f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f"scenario_{scenario_type}_summary.png"), dpi=160)
    plt.close(fig)


def plot_olv_imbalance(df, scenario_type, global_metrics, run_dir):
    if df.empty:
        return

    categories = [f"{cat}\n(n={int(n)})" for cat, n in zip(df["category"], df["samples"]) ]
    x = np.arange(len(categories))
    width = 0.16

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(categories) * 1.5), 8), sharex=True)
    ax = axes[0]
    for idx, metric in enumerate(METRIC_COLS):
        offsets = x + (idx - (len(METRIC_COLS) - 1) / 2) * width
        bars = ax.bar(offsets, df[metric].values, width=width, label=metric)
        if metric == "OpenLane-V2 Score":
            for bar in bars:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, min(v + 0.015, 0.99), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(global_metrics["OpenLane-V2 Score"], color="black", linestyle="--", linewidth=1.0, label="Global OLV2")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metric score")
    ax.set_title(f"{scenario_type.title()} - OLV Metrics By Category")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, title="Metrics")

    delta = df["OpenLane-V2 Score"] - global_metrics["OpenLane-V2 Score"]
    colors = ["#d62828" if v < 0 else "#2a9d8f" for v in delta.values]
    bars_delta = axes[1].bar(x, delta.values, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("Delta OLV2")
    axes[1].set_title("Category Gap vs Global OLV2 (negative = worse)")
    axes[1].grid(axis="y", alpha=0.25)

    for bar, v in zip(bars_delta, delta.values):
        va = "bottom" if v >= 0 else "top"
        y = v + 0.005 if v >= 0 else v - 0.005
        axes[1].text(bar.get_x() + bar.get_width() / 2, y, f"{v:+.3f}", ha="center", va=va, fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=20, ha="right")

    fig.text(
        0.01,
        0.01,
        "DET_l: lane detection, DET_t: traffic-element detection, TOP_ll: lane-lane topology, TOP_lt: lane-traffic topology",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f"scenario_{scenario_type}_olv_metrics.png"), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()

    # Known benign warning in OpenLane f-score interpolation for degenerate lanes.
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in true_divide",
        category=RuntimeWarning,
    )

    stream_paths = load_stream_paths(args.stream_dir)
    json_paths = load_json_paths(args.data_root, args.split)

    if args.max_samples > 0:
        stream_paths = stream_paths[:args.max_samples]
        json_paths = json_paths[:args.max_samples]

    keep_n = min(len(stream_paths), len(json_paths))
    if len(stream_paths) != len(json_paths):
        print(
            f"Warning: stream outputs ({len(stream_paths)}) and JSON files ({len(json_paths)}) differ; truncating to {keep_n}."
        )
    stream_paths = stream_paths[:keep_n]
    json_paths = json_paths[:keep_n]

    if keep_n == 0:
        raise RuntimeError("No samples to aggregate after filtering.")

    summary, token_to_labels, gt_dict, pred_results = aggregate(
        stream_paths,
        json_paths,
        args.split,
        workers=args.workers,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(args.out_dir, f"category_summary_{timestamp}")
    mmcv.mkdir_or_exist(run_dir)

    with open(osp.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    global_metrics, scenario_metric_tables = compute_olv_tables(token_to_labels, gt_dict, pred_results)

    global_df = pd.DataFrame([global_metrics])
    global_df.to_csv(osp.join(run_dir, "global_olv_metrics.csv"), index=False)

    imbalance_rows = []
    for scenario_type, metric_df in scenario_metric_tables.items():
        if metric_df.empty:
            continue
        for metric in METRIC_COLS:
            metric_df[f"delta_{metric}"] = metric_df[metric] - global_metrics[metric]
        metric_df.to_csv(osp.join(run_dir, f"scenario_{scenario_type}_olv_metrics.csv"), index=False)
        plot_olv_imbalance(metric_df, scenario_type, global_metrics, run_dir)
        imbalance_rows.extend(metric_df.to_dict(orient="records"))

    if imbalance_rows:
        pd.DataFrame(imbalance_rows).to_csv(osp.join(run_dir, "all_olv_imbalance.csv"), index=False)

    for scenario_type in ["curvature", "lighting", "occlusion"]:
        df = to_dataframe(summary.get(scenario_type, {}))
        df.to_csv(osp.join(run_dir, f"scenario_{scenario_type}.csv"), index=False)
        plot_scenario(df, scenario_type, run_dir)

    print("Saved category aggregation results to:", run_dir)
    print("Files:")
    print("  summary.json")
    print("  global_olv_metrics.csv")
    print("  all_olv_imbalance.csv (if categories available)")
    print("  scenario_*.csv")
    print("  scenario_*_summary.png")
    print("  scenario_*_olv_metrics.csv")
    print("  scenario_*_olv_metrics.png")


if __name__ == "__main__":
    main()
