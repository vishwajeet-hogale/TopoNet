import argparse
import concurrent.futures
import ctypes
import gc
import glob
import json
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import pandas as pd
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


METRIC_COLS_EVAL = ["OpenLane-V2 Score", "DET_l", "DET_t", "TOP_ll", "TOP_lt"]
METRIC_LABELS = {
    "OpenLane-V2 Score": "OLV2 Score",
    "DET_l": "DET_l",
    "DET_t": "DET_t",
    "TOP_ll": "TOP_ll",
    "TOP_lt": "TOP_lt",
}
SCENARIO_TITLES = {
    "curvature": "Road Curvature",
    "lighting": "Lighting Condition",
    "occlusion": "Occlusion Level",
    "topology_complexity": "Topology Complexity",
}


def _dataset_has_annotations(dataset):
    """Return True if the dataset appears to contain GT annotations."""
    if not hasattr(dataset, "data_infos") or not dataset.data_infos:
        return False
    first = dataset.data_infos[0]
    return isinstance(first, dict) and "annotation" in first


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from streamed TopoNet predictions"
    )
    parser.add_argument("config", help="Config file path")
    parser.add_argument(
        "--stream-dir",
        default="work_dirs/results/stream_outputs",
        help="Directory containing streamed .pkl outputs and optional manifest.txt",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/results",
        help="Directory where aggregated CSVs will be written",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override data.test.split in config (e.g. train, val, test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of samples loaded from stream outputs. -1 means all.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["auto", "memory", "lazy"],
        default="auto",
        help=(
            "How to access streamed prediction files. "
            "'memory' preloads once for fastest repeated evaluation, "
            "'lazy' loads on demand with lower RAM use, and "
            "'auto' chooses based on total pickle size."
        ),
    )
    parser.add_argument(
        "--max-preload-gb",
        type=float,
        default=None,
        help=(
            "Optional hard cap for auto-mode preloading in GB. "
            "If omitted, a conservative fraction of currently available RAM is used."
        ),
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=0,
        help="Number of worker threads used to preload pickles. 0 picks a reasonable default.",
    )
    parser.add_argument(
        "--lazy-cache-items",
        type=int,
        default=128,
        help=(
            "Number of predictions kept in an in-memory LRU cache when using --cache-mode lazy. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--eval",
        nargs="+",
        default=None,
        help="Override evaluation metrics list (e.g. OpenLane-V2)",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def _sanitize_eval_kwargs(eval_kwargs):
    """Remove EvalHook-only keys that dataset.evaluate does not accept."""
    cleaned = dict(eval_kwargs) if eval_kwargs is not None else {}
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        cleaned.pop(key, None)
    return cleaned


def _get_sample_scenario_labels(info):
    """Return {scenario_type: category_label} for a single sample's scenario_meta."""
    out = {}
    meta = info.get("scenario_meta", {})

    if "curvature" in meta:
        v = meta["curvature"].get("value_m_inv")
        t = meta["curvature"].get("thresholds_m_inv", {})
        if v is not None:
            if v <= t.get("straight", 0.003):
                out["curvature"] = "straight"
            elif v <= t.get("low", 0.008):
                out["curvature"] = "low curvature"
            elif v <= t.get("medium", 0.02):
                out["curvature"] = "medium curvature"
            else:
                out["curvature"] = "high curvature"

    if "topology_complexity" in meta:
        v = meta["topology_complexity"].get("value")
        if v is not None:
            if v <= 0.3:
                out["topology_complexity"] = "low topology complexity"
            elif v <= 0.6:
                out["topology_complexity"] = "medium topology complexity"
            else:
                out["topology_complexity"] = "high topology complexity"

    if "lighting" in meta:
        lbl = meta["lighting"].get("label")
        if lbl:
            out["lighting"] = lbl

    if "occlusion" in meta:
        lbl = meta["occlusion"].get("label")
        if lbl:
            out["occlusion"] = lbl

    return out


class _ProgressBar:
    def __init__(self, total, desc):
        self.total = max(int(total), 0)
        self.desc = desc
        self._bar = None
        self._done = 0

        if tqdm is not None:
            self._bar = tqdm(total=self.total, desc=desc, unit="item")
        else:
            print(f"{desc} ...")
            self._bar = mmcv.ProgressBar(self.total)

    def update(self, n=1):
        if self._bar is None:
            return
        if tqdm is not None:
            self._bar.update(n)
            return

        for _ in range(n):
            if self._done >= self.total:
                break
            self._bar.update()
            self._done += 1

    def close(self):
        if tqdm is not None and self._bar is not None:
            self._bar.close()


class _LazyPklList:
    """List-like object that loads .pkl files on demand instead of all at once."""

    def __init__(self, paths, cache_items=0):
        self._paths = list(paths)
        self._cache_items = max(int(cache_items), 0)
        self._cache = OrderedDict() if self._cache_items > 0 else None

    def _load_index(self, idx):
        if self._cache is None:
            return mmcv.load(self._paths[idx])

        if idx in self._cache:
            value = self._cache.pop(idx)
            self._cache[idx] = value
            return value

        value = mmcv.load(self._paths[idx])
        self._cache[idx] = value
        if len(self._cache) > self._cache_items:
            self._cache.popitem(last=False)
        return value

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _LazyPklList(self._paths[idx], cache_items=self._cache_items)
        return self._load_index(idx)

    def __iter__(self):
        for idx in range(len(self._paths)):
            yield self._load_index(idx)


def _resolve_stream_output_path(stream_dir, manifest_path, raw_path):
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
        "Could not resolve manifest entry to an existing pickle file: "
        f"{raw_path}"
    )


def _load_streamed_prediction_paths(stream_dir):
    manifest = osp.join(stream_dir, "manifest.txt")
    paths = []

    if osp.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                fp = line.strip()
                if fp:
                    fp = _resolve_stream_output_path(stream_dir, manifest, fp)
                    paths.append(fp)

    if not paths:
        paths = sorted(glob.glob(osp.join(stream_dir, "*.pkl")))

    if not paths:
        raise FileNotFoundError(f"No .pkl prediction files found in: {stream_dir}")

    return paths


def _format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _get_available_memory_bytes():
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)
    except (AttributeError, OSError, ValueError):
        return None


def _get_auto_preload_limit_bytes(max_preload_gb):
    if max_preload_gb is not None:
        return int(max_preload_gb * (1024 ** 3))

    available_bytes = _get_available_memory_bytes()
    if available_bytes is None:
        return int(2.0 * (1024 ** 3))

    # Keep the cache well below available RAM to leave room for the dataset,
    # evaluator allocations, Python overhead, and the OS page cache.
    return max(int(available_bytes * 0.35), int(1.0 * (1024 ** 3)))


def _resolve_load_workers(num_workers, num_items):
    if num_items <= 1:
        return 1
    if num_workers and num_workers > 0:
        return min(num_workers, num_items)
    cpu_count = os.cpu_count() or 1
    return min(max(4, cpu_count), 16, num_items)


def _is_openlane_json_dataset(dataset_cfg):
    dataset_type = dataset_cfg.get("type")
    return dataset_type == "OpenLaneJSONDataset"


def _configure_dataset_for_eval(dataset_cfg):
    dataset_cfg = dataset_cfg.copy()
    if _is_openlane_json_dataset(dataset_cfg):
        dataset_cfg["lazy_load"] = True
        dataset_cfg["pipeline"] = None
    return dataset_cfg


def _load_scenario_meta_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("scenario_meta", {})


def _materialize_predictions(paths, num_workers):
    outputs = [None] * len(paths)
    workers = _resolve_load_workers(num_workers, len(paths))
    progress = _ProgressBar(len(paths), "Loading streamed predictions")

    try:
        if workers == 1:
            for idx, path in enumerate(paths):
                outputs[idx] = mmcv.load(path)
                progress.update()
            return outputs

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(mmcv.load, path): idx for idx, path in enumerate(paths)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                outputs[idx] = future.result()
                progress.update()
    finally:
        progress.close()

    return outputs


def _build_scenarios(dataset):
    scenarios = {
        "curvature": {},
        "lighting": {},
        "occlusion": {},
        "topology_complexity": {},
    }

    progress = _ProgressBar(len(dataset), "Scanning scenario labels")
    try:
        for i, info in enumerate(dataset.data_infos):
            if "scenario_meta" not in info and "json_path" in info:
                info = dict(info)
                info["scenario_meta"] = _load_scenario_meta_from_json(info["json_path"])
                dataset.data_infos[i] = info
            for stype, label in _get_sample_scenario_labels(info).items():
                scenarios[stype].setdefault(label, []).append(i)
            progress.update()
    finally:
        progress.close()

    return scenarios


def _evaluate_subset(dataset, outputs, indices, eval_kwargs):
    original_data_infos = dataset.data_infos
    try:
        dataset.data_infos = [original_data_infos[i] for i in indices]
        subset_outputs = [outputs[i] for i in indices]
        metrics = dataset.evaluate(subset_outputs, **eval_kwargs)
        del subset_outputs
        gc.collect()
        return metrics
    finally:
        dataset.data_infos = original_data_infos


def _choose_output_container(paths, cache_mode, max_preload_gb, load_workers, lazy_cache_items):
    total_bytes = sum(osp.getsize(path) for path in paths)
    auto_limit_bytes = _get_auto_preload_limit_bytes(max_preload_gb)
    print(
        f"Discovered {len(paths)} prediction files "
        f"({_format_bytes(total_bytes)} total)."
    )

    if cache_mode == "memory":
        should_preload = True
    elif cache_mode == "lazy":
        should_preload = False
    else:
        print(
            "Auto cache limit: "
            f"{_format_bytes(auto_limit_bytes)} based on available RAM."
        )
        should_preload = total_bytes <= auto_limit_bytes

    if should_preload:
        print("Using memory cache for aggregation.")
        return _materialize_predictions(paths, load_workers)

    print(
        "Using lazy loading for aggregation because total pickle size exceeds "
        f"the auto preload threshold of {_format_bytes(auto_limit_bytes)}. "
        "Use --cache-mode memory to force preloading if RAM allows it."
    )
    if lazy_cache_items > 0:
        print(f"Enabled lazy LRU cache for up to {lazy_cache_items} predictions.")
    return _LazyPklList(paths, cache_items=lazy_cache_items)


def _plot_global_metrics(global_row, run_dir):
    metric_values = [global_row.get(metric) for metric in METRIC_COLS_EVAL]
    labels = [METRIC_LABELS[metric] for metric in METRIC_COLS_EVAL]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, metric_values, color="#2f6db2")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Global Metrics")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, metric_values):
        if value is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(osp.join(run_dir, "global_metrics.png"), dpi=160)
    plt.close(fig)


def _plot_scenario_table(scenario_type, df, run_dir):
    plot_df = df[METRIC_COLS_EVAL].copy()
    labels = [METRIC_LABELS[metric] for metric in METRIC_COLS_EVAL]
    categories = plot_df.index.tolist()
    x_positions = list(range(len(categories)))
    width = 0.8 / max(len(METRIC_COLS_EVAL), 1)

    fig_width = max(8, len(categories) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    for metric_idx, metric in enumerate(METRIC_COLS_EVAL):
        offsets = [x + (metric_idx - (len(METRIC_COLS_EVAL) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, plot_df[metric].tolist(), width=width, label=METRIC_LABELS[metric])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(SCENARIO_TITLES.get(scenario_type, scenario_type.replace("_", " ").title()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(3, len(labels)), frameon=False)

    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f"scenario_{scenario_type}.png"), dpi=160)
    plt.close(fig)


def _save_metric_plots(global_row, scenario_tables, run_dir):
    _plot_global_metrics(global_row, run_dir)
    for scenario_type, df in scenario_tables.items():
        if not df.empty:
            _plot_scenario_table(scenario_type, df, run_dir)


def evaluate_by_scenario(dataset, outputs, eval_kwargs, out_dir):
    metric_cols = METRIC_COLS_EVAL
    all_rows = []
    scenarios = _build_scenarios(dataset)
    total_eval_steps = 1 + sum(len(categories) for categories in scenarios.values())
    eval_progress = _ProgressBar(total_eval_steps, "Evaluating metric groups")

    print("\n==============================")
    print("GLOBAL METRICS")
    print("==============================")

    scenario_tables = {}
    try:
        global_metrics = dataset.evaluate(outputs, **eval_kwargs)
        eval_progress.update()
        print(global_metrics)
        gc.collect()

        global_row = {
            "scenario_type": "global",
            "category": "all",
            "samples": len(outputs),
        }
        global_row.update({k: global_metrics.get(k) for k in metric_cols})
        all_rows.append(global_row)

        print("\n==============================")
        print("SCENARIO BREAKDOWN")
        print("==============================")

        for scenario_type in scenarios:
            print(f"\n--- {scenario_type.upper()} ---")
            if not scenarios[scenario_type]:
                print("  No data available")
                continue

            rows = []
            for category in sorted(scenarios[scenario_type].keys()):
                indices = scenarios[scenario_type][category]
                metrics = _evaluate_subset(dataset, outputs, indices, eval_kwargs)
                eval_progress.update()

                row = {"category": category, "samples": len(indices)}
                row.update({k: metrics.get(k) for k in metric_cols})
                rows.append(row)

                combined_row = {"scenario_type": scenario_type, **row}
                all_rows.append(combined_row)

                print(f"  {category}: {len(indices)} samples")
                print(f"    Metrics: {metrics}")

            scenario_tables[scenario_type] = pd.DataFrame(rows).set_index("category")
    finally:
        eval_progress.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(out_dir, f"eval_{timestamp}")
    mmcv.mkdir_or_exist(run_dir)

    global_df = pd.DataFrame([global_row]).set_index("scenario_type")
    global_df.to_csv(osp.join(run_dir, "global_metrics.csv"))

    for scenario_type, df in scenario_tables.items():
        df.to_csv(osp.join(run_dir, f"scenario_{scenario_type}.csv"))

    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(osp.join(run_dir, "all_metrics.csv"), index=False)
    _save_metric_plots(global_row, scenario_tables, run_dir)

    print("\n==============================")
    print("SAVED RESULTS")
    print("==============================")
    print(f"Directory: {run_dir}")
    print(
        "Saved: global_metrics.csv, scenario_*.csv, all_metrics.csv, "
        "global_metrics.png, scenario_*.png"
    )

    return run_dir


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.split:
        cfg.data.test.split = args.split
        print(f"Overriding data.test.split = '{args.split}'")

    eval_kwargs = _sanitize_eval_kwargs(cfg.get("evaluation", {}).copy())
    if args.eval is not None:
        eval_kwargs["metric"] = args.eval

    dataset_cfg = _configure_dataset_for_eval(cfg.data.test)
    if _is_openlane_json_dataset(dataset_cfg):
        print("Building dataset in lightweight lazy mode for evaluation ...")
    else:
        print("Building dataset for evaluation ...")
    dataset = build_dataset(dataset_cfg)

    paths = _load_streamed_prediction_paths(args.stream_dir)
    print(f"Found {len(paths)} streamed predictions in: {args.stream_dir}")

    if args.max_samples > 0:
        paths = paths[: args.max_samples]
        print(f"Using first {len(paths)} predictions due to --max-samples")

    if len(dataset) != len(paths):
        keep_n = min(len(dataset), len(paths))
        print(
            f"Warning: dataset size ({len(dataset)}) != output count ({len(paths)}). "
            f"Truncating both to {keep_n}."
        )
        dataset.data_infos = dataset.data_infos[:keep_n]
        paths = paths[:keep_n]

    if (not _dataset_has_annotations(dataset)) and (not _is_openlane_json_dataset(dataset_cfg)):
        split_name = getattr(cfg.data.test, "split", "unknown")
        raise RuntimeError(
            "Selected dataset split does not include ground-truth annotations, so "
            "OpenLane-V2 metrics cannot be computed. "
            f"Current split: '{split_name}'.\n"
            "Use a split with annotations (typically 'val' or 'train'), e.g.:\n"
            "  python tools/aggregate_stream_metrics.py <config> --stream-dir <dir> --out-dir <dir> --split val\n"
            "Important: streamed predictions must come from the same split and order "
            "used for evaluation."
        )

    mmcv.mkdir_or_exist(osp.abspath(args.out_dir))
    outputs = _choose_output_container(
        paths,
        cache_mode=args.cache_mode,
        max_preload_gb=args.max_preload_gb,
        load_workers=args.load_workers,
        lazy_cache_items=args.lazy_cache_items,
    )
    evaluate_by_scenario(dataset, outputs, eval_kwargs, args.out_dir)


if __name__ == "__main__":
    main()
