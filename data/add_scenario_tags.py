#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenLane-V2 scenario tagger (curvature + topology complexity + lighting + occlusion)

Tags written:
-------------
- Curvature:
    "curvature_unknown"
    "straight"
    "low curvature left"
    "low curvature right"
    "medium curvature left"
    "medium curvature right"
    "high curvature left"
    "high curvature right"

- Topology complexity:
    "topology_unknown"
    "low topological complexity"
    "medium topological complexity"
    "high topological complexity"

- Lighting:
    "well lit"
    "poorly lit"
    "lighting_unknown"

- Occlusion:
    "low occlusion"
    "high occlusion"
    "occlusion_unknown"

Meta written:
-------------
scenario_meta.curvature
scenario_meta.topology_complexity
scenario_meta.lighting
scenario_meta.occlusion

Usage:
------
Dry run:
python add_scenario_tags.py \
  --dataset_root /Users/nikhil/Workspace/ScenarioTagsOLV2/OpenLane-V2/data/train \
  --dry_run

Write:
python add_scenario_tags.py \
  --dataset_root /Users/nikhil/Workspace/ScenarioTagsOLV2/OpenLane-V2/data/train
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# =========================================================
# Generic helpers
# =========================================================


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff(a: float, b: float) -> float:
    return angle_wrap(a - b)


def wrap_angle_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    xpad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(xpad, kernel, mode="valid")


def is_point_like(p: Any) -> bool:
    return (
        isinstance(p, (list, tuple))
        and len(p) >= 2
        and all(isinstance(v, (int, float)) for v in p[:2])
    )


def is_polyline_like(x: Any) -> bool:
    return isinstance(x, list) and len(x) >= 2 and all(is_point_like(p) for p in x)


def polyline_to_xy(poly: List[List[float]]) -> np.ndarray:
    arr = np.array(poly, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :2]


def remove_duplicate_points(xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(xy) <= 1:
        return xy
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = d > eps
    return xy[keep]


def resample_polyline(xy: np.ndarray, ds: float = 0.5) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xy = remove_duplicate_points(xy)
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total < ds:
        return xy.copy()

    s_new = np.arange(0.0, total + 1e-9, ds)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.stack([x_new, y_new], axis=1)


def smooth_polyline(xy: np.ndarray, window: int = 5) -> np.ndarray:
    if len(xy) < max(5, window):
        return xy
    x = moving_average(xy[:, 0], window)
    y = moving_average(xy[:, 1], window)
    return np.stack([x, y], axis=1)


def nearest_y_at_x(xy: np.ndarray, x_target: float) -> float:
    if len(xy) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(xy[:, 0] - x_target)))
    return float(xy[idx, 1])


def heading_at_x(xy: np.ndarray, x_target: float) -> float:
    if len(xy) < 3:
        return float("nan")
    idx = int(np.argmin(np.abs(xy[:, 0] - x_target)))
    i0 = max(0, idx - 1)
    i1 = min(len(xy) - 1, idx + 1)
    if i1 == i0:
        return float("nan")
    dx = xy[i1, 0] - xy[i0, 0]
    dy = xy[i1, 1] - xy[i0, 1]
    if abs(dx) + abs(dy) < 1e-8:
        return float("nan")
    return float(math.atan2(dy, dx))


def cluster_1d(values: List[float], eps: float) -> List[List[float]]:
    if not values:
        return []
    vals = sorted(values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= eps:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def dedupe_polylines(polys: List[np.ndarray], tol: float = 0.2) -> List[np.ndarray]:
    kept = []
    seen = set()
    for p in polys:
        if len(p) < 2:
            continue
        key = (
            round(float(p[0, 0]) / tol),
            round(float(p[0, 1]) / tol),
            round(float(p[-1, 0]) / tol),
            round(float(p[-1, 1]) / tol),
            len(p),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return kept


# =========================================================
# Schema-tolerant polyline / relation extraction
# =========================================================

KEY_HINTS = [
    "centerline",
    "center_line",
    "polyline",
    "points",
    "xyz",
    "geometry",
    "lane_centerline",
    "lane_center_line",
    "lane_points",
    "coords",
]

REL_KEYS = {
    "successor",
    "successors",
    "predecessor",
    "predecessors",
    "left",
    "right",
    "adjacent",
    "adjacent_left",
    "adjacent_right",
    "neighbor",
    "neighbour",
    "incoming",
    "outgoing",
    "connect",
    "connection",
    "merge",
    "diverge",
    "branch",
    "fork",
    "split",
    "join",
    "intersection",
}


def maybe_extract_polyline_from_dict(d: Dict[str, Any]) -> List[List[float]]:
    for k, v in d.items():
        kl = k.lower()
        if any(h in kl for h in KEY_HINTS) and is_polyline_like(v):
            return v

    for _, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                kkl = kk.lower()
                if any(h in kkl for h in KEY_HINTS) and is_polyline_like(vv):
                    return vv
    return []


def collect_polylines(obj: Any, out: List[List[List[float]]]) -> None:
    if isinstance(obj, dict):
        poly = maybe_extract_polyline_from_dict(obj)
        if poly:
            out.append(poly)
        for v in obj.values():
            collect_polylines(v, out)
    elif isinstance(obj, list):
        if is_polyline_like(obj):
            out.append(obj)  # type: ignore
        else:
            for it in obj:
                collect_polylines(it, out)


def collect_relation_counts(obj: Any, counts: Dict[str, int]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(r in kl for r in REL_KEYS):
                counts["relation_key_hits"] += 1
                if any(
                    x in kl
                    for x in [
                        "merge",
                        "diverge",
                        "fork",
                        "split",
                        "join",
                        "intersection",
                    ]
                ):
                    counts["event_key_hits"] += 1

                if isinstance(v, list):
                    counts["relation_list_items"] += len(v)
                elif isinstance(v, (int, str, dict)):
                    counts["relation_scalar_or_obj"] += 1

            collect_relation_counts(v, counts)
    elif isinstance(obj, list):
        for it in obj:
            collect_relation_counts(it, counts)


# =========================================================
# Curvature
# =========================================================


def forward_crop(
    xy: np.ndarray, x_min: float, x_max: float, y_abs_max: float
) -> np.ndarray:
    if len(xy) == 0:
        return xy
    m = (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) & (np.abs(xy[:, 1]) <= y_abs_max)
    return xy[m]


def curvature_samples(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(xy) < 4:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    dxy = np.diff(xy, axis=0)
    ds = np.linalg.norm(dxy, axis=1)
    if np.sum(ds > 1e-6) < 3:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    theta = np.arctan2(dxy[:, 1], dxy[:, 0])
    dtheta = wrap_angle_arr(np.diff(theta))
    ds_mid = 0.5 * (ds[:-1] + ds[1:])
    kappa = np.divide(dtheta, ds_mid, out=np.zeros_like(dtheta), where=ds_mid > 1e-6)

    s = np.concatenate(([0.0], np.cumsum(ds)))
    s_mid = 0.5 * (s[1:-1] + s[2:])
    return kappa, s_mid


def polyline_confidence(xy: np.ndarray, kappa: np.ndarray, target_len: float) -> float:
    if len(xy) < 2 or len(kappa) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    length = float(np.sum(seg))
    c_len = clamp(length / target_len, 0.0, 1.0)
    c_pts = clamp(len(kappa) / 20.0, 0.0, 1.0)
    return float(0.6 * c_len + 0.4 * c_pts)


def classify_curvature(k: float, thr: Dict[str, float]) -> str:
    a = abs(k)
    if a < thr["straight"]:
        return "straight"
    direction = "left" if k > 0 else "right"
    if a < thr["low"]:
        return f"low curvature {direction}"
    elif a < thr["medium"]:
        return f"medium curvature {direction}"
    else:
        return f"high curvature {direction}"


def compute_curvature(
    data: Dict[str, Any], args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    raw = []
    collect_polylines(data, raw)

    lanes = []
    for p in raw:
        xy = polyline_to_xy(p)
        if len(xy) >= 2:
            xy = resample_polyline(xy, ds=args.curv_resample_ds)
            if len(xy) >= 4:
                xy = smooth_polyline(xy, window=args.curv_smooth_window)
                lanes.append(xy)

    lanes = dedupe_polylines(lanes)

    lane_stats = []
    for xy in lanes:
        xy_f = forward_crop(
            xy,
            x_min=args.curv_forward_min,
            x_max=args.curv_forward_max,
            y_abs_max=args.curv_lateral_max,
        )
        if len(xy_f) < 4:
            continue

        kappa, _ = curvature_samples(xy_f)
        if len(kappa) < 2:
            continue

        k_med = float(np.median(kappa))
        conf = polyline_confidence(
            xy_f, kappa, target_len=(args.curv_forward_max - args.curv_forward_min)
        )
        y_mid = nearest_y_at_x(
            xy_f, (args.curv_forward_min + args.curv_forward_max) * 0.5
        )

        # Weight by confidence and proximity to ego centerline
        w_center = math.exp(-abs(y_mid) / 3.0)
        weight = max(1e-6, conf * w_center)
        lane_stats.append((k_med, conf, weight))

    if not lane_stats:
        return "curvature_unknown", {
            "value_m_inv": None,
            "confidence": 0.0,
            "method": "weighted_median_lane_curvature",
            "status": "no_valid_lanes",
            "thresholds_m_inv": {
                "straight": args.curv_thr_straight,
                "low": args.curv_thr_low,
                "medium": args.curv_thr_medium,
            },
            "forward_range_m": [args.curv_forward_min, args.curv_forward_max],
            "lateral_limit_m": args.curv_lateral_max,
        }

    ks = np.array([x[0] for x in lane_stats], dtype=np.float64)
    ws = np.array([x[2] for x in lane_stats], dtype=np.float64)
    order = np.argsort(ks)
    ks_s = ks[order]
    ws_s = ws[order]
    cdf = np.cumsum(ws_s) / np.sum(ws_s)
    idx = int(np.searchsorted(cdf, 0.5))
    k_frame = float(ks_s[min(idx, len(ks_s) - 1)])
    conf_frame = float(clamp(np.mean([x[1] for x in lane_stats]), 0.0, 1.0))

    thr = {
        "straight": args.curv_thr_straight,
        "low": args.curv_thr_low,
        "medium": args.curv_thr_medium,
    }
    tag = classify_curvature(k_frame, thr)

    return tag, {
        "value_m_inv": k_frame,
        "confidence": conf_frame,
        "method": "weighted_median_lane_curvature",
        "status": "ok",
        "thresholds_m_inv": thr,
        "forward_range_m": [args.curv_forward_min, args.curv_forward_max],
        "lateral_limit_m": args.curv_lateral_max,
        "num_lanes_used": int(len(lane_stats)),
    }


# =========================================================
# Topology complexity (v2 event-centric)
# =========================================================


def heading_modes(headings: List[float], mode_sep_deg: float) -> int:
    if not headings:
        return 0
    sep = math.radians(mode_sep_deg)
    modes = []
    for h in headings:
        assigned = False
        for m in modes:
            if abs(angle_diff(h, m)) <= sep:
                assigned = True
                break
        if not assigned:
            modes.append(h)
    return len(modes)


def compute_topology_complexity(
    data: Dict[str, Any], args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    raw = []
    collect_polylines(data, raw)

    lanes = []
    for p in raw:
        xy = polyline_to_xy(p)
        if len(xy) >= 2:
            xy = resample_polyline(xy, ds=args.topo_resample_ds)
            if len(xy) >= 2:
                lanes.append(xy)
    lanes = dedupe_polylines(lanes)

    near_y, far_y = [], []
    near_h, far_h = [], []
    heading_gate = math.radians(args.topo_heading_forward_gate_deg)

    for xy in lanes:
        if len(xy) < 3:
            continue
        xmin, xmax = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
        use_near = xmin <= args.topo_x_near <= xmax
        use_far = xmin <= args.topo_x_far <= xmax
        if not use_near and not use_far:
            continue

        if use_near:
            y = nearest_y_at_x(xy, args.topo_x_near)
            h = heading_at_x(xy, args.topo_x_near)
            if np.isfinite(y) and np.isfinite(h) and abs(y) <= args.topo_y_limit:
                if abs(angle_diff(h, 0.0)) <= heading_gate:
                    near_y.append(float(y))
                    near_h.append(float(h))

        if use_far:
            y = nearest_y_at_x(xy, args.topo_x_far)
            h = heading_at_x(xy, args.topo_x_far)
            if np.isfinite(y) and np.isfinite(h) and abs(y) <= args.topo_y_limit:
                if abs(angle_diff(h, 0.0)) <= heading_gate:
                    far_y.append(float(y))
                    far_h.append(float(h))

    near_clusters = cluster_1d(near_y, args.topo_lateral_eps)
    far_clusters = cluster_1d(far_y, args.topo_lateral_eps)
    n_near = len(near_clusters)
    n_far = len(far_clusters)

    fan_out = max(0, n_far - n_near)
    fan_in = max(0, n_near - n_far)

    fan_out_score = clamp(fan_out / 2.0, 0.0, 1.0)
    fan_in_score = clamp(fan_in / 2.0, 0.0, 1.0)
    merge_diverge_score = max(fan_out_score, fan_in_score)

    modes_near = heading_modes(near_h, args.topo_heading_mode_sep_deg)
    modes_far = heading_modes(far_h, args.topo_heading_mode_sep_deg)
    modes = max(modes_near, modes_far)

    if modes <= 1:
        inter_score = 0.0
    elif modes == 2:
        inter_score = 0.55
    else:
        inter_score = 1.0

    all_h = near_h + far_h
    if len(all_h) >= 3:
        h_arr = np.array(all_h, dtype=np.float64)
        mu = math.atan2(np.mean(np.sin(h_arr)), np.mean(np.cos(h_arr)))
        diffs = np.array([angle_diff(h, mu) for h in h_arr], dtype=np.float64)
        hdg_std_deg = float(np.degrees(np.std(diffs)))
    else:
        hdg_std_deg = 0.0

    parallel_penalty = 0.0
    if max(n_near, n_far) >= 4 and hdg_std_deg < 6.0 and merge_diverge_score < 0.25:
        parallel_penalty = 0.35

    geom_raw = max(0.65 * merge_diverge_score + 0.35 * inter_score, inter_score)
    geom_score = clamp(geom_raw - parallel_penalty, 0.0, 1.0)
    geom_conf = clamp(max(len(near_y), len(far_y)) / 8.0, 0.0, 1.0)

    rel_counts = {
        "relation_key_hits": 0,
        "relation_list_items": 0,
        "relation_scalar_or_obj": 0,
        "event_key_hits": 0,
    }
    collect_relation_counts(data, rel_counts)

    event_term = clamp(rel_counts["event_key_hits"] / 6.0, 0.0, 1.0)
    generic_term = clamp(
        (
            0.25 * rel_counts["relation_key_hits"]
            + 0.15 * rel_counts["relation_list_items"]
            + 0.1 * rel_counts["relation_scalar_or_obj"]
        )
        / 20.0,
        0.0,
        1.0,
    )
    rel_score = clamp(0.75 * event_term + 0.25 * generic_term, 0.0, 1.0)
    rel_conf = clamp(
        (rel_counts["event_key_hits"] + 0.3 * rel_counts["relation_key_hits"]) / 12.0,
        0.0,
        1.0,
    )

    has_geom = (len(near_y) + len(far_y)) > 0
    has_rel = rel_counts["relation_key_hits"] > 0

    if not has_geom and not has_rel:
        return "topology_unknown", {
            "value": None,
            "confidence": 0.0,
            "method": "event_centric_geometry_plus_relation_event",
            "status": "insufficient_signals",
            "geometry": {},
            "relations": rel_counts,
            "policy": {
                "low": "parallel/simple",
                "medium": "merge/diverge/exits",
                "high": "intersection/multi-branch",
            },
        }

    if has_geom and has_rel:
        score = 0.88 * geom_score + 0.12 * rel_score
        conf = 0.85 * geom_conf + 0.15 * rel_conf
    elif has_geom:
        score = geom_score
        conf = geom_conf
    else:
        score = rel_score
        conf = rel_conf

    score = float(clamp(score, 0.0, 1.0))
    conf = float(clamp(conf, 0.0, 1.0))

    # Policy mapping per your request
    if inter_score >= 0.9:
        tag = "high topological complexity"
    elif score >= 0.72:
        tag = "high topological complexity"
    elif merge_diverge_score >= 0.35:
        tag = "medium topological complexity"
    elif score >= 0.38:
        tag = "medium topological complexity"
    else:
        tag = "low topological complexity"

    meta = {
        "value": score,
        "confidence": conf,
        "method": "event_centric_geometry_plus_relation_event",
        "status": "ok",
        "geometry": {
            "near_cluster_count": n_near,
            "far_cluster_count": n_far,
            "fan_out": int(fan_out),
            "fan_in": int(fan_in),
            "merge_diverge_score": float(merge_diverge_score),
            "heading_modes_near": int(modes_near),
            "heading_modes_far": int(modes_far),
            "heading_modes_max": int(modes),
            "intersection_like_score": float(inter_score),
            "heading_std_deg": float(hdg_std_deg),
            "parallel_lane_penalty": float(parallel_penalty),
            "geometry_event_score": float(geom_score),
            "num_near_candidates": int(len(near_y)),
            "num_far_candidates": int(len(far_y)),
        },
        "relations": {
            **rel_counts,
            "relation_event_score": float(rel_score),
        },
        "policy": {
            "low": "parallel/simple",
            "medium": "merge/diverge/exits",
            "high": "intersection/multi-branch",
        },
        "params": {
            "x_near_m": args.topo_x_near,
            "x_far_m": args.topo_x_far,
            "y_limit_m": args.topo_y_limit,
            "heading_forward_gate_deg": args.topo_heading_forward_gate_deg,
            "lateral_cluster_eps_m": args.topo_lateral_eps,
            "heading_mode_sep_deg": args.topo_heading_mode_sep_deg,
        },
    }
    return tag, meta


# =========================================================
# Lighting
# =========================================================


def compute_lighting(
    image_path: Path, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    if cv2 is None:
        return "lighting_unknown", {
            "label": "lighting_unknown",
            "score": None,
            "confidence": 0.0,
            "method": "composite_luminance_contrast_darktail_v1",
            "status": "opencv_not_available",
            "image_path": str(image_path),
        }

    if not image_path.exists():
        return "lighting_unknown", {
            "label": "lighting_unknown",
            "score": None,
            "confidence": 0.0,
            "method": "composite_luminance_contrast_darktail_v1",
            "status": "image_missing",
            "image_path": str(image_path),
        }

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return "lighting_unknown", {
            "label": "lighting_unknown",
            "score": None,
            "confidence": 0.0,
            "method": "composite_luminance_contrast_darktail_v1",
            "status": "image_read_failed",
            "image_path": str(image_path),
        }

    # Luminance proxy: Lab L channel [0..255]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0

    mu = float(np.mean(L))
    p10 = float(np.percentile(L, 10))
    p90 = float(np.percentile(L, 90))
    spread = float(p90 - p10)
    dark_ratio = float(np.mean(L < args.light_dark_thresh))
    sat_low = float(np.mean(L < args.light_sat_low_thresh))

    # Composite low-light score
    score = (
        0.40 * (1.0 - mu)
        + 0.25 * (1.0 - p10)
        + 0.20 * dark_ratio
        + 0.15 * (1.0 - spread)
    )
    score = float(clamp(score, 0.0, 1.0))

    label = "poorly lit" if score >= args.light_score_thresh else "well lit"
    confidence = float(
        clamp(abs(score - args.light_score_thresh) / args.light_conf_band, 0.0, 1.0)
    )

    return label, {
        "label": label,
        "score": score,
        "confidence": confidence,
        "method": "composite_luminance_contrast_darktail_v1",
        "status": "ok",
        "image_path": str(image_path),
        "features": {
            "mean_luminance": mu,
            "p10_luminance": p10,
            "p90_luminance": p90,
            "contrast_spread": spread,
            "dark_ratio": dark_ratio,
            "saturated_dark_ratio": sat_low,
            "dark_thresh": args.light_dark_thresh,
            "sat_low_thresh": args.light_sat_low_thresh,
        },
        "thresholds": {
            "score_thresh_poorly_lit": args.light_score_thresh,
            "confidence_band": args.light_conf_band,
        },
    }


def resolve_image_path(json_path: Path, camera_name: str, ext: str) -> Path:
    """
    Expects JSON path like:
      .../<seq>/info/<stem>.json
    Image path:
      .../<seq>/image/<camera_name>/<stem>.<ext>
    """
    stem = json_path.stem
    seq_dir = json_path.parent.parent  # info -> seq
    return seq_dir / "image" / camera_name / f"{stem}.{ext}"


def resolve_frame_image_path(
    data: Dict[str, Any], json_path: Path, camera_name: str, ext: str
) -> Path:
    sensor = data.get("sensor")
    if isinstance(sensor, dict):
        camera = sensor.get(camera_name)
        if isinstance(camera, dict):
            image_path = camera.get("image_path")
            if isinstance(image_path, str) and image_path.strip():
                dataset_root = json_path.parent.parent.parent.parent
                return dataset_root / Path(image_path)
    return resolve_image_path(json_path, camera_name, ext)


# =========================================================
# Occlusion
# =========================================================

# YOLO COCO classes considered vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

K_FIXED = np.array(
    [
        [1.77753967e03, 0.00000000e00, 7.77762878e02],
        [0.00000000e00, 1.77753967e03, 1.01631311e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float64,
)

DIST_FIXED = np.array([-0.24479243, -0.19577468, 0.30131747], dtype=np.float64)

R_FIXED = np.array(
    [
        [-8.48589870e-04, 1.00005773e-02, 9.99949633e-01],
        [-9.99998983e-01, -1.15429954e-03, -8.37087508e-04],
        [1.14587004e-03, -9.99949327e-01, 1.00015467e-02],
    ],
    dtype=np.float64,
)

T_FIXED = np.array([1.63315125, 0.00800013, 1.38385219], dtype=np.float64)


def extract_camera_params(
    data: Dict[str, Any], camera_name: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    sensor = data.get("sensor")
    if not isinstance(sensor, dict):
        return None

    camera = sensor.get(camera_name)
    if not isinstance(camera, dict):
        return None

    intrinsic = camera.get("intrinsic")
    extrinsic = camera.get("extrinsic")
    if not isinstance(intrinsic, dict) or not isinstance(extrinsic, dict):
        return None

    try:
        K = np.array(intrinsic.get("K"), dtype=np.float64)
        distortion = np.array(intrinsic.get("distortion"), dtype=np.float64).reshape(-1)
        R = np.array(extrinsic.get("rotation"), dtype=np.float64)
        t = np.array(extrinsic.get("translation"), dtype=np.float64).reshape(-1)
    except Exception:
        return None

    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape[0] != 3:
        return None

    if distortion.size < 3:
        return None

    return K, distortion[:3], R, t[:3]


def collect_lane_arrays(data: Dict[str, Any]) -> List[np.ndarray]:
    raw: List[List[List[float]]] = []
    collect_polylines(data, raw)
    lanes = []
    for p in raw:
        arr = np.array(p, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            lanes.append(arr[:, :3] if arr.shape[1] >= 3 else arr[:, :2])
    return lanes


def project_lane_xyz_to_image_uv(
    lane_xyz: np.ndarray,
    K: np.ndarray,
    dist3: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    if lane_xyz.ndim != 2 or lane_xyz.shape[1] < 3 or lane_xyz.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xyz = lane_xyz[:, :3].astype(np.float64)
    uv = np.full((xyz.shape[0], 2), np.nan, dtype=np.float64)

    cam = np.linalg.pinv(R) @ (xyz.T - t.reshape(3, 1))
    valid = cam[2, :] > 1e-6
    if not np.any(valid):
        return uv

    points_on_image = K @ cam[:, valid]
    points_on_image /= points_on_image[2:3, :]
    uv[valid] = points_on_image[:2, :].T
    return uv


def project_3d_lanes_to_image(
    lanes_3d: List[np.ndarray],
    img_w: int,
    img_h: int,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[np.ndarray]:
    if camera_params is None:
        camera_params = (K_FIXED, DIST_FIXED, R_FIXED, T_FIXED)

    K, dist3, R, t = camera_params
    out = []
    for lane in lanes_3d:
        if lane.ndim != 2 or lane.shape[0] < 2 or lane.shape[1] < 3:
            continue
        uv = project_lane_xyz_to_image_uv(lane, K, dist3, R, t)
        mask = (
            np.isfinite(uv[:, 0])
            & np.isfinite(uv[:, 1])
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < img_w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < img_h)
        )
        pts = uv[mask]
        if pts.shape[0] >= 2:
            out.append(pts)
    return out


def resample_polyline_px(xy: np.ndarray, ds_px: float) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xy = remove_duplicate_points(xy)
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total < ds_px:
        return xy.copy()

    s_new = np.arange(0.0, total + 1e-9, ds_px)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.stack([x_new, y_new], axis=1)


def detect_vehicle_boxes(
    model: Any,
    image_bgr: np.ndarray,
    conf_thr: float,
    imgsz: int,
    device: str,
) -> List[Tuple[float, float, float, float, float, int]]:
    if model is None:
        return []

    results = model.predict(
        source=image_bgr, conf=conf_thr, imgsz=imgsz, device=device, verbose=False
    )
    out: List[Tuple[float, float, float, float, float, int]] = []
    if not results:
        return out

    boxes = results[0].boxes
    if boxes is None:
        return out

    xyxy = (
        boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    )
    conf = (
        boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    )
    cls = (
        boxes.cls.cpu().numpy().astype(int)
        if hasattr(boxes.cls, "cpu")
        else np.array(boxes.cls).astype(int)
    )

    for b, s, c in zip(xyxy, conf, cls):
        if int(c) not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = map(float, b)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out.append((x1, y1, x2, y2, float(s), int(c)))
    return out


def dilate_boxes(
    boxes: List[Tuple[float, float, float, float, float, int]],
    dilate_px: float,
    w: int,
    h: int,
) -> List[Tuple[float, float, float, float, float, int]]:
    out = []
    for x1, y1, x2, y2, s, c in boxes:
        out.append(
            (
                max(0.0, x1 - dilate_px),
                max(0.0, y1 - dilate_px),
                min(float(w - 1), x2 + dilate_px),
                min(float(h - 1), y2 + dilate_px),
                s,
                c,
            )
        )
    return out


def point_in_any_box(
    u: float, v: float, boxes: List[Tuple[float, float, float, float, float, int]]
) -> bool:
    for x1, y1, x2, y2, _, _ in boxes:
        if x1 <= u <= x2 and y1 <= v <= y2:
            return True
    return False


def compute_occlusion(
    lane_polys_img: List[np.ndarray],
    boxes: List[Tuple[float, float, float, float, float, int]],
    image_w: int,
    image_h: int,
    sample_ds_px: float,
    min_lane_length_px: float,
) -> Dict[str, Any]:
    total_samples = 0
    occ_samples = 0
    lanes_used = 0

    for xy in lane_polys_img:
        if len(xy) < 2:
            continue
        mask = (
            np.isfinite(xy[:, 0])
            & np.isfinite(xy[:, 1])
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < image_w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < image_h)
        )
        pts = xy[mask]
        if len(pts) < 2:
            continue

        vis_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if vis_len < min_lane_length_px:
            continue

        samples = resample_polyline_px(pts, ds_px=sample_ds_px)
        if len(samples) == 0:
            continue

        lanes_used += 1
        total_samples += len(samples)
        for u, v in samples:
            if point_in_any_box(float(u), float(v), boxes):
                occ_samples += 1

    ratio = None if total_samples == 0 else float(occ_samples / total_samples)
    return {
        "ratio": ratio,
        "total_samples": int(total_samples),
        "occluded_samples": int(occ_samples),
        "lanes_used": int(lanes_used),
    }


def compute_occlusion_for_frame(
    data: Dict[str, Any],
    image_path: Path,
    args: argparse.Namespace,
    model: Any,
) -> Tuple[str, Dict[str, Any]]:
    if cv2 is None:
        return "occlusion_unknown", {
            "value": None,
            "label": "occlusion_unknown",
            "threshold": args.occlusion_threshold,
            "confidence": 0.0,
            "method": "front_image_detector_lane_overlap",
            "status": "opencv_not_available",
            "debug": {"image_path": str(image_path)},
        }

    if model is None:
        return "occlusion_unknown", {
            "value": None,
            "label": "occlusion_unknown",
            "threshold": args.occlusion_threshold,
            "confidence": 0.0,
            "method": "front_image_detector_lane_overlap",
            "status": "detector_unavailable",
            "debug": {"image_path": str(image_path), "detector_model": args.det_model},
        }

    if not image_path.exists():
        return "occlusion_unknown", {
            "value": None,
            "label": "occlusion_unknown",
            "threshold": args.occlusion_threshold,
            "confidence": 0.0,
            "method": "front_image_detector_lane_overlap",
            "status": "image_missing",
            "debug": {"image_path": str(image_path)},
        }

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(image)
        return "occlusion_unknown", {
            "value": None,
            "label": "occlusion_unknown",
            "threshold": args.occlusion_threshold,
            "confidence": 0.0,
            "method": "front_image_detector_lane_overlap",
            "status": "image_read_failed",
            "debug": {"image_path": str(image_path)},
        }

    h, w = image.shape[:2]
    raw_lanes = dedupe_polylines(collect_lane_arrays(data), tol=0.5)

    lanes_3d = [
        lane[:, :3]
        for lane in raw_lanes
        if lane.ndim == 2 and lane.shape[1] >= 3 and lane.shape[0] >= 2
    ]
    lanes_2d = [
        lane[:, :2]
        for lane in raw_lanes
        if lane.ndim == 2 and lane.shape[1] == 2 and lane.shape[0] >= 2
    ]

    camera_params = extract_camera_params(data, args.camera_name)
    lane_polys_img = project_3d_lanes_to_image(
        lanes_3d, w, h, camera_params=camera_params
    )
    lane_source = (
        "json_3d_projected_sensor_calib"
        if camera_params is not None
        else "json_3d_projected_fixed_calib"
    )
    if len(lane_polys_img) == 0:
        lane_source = "json_2d_direct_fallback"
        filtered = []
        for xy in lanes_2d:
            mask = (
                np.isfinite(xy[:, 0])
                & np.isfinite(xy[:, 1])
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < w)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < h)
            )
            pts = xy[mask]
            if pts.shape[0] >= 2:
                filtered.append(pts)
        lane_polys_img = filtered

    boxes = detect_vehicle_boxes(
        model=model,
        image_bgr=image,
        conf_thr=args.bbox_score_thr,
        imgsz=args.det_imgsz,
        device=args.device,
    )
    boxes = dilate_boxes(boxes, args.bbox_dilate_px, w, h)

    occ_stats = compute_occlusion(
        lane_polys_img=lane_polys_img,
        boxes=boxes,
        image_w=w,
        image_h=h,
        sample_ds_px=args.sample_ds_px,
        min_lane_length_px=args.min_lane_length_px,
    )

    if (
        occ_stats["ratio"] is None
        or occ_stats["total_samples"] < args.min_total_samples
    ):
        tag, status = "low occlusion", "insufficient_lane_samples_forced_low"
    elif len(boxes) == 0:
        tag, status = "low occlusion", "no_vehicle_boxes"
    else:
        tag = (
            "high occlusion"
            if occ_stats["ratio"] >= args.occlusion_threshold
            else "low occlusion"
        )
        status = "ok"

    if tag == "occlusion_unknown":
        confidence = 0.0
    else:
        conf_sample = clamp(
            occ_stats["total_samples"] / float(max(1, args.min_total_samples * 3)),
            0.0,
            1.0,
        )
        conf_det = clamp(
            float(np.mean([b[4] for b in boxes])) if len(boxes) else 0.0, 0.0, 1.0
        )
        confidence = float(clamp(0.7 * conf_sample + 0.3 * conf_det, 0.0, 1.0))

    meta = {
        "value": occ_stats["ratio"],
        "label": tag,
        "threshold": args.occlusion_threshold,
        "confidence": confidence,
        "method": "front_image_detector_lane_overlap",
        "status": status,
        "debug": {
            "image_path": str(image_path),
            "image_size": {"width": int(w), "height": int(h)},
            "vehicle_boxes_used": int(len(boxes)),
            "total_samples": int(occ_stats["total_samples"]),
            "occluded_samples": int(occ_stats["occluded_samples"]),
            "lanes_used": int(occ_stats["lanes_used"]),
            "num_raw_lanes": len(raw_lanes),
            "num_3d_lanes": len(lanes_3d),
            "num_2d_lanes": len(lanes_2d),
            "lane_source": lane_source,
            "detector_model": args.det_model,
            "detector_imgsz": int(args.det_imgsz),
            "bbox_score_thr": float(args.bbox_score_thr),
            "bbox_dilate_px": float(args.bbox_dilate_px),
            "sample_ds_px": float(args.sample_ds_px),
            "min_lane_length_px": float(args.min_lane_length_px),
            "min_total_samples": int(args.min_total_samples),
        },
    }
    return tag, meta


# =========================================================
# JSON update helpers
# =========================================================

CURVATURE_TAGS = {
    "curvature_unknown",
    "straight",
    "low curvature left",
    "low curvature right",
    "medium curvature left",
    "medium curvature right",
    "high curvature left",
    "high curvature right",
}

TOPOLOGY_TAGS = {
    "topology_unknown",
    "low topological complexity",
    "medium topological complexity",
    "high topological complexity",
}

LIGHTING_TAGS = {
    "well lit",
    "poorly lit",
    "lighting_unknown",
}

OCCLUSION_TAGS = {
    "low occlusion",
    "high occlusion",
    "occlusion_unknown",
}


def upsert_tag_family(tags: List[Any], new_tag: str, family_set: set) -> List[Any]:
    kept = []
    for t in tags:
        if isinstance(t, str) and t.strip().lower() in family_set:
            continue
        kept.append(t)
    kept.append(new_tag)

    # de-dup preserve order
    out = []
    seen = set()
    for t in kept:
        key = t.strip().lower() if isinstance(t, str) else str(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def update_json_fields(
    data: Dict[str, Any],
    curvature_tag: str,
    curvature_meta: Dict[str, Any],
    topo_tag: str,
    topo_meta: Dict[str, Any],
    lighting_tag: str,
    lighting_meta: Dict[str, Any],
    occlusion_tag: str,
    occlusion_meta: Dict[str, Any],
) -> None:
    if "scenario_tags" not in data or not isinstance(data["scenario_tags"], list):
        data["scenario_tags"] = []

    tags = data["scenario_tags"]
    tags = upsert_tag_family(tags, curvature_tag, CURVATURE_TAGS)
    tags = upsert_tag_family(tags, topo_tag, TOPOLOGY_TAGS)
    tags = upsert_tag_family(tags, lighting_tag, LIGHTING_TAGS)
    tags = upsert_tag_family(tags, occlusion_tag, OCCLUSION_TAGS)
    data["scenario_tags"] = tags

    if "scenario_meta" not in data or not isinstance(data["scenario_meta"], dict):
        data["scenario_meta"] = {}

    data["scenario_meta"]["curvature"] = curvature_meta
    data["scenario_meta"]["topology_complexity"] = topo_meta
    data["scenario_meta"]["lighting"] = lighting_meta
    data["scenario_meta"]["occlusion"] = occlusion_meta


# =========================================================
# Batch processing
# =========================================================


def find_json_files(root: Path, info_only: bool) -> List[Path]:
    if info_only:
        return sorted([p for p in root.rglob("info/*.json") if p.is_file()])
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def process_file(
    jpath: Path, args: argparse.Namespace, detector_model: Any
) -> Tuple[bool, Dict[str, str]]:
    try:
        with jpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, {"error": f"read_error: {e}"}

    if not isinstance(data, dict):
        return False, {"error": "root_not_dict"}

    try:
        curv_tag, curv_meta = compute_curvature(data, args)
        topo_tag, topo_meta = compute_topology_complexity(data, args)

        img_path = resolve_frame_image_path(
            data, jpath, camera_name=args.camera_name, ext=args.image_ext
        )
        light_tag, light_meta = compute_lighting(img_path, args)
        occ_tag, occ_meta = compute_occlusion_for_frame(
            data, img_path, args, detector_model
        )

        update_json_fields(
            data=data,
            curvature_tag=curv_tag,
            curvature_meta=curv_meta,
            topo_tag=topo_tag,
            topo_meta=topo_meta,
            lighting_tag=light_tag,
            lighting_meta=light_meta,
            occlusion_tag=occ_tag,
            occlusion_meta=occ_meta,
        )

        if not args.dry_run:
            with jpath.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return True, {
            "curvature": curv_tag,
            "topology": topo_tag,
            "lighting": light_tag,
            "occlusion": occ_tag,
        }
    except Exception as e:
        return False, {"error": f"proc_error: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--all_json",
        action="store_true",
        help="Process all *.json files under dataset_root (default is info/*.json only).",
    )

    # image resolution
    parser.add_argument("--camera_name", type=str, default="ring_front_center")
    parser.add_argument("--image_ext", type=str, default="jpg")

    # curvature params
    parser.add_argument("--curv_forward_min", type=float, default=5.0)
    parser.add_argument("--curv_forward_max", type=float, default=40.0)
    parser.add_argument("--curv_lateral_max", type=float, default=20.0)
    parser.add_argument("--curv_resample_ds", type=float, default=0.5)
    parser.add_argument("--curv_smooth_window", type=int, default=5)
    parser.add_argument("--curv_thr_straight", type=float, default=0.003)
    parser.add_argument("--curv_thr_low", type=float, default=0.008)
    parser.add_argument("--curv_thr_medium", type=float, default=0.020)

    # topology params
    parser.add_argument("--topo_resample_ds", type=float, default=0.5)
    parser.add_argument("--topo_x_near", type=float, default=12.0)
    parser.add_argument("--topo_x_far", type=float, default=35.0)
    parser.add_argument("--topo_y_limit", type=float, default=30.0)
    parser.add_argument("--topo_heading_forward_gate_deg", type=float, default=65.0)
    parser.add_argument("--topo_lateral_eps", type=float, default=1.8)
    parser.add_argument("--topo_heading_mode_sep_deg", type=float, default=35.0)

    # lighting params
    parser.add_argument("--light_dark_thresh", type=float, default=0.16)
    parser.add_argument("--light_sat_low_thresh", type=float, default=0.04)
    parser.add_argument("--light_score_thresh", type=float, default=0.55)
    parser.add_argument("--light_conf_band", type=float, default=0.20)

    # occlusion / detector params
    parser.add_argument("--det_model", type=str, default="yolov8n.pt")
    parser.add_argument("--det_imgsz", type=int, default=960)
    parser.add_argument("--bbox_score_thr", type=float, default=0.25)
    parser.add_argument("--bbox_dilate_px", type=float, default=2.0)
    parser.add_argument(
        "--device", type=str, default="mps", help="mps (Apple Silicon), cpu, or cuda"
    )
    parser.add_argument("--sample_ds_px", type=float, default=3.0)
    parser.add_argument("--min_lane_length_px", type=float, default=30.0)
    parser.add_argument("--min_total_samples", type=int, default=80)
    parser.add_argument("--occlusion_threshold", type=float, default=0.20)

    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid dataset_root: {root}")

    files = find_json_files(root, info_only=(not args.all_json))
    print(f"[INFO] Found {len(files)} JSON files under: {root}")
    if cv2 is None:
        print(
            "[WARN] OpenCV (cv2) is not installed. Lighting tags will become 'lighting_unknown'."
        )
    if YOLO is None:
        print(
            "[WARN] Ultralytics is not installed. Occlusion tags will become 'occlusion_unknown'."
        )

    detector_model = None
    if YOLO is not None:
        try:
            detector_model = YOLO(args.det_model)
        except Exception as e:
            print(
                f"[WARN] Failed to load detector '{args.det_model}': {e}. Occlusion will be unknown."
            )

    ok, fail = 0, 0
    curv_counts: Dict[str, int] = {}
    topo_counts: Dict[str, int] = {}
    light_counts: Dict[str, int] = {}
    occ_counts: Dict[str, int] = {}

    for i, fp in enumerate(files, 1):
        success, result = process_file(fp, args, detector_model)
        if success:
            ok += 1
            c = result["curvature"]
            t = result["topology"]
            l = result["lighting"]
            o = result["occlusion"]

            unknowns = []
            if isinstance(c, str) and "unknown" in c.lower():
                unknowns.append(f"curvature={c}")
            if isinstance(t, str) and "unknown" in t.lower():
                unknowns.append(f"topology={t}")
            if isinstance(l, str) and "unknown" in l.lower():
                unknowns.append(f"lighting={l}")
            if isinstance(o, str) and "unknown" in o.lower():
                unknowns.append(f"occlusion={o}")
            if unknowns:
                print(f"[WARN] {fp}: unknown tags -> {', '.join(unknowns)}")

            curv_counts[c] = curv_counts.get(c, 0) + 1
            topo_counts[t] = topo_counts.get(t, 0) + 1
            light_counts[l] = light_counts.get(l, 0) + 1
            occ_counts[o] = occ_counts.get(o, 0) + 1
        else:
            fail += 1
            print(f"[WARN] {fp}: {result.get('error', 'unknown_error')}")

        if i % 200 == 0 or i == len(files):
            print(f"[PROGRESS] {i}/{len(files)} | ok={ok} fail={fail}")

    print("\n=== Summary ===")
    print(f"Total files: {len(files)}")
    print(f"Success:     {ok}")
    print(f"Failed:      {fail}")

    print("\nCurvature tags:")
    for k, v in sorted(curv_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    print("\nTopology tags:")
    for k, v in sorted(topo_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    print("\nLighting tags:")
    for k, v in sorted(light_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    print("\nOcclusion tags:")
    for k, v in sorted(occ_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    if args.dry_run:
        print("\n[NOTE] Dry run enabled: no files were modified.")


if __name__ == "__main__":
    main()
