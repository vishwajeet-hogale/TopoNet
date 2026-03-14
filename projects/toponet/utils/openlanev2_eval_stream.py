import numpy as np

from openlanev2.evaluation.distance import (
    chamfer_distance,
    frechet_distance,
    iou_distance,
    pairwise,
)
from openlanev2.evaluation.f_score import f1
from openlanev2.preprocessing import check_results
from openlanev2.utils import TRAFFIC_ELEMENT_ATTRIBUTE


THRESHOLDS_FRECHET = [1.0, 2.0, 3.0]
THRESHOLDS_IOU = [0.75]
THRESHOLD_RELATIONSHIP_CONFIDENCE = 0.5


def _pr_curve(recalls, precisions):
    recalls = np.asarray(recalls)[np.newaxis, :]
    precisions = np.asarray(precisions)[np.newaxis, :]

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    for i in range(num_scales):
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[i, recalls[i, :] >= thr]
            ap[i] += precs.max() if precs.size > 0 else 0
    ap /= 11
    return ap[0]


def _tpfp(gts, preds, confidences, distance_matrix, distance_threshold):
    assert len(preds) == len(confidences)

    num_gts = len(gts)
    num_preds = len(preds)

    tp = np.zeros((num_preds,), dtype=np.float32)
    fp = np.zeros((num_preds,), dtype=np.float32)
    idx_match_gt = np.ones((num_preds,), dtype=float) * np.nan

    if num_gts == 0:
        fp[...] = 1
        return tp, fp, idx_match_gt
    if num_preds == 0:
        return tp, fp, idx_match_gt

    dist_min = distance_matrix.min(0)
    dist_idx = distance_matrix.argmin(0)

    confidences_idx = np.argsort(-np.asarray(confidences))
    gt_covered = np.zeros(num_gts, dtype=bool)
    for pred_index in confidences_idx:
        if dist_min[pred_index] < distance_threshold:
            matched_gt = dist_idx[pred_index]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[pred_index] = 1
                idx_match_gt[pred_index] = matched_gt
            else:
                fp[pred_index] = 1
        else:
            fp[pred_index] = 1

    return tp, fp, idx_match_gt


def _inject(num_gt, pred, tp, idx_match_gt, confidence, distance_threshold, object_type):
    if tp.tolist() == []:
        pred[f"{object_type}_{distance_threshold}_idx_match_gt"] = []
        pred[f"{object_type}_{distance_threshold}_confidence"] = []
        pred[f"{object_type}_{distance_threshold}_confidence_thresholds"] = []
        return

    confidence = np.asarray(confidence)
    sorted_idx = np.argsort(-confidence)
    sorted_confidence = confidence[sorted_idx]
    tp_sorted = tp[sorted_idx]
    tps = np.cumsum(tp_sorted, axis=0)
    eps = np.finfo(np.float32).eps
    recalls = tps / np.maximum(num_gt, eps)
    try:
        taken = np.percentile(recalls, np.arange(10, 101, 10), method="closest_observation")
    except TypeError:
        # NumPy < 1.22 uses "interpolation" instead of "method".
        taken = np.percentile(recalls, np.arange(10, 101, 10), interpolation="nearest")
    taken_idx = {recall: i for i, recall in enumerate(recalls)}
    confidence_thresholds = sorted_confidence[np.asarray([taken_idx[t] for t in taken])]

    pred[f"{object_type}_{distance_threshold}_idx_match_gt"] = idx_match_gt
    pred[f"{object_type}_{distance_threshold}_confidence"] = confidence
    pred[f"{object_type}_{distance_threshold}_confidence_thresholds"] = confidence_thresholds


def _average_precision_per_vertex(gts, preds, confidences):
    assert len(np.unique(preds)) == len(preds) == len(confidences)

    num_gts = len(gts)
    num_preds = len(preds)
    tp = np.zeros((num_preds,), dtype=np.float32)
    fp = np.zeros((num_preds,), dtype=np.float32)

    if num_gts == num_preds == 0:
        return np.float32(1)
    if num_gts == 0 or num_preds == 0:
        return np.float32(0)

    gts = set(gts)
    confidences_idx = np.argsort(-confidences)
    preds = preds[confidences_idx]
    for pred_index, pred in enumerate(preds):
        if pred in gts:
            tp[pred_index] = 1
        else:
            fp[pred_index] = 1

    rel = tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    eps = np.finfo(np.float32).eps
    precisions = tp / np.maximum((tp + fp), eps)
    return np.dot(precisions, rel) / num_gts


def _ap_directed(gts, preds):
    assert gts.shape[0] == gts.shape[1] == preds.shape[0] == preds.shape[1]

    indices = np.arange(gts.shape[0])
    acc = []
    for gt, pred in zip(gts, preds):
        gt = indices[gt.astype(bool)]
        confidence = pred[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        pred = indices[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        acc.append(_average_precision_per_vertex(gt, pred, confidence))
    for gt, pred in zip(gts.T, preds.T):
        gt = indices[gt.astype(bool)]
        confidence = pred[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        pred = indices[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        acc.append(_average_precision_per_vertex(gt, pred, confidence))
    return acc


def _ap_undirected(gts, preds):
    assert gts.shape[0] == preds.shape[0] and gts.shape[1] == preds.shape[1]

    acc = []
    indices = np.arange(gts.shape[1])
    for gt, pred in zip(gts, preds):
        gt = indices[gt.astype(bool)]
        confidence = pred[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        pred = indices[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        acc.append(_average_precision_per_vertex(gt, pred, confidence))

    indices = np.arange(gts.shape[0])
    for gt, pred in zip(gts.T, preds.T):
        gt = indices[gt.astype(bool)]
        confidence = pred[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        pred = indices[pred > THRESHOLD_RELATIONSHIP_CONFIDENCE]
        acc.append(_average_precision_per_vertex(gt, pred, confidence))
    return acc


def _mAP_topology_lclc(gts, preds, distance_thresholds):
    acc = []
    for distance_threshold in distance_thresholds:
        for token in gts.keys():
            preds_topology_lclc_unmatched = preds[token]["topology_lclc"]
            idx_match_gt = preds[token][f"lane_centerline_{distance_threshold}_idx_match_gt"]
            gt_pred = {m: i for i, m in enumerate(idx_match_gt) if not np.isnan(m)}

            gts_topology_lclc = gts[token]["topology_lclc"]
            if 0 in gts_topology_lclc.shape:
                continue

            gt_indices = np.array(list(gt_pred.keys())).astype(int)
            pred_indices = np.array(list(gt_pred.values())).astype(int)
            preds_topology_lclc = np.ones_like(gts_topology_lclc, dtype=gts_topology_lclc.dtype) * np.nan
            xs = gt_indices[:, None].repeat(len(gt_indices), 1)
            ys = gt_indices[None, :].repeat(len(gt_indices), 0)
            preds_topology_lclc[xs, ys] = preds_topology_lclc_unmatched[pred_indices][:, pred_indices]
            preds_topology_lclc[np.isnan(preds_topology_lclc)] = (
                1 - gts_topology_lclc[np.isnan(preds_topology_lclc)]
            ) * (0.5 + np.finfo(np.float32).eps)

            acc.append(_ap_directed(gts=gts_topology_lclc, preds=preds_topology_lclc))

    if len(acc) == 0:
        return np.float32(0)
    return np.hstack(acc).mean()


def _mAP_topology_lcte(gts, preds, distance_thresholds):
    acc = []
    for distance_threshold_lane_centerline in distance_thresholds["lane_centerline"]:
        for distance_threshold_traffic_element in distance_thresholds["traffic_element"]:
            for token in gts.keys():
                preds_topology_lcte_unmatched = preds[token]["topology_lcte"]

                idx_match_gt_lane_centerline = preds[token][
                    f"lane_centerline_{distance_threshold_lane_centerline}_idx_match_gt"
                ]
                gt_pred_lane_centerline = {
                    m: i for i, m in enumerate(idx_match_gt_lane_centerline) if not np.isnan(m)
                }

                idx_match_gt_traffic_element = preds[token][
                    f"traffic_element_{distance_threshold_traffic_element}_idx_match_gt"
                ]
                gt_pred_traffic_element = {
                    m: i for i, m in enumerate(idx_match_gt_traffic_element) if not np.isnan(m)
                }

                gts_topology_lcte = gts[token]["topology_lcte"]
                if 0 in gts_topology_lcte.shape:
                    continue

                gt_indices_lc = np.array(list(gt_pred_lane_centerline.keys())).astype(int)
                pred_indices_lc = np.array(list(gt_pred_lane_centerline.values())).astype(int)
                gt_indices_te = np.array(list(gt_pred_traffic_element.keys())).astype(int)
                pred_indices_te = np.array(list(gt_pred_traffic_element.values())).astype(int)

                preds_topology_lcte = np.ones_like(gts_topology_lcte, dtype=gts_topology_lcte.dtype) * np.nan
                xs = gt_indices_lc[:, None].repeat(len(gt_indices_te), 1)
                ys = gt_indices_te[None, :].repeat(len(gt_indices_lc), 0)
                preds_topology_lcte[xs, ys] = preds_topology_lcte_unmatched[pred_indices_lc][:, pred_indices_te]
                preds_topology_lcte[np.isnan(preds_topology_lcte)] = (
                    1 - gts_topology_lcte[np.isnan(preds_topology_lcte)]
                ) * (0.5 + np.finfo(np.float32).eps)

                acc.append(_ap_undirected(gts=gts_topology_lcte, preds=preds_topology_lcte))

    if len(acc) == 0:
        return np.float32(0)
    return np.hstack(acc).mean()


def _finalize_ap(collected_tps, collected_fps, collected_confidences, num_gts):
    if not collected_tps:
        return np.float32(1 if num_gts == 0 else 0)

    confidences = np.hstack(collected_confidences)
    sorted_idx = np.argsort(-confidences)
    tps = np.hstack(collected_tps)[sorted_idx]
    fps = np.hstack(collected_fps)[sorted_idx]

    if len(tps) == num_gts == 0:
        return np.float32(1)

    tps = np.cumsum(tps, axis=0)
    fps = np.cumsum(fps, axis=0)
    eps = np.finfo(np.float32).eps
    recalls = tps / np.maximum(num_gts, eps)
    precisions = tps / np.maximum((tps + fps), eps)
    return _pr_curve(recalls=recalls, precisions=precisions)


def evaluate_centerline_stream(ground_truth, predictions, verbose=True):
    if predictions is None:
        raise ValueError("Predictions are required for streamed evaluation.")

    check_results(predictions)
    predictions = predictions["results"]

    gts = {}
    preds = {}
    for token in ground_truth.keys():
        gts[token] = ground_truth[token]["annotation"]
        preds[token] = predictions[token]["predictions"]

    assert set(gts.keys()) == set(preds.keys()), "#frame differs"

    lane_ap_state = {
        threshold: {"tp": [], "fp": [], "confidence": [], "num_gt": 0}
        for threshold in THRESHOLDS_FRECHET
    }
    te_attr_values = list(TRAFFIC_ELEMENT_ATTRIBUTE.values())
    te_ap_state = {
        attr: {threshold: {"tp": [], "fp": [], "confidence": [], "num_gt": 0} for threshold in THRESHOLDS_IOU}
        for attr in te_attr_values
    }

    for token in gts.keys():
        gt_lane_points = [gt["points"] for gt in gts[token]["lane_centerline"]]
        pred_lane_points = [pred["points"] for pred in preds[token]["lane_centerline"]]
        lane_confidences = [pred["confidence"] for pred in preds[token]["lane_centerline"]]

        lane_mask = pairwise(
            gt_lane_points,
            pred_lane_points,
            chamfer_distance,
            relax=True,
        ) < THRESHOLDS_FRECHET[-1]
        lane_distance_matrix = pairwise(
            gt_lane_points,
            pred_lane_points,
            frechet_distance,
            mask=lane_mask,
            relax=True,
        )

        for threshold in THRESHOLDS_FRECHET:
            tp, fp, idx_match_gt = _tpfp(
                gts=gt_lane_points,
                preds=pred_lane_points,
                confidences=lane_confidences,
                distance_matrix=lane_distance_matrix,
                distance_threshold=threshold,
            )
            lane_ap_state[threshold]["tp"].append(tp)
            lane_ap_state[threshold]["fp"].append(fp)
            lane_ap_state[threshold]["confidence"].append(lane_confidences)
            lane_ap_state[threshold]["num_gt"] += len(gt_lane_points)
            _inject(
                num_gt=len(gt_lane_points),
                pred=preds[token],
                tp=tp,
                idx_match_gt=idx_match_gt,
                confidence=lane_confidences,
                distance_threshold=threshold,
                object_type="lane_centerline",
            )

        gt_te_objects = gts[token]["traffic_element"]
        pred_te_objects = preds[token]["traffic_element"]
        te_distance_matrix = pairwise(
            [gt["points"] for gt in gt_te_objects],
            [pred["points"] for pred in pred_te_objects],
            iou_distance,
        )

        for attr in te_attr_values:
            gt_mask = [gt["attribute"] == attr for gt in gt_te_objects]
            pred_mask = [pred["attribute"] == attr for pred in pred_te_objects]
            filtered_distance_matrix = te_distance_matrix[gt_mask, :][:, pred_mask]
            gt_filtered = [gt for gt in gt_te_objects if gt["attribute"] == attr]
            pred_filtered = [pred for pred in pred_te_objects if pred["attribute"] == attr]
            confidences = [pred["confidence"] for pred in pred_filtered]

            for threshold in THRESHOLDS_IOU:
                tp, fp, _ = _tpfp(
                    gts=gt_filtered,
                    preds=pred_filtered,
                    confidences=confidences,
                    distance_matrix=filtered_distance_matrix,
                    distance_threshold=threshold,
                )
                te_ap_state[attr][threshold]["tp"].append(tp)
                te_ap_state[attr][threshold]["fp"].append(fp)
                te_ap_state[attr][threshold]["confidence"].append(confidences)
                te_ap_state[attr][threshold]["num_gt"] += len(gt_filtered)

        all_te_confidences = [pred["confidence"] for pred in pred_te_objects]
        for threshold in THRESHOLDS_IOU:
            tp, fp, idx_match_gt = _tpfp(
                gts=gt_te_objects,
                preds=pred_te_objects,
                confidences=all_te_confidences,
                distance_matrix=te_distance_matrix,
                distance_threshold=threshold,
            )
            _inject(
                num_gt=len(gt_te_objects),
                pred=preds[token],
                tp=tp,
                idx_match_gt=idx_match_gt,
                confidence=all_te_confidences,
                distance_threshold=threshold,
                object_type="traffic_element",
            )

    det_l = np.asarray([
        _finalize_ap(
            lane_ap_state[threshold]["tp"],
            lane_ap_state[threshold]["fp"],
            lane_ap_state[threshold]["confidence"],
            lane_ap_state[threshold]["num_gt"],
        )
        for threshold in THRESHOLDS_FRECHET
    ]).mean()

    det_t = np.hstack([
        np.asarray([
            _finalize_ap(
                te_ap_state[attr][threshold]["tp"],
                te_ap_state[attr][threshold]["fp"],
                te_ap_state[attr][threshold]["confidence"],
                te_ap_state[attr][threshold]["num_gt"],
            )
            for threshold in THRESHOLDS_IOU
        ])
        for attr in te_attr_values
    ]).mean()

    top_ll = _mAP_topology_lclc(gts, preds, THRESHOLDS_FRECHET)
    top_lt = _mAP_topology_lcte(
        gts,
        preds,
        {"lane_centerline": THRESHOLDS_FRECHET, "traffic_element": THRESHOLDS_IOU},
    )
    score = np.asarray([det_l, det_t, np.sqrt(top_ll), np.sqrt(top_lt)]).mean()

    return {
        "OpenLane-V2 Score": {
            "score": score,
            "DET_l": det_l,
            "DET_t": det_t,
            "TOP_ll": top_ll,
            "TOP_lt": top_lt,
        },
        "F-Score for 3D Lane": {
            "score": f1.bench_one_submit(gts=gts, preds=preds),
        },
    }