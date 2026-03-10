import argparse
import os
import os.path as osp
import warnings

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.utils import get_logger

from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test


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

def evaluate_by_scenario(dataset, outputs, eval_kwargs):

    print("\n==============================")
    print("GLOBAL METRICS")
    print("==============================")

    global_metrics = dataset.evaluate(outputs, **eval_kwargs)
    print(global_metrics)

    scenarios = {
        "curvature": {},
        "lighting": {},
        "occlusion": {},
        "topology_complexity": {}
    }

    # build scenario groups
    for i in range(len(dataset)):

        info = dataset.data_infos[i]

        if "scenario" not in info:
            continue

        scenario = info["scenario"]

        mapping = {
            "curvature": scenario.get("curvature"),
            "lighting": scenario.get("lighting"),
            "occlusion": scenario.get("occlusion"),
            "topology_complexity": scenario.get("topology_complexity")
        }

        for key, value in mapping.items():

            if value is None:
                continue

            if value not in scenarios[key]:
                scenarios[key][value] = []

            scenarios[key][value].append(i)

    print("\n==============================")
    print("SCENARIO BREAKDOWN")
    print("==============================")

    for scenario_type in scenarios:

        print("\n---", scenario_type.upper(), "---")

        for category in scenarios[scenario_type]:

            indices = scenarios[scenario_type][category]

            subset_outputs = [outputs[i] for i in indices]

            metrics = dataset.evaluate(subset_outputs, **eval_kwargs)

            print(category, metrics.get("OLS", metrics))


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
    # Dataset
    # -------------------------

    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # -------------------------
    # Model
    # -------------------------

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

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

        evaluate_by_scenario(dataset, outputs, eval_kwargs)


if __name__ == "__main__":
    main()