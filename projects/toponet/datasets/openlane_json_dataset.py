import os
import glob
import json

import mmcv
import numpy as np

from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset


def load_openlane_json(root_dir, split):

    data_infos = []

    split_dir = os.path.join(root_dir, split)

    segments = sorted(glob.glob(os.path.join(split_dir, "*")))

    for seg in segments:

        seg_id = os.path.basename(seg)

        info_dir = os.path.join(seg, "info")
        image_dir = os.path.join(seg, "image")

        json_files = sorted(glob.glob(info_dir + "/*.json"))

        for jf in json_files:

            with open(jf) as f:
                data = json.load(f)

            timestamp = os.path.basename(jf).replace(".json", "")

            info = {}

            info["segment_id"] = seg_id
            info["timestamp"] = timestamp

            info["image_dir"] = image_dir
            info["json_path"] = jf

            # camera images
            if "sensor" in data:
                info["cams"] = data["sensor"]

            # annotations
            if "annotation" in data:
                info["annotation"] = data["annotation"]

            # scenario tags (your addition)
            if "scenario" in data:
                info["scenario"] = data["scenario"]

            data_infos.append(info)

    return data_infos


@DATASETS.register_module()
class OpenLaneJSONDataset(Custom3DDataset):

    CLASSES = ("centerline",)

    def __init__(self,
                 data_root,
                 split="train",
                 pipeline=None,
                 test_mode=False,
                 **kwargs):

        self.split = split
        self.data_root = data_root

        self.data_infos = load_openlane_json(data_root, split)

        super().__init__(
            data_root=data_root,
            ann_file=None,
            pipeline=pipeline,
            classes=self.CLASSES,
            test_mode=test_mode,
            **kwargs
        )

    def __len__(self):
        return len(self.data_infos)

    def get_data_info(self, index):

        info = self.data_infos[index]

        data = dict()

        data["img_prefix"] = info["image_dir"]

        data["img_info"] = dict(
            filename=info["timestamp"]
        )

        if "annotation" in info:
            data["ann_info"] = info["annotation"]

        # include scenario tags
        if "scenario" in info:
            data["scenario"] = info["scenario"]

        return data

    def evaluate(self, results, **kwargs):

        print("Evaluation placeholder")

        # here you plug your scenario evaluation later

        return {}