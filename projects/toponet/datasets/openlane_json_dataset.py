import os
import copy
import glob
import json

import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS, PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.utils import build_from_cfg
from mmdet3d.datasets import Custom3DDataset
from openlanev2.evaluation import evaluate as openlanev2_evaluate
from openlanev2.utils import format_metric

from ..core.lane.util import fix_pts_interpolate


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
            
            # Store raw JSON data for processing in get_data_info
            info["sensor"] = data.get("sensor", {})
            info["pose"] = data.get("pose", {})

            # annotations
            if "annotation" in data:
                info["annotation"] = data["annotation"]
            
            # scenario tags
            if "scenario_meta" in data:
                info["scenario_meta"] = data["scenario_meta"]

            data_infos.append(info)

    return data_infos


def load_openlane_json_lazy(root_dir, split):
    """Collect only file paths/identifiers without reading JSON contents.

    Each entry is a lightweight stub with keys:
        segment_id, timestamp, json_path, image_dir, _lazy (=True)
    The actual JSON is loaded on first access via
    ``OpenLaneJSONDataset._ensure_info_loaded``.
    """
    data_infos = []
    split_dir = os.path.join(root_dir, split)
    segments = sorted(glob.glob(os.path.join(split_dir, "*")))
    for seg in segments:
        seg_id = os.path.basename(seg)
        info_dir = os.path.join(seg, "info")
        image_dir = os.path.join(seg, "image")
        json_files = sorted(glob.glob(info_dir + "/*.json"))
        for jf in json_files:
            timestamp = os.path.basename(jf).replace(".json", "")
            data_infos.append({
                "segment_id": seg_id,
                "timestamp": timestamp,
                "json_path": jf,
                "image_dir": image_dir,
                "_lazy": True,
            })
    return data_infos


@DATASETS.register_module()
class OpenLaneJSONDataset(Custom3DDataset):

    CLASSES = ("centerline",)

    def __init__(self,
                 data_root,
                 split="train",
                 pipeline=None,
                 test_mode=False,
                 modality=None,
                 lazy_load=False,
                 **kwargs):

        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.lazy_load = lazy_load
        self.modality = modality if modality is not None else dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False
        )

        # Load data infos — lazy mode only collects paths, no JSON I/O
        if lazy_load:
            self.data_infos = load_openlane_json_lazy(data_root, split)
        else:
            self.data_infos = load_openlane_json(data_root, split)
        
        # Build pipeline from config
        if pipeline is not None:
            if isinstance(pipeline, list):
                self.pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
            else:
                self.pipeline = pipeline
        else:
            self.pipeline = None

    def __len__(self):
        return len(self.data_infos)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_metas'] = {}
    
    @staticmethod
    def _ensure_info_loaded(info):
        """If *info* is a lazy stub, read its JSON and fill in full fields."""
        if not info.get('_lazy', False):
            return info
        with open(info['json_path']) as f:
            data = json.load(f)
        info['sensor'] = data.get('sensor', {})
        info['pose'] = data.get('pose', {})
        if 'annotation' in data:
            info['annotation'] = data['annotation']
        if 'scenario_meta' in data:
            info['scenario_meta'] = data['scenario_meta']
        info['_lazy'] = False
        return info

    def __getitem__(self, idx):
        """Get item from dataset."""
        # Materialise lazy stubs on first access
        self.data_infos[idx] = self._ensure_info_loaded(self.data_infos[idx])
        data = self.get_data_info(idx)
        if data is None:
            return None
        self.pre_pipeline(data)
        if self.pipeline is not None:
            data = self.pipeline(data)
        return data

    def get_data_info(self, index):
        """Get data info according to the given index.
        
        Args:
            index (int): Index of the sample data to get.
            
        Returns:
            dict: Data information that will be passed to the data preprocessing pipelines.
        """
        info = self.data_infos[index]
        
        input_dict = dict(
            sample_idx=info['timestamp'],
            scene_token=info['segment_id']
        )
        
        # Process camera data
        if self.modality['use_camera'] and 'sensor' in info:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            # Process sensor data for each camera
            for cam_name, cam_info in info['sensor'].items():
                if 'image_path' not in cam_info:
                    continue
                    
                image_path = cam_info['image_path']
                image_paths.append(os.path.join(self.data_root, image_path))
                
                # Get extrinsic and intrinsic matrices
                if 'extrinsic' in cam_info and 'intrinsic' in cam_info:
                    extrinsic = cam_info['extrinsic']
                    intrinsic = cam_info['intrinsic']
                    
                    # Compute lidar to camera rotation and translation
                    lidar2cam_r = np.linalg.inv(np.array(extrinsic['rotation']))
                    lidar2cam_t = cam_info['extrinsic']['translation'] @ lidar2cam_r.T
                    
                    # Build 4x4 lidar to camera matrix
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    
                    # Get intrinsic matrix
                    intrinsic_matrix = np.array(intrinsic['K'])
                    
                    # Build view padding matrix (4x4)
                    viewpad = np.eye(4)
                    viewpad[:intrinsic_matrix.shape[0], :intrinsic_matrix.shape[1]] = intrinsic_matrix
                    
                    # Compute lidar to image projection matrix
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    
                    lidar2img_rts.append(lidar2img_rt)
                    cam_intrinsics.append(viewpad)
                    lidar2cam_rts.append(lidar2cam_rt.T)
            
            if image_paths:
                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                        cam_intrinsic=cam_intrinsics,
                        lidar2cam=lidar2cam_rts,
                    ))
        
        # Process pose/can_bus data
        if 'pose' in info:
            pose = info['pose']
            can_bus = np.zeros(18)
            
            if 'translation' in pose:
                can_bus[:3] = np.array(pose['translation'])
            
            if 'rotation' in pose:
                input_dict['lidar2global_rotation'] = np.array(pose['rotation'])
                try:
                    rotation = Quaternion._from_matrix(np.array(pose['rotation']))
                    can_bus[3:7] = rotation
                    patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
                    if patch_angle < 0:
                        patch_angle += 360
                    can_bus[-2] = patch_angle / 180 * np.pi
                    can_bus[-1] = patch_angle
                except:
                    # Fallback if rotation matrix is invalid
                    pass
            
            input_dict['can_bus'] = can_bus
        
        # Add annotations if available
        if "annotation" in info:
            input_dict["ann_info"] = info["annotation"]
        
        # Add scenario tags if available
        if "scenario_meta" in info:
            input_dict["scenario_meta"] = info["scenario_meta"]
        
        return input_dict

    def format_openlanev2_gt(self):
        gt_dict = {}
        for idx in range(len(self.data_infos)):
            info = copy.deepcopy(self.data_infos[idx])
            key = (self.split, info['segment_id'], str(info['timestamp']))
            for lane in info['annotation']['lane_centerline']:
                lane['points'] = np.array(lane['points'], dtype=np.float32)
                if len(lane['points']) == 201:
                    lane['points'] = lane['points'][::20]  # downsample: 201 --> 11
            for te in info['annotation']['traffic_element']:
                te['points'] = np.array(te['points'], dtype=np.float32)
            info['annotation']['topology_lclc'] = np.array(info['annotation']['topology_lclc'], dtype=np.float32)
            info['annotation']['topology_lcte'] = np.array(info['annotation']['topology_lcte'], dtype=np.float32)
            gt_dict[key] = info
        return gt_dict

    def format_results(self, results):
        pred_dict = {}
        pred_dict['method'] = 'TopoNet'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'OpenDriveLab'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        for idx, result in enumerate(results):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_centerline=[],
                traffic_element=[],
                topology_lclc=None,
                topology_lcte=None
            )

            valid_indices = None
            if result['lane_results'] is not None:
                lane_results = result['lane_results']
                scores = lane_results[1]
                valid_indices = np.argsort(-scores)
                lanes = lane_results[0][valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)
                scores = scores[valid_indices]
                for pred_idx, (lane, score) in enumerate(zip(lanes, scores)):
                    points = fix_pts_interpolate(lane, 11)
                    lc_info = dict(
                        id=10000 + pred_idx,
                        points=points.astype(np.float32),
                        confidence=score.item()
                    )
                    pred_info['lane_centerline'].append(lc_info)

            te_valid_indices = None
            if result['bbox_results'] is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id=20000 + pred_idx,
                        category=1 if class_idx < 4 else 2,
                        attribute=class_idx,
                        points=te.reshape(2, 2).astype(np.float32),
                        confidence=score
                    )
                    pred_info['traffic_element'].append(te_info)

            if result['lclc_results'] is not None and valid_indices is not None:
                pred_info['topology_lclc'] = result['lclc_results'].astype(np.float32)[valid_indices][:, valid_indices]
            else:
                pred_info['topology_lclc'] = np.zeros((len(pred_info['lane_centerline']), len(pred_info['lane_centerline'])), dtype=np.float32)

            if result['lcte_results'] is not None and valid_indices is not None and te_valid_indices is not None:
                pred_info['topology_lcte'] = result['lcte_results'].astype(np.float32)[valid_indices][:, te_valid_indices]
            else:
                pred_info['topology_lcte'] = np.zeros((len(pred_info['lane_centerline']), len(pred_info['traffic_element'])), dtype=np.float32)

            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluation using OpenLane-V2 metrics."""
        from mmcv.utils import get_logger as _get_logger
        if logger is None:
            logger = _get_logger('mmdet')

        logger.info('Formatting ground truth...')
        gt_dict = self.format_openlanev2_gt()

        logger.info('Formatting predictions...')
        pred_dict = self.format_results(results)

        logger.info('Running openlanev2 evaluate...')
        metric_results = openlanev2_evaluate(gt_dict, pred_dict)
        format_metric(metric_results)
        metric_results = {
            'OpenLane-V2 Score': metric_results['OpenLane-V2 Score']['score'],
            'DET_l': metric_results['OpenLane-V2 Score']['DET_l'],
            'DET_t': metric_results['OpenLane-V2 Score']['DET_t'],
            'TOP_ll': metric_results['OpenLane-V2 Score']['TOP_ll'],
            'TOP_lt': metric_results['OpenLane-V2 Score']['TOP_lt'],
        }
        return metric_results