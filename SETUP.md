# TopoNet Setup And Run Guide

This file documents the exact workflow used in this workspace, without changing the original project README.

## Scope

- OS: Windows
- Shell: PowerShell
- Python env: `TopoNet` conda environment
- Dataset split used for streamed aggregation below: `train`
- Recommended aggregation mode on 8 GB RAM: `--cache-mode lazy`

## Expected Layout

```text
TopoNet/
├── ckpts/
│   └── toponet_r50_8x1_24e_olv2_subset_A.pth
├── data/
│   └── OpenLane-V2/
├── projects/
├── tools/
└── work_dirs/
```

## Environment Setup

Create and activate the environment:

```powershell
conda create -n TopoNet python=3.8 -y
conda activate TopoNet
```

Install PyTorch and CUDA-compatible dependencies:

```powershell
conda install cudatoolkit=11.1.1 -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc6
pip install -r requirements.txt
```

## Notes For Windows

- Run commands from the repository root.
- Keep `--workers-per-gpu 0` for evaluation commands on Windows.
- If a streamed run is interrupted, use `--resume` when rerunning `tools/test_scenario.py`.
- For large streamed outputs on an 8 GB machine, use `--cache-mode lazy` for aggregation.

## Dataset

Expected data location:

```text
data/OpenLane-V2/
├── train/
├── val/
├── test/
└── ...
```

## Checkpoint

Subset-A checkpoint expected at:

```text
ckpts/toponet_r50_8x1_24e_olv2_subset_A.pth
```

## Inference Workflow

### 1. Stream predictions sample-by-sample

Run sample-by-sample inference and write one `.pkl` per sample to `work_dirs/results/stream_outputs`:

```powershell
python tools/test_scenario.py projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py ckpts/toponet_r50_8x1_24e_olv2_subset_A.pth --eval chamfer --out-dir work_dirs/results --sample-by-sample --split train --resume
```

What this produces:

- `work_dirs/results/stream_outputs/*.pkl`
- `work_dirs/results/stream_outputs/manifest.txt`
- optional per-sample CSV summaries if enabled by the script

### 2. Smoke test aggregation on a small subset

Before running full aggregation, test the pipeline on a few samples:

```powershell
python tools/aggregate_stream_metrics.py projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py --stream-dir work_dirs/results/stream_outputs --out-dir work_dirs/results --split train --cache-mode lazy --max-samples 10
```

### 3. Aggregate all streamed train samples

Run the full aggregation once the smoke test succeeds:

````powershell
python tools/aggregate_stream_metrics.py projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py --stream-dir work_dirs/results/stream_outputs --out-dir work_dirs/results --split train --cache-mode lazy

If aggregation still strains RAM or stalls mid-progress, cap lazy cache size explicitly:

```powershell
python tools/aggregate_stream_metrics.py projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py --stream-dir work_dirs/results/stream_outputs --out-dir work_dirs/results --split train --cache-mode lazy --lazy-cache-items 32
````

````

## Aggregator Behavior

`tools/aggregate_stream_metrics.py` in this workspace has been updated to:

- support progress bars
- support RAM-aware caching and lazy loading
- load evaluation metadata lazily for the JSON dataset
- handle `manifest.txt` path resolution correctly on Windows
- save CSV summaries and PNG plots

Key flags:

| Flag                 | Meaning                                                      |
| -------------------- | ------------------------------------------------------------ |
| `--split train`      | Evaluate using the train split metadata and annotations      |
| `--cache-mode lazy`  | Read `.pkl` files on demand instead of preloading everything |
| `--lazy-cache-items` | Keep a small LRU cache in lazy mode (set `0` to disable cache) |
| `--max-samples N`    | Limit aggregation to the first `N` streamed predictions      |
| `--load-workers N`   | Number of worker threads for metadata or preload work        |
| `--max-preload-gb N` | Optional manual preload threshold for auto mode              |

## Expected Output

After aggregation completes, a new directory is created:

```text
work_dirs/results/eval_YYYYMMDD_HHMMSS/
````

Expected files:

```text
global_metrics.csv
all_metrics.csv
global_metrics.png
scenario_curvature.csv
scenario_curvature.png
scenario_lighting.csv
scenario_lighting.png
scenario_occlusion.csv
scenario_occlusion.png
scenario_topology_complexity.csv
scenario_topology_complexity.png
```

## Typical Terminal Flow

During aggregation you should see output similar to:

```text
Overriding data.test.split = 'train'
Building dataset in lightweight lazy mode for evaluation ...
Found 22477 streamed predictions in: work_dirs/results/stream_outputs
Loading evaluation metadata: ...
Discovered 22477 prediction files (... total).
Using lazy loading for aggregation ...
Scanning scenario labels: ...
Evaluating metric groups: ...
```

At the end it will print the output directory and saved files.

## Fast Recovery Commands

Resume interrupted inference:

```powershell
python tools/test_scenario.py projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py ckpts/toponet_r50_8x1_24e_olv2_subset_A.pth --eval chamfer --out-dir work_dirs/results --sample-by-sample --split train --resume
```

Rerun full aggregation:

```powershell
python tools/aggregate_stream_metrics.py projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py --stream-dir work_dirs/results/stream_outputs --out-dir work_dirs/results --split train --cache-mode lazy
```
