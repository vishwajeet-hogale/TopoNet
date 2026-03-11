"""
Quick script to load and test the dataset with just 3 folders
"""
import argparse
import os
import os.path as osp
import sys
import json

# Add parent directory to path so projects module can be imported
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="QuickTest dataset")
    parser.add_argument("--config", default="projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py", 
                        help="config file")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="data split to test")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="maximum number of samples to load")
    parser.add_argument("--data-root", default="data/OpenLane-V2/",
                        help="data root directory")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load config
    print("Loading config...")
    cfg = Config.fromfile(args.config)
    print(f"Config loaded from: {args.config}\n")
    
    # Update data root if specified
    if args.data_root:
        cfg.data_root = args.data_root
    
    print(f"Building dataset for split: {args.split}")
    print(f"Data root: {cfg.data_root}")
    
    # Get dataset config
    dataset_cfg = cfg.data[args.split]
    print(f"Dataset type: {dataset_cfg.type}")
    print(f"Pipeline steps: {len(dataset_cfg.pipeline)}\n")
    
    # Build the dataset
    print("Building dataset with mmdet3d...")
    print("(This loads ALL segments and JSON files from the split directory)")
    try:
        dataset = build_dataset(dataset_cfg)
        print(f"\n✓ Dataset built successfully!")
        print(f"Total samples available: {len(dataset)}")
        
        # Limit to max samples for testing
        if args.max_samples and len(dataset) > args.max_samples:
            from torch.utils.data import Subset
            dataset = Subset(dataset, range(args.max_samples))
            print(f"Limited to first {args.max_samples} samples for testing")
        
        print(f"Samples to test: {len(dataset)}\n")
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty!")
            return
        
        # Try loading first few samples with progress
        print("--- Loading samples ---")
        for i in range(min(3, len(dataset))):
            print(f"Loading sample {i}...", end=" ", flush=True)
            sample = dataset[i]
            print(f"✓ Loaded (keys: {list(sample.keys())})")
        
        # Show details of first sample
        print("\n--- First sample details ---")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: tensor of shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Try building dataloader
        print("\n--- Building dataloader ---")
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=0,  # Use 0 workers for testing
            dist=False,
            shuffle=False,
        )
        print(f"✓ DataLoader built successfully!")
        print(f"Number of batches: {len(data_loader)}")
        
        # Try loading one batch
        print("\n--- Loading first batch from dataloader ---")
        for batch_idx, batch in enumerate(data_loader):
            print(f"Batch {batch_idx} loaded successfully!")
            print(f"Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: tensor of shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                else:
                    print(f"  {key}: {type(value).__name__}")
            break
        
        print("\n✓✓✓ Dataset testing completed successfully! ✓✓✓")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

