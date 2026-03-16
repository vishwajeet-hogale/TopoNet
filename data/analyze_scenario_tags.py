#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze scenario tag distribution across all JSON files.

Outputs:
- Console summary
- analysis/scenario_tag_analysis/tag_counts.csv
- analysis/scenario_tag_analysis/num_tags_per_file.csv
- analysis/scenario_tag_analysis/summary.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def find_json_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def normalize_tag(tag: str) -> str:
    return " ".join(tag.strip().split()).lower()


def analyze_files(files: List[Path]) -> Dict[str, Any]:
    tag_counts: Counter[str] = Counter()
    tags_per_file: Counter[int] = Counter()
    combo_counts: Counter[str] = Counter()

    files_with_scenario_tags = 0
    files_without_scenario_tags = 0
    files_with_empty_scenario_tags = 0
    files_parsed = 0
    files_failed = 0

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            files_failed += 1
            continue

        if not isinstance(data, dict):
            files_failed += 1
            continue

        files_parsed += 1
        raw_tags = data.get("scenario_tags")

        if raw_tags is None:
            files_without_scenario_tags += 1
            tags_per_file[0] += 1
            continue

        if not isinstance(raw_tags, list):
            files_failed += 1
            continue

        files_with_scenario_tags += 1
        normalized_tags = sorted({
            normalize_tag(t)
            for t in raw_tags
            if isinstance(t, str) and t.strip()
        })

        if not normalized_tags:
            files_with_empty_scenario_tags += 1

        tags_per_file[len(normalized_tags)] += 1
        for t in normalized_tags:
            tag_counts[t] += 1

        combo_key = " | ".join(normalized_tags) if normalized_tags else "<empty>"
        combo_counts[combo_key] += 1

    return {
        "files_total": len(files),
        "files_parsed": files_parsed,
        "files_failed": files_failed,
        "files_with_scenario_tags": files_with_scenario_tags,
        "files_without_scenario_tags": files_without_scenario_tags,
        "files_with_empty_scenario_tags": files_with_empty_scenario_tags,
        "unique_tags": len(tag_counts),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tags_per_file": dict(sorted(tags_per_file.items(), key=lambda kv: kv[0])),
        "top_tag_combinations": [
            {"combination": k, "count": v}
            for k, v in combo_counts.most_common(25)
        ],
    }


def write_tag_counts_csv(tag_counts: Dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_tag", "datapoint_count"])
        for tag, count in tag_counts.items():
            writer.writerow([tag, count])


def write_tags_per_file_csv(tags_per_file: Dict[int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_tags_in_file", "file_count"])
        for n_tags, file_count in tags_per_file.items():
            writer.writerow([n_tags, file_count])


def print_summary(summary: Dict[str, Any]) -> None:
    print("=== Scenario Tag Analysis ===")
    print(f"Total JSON files:            {summary['files_total']}")
    print(f"Parsed JSON files:           {summary['files_parsed']}")
    print(f"Failed JSON files:           {summary['files_failed']}")
    print(f"With scenario_tags:          {summary['files_with_scenario_tags']}")
    print(f"Without scenario_tags:       {summary['files_without_scenario_tags']}")
    print(f"Empty scenario_tags lists:   {summary['files_with_empty_scenario_tags']}")
    print(f"Unique scenario tags:        {summary['unique_tags']}")

    print("\nTop tags:")
    tag_items = list(summary["tag_counts"].items())[:20]
    if not tag_items:
        print("  <none>")
    for tag, count in tag_items:
        print(f"  {tag:35s} {count}")

    print("\nTags per file distribution:")
    for n_tags, file_count in summary["tags_per_file"].items():
        print(f"  {n_tags:2d} tags: {file_count}")

    print("\nTop combinations:")
    combos = summary["top_tag_combinations"][:10]
    if not combos:
        print("  <none>")
    for c in combos:
        print(f"  {c['count']:6d}  {c['combination']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_root", type=str, required=True,
                        help="Root directory to recursively scan for JSON files.")
    parser.add_argument("--output_dir", type=str, default="analysis/scenario_tag_analysis",
                        help="Directory to store CSV/JSON summary outputs.")
    args = parser.parse_args()

    root = Path(args.target_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid target_root: {root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_json_files(root)
    summary = analyze_files(files)

    write_tag_counts_csv(summary["tag_counts"], output_dir / "tag_counts.csv")
    write_tags_per_file_csv(summary["tags_per_file"], output_dir / "num_tags_per_file.csv")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
