"""One-command orchestration for terrain-classifier training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from legged_gym import LEGGED_GYM_ROOT_DIR


APPROACH_MODULES = {
    "rbf_prototype": "legged_gym.scripts.depth_data_pipeline.train_rbf_prototype_classifier",
    "rbf_svm": "legged_gym.scripts.depth_data_pipeline.train_rbf_svm_classifier",
    "feature_nn": "legged_gym.scripts.depth_data_pipeline.train_feature_nn",
    "raw_depth_nn": "legged_gym.scripts.depth_data_pipeline.train_raw_depth_nn",
}


def _contains(folder: Path, names: Iterable[str]) -> bool:
    return folder.is_dir() and all((folder / f"{name}.pt").is_file() for name in names)


def _resolve_data_folder(
    explicit: Path | None, root: Path, candidates: tuple[str, ...], required: tuple[str, ...], kind: str,
) -> Path:
    if explicit is not None:
        folder = explicit.expanduser().resolve()
        if not _contains(folder, required):
            raise FileNotFoundError(f"{kind} dataset {folder} must contain {required}")
        return folder
    direct = [root / name for name in candidates] + [root]
    for folder in direct:
        if _contains(folder, required):
            return folder.resolve()
    matches = [folder for folder in root.iterdir() if _contains(folder, required)] if root.is_dir() else []
    if matches:
        return max(matches, key=lambda folder: folder.stat().st_mtime).resolve()
    raise FileNotFoundError(
        f"Could not find {kind} data below {root}; pass the corresponding explicit folder flag"
    )


def parse_args() -> argparse.Namespace:
    default_dataset = Path(os.environ.get(
        "TERRAIN_CLASSIFIER_DATASET",
        str(Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "processed_data"),
    ))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach", nargs="+", default=["all"],
        choices=["all", *APPROACH_MODULES],
        help="one or more approaches; 'all' runs every approach sequentially",
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset,
                        help="dataset root containing structural/ and bayesian/ folders")
    parser.add_argument("--classifier-data", type=Path,
                        help="explicit structural train/val/calibration/test folder")
    parser.add_argument("--bayesian-data", type=Path,
                        help="explicit ordered Bayes val/test (optionally train/calibration) folder")
    parser.add_argument("--output", type=Path,
                        help="suite output root; each approach receives a separate subfolder")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-batch-processing", action="store_true",
                        help="use each complete split as one batch; may exhaust RAM/VRAM")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    classifier_data = _resolve_data_folder(
        args.classifier_data, dataset_root, ("structural", "classifier"),
        ("train", "val", "calibration", "test"), "structural classifier",
    )
    bayesian_data = _resolve_data_folder(
        args.bayesian_data, dataset_root, ("bayesian", "bayes", "sequences"),
        ("val", "test"), "ordered Bayesian",
    )
    output_root = (args.output or (
        Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "full_models" /
        f"terrain_classifier_suite_{datetime.now():%Y%m%d_%H%M%S}"
    )).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    approaches = list(APPROACH_MODULES) if "all" in args.approach else list(dict.fromkeys(args.approach))
    manifest = {
        "dataset": str(dataset_root), "classifier_data": str(classifier_data),
        "bayesian_data": str(bayesian_data), "output": str(output_root),
        "batch_processing": not args.no_batch_processing, "batch_size": args.batch_size,
        "runs": {},
    }
    successful_outputs: list[Path] = []
    for approach in approaches:
        approach_output = output_root / approach
        command = [
            sys.executable, "-m", APPROACH_MODULES[approach],
            "--classifier_folder", str(classifier_data),
            "--bayesian_folder", str(bayesian_data),
            "--output_dir", str(approach_output),
            "--batch_size", str(args.batch_size),
        ]
        if args.no_batch_processing:
            command.append("--no_batch_processing")
        print(f"\n=== Training {approach} -> {approach_output} ===", flush=True)
        completed = subprocess.run(command, check=False)
        manifest["runs"][approach] = {
            "output": str(approach_output), "return_code": completed.returncode,
            "status": "completed" if completed.returncode == 0 else "failed",
        }
        with (output_root / "suite_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        if completed.returncode == 0:
            successful_outputs.append(approach_output)
        elif not args.continue_on_error:
            raise SystemExit(completed.returncode)

    if successful_outputs and not args.skip_comparison:
        comparison_prefix = output_root / "terrain_classifier_comparison"
        command = [
            sys.executable, "-m",
            "legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results",
            *map(str, successful_outputs), "--output", str(comparison_prefix),
        ]
        comparison = subprocess.run(command, check=False)
        manifest["comparison_return_code"] = comparison.returncode
        with (output_root / "suite_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        if comparison.returncode != 0 and not args.continue_on_error:
            raise SystemExit(comparison.returncode)
    print(f"\nSuite outputs: {output_root}")


if __name__ == "__main__":
    main()
