"""Compare deterministic and MC-Dropout terrain-classifier suite results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_ARCHITECTURES = {"feature_nn", "raw_depth_nn"}


def parse_result_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if "architecture" not in result or "deployments" not in result or "search" not in result:
        raise ValueError(f"{path} is not an uncertainty-aware NN result")
    result["result_file"] = str(path.resolve())
    return result


def discover_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif (path / "results.json").is_file():
            files.append(path / "results.json")
        elif path.is_dir():
            files.extend(path.glob("*/results.json"))
    results = [parse_result_file(path) for path in sorted(set(files))]
    if not results:
        raise FileNotFoundError("no NN results.json files found")
    return results


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    scalar = ("accuracy", "balanced_accuracy", "macro_f1", "transition_window_accuracy",
              "steady_state_accuracy", "mean_transition_delay", "false_transition_rate",
              "minimum_class_recall", "nll", "brier", "event_fraction", "event_precision",
              "expected_calibration_error",
              "event_recall", "mean_event_offset", "ambiguous_frame_fraction",
              "mean_ambiguity_run_length", "ambiguity_inside_transition_fraction",
              "ambiguity_outside_transition_fraction", "high_epistemic_frame_fraction")
    row = {f"{prefix}_{key}": metrics.get(key) for key in scalar}
    for key in ("per_class_precision", "per_class_recall", "missing_predicted_classes", "confusion_matrix"):
        row[f"{prefix}_{key}"] = json.dumps(metrics.get(key))
    return row


def _deployment_row(run, mode):
    value = run["deployments"][mode]
    ema = value.get("best_ema", {})
    ema_parameters = ema.get("parameters", {})
    return {"approach": f"{run['architecture']}_{mode}", "architecture": run["architecture"],
            "inference_mode": mode, "dropout_p": value["dropout_p"],
            "weight_decay": value["weight_decay"],
            "mc_samples": value["mc_samples"], "model_path": value["model_path"],
            "temporal_filter_path": value["temporal_filter_path"],
            "temporal_family": value["selected_temporal_filter_family"],
            "T_filter": value["T_filter"], "stable_stay": value["stable_stay"],
            "release_strength": value.get("release_strength"),
            "switch_margin": value.get("switch_margin"),
            "change_patience": value.get("change_patience"),
            "epistemic_percentile": value.get("epistemic_percentile"),
            "epistemic_threshold": value.get("epistemic_threshold"),
            "ambiguity_margin": value.get("ambiguity_margin"),
            "flatten_strength": value.get("flatten_strength"),
            "cv_selection_score": value["cv_metrics"]["mean_selection_score"],
            **_flatten_metrics("structural_validation", value["structural_metrics"]["structural_validation"]),
            **_flatten_metrics("structural_test", value["structural_metrics"]["structural_test"]),
            **_flatten_metrics("instantaneous_validation", value["sequential_instantaneous_validation"]),
            **_flatten_metrics("instantaneous_test", value["sequential_instantaneous_test"]),
            **_flatten_metrics("validation", value["validation_metrics"]),
            **_flatten_metrics("test", value["ordered_test_metrics"]),
            "ema_alpha": ema_parameters.get("ema_alpha"),
            "ema_patience": ema_parameters.get("change_patience"),
            "ema_filter_path": ema.get("filter_path"),
            "ema_cv_selection_score": ema.get("cv", {}).get("mean_selection_score"),
            **_flatten_metrics("ema_validation", ema.get("validation_metrics", {})),
            **_flatten_metrics("ema_test", ema.get("ordered_test_metrics", {})),
            "uncertainty_stats": json.dumps(value.get("uncertainty_stats", {}), sort_keys=True),
            "inference_runtime_seconds": json.dumps(value.get("inference_runtime_seconds", {}), sort_keys=True),
            "training_runtime_seconds": value.get("training_runtime_seconds"),
            "result_file": run["result_file"]}


def _stage_rows(run):
    rows = []
    for config_id, config in run["search"]["configs"].items():
        candidate = config["candidate"]
        for stage in ("stage1", "stage2", "stage3", "stage4", "ema"):
            record = config.get(stage)
            if not record:
                continue
            rows.append({"architecture": run["architecture"], "config_id": config_id,
                         "inference_mode": candidate["inference_mode"],
                         "dropout_p": candidate["dropout_p"], "mc_samples": candidate["mc_samples"],
                         "weight_decay": candidate["weight_decay"],
                         "stage": stage, "parameters": json.dumps(record["parameters"], sort_keys=True),
                         **_flatten_metrics("validation", record["validation_metrics"]),
                         **_flatten_metrics("test", record["ordered_test_metrics"])})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("terrain_classifier_comparison"))
    args = parser.parse_args()
    runs = discover_results(args.runs)
    present = {run["architecture"] for run in runs}
    if REQUIRED_ARCHITECTURES - present:
        print("Warning: missing " + ", ".join(sorted(REQUIRED_ARCHITECTURES - present)))
    winners = [_deployment_row(run, mode) for run in runs for mode in ("deterministic", "mc")]
    stage_rows = [row for run in runs for row in _stage_rows(run)]
    by_name = {row["approach"]: row for row in winners}
    deterministic = [row for row in winners if row["inference_mode"] == "deterministic"]
    mc = [row for row in winners if row["inference_mode"] == "mc"]
    best = lambda values: max(values, key=lambda row: row["cv_selection_score"])
    selections = {
        "feature_nn deterministic winner": by_name.get("feature_nn_deterministic"),
        "feature_nn MC winner": by_name.get("feature_nn_mc"),
        "raw_depth_nn deterministic winner": by_name.get("raw_depth_nn_deterministic"),
        "raw_depth_nn MC winner": by_name.get("raw_depth_nn_mc"),
        "global deterministic winner": best(deterministic) if deterministic else None,
        "global MC winner": best(mc) if mc else None,
        "global overall winner": best(winners),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output.with_name(args.output.name + "_winners.csv"), winners)
    _write_csv(args.output.with_name(args.output.name + "_stages.csv"), stage_rows)
    payload = {"approaches": winners, "stage_results": stage_rows, "selections": selections}
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    with args.output.with_name("suite_deployments.json").open("w", encoding="utf-8") as stream:
        json.dump(by_name, stream, indent=2)


if __name__ == "__main__":
    main()
