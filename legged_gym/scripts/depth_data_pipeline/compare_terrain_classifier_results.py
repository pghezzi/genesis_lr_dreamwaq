"""Compare deterministic and MC-Dropout terrain-classifier suite results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_ARCHITECTURES = {"feature_nn", "raw_depth_nn"}


def _score(value: Any, default: float = float("-inf")) -> float:
    """Return a sortable metric value from JSON results.

    ``save_results`` deliberately encodes non-finite values as ``null``.  The
    reporting step must therefore tolerate ``None`` rather than failing after
    a successful training run just because a metric is unavailable.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value else default  # NaN is the only float != itself.


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
    scalar = ("selection_score", "score_v2", "legacy_selection_score",
              "accuracy", "balanced_accuracy", "macro_f1", "transition_window_accuracy",
              "steady_state_accuracy", "mean_transition_delay", "false_transition_rate",
              "minimum_class_recall", "nll", "brier", "event_fraction", "event_precision",
              "expected_calibration_error",
              "event_recall", "mean_event_offset", "mean_switch_event_offset",
              "ambiguous_frame_fraction",
              "mean_ambiguity_run_length", "ambiguity_inside_transition_fraction",
              "ambiguity_outside_transition_fraction", "high_epistemic_frame_fraction",
              "switch_event_precision", "switch_event_recall",
              "true_transition_detection_recall", "mean_mc_agreement",
              "agreement_correct_frames", "agreement_incorrect_frames",
              "agreement_true_switch_events", "agreement_false_switch_events",
              "mean_beta", "beta_std", "beta_correct_frames", "beta_incorrect_frames",
              "beta_inside_transition_window", "beta_outside_transition_window",
              "mean_accumulated_evidence", "evidence_true_switch_events",
              "evidence_false_switch_events", "evidence_frames_above_threshold",
              "evidence_fraction_above_threshold", "uncertainty_error_auroc",
              "inference_latency_seconds", "effective_hz", "predictive_entropy_mean",
              "mutual_information_mean")
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
            "uncertainty_stage": value.get("selected_temporal_filter_family"),
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
            records = record.get("frontier", [record])
            best_id = record.get("best", {}).get("trial_id")
            for value in records:
                rows.append({"architecture": run["architecture"], "config_id": config_id,
                             "inference_mode": candidate["inference_mode"],
                             "dropout_p": candidate["dropout_p"], "mc_samples": candidate["mc_samples"],
                             "weight_decay": candidate["weight_decay"], "stage": stage,
                             "trial_id": value.get("trial_id"),
                             "selected": value.get("trial_id") == best_id,
                             "is_noop_baseline": value.get("is_noop_baseline", False),
                             "noop_verified": value.get("noop_verified"),
                             "selected_at_lower_boundary": value.get(
                                 "selected_at_lower_boundary", False),
                             "selected_at_upper_boundary": value.get(
                                 "selected_at_upper_boundary", False),
                             "parameters": json.dumps(value["parameters"], sort_keys=True),
                             **_flatten_metrics("validation", value["validation_metrics"]),
                             **_flatten_metrics("test", value["ordered_test_metrics"])})
        uncertainty = config.get("uncertainty_search")
        if uncertainty:
            for stage in ("adaptive_beta", "accumulated_evidence"):
                payload = uncertainty.get(stage)
                if not payload:
                    continue
                best_id = payload["best"]["trial_id"]
                for value in payload["frontier"]:
                    rows.append({"architecture": run["architecture"], "config_id": config_id,
                                 "inference_mode": candidate["inference_mode"],
                                 "dropout_p": candidate["dropout_p"],
                                 "mc_samples": candidate["mc_samples"],
                                 "weight_decay": candidate["weight_decay"], "stage": stage,
                                 "trial_id": value["trial_id"],
                                 "selected": value["trial_id"] == best_id,
                                 "is_noop_baseline": value.get("is_noop_baseline", False),
                                 "noop_verified": value.get("noop_verified"),
                                 "selected_at_lower_boundary": value.get(
                                     "selected_at_lower_boundary", False),
                                 "selected_at_upper_boundary": value.get(
                                     "selected_at_upper_boundary", False),
                                 "parameters": json.dumps(value["parameters"], sort_keys=True),
                                 **_flatten_metrics("validation", value["validation_metrics"]),
                                 **_flatten_metrics("test", value["ordered_test_metrics"])})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def _frontier_rows(run, pareto=False):
    rows = []
    for candidate_id, stages in run["search"].get("all_trials", {}).items():
        for stage, trials in stages.items():
            for trial in trials:
                flag = "on_pareto_frontier" if pareto else "on_stage_frontier"
                if not trial.get(flag):
                    continue
                rows.append({"architecture": run["architecture"], "candidate_id": candidate_id,
                             "stage": stage, "trial_id": trial.get("trial_id"),
                             "parent_trial_id": trial.get("parent_trial_id"),
                             "parent_stage": trial.get("parent_stage"),
                             "lineage": trial.get("lineage"), "score_v2": trial.get("score_v2"),
                             "is_noop_baseline": trial.get("is_noop_baseline", False),
                             "noop_verified": trial.get("noop_verified"),
                             "selected_at_lower_boundary": trial.get(
                                 "selected_at_lower_boundary", False),
                             "selected_at_upper_boundary": trial.get(
                                 "selected_at_upper_boundary", False),
                             "balanced_accuracy": trial.get("balanced_accuracy"),
                             "transition_window_accuracy": trial.get("transition_window_accuracy"),
                             "mean_transition_delay": trial.get("mean_transition_delay"),
                             "false_transition_rate": trial.get("false_transition_rate")})
    return rows


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
    best = lambda values: max(values, key=lambda row: _score(row.get("cv_selection_score")))
    selections = {
        "feature_nn deterministic winner": by_name.get("feature_nn_deterministic"),
        "feature_nn MC winner": by_name.get("feature_nn_mc"),
        "best uncertainty-aware stage for feature-NN MC": by_name.get("feature_nn_mc"),
        "raw_depth_nn deterministic winner": by_name.get("raw_depth_nn_deterministic"),
        "raw_depth_nn MC winner": by_name.get("raw_depth_nn_mc"),
        "best uncertainty-aware stage for raw-depth-NN MC": by_name.get("raw_depth_nn_mc"),
        "global deterministic winner": best(deterministic) if deterministic else None,
        "global MC winner": best(mc) if mc else None,
        "best MC filter overall": best(mc) if mc else None,
        "global overall winner": best(winners) if winners else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output.with_name(args.output.name + "_winners.csv"), winners)
    _write_csv(args.output.with_name(args.output.name + "_stages.csv"), stage_rows)
    payload = {"approaches": winners, "stage_results": stage_rows, "selections": selections}
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    with args.output.with_name("suite_deployments.json").open("w", encoding="utf-8") as stream:
        json.dump(by_name, stream, indent=2)

    output_dir = args.output.parent
    stage_frontiers = [row for run in runs for row in _frontier_rows(run)]
    pareto_frontiers = [row for run in runs for row in _frontier_rows(run, pareto=True)]
    experiment_a = [row for run in runs for row in run.get("experiment_A_search_results", [])]
    experiment_b, experiment_c = [], []
    selected_a, selected_b, selected_c = {}, {}, {}
    for run in runs:
        architecture = run["architecture"]
        selected_a[architecture] = {**run.get("experiment_A_selected_configs", {
            "deterministic_candidate_id": run["search"]["stage0"]["deterministic_selected"],
            "mc_candidate_id": run["search"]["stage0"]["mc_selected"],
            "selection_split": "ordered_validation",
        }), "run_directory": str(Path(run["result_file"]).parent)}
        selected_b[architecture] = {
            **run.get("experiment_B_selected_configs", {}),
            "run_directory": str(Path(run["result_file"]).parent)}
        selected_c[architecture] = {
            **run.get("experiment_C_selected_configs", {}),
            "run_directory": str(Path(run["result_file"]).parent)}
        b_configurations = run.get("experiment_B_selected_configs", {}).get(
            "configurations", run.get("experiment_B_selected_configs", {}))
        for mode, methods in b_configurations.items():
            if not isinstance(methods, dict):
                continue
            for method, record in methods.items():
                metrics = record.get("validation_metrics", {})
                experiment_b.append({"architecture": architecture, "inference_mode": mode,
                                     "method": method, **_flatten_metrics("validation", metrics)})
        chain = run.get("experiment_C_selected_configs", {}).get("controlled_C_chain", {})
        for stage, record in chain.items():
            if not isinstance(record, dict) or "validation_metrics" not in record:
                continue
            experiment_c.append({"architecture": architecture, "stage": stage,
                                 "parent_id": record.get("parent_id"),
                                 **_flatten_metrics("validation", record["validation_metrics"])})

    for name, rows in (("stage_frontiers", stage_frontiers),
                       ("pareto_frontiers", pareto_frontiers),
                       ("experiment_A_search_results", experiment_a),
                       ("experiment_B_search_results", experiment_b),
                       ("experiment_C_search_results", experiment_c)):
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", rows)
    _write_json(output_dir / "experiment_A_selected_configs.json", selected_a)
    _write_json(output_dir / "experiment_B_selected_configs.json", selected_b)
    _write_json(output_dir / "experiment_C_selected_configs.json", selected_c)

    if winners:
        best_overall = max(winners, key=lambda row: _score(row.get("validation_score_v2")))
        best_score = _score(best_overall.get("validation_score_v2"))
        near = [row for row in winners
                if best_score - _score(row.get("validation_score_v2")) <= 0.01]
        # A missing delay is never preferable to an observed delay.
        best_low_delay = min(
            near, key=lambda row: _score(row.get("validation_mean_transition_delay"), float("inf")))
        _write_json(output_dir / "best_overall_pipeline.json", best_overall)
        _write_json(output_dir / "best_low_delay_pipeline.json", best_low_delay)


if __name__ == "__main__":
    main()
