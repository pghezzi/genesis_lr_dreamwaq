"""Compare saved terrain-classifier runs without rerunning training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_METHODS = {"RBF Prototype", "RBF SVM", "feature NN", "raw-depth NN"}


def parse_result_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    required = {"method", "instantaneous", "bayes", "validation_score"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{path} is missing result fields: {sorted(missing)}")
    if "metrics" not in result["bayes"]:
        raise ValueError(f"{path} has no Bayes metrics")
    result["result_file"] = str(path.resolve())
    return result


def discover_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif (path / "results.json").is_file():
            files.append(path / "results.json")
        elif path.is_dir():
            files.extend(path.glob("*/results.json"))
    if not files:
        raise FileNotFoundError("no results.json files found")
    return [parse_result_file(path) for path in sorted(set(files))]


def _row(run: dict[str, Any], bayes: bool) -> dict[str, Any]:
    metrics = run["bayes"]["metrics"] if bayes else run["instantaneous"]
    filter_parameters = run["bayes"].get("filter_parameters", {})
    row = {
        "method": run["method"], "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"), "macro_f1": metrics.get("macro_f1"),
        "confusion_matrix": json.dumps(metrics.get("confusion_matrix")),
        "validation_score": run["bayes"].get("validation_score") if bayes else run.get("validation_score"),
        "nll": metrics.get("nll"), "brier": metrics.get("brier"),
        "mean_transition_delay": metrics.get("mean_transition_delay") if bayes else None,
        "false_transition_rate": metrics.get("false_transition_rate") if bayes else None,
        "best_hyperparameters": json.dumps(run.get("best_hyperparameters", {}), sort_keys=True),
        "selected_temperature": run["bayes"].get("selected_temperature") if bayes else None,
        "stay_probability": filter_parameters.get("stay_probability") if bayes else None,
        "transition_alpha": filter_parameters.get("transition_alpha") if bayes else None,
        "transition_matrix": json.dumps(filter_parameters.get("transition_matrix")) if bayes else None,
        "filter_parameters": json.dumps(filter_parameters, sort_keys=True) if bayes else None,
        "search_or_training_runtime": run.get("runtime_seconds", {}).get("search_or_training"),
        "bayes_search_runtime": run.get("runtime_seconds", {}).get("bayes_search"),
        "model_size_bytes": run.get("model", {}).get("model_size_bytes"),
        "prototype_count": run.get("model", {}).get("prototype_count"),
        "kernel_basis_size": run.get("model", {}).get("kernel_basis_size"),
        "parameter_count": run.get("model", {}).get("parameter_count"),
        "result_file": run["result_file"],
    }
    unfiltered = run["bayes"].get("unfiltered_metrics", run["instantaneous"])
    row["filter_accuracy_delta"] = (
        metrics.get("accuracy", 0.0) - unfiltered.get("accuracy", 0.0) if bayes else None
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_ranking(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    for rank, row in enumerate(sorted(rows, key=lambda item: item["balanced_accuracy"], reverse=True), 1):
        suffix = ""
        if row["filter_accuracy_delta"] is not None:
            suffix = f"  filter Δacc={row['filter_accuracy_delta']:+.4f}"
        print(
            f"{rank:2d}. {row['method']:<16} bal_acc={row['balanced_accuracy']:.4f} "
            f"acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}{suffix}"
        )


def _instantaneous_stage_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    stages = run.get("classifier_search_stages")
    if not stages:
        return [{**_row(run, False), "stage": "legacy", "selected_across_stages": True,
                 "structural_validation_accuracy": run.get("validation_score")}]
    rows = []
    selected = stages.get("selected_stage", "base")
    for name in ("stage1", "stage2", "base"):
        if name not in stages:
            continue
        stage = stages[name]
        validation = stage.get("validation_metrics", {})
        test = stage.get("structural_test_metrics", run.get("instantaneous", {}))
        rows.append({
            "method": run["method"], "stage": name,
            "selected_across_stages": name == selected,
            "structural_validation_accuracy": validation.get("accuracy"),
            "structural_validation_nll": validation.get("nll"),
            "structural_validation_brier": validation.get("brier"),
            "structural_test_accuracy": test.get("accuracy"),
            "structural_test_balanced_accuracy": test.get("balanced_accuracy"),
            "structural_test_macro_f1": test.get("macro_f1"),
            "parameters": json.dumps(stage.get("best_params", {}), sort_keys=True),
            "model_metadata": json.dumps(stage.get("model_metadata", {}), sort_keys=True),
            "result_file": run["result_file"],
        })
    selected_row = next((row for row in rows if row["stage"] == selected), None)
    if selected_row is not None and selected in {"stage1", "stage2"}:
        rows.append({**selected_row, "stage": "selected", "selected_from_stage": selected,
                     "selected_across_stages": True})
    return rows


def _temporal_row(run: dict[str, Any], family: str, record: dict[str, Any]) -> dict[str, Any]:
    validation = record.get("best_validation", {})
    test = record.get("test_metrics", {})
    return {
        "method": run["method"], "family": family,
        "validation_selection_score": validation.get("selection_score"),
        "ordered_test_accuracy": test.get("accuracy"),
        "ordered_test_balanced_accuracy": test.get("balanced_accuracy"),
        "ordered_test_macro_f1": test.get("macro_f1"),
        "transition_window_accuracy": test.get("transition_window_accuracy"),
        "steady_state_accuracy": test.get("steady_state_accuracy"),
        "mean_transition_delay": test.get("mean_transition_delay"),
        "false_transition_rate": test.get("false_transition_rate"),
        "minimum_class_recall": test.get("minimum_class_recall"),
        "missing_predicted_classes": json.dumps(test.get("missing_predicted_classes", [])),
        "per_class_precision": json.dumps(test.get("per_class_precision", {}), sort_keys=True),
        "per_class_recall": json.dumps(test.get("per_class_recall", {}), sort_keys=True),
        "confusion_matrix": json.dumps(test.get("confusion_matrix")),
        "parameters": json.dumps(record.get("parameters", {}), sort_keys=True),
        "result_file": run["result_file"],
    }


def _sequential_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    search = run.get("sequential_search")
    if search:
        return [_temporal_row(run, family, record)
                for family, record in search.get("stages", {}).items()]
    # Legacy Bayes-only runs remain parseable and appear as one temporal row.
    bayes = run["bayes"]
    return [_temporal_row(run, "legacy_bayes", {
        "best_validation": {"selection_score": bayes.get("validation_score")},
        "test_metrics": bayes.get("metrics", {}),
        "parameters": bayes.get("filter_parameters", {}),
    })]


def _best(rows: list[dict[str, Any]], families: set[str] | None = None) -> dict[str, Any] | None:
    eligible = [row for row in rows if families is None or row["family"] in families]
    eligible = [row for row in eligible if row.get("validation_selection_score") is not None]
    return max(eligible, key=lambda row: row["validation_selection_score"]) if eligible else None


def _instantaneous_key(row: dict[str, Any]) -> tuple[float, float, float]:
    accuracy = row.get("structural_validation_accuracy")
    nll = row.get("structural_validation_nll")
    brier = row.get("structural_validation_brier")
    return (float("-inf") if accuracy is None else accuracy,
            float("-inf") if nll is None else -nll,
            float("-inf") if brier is None else -brier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories, results.json files, or a parent directory")
    parser.add_argument("--output", type=Path, default=Path("terrain_classifier_comparison"))
    args = parser.parse_args()
    runs = discover_results(args.runs)
    present = {run["method"] for run in runs}
    missing = REQUIRED_METHODS - present
    if missing:
        print("Warning: no saved run supplied for " + ", ".join(sorted(missing)))
    instantaneous = [_row(run, False) for run in runs]
    bayesian = [_row(run, True) for run in runs]
    instantaneous_stages = [row for run in runs for row in _instantaneous_stage_rows(run)]
    temporal_stages = [row for run in runs for row in _sequential_rows(run)]
    bayes_families = {"stage1_fixed_bayes", "stage2_event_bayes", "stage3_ambiguity_bayes", "legacy_bayes"}
    ema_families = {"ema_logit_patience"}
    per_classifier = []
    for run in runs:
        rows = _sequential_rows(run)
        per_classifier.append({"method": run["method"], "result_file": run["result_file"],
                               "best_bayes": _best(rows, bayes_families),
                               "best_ema": _best(rows, ema_families),
                               "best_temporal_overall": _best(rows)})
    rbf_stage_rows = [r for r in instantaneous_stages if r["method"] in {"RBF Prototype", "RBF SVM"}]
    comparable_stage_bests = {}
    for stage in ("stage1", "stage2"):
        candidates = [r for r in rbf_stage_rows if r["stage"] == stage]
        if candidates:
            comparable_stage_bests[stage] = max(candidates, key=_instantaneous_key)
    global_selection = {
        "best_rbf_per_stage": comparable_stage_bests,
        "best_rbf_instantaneous_overall": max(
            [r for r in rbf_stage_rows if r["stage"] in {"stage1", "stage2"}],
            key=_instantaneous_key
        ) if rbf_stage_rows else None,
        "best_bayes": _best(temporal_stages, bayes_families),
        "best_ema": _best(temporal_stages, ema_families),
        "best_temporal_overall": _best(temporal_stages),
    }
    _print_ranking("Instantaneous structural-test ranking", instantaneous)
    _print_ranking("Bayesian ordered-test ranking", bayesian)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output.with_name(args.output.name + "_instantaneous.csv"), instantaneous)
    _write_csv(args.output.with_name(args.output.name + "_bayesian.csv"), bayesian)
    _write_csv(args.output.with_name(args.output.name + "_instantaneous_stages.csv"), instantaneous_stages)
    _write_csv(args.output.with_name(args.output.name + "_sequential_stages.csv"), temporal_stages)
    flat_best = []
    for item in per_classifier:
        for kind in ("best_bayes", "best_ema", "best_temporal_overall"):
            if item[kind]:
                flat_best.append({"selection": kind, **item[kind]})
    _write_csv(args.output.with_name(args.output.name + "_per_classifier_best.csv"), flat_best)
    with args.output.with_name(args.output.name + "_staged.json").open("w", encoding="utf-8") as stream:
        json.dump({"instantaneous_stages": instantaneous_stages,
                   "sequential_stages": temporal_stages,
                   "per_classifier_best": per_classifier,
                   "global_best": global_selection}, stream, indent=2)
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump({"instantaneous": instantaneous, "bayesian": bayesian,
                   "instantaneous_stages": instantaneous_stages,
                   "sequential_stages": temporal_stages,
                   "per_classifier_best": per_classifier,
                   "global_best": global_selection}, stream, indent=2)


if __name__ == "__main__":
    main()
