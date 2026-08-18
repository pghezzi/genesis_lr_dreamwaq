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
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    _print_ranking("Instantaneous structural-test ranking", instantaneous)
    _print_ranking("Bayesian ordered-test ranking", bayesian)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output.with_name(args.output.name + "_instantaneous.csv"), instantaneous)
    _write_csv(args.output.with_name(args.output.name + "_bayesian.csv"), bayesian)
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump({"instantaneous": instantaneous, "bayesian": bayesian}, stream, indent=2)


if __name__ == "__main__":
    main()
