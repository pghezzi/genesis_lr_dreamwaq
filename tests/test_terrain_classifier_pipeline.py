"""Small CPU sanity tests for the terrain-classification training utilities."""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results import parse_result_file
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    FeatureStandardizer, NeuralClassifierAdapter, RBFSVM, fit_nn,
    build_filter_from_search_result, build_hybrid_transition_cache,
    collect_classifier_scores_batched, search_bayes_filter_hyperparameters,
    search_rbf_svm_hyperparameters,
)


def _toy_data():
    torch.manual_seed(4)
    x0 = torch.randn(20, 3) - 1.5
    x1 = torch.randn(20, 3) + 1.5
    return torch.cat((x0, x1)), ["flat"] * 20 + ["rough"] * 20


def test_streaming_standardizer_and_save_load(tmp_path):
    x = torch.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    loader = DataLoader(TensorDataset(x, torch.arange(3)), batch_size=2)
    standardizer = FeatureStandardizer(eps=1e-4).fit(loader)
    transformed = standardizer.transform(x)
    assert torch.allclose(transformed.mean(0), torch.zeros(2), atol=1e-6)
    assert standardizer.std[1] == 1e-4
    path = tmp_path / "standardizer.pt"
    standardizer.save(path)
    restored = FeatureStandardizer.load(path)
    assert torch.allclose(restored.transform(x), transformed)


def test_rbf_svm_inference_and_save_load(tmp_path):
    x, labels = _toy_data()
    model = RBFSVM(3, gamma="scale", max_kernel_samples=16, epochs=8,
                   batch_size=16, early_stopping_patience=3)
    model.fit(x, labels, validation_inputs=x, validation_labels=labels)
    before = model.decision_function(x[:5])
    path = tmp_path / "svm.pt"
    model.save(path)
    restored = RBFSVM.load(path)
    assert restored.require_feature
    assert restored.class_ids == model.class_ids
    assert torch.allclose(restored.decision_function(x[:5]), before)


def test_fit_nn_callback_supports_feature_and_image_batches():
    for inputs, model in (
        (torch.randn(12, 4), nn.Linear(4, 2)),
        (torch.randn(12, 1, 3, 3), nn.Sequential(nn.Flatten(), nn.Linear(9, 2))),
    ):
        labels = ["a", "b"] * 6
        adapter = NeuralClassifierAdapter(model, ["a", "b"], fit_callback=fit_nn)
        adapter.fit(inputs, labels, val=(inputs, labels), epochs=2, batch_size=4, verbose=False)
        assert len(adapter.training_history["train_loss"]) == 2
        assert len(adapter.predict(inputs[:2])) == 2


def test_svm_search_reuses_basis_selection(monkeypatch):
    x, labels = _toy_data()
    calls = 0
    original = RBFSVM._stratified_basis_indices

    def counted(self, y, maximum):
        nonlocal calls
        calls += 1
        return original(self, y, maximum)

    monkeypatch.setattr(RBFSVM, "_stratified_basis_indices", counted)
    search_rbf_svm_hyperparameters(
        x, labels, x, labels,
        base_config={"feature_dim": 3, "epochs": 2, "early_stopping_patience": 1},
        search_space={"gamma": [0.1, 0.2], "max_kernel_samples": [8]},
    )
    assert calls == 1  # once for the shared distance/basis cache, not once per trial


def test_result_parser(tmp_path):
    result = {
        "method": "RBF SVM", "validation_score": 0.8,
        "instantaneous": {"accuracy": 0.7},
        "bayes": {"metrics": {"accuracy": 0.75}},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    parsed = parse_result_file(tmp_path)
    assert parsed["method"] == "RBF SVM"
    assert parsed["result_file"].endswith("results.json")


def test_hybrid_transition_cache_deduplicates_empirical_endpoint(tmp_path, monkeypatch):
    labels = [0, 1]
    truth = [0, 0, 1, 1, 0, 0]
    sequence_ids = [0] * len(truth)
    cache = build_hybrid_transition_cache(
        labels, truth, transition_training_sequence_ids=sequence_ids,
        stay_probabilities=[0.9, 0.97], transition_alphas=[0.0, 0.5, 1.0],
    )
    assert len(cache.matrices_by_parameters) == 6
    assert len(cache.unique_candidates) == 5  # alpha=1 is identical for both stays

    x = torch.tensor([[-1.0], [-0.5], [0.5], [1.0], [-0.7], [0.7]])
    model = RBFSVM(1, gamma=1.0, epochs=2, max_kernel_samples=None)
    model.fit(x, truth)
    scores = collect_classifier_scores_batched(model, x, batch_size=2)
    monkeypatch.setattr(
        model, "decision_function",
        lambda _: (_ for _ in ()).throw(AssertionError("search repeated classifier inference")),
    )
    results = search_bayes_filter_hyperparameters(
        model, None, truth, {0: 0.5, 1: 0.5}, sequence_ids=sequence_ids,
        filter_scores=scores,
        temperatures=[1.0], transition_cache=cache,
        evidence_powers=[0.5], min_evidence_powers=[0.1], confidence_gammas=[1.0],
    )
    assert len(results) == 5
    assert all("transition_alpha" in result and "transition_matrix" in result for result in results)
    filt, _ = build_filter_from_search_result(model, results[0], {0: 0.5, 1: 0.5})
    assert torch.allclose(filt.transition_matrix.cpu(), results[0]["transition_matrix"])
    checkpoint = tmp_path / "filter.pt"
    filt.save(checkpoint)
    restored = type(filt).load(checkpoint)
    assert restored.transition_source == "hybrid"
    assert restored.transition_alpha == results[0]["transition_alpha"]
