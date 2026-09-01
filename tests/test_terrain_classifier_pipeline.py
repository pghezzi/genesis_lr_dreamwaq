"""Small CPU sanity tests for the terrain-classification training utilities."""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results import parse_result_file
from legged_gym.scripts.depth_data_pipeline.util_func import collect_neural_logits_batched
from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import (
    CandidateReleaseBayesianTerrainFilter, accumulate_transition_evidence,
    FILTER_STAYS, load_temporal_trace, recompute_temporal_trace_metrics,
    mc_candidate_agreement, run_candidate_release_sequences,
    save_release_temporal_trace, select_stage_frontier,
    uncertainty_adaptive_beta, uncertainty_error_auroc,
    validation_mi_threshold,
)
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


def test_batched_mc_dropout_shapes_and_masks():
    model = nn.Sequential(nn.Linear(4, 12), nn.ELU(), nn.Dropout(0.5), nn.Linear(12, 3))
    adapter = NeuralClassifierAdapter(model, [0, 1, 2])
    inputs = torch.ones(5, 4)
    for samples in (10, 25, 50):
        logits, _ = collect_neural_logits_batched(
            adapter, inputs, batch_size=5, mc_samples=samples,
            mc_dropout=True, expanded_batch_size=64)
        assert logits.shape == (samples, 5, 3)
        assert not torch.allclose(logits[0], logits[1])
    deterministic, _ = collect_neural_logits_batched(adapter, inputs, batch_size=5)
    assert deterministic.shape == (1, 5, 3)
    assert torch.allclose(deterministic[0, 0], deterministic[0, 1])


def test_candidate_release_runner_resets_each_sequence():
    labels = [0, 1]
    filt = CandidateReleaseBayesianTerrainFilter(
        labels, {0: 0.5, 1: 0.5}, torch.tensor([[0.99, 0.01], [0.01, 0.99]]),
        release_strength=0.5, switch_margin=0.1, change_patience=1)
    probabilities = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    predictions, _, _, _ = run_candidate_release_sequences(
        filt, probabilities, probabilities, [0, 0, 1, 1])
    assert predictions[0] == 1
    assert predictions[2] == 0


def test_mc_agreement_mi_threshold_and_beta_endpoints():
    probabilities = torch.tensor([
        [[0.9, 0.1], [0.6, 0.4]],
        [[0.8, 0.2], [0.2, 0.8]],
        [[0.7, 0.3], [0.1, 0.9]],
        [[0.4, 0.6], [0.7, 0.3]],
    ])
    assert torch.allclose(mc_candidate_agreement(probabilities), torch.tensor([0.75, 0.5]))
    mi = torch.tensor([0.0, 0.1, 0.2, 0.3])
    assert validation_mi_threshold(mi, 100) == float(mi.max())
    assert torch.allclose(uncertainty_adaptive_beta(torch.tensor([0.0, 0.2]), 0.2, 0.25),
                          torch.tensor([1.0, 0.25]))
    assert torch.allclose(uncertainty_adaptive_beta(mi, 0.2, 1.0), torch.ones_like(mi))


def test_accumulated_evidence_candidate_changes_and_reset(tmp_path):
    value, candidate = accumulate_transition_evidence(0.4, 1, 1, 0.2, 0.5)
    assert value == 0.4 and candidate == 1
    value, candidate = accumulate_transition_evidence(value, candidate, 2, 0.1, 0.5)
    assert value == 0.1 and candidate == 2
    assert accumulate_transition_evidence(value, candidate, 2, 0.0, 0.5, False) == (0.0, None)

    filt = CandidateReleaseBayesianTerrainFilter(
        [0, 1], {0: 0.5, 1: 0.5}, torch.eye(2), release_strength=0.8,
        change_patience=99, epistemic_threshold=1.0,
        mi_scale=1.0, use_accumulated_evidence=True, evidence_decay=0.5,
        evidence_threshold=0.1)
    filt.update(torch.tensor([0.2, 0.8]), event_probabilities=torch.tensor([0.2, 0.8]),
                mutual_information=0.0)
    assert filt.last_event  # U3 triggers without applying discrete patience.
    filt.reset()
    assert filt.accumulated_evidence == 0.0 and filt.evidence_candidate_index is None
    path = tmp_path / "u3_filter.pt"
    filt.save(path)
    restored = CandidateReleaseBayesianTerrainFilter.load(path)
    assert restored.use_accumulated_evidence and restored.change_patience == 99


def test_adaptive_beta_one_matches_unattenuated_observation():
    kwargs = dict(labels=[0, 1], prior={0: 0.5, 1: 0.5},
                  stable_transition_matrix=torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
    baseline = CandidateReleaseBayesianTerrainFilter(**kwargs)
    adaptive_endpoint = CandidateReleaseBayesianTerrainFilter(
        **kwargs, beta_min=1.0, mi_scale=0.2)
    probability = torch.tensor([0.3, 0.7])
    baseline.update(probability, mutual_information=0.2)
    adaptive_endpoint.update(probability, mutual_information=0.2)
    assert torch.allclose(baseline.belief, adaptive_endpoint.belief)


def test_uncertainty_error_auroc_and_structured_frontier():
    assert uncertainty_error_auroc(
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
        torch.tensor([False, False, True, True])) == 1.0
    trials = []
    for index, (score, delay, transition, false_rate) in enumerate((
        (0.80, 4.0, 0.70, 0.04), (0.795, 1.0, 0.68, 0.06),
        (0.794, 3.0, 0.80, 0.05), (0.793, 2.0, 0.72, 0.01),
    )):
        trials.append({"trial_id": f"t{index}", "score_v2": score,
                       "selection_score": score, "balanced_accuracy": 0.8,
                       "transition_window_accuracy": transition,
                       "mean_transition_delay": delay,
                       "false_transition_rate": false_rate,
                       "stable_stay": 0.9 + index * 0.01})
    frontier = select_stage_frontier(trials)
    assert {trial["trial_id"] for trial in frontier} == {"t0", "t1", "t2", "t3"}
    assert all(trial["on_stage_frontier"] for trial in frontier)


def test_ambiguity_preserves_patience_and_decays_evidence():
    common = dict(labels=[0, 1, 2], prior={0: 0.8, 1: 0.1, 2: 0.1},
                  stable_transition_matrix=torch.eye(3), release_strength=0.2,
                  switch_margin=0.05, ambiguity_margin=0.2, flatten_strength=0.0)
    patience_filter = CandidateReleaseBayesianTerrainFilter(**common, change_patience=2)
    patience_filter.update(torch.tensor([0.3, 0.6, 0.1]),
                           event_probabilities=torch.tensor([0.3, 0.6, 0.1]))
    pending = patience_filter.pending_target_index
    count = patience_filter.pending_count
    patience_filter.update(torch.tensor([0.45, 0.50, 0.05]),
                           event_probabilities=torch.tensor([0.45, 0.50, 0.05]))
    assert patience_filter.pending_target_index == pending
    assert patience_filter.pending_count == count

    evidence_filter = CandidateReleaseBayesianTerrainFilter(
        **common, mi_scale=1.0, use_accumulated_evidence=True,
        evidence_decay=0.5, evidence_threshold=2.0)
    evidence_filter.update(torch.tensor([0.3, 0.6, 0.1]),
                           event_probabilities=torch.tensor([0.3, 0.6, 0.1]))
    evidence = evidence_filter.accumulated_evidence
    evidence_filter.update(torch.tensor([0.45, 0.50, 0.05]),
                           event_probabilities=torch.tensor([0.45, 0.50, 0.05]))
    assert evidence_filter.accumulated_evidence == evidence * 0.5

    uncertain_filter = CandidateReleaseBayesianTerrainFilter(
        **{**common, "ambiguity_margin": 0.01},
        epistemic_threshold=0.05, mi_scale=1.0,
        use_accumulated_evidence=True, evidence_decay=0.5,
        evidence_threshold=2.0)
    uncertain_filter.update(torch.tensor([0.40, 0.55, 0.05]),
                            event_probabilities=torch.tensor([0.40, 0.55, 0.05]),
                            mutual_information=0.0)
    evidence = uncertain_filter.accumulated_evidence
    uncertain_filter.update(torch.tensor([0.40, 0.55, 0.05]),
                            event_probabilities=torch.tensor([0.40, 0.55, 0.05]),
                            mutual_information=0.1)
    assert uncertain_filter.accumulated_evidence == evidence * 0.5


def test_compact_temporal_trace_round_trip_and_offline_metrics(tmp_path):
    assert 0.997 in FILTER_STAYS
    labels = [0, 1]
    truth = [0, 0, 1, 1, 1, 0]
    sequence_ids = [0, 0, 0, 0, 1, 1]
    logits = torch.tensor([
        [2.0, -1.0], [1.5, -0.5], [-0.5, 1.5], [-1.0, 2.0],
        [-1.0, 2.0], [2.0, -1.0],
    ])
    trial = {
        "trial_id": "trace-test", "parent_trial_id": None,
        "stage": "stage1_fixed_bayes", "family": "fixed_bayes",
        "lineage": "fixed_bayes", "T_filter": 1.0,
        "stable_stay": 0.997,
    }
    path = tmp_path / "trace.pt"
    save_release_temporal_trace(
        path, trial, logits, truth, sequence_ids, labels,
        inference_mode="deterministic", mc_samples=1)
    trace = load_temporal_trace(path)
    assert len(trace["sequence_id"]) == len(truth)
    assert trace["true_transition_mask"].tolist() == [False, False, True, False, False, True]
    assert trace["sequence_frame_index"].tolist() == [0, 1, 2, 3, 0, 1]
    assert torch.allclose(trace["filtered_posterior"].sum(1), torch.ones(len(truth)))
    assert torch.isnan(trace["mutual_information"]).all()
    assert torch.isnan(trace["predictive_entropy"]).all()
    metrics = recompute_temporal_trace_metrics(path, transition_window_radius=2)
    assert "confusion_matrix" in metrics and "mean_transition_delay" in metrics


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
        "method": "feature NN", "architecture": "feature_nn",
        "deployments": {"deterministic": {}, "mc": {}}, "search": {},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    parsed = parse_result_file(tmp_path)
    assert parsed["architecture"] == "feature_nn"
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
