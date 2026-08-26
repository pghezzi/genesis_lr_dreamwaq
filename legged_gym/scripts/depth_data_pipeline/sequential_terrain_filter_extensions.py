"""
Sequential terrain-filter extensions.

Contains:
  - EventConditionedBayesianTerrainFilter
  - AmbiguityAwareEventConditionedBayesianTerrainFilter
  - EMALogitPatienceFilter
  - sequence runners for the two Bayes extensions

These classes build on the existing BayesianTerrainFilter API.
"""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Hashable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    BayesianFilterStep,
    BayesianTerrainFilter,
    estimate_observation_matrix_from_probabilities,
    evaluate_predictions,
    make_persistent_transition_matrix,
    run_filter_sequences,
)


def _row_stochastic(
    matrix: torch.Tensor | Sequence,
    size: int,
    device: torch.device,
    eps: float,
    name: str,
) -> torch.Tensor:
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    if x.shape != (size, size):
        raise ValueError(f"{name} must have shape [{size},{size}]")
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    x = x.clamp_min(eps)
    return x / x.sum(dim=1, keepdim=True).clamp_min(eps)


class EventConditionedBayesianTerrainFilter(BayesianTerrainFilter):
    """
    Bayes filter with high persistence during stable operation and a separate,
    lower-persistence transition model during a detected terrain-change event.

    A change event is proposed when the instantaneous classifier's argmax differs
    from the currently emitted filtered state and its maximum probability exceeds
    `switch_confidence`. The same candidate must persist for `change_patience`
    consecutive clear frames before the low-persistence transition matrix is used.

    Patience gates the transition-model release; normal Bayes evidence is still
    processed on clear frames before patience has elapsed.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
        stable_transition_matrix: torch.Tensor | Sequence,
        switch_transition_matrix: torch.Tensor | Sequence,
        observation_matrix: Optional[torch.Tensor | Sequence] = None,
        *,
        switch_confidence: float = 0.70,
        change_patience: int = 2,
        evidence_power: float = 1.0,
        adaptive_evidence: bool = False,
        min_evidence_power: float = 1.0,
        confidence_gamma: float = 1.0,
        device: str | torch.device = "cpu",
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= switch_confidence <= 1.0:
            raise ValueError("switch_confidence must be in [0,1]")
        if int(change_patience) < 1:
            raise ValueError("change_patience must be >= 1")

        super().__init__(
            labels=labels,
            prior=prior,
            transition_matrix=stable_transition_matrix,
            observation_matrix=observation_matrix,
            evidence_power=evidence_power,
            adaptive_evidence=adaptive_evidence,
            min_evidence_power=min_evidence_power,
            confidence_gamma=confidence_gamma,
            device=device,
            eps=eps,
        )

        self.stable_transition_matrix = self.transition_matrix.clone()
        self.switch_transition_matrix = _row_stochastic(
            switch_transition_matrix,
            self.num_classes,
            self.device,
            self.eps,
            "switch_transition_matrix",
        )
        self.switch_confidence = float(switch_confidence)
        self.change_patience = int(change_patience)

        self.pending_target_index: Optional[int] = None
        self.pending_count = 0
        self.last_used_switch_transition = False
        self.current_output_index = int(self.belief.argmax())

    def _normalize_classifier_probabilities(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        q = torch.as_tensor(
            classifier_probabilities,
            dtype=torch.float32,
            device=self.device,
        ).flatten()
        if q.numel() != self.num_classes:
            raise ValueError(
                f"classifier_probabilities must have {self.num_classes} entries"
            )
        if not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError("classifier_probabilities must be finite and non-negative")
        q = q.clamp_min(self.eps)
        return q / q.sum().clamp_min(self.eps)

    def _select_transition_matrix(self, q: torch.Tensor) -> bool:
        candidate_index = int(q.argmax())
        candidate_confidence = float(q[candidate_index])

        disagreement = (
            candidate_index != self.current_output_index
            and candidate_confidence >= self.switch_confidence
        )

        if disagreement:
            if self.pending_target_index == candidate_index:
                self.pending_count += 1
            else:
                self.pending_target_index = candidate_index
                self.pending_count = 1
        else:
            self.pending_target_index = None
            self.pending_count = 0

        use_switch = disagreement and self.pending_count >= self.change_patience
        self.transition_matrix = (
            self.switch_transition_matrix
            if use_switch
            else self.stable_transition_matrix
        )
        self.last_used_switch_transition = bool(use_switch)
        return bool(use_switch)

    @torch.inference_mode()
    def update(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
        *,
        observation_quality: float = 1.0,
    ) -> BayesianFilterStep:
        q = self._normalize_classifier_probabilities(classifier_probabilities)
        previous_output = self.current_output_index

        self._select_transition_matrix(q)
        step = super().update(q, observation_quality=observation_quality)

        new_output = int(step.posterior.argmax())
        self.current_output_index = new_output

        if new_output != previous_output:
            self.pending_target_index = None
            self.pending_count = 0

        # Restore stable T after each step. _select_transition_matrix() will
        # explicitly reactivate switch T on the next qualifying frame.
        self.transition_matrix = self.stable_transition_matrix

        return BayesianFilterStep(
            self.labels[self.current_output_index],
            step.posterior,
            step.predicted_prior,
            step.observation_likelihood,
            step.classifier_probabilities,
            step.confidence,
            step.evidence_power,
        )

    def reset(
        self,
        prior: Optional[
            torch.Tensor | Sequence[float] | Mapping[Hashable, float]
        ] = None,
    ) -> torch.Tensor:
        belief = super().reset(prior)
        self.transition_matrix = self.stable_transition_matrix
        self.pending_target_index = None
        self.pending_count = 0
        self.last_used_switch_transition = False
        self.current_output_index = int(self.belief.argmax())
        return belief

    def _base_state_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "initial_prior": self.initial_prior.cpu(),
            "belief": self.belief.cpu(),
            "stable_transition_matrix": self.stable_transition_matrix.cpu(),
            "switch_transition_matrix": self.switch_transition_matrix.cpu(),
            "observation_matrix": self.observation_matrix.cpu(),
            "switch_confidence": self.switch_confidence,
            "change_patience": self.change_patience,
            "evidence_power": self.evidence_power,
            "adaptive_evidence": self.adaptive_evidence,
            "min_evidence_power": self.min_evidence_power,
            "confidence_gamma": self.confidence_gamma,
            "pending_target_index": self.pending_target_index,
            "pending_count": self.pending_count,
            "current_output_index": self.current_output_index,
            "eps": self.eps,
        }

    def save(self, path: str | Path) -> None:
        state = self._base_state_dict()
        state["filter_class"] = self.__class__.__name__
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "EventConditionedBayesianTerrainFilter":
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)

        filt = cls(
            labels=state["labels"],
            prior=state["initial_prior"],
            stable_transition_matrix=state["stable_transition_matrix"],
            switch_transition_matrix=state["switch_transition_matrix"],
            observation_matrix=state["observation_matrix"],
            switch_confidence=state["switch_confidence"],
            change_patience=state["change_patience"],
            evidence_power=state["evidence_power"],
            adaptive_evidence=state["adaptive_evidence"],
            min_evidence_power=state["min_evidence_power"],
            confidence_gamma=state["confidence_gamma"],
            device=device,
            eps=state["eps"],
        )
        filt.belief = state["belief"].to(filt.device)
        filt.pending_target_index = state.get("pending_target_index")
        filt.pending_count = int(state.get("pending_count", 0))
        filt.current_output_index = int(
            state.get("current_output_index", int(filt.belief.argmax()))
        )
        return filt


class AmbiguityAwareEventConditionedBayesianTerrainFilter(
    EventConditionedBayesianTerrainFilter
):
    """
    Event-conditioned Bayes filter with explicit ambiguous-frame handling.

    A frame is ambiguous when the top-two instantaneous classifier probabilities
    differ by <= `ambiguity_margin_threshold`.

    On an ambiguous frame:
      1) keep the current emitted terrain label;
      2) reset event-switch patience;
      3) do not apply class evidence from the ambiguous observation;
      4) propagate with the stable transition matrix;
      5) flatten the propagated belief toward the initial prior:

           b <- (1-lambda) * predicted + lambda * prior

    This allows uncertainty to reduce accumulated certainty without making
    uncertainty itself evidence for the previous terrain.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
        stable_transition_matrix: torch.Tensor | Sequence,
        switch_transition_matrix: torch.Tensor | Sequence,
        observation_matrix: Optional[torch.Tensor | Sequence] = None,
        *,
        switch_confidence: float = 0.70,
        change_patience: int = 2,
        ambiguity_margin_threshold: float = 0.20,
        ambiguity_flatten_strength: float = 0.25,
        evidence_power: float = 1.0,
        adaptive_evidence: bool = False,
        min_evidence_power: float = 1.0,
        confidence_gamma: float = 1.0,
        device: str | torch.device = "cpu",
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= ambiguity_margin_threshold <= 1.0:
            raise ValueError("ambiguity_margin_threshold must be in [0,1]")
        if not 0.0 <= ambiguity_flatten_strength <= 1.0:
            raise ValueError("ambiguity_flatten_strength must be in [0,1]")

        super().__init__(
            labels=labels,
            prior=prior,
            stable_transition_matrix=stable_transition_matrix,
            switch_transition_matrix=switch_transition_matrix,
            observation_matrix=observation_matrix,
            switch_confidence=switch_confidence,
            change_patience=change_patience,
            evidence_power=evidence_power,
            adaptive_evidence=adaptive_evidence,
            min_evidence_power=min_evidence_power,
            confidence_gamma=confidence_gamma,
            device=device,
            eps=eps,
        )

        self.ambiguity_margin_threshold = float(ambiguity_margin_threshold)
        self.ambiguity_flatten_strength = float(ambiguity_flatten_strength)
        self.last_was_ambiguous = False
        self.last_top2_margin = 1.0

    def top2_margin(self, q: torch.Tensor) -> float:
        if self.num_classes <= 1:
            return 1.0
        values = torch.topk(q, k=2).values
        return float(values[0] - values[1])

    @torch.inference_mode()
    def update(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
        *,
        observation_quality: float = 1.0,
    ) -> BayesianFilterStep:
        q = self._normalize_classifier_probabilities(classifier_probabilities)
        margin = self.top2_margin(q)

        self.last_top2_margin = margin
        self.last_was_ambiguous = margin <= self.ambiguity_margin_threshold

        if not self.last_was_ambiguous:
            return super().update(q, observation_quality=observation_quality)

        # Ambiguous frame: it cannot contribute to a pending state switch.
        self.pending_target_index = None
        self.pending_count = 0
        self.last_used_switch_transition = False
        self.transition_matrix = self.stable_transition_matrix

        predicted = self.belief @ self.stable_transition_matrix
        predicted = predicted / predicted.sum().clamp_min(self.eps)

        lam = self.ambiguity_flatten_strength
        posterior = (1.0 - lam) * predicted + lam * self.initial_prior
        posterior = posterior / posterior.sum().clamp_min(self.eps)
        self.belief = posterior

        # No class-specific evidence was applied. Return an all-ones likelihood
        # and beta=0 for diagnostics while preserving the current output label.
        likelihood = torch.ones(self.num_classes, device=self.device)
        confidence = self.entropy_confidence(q)

        return BayesianFilterStep(
            self.labels[self.current_output_index],
            posterior.clone(),
            predicted.clone(),
            likelihood,
            q.clone(),
            confidence,
            0.0,
        )

    def reset(
        self,
        prior: Optional[
            torch.Tensor | Sequence[float] | Mapping[Hashable, float]
        ] = None,
    ) -> torch.Tensor:
        belief = super().reset(prior)
        self.last_was_ambiguous = False
        self.last_top2_margin = 1.0
        return belief

    def save(self, path: str | Path) -> None:
        state = self._base_state_dict()
        state.update(
            {
                "filter_class": self.__class__.__name__,
                "ambiguity_margin_threshold": self.ambiguity_margin_threshold,
                "ambiguity_flatten_strength": self.ambiguity_flatten_strength,
                "last_was_ambiguous": self.last_was_ambiguous,
                "last_top2_margin": self.last_top2_margin,
            }
        )
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "AmbiguityAwareEventConditionedBayesianTerrainFilter":
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)

        filt = cls(
            labels=state["labels"],
            prior=state["initial_prior"],
            stable_transition_matrix=state["stable_transition_matrix"],
            switch_transition_matrix=state["switch_transition_matrix"],
            observation_matrix=state["observation_matrix"],
            switch_confidence=state["switch_confidence"],
            change_patience=state["change_patience"],
            ambiguity_margin_threshold=state["ambiguity_margin_threshold"],
            ambiguity_flatten_strength=state["ambiguity_flatten_strength"],
            evidence_power=state["evidence_power"],
            adaptive_evidence=state["adaptive_evidence"],
            min_evidence_power=state["min_evidence_power"],
            confidence_gamma=state["confidence_gamma"],
            device=device,
            eps=state["eps"],
        )
        filt.belief = state["belief"].to(filt.device)
        filt.pending_target_index = state.get("pending_target_index")
        filt.pending_count = int(state.get("pending_count", 0))
        filt.current_output_index = int(
            state.get("current_output_index", int(filt.belief.argmax()))
        )
        filt.last_was_ambiguous = bool(state.get("last_was_ambiguous", False))
        filt.last_top2_margin = float(state.get("last_top2_margin", 1.0))
        return filt


class EMALogitPatienceFilter:
    """
    Lightweight non-Bayesian temporal baseline.

    Smooths classifier scores/logits:
        z_bar_t = alpha*z_t + (1-alpha)*z_bar_{t-1}

    and changes the emitted label only after the same new argmax persists for
    `change_patience` consecutive frames. alpha=1 gives a patience-only gate.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        *,
        ema_alpha: float = 0.8,
        change_patience: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0,1]")
        if int(change_patience) < 1:
            raise ValueError("change_patience must be >= 1")
        self.labels = list(labels)
        self.ema_alpha = float(ema_alpha)
        self.change_patience = int(change_patience)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.ema_scores: Optional[torch.Tensor] = None
        self.current_output_index: Optional[int] = None
        self.pending_target_index: Optional[int] = None
        self.pending_count = 0

    @torch.inference_mode()
    def update(self, scores: torch.Tensor | Sequence[float]) -> Hashable:
        z = torch.as_tensor(scores, dtype=torch.float32, device=self.device).flatten()
        if z.numel() != len(self.labels):
            raise ValueError(f"scores must have {len(self.labels)} entries")

        if self.ema_scores is None:
            self.ema_scores = z.clone()
            self.current_output_index = int(self.ema_scores.argmax())
            return self.labels[self.current_output_index]

        a = self.ema_alpha
        self.ema_scores = a * z + (1.0 - a) * self.ema_scores
        candidate = int(self.ema_scores.argmax())

        if candidate == self.current_output_index:
            self.pending_target_index = None
            self.pending_count = 0
        else:
            if self.pending_target_index == candidate:
                self.pending_count += 1
            else:
                self.pending_target_index = candidate
                self.pending_count = 1

            if self.pending_count >= self.change_patience:
                self.current_output_index = candidate
                self.pending_target_index = None
                self.pending_count = 0

        return self.labels[self.current_output_index]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "labels": list(self.labels),
                "ema_alpha": self.ema_alpha,
                "change_patience": self.change_patience,
                "ema_scores": None if self.ema_scores is None else self.ema_scores.cpu(),
                "current_output_index": self.current_output_index,
                "pending_target_index": self.pending_target_index,
                "pending_count": self.pending_count,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "EMALogitPatienceFilter":
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        filt = cls(
            state["labels"],
            ema_alpha=state["ema_alpha"],
            change_patience=state["change_patience"],
            device=device,
        )
        if state.get("ema_scores") is not None:
            filt.ema_scores = state["ema_scores"].to(filt.device)
        filt.current_output_index = state.get("current_output_index")
        filt.pending_target_index = state.get("pending_target_index")
        filt.pending_count = int(state.get("pending_count", 0))
        return filt


@torch.inference_mode()
def run_event_conditioned_filter_sequences(
    terrain_filter: EventConditionedBayesianTerrainFilter,
    classifier_probabilities: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence] = None,
    observation_quality: Optional[torch.Tensor | Sequence[float]] = None,
    prior: Optional[
        torch.Tensor | Sequence[float] | Mapping[Hashable, float]
    ] = None,
):
    probabilities = torch.as_tensor(classifier_probabilities, dtype=torch.float32)
    n = probabilities.shape[0]
    ids = list(sequence_ids) if sequence_ids is not None else [0] * n
    qualities = (
        torch.ones(n)
        if observation_quality is None
        else torch.as_tensor(observation_quality, dtype=torch.float32).flatten()
    )

    predictions, posteriors, evidence_powers, switch_events = [], [], [], []
    previous_id = object()

    for i in range(n):
        if i == 0 or ids[i] != previous_id:
            terrain_filter.reset(prior)
        previous_id = ids[i]

        step = terrain_filter.update(
            probabilities[i],
            observation_quality=float(qualities[i]),
        )
        predictions.append(step.label)
        posteriors.append(step.posterior.cpu())
        evidence_powers.append(step.evidence_power)
        switch_events.append(terrain_filter.last_used_switch_transition)

    return (
        predictions,
        torch.stack(posteriors),
        torch.tensor(evidence_powers),
        torch.tensor(switch_events, dtype=torch.bool),
    )


@torch.inference_mode()
def run_ambiguity_aware_filter_sequences(
    terrain_filter: AmbiguityAwareEventConditionedBayesianTerrainFilter,
    classifier_probabilities: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence] = None,
    observation_quality: Optional[torch.Tensor | Sequence[float]] = None,
    prior: Optional[
        torch.Tensor | Sequence[float] | Mapping[Hashable, float]
    ] = None,
):
    probabilities = torch.as_tensor(classifier_probabilities, dtype=torch.float32)
    n = probabilities.shape[0]
    ids = list(sequence_ids) if sequence_ids is not None else [0] * n
    qualities = (
        torch.ones(n)
        if observation_quality is None
        else torch.as_tensor(observation_quality, dtype=torch.float32).flatten()
    )

    predictions, posteriors, evidence_powers = [], [], []
    switch_events, ambiguous_events, margins = [], [], []
    previous_id = object()

    for i in range(n):
        if i == 0 or ids[i] != previous_id:
            terrain_filter.reset(prior)
        previous_id = ids[i]

        step = terrain_filter.update(
            probabilities[i],
            observation_quality=float(qualities[i]),
        )
        predictions.append(step.label)
        posteriors.append(step.posterior.cpu())
        evidence_powers.append(step.evidence_power)
        switch_events.append(terrain_filter.last_used_switch_transition)
        ambiguous_events.append(terrain_filter.last_was_ambiguous)
        margins.append(terrain_filter.last_top2_margin)

    return (
        predictions,
        torch.stack(posteriors),
        torch.tensor(evidence_powers),
        torch.tensor(switch_events, dtype=torch.bool),
        torch.tensor(ambiguous_events, dtype=torch.bool),
        torch.tensor(margins, dtype=torch.float32),
    )


@torch.inference_mode()
def run_ema_logit_patience_sequences(
    terrain_filter: EMALogitPatienceFilter,
    classifier_scores: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence] = None,
):
    scores = torch.as_tensor(classifier_scores, dtype=torch.float32)
    ids = list(sequence_ids) if sequence_ids is not None else [0] * scores.shape[0]

    predictions = []
    previous_id = object()
    for i in range(scores.shape[0]):
        if i == 0 or ids[i] != previous_id:
            terrain_filter.reset()
        previous_id = ids[i]
        predictions.append(terrain_filter.update(scores[i]))
    return predictions


# Shared staged search used by every terrain-classifier trainer.  Keeping the
# recurrence here also keeps the deployable filter implementations and their
# search semantics in one place.
SEQUENTIAL_METRIC_CONFIG = {
    "transition_window_radius": 5,
    "transition_accuracy_weight": 0.35,
    "delay_weight": 0.002,
    "false_transition_weight": 0.05,
    "missing_class_penalty": 0.05,
}
TEMPERATURES = (0.25, 0.40, 0.50, 0.75, 1.0, 1.5)
OBSERVATION_MIXES = (0.0, 0.25)
STABLE_STAYS = (0.70, 0.85, 0.95, 0.98)
SWITCH_STAYS = (0.10, 0.30, 0.50, 0.70)
SWITCH_CONFIDENCES = (0.65, 0.80)
CHANGE_PATIENCES = (1, 2)
AMBIGUITY_THRESHOLDS = (0.10, 0.20, 0.30)
AMBIGUITY_FLATTEN_STRENGTHS = (0.10, 0.25, 0.40)
EMA_ALPHAS = (0.40, 0.60, 0.80, 1.0)
EMA_PATIENCES = (1, 2)
OBSERVATION_PSEUDOCOUNT = 0.5


def _transition_window_mask(truth: Sequence, sequence_ids: Sequence, radius: int) -> torch.Tensor:
    ids, mask = list(sequence_ids), torch.zeros(len(truth), dtype=torch.bool)
    for t in range(1, len(truth)):
        if ids[t] == ids[t - 1] and truth[t] != truth[t - 1]:
            for j in range(max(0, t - radius), min(len(truth), t + radius + 1)):
                if ids[j] == ids[t]:
                    mask[j] = True
    return mask


def _ambiguity_run_length(mask: torch.Tensor, sequence_ids: Sequence) -> float:
    ids, runs, run = list(sequence_ids), [], 0
    for i, flag in enumerate(mask.tolist()):
        if i and ids[i] != ids[i - 1]:
            if run:
                runs.append(run)
            run = 0
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return float(sum(runs) / len(runs)) if runs else 0.0


def evaluate_sequential_predictions(
    truth: Sequence, predictions: Sequence, labels: Sequence, sequence_ids: Sequence,
    metric_config: Mapping[str, float] = SEQUENTIAL_METRIC_CONFIG,
) -> dict[str, Any]:
    truth, predictions, labels, ids = list(truth), list(predictions), list(labels), list(sequence_ids)
    base = evaluate_predictions(truth, predictions, labels, sequence_ids=ids)
    matrix = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    index = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(truth, predictions):
        matrix[index[actual], index[predicted]] += 1
    recalls, precisions = {}, {}
    for i, label in enumerate(labels):
        tp = float(matrix[i, i])
        recalls[str(label)] = tp / max(float(matrix[i].sum()), 1.0)
        precisions[str(label)] = tp / max(float(matrix[:, i].sum()), 1.0)
    missing = [str(label) for i, label in enumerate(labels) if int(matrix[:, i].sum()) == 0]
    window = _transition_window_mask(truth, ids, int(metric_config["transition_window_radius"]))
    correct = torch.tensor([a == b for a, b in zip(truth, predictions)], dtype=torch.float32)
    transition_accuracy = float(correct[window].mean()) if bool(window.any()) else float("nan")
    steady_accuracy = float(correct[~window].mean()) if bool((~window).any()) else float("nan")
    transition_term = transition_accuracy if math.isfinite(transition_accuracy) else float(base.balanced_accuracy)
    score = ((1.0 - float(metric_config["transition_accuracy_weight"])) * float(base.balanced_accuracy)
             + float(metric_config["transition_accuracy_weight"]) * transition_term)
    if math.isfinite(float(base.mean_transition_delay)):
        score -= float(metric_config["delay_weight"]) * float(base.mean_transition_delay)
    if math.isfinite(float(base.false_transition_rate)):
        score -= float(metric_config["false_transition_weight"]) * float(base.false_transition_rate)
    score -= float(metric_config["missing_class_penalty"]) * len(missing)
    return {
        "selection_score": float(score), **base.as_dict(),
        "transition_window_accuracy": transition_accuracy,
        "steady_state_accuracy": steady_accuracy,
        "transition_window_frame_fraction": float(window.float().mean()),
        "per_class_recall": recalls, "per_class_precision": precisions,
        "minimum_class_recall": min(recalls.values()) if recalls else float("nan"),
        "missing_predicted_classes": missing, "confusion_matrix": matrix,
    }


def _probs(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    probabilities = F.softmax(torch.as_tensor(scores).cpu() / float(temperature), dim=1).clamp_min(1e-8)
    return probabilities / probabilities.sum(1, keepdim=True)


def _mixed_observation(soft: torch.Tensor, rho: float) -> torch.Tensor:
    return (1.0 - float(rho)) * torch.eye(soft.shape[0]) + float(rho) * soft


def _params(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metric_keys = {
        "selection_score", "accuracy", "balanced_accuracy", "macro_f1",
        "mean_transition_delay", "false_transition_rate", "transition_window_accuracy",
        "steady_state_accuracy", "transition_window_frame_fraction", "per_class_recall",
        "per_class_precision", "minimum_class_recall", "missing_predicted_classes",
        "confusion_matrix", "switch_transition_fraction", "ambiguous_frame_fraction",
        "mean_ambiguity_run_length",
    }
    return {k: v for k, v in candidate.items() if k not in metric_keys and k != "family"}


def _make_filter(candidate: Mapping[str, Any], labels: Sequence, prior: Mapping, observation: torch.Tensor):
    stable = make_persistent_transition_matrix(labels, candidate["stable_stay"], device="cpu")
    family = candidate["family"]
    if family == "stage1_fixed_bayes":
        return BayesianTerrainFilter(labels, prior, stable, observation, evidence_power=1.0,
                                     adaptive_evidence=False, min_evidence_power=1.0,
                                     confidence_gamma=1.0, device="cpu")
    switch = make_persistent_transition_matrix(labels, candidate["switch_stay"], device="cpu")
    common = dict(labels=labels, prior=prior, stable_transition_matrix=stable,
                  switch_transition_matrix=switch, observation_matrix=observation,
                  switch_confidence=candidate["switch_confidence"],
                  change_patience=candidate["change_patience"], evidence_power=1.0,
                  adaptive_evidence=False, min_evidence_power=1.0,
                  confidence_gamma=1.0, device="cpu")
    if family == "stage2_event_bayes":
        return EventConditionedBayesianTerrainFilter(**common)
    return AmbiguityAwareEventConditionedBayesianTerrainFilter(
        **common, ambiguity_margin_threshold=candidate["ambiguity_margin_threshold"],
        ambiguity_flatten_strength=candidate["ambiguity_flatten_strength"])


@torch.inference_mode()
def _evaluate_candidate(candidate, scores, truth, ids, labels, prior, soft_observations,
                        metric_config, probability_cache=None):
    family = candidate["family"]
    if family == "ema_logit_patience":
        filt = EMALogitPatienceFilter(labels, ema_alpha=candidate["ema_alpha"],
                                      change_patience=candidate["change_patience"])
        predictions = run_ema_logit_patience_sequences(filt, scores, sequence_ids=ids)
        return filt, evaluate_sequential_predictions(truth, predictions, labels, ids, metric_config)
    temperature, rho = candidate["temperature"], candidate["observation_mix"]
    observation = _mixed_observation(soft_observations[temperature], rho)
    filt = _make_filter(candidate, labels, prior, observation)
    probabilities = (_probs(scores, temperature) if probability_cache is None
                     else probability_cache[temperature])
    if family == "stage1_fixed_bayes":
        predictions, _, _ = run_filter_sequences(filt, probabilities, sequence_ids=ids)
        extras = {}
    elif family == "stage2_event_bayes":
        predictions, _, _, switches = run_event_conditioned_filter_sequences(filt, probabilities, sequence_ids=ids)
        extras = {"switch_transition_fraction": float(switches.float().mean())}
    else:
        predictions, _, _, switches, ambiguous, _ = run_ambiguity_aware_filter_sequences(
            filt, probabilities, sequence_ids=ids)
        extras = {"switch_transition_fraction": float(switches.float().mean()),
                  "ambiguous_frame_fraction": float(ambiguous.float().mean()),
                  "mean_ambiguity_run_length": _ambiguity_run_length(ambiguous, ids)}
    return filt, {**evaluate_sequential_predictions(truth, predictions, labels, ids, metric_config), **extras}


@torch.inference_mode()
def staged_sequential_search(
    classifier: Any, calibration_scores: torch.Tensor, calibration_labels: Sequence,
    validation_scores: torch.Tensor, validation_labels: Sequence, validation_sequence_ids: Sequence,
    test_scores: torch.Tensor, test_labels: Sequence, test_sequence_ids: Sequence,
    output_dir: str | Path, metric_config: Optional[Mapping[str, float]] = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Tune all temporal families on ordered validation and evaluate test once selected."""
    output_dir = Path(output_dir)
    config = dict(SEQUENTIAL_METRIC_CONFIG if metric_config is None else metric_config)
    labels = list(classifier.class_ids)
    prior = {label: 1.0 / len(labels) for label in labels}
    validation_labels = classifier._normalize_labels(validation_labels)
    test_labels = classifier._normalize_labels(test_labels)
    calibration_labels = classifier._normalize_labels(calibration_labels)
    val_probs = {t: _probs(validation_scores, t) for t in TEMPERATURES}
    soft = {t: estimate_observation_matrix_from_probabilities(
        _probs(calibration_scores, t), calibration_labels, labels, mode="soft",
        pseudocount=OBSERVATION_PSEUDOCOUNT, device="cpu") for t in TEMPERATURES}

    stage1 = []
    for t in TEMPERATURES:
        for rho in OBSERVATION_MIXES:
            for stay in STABLE_STAYS:
                candidate = {"family": "stage1_fixed_bayes", "temperature": t,
                             "observation_mix": rho, "stable_stay": stay,
                             "evidence_power": 1.0, "adaptive_evidence": False,
                             "observation_pseudocount": OBSERVATION_PSEUDOCOUNT}
                _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                    validation_sequence_ids, labels, prior, soft, config, val_probs)
                stage1.append({**candidate, **metrics})
    stage1.sort(key=lambda x: x["selection_score"], reverse=True)

    stage2, seen = [], set()
    for seed in stage1[:2]:
        for switch_stay in SWITCH_STAYS:
            if switch_stay >= seed["stable_stay"]:
                continue
            for confidence in SWITCH_CONFIDENCES:
                for patience in CHANGE_PATIENCES:
                    key = (seed["temperature"], seed["observation_mix"], seed["stable_stay"],
                           switch_stay, confidence, patience)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = {**_params(seed), "family": "stage2_event_bayes",
                                 "switch_stay": switch_stay, "switch_confidence": confidence,
                                 "change_patience": patience}
                    _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                        validation_sequence_ids, labels, prior, soft, config, val_probs)
                    stage2.append({**candidate, **metrics})
    stage2.sort(key=lambda x: x["selection_score"], reverse=True)

    stage3, seen = [], set()
    for seed in stage2[:2]:
        for threshold in AMBIGUITY_THRESHOLDS:
            for strength in AMBIGUITY_FLATTEN_STRENGTHS:
                key = (seed["temperature"], seed["observation_mix"], seed["stable_stay"],
                       seed["switch_stay"], seed["switch_confidence"],
                       seed["change_patience"], threshold, strength)
                if key in seen:
                    continue
                seen.add(key)
                candidate = {**_params(seed), "family": "stage3_ambiguity_bayes",
                             "ambiguity_margin_threshold": threshold,
                             "ambiguity_flatten_strength": strength}
                _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                    validation_sequence_ids, labels, prior, soft, config, val_probs)
                stage3.append({**candidate, **metrics})
    stage3.sort(key=lambda x: x["selection_score"], reverse=True)

    ema = []
    for alpha in EMA_ALPHAS:
        for patience in EMA_PATIENCES:
            candidate = {"family": "ema_logit_patience", "ema_alpha": alpha,
                         "change_patience": patience}
            _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                validation_sequence_ids, labels, prior, soft, config)
            ema.append({**candidate, **metrics})
    ema.sort(key=lambda x: x["selection_score"], reverse=True)

    bests = [stage1[0], stage2[0], stage3[0], ema[0]]
    test_soft = soft  # observation calibration remains structural calibration only
    test_probs = {t: _probs(test_scores, t) for t in TEMPERATURES}
    stage_records, filters = {}, {}
    names = ("stage1_fixed_bayes", "stage2_event_bayes", "stage3_ambiguity_bayes", "ema_logit_patience")
    for name, best in zip(names, bests):
        filt, test_metrics = _evaluate_candidate(best, test_scores, test_labels, test_sequence_ids,
                                                  labels, prior, test_soft, config, test_probs)
        filters[name] = filt
        stage_records[name] = {"best_validation": best, "test_metrics": test_metrics,
                               "parameters": _params(best)}
        filt.reset()
        filt.save(output_dir / f"{name}_filter.pt")
    best_bayes = max(bests[:3], key=lambda x: x["selection_score"])
    best_temporal = max(bests, key=lambda x: x["selection_score"])
    best_bayes_name = best_bayes["family"]
    filters[best_bayes_name].save(output_dir / "bayes_filter.pt")
    filters["ema_logit_patience"].save(output_dir / "ema_filter.pt")
    filters[best_temporal["family"]].save(output_dir / "best_temporal_filter.pt")
    schema = {"metric_config": config, "stages": stage_records,
              "best_bayes_stage": best_bayes_name, "best_bayes": best_bayes,
              "best_ema": ema[0], "best_temporal_overall": best_temporal}
    return schema, {"stage1_fixed_bayes": stage1, "stage2_event_bayes": stage2,
                    "stage3_ambiguity_bayes": stage3, "ema_logit_patience": ema}
