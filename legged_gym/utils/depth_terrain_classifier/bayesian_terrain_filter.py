"""Discrete Bayesian filtering for PCA/prototype terrain classifiers.

This module wraps an already-fitted terrain classifier with a lightweight hidden
Markov model (HMM) filtering step:

    classifier probability vector
        -> transition prediction
        -> confusion-aware observation likelihood
        -> confidence-adaptive evidence tempering
        -> normalized terrain posterior

The implementation is deliberately dependency-light: only PyTorch and the Python
standard library are required.  It is compatible with the ``DepthTerrainClassifier``
and ``IncrementalPCAPrototypeClassifier`` APIs from ``depth_terrain_classifier.py``.

Conventions
-----------
* ``transition[i, j] = P(x_t=j | x_{t-1}=i)``.  Rows sum to one.
* ``observation[i, j] = P(o_t=j | x_t=i)``.  Rows sum to one.
* Classifier outputs are soft observations ``q_t[j]`` over the same ordered labels.
* The confusion-aware likelihood is the soft compatibility

      L_t[i] = sum_j observation[i, j] * q_t[j].

  With an identity observation matrix this reduces to the classifier probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


# =============================================================================
# Small result containers
# =============================================================================


@dataclass
class BayesianFilterStep:
    """Diagnostic output for one Bayes-filter update."""

    label: str
    posterior: torch.Tensor
    predicted_prior: torch.Tensor
    observation_likelihood: torch.Tensor
    classifier_probabilities: torch.Tensor
    confidence: float
    evidence_power: float


@dataclass
class BayesianTerrainPrediction:
    """Combined instantaneous-classifier and Bayesian-filter prediction."""

    label: str
    instantaneous_label: str
    posterior: torch.Tensor
    classifier_probabilities: torch.Tensor
    labels: list[str]
    distances: torch.Tensor | None
    raw_features: torch.Tensor | None
    confidence: float
    evidence_power: float


@dataclass
class FilterEvaluation:
    """Dependency-free metrics for an ordered sequence evaluation."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: torch.Tensor
    labels: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


# =============================================================================
# Core discrete Bayes filter
# =============================================================================


class BayesianTerrainFilter:
    """Discrete Bayes filter over a fixed ordered set of terrain classes.

    Parameters
    ----------
    labels:
        Ordered class labels.  This order must match the classifier probability
        vector, transition matrix, observation matrix, and prior.
    prior:
        Manual initial terrain distribution, either a length-C sequence/tensor or a
        ``{label: probability}`` mapping.  Exact zeros are replaced by ``eps`` so a
        class can recover if later evidence strongly supports it.
    transition_matrix:
        Row-stochastic matrix ``T[i,j] = P(x_t=j | x_{t-1}=i)``.
    observation_matrix:
        Row-stochastic confusion/observation matrix
        ``O[i,j] = P(classifier observation=j | true terrain=i)``.  If omitted, an
        identity matrix is used.
    evidence_power:
        Maximum/fixed likelihood exponent.  Values below one soften observations;
        values above one sharpen them.
    adaptive_evidence:
        If true, interpolate the exponent from ``min_evidence_power`` to
        ``evidence_power`` using normalized entropy confidence.
    min_evidence_power:
        Evidence exponent used for a maximally uncertain classifier output.
    confidence_gamma:
        Shapes confidence before interpolation.  Values above one make the filter
        conservative except for very confident observations.
    eps:
        Numerical probability floor.

    Notes
    -----
    The filter does not require a learned dynamics model.  A manually specified
    persistent transition matrix is often sufficient to suppress isolated false
    positives at 10 Hz.
    """

    def __init__(
        self,
        labels: Sequence[str],
        prior: torch.Tensor | Sequence[float] | Mapping[str, float],
        transition_matrix: torch.Tensor | Sequence,
        observation_matrix: torch.Tensor | Sequence | None = None,
        *,
        evidence_power: float = 0.75,
        adaptive_evidence: bool = True,
        min_evidence_power: float = 0.20,
        confidence_gamma: float = 1.0,
        device: str | torch.device = "cpu",
        eps: float = 1e-8,
    ):
        self.labels = [str(label) for label in labels]
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be a non-empty sequence of unique names")

        self.device = torch.device(device)
        self.eps = float(eps)
        self.num_classes = len(self.labels)

        self.evidence_power = float(evidence_power)
        self.adaptive_evidence = bool(adaptive_evidence)
        self.min_evidence_power = float(min_evidence_power)
        self.confidence_gamma = float(confidence_gamma)

        if self.evidence_power < 0.0 or self.min_evidence_power < 0.0:
            raise ValueError("evidence powers must be non-negative")
        if self.min_evidence_power > self.evidence_power:
            raise ValueError("min_evidence_power cannot exceed evidence_power")
        if self.confidence_gamma <= 0.0:
            raise ValueError("confidence_gamma must be positive")

        self.initial_prior = make_manual_prior(
            self.labels, prior, device=self.device, eps=self.eps
        )
        self.transition_matrix = _validate_row_stochastic_matrix(
            transition_matrix,
            self.num_classes,
            name="transition_matrix",
            device=self.device,
            eps=self.eps,
        )

        if observation_matrix is None:
            observation_matrix = torch.eye(self.num_classes)
        self.observation_matrix = _validate_row_stochastic_matrix(
            observation_matrix,
            self.num_classes,
            name="observation_matrix",
            device=self.device,
            eps=self.eps,
        )

        self.belief = self.initial_prior.clone()

    # ------------------------------------------------------------------
    # Runtime filtering
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def update(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
        *,
        observation_quality: float = 1.0,
    ) -> BayesianFilterStep:
        """Advance the filter using one instantaneous classifier distribution.

        ``observation_quality`` is an optional scalar in [0,1].  Setting it below one
        weakens the observation update; zero performs only the transition prediction.
        This can be driven by depth validity, angular speed, or another sensor-quality
        measure without changing the filter equations.
        """
        q = _normalize_vector(
            classifier_probabilities,
            expected_size=self.num_classes,
            device=self.device,
            eps=self.eps,
            name="classifier_probabilities",
        )

        quality = float(observation_quality)
        if not 0.0 <= quality <= 1.0:
            raise ValueError("observation_quality must be in [0,1]")

        # HMM prediction: propagate the previous posterior through terrain dynamics.
        predicted = self.belief @ self.transition_matrix
        predicted = predicted / predicted.sum().clamp_min(self.eps)

        # Soft confusion-aware likelihood.  Each row of O describes the classifier
        # output expected under one true terrain state; dotting it with q yields the
        # compatibility between the current soft observation and that state.
        likelihood = self.observation_matrix @ q
        likelihood = likelihood.clamp_min(self.eps)

        confidence = self.entropy_confidence(q)
        beta = self._effective_evidence_power(confidence, quality)

        # Tempering controls likelihood ratios. beta=0 makes all states equally likely
        # under the observation, so the update falls back to the transition prediction.
        tempered_likelihood = likelihood.pow(beta)
        posterior = predicted * tempered_likelihood
        posterior = posterior / posterior.sum().clamp_min(self.eps)
        self.belief = posterior

        best_index = int(torch.argmax(posterior).item())
        return BayesianFilterStep(
            label=self.labels[best_index],
            posterior=posterior.clone(),
            predicted_prior=predicted.clone(),
            observation_likelihood=likelihood.clone(),
            classifier_probabilities=q.clone(),
            confidence=float(confidence),
            evidence_power=float(beta),
        )

    def reset(
        self,
        prior: torch.Tensor | Sequence[float] | Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        """Reset at an episode boundary, optionally with a new manual prior."""
        if prior is None:
            self.belief = self.initial_prior.clone()
        else:
            self.belief = make_manual_prior(
                self.labels, prior, device=self.device, eps=self.eps
            )
        return self.belief.clone()

    def predict_label(
        self,
        *,
        min_posterior: float = 0.0,
        min_margin: float = 0.0,
        fallback_label: str | None = None,
    ) -> str:
        """Return the MAP state, optionally requiring confidence and margin gates.

        The optional gates are useful when a downstream controller should retain a
        conservative fallback state until the posterior is decisive.  They do not
        alter the Bayesian belief itself.
        """
        top_values, top_indices = torch.topk(self.belief, k=min(2, self.num_classes))
        best_label = self.labels[int(top_indices[0])]
        best_probability = float(top_values[0])
        margin = (
            float(top_values[0] - top_values[1]) if self.num_classes > 1 else 1.0
        )

        if best_probability < min_posterior or margin < min_margin:
            if fallback_label is None:
                return best_label
            if fallback_label not in self.labels:
                raise ValueError(f"Unknown fallback_label: {fallback_label}")
            return fallback_label
        return best_label

    def entropy_confidence(self, probabilities: torch.Tensor | Sequence[float]) -> float:
        """Return normalized confidence in [0,1] from distribution entropy."""
        q = _normalize_vector(
            probabilities,
            expected_size=self.num_classes,
            device=self.device,
            eps=self.eps,
            name="probabilities",
        )
        if self.num_classes == 1:
            return 1.0
        entropy = -(q * q.clamp_min(self.eps).log()).sum()
        max_entropy = torch.log(
            torch.tensor(float(self.num_classes), device=self.device)
        )
        return float((1.0 - entropy / max_entropy).clamp(0.0, 1.0))

    def _effective_evidence_power(self, confidence: float, quality: float) -> float:
        if self.adaptive_evidence:
            shaped_confidence = confidence ** self.confidence_gamma
            beta = self.min_evidence_power + shaped_confidence * (
                self.evidence_power - self.min_evidence_power
            )
        else:
            beta = self.evidence_power
        return float(beta * quality)

    # ------------------------------------------------------------------
    # Configuration and persistence
    # ------------------------------------------------------------------

    def set_prior(
        self,
        prior: torch.Tensor | Sequence[float] | Mapping[str, float],
        *,
        reset: bool = True,
    ) -> None:
        self.initial_prior = make_manual_prior(
            self.labels, prior, device=self.device, eps=self.eps
        )
        if reset:
            self.belief = self.initial_prior.clone()

    def set_transition_matrix(self, matrix: torch.Tensor | Sequence) -> None:
        self.transition_matrix = _validate_row_stochastic_matrix(
            matrix,
            self.num_classes,
            name="transition_matrix",
            device=self.device,
            eps=self.eps,
        )

    def set_observation_matrix(self, matrix: torch.Tensor | Sequence) -> None:
        self.observation_matrix = _validate_row_stochastic_matrix(
            matrix,
            self.num_classes,
            name="observation_matrix",
            device=self.device,
            eps=self.eps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "initial_prior": self.initial_prior.cpu(),
            "belief": self.belief.cpu(),
            "transition_matrix": self.transition_matrix.cpu(),
            "observation_matrix": self.observation_matrix.cpu(),
            "evidence_power": self.evidence_power,
            "adaptive_evidence": self.adaptive_evidence,
            "min_evidence_power": self.min_evidence_power,
            "confidence_gamma": self.confidence_gamma,
            "eps": self.eps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if [str(v) for v in state["labels"]] != self.labels:
            raise ValueError("Saved filter labels/order do not match this filter")
        self.initial_prior = make_manual_prior(
            self.labels, state["initial_prior"], device=self.device, eps=self.eps
        )
        self.belief = _normalize_vector(
            state["belief"], self.num_classes, self.device, self.eps, "belief"
        )
        self.set_transition_matrix(state["transition_matrix"])
        self.set_observation_matrix(state["observation_matrix"])
        self.evidence_power = float(state["evidence_power"])
        self.adaptive_evidence = bool(state["adaptive_evidence"])
        self.min_evidence_power = float(state["min_evidence_power"])
        self.confidence_gamma = float(state["confidence_gamma"])

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), map_location=self.device, weights_only=False))


# =============================================================================
# Deployment wrapper for the existing depth-terrain classifier
# =============================================================================


class BayesianFilteredTerrainClassifier:
    """Combine a fitted terrain classifier with ``BayesianTerrainFilter``.

    The wrapper always requests *instantaneous* classifier probabilities
    (``temporal=False``) so that the prior EMA/hysteresis filter is not stacked with
    the Bayesian update.  The wrapped model is expected to expose the same API as
    ``DepthTerrainClassifier``.
    """

    def __init__(self, terrain_model: Any, bayes_filter: BayesianTerrainFilter):
        self.terrain_model = terrain_model
        self.filter = bayes_filter
        labels = get_classifier_labels(terrain_model)
        if labels != self.filter.labels:
            raise ValueError(
                "Classifier labels/order differ from Bayes-filter labels/order: "
                f"{labels} != {self.filter.labels}"
            )

    @torch.inference_mode()
    def predict_depth(
        self,
        depth_image: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
        *,
        observation_quality: float = 1.0,
    ) -> BayesianTerrainPrediction:
        prediction = self.terrain_model.predict_depth(
            depth_image,
            orientation_rpy=orientation_rpy,
            angular_velocity=angular_velocity,
            temporal=False,
        )
        if list(prediction.labels) != self.filter.labels:
            raise RuntimeError("Classifier class order changed; rebuild/reconfigure filter")

        step = self.filter.update(
            prediction.probabilities,
            observation_quality=observation_quality,
        )
        return BayesianTerrainPrediction(
            label=step.label,
            instantaneous_label=prediction.instantaneous_label,
            posterior=step.posterior,
            classifier_probabilities=step.classifier_probabilities,
            labels=list(self.filter.labels),
            distances=getattr(prediction, "distances", None),
            raw_features=getattr(prediction, "raw_features", None),
            confidence=step.confidence,
            evidence_power=step.evidence_power,
        )

    def reset(
        self,
        prior: torch.Tensor | Sequence[float] | Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        """Reset the Bayesian state at an episode/environment boundary."""
        return self.filter.reset(prior)


# =============================================================================
# Calibration utilities for an already-fitted classifier
# =============================================================================


def get_classifier_labels(model: Any) -> list[str]:
    """Return class order from a unified or low-level PCA terrain classifier."""
    if hasattr(model, "classifier") and hasattr(model.classifier, "class_ids"):
        labels = model.classifier.class_ids
    elif hasattr(model, "class_ids"):
        labels = model.class_ids
    else:
        raise TypeError("model does not expose classifier.class_ids or class_ids")
    labels = [str(v) for v in labels]
    if not labels:
        raise ValueError("classifier has no fitted classes")
    return labels


def make_manual_prior(
    labels: Sequence[str],
    prior: torch.Tensor | Sequence[float] | Mapping[str, float],
    *,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Create a normalized prior aligned to ``labels``.

    A mapping is safer than a bare vector because it makes class alignment explicit.
    Missing mapping entries receive ``eps`` rather than an irreversible exact zero.
    """
    labels = [str(v) for v in labels]
    if isinstance(prior, Mapping):
        unknown = set(str(k) for k in prior) - set(labels)
        if unknown:
            raise ValueError(f"Prior contains unknown labels: {sorted(unknown)}")
        values = [float(prior.get(label, eps)) for label in labels]
    else:
        values = prior
    return _normalize_vector(values, len(labels), torch.device(device), eps, "prior")


def make_persistent_transition_matrix(
    labels: Sequence[str],
    stay_probability: float = 0.95,
    *,
    transition_weights: Mapping[str, Mapping[str, float]] | None = None,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a row-stochastic persistent terrain transition matrix.

    Without ``transition_weights``, all off-diagonal transitions share the remaining
    probability equally.  A nested mapping can encode known topology, for example
    ``{"flat": {"stairs": 3, "rough": 1}}``.  Weights are relative and only affect
    the off-diagonal mass; unspecified destinations receive zero before flooring.
    """
    labels = [str(v) for v in labels]
    c = len(labels)
    if c == 0:
        raise ValueError("labels cannot be empty")
    if c == 1:
        return torch.ones(1, 1, dtype=torch.float32, device=device)
    if not 0.0 <= stay_probability < 1.0:
        raise ValueError("stay_probability must be in [0,1) for multiple classes")

    index = {label: i for i, label in enumerate(labels)}
    matrix = torch.zeros(c, c, dtype=torch.float32, device=device)
    for source in labels:
        i = index[source]
        matrix[i, i] = stay_probability
        remaining = 1.0 - stay_probability

        if transition_weights is None or source not in transition_weights:
            matrix[i] += remaining / (c - 1)
            matrix[i, i] = stay_probability
            continue

        row_weights = transition_weights[source]
        unknown = set(row_weights) - set(labels)
        if unknown:
            raise ValueError(
                f"Transition weights for '{source}' contain unknown labels: {unknown}"
            )
        weights = torch.tensor(
            [0.0 if destination == source else float(row_weights.get(destination, 0.0))
             for destination in labels],
            dtype=torch.float32,
            device=device,
        )
        if torch.any(weights < 0.0):
            raise ValueError("transition weights must be non-negative")
        if float(weights.sum()) <= 0.0:
            raise ValueError(f"No positive outgoing transition weight for '{source}'")
        matrix[i] += remaining * weights / weights.sum()
        matrix[i, i] = stay_probability

    return _validate_row_stochastic_matrix(
        matrix, c, "transition_matrix", torch.device(device), eps
    )


def estimate_transition_matrix_from_sequences(
    true_labels: Sequence[str],
    labels: Sequence[str],
    *,
    sequence_ids: Sequence[Any] | None = None,
    pseudocount: float = 1.0,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate ``P(x_t | x_{t-1})`` from ordered labeled sequences.

    Transitions are never counted across changes in ``sequence_ids``.  A pseudocount
    avoids zero-probability transitions in small datasets.
    """
    labels = [str(v) for v in labels]
    y = [str(v) for v in true_labels]
    if sequence_ids is None:
        sequence_ids = [0] * len(y)
    if len(y) != len(sequence_ids):
        raise ValueError("true_labels and sequence_ids must have equal length")
    if pseudocount < 0.0:
        raise ValueError("pseudocount must be non-negative")

    index = {label: i for i, label in enumerate(labels)}
    unknown = set(y) - set(labels)
    if unknown:
        raise ValueError(f"true_labels contain unknown classes: {sorted(unknown)}")

    counts = torch.full(
        (len(labels), len(labels)),
        float(pseudocount),
        dtype=torch.float32,
        device=device,
    )
    for t in range(1, len(y)):
        if sequence_ids[t] != sequence_ids[t - 1]:
            continue
        counts[index[y[t - 1]], index[y[t]]] += 1.0
    return _validate_row_stochastic_matrix(
        counts, len(labels), "transition_matrix", torch.device(device), eps
    )


def estimate_observation_matrix_from_probabilities(
    probabilities: torch.Tensor | Sequence,
    true_labels: Sequence[str],
    labels: Sequence[str],
    *,
    mode: str = "soft",
    pseudocount: float = 0.5,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate a row-normalized classifier observation/confusion matrix.

    ``mode='soft'`` (recommended) adds the full predicted probability vector to the
    row for the true class.  ``mode='hard'`` increments only the argmax prediction.
    The result follows ``O[true, observed]`` and can be passed directly to the filter.
    """
    labels = [str(v) for v in labels]
    probs = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    if probs.ndim != 2 or probs.shape[1] != len(labels):
        raise ValueError(f"probabilities must have shape [N,{len(labels)}]")
    if probs.shape[0] != len(true_labels):
        raise ValueError("probabilities and true_labels must have equal sample counts")
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
    if pseudocount < 0.0:
        raise ValueError("pseudocount must be non-negative")

    # Normalize defensively because probabilities may come from stored/rounded output.
    probs = probs.clamp_min(0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)

    index = {label: i for i, label in enumerate(labels)}
    true_labels = [str(v) for v in true_labels]
    unknown = set(true_labels) - set(labels)
    if unknown:
        raise ValueError(f"true_labels contain unknown classes: {sorted(unknown)}")

    counts = torch.full(
        (len(labels), len(labels)),
        float(pseudocount),
        dtype=torch.float32,
        device=device,
    )
    if mode == "soft":
        for row, truth in zip(probs, true_labels):
            counts[index[truth]] += row
    else:
        predicted = torch.argmax(probs, dim=1)
        for prediction, truth in zip(predicted.tolist(), true_labels):
            counts[index[truth], prediction] += 1.0

    return _validate_row_stochastic_matrix(
        counts, len(labels), "observation_matrix", torch.device(device), eps
    )


@torch.inference_mode()
def collect_classifier_probabilities(
    model: Any,
    *,
    depth_images: torch.Tensor | Sequence | None = None,
    features: torch.Tensor | Sequence | None = None,
    orientation_rpy: torch.Tensor | Sequence | None = None,
    angular_velocity: torch.Tensor | Sequence | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
    """Collect instantaneous probabilities from a fitted terrain classifier.

    Supply either raw ``depth_images`` for the unified model or engineered ``features``
    for the low-level PCA/prototype classifier.  Temporal filtering is intentionally
    excluded because the resulting observation matrix should describe the one-frame
    classifier used as the Bayes-filter sensor model.
    """
    if (depth_images is None) == (features is None):
        raise ValueError("Supply exactly one of depth_images or features")

    labels = get_classifier_labels(model)
    if depth_images is not None:
        if not hasattr(model, "predict_depth_batch"):
            raise TypeError("Raw depth input requires model.predict_depth_batch")
        _, probabilities, distances = model.predict_depth_batch(
            depth_images,
            orientation_rpy=orientation_rpy,
            angular_velocity=angular_velocity,
        )
    else:
        classifier = model.classifier if hasattr(model, "classifier") else model
        if not hasattr(classifier, "predict_proba"):
            raise TypeError("Model does not expose predict_proba")
        _, probabilities, distances = classifier.predict_proba(features)

    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise RuntimeError("Classifier returned probabilities inconsistent with labels")
    return probabilities, labels, distances


@torch.inference_mode()
def estimate_observation_matrix_from_classifier(
    model: Any,
    true_labels: Sequence[str],
    *,
    depth_images: torch.Tensor | Sequence | None = None,
    features: torch.Tensor | Sequence | None = None,
    orientation_rpy: torch.Tensor | Sequence | None = None,
    angular_velocity: torch.Tensor | Sequence | None = None,
    mode: str = "soft",
    pseudocount: float = 0.5,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Estimate the confusion-aware observation model from held-out labeled data.

    Returns ``(observation_matrix, instantaneous_probabilities, class_labels)``.
    Use deployment-like simulation sequences rather than the classifier's training
    set whenever possible so the matrix captures the false positives seen in practice.
    """
    probabilities, labels, _ = collect_classifier_probabilities(
        model,
        depth_images=depth_images,
        features=features,
        orientation_rpy=orientation_rpy,
        angular_velocity=angular_velocity,
    )
    observation = estimate_observation_matrix_from_probabilities(
        probabilities,
        true_labels,
        labels,
        mode=mode,
        pseudocount=pseudocount,
        device=device,
    )
    return observation, probabilities, labels


def build_bayesian_filter_from_classifier(
    model: Any,
    validation_true_labels: Sequence[str],
    *,
    manual_prior: torch.Tensor | Sequence[float] | Mapping[str, float],
    transition_matrix: torch.Tensor | Sequence | None = None,
    stay_probability: float = 0.95,
    transition_weights: Mapping[str, Mapping[str, float]] | None = None,
    depth_images: torch.Tensor | Sequence | None = None,
    features: torch.Tensor | Sequence | None = None,
    orientation_rpy: torch.Tensor | Sequence | None = None,
    angular_velocity: torch.Tensor | Sequence | None = None,
    observation_mode: str = "soft",
    observation_pseudocount: float = 0.5,
    evidence_power: float = 0.75,
    adaptive_evidence: bool = True,
    min_evidence_power: float = 0.20,
    confidence_gamma: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[BayesianTerrainFilter, torch.Tensor]:
    """Calibrate and construct a Bayes filter around an already-fitted classifier.

    The returned second value is the validation probability matrix, which is useful
    for diagnostics or filter hyperparameter selection.
    """
    observation, probabilities, labels = estimate_observation_matrix_from_classifier(
        model,
        validation_true_labels,
        depth_images=depth_images,
        features=features,
        orientation_rpy=orientation_rpy,
        angular_velocity=angular_velocity,
        mode=observation_mode,
        pseudocount=observation_pseudocount,
        device=device,
    )
    if transition_matrix is None:
        transition_matrix = make_persistent_transition_matrix(
            labels,
            stay_probability,
            transition_weights=transition_weights,
            device=device,
        )

    terrain_filter = BayesianTerrainFilter(
        labels,
        manual_prior,
        transition_matrix,
        observation,
        evidence_power=evidence_power,
        adaptive_evidence=adaptive_evidence,
        min_evidence_power=min_evidence_power,
        confidence_gamma=confidence_gamma,
        device=device,
    )
    return terrain_filter, probabilities


# =============================================================================
# Ordered-sequence evaluation and compact hyperparameter search
# =============================================================================


@torch.inference_mode()
def run_filter_sequences(
    terrain_filter: BayesianTerrainFilter,
    classifier_probabilities: torch.Tensor | Sequence,
    *,
    sequence_ids: Sequence[Any] | None = None,
    observation_quality: torch.Tensor | Sequence[float] | None = None,
    prior: torch.Tensor | Sequence[float] | Mapping[str, float] | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Run a filter over ordered samples, resetting at sequence boundaries.

    Returns labels, posterior matrix [N,C], and effective evidence powers [N].
    """
    probs = torch.as_tensor(classifier_probabilities, dtype=torch.float32)
    if probs.ndim != 2 or probs.shape[1] != terrain_filter.num_classes:
        raise ValueError("classifier_probabilities have the wrong shape")
    n = probs.shape[0]
    if sequence_ids is None:
        sequence_ids = [0] * n
    if len(sequence_ids) != n:
        raise ValueError("sequence_ids length must match probabilities")

    if observation_quality is None:
        qualities = torch.ones(n)
    else:
        qualities = torch.as_tensor(observation_quality, dtype=torch.float32).flatten()
        if qualities.numel() != n:
            raise ValueError("observation_quality length must match probabilities")

    predicted_labels: list[str] = []
    posteriors: list[torch.Tensor] = []
    powers: list[float] = []
    previous_sequence: Any = object()

    for i in range(n):
        if i == 0 or sequence_ids[i] != previous_sequence:
            terrain_filter.reset(prior)
        previous_sequence = sequence_ids[i]
        step = terrain_filter.update(
            probs[i], observation_quality=float(qualities[i].clamp(0.0, 1.0))
        )
        predicted_labels.append(step.label)
        posteriors.append(step.posterior.cpu())
        powers.append(step.evidence_power)

    return predicted_labels, torch.stack(posteriors), torch.tensor(powers)


def evaluate_predictions(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] | None = None,
) -> FilterEvaluation:
    """Compute accuracy, balanced accuracy, macro F1, and confusion matrix."""
    truth = [str(v) for v in true_labels]
    prediction = [str(v) for v in predicted_labels]
    if len(truth) != len(prediction):
        raise ValueError("true and predicted label counts differ")
    labels = list(labels or dict.fromkeys(truth + prediction))
    index = {label: i for i, label in enumerate(labels)}
    if set(truth + prediction) - set(labels):
        raise ValueError("labels do not cover all predictions and targets")

    confusion = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    for y, yhat in zip(truth, prediction):
        confusion[index[y], index[yhat]] += 1

    total = max(int(confusion.sum()), 1)
    accuracy = float(torch.diagonal(confusion).sum()) / total
    recalls: list[float] = []
    f1s: list[float] = []
    for i in range(len(labels)):
        tp = float(confusion[i, i])
        fn = float(confusion[i].sum()) - tp
        fp = float(confusion[:, i].sum()) - tp
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        recalls.append(recall)
        f1s.append(f1)

    return FilterEvaluation(
        accuracy=accuracy,
        balanced_accuracy=sum(recalls) / max(len(recalls), 1),
        macro_f1=sum(f1s) / max(len(f1s), 1),
        confusion_matrix=confusion,
        labels=labels,
    )


def search_filter_hyperparameters(
    classifier_probabilities: torch.Tensor | Sequence,
    true_labels: Sequence[str],
    labels: Sequence[str],
    observation_matrix: torch.Tensor | Sequence,
    manual_prior: torch.Tensor | Sequence[float] | Mapping[str, float],
    *,
    sequence_ids: Sequence[Any] | None = None,
    stay_probabilities: Sequence[float] = (0.90, 0.94, 0.97),
    evidence_powers: Sequence[float] = (0.50, 0.75, 1.0),
    min_evidence_powers: Sequence[float] = (0.10, 0.25),
    confidence_gammas: Sequence[float] = (1.0, 2.0),
    scoring: str = "balanced_accuracy",
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Grid-search the main filter parameters on ordered validation sequences.

    The observation matrix should be estimated on a separate calibration split when
    possible; otherwise scores can be optimistic.  Results are sorted best first.
    """
    if scoring not in {"accuracy", "balanced_accuracy", "macro_f1"}:
        raise ValueError("Unsupported scoring metric")

    results: list[dict[str, Any]] = []
    for stay, max_power, min_power, gamma in product(
        stay_probabilities,
        evidence_powers,
        min_evidence_powers,
        confidence_gammas,
    ):
        if min_power > max_power:
            continue
        transition = make_persistent_transition_matrix(labels, stay, device=device)
        filt = BayesianTerrainFilter(
            labels,
            manual_prior,
            transition,
            observation_matrix,
            evidence_power=max_power,
            adaptive_evidence=True,
            min_evidence_power=min_power,
            confidence_gamma=gamma,
            device=device,
        )
        predictions, _, _ = run_filter_sequences(
            filt,
            classifier_probabilities,
            sequence_ids=sequence_ids,
        )
        metrics = evaluate_predictions(true_labels, predictions, labels)
        result = {
            "stay_probability": float(stay),
            "evidence_power": float(max_power),
            "min_evidence_power": float(min_power),
            "confidence_gamma": float(gamma),
            **metrics.as_dict(),
        }
        results.append(result)

    return sorted(results, key=lambda item: item[scoring], reverse=True)


# =============================================================================
# Internal validation helpers
# =============================================================================


def _normalize_vector(
    values: torch.Tensor | Sequence[float],
    expected_size: int,
    device: torch.device,
    eps: float,
    name: str,
) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float32, device=device).flatten()
    if vector.numel() != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    if torch.any(vector < 0.0):
        raise ValueError(f"{name} must be non-negative")
    vector = vector.clamp_min(eps)
    total = vector.sum()
    if float(total) <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return vector / total


def _validate_row_stochastic_matrix(
    matrix: torch.Tensor | Sequence,
    expected_size: int,
    name: str,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    if tensor.shape != (expected_size, expected_size):
        raise ValueError(f"{name} must have shape [{expected_size},{expected_size}]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    if torch.any(tensor < 0.0):
        raise ValueError(f"{name} must be non-negative")
    row_sums = tensor.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0.0):
        raise ValueError(f"Every row of {name} must contain positive mass")
    tensor = tensor.clamp_min(eps)
    return tensor / tensor.sum(dim=1, keepdim=True)


__all__ = [
    "BayesianFilterStep",
    "BayesianTerrainPrediction",
    "FilterEvaluation",
    "BayesianTerrainFilter",
    "BayesianFilteredTerrainClassifier",
    "get_classifier_labels",
    "make_manual_prior",
    "make_persistent_transition_matrix",
    "estimate_transition_matrix_from_sequences",
    "estimate_observation_matrix_from_probabilities",
    "collect_classifier_probabilities",
    "estimate_observation_matrix_from_classifier",
    "build_bayesian_filter_from_classifier",
    "run_filter_sequences",
    "evaluate_predictions",
    "search_filter_hyperparameters",
]
