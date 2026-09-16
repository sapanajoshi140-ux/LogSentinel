 # Weighted Majority (Littlestone & Warmuth [9])
"""
DAWF — Drift-Aware Weighted Fusion.

Built directly on the Weighted Majority algorithm (Littlestone and Warmuth,
1994): each detector holds a weight; every time a detector is wrong, its
weight is multiplicatively reduced; weights are renormalized so they always
sum to 1. This runs continuously, online, at every prediction — that
continuous reweighting during deployment is the whole "drift-aware" part.
A standard ensemble fixes weights once at training time and never touches
them again; DAWF's entire point is not doing that.

Weighted Majority's formal guarantee: DAWF's cumulative mistakes are bounded
in terms of the BEST single detector's mistakes in hindsight, even under
adversarial / non-stationary conditions. That's real theoretical grounding,
not just an engineering hack — but see the caveats below, which are exactly
what your ablation and drift-robustness tests (src/evaluation/) exist to
check empirically rather than assume.

KNOWN FAILURE MODES TO TEST FOR (do not assume DAWF helps — measure it):
    - If one detector simply dominates everywhere, fusion converges to
      "mostly trust that one" and adds nothing over using it alone. Catch
      this with the ablation study (src/evaluation/ablation.py).
    - Weight updates only happen AFTER a mistake, so there's real lag at a
      sudden drift boundary. Test explicitly at the boundary, not just in
      aggregate (src/evaluation/drift.py).
    - If all detectors are fooled by the same anomaly type (correlated
      failure), fusion buys nothing — no weighting scheme fixes that.
    - Noisy ground-truth labels make every weight update noisier, which can
      make DAWF worse than a single well-tuned detector on noisy data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DAWFState:
    """Serializable snapshot of DAWF's weights — dump this per-batch during
    a drift-robustness run so you can plot weight trajectories over time
    and actually see whether/how fast DAWF reacts to a drift boundary."""
    detector_names: list[str]
    weights: list[float]
    n_updates: int = 0


class DAWF:
    """Drift-Aware Weighted Fusion over an arbitrary number of detectors.

    Usage:
        dawf = DAWF(detector_names=["ewma", "markov", "bilstm"], beta=0.5)
        for scores_by_detector, true_label in stream:
            fused_score, fused_pred = dawf.predict(scores_by_detector, threshold=0.5)
            dawf.update(scores_by_detector, true_label, threshold=0.5)
    """

    def __init__(
        self,
        detector_names: list[str],
        beta: float = 0.5,
        min_weight: float = 0.01,
    ):
        """
        beta: penalty factor applied to a detector's weight when it is
            wrong (weight *= beta). Lower beta = harsher, faster-reacting
            penalty; beta close to 1 = slower, more stable adaptation.
            This is the knob that trades off reaction speed against
            stability at a drift boundary — sweep it in your drift test.
        min_weight: floor so a detector that's had a bad run can still
            recover if conditions change back. Without a floor, a detector
            that hits zero weight can never regain influence even if it
            becomes the best one again — which would be a hidden footgun,
            not a feature, given the whole point is adapting to drift in
            EITHER direction.
        """
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if not 0 < min_weight < 1 / len(detector_names):
            raise ValueError("min_weight must leave room for renormalization")

        self.detector_names = list(detector_names)
        self.beta = beta
        self.min_weight = min_weight
        self.weights = np.ones(len(detector_names)) / len(detector_names)
        self.n_updates = 0
        self._history: list[DAWFState] = []

    def fuse_scores(self, scores_by_detector: dict[str, float]) -> float:
        """Weighted average of per-detector anomaly scores."""
        scores = np.array([scores_by_detector[name] for name in self.detector_names])
        return float(np.dot(self.weights, scores))

    def predict(
        self, scores_by_detector: dict[str, float], threshold: float
    ) -> tuple[float, int]:
        fused = self.fuse_scores(scores_by_detector)
        return fused, int(fused >= threshold)

    def update(
        self,
        scores_by_detector: dict[str, float],
        true_label: int,
        threshold: float,
    ) -> None:
        """Weighted Majority weight update given the now-known true label.

        Call this AFTER predict(), once ground truth becomes available —
        exactly like a real deployment would learn from a confirmed
        incident, not before.
        """
        for i, name in enumerate(self.detector_names):
            detector_pred = int(scores_by_detector[name] >= threshold)
            if detector_pred != true_label:
                self.weights[i] *= self.beta

        # Floor, then renormalize so weights always sum to 1.
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights /= self.weights.sum()
        self.n_updates += 1

        self._history.append(
            DAWFState(
                detector_names=list(self.detector_names),
                weights=self.weights.tolist(),
                n_updates=self.n_updates,
            )
        )

    def weight_trajectory(self) -> list[DAWFState]:
        """Full history of weight snapshots — plot this against a known
        drift boundary to see whether/how fast DAWF actually reacts.
        This is the evidence for the "drift-aware" claim, not an assertion."""
        return self._history

    def current_weights(self) -> dict[str, float]:
        return dict(zip(self.detector_names, self.weights.tolist()))
