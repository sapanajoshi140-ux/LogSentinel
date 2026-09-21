"""
DAWF — Drift-Aware Weighted Fusion.

RELATIONSHIP TO THE MID-PROGRESS EVALUATION (src/pipeline/evaluate.py)
    DAWF's fuse_scores() is a plain weighted average of whatever is handed
    to it in scores_by_detector -- it does not know or care whether those
    numbers are raw detector output or calibrated probabilities. That
    means DAWF is only meaningful when every value it receives is already
    on the same scale. EWMA's raw z-scores and Markov's raw surprisal
    values are NOT on the same scale (see scripts/try_dawf.py for the
    real numbers), so feeding them to DAWF directly would silently mix
    incompatible units -- DAWF must only ever be called with scores that
    were calibrated first (e.g. via src/calibration/platt.py, fit on the
    VALIDATION split, exactly as scripts/try_dawf.py already does).

    For this mid-progress checkpoint, the PRIMARY reported evaluation
    (src/pipeline/evaluate.py) deliberately does NOT use DAWF. It uses
    independent per-detector thresholds instead (EWMA-only, Markov-only,
    OR, AND) -- each detector's own raw score against its own threshold,
    never mixed with the other detector's scale at all. This is simpler,
    needs no calibration step, and is easier to defend/explain at this
    stage of the project. DAWF is preserved below, unchanged, as the
    calibrated-fusion path for later -- BiLSTM's addition is the natural
    point to bring it back in, once there are three genuinely different
    detectors worth adaptively weighting rather than two.

WHAT IT DOES, IN PLAIN TERMS
    Each detector (EWMA, Markov, BiLSTM) gets a trust "weight". Every time
    we learn the true answer for a block, we check who was right and who
    was wrong, and adjust trust accordingly. This happens continuously,
    live, while the system runs -- that ongoing adjustment is the whole
    "drift-aware" part. A normal ensemble picks fixed weights once during
    training and never touches them again; DAWF keeps adjusting forever.

    It's built on the same idea as the classic Weighted Majority algorithm
    (Littlestone and Warmuth, 1994): mistakes cost trust, being right keeps
    it. This version adds one important correction, found by testing on
    real data -- explained below.

THE REAL BUG THIS VERSION FIXES
    Anomaly data is almost always lopsided -- on real HDFS data, only
    ~1.3% of blocks were actually anomalous. That causes a trap: a
    detector that plays it safe and rarely says "anomaly" (like EWMA
    turned out to be) will be "right" most of the time by default, simply
    because most things really are normal -- not because it's actually
    good at catching anomalies. A detector that's more willing to flag
    things (like Markov) makes more total mistakes along the way, even
    though it's genuinely better at the actual job (catching anomalies).

    A first attempt at fixing this just made "missing an anomaly" cost a
    bit more than "a false alarm" -- but on the real numbers (EWMA: 764
    misses, 4 false alarms; Markov: 575 misses, 446 false alarms), that
    still wasn't enough: Markov still had way more TOTAL mistakes, so it
    still lost the trust competition even with the harsher per-mistake
    penalty.

    The actual fix: the cost of a mistake is scaled by how RARE that
    outcome's true class is, using a running estimate learned from the
    data itself (this is the same idea as "balanced" class weighting in
    standard ML). Since anomalies are rare, missing one costs much more
    per-occurrence than a false alarm on a normal block -- scaled by
    exactly how rare anomalies actually are, not a fixed guessed number.
    Verified against the real counts above: this correctly flips trust
    toward Markov, the actually-better detector.

WHY THIS TRACKS COST, NOT WEIGHT, INTERNALLY
    A naive implementation multiplies the weight by a penalty factor every
    single mistake. Over tens of thousands of real predictions, that
    number underflows to exactly 0.0 in floating point regardless of how
    gently it's tuned -- it's not a tuning problem, it's unavoidable after
    enough multiplications. Instead, this version adds up a running COST
    per detector (addition never underflows) and only converts cost into
    an actual weight when needed, using a numerically stable formula.

KNOWN FAILURE MODES TO TEST FOR (do not assume DAWF helps — measure it):
    - If one detector simply dominates everywhere, fusion converges to
      "mostly trust that one" and adds nothing over using it alone. Catch
      this with the ablation study (src/evaluation/ablation.py).
    - Weight updates only happen AFTER a mistake, so there's real lag at a
      sudden drift boundary. Test explicitly at the boundary, not just in
      aggregate (src/evaluation/drift.py).
    - If all detectors are fooled by the same anomaly type (correlated
      failure), fusion buys nothing — no weighting scheme fixes that.
    - Given enough data, weights will naturally converge strongly toward
      whichever detector is empirically better -- that's the intended
      behaviour, not a bug. min_weight exists so a detector that's
      currently losing can still recover if conditions change later
      (real drift-adaptation), rather than being permanently locked out.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        dawf = DAWF(detector_names=["ewma", "markov", "bilstm"])
        for scores_by_detector, true_label in stream:
            fused_score, fused_pred = dawf.predict(scores_by_detector, threshold=0.5)
            dawf.update(scores_by_detector, true_label, threshold=0.5)
    """

    def __init__(
        self,
        detector_names: list[str],
        eta: float = 0.02,
        min_weight: float = 0.01,
        prior_smoothing: float = 1.0,
    ):
        """
        eta: learning rate -- how strongly a single mistake's (class-
            balanced) cost affects trust. Higher = reacts faster but
            noisier; lower = smoother but slower to react to a real
            change. 0.02 was tuned against real HDFS data so a clear,
            sustained performance gap becomes decisive within a few
            hundred to a few thousand updates, without one-off mistakes
            causing wild swings.
        min_weight: floor so a currently-losing detector can still recover
            if conditions change later (this is the actual "drift-aware"
            safety net) -- without it, a detector that hits ~0 trust could
            never regain influence even if it became the best one again.
        prior_smoothing: Laplace-style smoothing (default 1.0) for the
            running estimate of how common anomalies vs normals are, so
            the cost calculation doesn't blow up on the very first few
            updates before enough data has been seen.
        """
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if not 0 < min_weight < 1 / len(detector_names):
            raise ValueError("min_weight must leave room for renormalization")

        self.detector_names = list(detector_names)
        self.eta = eta
        self.min_weight = min_weight

        self._cumulative_cost = np.zeros(len(detector_names))
        self._n_pos = prior_smoothing  # running (smoothed) count of true anomalies seen
        self._n_neg = prior_smoothing  # running (smoothed) count of true normals seen
        self.n_updates = 0
        self._history: list[DAWFState] = []

    # -- weights, computed on demand from cumulative cost --------------------

    @property
    def weights(self) -> np.ndarray:
        """Numerically stable conversion from cumulative cost to a
        normalized weight vector. Subtracting the minimum cost before
        exponentiating means the best-performing detector's exponent is
        always 0 (weight contribution exactly 1), and a badly losing
        detector's exponent just underflows cleanly to 0.0 -- no
        "0.5 multiplied by itself 80,000 times" underflow risk.

        The floor is applied via water-filling, not a naive floor-then-
        renormalize: naively doing max(w, min_weight) and then dividing
        the WHOLE vector by its new sum pushes the just-floored weight
        back below min_weight again whenever another detector's weight is
        large (e.g. floor=0.05 but the other detector is at 1.0 -> naive
        renormalization gives 0.05/1.05=0.0476, violating the floor).
        Water-filling instead sets floored entries to EXACTLY min_weight
        and rescales only the remaining entries to absorb what's left, so
        every entry actually respects the floor and the vector still sums
        to 1."""
        adjusted = self._cumulative_cost - self._cumulative_cost.min()
        w = np.exp(-adjusted)
        w = w / w.sum()

        for _ in range(len(w)):  # at most one pass per detector is ever needed
            below = w < self.min_weight
            if not below.any():
                break
            w[below] = self.min_weight
            remaining_budget = 1.0 - w[below].sum()
            above = ~below
            above_sum = w[above].sum()
            if above_sum > 0:
                w[above] = w[above] / above_sum * remaining_budget

        return w

    def current_weights(self) -> dict[str, float]:
        return dict(zip(self.detector_names, self.weights.tolist()))

    # -- prediction -----------------------------------------------------------

    def fuse_scores(self, scores_by_detector: dict[str, float]) -> float:
        """Weighted average of per-detector anomaly scores/probabilities."""
        scores = np.array([scores_by_detector[name] for name in self.detector_names])
        return float(np.dot(self.weights, scores))

    def predict(
        self, scores_by_detector: dict[str, float], threshold: float
    ) -> tuple[float, int]:
        fused = self.fuse_scores(scores_by_detector)
        return fused, int(fused >= threshold)

    # -- learning ---------------------------------------------------------------

    def update(
        self,
        scores_by_detector: dict[str, float],
        true_label: int,
        threshold: float,
    ) -> None:
        """Class-balanced cost update, given the now-known true label.

        Call this AFTER predict(), once ground truth becomes available —
        exactly like a real deployment would learn from a confirmed
        incident, not before.
        """
        # Cost of THIS event's class, using the class balance observed
        # so far (not the final/global balance -- keeps this a genuine
        # streaming algorithm that would also adapt if the anomaly rate
        # itself drifted over time).
        total = self._n_pos + self._n_neg
        cost_if_wrong = (
            total / (2 * self._n_pos) if true_label == 1
            else total / (2 * self._n_neg)
        )

        for i, name in enumerate(self.detector_names):
            detector_pred = int(scores_by_detector[name] >= threshold)
            if detector_pred != true_label:
                self._cumulative_cost[i] += self.eta * cost_if_wrong

        # Update the running class-balance estimate AFTER using it above,
        # so this event's cost was computed from what was known before it.
        if true_label == 1:
            self._n_pos += 1
        else:
            self._n_neg += 1

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