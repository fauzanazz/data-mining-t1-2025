"""Rule quality evaluator for SP mining results."""

import statistics

from sp_mining.core.interfaces import (
    SPAlgorithmResult,
    SPEvaluator,
    SPEvaluationResult,
    Sequence,
)


class SPRuleQualityEvaluator(SPEvaluator):
    """Evaluates the quality of discovered sequential rules.

    Quality metrics provide insights into the interestingness
    and reliability of the discovered rules.

    Metrics computed:
    - avg_confidence: Average confidence of all rules
    - avg_lift: Average lift of all rules
    - avg_support: Average support of all rules
    - avg_pattern_length: Average length of patterns
    - num_high_confidence_rules: Rules with confidence >= threshold
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.8,
        high_lift_threshold: float = 2.0,
    ) -> None:
        """Initialize quality evaluator.

        Args:
            high_confidence_threshold: Threshold for high confidence rules.
            high_lift_threshold: Threshold for high lift rules.
        """
        self._high_confidence_threshold = high_confidence_threshold
        self._high_lift_threshold = high_lift_threshold

    @property
    def name(self) -> str:
        return "SPRuleQuality"

    def evaluate(
        self,
        result: SPAlgorithmResult,
        sequences: list[Sequence],
    ) -> SPEvaluationResult:
        """Evaluate rule quality metrics.

        Args:
            result: Algorithm execution result.
            sequences: Original sequences.

        Returns:
            SPEvaluationResult with quality metrics.
        """
        patterns = result.patterns
        rules = result.rules

        # Pattern statistics
        pattern_lengths = [len(p.elements) for p in patterns]
        avg_pattern_length = (
            statistics.mean(pattern_lengths) if pattern_lengths else 0.0
        )
        max_pattern_length = max(pattern_lengths) if pattern_lengths else 0

        pattern_supports = [p.support for p in patterns]
        avg_pattern_support = (
            statistics.mean(pattern_supports) if pattern_supports else 0.0
        )

        if not rules:
            return SPEvaluationResult(
                metrics={
                    "avg_confidence": 0.0,
                    "avg_lift": 0.0,
                    "avg_support": 0.0,
                    "avg_pattern_length": avg_pattern_length,
                    "max_pattern_length": float(max_pattern_length),
                    "num_high_confidence_rules": 0,
                    "num_high_lift_rules": 0,
                },
                details={
                    "total_patterns": len(patterns),
                    "total_rules": 0,
                    "pattern_length_distribution": self._get_length_distribution(
                        pattern_lengths
                    ),
                },
            )

        # Rule statistics
        confidences = [r.confidence for r in rules]
        lifts = [r.lift for r in rules if r.lift != float("inf")]
        supports = [r.support for r in rules]

        avg_confidence = statistics.mean(confidences)
        avg_lift = statistics.mean(lifts) if lifts else 0.0
        avg_support = statistics.mean(supports)

        high_confidence_rules = sum(
            1 for r in rules if r.confidence >= self._high_confidence_threshold
        )
        high_lift_rules = sum(
            1 for r in rules if r.lift >= self._high_lift_threshold
        )

        return SPEvaluationResult(
            metrics={
                "avg_confidence": avg_confidence,
                "avg_lift": avg_lift,
                "avg_support": avg_support,
                "avg_pattern_length": avg_pattern_length,
                "max_pattern_length": float(max_pattern_length),
                "num_high_confidence_rules": high_confidence_rules,
                "num_high_lift_rules": high_lift_rules,
            },
            details={
                "total_patterns": len(patterns),
                "total_rules": len(rules),
                "pattern_length_distribution": self._get_length_distribution(
                    pattern_lengths
                ),
                "confidence_distribution": self._get_confidence_distribution(
                    confidences
                ),
                "std_confidence": (
                    statistics.stdev(confidences) if len(confidences) > 1 else 0.0
                ),
            },
        )

    def _get_length_distribution(
        self, lengths: list[int]
    ) -> dict[int, int]:
        """Get distribution of pattern lengths."""
        dist: dict[int, int] = {}
        for length in lengths:
            dist[length] = dist.get(length, 0) + 1
        return dist

    def _get_confidence_distribution(
        self, confidences: list[float]
    ) -> dict[str, int]:
        """Get distribution of confidence values."""
        dist = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        for c in confidences:
            if c < 0.2:
                dist["0.0-0.2"] += 1
            elif c < 0.4:
                dist["0.2-0.4"] += 1
            elif c < 0.6:
                dist["0.4-0.6"] += 1
            elif c < 0.8:
                dist["0.6-0.8"] += 1
            else:
                dist["0.8-1.0"] += 1
        return dist
