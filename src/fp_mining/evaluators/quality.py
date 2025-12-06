"""Rule quality evaluator for FP mining results."""

import statistics
from typing import Any

from fp_mining.core.interfaces import (
    AlgorithmResult,
    Evaluator,
    EvaluationResult,
)


class RuleQualityEvaluator(Evaluator):
    """Evaluates the quality of discovered association rules.

    Quality metrics provide insights into the interestingness and
    reliability of the discovered rules.

    Metrics computed:
    - avg_confidence: Average confidence of all rules
    - avg_lift: Average lift of all rules
    - avg_support: Average support of all rules
    - num_high_confidence_rules: Rules with confidence >= threshold
    - num_high_lift_rules: Rules with lift >= threshold
    - redundancy_ratio: Ratio of potentially redundant rules
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
        """Return the evaluator name."""
        return "RuleQuality"

    def evaluate(
        self,
        result: AlgorithmResult,
        transactions: list[list[str]],
    ) -> EvaluationResult:
        """Evaluate rule quality metrics.

        Args:
            result: Algorithm execution result.
            transactions: Original transactions.

        Returns:
            EvaluationResult with quality metrics.
        """
        rules = result.rules

        if not rules:
            return EvaluationResult(
                metrics={
                    "avg_confidence": 0.0,
                    "avg_lift": 0.0,
                    "avg_support": 0.0,
                    "num_high_confidence_rules": 0,
                    "num_high_lift_rules": 0,
                    "redundancy_ratio": 0.0,
                },
                details={
                    "total_rules": 0,
                    "confidence_distribution": {},
                    "lift_distribution": {},
                },
            )

        # Calculate averages
        confidences = [r.confidence for r in rules]
        lifts = [r.lift for r in rules]
        supports = [r.support for r in rules]

        avg_confidence = statistics.mean(confidences)
        avg_lift = statistics.mean(lifts)
        avg_support = statistics.mean(supports)

        # Count high quality rules
        high_confidence_rules = sum(
            1 for r in rules if r.confidence >= self._high_confidence_threshold
        )
        high_lift_rules = sum(
            1 for r in rules if r.lift >= self._high_lift_threshold
        )

        # Calculate redundancy (rules with same consequent)
        consequent_counts: dict[frozenset[str], int] = {}
        for rule in rules:
            consequent_counts[rule.consequent] = consequent_counts.get(rule.consequent, 0) + 1

        redundant_rules = sum(count - 1 for count in consequent_counts.values() if count > 1)
        redundancy_ratio = redundant_rules / len(rules) if rules else 0.0

        # Create distributions
        def get_distribution(values: list[float]) -> dict[str, int]:
            dist: dict[str, int] = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
            for v in values:
                if v < 0.2:
                    dist["0.0-0.2"] += 1
                elif v < 0.4:
                    dist["0.2-0.4"] += 1
                elif v < 0.6:
                    dist["0.4-0.6"] += 1
                elif v < 0.8:
                    dist["0.6-0.8"] += 1
                else:
                    dist["0.8-1.0"] += 1
            return dist

        return EvaluationResult(
            metrics={
                "avg_confidence": avg_confidence,
                "avg_lift": avg_lift,
                "avg_support": avg_support,
                "num_high_confidence_rules": high_confidence_rules,
                "num_high_lift_rules": high_lift_rules,
                "redundancy_ratio": redundancy_ratio,
            },
            details={
                "total_rules": len(rules),
                "confidence_distribution": get_distribution(confidences),
                "lift_distribution": {
                    "<1.0": sum(1 for l in lifts if l < 1.0),
                    "1.0-2.0": sum(1 for l in lifts if 1.0 <= l < 2.0),
                    "2.0-5.0": sum(1 for l in lifts if 2.0 <= l < 5.0),
                    ">5.0": sum(1 for l in lifts if l >= 5.0),
                },
                "std_confidence": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                "std_lift": statistics.stdev(lifts) if len(lifts) > 1 else 0.0,
            },
        )
