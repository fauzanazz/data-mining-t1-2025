"""Coverage evaluator for SP mining results."""

from sp_mining.core.interfaces import (
    SPAlgorithmResult,
    SPEvaluator,
    SPEvaluationResult,
    Sequence,
    Itemset,
)


class SPCoverageEvaluator(SPEvaluator):
    """Evaluates the coverage of discovered sequential patterns.

    Coverage metrics measure how well the discovered patterns
    cover the original sequence data.

    Metrics computed:
    - item_coverage: Fraction of unique items in patterns
    - sequence_coverage: Fraction of sequences matched by patterns
    - pattern_diversity: Ratio of unique pattern elements to total
    """

    @property
    def name(self) -> str:
        return "SPCoverage"

    def _is_subsequence(
        self,
        pattern_elements: tuple[Itemset, ...],
        sequence: Sequence,
    ) -> bool:
        """Check if pattern is subsequence of sequence."""
        if not pattern_elements:
            return True
        if not sequence.elements:
            return False

        pattern_idx = 0
        for element in sequence.elements:
            if pattern_elements[pattern_idx].items.issubset(element.items):
                pattern_idx += 1
                if pattern_idx >= len(pattern_elements):
                    return True
        return False

    def evaluate(
        self,
        result: SPAlgorithmResult,
        sequences: list[Sequence],
    ) -> SPEvaluationResult:
        """Evaluate coverage metrics.

        Args:
            result: Algorithm execution result.
            sequences: Original sequences.

        Returns:
            SPEvaluationResult with coverage metrics.
        """
        # Get all unique items from sequences
        all_items: set[str] = set()
        for seq in sequences:
            for element in seq.elements:
                all_items.update(element.items)

        # Get items covered by patterns
        covered_items: set[str] = set()
        for pattern in result.patterns:
            for element in pattern.elements:
                covered_items.update(element.items)

        # Calculate item coverage
        item_coverage = len(covered_items) / len(all_items) if all_items else 0.0

        # Calculate sequence coverage
        sequences_matched = 0
        for seq in sequences:
            for pattern in result.patterns:
                if self._is_subsequence(pattern.elements, seq):
                    sequences_matched += 1
                    break

        sequence_coverage = sequences_matched / len(sequences) if sequences else 0.0

        # Calculate pattern diversity
        all_pattern_elements: set[frozenset[str]] = set()
        total_elements = 0
        for pattern in result.patterns:
            for element in pattern.elements:
                all_pattern_elements.add(element.items)
                total_elements += 1

        pattern_diversity = (
            len(all_pattern_elements) / total_elements
            if total_elements > 0
            else 0.0
        )

        # Rule coverage
        sequences_with_rules = 0
        for seq in sequences:
            for rule in result.rules:
                if self._is_subsequence(rule.antecedent, seq):
                    sequences_with_rules += 1
                    break

        rule_coverage = sequences_with_rules / len(sequences) if sequences else 0.0

        return SPEvaluationResult(
            metrics={
                "item_coverage": item_coverage,
                "sequence_coverage": sequence_coverage,
                "pattern_diversity": pattern_diversity,
                "rule_coverage": rule_coverage,
            },
            details={
                "total_items": len(all_items),
                "covered_items": len(covered_items),
                "total_sequences": len(sequences),
                "sequences_matched": sequences_matched,
                "unique_pattern_elements": len(all_pattern_elements),
                "total_pattern_elements": total_elements,
            },
        )
