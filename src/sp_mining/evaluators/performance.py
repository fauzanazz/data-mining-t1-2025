"""Performance evaluator for SP mining results."""

import statistics

from sp_mining.core.interfaces import (
    SPAlgorithmResult,
    SPEvaluator,
    SPEvaluationResult,
    Sequence,
)


class SPPerformanceEvaluator(SPEvaluator):
    """Evaluates the performance characteristics of SP mining.

    Performance metrics help understand the efficiency and scalability
    of the algorithm execution.

    Metrics computed:
    - execution_time: Total execution time in seconds
    - patterns_per_second: Rate of pattern discovery
    - rules_per_second: Rate of rule generation
    - sequences_per_second: Processing throughput
    - avg_sequence_length: Average sequence length in dataset
    """

    @property
    def name(self) -> str:
        return "SPPerformance"

    def evaluate(
        self,
        result: SPAlgorithmResult,
        sequences: list[Sequence],
    ) -> SPEvaluationResult:
        """Evaluate performance metrics.

        Args:
            result: Algorithm execution result.
            sequences: Original sequences.

        Returns:
            SPEvaluationResult with performance metrics.
        """
        exec_time = result.execution_time
        num_patterns = len(result.patterns)
        num_rules = len(result.rules)
        num_sequences = len(sequences)

        # Calculate rates
        patterns_per_second = num_patterns / exec_time if exec_time > 0 else 0.0
        rules_per_second = num_rules / exec_time if exec_time > 0 else 0.0
        sequences_per_second = num_sequences / exec_time if exec_time > 0 else 0.0

        # Sequence statistics
        sequence_lengths = [len(seq.elements) for seq in sequences]
        avg_sequence_length = (
            statistics.mean(sequence_lengths) if sequence_lengths else 0.0
        )
        max_sequence_length = max(sequence_lengths) if sequence_lengths else 0

        # Total items in sequences
        total_items = sum(seq.total_items() for seq in sequences)
        items_per_sequence = total_items / num_sequences if num_sequences > 0 else 0.0

        # Pattern complexity
        pattern_sizes = [len(p.elements) for p in result.patterns]
        avg_pattern_size = statistics.mean(pattern_sizes) if pattern_sizes else 0.0

        return SPEvaluationResult(
            metrics={
                "execution_time": exec_time,
                "patterns_per_second": patterns_per_second,
                "rules_per_second": rules_per_second,
                "sequences_per_second": sequences_per_second,
                "avg_sequence_length": avg_sequence_length,
                "avg_pattern_size": avg_pattern_size,
                "items_per_sequence": items_per_sequence,
            },
            details={
                "total_patterns": num_patterns,
                "total_rules": num_rules,
                "total_sequences": num_sequences,
                "max_sequence_length": max_sequence_length,
                "total_items_in_sequences": total_items,
                "sequence_length_distribution": self._get_distribution(
                    sequence_lengths
                ),
            },
        )

    def _get_distribution(self, values: list[int]) -> dict[str, int]:
        """Get distribution of values in ranges."""
        dist = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, ">20": 0}
        for v in values:
            if v == 1:
                dist["1"] += 1
            elif v <= 5:
                dist["2-5"] += 1
            elif v <= 10:
                dist["6-10"] += 1
            elif v <= 20:
                dist["11-20"] += 1
            else:
                dist[">20"] += 1
        return dist
